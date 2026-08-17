# [Phase4] 민원처리질문(5번에서 intent="민원처리"로 판정된 경우) 전용 3단계 템플릿 답변.
# 절차안내 -> 서류안내 -> 페이지연결 순서로 구조화하되, 근거·URL 검증 로직은
# generation/answer_generation.py의 기존 함수를 그대로 재사용한다.
from __future__ import annotations

from core.config import HCX_RAG_MIN_SCORE_RERANK
from core.hcx_api import hcx_chat_text
from generation.answer_generation import (
    _allowed_evidence_values,
    _remove_unsupported_urls_and_phones,
    _verify_numeric_claims,
)


def generate_complaint_template_answer(
    question: str,
    search_results: list[dict],
    *,
    min_score: float = HCX_RAG_MIN_SCORE_RERANK,
) -> str:
    """호출 전제: classification.pipeline.classify()가 intent="민원처리"로 확정한
    질문에만 써야 한다. 정보질문(ELIGIBILITY/AMOUNT 등)에 쓰면 절차가 아닌 내용도
    "① 신청 절차" 라벨 아래 억지로 욱여넣는 오출력이 나온다(세션에서 실측 확인됨)."""
    if (
        not search_results
        or search_results[0]["score"] < min_score
    ):
        return (
            "검색된 공식 근거의 관련도가 충분하지 않아 "
            "현재 수집 데이터만으로는 정확하게 답할 수 없습니다."
        )

    evidence_blocks = []
    for rank, result in enumerate(search_results, start=1):
        chunk = result["chunk"]
        evidence_blocks.append("\n".join([
            f"[근거 {rank}]",
            f"문서 ID: {chunk.get('document_id')}",
            f"청크 ID: {chunk.get('chunk_id')}",
            f"업무: {chunk.get('business_function')}",
            f"출처: {chunk.get('source_url')}",
            "내용:",
            chunk.get("content", ""),
        ]))

    allowed_urls, allowed_phones = _allowed_evidence_values(search_results)
    evidence_text = "\n\n".join(evidence_blocks)

    raw_answer = hcx_chat_text(
        system_prompt=(
            "당신은 예금보험공사 공식 문서 기반 RAG 답변기입니다. "
            "사용자는 민원처리(신청·접수 등 절차)에 대해 질문했습니다. "
            "제공된 근거에 명시된 정보만 사용해 반드시 아래 3단계 형식으로 "
            "답변하세요. 근거에 없는 내용은 절대 만들지 마세요:\n\n"
            "① 신청 절차: 신청 방법과 절차를 근거에 있는 대로 설명\n"
            "② 필요 서류: 근거에 명시된 구비서류 목록을 그대로 나열 "
            "(근거에 서류 정보가 없으면 이 항목은 생략)\n"
            "③ 신청 페이지: 아래 [답변에 사용 가능한 URL] 중 이 문의와 "
            "관련된 URL을 안내 (다운로드 링크가 아니라 안내 페이지이므로 "
            "'이 링크에서 서식을 받으세요'가 아니라 '이 페이지에서 확인하세요' "
            "식으로 표현)\n\n"
            "특히 금액·기한·서류명 등 구체적인 내용은 근거 원문에 그대로 "
            "적힌 표현만 사용하세요. 근거가 부족한 항목은 억지로 채우지 "
            "말고 생략하세요."
        ),
        user_prompt=f"""
[사용자 질문]
{question}

[검색된 공식 근거]
{evidence_text}

[답변에 사용 가능한 URL]
{sorted(allowed_urls)}

[답변에 사용 가능한 전화번호]
{sorted(allowed_phones)}

한국어로 답변하세요. URL과 전화번호는 위 허용 목록에 있는 값만 그대로 사용할 수 있습니다.
""".strip(),
        max_tokens=900,
        temperature=0.0,
    )

    verified_answer = _verify_numeric_claims(evidence_text, raw_answer)
    answer = _remove_unsupported_urls_and_phones(
        verified_answer, allowed_urls, allowed_phones
    )

    used_chunks = [result["chunk"].get("chunk_id") for result in search_results]
    appendix = "\n\n근거 청크: " + ", ".join(used_chunks)
    return answer + appendix
