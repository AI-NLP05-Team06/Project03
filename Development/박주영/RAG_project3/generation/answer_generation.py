# [Generation] 검색된 근거 청크만 사용해 HCX로 그라운딩된 답변을 생성합니다.
# (검색 로직과 분리: retrieval/은 "무엇을 근거로 쓸지"만 고르고, 여기서는 "그 근거로 어떻게 답할지"를 다룹니다)
import re

from core.config import *
from core.hcx_api import hcx_chat_text


def _allowed_evidence_values(
    search_results: list[dict],
) -> tuple[set[str], set[str]]:
    urls: set[str] = set()
    phones: set[str] = set()
    phone_pattern = re.compile(
        r"(?:0\d{1,2}[-)]\s*)?\d{3,4}-\d{4}"
    )

    for result in search_results:
        chunk = result["chunk"]

        for value in [
            chunk.get("source_url"),
            chunk.get("official_download_url"),
        ]:
            if value:
                urls.add(value.rstrip("/"))

        for link in chunk.get("related_links", []):
            if link.get("url"):
                urls.add(link["url"].rstrip("/"))

        phones.update(
            phone_pattern.findall(chunk.get("content", ""))
        )

    return urls, phones


def _remove_unsupported_urls_and_phones(
    answer: str,
    allowed_urls: set[str],
    allowed_phones: set[str],
) -> str:
    url_pattern = re.compile(r"https?://[^\s)\]}>]+")
    phone_pattern = re.compile(
        r"(?:0\d{1,2}[-)]\s*)?\d{3,4}-\d{4}"
    )

    def url_replacer(match: re.Match) -> str:
        candidate = match.group(0).rstrip(".")
        return (
            candidate
            if candidate.rstrip("/") in allowed_urls
            else ""
        )

    def phone_replacer(match: re.Match) -> str:
        candidate = match.group(0)
        return candidate if candidate in allowed_phones else ""

    answer = url_pattern.sub(url_replacer, answer)
    answer = phone_pattern.sub(phone_replacer, answer)
    # URL/전화번호가 "(...)"나 "[...]" 안에 단독으로 있었다면 빈 괄호가 남으므로 같이 정리합니다.
    answer = re.sub(r"[(\[][ \t]*[)\]]", "", answer)
    answer = re.sub(r"[ \t]{2,}", " ", answer)
    answer = re.sub(r"[ \t]+\n", "\n", answer)
    answer = re.sub(r"\n{3,}", "\n\n", answer)
    return answer.strip()


def _verify_numeric_claims(evidence_text: str, draft_answer: str) -> str:
    """URL/전화번호와 달리 금액·한도·기간·비율은 표현이 다양해 화이트리스트로
    못 거르므로, LLM에게 근거 원문과 한 문장씩 대조시켜 근거에 없는 수치를
    지우거나 고치게 하는 2차 검수 패스입니다."""
    return hcx_chat_text(
        system_prompt=(
            "당신은 아래 초안 답변이 근거 원문에 실제로 있는 내용만 담고 "
            "있는지 검증하고 고치는 검수자입니다. 초안에 등장하는 금액, "
            "한도, 기간, 비율 등 구체적인 수치를 근거 원문과 하나씩 "
            "대조하세요. 근거 원문에 그대로 없는 수치가 있으면 그 문장을 "
            "삭제하거나 근거에 있는 값으로 고치세요. 근거에 실제로 있는 "
            "내용, 문장 구조, 어투는 그대로 유지하고 새로운 내용을 "
            "추가하지 마세요."
        ),
        user_prompt=f"""
[검색된 공식 근거]
{evidence_text}

[검수 대상 초안 답변]
{draft_answer}

위 초안에서 근거 원문에 없는 구체적 수치를 찾아 삭제하거나 근거에 있는 값으로 고친 최종 답변만 출력하세요. 다른 설명이나 "수정했습니다" 같은 말은 출력하지 마세요.
""".strip(),
        max_tokens=900,
        temperature=0.0,
    )


def generate_grounded_hcx_answer(
    question: str,
    search_results: list[dict],
    *,
    min_score: float = HCX_RAG_MIN_SCORE_RERANK,
) -> str:
    # 기본값은 hybrid_rerank_search()의 reranker 점수(0~1) 스케일 기준.
    # Dense 단독 결과를 넣는다면 min_score=HCX_RAG_MIN_SCORE로 넘길 것.
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

    allowed_urls, allowed_phones = _allowed_evidence_values(
        search_results
    )
    evidence_text = "\n\n".join(evidence_blocks)

    raw_answer = hcx_chat_text(
        system_prompt=(
            "당신은 예금보험공사 공식 문서 기반 RAG 답변기입니다. "
            "제공된 근거에 명시된 정보만 사용하세요. "
            "근거에 없는 금액, 기한, 조건, 기관, URL, 전화번호를 "
            "만들지 마세요. 근거가 부족하면 확인할 수 없다고 답하세요. "
            "출처 목록은 작성하지 마세요. "
            "특히 금액·한도·기간·비율 등 구체적인 수치는 사전 학습된 "
            "지식이나 기억이 아니라 반드시 근거 원문에 그대로 적힌 "
            "표현만 사용하세요. 같은 항목에 대해 근거마다 다른 수치가 "
            "나오면 임의로 하나를 고르지 말고, 근거 원문에 있는 그대로 "
            "각각 언급하거나 어느 근거의 수치인지 밝히세요."
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
        verified_answer,
        allowed_urls,
        allowed_phones,
    )

    used_chunks = [
        result["chunk"].get("chunk_id")
        for result in search_results
    ]
    source_urls = list(dict.fromkeys(
        result["chunk"].get("source_url")
        for result in search_results
        if result["chunk"].get("source_url")
    ))

    appendix = "\n\n근거 청크: " + ", ".join(used_chunks)
    if source_urls:
        appendix += "\n공식 출처:\n" + "\n".join(
            f"- {url}" for url in source_urls
        )

    return answer + appendix


print("답변 생성(Grounded HCX Answer) 모듈 준비 완료")
