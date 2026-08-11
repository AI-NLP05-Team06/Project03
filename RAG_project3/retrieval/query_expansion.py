# [Phase6] 검색 고도화: Multi-Query(RAG-Fusion), HyDE. 질문 자체를 검색 전에
# 여러 형태로 바꿔서(또는 가상 답변으로) 검색 재현율을 높이는 게 목적이다.
# 세션2에서 정한 대로 reranker 이전 raw recall@k로 효과를 검증한다(6번 evaluate 참고).
from __future__ import annotations

from core.hcx_api import hcx_chat_text, hcx_embed_text

MULTI_QUERY_COUNT = 3

_MULTI_QUERY_SYSTEM_PROMPT = f"""당신은 검색 성능을 높이기 위한 질의 확장기입니다.
사용자 질문 하나를 받아서, 같은 의도를 유지하되 표현이 서로 다른 검색용 질문
{MULTI_QUERY_COUNT}개를 만드세요.

- 동의어·유사 표현으로 바꿔쓰거나, 질문의 다른 측면을 강조하는 방식으로 변형하세요.
- 원래 질문에 없는 새로운 조건이나 사실을 추가하지 마세요.
- 한 줄에 질문 하나씩, 번호나 설명 없이 질문 문장만 출력하세요."""

_HYDE_SYSTEM_PROMPT = """당신은 예금보험공사 업무에 대한 가상의 답변을 작성하는
도우미입니다. 사용자 질문에 대해, 실제 사실 여부와 무관하게 "이런 내용일 것
같다"는 그럴듯한 답변을 2~3문장으로 작성하세요. 이 답변은 실제 사용자에게
보여주지 않고 검색 임베딩 용도로만 씁니다 — 사실 확인은 필요 없고, 질문과
관련된 용어·맥락이 풍부하게 들어간 문장이면 됩니다."""


def generate_multi_queries(question: str, *, n: int = MULTI_QUERY_COUNT) -> list[str]:
    raw = hcx_chat_text(
        system_prompt=_MULTI_QUERY_SYSTEM_PROMPT,
        user_prompt=question,
        max_tokens=300,
        temperature=0.7,
    )
    variants = [line.strip() for line in raw.splitlines() if line.strip()]
    return variants[:n] or [question]


def generate_hyde_passage(question: str) -> str:
    return hcx_chat_text(
        system_prompt=_HYDE_SYSTEM_PROMPT,
        user_prompt=question,
        max_tokens=200,
        temperature=0.3,
    ).strip()


def embed_hyde_passage(question: str) -> list[float]:
    passage = generate_hyde_passage(question)
    return hcx_embed_text(passage)


def reciprocal_rank_fusion(
    ranked_id_lists: list[list[str]],
    *,
    k: int = 60,
) -> list[str]:
    """여러 검색 결과(청크 ID 순위 리스트)를 RRF로 하나의 순위로 합친다.
    score(id) = sum(1 / (k + rank)), rank는 1부터 시작."""
    scores: dict[str, float] = {}
    for ranked_ids in ranked_id_lists:
        for rank, chunk_id in enumerate(ranked_ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)
