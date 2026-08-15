# [Rerank-LLM] HCX-005(디코더 전용 LLM)를 listwise reranker로 사용합니다.
# cross-encoder(retrieval/reranker.py)는 후보 하나하나를 독립적으로 채점하지만,
# 이건 후보 전체를 한 프롬프트에 다 보여주고 모델이 서로 비교하며 순서를 매기게 합니다.
from __future__ import annotations

import json
import re

from core.hcx_api import hcx_chat_text

LLM_RERANK_CONTENT_MAX_CHARS = 300

_SYSTEM_PROMPT = (
    "당신은 검색 결과 재정렬 시스템입니다. "
    "사용자 질문과 후보 문서 목록이 주어지면, 질문에 대한 답으로 가장 관련성 높은 "
    "순서대로 문서 ID를 정렬하세요. "
    '다른 설명 없이 JSON 배열(예: ["ID1", "ID2", ...])만 출력하세요.'
)


def _build_candidate_block(chunk: dict) -> str:
    parts = [
        chunk.get("title") or "",
        chunk.get("section_title") or "",
        (chunk.get("content") or "")[:LLM_RERANK_CONTENT_MAX_CHARS],
    ]
    body = "\n".join(part for part in parts if part)
    return f"[{chunk['chunk_id']}]\n{body}"


def _parse_ranked_ids(raw_response: str, valid_ids: set[str]) -> list[str]:
    """모델 응답에서 JSON 배열만 추출하고, 후보에 실제로 있는 id만 남깁니다.
    (모델이 없는 id를 지어내거나 형식을 어겨도 안전하게 무시)"""
    match = re.search(r"\[.*\]", raw_response, re.DOTALL)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []

    seen: set[str] = set()
    ranked_ids: list[str] = []
    for item in parsed:
        chunk_id = str(item)
        if chunk_id in valid_ids and chunk_id not in seen:
            ranked_ids.append(chunk_id)
            seen.add(chunk_id)
    return ranked_ids


def llm_rerank_candidates(
    question: str,
    candidates: list[dict],
    *,
    top_k: int = 5,
) -> list[dict]:
    """candidates: [{"score":..., "chunk":...}, ...].
    HCX-005에게 후보 전체를 한 번에 보여주고 관련도 순으로 정렬해달라고 요청합니다.
    모델이 언급하지 않은 후보는 원래(hybrid) 순서 그대로 뒤에 이어붙입니다."""
    if not candidates:
        return []

    chunk_by_id = {c["chunk"]["chunk_id"]: c["chunk"] for c in candidates}
    valid_ids = set(chunk_by_id.keys())

    candidate_blocks = "\n\n".join(
        _build_candidate_block(c["chunk"]) for c in candidates
    )
    user_prompt = f"""[질문]
{question}

[후보 문서]
{candidate_blocks}

관련도 높은 순서대로 chunk_id를 JSON 배열로만 출력하세요.""".strip()

    raw_response = hcx_chat_text(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_tokens=500,
        temperature=0.0,
    )

    ranked_ids = _parse_ranked_ids(raw_response, valid_ids)
    for c in candidates:
        chunk_id = c["chunk"]["chunk_id"]
        if chunk_id not in ranked_ids:
            ranked_ids.append(chunk_id)

    reranked = [
        {"score": float(len(ranked_ids) - rank), "chunk": chunk_by_id[chunk_id]}
        for rank, chunk_id in enumerate(ranked_ids)
    ]
    return reranked[:top_k]


# 튜닝 결과 확정 전 기본값: cross-encoder와 같은 hybrid pool을 재사용하되,
# LLM 프롬프트 토큰 비용 때문에 후보 수는 더 적게(15개) 시작합니다.
HYBRID_POOL_FOR_LLM_RERANK = 50
LLM_RERANK_POOL_SIZE = 15


def hybrid_llm_rerank_search(
    question: str,
    *,
    top_k: int = 5,
    business_function: str | None = None,
    hybrid_pool_size: int = HYBRID_POOL_FOR_LLM_RERANK,
    rerank_pool_size: int = LLM_RERANK_POOL_SIZE,
) -> list[dict]:
    """Hybrid(pool=50) 융합 결과 상위 rerank_pool_size개를 HCX-005 listwise reranker로 재정렬합니다."""
    from retrieval.hybrid_search import hybrid_search

    fused = hybrid_search(
        question,
        top_k=rerank_pool_size,
        business_function=business_function,
        candidate_pool_size=hybrid_pool_size,
    )
    return llm_rerank_candidates(question, fused, top_k=top_k)


print("LLM Reranker(HCX-005) 모듈 준비 완료")
