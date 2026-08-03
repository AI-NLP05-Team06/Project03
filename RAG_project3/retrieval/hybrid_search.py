# [Hybrid] Dense(semantic_search_hcx) + BM25(bm25_search) 결과를 Min-Max 정규화 후 가중합으로 합칩니다.
from __future__ import annotations

from retrieval.bm25_search import bm25_search
from retrieval.search_answer import semantic_search_hcx

# 튜닝 결과 확정값 (tune_hybrid.py 단계적 스윕 + 5단위 정밀 그리드에서 선정):
# candidate_pool_size=10, fusion=Weighted-sum(Min-Max), Dense:BM25=0.7:0.3
CANDIDATE_POOL_SIZE = 10
FUSION_WEIGHTS = (0.7, 0.3)


def _normalize_scores(results: list[dict]) -> dict[str, float]:
    """리스트 안에서 점수를 0~1로 Min-Max 정규화합니다."""
    if not results:
        return {}

    scores = [result["score"] for result in results]
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12:
        return {result["chunk"]["chunk_id"]: 1.0 for result in results}

    return {
        result["chunk"]["chunk_id"]: (result["score"] - lo) / (hi - lo)
        for result in results
    }


def weighted_sum_fusion(
    dense_results: list[dict],
    bm25_results: list[dict],
    *,
    weights: tuple[float, float] = FUSION_WEIGHTS,
) -> list[dict]:
    """각 리스트는 {"score":..., "chunk":...} 형태로, 이미 점수 내림차순 정렬돼 있어야 합니다."""
    dense_norm = _normalize_scores(dense_results)
    bm25_norm = _normalize_scores(bm25_results)

    chunk_by_id = {
        result["chunk"]["chunk_id"]: result["chunk"]
        for result in dense_results + bm25_results
    }
    all_ids = set(dense_norm) | set(bm25_norm)

    fused = [
        {
            "score": (
                weights[0] * dense_norm.get(chunk_id, 0.0)
                + weights[1] * bm25_norm.get(chunk_id, 0.0)
            ),
            "chunk": chunk_by_id[chunk_id],
        }
        for chunk_id in all_ids
    ]
    fused.sort(key=lambda item: item["score"], reverse=True)
    return fused


def hybrid_search(
    question: str,
    *,
    top_k: int = 5,
    business_function: str | None = None,
    candidate_pool_size: int = CANDIDATE_POOL_SIZE,
) -> list[dict]:
    dense_results = semantic_search_hcx(
        question,
        top_k=candidate_pool_size,
        business_function=business_function,
        min_score=None,
    )
    bm25_results = bm25_search(
        question,
        top_k=candidate_pool_size,
        business_function=business_function,
    )

    fused = weighted_sum_fusion(dense_results, bm25_results)
    return fused[:top_k]


print("Hybrid 검색(Weighted-sum, Min-Max) 준비 완료")
