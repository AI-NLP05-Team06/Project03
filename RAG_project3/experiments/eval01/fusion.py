# [Experiment/eval01] evaluation/tune_hybrid.py의 rrf_fuse / weighted_sum_fuse(minmax)와
# 동일한 로직입니다. 원본은 스크립트(import 시 gold set 로드+캐시 구축까지 실행)라
# 순수 융합 함수만 이 모듈로 옮겨왔습니다(원본 파일은 손대지 않음).
from __future__ import annotations


def rrf_fuse(
    dense_results: list[dict],
    bm25_results: list[dict],
    *,
    k: int,
    weights: tuple[float, float],
) -> list[dict]:
    fused_scores: dict[str, float] = {}
    chunk_by_id: dict[str, dict] = {}

    for ranked_list, weight in zip([dense_results, bm25_results], weights):
        for rank, result in enumerate(ranked_list, start=1):
            chunk = result["chunk"]
            chunk_id = chunk["chunk_id"]
            chunk_by_id[chunk_id] = chunk
            fused_scores[chunk_id] = (
                fused_scores.get(chunk_id, 0.0) + weight / (k + rank)
            )

    fused = [
        {"score": score, "chunk": chunk_by_id[chunk_id]}
        for chunk_id, score in fused_scores.items()
    ]
    fused.sort(key=lambda item: item["score"], reverse=True)
    return fused


def _minmax_normalize(results: list[dict]) -> dict[str, float]:
    if not results:
        return {}

    scores = [r["score"] for r in results]
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12:
        return {r["chunk"]["chunk_id"]: 1.0 for r in results}
    return {
        r["chunk"]["chunk_id"]: (r["score"] - lo) / (hi - lo)
        for r in results
    }


def weighted_sum_fuse_minmax(
    dense_results: list[dict],
    bm25_results: list[dict],
    *,
    weights: tuple[float, float],
) -> list[dict]:
    dense_norm = _minmax_normalize(dense_results)
    bm25_norm = _minmax_normalize(bm25_results)

    chunk_by_id = {
        r["chunk"]["chunk_id"]: r["chunk"]
        for r in dense_results + bm25_results
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
