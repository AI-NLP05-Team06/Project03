# [Rerank] 로컬 BGE-Reranker-v2-m3(cross-encoder)로 하이브리드 후보를 재정렬합니다.
from __future__ import annotations

import os

# transformers가 (안 쓰는) tensorflow를 임포트하려다 이 환경의 깨진 TF 설치와
# 충돌하는 문제를 막기 위해, FlagEmbedding을 불러오기 전에 미리 꺼둡니다.
os.environ.setdefault("USE_TF", "0")

from FlagEmbedding import FlagReranker  # pyright: ignore[reportMissingImports]

RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
CONTENT_MAX_CHARS = 300
RERANK_MAX_LENGTH = 256

_reranker: FlagReranker | None = None


def _get_reranker() -> FlagReranker:
    global _reranker
    if _reranker is None:
        _reranker = FlagReranker(RERANKER_MODEL_NAME, use_fp16=True)
    return _reranker


def _build_rerank_document(chunk: dict) -> str:
    parts = [
        chunk.get("title") or "",
        chunk.get("section_title") or "",
        (chunk.get("content") or "")[:CONTENT_MAX_CHARS],
    ]
    return "\n".join(part for part in parts if part)


def score_candidates(
    question: str,
    candidates: list[dict],
) -> dict[str, float]:
    """candidates 각각을 독립적으로 채점(질문-청크 쌍 하나만 봄)해서
    {chunk_id: rerank_score} 형태로 반환합니다. 후보 집합이 바뀌어도
    이미 채점한 chunk_id의 점수는 그대로 재사용할 수 있습니다."""
    if not candidates:
        return {}

    reranker = _get_reranker()
    pairs = [
        (question, _build_rerank_document(result["chunk"]))
        for result in candidates
    ]
    # normalize=True: sigmoid를 적용해 0~1 스케일로 반환합니다. 정렬 순서는
    # (sigmoid가 단조증가라) raw logit과 완전히 동일하므로 hit@3/mrr@10 등
    # 기존 튜닝 결과에는 영향이 없고, generate_grounded_hcx_answer의
    # 최소 점수 임계값(HCX_RAG_MIN_SCORE_RERANK)을 사람이 해석 가능한
    # 스케일로 쓰기 위해 바꿨습니다.
    rerank_scores = reranker.compute_score(
        pairs, normalize=True, max_length=RERANK_MAX_LENGTH,
    )
    if isinstance(rerank_scores, float):
        rerank_scores = [rerank_scores]

    return {
        result["chunk"]["chunk_id"]: float(rerank_score)
        for result, rerank_score in zip(candidates, rerank_scores)
    }


def rerank_candidates(
    question: str,
    candidates: list[dict],
    *,
    top_k: int = 5,
) -> list[dict]:
    """candidates: [{"score":..., "chunk":...}, ...] (이미 어떤 순서든 상관없음).
    cross-encoder 점수로 재정렬한 뒤 top_k를 반환합니다."""
    if not candidates:
        return []

    scores_by_id = score_candidates(question, candidates)
    reranked = [
        {"score": scores_by_id[result["chunk"]["chunk_id"]], "chunk": result["chunk"]}
        for result in candidates
    ]
    reranked.sort(key=lambda item: item["score"], reverse=True)
    return reranked[:top_k]


# 확정 파이프라인 (evaluation/tune_reranker.py 스윕 결과):
# Hybrid(pool=50)로 후보를 뽑고, 그중 상위 25개만 cross-encoder로 재정렬합니다.
HYBRID_POOL_FOR_RERANK = 50
RERANK_POOL_SIZE = 25


def hybrid_rerank_search(
    question: str,
    *,
    top_k: int = 5,
    business_function: str | None = None,
    hybrid_pool_size: int = HYBRID_POOL_FOR_RERANK,
    rerank_pool_size: int = RERANK_POOL_SIZE,
) -> list[dict]:
    """확정된 최종 검색 파이프라인: Hybrid(Dense+BM25, pool=50) -> 상위 25개
    cross-encoder 재정렬 -> top_k. 프로덕션(run_kdic_rag)과 평가 스크립트가
    공유하는 단일 진입점입니다."""
    from retrieval.hybrid_search import hybrid_search

    fused = hybrid_search(
        question,
        top_k=rerank_pool_size,
        business_function=business_function,
        candidate_pool_size=hybrid_pool_size,
    )
    return rerank_candidates(question, fused, top_k=top_k)


print("Reranker(BGE-Reranker-v2-m3) 모듈 준비 완료")
