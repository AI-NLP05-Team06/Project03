# [Rerank] 로컬 BGE-Reranker-v2-m3(cross-encoder)로 하이브리드 후보를 재정렬합니다.
from __future__ import annotations

import os

# transformers가 (안 쓰는) tensorflow를 임포트하려다 이 환경의 깨진 TF 설치와
# 충돌하는 문제를 막기 위해, FlagEmbedding을 불러오기 전에 미리 꺼둡니다.
os.environ.setdefault("USE_TF", "0")

from FlagEmbedding import FlagReranker

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
    rerank_scores = reranker.compute_score(
        pairs, normalize=False, max_length=RERANK_MAX_LENGTH,
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


print("Reranker(BGE-Reranker-v2-m3) 모듈 준비 완료")
