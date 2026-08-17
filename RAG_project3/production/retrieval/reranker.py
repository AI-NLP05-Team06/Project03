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


def hybrid_rerank_search_hedged(
    question: str,
    original_question: str,
    *,
    top_k: int = 5,
    business_function: str | None = None,
    hybrid_pool_size: int = HYBRID_POOL_FOR_RERANK,
    rerank_pool_size: int = RERANK_POOL_SIZE,
) -> list[dict]:
    """복합질의의 하위질문 검색 전용(compound_answer.py에서 사용). 하위질문
    자체의 하이브리드 후보 + 원문(분해 전 전체 질문)의 하이브리드 후보를
    합쳐서 재정렬합니다.

    eval01 실험(experiment/eval01/run_bench.py)에서 확인된 문제: 하위질문이
    원문에 있던 다른 부분의 맥락(단어)을 잃으면 검색이 흔들리는 경우가 있음
    (예: "예금 지급정지와 은행 파산은... 각각 언제 보험금을 주나요?"를 분해한
    "은행 파산이 일어났을 때..." 하위질문 혼자만으로는 반복문구 많은 엉뚱한
    문서로 쏠리는데, 원문에 있던 "지급정지"라는 단어가 임베딩을 붙잡아주는
    앵커 역할을 하고 있었음). 원문 검색 후보를 안전망으로 같이 넣으면 이
    현상이 크게 줄어듦(eval01 실측: raw 대비 격차 70~85% 감소, 일반 질문
    셋에서는 오히려 raw를 앞서기도 함).

    eval01의 하위질문별 병합(_merge_subquery_results)은 각 하위질문이 자기
    후보군 안에서만 Min-Max 정규화된 점수를 직접 비교하는 방식이라 "병합손해"
    (후보군이 약한 하위질문의 1등이 실제로 더 관련있는 다른 후보를 밀어내는
    문제)가 있었는데, 여기서는 점수가 아니라 후보 "집합"만 합친 뒤
    Cross-Encoder로 하위질문 기준 한 번에 다시 채점하므로 그 정규화 불일치
    문제 자체가 생기지 않는다(그래서 프로덕션 버전은 eval01보다 더 안전함).

    question == original_question(단일질의)이면 의미 없는 중복 검색이 되므로
    호출부에서 복합질의일 때만 써야 한다."""
    from retrieval.hybrid_search import hybrid_search

    fused_subq = hybrid_search(
        question,
        top_k=rerank_pool_size,
        business_function=business_function,
        candidate_pool_size=hybrid_pool_size,
    )
    fused_original = hybrid_search(
        original_question,
        top_k=rerank_pool_size,
        business_function=business_function,
        candidate_pool_size=hybrid_pool_size,
    )

    merged_by_id: dict[str, dict] = {}
    for item in fused_subq + fused_original:
        chunk_id = item["chunk"]["chunk_id"]
        if chunk_id not in merged_by_id:
            merged_by_id[chunk_id] = item

    return rerank_candidates(question, list(merged_by_id.values()), top_k=top_k)


print("Reranker(BGE-Reranker-v2-m3) 모듈 준비 완료")
