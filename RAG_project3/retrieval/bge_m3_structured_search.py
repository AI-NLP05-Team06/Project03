# [Dense-Structured] BGE-M3를 로컬로 로드해서 Dense+Sparse+Multi-vector(ColBERT)를
# 각각 채점한 뒤 가중합으로 결합합니다. 청크 쪽은 scripts/build_bge_m3_embeddings.py +
# scripts/convert_bge_m3_embeddings_to_npz.py로 미리 만들어둔 걸 그대로 씁니다.
from __future__ import annotations

import json
import os

os.environ.setdefault("USE_TF", "0")

import numpy as np
from FlagEmbedding import BGEM3FlagModel

from core.config import BGE_M3_DATA_ROOT
from core.load_data import RESULT

NPZ_PATH = BGE_M3_DATA_ROOT / "chunk_embeddings_bge_m3_structured.npz"
SPARSE_JSON_PATH = BGE_M3_DATA_ROOT / "chunk_embeddings_bge_m3_sparse.json"

_npz = np.load(NPZ_PATH, allow_pickle=True)
CHUNK_IDS: list[str] = [str(cid) for cid in _npz["chunk_ids"]]
DENSE: np.ndarray = _npz["dense"]
COLBERT: np.ndarray = _npz["colbert"]
COLBERT_OFFSETS: np.ndarray = _npz["colbert_offsets"]

with open(SPARSE_JSON_PATH, encoding="utf-8") as f:
    SPARSE_BY_ID: dict[str, dict[str, float]] = json.load(f)

SPARSE_LIST = [SPARSE_BY_ID[cid] for cid in CHUNK_IDS]
CHUNK_BY_ID = {chunk["chunk_id"]: chunk for chunk in RESULT["chunks"]}

_DENSE_NORM = DENSE / np.linalg.norm(DENSE, axis=1, keepdims=True)

print("BGE-M3 structured 임베딩 로드 완료:", len(CHUNK_IDS), "청크", flush=True)

_model: BGEM3FlagModel | None = None


def _get_model() -> BGEM3FlagModel:
    global _model
    if _model is None:
        _model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    return _model


# 튜닝 결과 확정값 (evaluation/tune_bge_m3_structured.py, 120문항 4x4x4 그리드에서 선정):
# dense:sparse:colbert = 0.5:0.0:1.5 — sparse는 오히려 도움 안 돼서 0으로, colbert(multi-vector)가
# 가장 크게 기여함. hit@3 0.742→0.758, mrr@10 0.632→0.637 (기본값 1:0.3:1 대비 소폭 개선).
DEFAULT_WEIGHTS = (0.5, 0.0, 1.5)


def compute_mode_scores(question: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """질문 1건을 인코딩해서 (dense_scores, sparse_scores, colbert_scores) 배열 3개를
    반환합니다(각각 길이 len(CHUNK_IDS)). 가중치 튜닝 시 질문당 한 번만 호출해서
    캐싱해두고, 이후 가중치 조합만 바꿔가며 재계산 없이 스윕할 수 있습니다."""
    model = _get_model()
    q_out = model.encode(
        [question],
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True,
    )
    q_dense = q_out["dense_vecs"][0]
    q_dense_norm = q_dense / np.linalg.norm(q_dense)
    q_sparse = q_out["lexical_weights"][0]
    q_colbert = q_out["colbert_vecs"][0]

    dense_scores = _DENSE_NORM @ q_dense_norm

    sparse_scores = np.asarray(
        model.compute_lexical_matching_score([q_sparse], SPARSE_LIST)
    )[0]

    colbert_scores = np.zeros(len(CHUNK_IDS), dtype=np.float32)
    for i in range(len(CHUNK_IDS)):
        start, end = COLBERT_OFFSETS[i], COLBERT_OFFSETS[i + 1]
        p_reps = COLBERT[start:end]
        colbert_scores[i] = float(model.colbert_score(q_colbert, p_reps))

    return dense_scores, sparse_scores, colbert_scores


def compute_colbert_scores(question: str) -> np.ndarray:
    """ColBERT(multi-vector) 점수만 계산합니다(길이 len(CHUNK_IDS), CHUNK_IDS 순서와 동일).
    다른 첫단계 검색(예: 기존 Hybrid의 Dense+BM25)에 세 번째 축으로 얹을 때 씁니다."""
    model = _get_model()
    q_out = model.encode([question], return_dense=False, return_sparse=False, return_colbert_vecs=True)
    q_colbert = q_out["colbert_vecs"][0]

    colbert_scores = np.zeros(len(CHUNK_IDS), dtype=np.float32)
    for i in range(len(CHUNK_IDS)):
        start, end = COLBERT_OFFSETS[i], COLBERT_OFFSETS[i + 1]
        p_reps = COLBERT[start:end]
        colbert_scores[i] = float(model.colbert_score(q_colbert, p_reps))
    return colbert_scores


def combine_mode_scores(
    dense_scores: np.ndarray,
    sparse_scores: np.ndarray,
    colbert_scores: np.ndarray,
    *,
    weights: tuple[float, float, float] = DEFAULT_WEIGHTS,
    top_k: int = 5,
    business_function: str | None = None,
) -> list[dict]:
    """이미 계산된 모드별 점수를 가중합으로 결합해서 top_k를 반환합니다."""
    combined = (
        weights[0] * dense_scores
        + weights[1] * sparse_scores
        + weights[2] * colbert_scores
    )

    order = np.argsort(-combined)
    results: list[dict] = []
    for idx in order:
        chunk_id = CHUNK_IDS[idx]
        chunk = CHUNK_BY_ID.get(chunk_id)
        if chunk is None:
            continue
        if business_function and chunk.get("business_function") != business_function:
            continue
        results.append({"score": float(combined[idx]), "chunk": chunk})
        if len(results) >= top_k:
            break
    return results


def bge_m3_structured_search(
    question: str,
    *,
    top_k: int = 5,
    business_function: str | None = None,
    weights: tuple[float, float, float] = DEFAULT_WEIGHTS,
) -> list[dict]:
    """Dense+Sparse+Multi-vector(ColBERT)를 각각 채점해서 가중합으로 결합합니다."""
    dense_scores, sparse_scores, colbert_scores = compute_mode_scores(question)
    return combine_mode_scores(
        dense_scores,
        sparse_scores,
        colbert_scores,
        weights=weights,
        top_k=top_k,
        business_function=business_function,
    )


print("BGE-M3 Structured(Dense+Sparse+Multi-vector) 검색 모듈 준비 완료")
