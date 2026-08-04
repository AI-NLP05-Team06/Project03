# [Step 1/2] 확정 3-way 조합(Dense=1.0, BM25=0.15, ColBERT=1.5)으로 문항당 top-30 후보를
# 뽑아 JSON으로 저장합니다. reranker(구버전 transformers)와 같은 프로세스에서 못 써서
# 2단계로 나눈 것 중 1단계 — 이 스크립트는 "최신 transformers" 환경에서 실행하세요.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import time

from core.config import *
from core.integrity_check import *
from retrieval.search_answer import semantic_search_hcx
from retrieval.bm25_search import bm25_search
from retrieval.bge_m3_structured_search import CHUNK_IDS, compute_colbert_scores
from evaluation.eval_search import load_gold_set

POOL_SIZE = 30
N_CHUNKS = len(CHUNK_IDS)
CHUNK_ID_TO_IDX = {cid: i for i, cid in enumerate(CHUNK_IDS)}
WEIGHTS = (1.0, 0.15, 1.5)  # dense, bm25, colbert (확정값: evaluation/tune_hybrid_colbert.py)

OUT_PATH = BGE_M3_DATA_ROOT / "hybrid_colbert_candidates.json"

gold_records = load_gold_set()
print("평가 문항 수:", len(gold_records), flush=True)


def _minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return np.ones_like(arr)
    return (arr - lo) / (hi - lo)


def _results_to_array(results: list[dict]) -> np.ndarray:
    arr = np.zeros(N_CHUNKS, dtype=np.float32)
    for r in results:
        idx = CHUNK_ID_TO_IDX.get(r["chunk"]["chunk_id"])
        if idx is not None:
            arr[idx] = r["score"]
    return arr


start = time.perf_counter()
cache = []
for i, record in enumerate(gold_records, 1):
    dense_results = semantic_search_hcx(
        record["question"], top_k=N_CHUNKS, business_function=None, min_score=None,
    )
    bm25_results = bm25_search(record["question"], top_k=N_CHUNKS, business_function=None)
    colbert_arr = compute_colbert_scores(record["question"])

    dense_norm = _minmax(_results_to_array(dense_results))
    bm25_norm = _minmax(_results_to_array(bm25_results))
    colbert_norm = _minmax(colbert_arr)

    combined = WEIGHTS[0] * dense_norm + WEIGHTS[1] * bm25_norm + WEIGHTS[2] * colbert_norm
    order = np.argsort(-combined)[:POOL_SIZE]

    cache.append({
        "evaluation_id": record["evaluation_id"],
        "candidates": [
            {"score": float(combined[idx]), "chunk_id": CHUNK_IDS[idx]}
            for idx in order
        ],
    })
    if i % 20 == 0:
        print(f"  진행: {i}/{len(gold_records)} (경과 {time.perf_counter()-start:.0f}초)", flush=True)

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(cache, f, ensure_ascii=False)

print("저장 완료:", OUT_PATH, flush=True)
print("총 소요:", round(time.perf_counter() - start, 1), "초", flush=True)
