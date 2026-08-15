# [실험] title을 포함시킨 Dense 임베딩(scripts/build_dense_structured_embeddings.py)으로
# 기존 확정 Hybrid(Dense:BM25=7:3, Min-Max, pool=10)를 다시 구성해 120문항 전체로 평가하고,
# content-only 기존 확정값(hybrid_wsum_minmax_w70_30_pool10_norerank_noexp)과 비교합니다.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import json

from core.config import *
from core.integrity_check import *
from core.hcx_api import hcx_embed_text
from retrieval.bm25_search import bm25_search
from retrieval.hybrid_search import weighted_sum_fusion, CANDIDATE_POOL_SIZE, FUSION_WEIGHTS
from evaluation.eval_search import load_gold_set, evaluate_search, log_result

STRUCTURED_PATH = EXPERIMENTS_ROOT / "chunk_embeddings_dense_structured.jsonl"
COMBO_NAME = "hybrid_dense_structured_title_w70_30_pool10_norerank_noexp"
BASELINE_COMBO_NAME = "hybrid_wsum_minmax_w70_30_pool10_norerank_noexp"

chunk_by_id = {c["chunk_id"]: c for c in RESULT["chunks"]}

structured_vectors: dict[str, np.ndarray] = {}
with open(STRUCTURED_PATH, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        structured_vectors[rec["chunk_id"]] = np.asarray(rec["embedding"], dtype=np.float32)

print("구조화(title+content) 임베딩 로드:", len(structured_vectors), "개")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def dense_search_structured(
    question: str,
    *,
    top_k: int = 5,
    business_function: str | None = None,
    min_score: float | None = None,
) -> list[dict]:
    query_vector = np.asarray(hcx_embed_text(question), dtype=np.float32)
    scored = []
    for chunk_id, vector in structured_vectors.items():
        chunk = chunk_by_id.get(chunk_id)
        if not chunk:
            continue
        if business_function and chunk.get("business_function") != business_function:
            continue
        score = cosine_similarity(query_vector, vector)
        if min_score is not None and score < min_score:
            continue
        scored.append({"score": score, "chunk": chunk})
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def hybrid_search_structured(
    question: str,
    *,
    top_k: int = 5,
    business_function: str | None = None,
    candidate_pool_size: int = CANDIDATE_POOL_SIZE,
) -> list[dict]:
    dense_results = dense_search_structured(
        question, top_k=candidate_pool_size, business_function=business_function,
    )
    bm25_results = bm25_search(
        question, top_k=candidate_pool_size, business_function=business_function,
    )
    fused = weighted_sum_fusion(dense_results, bm25_results, weights=FUSION_WEIGHTS)
    return fused[:top_k]


gold_records = load_gold_set()
print("평가 문항 수:", len(gold_records))

eval_result = evaluate_search(
    lambda q: hybrid_search_structured(q, top_k=5, business_function=None),
    gold_records,
)
print("\n=== Hybrid(title+content Dense, 7:3) 결과 ===")
print(json.dumps(eval_result["summary"], ensure_ascii=False, indent=2))
log_result(COMBO_NAME, eval_result["summary"])

detail_path = DETAIL_ROOT / f"{COMBO_NAME}.csv"
eval_result["detail"].to_csv(detail_path, index=False, encoding="utf-8-sig")
print("문항별 상세 저장:", detail_path)

# ============================================================
# 기존 확정값(content-only)과 비교
# ============================================================
existing_log = pd.read_csv(LOG_ROOT / "search_eval_log.csv")
baseline_row = existing_log[existing_log["combo"] == BASELINE_COMBO_NAME]

if baseline_row.empty:
    print(f"\n[경고] 베이스라인 '{BASELINE_COMBO_NAME}'을 로그에서 못 찾았습니다.")
else:
    baseline = baseline_row.iloc[0].to_dict()
    new = eval_result["summary"]
    print("\n=== 비교: content-only(기존 확정) vs title+content(신규) ===")
    for metric in ("hit@3", "recall@5", "mrr@10", "map@10", "ndcg@5"):
        print(f"  {metric}: {baseline[metric]:.4f} -> {new[metric]:.4f} (차이 {new[metric]-baseline[metric]:+.4f})")
