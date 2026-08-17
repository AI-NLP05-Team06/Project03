# [Step 2/2] evaluation/cache_hybrid_colbert_candidates.py가 저장한 Hybrid+ColBERT(3-way)
# top-30 후보를 cross-encoder(BGE-Reranker-v2-m3)로 재정렬하고 평가합니다.
# 이 스크립트는 "구버전 transformers(4.46.3)" 환경에서 실행하세요 (reranker용).
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "production"))

import json

from core.config import *
from core.integrity_check import *
from retrieval.reranker import rerank_candidates
from experiments.evaluation.eval_search import *

CANDIDATES_PATH = BGE_M3_DATA_ROOT / "hybrid_colbert_candidates.json"
COMBO_NAME = "hybrid_colbert_d1.0_b0.15_c1.5_rerank"
TOP_K_FINAL = 5

with open(CANDIDATES_PATH, encoding="utf-8") as f:
    cached_candidates = json.load(f)
candidates_by_eval_id = {c["evaluation_id"]: c["candidates"] for c in cached_candidates}

chunk_by_id = {chunk["chunk_id"]: chunk for chunk in RESULT["chunks"]}
gold_records = load_gold_set()
print("평가 문항 수:", len(gold_records), flush=True)


def rerank_search_fn(record: dict) -> list[dict]:
    cached = candidates_by_eval_id[record["evaluation_id"]]
    candidates = [
        {"score": c["score"], "chunk": chunk_by_id[c["chunk_id"]]}
        for c in cached
        if c["chunk_id"] in chunk_by_id
    ]
    return rerank_candidates(record["question"], candidates, top_k=TOP_K_FINAL)


rows = []
for record in gold_records:
    ranked = rerank_search_fn(record)
    ranked_ids = [r["chunk"]["chunk_id"] for r in ranked]

    gold_ids = record["gold_chunk_ids"]
    primary_ids = record["primary_chunk_ids"]
    supporting_ids = record["supporting_chunk_ids"]

    precision5 = precision_at_k(ranked_ids, gold_ids, 5)
    recall5 = recall_at_k(ranked_ids, gold_ids, 5)

    row = {
        "evaluation_id": record["evaluation_id"],
        "business_function": record["business_function"],
        "hit@3": hit_at_k(ranked_ids, gold_ids, 3),
        "recall@5": recall5,
        "mrr@10": mrr_at_k(ranked_ids, gold_ids, 10),
        "map@10": average_precision_at_k(ranked_ids, gold_ids, 10),
        "precision@5": precision5,
        "f1@5": f1_at_k(precision5, recall5),
        "ndcg@5": ndcg_at_k(ranked_ids, primary_ids, supporting_ids, 5),
    }
    complete_applicable = (
        record["multi_chunk_required"] and 0 < len(primary_ids) <= 5
    )
    row["complete@5"] = (
        complete_at_k(ranked_ids, primary_ids, 5)
        if complete_applicable else None
    )
    rows.append(row)

detail_df = pd.DataFrame(rows)
summary: dict = {"n_questions": len(detail_df)}
for metric in ("hit@3", "recall@5", "mrr@10", "map@10", "precision@5", "f1@5", "ndcg@5"):
    summary[metric] = round(float(detail_df[metric].mean()), 4)
mask = detail_df["complete@5"].notna()
summary["complete@5"] = round(float(detail_df.loc[mask, "complete@5"].mean()), 4) if mask.any() else None
summary["complete@5_n"] = int(mask.sum())

print("\n=== Hybrid+ColBERT(3-way) + Reranker 결과 ===")
print(json.dumps(summary, ensure_ascii=False, indent=2))

log_result(COMBO_NAME, summary)

detail_path = DETAIL_ROOT / f"{COMBO_NAME}.csv"
detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
print("문항별 상세 결과 저장:", detail_path)

domain_metrics = ["hit@3", "recall@5", "mrr@10", "map@10", "precision@5", "f1@5", "ndcg@5"]
domain_df = detail_df.groupby("business_function")[domain_metrics].mean().round(4)
domain_df["n"] = detail_df.groupby("business_function").size()
domain_path = DOMAIN_BREAKDOWN_ROOT / f"{COMBO_NAME}.csv"
domain_df.to_csv(domain_path, encoding="utf-8-sig")
print("도메인별 결과 저장:", domain_path)
print(domain_df)
