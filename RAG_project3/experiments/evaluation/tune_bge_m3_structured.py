# [Tune] BGE-M3 Dense/Sparse/ColBERT 모드별 점수를 질문당 한 번만 계산해서 캐싱해두고,
# 세 가중치(dense:sparse:colbert) 조합만 바꿔가며 스윕합니다(재계산 없음, HCX API 불필요).
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "production"))

import time

from core.config import *
from core.integrity_check import *
from retrieval.bge_m3_structured_search import (
    CHUNK_IDS,
    CHUNK_BY_ID,
    compute_mode_scores,
    combine_mode_scores,
)
from experiments.evaluation.eval_search import (
    load_gold_set,
    hit_at_k,
    recall_at_k,
    mrr_at_k,
    average_precision_at_k,
    complete_at_k,
    ndcg_at_k,
    precision_at_k,
    f1_at_k,
    log_result,
)

TOP_K_FINAL = 5

DENSE_OPTIONS = [0.5, 1.0, 1.5, 2.0]
SPARSE_OPTIONS = [0.0, 0.15, 0.3, 0.5]
COLBERT_OPTIONS = [0.5, 1.0, 1.5, 2.0]

gold_records = load_gold_set()
print("평가 문항 수:", len(gold_records), flush=True)


# ============================================================
# 1) 질문당 한 번만: dense/sparse/colbert 모드별 점수 캐싱
# ============================================================
start = time.perf_counter()
cache: list[tuple[dict, tuple]] = []
for i, record in enumerate(gold_records, 1):
    mode_scores = compute_mode_scores(record["question"])
    cache.append((record, mode_scores))
    if i % 20 == 0:
        elapsed = time.perf_counter() - start
        print(f"  진행: {i}/{len(gold_records)} (경과 {elapsed:.0f}초)", flush=True)

print("캐시 완료:", len(cache), "건, 총", round(time.perf_counter() - start, 1), "초", flush=True)


# ============================================================
# 2) 평가 함수
# ============================================================
def evaluate_weights(weights: tuple[float, float, float]) -> dict:
    rows = []
    for record, (dense_scores, sparse_scores, colbert_scores) in cache:
        fused = combine_mode_scores(
            dense_scores, sparse_scores, colbert_scores,
            weights=weights, top_k=TOP_K_FINAL,
        )
        ranked_ids = [r["chunk"]["chunk_id"] for r in fused]

        gold_ids = record["gold_chunk_ids"]
        primary_ids = record["primary_chunk_ids"]
        supporting_ids = record["supporting_chunk_ids"]

        precision5 = precision_at_k(ranked_ids, gold_ids, 5)
        recall5 = recall_at_k(ranked_ids, gold_ids, 5)

        row = {
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

    df = pd.DataFrame(rows)
    summary: dict = {"n_questions": len(df)}
    for metric in (
        "hit@3", "recall@5", "mrr@10", "map@10",
        "precision@5", "f1@5", "ndcg@5",
    ):
        summary[metric] = round(float(df[metric].mean()), 4)

    mask = df["complete@5"].notna()
    summary["complete@5"] = (
        round(float(df.loc[mask, "complete@5"].mean()), 4)
        if mask.any() else None
    )
    summary["complete@5_n"] = int(mask.sum())
    return summary


# ============================================================
# 3) 가중치 그리드 스윕 (dense x sparse x colbert = 4x4x4 = 64개)
# ============================================================
summaries: dict[str, dict] = {}
for dense_w in DENSE_OPTIONS:
    for sparse_w in SPARSE_OPTIONS:
        for colbert_w in COLBERT_OPTIONS:
            weights = (dense_w, sparse_w, colbert_w)
            combo_name = f"bge_m3_structured_d{dense_w}_s{sparse_w}_c{colbert_w}"
            summary = evaluate_weights(weights)
            summaries[combo_name] = summary
            log_result(combo_name, summary)

result_df = pd.DataFrame(summaries).T.sort_values("mrr@10", ascending=False)
print("\n=== 가중치 스윕 결과 (상위 15개, mrr@10 내림차순) ===")
print(result_df[["mrr@10", "recall@5", "ndcg@5", "hit@3", "map@10"]].head(15))

best_name = result_df.index[0]
print(f"\n>>> 최고 조합: {best_name} (mrr@10={summaries[best_name]['mrr@10']})")
print("성능:", json.dumps(summaries[best_name], ensure_ascii=False, indent=2))
