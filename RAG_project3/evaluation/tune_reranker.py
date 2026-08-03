# [Tune] Hybrid(pool=30) 융합 결과를 cross-encoder(BGE-Reranker-v2-m3)로 재정렬합니다.
# 후보 하나하나는 독립적으로 채점되므로, 질문당 한 번만 채점해서 캐싱해두고
# N(재정렬에 포함할 후보 개수)만 슬라이스로 바꿔가며 스윕합니다.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from core.config import *
from core.integrity_check import *
from retrieval.hybrid_search import hybrid_search
from retrieval.reranker import score_candidates
from evaluation.eval_search import (
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

HYBRID_POOL_FOR_RERANK = 30
MAX_RERANK_N = 20  # 빠른 방향성 확인용: N=20 하나만 채점 (전체 스윕 시 30으로 복원)
N_OPTIONS = [20]  # 빠른 방향성 확인용 spot-check (전체 스윕은 [10,15,20,25,30]로 복원)
TOP_K_FINAL = 5

# reranker 효과를 볼 만한 도메인만 선별 (hidden_assets_report는 이미 hit@3=0.96으로
# 거의 다 맞혀서 제외, unclaimed_funds도 이미 양호해서 제외)
TARGET_DOMAINS = {
    "debt_adjustment",
    "deposit_insurance_payout",
    "deposit_protection",
    "mistaken_transfer",
}

SAMPLE_SIZE = 20  # 빠른 방향성 확인용 표본 (전체 79문항 대신 일부만)

gold_records = load_gold_set()
gold_records = [r for r in gold_records if r["business_function"] in TARGET_DOMAINS]
gold_records = gold_records[:SAMPLE_SIZE]
print("평가 문항 수(도메인 필터 + 표본):", len(gold_records), flush=True)


# ============================================================
# 1) 질문당 한 번만: hybrid(pool=30) 융합 결과 top-30 + rerank 점수 캐싱
# ============================================================
import time as _time

_start = _time.perf_counter()
cache: list[tuple[dict, list[dict], dict[str, float]]] = []
for i, record in enumerate(gold_records, 1):
    fused = hybrid_search(
        record["question"],
        top_k=MAX_RERANK_N,
        business_function=None,
        candidate_pool_size=HYBRID_POOL_FOR_RERANK,
    )
    rerank_scores = score_candidates(record["question"], fused)
    cache.append((record, fused, rerank_scores))

    elapsed = _time.perf_counter() - _start
    avg = elapsed / i
    remaining = avg * (len(gold_records) - i)
    print(
        f"  진행: {i}/{len(gold_records)} "
        f"(경과 {elapsed:.0f}초, 평균 {avg:.1f}초/건, 예상 잔여 {remaining:.0f}초)",
        flush=True,
    )

print("캐시 완료:", len(cache), "건 (hybrid pool=", HYBRID_POOL_FOR_RERANK, ", 채점 top=", MAX_RERANK_N, ")", flush=True)


# ============================================================
# 2) 평가 함수
# ============================================================
def evaluate_ranked_ids_list(ranked_ids_by_record: list[tuple[dict, list[str]]]) -> dict:
    rows = []
    for record, ranked_ids in ranked_ids_by_record:
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
# 3) baseline: rerank 없이 hybrid(pool=30) 그대로 top-5
# ============================================================
baseline_pairs = [
    (record, [r["chunk"]["chunk_id"] for r in fused[:TOP_K_FINAL]])
    for record, fused, _ in cache
]
baseline_summary = evaluate_ranked_ids_list(baseline_pairs)
log_result("hybrid_pool30_norerank_4dom", baseline_summary)
print("\n=== baseline: hybrid_pool30_norerank_4dom ===")
print(json.dumps(baseline_summary, ensure_ascii=False, indent=2))


# ============================================================
# 4) N 스윕: 캐시된 rerank 점수로 top-N만 재정렬 후 top-5
# ============================================================
summaries: dict[str, dict] = {"hybrid_pool30_norerank_4dom": baseline_summary}

for n in N_OPTIONS:
    pairs = []
    for record, fused, rerank_scores in cache:
        candidates_n = fused[:n]
        reranked = sorted(
            candidates_n,
            key=lambda r: rerank_scores[r["chunk"]["chunk_id"]],
            reverse=True,
        )
        ranked_ids = [r["chunk"]["chunk_id"] for r in reranked[:TOP_K_FINAL]]
        pairs.append((record, ranked_ids))

    combo_name = f"rerank_N{n}_4dom"
    summary = evaluate_ranked_ids_list(pairs)
    summaries[combo_name] = summary
    log_result(combo_name, summary)

result_df = pd.DataFrame(summaries).T.sort_values("mrr@10", ascending=False)
print("\n=== N 스윕 결과 (mrr@10 내림차순) ===")
print(result_df[["mrr@10", "recall@5", "ndcg@5", "hit@3", "map@10"]])

best_name = result_df.index[0]
print(f"\n>>> 최고 조합: {best_name} (mrr@10={summaries[best_name]['mrr@10']})")
print("성능:", json.dumps(summaries[best_name], ensure_ascii=False, indent=2))
