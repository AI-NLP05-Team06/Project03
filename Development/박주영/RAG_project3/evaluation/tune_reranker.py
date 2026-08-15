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

HYBRID_POOL_FOR_RERANK = 50
MAX_RERANK_N = 50
N_OPTIONS = [10, 15, 20, 25, 30, 40, 50]
TOP_K_FINAL = 5

# GPU(Colab)에서는 CPU 대비 훨씬 빠르므로 도메인/표본 제한 없이 전체 120문항으로 스윕합니다.
# (로컬 CPU에서 빠르게 방향성만 볼 땐 아래 두 줄의 주석을 풀어서 4개 도메인·표본만 쓰세요.)
# TARGET_DOMAINS = {"debt_adjustment", "deposit_insurance_payout", "deposit_protection", "mistaken_transfer"}
# gold_records = [r for r in gold_records if r["business_function"] in TARGET_DOMAINS][:20]

gold_records = load_gold_set()
print("평가 문항 수(전체):", len(gold_records), flush=True)


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
def _per_question_rows(ranked_ids_by_record: list[tuple[dict, list[str]]]) -> pd.DataFrame:
    rows = []
    for record, ranked_ids in ranked_ids_by_record:
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
    return pd.DataFrame(rows)


def evaluate_ranked_ids_list(ranked_ids_by_record: list[tuple[dict, list[str]]]) -> dict:
    df = _per_question_rows(ranked_ids_by_record)
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


def domain_breakdown(ranked_ids_by_record: list[tuple[dict, list[str]]]) -> pd.DataFrame:
    df = _per_question_rows(ranked_ids_by_record)
    metrics = ["hit@3", "recall@5", "mrr@10", "map@10", "precision@5", "f1@5", "ndcg@5"]
    g = df.groupby("business_function")[metrics].mean().round(4)
    g["n"] = df.groupby("business_function").size()
    return g


# ============================================================
# 3) baseline: rerank 없이 hybrid(pool=30) 그대로 top-5
# ============================================================
baseline_pairs = [
    (record, [r["chunk"]["chunk_id"] for r in fused[:TOP_K_FINAL]])
    for record, fused, _ in cache
]
baseline_summary = evaluate_ranked_ids_list(baseline_pairs)
log_result("hybrid_pool30_norerank_full", baseline_summary)
print("\n=== baseline: hybrid_pool30_norerank_full ===")
print(json.dumps(baseline_summary, ensure_ascii=False, indent=2))


# ============================================================
# 4) N 스윕: 캐시된 rerank 점수로 top-N만 재정렬 후 top-5
# ============================================================
summaries: dict[str, dict] = {"hybrid_pool30_norerank_full": baseline_summary}
pairs_by_n: dict[int, list[tuple[dict, list[str]]]] = {}

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
    pairs_by_n[n] = pairs

    combo_name = f"rerank_N{n}_full"
    summary = evaluate_ranked_ids_list(pairs)
    summaries[combo_name] = summary
    log_result(combo_name, summary)

result_df = pd.DataFrame(summaries).T.sort_values("mrr@10", ascending=False)
print("\n=== N 스윕 결과 (mrr@10 내림차순) ===")
print(result_df[["mrr@10", "recall@5", "ndcg@5", "hit@3", "map@10"]])

best_name = result_df.index[0]
print(f"\n>>> 최고 조합: {best_name} (mrr@10={summaries[best_name]['mrr@10']})")
print("성능:", json.dumps(summaries[best_name], ensure_ascii=False, indent=2))


# ============================================================
# 5) 도메인별 비교: reranker 안 썼을 때(baseline) vs 확정 N(=25) 썼을 때
# ============================================================
CONFIRMED_N = 25

baseline_domain = domain_breakdown(baseline_pairs)
rerank_domain = domain_breakdown(pairs_by_n[CONFIRMED_N])

print(f"\n=== 도메인별 비교: baseline(rerank 없음) vs rerank_N{CONFIRMED_N} ===")
compare = baseline_domain[["hit@3", "recall@5", "mrr@10", "ndcg@5"]].add_suffix("_baseline").join(
    rerank_domain[["hit@3", "recall@5", "mrr@10", "ndcg@5"]].add_suffix(f"_rerank_N{CONFIRMED_N}")
).join(rerank_domain[["n"]])
print(compare)

baseline_domain.to_csv(DOMAIN_BREAKDOWN_ROOT / "hybrid_norerank_full.csv", encoding="utf-8-sig")
rerank_domain.to_csv(DOMAIN_BREAKDOWN_ROOT / f"rerank_N{CONFIRMED_N}_full.csv", encoding="utf-8-sig")
compare.to_csv(DOMAIN_BREAKDOWN_ROOT / "rerank_vs_norerank.csv", encoding="utf-8-sig")
print("\n도메인별 CSV 저장 완료:")
print(" -", DOMAIN_BREAKDOWN_ROOT / "hybrid_norerank_full.csv")
print(" -", DOMAIN_BREAKDOWN_ROOT / f"rerank_N{CONFIRMED_N}_full.csv")
print(" -", DOMAIN_BREAKDOWN_ROOT / "rerank_vs_norerank.csv")
