# [Tune] Hybrid 융합 결과를 HCX-005(디코더 전용 LLM listwise reranker)로 재정렬합니다.
# cross-encoder와 달리 후보 전체를 한 프롬프트로 같이 보여주는 방식이라, N(후보 개수)마다
# API를 다시 호출해야 합니다(캐싱 불가) — 그래서 비용을 고려해 4개 도메인만 먼저 봅니다.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from core.config import *
from core.integrity_check import *
from retrieval.hybrid_search import hybrid_search
from retrieval.llm_reranker import llm_rerank_candidates
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

HYBRID_POOL_FOR_LLM_RERANK = 50
LLM_RERANK_N = 15  # cross-encoder(N=25)와 달리 호출당 비용이 있어 더 적게 시작
TOP_K_FINAL = 5

# reranker 효과를 볼 만한 도메인만 선별 (hidden_assets_report·unclaimed_funds는 이미 양호해서 제외)
TARGET_DOMAINS = {
    "debt_adjustment",
    "deposit_insurance_payout",
    "deposit_protection",
    "mistaken_transfer",
}

gold_records = load_gold_set()
gold_records = [r for r in gold_records if r["business_function"] in TARGET_DOMAINS]
print("평가 문항 수(도메인 필터):", len(gold_records), flush=True)


# ============================================================
# 1) baseline(rerank 없음)과 LLM rerank 결과를 문항별로 수집
# ============================================================
import time as _time

_start = _time.perf_counter()
baseline_pairs: list[tuple[dict, list[str]]] = []
llm_pairs: list[tuple[dict, list[str]]] = []

for i, record in enumerate(gold_records, 1):
    fused = hybrid_search(
        record["question"],
        top_k=LLM_RERANK_N,
        business_function=None,
        candidate_pool_size=HYBRID_POOL_FOR_LLM_RERANK,
    )
    baseline_pairs.append(
        (record, [r["chunk"]["chunk_id"] for r in fused[:TOP_K_FINAL]])
    )

    reranked = llm_rerank_candidates(record["question"], fused, top_k=TOP_K_FINAL)
    llm_pairs.append((record, [r["chunk"]["chunk_id"] for r in reranked]))

    elapsed = _time.perf_counter() - _start
    avg = elapsed / i
    remaining = avg * (len(gold_records) - i)
    print(
        f"  진행: {i}/{len(gold_records)} "
        f"(경과 {elapsed:.0f}초, 평균 {avg:.1f}초/건, 예상 잔여 {remaining:.0f}초)",
        flush=True,
    )

print("완료:", len(llm_pairs), "건 (hybrid pool=", HYBRID_POOL_FOR_LLM_RERANK, ", LLM rerank N=", LLM_RERANK_N, ")", flush=True)


# ============================================================
# 2) 평가 함수 (전체 + 도메인별)
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
# 3) 전체 비교: baseline vs LLM rerank
# ============================================================
baseline_summary = evaluate_ranked_ids_list(baseline_pairs)
llm_summary = evaluate_ranked_ids_list(llm_pairs)

log_result("hybrid_pool50_norerank_4dom", baseline_summary)
log_result(f"llm_rerank_N{LLM_RERANK_N}_4dom", llm_summary)

print("\n=== 전체 비교 ===")
compare_df = pd.DataFrame(
    {"baseline(rerank 없음)": baseline_summary, f"llm_rerank_N{LLM_RERANK_N}": llm_summary}
).T
print(compare_df[["hit@3", "recall@5", "mrr@10", "map@10", "ndcg@5"]])


# ============================================================
# 4) 도메인별 비교
# ============================================================
baseline_domain = domain_breakdown(baseline_pairs)
llm_domain = domain_breakdown(llm_pairs)

print("\n=== 도메인별 비교: baseline vs LLM rerank ===")
domain_compare = baseline_domain[["hit@3", "recall@5", "mrr@10", "ndcg@5"]].add_suffix("_baseline").join(
    llm_domain[["hit@3", "recall@5", "mrr@10", "ndcg@5"]].add_suffix(f"_llm_N{LLM_RERANK_N}")
).join(llm_domain[["n"]])
print(domain_compare)

baseline_domain.to_csv(RESULT_ROOT / "domain_breakdown_hybrid_norerank_4dom.csv", encoding="utf-8-sig")
llm_domain.to_csv(RESULT_ROOT / f"domain_breakdown_llm_rerank_N{LLM_RERANK_N}_4dom.csv", encoding="utf-8-sig")
domain_compare.to_csv(RESULT_ROOT / "domain_breakdown_llm_rerank_vs_norerank.csv", encoding="utf-8-sig")
print("\n도메인별 CSV 저장 완료:", RESULT_ROOT)
