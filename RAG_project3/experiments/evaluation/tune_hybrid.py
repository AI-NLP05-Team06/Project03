# [Tune] Dense/BM25 후보를 질문당 한 번만(pool=50) 가져와 캐시해두고,
# 단계적(OFAT) 탐색으로 하이브리드 검색의 4개 결정 사항을 확정합니다.
#   1단계: candidate_pool_size 결정 (fusion=RRF, k=60, weights=0.5:0.5 고정)
#   2단계: fusion 방식 결정 (RRF vs Weighted-sum[minmax] vs Weighted-sum[zscore], weights=0.5:0.5 고정)
#   3단계: weights(Dense:BM25) 결정 (1,2단계 값 고정)
# k(RRF 상수)는 원 논문의 표준값 60으로 고정하고 별도로 스윕하지 않습니다.
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "production"))

import statistics

from core.config import *
from core.integrity_check import *
from retrieval.bm25_search import *
from retrieval.search_answer import *
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

MAX_POOL = 50
TOP_K_FINAL = 10
RRF_K_STANDARD = 60

POOL_SIZES = [10, 20, 30, 50]
NEUTRAL_WEIGHTS = (0.5, 0.5)
WEIGHT_OPTIONS = [
    (0.5, 0.5),
    (0.6, 0.4),
    (0.7, 0.3),
    (0.8, 0.2),
    (0.9, 0.1),
]

gold_records = load_gold_set()
print("평가 문항 수:", len(gold_records))


# ============================================================
# 1) Dense/BM25 후보를 질문당 한 번만 pool=50으로 가져와서 캐시
#    (이후 pool_size를 줄이는 건 캐시를 앞에서부터 자르기만 하면 됨)
# ============================================================
cache: list[tuple[dict, list[dict], list[dict]]] = []
for record in gold_records:
    dense_results = semantic_search_hcx(
        record["question"],
        top_k=MAX_POOL,
        business_function=None,
        min_score=None,
    )
    bm25_results = bm25_search(
        record["question"],
        top_k=MAX_POOL,
        business_function=None,
    )
    cache.append((record, dense_results, bm25_results))

print("후보 캐시 완료:", len(cache), "건 (pool=", MAX_POOL, ")")


# ============================================================
# 2) Fusion 함수들
# ============================================================
def rrf_fuse(
    dense_results: list[dict],
    bm25_results: list[dict],
    *,
    k: int,
    weights: tuple[float, float],
) -> list[dict]:
    fused_scores: dict[str, float] = {}
    chunk_by_id: dict[str, dict] = {}

    for ranked_list, weight in zip([dense_results, bm25_results], weights):
        for rank, result in enumerate(ranked_list, start=1):
            chunk = result["chunk"]
            chunk_id = chunk["chunk_id"]
            chunk_by_id[chunk_id] = chunk
            fused_scores[chunk_id] = (
                fused_scores.get(chunk_id, 0.0) + weight / (k + rank)
            )

    fused = [
        {"score": score, "chunk": chunk_by_id[chunk_id]}
        for chunk_id, score in fused_scores.items()
    ]
    fused.sort(key=lambda item: item["score"], reverse=True)
    return fused


def _normalize_scores(results: list[dict], method: str) -> dict[str, float]:
    if not results:
        return {}

    scores = [r["score"] for r in results]

    if method == "minmax":
        lo, hi = min(scores), max(scores)
        if hi - lo < 1e-12:
            return {r["chunk"]["chunk_id"]: 1.0 for r in results}
        return {
            r["chunk"]["chunk_id"]: (r["score"] - lo) / (hi - lo)
            for r in results
        }

    if method == "zscore":
        mean = statistics.mean(scores)
        std = statistics.pstdev(scores)
        if std < 1e-12:
            return {r["chunk"]["chunk_id"]: 0.0 for r in results}
        return {
            r["chunk"]["chunk_id"]: (r["score"] - mean) / std
            for r in results
        }

    raise ValueError(f"알 수 없는 정규화 방식: {method}")


def weighted_sum_fuse(
    dense_results: list[dict],
    bm25_results: list[dict],
    *,
    norm_method: str,
    weights: tuple[float, float],
) -> list[dict]:
    dense_norm = _normalize_scores(dense_results, norm_method)
    bm25_norm = _normalize_scores(bm25_results, norm_method)

    chunk_by_id = {
        r["chunk"]["chunk_id"]: r["chunk"]
        for r in dense_results + bm25_results
    }
    all_ids = set(dense_norm) | set(bm25_norm)

    fused = [
        {
            "score": (
                weights[0] * dense_norm.get(chunk_id, 0.0)
                + weights[1] * bm25_norm.get(chunk_id, 0.0)
            ),
            "chunk": chunk_by_id[chunk_id],
        }
        for chunk_id in all_ids
    ]
    fused.sort(key=lambda item: item["score"], reverse=True)
    return fused


# ============================================================
# 3) 평가 하니스: pool_size로 캐시를 잘라서 fuse_fn 적용 후 지표 계산
# ============================================================
def evaluate_config(pool_size: int, fuse_fn) -> dict:
    rows = []

    for record, dense_results, bm25_results in cache:
        dense_slice = dense_results[:pool_size]
        bm25_slice = bm25_results[:pool_size]
        fused = fuse_fn(dense_slice, bm25_slice)
        ranked_ids = [r["chunk"]["chunk_id"] for r in fused[:TOP_K_FINAL]]

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


def run_stage(
    stage_name: str,
    configs: list[tuple[str, int, object]],
    *,
    top_n: int | None = None,
) -> tuple[str, dict, dict[str, dict]]:
    """configs: [(combo_name, pool_size, fuse_fn), ...]. mrr@10 기준 최고 조합을 반환."""
    print(f"\n{'=' * 60}\n{stage_name}\n{'=' * 60}")
    summaries: dict[str, dict] = {}
    for combo_name, pool_size, fuse_fn in configs:
        summary = evaluate_config(pool_size, fuse_fn)
        summaries[combo_name] = summary
        log_result(combo_name, summary)

    result_df = pd.DataFrame(summaries).T
    result_df = result_df.sort_values("mrr@10", ascending=False)
    printed_df = result_df.head(top_n) if top_n else result_df
    print(printed_df[["mrr@10", "recall@5", "ndcg@5", "hit@3", "map@10"]])

    best_name = result_df.index[0]
    print(f"\n>>> {stage_name} 승자: {best_name} (mrr@10={summaries[best_name]['mrr@10']})")
    return best_name, summaries[best_name], summaries


# ============================================================
# 4) 1단계: candidate_pool_size 결정
#    fusion=RRF(k=60), weights=0.5:0.5 고정
# ============================================================
stage1_configs = [
    (
        f"stage1_pool{pool_size}",
        pool_size,
        lambda d, b: rrf_fuse(d, b, k=RRF_K_STANDARD, weights=NEUTRAL_WEIGHTS),
    )
    for pool_size in POOL_SIZES
]
best_pool_name, _, _ = run_stage("1단계: candidate_pool_size 결정", stage1_configs)
best_pool_size = int(best_pool_name.replace("stage1_pool", ""))


# ============================================================
# 5) 2단계: fusion 방식 결정
#    best_pool_size 고정, weights=0.5:0.5 고정
# ============================================================
stage2_configs = [
    (
        "stage2_rrf_k60",
        best_pool_size,
        lambda d, b: rrf_fuse(d, b, k=RRF_K_STANDARD, weights=NEUTRAL_WEIGHTS),
    ),
    (
        "stage2_wsum_minmax",
        best_pool_size,
        lambda d, b: weighted_sum_fuse(d, b, norm_method="minmax", weights=NEUTRAL_WEIGHTS),
    ),
    (
        "stage2_wsum_zscore",
        best_pool_size,
        lambda d, b: weighted_sum_fuse(d, b, norm_method="zscore", weights=NEUTRAL_WEIGHTS),
    ),
]
best_fusion_name, _, _ = run_stage("2단계: fusion 방식 결정", stage2_configs)


# ============================================================
# 6) 3단계: weights(Dense:BM25) 결정
#    best_pool_size + best_fusion 방식 고정
# ============================================================
def make_fuse_fn(fusion_name: str, weights: tuple[float, float]):
    if fusion_name == "stage2_rrf_k60":
        return lambda d, b: rrf_fuse(d, b, k=RRF_K_STANDARD, weights=weights)
    if fusion_name == "stage2_wsum_minmax":
        return lambda d, b: weighted_sum_fuse(d, b, norm_method="minmax", weights=weights)
    if fusion_name == "stage2_wsum_zscore":
        return lambda d, b: weighted_sum_fuse(d, b, norm_method="zscore", weights=weights)
    raise ValueError(fusion_name)


stage3_configs = [
    (
        f"stage3_w{int(w[0]*10)}{int(w[1]*10)}",
        best_pool_size,
        make_fuse_fn(best_fusion_name, w),
    )
    for w in WEIGHT_OPTIONS
]
best_weight_name, best_weight_summary, _ = run_stage("3단계: weights(Dense:BM25) 결정", stage3_configs)


# ============================================================
# 7) 4단계: pool_size × weights 정밀(5단위) 그리드
#    best_fusion_name 고정, pool_size ∈ {5,10,...,50}, Dense 비율 ∈ {50%,55%,...,95%}
# ============================================================
FINE_POOL_SIZES = list(range(5, 51, 5))
FINE_DENSE_PCTS = list(range(50, 100, 5))
FINE_WEIGHT_OPTIONS = [(pct / 100, (100 - pct) / 100) for pct in FINE_DENSE_PCTS]

def _stage4_combo_name(pool_size: int, dense_pct: int) -> str:
    return f"stage4_pool{pool_size}_w{dense_pct:02d}{100 - dense_pct:02d}"


stage4_configs = [
    (
        _stage4_combo_name(pool_size, pct),
        pool_size,
        make_fuse_fn(best_fusion_name, weights),
    )
    for pool_size in FINE_POOL_SIZES
    for pct, weights in zip(FINE_DENSE_PCTS, FINE_WEIGHT_OPTIONS)
]
best_fine_name, best_fine_summary, stage4_summary_by_name = run_stage(
    "4단계: pool_size × weights 정밀(5단위) 그리드",
    stage4_configs,
    top_n=15,
)

# pool_size(행) x Dense비율(열) 히트맵 형태로 mrr@10 전체를 한 번에 확인
heatmap_rows = []
for pool_size in FINE_POOL_SIZES:
    row = {"pool_size": pool_size}
    for pct in FINE_DENSE_PCTS:
        row[f"D{pct}"] = stage4_summary_by_name[_stage4_combo_name(pool_size, pct)]["mrr@10"]
    heatmap_rows.append(row)

heatmap_df = pd.DataFrame(heatmap_rows).set_index("pool_size")
print("\n--- mrr@10 히트맵 (행=pool_size, 열=Dense 비율%) ---")
print(heatmap_df)

# 파싱: stage4 승자에서 pool_size, weights 복원
_pool_part, _w_part = best_fine_name.replace("stage4_pool", "").split("_w")
fine_best_pool_size = int(_pool_part)
fine_dense_pct = int(_w_part[:2])
fine_bm25_pct = 100 - fine_dense_pct
fine_best_weights = (fine_dense_pct / 100, fine_bm25_pct / 100)


# ============================================================
# 8) 최종 확정값 요약 (3단계 결과 vs 4단계 정밀 그리드 결과 비교)
# ============================================================
coarse_weights = WEIGHT_OPTIONS[
    [f"stage3_w{int(w[0]*10)}{int(w[1]*10)}" for w in WEIGHT_OPTIONS].index(best_weight_name)
]

if best_fine_summary["mrr@10"] >= best_weight_summary["mrr@10"]:
    final_pool_size = fine_best_pool_size
    final_weights = fine_best_weights
    final_summary = best_fine_summary
    final_source = "4단계 정밀 그리드"
else:
    final_pool_size = best_pool_size
    final_weights = coarse_weights
    final_summary = best_weight_summary
    final_source = "3단계 거친 스윕"

print(f"\n{'=' * 60}\n최종 확정값 ({final_source} 기준)\n{'=' * 60}")
print("candidate_pool_size:", final_pool_size)
print("fusion 방식:", best_fusion_name)
print("weights (Dense:BM25):", final_weights)
print("성능:", json.dumps(final_summary, ensure_ascii=False, indent=2))
