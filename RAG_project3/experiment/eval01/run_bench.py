# [Experiment/eval01] Evaluation_DataSet01.xlsx 기준으로 "질의분석(decompose_query)을
# 거친 뒤 검색"까지 파이프라인 그대로 태워서 5가지 검색기법을 비교합니다.
#   1. dense v2 structured 단독 (title+content 임베딩)
#   2. hybrid 7:3 RRF   (dense v2 structured + BM25-Nori, rrf k=10)
#   3. hybrid 7:3 MinMax
#   4. hybrid 9:1 RRF
#   5. hybrid 9:1 MinMax
# (dense/bm25 후보 풀 depth=20)
#
# 기존 코드는 전혀 수정하지 않고, 필요한 순수 함수만 이 experiment/ 폴더 안으로
# 옮겨와서 조립했습니다 (dense_structured_search.py, bm25_nori_search.py, fusion.py).
#
# 하위질문 병합 규칙: decompose_query()가 하위질문을 여러 개로 쪼갠 경우(전체 문항의
# 소수), 하위질문별로 검색한 뒤 같은 청크는 최고 점수로 병합하고 점수 내림차순으로
# 최종 랭킹을 만듭니다. 단일질의(하위질문 1개, 대다수)는 사실상 그 결과 그대로입니다.
#
# latency는 "질의분석 넘어가서 검색까지"의 종단 시간입니다: decompose_query 호출
# 1회(방법 5개가 공유하는 단계라 한 번만 측정해 그대로 더함) + 하위질문마다
# dense/bm25 후보 풀 조회 시간의 합. 융합(RRF/MinMax) 자체는 이미 가져온 후보
# 리스트에 대한 순수 연산이라 무시할 수준(<1ms)이라 별도로 재지 않습니다.
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd

from classification.decomposition import decompose_query
from evaluation.eval_search import (
    load_gold_set,
    hit_at_k,
    recall_at_k,
    precision_at_k,
    mrr_at_k,
    average_precision_at_k,
    f1_at_k,
)

from experiment.eval01.dense_structured_search import dense_structured_search
from experiment.eval01.bm25_nori_search import bm25_nori_search
from experiment.eval01.fusion import rrf_fuse, weighted_sum_fuse_minmax

GOLD_PATH = Path("data/Evaluation_DataSet01.xlsx")
GOLD_SHEET = "평가데이터셋 V4_1"

DEPTH = 20    # dense/bm25 후보 풀 크기
RRF_K = 10    # RRF 상수(사용자 지정값 — 기존 튜닝 확정값 60이 아님)
FINAL_K = 10  # 병합 후 최종 랭킹 길이(hit@3/recall@5 계산에 충분한 여유)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

METHODS = [
    ("dense_v2_structured", False, None),
    ("hybrid_rrf_7_3", True, lambda d, b: rrf_fuse(d, b, k=RRF_K, weights=(0.7, 0.3))),
    ("hybrid_minmax_7_3", True, lambda d, b: weighted_sum_fuse_minmax(d, b, weights=(0.7, 0.3))),
    ("hybrid_rrf_9_1", True, lambda d, b: rrf_fuse(d, b, k=RRF_K, weights=(0.9, 0.1))),
    ("hybrid_minmax_9_1", True, lambda d, b: weighted_sum_fuse_minmax(d, b, weights=(0.9, 0.1))),
]


def build_subq_cache(gold_records: list[dict], *, limit: int | None = None) -> dict:
    """문항마다 decompose_query 1회 + 하위질문마다 dense/bm25 후보 풀을 한 번만
    가져와서 캐싱합니다. 5가지 방법이 이 캐시를 그대로 재사용하므로 API 호출이
    문항 수 x 방법 수가 아니라 문항 수 x 하위질문 수 만큼만 발생합니다."""
    records = gold_records[:limit] if limit else gold_records
    cache: dict[str, dict] = {}

    for i, record in enumerate(records, 1):
        eid = record["evaluation_id"]

        t0 = time.perf_counter()
        subqs = decompose_query(record["question"])
        decompose_latency = time.perf_counter() - t0

        per_subq = {}
        for subq in subqs:
            t0 = time.perf_counter()
            dense_pool = dense_structured_search(subq, top_k=DEPTH, business_function=None)
            dense_latency = time.perf_counter() - t0

            t0 = time.perf_counter()
            bm25_pool = bm25_nori_search(subq, top_k=DEPTH, business_function=None)
            bm25_latency = time.perf_counter() - t0

            per_subq[subq] = {
                "dense_pool": dense_pool,
                "bm25_pool": bm25_pool,
                "dense_latency": dense_latency,
                "bm25_latency": bm25_latency,
            }

        cache[eid] = {
            "subqs": subqs,
            "decompose_latency": decompose_latency,
            "per_subq": per_subq,
        }

        if i % 20 == 0:
            print(f"  캐시 구축: {i}/{len(records)}")

    print(f"캐시 구축 완료: {len(records)}문항")
    return cache


def _merge_subquery_results(per_subq_ranked: list[list[dict]], final_k: int) -> list[str]:
    best_score: dict[str, float] = {}
    for ranked in per_subq_ranked:
        for item in ranked:
            cid = item["chunk"]["chunk_id"]
            if cid not in best_score or item["score"] > best_score[cid]:
                best_score[cid] = item["score"]
    ordered = sorted(best_score.items(), key=lambda kv: kv[1], reverse=True)
    return [cid for cid, _ in ordered[:final_k]]


def run_method(
    name: str,
    uses_bm25: bool,
    fuse_fn,
    gold_records: list[dict],
    cache: dict,
) -> tuple[dict, pd.DataFrame]:
    rows = []
    for record in gold_records:
        eid = record["evaluation_id"]
        entry = cache[eid]

        per_subq_ranked = []
        latency = entry["decompose_latency"]
        for subq in entry["subqs"]:
            pools = entry["per_subq"][subq]
            latency += pools["dense_latency"]
            if uses_bm25:
                latency += pools["bm25_latency"]
                ranked = fuse_fn(pools["dense_pool"], pools["bm25_pool"])
            else:
                ranked = pools["dense_pool"]
            per_subq_ranked.append(ranked)

        ranked_ids = _merge_subquery_results(per_subq_ranked, FINAL_K)
        gold_ids = record["gold_chunk_ids"]

        precision5 = precision_at_k(ranked_ids, gold_ids, 5)
        recall5 = recall_at_k(ranked_ids, gold_ids, 5)
        rows.append({
            "evaluation_id": eid,
            "n_sub_questions": len(entry["subqs"]),
            "latency_sec": latency,
            "hit@3": hit_at_k(ranked_ids, gold_ids, 3),
            "recall@5": recall5,
            "precision@5": precision5,
            "f1@5": f1_at_k(precision5, recall5),
            "mrr@10": mrr_at_k(ranked_ids, gold_ids, 10),
            "map@10": average_precision_at_k(ranked_ids, gold_ids, 10),
        })

    detail_df = pd.DataFrame(rows)
    summary = {"method": name, "n_questions": len(detail_df)}
    for metric in ("latency_sec", "hit@3", "recall@5", "precision@5", "f1@5", "mrr@10", "map@10"):
        summary[metric] = round(float(detail_df[metric].mean()), 4)

    return summary, detail_df


def evaluate(*, limit: int | None = None) -> pd.DataFrame:
    gold_records = load_gold_set(path=GOLD_PATH, sheet_name=GOLD_SHEET)
    print(f"평가 문항 수: {len(gold_records)}" + (f" (상위 {limit}건만 시험 실행)" if limit else ""))

    cache = build_subq_cache(gold_records, limit=limit)
    records = gold_records[:limit] if limit else gold_records

    summaries = []
    for name, uses_bm25, fuse_fn in METHODS:
        summary, detail_df = run_method(name, uses_bm25, fuse_fn, records, cache)
        summaries.append(summary)
        detail_path = RESULTS_DIR / f"detail_{name}.csv"
        detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
        print(f"\n=== {name} ===")
        print(summary)
        print("문항별 상세 저장:", detail_path)

    summary_df = pd.DataFrame(summaries)
    summary_path = RESULTS_DIR / "summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print("\n요약 저장:", summary_path)
    print(summary_df.to_string(index=False))
    return summary_df


if __name__ == "__main__":
    evaluate()
