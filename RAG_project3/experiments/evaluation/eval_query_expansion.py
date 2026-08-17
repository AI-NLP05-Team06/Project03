# [Phase6] Multi-Query/HyDE가 reranker 이전 raw recall@k를 baseline 대비
# 얼마나 올리는지 측정한다. (세션2 결정: reranker는 이미 있는 후보를 재정렬만
# 하므로, 확장의 진짜 효과는 reranker 전 pool 단계에서 봐야 함)
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "production"))

import random

from core.config import *
from core.integrity_check import *
from retrieval.hybrid_search import hybrid_search
from retrieval.query_expansion import (
    generate_hyde_passage,
    generate_multi_queries,
    generate_stepback_query,
    score_sum_fusion,
)
from experiments.evaluation.eval_search import load_gold_set, recall_at_k, hit_at_k

SAMPLE_PER_DOMAIN = 5  # 골드셋 전체(120건)는 비용이 크니 우선 표본으로
RANDOM_SEED = 42
TOP_K = 5
POOL_K = 20  # RRF 병합 전, variant별로 넉넉히 뽑아둘 pool 크기


def _search(question: str, top_k: int = POOL_K) -> list[dict]:
    return hybrid_search(question, top_k=top_k, business_function=None)


def _ranked_ids(question: str, top_k: int = POOL_K) -> list[str]:
    return [r["chunk"]["chunk_id"] for r in _search(question, top_k)]


def evaluate_baseline(question: str) -> list[str]:
    return _ranked_ids(question)


def evaluate_multi_query(question: str) -> list[str]:
    variants = generate_multi_queries(question)
    results_lists = [_search(v) for v in variants]
    return score_sum_fusion(results_lists)


def evaluate_hyde(question: str) -> list[str]:
    passage = generate_hyde_passage(question)
    return _ranked_ids(passage)


def evaluate_stepback(question: str) -> list[str]:
    stepback_query = generate_stepback_query(question)
    return _ranked_ids(stepback_query)


def main() -> None:
    random.seed(RANDOM_SEED)
    gold_records = load_gold_set()

    by_domain: dict[str, list[dict]] = {}
    for record in gold_records:
        by_domain.setdefault(record["business_function"], []).append(record)

    sampled_records: list[dict] = []
    for domain, records in by_domain.items():
        sampled_records.extend(
            random.sample(records, min(SAMPLE_PER_DOMAIN, len(records)))
        )
    print(f"평가 문항 수: {len(sampled_records)} (도메인 {len(by_domain)}개 x 최대 {SAMPLE_PER_DOMAIN}개)")

    out_path = EXPERIMENTS_ROOT / "query_expansion_eval_stepback.csv"
    rows = []
    done_ids: set = set()
    if out_path.exists():
        existing_df = pd.read_csv(out_path)
        rows = existing_df.to_dict("records")
        done_ids = set(existing_df["evaluation_id"])
        print(f"이어서 진행: 이미 완료된 문항 {len(done_ids)}건 건너뜀")

    for i, record in enumerate(sampled_records, 1):
        if record["evaluation_id"] in done_ids:
            continue
        question = record["question"]
        gold_ids = record["gold_chunk_ids"]

        baseline_ids = evaluate_baseline(question)
        stepback_ids = evaluate_stepback(question)

        rows.append({
            "evaluation_id": record["evaluation_id"],
            "baseline_recall@5": recall_at_k(baseline_ids, gold_ids, TOP_K),
            "stepback_recall@5": recall_at_k(stepback_ids, gold_ids, TOP_K),
            "baseline_hit@5": hit_at_k(baseline_ids, gold_ids, TOP_K),
            "stepback_hit@5": hit_at_k(stepback_ids, gold_ids, TOP_K),
        })
        print(f"  {i}/{len(sampled_records)} | baseline={rows[-1]['baseline_recall@5']:.2f} "
              f"stepback={rows[-1]['stepback_recall@5']:.2f}")
        pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")

    df = pd.DataFrame(rows)
    print("\n=== 평균 recall@5 / hit@5 ===")
    print(df[[c for c in df.columns if c != "evaluation_id"]].mean())
    print(f"\n상세 저장: {out_path}")


if __name__ == "__main__":
    main()
