# [Phase6] Multi-Query/HyDE가 reranker 이전 raw recall@k를 baseline 대비
# 얼마나 올리는지 측정한다. (세션2 결정: reranker는 이미 있는 후보를 재정렬만
# 하므로, 확장의 진짜 효과는 reranker 전 pool 단계에서 봐야 함)
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import random

from core.config import *
from core.integrity_check import *
from retrieval.hybrid_search import hybrid_search
from retrieval.query_expansion import (
    generate_hyde_passage,
    generate_multi_queries,
    reciprocal_rank_fusion,
)
from evaluation.eval_search import load_gold_set, recall_at_k, hit_at_k

SAMPLE_PER_DOMAIN = 5  # 골드셋 전체(120건)는 비용이 크니 우선 표본으로
RANDOM_SEED = 42
TOP_K = 5
POOL_K = 20  # RRF 병합 전, variant별로 넉넉히 뽑아둘 pool 크기


def _ranked_ids(question: str, top_k: int = POOL_K) -> list[str]:
    results = hybrid_search(question, top_k=top_k, business_function=None)
    return [r["chunk"]["chunk_id"] for r in results]


def evaluate_baseline(question: str) -> list[str]:
    return _ranked_ids(question)


def evaluate_multi_query(question: str) -> list[str]:
    variants = generate_multi_queries(question)
    ranked_lists = [_ranked_ids(v) for v in variants]
    return reciprocal_rank_fusion(ranked_lists)


def evaluate_hyde(question: str) -> list[str]:
    passage = generate_hyde_passage(question)
    return _ranked_ids(passage)


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

    rows = []
    for i, record in enumerate(sampled_records, 1):
        question = record["question"]
        gold_ids = record["gold_chunk_ids"]

        baseline_ids = evaluate_baseline(question)
        multi_query_ids = evaluate_multi_query(question)
        hyde_ids = evaluate_hyde(question)

        rows.append({
            "evaluation_id": record["evaluation_id"],
            "baseline_recall@5": recall_at_k(baseline_ids, gold_ids, TOP_K),
            "multi_query_recall@5": recall_at_k(multi_query_ids, gold_ids, TOP_K),
            "hyde_recall@5": recall_at_k(hyde_ids, gold_ids, TOP_K),
            "baseline_hit@5": hit_at_k(baseline_ids, gold_ids, TOP_K),
            "multi_query_hit@5": hit_at_k(multi_query_ids, gold_ids, TOP_K),
            "hyde_hit@5": hit_at_k(hyde_ids, gold_ids, TOP_K),
        })
        print(f"  {i}/{len(sampled_records)} | baseline={rows[-1]['baseline_recall@5']:.2f} "
              f"multi_query={rows[-1]['multi_query_recall@5']:.2f} hyde={rows[-1]['hyde_recall@5']:.2f}")

    df = pd.DataFrame(rows)
    print("\n=== 평균 recall@5 / hit@5 ===")
    print(df[[c for c in df.columns if c != "evaluation_id"]].mean())

    out_path = EXPERIMENTS_ROOT / "query_expansion_eval.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n상세 저장: {out_path}")


if __name__ == "__main__":
    main()
