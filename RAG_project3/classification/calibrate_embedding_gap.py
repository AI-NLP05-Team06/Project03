# [7/9] rule 티어가 놓친 문항만 대상으로 임베딩 gap과 정답 여부의 상관관계를 봐서
# gap_threshold를 데이터 기반으로 잡는다. (evaluation/calibrate_gap_threshold.py와 동일한 방식)
from __future__ import annotations

import pandas as pd

from classification.data_prep import load_routing_eval_set
from classification.embedding_tier import _rank_categories, _BUSINESS_FUNCTION_VECTORS, _INTENT_VECTORS
from classification.rules import classify_business_function_rule, classify_intent_rule


def calibrate_business_function() -> None:
    records = load_routing_eval_set()
    targets = [
        r for r in records
        if r["business_functions"] and classify_business_function_rule(r["question"]) is None
    ]

    rows = []
    for r in targets:
        ranked = _rank_categories(r["question"], _BUSINESS_FUNCTION_VECTORS)
        top1_category, top1_score = ranked[0]
        gap = top1_score - ranked[1][1]
        rows.append({
            "evaluation_id": r["evaluation_id"],
            "correct": top1_category in r["business_functions"],
            "top1_score": top1_score,
            "gap": gap,
        })

    df = pd.DataFrame(rows)
    print(f"=== 업무 분류: rule 미해결 {len(df)}건 ===")
    print(df.groupby("correct")["gap"].describe())
    bins = [0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 1.0]
    df["gap_bin"] = pd.cut(df["gap"], bins=bins)
    print(df.groupby("gap_bin", observed=True)["correct"].agg(["mean", "count"]))
    print()


def calibrate_intent() -> None:
    records = load_routing_eval_set()
    targets = [
        r for r in records
        if r["gold_intent"] is not None and classify_intent_rule(r["question"]) is None
    ]

    rows = []
    for r in targets:
        ranked = _rank_categories(r["question"], _INTENT_VECTORS)
        top1_category, top1_score = ranked[0]
        gap = top1_score - ranked[1][1]
        rows.append({
            "evaluation_id": r["evaluation_id"],
            "correct": top1_category == r["gold_intent"],
            "top1_score": top1_score,
            "gap": gap,
        })

    df = pd.DataFrame(rows)
    print(f"=== 의도 분류: rule 미해결 {len(df)}건 ===")
    print(df.groupby("correct")["gap"].describe())
    bins = [0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 1.0]
    df["gap_bin"] = pd.cut(df["gap"], bins=bins)
    print(df.groupby("gap_bin", observed=True)["correct"].agg(["mean", "count"]))


if __name__ == "__main__":
    calibrate_business_function()
    calibrate_intent()
