# [9/9] rule -> embedding -> LLM 전체 cascade를 270건에 돌려서 최종 라우팅·업무·의도 정확도를 잰다.
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from classification.data_prep import load_routing_eval_set
from classification.pipeline import classify

DETAIL_PATH = Path("data/evaluate_pipeline_detail.csv")


def _gold_route(r: dict) -> str:
    if r["route_type"] == "DIRECT_RESPONSE":
        return "DIRECT_RESPONSE"
    if r["route_type"] == "OUT_OF_SCOPE":
        return "OUT_OF_SCOPE"
    if r["should_clarify"]:
        return "CLARIFY"
    return "RETRIEVE"  # RETRIEVE + business는 알지만 다른 슬롯이 빠진 CLARIFY(스코프 밖) 모두 포함


def evaluate() -> None:
    records = load_routing_eval_set()

    route_correct = 0
    confusion: Counter[tuple[str, str]] = Counter()
    bf_targets = bf_correct = 0
    intent_targets = intent_correct = 0
    detail_rows = []

    for i, r in enumerate(records, 1):
        gold_route = _gold_route(r)
        result = classify(r["question"])
        pred_route = result["route"]

        confusion[(gold_route, pred_route)] += 1
        if pred_route == gold_route:
            route_correct += 1

        is_bf_target = gold_route == "RETRIEVE" and pred_route == "RETRIEVE" and bool(r["business_functions"])
        bf_correct_flag = None
        intent_correct_flag = None
        if is_bf_target:
            bf_targets += 1
            bf_correct_flag = result["business_function"] in r["business_functions"]
            bf_correct += bf_correct_flag
            if r["gold_intent"] is not None:
                intent_targets += 1
                intent_correct_flag = result["intent"] == r["gold_intent"]
                intent_correct += intent_correct_flag

        detail_rows.append({
            "evaluation_id": r["evaluation_id"],
            "question": r["question"],
            "gold_route": gold_route,
            "pred_route": pred_route,
            "gold_business_functions": r["business_functions"],
            "pred_business_function": result.get("business_function"),
            "bf_correct": bf_correct_flag,
            "gold_intent": r["gold_intent"],
            "pred_intent": result.get("intent"),
            "intent_correct": intent_correct_flag,
        })

        if i % 10 == 0:
            pd.DataFrame(detail_rows).to_csv(DETAIL_PATH, index=False, encoding="utf-8-sig")
        if i % 50 == 0:
            print(f"  진행: {i}/{len(records)}")

    pd.DataFrame(detail_rows).to_csv(DETAIL_PATH, index=False, encoding="utf-8-sig")
    print(f"\n문항별 상세 저장: {DETAIL_PATH}")

    print()
    print(f"=== 최종 라우팅 정확도: {route_correct}/{len(records)} ({route_correct/len(records):.1%}) ===")
    print("gold_route -> pred_route 컨퓨전 매트릭스 (틀린 것만):")
    for (gold, pred), count in sorted(confusion.items()):
        if gold != pred:
            print(f"  {gold} -> {pred}: {count}건")

    print()
    print(f"=== 업무(도메인) 최종 정확도 (RETRIEVE로 맞게 라우팅된 것 중): "
          f"{bf_correct}/{bf_targets} ({bf_correct/bf_targets:.1%}) ===")
    print(f"=== 의도 최종 정확도: {intent_correct}/{intent_targets} ({intent_correct/intent_targets:.1%}) ===")


if __name__ == "__main__":
    evaluate()
