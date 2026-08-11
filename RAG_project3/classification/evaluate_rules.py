# [6/6] rule 티어만으로 270건 전체를 돌려서 커버리지·정확도를 측정한다.
from __future__ import annotations

from classification.data_prep import load_routing_eval_set
from classification.rules import (
    classify_business_function_rule,
    classify_intent_rule,
    is_direct_response,
    needs_clarification_rule,
)


def evaluate() -> None:
    records = load_routing_eval_set()

    # --- 1) DIRECT_RESPONSE 게이트 (파이프라인에서 제일 먼저 도는 체크) ---
    tp = fp = fn = tn = 0
    for r in records:
        pred = is_direct_response(r["question"])
        gold = r["route_type"] == "DIRECT_RESPONSE"
        if pred and gold:
            tp += 1
        elif pred and not gold:
            fp += 1
        elif not pred and gold:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    print("=== 1) DIRECT_RESPONSE 게이트 ===")
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}  Precision={precision:.2f} Recall={recall:.2f}")
    if fp:
        print("  [오탐(FP)] 진짜 업무질문인데 인사말로 오분류:")
        for r in records:
            if is_direct_response(r["question"]) and r["route_type"] != "DIRECT_RESPONSE":
                print("   -", r["question"], f"[{r['route_type']}]")

    # DIRECT_RESPONSE로 걸러진 건 이후 단계로 안 넘어간다고 가정하고 제외
    remaining = [r for r in records if not is_direct_response(r["question"])]

    # --- 2) 업무(도메인) 분류 ---
    # gold: business_functions가 비어있지 않은 문항만 평가 대상 (OUT_OF_SCOPE·
    # business_function이 missing_slot인 CLARIFY는 애초에 정답 업무가 없음)
    bf_targets = [r for r in remaining if r["business_functions"]]
    bf_correct = bf_covered = 0
    for r in bf_targets:
        pred = classify_business_function_rule(r["question"])
        if pred is not None:
            bf_covered += 1
            if pred in r["business_functions"]:
                bf_correct += 1
    print()
    print("=== 2) 업무(도메인) 분류 ===")
    print(f"평가 대상 {len(bf_targets)}건 중 rule 매칭(커버리지) {bf_covered}건 "
          f"({bf_covered/len(bf_targets):.1%})")
    if bf_covered:
        print(f"매칭된 것 중 정답률 {bf_correct}/{bf_covered} ({bf_correct/bf_covered:.1%})")

    # --- 3) 의도(정보/민원처리) 분류: RETRIEVE(=gold_intent 있는) 문항만 ---
    intent_targets = [r for r in remaining if r["gold_intent"] is not None]
    intent_correct = intent_covered = 0
    for r in intent_targets:
        pred = classify_intent_rule(r["question"])
        if pred is not None:
            intent_covered += 1
            if pred == r["gold_intent"]:
                intent_correct += 1
    print()
    print("=== 3) 의도(정보/민원처리) 분류 ===")
    print(f"평가 대상 {len(intent_targets)}건 중 rule 매칭(커버리지) {intent_covered}건 "
          f"({intent_covered/len(intent_targets):.1%})")
    if intent_covered:
        print(f"매칭된 것 중 정답률 {intent_correct}/{intent_covered} ({intent_correct/intent_covered:.1%})")

    # --- 4) 되묻기(CLARIFY) 규칙 ---
    tp = fp = fn = tn = 0
    for r in remaining:
        pred = needs_clarification_rule(r["question"])
        gold = r["should_clarify"]
        if pred and gold:
            tp += 1
        elif pred and not gold:
            fp += 1
        elif not pred and gold:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    print()
    print("=== 4) 되묻기(CLARIFY, 업무 불명확 조건 하나) ===")
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}  Precision={precision:.2f} Recall={recall:.2f}")
    if fp:
        by_route = {}
        for r in remaining:
            if needs_clarification_rule(r["question"]) and not r["should_clarify"]:
                by_route[r["route_type"]] = by_route.get(r["route_type"], 0) + 1
        print("  [오탐(FP) route_type 분포]", by_route)


if __name__ == "__main__":
    evaluate()
