# [7/9] rule -> embedding 2단계 누적 정확도를 측정한다. (LLM 티어 붙기 전 중간 점검용)
from __future__ import annotations

from classification.data_prep import load_routing_eval_set
from classification.embedding_tier import (
    classify_business_function_embedding,
    classify_intent_embedding,
)
from classification.rules import classify_business_function_rule, classify_intent_rule


def evaluate() -> None:
    records = load_routing_eval_set()

    # --- 업무(도메인) ---
    bf_targets = [r for r in records if r["business_functions"]]
    bf_covered = bf_correct = 0
    bf_still_unresolved = 0
    for r in bf_targets:
        pred = classify_business_function_rule(r["question"])
        if pred is None:
            pred, _, _ = classify_business_function_embedding(r["question"])
        if pred is not None:
            bf_covered += 1
            if pred in r["business_functions"]:
                bf_correct += 1
        else:
            bf_still_unresolved += 1

    print("=== 업무(도메인) 분류: rule -> embedding 누적 ===")
    print(f"평가 대상 {len(bf_targets)}건 중 커버리지 {bf_covered}건 ({bf_covered/len(bf_targets):.1%}), "
          f"정답률 {bf_correct}/{bf_covered} ({bf_correct/bf_covered:.1%})")
    print(f"여전히 미해결(LLM 티어로 넘길 것) {bf_still_unresolved}건")

    # --- 의도(정보/민원처리) ---
    intent_targets = [r for r in records if r["gold_intent"] is not None]
    intent_covered = intent_correct = 0
    intent_still_unresolved = 0
    for r in intent_targets:
        pred = classify_intent_rule(r["question"])
        if pred is None:
            pred, _, _ = classify_intent_embedding(r["question"])
        if pred is not None:
            intent_covered += 1
            if pred == r["gold_intent"]:
                intent_correct += 1
        else:
            intent_still_unresolved += 1

    print()
    print("=== 의도(정보/민원처리) 분류: rule -> embedding 누적 ===")
    print(f"평가 대상 {len(intent_targets)}건 중 커버리지 {intent_covered}건 ({intent_covered/len(intent_targets):.1%}), "
          f"정답률 {intent_correct}/{intent_covered} ({intent_correct/intent_covered:.1%})")
    print(f"여전히 미해결(LLM 티어로 넘길 것) {intent_still_unresolved}건")


if __name__ == "__main__":
    evaluate()
