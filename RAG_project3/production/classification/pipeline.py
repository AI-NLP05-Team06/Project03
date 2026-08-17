# [8-9/9] Phase1 전체 cascade: DIRECT_RESPONSE 게이트 -> 업무(rule->임베딩->LLM) ->
# (RETRIEVE면) 의도(rule->임베딩->LLM). 최종적으로 5번(업무+의도 통합분류)의 단일 진입점.
from __future__ import annotations

from classification.embedding_tier import (
    classify_business_function_embedding,
    classify_intent_embedding,
)
from classification.llm_tier import classify_business_and_intent_llm, classify_intent_llm
from classification.rules import (
    classify_business_function_rule,
    classify_intent_rule,
    is_direct_response,
)


def classify(question: str) -> dict:
    if is_direct_response(question):
        return {"route": "DIRECT_RESPONSE", "business_function": None, "intent": None}

    business = classify_business_function_rule(question)
    tier = "rule"
    if business is None:
        business, _, _ = classify_business_function_embedding(question)
        tier = "embedding"

    if business is None:
        llm_result = classify_business_and_intent_llm(question)
        business = llm_result["business_function"]
        tier = "llm"

        if business == "OUT_OF_SCOPE":
            return {"route": "OUT_OF_SCOPE", "business_function": None, "intent": None}
        if business == "CLARIFY":
            return {"route": "CLARIFY", "business_function": None, "intent": None}

        # LLM이 업무까지 특정한 경우, 의도도 같은 호출에서 이미 나왔으면 그대로 쓴다.
        intent = llm_result.get("intent")
        if intent in ("정보", "민원처리"):
            return {"route": "RETRIEVE", "business_function": business, "intent": intent, "tier": tier}

    # 업무는 정해졌는데 의도가 아직 없는 경우 (rule/임베딩으로 업무만 풀렸거나,
    # LLM이 업무는 줬는데 의도를 안 준 경우) -> 의도만 rule -> 임베딩 -> LLM으로 마저 판단
    intent = classify_intent_rule(question)
    if intent is None:
        intent, _, _ = classify_intent_embedding(question)
    if intent is None:
        intent = classify_intent_llm(question)

    return {"route": "RETRIEVE", "business_function": business, "intent": intent, "tier": tier}
