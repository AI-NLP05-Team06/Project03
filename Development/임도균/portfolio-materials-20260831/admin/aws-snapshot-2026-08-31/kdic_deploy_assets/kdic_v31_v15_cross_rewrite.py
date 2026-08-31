from __future__ import annotations

import copy
import math
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol, Sequence


PIPELINE_VERSION = "KDIC_V31_ANALYSIS_V15_CROSS_REWRITE_2026_08_18"
NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d+)?")
TOKEN_PATTERN = re.compile(r"[가-힣A-Za-z0-9]+")
NEGATION_TERMS = ("아닌", "아니", "않", "못", "제외", "불가", "없이", "없", "미해당")


class V31AnalyzerProtocol(Protocol):
    def run(
        self,
        query: str,
        *,
        conversation_state: Mapping[str, Any] | None = None,
        manual_selection: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class CrossRewritePolicy:
    original_weight: float = 0.40
    rewritten_total_weight: float = 0.60
    max_rewritten_queries: int = 4
    minimum_business_confidence: float = 0.70
    minimum_token_overlap: float = 0.25
    allow_soft_business_hint: bool = True

    def __post_init__(self) -> None:
        if not math.isclose(
            self.original_weight + self.rewritten_total_weight,
            1.0,
            abs_tol=1e-9,
        ):
            raise ValueError("원문과 재작성 질의의 가중치 합은 1이어야 합니다.")
        if self.max_rewritten_queries < 2:
            raise ValueError("교차업무 재작성 최대 개수는 2 이상이어야 합니다.")
        if not 0.0 <= self.minimum_business_confidence <= 1.0:
            raise ValueError("minimum_business_confidence는 0~1이어야 합니다.")
        if not 0.0 <= self.minimum_token_overlap <= 1.0:
            raise ValueError("minimum_token_overlap은 0~1이어야 합니다.")


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _compact(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).lower()


def _ordered_unique(values: Sequence[Any]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_text(value)
        key = _compact(text)
        if text and key not in seen:
            seen.add(key)
            output.append(text)
    return output


def _normalize_route(value: Any) -> str:
    text = str(value or "").strip().upper()
    mapping = {
        "RETRIEVE": "RETRIEVE",
        "RETRIEVE_RELAXED": "RETRIEVE",
        "CLARIFY": "CLARIFY",
        "DIRECT": "DIRECT_RESPONSE",
        "DIRECT_RESPONSE": "DIRECT_RESPONSE",
        "OUT_OF_SCOPE": "OUT_OF_SCOPE",
        "OOS": "OUT_OF_SCOPE",
    }
    return mapping.get(text, text)


def _businesses_from_needs(needs: Sequence[Mapping[str, Any]]) -> list[str]:
    return _ordered_unique(
        [need.get("business_function") for need in needs if need.get("business_function")]
    )


def _explicit_businesses(v31_module: Any, original: str) -> list[str]:
    finder = getattr(v31_module, "find_businesses", None)
    if not callable(finder):
        return []
    return _ordered_unique(list(finder(original) or []))


def detect_cross_business(
    *,
    original: str,
    needs: Sequence[Mapping[str, Any]],
    v31_module: Any,
) -> dict[str, Any]:
    explicit_businesses = _explicit_businesses(v31_module, original)
    need_businesses = _businesses_from_needs(needs)
    if len(explicit_businesses) >= 2:
        businesses = explicit_businesses
        source = "EXPLICIT_ORIGINAL_TERMS"
    elif len(need_businesses) >= 2:
        businesses = need_businesses
        source = "V31_STRUCTURED_NEEDS"
    else:
        businesses = _ordered_unique(explicit_businesses + need_businesses)
        source = "SINGLE_OR_UNKNOWN_BUSINESS"
    return {
        "is_cross_business": len(businesses) >= 2,
        "businesses": businesses,
        "explicit_businesses": explicit_businesses,
        "need_businesses": need_businesses,
        "evidence_source": source,
    }


def _token_overlap(original: str, rewritten: str) -> float:
    original_tokens = set(TOKEN_PATTERN.findall(original.lower()))
    rewritten_tokens = set(TOKEN_PATTERN.findall(rewritten.lower()))
    if not original_tokens:
        return 1.0
    return len(original_tokens & rewritten_tokens) / len(original_tokens)


def validate_cross_rewrite(
    *,
    original: str,
    needs: Sequence[Mapping[str, Any]],
    expected_businesses: Sequence[str],
    policy: CrossRewritePolicy,
) -> dict[str, Any]:
    issues: list[str] = []
    accepted_needs: list[dict[str, Any]] = []
    seen_queries: set[str] = set()

    for raw_need in needs[: policy.max_rewritten_queries]:
        need = copy.deepcopy(dict(raw_need))
        query = _clean_text(need.get("query"))
        business = _clean_text(need.get("business_function"))
        confidence = float(need.get("business_confidence") or 0.0)
        if not query:
            issues.append("EMPTY_REWRITTEN_QUERY")
            continue
        key = _compact(query)
        if key in seen_queries:
            issues.append("DUPLICATE_REWRITTEN_QUERY")
            continue
        seen_queries.add(key)
        if len(query) < 5:
            issues.append("REWRITTEN_QUERY_TOO_SHORT")
        if key == _compact(original):
            issues.append("REWRITTEN_QUERY_EQUALS_ORIGINAL")
        if not business:
            issues.append("REWRITTEN_QUERY_WITHOUT_BUSINESS")
        if confidence < policy.minimum_business_confidence:
            issues.append("LOW_BUSINESS_CONFIDENCE")
        need["query"] = query
        need["business_function"] = business or None
        need["business_confidence"] = confidence
        need["token_overlap"] = _token_overlap(original, query)
        accepted_needs.append(need)

    if len(accepted_needs) < 2:
        issues.append("TOO_FEW_REWRITTEN_QUERIES")

    covered_businesses = _businesses_from_needs(accepted_needs)
    missing_businesses = [
        business for business in expected_businesses if business not in covered_businesses
    ]
    if missing_businesses:
        issues.append("MISSING_CROSS_BUSINESS_COVERAGE")

    reconstructed = " ".join(str(need.get("query") or "") for need in accepted_needs)
    original_numbers = NUMBER_PATTERN.findall(original)
    missing_numbers = [number for number in original_numbers if number not in reconstructed]
    if missing_numbers:
        issues.append("MISSING_NUMERIC_CONSTRAINT")

    original_negations = [term for term in NEGATION_TERMS if term in original]
    missing_negations = [term for term in original_negations if term not in reconstructed]
    if missing_negations:
        issues.append("MISSING_NEGATION_CONSTRAINT")

    combined_overlap = _token_overlap(original, reconstructed)
    if combined_overlap < policy.minimum_token_overlap:
        issues.append("LOW_ORIGINAL_TOKEN_COVERAGE")

    safety_issues = []
    for need in accepted_needs:
        safety_issues.extend(need.get("hard_filter_denial_reasons") or [])
        if need.get("model_rule_conflict"):
            issues.append("MODEL_RULE_CONFLICT")
        if str(need.get("decomposition_status") or "") not in {"", "COMPLETE"}:
            issues.append("INCOMPLETE_V31_DECOMPOSITION")

    issues = _ordered_unique(issues)
    return {
        "accepted": not issues,
        "issues": issues,
        "rewritten_needs": accepted_needs,
        "expected_businesses": list(expected_businesses),
        "covered_businesses": covered_businesses,
        "missing_businesses": missing_businesses,
        "missing_numbers": missing_numbers,
        "missing_negations": missing_negations,
        "combined_token_overlap": combined_overlap,
        "v31_filter_safety_notes": _ordered_unique(safety_issues),
    }


def _original_plan(original: str) -> dict[str, Any]:
    return {
        "need_id": "FUSED",
        "variant_id": "ORIGINAL",
        "query_source": "ORIGINAL",
        "semantic_query": original,
        "keyword_query": original,
        "query_weight": 1.0,
        "business_function": None,
        "business_filter": {
            "mode": "NONE",
            "value": None,
            "soft_hint": None,
            "denial_reasons": ["HARD_DISABLED_BY_CROSS_REWRITE_POLICY"],
        },
    }


def build_search_plans(
    *,
    original: str,
    rewrite_validation: Mapping[str, Any],
    policy: CrossRewritePolicy,
) -> list[dict[str, Any]]:
    if not rewrite_validation.get("accepted"):
        return [_original_plan(original)]
    rewritten_needs = list(rewrite_validation.get("rewritten_needs") or [])
    sub_weight = policy.rewritten_total_weight / len(rewritten_needs)
    plans = [{
        **_original_plan(original),
        "variant_id": "ORIGINAL_ANCHOR",
        "query_source": "ORIGINAL_ANCHOR",
        "query_weight": policy.original_weight,
    }]
    for index, need in enumerate(rewritten_needs, start=1):
        business = _clean_text(need.get("business_function")) or None
        query = _clean_text(need.get("query"))
        plans.append({
            "need_id": str(need.get("need_id") or f"N{index}"),
            "variant_id": f"REWRITTEN_{index:02d}",
            "query_source": "V31_REWRITTEN_SUBQUERY",
            "semantic_query": query,
            "keyword_query": query,
            "query_weight": sub_weight,
            "business_function": business,
            "intent": need.get("intent"),
            "business_filter": {
                "mode": "SOFT" if policy.allow_soft_business_hint and business else "NONE",
                "value": None,
                "soft_hint": business if policy.allow_soft_business_hint else None,
                "denial_reasons": ["HARD_DISABLED_BY_CROSS_REWRITE_POLICY"],
            },
        })
    return plans


def query_plans_are_valid(route: str, plans: Sequence[Mapping[str, Any]]) -> bool:
    if route != "RETRIEVE":
        return len(plans) == 0
    if not plans:
        return False
    if not math.isclose(
        sum(float(plan.get("query_weight") or 0.0) for plan in plans),
        1.0,
        abs_tol=1e-9,
    ):
        return False
    for plan in plans:
        if not _clean_text(plan.get("semantic_query")):
            return False
        if not _clean_text(plan.get("keyword_query")):
            return False
        if float(plan.get("query_weight") or 0.0) <= 0:
            return False
        if str((plan.get("business_filter") or {}).get("mode") or "") == "HARD":
            return False
    return True


class KDICV31V15CrossRewriteAnalyzer:
    def __init__(
        self,
        v31_analyzer: V31AnalyzerProtocol,
        v31_module: Any,
        policy: CrossRewritePolicy | None = None,
    ) -> None:
        self.v31_analyzer = v31_analyzer
        self.v31_module = v31_module
        self.policy = policy or CrossRewritePolicy()

    def run(
        self,
        query: str,
        *,
        conversation_state: Mapping[str, Any] | None = None,
        manual_selection: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        base_started = time.perf_counter()
        base = self.v31_analyzer.run(
            query,
            conversation_state=conversation_state,
            manual_selection=manual_selection,
        )
        base_latency_ms = (time.perf_counter() - base_started) * 1000
        original = _clean_text(base.get("original_query") or query)
        analysis = dict(base.get("analysis") or {})
        v31_route = str(analysis.get("route") or "")
        route = _normalize_route(v31_route)
        needs = list(analysis.get("needs") or [])

        cross = detect_cross_business(
            original=original,
            needs=needs,
            v31_module=self.v31_module,
        )
        rewrite_called = bool(route == "RETRIEVE" and cross["is_cross_business"])
        if rewrite_called:
            validation = validate_cross_rewrite(
                original=original,
                needs=needs,
                expected_businesses=cross["businesses"],
                policy=self.policy,
            )
        else:
            validation = {
                "accepted": False,
                "issues": ["NOT_CROSS_BUSINESS"] if route == "RETRIEVE" else ["NO_RETRIEVAL_ROUTE"],
                "rewritten_needs": [],
                "expected_businesses": cross["businesses"],
                "covered_businesses": [],
                "missing_businesses": [],
                "missing_numbers": [],
                "missing_negations": [],
                "combined_token_overlap": 0.0,
                "v31_filter_safety_notes": [],
            }

        rewrite_accepted = bool(rewrite_called and validation["accepted"])
        if route == "RETRIEVE":
            plans = build_search_plans(
                original=original,
                rewrite_validation=validation if rewrite_accepted else {"accepted": False},
                policy=self.policy,
            )
        else:
            plans = []

        query_type = (
            "CROSS_BUSINESS"
            if cross["is_cross_business"]
            else ("SAME_BUSINESS_MULTI" if len(needs) >= 2 else "SINGLE")
        ) if route == "RETRIEVE" else "NO_RETRIEVAL"
        plan_valid = query_plans_are_valid(route, plans)
        warnings = list(base.get("validation_warnings") or [])
        if not plan_valid:
            warnings.append("COMBINED_QUERY_PLAN_INVALID")

        return {
            "pipeline_version": PIPELINE_VERSION,
            "analysis_status": "OK" if plan_valid else "INVALID_PLAN",
            "original_query": original,
            "normalized_query": base.get("normalized_query") or original,
            "route": route,
            "v31_route": v31_route,
            "query_type": query_type,
            "v31_analysis": analysis,
            "v31_model_needs": base.get("model_needs") or [],
            "v31_rule_actions": base.get("rule_actions") or [],
            "v31_gate_reasons": base.get("gate_reasons") or [],
            "cross_business": cross,
            "rewrite_called": rewrite_called,
            "rewrite_accepted": rewrite_accepted,
            "rewrite_validation": validation,
            "fallback_to_original": bool(route == "RETRIEVE" and rewrite_called and not rewrite_accepted),
            "search_plans": plans,
            "query_plan_valid": plan_valid,
            "hard_filter_count": sum(
                1 for plan in plans
                if str((plan.get("business_filter") or {}).get("mode") or "") == "HARD"
            ),
            "validation_warnings": _ordered_unique(warnings),
            "runtime": {
                "v31_analysis_latency_ms": base_latency_ms,
                "wrapper_latency_ms": (time.perf_counter() - started) * 1000 - base_latency_ms,
                "total_latency_ms": (time.perf_counter() - started) * 1000,
                "v31_runtime": base.get("runtime") or {},
            },
            "policy": asdict(self.policy),
        }


def analyze_query(
    query: str,
    *,
    analyzer: KDICV31V15CrossRewriteAnalyzer,
    conversation_state: Mapping[str, Any] | None = None,
    manual_selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return analyzer.run(
        query,
        conversation_state=conversation_state,
        manual_selection=manual_selection,
    )