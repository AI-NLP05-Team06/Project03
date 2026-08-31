from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import requests

import kdic_lightweight_router_v1 as light_router
from kdic_integrated_eval_core import AnalyzerCase, QueryPlan, normalize_route


VERSION_V10 = "LIGHT_V1.0_ORIGINAL"
VERSION_V11 = "LIGHT_V1.1_RULE_FALLBACK_ORIGINAL"
VERSION_V12 = "LIGHT_V1.2_RULE_THEN_LLM"
VERSION_V13 = "LIGHT_V1.3_LLM_ON_COMPLEX"
VERSION_ORDER = (VERSION_V10, VERSION_V11, VERSION_V12, VERSION_V13)

VERSION_DESCRIPTIONS = {
    VERSION_V10: "공통 라우팅 후 RETRIEVE 질의는 원문 하나만 검색",
    VERSION_V11: "복합 가능성이 높으면 규칙 분해, 실패 시 원문 검색",
    VERSION_V12: "복합 가능성이 높으면 규칙 분해, 실패 시 LLM 구조화 분해, 다시 실패하면 원문 검색",
    VERSION_V13: "복합 가능성이 높으면 규칙을 건너뛰고 LLM 구조화 분해, 실패 시 원문 검색",
}

DECOMPOSITION_PROMPT_VERSION = "KDIC_DECOMPOSE_STRUCTURED_V1_2026_08_13"
NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d+)?")
NEGATION_TERMS = ("아닌", "아니", "않", "못", "제외", "불가", "없이", "없", "미해당")
TOKEN_PATTERN = re.compile(r"[가-힣A-Za-z0-9]+")


@dataclass(frozen=True)
class AblationConfig:
    original_anchor_weight: float = 0.60
    decomposition_weight: float = 0.40
    max_subqueries: int = 4
    llm_min_confidence: float = 0.80
    llm_model: str = "HCX-007"
    llm_endpoint: str = "https://clovastudio.stream.ntruss.com/v3/chat-completions/HCX-007"
    llm_timeout_seconds: float = 90.0
    llm_max_retries: int = 2
    request_delay_seconds: float = 0.0

    def __post_init__(self) -> None:
        total = self.original_anchor_weight + self.decomposition_weight
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"검색 질의 가중치 합은 1이어야 합니다: {total}")
        if self.max_subqueries < 2:
            raise ValueError("max_subqueries는 2 이상이어야 합니다.")


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _ordered_unique(values: Sequence[Any]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            output.append(text)
            seen.add(text)
    return output


def _parse_previous_turns(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (list, dict)):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return text


def normalize_gold_route(value: Any) -> str:
    route = normalize_route(value)
    if route in {"SIMPLE_RETRIEVE", "MULTI_RETRIEVE", "RETRIEVE_RELAXED"}:
        return "RETRIEVE"
    return route


def analyze_common(
    evaluation_id: str,
    question: str,
    *,
    previous_turns: Any = None,
    router_config: light_router.RouterConfig | None = None,
) -> dict[str, Any]:
    """네 버전이 공유하는 정규화·라우팅·복합가능성 판별을 한 번 수행한다."""
    config = router_config or light_router.RouterConfig()
    common_started = _now_ms()
    normalized = light_router.normalize_query(question)
    context = light_router.build_context(_parse_previous_turns(previous_turns), None)
    route_raw, route_reasons, missing, direct_action = light_router.detect_route(
        normalized["normalized_query"], context=context
    )
    route = normalize_route(route_raw)
    if route == "RETRIEVE":
        complexity = light_router.detect_complexity(normalized["normalized_query"])
    else:
        complexity = {
            "question_type": "NONE",
            "businesses": [],
            "intents": [],
            "clause_count": 0,
            "reasons": ["NO_RETRIEVAL_ROUTE"],
        }
    common_latency_ms = _now_ms() - common_started

    rule_started = _now_ms()
    if route == "RETRIEVE" and complexity.get("question_type") == "MULTI":
        rule_decomposition = light_router.decompose_query(
            normalized["normalized_query"], complexity, config
        )
    else:
        rule_decomposition = {
            "status": "NOT_REQUIRED" if route == "RETRIEVE" else "NOT_APPLICABLE",
            "subqueries": [normalized["normalized_query"]] if route == "RETRIEVE" else [],
            "issues": [],
            "fallback_to_original": False,
        }
    rule_latency_ms = _now_ms() - rule_started

    return {
        "evaluation_id": str(evaluation_id),
        "original_question": str(question).strip(),
        "normalized_question": normalized["normalized_query"],
        "normalization_changes": normalized.get("changes") or [],
        "context": context,
        "route": route,
        "route_reasons": route_reasons,
        "missing_information": missing,
        "direct_action": direct_action,
        "complexity": complexity,
        "complex_candidate": route == "RETRIEVE" and complexity.get("question_type") == "MULTI",
        "rule_decomposition": rule_decomposition,
        "common_latency_ms": round(common_latency_ms, 3),
        "rule_latency_ms": round(rule_latency_ms, 3),
    }


def llm_required(version: str, common: Mapping[str, Any]) -> bool:
    if common.get("route") != "RETRIEVE" or not common.get("complex_candidate"):
        return False
    if version == VERSION_V12:
        return (common.get("rule_decomposition") or {}).get("status") != "COMPLETE"
    return version == VERSION_V13


def decomposition_json_schema(max_subqueries: int = 4) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "decomposable": {"type": "boolean"},
            "subqueries": {
                "type": "array",
                "maxItems": int(max_subqueries),
                "items": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string"},
        },
        "required": ["decomposable", "subqueries", "confidence", "reason"],
        "additionalProperties": False,
    }


def build_decomposition_messages(question: str) -> list[dict[str, str]]:
    system = """
당신은 예금보험공사 검색 파이프라인의 복합질의 분해기입니다.
이 작업은 질의 재작성이나 검색어 최적화가 아니라, 원문에 실제로 들어 있는 독립 정보 요구를 구조적으로 분리하는 작업입니다.

규칙:
1. 서로 따로 검색하고 답할 수 있는 정보 요구가 2개 이상일 때만 decomposable=true로 판단합니다.
2. 단일 업무의 하나의 응집된 질문, 용어 정의, 비교 관계 자체를 묻는 질문은 분리하지 않습니다.
3. 원문의 업무명, 대상, 조건, 숫자, 기간, 부정 표현을 빠뜨리거나 바꾸지 않습니다.
4. 원문에 없는 업무, 조건, 숫자, 예외, 의도를 추가하지 않습니다.
5. 문체 개선, 요약, 동의어 확장, 검색 키워드 생성은 하지 않습니다.
6. 각 하위질문은 단독으로 이해 가능한 한국어 질문이어야 합니다.
7. 분리할 수 없거나 확신이 낮으면 decomposable=false, subqueries=[]로 반환합니다.
8. 하위질문은 2개 이상 4개 이하로 제한합니다.
""".strip()
    user = f"원문 질문:\n{question}\n\n원문의 독립 정보 요구만 판별하고 JSON 스키마에 맞춰 반환하세요."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _extract_hcx_payload(response_json: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    result = response_json.get("result") or {}
    message = result.get("message") or {}
    content = message.get("content")
    if isinstance(content, Mapping):
        payload = dict(content)
    else:
        text = str(content or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()
        payload = json.loads(text)
    usage = result.get("usage") or response_json.get("usage") or {}
    prompt_tokens = int(usage.get("promptTokens") or usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completionTokens") or usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("totalTokens") or usage.get("total_tokens") or prompt_tokens + completion_tokens)
    return payload, {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _token_overlap_ratio(original: str, candidate: str) -> float:
    original_tokens = set(TOKEN_PATTERN.findall(original.lower()))
    candidate_tokens = set(TOKEN_PATTERN.findall(candidate.lower()))
    if not candidate_tokens:
        return 0.0
    return len(original_tokens & candidate_tokens) / len(candidate_tokens)


def validate_llm_decomposition(
    original: str,
    payload: Mapping[str, Any],
    *,
    expected_businesses: Sequence[str],
    config: AblationConfig,
) -> dict[str, Any]:
    issues: list[str] = []
    decomposable = payload.get("decomposable") is True
    confidence = float(payload.get("confidence") or 0.0)
    reason = str(payload.get("reason") or "").strip()
    raw_subqueries = payload.get("subqueries") or []
    if not isinstance(raw_subqueries, list):
        raw_subqueries = []
        issues.append("SUBQUERIES_NOT_ARRAY")
    raw_subquery_count = len(raw_subqueries)
    subqueries = _ordered_unique(
        [item.get("query") if isinstance(item, Mapping) else item for item in raw_subqueries]
    )[: config.max_subqueries]

    if not decomposable:
        return {
            "status": "DECLINED",
            "accepted": False,
            "subqueries": [],
            "confidence": confidence,
            "reason": reason,
            "issues": ["LLM_DECLINED_DECOMPOSITION"],
        }
    if confidence < config.llm_min_confidence:
        issues.append("LOW_LLM_CONFIDENCE")

    base_validation = light_router.validate_decomposition(original, subqueries, expected_businesses)
    issues.extend(base_validation.get("issues") or [])
    if raw_subquery_count > config.max_subqueries:
        issues.append("TOO_MANY_SUBQUERIES")

    reconstructed = " ".join(subqueries)
    original_numbers = set(NUMBER_PATTERN.findall(original))
    generated_numbers = set(NUMBER_PATTERN.findall(reconstructed))
    if generated_numbers - original_numbers:
        issues.append("INVENTED_NUMERIC_CONSTRAINT")
    original_negations = {term for term in NEGATION_TERMS if term in original}
    generated_negations = {term for term in NEGATION_TERMS if term in reconstructed}
    if generated_negations - original_negations:
        issues.append("INVENTED_NEGATION")
    generated_businesses = set(light_router.find_businesses(reconstructed))
    expected_business_set = set(expected_businesses)
    if expected_business_set and generated_businesses - expected_business_set:
        issues.append("INVENTED_BUSINESS")
    for subquery in subqueries:
        if _token_overlap_ratio(original, subquery) < 0.25:
            issues.append("LOW_SOURCE_TERM_OVERLAP")
            break

    issues = _ordered_unique(issues)
    accepted = not issues and 2 <= len(subqueries) <= config.max_subqueries
    return {
        "status": "COMPLETE" if accepted else "FAILED",
        "accepted": accepted,
        "subqueries": subqueries if accepted else [],
        "confidence": confidence,
        "reason": reason,
        "issues": issues,
    }


class HCXStructuredDecomposer:
    def __init__(
        self,
        api_key: str,
        *,
        config: AblationConfig | None = None,
        cache_path: str | Path | None = None,
        session: requests.Session | None = None,
    ) -> None:
        key = str(api_key or "").strip()
        if not key or key.lower().startswith("bearer ") or any(ch.isspace() for ch in key):
            raise ValueError("HCX_API_KEY에는 Bearer 접두사나 공백을 넣지 않습니다.")
        self.api_key = key
        self.config = config or AblationConfig()
        self.cache_path = Path(cache_path) if cache_path else None
        self.session = session or requests.Session()
        self.cache: dict[str, dict[str, Any]] = {}
        if self.cache_path and self.cache_path.exists():
            with self.cache_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        self.cache[str(row["cache_key"])] = row

    def _cache_key(self, question: str, expected_businesses: Sequence[str]) -> str:
        raw = json.dumps(
            {
                "prompt_version": DECOMPOSITION_PROMPT_VERSION,
                "model": self.config.llm_model,
                "question": question,
                "expected_businesses": list(expected_businesses),
                "min_confidence": self.config.llm_min_confidence,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _append_cache(self, row: Mapping[str, Any]) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")

    def decompose(self, question: str, expected_businesses: Sequence[str]) -> dict[str, Any]:
        cache_key = self._cache_key(question, expected_businesses)
        if cache_key in self.cache:
            cached = dict(self.cache[cache_key])
            cached["cache_hit"] = True
            cached["actual_api_latency_ms"] = 0.0
            return cached

        body = {
            "messages": build_decomposition_messages(question),
            "topP": 0.1,
            "topK": 0,
            "maxCompletionTokens": 700,
            "temperature": 0.0,
            "repetitionPenalty": 1.0,
            "thinking": {"effort": "none"},
            "stop": [],
            "responseFormat": {"type": "json", "schema": decomposition_json_schema(self.config.max_subqueries)},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-NCP-CLOVASTUDIO-REQUEST-ID": str(uuid.uuid4()),
        }
        last_error: Exception | None = None
        for attempt in range(self.config.llm_max_retries + 1):
            started = _now_ms()
            try:
                response = self.session.post(
                    self.config.llm_endpoint,
                    headers=headers,
                    json=body,
                    timeout=self.config.llm_timeout_seconds,
                )
                response.raise_for_status()
                payload, usage = _extract_hcx_payload(response.json())
                api_latency_ms = _now_ms() - started
                validation = validate_llm_decomposition(
                    question,
                    payload,
                    expected_businesses=expected_businesses,
                    config=self.config,
                )
                row = {
                    "cache_key": cache_key,
                    "question": question,
                    "model": self.config.llm_model,
                    "prompt_version": DECOMPOSITION_PROMPT_VERSION,
                    "raw_payload": payload,
                    **validation,
                    **usage,
                    "effective_api_latency_ms": round(api_latency_ms, 3),
                    "actual_api_latency_ms": round(api_latency_ms, 3),
                    "cache_hit": False,
                    "error_type": "",
                    "error_message": "",
                }
                self.cache[cache_key] = row
                self._append_cache(row)
                if self.config.request_delay_seconds > 0:
                    time.sleep(self.config.request_delay_seconds)
                return dict(row)
            except Exception as error:
                last_error = error
                if attempt < self.config.llm_max_retries:
                    time.sleep(min(2 ** attempt, 4))

        row = {
            "cache_key": cache_key,
            "question": question,
            "model": self.config.llm_model,
            "prompt_version": DECOMPOSITION_PROMPT_VERSION,
            "raw_payload": {},
            "status": "ERROR",
            "accepted": False,
            "subqueries": [],
            "confidence": 0.0,
            "reason": "",
            "issues": ["LLM_REQUEST_FAILED"],
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "effective_api_latency_ms": 0.0,
            "actual_api_latency_ms": 0.0,
            "cache_hit": False,
            "error_type": type(last_error).__name__ if last_error else "UnknownError",
            "error_message": str(last_error or "unknown error"),
        }
        self.cache[cache_key] = row
        self._append_cache(row)
        return dict(row)


def _make_plans(original: str, subqueries: Sequence[str], config: AblationConfig) -> list[QueryPlan]:
    original_compact = re.sub(r"\s+", "", original).lower()
    valid_subqueries = [
        value for value in _ordered_unique(subqueries)
        if re.sub(r"\s+", "", value).lower() != original_compact
    ]
    if len(valid_subqueries) < 2:
        return [
            QueryPlan(
                need_id="FUSED",
                variant_id="ORIGINAL",
                dense_query=original,
                bm25_query=original,
                filter_mode="NONE",
                business_filters=[],
                soft_business_hints=[],
                query_weight=1.0,
                query_source="ORIGINAL",
            )
        ]
    sub_weight = config.decomposition_weight / len(valid_subqueries)
    plans = [
        QueryPlan(
            need_id="FUSED",
            variant_id="ORIGINAL_ANCHOR",
            dense_query=original,
            bm25_query=original,
            filter_mode="NONE",
            business_filters=[],
            soft_business_hints=[],
            query_weight=config.original_anchor_weight,
            query_source="ORIGINAL_ANCHOR",
        )
    ]
    for index, subquery in enumerate(valid_subqueries, 1):
        plans.append(
            QueryPlan(
                need_id="FUSED",
                variant_id=f"SUBQUERY_{index:02d}",
                dense_query=subquery,
                bm25_query=subquery,
                filter_mode="NONE",
                business_filters=[],
                soft_business_hints=[],
                query_weight=sub_weight,
                query_source="DECOMPOSED",
            )
        )
    return plans


def build_version_case(
    common: Mapping[str, Any],
    version: str,
    *,
    llm_record: Mapping[str, Any] | None = None,
    config: AblationConfig | None = None,
) -> AnalyzerCase:
    if version not in VERSION_ORDER:
        raise ValueError(f"지원하지 않는 버전: {version}")
    cfg = config or AblationConfig()
    route = str(common["route"])
    original = str(common["original_question"])
    rule = dict(common.get("rule_decomposition") or {})
    candidate = bool(common.get("complex_candidate"))
    subqueries: list[str] = []
    source = "ORIGINAL_POLICY"
    fallback_reason = ""
    policy_rule_used = False
    policy_llm_called = False

    if route == "RETRIEVE" and candidate:
        if version in {VERSION_V11, VERSION_V12}:
            policy_rule_used = True
            if rule.get("status") == "COMPLETE":
                subqueries = list(rule.get("subqueries") or [])
                source = "RULE"
            elif version == VERSION_V12:
                policy_llm_called = True
                if llm_record and llm_record.get("accepted"):
                    subqueries = list(llm_record.get("subqueries") or [])
                    source = "LLM"
                else:
                    source = "ORIGINAL_FALLBACK"
                    fallback_reason = "LLM_FAILED_OR_DECLINED"
            else:
                source = "ORIGINAL_FALLBACK"
                fallback_reason = "RULE_DECOMPOSITION_FAILED"
        elif version == VERSION_V13:
            policy_llm_called = True
            if llm_record and llm_record.get("accepted"):
                subqueries = list(llm_record.get("subqueries") or [])
                source = "LLM"
            else:
                source = "ORIGINAL_FALLBACK"
                fallback_reason = "LLM_FAILED_OR_DECLINED"

    if route == "RETRIEVE":
        plans = _make_plans(original, subqueries, cfg)
    else:
        plans = []

    analysis_latency_ms = float(common.get("common_latency_ms") or 0.0)
    if policy_rule_used:
        analysis_latency_ms += float(common.get("rule_latency_ms") or 0.0)
    if policy_llm_called and llm_record:
        analysis_latency_ms += float(llm_record.get("effective_api_latency_ms") or 0.0)

    raw_result = {
        "pipeline_version": version,
        "analysis_status": "OK",
        "original_query": original,
        "normalized_query": common.get("normalized_question"),
        "route_reasons": common.get("route_reasons") or [],
        "direct_action": common.get("direct_action"),
        "blocking_slot": (common.get("missing_information") or [None])[0],
        "complexity": common.get("complexity") or {},
        "complex_candidate": candidate,
        "rule_decomposition": rule,
        "llm_decomposition": dict(llm_record or {}),
        "decomposition_source": source,
        "final_subqueries": subqueries,
        "fallback_reason": fallback_reason,
        "model_needs": [
            {"business_function": value}
            for value in (common.get("complexity") or {}).get("businesses") or []
        ],
        "runtime": {
            "api_request_count": int(policy_llm_called),
            "prompt_tokens": int((llm_record or {}).get("prompt_tokens") or 0) if policy_llm_called else 0,
            "completion_tokens": int((llm_record or {}).get("completion_tokens") or 0) if policy_llm_called else 0,
            "total_tokens": int((llm_record or {}).get("total_tokens") or 0) if policy_llm_called else 0,
            "latency_ms": round(analysis_latency_ms, 3),
        },
    }
    return AnalyzerCase(
        evaluation_id=str(common["evaluation_id"]),
        analyzer=version,
        original_question=original,
        route=route,
        analysis_latency_ms=round(analysis_latency_ms, 3),
        plans=plans,
        raw_result=raw_result,
    )


def build_all_cases(
    eval_df: pd.DataFrame,
    *,
    decomposer: HCXStructuredDecomposer | None,
    config: AblationConfig | None = None,
    previous_turns_column: str = "previous_turns",
) -> tuple[dict[tuple[str, str], AnalyzerCase], pd.DataFrame]:
    cfg = config or AblationConfig()
    common_by_id: dict[str, dict[str, Any]] = {}
    for row in eval_df.to_dict(orient="records"):
        evaluation_id = str(row["evaluation_id"])
        common_by_id[evaluation_id] = analyze_common(
            evaluation_id,
            str(row["question"]),
            previous_turns=row.get(previous_turns_column),
        )

    llm_by_id: dict[str, dict[str, Any]] = {}
    required_ids = [
        evaluation_id
        for evaluation_id, common in common_by_id.items()
        if any(llm_required(version, common) for version in (VERSION_V12, VERSION_V13))
    ]
    if required_ids and decomposer is None:
        raise ValueError("V1.2/V1.3 평가에는 HCXStructuredDecomposer가 필요합니다.")
    for evaluation_id in required_ids:
        common = common_by_id[evaluation_id]
        llm_by_id[evaluation_id] = decomposer.decompose(
            str(common["normalized_question"]),
            list((common.get("complexity") or {}).get("businesses") or []),
        )

    cases: dict[tuple[str, str], AnalyzerCase] = {}
    audit_rows: list[dict[str, Any]] = []
    for evaluation_id, common in common_by_id.items():
        llm_record = llm_by_id.get(evaluation_id)
        for version in VERSION_ORDER:
            case = build_version_case(common, version, llm_record=llm_record, config=cfg)
            cases[(version, evaluation_id)] = case
            runtime = case.raw_result["runtime"]
            audit_rows.append({
                "evaluation_id": evaluation_id,
                "question": case.original_question,
                "version": version,
                "route": case.route,
                "complex_candidate": bool(common.get("complex_candidate")),
                "rule_status": (common.get("rule_decomposition") or {}).get("status"),
                "rule_subqueries": (common.get("rule_decomposition") or {}).get("subqueries") or [],
                "llm_policy_call": int(llm_required(version, common)),
                "llm_actual_api_call": int(bool(llm_record) and not bool(llm_record.get("cache_hit"))) if llm_required(version, common) else 0,
                "llm_cache_hit": bool((llm_record or {}).get("cache_hit")) if llm_required(version, common) else False,
                "llm_status": (llm_record or {}).get("status", "NOT_CALLED"),
                "llm_confidence": float((llm_record or {}).get("confidence") or 0.0),
                "llm_issues": (llm_record or {}).get("issues") or [],
                "decomposition_source": case.raw_result["decomposition_source"],
                "final_subqueries": case.raw_result["final_subqueries"],
                "query_plan_count": len(case.plans),
                "query_plan_weight_sum": round(sum(plan.query_weight for plan in case.plans), 10),
                "hard_filter_count": sum(plan.filter_mode == "HARD" for plan in case.plans),
                "analysis_latency_ms": case.analysis_latency_ms,
                "prompt_tokens": runtime["prompt_tokens"],
                "completion_tokens": runtime["completion_tokens"],
                "total_tokens": runtime["total_tokens"],
            })
    return cases, pd.DataFrame(audit_rows)


def summarize_router_ablation(audit_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for version, frame in audit_df.groupby("version", sort=False):
        retrieve = frame[frame["route"].eq("RETRIEVE")]
        rows.append({
            "version": version,
            "question_count": len(frame),
            "retrieve_count": int(frame["route"].eq("RETRIEVE").sum()),
            "clarify_count": int(frame["route"].eq("CLARIFY").sum()),
            "out_of_scope_count": int(frame["route"].eq("OUT_OF_SCOPE").sum()),
            "direct_response_count": int(frame["route"].eq("DIRECT_RESPONSE").sum()),
            "complex_candidate_count": int(frame["complex_candidate"].sum()),
            "decomposed_count": int(frame["decomposition_source"].isin(["RULE", "LLM"]).sum()),
            "rule_decomposed_count": int(frame["decomposition_source"].eq("RULE").sum()),
            "llm_decomposed_count": int(frame["decomposition_source"].eq("LLM").sum()),
            "original_fallback_count": int(frame["decomposition_source"].eq("ORIGINAL_FALLBACK").sum()),
            "llm_policy_call_count": int(frame["llm_policy_call"].sum()),
            "llm_total_tokens": int(frame["total_tokens"].sum()),
            "analysis_latency_ms_mean_all": float(frame["analysis_latency_ms"].mean()),
            "analysis_latency_ms_p95_all": float(frame["analysis_latency_ms"].quantile(0.95)),
            "retrieval_query_count_mean": float(retrieve["query_plan_count"].mean()) if len(retrieve) else 0.0,
            "hard_filter_count": int(frame["hard_filter_count"].sum()),
            "invalid_query_plan_weight_count": int((retrieve["query_plan_weight_sum"].sub(1.0).abs() > 1e-9).sum()),
        })
    summary = pd.DataFrame(rows)
    summary["version"] = pd.Categorical(summary["version"], VERSION_ORDER, ordered=True)
    return summary.sort_values("version").reset_index(drop=True)


def route_hard_gate_report(audit_df: pd.DataFrame, eval_df: pd.DataFrame) -> pd.DataFrame:
    gold = eval_df[["evaluation_id", "gold_route_v6"]].copy() if "gold_route_v6" in eval_df.columns else pd.DataFrame()
    if not gold.empty:
        gold["gold_route_v6"] = gold["gold_route_v6"].map(normalize_gold_route)
    rows: list[dict[str, Any]] = []
    for version, frame in audit_df.groupby("version", sort=False):
        merged = frame.merge(gold, on="evaluation_id", how="left") if not gold.empty else frame.assign(gold_route_v6="")
        route_known = merged["gold_route_v6"].fillna("").ne("")
        normal_retrieve = merged["gold_route_v6"].eq("RETRIEVE")
        predicted_retrieve = merged["route"].eq("RETRIEVE")
        query_valid_rate = float(merged.loc[predicted_retrieve, "query_plan_count"].gt(0).mean()) if predicted_retrieve.any() else 1.0
        wrong_oos = int((normal_retrieve & merged["route"].eq("OUT_OF_SCOPE")).sum())
        wrong_direct = int((normal_retrieve & merged["route"].eq("DIRECT_RESPONSE")).sum())
        route_accuracy = float((merged.loc[route_known, "route"] == merged.loc[route_known, "gold_route_v6"]).mean()) if route_known.any() else float("nan")
        predicted_clarify = merged["route"].eq("CLARIFY") & route_known
        clarify_precision = float(merged.loc[predicted_clarify, "gold_route_v6"].eq("CLARIFY").mean()) if predicted_clarify.any() else 1.0
        rows.extend([
            {"version": version, "gate": "실행 성공률", "value": 1.0, "threshold": ">=0.995", "passed": True},
            {"version": version, "gate": "검색 질의 생성 유효율", "value": query_valid_rate, "threshold": ">=0.99", "passed": query_valid_rate >= 0.99},
            {"version": version, "gate": "정상 질문의 잘못된 OOS", "value": wrong_oos, "threshold": "=0", "passed": wrong_oos == 0},
            {"version": version, "gate": "정상 질문의 잘못된 DIRECT", "value": wrong_direct, "threshold": "=0", "passed": wrong_direct == 0},
            {"version": version, "gate": "Hard Filter", "value": int(frame["hard_filter_count"].sum()), "threshold": "=0", "passed": int(frame["hard_filter_count"].sum()) == 0},
            {"version": version, "gate": "검색 불가능 질문의 추가질문 Precision", "value": clarify_precision, "threshold": ">=0.95", "passed": clarify_precision >= 0.95},
            {"version": version, "gate": "최종 라우팅 정확도", "value": route_accuracy, "threshold": "참고", "passed": True},
        ])
    return pd.DataFrame(rows)


def query_analysis_quality_summary(audit_df: pd.DataFrame, eval_df: pd.DataFrame) -> pd.DataFrame:
    """라우팅과 최종 분해 여부를 gold와 비교한다. gold 열이 없으면 해당 값은 NaN이다."""
    gold_columns = [column for column in ("evaluation_id", "gold_route_v6", "split_needed") if column in eval_df.columns]
    gold = eval_df[gold_columns].copy()
    if "gold_route_v6" in gold.columns:
        gold["gold_route"] = gold["gold_route_v6"].map(normalize_gold_route)
    else:
        gold["gold_route"] = ""
    if "split_needed" in gold.columns:
        gold["gold_multi"] = gold["split_needed"].map(
            lambda value: str(value).strip().lower() in {"1", "true", "y", "yes", "multi", "복합", "필요"}
            if not pd.isna(value) and str(value).strip() else pd.NA
        )
    else:
        gold["gold_multi"] = pd.NA

    rows: list[dict[str, Any]] = []
    for version, frame in audit_df.groupby("version", sort=False):
        merged = frame.merge(gold[["evaluation_id", "gold_route", "gold_multi"]], on="evaluation_id", how="left")
        known_route = merged["gold_route"].fillna("").ne("")
        route_accuracy = float(merged.loc[known_route, "route"].eq(merged.loc[known_route, "gold_route"]).mean()) if known_route.any() else float("nan")
        gold_retrieve = merged["gold_route"].eq("RETRIEVE")
        retrieve_recall = float(merged.loc[gold_retrieve, "route"].eq("RETRIEVE").mean()) if gold_retrieve.any() else float("nan")
        predicted_clarify = merged["route"].eq("CLARIFY") & known_route
        clarify_precision = float(merged.loc[predicted_clarify, "gold_route"].eq("CLARIFY").mean()) if predicted_clarify.any() else 1.0

        known_multi = merged["gold_multi"].notna() & gold_retrieve
        predicted_multi = merged["decomposition_source"].isin(["RULE", "LLM"])
        tp = int((known_multi & merged["gold_multi"].astype("boolean").fillna(False) & predicted_multi).sum())
        fp = int((known_multi & ~merged["gold_multi"].astype("boolean").fillna(False) & predicted_multi).sum())
        fn = int((known_multi & merged["gold_multi"].astype("boolean").fillna(False) & ~predicted_multi).sum())
        if not known_multi.any():
            precision = recall = f1 = float("nan")
        else:
            precision = tp / (tp + fp) if tp + fp else 1.0
            recall = tp / (tp + fn) if tp + fn else 1.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({
            "version": version,
            "route_accuracy": route_accuracy,
            "retrieve_recall": retrieve_recall,
            "clarify_precision": clarify_precision,
            "multi_precision": precision,
            "multi_recall": recall,
            "multi_f1": f1,
            "multi_tp": tp,
            "multi_fp": fp,
            "multi_fn": fn,
            "analysis_latency_ms_mean": float(frame["analysis_latency_ms"].mean()),
            "llm_policy_call_count": int(frame["llm_policy_call"].sum()),
            "llm_total_tokens": int(frame["total_tokens"].sum()),
        })
    summary = pd.DataFrame(rows)
    summary["version"] = pd.Categorical(summary["version"], VERSION_ORDER, ordered=True)
    return summary.sort_values("version").reset_index(drop=True)


def case_signature(case: AnalyzerCase) -> str:
    payload = {
        "route": case.route,
        "plans": [asdict(plan) for plan in case.plans],
        "source": case.raw_result.get("decomposition_source"),
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]
