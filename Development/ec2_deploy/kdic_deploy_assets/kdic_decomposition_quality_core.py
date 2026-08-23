from __future__ import annotations

import hashlib
import json
import math
import re
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd
try:
    import requests
except ImportError:  # 로컬 정적 검증 환경에서는 HTTP 호출을 사용하지 않을 수 있습니다.
    requests = None  # type: ignore[assignment]

import kdic_lightweight_router_v1 as light_router


DATE_PREFIX = "2026-08-14"

BASELINE = "V1.5_BASELINE"
QUALITY = "V1.5_Q"
RETRY = "V1.5_R"
QUALITY_RETRY = "V1.5_QR"
CONDITION_ORDER = (BASELINE, QUALITY, RETRY, QUALITY_RETRY)
CONDITION_LABELS = {
    BASELINE: "V1.5 Baseline",
    QUALITY: "V1.5-Q 품질개선",
    RETRY: "V1.5-R 교정재시도",
    QUALITY_RETRY: "V1.5-QR 품질개선+재시도",
}

PROMPT_VERSION = "KDIC_DECOMPOSITION_QUALITY_V1_2026_08_14"
REPAIR_PROMPT_VERSION = "KDIC_DECOMPOSITION_REPAIR_V1_2026_08_14"

INTENT_VALUES = (
    "OVERVIEW", "ELIGIBILITY", "AMOUNT", "APPLICATION", "DOCUMENTS",
    "TIME", "STATUS", "CALCULATION", "EXCEPTION", "OTHER",
)
INTENT_PATTERNS: dict[str, tuple[str, ...]] = {
    "AMOUNT": (r"한도", r"얼마", r"금액", r"최대", r"최소", r"몇\s*원", r"비율"),
    "APPLICATION": (
        r"신청\s*(?:방법|절차)", r"신청하려면", r"접수\s*(?:방법|절차)?",
        r"어떻게\s*(?:신청|받|진행)",
    ),
    "DOCUMENTS": (r"서류", r"준비물", r"증빙", r"제출"),
    "TIME": (r"언제", r"기간", r"기한", r"시점", r"며칠", r"몇\s*개월"),
    "STATUS": (r"조회", r"확인", r"찾(?:는|을|아)", r"남았(?:는지|나요)"),
    "CALCULATION": (r"계산", r"산정", r"합산"),
    "EXCEPTION": (r"제외", r"예외", r"불가", r"해당하지", r"받지\s*못"),
    "ELIGIBILITY": (r"대상", r"자격", r"조건", r"누가", r"가능한지", r"받을\s*수\s*있"),
    "OVERVIEW": (r"무엇(?:인가요|인지|이죠)?", r"의미", r"차이", r"어떤\s*제도", r"설명"),
}
SUBJECT_TERMS = (
    "상속인", "본인", "대리인", "법인", "개인", "채무자", "송금인", "수취인",
    "미성년자", "친권자", "외국인", "고인", "피상속인", "금융회사",
)
NEGATION_TERMS = ("아닌", "아니", "않", "못", "제외", "불가", "없이", "없", "미해당", "뿐 아니라")
NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d+)?(?:\s*(?:원|만원|천만원|억원|%|퍼센트|년|개월|일|회))?")
TOKEN_PATTERN = re.compile(r"[가-힣A-Za-z0-9]+")
UNRESOLVED_REFERENCE_PATTERN = re.compile(r"(?:그것|그거|이것|해당\s*(?:제도|경우|업무)|그\s*제도|앞의\s*내용)")


@dataclass(frozen=True)
class QualityConfig:
    llm_endpoint: str = "https://clovastudio.stream.ntruss.com/v3/chat-completions/HCX-007"
    llm_model: str = "HCX-007"
    llm_timeout_seconds: float = 120.0
    transport_retries: int = 2
    semantic_retries: int = 1
    llm_min_confidence: float = 0.80
    max_subqueries: int = 4
    request_delay_seconds: float = 0.0
    original_weight: float = 0.40
    subquery_total_weight: float = 0.60

    def __post_init__(self) -> None:
        if abs(self.original_weight + self.subquery_total_weight - 1.0) > 1e-9:
            raise ValueError("원문과 하위질의 가중치 합은 1이어야 합니다.")
        if self.semantic_retries != 1:
            raise ValueError("이번 실험의 의미 교정 재시도는 정확히 1회로 고정합니다.")


def ordered_unique(values: Sequence[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def find_intents(text: str) -> list[str]:
    output: list[str] = []
    for intent, patterns in INTENT_PATTERNS.items():
        if any(re.search(pattern, text) for pattern in patterns):
            output.append(intent)
    if len(output) > 1 and "OVERVIEW" in output:
        output.remove("OVERVIEW")
    return output


def source_features(question: str, expected_businesses: Sequence[str]) -> dict[str, Any]:
    matches = light_router.find_business_matches(question)
    evidence_by_business = {
        str(row["business_function"]): ordered_unique(row.get("evidence") or [])
        for row in matches
    }
    return {
        "expected_businesses": ordered_unique(expected_businesses),
        "expected_intents": find_intents(question),
        "numbers": ordered_unique(NUMBER_PATTERN.findall(question)),
        "negations": [term for term in NEGATION_TERMS if term in question],
        "subjects": [term for term in SUBJECT_TERMS if term in question],
        "business_evidence": evidence_by_business,
    }


def quality_json_schema(max_subqueries: int) -> dict[str, Any]:
    business_values = sorted(light_router.BUSINESS_FUNCTIONS)
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "decomposable": {"type": "boolean"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string"},
            "subqueries": {
                "type": "array",
                "minItems": 0,
                "maxItems": max_subqueries,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "query": {"type": "string"},
                        "business_function": {"type": "string", "enum": business_values},
                        "intent": {"type": "string", "enum": list(INTENT_VALUES)},
                        "preserved_terms": {"type": "array", "items": {"type": "string"}},
                        "preserved_constraints": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "query", "business_function", "intent",
                        "preserved_terms", "preserved_constraints",
                    ],
                },
            },
        },
        "required": ["decomposable", "confidence", "reason", "subqueries"],
    }


def baseline_json_schema(max_subqueries: int) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "decomposable": {"type": "boolean"},
            "subqueries": {
                "type": "array",
                "maxItems": max_subqueries,
                "items": {
                    "oneOf": [
                        {"type": "string"},
                        {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    ]
                },
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string"},
        },
        "required": ["decomposable", "subqueries", "confidence", "reason"],
    }


def build_quality_messages(question: str, expected_businesses: Sequence[str]) -> list[dict[str, str]]:
    features = source_features(question, expected_businesses)
    system = """당신은 예금보험공사 검색용 교차업무 질의 구조화 분해기입니다.
라우터가 제시한 업무들은 확정 정답이 아니라 분해 필요성을 검토할 후보입니다.
서로 다른 업무 용어가 보여도 하나의 사건·절차·대상을 비교하거나 설명하는 단일 정보요구라면 decomposable=false와 빈 subqueries를 반환합니다.
서로 독립적으로 검색해야 할 업무별 정보요구가 둘 이상일 때만 decomposable=true로 분해합니다.
원문을 요약하거나 일반화하지 말고, 서로 다른 업무별 독립 검색 질의로만 분리합니다.
각 하위질의는 하나의 업무와 하나의 주된 요청 의도만 담당해야 합니다.
원문의 전문용어, 숫자, 금액, 기간, 부정·제외 표현, 사용자 주체와 조건을 보존합니다.
원문에 없는 업무·숫자·조건을 추가하지 않습니다.
'그것', '해당 경우'처럼 원문 없이 이해할 수 없는 표현을 사용하지 않습니다.
동일 의미의 하위질의를 중복 생성하지 않습니다.
반드시 지정된 JSON Schema만 출력합니다."""
    user = json.dumps({
        "question": question,
        "router_expected_businesses": list(expected_businesses),
        "source_features_to_preserve": features,
        "instruction": "먼저 실제 독립 정보요구가 둘 이상인지 판정하세요. 맞을 때만 각 업무를 담당하는 2~4개 하위질의로 분해하세요.",
    }, ensure_ascii=False, indent=2)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_repair_messages(
    question: str,
    expected_businesses: Sequence[str],
    previous_payload: Mapping[str, Any],
    issues: Sequence[str],
    *,
    quality_mode: bool,
) -> list[dict[str, str]]:
    system = """당신은 검색용 질의 분해 결과 교정기입니다.
직전 결과 전체를 새로 창작하지 말고, 검증기가 지적한 오류만 수정합니다.
누락된 업무·요청·전문용어·숫자·부정·주체를 복원하고 새 정보는 만들지 않습니다.
라우터 업무 후보는 확정 정답이 아닙니다. 하나의 정보요구라면 decomposable=false로 판단하며 억지로 분해하지 않습니다.
교정 결과도 검증을 통과하지 못하면 폐기되므로 억지로 분해하지 않습니다.
반드시 지정된 JSON Schema만 출력합니다."""
    user = json.dumps({
        "question": question,
        "router_expected_businesses": list(expected_businesses),
        "source_features_to_preserve": source_features(question, expected_businesses),
        "previous_payload": dict(previous_payload),
        "validation_issues": list(issues),
        "output_mode": "quality_structured" if quality_mode else "baseline_structured",
        "instruction": "검증 오류를 정확히 수정하여 2~4개의 독립 검색 질의를 반환하세요.",
    }, ensure_ascii=False, indent=2)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def extract_queries(payload: Mapping[str, Any]) -> list[str]:
    raw = payload.get("subqueries") or []
    if not isinstance(raw, list):
        return []
    return ordered_unique([
        item.get("query") if isinstance(item, Mapping) else item
        for item in raw
    ])


def _token_overlap_ratio(original: str, candidate: str) -> float:
    original_tokens = set(TOKEN_PATTERN.findall(original.lower()))
    candidate_tokens = set(TOKEN_PATTERN.findall(candidate.lower()))
    if not candidate_tokens:
        return 0.0
    return len(original_tokens & candidate_tokens) / len(candidate_tokens)


def validate_baseline_decomposition(
    question: str,
    payload: Mapping[str, Any],
    *,
    expected_businesses: Sequence[str],
    config: QualityConfig,
) -> dict[str, Any]:
    """기존 V1.5 검증 규칙을 독립적으로 재현한다.

    이 함수는 개선 검증기의 비교 기준이므로 기존 실험 모듈을 import하지 않는다.
    """
    issues: list[str] = []
    decomposable = payload.get("decomposable") is True
    confidence = float(payload.get("confidence") or 0.0)
    reason = str(payload.get("reason") or "").strip()
    raw_subqueries = payload.get("subqueries") or []
    if not isinstance(raw_subqueries, list):
        raw_subqueries = []
        issues.append("SUBQUERIES_NOT_ARRAY")
    raw_subquery_count = len(raw_subqueries)
    subqueries = extract_queries({"subqueries": raw_subqueries})[: config.max_subqueries]

    if not decomposable:
        return {
            "status": "DECLINED",
            "accepted": False,
            "subqueries": [],
            "candidate_subqueries": subqueries,
            "confidence": confidence,
            "reason": reason,
            "issues": ["LLM_DECLINED_DECOMPOSITION"],
            "checks": content_checks(question, payload, expected_businesses),
        }
    if confidence < config.llm_min_confidence:
        issues.append("LOW_LLM_CONFIDENCE")

    base_validation = light_router.validate_decomposition(
        question, subqueries, expected_businesses
    )
    issues.extend(base_validation.get("issues") or [])
    if raw_subquery_count > config.max_subqueries:
        issues.append("TOO_MANY_SUBQUERIES")

    reconstructed = " ".join(subqueries)
    original_numbers = set(NUMBER_PATTERN.findall(question))
    generated_numbers = set(NUMBER_PATTERN.findall(reconstructed))
    if generated_numbers - original_numbers:
        issues.append("INVENTED_NUMERIC_CONSTRAINT")
    original_negations = {term for term in NEGATION_TERMS if term in question}
    generated_negations = {term for term in NEGATION_TERMS if term in reconstructed}
    if generated_negations - original_negations:
        issues.append("INVENTED_NEGATION")
    generated_businesses = set(light_router.find_businesses(reconstructed))
    expected_business_set = set(expected_businesses)
    if expected_business_set and generated_businesses - expected_business_set:
        issues.append("INVENTED_BUSINESS")
    if any(_token_overlap_ratio(question, subquery) < 0.25 for subquery in subqueries):
        issues.append("LOW_SOURCE_TERM_OVERLAP")

    issues = ordered_unique(issues)
    accepted = not issues and 2 <= len(subqueries) <= config.max_subqueries
    return {
        "status": "COMPLETE" if accepted else "FAILED",
        "accepted": accepted,
        "subqueries": subqueries if accepted else [],
        "candidate_subqueries": subqueries,
        "confidence": confidence,
        "reason": reason,
        "issues": issues,
        "checks": content_checks(question, payload, expected_businesses),
    }


def content_checks(
    question: str,
    payload: Mapping[str, Any],
    expected_businesses: Sequence[str],
) -> dict[str, Any]:
    queries = extract_queries(payload)
    reconstructed = " ".join(queries)
    features = source_features(question, expected_businesses)
    generated_businesses = set(light_router.find_businesses(reconstructed))
    expected_business_set = set(features["expected_businesses"])
    generated_intents = set(find_intents(reconstructed))
    expected_intents = set(features["expected_intents"])
    numbers_preserved = all(value in reconstructed for value in features["numbers"])
    negations_preserved = all(value in reconstructed for value in features["negations"])
    subjects_preserved = all(value in reconstructed for value in features["subjects"])
    standalone = all(
        not UNRESOLVED_REFERENCE_PATTERN.search(query)
        or bool(light_router.find_businesses(query))
        for query in queries
    )
    atomic = all(len(set(light_router.find_businesses(query)) & expected_business_set) <= 1 for query in queries)
    return {
        "query_count": len(queries),
        "business_coverage": expected_business_set.issubset(generated_businesses),
        "request_coverage": expected_intents.issubset(generated_intents),
        "numeric_preservation": numbers_preserved,
        "negation_preservation": negations_preserved,
        "subject_preservation": subjects_preserved,
        "standalone_subqueries": standalone,
        "atomic_subqueries": atomic,
        "invented_business_count": len(generated_businesses - expected_business_set),
    }


def validate_quality_decomposition(
    question: str,
    payload: Mapping[str, Any],
    *,
    expected_businesses: Sequence[str],
    config: QualityConfig,
) -> dict[str, Any]:
    base = validate_baseline_decomposition(
        question, payload, expected_businesses=expected_businesses, config=config
    )
    issues = list(base.get("issues") or [])
    queries = extract_queries(payload)
    raw_items = payload.get("subqueries") if isinstance(payload.get("subqueries"), list) else []
    structured_items = [item for item in raw_items if isinstance(item, Mapping)]
    if len(structured_items) != len(raw_items):
        issues.append("MISSING_STRUCTURED_SUBQUERY_FIELDS")

    assigned_businesses: list[str] = []
    assigned_intents: list[str] = []
    pairs: list[tuple[str, str]] = []
    for item in structured_items:
        business = str(item.get("business_function") or "").strip()
        intent = str(item.get("intent") or "").strip()
        query = str(item.get("query") or "").strip()
        assigned_businesses.append(business)
        assigned_intents.append(intent)
        pairs.append((business, intent))
        if business not in expected_businesses:
            issues.append("INVENTED_OR_WRONG_ASSIGNED_BUSINESS")
        if intent not in INTENT_VALUES:
            issues.append("INVALID_ASSIGNED_INTENT")
        detected = set(light_router.find_businesses(query))
        if business and business not in detected:
            issues.append("ASSIGNED_BUSINESS_NOT_EXPLICIT_IN_QUERY")
        if len(detected & set(expected_businesses)) > 1:
            issues.append("NON_ATOMIC_SUBQUERY")
        if UNRESOLVED_REFERENCE_PATTERN.search(query) and not detected:
            issues.append("NON_STANDALONE_SUBQUERY")

    if set(expected_businesses) - set(assigned_businesses):
        issues.append("MISSING_ASSIGNED_BUSINESS_COVERAGE")
    expected_intents = set(find_intents(question))
    if expected_intents - set(assigned_intents):
        issues.append("MISSING_REQUEST_COVERAGE")
    if len(pairs) != len(set(pairs)):
        issues.append("DUPLICATE_BUSINESS_INTENT_PAIR")

    checks = content_checks(question, payload, expected_businesses)
    if not checks["request_coverage"]:
        issues.append("MISSING_REQUEST_TERMS_IN_QUERY")
    if not checks["subject_preservation"]:
        issues.append("MISSING_SUBJECT_CONSTRAINT")
    if not checks["standalone_subqueries"]:
        issues.append("NON_STANDALONE_SUBQUERY")
    if not checks["atomic_subqueries"]:
        issues.append("NON_ATOMIC_SUBQUERY")
    issues = ordered_unique(issues)
    accepted = bool(payload.get("decomposable") is True) and not issues and 2 <= len(queries) <= config.max_subqueries
    return {
        "status": "COMPLETE" if accepted else "FAILED",
        "accepted": accepted,
        "subqueries": queries if accepted else [],
        "candidate_subqueries": queries,
        "confidence": float(payload.get("confidence") or 0.0),
        "reason": str(payload.get("reason") or "").strip(),
        "issues": issues,
        "checks": checks,
    }


class HCXQualityCaller:
    def __init__(
        self,
        api_key: str,
        *,
        config: QualityConfig | None = None,
        cache_path: str | Path | None = None,
        session: Any | None = None,
    ) -> None:
        key = str(api_key or "").strip()
        if not key or key.lower().startswith("bearer ") or any(char.isspace() for char in key):
            raise ValueError("HCX_API_KEY 형식을 확인하세요.")
        self.api_key = key
        self.config = config or QualityConfig()
        self.cache_path = Path(cache_path) if cache_path else None
        if session is None and requests is None:
            raise RuntimeError("HCX 호출에는 requests 패키지가 필요합니다.")
        self.session = session or requests.Session()
        self.cache: dict[str, dict[str, Any]] = {}
        if self.cache_path and self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.cache[str(row["cache_key"])] = row

    def _append(self, row: Mapping[str, Any]) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")

    def _call(
        self,
        *,
        cache_payload: Mapping[str, Any],
        messages: Sequence[Mapping[str, str]],
        schema: Mapping[str, Any],
        validator: Callable[[Mapping[str, Any]], dict[str, Any]],
        prompt_version: str,
    ) -> dict[str, Any]:
        from kdic_lightweight_query_ablation_core import _extract_hcx_payload

        cache_key = hashlib.sha256(
            json.dumps(cache_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if cache_key in self.cache:
            row = dict(self.cache[cache_key])
            row["cache_hit"] = True
            row["actual_api_latency_ms"] = 0.0
            return row

        body = {
            "messages": list(messages),
            "topP": 0.1,
            "topK": 0,
            "maxCompletionTokens": 900,
            "temperature": 0.0,
            "repetitionPenalty": 1.0,
            "thinking": {"effort": "none"},
            "stop": [],
            "responseFormat": {"type": "json", "schema": dict(schema)},
        }
        last_error: Exception | None = None
        for attempt in range(self.config.transport_retries + 1):
            started = time.perf_counter()
            try:
                response = self.session.post(
                    self.config.llm_endpoint,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "X-NCP-CLOVASTUDIO-REQUEST-ID": str(uuid.uuid4()),
                    },
                    json=body,
                    timeout=self.config.llm_timeout_seconds,
                )
                response.raise_for_status()
                payload, usage = _extract_hcx_payload(response.json())
                validation = validator(payload)
                latency = (time.perf_counter() - started) * 1000
                row = {
                    "cache_key": cache_key,
                    "model": self.config.llm_model,
                    "prompt_version": prompt_version,
                    "raw_payload": payload,
                    **validation,
                    **usage,
                    "effective_api_latency_ms": round(latency, 3),
                    "actual_api_latency_ms": round(latency, 3),
                    "cache_hit": False,
                    "transport_attempts": attempt + 1,
                    "error_type": "",
                    "error_message": "",
                }
                self.cache[cache_key] = row
                self._append(row)
                if self.config.request_delay_seconds:
                    time.sleep(self.config.request_delay_seconds)
                return dict(row)
            except Exception as error:
                last_error = error
                if attempt < self.config.transport_retries:
                    time.sleep(min(2 ** attempt, 4))

        row = {
            "cache_key": cache_key,
            "model": self.config.llm_model,
            "prompt_version": prompt_version,
            "raw_payload": {},
            "status": "ERROR",
            "accepted": False,
            "subqueries": [],
            "candidate_subqueries": [],
            "confidence": 0.0,
            "reason": "",
            "issues": ["LLM_REQUEST_FAILED"],
            "checks": {},
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "effective_api_latency_ms": 0.0,
            "actual_api_latency_ms": 0.0,
            "cache_hit": False,
            "transport_attempts": self.config.transport_retries + 1,
            "error_type": type(last_error).__name__ if last_error else "UnknownError",
            "error_message": str(last_error or "unknown error"),
        }
        self.cache[cache_key] = row
        self._append(row)
        return dict(row)

    def quality_first(self, question: str, expected_businesses: Sequence[str]) -> dict[str, Any]:
        return self._call(
            cache_payload={
                "prompt_version": PROMPT_VERSION,
                "model": self.config.llm_model,
                "question": question,
                "expected_businesses": list(expected_businesses),
            },
            messages=build_quality_messages(question, expected_businesses),
            schema=quality_json_schema(self.config.max_subqueries),
            validator=lambda payload: validate_quality_decomposition(
                question, payload, expected_businesses=expected_businesses, config=self.config
            ),
            prompt_version=PROMPT_VERSION,
        )

    def repair(
        self,
        question: str,
        expected_businesses: Sequence[str],
        first_record: Mapping[str, Any],
        *,
        quality_mode: bool,
    ) -> dict[str, Any]:
        issues = list(first_record.get("issues") or [])
        previous_payload = dict(first_record.get("raw_payload") or {})
        if quality_mode:
            schema = quality_json_schema(self.config.max_subqueries)
            validator = lambda payload: validate_quality_decomposition(
                question, payload, expected_businesses=expected_businesses, config=self.config
            )
        else:
            schema = baseline_json_schema(self.config.max_subqueries)
            validator = lambda payload: validate_baseline_decomposition(
                question, payload,
                expected_businesses=expected_businesses,
                config=self.config,
            )
        return self._call(
            cache_payload={
                "prompt_version": REPAIR_PROMPT_VERSION,
                "model": self.config.llm_model,
                "question": question,
                "expected_businesses": list(expected_businesses),
                "quality_mode": quality_mode,
                "issues": issues,
                "previous_payload": previous_payload,
            },
            messages=build_repair_messages(
                question, expected_businesses, previous_payload, issues, quality_mode=quality_mode
            ),
            schema=schema,
            validator=validator,
            prompt_version=REPAIR_PROMPT_VERSION,
        )


def normalize_baseline_record(
    question: str,
    expected_businesses: Sequence[str],
    record: Mapping[str, Any],
) -> dict[str, Any]:
    output = dict(record)
    output.setdefault("candidate_subqueries", extract_queries(record.get("raw_payload") or {}))
    output.setdefault("checks", content_checks(question, record.get("raw_payload") or {}, expected_businesses))
    return output


def _condition_record(
    condition: str,
    first: Mapping[str, Any],
    final: Mapping[str, Any],
    *,
    retry_called: bool,
) -> dict[str, Any]:
    first_latency = float(first.get("effective_api_latency_ms") or 0.0)
    retry_latency = float(final.get("effective_api_latency_ms") or 0.0) if retry_called else 0.0
    first_tokens = int(first.get("total_tokens") or 0)
    retry_tokens = int(final.get("total_tokens") or 0) if retry_called else 0
    return {
        "condition": condition,
        "condition_label": CONDITION_LABELS[condition],
        "first_status": first.get("status"),
        "first_accepted": bool(first.get("accepted")),
        "first_confidence": float(first.get("confidence") or 0.0),
        "first_issues": list(first.get("issues") or []),
        "first_candidate_subqueries": list(first.get("candidate_subqueries") or first.get("subqueries") or []),
        "first_checks": dict(first.get("checks") or {}),
        "retry_called": retry_called,
        "retry_count": int(retry_called),
        "retry_success": bool(retry_called and final.get("accepted")),
        "retry_status": final.get("status") if retry_called else "NOT_CALLED",
        "retry_issues": list(final.get("issues") or []) if retry_called else [],
        "final_status": final.get("status"),
        "final_accepted": bool(final.get("accepted")),
        "final_confidence": float(final.get("confidence") or 0.0),
        "final_issues": list(final.get("issues") or []),
        "final_subqueries": list(final.get("subqueries") or []),
        "final_candidate_subqueries": list(final.get("candidate_subqueries") or final.get("subqueries") or []),
        "final_checks": dict(final.get("checks") or {}),
        "fallback_to_original": not bool(final.get("accepted")),
        "analysis_api_latency_ms": first_latency + retry_latency,
        "prompt_tokens": int(first.get("prompt_tokens") or 0) + (int(final.get("prompt_tokens") or 0) if retry_called else 0),
        "completion_tokens": int(first.get("completion_tokens") or 0) + (int(final.get("completion_tokens") or 0) if retry_called else 0),
        "total_tokens": first_tokens + retry_tokens,
        "api_request_count": 1 + int(retry_called),
        "first_raw_payload": dict(first.get("raw_payload") or {}),
        "retry_raw_payload": dict(final.get("raw_payload") or {}) if retry_called else {},
    }


def should_semantic_retry(record: Mapping[str, Any]) -> bool:
    """검증 가능한 생성 오류만 한 번 교정하고, 모델의 분해 거절은 존중한다."""
    status = str(record.get("status") or "").upper()
    issues = set(record.get("issues") or [])
    if bool(record.get("accepted")):
        return False
    if status in {"DECLINED", "ERROR"}:
        return False
    if "LLM_DECLINED_DECOMPOSITION" in issues or "LLM_REQUEST_FAILED" in issues:
        return False
    return True


def run_candidate_conditions(
    question: str,
    expected_businesses: Sequence[str],
    *,
    baseline_decomposer: Any,
    quality_caller: HCXQualityCaller,
) -> list[dict[str, Any]]:
    baseline_first = normalize_baseline_record(
        question, expected_businesses,
        baseline_decomposer.decompose(question, expected_businesses),
    )
    quality_first = quality_caller.quality_first(question, expected_businesses)

    if not should_semantic_retry(baseline_first):
        baseline_final = baseline_first
        baseline_retry_called = False
    else:
        baseline_final = quality_caller.repair(
            question, expected_businesses, baseline_first, quality_mode=False
        )
        baseline_retry_called = True

    if not should_semantic_retry(quality_first):
        quality_final = quality_first
        quality_retry_called = False
    else:
        quality_final = quality_caller.repair(
            question, expected_businesses, quality_first, quality_mode=True
        )
        quality_retry_called = True

    return [
        _condition_record(BASELINE, baseline_first, baseline_first, retry_called=False),
        _condition_record(QUALITY, quality_first, quality_first, retry_called=False),
        _condition_record(RETRY, baseline_first, baseline_final, retry_called=baseline_retry_called),
        _condition_record(QUALITY_RETRY, quality_first, quality_final, retry_called=quality_retry_called),
    ]


def make_query_plans(original: str, subqueries: Sequence[str], config: QualityConfig) -> list[Any]:
    from kdic_integrated_eval_core import QueryPlan

    cleaned = ordered_unique(subqueries)
    compact_original = re.sub(r"\s+", "", original).lower()
    cleaned = [item for item in cleaned if re.sub(r"\s+", "", item).lower() != compact_original]
    if len(cleaned) < 2:
        return [QueryPlan(
            need_id="FUSED", variant_id="ORIGINAL", dense_query=original, bm25_query=original,
            filter_mode="NONE", business_filters=[], soft_business_hints=[], query_weight=1.0,
            query_source="ORIGINAL_FALLBACK",
        )]
    each = config.subquery_total_weight / len(cleaned)
    plans = [QueryPlan(
        need_id="FUSED", variant_id="ORIGINAL_ANCHOR", dense_query=original, bm25_query=original,
        filter_mode="NONE", business_filters=[], soft_business_hints=[],
        query_weight=config.original_weight, query_source="ORIGINAL_ANCHOR",
    )]
    plans.extend(QueryPlan(
        need_id="FUSED", variant_id=f"SUBQUERY_{index:02d}", dense_query=query, bm25_query=query,
        filter_mode="NONE", business_filters=[], soft_business_hints=[],
        query_weight=each, query_source="DECOMPOSED",
    ) for index, query in enumerate(cleaned, 1))
    return plans


def build_condition_case(
    common: Mapping[str, Any],
    condition_record: Mapping[str, Any] | None,
    condition: str,
    *,
    config: QualityConfig,
) -> Any:
    from kdic_integrated_eval_core import AnalyzerCase

    route = str(common["route"])
    original = str(common["original_question"])
    cross_candidate = bool(common.get("complex_candidate")) and len(
        ordered_unique((common.get("complexity") or {}).get("businesses") or [])
    ) >= 2
    record = dict(condition_record or {})
    accepted = bool(cross_candidate and record.get("final_accepted"))
    subqueries = list(record.get("final_subqueries") or []) if accepted else []
    plans = make_query_plans(original, subqueries, config) if route == "RETRIEVE" else []
    analysis_latency = float(common.get("common_latency_ms") or 0.0) + float(record.get("analysis_api_latency_ms") or 0.0)
    raw_result = {
        "pipeline_version": condition,
        "analysis_status": "OK",
        "original_query": original,
        "normalized_query": common.get("normalized_question"),
        "route_reasons": common.get("route_reasons") or [],
        "complex_candidate": bool(common.get("complex_candidate")),
        "cross_business_candidate": cross_candidate,
        "businesses": (common.get("complexity") or {}).get("businesses") or [],
        "decomposition_condition": condition,
        "decomposition_record": record,
        "decomposition_source": "LLM" if accepted else ("ORIGINAL_FALLBACK" if cross_candidate else "ORIGINAL_POLICY"),
        "final_subqueries": subqueries,
        "fusion_policy": "WEIGHTED_RRF",
        "original_weight": config.original_weight if accepted else 1.0,
        "subquery_total_weight": config.subquery_total_weight if accepted else 0.0,
        "runtime": {
            "api_request_count": int(record.get("api_request_count") or 0),
            "prompt_tokens": int(record.get("prompt_tokens") or 0),
            "completion_tokens": int(record.get("completion_tokens") or 0),
            "total_tokens": int(record.get("total_tokens") or 0),
            "latency_ms": analysis_latency,
        },
    }
    return AnalyzerCase(
        evaluation_id=str(common["evaluation_id"]), analyzer=condition,
        original_question=original, route=route,
        analysis_latency_ms=round(analysis_latency, 3), plans=plans, raw_result=raw_result,
    )


def summarize_decomposition(audit_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for condition in CONDITION_ORDER:
        frame = audit_df[audit_df["condition"].eq(condition)]
        candidates = frame[frame["cross_business_candidate"]]
        true_cross = candidates[candidates["gold_cross_business"]]
        boundary = candidates[~candidates["gold_cross_business"]]
        retries = candidates[candidates["retry_called"]]
        rows.append({
            "condition": condition,
            "condition_label": CONDITION_LABELS[condition],
            "all_question_count": len(frame),
            "predicted_cross_business_count": len(candidates),
            "gold_cross_business_count": int(frame["gold_cross_business"].sum()),
            "cross_business_true_positive_count": int((frame["cross_business_candidate"] & frame["gold_cross_business"]).sum()),
            "cross_business_false_positive_count": int((frame["cross_business_candidate"] & ~frame["gold_cross_business"]).sum()),
            "cross_business_false_negative_count": int((~frame["cross_business_candidate"] & frame["gold_cross_business"]).sum()),
            "first_accept_count": int(candidates["first_accepted"].sum()),
            "final_accept_count": int(candidates["final_accepted"].sum()),
            "true_cross_first_accept_rate": float(true_cross["first_accepted"].mean()) if len(true_cross) else math.nan,
            "true_cross_final_accept_rate": float(true_cross["final_accepted"].mean()) if len(true_cross) else math.nan,
            "boundary_wrong_accept_count": int(boundary["final_accepted"].sum()),
            "retry_call_count": int(candidates["retry_called"].sum()),
            "retry_success_count": int(candidates["retry_success"].sum()),
            "retry_success_rate": float(retries["retry_success"].mean()) if len(retries) else math.nan,
            "fallback_count": int(candidates["fallback_to_original"].sum()),
            "business_coverage_rate": float(candidates["check_business_coverage"].mean()),
            "request_coverage_rate": float(candidates["check_request_coverage"].mean()),
            "constraint_preservation_rate": float((
                candidates["check_numeric_preservation"]
                & candidates["check_negation_preservation"]
                & candidates["check_subject_preservation"]
            ).mean()),
            "standalone_rate": float(candidates["check_standalone_subqueries"].mean()),
            "atomic_rate": float(candidates["check_atomic_subqueries"].mean()),
            "analysis_latency_ms_mean_all": float(frame["analysis_latency_ms"].mean()),
            "analysis_latency_ms_p95_all": float(frame["analysis_latency_ms"].quantile(.95)),
            "analysis_latency_ms_mean_candidates": float(candidates["analysis_latency_ms"].mean()),
            "total_tokens": int(frame["total_tokens"].sum()),
        })
    return pd.DataFrame(rows)


def decomposition_hard_gates(audit_df: pd.DataFrame, eval_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for condition in CONDITION_ORDER:
        frame = audit_df[audit_df["condition"].eq(condition)]
        candidates = frame[frame["cross_business_candidate"]]
        boundary = candidates[~candidates["gold_cross_business"]]
        retry_failed = candidates[candidates["retry_called"] & ~candidates["retry_success"]]
        accepted = candidates[candidates["final_accepted"]]
        gates = [
            ("실행 성공률", float(frame["execution_success"].mean()), ">=0.995", float(frame["execution_success"].mean()) >= .995),
            ("검색 질의 생성 유효율", float(frame.loc[frame["route"].eq("RETRIEVE"), "query_plan_valid"].mean()), ">=0.99", float(frame.loc[frame["route"].eq("RETRIEVE"), "query_plan_valid"].mean()) >= .99),
            ("정상 질문의 잘못된 OOS", int(frame["false_oos"].sum()), "=0", int(frame["false_oos"].sum()) == 0),
            ("정상 질문의 잘못된 DIRECT", int(frame["false_direct"].sum()), "=0", int(frame["false_direct"].sum()) == 0),
            ("Hard Filter", int(frame["hard_filter_count"].sum()), "=0", int(frame["hard_filter_count"].sum()) == 0),
            ("비교차 업무 분해 승인", int(boundary["final_accepted"].sum()), "=0", int(boundary["final_accepted"].sum()) == 0),
            ("승인 결과 검증 오류", int(accepted["final_issues"].map(bool).sum()), "=0", int(accepted["final_issues"].map(bool).sum()) == 0),
            ("재시도 최대 1회 초과", int((candidates["retry_count"] > 1).sum()), "=0", int((candidates["retry_count"] > 1).sum()) == 0),
            ("재시도 실패 후 원문 fallback", float(retry_failed["fallback_to_original"].mean()) if len(retry_failed) else 1.0, "=1.0", bool(retry_failed["fallback_to_original"].all()) if len(retry_failed) else True),
            (
                "질의 가중치 합 오류",
                int((
                    frame.loc[frame["route"].eq("RETRIEVE"), "query_plan_weight_sum"]
                    .sub(1.0).abs() > 1e-9
                ).sum()),
                "=0",
                int((
                    frame.loc[frame["route"].eq("RETRIEVE"), "query_plan_weight_sum"]
                    .sub(1.0).abs() > 1e-9
                ).sum()) == 0,
            ),
        ]
        for gate, value, threshold, passed in gates:
            rows.append({
                "condition": condition,
                "condition_label": CONDITION_LABELS[condition],
                "gate": gate,
                "value": value,
                "threshold": threshold,
                "passed": bool(passed),
            })
    return pd.DataFrame(rows)


def serialize_nested(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in output.columns:
        if output[column].map(lambda value: isinstance(value, (list, dict, tuple))).any():
            output[column] = output[column].map(
                lambda value: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (list, dict, tuple)) else value
            )
    return output
