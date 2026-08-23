from __future__ import annotations

import json
import math
import os
import random
import re
import time
import unicodedata
import uuid
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import requests

PIPELINE_VERSION = "KDIC_LIGHTWEIGHT_RAG_QUERY_ANALYZER_V1_2026_08_11"

BUSINESS_FUNCTIONS = [
    "예금자보호제도",
    "예금보험금 안내",
    "고객 미수령금 신청",
    "착오송금 반환 신청",
    "채무조정 안내",
    "은닉재산 신고",
]
INTENTS = ["AMOUNT", "ELIGIBILITY", "TIME", "APPLICATION", "OVERVIEW", "STATUS", "DOCUMENTS", "CONTACT"]
ROUTES = ["RETRIEVE", "CLARIFY", "DIRECT", "OUT_OF_SCOPE"]
APPLICANT_TYPES = ["SELF", "PROXY", "HEIR", "LEGAL_REPRESENTATIVE", "CORPORATION"]
USER_ROLES = ["DEPOSITOR", "SENDER", "RECIPIENT", "DEBTOR", "REPORTER", "CLAIMANT", "GENERAL_USER"]
MISSING_FIELDS = ["business_function", "applicant_type", "target_type", "case_details"]

INTENT_ALIASES = {
    "신청 방법": "APPLICATION", "신청 절차": "APPLICATION", "접수 방법": "APPLICATION",
    "필요 서류": "DOCUMENTS", "구비서류": "DOCUMENTS", "준비 서류": "DOCUMENTS",
    "자격": "ELIGIBILITY", "대상": "ELIGIBILITY", "조건": "ELIGIBILITY",
    "금액": "AMOUNT", "한도": "AMOUNT", "보호한도": "AMOUNT",
    "기간": "TIME", "기한": "TIME", "처리 기간": "TIME",
    "조회": "STATUS", "진행 상태": "STATUS", "처리 상태": "STATUS",
    "문의": "CONTACT", "연락처": "CONTACT",
    "안내": "OVERVIEW", "개요": "OVERVIEW", "무엇": "OVERVIEW",
}

BUSINESS_KEYWORDS = {
    "예금자보호제도": ["예금자보호", "보호한도", "보호대상", "예금 보호"],
    "예금보험금 안내": ["예금보험금", "보험금 지급", "보험사고", "가지급금"],
    "고객 미수령금 신청": ["미수령금", "파산배당금", "개산지급금 정산금", "상속인 금융거래 조회"],
    "착오송금 반환 신청": ["착오송금", "잘못 보낸 돈", "잘못 송금", "착오 송금"],
    "채무조정 안내": ["채무조정", "신용회복지원", "파산선고", "면책", "채무감면"],
    "은닉재산 신고": ["은닉재산", "은닉 재산"],
}

EXPLICIT_TYPO_MAP = (
    ("예금보헝금", "예금보험금"),
    ("예금보혐금", "예금보험금"),
    ("착오송금반한", "착오송금 반환"),
    ("미수령금신정", "미수령금 신청"),
)

@dataclass(frozen=True)
class PipelineConfig:
    model: str = "HCX-007"
    base_url: str = "https://clovastudio.stream.ntruss.com"
    timeout_seconds: float = 90.0
    max_api_attempts: int = 2
    max_completion_tokens: int = 1400
    temperature: float = 0.1
    top_p: float = 0.8
    top_k: int = 0
    request_interval_seconds: float = 0.3
    max_context_turns: int = 2
    intent_soft_boost: float = 0.15

class HCXAPIError(RuntimeError):
    def __init__(self, message: str, *, error_type: str, telemetry: dict[str, Any]):
        super().__init__(message)
        self.error_type = error_type
        self.telemetry = telemetry

def get_hcx_api_key(secret_name: str = "HCX_API_KEY") -> str:
    try:
        from google.colab import userdata
        value = str(userdata.get(secret_name) or "").strip()
    except ImportError:
        value = os.getenv(secret_name, "").strip()
    if not value:
        raise ValueError(f"Colab Secrets 또는 환경변수에 {secret_name}가 없습니다.")
    if value.lower().startswith("bearer ") or any(ch.isspace() for ch in value):
        raise ValueError(f"{secret_name}에는 Bearer 접두사 없이 API 키 값만 저장하세요.")
    return value


FOLLOW_UP_PATTERN = re.compile(r"(?:그럼|그러면|그거|그건|그 경우|이거|이건|이 경우|앞서|방금|그때|그것|그 서류|그 신청)")

def normalize_query(text: str) -> dict[str, Any]:
    original = str(text or "")
    value = unicodedata.normalize("NFKC", original)
    changes = []
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", value)
    if cleaned != value:
        changes.append("CONTROL_CHARACTER")
        value = cleaned
    cleaned = re.sub(r"([!?ㅋㅎㅠㅜ])\1{2,}", r"\1\1", value)
    if cleaned != value:
        changes.append("REPEATED_CHARACTER")
        value = cleaned
    for wrong, correct in EXPLICIT_TYPO_MAP:
        if wrong in value:
            value = value.replace(wrong, correct)
            changes.append("EXPLICIT_TYPO")
    cleaned = re.sub(r"\s+", " ", value).strip()
    if cleaned != value:
        changes.append("WHITESPACE")
    if not cleaned:
        raise ValueError("사용자 질의가 비어 있습니다.")
    return {"original_query": original, "normalized_query": cleaned, "changes": changes}

def build_context(query: str, conversation_state: dict[str, Any] | None, max_turns: int) -> dict[str, Any]:
    state = dict(conversation_state or {})
    if not FOLLOW_UP_PATTERN.search(query):
        return {"used": False, "confirmed": {}, "recent_turns": []}
    confirmed = state.get("confirmed") if isinstance(state.get("confirmed"), dict) else {}
    turns = state.get("recent_turns") if isinstance(state.get("recent_turns"), list) else []
    return {"used": True, "confirmed": confirmed, "recent_turns": turns[-max_turns:]}

def exact_fullmatch(pattern: str, query: str) -> bool:
    return re.fullmatch(pattern, query.strip(), flags=re.I) is not None

def detect_fast_path(query: str, conversation_state: dict[str, Any] | None = None) -> dict[str, Any] | None:
    # 문장 전체가 규칙에 일치할 때만 처리해 실제 질문을 잘라내지 않는다.
    detectors = []
    if exact_fullmatch(r"(?:안녕|안녕하세요|반갑습니다|반가워요)[.!?]*", query):
        detectors.append({"route": "DIRECT", "action": "GREETING"})
    if exact_fullmatch(r"(?:고마워요|고맙습니다|감사합니다|도움이 됐어요|알겠습니다)[.!?]*", query):
        detectors.append({"route": "DIRECT", "action": "ACKNOWLEDGEMENT"})
    if exact_fullmatch(r"(?:무슨 질문을 할 수 있나요|지원하는 업무를 (?:알려주세요|목록으로 보여주세요)|이 챗봇은 어떻게 사용하면 되나요)[.!?]*", query):
        detectors.append({"route": "DIRECT", "action": "CAPABILITY_GUIDE"})
    has_previous = bool((conversation_state or {}).get("has_previous_answer"))
    if has_previous and exact_fullmatch(r"(?:쉽게 설명해 주세요|핵심만 알려주세요|표로 정리해 주세요|더 자세히 설명해 주세요)[.!?]*", query):
        detectors.append({"route": "DIRECT", "action": "REFORMAT_PREVIOUS_ANSWER"})
    if exact_fullmatch(r"(?:오늘|내일|이번 주말)?\s*(?:서울 )?(?:날씨|기온|미세먼지)(?:를|가|은|는)?.*", query):
        detectors.append({"route": "OUT_OF_SCOPE", "action": "EXPLICIT_WEATHER"})
    if exact_fullmatch(r"(?:주식 종목을 추천해 주세요|로또 번호를 알려주세요)[.!?]*", query):
        detectors.append({"route": "OUT_OF_SCOPE", "action": "EXPLICIT_NON_KDIC"})
    return detectors[0] if len(detectors) == 1 else None

def find_businesses(text: str) -> list[str]:
    compact = re.sub(r"\s+", "", text)
    found = []
    for business, keywords in BUSINESS_KEYWORDS.items():
        if any(re.sub(r"\s+", "", keyword) in compact for keyword in keywords):
            found.append(business)
    return found

def normalize_intent(value: Any) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    if text in INTENTS:
        return text
    if text.upper() in INTENTS:
        return text.upper()
    return INTENT_ALIASES.get(text)


def query_analysis_schema() -> dict[str, Any]:
    # HCX-007 공식 지원 타입에 null이 없으므로 UNKNOWN/빈 문자열을 sentinel로 쓴다.
    business_value = {"type": "string", "enum": [*BUSINESS_FUNCTIONS, "UNKNOWN"]}
    intent_value = {"type": "string", "enum": [*INTENTS, "UNKNOWN"]}
    applicant_value = {"type": "string", "enum": [*APPLICANT_TYPES, "UNKNOWN"]}
    user_role_value = {"type": "string", "enum": [*USER_ROLES, "UNKNOWN"]}
    return {
        "type": "object",
        "properties": {
            "route": {"type": "string", "enum": ROUTES},
            "needs": {
                "type": "array",
                "maxItems": 6,
                "items": {
                    "type": "object",
                    "properties": {
                        "need_id": {"type": "string"},
                        "query": {"type": "string"},
                        "business_function": business_value,
                        "business_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "intent": intent_value,
                        "intent_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "user_role": user_role_value,
                        "applicant_type": applicant_value,
                        "target_type": {"type": "string"},
                        "case_details": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "need_id", "query", "business_function", "business_confidence",
                        "intent", "intent_confidence", "user_role",
                        "applicant_type", "target_type", "case_details",
                    ],
                },
            },
            "missing_information": {"type": "array", "items": {"type": "string", "enum": MISSING_FIELDS}},
        },
        "required": ["route", "needs", "missing_information"],
    }

SYSTEM_PROMPT = f"""
역할: 예금보험공사 RAG 시스템의 경량 질의 분석기.
목적: 사용자 요구를 검색 가능한 최소 Need로 나누고 각 Need의 검색어·업무·의도·핵심 조건을 반환한다.

규칙:
1. 서로 다른 정보 요구는 N1, N2, ...로 분리한다.
2. query는 해당 Need를 독립적으로 검색할 수 있는 한국어 문장으로 쓴다.
3. 원문이 이미 독립적이면 불필요하게 바꾸지 않는다.
4. 확정 context는 후속 질문일 때만 사용한다.
5. 근거가 없는 범주형 값은 UNKNOWN, target_type은 빈 문자열, 목록은 []로 두고 추측하지 않는다.
   business_confidence와 intent_confidence는 각각 해당 분류가 맞을 확률을 0~1로 쓴다.
6. 업무가 불확실해도 원문으로 검색 가능하면 RETRIEVE다.
7. 사용자만 제공할 수 있는 필수 정보가 없어 정답 대상이 바뀐 때만 CLARIFY다.
8. 인사·감사·사용법·이전 답변 재구성은 DIRECT다.
9. 예금보험공사 업무와 명확히 무관한 요청은 OUT_OF_SCOPE다.

허용 business_function: {BUSINESS_FUNCTIONS}
허용 intent: {INTENTS}
허용 user_role: {USER_ROLES}
허용 applicant_type: {APPLICANT_TYPES}
"""

class HCX007StructuredClient:
    def __init__(self, api_key: str, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.api_key = api_key
        self.session = requests.Session()

    @property
    def url(self) -> str:
        return f"{self.config.base_url}/v3/chat-completions/{self.config.model}"

    def analyze(self, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        started = time.perf_counter()
        attempts = []
        total_tokens = 0
        last_error = None
        body = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "topP": self.config.top_p,
            "topK": self.config.top_k,
            "maxCompletionTokens": self.config.max_completion_tokens,
            "temperature": self.config.temperature,
            "repetitionPenalty": 1.05,
            "thinking": {"effort": "none"},
            "stop": [],
            "responseFormat": {"type": "json", "schema": query_analysis_schema()},
        }
        for attempt in range(1, self.config.max_api_attempts + 1):
            attempt_started = time.perf_counter()
            request_id = str(uuid.uuid4())
            try:
                response = self.session.post(
                    self.url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "X-NCP-CLOVASTUDIO-REQUEST-ID": request_id,
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                    json=body,
                    timeout=self.config.timeout_seconds,
                )
                if response.status_code >= 400:
                    try:
                        error_body = response.json()
                    except Exception:
                        error_body = {"text": response.text[:1000]}
                    retryable = response.status_code in {408, 429, 500, 502, 503, 504}
                    attempts.append({
                        "attempt": attempt, "request_id": request_id, "http_status": response.status_code,
                        "retryable": retryable, "error": error_body,
                        "latency_ms": round((time.perf_counter() - attempt_started) * 1000, 3),
                    })
                    last_error = f"HTTP {response.status_code}: {error_body}"
                    if retryable and attempt < self.config.max_api_attempts:
                        retry_after = response.headers.get("Retry-After")
                        delay = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else 1.5 + random.random()
                        time.sleep(min(delay, 10.0))
                        continue
                    raise HCXAPIError(last_error, error_type="API_RETRYABLE" if retryable else "API_FATAL", telemetry={
                        "api_request_count": len(attempts), "attempts": attempts, "total_tokens": total_tokens,
                        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                    })

                envelope = response.json()
                result = envelope.get("result", envelope)
                usage = result.get("usage") or {}
                used = int(usage.get("totalTokens") or 0)
                total_tokens += used
                content = str((result.get("message") or {}).get("content") or "")
                parsed = json.loads(content)
                if not isinstance(parsed, dict):
                    raise TypeError("모델 결과의 최상위가 object가 아닙니다.")
                attempts.append({
                    "attempt": attempt, "request_id": request_id, "http_status": response.status_code,
                    "tokens": used, "finish_reason": result.get("finishReason"),
                    "latency_ms": round((time.perf_counter() - attempt_started) * 1000, 3),
                })
                return parsed, {
                    "api_request_count": len(attempts), "attempts": attempts, "total_tokens": total_tokens,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                }
            except HCXAPIError:
                raise
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                attempts.append({
                    "attempt": attempt, "request_id": request_id, "retryable": True, "error": last_error,
                    "latency_ms": round((time.perf_counter() - attempt_started) * 1000, 3),
                })
                if attempt < self.config.max_api_attempts:
                    time.sleep(1.5 + random.random())
                    continue
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                attempts.append({
                    "attempt": attempt, "request_id": request_id, "retryable": False, "error": last_error,
                    "latency_ms": round((time.perf_counter() - attempt_started) * 1000, 3),
                })
                break
        raise HCXAPIError(last_error or "HCX-007 호출 실패", error_type="API_OR_PARSE_ERROR", telemetry={
            "api_request_count": len(attempts), "attempts": attempts, "total_tokens": total_tokens,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        })


def validate_analysis(raw: dict[str, Any], normalized_query: str) -> tuple[dict[str, Any], list[str]]:
    warnings = []
    route = raw.get("route")
    if route not in ROUTES:
        route = "RETRIEVE"
        warnings.append("INVALID_ROUTE_TO_RETRIEVE")
    rows = raw.get("needs")
    if not isinstance(rows, list):
        rows = []
        warnings.append("NEEDS_NOT_LIST")
    needs = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            warnings.append(f"N{index}_NOT_OBJECT")
            continue
        query = str(row.get("query") or "").strip()
        if not query:
            query = normalized_query
            warnings.append(f"N{index}_EMPTY_QUERY_USED_ORIGINAL")
        business = row.get("business_function")
        if business not in BUSINESS_FUNCTIONS:
            if business not in (None, "", "UNKNOWN"):
                warnings.append(f"N{index}_INVALID_BUSINESS_TO_NULL")
            business = None
        intent = normalize_intent(row.get("intent"))
        if row.get("intent") not in (None, "", "UNKNOWN") and intent is None:
            warnings.append(f"N{index}_INVALID_INTENT_TO_NULL")
        applicant = row.get("applicant_type")
        if applicant not in APPLICANT_TYPES:
            applicant = None
        user_role = row.get("user_role")
        if user_role not in USER_ROLES:
            user_role = None
        try:
            business_confidence = min(1.0, max(0.0, float(row.get("business_confidence") or 0.0)))
        except (TypeError, ValueError):
            business_confidence = 0.0
            warnings.append(f"N{index}_INVALID_BUSINESS_CONFIDENCE_TO_ZERO")
        try:
            intent_confidence = min(1.0, max(0.0, float(row.get("intent_confidence") or 0.0)))
        except (TypeError, ValueError):
            intent_confidence = 0.0
            warnings.append(f"N{index}_INVALID_INTENT_CONFIDENCE_TO_ZERO")
        target = row.get("target_type")
        target = str(target).strip() if target not in (None, "") else None
        cases = row.get("case_details")
        cases = [str(x).strip() for x in cases if str(x).strip()] if isinstance(cases, list) else []
        needs.append({
            "need_id": f"N{index}", "query": query, "business_function": business,
            "business_confidence": business_confidence,
            "intent": intent, "intent_confidence": intent_confidence,
            "user_role": user_role, "applicant_type": applicant, "target_type": target,
            "case_details": cases,
        })
    if route == "RETRIEVE" and not needs:
        needs = [{
            "need_id": "N1", "query": normalized_query, "business_function": None,
            "business_confidence": 0.0, "intent": None, "intent_confidence": 0.0,
            "user_role": None, "applicant_type": None, "target_type": None, "case_details": [],
        }]
        warnings.append("EMPTY_RETRIEVE_NEEDS_USED_ORIGINAL")
    if route in {"DIRECT", "OUT_OF_SCOPE"}:
        needs = []
    missing = raw.get("missing_information")
    missing = [x for x in missing if x in MISSING_FIELDS] if isinstance(missing, list) else []
    return {"route": route, "needs": needs, "missing_information": missing}, warnings

def build_keyword_query(need: dict[str, Any]) -> str:
    values = [
        need.get("business_function"), need.get("intent"), need.get("user_role"),
        need.get("applicant_type"), need.get("target_type"),
    ]
    values.extend(need.get("case_details") or [])
    return " ".join(str(x) for x in values if x)

def determine_filter_policy(need: dict[str, Any], original: str, context: dict[str, Any], manual: dict[str, Any]) -> dict[str, Any]:
    business = need.get("business_function")
    if business not in BUSINESS_FUNCTIONS:
        return {"mode": "NONE", "value": None, "soft_hint": None, "evidence": "UNKNOWN"}
    if manual.get("business_function") == business:
        return {"mode": "HARD", "value": business, "soft_hint": None, "evidence": "MANUAL"}
    explicit = business in find_businesses(original)
    if explicit:
        return {"mode": "HARD", "value": business, "soft_hint": None, "evidence": "ORIGINAL"}
    if context.get("confirmed", {}).get("business_function") == business:
        return {"mode": "HARD", "value": business, "soft_hint": None, "evidence": "CONTEXT"}
    if float(need.get("business_confidence") or 0.0) >= 0.65:
        return {"mode": "SOFT", "value": None, "soft_hint": business, "evidence": "MODEL"}
    return {"mode": "NONE", "value": None, "soft_hint": None, "evidence": "LOW_CONFIDENCE_MODEL"}

def build_query_plans(analysis: dict[str, Any], original: str, context: dict[str, Any], manual: dict[str, Any], config: PipelineConfig) -> list[dict[str, Any]]:
    if analysis["route"] != "RETRIEVE":
        return []
    plans = []
    for need in analysis["needs"]:
        policy = determine_filter_policy(need, original, context, manual)
        plans.append({
            "need_id": need["need_id"],
            "semantic_query": need["query"],
            "keyword_query": build_keyword_query(need) or need["query"],
            "business_filter": policy,
            "intent_boost": {
                "mode": "SOFT" if need.get("intent") in INTENTS else "NONE",
                "value": need.get("intent"),
                "weight": config.intent_soft_boost if need.get("intent") in INTENTS else 0.0,
            },
            "entities": {
                "user_role": need.get("user_role"),
                "applicant_type": need.get("applicant_type"),
                "target_type": need.get("target_type"),
                "case_details": need.get("case_details") or [],
            },
        })
    return plans

GENERIC_CLARIFY_PATTERN = re.compile(r"^(?:신청 방법|필요한 서류|제출해야 하는 서류|신청 기한|처리 기간|문의처)(?:을|가|은|는|이|가)?(?: 어떻게 되나요| 언제까지인가요| 알려주세요| 무엇인가요)?[.!?]*$")

def fallback_analysis(normalized_query: str, context: dict[str, Any], reason: str) -> dict[str, Any]:
    businesses = find_businesses(normalized_query)
    confirmed_business = context.get("confirmed", {}).get("business_function")
    if not businesses and not confirmed_business and GENERIC_CLARIFY_PATTERN.fullmatch(normalized_query):
        return {
            "route": "CLARIFY", "needs": [], "missing_information": ["business_function"],
            "fallback_reason": reason,
        }
    return {
        "route": "RETRIEVE",
        "needs": [{
            "need_id": "N1", "query": normalized_query,
            "business_function": businesses[0] if len(businesses) == 1 else confirmed_business,
            "business_confidence": 1.0 if len(businesses) == 1 else 0.0,
            "intent": None, "intent_confidence": 0.0,
            "user_role": None, "applicant_type": None, "target_type": None, "case_details": [],
        }],
        "missing_information": [], "fallback_reason": reason,
    }

class KDICLightweightRAGAnalyzer:
    def __init__(self, client: HCX007StructuredClient, config: PipelineConfig | None = None):
        self.client = client
        self.config = config or client.config

    def run(self, query: str, *, conversation_state=None, manual_selection=None) -> dict[str, Any]:
        started = time.perf_counter()
        normalized = normalize_query(query)
        original = normalized["original_query"]
        text = normalized["normalized_query"]
        manual = dict(manual_selection or {})
        context = build_context(text, conversation_state, self.config.max_context_turns)

        fast = detect_fast_path(text, conversation_state)
        if fast:
            analysis = {"route": fast["route"], "needs": [], "missing_information": []}
            return {
                "pipeline_version": PIPELINE_VERSION, "analysis_status": "FAST_PATH",
                "original_query": original, "normalized_query": text, "context": context,
                "analysis": analysis, "fast_path": fast, "query_plans": [],
                "runtime": {
                    "api_request_count": 0, "total_tokens": 0,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 3), "attempts": [],
                },
            }

        payload = {
            "query": text,
            "confirmed_context": context["confirmed"],
            "recent_turns": context["recent_turns"],
            "manual_selection": manual,
        }
        try:
            raw, telemetry = self.client.analyze(payload)
            analysis, warnings = validate_analysis(raw, text)
            status = "REPAIRED" if warnings else "OK"
        except HCXAPIError as exc:
            telemetry = exc.telemetry
            analysis = fallback_analysis(text, context, f"{exc.error_type}: {exc}")
            warnings = ["MODEL_ANALYSIS_FAILED_USED_FALLBACK"]
            status = "FALLBACK"

        plans = build_query_plans(analysis, original, context, manual, self.config)
        return {
            "pipeline_version": PIPELINE_VERSION, "analysis_status": status,
            "original_query": original, "normalized_query": text, "context": context,
            "analysis": analysis, "validation_warnings": warnings, "query_plans": plans,
            "runtime": {
                **telemetry,
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            },
        }


PIPELINE_VERSION = "KDIC_LIGHTWEIGHT_RAG_QUERY_ANALYZER_V3_1_2026_08_12"
HARD_BUSINESS_CONFIDENCE_THRESHOLD = 0.98
HARD_BUSINESS_MARGIN_THRESHOLD = 0.20
V3_ROUTES = ["RETRIEVE", "RETRIEVE_RELAXED", "CLARIFY", "DIRECT", "OUT_OF_SCOPE"]

BUSINESS_KEYWORDS = {
    "예금자보호제도": ["예금자보호", "보호한도", "보호대상", "예금 보호", "보호 한도"],
    "예금보험금 안내": ["예금보험금", "보험금 지급", "보험사고", "가지급금", "개산지급금", "1종 보험사고", "2종 보험사고"],
    "고객 미수령금 신청": ["미수령금", "파산배당금", "개산지급금 정산금", "지급대행점", "상속인 금융거래 조회", "상속인 금융거래 조회서비스"],
    "착오송금 반환 신청": ["착오송금", "잘못 보낸 돈", "잘못 송금", "착오 송금", "반환지원", "매입계약", "지급명령", "강제집행", "송금인", "수취인"],
    "채무조정 안내": ["채무조정", "신용회복지원", "파산선고", "면책", "채무감면", "개인회생", "개인파산", "워크아웃", "변제기간", "부채증명원", "채무정보"],
    "은닉재산 신고": ["은닉재산", "은닉 재산", "금융부실관련자", "부실관련자", "차명 재산", "차명재산", "신고 포상금"],
}

DIRECT_META_PATTERN = re.compile(
    r"^(?:안녕|안녕하세요|반갑습니다|반가워요|고마워요|고맙습니다|감사합니다|도움이 됐어요|"
    r"알겠습니다|무슨 질문을 할 수 있나요|지원하는 업무를 (?:알려주세요|목록으로 보여주세요)|"
    r"이 챗봇은 어떻게 사용하면 되나요|답변을 쉽게 설명해 줄 수 있나요|"
    r"전문가 수준으로 자세히 설명해 주세요|긴 설명보다 핵심 내용만 먼저 알려주세요|"
    r"질문을 잘못 입력했어요[.]? 다시 물어볼게요)[.!?]*$", re.I,
)
OOS_EXACT_PATTERN = re.compile(
    r".*(?:코스피|비트코인|주택담보대출 금리|신용점수|실손보험|국민연금|보이스피싱|"
    r"은행 계좌를 새로|해외송금 수수료|카드 결제|전세대출|세금 환급|퇴직금|"
    r"개인정보 유출|상속세|환율|주식 투자|신용카드 연회비|사업자등록|서울 날씨|"
    r"이번 주말.*날씨).*", re.I,
)
USER_ROLE_PATTERNS = [
    ("SENDER", r"송금인|돈을 보낸 사람"), ("RECIPIENT", r"수취인|돈을 받은 사람"),
    ("DEBTOR", r"채무자"), ("REPORTER", r"신고자|신고인"),
    ("CLAIMANT", r"청구인"), ("DEPOSITOR", r"예금자(?!보호)"),
]
APPLICANT_PATTERNS = [
    ("LEGAL_REPRESENTATIVE", r"법정대리인|후견인"),
    ("PROXY", r"대리인|대신해 신청|대리 신청|위임받"),
    ("HEIR", r"상속인"), ("CORPORATION", r"법인(?: 명의|이|으로| 신청)"),
    ("SELF", r"본인이 직접|본인 신청|제가 직접|직접 신청"),
]

def extract_explicit_value(text: str, patterns: list[tuple[str, str]]) -> str | None:
    for value, pattern in patterns:
        if re.search(pattern, text, flags=re.I):
            return value
    return None

def detect_fast_path_v3(query: str, conversation_state: dict[str, Any] | None = None) -> dict[str, Any] | None:
    text = query.strip()
    if DIRECT_META_PATTERN.fullmatch(text):
        return {"route": "DIRECT", "action": "META_OR_SOCIAL"}
    has_previous = bool((conversation_state or {}).get("has_previous_answer"))
    if has_previous and exact_fullmatch(
        r"(?:쉽게 설명해 주세요|핵심만 알려주세요|표로 정리해 주세요|더 자세히 설명해 주세요)[.!?]*", text
    ):
        return {"route": "DIRECT", "action": "REFORMAT_PREVIOUS_ANSWER"}
    if OOS_EXACT_PATTERN.fullmatch(text) and not find_businesses(text):
        return {"route": "OUT_OF_SCOPE", "action": "EXPLICIT_NON_KDIC"}
    return None

# Intent 규칙은 단어가 아니라 완성 구문을 우선한다.
HIGH_PRECISION_INTENT_RULES = [
    ("DOCUMENTS", [r"필요한?\s*(?:서류|증빙)", r"제출(?:해야 하는|할)?\s*서류", r"(?:서류|증빙)(?:가|는|은)?\s*무엇", r"구비\s*서류", r"신분증", r"위임장", r"준비\s*서류"]),
    ("STATUS", [r"어디(?:서|에서)\s*(?:확인|조회)", r"조회\s*(?:방법|결과)", r"처리\s*결과", r"진행\s*상황", r"지급\s*정보.*보는\s*방법"]),
    ("APPLICATION", [r"신청\s*(?:방법|절차)", r"접수\s*(?:방법|절차)", r"제출\s*방법", r"신고\s*채널", r"어떻게\s*(?:신청|접수)", r"(?:온라인|방문|직접).*신청.*(?:가능|할\s*수)", r"취소.*방법", r"철회.*방법"]),
    ("TIME", [r"언제(?:부터|까지)", r"신청.*(?:기한|기간)", r"처리\s*기간", r"소요\s*(?:기간|시간)", r"얼마나\s*걸"]),
    ("AMOUNT", [r"보호\s*한도", r"지급\s*금액", r"금액\s*계산", r"금액.*얼마(?:여야|이어야)", r"계산\s*(?:기준|방법)", r"수수료\s*(?:금액|비용)", r"얼마나\s*(?:감면|지급|보상|돌려|받)"]),
    ("CONTACT", [r"연락처", r"전화번호", r"문의처", r"어디로\s*연락", r"어느\s*기관.*문의"]),
    ("ELIGIBILITY", [r"신청\s*(?:대상|자격|요건)", r"가능한\s*대상", r"제외되는?\s*경우", r"받을\s*수\s*있", r"신청할\s*수\s*있", r"포함되", r"어떤\s*경우.*(?:지급|지원|보호)", r"(?:대상|자격)에\s*해당", r"(?:예금|계좌|금융상품|상품|원금|이자).{0,20}보호(?:가)?\s*되", r"지원\s*대상"]),
    ("OVERVIEW", [r"무엇(?:인가요|인지)", r"의미", r"정의", r"차이", r"종류", r"개요", r"설명"]),
]
WEAK_INTENT_RULES = [
    ("DOCUMENTS", [r"서류", r"증빙"]), ("STATUS", [r"조회", r"확인"]),
    ("APPLICATION", [r"신청", r"접수", r"절차", r"제출"]),
    ("TIME", [r"기간", r"기한", r"시점"]), ("AMOUNT", [r"한도", r"금액", r"계산", r"비용", r"포상금"]),
    ("CONTACT", [r"연락", r"문의", r"전화"]), ("ELIGIBILITY", [r"대상", r"자격", r"요건", r"조건", r"가능"]),
    ("OVERVIEW", [r"설명", r"관계", r"방식"]),
]

def match_intent(text: str, rules: list[tuple[str, list[str]]]) -> tuple[str | None, str | None]:
    for intent, patterns in rules:
        for pattern in patterns:
            if re.search(pattern, text, flags=re.I):
                return intent, pattern
    return None, None

def match_all_intents(text: str, rules: list[tuple[str, list[str]]]) -> list[tuple[str, str]]:
    matches = []
    for intent, patterns in rules:
        for pattern in patterns:
            if re.search(pattern, text, flags=re.I):
                matches.append((intent, pattern))
                break
    return matches

def resolve_intent_v3(text: str, model_intent: str | None) -> tuple[str | None, str, str | None]:
    high_matches = match_all_intents(text, HIGH_PRECISION_INTENT_RULES)
    high_intents = list(dict.fromkeys(intent for intent, _ in high_matches))
    if len(high_intents) > 1:
        if model_intent in INTENTS:
            pattern = next((p for i, p in high_matches if i == model_intent), high_matches[0][1])
            return model_intent, "RULE_AMBIGUOUS_MODEL_KEPT", pattern
        return None, "RULE_AMBIGUOUS_UNKNOWN", high_matches[0][1]
    if len(high_intents) == 1:
        high, pattern = high_matches[0]
        if model_intent == high:
            return high, "RULE_CONFIRMED", pattern
        return high, "RULE_OVERRIDE", pattern
    if model_intent in INTENTS:
        weak, weak_pattern = match_intent(text, WEAK_INTENT_RULES)
        if weak and weak != model_intent:
            return model_intent, "RULE_CONFLICT_MODEL_KEPT", weak_pattern
        return model_intent, "MODEL", None
    weak, weak_pattern = match_intent(text, WEAK_INTENT_RULES)
    return weak, "RULE_FILLED_UNKNOWN" if weak else "UNKNOWN", weak_pattern

def query_analysis_schema_v3() -> dict[str, Any]:
    business_value = {"type": "string", "enum": [*BUSINESS_FUNCTIONS, "UNKNOWN"]}
    intent_value = {"type": "string", "enum": [*INTENTS, "UNKNOWN"]}
    return {
        "type": "object",
        "properties": {
            "needs": {
                "type": "array", "maxItems": 6,
                "items": {
                    "type": "object",
                    "properties": {
                        "need_id": {"type": "string"}, "query": {"type": "string"},
                        "business_function": business_value, "intent": intent_value,
                        "target_type": {"type": "string"},
                        "case_details": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["need_id", "query", "business_function", "intent", "target_type", "case_details"],
                },
            },
        },
        "required": ["needs"],
    }

SYSTEM_PROMPT_V3 = f"""
역할: 예금보험공사 RAG의 Atomic Need 분석기.
출력: JSON Schema만 준수한다. Route와 확인 질문은 결정하지 않는다.

업무 사전:
- 예금자보호제도: 보호대상, 보호한도, 금융상품 보호 여부
- 예금보험금 안내: 보험사고, 예금보험금, 가지급금, 개산지급금
- 고객 미수령금 신청: 미수령금, 파산배당금, 정산금, 지급대행점, 상속인 금융거래 조회
- 착오송금 반환 신청: 반환지원, 매입계약, 지급명령, 강제집행
- 채무조정 안내: 개인회생, 개인파산, 워크아웃, 신용회복지원, 면책, 부채증명원
- 은닉재산 신고: 금융부실관련자, 차명재산, 신고, 포상금

규칙:
1. 사용자에게 다시 질문하지 말고 현재 입력에서 검색할 Need를 최대한 생성한다.
2. 서로 다른 정보 요구는 N1, N2로 분리한다. 서류와 제출방법은 서로 다른 Need다.
3. 같은 Intent여도 검색 대상·상황이 다르면 분리한다.
4. 비교 질문은 대상별 Cartesian 분리보다 정보 차원별로 분리한다.
   예: 개인회생과 워크아웃의 조건과 변제방식 → 조건 비교, 변제방식 비교의 두 Need.
5. query는 해당 Need만 독립 검색 가능한 문장으로 쓰고 원문 의미를 보존한다.
6. 원문이 이미 독립적이면 불필요하게 재작성하지 않는다.
7. 업무나 Intent가 불확실하면 UNKNOWN을 사용하되 Need를 삭제하지 않는다.
8. target_type과 case_details는 원문에 명시된 검색 조건만 기록한다.

허용 business_function: {BUSINESS_FUNCTIONS}
허용 intent: {INTENTS}
"""

class HCX007AtomicNeedClientV3(HCX007StructuredClient):
    def analyze(self, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        started = time.perf_counter()
        attempts = []
        totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        last_error = None
        body = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_V3},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "topP": self.config.top_p, "topK": self.config.top_k,
            "maxCompletionTokens": self.config.max_completion_tokens,
            "temperature": self.config.temperature, "repetitionPenalty": 1.05,
            "thinking": {"effort": "none"}, "stop": [],
            "responseFormat": {"type": "json", "schema": query_analysis_schema_v3()},
        }
        for attempt in range(1, self.config.max_api_attempts + 1):
            attempt_started = time.perf_counter()
            request_id = str(uuid.uuid4())
            try:
                response = self.session.post(
                    self.url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "X-NCP-CLOVASTUDIO-REQUEST-ID": request_id,
                        "Content-Type": "application/json", "Accept": "application/json",
                    },
                    json=body, timeout=self.config.timeout_seconds,
                )
                if response.status_code >= 400:
                    try:
                        error_body = response.json()
                    except Exception:
                        error_body = {"text": response.text[:1000]}
                    retryable = response.status_code in {408, 429, 500, 502, 503, 504}
                    attempts.append({
                        "attempt": attempt, "request_id": request_id, "http_status": response.status_code,
                        "retryable": retryable, "error": error_body,
                        "latency_ms": round((time.perf_counter() - attempt_started) * 1000, 3),
                    })
                    last_error = f"HTTP {response.status_code}: {error_body}"
                    if retryable and attempt < self.config.max_api_attempts:
                        retry_after = response.headers.get("Retry-After")
                        delay = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else 1.5 + random.random()
                        time.sleep(min(delay, 10.0))
                        continue
                    raise HCXAPIError(last_error, error_type="API_RETRYABLE" if retryable else "API_FATAL", telemetry={
                        "api_request_count": len(attempts), "attempts": attempts, **totals,
                        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                    })
                envelope = response.json()
                result = envelope.get("result", envelope)
                usage = result.get("usage") or {}
                used = {
                    "prompt_tokens": int(usage.get("promptTokens") or 0),
                    "completion_tokens": int(usage.get("completionTokens") or 0),
                    "total_tokens": int(usage.get("totalTokens") or 0),
                }
                for key in totals:
                    totals[key] += used[key]
                content = str((result.get("message") or {}).get("content") or "")
                parsed = json.loads(content)
                if not isinstance(parsed, dict):
                    raise TypeError("모델 결과의 최상위가 object가 아닙니다.")
                attempts.append({
                    "attempt": attempt, "request_id": request_id, "http_status": response.status_code,
                    **used, "finish_reason": result.get("finishReason"),
                    "latency_ms": round((time.perf_counter() - attempt_started) * 1000, 3),
                })
                return parsed, {
                    "api_request_count": len(attempts), "attempts": attempts, **totals,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                }
            except HCXAPIError:
                raise
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                attempts.append({
                    "attempt": attempt, "request_id": request_id, "retryable": True, "error": last_error,
                    "latency_ms": round((time.perf_counter() - attempt_started) * 1000, 3),
                })
                if attempt < self.config.max_api_attempts:
                    time.sleep(1.5 + random.random())
                    continue
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                attempts.append({
                    "attempt": attempt, "request_id": request_id, "retryable": False, "error": last_error,
                    "latency_ms": round((time.perf_counter() - attempt_started) * 1000, 3),
                })
                break
        raise HCXAPIError(last_error or "HCX-007 호출 실패", error_type="API_OR_PARSE_ERROR", telemetry={
            "api_request_count": len(attempts), "attempts": attempts, **totals,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        })

def validate_atomic_needs_v3(raw: dict[str, Any], normalized_query: str) -> tuple[list[dict[str, Any]], list[str]]:
    warnings = []
    rows = raw.get("needs") if isinstance(raw.get("needs"), list) else []
    needs = []
    for index, row in enumerate(rows[:6], 1):
        if not isinstance(row, dict):
            warnings.append(f"N{index}_NOT_OBJECT")
            continue
        query = str(row.get("query") or normalized_query).strip() or normalized_query
        business = row.get("business_function")
        business = business if business in BUSINESS_FUNCTIONS else None
        model_intent = normalize_intent(row.get("intent"))
        target = str(row.get("target_type") or "").strip() or None
        cases = row.get("case_details")
        cases = [str(x).strip() for x in cases if str(x).strip()] if isinstance(cases, list) else []
        needs.append({
            "need_id": f"N{index}", "query": query,
            "business_function": business, "business_source": "MODEL" if business else None,
            "model_intent": model_intent, "intent": model_intent, "intent_source": "MODEL" if model_intent else "UNKNOWN",
            "intent_rule_pattern": None, "target_type": target, "case_details": cases,
            "user_role": None, "user_role_source": None,
            "applicant_type": None, "applicant_type_source": None,
        })
    if not needs:
        warnings.append("EMPTY_MODEL_NEEDS_RECOVERED")
        needs = [create_retrieve_need_v3(normalized_query)]
    return needs, warnings

def create_retrieve_need_v3(query: str, business: str | None = None) -> dict[str, Any]:
    intent, source, pattern = resolve_intent_v3(query, None)
    return {
        "need_id": "N1", "query": query,
        "business_function": business, "business_source": "ORIGINAL" if business else None,
        "model_intent": None, "intent": intent, "intent_source": source, "intent_rule_pattern": pattern,
        "target_type": None, "case_details": [],
        "user_role": None, "user_role_source": None,
        "applicant_type": None, "applicant_type_source": None,
    }

GENERIC_BUSINESS_CLARIFY_PATTERN = re.compile(
    r"^(?:신청\s*(?:방법|자격|기한|과정)|접수\s*(?:방법|절차)|제출해야 하는 서류|필요한 서류|"
    r"조회는 어디에서|온라인으로 신청|방문해서 접수|접수 후 처리 기간|신청 대상|신청 자격과 제외 조건|"
    r"신청 과정에서 수수료나 비용|처리 결과는 어디에서 확인|이미 접수한 신청을 취소|"
    r"문의하거나 접수하려면 어느 기관|제가 신청 대상에 해당|본인 대신 대리인이 신청|"
    r"상속인이 신청하거나 받을 수).*$", re.I,
)
TARGET_DEMONSTRATIVE_PATTERN_V3 = re.compile(
    r"(?:^|[\s,.(])(?:제가\s*(?:말한|가입한|가진|본)\s*)?이\s*(?:금융상품|계좌|상품|돈|송금|거래|금액|채무|재산)(?:을|를|이|가|도|은|는|의|이나|과|와|\s|[,.!?]|$)"
)
TARGET_REFERENCE_PHRASES_V3 = [
    "제가 가입한 상품", "어떤 송금 건", "어떤 예금에 대해", "어떤 돈을 신청", "어떤 예금이나 금융상품",
]
APPLICANT_REFERENCE_PHRASES_V3 = [
    "제 신청 유형", "제 신청 자격", "제 경우", "누구를 신청인", "누가 방문", "신고 주체 유형", "신청인란",
]
CASE_REFERENCE_PHRASES_V3 = [
    "제 상황", "현재 상황", "제 채무 상태", "제 신고 상황", "반려", "거절", "보완 요청", "진행되지 않",
    "여러 금융회사에 예금", "여러 계좌에 나뉘",
]

def has_resolved_target(query: str, needs: list[dict[str, Any]]) -> bool:
    if any(n.get("target_type") for n in needs):
        return True
    demonstrative = TARGET_DEMONSTRATIVE_PATTERN_V3.search(query)
    if not demonstrative:
        return False
    phrase = demonstrative.group(0)
    # 지시 표현 자체만 있을 뿐 구체 상품명·거래정보는 없으므로 unresolved로 본다.
    return False if phrase else True

def detect_blocking_slot_v3(
    query: str, needs: list[dict[str, Any]], context: dict[str, Any]
) -> tuple[str | None, str | None]:
    confirmed = context.get("confirmed", {}) if context.get("used") else {}
    if GENERIC_BUSINESS_CLARIFY_PATTERN.fullmatch(query.strip()) and not find_businesses(query) and not confirmed.get("business_function"):
        return "business_function", "GENERIC_BUSINESS_EXACT_MATCH"
    if any(x in query for x in APPLICANT_REFERENCE_PHRASES_V3):
        if not confirmed.get("applicant_type") and not extract_explicit_value(query, APPLICANT_PATTERNS):
            return "applicant_type", "UNRESOLVED_APPLICANT_REFERENCE"
    if any(x in query for x in TARGET_REFERENCE_PHRASES_V3) or TARGET_DEMONSTRATIVE_PATTERN_V3.search(query):
        if not confirmed.get("target_type") and not has_resolved_target(query, needs):
            return "target_type", "UNRESOLVED_TARGET_REFERENCE"
    if any(x in query for x in CASE_REFERENCE_PHRASES_V3):
        return "case_details", "PERSONAL_CASE_REQUIRES_DETAILS"
    return None, None

STRONG_BUSINESS_KEYWORDS_V31 = {
    "예금자보호제도": ["예금자보호", "보호한도", "예금 보호", "보호 한도"],
    "예금보험금 안내": ["예금보험금", "보험금 지급", "1종 보험사고", "2종 보험사고"],
    "고객 미수령금 신청": ["미수령금", "파산배당금", "지급대행점", "상속인 금융거래 조회"],
    "착오송금 반환 신청": ["착오송금", "착오 송금", "반환지원", "매입계약", "지급명령", "강제집행"],
    "채무조정 안내": ["채무조정", "신용회복지원", "개인회생", "개인파산", "워크아웃", "부채증명원"],
    "은닉재산 신고": ["은닉재산", "은닉 재산", "금융부실관련자", "차명재산", "차명 재산", "신고 포상금"],
}
WEAK_OR_CROSS_BUSINESS_TERMS_V31 = {
    "예금보험금 안내": ["보험사고", "가지급금", "개산지급금"],
    "고객 미수령금 신청": ["개산지급금 정산금"],
}
MULTI_DECOMPOSITION_SIGNAL_V31 = re.compile(
    r"각각|뿐만\s*아니라|함께\s*알려|동시에\s*알려|"
    r"(?:와|과).{0,30}(?:관계|차이|비교)|"
    r"(?:대상|기준|방법|절차|서류|금액|시점).{0,25}(?:와|과|그리고).{0,25}(?:대상|기준|방법|절차|서류|금액|시점)",
    re.I,
)

def matched_terms_v31(text: str, mapping: dict[str, list[str]]) -> dict[str, list[str]]:
    lowered = (text or "").lower()
    return {
        business: [term for term in terms if term.lower() in lowered]
        for business, terms in mapping.items()
        if any(term.lower() in lowered for term in terms)
    }

def decomposition_status_v31(original: str, needs: list[dict[str, Any]]) -> str:
    if not needs or any(not (need.get("query") or "").strip() for need in needs):
        return "PARTIAL"
    if len(needs) == 1 and MULTI_DECOMPOSITION_SIGNAL_V31.search(original):
        return "PARTIAL"
    return "COMPLETE"

def apply_business_and_intent_rules_v31(
    needs: list[dict[str, Any]], original: str
) -> tuple[list[dict[str, Any]], list[str]]:
    actions = []
    need_count = len(needs)
    for need in needs:
        query = need.get("query") or original
        explicit_businesses = find_businesses(query)
        # V3의 오류 원인이었던 복합 질문 원문 업무의 전체 need 전파를 금지한다.
        if not explicit_businesses and need_count == 1:
            explicit_businesses = find_businesses(original)
        if len(explicit_businesses) == 1 and explicit_businesses[0] != need.get("business_function"):
            need["model_business_function"] = need.get("business_function")
            need["business_function"] = explicit_businesses[0]
            need["business_source"] = "ORIGINAL"
            actions.append(f"{need['need_id']}_BUSINESS_ORIGINAL_OVERRIDE")

        model_intent = need.get("model_intent")
        final_intent, source, pattern = resolve_intent_v3(query, model_intent)
        # V3 FULL에서 N2 override는 개선 0, 회귀 3이었다. 알려진 모델 intent는 보존한다.
        if need.get("need_id") != "N1" and source == "RULE_OVERRIDE" and model_intent in INTENTS:
            final_intent = model_intent
            source = "RULE_CONFLICT_MODEL_KEPT_V31"
            actions.append(f"{need['need_id']}_RULE_OVERRIDE_BLOCKED_V31")
        need["intent"] = final_intent
        need["intent_source"] = source
        need["intent_rule_pattern"] = pattern
        if source in {
            "RULE_OVERRIDE", "RULE_FILLED_UNKNOWN", "RULE_CONFLICT_MODEL_KEPT",
            "RULE_CONFLICT_MODEL_KEPT_V31", "RULE_AMBIGUOUS_MODEL_KEPT", "RULE_AMBIGUOUS_UNKNOWN",
        }:
            actions.append(f"{need['need_id']}_{source}")

    deduped, seen = [], set()
    for need in needs:
        key = (
            need.get("business_function"), need.get("intent"),
            normalize_query(need.get("query") or "")["normalized_query"],
            need.get("target_type"), tuple(sorted(need.get("case_details") or [])),
        )
        if key in seen:
            actions.append("EXACT_DUPLICATE_NEED_COLLAPSED")
            continue
        seen.add(key)
        deduped.append(need)
    for index, need in enumerate(deduped, 1):
        need["need_id"] = f"N{index}"
    return deduped, list(dict.fromkeys(actions))

def annotate_business_safety_v31(
    needs: list[dict[str, Any]], original: str
) -> list[dict[str, Any]]:
    decomposition = decomposition_status_v31(original, needs)
    need_count = len(needs)
    original_candidates = find_businesses(original)
    for need in needs:
        query = need.get("query") or original
        business = need.get("business_function")
        source = need.get("business_source") or "UNKNOWN"
        query_candidates = find_businesses(query)
        candidates = query_candidates or (original_candidates if need_count == 1 else [])
        candidates = list(dict.fromkeys(candidates))
        strong_matches = matched_terms_v31(query, STRONG_BUSINESS_KEYWORDS_V31)
        weak_matches = matched_terms_v31(query, WEAK_OR_CROSS_BUSINESS_TERMS_V31)
        if not strong_matches and need_count == 1:
            strong_matches = matched_terms_v31(original, STRONG_BUSINESS_KEYWORDS_V31)
        if not weak_matches and need_count == 1:
            weak_matches = matched_terms_v31(original, WEAK_OR_CROSS_BUSINESS_TERMS_V31)

        model_business = need.get("model_business_function")
        model_rule_conflict = bool(model_business and business and model_business != business)
        cross_business_ambiguity = len(candidates) > 1
        strong_for_business = business in strong_matches

        if source == "MANUAL":
            confidence = 1.0
        elif source == "CONTEXT":
            confidence = 0.995
        elif strong_for_business and len(candidates) == 1:
            confidence = 0.99
        elif business in candidates and len(candidates) == 1:
            confidence = 0.90
        elif business:
            confidence = 0.70
        else:
            confidence = 0.0
        candidate_margin = 1.0 if len(candidates) <= 1 else 0.0

        denial_reasons = []
        if decomposition != "COMPLETE": denial_reasons.append("INCOMPLETE_DECOMPOSITION")
        if not business: denial_reasons.append("NO_BUSINESS_CANDIDATE")
        if len(candidates) > 1: denial_reasons.append("MULTIPLE_BUSINESS_CANDIDATES")
        if model_rule_conflict: denial_reasons.append("MODEL_RULE_CONFLICT")
        if cross_business_ambiguity: denial_reasons.append("CROSS_BUSINESS_AMBIGUITY")
        if source not in {"MANUAL", "CONTEXT"} and not strong_for_business:
            denial_reasons.append("NO_STRONG_EXPLICIT_EVIDENCE")
        if confidence < HARD_BUSINESS_CONFIDENCE_THRESHOLD:
            denial_reasons.append("LOW_BUSINESS_CONFIDENCE")
        if candidate_margin < HARD_BUSINESS_MARGIN_THRESHOLD:
            denial_reasons.append("LOW_CANDIDATE_MARGIN")

        need["business_candidates"] = [
            {
                "value": candidate,
                "confidence": 0.99 if candidate in strong_matches else 0.80,
                "strong_evidence": strong_matches.get(candidate, []),
                "weak_evidence": weak_matches.get(candidate, []),
            }
            for candidate in candidates
        ]
        need["decomposition_status"] = decomposition
        need["model_rule_conflict"] = model_rule_conflict
        need["cross_business_ambiguity"] = cross_business_ambiguity
        need["business_confidence"] = confidence
        need["business_candidate_margin"] = candidate_margin
        need["hard_filter_eligible"] = not denial_reasons
        need["hard_filter_denial_reasons"] = list(dict.fromkeys(denial_reasons))
    return needs

def decide_route_v3(
    query: str, needs: list[dict[str, Any]], context: dict[str, Any]
) -> tuple[str, list[str], str | None]:
    blocking_slot, reason = detect_blocking_slot_v3(query, needs, context)
    if blocking_slot:
        return "CLARIFY", [reason], blocking_slot
    businesses = [n.get("business_function") for n in needs if n.get("business_function") in BUSINESS_FUNCTIONS]
    if not businesses or any(n.get("business_function") not in BUSINESS_FUNCTIONS for n in needs):
        return "RETRIEVE_RELAXED", ["BUSINESS_UNCERTAIN_BROAD_RETRIEVAL"], None
    return "RETRIEVE", ["SEARCH_CONDITIONS_AVAILABLE"], None

def enrich_evidence_v3(
    needs: list[dict[str, Any]], original: str, context: dict[str, Any], manual: dict[str, Any]
) -> list[dict[str, Any]]:
    explicit_businesses = find_businesses(original)
    explicit_role = extract_explicit_value(original, USER_ROLE_PATTERNS)
    explicit_applicant = extract_explicit_value(original, APPLICANT_PATTERNS)
    confirmed = context.get("confirmed", {}) if context.get("used") else {}
    for need in needs:
        business = need.get("business_function")
        if manual.get("business_function") == business:
            need["business_source"] = "MANUAL"
        elif business in explicit_businesses:
            need["business_source"] = "ORIGINAL"
        elif confirmed.get("business_function") == business:
            need["business_source"] = "CONTEXT"
        elif business:
            need["business_source"] = need.get("business_source") or "MODEL"
        if manual.get("user_role") in USER_ROLES:
            need["user_role"], need["user_role_source"] = manual["user_role"], "MANUAL"
        elif explicit_role:
            need["user_role"], need["user_role_source"] = explicit_role, "ORIGINAL"
        elif confirmed.get("user_role") in USER_ROLES:
            need["user_role"], need["user_role_source"] = confirmed["user_role"], "CONTEXT"
        if manual.get("applicant_type") in APPLICANT_TYPES:
            need["applicant_type"], need["applicant_type_source"] = manual["applicant_type"], "MANUAL"
        elif explicit_applicant:
            need["applicant_type"], need["applicant_type_source"] = explicit_applicant, "ORIGINAL"
        elif confirmed.get("applicant_type") in APPLICANT_TYPES:
            need["applicant_type"], need["applicant_type_source"] = confirmed["applicant_type"], "CONTEXT"
    return needs

def build_keyword_query_v3(need: dict[str, Any]) -> str:
    values = [need.get("business_function"), need.get("intent"), need.get("target_type")]
    values.extend(need.get("case_details") or [])
    if need.get("user_role_source") in {"ORIGINAL", "CONTEXT", "MANUAL"}:
        values.append(need.get("user_role"))
    if need.get("applicant_type_source") in {"ORIGINAL", "CONTEXT", "MANUAL"}:
        values.append(need.get("applicant_type"))
    return " ".join(str(x) for x in values if x and x != "UNKNOWN")

def build_query_plans_v31(analysis: dict[str, Any], config: PipelineConfig) -> list[dict[str, Any]]:
    if analysis["route"] not in {"RETRIEVE", "RETRIEVE_RELAXED"}:
        return []
    relaxed = analysis["route"] == "RETRIEVE_RELAXED"
    plans = []
    for need in analysis["needs"]:
        business, source = need.get("business_function"), need.get("business_source")
        eligible = bool(need.get("hard_filter_eligible"))
        denial_reasons = need.get("hard_filter_denial_reasons") or []
        if relaxed or not business:
            business_filter = {
                "mode": "NONE", "value": None, "soft_hint": business,
                "candidates": need.get("business_candidates") or [],
                "evidence": source or "UNKNOWN", "denial_reasons": denial_reasons,
            }
            fallback_chain = []
        elif eligible:
            business_filter = {
                "mode": "HARD", "value": business, "soft_hint": None,
                "candidates": need.get("business_candidates") or [],
                "evidence": source, "denial_reasons": [],
            }
            fallback_chain = ["SOFT", "NONE"]
        else:
            business_filter = {
                "mode": "SOFT", "value": None, "soft_hint": business,
                "candidates": need.get("business_candidates") or [],
                "evidence": source or "UNKNOWN", "denial_reasons": denial_reasons,
            }
            fallback_chain = ["NONE"]
        intent = need.get("intent")
        intent_weight = 0.20 if need.get("intent_source") in {
            "RULE_OVERRIDE", "RULE_CONFIRMED", "RULE_FILLED_UNKNOWN"
        } else config.intent_soft_boost
        plans.append({
            "need_id": need["need_id"],
            "retrieval_mode": "RELAXED" if relaxed else "STANDARD",
            "semantic_query": need.get("query"),
            "keyword_query": build_keyword_query_v3(need) or need.get("query"),
            "business_filter": business_filter,
            "filter_safety": {
                "hard_filter_eligible": eligible,
                "decomposition_status": need.get("decomposition_status"),
                "model_rule_conflict": need.get("model_rule_conflict"),
                "cross_business_ambiguity": need.get("cross_business_ambiguity"),
                "business_confidence": need.get("business_confidence"),
                "candidate_margin": need.get("business_candidate_margin"),
                "denial_reasons": denial_reasons,
            },
            "fallback_policy": {
                "enabled": bool(fallback_chain),
                "on": ["NO_RESULTS", "LOW_TOP_SCORE", "LOW_COVERAGE"],
                "next_filter_modes": fallback_chain,
                "fail_open": True,
            },
            "intent_boost": {
                "mode": "SOFT" if intent in INTENTS else "NONE", "value": intent,
                "weight": intent_weight if intent in INTENTS else 0.0,
                "evidence": need.get("intent_source"),
            },
            "entities": {
                "user_role": need.get("user_role"), "user_role_source": need.get("user_role_source"),
                "applicant_type": need.get("applicant_type"), "applicant_type_source": need.get("applicant_type_source"),
                "target_type": need.get("target_type"), "case_details": need.get("case_details") or [],
            },
        })
    return plans

def relax_query_plan_v31(plan: dict[str, Any], reason: str) -> dict[str, Any]:
    """검색기가 결과 부족 시 호출할 수 있는 fail-open helper."""
    relaxed_plan = json.loads(json.dumps(plan, ensure_ascii=False))
    current_mode = relaxed_plan.get("business_filter", {}).get("mode")
    if current_mode == "HARD":
        relaxed_plan["business_filter"]["mode"] = "SOFT"
        relaxed_plan["business_filter"]["soft_hint"] = relaxed_plan["business_filter"].get("value")
        relaxed_plan["business_filter"]["value"] = None
    elif current_mode == "SOFT":
        relaxed_plan["business_filter"]["mode"] = "NONE"
        relaxed_plan["retrieval_mode"] = "RELAXED"
    relaxed_plan.setdefault("fallback_history", []).append({"from": current_mode, "reason": reason})
    return relaxed_plan

class KDICLightweightRAGAnalyzerV31:
    def __init__(self, client: HCX007AtomicNeedClientV3, config: PipelineConfig | None = None):
        self.client = client
        self.config = config or client.config

    def run(self, query: str, *, conversation_state=None, manual_selection=None) -> dict[str, Any]:
        started = time.perf_counter()
        normalized = normalize_query(query)
        original, text = normalized["original_query"], normalized["normalized_query"]
        manual = dict(manual_selection or {})
        context = build_context(text, conversation_state, self.config.max_context_turns)
        fast = detect_fast_path_v3(text, conversation_state)
        if fast:
            analysis = {"route": fast["route"], "needs": [], "missing_information": []}
            return {
                "pipeline_version": PIPELINE_VERSION, "analysis_status": "FAST_PATH",
                "original_query": original, "normalized_query": text, "context": context,
                "gate_reasons": ["FAST_PATH"], "rule_actions": [], "blocking_slot": None,
                "analysis": analysis, "fast_path": fast, "validation_warnings": [], "query_plans": [],
                "runtime": {"api_request_count": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                            "latency_ms": round((time.perf_counter() - started) * 1000, 3), "attempts": []},
            }
        payload = {"query": text, "confirmed_context": context["confirmed"], "recent_turns": context["recent_turns"], "manual_selection": manual}
        try:
            raw, telemetry = self.client.analyze(payload)
            needs, warnings = validate_atomic_needs_v3(raw, text)
            status = "REPAIRED" if warnings else "OK"
        except HCXAPIError as exc:
            telemetry = exc.telemetry
            needs = [create_retrieve_need_v3(text, find_businesses(text)[0] if len(find_businesses(text)) == 1 else None)]
            warnings = ["MODEL_ANALYSIS_FAILED_USED_FALLBACK"]
            status = "FALLBACK"
        model_needs = json.loads(json.dumps(needs, ensure_ascii=False))
        needs, rule_actions = apply_business_and_intent_rules_v31(needs, original)
        needs = enrich_evidence_v3(needs, original, context, manual)
        needs = annotate_business_safety_v31(needs, original)
        route, gate_reasons, blocking_slot = decide_route_v3(text, needs, context)
        analysis_needs = [] if route == "CLARIFY" else needs
        analysis = {"route": route, "needs": analysis_needs, "missing_information": [blocking_slot] if blocking_slot else []}
        plans = build_query_plans_v31(analysis, self.config)
        return {
            "pipeline_version": PIPELINE_VERSION, "analysis_status": status,
            "original_query": original, "normalized_query": text, "context": context,
            "model_needs": model_needs, "gate_reasons": gate_reasons,
            "rule_actions": rule_actions, "blocking_slot": blocking_slot,
            "analysis": analysis, "validation_warnings": warnings, "query_plans": plans,
            "runtime": {**telemetry, "latency_ms": round((time.perf_counter() - started) * 1000, 3)},
        }