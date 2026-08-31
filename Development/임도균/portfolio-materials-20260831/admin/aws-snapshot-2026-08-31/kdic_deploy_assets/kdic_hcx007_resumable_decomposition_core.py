from __future__ import annotations

import email.utils
import hashlib
import json
import random
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import requests

import kdic_decomposition_quality_core as quality_core
from kdic_decomposition_quality_core import (
    BASELINE,
    CONDITION_LABELS,
    QUALITY,
    QUALITY_RETRY,
    RETRY,
    QualityConfig,
    baseline_json_schema,
    build_quality_messages,
    build_repair_messages,
    content_checks,
    extract_queries,
    quality_json_schema,
    should_semantic_retry,
    validate_baseline_decomposition,
    validate_quality_decomposition,
)
from kdic_lightweight_query_ablation_core import (
    AblationConfig,
    DECOMPOSITION_PROMPT_VERSION,
    _extract_hcx_payload,
    build_decomposition_messages,
    decomposition_json_schema,
    validate_llm_decomposition,
)


HCX_DECOMPOSITION_MODEL = "HCX-007"
HCX_DECOMPOSITION_ENDPOINT = (
    "https://clovastudio.stream.ntruss.com/v3/chat-completions/HCX-007"
)
REDESIGN_PROMPT_VERSION = "KDIC_DECOMPOSITION_QUALITY_V2_HCX007_2026_08_14"
REDESIGN_REPAIR_PROMPT_VERSION = "KDIC_DECOMPOSITION_REPAIR_V2_HCX007_2026_08_14"


@dataclass(frozen=True)
class TransportPolicy:
    request_delay_seconds: float = 1.5
    max_transport_retries: int = 5
    base_backoff_seconds: float = 2.0
    max_backoff_seconds: float = 32.0
    jitter_seconds: float = 0.5
    consecutive_429_cooldown_threshold: int = 3
    cooldown_seconds: float = 60.0
    timeout_seconds: float = 120.0


def _valid_api_key(value: str) -> str:
    key = str(value or "").strip()
    if not key or key.lower().startswith("bearer ") or any(ch.isspace() for ch in key):
        raise ValueError("HCX_API_KEY에는 Bearer 접두사나 공백을 넣지 않습니다.")
    return key


def _cache_key(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ValidOnlyJsonlCache:
    """정상 응답만 재사용하고 ERROR 행은 감사 기록으로만 남긴다."""

    def __init__(
        self,
        active_path: str | Path,
        *,
        seed_paths: Sequence[str | Path] = (),
    ) -> None:
        self.active_path = Path(active_path)
        self.rows: dict[str, dict[str, Any]] = {}
        self.origin_by_key: dict[str, str] = {}
        for source in [*map(Path, seed_paths), self.active_path]:
            if not source.is_file():
                continue
            for line in source.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                key = str(row.get("cache_key") or "")
                if not key or str(row.get("status") or "").upper() == "ERROR":
                    continue
                self.rows[key] = row
                self.origin_by_key[key] = str(source)

    def get(self, key: str) -> dict[str, Any] | None:
        if key not in self.rows:
            return None
        row = dict(self.rows[key])
        row["cache_hit"] = True
        row["actual_api_latency_ms"] = 0.0
        row["cache_origin"] = self.origin_by_key.get(key, "")
        if Path(self.origin_by_key.get(key, "")) != self.active_path:
            promoted = dict(row)
            promoted["promoted_from_seed_cache"] = True
            self.append(promoted, reusable=True)
        return row

    def append(self, row: Mapping[str, Any], *, reusable: bool) -> None:
        payload = dict(row)
        self.active_path.parent.mkdir(parents=True, exist_ok=True)
        with self.active_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        key = str(payload.get("cache_key") or "")
        if reusable and key:
            self.rows[key] = payload
            self.origin_by_key[key] = str(self.active_path)


def _retry_after_seconds(response: requests.Response) -> float | None:
    raw = str(response.headers.get("Retry-After") or "").strip()
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        try:
            parsed = email.utils.parsedate_to_datetime(raw)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds())
        except Exception:
            return None


class RobustHCXTransport:
    def __init__(
        self,
        api_key: str,
        *,
        endpoint: str = HCX_DECOMPOSITION_ENDPOINT,
        policy: TransportPolicy | None = None,
        session: requests.Session | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        monotonic_fn: Callable[[], float] = time.monotonic,
        random_fn: Callable[[], float] = random.random,
    ) -> None:
        self.api_key = _valid_api_key(api_key)
        self.endpoint = endpoint
        self.policy = policy or TransportPolicy()
        self.session = session or requests.Session()
        self.sleep_fn = sleep_fn
        self.monotonic_fn = monotonic_fn
        self.random_fn = random_fn
        self.last_request_at: float | None = None
        self.consecutive_429 = 0

    def _pace(self) -> float:
        if self.last_request_at is None:
            return 0.0
        remaining = self.policy.request_delay_seconds - (
            self.monotonic_fn() - self.last_request_at
        )
        if remaining > 0:
            self.sleep_fn(remaining)
            return remaining
        return 0.0

    def post_json(self, body: Mapping[str, Any]) -> dict[str, Any]:
        logical_started = self.monotonic_fn()
        total_sleep_seconds = 0.0
        service_latency_ms = 0.0
        attempts = 0
        last_status: int | None = None
        last_error_type = ""
        last_error_message = ""
        last_response_body = ""

        for retry_index in range(self.policy.max_transport_retries + 1):
            total_sleep_seconds += self._pace()
            attempts += 1
            request_started = self.monotonic_fn()
            try:
                response = self.session.post(
                    self.endpoint,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "X-NCP-CLOVASTUDIO-REQUEST-ID": str(uuid.uuid4()),
                    },
                    json=dict(body),
                    timeout=self.policy.timeout_seconds,
                )
                self.last_request_at = self.monotonic_fn()
                service_latency_ms += (self.last_request_at - request_started) * 1000
                last_status = int(response.status_code)
                last_response_body = str(response.text or "")[:8000]

                if 200 <= response.status_code < 300:
                    self.consecutive_429 = 0
                    payload, usage = _extract_hcx_payload(response.json())
                    return {
                        "transport_ok": True,
                        "payload": payload,
                        "usage": usage,
                        "http_status": last_status,
                        "transport_attempts": attempts,
                        "service_latency_ms": round(service_latency_ms, 3),
                        "transport_sleep_ms": round(total_sleep_seconds * 1000, 3),
                        "actual_api_latency_ms": round(
                            (self.monotonic_fn() - logical_started) * 1000, 3
                        ),
                        "error_type": "",
                        "error_message": "",
                        "error_response_body": "",
                    }

                last_error_type = "HTTP_ERROR"
                last_error_message = f"HTTP {response.status_code}"
                retryable = response.status_code == 429 or response.status_code >= 500
                if response.status_code == 429:
                    self.consecutive_429 += 1
                else:
                    self.consecutive_429 = 0
                if not retryable or retry_index >= self.policy.max_transport_retries:
                    break

                if (
                    response.status_code == 429
                    and self.consecutive_429
                    >= self.policy.consecutive_429_cooldown_threshold
                ):
                    wait_seconds = self.policy.cooldown_seconds
                    self.consecutive_429 = 0
                else:
                    retry_after = _retry_after_seconds(response)
                    exponential = min(
                        self.policy.max_backoff_seconds,
                        self.policy.base_backoff_seconds * (2**retry_index),
                    )
                    wait_seconds = retry_after if retry_after is not None else exponential
                    wait_seconds += self.random_fn() * self.policy.jitter_seconds
                self.sleep_fn(wait_seconds)
                total_sleep_seconds += wait_seconds
            except Exception as error:
                self.last_request_at = self.monotonic_fn()
                service_latency_ms += (self.last_request_at - request_started) * 1000
                last_error_type = type(error).__name__
                last_error_message = str(error)
                if retry_index >= self.policy.max_transport_retries:
                    break
                wait_seconds = min(
                    self.policy.max_backoff_seconds,
                    self.policy.base_backoff_seconds * (2**retry_index),
                ) + self.random_fn() * self.policy.jitter_seconds
                self.sleep_fn(wait_seconds)
                total_sleep_seconds += wait_seconds

        return {
            "transport_ok": False,
            "payload": {},
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "http_status": last_status,
            "transport_attempts": attempts,
            "service_latency_ms": round(service_latency_ms, 3),
            "transport_sleep_ms": round(total_sleep_seconds * 1000, 3),
            "actual_api_latency_ms": round(
                (self.monotonic_fn() - logical_started) * 1000, 3
            ),
            "error_type": last_error_type or "UNKNOWN_TRANSPORT_ERROR",
            "error_message": last_error_message or "unknown transport error",
            "error_response_body": last_response_body,
        }


def _error_row(
    cache_key: str,
    *,
    question: str,
    model: str,
    prompt_version: str,
    transport: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "cache_key": cache_key,
        "question": question,
        "model": model,
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
        **dict(transport.get("usage") or {}),
        "effective_api_latency_ms": float(transport.get("actual_api_latency_ms") or 0.0),
        "cache_hit": False,
        **{
            key: transport.get(key)
            for key in (
                "http_status", "transport_attempts", "service_latency_ms",
                "transport_sleep_ms", "actual_api_latency_ms", "error_type",
                "error_message", "error_response_body",
            )
        },
    }


class ResumableBaselineDecomposer:
    def __init__(
        self,
        api_key: str,
        *,
        cache_path: str | Path,
        seed_cache_paths: Sequence[str | Path] = (),
        config: AblationConfig | None = None,
        transport_policy: TransportPolicy | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.config = config or AblationConfig(
            llm_endpoint=HCX_DECOMPOSITION_ENDPOINT,
            llm_model=HCX_DECOMPOSITION_MODEL,
        )
        if self.config.llm_model != HCX_DECOMPOSITION_MODEL:
            raise ValueError("Baseline 구조화 분해 모델은 HCX-007이어야 합니다.")
        self.cache = ValidOnlyJsonlCache(cache_path, seed_paths=seed_cache_paths)
        self.transport = RobustHCXTransport(
            api_key, endpoint=HCX_DECOMPOSITION_ENDPOINT,
            policy=transport_policy, session=session,
        )

    def _key(self, question: str, expected_businesses: Sequence[str]) -> str:
        return _cache_key({
            "prompt_version": DECOMPOSITION_PROMPT_VERSION,
            "model": self.config.llm_model,
            "question": question,
            "expected_businesses": list(expected_businesses),
            "min_confidence": self.config.llm_min_confidence,
        })

    def decompose(self, question: str, expected_businesses: Sequence[str]) -> dict[str, Any]:
        key = self._key(question, expected_businesses)
        cached = self.cache.get(key)
        if cached is not None:
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
            "responseFormat": {
                "type": "json",
                "schema": decomposition_json_schema(self.config.max_subqueries),
            },
        }
        transport = self.transport.post_json(body)
        if not transport["transport_ok"]:
            row = _error_row(
                key, question=question, model=self.config.llm_model,
                prompt_version=DECOMPOSITION_PROMPT_VERSION, transport=transport,
            )
            self.cache.append(row, reusable=False)
            return row
        validation = validate_llm_decomposition(
            question,
            transport["payload"],
            expected_businesses=expected_businesses,
            config=self.config,
        )
        row = {
            "cache_key": key,
            "question": question,
            "model": self.config.llm_model,
            "prompt_version": DECOMPOSITION_PROMPT_VERSION,
            "raw_payload": transport["payload"],
            **validation,
            **transport["usage"],
            "effective_api_latency_ms": transport["actual_api_latency_ms"],
            "cache_hit": False,
            **{
                field: transport.get(field)
                for field in (
                    "http_status", "transport_attempts", "service_latency_ms",
                    "transport_sleep_ms", "actual_api_latency_ms", "error_type",
                    "error_message", "error_response_body",
                )
            },
        }
        self.cache.append(row, reusable=True)
        return row


class ResumableQualityCaller:
    def __init__(
        self,
        api_key: str,
        *,
        cache_path: str | Path,
        config: QualityConfig,
        transport_policy: TransportPolicy | None = None,
        session: requests.Session | None = None,
    ) -> None:
        if config.llm_model != HCX_DECOMPOSITION_MODEL:
            raise ValueError("개선 구조화 분해 모델은 HCX-007이어야 합니다.")
        self.config = config
        self.cache = ValidOnlyJsonlCache(cache_path)
        self.transport = RobustHCXTransport(
            api_key, endpoint=HCX_DECOMPOSITION_ENDPOINT,
            policy=transport_policy, session=session,
        )

    def _call(
        self,
        *,
        key_payload: Mapping[str, Any],
        question: str,
        prompt_version: str,
        messages: Sequence[Mapping[str, str]],
        schema: Mapping[str, Any],
        validator: Callable[[Mapping[str, Any]], dict[str, Any]],
    ) -> dict[str, Any]:
        key = _cache_key(key_payload)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
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
        transport = self.transport.post_json(body)
        if not transport["transport_ok"]:
            row = _error_row(
                key, question=question, model=self.config.llm_model,
                prompt_version=prompt_version, transport=transport,
            )
            self.cache.append(row, reusable=False)
            return row
        validation = validator(transport["payload"])
        row = {
            "cache_key": key,
            "question": question,
            "model": self.config.llm_model,
            "prompt_version": prompt_version,
            "raw_payload": transport["payload"],
            **validation,
            **transport["usage"],
            "effective_api_latency_ms": transport["actual_api_latency_ms"],
            "cache_hit": False,
            **{
                field: transport.get(field)
                for field in (
                    "http_status", "transport_attempts", "service_latency_ms",
                    "transport_sleep_ms", "actual_api_latency_ms", "error_type",
                    "error_message", "error_response_body",
                )
            },
        }
        self.cache.append(row, reusable=True)
        return row

    def quality_first(self, question: str, expected_businesses: Sequence[str]) -> dict[str, Any]:
        return self._call(
            key_payload={
                "prompt_version": REDESIGN_PROMPT_VERSION,
                "model": self.config.llm_model,
                "question": question,
                "expected_businesses": list(expected_businesses),
            },
            question=question,
            prompt_version=REDESIGN_PROMPT_VERSION,
            messages=build_quality_messages(question, expected_businesses),
            schema=quality_json_schema(self.config.max_subqueries),
            validator=lambda payload: validate_quality_decomposition(
                question, payload,
                expected_businesses=expected_businesses,
                config=self.config,
            ),
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
                question, payload,
                expected_businesses=expected_businesses,
                config=self.config,
            )
        else:
            schema = baseline_json_schema(self.config.max_subqueries)
            validator = lambda payload: validate_baseline_decomposition(
                question, payload,
                expected_businesses=expected_businesses,
                config=self.config,
            )
        return self._call(
            key_payload={
                "prompt_version": REDESIGN_REPAIR_PROMPT_VERSION,
                "model": self.config.llm_model,
                "question": question,
                "expected_businesses": list(expected_businesses),
                "quality_mode": quality_mode,
                "issues": issues,
                "previous_payload": previous_payload,
            },
            question=question,
            prompt_version=REDESIGN_REPAIR_PROMPT_VERSION,
            messages=build_repair_messages(
                question, expected_businesses, previous_payload, issues,
                quality_mode=quality_mode,
            ),
            schema=schema,
            validator=validator,
        )


def _normalize_baseline_record(
    question: str,
    expected_businesses: Sequence[str],
    record: Mapping[str, Any],
) -> dict[str, Any]:
    output = dict(record)
    output.setdefault(
        "candidate_subqueries",
        extract_queries(record.get("raw_payload") or {}),
    )
    output.setdefault(
        "checks",
        content_checks(question, record.get("raw_payload") or {}, expected_businesses),
    )
    return output


def _condition_record(
    condition: str,
    first: Mapping[str, Any],
    final: Mapping[str, Any],
    *,
    retry_called: bool,
) -> dict[str, Any]:
    final_record = dict(final)
    first_latency = float(first.get("effective_api_latency_ms") or 0.0)
    retry_latency = float(final.get("effective_api_latency_ms") or 0.0) if retry_called else 0.0
    return {
        "condition": condition,
        "condition_label": CONDITION_LABELS[condition],
        "first_status": first.get("status"),
        "first_accepted": bool(first.get("accepted")),
        "first_confidence": float(first.get("confidence") or 0.0),
        "first_issues": list(first.get("issues") or []),
        "first_candidate_subqueries": list(
            first.get("candidate_subqueries") or first.get("subqueries") or []
        ),
        "first_checks": dict(first.get("checks") or {}),
        "first_http_status": first.get("http_status"),
        "first_error_type": first.get("error_type") or "",
        "first_error_message": first.get("error_message") or "",
        "first_error_response_body": first.get("error_response_body") or "",
        "first_transport_attempts": int(first.get("transport_attempts") or 0),
        "first_cache_hit": bool(first.get("cache_hit")),
        "first_cache_origin": first.get("cache_origin") or "",
        "retry_called": retry_called,
        "retry_count": int(retry_called),
        "retry_success": bool(retry_called and final.get("accepted")),
        "retry_status": final.get("status") if retry_called else "NOT_CALLED",
        "retry_issues": list(final.get("issues") or []) if retry_called else [],
        "retry_http_status": final.get("http_status") if retry_called else None,
        "retry_error_type": final.get("error_type") if retry_called else "",
        "retry_transport_attempts": int(final.get("transport_attempts") or 0) if retry_called else 0,
        "final_status": final.get("status"),
        "final_accepted": bool(final.get("accepted")),
        "final_confidence": float(final.get("confidence") or 0.0),
        "final_issues": list(final.get("issues") or []),
        "final_subqueries": list(final.get("subqueries") or []),
        "final_candidate_subqueries": list(
            final.get("candidate_subqueries") or final.get("subqueries") or []
        ),
        "final_checks": dict(final.get("checks") or {}),
        "fallback_to_original": not bool(final.get("accepted")),
        "analysis_api_latency_ms": first_latency + retry_latency,
        "actual_api_latency_ms": float(first.get("actual_api_latency_ms") or 0.0)
        + (float(final.get("actual_api_latency_ms") or 0.0) if retry_called else 0.0),
        "prompt_tokens": int(first.get("prompt_tokens") or 0)
        + (int(final.get("prompt_tokens") or 0) if retry_called else 0),
        "completion_tokens": int(first.get("completion_tokens") or 0)
        + (int(final.get("completion_tokens") or 0) if retry_called else 0),
        "total_tokens": int(first.get("total_tokens") or 0)
        + (int(final.get("total_tokens") or 0) if retry_called else 0),
        "logical_api_request_count": 1 + int(retry_called),
        # 기존 build_condition_case가 읽는 호환 필드입니다. 의미는 HTTP 재시도
        # 횟수가 아니라 첫 구조화 호출 + 선택적 의미 교정 호출 수입니다.
        "api_request_count": 1 + int(retry_called),
        "transport_attempt_count": int(first.get("transport_attempts") or 0)
        + (int(final.get("transport_attempts") or 0) if retry_called else 0),
        "first_raw_payload": dict(first.get("raw_payload") or {}),
        "retry_raw_payload": dict(final_record.get("raw_payload") or {}) if retry_called else {},
    }


def run_resumable_candidate_conditions(
    question: str,
    expected_businesses: Sequence[str],
    *,
    baseline_decomposer: ResumableBaselineDecomposer,
    quality_caller: ResumableQualityCaller,
) -> list[dict[str, Any]]:
    baseline_first = _normalize_baseline_record(
        question,
        expected_businesses,
        baseline_decomposer.decompose(question, expected_businesses),
    )
    quality_first = quality_caller.quality_first(question, expected_businesses)

    if should_semantic_retry(baseline_first):
        baseline_final = quality_caller.repair(
            question, expected_businesses, baseline_first, quality_mode=False
        )
        baseline_retry_called = True
    else:
        baseline_final = baseline_first
        baseline_retry_called = False

    if should_semantic_retry(quality_first):
        quality_final = quality_caller.repair(
            question, expected_businesses, quality_first, quality_mode=True
        )
        quality_retry_called = True
    else:
        quality_final = quality_first
        quality_retry_called = False

    return [
        _condition_record(BASELINE, baseline_first, baseline_first, retry_called=False),
        _condition_record(QUALITY, quality_first, quality_first, retry_called=False),
        _condition_record(RETRY, baseline_first, baseline_final, retry_called=baseline_retry_called),
        _condition_record(
            QUALITY_RETRY, quality_first, quality_final,
            retry_called=quality_retry_called,
        ),
    ]


def component_gate_rows(audit_df: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition, frame in audit_df.groupby("condition", sort=False):
        candidates = frame[frame["cross_business_candidate"].astype(bool)]
        first_success = candidates["first_status"].ne("ERROR")
        http_400_count = int(candidates["first_http_status"].eq(400).sum())
        final_error_count = int(candidates["final_status"].eq("ERROR").sum())
        error_latency_missing = int((
            candidates["first_status"].eq("ERROR")
            & candidates["actual_api_latency_ms"].le(0)
        ).sum())
        checks = [
            ("structured_call_success_rate", float(first_success.mean()), 0.995, float(first_success.mean()) >= .995),
            ("candidate_evaluable_count", int(first_success.sum()), len(candidates), int(first_success.sum()) == len(candidates)),
            ("http_400_count", http_400_count, 0, http_400_count == 0),
            ("final_transport_error_count", final_error_count, 0, final_error_count == 0),
            ("error_latency_missing_count", error_latency_missing, 0, error_latency_missing == 0),
            ("retry_limit_exceeded_count", int((candidates["retry_count"] > 1).sum()), 0, int((candidates["retry_count"] > 1).sum()) == 0),
        ]
        for gate, value, threshold, passed in checks:
            rows.append({
                "condition": condition,
                "condition_label": CONDITION_LABELS.get(condition, condition),
                "gate": gate,
                "value": value,
                "threshold": threshold,
                "passed": bool(passed),
            })
    return rows
