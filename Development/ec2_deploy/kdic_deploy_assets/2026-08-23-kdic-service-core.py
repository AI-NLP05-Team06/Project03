from __future__ import annotations

import ast
import copy
import importlib
import importlib.util
import inspect
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Protocol, Sequence


class PipelineNotConfiguredError(RuntimeError):
    """Raised when the API has no KDIC pipeline callable attached."""


class PipelineCallable(Protocol):
    def __call__(
        self,
        question: str,
        state: MutableMapping[str, Any],
        progress: Callable[[int, str], None] | None = None,
    ) -> Mapping[str, Any]: ...


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _clean_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values: Sequence[Any] = [value]
    elif isinstance(value, Sequence):
        values = value
    else:
        values = [value]
    output: list[str] = []
    for item in values:
        text = _clean_text(item)
        if text and text not in output:
            output.append(text)
    return output


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _source_rows(raw: Any) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in raw or []:
        if not isinstance(row, Mapping):
            continue
        title = _clean_text(
            row.get("title")
            or row.get("document_title")
            or row.get("name")
            or "공식 안내"
        )
        url = _clean_text(row.get("url") or row.get("source_url"))
        key = (title, url)
        if key in seen:
            continue
        seen.add(key)
        output.append({"title": title, "url": url})
    return output


def _action_link_rows(raw: Any) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for row in raw or []:
        if not isinstance(row, Mapping):
            continue
        url = _clean_text(row.get("url"))
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        output.append(
            {
                "link_id": _clean_text(row.get("link_id")),
                "label": _clean_text(
                    row.get("label") or row.get("button_label") or "공식 서비스 열기"
                ),
                "url": url,
                "description": _clean_text(row.get("description")),
                "requires_auth": bool(
                    row.get("requires_auth")
                    or str(row.get("channel") or "").upper() == "WEB_AUTH"
                ),
                "action_type": _clean_text(row.get("action_type")),
            }
        )
    return output


def _analysis_from_result(result: Mapping[str, Any]) -> dict[str, Any]:
    common = _mapping(result.get("common"))
    return _mapping(result.get("analysis")) or _mapping(common.get("analysis"))


def _common_from_result(result: Mapping[str, Any]) -> dict[str, Any]:
    return _mapping(result.get("common"))


def _route_from_result(result: Mapping[str, Any]) -> str:
    route = _clean_text(result.get("route")).upper()
    if route == "C_POLICY_TARGET":
        return "RETRIEVE"
    if route:
        return route
    return _clean_text(_common_from_result(result).get("route") or "RETRIEVE").upper()


def _normalize_answer_text(value: Any) -> str:
    """Return markdown text without leaking Python list representations."""

    if isinstance(value, (list, tuple)):
        sections = [str(item).strip() for item in value if isinstance(item, str) and item.strip()]
        return "\n\n".join(sections)
    text = str(value or "").strip()
    if len(text) >= 2 and text[0] in "[(" and text[-1] in "])":
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
            parsed = None
        if isinstance(parsed, (list, tuple)) and all(
            isinstance(item, str) for item in parsed
        ):
            sections = [item.strip() for item in parsed if item.strip()]
            if sections:
                return "\n\n".join(sections)
    return text


def _answer_from_result(result: Mapping[str, Any]) -> str:
    payload = _mapping(result.get("payload"))
    common = _common_from_result(result)
    candidates = [
        result.get("answer"),
        result.get("display_answer"),
        result.get("basic_answer"),
        payload.get("answer"),
        result.get("route_message"),
        common.get("route_message"),
    ]
    for candidate in candidates:
        text = _normalize_answer_text(candidate)
        if text:
            return text
    return "답변을 준비하지 못했습니다. 잠시 후 다시 시도해 주세요."


def _businesses_from_result(result: Mapping[str, Any]) -> list[str]:
    analysis = _analysis_from_result(result)
    complexity = _mapping(analysis.get("complexity"))
    values = (
        analysis.get("businesses")
        or complexity.get("businesses")
        or result.get("businesses")
        or []
    )
    return _clean_list(values)


def _clarification_options(result: Mapping[str, Any]) -> list[str]:
    analysis = _analysis_from_result(result)
    resolution = _mapping(analysis.get("context_resolution"))
    pending = _mapping(
        resolution.get("pending_clarification")
        or resolution.get("pending")
        or analysis.get("pending_clarification")
    )
    return _clean_list(
        result.get("clarification_options")
        or pending.get("options")
        or resolution.get("options")
        or analysis.get("options")
    )


FOLLOWUP_KEYWORDS: dict[str, tuple[str, ...]] = {
    "착오송금": ("신청 대상", "신청 기한", "필요 서류", "처리 절차", "회수 비용"),
    "예금보험금": ("신청 방법", "필요 서류", "지급 절차", "지급 시기", "보호 한도"),
    "예금자보호": ("보호 대상", "보호 한도", "제외 상품", "금융회사별 계산"),
    "미수령금": ("조회 방법", "신청 방법", "필요 서류", "지급 절차"),
    "채무조정": ("신청 대상", "신청 방법", "필요 서류", "조정 절차"),
    "은닉재산": ("신고 방법", "포상금 기준", "필요 자료", "처리 절차"),
}


FOLLOWUP_CANONICAL_BUSINESSES: dict[str, str] = {
    "착오송금": "착오송금 반환 신청",
    "예금보험금": "예금보험금 안내",
    "예금자보호": "예금자보호제도",
    "미수령금": "고객 미수령금 신청",
    "채무조정": "채무조정 안내",
    "은닉재산": "은닉재산 신고",
}
FOLLOWUP_QUERY_OVERRIDES: dict[tuple[str, str], str] = {
    ("예금자보호", "제외 상품"): (
        "예금자보호제도에서 보호되지 않는 금융상품을 알려주세요."
    ),
    ("미수령금", "조회 방법"): "본인 명의 고객 미수령금 조회 방법을 알려주세요.",
    ("미수령금", "필요 서류"): (
        "본인 명의 고객 미수령금 신청에 필요한 서류를 알려주세요."
    ),
    ("예금보험금", "보호 한도"): (
        "예금보험금으로 지급되는 1인당 최대 금액을 알려주세요."
    ),
}
SUGGESTION_CACHE_SCHEMA_VERSION = "kdic-suggestion-answer-bundle-v5.1"
BASIS_EXPLANATION_SCHEMA_VERSION = "kdic-basis-explanation-v2"
_SUGGESTION_BY_ID: dict[str, dict[str, str]] = {}
_SUGGESTIONS_BY_BUSINESS_KEY: dict[str, list[dict[str, str]]] = {}


def _suggestion_id(business: str, label: str, query: str) -> str:
    return "SQ-" + uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"kdic://suggestion/{business}/{label}/{query}",
    ).hex[:16].upper()


def _build_suggestion_registry() -> None:
    for business_key, labels in FOLLOWUP_KEYWORDS.items():
        business = FOLLOWUP_CANONICAL_BUSINESSES[business_key]
        rows: list[dict[str, str]] = []
        for label in labels:
            query = FOLLOWUP_QUERY_OVERRIDES.get(
                (business_key, label), f"{business}의 {label}을 알려주세요."
            )
            record = {
                "suggestion_id": _suggestion_id(business, label, query),
                "business_key": business_key,
                "business": business,
                "label": label,
                "query": query,
            }
            _SUGGESTION_BY_ID[record["suggestion_id"]] = record
            rows.append(record)
        _SUGGESTIONS_BY_BUSINESS_KEY[business_key] = rows


_build_suggestion_registry()


def suggestion_catalog() -> list[dict[str, str]]:
    return [
        copy.deepcopy(row)
        for rows in _SUGGESTIONS_BY_BUSINESS_KEY.values()
        for row in rows
    ]


def resolve_registered_suggestion(
    question: str,
    suggestion_id: str = "",
) -> dict[str, str] | None:
    """Accept cache access only when the server ID and canonical query both match."""

    clean_id = _clean_text(suggestion_id).upper()
    record = _SUGGESTION_BY_ID.get(clean_id)
    if record is None or record["query"] != _clean_text(question):
        return None
    return copy.deepcopy(record)


def suggestion_registry_stats() -> dict[str, Any]:
    return {
        "schema_version": SUGGESTION_CACHE_SCHEMA_VERSION,
        "registered_questions": len(_SUGGESTION_BY_ID),
        "business_count": len(_SUGGESTIONS_BY_BUSINESS_KEY),
        "requires_exact_id_and_query": True,
    }


def _followup_keywords(businesses: Sequence[str]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for business in businesses:
        for key, rows in _SUGGESTIONS_BY_BUSINESS_KEY.items():
            if key not in business:
                continue
            for record in rows:
                item = copy.deepcopy(record)
                if item not in output:
                    output.append(item)
    return output[:5]


def _sources_from_result(result: Mapping[str, Any]) -> list[dict[str, str]]:
    direct = _source_rows(result.get("official_sources") or result.get("sources"))
    if direct:
        return direct
    common = _common_from_result(result)
    pack = _mapping(common.get("evidence_pack") or result.get("evidence_pack"))
    return _source_rows(pack.get("sources"))


def _latency_from_result(result: Mapping[str, Any]) -> dict[str, float]:
    common = _common_from_result(result)
    latency = _mapping(result.get("latency_ms"))
    if not latency:
        latency = _mapping(common.get("latency_ms"))
    click = _mapping(result.get("latency"))
    if click:
        latency = {**latency, **click}
    return {str(key): _number(value) for key, value in latency.items()}


def normalize_public_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize legacy B, C and D-C notebook results for the HTML client."""

    route = _route_from_result(result)
    businesses = _businesses_from_result(result)
    action_links = _action_link_rows(
        result.get("action_links") or _common_from_result(result).get("action_links")
    )
    analysis = _analysis_from_result(result)
    payload = _mapping(result.get("payload"))
    return {
        "route": route,
        "answer": _answer_from_result(result),
        "businesses": businesses,
        "keywords": _followup_keywords(businesses) if route == "RETRIEVE" else [],
        "clarification_options": _clarification_options(result)
        if route == "CLARIFY"
        else [],
        "sources": _sources_from_result(result),
        "action_links": action_links if route == "RETRIEVE" else [],
        "latency_seconds": {
            key: round(value / 1000.0, 3)
            for key, value in _latency_from_result(result).items()
        },
        "context_used": bool(analysis.get("context_used")),
        "answer_system": _clean_text(
            result.get("variant")
            or payload.get("system")
            or result.get("answer_system")
        ),
        "coverage_status": _clean_text(payload.get("coverage_status")),
        "validation_passed": payload.get("validation_passed"),
    }


_BASIS_STOPWORDS = {
    "그리고", "그러나", "대한", "관련", "안내", "어떻게", "알려주세요", "무엇인가요",
    "있습니다", "합니다", "되는", "위한", "경우", "질문", "답변", "공식",
}


def _basis_public_text(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"\[(?:[^\]\s]*_)?chunk_?\d+\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b[A-Za-z0-9-]+_chunk_?\d+\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\b(?:evidence|chunk|parent|need)_?id\s*[:=]\s*[^\s,;]+",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"(?<![A-Za-z0-9])E\d+(?![A-Za-z0-9])", " ", text)
    text = re.sub(r"(?m)^\s*#{1,6}\s*", "", text)
    text = re.sub(r"(?i)\b(?:Evidence Pack|Reranker)\b", " ", text)
    return _clean_text(text)


def _basis_terms(value: Any) -> set[str]:
    return {
        token
        for token in re.findall(r"[가-힣A-Za-z0-9]{2,}", _basis_public_text(value).lower())
        if token not in _BASIS_STOPWORDS
    }


def _basis_relevant_excerpt(content: Any, question: str, answer: str) -> str:
    raw = str(content or "")
    candidates = re.split(
        r"(?=\[(?:[^\]\s]*_)?chunk_?\d+\])|(?=(?:^|\s)#{1,6}\s*Q\.\s*)",
        raw,
        flags=re.IGNORECASE,
    )
    candidates = [_basis_public_text(value) for value in candidates]
    candidates = [value for value in candidates if len(value) >= 20]
    if not candidates:
        candidates = [_basis_public_text(raw)] if _basis_public_text(raw) else []
    target_terms = _basis_terms(question) | _basis_terms(answer)

    def score(value: str) -> tuple[int, int]:
        overlap = len(_basis_terms(value) & target_terms)
        question_bonus = 4 if question and _clean_text(question)[:35] in value else 0
        return overlap + question_bonus, -len(value)

    selected = max(candidates, key=score, default="")
    q_parts = re.split(r"(?:^|\s)Q\.\s*", selected, flags=re.IGNORECASE)
    if len(q_parts) > 1 and _clean_text(q_parts[-1]):
        selected = _clean_text(q_parts[-1])
    selected = re.sub(r"^\d{1,2}[.)]\s*", "", selected)
    return selected[:320] + ("…" if len(selected) > 320 else "")


def _basis_answer_point(answer: str, excerpt: str, fallback_title: str) -> str:
    raw_answer = re.sub(r"https?://\S+", " ", str(answer or ""))
    candidates = []
    for line in raw_answer.splitlines() or [raw_answer]:
        for value in re.split(r"(?<=[.!?])\s+", line):
            clean_value = _basis_public_text(value)
            if 12 <= len(clean_value) <= 220:
                candidates.append(clean_value)
    excerpt_terms = _basis_terms(excerpt)
    if candidates:
        selected = max(candidates, key=lambda value: len(_basis_terms(value) & excerpt_terms))
        return selected[:180]
    title = _basis_public_text(fallback_title)
    if "?" in title:
        title = title.split("?", 1)[0] + "?"
    return (title or "답변에서 안내한 핵심 내용")[:180]


def _basis_user_meaning(answer_point: str, evidence_summary: str) -> str:
    text = f"{answer_point} {evidence_summary}"
    if re.search(r"기간|며칠|개월|소요|시기|언제", text):
        return "신청일과 현재 진행 상태를 기준으로 예상 처리 기간을 확인할 때 필요한 정보입니다."
    if re.search(r"금액|한도|원|포상금|비율", text):
        return "본인의 금액이 안내된 범위와 계산 기준에 해당하는지 확인할 때 필요한 정보입니다."
    if re.search(r"대상|자격|조건|요건", text):
        return "본인의 상황이 신청 대상과 조건에 해당하는지 판단할 때 확인해야 하는 정보입니다."
    if re.search(r"서류|증빙|준비", text):
        return "신청 전에 빠뜨리지 않고 준비해야 할 자료를 확인하는 데 필요한 정보입니다."
    if re.search(r"신청|조회|절차|방법", text):
        return "실제 신청이나 조회를 진행할 때 순서와 확인사항을 파악하는 데 필요한 정보입니다."
    return "답변의 핵심 내용을 본인의 상황에 적용하기 전에 공식 기준과 대조할 때 필요한 정보입니다."


def default_basis_from_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Build a verified user explanation from the Evidence Pack without an LLM."""

    common = _common_from_result(result)
    payload = _mapping(result.get("payload"))
    pack = _mapping(
        result.get("augmented_pack")
        or common.get("dc_augmented_pack")
        or common.get("evidence_pack")
        or result.get("evidence_pack")
    )
    allowed_ids = set(_clean_list(payload.get("used_evidence_ids") or result.get("used_evidence_ids")))
    answer = _answer_from_result(result)
    question = _clean_text(
        result.get("question")
        or common.get("resolved_question")
        or common.get("current_question")
    )
    mappings: list[dict[str, Any]] = []
    seen_items: set[str] = set()
    for evidence in pack.get("evidence") or []:
        if not isinstance(evidence, Mapping):
            continue
        evidence_id = _clean_text(evidence.get("evidence_id"))
        if allowed_ids and evidence_id not in allowed_ids:
            continue
        title = _clean_text(
            evidence.get("section_title")
            or evidence.get("document_title")
            or evidence.get("title")
            or evidence_id
            or "공식 근거"
        )
        raw_content = (
            evidence.get("content")
            or evidence.get("parent_context")
            or evidence.get("context")
            or evidence.get("text")
        )
        content = _basis_relevant_excerpt(raw_content, question, answer)
        if not content:
            continue
        answer_point = _basis_answer_point(answer, content, title)
        signature = re.sub(r"\W+", "", f"{answer_point}{content}").lower()[:180]
        if not signature or signature in seen_items:
            continue
        seen_items.add(signature)
        mappings.append(
            {
                "claim": answer_point,
                "reason": content,
                "answer_point": answer_point,
                "evidence_summary": content,
                "user_meaning": _basis_user_meaning(answer_point, content),
                "caveat": "",
                "evidence_ids": [evidence_id] if evidence_id else [],
            }
        )
        if len(mappings) >= 4:
            break
    if not mappings:
        for source in _sources_from_result(result)[:5]:
            mappings.append(
                {
                    "claim": source["title"],
                    "reason": "답변에 연결된 공식 원문 페이지입니다.",
                    "answer_point": source["title"],
                    "evidence_summary": "답변에 연결된 예금보험공사 공식 안내입니다.",
                    "user_meaning": "세부 조건과 최신 안내는 연결된 공식 페이지에서 다시 확인할 수 있습니다.",
                    "caveat": "",
                    "evidence_ids": [],
                }
            )
    missing = _clean_list(payload.get("missing_information"))
    businesses = _businesses_from_result(result)
    subject = "·".join(businesses[:3]) if businesses else "질문과 관련된 업무"
    return {
        "schema_version": BASIS_EXPLANATION_SCHEMA_VERSION,
        "summary": f"{subject} 답변에 실제로 사용된 공식 정보와 사용자가 확인할 내용을 연결했습니다.",
        "items": copy.deepcopy(mappings),
        "mappings": mappings,
        "conditions": [],
        "exceptions": [],
        "checkpoints": missing,
        "limitations": missing,
        "additional_information_needed": missing,
        "sources": _sources_from_result(result),
    }


def load_entrypoint(value: str) -> Callable[..., Mapping[str, Any]]:
    """Load `module:function` or `C:/path/file.py:function`."""

    entrypoint = _clean_text(value)
    if ":" not in entrypoint:
        raise ValueError("KDIC_PIPELINE_ENTRYPOINT는 module:function 형식이어야 합니다.")
    module_value, function_name = entrypoint.rsplit(":", 1)
    candidate = Path(module_value)
    if candidate.suffix.lower() == ".py" or candidate.exists():
        resolved = candidate.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        spec = importlib.util.spec_from_file_location(
            f"kdic_runtime_{uuid.uuid4().hex}", resolved
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"파이프라인 파일을 불러올 수 없습니다: {resolved}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_value)
    function = getattr(module, function_name, None)
    if not callable(function):
        raise TypeError(f"호출 가능한 함수를 찾지 못했습니다: {entrypoint}")
    return function


class PipelineRuntime:
    def __init__(self, pipeline: Callable[..., Mapping[str, Any]] | None = None):
        self._lock = threading.RLock()
        self._pipeline = pipeline

    @property
    def configured(self) -> bool:
        with self._lock:
            return callable(self._pipeline)

    @property
    def name(self) -> str:
        with self._lock:
            pipeline = self._pipeline
        if pipeline is None:
            return "UNCONFIGURED"
        return _clean_text(getattr(pipeline, "name", None) or pipeline.__class__.__name__)

    @property
    def cache_namespace(self) -> str:
        """Identify answer-affecting runtime code so stale bundles cannot be reused."""

        with self._lock:
            pipeline = self._pipeline
        build = getattr(pipeline, "build_info", None) if pipeline is not None else None
        build = dict(build) if isinstance(build, Mapping) else {}
        answer_cache_revision = getattr(pipeline, "answer_cache_revision", "") if pipeline is not None else ""
        if callable(answer_cache_revision):
            answer_cache_revision = answer_cache_revision()
        return ":".join(
            value
            for value in (
                self.name,
                _clean_text(build.get("build_sha256")),
                _clean_text(build.get("overlay_revision")),
                _clean_text(answer_cache_revision),
            )
            if value
        )

    def set(self, pipeline: Callable[..., Mapping[str, Any]]) -> None:
        if not callable(pipeline):
            raise TypeError("pipeline은 호출 가능해야 합니다.")
        with self._lock:
            self._pipeline = pipeline

    def configure(self, api_key: str) -> None:
        with self._lock:
            pipeline = self._pipeline
        if pipeline is None:
            raise PipelineNotConfiguredError("KDIC 파이프라인이 연결되지 않았습니다.")
        configure = getattr(pipeline, "configure", None)
        if callable(configure):
            configure(api_key)

    def basis(self, result: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            pipeline = self._pipeline
        fallback = default_basis_from_result(result)
        basis = getattr(pipeline, "basis", None) if pipeline is not None else None
        if callable(basis):
            try:
                parameters = inspect.signature(basis).parameters
            except (TypeError, ValueError):
                parameters = {}
            payload = basis(result, base_basis=fallback) if "base_basis" in parameters else basis(result)
            if isinstance(payload, Mapping):
                return dict(payload)
        return fallback

    def record_cached_turn(
        self,
        question: str,
        answer: str,
        state: MutableMapping[str, Any],
    ) -> None:
        """Keep a cache-hit turn visible to the next context-dependent question."""

        with self._lock:
            pipeline = self._pipeline
        recorder = getattr(pipeline, "record_cached_turn", None) if pipeline else None
        if callable(recorder):
            recorder(question, answer, state)
            return
        state.setdefault("turns", []).extend(
            [
                {"role": "user", "content": _clean_text(question)},
                {"role": "assistant", "content": str(answer or "").strip()},
            ]
        )

    def run(
        self,
        question: str,
        state: MutableMapping[str, Any],
        progress: Callable[[int, str], None],
    ) -> dict[str, Any]:
        with self._lock:
            pipeline = self._pipeline
        if pipeline is None:
            raise PipelineNotConfiguredError(
                "KDIC 파이프라인이 연결되지 않았습니다. 실행 가이드의 어댑터 연결 단계를 먼저 수행하세요."
            )

        signature = inspect.signature(pipeline)
        kwargs: dict[str, Any] = {}
        if "state" in signature.parameters:
            kwargs["state"] = state
        if "progress" in signature.parameters:
            kwargs["progress"] = progress
        elif "progress_callback" in signature.parameters:
            kwargs["progress_callback"] = progress

        if kwargs:
            result = pipeline(question, **kwargs)
        else:
            parameter_count = len(signature.parameters)
            if parameter_count >= 3:
                result = pipeline(question, state, progress)
            elif parameter_count >= 2:
                result = pipeline(question, state)
            else:
                result = pipeline(question)
        if not isinstance(result, Mapping):
            raise TypeError("KDIC 파이프라인 결과는 dict 형식이어야 합니다.")
        return dict(result)


@dataclass
class SessionRecord:
    session_id: str
    state: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)


class InMemorySessionStore:
    """Development store. Replace with Redis/DB before multi-replica deployment."""

    def __init__(self, ttl_seconds: int = 86_400, max_sessions: int = 2_000):
        self.ttl_seconds = max(60, int(ttl_seconds))
        self.max_sessions = max(10, int(max_sessions))
        self._lock = threading.RLock()
        self._records: dict[str, SessionRecord] = {}

    def _cleanup_locked(self) -> None:
        cutoff = time.time() - self.ttl_seconds
        expired = [
            key for key, row in self._records.items() if row.updated_at < cutoff
        ]
        for key in expired:
            self._records.pop(key, None)
        if len(self._records) > self.max_sessions:
            ordered = sorted(self._records.values(), key=lambda row: row.updated_at)
            for row in ordered[: len(self._records) - self.max_sessions]:
                self._records.pop(row.session_id, None)

    def get(self, session_id: str) -> SessionRecord:
        key = _clean_text(session_id)
        if not key or len(key) > 200:
            raise ValueError("유효한 session_id가 필요합니다.")
        with self._lock:
            self._cleanup_locked()
            record = self._records.get(key)
            if record is None:
                record = SessionRecord(session_id=key)
                self._records[key] = record
            record.updated_at = time.time()
            return record

    def reset(self, session_id: str) -> None:
        record = self.get(session_id)
        with record.lock:
            record.state.clear()
            record.updated_at = time.time()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            self._cleanup_locked()
            return {
                "backend": "memory",
                "session_count": len(self._records),
                "ttl_seconds": self.ttl_seconds,
                "max_sessions": self.max_sessions,
            }


@dataclass
class CachedAnswerBundle:
    cache_key: str
    suggestion_id: str
    business: str
    keyword: str
    question: str
    public_result: dict[str, Any]
    raw_result: dict[str, Any]
    basis_result: dict[str, Any]
    pipeline_name: str
    runtime_revision: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    hit_count: int = 0


class InMemorySuggestionAnswerCache:
    """Validated recommendation answer bundles for local development and Colab."""

    def __init__(self, ttl_seconds: int = 2_592_000, max_entries: int = 200):
        self.ttl_seconds = max(60, int(ttl_seconds))
        self.max_entries = max(10, int(max_entries))
        self._lock = threading.RLock()
        self._records: dict[str, CachedAnswerBundle] = {}
        self._hits = 0
        self._misses = 0
        self._stores = 0

    def _cleanup_locked(self) -> None:
        cutoff = time.time() - self.ttl_seconds
        for key in [
            key for key, row in self._records.items() if row.updated_at < cutoff
        ]:
            self._records.pop(key, None)
        if len(self._records) > self.max_entries:
            ordered = sorted(self._records.values(), key=lambda row: row.updated_at)
            for row in ordered[: len(self._records) - self.max_entries]:
                self._records.pop(row.cache_key, None)

    def get(self, cache_key: str) -> CachedAnswerBundle | None:
        clean_key = _clean_text(cache_key)
        with self._lock:
            self._cleanup_locked()
            row = self._records.get(clean_key)
            if row is None:
                self._misses += 1
                return None
            row.updated_at = time.time()
            row.hit_count += 1
            self._hits += 1
            return copy.deepcopy(row)

    def peek(self, cache_key: str) -> CachedAnswerBundle | None:
        with self._lock:
            self._cleanup_locked()
            row = self._records.get(_clean_text(cache_key))
            return copy.deepcopy(row) if row is not None else None

    def put(self, bundle: CachedAnswerBundle) -> None:
        with self._lock:
            self._cleanup_locked()
            self._records[bundle.cache_key] = copy.deepcopy(bundle)
            self._stores += 1
            self._cleanup_locked()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            self._cleanup_locked()
            total = self._hits + self._misses
            return {
                "backend": "memory",
                "schema_version": SUGGESTION_CACHE_SCHEMA_VERSION,
                "entry_count": len(self._records),
                "hits": self._hits,
                "misses": self._misses,
                "stores": self._stores,
                "hit_rate": round(self._hits / total, 4) if total else 0.0,
                "ttl_seconds": self.ttl_seconds,
                "max_entries": self.max_entries,
            }


@dataclass
class JobRecord:
    job_id: str
    session_id: str
    question: str
    status: str = "queued"
    progress: int = 2
    stage: str = "질문을 전달했습니다."
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    suggestion_id: str = ""
    cache_key: str = ""
    result: dict[str, Any] | None = None
    raw_result: dict[str, Any] | None = None
    error: str = ""

    def public(self) -> dict[str, Any]:
        payload = {
            "job_id": self.job_id,
            "session_id": self.session_id,
            "question": self.question,
            "status": self.status,
            "progress": self.progress,
            "stage": self.stage,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if self.result is not None:
            payload["result"] = copy.deepcopy(self.result)
        if self.error:
            payload["error"] = self.error
        return payload


class InMemoryJobStore:
    """Development job store with bounded retention."""

    def __init__(self, ttl_seconds: int = 86_400, max_jobs: int = 5_000):
        self.ttl_seconds = max(300, int(ttl_seconds))
        self.max_jobs = max(50, int(max_jobs))
        self._lock = threading.RLock()
        self._records: dict[str, JobRecord] = {}

    def _cleanup_locked(self) -> None:
        cutoff = time.time() - self.ttl_seconds
        removable = [
            key
            for key, row in self._records.items()
            if row.updated_at < cutoff and row.status in {"done", "error"}
        ]
        for key in removable:
            self._records.pop(key, None)
        if len(self._records) > self.max_jobs:
            finished = sorted(
                (
                    row
                    for row in self._records.values()
                    if row.status in {"done", "error"}
                ),
                key=lambda row: row.updated_at,
            )
            for row in finished[: max(0, len(self._records) - self.max_jobs)]:
                self._records.pop(row.job_id, None)

    def create(
        self,
        session_id: str,
        question: str,
        suggestion_id: str = "",
        cache_key: str = "",
    ) -> JobRecord:
        record = JobRecord(
            job_id=uuid.uuid4().hex,
            session_id=_clean_text(session_id),
            question=_clean_text(question),
            suggestion_id=_clean_text(suggestion_id).upper(),
            cache_key=_clean_text(cache_key),
        )
        with self._lock:
            self._cleanup_locked()
            self._records[record.job_id] = record
        return record

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            self._cleanup_locked()
            return self._records.get(_clean_text(job_id))

    def update(self, job_id: str, **values: Any) -> JobRecord:
        with self._lock:
            record = self._records.get(job_id)
            if record is None:
                raise KeyError(job_id)
            for key, value in values.items():
                if hasattr(record, key):
                    setattr(record, key, value)
            record.updated_at = time.time()
            return record

    def list_public(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            self._cleanup_locked()
            rows = sorted(
                self._records.values(), key=lambda row: row.created_at, reverse=True
            )[: max(1, min(int(limit), 500))]
            return [row.public() for row in rows]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            self._cleanup_locked()
            statuses: dict[str, int] = {}
            for row in self._records.values():
                statuses[row.status] = statuses.get(row.status, 0) + 1
            return {
                "backend": "memory",
                "job_count": len(self._records),
                "statuses": statuses,
                "ttl_seconds": self.ttl_seconds,
                "max_jobs": self.max_jobs,
            }


class KDICJobService:
    def __init__(
        self,
        runtime: PipelineRuntime,
        sessions: InMemorySessionStore | None = None,
        jobs: InMemoryJobStore | None = None,
        suggestion_cache: InMemorySuggestionAnswerCache | None = None,
        max_workers: int = 2,
    ):
        self.runtime = runtime
        self.sessions = sessions or InMemorySessionStore()
        self.jobs = jobs or InMemoryJobStore()
        self.suggestion_cache = suggestion_cache or InMemorySuggestionAnswerCache()
        self.executor = ThreadPoolExecutor(
            max_workers=max(1, int(max_workers)), thread_name_prefix="kdic-pipeline"
        )

    def _cache_key(self, suggestion_id: str) -> str:
        return ":".join(
            [
                SUGGESTION_CACHE_SCHEMA_VERSION,
                self.runtime.cache_namespace,
                _clean_text(suggestion_id).upper(),
            ]
        )

    @staticmethod
    def _cache_eligibility(public: Mapping[str, Any]) -> tuple[bool, str]:
        if _clean_text(public.get("route")).upper() != "RETRIEVE":
            return False, "ROUTE_NOT_RETRIEVE"
        if not str(public.get("answer") or "").strip():
            return False, "EMPTY_ANSWER"
        if bool(public.get("context_used")):
            return False, "CONTEXT_DEPENDENT_ANSWER"
        if _clean_text(public.get("coverage_status")).upper() in {
            "INSUFFICIENT",
            "EVIDENCE_INSUFFICIENT",
        }:
            return False, "INSUFFICIENT_COVERAGE"
        if public.get("validation_passed") is False:
            return False, "VALIDATION_FAILED"
        if not list(public.get("sources") or []):
            return False, "NO_OFFICIAL_SOURCES"
        return True, "VALIDATED_STANDALONE_RETRIEVE"

    def submit(
        self,
        session_id: str,
        question: str,
        suggestion_id: str = "",
    ) -> str:
        clean_question = _clean_text(question)
        if not clean_question:
            raise ValueError("질문이 비어 있습니다.")
        if len(clean_question) > 4_000:
            raise ValueError("질문은 4,000자 이하여야 합니다.")
        session = self.sessions.get(session_id)
        suggestion = resolve_registered_suggestion(clean_question, suggestion_id)
        clean_suggestion_id = suggestion.get("suggestion_id", "") if suggestion else ""
        cache_key = self._cache_key(clean_suggestion_id) if clean_suggestion_id else ""
        record = self.jobs.create(
            session.session_id,
            clean_question,
            suggestion_id=clean_suggestion_id,
            cache_key=cache_key,
        )
        if cache_key:
            lookup_started = time.perf_counter()
            bundle = self.suggestion_cache.get(cache_key)
            lookup_ms = (time.perf_counter() - lookup_started) * 1000.0
            if bundle is not None and bundle.question == clean_question:
                try:
                    public = copy.deepcopy(bundle.public_result)
                    public["answer"] = _normalize_answer_text(public.get("answer")) or (
                        "답변을 준비하지 못했습니다. 잠시 후 다시 시도해 주세요."
                    )
                    public["origin_latency_seconds"] = copy.deepcopy(
                        public.get("latency_seconds") or {}
                    )
                    public["latency_seconds"] = {
                        "추천 답변 캐시 조회": round(lookup_ms / 1000.0, 4)
                    }
                    public["suggestion_cache"] = {
                        "eligible": True,
                        "hit": True,
                        "stored": True,
                        "suggestion_id": clean_suggestion_id,
                        "source": str(
                            self.suggestion_cache.stats().get("backend") or "cache"
                        ).upper()
                        + "_ANSWER_BUNDLE",
                        "lookup_ms": round(lookup_ms, 3),
                        "age_seconds": round(time.time() - bundle.created_at, 3),
                        "skipped_stages": [
                            "질의분석",
                            "질문 임베딩",
                            "검색",
                            "BAAI Reranker",
                            "답변 LLM",
                        ],
                    }
                    with session.lock:
                        self.runtime.record_cached_turn(
                            clean_question,
                            public["answer"],
                            session.state,
                        )
                        session.updated_at = time.time()
                        save = getattr(self.sessions, "save", None)
                        if callable(save):
                            save(session)
                    self.jobs.update(
                        record.job_id,
                        status="done",
                        progress=100,
                        stage="저장된 검증 답변을 불러왔습니다.",
                        result=public,
                        raw_result=copy.deepcopy(bundle.raw_result),
                    )
                    return record.job_id
                except Exception:
                    # If state recording is incompatible, run the live pipeline
                    # so the next context-dependent question remains correct.
                    pass
        self.executor.submit(self._run, record.job_id)
        return record.job_id

    def _progress(self, job_id: str, value: int, stage: str) -> None:
        progress = max(2, min(99, int(value)))
        self.jobs.update(job_id, progress=progress, stage=_clean_text(stage))

    def _run(self, job_id: str) -> None:
        record = self.jobs.get(job_id)
        if record is None:
            return
        try:
            self.jobs.update(
                job_id,
                status="running",
                progress=5,
                stage="질문의 핵심 내용을 확인하고 있습니다.",
            )
            session = self.sessions.get(record.session_id)
            # Only turns from the same conversation are serialized. Other sessions
            # can run concurrently up to max_workers.
            with session.lock:
                raw = self.runtime.run(
                    record.question,
                    session.state,
                    lambda value, stage: self._progress(job_id, value, stage),
                )
                session.updated_at = time.time()
                # PostgresSessionStore mutates a snapshot, not the stored row, so
                # it needs an explicit save(). InMemorySessionStore keeps state by
                # reference and has no save() -- skip it there.
                save = getattr(self.sessions, "save", None)
                if callable(save):
                    save(session)
            public = normalize_public_result(raw)
            if record.cache_key and record.suggestion_id:
                eligible, reason = self._cache_eligibility(public)
                stored = False
                if eligible:
                    suggestion = _SUGGESTION_BY_ID[record.suggestion_id]
                    basis_result = self.runtime.basis(raw)
                    self.suggestion_cache.put(
                        CachedAnswerBundle(
                            cache_key=record.cache_key,
                            suggestion_id=record.suggestion_id,
                            business=suggestion["business"],
                            keyword=suggestion["label"],
                            question=record.question,
                            public_result=copy.deepcopy(public),
                            raw_result=copy.deepcopy(raw),
                            basis_result=copy.deepcopy(basis_result),
                            pipeline_name=self.runtime.name,
                            runtime_revision=self.runtime.cache_namespace,
                        )
                    )
                    stored = True
                public["suggestion_cache"] = {
                    "eligible": eligible,
                    "hit": False,
                    "stored": stored,
                    "suggestion_id": record.suggestion_id,
                    "source": "LIVE_PIPELINE",
                    "reason": reason,
                    "skipped_stages": [],
                }
            self.jobs.update(
                job_id,
                status="done",
                progress=100,
                stage="답변을 준비했습니다.",
                result=public,
                raw_result=raw,
            )
        except Exception as error:  # The API exposes a sanitized class + message.
            self.jobs.update(
                job_id,
                status="error",
                progress=100,
                stage="처리 중 오류가 발생했습니다.",
                error=f"{type(error).__name__}: {error}",
            )

    def basis(self, job_id: str) -> dict[str, Any]:
        record = self.jobs.get(job_id)
        if record is None:
            raise KeyError(job_id)
        if record.status != "done" or record.raw_result is None:
            raise RuntimeError("완료된 답변만 근거를 조회할 수 있습니다.")
        if record.cache_key:
            bundle = self.suggestion_cache.peek(record.cache_key)
            if (
                bundle is not None
                and bundle.basis_result
                and _clean_text(bundle.basis_result.get("schema_version"))
                == BASIS_EXPLANATION_SCHEMA_VERSION
            ):
                return copy.deepcopy(bundle.basis_result)
        return self.runtime.basis(record.raw_result)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=False)


class DemoKDICPipeline:
    """A deterministic UI smoke-test pipeline. It is never enabled by default."""

    name = "DEMO_KDIC_PIPELINE"

    def __call__(
        self,
        question: str,
        state: MutableMapping[str, Any],
        progress: Callable[[int, str], None] | None = None,
    ) -> Mapping[str, Any]:
        progress = progress or (lambda *_: None)
        progress(20, "질의 경로를 확인했습니다.")
        text = _clean_text(question)
        if any(word in text.lower() for word in ("안녕", "고마워")):
            route = "DIRECT_RESPONSE"
            answer = "안녕하세요. 예금보험공사 관련 내용을 질문해 주세요."
        elif any(word in text for word in ("날씨", "주식", "맛집")):
            route = "OUT_OF_SCOPE"
            answer = "이 챗봇은 예금보험공사 업무 범위의 질문을 안내합니다."
        elif text in {"얼마나 걸리나요?", "신청 서류는?"}:
            route = "CLARIFY"
            answer = "어떤 업무에 관한 질문인지 선택해 주세요."
        else:
            route = "RETRIEVE"
            answer = (
                "이것은 화면과 API 연결을 확인하기 위한 데모 답변입니다. "
                "실제 운영에서는 최신 V1.5 + C/D-C 파이프라인을 연결해야 합니다."
            )
        progress(70, "공식 근거를 정리했습니다.")
        state.setdefault("turns", []).extend(
            [
                {"role": "user", "content": text},
                {"role": "assistant", "content": answer},
            ]
        )
        analysis = {
            "businesses": ["착오송금 반환지원"] if route == "RETRIEVE" else [],
            "context_used": False,
            "context_resolution": {
                "pending_clarification": {
                    "options": ["착오송금 반환지원", "채무조정"]
                }
            },
        }
        return {
            "route": route,
            "answer": answer,
            "analysis": analysis,
            "sources": [
                {
                    "title": "예금보험공사 공식 홈페이지",
                    "url": "https://www.kdic.or.kr/",
                }
            ]
            if route == "RETRIEVE"
            else [],
            "action_links": [],
            "latency_ms": {"질의분석": 1.0, "검색": 2.0, "답변": 3.0},
        }

