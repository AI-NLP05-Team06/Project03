from __future__ import annotations

"""V1.5 문맥 개선 정책: 현재 질문 우선, 규칙 우선, LLM은 경계 사례만."""

import copy
import re
import time
from typing import Any, Callable, Mapping, Sequence


BUSINESS_PATTERNS: dict[str, tuple[str, ...]] = {
    "예금자보호": ("예금자보호", "보호한도", "보호 대상", "보호대상"),
    "예금보험금": ("예금보험금", "보험금 지급", "보험금 신청"),
    "고객 미수령금": ("고객 미수령금", "미수령금", "미수령 예금"),
    "착오송금 반환지원": ("착오송금 반환지원", "착오송금 반환", "착오송금", "잘못 송금", "잘못 보낸 돈", "잘못 받은 돈"),
    "채무조정": ("채무조정", "신용회복 지원", "채무 감면", "상환 유예"),
    "은닉재산 신고": ("은닉재산 신고", "은닉재산", "숨긴 재산 신고"),
}

INTENT_TERMS = (
    "한도", "대상", "자격", "신청", "서류", "절차", "방법", "기간", "기한",
    "얼마", "언제", "비용", "수수료", "왜", "이유", "종류", "조건", "예외",
)
EXCLUSION_PATTERN = re.compile(r"(?:말고|제외(?:하고|한|해|해서)?|빼고)")
CORRECTION_PATTERN = re.compile(r"(?:아니고|아니라|정정)")
CANCEL_PATTERN = re.compile(r"^(?:그만|취소|됐어|괜찮아|필요\s*없어)[.!?\s]*$")
STRONG_FOLLOWUP_PATTERN = re.compile(
    r"^(?:(?:그럼|그러면|그건|그거|그 경우|이건|이거|여기서)\s*)?"
    r"(?:얼마나\s*걸리나요?|기간(?:은|이)?(?:요)?|언제(?:까지)?(?:인가요|예요)?|"
    r"(?:서류|준비물|신청|절차|방법|대상|자격|금액|한도|이유|수취인|송금인)"
    r"(?:은|는|이)?(?:요)?|왜(?:요)?)\s*[?.!]*$"
)
AMBIGUOUS_REFERENCE_PATTERN = re.compile(
    r"(?:그때|아까|이전에|그쪽|그 부분|그 내용|그거 말고|신청하는 쪽|처리하는 쪽)"
)


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def detect_businesses(text: str) -> list[str]:
    cleaned = _clean(text)
    found: list[tuple[int, str]] = []
    for business, terms in BUSINESS_PATTERNS.items():
        positions = [cleaned.find(term) for term in terms if term in cleaned]
        if positions:
            found.append((min(positions), business))
    return [business for _, business in sorted(found)]


def _current_question_complete(question: str, explicit_businesses: Sequence[str]) -> bool:
    if not explicit_businesses:
        return False
    has_intent = any(term in question for term in INTENT_TERMS)
    has_explanation = bool(re.search(r"(?:알려|설명|궁금|무엇|뭔가|어떤)", question))
    unresolved_reference = bool(re.search(r"(?:그거|그건|그것|그 경우|저거|그쪽)(?!\s*말고)", question))
    return bool((has_intent or has_explanation or len(question) >= 10) and not unresolved_reference)


def _selected_pending(question: str, pending: Mapping[str, Any]) -> str | None:
    options = [_clean(value) for value in pending.get("options") or []]
    match = re.fullmatch(r"(?:선택지\s*)?(\d+)(?:번)?[.!?\s]*", re.sub(r"\s+", "", question))
    if match:
        index = int(match.group(1)) - 1
        if 0 <= index < len(options):
            return options[index]
    for option in options:
        if option and (option in question or question in option):
            return option
    return None


def _excluded_explicit_businesses(question: str, businesses: Sequence[str]) -> list[str]:
    if not EXCLUSION_PATTERN.search(question):
        return []
    output: list[str] = []
    for business in businesses:
        positions = [question.find(term) for term in BUSINESS_PATTERNS[business] if term in question]
        if not positions:
            continue
        start = min(positions)
        if EXCLUSION_PATTERN.search(question[start:start + 40]):
            output.append(business)
    return output


def _clarify(
    *,
    question: str,
    state: dict[str, Any],
    reason: str,
    message: str,
    options: Sequence[str],
    missing_slots: Sequence[str],
    started: float,
    llm_trace: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    pending = {
        "original_question": question, "reason": reason,
        "options": list(options), "missing_slots": list(missing_slots),
    }
    state["pending_clarification"] = pending
    return {
        "route": "CLARIFY", "dialogue_act": "CLARIFY",
        "original_question": question, "resolved_question": "",
        "current_question_complete": False, "context_used": False,
        "reason": reason, "clarification_message": message,
        "active_businesses": list(state.get("active_businesses") or []),
        "excluded_businesses": list(state.get("excluded_businesses") or []),
        "actor_role": state.get("actor_role"), "missing_slots": list(missing_slots),
        "pending_clarification": pending, "llm_judgment": dict(llm_trace or {}),
        "latency_ms": (time.perf_counter() - started) * 1000,
    }


def _validated_llm_decision(
    raw: Mapping[str, Any],
    *,
    explicit_businesses: Sequence[str],
    active_businesses: Sequence[str],
) -> dict[str, Any] | None:
    allowed_acts = {"NEW_TOPIC", "FOLLOW_UP", "CORRECTION", "EXCLUSION", "AMBIGUOUS"}
    act = _clean(raw.get("dialogue_act")).upper()
    confidence = float(raw.get("confidence") or 0.0)
    selected = [_clean(value) for value in raw.get("selected_businesses") or [] if _clean(value)]
    if act not in allowed_acts or confidence < 0.85:
        return None
    allowed_businesses = set(explicit_businesses) | set(active_businesses)
    if selected and not set(selected).issubset(allowed_businesses):
        return None
    if explicit_businesses and selected and set(selected) != set(explicit_businesses):
        return None
    return {
        "dialogue_act": act,
        "current_question_complete": bool(raw.get("current_question_complete")),
        "context_required": bool(raw.get("context_required")),
        "selected_businesses": selected,
        "excluded_businesses": [_clean(value) for value in raw.get("excluded_businesses") or [] if _clean(value)],
        "actor_role": _clean(raw.get("actor_role")) or None,
        "missing_slots": [_clean(value) for value in raw.get("missing_slots") or [] if _clean(value)],
        "confidence": confidence,
        "reason_code": _clean(raw.get("reason_code")),
    }


def new_context_state() -> dict[str, Any]:
    return {
        "turns": [], "active_businesses": [], "excluded_businesses": [],
        "actor_role": None, "pending_clarification": None,
        "last_resolved_question": "",
    }


def resolve_context_v2(
    question: str,
    *,
    state: dict[str, Any],
    llm_classifier: Callable[[str, Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    original = _clean(question)
    if not original:
        raise ValueError("질문이 비어 있습니다.")
    for key, default in new_context_state().items():
        state.setdefault(key, copy.deepcopy(default))

    if CANCEL_PATTERN.fullmatch(original):
        state["pending_clarification"] = None
        return {
            "route": "DIRECT_RESPONSE", "dialogue_act": "CANCEL",
            "original_question": original, "resolved_question": "",
            "current_question_complete": True, "context_used": False,
            "reason": "EXPLICIT_CANCEL", "direct_response": "알겠습니다. 현재 요청을 중단했습니다.",
            "active_businesses": list(state["active_businesses"]),
            "excluded_businesses": list(state["excluded_businesses"]),
            "actor_role": state.get("actor_role"), "missing_slots": [],
            "pending_clarification": None, "llm_judgment": {},
            "latency_ms": (time.perf_counter() - started) * 1000,
        }

    explicit = detect_businesses(original)
    complete = _current_question_complete(original, explicit)
    pending = state.get("pending_clarification") or {}
    selected = _selected_pending(original, pending) if pending else None

    if selected:
        base = _clean(pending.get("original_question"))
        if selected in {"송금인", "보낸 사람"}:
            state["actor_role"] = "SENDER"
            prefix = f"{state['active_businesses'][0]}에 관하여 " if len(state["active_businesses"]) == 1 else ""
            resolved = f"{prefix}{base} 송금인 기준"
        elif selected in {"수취인", "받은 사람"}:
            state["actor_role"] = "RECIPIENT"
            prefix = f"{state['active_businesses'][0]}에 관하여 " if len(state["active_businesses"]) == 1 else ""
            resolved = f"{prefix}{base} 수취인 기준"
        else:
            state["active_businesses"] = [selected]
            resolved = f"{selected}에 관하여 {base}"
        state["pending_clarification"] = None
        state["last_resolved_question"] = resolved
        return {
            "route": "CONTINUE", "dialogue_act": "SELECT_OPTION",
            "original_question": original, "resolved_question": resolved,
            "current_question_complete": False, "context_used": True,
            "reason": "PENDING_OPTION_MATCH", "clarification_message": "",
            "active_businesses": list(state["active_businesses"]),
            "excluded_businesses": list(state["excluded_businesses"]),
            "actor_role": state.get("actor_role"), "missing_slots": [],
            "pending_clarification": None, "llm_judgment": {},
            "latency_ms": (time.perf_counter() - started) * 1000,
        }

    excluded_now = _excluded_explicit_businesses(original, explicit)
    if excluded_now:
        state["excluded_businesses"] = list(dict.fromkeys(state["excluded_businesses"] + excluded_now))
        state["active_businesses"] = [
            business for business in state["active_businesses"] if business not in excluded_now
        ]
        remaining = [business for business in explicit if business not in excluded_now]
        state["pending_clarification"] = None
        if not remaining:
            return _clarify(
                question=original, state=state, reason="EXCLUSION_WITHOUT_REPLACEMENT",
                message=f"{', '.join(excluded_now)} 업무는 제외하겠습니다. 대신 어떤 업무를 안내할까요?",
                options=[business for business in BUSINESS_PATTERNS if business not in state["excluded_businesses"]],
                missing_slots=["business_function"], started=started,
            )
        state["active_businesses"] = remaining
        state["last_resolved_question"] = original
        return {
            "route": "CONTINUE", "dialogue_act": "CORRECTION",
            "original_question": original, "resolved_question": original,
            "current_question_complete": True, "context_used": False,
            "reason": "EXPLICIT_EXCLUSION_WITH_REPLACEMENT", "clarification_message": "",
            "active_businesses": list(state["active_businesses"]),
            "excluded_businesses": list(state["excluded_businesses"]),
            "actor_role": state.get("actor_role"), "missing_slots": [],
            "pending_clarification": None, "llm_judgment": {},
            "latency_ms": (time.perf_counter() - started) * 1000,
        }

    # 현재 질문이 독립적으로 완결되면 이전 pending과 업무를 무조건 덮어쓴다.
    if complete:
        state["active_businesses"] = explicit
        state["pending_clarification"] = None
        state["last_resolved_question"] = original
        return {
            "route": "CONTINUE", "dialogue_act": "CORRECTION" if CORRECTION_PATTERN.search(original) else "NEW_TOPIC",
            "original_question": original, "resolved_question": original,
            "current_question_complete": True, "context_used": False,
            "reason": "CURRENT_QUESTION_COMPLETE", "clarification_message": "",
            "active_businesses": list(state["active_businesses"]),
            "excluded_businesses": list(state["excluded_businesses"]),
            "actor_role": state.get("actor_role"), "missing_slots": [],
            "pending_clarification": None, "llm_judgment": {},
            "latency_ms": (time.perf_counter() - started) * 1000,
        }

    if STRONG_FOLLOWUP_PATTERN.fullmatch(original) and not explicit:
        active = list(state.get("active_businesses") or [])
        if len(active) == 1:
            if active[0] == "착오송금 반환지원" and re.search(r"(?:얼마나\s*걸|기간|언제)", original):
                return _clarify(
                    question=original, state=state, reason="MISTAKEN_TRANSFER_TIME_SCOPE_AMBIGUOUS",
                    message="착오송금의 어느 기간을 묻는지 확인이 필요합니다. 송금인의 반환지원 처리기간과 수취인의 자진반환 관련 기간 중 선택해 주세요.",
                    options=["송금인", "수취인"], missing_slots=["actor_role", "process_stage"], started=started,
                )
            resolved = f"{active[0]}에 관하여 {original}"
            state["pending_clarification"] = None
            state["last_resolved_question"] = resolved
            return {
                "route": "CONTINUE", "dialogue_act": "FOLLOW_UP",
                "original_question": original, "resolved_question": resolved,
                "current_question_complete": False, "context_used": True,
                "reason": "UNIQUE_ACTIVE_BUSINESS", "clarification_message": "",
                "active_businesses": active, "excluded_businesses": list(state["excluded_businesses"]),
                "actor_role": state.get("actor_role"), "missing_slots": [],
                "pending_clarification": None, "llm_judgment": {},
                "latency_ms": (time.perf_counter() - started) * 1000,
            }
        return _clarify(
            question=original, state=state,
            reason="FOLLOW_UP_WITHOUT_UNIQUE_BUSINESS",
            message="어떤 업무에 관한 후속 질문인지 알려주세요.",
            options=active or list(BUSINESS_PATTERNS), missing_slots=["business_function"], started=started,
        )

    # 규칙 경계 사례에서만 LLM을 호출한다.
    if AMBIGUOUS_REFERENCE_PATTERN.search(original) and llm_classifier is not None:
        raw = dict(llm_classifier(original, state) or {})
        decision = _validated_llm_decision(raw, explicit_businesses=explicit, active_businesses=state["active_businesses"])
        trace = {"called": True, "raw": raw, "accepted": bool(decision), "decision": decision}
        if decision and decision["dialogue_act"] == "FOLLOW_UP" and len(decision["selected_businesses"] or state["active_businesses"]) == 1:
            business = (decision["selected_businesses"] or state["active_businesses"])[0]
            resolved = f"{business}에 관하여 {original}"
            state["active_businesses"] = [business]
            state["pending_clarification"] = None
            state["last_resolved_question"] = resolved
            return {
                "route": "CONTINUE", "dialogue_act": "FOLLOW_UP",
                "original_question": original, "resolved_question": resolved,
                "current_question_complete": False, "context_used": True,
                "reason": "LLM_STRUCTURED_FOLLOW_UP", "clarification_message": "",
                "active_businesses": [business], "excluded_businesses": list(state["excluded_businesses"]),
                "actor_role": decision.get("actor_role"), "missing_slots": [],
                "pending_clarification": None, "llm_judgment": trace,
                "latency_ms": (time.perf_counter() - started) * 1000,
            }
        if decision and decision["current_question_complete"]:
            state["pending_clarification"] = None
            state["last_resolved_question"] = original
            return {
                "route": "CONTINUE", "dialogue_act": decision["dialogue_act"],
                "original_question": original, "resolved_question": original,
                "current_question_complete": True, "context_used": False,
                "reason": "LLM_STRUCTURED_NEW_TOPIC", "clarification_message": "",
                "active_businesses": list(state["active_businesses"]),
                "excluded_businesses": list(state["excluded_businesses"]),
                "actor_role": decision.get("actor_role"), "missing_slots": [],
                "pending_clarification": None, "llm_judgment": trace,
                "latency_ms": (time.perf_counter() - started) * 1000,
            }
        return _clarify(
            question=original, state=state, reason="AMBIGUOUS_AFTER_STRUCTURED_JUDGMENT",
            message="현재 질문이 이전 내용의 후속 질문인지 새로운 질문인지 확인해 주세요.",
            options=list(state["active_businesses"]), missing_slots=["dialogue_target"], started=started, llm_trace=trace,
        )

    # 문맥 필요성이 명확하지 않으면 이전 상태를 섞지 않고 원문을 V1.5에 전달한다.
    state["pending_clarification"] = None if pending else state["pending_clarification"]
    state["last_resolved_question"] = original
    return {
        "route": "CONTINUE", "dialogue_act": "NEW_QUESTION_UNCHANGED",
        "original_question": original, "resolved_question": original,
        "current_question_complete": False, "context_used": False,
        "reason": "CONTEXT_NOT_PROVEN_USE_ORIGINAL", "clarification_message": "",
        "active_businesses": list(state["active_businesses"]),
        "excluded_businesses": list(state["excluded_businesses"]),
        "actor_role": state.get("actor_role"), "missing_slots": [],
        "pending_clarification": state.get("pending_clarification"), "llm_judgment": {},
        "latency_ms": (time.perf_counter() - started) * 1000,
    }