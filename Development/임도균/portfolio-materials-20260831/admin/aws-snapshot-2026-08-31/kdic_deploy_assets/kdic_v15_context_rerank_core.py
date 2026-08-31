from __future__ import annotations

import re
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


BUSINESS_LABELS = (
    "예금자보호제도",
    "예금보험금 안내",
    "고객 미수령금 신청",
    "착오송금 반환 신청",
    "채무조정 안내",
    "은닉재산 신고",
)

BUSINESS_ALIASES: dict[str, tuple[str, ...]] = {
    "예금자보호제도": (
        "예금자보호", "보호한도", "보호 대상", "보호대상",
    ),
    "예금보험금 안내": (
        "예금보험금", "보험금 지급", "보험사고",
    ),
    "고객 미수령금 신청": (
        "미수령금", "파산배당금", "개산지급금 정산금",
    ),
    "착오송금 반환 신청": (
        "착오송금", "잘못 송금", "잘못송금", "반환지원",
    ),
    "채무조정 안내": (
        "채무조정", "신용회복", "채무감면",
    ),
    "은닉재산 신고": (
        "은닉재산", "은닉 재산",
    ),
}

INTENT_ONLY_PATTERN = re.compile(
    r"(?:신청|접수|서류|구비서류|준비물|조건|자격|대상|방법|절차|"
    r"기간|기한|금액|한도|조회|상태|연락처|전화번호)"
)
REFERENCE_PATTERN = re.compile(
    r"(?:^|\s)(?:그거|이거|그것|이것|그\s*신청|해당\s*신청|그\s*경우|"
    r"해당\s*경우|그러면|그럼|거기는|거기서)(?:\s|$|[?!.])"
)
SELECTION_PATTERN = re.compile(r"^\s*(\d{1,2})\s*(?:번)?\s*$")


def _clean_text(value: Any) -> str:
    text = str(value or "").replace("\x00", " ")
    return re.sub(r"\s+", " ", text).strip()


def _ordered_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        cleaned = _clean_text(value)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            output.append(cleaned)
    return output


def detect_businesses(text: str) -> list[str]:
    cleaned = _clean_text(text).lower()
    return [
        business
        for business, aliases in BUSINESS_ALIASES.items()
        if any(alias.lower() in cleaned for alias in aliases)
    ]


def _user_messages(previous_turns: Any) -> list[str]:
    if not isinstance(previous_turns, Sequence) or isinstance(previous_turns, (str, bytes)):
        return []
    output: list[str] = []
    for turn in previous_turns:
        if not isinstance(turn, Mapping):
            continue
        role = _clean_text(turn.get("role")).lower()
        if role:
            if role != "user":
                continue
            content = _clean_text(turn.get("content"))
            if content:
                output.append(content)
            continue
        user = _clean_text(turn.get("user") or turn.get("query"))
        if user:
            output.append(user)
    return output


def latest_context_businesses(
    previous_turns: Any,
    *,
    confirmed_businesses: Sequence[str] | None = None,
    detector: Callable[[str], list[str]] = detect_businesses,
) -> list[str]:
    confirmed = _ordered_unique(confirmed_businesses or [])
    if confirmed:
        return confirmed
    for message in reversed(_user_messages(previous_turns)):
        found = _ordered_unique(detector(message))
        if found:
            return found
    return []


def _is_context_dependent(question: str) -> bool:
    compact_length = len(re.sub(r"\s+", "", question))
    short_intent_only = compact_length <= 30 and bool(INTENT_ONLY_PATTERN.search(question))
    has_reference = bool(REFERENCE_PATTERN.search(question))
    return short_intent_only or has_reference


def _clarification_message(candidates: Sequence[str], *, repeated: bool = False) -> str:
    candidates = _ordered_unique(candidates) or list(BUSINESS_LABELS)
    prefix = (
        "아직 어떤 업무를 말씀하시는지 확인하기 어렵습니다."
        if repeated
        else "어떤 업무에 관한 질문인지 확인이 필요합니다."
    )
    lines = [prefix, "", "아래에서 선택하거나 업무명을 직접 입력해 주세요.", ""]
    lines.extend(f"{index}. {business}" for index, business in enumerate(candidates, start=1))
    return "\n".join(lines)


def _pending_payload(
    *,
    original_question: str,
    candidates: Sequence[str],
    clarification_count: int,
) -> dict[str, Any]:
    return {
        "active": True,
        "original_question": original_question,
        "missing_slots": ["business_function"],
        "business_candidates": _ordered_unique(candidates) or list(BUSINESS_LABELS),
        "clarification_count": int(clarification_count),
    }


def resolve_conversational_question(
    question: str,
    *,
    previous_turns: Any = None,
    pending_clarification: Mapping[str, Any] | None = None,
    confirmed_businesses: Sequence[str] | None = None,
    detector: Callable[[str], list[str]] = detect_businesses,
) -> dict[str, Any]:
    """검색 전 문맥을 보수적으로 복원하거나 CLARIFY를 반환한다.

    업무를 추정할 근거가 하나로 수렴하지 않으면 검색을 허용하지 않는다.
    """
    started = time.perf_counter()
    original = _clean_text(question)
    if not original:
        raise ValueError("사용자 질문이 비어 있습니다.")

    explicit_businesses = _ordered_unique(detector(original))
    pending = dict(pending_clarification or {})
    pending_active = bool(pending.get("active"))

    if pending_active:
        candidates = _ordered_unique(pending.get("business_candidates") or BUSINESS_LABELS)
        selected: list[str] = []
        numeric = SELECTION_PATTERN.fullmatch(original)
        if numeric:
            index = int(numeric.group(1)) - 1
            if 0 <= index < len(candidates):
                selected = [candidates[index]]
        if not selected:
            selected = [item for item in explicit_businesses if item in candidates]
        if not selected and len(explicit_businesses) == 1:
            selected = explicit_businesses

        if len(selected) == 1:
            pending_question = _clean_text(pending.get("original_question"))
            is_short_selection = len(re.sub(r"\s+", "", original)) <= 20
            resolved = (
                f"{selected[0]} {pending_question}"
                if pending_question and is_short_selection
                else original
            )
            return {
                "route": "RETRIEVE",
                "original_question": original,
                "resolved_question": resolved,
                "context_used": True,
                "context_businesses": selected,
                "resolution_reason": "PENDING_CLARIFICATION_RESOLVED",
                "clarification_message": "",
                "pending_clarification": None,
                "latency_ms": (time.perf_counter() - started) * 1000,
            }

        count = int(pending.get("clarification_count") or 1) + 1
        return {
            "route": "CLARIFY",
            "original_question": original,
            "resolved_question": "",
            "context_used": False,
            "context_businesses": candidates,
            "resolution_reason": "PENDING_CLARIFICATION_UNRESOLVED",
            "clarification_message": _clarification_message(candidates, repeated=True),
            "pending_clarification": _pending_payload(
                original_question=_clean_text(pending.get("original_question")) or original,
                candidates=candidates,
                clarification_count=count,
            ),
            "escalation_recommended": count >= 2,
            "latency_ms": (time.perf_counter() - started) * 1000,
        }

    if explicit_businesses:
        return {
            "route": "CONTINUE",
            "original_question": original,
            "resolved_question": original,
            "context_used": False,
            "context_businesses": explicit_businesses,
            "resolution_reason": "EXPLICIT_BUSINESS_IN_CURRENT_QUESTION",
            "clarification_message": "",
            "pending_clarification": None,
            "latency_ms": (time.perf_counter() - started) * 1000,
        }

    if not _is_context_dependent(original):
        return {
            "route": "CONTINUE",
            "original_question": original,
            "resolved_question": original,
            "context_used": False,
            "context_businesses": [],
            "resolution_reason": "STANDALONE_OR_BASE_ROUTER_DECISION",
            "clarification_message": "",
            "pending_clarification": None,
            "latency_ms": (time.perf_counter() - started) * 1000,
        }

    context_businesses = latest_context_businesses(
        previous_turns,
        confirmed_businesses=confirmed_businesses,
        detector=detector,
    )
    if len(context_businesses) == 1:
        return {
            "route": "RETRIEVE",
            "original_question": original,
            "resolved_question": f"{context_businesses[0]} {original}",
            "context_used": True,
            "context_businesses": context_businesses,
            "resolution_reason": "UNIQUE_PREVIOUS_BUSINESS",
            "clarification_message": "",
            "pending_clarification": None,
            "latency_ms": (time.perf_counter() - started) * 1000,
        }

    candidates = context_businesses or list(BUSINESS_LABELS)
    reason = "MULTIPLE_PREVIOUS_BUSINESSES" if len(context_businesses) > 1 else "BUSINESS_NOT_SPECIFIED"
    return {
        "route": "CLARIFY",
        "original_question": original,
        "resolved_question": "",
        "context_used": False,
        "context_businesses": context_businesses,
        "resolution_reason": reason,
        "clarification_message": _clarification_message(candidates),
        "pending_clarification": _pending_payload(
            original_question=original,
            candidates=candidates,
            clarification_count=1,
        ),
        "escalation_recommended": False,
        "latency_ms": (time.perf_counter() - started) * 1000,
    }


def _predict_scores(model: Any, pairs: list[list[str]], *, batch_size: int) -> np.ndarray:
    if hasattr(model, "predict"):
        raw = model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
    elif hasattr(model, "compute_score"):
        raw = model.compute_score(pairs, batch_size=batch_size, normalize=True)
    else:
        raise TypeError("Reranker 모델에 predict 또는 compute_score 메서드가 없습니다.")
    scores = np.atleast_1d(np.asarray(raw, dtype=np.float32)).reshape(-1)
    if len(scores) != len(pairs):
        raise RuntimeError(
            f"Reranker 점수 개수 불일치: pairs={len(pairs)}, scores={len(scores)}"
        )
    return scores


def rerank_candidates(
    question: str,
    candidates: Sequence[Mapping[str, Any]],
    *,
    chunks_by_id: Mapping[str, Mapping[str, Any]],
    model: Any,
    text_builder: Callable[[Mapping[str, Any]], str],
    candidate_depth: int = 20,
    final_top_k: int = 5,
    batch_size: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Hybrid 상위 후보를 CrossEncoder로 재정렬한다."""
    if candidate_depth < final_top_k or final_top_k < 1:
        raise ValueError("candidate_depth는 final_top_k 이상이어야 합니다.")
    started = time.perf_counter()
    prepared: list[dict[str, Any]] = []
    pairs: list[list[str]] = []
    seen: set[str] = set()
    for base_rank, item in enumerate(candidates[:candidate_depth], start=1):
        chunk_id = _clean_text(item.get("chunk_id"))
        if not chunk_id or chunk_id in seen:
            continue
        seen.add(chunk_id)
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            raise KeyError(f"Reranker 후보 청크가 corpus에 없습니다: {chunk_id}")
        passage = _clean_text(text_builder(chunk))
        if not passage:
            continue
        prepared.append({**dict(item), "pre_rerank_rank": base_rank})
        pairs.append([_clean_text(question), passage])

    if not prepared:
        raise RuntimeError("Reranker에 전달할 유효 후보가 없습니다.")
    scores = _predict_scores(model, pairs, batch_size=batch_size)
    scored = [
        {**row, "reranker_score": float(score)}
        for row, score in zip(prepared, scores)
    ]
    ordered = sorted(
        scored,
        key=lambda row: (
            -float(row["reranker_score"]),
            int(row["pre_rerank_rank"]),
            str(row["chunk_id"]),
        ),
    )
    final = [
        {**row, "rank": rank}
        for rank, row in enumerate(ordered[:final_top_k], start=1)
    ]
    return final, {
        "latency_ms": (time.perf_counter() - started) * 1000,
        "candidate_count": len(prepared),
        "returned_count": len(final),
        "batch_size": int(batch_size),
        "question": _clean_text(question),
    }
