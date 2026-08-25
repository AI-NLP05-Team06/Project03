"""KDIC EC2 production overlay generated from the approved 2026-08-21 notebook.

Source SHA-256: F9A908D62A43EA3A3566A5D8DF0E982F214373FFF96470A749DC1EFE79E25083
Policy: C_DEFAULT_DC2_COMPARE_ONLY_V1
This module is executed inside kdic_pipeline_engine globals.
"""
from __future__ import annotations

import kdic_lightweight_router_v1 as light_router

KDIC_PRODUCTION_OVERLAY_SOURCE_SHA256 = "F9A908D62A43EA3A3566A5D8DF0E982F214373FFF96470A749DC1EFE79E25083"
KDIC_PRODUCTION_OVERLAY_POLICY = "C_DEFAULT_DC2_COMPARE_ONLY_V1"
KDIC_PRODUCTION_OVERLAY_REVISION = "2026-08-25-ui-number-and-contact-guard-v5"


# ==== overlay: how-word-classification ====

HOW_COMPARE_PATTERN_V1 = re.compile(r"어떻게\s*(?:다르|다른|달라|구분|비교)|어떤\s*차이", re.I)

HOW_PROCEDURE_PATTERN_V1 = re.compile(
    r"어떻게\s*(?:신청|접수|청구|조회|확인|신고|받|진행|해야|하나)|"
    r"(?:신청|접수|청구|조회|확인|신고|받|진행)하려면\s*어떻게|방법|절차",
    re.I,
)

def classify_how_usage_v1(question: str) -> str:
    text = str(question or "")
    if HOW_COMPARE_PATTERN_V1.search(text):
        return "COMPARE_HOW"
    if HOW_PROCEDURE_PATTERN_V1.search(text):
        return "PROCEDURE_HOW"
    return "GENERIC_HOW" if "어떻게" in text else "NONE"

# ==== overlay: answer-bd-core ====


import json
import random
import re
import time
from collections import OrderedDict
from typing import Any, Mapping, Sequence


ANSWER_EVIDENCE_TOTAL_MAX_CHARS = 14_000
ANSWER_EVIDENCE_RANK_BUDGETS = (4_000, 3_500, 3_000, 2_000, 1_500)
ANSWER_CACHE_ENABLED_FOR_COMPARISON = False
ANSWER_PROMPT_VERSION = "bd-low-latency-v1"


def _truncate_at_boundary_v1(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    limited = text[:limit]
    boundary = max(limited.rfind("\n"), limited.rfind(" "))
    if boundary >= int(limit * 0.75):
        limited = limited[:boundary]
    return limited.rstrip()


def _proximity_order_v1(context_ids: Sequence[str], matched_ids: Sequence[str]) -> list[str]:
    context = list(dict.fromkeys(str(value) for value in context_ids if str(value)))
    matched = list(dict.fromkeys(str(value) for value in matched_ids if str(value)))
    output = [value for value in matched if value in context]
    for matched_id in matched:
        if matched_id not in context:
            output.append(matched_id)
            continue
        center = context.index(matched_id)
        for distance in range(1, len(context) + 1):
            for index in (center - distance, center + distance):
                if 0 <= index < len(context) and context[index] not in output:
                    output.append(context[index])
    output.extend(value for value in context if value not in output)
    return output


def build_compact_parent_evidence_pack_v1(
    question: str,
    search_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_parent: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for result in search_results:
        child = dict(result.get("chunk") or {})
        child_id = str(result.get("chunk_id") or child.get("chunk_id") or "")
        parent_id = str(result.get("parent_doc_id") or _parent_id_for_chunk(child))
        row = by_parent.setdefault(parent_id, {
            "rank": int(result.get("rank") or len(by_parent) + 1),
            "parent_id": parent_id,
            "representative_chunk_id": child_id,
            "matched_child_ids": [],
            "matched_child_ranks": [],
            "context_chunk_ids": list(result.get("parent_context_chunk_ids") or [child_id]),
            "document_title": _clean_text(child.get("title") or child.get("document_title")),
            "source_url": _clean_text(child.get("source_url")),
        })
        row["matched_child_ids"].append(child_id)
        row["matched_child_ranks"].append(int(result.get("rank") or 0))

    evidence: list[dict[str, Any]] = []
    sources: OrderedDict[str, dict[str, Any]] = OrderedDict()
    total_remaining = ANSWER_EVIDENCE_TOTAL_MAX_CHARS
    for parent_index, row in enumerate(by_parent.values()):
        if total_remaining <= 0 or parent_index >= len(ANSWER_EVIDENCE_RANK_BUDGETS):
            break
        parent_budget = min(ANSWER_EVIDENCE_RANK_BUDGETS[parent_index], total_remaining)
        ordered_ids = _proximity_order_v1(
            row["context_chunk_ids"], row["matched_child_ids"]
        )
        parts: list[str] = []
        included_ids: list[str] = []
        section_titles: list[str] = []
        remaining = parent_budget
        for chunk_id in ordered_ids:
            chunk = CHUNKS_BY_ID.get(str(chunk_id))
            if chunk is None:
                continue
            title = _clean_text(chunk.get("title"))
            section = _clean_text(chunk.get("section_title"))
            if section and section not in section_titles:
                section_titles.append(section)
            label = " / ".join(value for value in (title, section) if value)
            part = f"[{chunk_id}] {label}\n{_clean_text(chunk.get('content'))}".strip()
            separator_cost = 2 if parts else 0
            if remaining <= separator_cost:
                break
            part = _truncate_at_boundary_v1(part, remaining - separator_cost)
            if not part:
                break
            parts.append(part)
            included_ids.append(str(chunk_id))
            remaining -= len(part) + separator_cost
            if remaining < 120:
                break

        content = "\n\n".join(parts)
        if not content:
            continue
        evidence_id = f"E{len(evidence) + 1}"
        evidence.append({
            "evidence_id": evidence_id,
            "rank": int(row["rank"]),
            "chunk_id": row["representative_chunk_id"],
            "parent_id": row["parent_id"],
            "context_chunk_ids": included_ids,
            "matched_child_ids": list(dict.fromkeys(row["matched_child_ids"])),
            "matched_child_ranks": sorted(set(row["matched_child_ranks"])),
            "document_title": row["document_title"],
            "section_title": " · ".join(section_titles),
            "content": content,
            "context_char_count": len(content),
            "context_truncated": len(included_ids) < len(ordered_ids),
            "source_url": row["source_url"],
        })
        total_remaining -= len(content)
        url = row["source_url"]
        if url:
            source = sources.setdefault(url, {
                "source_id": f"S{len(sources) + 1}",
                "title": row["document_title"] or "공식 출처",
                "source_url": url,
                "evidence_ids": [],
            })
            source["evidence_ids"].append(evidence_id)

    if not evidence:
        raise ValueError("저지연 Evidence Pack을 만들 근거가 없습니다.")
    return {
        "question": _clean_text(question),
        "search_parent_context_max_chars": PARENT_CONTEXT_MAX_CHARS,
        "answer_evidence_total_max_chars": ANSWER_EVIDENCE_TOTAL_MAX_CHARS,
        "answer_prompt_version": ANSWER_PROMPT_VERSION,
        "evidence": evidence,
        "sources": list(sources.values()),
    }


RELATION_QUESTION_PATTERN_V1 = re.compile(
    r"(?:같이|함께|동시에|동시\s*신청|한\s*번에|병행|연계|둘\s*다|두\s*(?:제도|업무)\s*모두)", re.I
)
RELATION_POSITIVE_ANSWER_PATTERN_V1 = re.compile(
    r"(?:(?:같이|함께|동시에|한\s*번에|병행).{0,18}(?:신청|이용|처리).{0,12}(?:가능|불가능|할\s*수\s*(?:있|없))|(?:따로|각각).{0,18}(?:신청|진행).{0,12}(?:해야|하여야)|동일한?\s*지급대행점.{0,20}동시)", re.I
)
RELATION_EVIDENCE_PATTERN_V1 = re.compile(
    r"(?:동시\s*신청|한\s*번에\s*신청|함께\s*신청|같이\s*신청|병행\s*(?:신청|이용)|중복\s*신청)", re.I
)


def relation_constraint_v1(question: str, pack: Mapping[str, Any]) -> dict[str, Any]:
    relation_question = bool(RELATION_QUESTION_PATTERN_V1.search(question))
    direct_rows = [
        str(row.get("evidence_id"))
        for row in pack.get("evidence") or []
        if RELATION_EVIDENCE_PATTERN_V1.search(str(row.get("content") or ""))
    ]
    return {
        "relation_question": relation_question,
        "direct_relation_evidence_ids": direct_rows,
        "may_affirm_joint_application": bool(direct_rows),
        "rule": (
            "두 제도의 동시·병행 가능성을 직접 명시한 동일 Evidence가 없으면 "
            "가능하다고 단정하지 않고 확인되지 않는다고 답한다."
        ),
    }


B_LOW_LATENCY_SYSTEM_PROMPT = """
당신은 예금보험공사 공식 문서 기반 답변 시스템입니다.

1. 사용자 질문과 제공된 Evidence Pack의 사실만 사용하세요.
2. 질문에 직접 답하고 필요한 대상·조건·예외·금액·기간·절차를 설명하세요.
3. Evidence에 없는 사실이나 서로 다른 제도의 조건을 임의로 결합하지 마세요.
4. 동시·병행 신청 질문은 같은 Evidence가 그 관계를 직접 명시할 때만 가능하다고 답하세요.
5. 별도 문서가 각 제도의 자격을 각각 설명한다는 사실만으로 동시 신청 가능성을 추론하지 마세요.
6. 직접 관계 근거가 없으면 확인되지 않는다고 답하고 coverage_status를 PARTIAL로 두세요.
7. 문장 수는 제한하지 않되 질문하지 않은 배경 설명과 중복은 넣지 마세요.
8. 근거 문장 끝에 [E1] 형식으로 실제 Evidence ID를 표시하세요.
9. 지정된 JSON 객체 하나만 출력하고 문자열 내부 줄바꿈은 \\n으로 이스케이프하세요.
""".strip()


D_SKELETON_SYSTEM_PROMPT = """
당신은 예금보험공사 공식 문서에서 답변에 필요한 사실 구조만 추출하는 분석기입니다.

1. 사용자 질문과 동일 Evidence Pack에 명시된 사실만 사용하세요.
2. 최종 사용자 문장이 아니라 Answer Skeleton JSON만 작성하세요.
3. 각 answer_item에는 질문이 요구한 항목 하나와 실제 evidence_ids를 연결하세요.
4. 서로 다른 제도의 조건을 임의로 결합하지 마세요.
5. 동시·병행 신청 관계는 같은 Evidence가 그 관계를 직접 명시할 때만 claim으로 채택하세요.
6. 직접 관계 근거가 없으면 uncertainties에 기록하고 가능하다고 추론하지 마세요.
7. 문서 충돌은 conflicts, 확인 불가는 uncertainties에 기록하세요.
8. JSON 객체 하나만 출력하고 문자열 내부 줄바꿈은 \\n으로 이스케이프하세요.
""".strip()


D_FINAL_SYSTEM_PROMPT = """
당신은 예금보험공사 공식 문서 기반 질의응답 시스템입니다.

1. 제공된 Answer Skeleton과 동일 Evidence Pack의 사실만 사용하세요.
2. 질문에 직접 답하고 Skeleton의 항목에 필요한 조건·예외·금액·기간·절차를 설명하세요.
3. Skeleton 또는 Evidence에 없는 사실을 추정하지 마세요.
4. 동시·병행 가능성에 직접 근거가 없으면 가능하다고 단정하지 마세요.
5. 문장 수는 제한하지 않되 질문하지 않은 배경 설명과 중복은 넣지 마세요.
6. 각 주장에는 Skeleton이 허용한 [E1] 형식의 Evidence ID만 표시하세요.
7. JSON, Skeleton, 내부 구현, 검색 점수는 답변에 언급하지 마세요.
""".strip()


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _usage_dict(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    return {
        key: int(getattr(usage, key, 0) or 0)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def _call_answer_api_v1(
    *, system_prompt: str, user_prompt: str, max_tokens: int
) -> tuple[str, dict[str, int], float, dict[str, Any]]:
    global _LAST_ANSWER_API_TRACE
    started = time.perf_counter()
    response = ANSWER_HCX_CLIENT.chat.completions.create(
        model=HCX_CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    wall_ms = (time.perf_counter() - started) * 1000
    content = response.choices[0].message.content
    if not content or not str(content).strip():
        raise RuntimeError("HCX 답변 출력이 비어 있습니다.")
    return str(content), _usage_dict(response), wall_ms, dict(_LAST_ANSWER_API_TRACE)


def _merge_usage_v1(*values: Mapping[str, Any]) -> dict[str, int]:
    return {
        key: sum(int(value.get(key) or 0) for value in values)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def _relation_safe_answer_v1(
    answer: str,
    constraint: Mapping[str, Any],
) -> tuple[str, bool]:
    if (
        constraint.get("relation_question")
        and not constraint.get("may_affirm_joint_application")
        and RELATION_POSITIVE_ANSWER_PATTERN_V1.search(answer)
    ):
        return (
            "현재 검색된 공식 문서 근거만으로 두 제도를 동시에 또는 병행하여 "
            "신청할 수 있는지는 확인되지 않습니다. 각 제도의 개별 신청 요건은 "
            "확인할 수 있지만, 그것만으로 동시 신청 가능성을 단정할 수는 없습니다.",
            True,
        )
    return answer, False


def generate_answer_b_low_latency_v1(
    question: str,
    pack: Mapping[str, Any],
) -> dict[str, Any]:
    constraint = relation_constraint_v1(question, pack)
    prompt = f"""[사용자 질문]\n{_clean_text(question)}\n\n[관계 주장 안전조건]\n{_compact_json(constraint)}\n\n[Basic Evidence Pack]\n{_compact_json(pack)}\n\n[출력 JSON]\n{{\"answer\":\"근거 문장에 [E1] 표시\",\"used_evidence_ids\":[\"E1\"],\"coverage_status\":\"SUFFICIENT|PARTIAL|INSUFFICIENT\",\"missing_information\":[]}}"""
    raw, usage, first_ms, first_trace = _call_answer_api_v1(
        system_prompt=B_LOW_LATENCY_SYSTEM_PROMPT,
        user_prompt=prompt,
        max_tokens=1600,
    )
    attempts = [{"stage": "initial", "latency_ms": first_ms, "trace": first_trace}]
    total_usage = dict(usage)
    total_ms = first_ms
    try:
        raw_payload = answer_b_core._extract_json_object(raw)
        requested_ids = answer_b_core._clean_list(raw_payload.get("used_evidence_ids"))
        allowed = answer_b_core._allowed_evidence(pack)
        valid_ids = [value for value in requested_ids if value in allowed]
        answer = answer_b_core._strip_model_urls(answer_b_core._clean(raw_payload.get("answer")))
        if not answer:
            raise ValueError("answer가 비어 있습니다.")
        if not valid_ids:
            valid_ids = [f"E{n}" for n in re.findall(r"\[E(\d+)\]", answer) if f"E{n}" in allowed]
        if not valid_ids:
            raise ValueError("유효 Evidence ID가 없습니다.")
        payload = {
            "answer": answer,
            "used_evidence_ids": list(dict.fromkeys(valid_ids)),
            "used_chunk_ids": [allowed[value] for value in dict.fromkeys(valid_ids)],
            "coverage_status": str(raw_payload.get("coverage_status") or "PARTIAL").upper(),
            "missing_information": answer_b_core._clean_list(raw_payload.get("missing_information")),
        }
        payload = answer_b_core.validate_basic_answer(payload, pack)
    except (ValueError, TypeError):
        try:
            payload = answer_b_core._recover_basic_answer_from_raw(raw, pack)
            attempts.append({"stage": "local_raw_recovery", "latency_ms": 0.0})
        except (ValueError, TypeError):
            repair_prompt = f"""다음 출력을 사실 변경 없이 올바른 JSON 객체로만 고치세요. Evidence Pack 밖의 ID를 만들지 마세요.\n\n[원래 요청]\n{prompt}\n\n[교정 대상]\n{raw[:6000]}"""
            repaired, repair_usage, repair_ms, repair_trace = _call_answer_api_v1(
                system_prompt=B_LOW_LATENCY_SYSTEM_PROMPT,
                user_prompt=repair_prompt,
                max_tokens=1600,
            )
            total_usage = _merge_usage_v1(total_usage, repair_usage)
            total_ms += repair_ms
            attempts.append({"stage": "repair", "latency_ms": repair_ms, "trace": repair_trace})
            parsed = answer_b_core._extract_json_object(repaired)
            allowed = answer_b_core._allowed_evidence(pack)
            ids = [value for value in answer_b_core._clean_list(parsed.get("used_evidence_ids")) if value in allowed]
            payload = answer_b_core.validate_basic_answer({
                "answer": parsed.get("answer"),
                "used_evidence_ids": ids,
                "used_chunk_ids": [allowed[value] for value in ids],
                "coverage_status": parsed.get("coverage_status") or "PARTIAL",
                "missing_information": parsed.get("missing_information") or [],
            }, pack)

    safe_answer, guard_applied = _relation_safe_answer_v1(payload["answer"], constraint)
    payload["answer"] = safe_answer
    if guard_applied:
        payload["coverage_status"] = "PARTIAL"
        payload["missing_information"] = list(dict.fromkeys(
            list(payload.get("missing_information") or [])
            + ["두 제도의 동시·병행 신청 가능 여부를 직접 명시한 공식 근거"]
        ))
    return {
        **payload,
        "system": "B",
        "latency_ms": total_ms,
        "usage": total_usage,
        "api_calls": sum(1 for row in attempts if row["stage"] in {"initial", "repair"}),
        "attempts": attempts,
        "relation_constraint": constraint,
        "relation_guard_applied": guard_applied,
    }


def _validate_d_skeleton_v1(
    raw: Mapping[str, Any], pack: Mapping[str, Any]
) -> dict[str, Any]:
    allowed = answer_b_core._allowed_evidence(pack)
    items: list[dict[str, Any]] = []
    for index, item in enumerate(raw.get("answer_items") or [], start=1):
        if not isinstance(item, Mapping):
            continue
        claim = answer_b_core._clean(item.get("claim"))
        ids = [value for value in answer_b_core._clean_list(item.get("evidence_ids")) if value in allowed]
        if claim and ids:
            items.append({
                "item_id": f"A{len(items) + 1}",
                "topic": answer_b_core._clean(item.get("topic")) or f"답변 항목 {index}",
                "claim": claim,
                "conditions": answer_b_core._clean_list(item.get("conditions")),
                "details": answer_b_core._clean_list(item.get("details")),
                "evidence_ids": list(dict.fromkeys(ids)),
            })
    core = answer_b_core._clean(raw.get("core_answer"))
    if not core or not items:
        raise ValueError("D안 Skeleton의 핵심 답변 또는 유효 항목이 없습니다.")
    coverage = answer_b_core._clean(raw.get("coverage_status")).upper()
    if coverage not in answer_b_core.ALLOWED_COVERAGE_STATUS:
        coverage = "PARTIAL"
    return {
        "core_answer": core,
        "answer_items": items,
        "uncertainties": answer_b_core._clean_list(raw.get("uncertainties")),
        "conflicts": answer_b_core._clean_list(raw.get("conflicts")),
        "coverage_status": coverage,
    }


def generate_answer_d_v1(
    question: str,
    pack: Mapping[str, Any],
) -> dict[str, Any]:
    constraint = relation_constraint_v1(question, pack)
    skeleton_prompt = f"""[사용자 질문]\n{_clean_text(question)}\n\n[관계 주장 안전조건]\n{_compact_json(constraint)}\n\n[Evidence Pack]\n{_compact_json(pack)}\n\n[출력 JSON]\n{{\"core_answer\":\"핵심 결론\",\"answer_items\":[{{\"topic\":\"항목\",\"claim\":\"근거 사실\",\"conditions\":[],\"details\":[],\"evidence_ids\":[\"E1\"]}}],\"uncertainties\":[],\"conflicts\":[],\"coverage_status\":\"SUFFICIENT|PARTIAL|INSUFFICIENT\"}}"""
    raw_skeleton, usage1, skeleton_ms, trace1 = _call_answer_api_v1(
        system_prompt=D_SKELETON_SYSTEM_PROMPT,
        user_prompt=skeleton_prompt,
        max_tokens=2000,
    )
    skeleton = _validate_d_skeleton_v1(
        answer_b_core._extract_json_object(raw_skeleton), pack
    )
    final_prompt = f"""[사용자 질문]\n{_clean_text(question)}\n\n[관계 주장 안전조건]\n{_compact_json(constraint)}\n\n[Answer Skeleton]\n{_compact_json(skeleton)}\n\n[동일 Evidence Pack]\n{_compact_json(pack)}\n\n위 근거 범위에서 사용자용 최종 답변을 작성하세요."""
    raw_answer, usage2, final_ms, trace2 = _call_answer_api_v1(
        system_prompt=D_FINAL_SYSTEM_PROMPT,
        user_prompt=final_prompt,
        max_tokens=1600,
    )
    answer = answer_b_core._strip_model_urls(str(raw_answer).strip())
    if not answer:
        raise ValueError("D안 최종 답변이 비어 있습니다.")
    safe_answer, guard_applied = _relation_safe_answer_v1(answer, constraint)
    return {
        "system": "D",
        "answer": safe_answer,
        "skeleton": skeleton,
        "coverage_status": "PARTIAL" if guard_applied else skeleton["coverage_status"],
        "latency_ms": skeleton_ms + final_ms,
        "skeleton_latency_ms": skeleton_ms,
        "final_latency_ms": final_ms,
        "usage": _merge_usage_v1(usage1, usage2),
        "api_calls": 2,
        "attempts": [
            {"stage": "skeleton", "latency_ms": skeleton_ms, "trace": trace1},
            {"stage": "final", "latency_ms": final_ms, "trace": trace2},
        ],
        "relation_constraint": constraint,
        "relation_guard_applied": guard_applied,
    }

# ==== overlay: answer-bd-compare ====


import copy
import html
import time
from typing import Any, Mapping, Sequence


def new_bd_comparison_state() -> dict[str, Any]:
    return new_context_state()


def _current_question_scoped_analysis_v1(
    question: str,
    analysis: Mapping[str, Any],
) -> dict[str, Any]:
    """Prevent explicit current-question businesses from inheriting old scope."""

    output = dict(analysis)
    if str(output.get("route") or "").upper() != "RETRIEVE":
        return output
    try:
        current_businesses = order_businesses_by_question_p0_v1(
            question,
            list(light_router.find_businesses(question) or []),
        )
    except Exception:
        current_businesses = []
    if not 1 <= len(current_businesses) <= NEED_BATCH_MAX_BUSINESSES_V5:
        return output

    if len(current_businesses) == 1:
        subqueries: list[str] = []
        plans = [{"query": question, "weight": 1.0, "source": "ORIGINAL"}]
    else:
        subqueries = build_p0_cross_business_subqueries_v1(question, current_businesses)
        if len(subqueries) != len(current_businesses):
            return output
        sub_weight = V15_SUBQUERY_TOTAL_WEIGHT / len(subqueries)
        plans = [{
            "query": question,
            "weight": V15_ORIGINAL_WEIGHT,
            "source": "ORIGINAL_ANCHOR",
        }]
        plans.extend({
            "query": subquery,
            "weight": sub_weight,
            "source": "P0_RULE_DECOMPOSED",
        } for subquery in subqueries)

    output.update({
        "resolved_question": question,
        "normalized_question": question,
        "businesses": current_businesses,
        "question_businesses": current_businesses,
        "cross_business_candidate": len(current_businesses) >= 2,
        "p0_cross_preserved": len(current_businesses) >= 2,
        "decomposition_source": (
            "P0_RULE_CROSS_PRESERVED" if len(current_businesses) >= 2 else "NOT_REQUIRED"
        ),
        "decomposition_accepted": bool(subqueries),
        "subqueries": subqueries,
        "plans": plans,
        "query_plan_valid": True,
        "context_used": False,
        "context_override_reason": "EXPLICIT_CURRENT_BUSINESS_SCOPE",
    })
    return output


def _cross_business_scope_count_v5(
    analysis: Mapping[str, Any],
    plans: Sequence[Mapping[str, Any]],
    question: str = "",
) -> tuple[int, list[str], int]:
    # analysis.businesses may include active businesses inherited from earlier
    # turns. Capacity is a property of the current question, so explicit current
    # labels and the current decomposition plans take precedence.
    try:
        question_businesses = [
            _clean_text(value)
            for value in light_router.find_businesses(str(question or "")) or []
            if _clean_text(value)
        ]
    except Exception:
        question_businesses = []
    plan_businesses = []
    for plan in plans:
        if "DECOMPOSED" not in str(plan.get("source") or "").upper():
            continue
        try:
            matched = list(light_router.find_businesses(str(plan.get("query") or "")) or [])
        except Exception:
            matched = []
        if len(matched) == 1 and _clean_text(matched[0]):
            plan_businesses.append(_clean_text(matched[0]))
    decomposed_count = sum(
        1 for plan in plans if "DECOMPOSED" in str(plan.get("source") or "").upper()
    )
    current_businesses = list(dict.fromkeys(question_businesses + plan_businesses))
    if current_businesses or decomposed_count:
        businesses = current_businesses
        scope_count = max(len(current_businesses), decomposed_count)
    else:
        # Compatibility fallback for analyzers that do not expose a current
        # question and do not emit decomposed plans.
        businesses = list(dict.fromkeys(
            _clean_text(value)
            for value in analysis.get("businesses") or []
            if _clean_text(value)
        ))
        scope_count = len(businesses)
    return scope_count, businesses, decomposed_count


def prepare_common_retrieval_v1(
    question: str,
    *,
    state: dict[str, Any],
) -> dict[str, Any]:
    global _LAST_RERANK_TRACE, _LAST_PARENT_CHILD_TRACE
    total_started = time.perf_counter()
    _LAST_RERANK_TRACE = {}
    _LAST_PARENT_CHILD_TRACE = {}

    analysis = _current_question_scoped_analysis_v1(
        question,
        ANALYZER_FUNCTIONS["v15"](question, state),
    )
    analysis_ms = (time.perf_counter() - total_started) * 1000
    if analysis["route"] != "RETRIEVE":
        return {
            "question": question,
            "resolved_question": analysis.get("resolved_question") or "",
            "route": analysis["route"],
            "analysis": analysis,
            "route_message": _route_message(analysis),
            "latency_ms": {
                "질의분석": analysis_ms,
                "공통 준비 전체": (time.perf_counter() - total_started) * 1000,
            },
        }

    plans = list(analysis.get("plans") or [])
    requested_scope_count, detected_businesses, decomposed_plan_count = _cross_business_scope_count_v5(
        analysis,
        plans,
        question,
    )
    if requested_scope_count > 3:
        return {
            "question": question,
            "resolved_question": analysis.get("resolved_question") or question,
            "route": "SIMPLIFY_QUERY",
            "analysis": analysis,
            "plans": plans,
            "route_message": (
                "한 번에 최대 3개 업무까지 답변할 수 있습니다. 정확한 업무별 근거를 제공하기 위해 "
                "질문을 간소화하여 최대 3개 업무로 다시 질문해 주세요. "
                "나머지 업무는 다음 질문으로 나누어 주시면 이어서 안내해 드리겠습니다."
            ),
            "capacity_audit": {
                "passed": False,
                "requested_scope_count": requested_scope_count,
                "maximum_cross_business_count": 3,
                "detected_businesses": detected_businesses,
                "decomposed_plan_count": decomposed_plan_count,
            },
            "latency_ms": {
                "질의분석": analysis_ms,
                "공통 준비 전체": (time.perf_counter() - total_started) * 1000,
            },
        }
    if not analysis.get("query_plan_valid") or not _comparison_plans_valid("RETRIEVE", plans):
        raise RuntimeError("RETRIEVE 검색 계획이 유효하지 않습니다.")

    search_started = time.perf_counter()
    search_results, per_query = fuse_query_results(plans)
    child_search_ms = (time.perf_counter() - search_started) * 1000
    reranker_trace = dict(_LAST_RERANK_TRACE)

    parent_started = time.perf_counter()
    search_results = expand_parent_context(search_results)
    parent_ms = (time.perf_counter() - parent_started) * 1000
    parent_trace = dict(_LAST_PARENT_CHILD_TRACE)

    answer_question = str(analysis.get("resolved_question") or question)
    pack_started = time.perf_counter()
    pack = build_compact_parent_evidence_pack_v1(answer_question, search_results)
    pack_ms = (time.perf_counter() - pack_started) * 1000
    evidence_gate = (
        audit_need_evidence_pack_v5(pack)
        if str(pack.get("retrieval_strategy") or "") == "NEED_BATCH_RERANK_V5"
        else {"passed": True, "needs": []}
    )
    per_query_latency = _query_latency_rows(per_query)
    common_ms = (time.perf_counter() - total_started) * 1000
    evidence_gate_message = format_need_score_gate_message_v5(evidence_gate)
    return {
        "question": question,
        "resolved_question": answer_question,
        "route": analysis["route"] if evidence_gate.get("passed") else "EVIDENCE_INSUFFICIENT",
        "route_message": None if evidence_gate.get("passed") else evidence_gate_message,
        "analysis": analysis,
        "plans": plans,
        "search_results": search_results,
        "per_query": per_query,
        "per_query_latency": per_query_latency,
        "reranker": reranker_trace,
        "parent_child": parent_trace,
        "evidence_pack": pack,
        "evidence_pack_sha256": evidence_pack_sha256(pack),
        "evidence_quota_audit": evidence_gate,
        "evidence_chars": sum(int(row.get("context_char_count") or 0) for row in pack["evidence"]),
        "latency_ms": {
            "문맥 처리": float(analysis.get("context_latency_ms") or 0),
            "질의분석": analysis_ms,
            "검색": child_search_ms + parent_ms,
            "질문 임베딩": sum(row["embedding_latency_ms"] for row in per_query_latency),
            "Dense 계산": sum(row["dense_compute_latency_ms"] for row in per_query_latency),
            "BM25": sum(row["bm25_latency_ms"] for row in per_query_latency),
            "BAAI Reranker": float(reranker_trace.get("latency_ms") or 0),
            "Parent-Child8192": parent_ms,
            "저지연 Evidence Pack": pack_ms,
            "공통 준비 전체": common_ms,
        },
    }


def _trace_totals_v1(payload: Mapping[str, Any]) -> dict[str, float | int]:
    waits = 0.0
    rate_limits = 0
    for stage in payload.get("attempts") or []:
        trace = stage.get("trace") or {}
        waits += float(trace.get("total_wait_ms") or 0)
        rate_limits += sum(
            1 for attempt in trace.get("attempts") or []
            if attempt.get("status") == "RATE_LIMIT_429"
        )
    return {"wait_ms": waits, "rate_limit_429_count": rate_limits}


_BD_ORDER_COUNTER = 0


def compare_answer_systems_v1(
    question: str,
    *,
    state: dict[str, Any],
    order: str = "ALTERNATE",
) -> dict[str, Any]:
    global _BD_ORDER_COUNTER
    common = prepare_common_retrieval_v1(question, state=state)
    if common["route"] != "RETRIEVE":
        return {"common": common, "answers": {}, "summary": pd.DataFrame()}

    normalized_order = str(order or "ALTERNATE").upper()
    if normalized_order == "ALTERNATE":
        normalized_order = "B_FIRST" if _BD_ORDER_COUNTER % 2 == 0 else "D_FIRST"
        _BD_ORDER_COUNTER += 1
    sequence = ("B", "D") if normalized_order == "B_FIRST" else ("D", "B")
    answers: dict[str, dict[str, Any]] = {}
    for system in sequence:
        if system == "B":
            answers[system] = generate_answer_b_low_latency_v1(
                common["resolved_question"], common["evidence_pack"]
            )
        else:
            answers[system] = generate_answer_d_v1(
                common["resolved_question"], common["evidence_pack"]
            )

    common_ms = float(common["latency_ms"]["공통 준비 전체"])
    rows = []
    for system in ("B", "D"):
        payload = answers[system]
        trace = _trace_totals_v1(payload)
        usage = payload.get("usage") or {}
        rows.append({
            "답변안": system,
            "실행순서": sequence.index(system) + 1,
            "Evidence문자": common["evidence_chars"],
            "API호출": int(payload.get("api_calls") or 0),
            "429횟수": int(trace["rate_limit_429_count"]),
            "호출간격·429대기(ms)": float(trace["wait_ms"]),
            "입력토큰": int(usage.get("prompt_tokens") or 0),
            "출력토큰": int(usage.get("completion_tokens") or 0),
            "답변지연(ms)": float(payload.get("latency_ms") or 0),
            "가상E2E(ms)": common_ms + float(payload.get("latency_ms") or 0),
            "답변글자": len(str(payload.get("answer") or "")),
            "Coverage": payload.get("coverage_status"),
            "관계안전가드": bool(payload.get("relation_guard_applied")),
        })
    summary = pd.DataFrame(rows)
    state.setdefault("turns", []).extend([
        {"role": "user", "content": question},
        {"role": "assistant", "content": user_visible_answer(answers["B"]["answer"])},
    ])
    return {
        "common": common,
        "answers": answers,
        "execution_order": list(sequence),
        "summary": summary,
    }


def _sources_from_pack_v1(pack: Mapping[str, Any]) -> str:
    lines = ["### 공통 공식 출처", ""]
    for source in pack.get("sources") or []:
        title = str(source.get("title") or "공식 출처")
        url = str(source.get("source_url") or "")
        if url:
            lines.append(f"- [{title}]({url})")
    return "\n".join(lines) if len(lines) > 2 else ""


def render_bd_comparison_v1(result: Mapping[str, Any], *, show_pack: bool = False) -> None:
    common = result["common"]
    if common["route"] != "RETRIEVE":
        display(Markdown(f"### {common['route']}\n\n{common.get('route_message', '')}"))
        return
    answers = result["answers"]
    display(Markdown("## 답변 B안\n\n" + user_visible_answer(answers["B"]["answer"])))
    display(Markdown("## 답변 D안\n\n" + user_visible_answer(answers["D"]["answer"])))
    source_md = _sources_from_pack_v1(common["evidence_pack"])
    if source_md:
        display(Markdown(source_md))
    display(Markdown("### B/D 레이턴시·토큰 비교"))
    display(result["summary"])
    display(Markdown(analysis_markdown_compare({
        "analyzer_label": "V1.5 관계형 멀티턴 개선",
        "analysis": common["analysis"],
    })))
    display(Markdown(retrieval_markdown_compare({"search_results": common["search_results"]})))
    display(Markdown(latency_markdown_compare({"latency_ms": common["latency_ms"]})))
    if show_pack:
        display(JSON(common["evidence_pack"], expanded=False))

# ==== overlay: need-batch-rerank-v5 ====

# V5: 교차업무 Need-aware Batch Reranking
from collections import OrderedDict, defaultdict
from typing import Any, Mapping, Sequence

NEED_BATCH_MAX_EVIDENCE_V5 = 9
NEED_BATCH_REQUIRED_TOP_K_V5 = 3
NEED_BATCH_SCORE_GATE_ENABLED_V5 = bool(globals().get("NEED_BATCH_SCORE_GATE_ENABLED_V5", False))
NEED_BATCH_MIN_RERANKER_SCORE_V5 = float(globals().get("NEED_BATCH_MIN_RERANKER_SCORE_V5", 0.25))
NEED_BATCH_MAX_BUSINESSES_V5 = 3
NEED_BATCH_MIN_DISTINCT_PARENTS_V5 = 2
NEED_BATCH_MAX_EVIDENCE_PER_PARENT_V5 = 2
NEED_BATCH_GLOBAL_RERANK_WEIGHT_V5 = 0.70
NEED_BATCH_GLOBAL_RRF_WEIGHT_V5 = 0.30
ANSWER_EVIDENCE_RANK_BUDGETS_V5 = (2_400, 2_200, 2_000, 1_800, 1_600, 1_400, 1_000, 800, 800)
ANSWER_PROMPT_VERSION = (
    f"need-batch-rerank-v5-top3-gate{int(NEED_BATCH_SCORE_GATE_ENABLED_V5)}"
    f"-score{NEED_BATCH_MIN_RERANKER_SCORE_V5:.6f}"
)

_FUSE_QUERY_RESULTS_V4_1 = globals().get("_FUSE_QUERY_RESULTS_V4_1") or fuse_query_results
_LAST_NEED_BATCH_CONTEXT_V5: dict[str, Any] = {}


def need_score_gate_passed_v5(
    score: float,
    *,
    enabled: bool | None = None,
    threshold: float | None = None,
) -> bool:
    gate_enabled = NEED_BATCH_SCORE_GATE_ENABLED_V5 if enabled is None else bool(enabled)
    minimum = NEED_BATCH_MIN_RERANKER_SCORE_V5 if threshold is None else float(threshold)
    return bool(not gate_enabled or float(score) >= minimum)


def need_score_gate_metadata_v5() -> dict[str, Any]:
    return {
        "score_gate_enabled": bool(NEED_BATCH_SCORE_GATE_ENABLED_V5),
        "configured_minimum_need_reranker_score": float(NEED_BATCH_MIN_RERANKER_SCORE_V5),
        "effective_minimum_need_reranker_score": (
            float(NEED_BATCH_MIN_RERANKER_SCORE_V5)
            if NEED_BATCH_SCORE_GATE_ENABLED_V5
            else None
        ),
        "minimum_need_reranker_score": (
            float(NEED_BATCH_MIN_RERANKER_SCORE_V5)
            if NEED_BATCH_SCORE_GATE_ENABLED_V5
            else None
        ),
    }


def _source_is_decomposed_v5(plan: Mapping[str, Any]) -> bool:
    return "DECOMPOSED" in str(plan.get("source") or "").upper()


def _source_is_original_v5(plan: Mapping[str, Any]) -> bool:
    return "ORIGINAL" in str(plan.get("source") or "").upper()


def _need_business_v5(question: str) -> str:
    try:
        values = list(light_router.find_businesses(question) or [])
    except Exception:
        values = []
    return str(values[0]) if len(values) == 1 else ""


def _row_parent_key_v5(row: Mapping[str, Any]) -> str:
    chunk = dict(row.get("chunk") or {})
    return str(row.get("parent_doc_id") or _parent_id_for_chunk(chunk) or row.get("chunk_id") or "")


def _row_url_key_v5(row: Mapping[str, Any]) -> str:
    chunk = dict(row.get("chunk") or {})
    return _clean_text(chunk.get("source_url"))


def _minmax_values_v5(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    numbers = np.asarray(values, dtype=np.float64)
    low, high = float(numbers.min()), float(numbers.max())
    if abs(high - low) <= 1e-12:
        return [1.0 for _ in values]
    return [float(value) for value in ((numbers - low) / (high - low))]


def _fused_candidates_v5(
    all_hits: Sequence[tuple[int, Mapping[str, Any], Sequence[Mapping[str, Any]]]],
    rrf_k: int,
) -> list[dict[str, Any]]:
    fused: dict[str, dict[str, Any]] = {}
    for plan_index, plan, hits in all_hits:
        for hit in hits:
            chunk_id = str(hit["chunk_id"])
            row = fused.setdefault(chunk_id, {
                "chunk_id": chunk_id,
                "query_fusion_score": 0.0,
                "best_minmax_score": 0.0,
                "dense_rank": None,
                "bm25_rank": None,
                "matched_queries": [],
                "chunk": hit["chunk"],
            })
            row["query_fusion_score"] += float(plan["weight"]) / (rrf_k + int(hit["rank"]))
            row["best_minmax_score"] = max(
                float(row["best_minmax_score"]), float(hit.get("minmax_score") or 0.0)
            )
            if row["dense_rank"] is None or (
                hit.get("dense_rank") is not None and int(hit["dense_rank"]) < int(row["dense_rank"])
            ):
                row["dense_rank"] = hit.get("dense_rank")
            if row["bm25_rank"] is None or (
                hit.get("bm25_rank") is not None and int(hit["bm25_rank"]) < int(row["bm25_rank"])
            ):
                row["bm25_rank"] = hit.get("bm25_rank")
            row["matched_queries"].append({
                "plan_index": plan_index,
                "query": str(plan.get("query") or ""),
                "source": str(plan.get("source") or ""),
                "rank": int(hit["rank"]),
                "weight": float(plan["weight"]),
            })
    return sorted(
        fused.values(),
        key=lambda row: (
            -float(row["query_fusion_score"]),
            -float(row["best_minmax_score"]),
            str(row["chunk_id"]),
        ),
    )


def _can_select_need_row_v5(
    row: Mapping[str, Any],
    *,
    selected_chunks: set[str],
    selected_parents: set[str],
    selected_urls: set[str],
) -> bool:
    chunk_id = str(row.get("chunk_id") or "")
    parent_id = _row_parent_key_v5(row)
    source_url = _row_url_key_v5(row)
    if not chunk_id or chunk_id in selected_chunks:
        return False
    if parent_id and parent_id in selected_parents:
        return False
    # 같은 공식 URL 안의 서로 다른 Parent는 별도 근거로 허용합니다.
    # 중복 방지는 chunk_id와 parent_id 기준으로만 수행합니다.
    return True


def _mark_selected_need_row_v5(
    row: Mapping[str, Any],
    *,
    selected_chunks: set[str],
    selected_parents: set[str],
    selected_urls: set[str],
) -> None:
    selected_chunks.add(str(row.get("chunk_id") or ""))
    parent_id = _row_parent_key_v5(row)
    source_url = _row_url_key_v5(row)
    if parent_id:
        selected_parents.add(parent_id)
    if source_url:
        selected_urls.add(source_url)


def fuse_query_results(
    plans: list[dict[str, Any]],
    *,
    top_k: int = FINAL_TOP_K,
    rrf_k: int = QUERY_FUSION_RRF_K,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """교차업무만 Need-aware batch rerank, 나머지는 검증된 V4.1 검색을 유지합니다."""
    global _LAST_RERANK_TRACE, _LAST_NEED_BATCH_CONTEXT_V5
    if not plans:
        raise ValueError("검색 계획이 없습니다.")
    if not math.isclose(sum(float(plan["weight"]) for plan in plans), 1.0, abs_tol=1e-9):
        raise ValueError("검색 계획 가중치 합은 1이어야 합니다.")

    decomposed_plans = [plan for plan in plans if _source_is_decomposed_v5(plan)]
    if len(decomposed_plans) < 2:
        _LAST_NEED_BATCH_CONTEXT_V5 = {
            "strategy": "V4_1_STANDARD_RERANK",
            "needs": [],
            "max_evidence": int(top_k),
        }
        return _FUSE_QUERY_RESULTS_V4_1(plans, top_k=top_k, rrf_k=rrf_k)

    original_plan = next((plan for plan in plans if _source_is_original_v5(plan)), plans[0])
    original_question = _clean_text(original_plan.get("query"))
    needs = [
        {
            "need_id": f"N{index}",
            "plan_index": plans.index(plan) + 1,
            "question": _clean_text(plan.get("query")),
            "business_function": _need_business_v5(str(plan.get("query") or "")),
        }
        for index, plan in enumerate(decomposed_plans, start=1)
    ]
    need_by_plan_index = {int(row["plan_index"]): row for row in needs}

    per_query: list[dict[str, Any]] = []
    all_hits: list[tuple[int, Mapping[str, Any], Sequence[Mapping[str, Any]]]] = []
    hits_by_plan_index: dict[int, list[dict[str, Any]]] = {}
    for plan_index, plan in enumerate(plans, start=1):
        hits = hybrid_minmax_search(str(plan["query"]), top_k=CANDIDATE_DEPTH)
        trace = dict(_V15_LAST_QUERY_TRACE)
        hits_by_plan_index[plan_index] = hits
        all_hits.append((plan_index, plan, hits))
        per_query.append({
            **plan,
            "plan_index": plan_index,
            "latency_ms": float(trace.get("query_total_latency_ms") or 0.0),
            "latency_breakdown_ms": trace,
            "hits": hits,
        })

    fusion_started = time.perf_counter()
    fused = _fused_candidates_v5(all_hits, rrf_k)[:RERANKER_CANDIDATE_DEPTH]
    fusion_latency_ms = (time.perf_counter() - fusion_started) * 1000

    pair_texts: list[list[str]] = []
    pair_refs: list[dict[str, Any]] = []
    for need in needs:
        for hit in hits_by_plan_index[int(need["plan_index"])]:
            pair_texts.append([str(need["question"]), _reranker_passage(hit["chunk"])])
            pair_refs.append({"kind": "need", "need_id": need["need_id"], "row": hit})
    for row in fused:
        pair_texts.append([original_question, _reranker_passage(row["chunk"])])
        pair_refs.append({"kind": "global", "row": row})
    if not pair_texts:
        raise RuntimeError("Need-aware Reranker 입력 쌍이 없습니다.")

    rerank_started = time.perf_counter()
    pair_scores = np.asarray(
        RERANKER_MODEL.predict(pair_texts, batch_size=RERANKER_BATCH_SIZE),
        dtype=np.float64,
    ).reshape(-1)
    rerank_latency_ms = (time.perf_counter() - rerank_started) * 1000
    if len(pair_scores) != len(pair_refs):
        raise RuntimeError("BAAI Reranker 점수 개수가 후보 쌍 개수와 다릅니다.")

    need_scored: dict[str, list[dict[str, Any]]] = defaultdict(list)
    global_scored: list[dict[str, Any]] = []
    for ref, score in zip(pair_refs, pair_scores.tolist()):
        if ref["kind"] == "need":
            need = next(value for value in needs if value["need_id"] == ref["need_id"])
            row = {**dict(ref["row"]), "need_reranker_score": float(score), **need}
            need_scored[str(need["need_id"])].append(row)
        else:
            global_scored.append({**dict(ref["row"]), "global_reranker_score": float(score)})
    for need_id in need_scored:
        need_scored[need_id].sort(
            key=lambda row: (-float(row["need_reranker_score"]), int(row.get("rank") or 10**9), str(row["chunk_id"]))
        )

    required_top_k = NEED_BATCH_REQUIRED_TOP_K_V5
    selected: list[dict[str, Any]] = []
    selected_chunks: set[str] = set()
    selected_parents: set[str] = set()
    selected_urls: set[str] = set()
    coverage_rows: list[dict[str, Any]] = []

    for need in needs:
        candidates = list(need_scored.get(str(need["need_id"]), []))
        business = str(need.get("business_function") or "")
        structural_candidates = [
            row for row in candidates
            if business
            and _clean_text((row.get("chunk") or {}).get("business_function")) == business
            and _clean_text((row.get("chunk") or {}).get("content"))
        ]
        domain_candidates = [
            row for row in structural_candidates
            if need_score_gate_passed_v5(float(row.get("need_reranker_score") or 0.0))
        ]
        selected_for_need = 0
        selected_score_rows: list[dict[str, Any]] = []
        strict_parent_rejected_count = 0
        fallback_rejected_count = 0
        need_parent_counts: dict[str, int] = defaultdict(int)
        # 업무가 검출됐을 때는 다른 업무 청크로 필수 자리를 채우지 않습니다.
        def _append_required_row_v5(row: Mapping[str, Any], *, parent_fallback: bool) -> None:
            nonlocal selected_for_need
            selection_reason = "PARENT_FALLBACK" if parent_fallback else "DISTINCT_PARENT"
            chosen = {
                **row,
                "selection_type": "NEED_REQUIRED",
                "selection_reason": selection_reason,
                "parent_fallback": bool(parent_fallback),
                "need_ids": [str(need["need_id"])],
                "need_queries": [str(need["question"])],
                "need_businesses": [business] if business else [],
                "reranker_score": float(row["need_reranker_score"]),
            }
            fused_row = next((value for value in fused if value["chunk_id"] == row["chunk_id"]), None)
            if fused_row:
                chosen["query_fusion_score"] = float(fused_row["query_fusion_score"])
                chosen["matched_queries"] = list(fused_row["matched_queries"])
            selected.append(chosen)
            _mark_selected_need_row_v5(
                chosen,
                selected_chunks=selected_chunks,
                selected_parents=selected_parents,
                selected_urls=selected_urls,
            )
            parent_id = _row_parent_key_v5(chosen)
            if parent_id:
                need_parent_counts[parent_id] += 1
            selected_for_need += 1
            selected_score_rows.append({
                "chunk_id": str(chosen.get("chunk_id") or ""),
                "parent_id": parent_id,
                "need_reranker_score": float(chosen.get("need_reranker_score") or 0.0),
                "selection_reason": selection_reason,
                "parent_fallback": bool(parent_fallback),
            })

        # 1차: 서로 다른 Parent를 우선 선택합니다.
        for row in domain_candidates:
            if selected_for_need >= required_top_k:
                break
            if not _can_select_need_row_v5(
                row,
                selected_chunks=selected_chunks,
                selected_parents=selected_parents,
                selected_urls=selected_urls,
            ):
                strict_parent_rejected_count += 1
                continue
            _append_required_row_v5(row, parent_fallback=False)

        # 2차: 서로 다른 Parent를 2개 이상 확보한 업무만 동일 Parent의 다른 chunk로 1자리를 보충합니다.
        if (
            selected_for_need < required_top_k
            and len(need_parent_counts) >= NEED_BATCH_MIN_DISTINCT_PARENTS_V5
        ):
            for row in domain_candidates:
                if selected_for_need >= required_top_k:
                    break
                chunk_id = str(row.get("chunk_id") or "")
                parent_id = _row_parent_key_v5(row)
                if not chunk_id or chunk_id in selected_chunks:
                    fallback_rejected_count += 1
                    continue
                if not parent_id or parent_id not in need_parent_counts:
                    continue
                if need_parent_counts[parent_id] >= NEED_BATCH_MAX_EVIDENCE_PER_PARENT_V5:
                    fallback_rejected_count += 1
                    continue
                _append_required_row_v5(row, parent_fallback=True)
        coverage_rows.append({
            **need,
            "candidate_count": len(candidates),
            "structural_candidate_count": len(structural_candidates),
            "domain_candidate_count": len(domain_candidates),
            "top_structural_candidate_scores": [
                float(row.get("need_reranker_score") or 0.0)
                for row in structural_candidates[:5]
            ],
            "selected_score_rows": selected_score_rows,
            "distinct_parent_count": len(need_parent_counts),
            "parent_fallback_count": sum(bool(row.get("parent_fallback")) for row in selected_score_rows),
            "minimum_distinct_parents": NEED_BATCH_MIN_DISTINCT_PARENTS_V5,
            "maximum_evidence_per_parent": NEED_BATCH_MAX_EVIDENCE_PER_PARENT_V5,
            "exclusion_counts": {
                "business_or_content": max(0, len(candidates) - len(structural_candidates)),
                "score_gate": max(0, len(structural_candidates) - len(domain_candidates)),
                "duplicate_chunk_or_parent": strict_parent_rejected_count,
                "parent_fallback_rejected": fallback_rejected_count,
            },
            **need_score_gate_metadata_v5(),
            "required_count": required_top_k,
            "selected_count": selected_for_need,
            "sufficient": bool(
                selected_for_need >= required_top_k
                and len(need_parent_counts) >= NEED_BATCH_MIN_DISTINCT_PARENTS_V5
            ),
        })

    rerank_norm = _minmax_values_v5([float(row["global_reranker_score"]) for row in global_scored])
    rrf_norm = _minmax_values_v5([float(row["query_fusion_score"]) for row in global_scored])
    optional_rows: list[dict[str, Any]] = []
    expected_businesses = {str(row["business_function"]) for row in needs if row.get("business_function")}
    for row, rerank_value, rrf_value in zip(global_scored, rerank_norm, rrf_norm):
        chunk_business = _clean_text((row.get("chunk") or {}).get("business_function"))
        if expected_businesses and chunk_business not in expected_businesses:
            continue
        matched_need_ids = [
            str(need_by_plan_index[int(match["plan_index"])]["need_id"])
            for match in row.get("matched_queries") or []
            if int(match.get("plan_index") or 0) in need_by_plan_index
        ]
        optional_rows.append({
            **row,
            "selection_type": "GLOBAL_OPTIONAL",
            "need_ids": list(dict.fromkeys(matched_need_ids)),
            "need_queries": [
                str(next(value for value in needs if value["need_id"] == need_id)["question"])
                for need_id in dict.fromkeys(matched_need_ids)
            ],
            "need_businesses": [chunk_business] if chunk_business else [],
            "global_reranker_norm": float(rerank_value),
            "query_fusion_norm": float(rrf_value),
            "composite_score": (
                NEED_BATCH_GLOBAL_RERANK_WEIGHT_V5 * float(rerank_value)
                + NEED_BATCH_GLOBAL_RRF_WEIGHT_V5 * float(rrf_value)
            ),
            "reranker_score": float(row["global_reranker_score"]),
        })
    optional_rows.sort(
        key=lambda row: (-float(row["composite_score"]), -float(row["global_reranker_score"]), str(row["chunk_id"]))
    )
    for row in optional_rows:
        if len(selected) >= NEED_BATCH_MAX_EVIDENCE_V5:
            break
        if not _can_select_need_row_v5(
            row,
            selected_chunks=selected_chunks,
            selected_parents=selected_parents,
            selected_urls=selected_urls,
        ):
            continue
        selected.append(row)
        _mark_selected_need_row_v5(
            row,
            selected_chunks=selected_chunks,
            selected_parents=selected_parents,
            selected_urls=selected_urls,
        )

    # 필수 선택 부족 여부는 Evidence Pack 감사에서 하드 게이트로 처리합니다.
    final = []
    for rank, row in enumerate(selected[:NEED_BATCH_MAX_EVIDENCE_V5], start=1):
        final.append({
            **row,
            "rank": rank,
            "minmax_score": float(row.get("best_minmax_score") or row.get("minmax_score") or 0.0),
        })
    if not final:
        raise RuntimeError("Need-aware 선택 후 남은 검색 근거가 없습니다.")

    _LAST_NEED_BATCH_CONTEXT_V5 = {
        "strategy": "NEED_BATCH_RERANK_V5",
        "original_question": original_question,
        "needs": coverage_rows,
        "required_top_k": required_top_k,
        "max_evidence": NEED_BATCH_MAX_EVIDENCE_V5,
        "selected_count": len(final),
    }
    _LAST_RERANK_TRACE = {
        "latency_ms": rerank_latency_ms,
        "model": RERANKER_MODEL_NAME,
        "device": RERANKER_DEVICE,
        "strategy": "NEED_BATCH_RERANK_V5",
        "candidate_count": len(pair_texts),
        "returned_count": len(final),
        "batch_size": RERANKER_BATCH_SIZE,
        "need_count": len(needs),
        "need_pair_count": sum(1 for ref in pair_refs if ref["kind"] == "need"),
        "global_pair_count": sum(1 for ref in pair_refs if ref["kind"] == "global"),
        "required_top_k": required_top_k,
        "max_evidence": NEED_BATCH_MAX_EVIDENCE_V5,
        "need_coverage": coverage_rows,
        "all_needs_sufficient": all(bool(row["sufficient"]) for row in coverage_rows),
        "global_composite_weights": {
            "original_reranker": NEED_BATCH_GLOBAL_RERANK_WEIGHT_V5,
            "weighted_rrf": NEED_BATCH_GLOBAL_RRF_WEIGHT_V5,
        },
    }
    for row in per_query:
        row["query_fusion_latency_ms"] = fusion_latency_ms
        row["reranker_latency_ms"] = rerank_latency_ms
        row["reranker_candidate_count"] = len(pair_texts)
    return final, per_query


def build_compact_parent_evidence_pack_v1(
    question: str,
    search_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """최대 9개 Evidence와 업무별 3개 need-근거 및 제한적 Parent fallback을 보존하는 V5 Pack."""
    by_parent: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for result in search_results:
        child = dict(result.get("chunk") or {})
        child_id = str(result.get("chunk_id") or child.get("chunk_id") or "")
        parent_id = str(result.get("parent_doc_id") or _parent_id_for_chunk(child))
        parent_fallback = bool(result.get("parent_fallback"))
        evidence_group_key = (
            f"{parent_id}::fallback::{child_id}"
            if parent_fallback
            else f"{parent_id}::primary"
        )
        context_chunk_ids = (
            [child_id]
            if parent_fallback
            else list(result.get("parent_context_chunk_ids") or [child_id])
        )
        row = by_parent.setdefault(evidence_group_key, {
            "rank": int(result.get("rank") or len(by_parent) + 1),
            "parent_id": parent_id,
            "representative_chunk_id": child_id,
            "matched_child_ids": [],
            "matched_child_ranks": [],
            "context_chunk_ids": context_chunk_ids,
            "document_title": _clean_text(child.get("title") or child.get("document_title")),
            "source_url": _clean_text(child.get("source_url")),
            "need_ids": [],
            "need_queries": [],
            "need_businesses": [],
            "selection_types": [],
            "selection_reasons": [],
            "parent_fallback": parent_fallback,
            "need_reranker_scores": [],
        })
        row["matched_child_ids"].append(child_id)
        row["matched_child_ranks"].append(int(result.get("rank") or 0))
        row["need_ids"].extend(str(value) for value in result.get("need_ids") or [])
        row["need_queries"].extend(str(value) for value in result.get("need_queries") or [])
        row["need_businesses"].extend(str(value) for value in result.get("need_businesses") or [])
        row["selection_types"].append(str(result.get("selection_type") or "STANDARD"))
        row["selection_reasons"].append(str(result.get("selection_reason") or "STANDARD"))
        if str(result.get("selection_type") or "") == "NEED_REQUIRED":
            row["need_reranker_scores"].append(float(result.get("need_reranker_score") or 0.0))

    evidence: list[dict[str, Any]] = []
    sources: OrderedDict[str, dict[str, Any]] = OrderedDict()
    total_remaining = ANSWER_EVIDENCE_TOTAL_MAX_CHARS
    for parent_index, row in enumerate(by_parent.values()):
        if total_remaining <= 0 or parent_index >= len(ANSWER_EVIDENCE_RANK_BUDGETS_V5):
            break
        parent_budget = min(ANSWER_EVIDENCE_RANK_BUDGETS_V5[parent_index], total_remaining)
        ordered_ids = _proximity_order_v1(row["context_chunk_ids"], row["matched_child_ids"])
        parts: list[str] = []
        included_ids: list[str] = []
        section_titles: list[str] = []
        remaining = parent_budget
        for chunk_id in ordered_ids:
            chunk = CHUNKS_BY_ID.get(str(chunk_id))
            if chunk is None:
                continue
            title = _clean_text(chunk.get("title"))
            section = _clean_text(chunk.get("section_title"))
            if section and section not in section_titles:
                section_titles.append(section)
            label = " / ".join(value for value in (title, section) if value)
            part = f"[{chunk_id}] {label}\n{_clean_text(chunk.get('content'))}".strip()
            separator_cost = 2 if parts else 0
            if remaining <= separator_cost:
                break
            part = _truncate_at_boundary_v1(part, remaining - separator_cost)
            if not part:
                break
            parts.append(part)
            included_ids.append(str(chunk_id))
            remaining -= len(part) + separator_cost
            if remaining < 120:
                break
        content = "\n\n".join(parts)
        if not content:
            continue
        evidence_id = f"E{len(evidence) + 1}"
        evidence.append({
            "evidence_id": evidence_id,
            "rank": int(row["rank"]),
            "chunk_id": row["representative_chunk_id"],
            "parent_id": row["parent_id"],
            "context_chunk_ids": included_ids,
            "matched_child_ids": list(dict.fromkeys(row["matched_child_ids"])),
            "matched_child_ranks": sorted(set(row["matched_child_ranks"])),
            "document_title": row["document_title"],
            "section_title": " · ".join(section_titles),
            "content": content,
            "context_char_count": len(content),
            "context_truncated": len(included_ids) < len(ordered_ids),
            "source_url": row["source_url"],
            "need_ids": list(dict.fromkeys(row["need_ids"])),
            "need_queries": list(dict.fromkeys(row["need_queries"])),
            "need_businesses": list(dict.fromkeys(row["need_businesses"])),
            "selection_types": list(dict.fromkeys(row["selection_types"])),
            "selection_reasons": list(dict.fromkeys(row["selection_reasons"])),
            "parent_fallback": bool(row["parent_fallback"]),
            "need_reranker_score": max(row["need_reranker_scores"], default=0.0),
        })
        total_remaining -= len(content)
        if row["source_url"]:
            source = sources.setdefault(row["source_url"], {
                "source_id": f"S{len(sources) + 1}",
                "title": row["document_title"] or "공식 출처",
                "source_url": row["source_url"],
                "evidence_ids": [],
            })
            source["evidence_ids"].append(evidence_id)

    if not evidence:
        raise ValueError("V5 Evidence Pack을 만들 근거가 없습니다.")
    evidence_ids = {row["evidence_id"] for row in evidence}
    need_rows = []
    for need in _LAST_NEED_BATCH_CONTEXT_V5.get("needs") or []:
        need_id = str(need.get("need_id") or "")
        business = _clean_text(need.get("business_function"))
        linked_rows = [
            row for row in evidence
            if need_id in set(row.get("need_ids") or [])
            and business
            and business in set(row.get("need_businesses") or [])
            and "NEED_REQUIRED" in set(row.get("selection_types") or [])
            and _clean_text(row.get("content"))
            and need_score_gate_passed_v5(float(row.get("need_reranker_score") or 0.0))
        ]
        linked = [row["evidence_id"] for row in linked_rows]
        distinct_parent_count = len({str(row.get("parent_id") or "") for row in linked_rows if row.get("parent_id")})
        parent_fallback_count = sum(bool(row.get("parent_fallback")) for row in linked_rows)
        need_rows.append({
            **dict(need),
            "required_count": NEED_BATCH_REQUIRED_TOP_K_V5,
            "selected_count": len(linked),
            "distinct_parent_count": distinct_parent_count,
            "parent_fallback_count": parent_fallback_count,
            "minimum_distinct_parents": NEED_BATCH_MIN_DISTINCT_PARENTS_V5,
            "maximum_evidence_per_parent": NEED_BATCH_MAX_EVIDENCE_PER_PARENT_V5,
            "sufficient": bool(
                len(linked) >= NEED_BATCH_REQUIRED_TOP_K_V5
                and distinct_parent_count >= NEED_BATCH_MIN_DISTINCT_PARENTS_V5
            ),
            **need_score_gate_metadata_v5(),
            "evidence_ids": [value for value in linked if value in evidence_ids],
        })
    shared_evidence_ids = [
        row["evidence_id"] for row in evidence
        if not row.get("need_ids") or "GLOBAL_OPTIONAL" in set(row.get("selection_types") or [])
    ]
    return {
        "question": _clean_text(question),
        "retrieval_strategy": str(_LAST_NEED_BATCH_CONTEXT_V5.get("strategy") or "V4_1_STANDARD_RERANK"),
        "search_parent_context_max_chars": PARENT_CONTEXT_MAX_CHARS,
        "answer_evidence_total_max_chars": ANSWER_EVIDENCE_TOTAL_MAX_CHARS,
        "answer_prompt_version": ANSWER_PROMPT_VERSION,
        "required_evidence_per_business": NEED_BATCH_REQUIRED_TOP_K_V5,
        "minimum_distinct_parents_per_business": NEED_BATCH_MIN_DISTINCT_PARENTS_V5,
        "maximum_evidence_per_parent": NEED_BATCH_MAX_EVIDENCE_PER_PARENT_V5,
        **need_score_gate_metadata_v5(),
        "maximum_cross_business_count": NEED_BATCH_MAX_BUSINESSES_V5,
        "needs": need_rows,
        "shared_evidence_ids": shared_evidence_ids,
        "evidence": evidence,
        "sources": list(sources.values()),
    }


def audit_need_evidence_pack_v5(pack: Mapping[str, Any]) -> dict[str, Any]:
    """교차업무 Evidence Pack의 업무별 Top-3 구조와 선택적 점수 게이트 감사."""
    pack = dict(pack or {})
    needs = [dict(row) for row in pack.get("needs") or []]
    evidence = [dict(row) for row in pack.get("evidence") or []]
    evidence_by_id = {str(row.get("evidence_id") or ""): row for row in evidence}
    need_ids = [_clean_text(row.get("need_id")) for row in needs]
    businesses = [_clean_text(row.get("business_function")) for row in needs]
    structure_valid = bool(
        2 <= len(needs) <= NEED_BATCH_MAX_BUSINESSES_V5
        and all(need_ids)
        and len(set(need_ids)) == len(need_ids)
        and all(businesses)
        and len(set(businesses)) == len(businesses)
    )
    duplicate_chunks = len({str(row.get("chunk_id") or "") for row in evidence}) != len(evidence)
    parent_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evidence:
        parent_id = str(row.get("parent_id") or "")
        if parent_id:
            parent_groups[parent_id].append(row)
    duplicate_parents = any(len(rows) > 1 for rows in parent_groups.values())
    invalid_duplicate_parent_ids: list[str] = []
    parent_limit_exceeded = False
    for parent_id, rows in parent_groups.items():
        if len(rows) <= 1:
            continue
        fallback_rows = [row for row in rows if bool(row.get("parent_fallback"))]
        primary_rows = [row for row in rows if not bool(row.get("parent_fallback"))]
        if len(rows) > NEED_BATCH_MAX_EVIDENCE_PER_PARENT_V5:
            parent_limit_exceeded = True
        if (
            len(rows) > NEED_BATCH_MAX_EVIDENCE_PER_PARENT_V5
            or len(primary_rows) != 1
            or len(fallback_rows) != len(rows) - 1
        ):
            invalid_duplicate_parent_ids.append(parent_id)
    parent_fallback_used = any(bool(row.get("parent_fallback")) for row in evidence)
    parent_fallback_valid = bool(
        not invalid_duplicate_parent_ids
        and all(str(row.get("parent_id") or "") for row in evidence)
        and all(
            not bool(row.get("parent_fallback")) or len(parent_groups.get(str(row.get("parent_id") or ""), [])) > 1
            for row in evidence
        )
    )
    need_audits = []
    for need, need_id, business in zip(needs, need_ids, businesses):
        evidence_audits = []
        for evidence_id in list(dict.fromkeys(str(value) for value in need.get("evidence_ids") or [])):
            row = evidence_by_id.get(evidence_id)
            exists = row is not None
            row = dict(row or {})
            score = float(row.get("need_reranker_score") or 0.0)
            score_gate_passed = need_score_gate_passed_v5(score)
            parent_id = str(row.get("parent_id") or "")
            parent_rows = parent_groups.get(parent_id, [])
            parent_structure_valid = bool(
                parent_id
                and parent_id not in set(invalid_duplicate_parent_ids)
                and (
                    (len(parent_rows) == 1 and not bool(row.get("parent_fallback")))
                    or (len(parent_rows) > 1 and parent_fallback_valid)
                )
            )
            qualified = bool(
                exists
                and need_id in set(str(value) for value in row.get("need_ids") or [])
                and business in set(_clean_text(value) for value in row.get("need_businesses") or [])
                and "NEED_REQUIRED" in set(str(value) for value in row.get("selection_types") or [])
                and _clean_text(row.get("content"))
                and score_gate_passed
                and parent_structure_valid
            )
            evidence_audits.append({
                "evidence_id": evidence_id,
                "need_reranker_score": score,
                "score_gate_passed": score_gate_passed,
                "parent_id": parent_id,
                "parent_fallback": bool(row.get("parent_fallback")),
                "selection_reasons": list(row.get("selection_reasons") or []),
                "parent_structure_valid": parent_structure_valid,
                "qualified": qualified,
            })
        qualified_ids = [row["evidence_id"] for row in evidence_audits if row["qualified"]]
        qualified_parent_ids = [
            str(row.get("parent_id") or "")
            for row in evidence_audits
            if row.get("qualified") and row.get("parent_id")
        ]
        qualified_parent_counts = {
            parent_id: qualified_parent_ids.count(parent_id)
            for parent_id in set(qualified_parent_ids)
        }
        distinct_parent_count = len(qualified_parent_counts)
        parent_fallback_count = sum(
            bool(row.get("parent_fallback"))
            for row in evidence_audits
            if row.get("qualified")
        )
        parent_diversity_passed = bool(
            distinct_parent_count >= NEED_BATCH_MIN_DISTINCT_PARENTS_V5
            and all(count <= NEED_BATCH_MAX_EVIDENCE_PER_PARENT_V5 for count in qualified_parent_counts.values())
        )
        need_audits.append({
            "need_id": need_id,
            "business_function": business,
            "candidate_count": int(need.get("candidate_count") or 0),
            "structural_candidate_count": int(need.get("structural_candidate_count") or 0),
            "top_structural_candidate_scores": [
                float(value) for value in need.get("top_structural_candidate_scores") or []
            ],
            "selected_score_rows": list(need.get("selected_score_rows") or []),
            "exclusion_counts": dict(need.get("exclusion_counts") or {}),
            "distinct_parent_count": distinct_parent_count,
            "parent_fallback_count": parent_fallback_count,
            "minimum_distinct_parents": NEED_BATCH_MIN_DISTINCT_PARENTS_V5,
            "maximum_evidence_per_parent": NEED_BATCH_MAX_EVIDENCE_PER_PARENT_V5,
            "parent_diversity_passed": parent_diversity_passed,
            "required_count": NEED_BATCH_REQUIRED_TOP_K_V5,
            "qualified_evidence_count": len(qualified_ids),
            "qualified_evidence_ids": qualified_ids,
            "passed": bool(
                business
                and len(qualified_ids) >= NEED_BATCH_REQUIRED_TOP_K_V5
                and parent_diversity_passed
            ),
            "evidence_audits": evidence_audits,
        })
    passed = bool(
        structure_valid
        and not duplicate_chunks
        and parent_fallback_valid
        and need_audits
        and all(row["passed"] for row in need_audits)
    )
    return {
        "passed": passed,
        "structure_valid": structure_valid,
        "need_count": len(needs),
        "business_count": len(set(value for value in businesses if value)),
        "required_evidence_per_business": NEED_BATCH_REQUIRED_TOP_K_V5,
        "minimum_distinct_parents_per_business": NEED_BATCH_MIN_DISTINCT_PARENTS_V5,
        "maximum_evidence_per_parent": NEED_BATCH_MAX_EVIDENCE_PER_PARENT_V5,
        **need_score_gate_metadata_v5(),
        "maximum_cross_business_count": NEED_BATCH_MAX_BUSINESSES_V5,
        "capacity_exceeded": len(needs) > NEED_BATCH_MAX_BUSINESSES_V5,
        "duplicate_chunks": duplicate_chunks,
        "duplicate_parents": duplicate_parents,
        "parent_fallback_used": parent_fallback_used,
        "parent_fallback_valid": parent_fallback_valid,
        "parent_limit_exceeded": parent_limit_exceeded,
        "invalid_duplicate_parent_ids": invalid_duplicate_parent_ids,
        "needs": need_audits,
    }


def need_score_summary_rows_v5(pack_or_audit: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = dict(pack_or_audit or {})
    audit = (
        value
        if value.get("needs") and "required_evidence_per_business" in value
        else audit_need_evidence_pack_v5(value)
    )
    rows = []
    for need in audit.get("needs") or []:
        top_scores = [float(value) for value in need.get("top_structural_candidate_scores") or []]
        if not top_scores:
            top_scores = [
                float(row.get("need_reranker_score") or 0.0)
                for row in need.get("evidence_audits") or []
            ]
        selected_scores = [
            float(row.get("need_reranker_score") or 0.0)
            for row in need.get("evidence_audits") or []
            if row.get("qualified")
        ]
        rows.append({
            "업무": str(need.get("business_function") or need.get("need_id") or "업무"),
            "선택/필수": f"{int(need.get('qualified_evidence_count') or 0)}/{int(need.get('required_count') or NEED_BATCH_REQUIRED_TOP_K_V5)}",
            "선택 점수": ", ".join(f"{value:.4f}" for value in selected_scores) or "-",
            "업무일치 후보 상위점수": ", ".join(f"{value:.4f}" for value in top_scores[:5]) or "-",
            "점수 게이트": (
                f"활성 · {NEED_BATCH_MIN_RERANKER_SCORE_V5:.4f} 이상"
                if NEED_BATCH_SCORE_GATE_ENABLED_V5
                else "비활성 · 점수는 표시만 함"
            ),
            "Parent 구성": (
                f"서로 다른 Parent {int(need.get('distinct_parent_count') or 0)}개"
                f"/{int(need.get('minimum_distinct_parents') or NEED_BATCH_MIN_DISTINCT_PARENTS_V5)}개 이상"
                f" · fallback {int(need.get('parent_fallback_count') or 0)}개"
            ),
            "제외 사유별 개수": dict(need.get("exclusion_counts") or {}),
            "통과": bool(need.get("passed")),
        })
    return rows


def format_need_score_gate_message_v5(audit: Mapping[str, Any]) -> str:
    rows = need_score_summary_rows_v5(audit)
    score_lines = [
        f"- {row['업무']}: 후보 상위점수 {row['업무일치 후보 상위점수']} · 선택 {row['선택/필수']} · {row['Parent 구성']} · {row['점수 게이트']}"
        for row in rows
    ]
    failed = [str(row.get("business_function") or row.get("need_id") or "업무") for row in audit.get("needs") or [] if not row.get("passed")]
    requirement = (
        f"점수 {NEED_BATCH_MIN_RERANKER_SCORE_V5:.4f} 이상의 공식 근거"
        if NEED_BATCH_SCORE_GATE_ENABLED_V5
        else "점수 임계점을 적용하지 않은 업무 일치 공식 근거"
    )
    return (
        "**업무별 Reranker 점수**\n\n"
        + ("\n".join(score_lines) if score_lines else "- 점수 정보 없음")
        + f"\n\n업무별로 {requirement} {NEED_BATCH_REQUIRED_TOP_K_V5}개를 확보하지 못해 답변을 생성하지 않았습니다. "
        + "질문의 대상이나 조건을 조금 더 구체적으로 작성해 다시 질문해 주세요."
        + (" 근거가 부족한 업무: " + ", ".join(failed) if failed else "")
    )


def _make_need_gate_fixture_v5(businesses: Sequence[str], *, counts: Sequence[int] | None = None, low_score: tuple[int, int, float] | None = None) -> dict[str, Any]:
    counts = list(counts or [NEED_BATCH_REQUIRED_TOP_K_V5] * len(businesses))
    evidence = []
    needs = []
    for business_index, (business, count) in enumerate(zip(businesses, counts), start=1):
        need_id = f"N{business_index}"
        linked_ids = []
        for evidence_index in range(count):
            evidence_id = f"E{len(evidence) + 1}"
            score = NEED_BATCH_MIN_RERANKER_SCORE_V5
            if low_score and low_score[:2] == (business_index, evidence_index):
                score = float(low_score[2])
            linked_ids.append(evidence_id)
            evidence.append({
                "evidence_id": evidence_id,
                "chunk_id": f"C-{business_index}-{evidence_index}",
                "parent_id": f"P-{business_index}-{evidence_index}",
                "source_url": "https://www.kdic.or.kr/shared-official-page",
                "content": f"{business} 공식 근거 {evidence_index + 1}",
                "need_ids": [need_id],
                "need_businesses": [business] if business else [],
                "selection_types": ["NEED_REQUIRED"],
                "need_reranker_score": score,
            })
        needs.append({"need_id": need_id, "business_function": business, "evidence_ids": linked_ids})
    return {"retrieval_strategy": "NEED_BATCH_RERANK_V5", "needs": needs, "evidence": evidence}

# ==== overlay: dc-comparison-core ====

# D-C 1Call vs 2Call 공통 구조와 검증기

import copy
import json
import re
import time
from typing import Any, Mapping, Sequence


DC_PROMPT_VERSION_V1 = "dc-fact-pack-one-vs-two-call-tagged-v3-p0-rule-cross"
DC_ONECALL_MAX_TOKENS_V1 = 2400
DC_SKELETON_MAX_TOKENS_V1 = 1800
DC_FINAL_MAX_TOKENS_V1 = 1600

DC_FACT_SAFETY_RULES_V1 = """
[Fact Index 검증 규칙]
1. 원본 Evidence를 기본 근거로 사용하고 활성화된 Fact Index는 혼동 방지와 검증에만 사용하세요.
2. verified_claims는 해당 claim_id와 source_chunk_ids 범위에서만 사용하세요.
3. forbidden_claims에 해당하는 내용을 주장하지 마세요.
4. Fact Index와 원본 Evidence가 충돌하면 임의로 선택하지 말고 불확실성 또는 충돌로 표시하세요.
5. Fact Index가 없으면 원본 Evidence 범위에서만 답하세요.
6. Fact claim을 사용한 항목에는 [FI-CAND-001:F1] 형태의 실제 ID를 연결하세요.
""".strip()

DC_SKELETON_SYSTEM_PROMPT_V1 = (
    D2_SKELETON_SYSTEM_PROMPT
    + "\n\n"
    + DC_FACT_SAFETY_RULES_V1
    + "\n\n"
    + "ANSWERED와 PARTIAL에는 evidence_ids 또는 fact_claim_ids 중 하나 이상의 실제 근거를 연결하세요."
)

DC_FINAL_SYSTEM_PROMPT_V1 = (
    D_STRUCTURED_FINAL_SYSTEM_PROMPT_V3
    + "\n\n"
    + DC_FACT_SAFETY_RULES_V1
    + "\n최종답변은 Skeleton이 허용한 Evidence ID와 Fact Claim ID만 사용하세요."
)

DC_ONECALL_SYSTEM_PROMPT_V1 = (
    "당신은 예금보험공사 공식 문서 기반 Answer Skeleton 및 최종답변 생성기입니다.\n\n"
    "1. 모든 Answer Need에 정확히 하나의 answer_item을 만드세요.\n"
    "2. 각 item은 ANSWERED, PARTIAL, UNSUPPORTED 중 하나이며 실제 evidence_ids 또는 fact_claim_ids를 연결하세요.\n"
    "3. 동일한 호출에서 Answer Skeleton과 사용자용 Markdown 답변을 함께 생성하세요.\n"
    "4. 최종 answer는 answer_skeleton의 claim·conditions·details와 허용 근거만 사용하세요.\n"
    "5. 한 업무의 조건을 다른 업무에 적용하지 말고, 근거 없는 동시·병행 가능성을 추론하지 마세요.\n"
    "6. 전체를 하나의 JSON으로 감싸지 마세요. 아래 SKELETON_JSON과 FINAL_ANSWER 태그 두 개를 정확히 출력하세요.\n"
    "7. SKELETON_JSON 내부만 유효한 JSON 객체로 작성하고 FINAL_ANSWER에는 일반 Markdown을 작성하세요.\n"
    "8. 코드 펜스와 두 태그 밖의 설명은 출력하지 마세요.\n\n"
    + DC_FACT_SAFETY_RULES_V1
    + "\n\n"
    + MARKDOWN_FORMAT_RULES_V3
    + "\n\n"
    + ACTION_LINK_PROMPT_RULE_V1
    + "\n\n"
    + NEED_COVERAGE_PROMPT_V5
)


def extract_dc_answer_needs_v1(
    question: str,
    pack: Mapping[str, Any],
) -> list[dict[str, Any]]:
    retrieval_needs = [
        dict(row) for row in pack.get("needs") or []
        if str(row.get("need_id") or "") and str(row.get("question") or "")
    ]
    if len(retrieval_needs) >= 2:
        return [
            {
                "need_id": str(row["need_id"]),
                "need_type": "CROSS_BUSINESS",
                "label": str(row.get("business_function") or row.get("question") or row["need_id"]),
                "question_part": str(row.get("question") or ""),
                "retrieval_evidence_ids": list(row.get("evidence_ids") or []),
            }
            for row in retrieval_needs
        ]
    return extract_answer_needs_v2(question)


def validate_dc_skeleton_v1(
    raw: Mapping[str, Any],
    answer_needs: Sequence[Mapping[str, Any]],
    pack: Mapping[str, Any],
) -> dict[str, Any]:
    allowed_evidence = set(answer_b_core._allowed_evidence(pack))
    allowed_facts = set(_allowed_fact_claims_v3(pack))
    raw_items = raw.get("answer_items") if isinstance(raw.get("answer_items"), list) else []
    by_need = {
        str(item.get("need_id") or ""): item
        for item in raw_items
        if isinstance(item, Mapping)
    }
    items: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    invalid_evidence_ids: list[str] = []
    invalid_fact_claim_ids: list[str] = []
    for need in answer_needs:
        need_id = str(need["need_id"])
        source = by_need.get(need_id) or {}
        status = str(source.get("status") or "UNSUPPORTED").upper()
        if status not in NEED_STATUS_VALUES_V2:
            status = "UNSUPPORTED"
        requested_evidence = answer_b_core._clean_list(source.get("evidence_ids"))
        requested_facts = _clean_fact_claim_keys_v3(source.get("fact_claim_ids"))
        invalid_evidence_ids.extend(value for value in requested_evidence if value not in allowed_evidence)
        invalid_fact_claim_ids.extend(value for value in requested_facts if value not in allowed_facts)
        evidence_ids = [value for value in requested_evidence if value in allowed_evidence]
        fact_claim_ids = [value for value in requested_facts if value in allowed_facts]
        claim = answer_b_core._clean(source.get("claim"))
        if status == "ANSWERED" and not evidence_ids and not fact_claim_ids:
            status = "PARTIAL" if claim else "UNSUPPORTED"
        if status == "UNSUPPORTED" and not claim:
            claim = f"{need['label']}은 현재 근거로 확인되지 않습니다."
        item = {
            "need_id": need_id,
            "need_type": str(need.get("need_type") or "GENERAL"),
            "topic": answer_b_core._clean(source.get("topic")) or str(need["label"]),
            "status": status,
            "claim": claim,
            "conditions": answer_b_core._clean_list(source.get("conditions")),
            "details": answer_b_core._clean_list(source.get("details")),
            "evidence_ids": list(dict.fromkeys(evidence_ids)),
            "fact_claim_ids": list(dict.fromkeys(fact_claim_ids)),
            "missing_reason": answer_b_core._clean(source.get("missing_reason")),
        }
        items.append(item)
        coverage_rows.append({
            "need_id": need_id,
            "need_type": item["need_type"],
            "label": str(need["label"]),
            "status": status,
            "evidence_ids": item["evidence_ids"],
            "fact_claim_ids": item["fact_claim_ids"],
            "missing_reason": item["missing_reason"],
        })
    program = calculate_program_coverage_v2(coverage_rows)
    return {
        "core_answer": answer_b_core._clean(raw.get("core_answer")),
        "answer_items": items,
        "need_coverage": coverage_rows,
        "uncertainties": answer_b_core._clean_list(raw.get("uncertainties")),
        "conflicts": answer_b_core._clean_list(raw.get("conflicts")),
        "invalid_evidence_ids": list(dict.fromkeys(invalid_evidence_ids)),
        "invalid_fact_claim_ids": list(dict.fromkeys(invalid_fact_claim_ids)),
        "reference_validation_passed": not invalid_evidence_ids and not invalid_fact_claim_ids,
        **program,
    }


def filter_augmented_pack_for_dc_v1(
    pack: Mapping[str, Any],
    skeleton: Mapping[str, Any],
) -> dict[str, Any]:
    used_evidence_ids = {
        value
        for item in skeleton.get("answer_items") or []
        for value in item.get("evidence_ids") or []
    }
    used_fact_ids = {
        value
        for item in skeleton.get("answer_items") or []
        for value in item.get("fact_claim_ids") or []
    }
    evidence = [
        copy.deepcopy(dict(row))
        for row in pack.get("evidence") or []
        if str(row.get("evidence_id") or "") in used_evidence_ids
    ]
    source_urls = {str(row.get("source_url") or "") for row in evidence}
    fact_index = copy.deepcopy(dict(pack.get("fact_index") or {}))
    filtered_supplements = []
    for supplement in fact_index.get("supplements") or []:
        fact_id = str(supplement.get("fact_index_id") or "")
        selected_claims = [
            copy.deepcopy(dict(claim))
            for claim in supplement.get("verified_claims") or []
            if f"{fact_id}:{claim.get('claim_id')}" in used_fact_ids
        ]
        if selected_claims or supplement.get("forbidden_claims"):
            filtered_supplements.append({
                **copy.deepcopy(dict(supplement)),
                "verified_claims": selected_claims,
            })
    fact_index["supplements"] = filtered_supplements
    fact_index["supplement_count"] = len(filtered_supplements)
    return {
        **copy.deepcopy(dict(pack)),
        "evidence": evidence,
        "sources": [
            copy.deepcopy(dict(row))
            for row in pack.get("sources") or []
            if str(row.get("source_url") or "") in source_urls
        ],
        "fact_index": fact_index,
        "filtered_for_dc": True,
        "original_evidence_count": len(pack.get("evidence") or []),
        "selected_evidence_count": len(evidence),
        "selected_fact_claim_count": len(used_fact_ids),
    }


def _extract_json_relaxed_dc_v1(raw: str) -> dict[str, Any]:
    try:
        value = answer_b_core._extract_json_object(raw)
        if isinstance(value, Mapping):
            return dict(value)
    except Exception:
        pass
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(raw or "").strip(), flags=re.I | re.S)
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start:end + 1]
        try:
            value = json.loads(candidate, strict=False)
        except Exception as error:
            raise ValueError(f"D-C JSON 로컬 복구 실패: {error}") from error
        if isinstance(value, Mapping):
            return dict(value)
    raise ValueError("D-C 구조화 JSON 객체를 찾지 못했습니다.")


def _onecall_raw_preview_dc_v2(raw: Any, limit: int = 900) -> str:
    value = str(raw or "").replace("\x00", "").strip()
    return value[:limit].replace("\r", "\\r").replace("\n", "\\n")


def _parse_onecall_output_dc_v2(raw: str) -> dict[str, Any]:
    """태그 형식을 우선 처리하고 기존 중첩 JSON도 하위 호환으로 허용합니다."""
    text = str(raw or "").strip()
    if not text:
        raise ValueError("ONECALL_OUTPUT_CONTRACT_FAILED: HCX 응답이 비어 있습니다.")

    # v1.1 형식과 이미 정상적으로 중첩 JSON을 반환하는 응답을 계속 지원합니다.
    try:
        legacy = _extract_json_relaxed_dc_v1(text)
        legacy_skeleton = legacy.get("answer_skeleton") or legacy.get("skeleton")
        if isinstance(legacy_skeleton, Mapping) and str(legacy.get("answer") or "").strip():
            return {
                **legacy,
                "answer_skeleton": dict(legacy_skeleton),
                "answer": str(legacy.get("answer") or "").strip(),
                "output_contract": "LEGACY_NESTED_JSON",
            }
    except Exception:
        pass

    skeleton_match = re.search(
        r"<SKELETON_JSON>\s*(.*?)\s*</SKELETON_JSON>",
        text,
        flags=re.I | re.S,
    )
    answer_match = re.search(
        r"<FINAL_ANSWER>\s*(.*?)\s*</FINAL_ANSWER>",
        text,
        flags=re.I | re.S,
    )
    missing_tags = []
    if skeleton_match is None:
        missing_tags.append("SKELETON_JSON")
    if answer_match is None:
        missing_tags.append("FINAL_ANSWER")
    if missing_tags:
        raise ValueError(
            "ONECALL_OUTPUT_CONTRACT_FAILED: 필수 태그 누락="
            + ",".join(missing_tags)
            + "; raw_preview="
            + _onecall_raw_preview_dc_v2(text)
        )

    skeleton_text = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        skeleton_match.group(1).strip(),
        flags=re.I | re.S,
    ).strip()
    try:
        skeleton = _extract_json_relaxed_dc_v1(skeleton_text)
    except Exception as error:
        raise ValueError(
            "ONECALL_OUTPUT_CONTRACT_FAILED: SKELETON_JSON 파싱 실패="
            + str(error)
            + "; raw_preview="
            + _onecall_raw_preview_dc_v2(text)
        ) from error
    answer = answer_match.group(1).strip()
    if not answer:
        raise ValueError(
            "ONECALL_OUTPUT_CONTRACT_FAILED: FINAL_ANSWER가 비어 있습니다.; raw_preview="
            + _onecall_raw_preview_dc_v2(text)
        )
    return {
        "answer_skeleton": skeleton,
        "answer": answer,
        "used_evidence_ids": _DC_EVIDENCE_REF_PATTERN_V1.findall(answer),
        "used_fact_claim_ids": _DC_FACT_REF_PATTERN_V1.findall(answer),
        "output_contract": "TAGGED_SKELETON_AND_MARKDOWN_V2",
    }


_DC_EVIDENCE_REF_PATTERN_V1 = re.compile(r"\[(E\d+)\]")
_DC_FACT_REF_PATTERN_V1 = re.compile(r"\[((?:FI-[A-Z0-9-]+):F\d+)\]", re.I)


def audit_dc_final_references_v1(
    answer: str,
    skeleton: Mapping[str, Any],
    pack: Mapping[str, Any],
    *,
    explicit_evidence_ids: Sequence[str] = (),
    explicit_fact_claim_ids: Sequence[str] = (),
) -> dict[str, Any]:
    allowed_evidence = set(answer_b_core._allowed_evidence(pack))
    allowed_facts = set(_allowed_fact_claims_v3(pack))
    skeleton_evidence = {
        value for item in skeleton.get("answer_items") or [] for value in item.get("evidence_ids") or []
    }
    skeleton_facts = {
        value for item in skeleton.get("answer_items") or [] for value in item.get("fact_claim_ids") or []
    }
    requested_evidence = list(explicit_evidence_ids) + _DC_EVIDENCE_REF_PATTERN_V1.findall(str(answer or ""))
    requested_facts = list(explicit_fact_claim_ids) + _DC_FACT_REF_PATTERN_V1.findall(str(answer or ""))
    requested_evidence = list(dict.fromkeys(str(value) for value in requested_evidence))
    requested_facts = list(dict.fromkeys(str(value) for value in requested_facts))
    local_recovery = False
    if not requested_evidence and skeleton_evidence:
        requested_evidence = sorted(skeleton_evidence)
        local_recovery = True
    if not requested_facts and skeleton_facts:
        requested_facts = sorted(skeleton_facts)
        local_recovery = True
    invalid_evidence = [value for value in requested_evidence if value not in allowed_evidence]
    invalid_facts = [value for value in requested_facts if value not in allowed_facts]
    outside_skeleton_evidence = [value for value in requested_evidence if value in allowed_evidence and value not in skeleton_evidence]
    outside_skeleton_facts = [value for value in requested_facts if value in allowed_facts and value not in skeleton_facts]
    return {
        "used_evidence_ids": [value for value in requested_evidence if value in allowed_evidence and value in skeleton_evidence],
        "used_fact_claim_ids": [value for value in requested_facts if value in allowed_facts and value in skeleton_facts],
        "invalid_evidence_ids": invalid_evidence,
        "invalid_fact_claim_ids": invalid_facts,
        "outside_skeleton_evidence_ids": outside_skeleton_evidence,
        "outside_skeleton_fact_claim_ids": outside_skeleton_facts,
        "local_reference_recovery": local_recovery,
        "reference_consistency_passed": not (
            invalid_evidence or invalid_facts or outside_skeleton_evidence or outside_skeleton_facts
        ),
    }


def audit_numeric_support_dc_v1(answer: str, pack: Mapping[str, Any]) -> dict[str, Any]:
    augmented = copy.deepcopy(dict(pack))
    evidence = list(augmented.get("evidence") or [])
    fact_statements = [
        str(claim.get("statement") or "")
        for supplement in (augmented.get("fact_index") or {}).get("supplements") or []
        for claim in supplement.get("verified_claims") or []
    ]
    if fact_statements:
        evidence.append({"evidence_id": "FACT-INDEX", "content": " ".join(fact_statements)})
    augmented["evidence"] = evidence
    return audit_numeric_support_v2(answer, augmented)


def audit_forbidden_claims_dc_v1(answer: str, pack: Mapping[str, Any]) -> dict[str, Any]:
    normalized_answer = re.sub(r"\s+", "", str(answer or "")).lower()
    hits = []
    for supplement in (pack.get("fact_index") or {}).get("supplements") or []:
        fact_id = str(supplement.get("fact_index_id") or "")
        for claim in supplement.get("forbidden_claims") or []:
            normalized_claim = re.sub(r"\s+", "", str(claim or "")).lower()
            if len(normalized_claim) >= 8 and normalized_claim in normalized_answer:
                hits.append({"fact_index_id": fact_id, "forbidden_claim": str(claim)})
    return {"forbidden_claim_hits": hits, "forbidden_claim_check_passed": not hits}


def _dc_skeleton_prompt_v1(
    question: str,
    answer_needs: Sequence[Mapping[str, Any]],
    relation_constraint: Mapping[str, Any],
    augmented_pack: Mapping[str, Any],
) -> str:
    return f"""[사용자 질문]\n{_clean_text(question)}\n\n[Answer Needs]\n{_compact_json(answer_needs)}\n\n[관계 주장 안전조건]\n{_compact_json(relation_constraint)}\n\n[C안 Fact Index 보강 Evidence Pack]\n{_compact_json(augmented_pack)}\n\n[출력 JSON]\n{{\"core_answer\":\"핵심 결론\",\"answer_items\":[{{\"need_id\":\"N1\",\"topic\":\"항목\",\"status\":\"ANSWERED|PARTIAL|UNSUPPORTED\",\"claim\":\"근거 사실\",\"conditions\":[],\"details\":[],\"evidence_ids\":[\"E1\"],\"fact_claim_ids\":[\"FI-CAND-001:F1\"],\"missing_reason\":\"\"}}],\"uncertainties\":[],\"conflicts\":[]}}"""


def generate_dc_twocall_v1(
    question: str,
    augmented_pack: Mapping[str, Any],
) -> dict[str, Any]:
    total_started = time.perf_counter()
    need_started = time.perf_counter()
    answer_needs = extract_dc_answer_needs_v1(question, augmented_pack)
    relation_constraint = relation_constraint_v1(question, augmented_pack)
    need_ms = (time.perf_counter() - need_started) * 1000

    skeleton_prompt_started = time.perf_counter()
    skeleton_prompt = _dc_skeleton_prompt_v1(question, answer_needs, relation_constraint, augmented_pack)
    skeleton_prompt_ms = (time.perf_counter() - skeleton_prompt_started) * 1000
    raw_skeleton, usage1, skeleton_api_ms, trace1 = _call_answer_api_v1(
        system_prompt=DC_SKELETON_SYSTEM_PROMPT_V1,
        user_prompt=skeleton_prompt,
        max_tokens=DC_SKELETON_MAX_TOKENS_V1,
    )

    validation_started = time.perf_counter()
    skeleton = validate_dc_skeleton_v1(
        _extract_json_relaxed_dc_v1(raw_skeleton), answer_needs, augmented_pack
    )
    validation_ms = (time.perf_counter() - validation_started) * 1000
    selection_started = time.perf_counter()
    selected_pack = filter_augmented_pack_for_dc_v1(augmented_pack, skeleton)
    selection_ms = (time.perf_counter() - selection_started) * 1000

    final_prompt_started = time.perf_counter()
    final_prompt = f"""[사용자 질문]\n{_clean_text(question)}\n\n[Answer Needs]\n{_compact_json(answer_needs)}\n\n[관계 주장 안전조건]\n{_compact_json(relation_constraint)}\n\n[검증된 Answer Skeleton]\n{_compact_json(skeleton)}\n\n[Skeleton 선택 C안 Evidence Pack]\n{_compact_json(selected_pack)}\n\n모든 Answer Need를 반영한 최종 Markdown 답변만 작성하세요."""
    final_prompt_ms = (time.perf_counter() - final_prompt_started) * 1000
    raw_answer, usage2, final_api_ms, trace2 = _call_answer_api_v1(
        system_prompt=DC_FINAL_SYSTEM_PROMPT_V1,
        user_prompt=final_prompt,
        max_tokens=DC_FINAL_MAX_TOKENS_V1,
    )

    post_started = time.perf_counter()
    raw_answer = answer_b_core._strip_model_urls(str(raw_answer).strip())
    if not raw_answer:
        raise ValueError("D-C 2Call 최종 답변이 비어 있습니다.")
    reference_audit = audit_dc_final_references_v1(raw_answer, skeleton, augmented_pack)
    safe_answer, guard_applied = _relation_safe_answer_v1(raw_answer, relation_constraint)
    safe_answer = normalize_answer_markdown_v3(safe_answer)
    numeric_audit = audit_numeric_support_dc_v1(safe_answer, selected_pack)
    forbidden_audit = audit_forbidden_claims_dc_v1(safe_answer, selected_pack)
    post_ms = (time.perf_counter() - post_started) * 1000

    skeleton_trace = _trace_parts_v3(trace1)
    final_trace = _trace_parts_v3(trace2)
    total_ms = (time.perf_counter() - total_started) * 1000
    validation_passed = bool(
        skeleton.get("reference_validation_passed")
        and reference_audit["reference_consistency_passed"]
        and numeric_audit["numeric_support_passed"]
        and forbidden_audit["forbidden_claim_check_passed"]
        and not guard_applied
    )
    coverage = skeleton["coverage_status"]
    if guard_applied or not validation_passed:
        coverage = "PARTIAL" if coverage != "INSUFFICIENT" else coverage
    return {
        "system": "D-C 2Call",
        "answer": safe_answer,
        "answer_needs": answer_needs,
        "skeleton": skeleton,
        "need_coverage": skeleton["need_coverage"],
        "coverage_status": coverage,
        "strict_need_coverage_rate": skeleton["strict_need_coverage_rate"],
        "answerable_need_coverage_rate": skeleton["answerable_need_coverage_rate"],
        "selected_evidence_pack": selected_pack,
        "selected_evidence_count": len(selected_pack.get("evidence") or []),
        "used_evidence_ids": reference_audit["used_evidence_ids"],
        "used_fact_claim_ids": reference_audit["used_fact_claim_ids"],
        "reference_audit": reference_audit,
        "numeric_audit": numeric_audit,
        "forbidden_claim_audit": forbidden_audit,
        "validation_passed": validation_passed,
        "relation_constraint": relation_constraint,
        "relation_guard_applied": guard_applied,
        "latency_ms": total_ms,
        "skeleton_latency_ms": skeleton_api_ms,
        "final_latency_ms": final_api_ms,
        "usage": _merge_usage_v1(usage1, usage2),
        "api_calls": 2,
        "attempts": [
            {"stage": "dc_skeleton", "latency_ms": skeleton_api_ms, "trace": trace1},
            {"stage": "dc_final", "latency_ms": final_api_ms, "trace": trace2},
        ],
        "stage_latency_ms": {
            "need_extraction_ms": need_ms,
            "skeleton_prompt_build_ms": skeleton_prompt_ms,
            "skeleton_api_wall_ms": skeleton_trace["api_wall_ms"],
            "skeleton_pacing_wait_ms": skeleton_trace["pacing_wait_ms"],
            "skeleton_retry_wait_ms": skeleton_trace["retry_wait_ms"],
            "skeleton_estimated_service_ms": skeleton_trace["estimated_service_ms"],
            "skeleton_validation_ms": validation_ms,
            "evidence_selection_ms": selection_ms,
            "final_prompt_build_ms": final_prompt_ms,
            "final_api_wall_ms": final_trace["api_wall_ms"],
            "final_pacing_wait_ms": final_trace["pacing_wait_ms"],
            "final_retry_wait_ms": final_trace["retry_wait_ms"],
            "final_estimated_service_ms": final_trace["estimated_service_ms"],
            "postprocess_ms": post_ms,
            "answer_total_ms": total_ms,
        },
    }


def generate_dc_onecall_v1(
    question: str,
    augmented_pack: Mapping[str, Any],
) -> dict[str, Any]:
    total_started = time.perf_counter()
    need_started = time.perf_counter()
    answer_needs = extract_dc_answer_needs_v1(question, augmented_pack)
    relation_constraint = relation_constraint_v1(question, augmented_pack)
    need_ms = (time.perf_counter() - need_started) * 1000

    prompt_started = time.perf_counter()
    prompt = f"""[사용자 질문]\n{_clean_text(question)}\n\n[Answer Needs]\n{_compact_json(answer_needs)}\n\n[관계 주장 안전조건]\n{_compact_json(relation_constraint)}\n\n[C안 Fact Index 보강 Evidence Pack]\n{_compact_json(augmented_pack)}\n\n[반드시 지킬 출력 형식]\n<SKELETON_JSON>\n{{\"core_answer\":\"핵심 결론\",\"answer_items\":[{{\"need_id\":\"N1\",\"topic\":\"항목\",\"status\":\"ANSWERED|PARTIAL|UNSUPPORTED\",\"claim\":\"근거 사실\",\"conditions\":[],\"details\":[],\"evidence_ids\":[\"E1\"],\"fact_claim_ids\":[\"FI-CAND-001:F1\"],\"missing_reason\":\"\"}}],\"uncertainties\":[],\"conflicts\":[]}}\n</SKELETON_JSON>\n<FINAL_ANSWER>\n결론을 먼저 작성한 최종 Markdown 답변. 근거 문장에는 [E1] 또는 [FI-CAND-001:F1]을 표시합니다.\n</FINAL_ANSWER>"""
    prompt_ms = (time.perf_counter() - prompt_started) * 1000
    raw, usage, api_ms, trace = _call_answer_api_v1(
        system_prompt=DC_ONECALL_SYSTEM_PROMPT_V1,
        user_prompt=prompt,
        max_tokens=DC_ONECALL_MAX_TOKENS_V1,
    )

    validation_started = time.perf_counter()
    parsed = _parse_onecall_output_dc_v2(raw)
    raw_skeleton = parsed.get("answer_skeleton") or {}
    if not isinstance(raw_skeleton, Mapping):
        raise ValueError("D-C 1Call answer_skeleton이 JSON 객체가 아닙니다.")
    skeleton = validate_dc_skeleton_v1(raw_skeleton, answer_needs, augmented_pack)
    # JSON answer 문자열의 Markdown 줄바꿈을 보존합니다. _clean()은 모든 공백을
    # 한 칸으로 합치므로 번호 목록과 글머리표가 한 문단이 될 수 있습니다.
    raw_answer = answer_b_core._strip_model_urls(str(parsed.get("answer") or "").strip())
    if not raw_answer:
        raise ValueError("D-C 1Call 최종 답변이 비어 있습니다.")
    reference_audit = audit_dc_final_references_v1(
        raw_answer,
        skeleton,
        augmented_pack,
        explicit_evidence_ids=answer_b_core._clean_list(parsed.get("used_evidence_ids")),
        explicit_fact_claim_ids=_clean_fact_claim_keys_v3(parsed.get("used_fact_claim_ids")),
    )
    selected_pack = filter_augmented_pack_for_dc_v1(augmented_pack, skeleton)
    validation_ms = (time.perf_counter() - validation_started) * 1000

    post_started = time.perf_counter()
    safe_answer, guard_applied = _relation_safe_answer_v1(raw_answer, relation_constraint)
    safe_answer = normalize_answer_markdown_v3(safe_answer)
    numeric_audit = audit_numeric_support_dc_v1(safe_answer, selected_pack)
    forbidden_audit = audit_forbidden_claims_dc_v1(safe_answer, selected_pack)
    post_ms = (time.perf_counter() - post_started) * 1000
    trace_parts = _trace_parts_v3(trace)
    total_ms = (time.perf_counter() - total_started) * 1000
    validation_passed = bool(
        skeleton.get("reference_validation_passed")
        and reference_audit["reference_consistency_passed"]
        and numeric_audit["numeric_support_passed"]
        and forbidden_audit["forbidden_claim_check_passed"]
        and not guard_applied
    )
    requested_coverage = str(parsed.get("coverage_status") or skeleton["coverage_status"]).upper()
    coverage = skeleton["coverage_status"]
    if requested_coverage == "INSUFFICIENT" and coverage == "SUFFICIENT":
        coverage = "PARTIAL"
    if guard_applied or not validation_passed:
        coverage = "PARTIAL" if coverage != "INSUFFICIENT" else coverage
    return {
        "system": "D-C 1Call",
        "answer": safe_answer,
        "answer_needs": answer_needs,
        "skeleton": skeleton,
        "need_coverage": skeleton["need_coverage"],
        "coverage_status": coverage,
        "strict_need_coverage_rate": skeleton["strict_need_coverage_rate"],
        "answerable_need_coverage_rate": skeleton["answerable_need_coverage_rate"],
        "selected_evidence_pack": selected_pack,
        "selected_evidence_count": len(selected_pack.get("evidence") or []),
        "used_evidence_ids": reference_audit["used_evidence_ids"],
        "used_fact_claim_ids": reference_audit["used_fact_claim_ids"],
        "reference_audit": reference_audit,
        "numeric_audit": numeric_audit,
        "forbidden_claim_audit": forbidden_audit,
        "validation_passed": validation_passed,
        "output_contract": parsed.get("output_contract"),
        "relation_constraint": relation_constraint,
        "relation_guard_applied": guard_applied,
        "latency_ms": total_ms,
        "skeleton_latency_ms": api_ms,
        "final_latency_ms": 0.0,
        "usage": usage,
        "api_calls": 1,
        "attempts": [{"stage": "dc_skeleton_and_final", "latency_ms": api_ms, "trace": trace}],
        "stage_latency_ms": {
            "need_extraction_ms": need_ms,
            "combined_prompt_build_ms": prompt_ms,
            "combined_api_wall_ms": trace_parts["api_wall_ms"],
            "combined_pacing_wait_ms": trace_parts["pacing_wait_ms"],
            "combined_retry_wait_ms": trace_parts["retry_wait_ms"],
            "combined_estimated_service_ms": trace_parts["estimated_service_ms"],
            "combined_json_skeleton_validation_ms": validation_ms,
            "postprocess_ms": post_ms,
            "answer_total_ms": total_ms,
        },
    }

# ==== overlay: dc-comparison-execution ====

# 공통 검색·Fact Index 1회 캐시와 D-C 개별 실행기
def new_dc_controller_state_v1() -> dict[str, Any]:
    state = new_bcd_controller_state_c1()
    state["events"] = []
    return state


def _prepare_dc_common_v1(
    question: str,
    holder: dict[str, Any],
) -> tuple[dict[str, Any], bool, float, float]:
    common, cache_hit, common_this_click_ms = _prepare_or_reuse_common_v3(question, holder)
    fact_this_click_ms = 0.0
    if common.get("route") == "RETRIEVE" and "dc_augmented_pack" not in common:
        started = time.perf_counter()
        matched_records, fact_audit = match_fact_index_c1(common)
        augmented_pack = build_fact_augmented_pack_c1(common["evidence_pack"], matched_records)
        fact_this_click_ms = (time.perf_counter() - started) * 1000
        common["dc_matched_fact_records"] = matched_records
        common["dc_fact_audit"] = fact_audit
        common["dc_augmented_pack"] = augmented_pack
        common["dc_augmented_pack_sha256"] = _stable_json_hash_c1(augmented_pack)
        common.setdefault("latency_ms", {})["Fact Index 보강"] = fact_this_click_ms
        common["latency_ms"]["공통 준비 전체"] = (
            float(common["latency_ms"].get("공통 준비 전체") or 0) + fact_this_click_ms
        )
        common_this_click_ms += fact_this_click_ms
    return common, cache_hit, common_this_click_ms, fact_this_click_ms


def _dc_answer_cache_key_v1(
    variant: str,
    common: Mapping[str, Any],
) -> str:
    return _stable_json_hash_c1({
        "variant": variant,
        "resolved_question": common.get("resolved_question"),
        "augmented_pack_sha256": common.get("dc_augmented_pack_sha256"),
        "prompt_version": DC_PROMPT_VERSION_V1,
    })


def _official_sources_dc_v1(common: Mapping[str, Any]) -> list[dict[str, str]]:
    sources = []
    seen = set()
    for row in (common.get("evidence_pack") or {}).get("sources") or []:
        url = str(row.get("source_url") or "")
        if url and url not in seen:
            seen.add(url)
            sources.append({"title": str(row.get("title") or "공식 출처"), "url": url})
    for record in common.get("dc_matched_fact_records") or []:
        title = str(record.get("document_title") or record.get("fact_index_id") or "Fact Index 공식 근거")
        for url in record.get("source_urls") or []:
            url = str(url)
            if url and url not in seen:
                seen.add(url)
                sources.append({"title": title, "url": url})
    return sources


def execute_dc_variant_v1(
    variant: str,
    question: str,
    holder: dict[str, Any],
    *,
    force_answer_regeneration: bool = False,
) -> dict[str, Any]:
    variant = str(variant).upper()
    if variant not in {"DC_2CALL", "DC_1CALL"}:
        raise ValueError(f"지원하지 않는 D-C 답변안: {variant}")
    click_started = time.perf_counter()
    gate_start = len(HCX_SHARED_GATE_V3.history)
    common, common_cache_hit, common_this_click_ms, fact_this_click_ms = _prepare_dc_common_v1(
        question, holder
    )
    if common.get("route") != "RETRIEVE":
        return {
            "variant": variant,
            "route": common.get("route"),
            "route_message": common.get("route_message"),
            "common": common,
            "common_cache_hit": common_cache_hit,
            "answer_cache_hit": False,
            "latency": {
                "common_this_click_ms": common_this_click_ms,
                "fact_index_match_ms": fact_this_click_ms,
                "answer_ms": 0.0,
                "click_wall_ms": (time.perf_counter() - click_started) * 1000,
            },
            "api_trace": _trace_summary_since_v3(gate_start),
        }

    cache_key = _dc_answer_cache_key_v1(variant, common)
    cached = holder["answer_cache"].get(cache_key)
    if cached is not None and not force_answer_regeneration:
        payload = copy.deepcopy(cached)
        answer_cache_hit = True
        answer_ms = 0.0
    else:
        answer_cache_hit = False
        answer_started = time.perf_counter()
        if variant == "DC_2CALL":
            payload = generate_dc_twocall_v1(
                common["resolved_question"], common["dc_augmented_pack"]
            )
        else:
            payload = generate_dc_onecall_v1(
                common["resolved_question"], common["dc_augmented_pack"]
            )
        answer_ms = (time.perf_counter() - answer_started) * 1000
        holder["answer_cache"][cache_key] = copy.deepcopy(payload)

    if not holder.get("committed"):
        holder["conversation"].setdefault("turns", []).extend([
            {"role": "user", "content": _clean_text(question)},
            {"role": "assistant", "content": normalize_answer_markdown_v3(payload.get("answer"))},
        ])
        holder["committed"] = True
        holder["committed_variant"] = variant

    trace = _trace_summary_since_v3(gate_start)
    usage = _variant_usage_v3(payload)
    stage = payload.get("stage_latency_ms") or {}
    result = {
        "variant": variant,
        "route": "RETRIEVE",
        "common": common,
        "payload": payload,
        "augmented_pack": common["dc_augmented_pack"],
        "matched_fact_records": common.get("dc_matched_fact_records") or [],
        "fact_audit": common.get("dc_fact_audit") or [],
        "common_cache_hit": common_cache_hit,
        "answer_cache_hit": answer_cache_hit,
        "committed_variant": holder.get("committed_variant"),
        "official_sources": _official_sources_dc_v1(common),
        "action_links": action_links_for_streamlit_v1(common.get("action_links") or []),
        "latency": {
            "stored_common_pipeline_ms": float((common.get("latency_ms") or {}).get("공통 준비 전체") or 0),
            "common_this_click_ms": common_this_click_ms,
            "fact_index_match_ms": fact_this_click_ms,
            "answer_ms": answer_ms,
            "click_wall_ms": (time.perf_counter() - click_started) * 1000,
        },
        "api_trace": trace,
        "usage": usage,
        "circuit": hcx_circuit_status_c1(),
    }
    event = {
        "question": _clean_text(question),
        "variant": variant,
        "answer_api_calls": int(payload.get("api_calls") or 0),
        "common_cache_hit": common_cache_hit,
        "answer_cache_hit": answer_cache_hit,
        "fact_index_count": len(result["matched_fact_records"]),
        "fact_index_ids": " | ".join(str(row.get("fact_index_id") or "") for row in result["matched_fact_records"]),
        "stored_common_pipeline_ms": result["latency"]["stored_common_pipeline_ms"],
        "common_this_click_ms": common_this_click_ms,
        "fact_index_match_ms": fact_this_click_ms,
        "answer_total_ms": float(stage.get("answer_total_ms") or 0),
        "skeleton_api_wall_ms": float(stage.get("skeleton_api_wall_ms") or 0),
        "final_api_wall_ms": float(stage.get("final_api_wall_ms") or 0),
        "combined_api_wall_ms": float(stage.get("combined_api_wall_ms") or 0),
        "answer_pacing_wait_ms": sum(float(value) for key, value in stage.items() if key.endswith("pacing_wait_ms")),
        "answer_retry_wait_ms": sum(float(value) for key, value in stage.items() if key.endswith("retry_wait_ms")),
        "answer_estimated_service_ms": sum(float(value) for key, value in stage.items() if key.endswith("estimated_service_ms")),
        "click_wall_ms": result["latency"]["click_wall_ms"],
        **usage,
        "coverage_status": payload.get("coverage_status"),
        "strict_need_coverage_rate": float(payload.get("strict_need_coverage_rate") or 0),
        "answerable_need_coverage_rate": float(payload.get("answerable_need_coverage_rate") or 0),
        "reference_consistency_passed": bool((payload.get("reference_audit") or {}).get("reference_consistency_passed")),
        "numeric_support_passed": bool((payload.get("numeric_audit") or {}).get("numeric_support_passed")),
        "forbidden_claim_check_passed": bool((payload.get("forbidden_claim_audit") or {}).get("forbidden_claim_check_passed")),
        "relation_guard_applied": bool(payload.get("relation_guard_applied")),
        "validation_passed": bool(payload.get("validation_passed")),
        "output_contract": str(payload.get("output_contract") or "TWO_CALL_SEPARATE_OUTPUT"),
        "answer_chars": len(str(payload.get("answer") or "")),
    }
    holder["events"].append(event)
    result["event"] = event
    return result

# ==== overlay: dc-cross-specialized ====

# D안 교차업무 전용 프롬프트·검증·호출 게이트
DC_CROSS_PROMPT_VERSION_V1 = "dc-cross-business-specialized-tagged-v2-p0-rule-cross-how"
DC_RESPONSE_MODES_V1 = {"SEPARATE", "COMPARE", "RELATION", "SEQUENCE"}


def classify_dc_response_mode_v1(question: str) -> str:
    text = _clean_text(question)
    how_usage = classify_how_usage_v1(text)
    if re.search(r"먼저|다음|그\s*후|이후|뒤에|하고\s*나서|한\s*뒤|순서", text):
        return "SEQUENCE"
    if how_usage == "COMPARE_HOW" or re.search(r"차이|다른가|비교|같은\s*(?:건|것)|구분", text):
        return "COMPARE"
    if re.search(r"동시에|같이|함께|한\s*번에|둘\s*다|모두\s*신청|연계|병행|받을\s*수\s*있", text):
        return "RELATION"
    return "SEPARATE"


def order_businesses_by_question_p0_v1(
    question: str,
    businesses: Sequence[str],
) -> list[str]:
    """Keep business sections in the same order they appear in the question."""

    positioned = []
    for original_index, business in enumerate(
        dict.fromkeys(str(value) for value in businesses if str(value))
    ):
        terms = list((light_router.BUSINESS_KEYWORDS or {}).get(business) or [])
        positions = [
            question.find(term)
            for term in terms
            if term and question.find(term) >= 0
        ]
        positioned.append((
            min(positions) if positions else len(question) + original_index,
            original_index,
            business,
        ))
    return [business for _position, _index, business in sorted(positioned)]


def _business_local_segments_p0_v1(
    question: str,
    businesses: Sequence[str],
) -> dict[str, str]:
    anchors = []
    for business in businesses:
        terms = list((light_router.BUSINESS_KEYWORDS or {}).get(business) or [])
        positions = [
            (question.find(term), term)
            for term in terms
            if term and question.find(term) >= 0
        ]
        if positions:
            position, term = min(positions, key=lambda row: row[0])
            anchors.append((position, business, term))
    anchors.sort(key=lambda row: row[0])
    segments = {}
    for index, (start, business, _term) in enumerate(anchors):
        end = anchors[index + 1][0] if index + 1 < len(anchors) else len(question)
        segments[business] = question[start:end]
    return segments


def _business_need_topic_p0_v1(local_text: str, full_question: str) -> str:
    local = str(local_text or "")
    full = str(full_question or "")
    if re.search(r"포상금|회수기여|신고.*금액", local):
        return "신고 포상금 지급 기준과 금액"
    if re.search(r"조회|확인", local):
        return "조회 방법과 확인 절차"
    if re.search(r"서류|구비|준비물|양식", local):
        return "필요 서류와 준비 절차"
    if re.search(r"지급\s*조건", local):
        return "지급 조건"
    if re.search(r"대상|자격|조건", local) or re.search(r"신청\s*대상|자격", full):
        return "신청 대상과 자격 조건"
    if re.search(r"한도|금액|얼마", local):
        return "금액과 한도"
    if re.search(r"기간|기한|언제|얼마나\s*걸", local):
        return "신청 기한과 처리 기간"
    if re.search(r"신청|접수|청구|신고|하려면", local):
        return "신청 대상 조건과 신청 절차"
    if re.search(r"동시에|같이|함께|한\s*번에|병행", full):
        return "신청 대상 조건과 신청 절차"
    if HOW_COMPARE_PATTERN_V1.search(full):
        return "제도 개요와 주요 대상 및 이용 목적"
    return "제도 개요와 이용 절차"


def build_p0_cross_business_subqueries_v1(
    question: str,
    businesses: Sequence[str],
) -> list[str]:
    ordered = list(dict.fromkeys(str(value) for value in businesses if str(value)))
    if not 2 <= len(ordered) <= NEED_BATCH_MAX_BUSINESSES_V5:
        return []
    segments = _business_local_segments_p0_v1(question, ordered)
    subqueries = [
        f"{business} {_business_need_topic_p0_v1(segments.get(business, ''), question)}"
        for business in ordered
    ]
    detected = [list(light_router.find_businesses(query) or []) for query in subqueries]
    if any(values != [business] for values, business in zip(detected, ordered)):
        return []
    return subqueries


def is_cross_business_dc_v1(common: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    analysis = dict(common.get("analysis") or {})
    pack = dict(common.get("evidence_pack") or {})
    needs = [dict(row) for row in pack.get("needs") or []]
    strategy = str(pack.get("retrieval_strategy") or "")
    quota_audit = dict(common.get("evidence_quota_audit") or audit_need_evidence_pack_v5(pack))
    need_businesses = [_clean_text(row.get("business_function")) for row in needs]
    analysis_businesses = [_clean_text(value) for value in analysis.get("businesses") or [] if _clean_text(value)]
    gate_question = str(common.get("resolved_question") or common.get("question") or "")
    question_businesses = order_businesses_by_question_p0_v1(
        gate_question,
        [_clean_text(value) for value in light_router.find_businesses(gate_question) or [] if _clean_text(value)],
    )
    businesses = list(dict.fromkeys(analysis_businesses + question_businesses + need_businesses))
    raw_analysis = dict(analysis.get("raw_analysis") or {})
    cross_business_candidate = bool(
        analysis.get("p0_cross_preserved")
        or raw_analysis.get("p0_cross_preserved")
        or raw_analysis.get("cross_business_candidate")
        or len(question_businesses) >= 2
        or len(set(need_businesses)) >= 2
    )
    structure_valid = bool(
        2 <= len(businesses) <= NEED_BATCH_MAX_BUSINESSES_V5
        and len(needs) == len(businesses)
        and all(need_businesses)
        and len(set(need_businesses)) == len(need_businesses)
        and set(need_businesses) == set(businesses)
    )
    passed = bool(
        common.get("route") == "RETRIEVE"
        and cross_business_candidate
        and structure_valid
        and strategy == "NEED_BATCH_RERANK_V5"
        and quota_audit.get("passed")
    )
    return passed, {
        "passed": passed,
        "cross_business_candidate": cross_business_candidate,
        "p0_cross_preserved": bool(analysis.get("p0_cross_preserved") or raw_analysis.get("p0_cross_preserved")),
        "structure_valid": structure_valid,
        "business_count": len(businesses),
        "businesses": businesses,
        "analysis_businesses": analysis_businesses,
        "question_businesses": question_businesses,
        "need_businesses": need_businesses,
        "need_count": len(needs),
        "need_ids": [str(row.get("need_id") or "") for row in needs],
        "retrieval_strategy": strategy,
        "required_evidence_per_business": NEED_BATCH_REQUIRED_TOP_K_V5,
        **need_score_gate_metadata_v5(),
        "maximum_cross_business_count": NEED_BATCH_MAX_BUSINESSES_V5,
        "evidence_quota_audit": quota_audit,
        "query_type": str(analysis.get("query_type") or ""),
        "decomposition_accepted": bool(analysis.get("decomposition_or_rewrite_accepted")),
    }


DC_CROSS_SKELETON_RULES_V1 = """
[교차업무 Answer Skeleton 규칙]
1. response_mode은 제공된 expected_response_mode과 동일하게 작성하세요.
2. 모든 업무별 need_id에 정확히 하나의 answer_item을 작성하세요.
3. 각 item에 business_function을 명시하고 그 Need에 연결된 Evidence를 우선 사용하세요.
4. N1의 대상·조건·금액·기간·서류·절차를 N2에 적용하지 마세요.
5. 특정 Need의 근거가 부족하면 다른 Need의 근거로 채우지 말고 PARTIAL 또는 UNSUPPORTED로 표시하세요.
6. SEPARATE는 각 업무를 독립적으로 안내하고 요구하지 않은 비교를 만들지 마세요.
7. COMPARE는 각 업무의 독립 설명을 먼저 확보하고 근거가 있는 차이만 cross_need_relation에 작성하세요.
8. RELATION은 두 업무의 개별 자격만으로 동시·병행·인과 관계를 추론하지 마세요.
9. SEQUENCE는 사용자가 제시한 업무 순서를 보존하고 각 단계의 업무명을 명시하세요.
10. cross_need_relation의 supported=true는 관계를 직접 뒷받침하는 Evidence 또는 Fact Claim이 있을 때만 허용합니다.
""".strip()

DC_CROSS_FINAL_RULES_V1 = """
[교차업무 최종답변 규칙]
- 모든 Need를 업무명 소제목으로 구분하세요.
- 수치·기간·대상·조건·서류 앞에는 적용 업무를 분명히 표시하세요.
- SEPARATE: 업무별 답변만 제공하고 불필요한 공통점·차이점을 만들지 마세요.
- COMPARE: 각 업무 설명 뒤에 질문과 관련된 주요 차이를 정리하세요.
- RELATION: 관계 판단을 결론에서 먼저 말하고, 직접 근거가 없으면 확인되지 않는다고 답하세요.
- SEQUENCE: 사용자가 요청한 순서대로 단계와 Action을 구분하세요.
- 한 업무의 Evidence를 모든 업무의 공통 근거처럼 사용하지 마세요.
""".strip()

DC_SKELETON_SYSTEM_PROMPT_V1 = (
    D2_SKELETON_SYSTEM_PROMPT
    + "\n\n" + DC_FACT_SAFETY_RULES_V1
    + "\n\n" + DC_CROSS_SKELETON_RULES_V1
)
DC_FINAL_SYSTEM_PROMPT_V1 = (
    D_STRUCTURED_FINAL_SYSTEM_PROMPT_V3
    + "\n\n" + DC_FACT_SAFETY_RULES_V1
    + "\n\n" + DC_CROSS_FINAL_RULES_V1
)
DC_ONECALL_SYSTEM_PROMPT_V1 = (
    "당신은 예금보험공사 교차업무 복합질의 전용 Answer Skeleton 및 최종답변 생성기입니다.\n"
    "반드시 SKELETON_JSON과 FINAL_ANSWER 두 태그만 출력하세요.\n"
    "SKELETON_JSON 내부만 JSON이며 FINAL_ANSWER는 Markdown입니다.\n\n"
    + DC_FACT_SAFETY_RULES_V1
    + "\n\n" + DC_CROSS_SKELETON_RULES_V1
    + "\n\n" + DC_CROSS_FINAL_RULES_V1
    + "\n\n" + ACTION_LINK_PROMPT_RULE_V1
)


_VALIDATE_DC_SKELETON_GENERAL_V1 = globals().get("_VALIDATE_DC_SKELETON_GENERAL_V1") or validate_dc_skeleton_v1


def _fact_claim_business_map_dc_v1(pack: Mapping[str, Any]) -> dict[str, str]:
    output = {}
    for supplement in (pack.get("fact_index") or {}).get("supplements") or []:
        fact_id = str(supplement.get("fact_index_id") or "")
        business = str(supplement.get("business_function") or "")
        for claim in supplement.get("verified_claims") or []:
            output[f"{fact_id}:{claim.get('claim_id')}"] = business
    return output


def validate_dc_skeleton_v1(
    raw: Mapping[str, Any],
    answer_needs: Sequence[Mapping[str, Any]],
    pack: Mapping[str, Any],
) -> dict[str, Any]:
    result = _VALIDATE_DC_SKELETON_GENERAL_V1(raw, answer_needs, pack)
    need_map = {str(row.get("need_id") or ""): dict(row) for row in answer_needs}
    fact_business = _fact_claim_business_map_dc_v1(pack)
    evidence_violations = []
    fact_violations = []
    coverage_rows = []
    for item in result.get("answer_items") or []:
        need = need_map.get(str(item.get("need_id") or ""), {})
        business = str(need.get("label") or "")
        allowed_need_evidence = set(need.get("retrieval_evidence_ids") or [])
        original_evidence = list(item.get("evidence_ids") or [])
        if allowed_need_evidence:
            outside = [value for value in original_evidence if value not in allowed_need_evidence]
            evidence_violations.extend({"need_id": item["need_id"], "evidence_id": value} for value in outside)
            item["evidence_ids"] = [value for value in original_evidence if value in allowed_need_evidence]
        original_facts = list(item.get("fact_claim_ids") or [])
        outside_facts = [
            value for value in original_facts
            if fact_business.get(value) and business and fact_business.get(value) != business
        ]
        fact_violations.extend({"need_id": item["need_id"], "fact_claim_id": value} for value in outside_facts)
        item["fact_claim_ids"] = [value for value in original_facts if value not in outside_facts]
        item["business_function"] = business
        if item["status"] == "ANSWERED" and not item["evidence_ids"] and not item["fact_claim_ids"]:
            item["status"] = "PARTIAL" if item.get("claim") else "UNSUPPORTED"
        coverage_rows.append({
            "need_id": item["need_id"],
            "need_type": item.get("need_type"),
            "label": business,
            "business_function": business,
            "status": item["status"],
            "evidence_ids": item["evidence_ids"],
            "fact_claim_ids": item["fact_claim_ids"],
            "missing_reason": item.get("missing_reason"),
        })
    relation_raw = raw.get("cross_need_relation") if isinstance(raw.get("cross_need_relation"), Mapping) else {}
    allowed_evidence = set(answer_b_core._allowed_evidence(pack))
    allowed_facts = set(_allowed_fact_claims_v3(pack))
    relation_evidence = [
        value for value in answer_b_core._clean_list(relation_raw.get("evidence_ids"))
        if value in allowed_evidence
    ]
    relation_facts = [
        value for value in _clean_fact_claim_keys_v3(relation_raw.get("fact_claim_ids"))
        if value in allowed_facts
    ]
    relation_supported = bool(relation_raw.get("supported") and (relation_evidence or relation_facts))
    response_mode = str(raw.get("response_mode") or "SEPARATE").upper()
    if response_mode not in DC_RESPONSE_MODES_V1:
        response_mode = "SEPARATE"
    program = calculate_program_coverage_v2(coverage_rows)
    scope_passed = not evidence_violations and not fact_violations
    result.update({
        "response_mode": response_mode,
        "answer_items": result["answer_items"],
        "need_coverage": coverage_rows,
        "cross_need_relation": {
            "requested": bool(relation_raw.get("requested")),
            "supported": relation_supported,
            "claim": answer_b_core._clean(relation_raw.get("claim")),
            "evidence_ids": list(dict.fromkeys(relation_evidence)),
            "fact_claim_ids": list(dict.fromkeys(relation_facts)),
            "missing_reason": answer_b_core._clean(relation_raw.get("missing_reason")),
        },
        "cross_need_evidence_violations": evidence_violations,
        "cross_need_fact_violations": fact_violations,
        "cross_need_scope_passed": scope_passed,
        "reference_validation_passed": bool(result.get("reference_validation_passed") and scope_passed),
        **program,
    })
    return result


def _dc_skeleton_prompt_v1(
    question: str,
    answer_needs: Sequence[Mapping[str, Any]],
    relation_constraint: Mapping[str, Any],
    augmented_pack: Mapping[str, Any],
) -> str:
    expected_mode = classify_dc_response_mode_v1(question)
    return f"""[사용자 질문]\n{_clean_text(question)}\n\n[expected_response_mode]\n{expected_mode}\n\n[업무별 Answer Needs]\n{_compact_json(answer_needs)}\n\n[관계 주장 안전조건]\n{_compact_json(relation_constraint)}\n\n[C안 Fact Index 보강 Evidence Pack]\n{_compact_json(augmented_pack)}\n\n[출력 JSON]\n{{\"response_mode\":\"{expected_mode}\",\"core_answer\":\"핵심 결론\",\"answer_items\":[{{\"need_id\":\"N1\",\"business_function\":\"업무명\",\"topic\":\"항목\",\"status\":\"ANSWERED|PARTIAL|UNSUPPORTED\",\"claim\":\"근거 사실\",\"conditions\":[],\"details\":[],\"evidence_ids\":[\"E1\"],\"fact_claim_ids\":[\"FI-CAND-001:F1\"],\"missing_reason\":\"\"}}],\"cross_need_relation\":{{\"requested\":false,\"supported\":false,\"claim\":\"\",\"evidence_ids\":[],\"fact_claim_ids\":[],\"missing_reason\":\"\"}},\"uncertainties\":[],\"conflicts\":[]}}"""


_GENERATE_DC_TWOCALL_GENERAL_V1 = globals().get("_GENERATE_DC_TWOCALL_GENERAL_V1") or generate_dc_twocall_v1


def generate_dc_twocall_v1(
    question: str,
    augmented_pack: Mapping[str, Any],
) -> dict[str, Any]:
    if len(augmented_pack.get("needs") or []) < 2:
        raise RuntimeError("D_CROSS_ONLY_POLICY: 단일·동일업무 질문은 C안 대상입니다.")
    result = _GENERATE_DC_TWOCALL_GENERAL_V1(question, augmented_pack)
    expected = classify_dc_response_mode_v1(question)
    actual = str((result.get("skeleton") or {}).get("response_mode") or "SEPARATE")
    mode_passed = actual == expected
    result.update({
        "response_mode_expected": expected,
        "response_mode_actual": actual,
        "response_mode_validation_passed": mode_passed,
        "cross_business_prompt": True,
        "validation_passed": bool(result.get("validation_passed") and mode_passed),
    })
    if not mode_passed and result.get("coverage_status") == "SUFFICIENT":
        result["coverage_status"] = "PARTIAL"
    return result


def generate_dc_onecall_v1(
    question: str,
    augmented_pack: Mapping[str, Any],
) -> dict[str, Any]:
    if len(augmented_pack.get("needs") or []) < 2:
        raise RuntimeError("D_CROSS_ONLY_POLICY: 단일·동일업무 질문은 C안 대상입니다.")
    total_started = time.perf_counter()
    need_started = time.perf_counter()
    answer_needs = extract_dc_answer_needs_v1(question, augmented_pack)
    relation_constraint = relation_constraint_v1(question, augmented_pack)
    expected_mode = classify_dc_response_mode_v1(question)
    need_ms = (time.perf_counter() - need_started) * 1000
    prompt_started = time.perf_counter()
    prompt = f"""[사용자 질문]\n{_clean_text(question)}\n\n[expected_response_mode]\n{expected_mode}\n\n[업무별 Answer Needs]\n{_compact_json(answer_needs)}\n\n[관계 주장 안전조건]\n{_compact_json(relation_constraint)}\n\n[C안 Fact Index 보강 Evidence Pack]\n{_compact_json(augmented_pack)}\n\n[반드시 지킬 출력 형식]\n<SKELETON_JSON>\n{{\"response_mode\":\"{expected_mode}\",\"core_answer\":\"핵심 결론\",\"answer_items\":[{{\"need_id\":\"N1\",\"business_function\":\"업무명\",\"topic\":\"항목\",\"status\":\"ANSWERED|PARTIAL|UNSUPPORTED\",\"claim\":\"근거 사실\",\"conditions\":[],\"details\":[],\"evidence_ids\":[\"E1\"],\"fact_claim_ids\":[\"FI-CAND-001:F1\"],\"missing_reason\":\"\"}}],\"cross_need_relation\":{{\"requested\":false,\"supported\":false,\"claim\":\"\",\"evidence_ids\":[],\"fact_claim_ids\":[],\"missing_reason\":\"\"}},\"uncertainties\":[],\"conflicts\":[]}}\n</SKELETON_JSON>\n<FINAL_ANSWER>\nresponse_mode에 맞춰 모든 업무 Need를 구분한 Markdown 답변\n</FINAL_ANSWER>"""
    prompt_ms = (time.perf_counter() - prompt_started) * 1000
    raw, usage, api_ms, trace = _call_answer_api_v1(
        system_prompt=DC_ONECALL_SYSTEM_PROMPT_V1,
        user_prompt=prompt,
        max_tokens=DC_ONECALL_MAX_TOKENS_V1,
    )
    validation_started = time.perf_counter()
    parsed = _parse_onecall_output_dc_v2(raw)
    raw_skeleton = parsed.get("answer_skeleton") or {}
    if not isinstance(raw_skeleton, Mapping):
        raise ValueError("D-C 교차업무 1Call answer_skeleton이 JSON 객체가 아닙니다.")
    skeleton = validate_dc_skeleton_v1(raw_skeleton, answer_needs, augmented_pack)
    raw_answer = answer_b_core._strip_model_urls(str(parsed.get("answer") or "").strip())
    if not raw_answer:
        raise ValueError("D-C 교차업무 1Call 최종답변이 비어 있습니다.")
    reference_audit = audit_dc_final_references_v1(
        raw_answer,
        skeleton,
        augmented_pack,
        explicit_evidence_ids=answer_b_core._clean_list(parsed.get("used_evidence_ids")),
        explicit_fact_claim_ids=_clean_fact_claim_keys_v3(parsed.get("used_fact_claim_ids")),
    )
    selected_pack = filter_augmented_pack_for_dc_v1(augmented_pack, skeleton)
    validation_ms = (time.perf_counter() - validation_started) * 1000
    post_started = time.perf_counter()
    safe_answer, guard_applied = _relation_safe_answer_v1(raw_answer, relation_constraint)
    safe_answer = normalize_answer_markdown_v3(safe_answer)
    numeric_audit = audit_numeric_support_dc_v1(safe_answer, selected_pack)
    forbidden_audit = audit_forbidden_claims_dc_v1(safe_answer, selected_pack)
    post_ms = (time.perf_counter() - post_started) * 1000
    trace_parts = _trace_parts_v3(trace)
    total_ms = (time.perf_counter() - total_started) * 1000
    actual_mode = str(skeleton.get("response_mode") or "SEPARATE")
    mode_passed = actual_mode == expected_mode
    validation_passed = bool(
        skeleton.get("reference_validation_passed")
        and skeleton.get("cross_need_scope_passed")
        and reference_audit["reference_consistency_passed"]
        and numeric_audit["numeric_support_passed"]
        and forbidden_audit["forbidden_claim_check_passed"]
        and not guard_applied
        and mode_passed
    )
    coverage = skeleton["coverage_status"]
    if not validation_passed and coverage == "SUFFICIENT":
        coverage = "PARTIAL"
    return {
        "system": "D-C 1Call · 교차업무 전용",
        "answer": safe_answer,
        "answer_needs": answer_needs,
        "skeleton": skeleton,
        "need_coverage": skeleton["need_coverage"],
        "coverage_status": coverage,
        "strict_need_coverage_rate": skeleton["strict_need_coverage_rate"],
        "answerable_need_coverage_rate": skeleton["answerable_need_coverage_rate"],
        "selected_evidence_pack": selected_pack,
        "selected_evidence_count": len(selected_pack.get("evidence") or []),
        "used_evidence_ids": reference_audit["used_evidence_ids"],
        "used_fact_claim_ids": reference_audit["used_fact_claim_ids"],
        "reference_audit": reference_audit,
        "numeric_audit": numeric_audit,
        "forbidden_claim_audit": forbidden_audit,
        "validation_passed": validation_passed,
        "relation_constraint": relation_constraint,
        "relation_guard_applied": guard_applied,
        "response_mode_expected": expected_mode,
        "response_mode_actual": actual_mode,
        "response_mode_validation_passed": mode_passed,
        "cross_business_prompt": True,
        "output_contract": parsed.get("output_contract"),
        "latency_ms": total_ms,
        "skeleton_latency_ms": api_ms,
        "final_latency_ms": 0.0,
        "usage": usage,
        "api_calls": 1,
        "attempts": [{"stage": "dc_cross_skeleton_and_final", "latency_ms": api_ms, "trace": trace}],
        "stage_latency_ms": {
            "need_extraction_ms": need_ms,
            "combined_prompt_build_ms": prompt_ms,
            "combined_api_wall_ms": trace_parts["api_wall_ms"],
            "combined_pacing_wait_ms": trace_parts["pacing_wait_ms"],
            "combined_retry_wait_ms": trace_parts["retry_wait_ms"],
            "combined_estimated_service_ms": trace_parts["estimated_service_ms"],
            "combined_json_skeleton_validation_ms": validation_ms,
            "postprocess_ms": post_ms,
            "answer_total_ms": total_ms,
        },
    }


_EXECUTE_DC_GENERAL_V1 = globals().get("_EXECUTE_DC_GENERAL_V1") or execute_dc_variant_v1


def execute_dc_variant_v1(
    variant: str,
    question: str,
    holder: dict[str, Any],
    *,
    force_answer_regeneration: bool = False,
) -> dict[str, Any]:
    total_started = time.perf_counter()
    common, original_cache_hit, original_common_ms, fact_ms = _prepare_dc_common_v1(question, holder)
    if common.get("route") != "RETRIEVE":
        return {
            "variant": str(variant).upper(),
            "route": common.get("route"),
            "route_message": common.get("route_message"),
            "common": common,
            "common_cache_hit": original_cache_hit,
            "answer_cache_hit": False,
            "latency": {
                "common_this_click_ms": original_common_ms,
                "fact_index_match_ms": fact_ms,
                "answer_ms": 0.0,
                "click_wall_ms": (time.perf_counter() - total_started) * 1000,
            },
            "api_trace": {"logical_api_calls": 0, "physical_http_attempts": 0, "traces": []},
        }
    cross_passed, cross_audit = is_cross_business_dc_v1(common)
    common["dc_cross_business_gate"] = cross_audit
    if not cross_passed:
        gate_route = (
            "CROSS_BUSINESS_STRUCTURE_INVALID"
            if cross_audit.get("cross_business_candidate")
            else "C_POLICY_TARGET"
        )
        gate_message = (
            "교차업무 질문으로 확인했지만 업무별 검색 구조나 필수 근거가 완전하지 않아 답변을 생성하지 않았습니다. "
            "질문에 포함할 업무를 최대 3개까지 명확히 적어 다시 질문해 주세요."
            if gate_route == "CROSS_BUSINESS_STRUCTURE_INVALID"
            else (
                "이 질문은 교차업무 복합질의가 아니므로 D안을 호출하지 않았습니다. "
                "최종 운영 정책에서는 C안으로 답변합니다."
            )
        )
        event = {
            "question": _clean_text(question),
            "variant": str(variant).upper(),
            "answer_api_calls": 0,
            "route": gate_route,
            "cross_business_gate_passed": False,
            "cross_business_candidate": bool(cross_audit.get("cross_business_candidate")),
            "p0_cross_preserved": bool(cross_audit.get("p0_cross_preserved")),
            "business_count": cross_audit["business_count"],
            "need_count": cross_audit["need_count"],
            "common_this_click_ms": original_common_ms,
            "click_wall_ms": (time.perf_counter() - total_started) * 1000,
        }
        holder["events"].append(event)
        return {
            "variant": str(variant).upper(),
            "route": gate_route,
            "route_message": gate_message,
            "common": common,
            "common_cache_hit": original_cache_hit,
            "answer_cache_hit": False,
            "latency": {
                "common_this_click_ms": original_common_ms,
                "fact_index_match_ms": fact_ms,
                "answer_ms": 0.0,
                "click_wall_ms": event["click_wall_ms"],
            },
            "api_trace": {"logical_api_calls": 0, "physical_http_attempts": 0, "traces": []},
            "cross_business_gate": cross_audit,
            "event": event,
        }
    result = _EXECUTE_DC_GENERAL_V1(
        variant,
        question,
        holder,
        force_answer_regeneration=force_answer_regeneration,
    )
    result["common_cache_hit"] = original_cache_hit
    result["latency"]["common_this_click_ms"] = original_common_ms
    result["latency"]["fact_index_match_ms"] = fact_ms
    result["latency"]["click_wall_ms"] = (time.perf_counter() - total_started) * 1000
    result["cross_business_gate"] = cross_audit
    payload = result.get("payload") or {}
    event = result.get("event") or {}
    event.update({
        "common_cache_hit": original_cache_hit,
        "common_this_click_ms": original_common_ms,
        "fact_index_match_ms": fact_ms,
        "click_wall_ms": result["latency"]["click_wall_ms"],
        "cross_business_gate_passed": True,
        "cross_business_candidate": bool(cross_audit.get("cross_business_candidate")),
        "p0_cross_preserved": bool(cross_audit.get("p0_cross_preserved")),
        "response_mode_expected": payload.get("response_mode_expected"),
        "response_mode_actual": payload.get("response_mode_actual"),
        "response_mode_validation_passed": payload.get("response_mode_validation_passed"),
        "cross_need_scope_passed": (payload.get("skeleton") or {}).get("cross_need_scope_passed"),
    })
    return result

# ==== overlay: dc-cross-safety ====

# v1.2: 1Call 일반답변 fallback + 적용 대상 특수성 안전가드
DC_APPLICABILITY_SCOPE_RULES_V2 = """
[적용 대상·상황 특수성 규칙]
- 질문에 상속인·사망·피상속인이 없으면 상속인 금융거래조회를 일반적인 미수령금 조회 방법으로 설명하지 마세요.
- 질문이 일반적이면 일반 안내 근거를 우선 사용하고, 상속인·대리인·법인·미성년자 근거는 반드시 '해당 경우'로 한정하세요.
- 특정 역할이나 상황의 Evidence를 사용하면 answer_item의 applicability_scope를 SPECIAL_CASE로 표시하고 applies_to를 작성하세요.
- GENERAL 질문에 SPECIAL_CASE를 제도의 정의·대표 절차·유일한 방법처럼 확대하지 마세요.
- 미수령금 전체를 상속인이 받을 돈 또는 주로 상속 절차로 처리되는 돈이라고 설명하지 마세요.
""".strip()

DC_SKELETON_SYSTEM_PROMPT_V1 = DC_SKELETON_SYSTEM_PROMPT_V1 + "\n\n" + DC_APPLICABILITY_SCOPE_RULES_V2
DC_FINAL_SYSTEM_PROMPT_V1 = DC_FINAL_SYSTEM_PROMPT_V1 + "\n\n" + DC_APPLICABILITY_SCOPE_RULES_V2
DC_ONECALL_SYSTEM_PROMPT_V1 = DC_ONECALL_SYSTEM_PROMPT_V1 + "\n\n" + DC_APPLICABILITY_SCOPE_RULES_V2


_PARSE_ONECALL_STRICT_BEFORE_FALLBACK_V2 = globals().get("_PARSE_ONECALL_STRICT_BEFORE_FALLBACK_V2") or _parse_onecall_output_dc_v2


def _parse_onecall_output_dc_v2(raw: str) -> dict[str, Any]:
    """정상 태그·기존 JSON을 우선 사용하고, 일반 Markdown은 감사 가능한 fallback으로 보존합니다."""
    try:
        return _PARSE_ONECALL_STRICT_BEFORE_FALLBACK_V2(raw)
    except ValueError as error:
        answer = str(raw or "").strip()
        if not answer:
            raise
        return {
            "answer_skeleton": {
                "response_mode": "SEPARATE",
                "core_answer": "",
                "answer_items": [],
                "cross_need_relation": {
                    "requested": False,
                    "supported": False,
                    "claim": "",
                    "evidence_ids": [],
                    "fact_claim_ids": [],
                    "missing_reason": "모델이 Answer Skeleton 출력 계약을 따르지 않음",
                },
                "uncertainties": ["Answer Skeleton이 모델 출력에서 누락됨"],
                "conflicts": [],
            },
            "answer": answer,
            "used_evidence_ids": [],
            "used_fact_claim_ids": [],
            "output_contract": "PLAIN_ANSWER_CONTRACT_FALLBACK",
            "output_contract_passed": False,
            "output_contract_error": str(error),
        }


def build_program_fallback_skeleton_dc_v2(
    question: str,
    answer_needs: Sequence[Mapping[str, Any]],
    pack: Mapping[str, Any],
) -> dict[str, Any]:
    expected_mode = classify_dc_response_mode_v1(question)
    items = []
    for need in answer_needs:
        evidence_ids = list(dict.fromkeys(
            str(value) for value in need.get("retrieval_evidence_ids") or [] if str(value)
        ))[:2]
        items.append({
            "need_id": str(need.get("need_id") or ""),
            "business_function": str(need.get("label") or ""),
            "topic": str(need.get("question_part") or need.get("label") or ""),
            "status": "PARTIAL",
            "claim": "모델이 Answer Skeleton을 반환하지 않아 최종답변만 로컬 검증합니다.",
            "conditions": [],
            "details": [],
            "evidence_ids": evidence_ids,
            "fact_claim_ids": [],
            "missing_reason": "ONECALL_OUTPUT_CONTRACT_FAILED",
            "applicability_scope": "UNVERIFIED",
            "applies_to": [],
        })
    raw = {
        "response_mode": expected_mode,
        "core_answer": "",
        "answer_items": items,
        "cross_need_relation": {
            "requested": expected_mode in {"COMPARE", "RELATION"},
            "supported": False,
            "claim": "",
            "evidence_ids": [],
            "fact_claim_ids": [],
            "missing_reason": "모델 Skeleton 누락으로 관계 근거 구조를 검증할 수 없음",
        },
        "uncertainties": ["Answer Skeleton이 프로그램 fallback으로 생성됨"],
        "conflicts": [],
    }
    return validate_dc_skeleton_v1(raw, answer_needs, pack)


DC_SCOPE_ROLE_PATTERNS_V2 = {
    "INHERITANCE": re.compile(r"상속|상속인|피상속인|사망|유족"),
    "PROXY": re.compile(r"대리인|위임|대리\s*신청"),
    "SENDER": re.compile(r"송금인|착오송금인|돈을\s*보낸"),
    "RECIPIENT": re.compile(r"수취인|돈을\s*받은|잘못\s*받"),
    "MINOR": re.compile(r"미성년자|친권자"),
    "CORPORATION": re.compile(r"법인|사업자|대표자"),
}

DC_HIGH_RISK_GENERALIZATION_PATTERNS_V2 = [
    (
        "UNCLAIMED_FUNDS_INHERITANCE_GENERALIZATION",
        re.compile(r"미수령금(?:은|이란|의 경우).{0,45}(?:주로\s*)?(?:상속\s*절차|상속인이\s*받아야|상속을\s*통해).{0,30}(?:처리|수령|돈)", re.I),
    ),
    (
        "UNCLAIMED_FUNDS_INHERITANCE_ONLY_METHOD",
        re.compile(r"미수령금.{0,35}(?:상속인\s*금융거래조회).{0,25}(?:로만|유일|통해서만)", re.I),
    ),
]


def audit_applicability_scope_dc_v2(
    question: str,
    answer: str,
) -> dict[str, Any]:
    question_roles = {
        role for role, pattern in DC_SCOPE_ROLE_PATTERNS_V2.items() if pattern.search(str(question or ""))
    }
    answer_roles = {
        role for role, pattern in DC_SCOPE_ROLE_PATTERNS_V2.items() if pattern.search(str(answer or ""))
    }
    unrequested_answer_roles = sorted(answer_roles - question_roles)
    hits = []
    if "INHERITANCE" not in question_roles:
        for rule_id, pattern in DC_HIGH_RISK_GENERALIZATION_PATTERNS_V2:
            match = pattern.search(str(answer or ""))
            if match:
                hits.append({"rule_id": rule_id, "matched_text": match.group(0)})
        if (
            "INHERITANCE" in answer_roles
            and re.search(r"미수령금", str(question or ""))
        ):
            hits.append({
                "rule_id": "UNREQUESTED_INHERITANCE_SCOPE",
                "matched_text": "질문에 없는 상속·피상속인 절차",
            })
    return {
        "question_roles": sorted(question_roles),
        "answer_roles": sorted(answer_roles),
        "unrequested_answer_roles": unrequested_answer_roles,
        "high_risk_generalization_hits": hits,
        "applicability_scope_passed": not hits,
    }


def apply_applicability_scope_guard_dc_v2(
    question: str,
    answer: str,
) -> tuple[str, dict[str, Any], bool]:
    audit = audit_applicability_scope_dc_v2(question, answer)
    if audit["applicability_scope_passed"]:
        return answer, audit, False
    corrected = str(answer or "")
    corrected = re.sub(
        r"미수령금은\s*주로\s*상속\s*절차를\s*통해\s*처리됩니다\.?",
        "미수령금의 일반 조회·신청 절차와 상속인에게 적용되는 별도 조회 절차는 구분해야 합니다.",
        corrected,
        flags=re.I,
    )
    corrected = re.sub(
        r"미수령금은\s*상속인이\s*받아야\s*할\s*돈(?:을\s*의미합니다)?\.?",
        "미수령금은 예금자 등이 찾아가지 않은 금액이며, 상속인 조회는 사망한 예금자와 관련된 특수한 경우입니다.",
        corrected,
        flags=re.I,
    )
    notice = (
        "**적용 대상 안내:** 질문에 상속 상황이 명시되지 않았으므로, 상속인 금융거래조회 절차를 "
        "일반적인 미수령금 조회 방법으로 단정하지 않습니다. 상속인의 경우에만 별도 절차가 적용될 수 있습니다."
    )
    corrected = notice + "\n\n" + corrected
    audit["guard_notice_added"] = True
    return corrected, audit, True


_GENERATE_DC_ONECALL_BEFORE_SAFETY_V2 = globals().get("_GENERATE_DC_ONECALL_BEFORE_SAFETY_V2") or generate_dc_onecall_v1
_GENERATE_DC_TWOCALL_BEFORE_SAFETY_V2 = globals().get("_GENERATE_DC_TWOCALL_BEFORE_SAFETY_V2") or generate_dc_twocall_v1


def _rebuild_plain_fallback_payload_dc_v2(
    result: dict[str, Any],
    question: str,
    augmented_pack: Mapping[str, Any],
) -> dict[str, Any]:
    answer_needs = list(result.get("answer_needs") or extract_dc_answer_needs_v1(question, augmented_pack))
    skeleton = build_program_fallback_skeleton_dc_v2(question, answer_needs, augmented_pack)
    selected_pack = filter_augmented_pack_for_dc_v1(augmented_pack, skeleton)
    answer = str(result.get("answer") or "").strip()
    reference_audit = audit_dc_final_references_v1(answer, skeleton, augmented_pack)
    numeric_audit = audit_numeric_support_dc_v1(answer, selected_pack)
    forbidden_audit = audit_forbidden_claims_dc_v1(answer, selected_pack)
    expected_mode = classify_dc_response_mode_v1(question)
    result.update({
        "skeleton": skeleton,
        "need_coverage": skeleton["need_coverage"],
        "coverage_status": "PARTIAL",
        "strict_need_coverage_rate": 0.0,
        "answerable_need_coverage_rate": 1.0 if answer_needs else 0.0,
        "selected_evidence_pack": selected_pack,
        "selected_evidence_count": len(selected_pack.get("evidence") or []),
        "used_evidence_ids": reference_audit["used_evidence_ids"],
        "used_fact_claim_ids": reference_audit["used_fact_claim_ids"],
        "reference_audit": reference_audit,
        "numeric_audit": numeric_audit,
        "forbidden_claim_audit": forbidden_audit,
        "validation_passed": False,
        "response_mode_expected": expected_mode,
        "response_mode_actual": expected_mode,
        "response_mode_validation_passed": True,
        "output_contract": "PLAIN_ANSWER_CONTRACT_FALLBACK",
        "output_contract_passed": False,
        "skeleton_source": "PROGRAM_FALLBACK",
        "plain_answer_fallback": True,
    })
    return result


def _apply_scope_safety_to_result_dc_v2(
    result: dict[str, Any],
    question: str,
) -> dict[str, Any]:
    guarded_answer, scope_audit, guard_applied = apply_applicability_scope_guard_dc_v2(
        question, str(result.get("answer") or "")
    )
    result["answer"] = normalize_answer_markdown_v3(guarded_answer)
    result["applicability_scope_audit"] = scope_audit
    result["applicability_scope_guard_applied"] = guard_applied
    if guard_applied:
        result["validation_passed"] = False
        if result.get("coverage_status") == "SUFFICIENT":
            result["coverage_status"] = "PARTIAL"
    result.setdefault("output_contract_passed", True)
    result.setdefault("skeleton_source", "MODEL")
    result.setdefault("plain_answer_fallback", False)
    return result


def generate_dc_onecall_v1(
    question: str,
    augmented_pack: Mapping[str, Any],
) -> dict[str, Any]:
    result = _GENERATE_DC_ONECALL_BEFORE_SAFETY_V2(question, augmented_pack)
    if str(result.get("output_contract") or "") == "PLAIN_ANSWER_CONTRACT_FALLBACK":
        result = _rebuild_plain_fallback_payload_dc_v2(result, question, augmented_pack)
    return _apply_scope_safety_to_result_dc_v2(result, question)


def generate_dc_twocall_v1(
    question: str,
    augmented_pack: Mapping[str, Any],
) -> dict[str, Any]:
    result = _GENERATE_DC_TWOCALL_BEFORE_SAFETY_V2(question, augmented_pack)
    result["output_contract_passed"] = True
    result["skeleton_source"] = "MODEL"
    result["plain_answer_fallback"] = False
    return _apply_scope_safety_to_result_dc_v2(result, question)


_EXECUTE_DC_BEFORE_SAFETY_AUDIT_V2 = globals().get("_EXECUTE_DC_BEFORE_SAFETY_AUDIT_V2") or execute_dc_variant_v1


def execute_dc_variant_v1(
    variant: str,
    question: str,
    holder: dict[str, Any],
    *,
    force_answer_regeneration: bool = False,
) -> dict[str, Any]:
    result = _EXECUTE_DC_BEFORE_SAFETY_AUDIT_V2(
        variant,
        question,
        holder,
        force_answer_regeneration=force_answer_regeneration,
    )
    payload = result.get("payload") or {}
    event = result.get("event")
    if isinstance(event, dict) and payload:
        event.update({
            "output_contract_passed": bool(payload.get("output_contract_passed")),
            "skeleton_source": str(payload.get("skeleton_source") or ""),
            "plain_answer_fallback": bool(payload.get("plain_answer_fallback")),
            "applicability_scope_passed": bool(
                (payload.get("applicability_scope_audit") or {}).get("applicability_scope_passed")
            ),
            "applicability_scope_guard_applied": bool(payload.get("applicability_scope_guard_applied")),
        })
    return result

# ==== overlay: dc-cross-tagged-guard ====

# v1.3: 태그 포함 fallback 정리 + 미지원 관계 추론 차단 + Registry 내부 안내 미노출
_PARSE_ONECALL_BEFORE_TAGGED_FALLBACK_V3 = globals().get("_PARSE_ONECALL_BEFORE_TAGGED_FALLBACK_V3") or _parse_onecall_output_dc_v2


def _extract_final_answer_from_tagged_raw_dc_v3(raw: str) -> str:
    """구조화 Skeleton이 실패해도 FINAL_ANSWER 사용자 답변만 안전하게 분리합니다."""
    text = str(raw or "").strip()
    if not text:
        return ""
    patterns = [
        re.compile(
            r"<\s*FINAL_ANSWER\s*>\s*(.*?)(?:<\s*/\s*FINAL_ANSWER\s*>|\Z)",
            re.I | re.S,
        ),
        re.compile(
            r"\[\s*FINAL_ANSWER\s*\]\s*(.*?)(?=\n\s*\[\s*(?:SKELETON_JSON|FINAL_ANSWER)\s*\]|\Z)",
            re.I | re.S,
        ),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            answer = match.group(1).strip()
            answer = re.sub(r"<\s*/\s*FINAL_ANSWER\s*>\s*$", "", answer, flags=re.I).strip()
            if answer:
                return answer
    return ""


def _parse_onecall_output_dc_v2(raw: str) -> dict[str, Any]:
    """정상 구조화 결과를 우선 사용하고, 실패 시 사용자용 최종답변만 보존합니다."""
    parsed = _PARSE_ONECALL_BEFORE_TAGGED_FALLBACK_V3(raw)
    if str(parsed.get("output_contract") or "") != "PLAIN_ANSWER_CONTRACT_FALLBACK":
        return parsed

    final_answer = _extract_final_answer_from_tagged_raw_dc_v3(raw)
    if final_answer:
        parsed["answer"] = final_answer
        parsed["fallback_kind"] = "TAGGED_INVALID_SKELETON_FINAL_ONLY"
        parsed["tagged_final_answer_extracted"] = True
    else:
        parsed["fallback_kind"] = "PLAIN_MARKDOWN"
        parsed["tagged_final_answer_extracted"] = False
    return parsed


DC_RELATION_TERM_PATTERN_V3 = re.compile(r"(?:동시|한\s*번에|같이|함께|따로|각각)", re.I)
DC_RELATION_SPECULATION_PATTERN_V3 = re.compile(
    r"(?:가능성(?:이)?\s*(?:높|있)|것으로\s*(?:보|추정)|것\s*같|추측)",
    re.I,
)
DC_RELATION_UNSUPPORTED_NOTICE_V3 = (
    "제공된 공식 근거만으로는 두 항목의 동시 처리 또는 동시 신청 가능 여부를 확인할 수 없습니다."
)


def _remove_unsupported_relation_speculation_dc_v3(
    question: str,
    answer: str,
    skeleton: Mapping[str, Any],
) -> tuple[str, dict[str, Any], bool]:
    expected_mode = classify_dc_response_mode_v1(question)
    relation = skeleton.get("cross_need_relation") or {}
    relation_requested = expected_mode in {"RELATION", "COMPARE"} or bool(relation.get("requested"))
    relation_supported = bool(relation.get("supported"))
    audit = {
        "response_mode": expected_mode,
        "relation_requested": relation_requested,
        "relation_supported": relation_supported,
        "removed_sentences": [],
        "unsupported_relation_speculation_passed": True,
    }
    if not relation_requested or relation_supported:
        return str(answer or ""), audit, False

    text = str(answer or "").strip()
    if not text:
        return text, audit, False

    parts = re.split(r"(?<=[.!?])\s+", text)
    kept = []
    for part in parts:
        sentence = part.strip()
        if not sentence:
            continue
        if (
            DC_RELATION_TERM_PATTERN_V3.search(sentence)
            and DC_RELATION_SPECULATION_PATTERN_V3.search(sentence)
        ):
            audit["removed_sentences"].append(sentence)
            continue
        kept.append(sentence)

    if not audit["removed_sentences"]:
        return text, audit, False

    corrected = "\n\n".join(kept).strip()
    if not re.search(r"(?:공식\s*근거|직접적인\s*(?:언급|근거)).{0,30}(?:확인|명시).{0,10}(?:않|없)", corrected):
        corrected = (corrected + "\n\n" + DC_RELATION_UNSUPPORTED_NOTICE_V3).strip()
    audit["unsupported_relation_speculation_passed"] = False
    return corrected, audit, True


_GENERATE_DC_ONECALL_BEFORE_RELATION_GUARD_V3 = globals().get("_GENERATE_DC_ONECALL_BEFORE_RELATION_GUARD_V3") or generate_dc_onecall_v1
_GENERATE_DC_TWOCALL_BEFORE_RELATION_GUARD_V3 = globals().get("_GENERATE_DC_TWOCALL_BEFORE_RELATION_GUARD_V3") or generate_dc_twocall_v1


def _apply_relation_speculation_guard_to_result_dc_v3(
    result: dict[str, Any],
    question: str,
    augmented_pack: Mapping[str, Any],
) -> dict[str, Any]:
    corrected, audit, applied = _remove_unsupported_relation_speculation_dc_v3(
        question,
        str(result.get("answer") or ""),
        result.get("skeleton") or {},
    )
    result["answer"] = normalize_answer_markdown_v3(corrected)
    result["relation_speculation_audit"] = audit
    result["relation_speculation_guard_applied"] = applied
    result["fallback_answer_sanitized"] = bool(
        result.get("plain_answer_fallback")
        and "SKELETON_JSON" not in str(result.get("answer") or "")
        and "FINAL_ANSWER" not in str(result.get("answer") or "")
    )
    if applied:
        result["validation_passed"] = False
        if result.get("coverage_status") == "SUFFICIENT":
            result["coverage_status"] = "PARTIAL"
        selected_pack = result.get("selected_evidence_pack") or augmented_pack
        result["reference_audit"] = audit_dc_final_references_v1(
            result["answer"], result.get("skeleton") or {}, augmented_pack
        )
        result["numeric_audit"] = audit_numeric_support_dc_v1(result["answer"], selected_pack)
        result["forbidden_claim_audit"] = audit_forbidden_claims_dc_v1(result["answer"], selected_pack)
    return result


def generate_dc_onecall_v1(
    question: str,
    augmented_pack: Mapping[str, Any],
) -> dict[str, Any]:
    result = _GENERATE_DC_ONECALL_BEFORE_RELATION_GUARD_V3(question, augmented_pack)
    return _apply_relation_speculation_guard_to_result_dc_v3(result, question, augmented_pack)


def generate_dc_twocall_v1(
    question: str,
    augmented_pack: Mapping[str, Any],
) -> dict[str, Any]:
    result = _GENERATE_DC_TWOCALL_BEFORE_RELATION_GUARD_V3(question, augmented_pack)
    return _apply_relation_speculation_guard_to_result_dc_v3(result, question, augmented_pack)


_ACTION_LINKS_MARKDOWN_BEFORE_NOTICE_REMOVAL_V3 = globals().get("_ACTION_LINKS_MARKDOWN_BEFORE_NOTICE_REMOVAL_V3") or action_links_markdown_v1


def action_links_markdown_v1(action_links: Sequence[Mapping[str, Any]]) -> str:
    """Registry 검증은 유지하고 내부 구현 설명 문구만 사용자 화면에서 제외합니다."""
    rendered = _ACTION_LINKS_MARKDOWN_BEFORE_NOTICE_REMOVAL_V3(action_links)
    visible_lines = [
        line
        for line in str(rendered or "").splitlines()
        if "Action Link Registry" not in line
    ]
    return "\n".join(visible_lines).rstrip()


_EXECUTE_DC_BEFORE_V3_AUDIT = globals().get("_EXECUTE_DC_BEFORE_V3_AUDIT") or execute_dc_variant_v1


def execute_dc_variant_v1(
    variant: str,
    question: str,
    holder: dict[str, Any],
    *,
    force_answer_regeneration: bool = False,
) -> dict[str, Any]:
    result = _EXECUTE_DC_BEFORE_V3_AUDIT(
        variant,
        question,
        holder,
        force_answer_regeneration=force_answer_regeneration,
    )
    payload = result.get("payload") or {}
    event = result.get("event")
    if isinstance(event, dict) and payload:
        event.update({
            "fallback_answer_sanitized": bool(payload.get("fallback_answer_sanitized")),
            "relation_speculation_guard_applied": bool(payload.get("relation_speculation_guard_applied")),
            "unsupported_relation_speculation_passed": bool(
                (payload.get("relation_speculation_audit") or {}).get(
                    "unsupported_relation_speculation_passed", True
                )
            ),
        })
    return result

# ==== overlay: c-direct-runtime ====

# C안 직접답변 1Call + D-C 1Call + D-C 2Call 세 방식 비교 UI
C_THREEWAY_PROMPT_VERSION_V1 = "c-direct-cross-3perbusiness-v1"
C_THREEWAY_MAX_TOKENS_V1 = 2400
C_THREEWAY_VARIANTS_V1 = ("C_1CALL", "DC_1CALL", "DC_2CALL")
C_CROSS_DIRECT_SYSTEM_PROMPT_V1 = (
    C_STRUCTURED_SYSTEM_PROMPT_V3
    + "\n\n[교차업무 C안 직접답변 규칙]\n"
    + "- Answer Skeleton을 만들지 말고 Evidence Pack 전체에서 최종 답변을 직접 생성하세요.\n"
    + "- Evidence Pack의 모든 Need를 업무명 소제목으로 구분하고 질문에 나온 순서를 보존하세요.\n"
    + "- 각 업무 문단에는 그 Need에 연결된 Evidence 또는 동일 업무 Fact Claim을 최소 하나 인용하세요.\n"
    + "- 서로 다른 업무의 대상·조건·금액·기간·서류·절차를 섞지 마세요.\n"
    + "- COMPARE·RELATION·SEQUENCE 관계는 직접 근거가 있을 때만 단정하세요.\n"
    + "- response_mode은 제공된 expected_response_mode과 정확히 같아야 합니다.\n"
    + DC_APPLICABILITY_SCOPE_RULES_V2
)


def audit_c_direct_references_v1(answer: str, payload: Mapping[str, Any], pack: Mapping[str, Any]) -> dict[str, Any]:
    allowed_evidence = set(answer_b_core._allowed_evidence(pack))
    allowed_facts = set(_allowed_fact_claims_v3(pack))
    inline_evidence = set(re.findall(r"\[(E\d+)\]", str(answer or "")))
    inline_facts = set(re.findall(r"\[(FI-CAND-\d+:F\d+)\]", str(answer or "")))
    declared_evidence = set(answer_b_core._clean_list(payload.get("declared_evidence_ids")))
    declared_facts = set(_clean_fact_claim_keys_v3(payload.get("declared_fact_claim_ids")))
    invalid_evidence = sorted(inline_evidence - allowed_evidence)
    invalid_facts = sorted(inline_facts - allowed_facts)
    declared_not_inline_evidence = sorted((declared_evidence & allowed_evidence) - inline_evidence)
    declared_not_inline_facts = sorted((declared_facts & allowed_facts) - inline_facts)
    valid_evidence = sorted(inline_evidence & allowed_evidence)
    valid_facts = sorted(inline_facts & allowed_facts)
    passed = bool(
        (valid_evidence or valid_facts)
        and not invalid_evidence
        and not invalid_facts
        and not declared_not_inline_evidence
        and not declared_not_inline_facts
    )
    return {
        "reference_consistency_passed": passed,
        "valid_evidence_ids": valid_evidence,
        "valid_fact_claim_ids": valid_facts,
        "invalid_evidence_ids": invalid_evidence,
        "invalid_fact_claim_ids": invalid_facts,
        "declared_not_inline_evidence_ids": declared_not_inline_evidence,
        "declared_not_inline_fact_claim_ids": declared_not_inline_facts,
    }


def audit_c_direct_need_citations_v1(
    payload: Mapping[str, Any],
    pack: Mapping[str, Any],
    *,
    cross_required: bool,
) -> dict[str, Any]:
    needs = [dict(row) for row in pack.get("needs") or []]
    quota_audit = audit_need_evidence_pack_v5(pack) if cross_required else {"passed": True, "needs": []}
    reference_audit = dict(payload.get("reference_audit") or {})
    cited_evidence = set(reference_audit.get("valid_evidence_ids") or [])
    cited_facts = set(reference_audit.get("valid_fact_claim_ids") or [])
    fact_business = _fact_claim_business_map_dc_v1(pack)
    rows = []
    for need in needs:
        need_id = _clean_text(need.get("need_id"))
        business = _clean_text(need.get("business_function"))
        allowed_evidence = set(str(value) for value in need.get("evidence_ids") or [])
        matched_evidence = sorted(cited_evidence & allowed_evidence)
        matched_facts = sorted(
            fact_id for fact_id in cited_facts
            if _clean_text(fact_business.get(fact_id)) == business
        )
        rows.append({
            "need_id": need_id,
            "business_function": business,
            "cited_evidence_ids": matched_evidence,
            "cited_fact_claim_ids": matched_facts,
            "passed": bool(business and (matched_evidence or matched_facts)),
        })
    covered = sum(1 for row in rows if row["passed"])
    need_count = len(rows)
    passed = bool(
        reference_audit.get("reference_consistency_passed")
        and (not cross_required or (quota_audit.get("passed") and need_count >= 2 and covered == need_count))
    )
    return {
        "passed": passed,
        "cross_required": cross_required,
        "need_count": need_count,
        "covered_need_count": covered,
        "strict_need_coverage_rate": (covered / need_count) if need_count else 1.0,
        "answerable_need_coverage_rate": (covered / need_count) if need_count else 1.0,
        "required_evidence_per_business": NEED_BATCH_REQUIRED_TOP_K_V5 if cross_required else 0,
        "score_gate_enabled": bool(NEED_BATCH_SCORE_GATE_ENABLED_V5) if cross_required else False,
        "configured_minimum_need_reranker_score": NEED_BATCH_MIN_RERANKER_SCORE_V5 if cross_required else None,
        "effective_minimum_need_reranker_score": (
            NEED_BATCH_MIN_RERANKER_SCORE_V5
            if cross_required and NEED_BATCH_SCORE_GATE_ENABLED_V5
            else None
        ),
        "minimum_need_reranker_score": (
            NEED_BATCH_MIN_RERANKER_SCORE_V5
            if cross_required and NEED_BATCH_SCORE_GATE_ENABLED_V5
            else None
        ),
        "quota_audit_passed": bool(quota_audit.get("passed")),
        "needs": rows,
    }


def generate_c_direct_threeway_v1(question: str, augmented_pack: Mapping[str, Any]) -> dict[str, Any]:
    total_started = time.perf_counter()
    expected_mode = classify_dc_response_mode_v1(question)
    answer_needs = [
        {
            "need_id": _clean_text(row.get("need_id")),
            "business_function": _clean_text(row.get("business_function")),
            "subquery": _clean_text(row.get("subquery")),
            "evidence_ids": list(row.get("evidence_ids") or []),
        }
        for row in augmented_pack.get("needs") or []
    ]
    relation_constraint = relation_constraint_v1(question, augmented_pack)
    prompt_started = time.perf_counter()
    prompt = f"""[사용자 질문]\n{_clean_text(question)}\n\n[expected_response_mode]\n{expected_mode}\n\n[업무별 Answer Needs]\n{_compact_json(answer_needs)}\n\n[관계 주장 안전조건]\n{_compact_json(relation_constraint)}\n\n[동일 공통 Fact Index 보강 Evidence Pack]\n{_compact_json(augmented_pack)}\n\n[출력 JSON]\n{{\"response_mode\":\"{expected_mode}\",\"answer\":\"업무별 소제목을 포함한 최종 Markdown 답변\",\"used_evidence_ids\":[\"E1\"],\"used_fact_claim_ids\":[\"FI-CAND-001:F1\"],\"coverage_status\":\"SUFFICIENT|PARTIAL|INSUFFICIENT\",\"missing_information\":[]}}"""
    prompt_ms = (time.perf_counter() - prompt_started) * 1000
    raw, usage, api_ms, trace = _call_answer_api_v1(
        system_prompt=C_CROSS_DIRECT_SYSTEM_PROMPT_V1,
        user_prompt=prompt,
        max_tokens=C_THREEWAY_MAX_TOKENS_V1,
    )
    parse_started = time.perf_counter()
    local_recovery = False
    try:
        parsed = answer_b_core._extract_json_object(raw)
        answer = answer_b_core._strip_model_urls(str(parsed.get("answer") or "").strip())
        if not answer:
            raise ValueError("C안 직접답변이 비어 있습니다.")
        actual_mode = str(parsed.get("response_mode") or "").upper()
        declared_evidence = [
            value for value in answer_b_core._clean_list(parsed.get("used_evidence_ids"))
            if value in answer_b_core._allowed_evidence(augmented_pack)
        ]
        declared_facts = [
            value for value in _clean_fact_claim_keys_v3(parsed.get("used_fact_claim_ids"))
            if value in _allowed_fact_claims_v3(augmented_pack)
        ]
        coverage = str(parsed.get("coverage_status") or "PARTIAL").upper()
        if coverage not in answer_b_core.ALLOWED_COVERAGE_STATUS:
            coverage = "PARTIAL"
        missing = answer_b_core._clean_list(parsed.get("missing_information"))
    except (ValueError, TypeError, json.JSONDecodeError):
        local_recovery = True
        answer = _extract_final_answer_from_tagged_raw_dc_v3(raw) or answer_b_core._strip_model_urls(str(raw).strip())
        if not answer:
            raise ValueError("C안 직접답변 출력이 비어 있습니다.")
        actual_mode = "UNKNOWN"
        declared_evidence = []
        declared_facts = []
        coverage = "PARTIAL"
        missing = ["C안 JSON 출력 계약을 로컬 복구함"]
    parse_ms = (time.perf_counter() - parse_started) * 1000

    post_started = time.perf_counter()
    answer, relation_guard_applied = _relation_safe_answer_v1(answer, relation_constraint)
    answer, applicability_audit, applicability_guard_applied = apply_applicability_scope_guard_dc_v2(question, answer)
    answer = normalize_answer_markdown_v3(answer)
    provisional = {
        "declared_evidence_ids": declared_evidence,
        "declared_fact_claim_ids": declared_facts,
    }
    reference_audit = audit_c_direct_references_v1(answer, provisional, augmented_pack)
    cross_required = len(augmented_pack.get("needs") or []) >= 2
    provisional["reference_audit"] = reference_audit
    need_citation_audit = audit_c_direct_need_citations_v1(
        provisional, augmented_pack, cross_required=cross_required
    )
    numeric_audit = audit_numeric_support_dc_v1(answer, augmented_pack)
    forbidden_claim_audit = audit_forbidden_claims_dc_v1(answer, augmented_pack)
    response_mode_passed = actual_mode == expected_mode
    output_contract_passed = not local_recovery
    validation_passed = bool(
        need_citation_audit.get("passed")
        and reference_audit.get("reference_consistency_passed")
        and numeric_audit.get("numeric_support_passed")
        and forbidden_claim_audit.get("forbidden_claim_check_passed")
        and applicability_audit.get("applicability_scope_passed")
        and response_mode_passed
        and output_contract_passed
        and not relation_guard_applied
    )
    if not validation_passed and coverage == "SUFFICIENT":
        coverage = "PARTIAL"
    post_ms = (time.perf_counter() - post_started) * 1000
    trace_parts = _trace_parts_v3(trace)
    total_ms = (time.perf_counter() - total_started) * 1000
    return {
        "system": "C",
        "answer": answer,
        "answer_needs": answer_needs,
        "used_evidence_ids": reference_audit["valid_evidence_ids"],
        "used_chunk_ids": [
            answer_b_core._allowed_evidence(augmented_pack)[value]
            for value in reference_audit["valid_evidence_ids"]
        ],
        "used_fact_claim_ids": reference_audit["valid_fact_claim_ids"],
        "used_fact_index_ids": list(dict.fromkeys(
            value.split(":", 1)[0] for value in reference_audit["valid_fact_claim_ids"]
        )),
        "coverage_status": coverage,
        "missing_information": missing,
        "strict_need_coverage_rate": need_citation_audit["strict_need_coverage_rate"],
        "answerable_need_coverage_rate": need_citation_audit["answerable_need_coverage_rate"],
        "need_citation_audit": need_citation_audit,
        "reference_audit": reference_audit,
        "numeric_audit": numeric_audit,
        "forbidden_claim_audit": forbidden_claim_audit,
        "applicability_scope_audit": applicability_audit,
        "applicability_scope_guard_applied": applicability_guard_applied,
        "relation_constraint": relation_constraint,
        "relation_guard_applied": relation_guard_applied,
        "response_mode_expected": expected_mode,
        "response_mode_actual": actual_mode,
        "response_mode_validation_passed": response_mode_passed,
        "validation_passed": validation_passed,
        "output_contract": "C_DIRECT_JSON",
        "output_contract_passed": output_contract_passed,
        "skeleton_source": "NONE_DIRECT_C",
        "plain_answer_fallback": local_recovery,
        "selected_evidence_pack": augmented_pack,
        "latency_ms": total_ms,
        "usage": usage,
        "api_calls": 1,
        "attempts": [{"stage": "answer_c_direct", "latency_ms": api_ms, "trace": trace}],
        "local_recovery": local_recovery,
        "stage_latency_ms": {
            "direct_prompt_build_ms": prompt_ms,
            "direct_api_wall_ms": float(trace_parts.get("api_wall_ms") or api_ms),
            "direct_pacing_wait_ms": float(trace_parts.get("pacing_wait_ms") or 0),
            "direct_retry_wait_ms": float(trace_parts.get("retry_wait_ms") or 0),
            "direct_estimated_service_ms": float(trace_parts.get("estimated_service_ms") or 0),
            "direct_parse_validation_ms": parse_ms,
            "postprocess_ms": post_ms,
            "answer_total_ms": total_ms,
        },
    }


def _c_threeway_cache_key_v1(common: Mapping[str, Any]) -> str:
    return _stable_json_hash_c1({
        "variant": "C_1CALL",
        "resolved_question": common.get("resolved_question"),
        "augmented_pack_sha256": common.get("dc_augmented_pack_sha256"),
        "prompt_version": C_THREEWAY_PROMPT_VERSION_V1,
    })


def _execute_c_threeway_v1(
    question: str,
    holder: dict[str, Any],
    *,
    force_answer_regeneration: bool = False,
) -> dict[str, Any]:
    total_started = time.perf_counter()
    gate_start = len(HCX_SHARED_GATE_V3.history)
    common, common_cache_hit, common_this_click_ms, fact_this_click_ms = _prepare_dc_common_v1(question, holder)
    if common.get("route") != "RETRIEVE":
        return {
            "variant": "C_1CALL",
            "route": common.get("route"),
            "route_message": common.get("route_message"),
            "common": common,
            "common_cache_hit": common_cache_hit,
            "answer_cache_hit": False,
            "latency": {
                "common_this_click_ms": common_this_click_ms,
                "fact_index_match_ms": fact_this_click_ms,
                "answer_ms": 0.0,
                "click_wall_ms": (time.perf_counter() - total_started) * 1000,
            },
            "api_trace": _trace_summary_since_v3(gate_start),
        }

    cross_passed, cross_audit = is_cross_business_dc_v1(common)
    common["dc_cross_business_gate"] = cross_audit
    cross_candidate = bool(cross_audit.get("cross_business_candidate"))
    if cross_candidate and not cross_passed:
        event = {
            "question": _clean_text(question),
            "variant": "C_1CALL",
            "answer_api_calls": 0,
            "route": "CROSS_BUSINESS_STRUCTURE_INVALID",
            "cross_business_gate_required": True,
            "cross_business_gate_passed": False,
            "business_count": int(cross_audit.get("business_count") or 0),
            "need_count": int(cross_audit.get("need_count") or 0),
            "common_this_click_ms": common_this_click_ms,
            "click_wall_ms": (time.perf_counter() - total_started) * 1000,
        }
        holder["events"].append(event)
        return {
            "variant": "C_1CALL",
            "route": "CROSS_BUSINESS_STRUCTURE_INVALID",
            "route_message": (
                "C안도 교차업무 질문에서는 업무별 필수 근거 3개와 서로 다른 Parent 2개 이상이 필요합니다. "
                "세 번째 근거는 같은 Parent의 다른 chunk를 제한적으로 사용할 수 있습니다. "
                + (
                    f"현재 점수 게이트는 {NEED_BATCH_MIN_RERANKER_SCORE_V5:.4f} 이상으로 활성화되어 있습니다. "
                    if NEED_BATCH_SCORE_GATE_ENABLED_V5
                    else "현재 점수 게이트는 비활성화되어 점수는 차단에 사용하지 않습니다. "
                )
                + "질문에 포함할 업무를 최대 3개까지 명확히 적어 다시 질문해 주세요."
            ),
            "common": common,
            "common_cache_hit": common_cache_hit,
            "answer_cache_hit": False,
            "latency": {
                "common_this_click_ms": common_this_click_ms,
                "fact_index_match_ms": fact_this_click_ms,
                "answer_ms": 0.0,
                "click_wall_ms": event["click_wall_ms"],
            },
            "api_trace": _trace_summary_since_v3(gate_start),
            "cross_business_gate": cross_audit,
            "event": event,
        }

    cache_key = _c_threeway_cache_key_v1(common)
    cached = holder["answer_cache"].get(cache_key)
    if cached is not None and not force_answer_regeneration:
        payload = copy.deepcopy(cached)
        answer_cache_hit = True
        answer_ms = 0.0
    else:
        answer_cache_hit = False
        answer_started = time.perf_counter()
        payload = generate_c_direct_threeway_v1(common["resolved_question"], common["dc_augmented_pack"])
        answer_ms = (time.perf_counter() - answer_started) * 1000
        holder["answer_cache"][cache_key] = copy.deepcopy(payload)

    if not holder.get("committed"):
        holder["conversation"].setdefault("turns", []).extend([
            {"role": "user", "content": _clean_text(question)},
            {"role": "assistant", "content": normalize_answer_markdown_v3(payload.get("answer"))},
        ])
        holder["committed"] = True
        holder["committed_variant"] = "C_1CALL"

    stage = payload.get("stage_latency_ms") or {}
    usage = _variant_usage_v3(payload)
    result = {
        "variant": "C_1CALL",
        "route": "RETRIEVE",
        "common": common,
        "payload": payload,
        "augmented_pack": common["dc_augmented_pack"],
        "matched_fact_records": common.get("dc_matched_fact_records") or [],
        "fact_audit": common.get("dc_fact_audit") or [],
        "common_cache_hit": common_cache_hit,
        "answer_cache_hit": answer_cache_hit,
        "committed_variant": holder.get("committed_variant"),
        "official_sources": _official_sources_dc_v1(common),
        "action_links": action_links_for_streamlit_v1(common.get("action_links") or []),
        "cross_business_gate": cross_audit,
        "shared_augmented_pack_sha256": common.get("dc_augmented_pack_sha256"),
        "latency": {
            "stored_common_pipeline_ms": float((common.get("latency_ms") or {}).get("공통 준비 전체") or 0),
            "common_this_click_ms": common_this_click_ms,
            "fact_index_match_ms": fact_this_click_ms,
            "answer_ms": answer_ms,
            "click_wall_ms": (time.perf_counter() - total_started) * 1000,
        },
        "api_trace": _trace_summary_since_v3(gate_start),
        "usage": usage,
        "circuit": hcx_circuit_status_c1(),
    }
    event = {
        "question": _clean_text(question),
        "variant": "C_1CALL",
        "answer_api_calls": int(payload.get("api_calls") or 0),
        "common_cache_hit": common_cache_hit,
        "answer_cache_hit": answer_cache_hit,
        "fact_index_count": len(result["matched_fact_records"]),
        "fact_index_ids": " | ".join(str(row.get("fact_index_id") or "") for row in result["matched_fact_records"]),
        "shared_augmented_pack_sha256": common.get("dc_augmented_pack_sha256"),
        "required_evidence_per_business": NEED_BATCH_REQUIRED_TOP_K_V5 if cross_candidate else 0,
        "score_gate_enabled": bool(NEED_BATCH_SCORE_GATE_ENABLED_V5) if cross_candidate else False,
        "configured_minimum_need_reranker_score": NEED_BATCH_MIN_RERANKER_SCORE_V5 if cross_candidate else None,
        "effective_minimum_need_reranker_score": (
            NEED_BATCH_MIN_RERANKER_SCORE_V5
            if cross_candidate and NEED_BATCH_SCORE_GATE_ENABLED_V5
            else None
        ),
        "minimum_need_reranker_score": (
            NEED_BATCH_MIN_RERANKER_SCORE_V5
            if cross_candidate and NEED_BATCH_SCORE_GATE_ENABLED_V5
            else None
        ),
        "stored_common_pipeline_ms": result["latency"]["stored_common_pipeline_ms"],
        "common_this_click_ms": common_this_click_ms,
        "fact_index_match_ms": fact_this_click_ms,
        "answer_total_ms": float(stage.get("answer_total_ms") or 0),
        "direct_api_wall_ms": float(stage.get("direct_api_wall_ms") or 0),
        "answer_pacing_wait_ms": float(stage.get("direct_pacing_wait_ms") or 0),
        "answer_retry_wait_ms": float(stage.get("direct_retry_wait_ms") or 0),
        "answer_estimated_service_ms": float(stage.get("direct_estimated_service_ms") or 0),
        "click_wall_ms": result["latency"]["click_wall_ms"],
        **usage,
        "coverage_status": payload.get("coverage_status"),
        "strict_need_coverage_rate": float(payload.get("strict_need_coverage_rate") or 0),
        "answerable_need_coverage_rate": float(payload.get("answerable_need_coverage_rate") or 0),
        "reference_consistency_passed": bool((payload.get("reference_audit") or {}).get("reference_consistency_passed")),
        "numeric_support_passed": bool((payload.get("numeric_audit") or {}).get("numeric_support_passed")),
        "forbidden_claim_check_passed": bool((payload.get("forbidden_claim_audit") or {}).get("forbidden_claim_check_passed")),
        "validation_passed": bool(payload.get("validation_passed")),
        "output_contract": str(payload.get("output_contract") or "C_DIRECT_JSON"),
        "output_contract_passed": bool(payload.get("output_contract_passed")),
        "skeleton_source": "NONE_DIRECT_C",
        "cross_business_gate_required": cross_candidate,
        "cross_business_gate_passed": bool(cross_passed) if cross_candidate else True,
        "cross_business_candidate": cross_candidate,
        "p0_cross_preserved": bool(cross_audit.get("p0_cross_preserved")),
        "business_count": int(cross_audit.get("business_count") or 0),
        "need_count": int(cross_audit.get("need_count") or 0),
        "response_mode_expected": payload.get("response_mode_expected"),
        "response_mode_actual": payload.get("response_mode_actual"),
        "response_mode_validation_passed": payload.get("response_mode_validation_passed"),
        "cross_need_scope_passed": bool((payload.get("need_citation_audit") or {}).get("passed")),
        "answer_chars": len(str(payload.get("answer") or "")),
    }
    holder["events"].append(event)
    result["event"] = event
    return result


_EXECUTE_DC_BEFORE_C_THREEWAY_V1 = globals().get("_EXECUTE_DC_BEFORE_C_THREEWAY_V1") or execute_dc_variant_v1


def execute_dc_variant_v1(
    variant: str,
    question: str,
    holder: dict[str, Any],
    *,
    force_answer_regeneration: bool = False,
) -> dict[str, Any]:
    normalized_variant = str(variant).upper()
    if normalized_variant == "C_1CALL":
        return _execute_c_threeway_v1(
            question, holder, force_answer_regeneration=force_answer_regeneration
        )
    if normalized_variant not in {"DC_1CALL", "DC_2CALL"}:
        raise ValueError(f"지원하지 않는 세 방식 비교안: {normalized_variant}")
    result = _EXECUTE_DC_BEFORE_C_THREEWAY_V1(
        normalized_variant,
        question,
        holder,
        force_answer_regeneration=force_answer_regeneration,
    )
    common = result.get("common") or {}
    pack_hash = common.get("dc_augmented_pack_sha256")
    result["shared_augmented_pack_sha256"] = pack_hash
    event = result.get("event")
    if isinstance(event, dict):
        event["shared_augmented_pack_sha256"] = pack_hash
        event["required_evidence_per_business"] = (
            NEED_BATCH_REQUIRED_TOP_K_V5 if event.get("cross_business_candidate") else 0
        )
        event["score_gate_enabled"] = bool(
            NEED_BATCH_SCORE_GATE_ENABLED_V5 and event.get("cross_business_candidate")
        )
        event["configured_minimum_need_reranker_score"] = (
            NEED_BATCH_MIN_RERANKER_SCORE_V5 if event.get("cross_business_candidate") else None
        )
        event["effective_minimum_need_reranker_score"] = (
            NEED_BATCH_MIN_RERANKER_SCORE_V5
            if NEED_BATCH_SCORE_GATE_ENABLED_V5 and event.get("cross_business_candidate")
            else None
        )
        event["minimum_need_reranker_score"] = event["effective_minimum_need_reranker_score"]
    return result

# ==== overlay: production-routing ====

# 운영 정책: C 기본, 실제 다중업무 비교형만 D-C 2Call
KDIC_RUNTIME_POLICY_VERSION_V1 = "C_DEFAULT_DC2_COMPARE_ONLY_V1"
KDIC_RUNTIME_SOURCE_NOTEBOOK_V1 = (
    "2026-08-21-KDIC-D-교차업무전용-1Call-vs-2Call-v1_3-Colab.ipynb"
)
KDIC_RUNTIME_BUILD_V1 = {
    "source_notebook": KDIC_RUNTIME_SOURCE_NOTEBOOK_V1,
    "routing_policy": KDIC_RUNTIME_POLICY_VERSION_V1,
    "default_variant": "C_1CALL",
    "comparison_variant": "DC_2CALL",
    "dc1_enabled": False,
    "score_gate_enabled": bool(NEED_BATCH_SCORE_GATE_ENABLED_V5),
    "configured_minimum_need_reranker_score": float(NEED_BATCH_MIN_RERANKER_SCORE_V5),
    "maximum_cross_business_count": int(NEED_BATCH_MAX_BUSINESSES_V5),
    "required_evidence_per_business": int(NEED_BATCH_REQUIRED_TOP_K_V5),
}


def select_production_variant_v1(
    *,
    common_route: str,
    response_mode: str,
    cross_business_candidate: bool,
    cross_business_passed: bool,
    structure_valid: bool,
) -> tuple[str, str]:
    """Return the only production-safe variant and a stable audit reason."""

    route = str(common_route or "").upper()
    mode = str(response_mode or "SEPARATE").upper()
    if (
        route == "RETRIEVE"
        and mode == "COMPARE"
        and cross_business_candidate
        and cross_business_passed
        and structure_valid
    ):
        return "DC_2CALL", "CROSS_BUSINESS_COMPARE"
    if mode == "COMPARE" and cross_business_candidate and not structure_valid:
        return "C_1CALL", "COMPARE_STRUCTURE_INVALID_C_HARD_GATE"
    if mode == "COMPARE" and cross_business_candidate and not cross_business_passed:
        return "C_1CALL", "COMPARE_EVIDENCE_GATE_FAILED_C_HARD_GATE"
    if route != "RETRIEVE":
        return "C_1CALL", f"NON_RETRIEVE_{route or 'UNKNOWN'}"
    return "C_1CALL", "C_DEFAULT"


def production_route_decision_v1(
    question: str,
    holder: dict[str, Any],
) -> dict[str, Any]:
    """Prepare common retrieval once, then select C or comparison-only DC2."""

    common, common_cache_hit, common_this_click_ms, fact_this_click_ms = _prepare_dc_common_v1(
        question, holder
    )
    common_route = str(common.get("route") or "")
    resolved_question = str(common.get("resolved_question") or question or "")
    response_mode = classify_dc_response_mode_v1(resolved_question)
    if common_route == "RETRIEVE":
        cross_business_passed, cross_audit = is_cross_business_dc_v1(common)
    else:
        cross_business_passed, cross_audit = False, {
            "passed": False,
            "cross_business_candidate": False,
            "p0_cross_preserved": False,
            "structure_valid": True,
            "business_count": 0,
            "need_count": 0,
        }
    cross_business_candidate = bool(
        cross_audit.get("cross_business_candidate")
        or cross_audit.get("p0_cross_preserved")
    )
    structure_valid = bool(cross_audit.get("structure_valid", True))
    selected_variant, selection_reason = select_production_variant_v1(
        common_route=common_route,
        response_mode=response_mode,
        cross_business_candidate=cross_business_candidate,
        cross_business_passed=bool(cross_business_passed),
        structure_valid=structure_valid,
    )
    return {
        "policy_version": KDIC_RUNTIME_POLICY_VERSION_V1,
        "selected_variant": selected_variant,
        "selection_reason": selection_reason,
        "common_route": common_route,
        "response_mode": response_mode,
        "cross_business_candidate": cross_business_candidate,
        "cross_business_passed": bool(cross_business_passed),
        "structure_valid": structure_valid,
        "p0_cross_preserved": bool(cross_audit.get("p0_cross_preserved")),
        "business_count": int(cross_audit.get("business_count") or 0),
        "need_count": int(cross_audit.get("need_count") or 0),
        "common_cache_hit": bool(common_cache_hit),
        "common_this_click_ms": float(common_this_click_ms or 0),
        "fact_index_this_click_ms": float(fact_this_click_ms or 0),
        "dc1_enabled": False,
        "cross_business_audit": cross_audit,
    }


def execute_production_variant_v1(
    question: str,
    holder: dict[str, Any],
    *,
    force_answer_regeneration: bool = False,
) -> dict[str, Any]:
    """Single server entrypoint: C by default, DC2 only for valid comparisons."""

    decision = production_route_decision_v1(question, holder)
    selected_variant = str(decision["selected_variant"])
    if selected_variant == "DC_1CALL":
        raise RuntimeError("DC_1CALL is disabled in the production routing policy.")
    result = execute_dc_variant_v1(
        selected_variant,
        question,
        holder,
        force_answer_regeneration=force_answer_regeneration,
    )
    if not isinstance(result, Mapping):
        raise TypeError("Production KDIC execution result must be a mapping.")
    output = dict(result)
    output["production_routing"] = decision
    output["runtime_build"] = dict(KDIC_RUNTIME_BUILD_V1)
    output["answer_system"] = str(output.get("variant") or selected_variant)
    event = output.get("event")
    if isinstance(event, dict):
        event["production_policy_version"] = KDIC_RUNTIME_POLICY_VERSION_V1
        event["production_selected_variant"] = selected_variant
        event["production_selection_reason"] = decision["selection_reason"]
        event["dc1_enabled"] = False
    return output


_NORMALIZE_ANSWER_BEFORE_OFFICIAL_CONTACT_GUARD_V5 = normalize_answer_markdown_v3


def normalize_answer_markdown_v3(text: Any) -> str:
    """Preserve answer formatting and repair one invalidly merged KDIC phone number."""

    value = _NORMALIZE_ANSWER_BEFORE_OFFICIAL_CONTACT_GUARD_V5(text)
    return re.sub(r"(?<!\d)02[\s-]*1588[\s-]*0037(?!\d)", "1588-0037", value)


KDIC_RUNTIME_BUILD_V1.update({
    "build_sha256": KDIC_PRODUCTION_OVERLAY_SOURCE_SHA256,
    "overlay_file": "2026-08-25-kdic-production-overlay.py",
    "overlay_revision": KDIC_PRODUCTION_OVERLAY_REVISION,
})
