from __future__ import annotations

"""KDIC 답변 B v2: 기본 답변과 동일 Evidence Pack 기반 근거 상세설명."""

import hashlib
import json
import re
import time
from collections import OrderedDict
from typing import Any, Callable, Mapping, Sequence


ALLOWED_COVERAGE_STATUS = {"SUFFICIENT", "PARTIAL", "INSUFFICIENT"}

BASIC_ANSWER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "answer", "used_evidence_ids", "used_chunk_ids",
        "coverage_status", "missing_information",
    ],
    "properties": {
        "answer": {"type": "string", "minLength": 1},
        "used_evidence_ids": {
            "type": "array", "minItems": 1, "items": {"type": "string"},
        },
        "used_chunk_ids": {
            "type": "array", "minItems": 1, "items": {"type": "string"},
        },
        "coverage_status": {
            "type": "string", "enum": ["SUFFICIENT", "PARTIAL", "INSUFFICIENT"],
        },
        "missing_information": {"type": "array", "items": {"type": "string"}},
    },
}

EVIDENCE_EXPLANATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "explanation_summary", "claim_evidence_map", "conditions",
        "exceptions", "limitations", "additional_information_needed",
    ],
    "properties": {
        "explanation_summary": {"type": "string", "minLength": 1},
        "claim_evidence_map": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["claim", "evidence_ids", "chunk_ids", "relevance_reason"],
                "properties": {
                    "claim": {"type": "string", "minLength": 1},
                    "evidence_ids": {
                        "type": "array", "minItems": 1, "items": {"type": "string"},
                    },
                    "chunk_ids": {
                        "type": "array", "minItems": 1, "items": {"type": "string"},
                    },
                    "relevance_reason": {"type": "string", "minLength": 1},
                },
            },
        },
        "conditions": {"type": "array", "items": {"type": "string"}},
        "exceptions": {"type": "array", "items": {"type": "string"}},
        "limitations": {"type": "array", "items": {"type": "string"}},
        "additional_information_needed": {
            "type": "array", "items": {"type": "string"},
        },
    },
}

BASIC_ANSWER_SYSTEM_PROMPT = """
당신은 예금보험공사 공식 문서 기반 답변 시스템입니다.

반드시 지킬 규칙:
1. 사용자 질문과 Basic Evidence Pack에 있는 내용만 사용합니다.
2. 질문에 대한 결론을 먼저 제시하는 일반적인 기본 답변을 작성합니다.
3. 필요한 조건·금액·기간·절차·예외를 질문 범위 안에서 포함합니다.
4. 서로 다른 제도나 대상을 임의로 결합하지 않습니다.
5. Evidence에 없는 사실·URL·전화번호·해석을 추가하지 않습니다.
6. 사용한 문장 끝에 [E1] 형식으로 Evidence ID를 표시합니다.
7. 특정 값을 판단할 정보가 부족하면 추측하지 말고 missing_information에 기록합니다.
8. 지정된 JSON 객체 하나만 출력합니다.
9. JSON 문자열 내부의 줄바꿈·탭은 실제 제어문자가 아니라 \\n·\\t로 이스케이프합니다.
""".strip()

EVIDENCE_EXPLANATION_SYSTEM_PROMPT = """
당신은 예금보험공사 답변의 문서 근거를 설명하는 시스템입니다.

반드시 지킬 규칙:
1. 기본 답변 생성에 사용한 동일한 Basic Evidence Pack만 사용합니다.
2. 기본 답변의 핵심 주장과 Evidence ID·Chunk ID의 연결 관계를 설명합니다.
3. 각 Evidence가 질문과 해당 주장에 관련되는 이유를 문서 내용 기준으로 설명합니다.
4. 적용 조건·예외·근거 한계·추가 필요 정보를 구분합니다.
5. 기본 답변과 모순되는 새로운 결론을 만들지 않습니다.
6. 모델의 숨겨진 사고과정이나 내부 추론을 서술하지 않습니다.
7. Evidence에서 사용자가 확인할 수 있는 근거 관계만 설명합니다.
8. Evidence에 없는 사실·URL·전화번호를 추가하지 않습니다.
9. 지정된 JSON 객체 하나만 출력합니다.
10. JSON 문자열 내부의 줄바꿈·탭은 실제 제어문자가 아니라 \\n·\\t로 이스케이프합니다.
""".strip()


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _clean_list(value: Any) -> list[str]:
    values = value if isinstance(value, list) else [value] if value not in (None, "") else []
    output: list[str] = []
    for item in values:
        text = _clean(item)
        if text and text not in output:
            output.append(text)
    return output


def _strip_model_urls(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\(https?://[^)]+\)", r"\1", text)
    return re.sub(r"https?://[^\s)\]}>]+", "", text).strip()


def _escape_control_chars_inside_json_strings(text: str) -> str:
    """JSON 문자열 리터럴 안의 비이스케이프 제어문자만 안전하게 교정한다.

    객체 필드 사이의 정상 줄바꿈은 그대로 두고, 따옴표 안의 LF/CR/TAB 및
    U+0000~U+001F만 JSON 표준 이스케이프 형태로 바꾼다.
    """
    output: list[str] = []
    in_string = False
    escaped = False
    for character in str(text or ""):
        if not in_string:
            output.append(character)
            if character == '"':
                in_string = True
            continue

        if escaped:
            output.append(character)
            escaped = False
            continue
        if character == "\\":
            output.append(character)
            escaped = True
            continue
        if character == '"':
            output.append(character)
            in_string = False
            continue
        if character == "\n":
            output.append("\\n")
        elif character == "\r":
            output.append("\\r")
        elif character == "\t":
            output.append("\\t")
        elif ord(character) < 0x20:
            output.append(f"\\u{ord(character):04x}")
        else:
            output.append(character)
    return "".join(output)


def _json_candidates(text: str) -> list[str]:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    backslash_repaired = re.sub(
        r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', cleaned
    )
    candidates = [
        cleaned,
        backslash_repaired,
        _escape_control_chars_inside_json_strings(cleaned),
        _escape_control_chars_inside_json_strings(backslash_repaired),
    ]
    return list(dict.fromkeys(candidates))


def _extract_json_object(text: str) -> dict[str, Any]:
    last_error: Exception | None = None
    for candidate in _json_candidates(text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as error:
            last_error = error
            start = candidate.find("{")
            if start < 0:
                continue
            try:
                parsed, _ = json.JSONDecoder().raw_decode(candidate[start:])
            except json.JSONDecodeError as nested:
                last_error = nested
                continue
        if not isinstance(parsed, dict):
            raise TypeError("구조화 답변의 최상위 값은 JSON 객체여야 합니다.")
        return parsed
    raise ValueError(f"구조화 답변 JSON 파싱 실패: {last_error}") from last_error


def _decode_raw_answer_text(text: str) -> str:
    """객체 복구가 불가능할 때 JSON 문자열 또는 일반 본문만 보수적으로 꺼낸다."""
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    for candidate in _json_candidates(cleaned):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, str):
            return parsed.strip()
        if isinstance(parsed, Mapping) and _clean(parsed.get("answer")):
            return str(parsed.get("answer")).strip()
    if cleaned.startswith("{"):
        return ""
    return cleaned.strip('"').strip()


def build_basic_evidence_pack(
    question: str,
    search_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """LLM 없이 검색 근거의 경계와 출처를 결정적으로 정돈한다."""
    evidence: list[dict[str, Any]] = []
    sources_by_url: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for index, result in enumerate(search_results, start=1):
        chunk = result.get("chunk") or result.get("original_chunk") or {}
        if not isinstance(chunk, Mapping):
            raise TypeError(f"검색 결과 {index}의 chunk가 객체가 아닙니다.")
        rank = int(result.get("rank") or index)
        chunk_id = _clean(result.get("chunk_id") or chunk.get("chunk_id"))
        if not chunk_id:
            raise ValueError(f"검색 결과 {index}에 chunk_id가 없습니다.")
        source_url = _clean(chunk.get("source_url"))
        source_id = None
        if source_url:
            if source_url not in sources_by_url:
                sources_by_url[source_url] = {
                    "source_id": f"S{len(sources_by_url) + 1}",
                    "title": _clean(chunk.get("title") or chunk.get("document_title")),
                    "source_url": source_url,
                }
            source_id = sources_by_url[source_url]["source_id"]
        evidence.append({
            "evidence_id": f"E{index}",
            "rank": rank,
            "chunk_id": chunk_id,
            "parent_id": _clean(result.get("parent_id") or chunk.get("parent_doc_id")) or None,
            "context_chunk_ids": list(result.get("context_chunk_ids") or chunk.get("context_chunk_ids") or [chunk_id]),
            "document_title": _clean(chunk.get("title") or chunk.get("document_title")),
            "section_title": _clean(chunk.get("section_title")),
            "content": _clean(chunk.get("content")),
            "source_id": source_id,
            "source_url": source_url,
        })
    if not evidence:
        raise ValueError("Basic Evidence Pack을 만들 검색 결과가 없습니다.")
    return {
        "question": _clean(question),
        "evidence": evidence,
        "sources": list(sources_by_url.values()),
    }


def evidence_pack_sha256(pack: Mapping[str, Any]) -> str:
    raw = json.dumps(pack, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _allowed_evidence(pack: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(row["evidence_id"]): str(row["chunk_id"])
        for row in pack.get("evidence") or []
    }


def _allowed_chunks_by_evidence(pack: Mapping[str, Any]) -> dict[str, set[str]]:
    output: dict[str, set[str]] = {}
    for row in pack.get("evidence") or []:
        evidence_id = str(row["evidence_id"])
        values = {str(row["chunk_id"])}
        values.update(str(item) for item in row.get("context_chunk_ids") or [] if str(item))
        output[evidence_id] = values
    return output


def validate_basic_answer(
    payload: Mapping[str, Any],
    pack: Mapping[str, Any],
) -> dict[str, Any]:
    answer = _strip_model_urls(_clean(payload.get("answer")))
    if not answer:
        raise ValueError("기본 답변 본문이 비어 있습니다.")
    allowed = _allowed_evidence(pack)
    requested_ids = _clean_list(payload.get("used_evidence_ids"))
    invalid_ids = [item for item in requested_ids if item not in allowed]
    if invalid_ids:
        raise ValueError(f"Evidence Pack에 없는 Evidence ID: {invalid_ids}")
    used_ids = list(requested_ids)
    for number in re.findall(r"\[E(\d+)\]", answer):
        evidence_id = f"E{number}"
        if evidence_id in allowed and evidence_id not in used_ids:
            used_ids.append(evidence_id)
    if not used_ids:
        raise ValueError("기본 답변에 유효한 Evidence ID가 없습니다.")
    allowed_chunks = _allowed_chunks_by_evidence(pack)
    permitted_chunks = set().union(*(allowed_chunks[item] for item in used_ids))
    model_chunks = _clean_list(payload.get("used_chunk_ids"))
    if not model_chunks or not set(model_chunks).issubset(permitted_chunks):
        raise ValueError("기본 답변의 Chunk ID가 사용 Evidence와 일치하지 않습니다.")
    coverage = _clean(payload.get("coverage_status")).upper()
    if coverage not in ALLOWED_COVERAGE_STATUS:
        raise ValueError(f"허용되지 않은 coverage_status: {coverage}")
    return {
        "answer": answer,
        "used_evidence_ids": used_ids,
        "used_chunk_ids": model_chunks,
        "coverage_status": coverage,
        "missing_information": _clean_list(payload.get("missing_information")),
    }


def validate_evidence_explanation(
    payload: Mapping[str, Any],
    pack: Mapping[str, Any],
) -> dict[str, Any]:
    summary = _strip_model_urls(_clean(payload.get("explanation_summary")))
    if not summary:
        raise ValueError("근거 상세설명의 요약이 비어 있습니다.")
    allowed = _allowed_evidence(pack)
    allowed_chunks = _allowed_chunks_by_evidence(pack)
    mappings: list[dict[str, Any]] = []
    for index, raw in enumerate(payload.get("claim_evidence_map") or [], start=1):
        if not isinstance(raw, Mapping):
            raise TypeError(f"claim_evidence_map {index}가 객체가 아닙니다.")
        claim = _strip_model_urls(_clean(raw.get("claim")))
        reason = _strip_model_urls(_clean(raw.get("relevance_reason")))
        evidence_ids = _clean_list(raw.get("evidence_ids"))
        if not claim or not reason or not evidence_ids:
            raise ValueError(f"claim_evidence_map {index}의 필수값이 비었습니다.")
        invalid = [item for item in evidence_ids if item not in allowed]
        if invalid:
            raise ValueError(f"상세설명에 Pack 밖의 Evidence ID가 있습니다: {invalid}")
        permitted_chunks = set().union(*(allowed_chunks[item] for item in evidence_ids))
        model_chunks = _clean_list(raw.get("chunk_ids"))
        if not model_chunks or not set(model_chunks).issubset(permitted_chunks):
            raise ValueError("상세설명의 Chunk ID가 Evidence ID와 일치하지 않습니다.")
        mappings.append({
            "claim": claim,
            "evidence_ids": evidence_ids,
            "chunk_ids": model_chunks,
            "relevance_reason": reason,
        })
    if not mappings:
        raise ValueError("유효한 주장-Evidence 연결이 없습니다.")
    return {
        "explanation_summary": summary,
        "claim_evidence_map": mappings,
        "conditions": _clean_list(payload.get("conditions")),
        "exceptions": _clean_list(payload.get("exceptions")),
        "limitations": _clean_list(payload.get("limitations")),
        "additional_information_needed": _clean_list(payload.get("additional_information_needed")),
    }


def _structured_output_is_unsupported(error: Exception) -> bool:
    if type(error).__name__ != "BadRequestError":
        return False
    message = str(error).lower()
    return any(marker in message for marker in (
        "response_format", "json_schema", "json_object", "unsupported",
        "not supported", "convert error",
    ))


def _structured_output_capability_cache(client: Any) -> dict[str, bool]:
    """동일 HCX client에서 확인한 response_format 지원 여부를 보존한다."""
    attribute = "_kdic_structured_output_capability"
    cache = getattr(client, attribute, None)
    if isinstance(cache, dict):
        return cache
    cache = {}
    try:
        setattr(client, attribute, cache)
    except Exception:
        # 일부 client wrapper가 속성 설정을 막아도 정답 생성은 계속한다.
        pass
    return cache


def _call_model(
    *, client: Any, model: str, system_prompt: str, user_prompt: str,
    max_tokens: int, response_format: Mapping[str, Any] | None,
) -> tuple[str, dict[str, int], float]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    if response_format is not None:
        kwargs["response_format"] = dict(response_format)
    started = time.perf_counter()
    response = client.chat.completions.create(**kwargs)
    latency_ms = (time.perf_counter() - started) * 1000
    content = response.choices[0].message.content
    if not content or not str(content).strip():
        raise RuntimeError("HCX 구조화 답변 출력이 비어 있습니다.")
    usage_obj = getattr(response, "usage", None)
    usage = {
        key: int(getattr(usage_obj, key, 0) or 0)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    return str(content), usage, latency_ms


def _call_structured(
    *, client: Any, model: str, system_prompt: str, user_prompt: str,
    schema_name: str, schema: Mapping[str, Any], max_tokens: int,
    validator: Callable[[Mapping[str, Any]], dict[str, Any]],
    raw_recovery: Callable[[str], dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, int], float, list[dict[str, Any]]]:
    capability_cache = _structured_output_capability_cache(client)
    if capability_cache.get(model) is False:
        formats: list[tuple[str, Mapping[str, Any] | None]] = [("prompt", None)]
    else:
        formats = [
            ("json_schema", {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name, "strict": True, "schema": dict(schema),
                },
            }),
            ("json_object", {"type": "json_object"}),
            ("prompt", None),
        ]
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    total_latency_ms = 0.0
    attempts: list[dict[str, Any]] = []
    raw_outputs: list[str] = []
    unsupported_formats: set[str] = set()
    for format_name, response_format in formats:
        previous_output = ""
        previous_error = ""
        for repair_index in range(3):
            prompt = user_prompt if repair_index == 0 else f"""
[작업]
직전 출력은 JSON 파싱 또는 Evidence 검증에 실패했습니다.
원래 Evidence의 사실 범위를 바꾸지 말고 지정된 JSON 객체로 한 번만 교정하세요.

[검증 실패 이유]
{previous_error}

[원래 요청]
{user_prompt}

[직전 출력]
{previous_output[:6000]}
""".strip()
            try:
                raw, usage, latency_ms = _call_model(
                    client=client, model=model, system_prompt=system_prompt,
                    user_prompt=prompt, max_tokens=max_tokens,
                    response_format=response_format,
                )
            except Exception as error:
                if response_format is not None and _structured_output_is_unsupported(error):
                    unsupported_formats.add(format_name)
                    if {"json_schema", "json_object"}.issubset(unsupported_formats):
                        capability_cache[model] = False
                    attempts.append({
                        "format": format_name, "repair_index": repair_index,
                        "valid": False, "fallback_reason": "STRUCTURED_OUTPUT_UNSUPPORTED",
                        "error": f"{type(error).__name__}: {error}",
                    })
                    break
                raise
            raw_outputs.append(raw)
            if response_format is not None:
                capability_cache[model] = True
            total_latency_ms += latency_ms
            for key in total_usage:
                total_usage[key] += int(usage.get(key) or 0)
            try:
                validated = validator(_extract_json_object(raw))
            except (ValueError, TypeError) as error:
                previous_output = raw
                previous_error = f"{type(error).__name__}: {error}"
                attempts.append({
                    "format": format_name, "repair_index": repair_index,
                    "valid": False, "error": f"{type(error).__name__}: {error}",
                    "raw_output_preview": raw[:2000], "latency_ms": latency_ms,
                })
                continue
            attempts.append({
                "format": format_name, "repair_index": repair_index,
                "valid": True, "latency_ms": latency_ms,
            })
            return validated, total_usage, total_latency_ms, attempts
    if raw_recovery is not None:
        recovery_errors: list[str] = []
        for raw in reversed(list(dict.fromkeys(raw_outputs))):
            try:
                recovered = raw_recovery(raw)
            except (ValueError, TypeError) as error:
                recovery_errors.append(f"{type(error).__name__}: {error}")
                continue
            attempts.append({
                "format": "raw_text_recovery", "repair_index": None,
                "valid": True, "fallback_reason": "STRUCTURED_METADATA_RECOVERED",
                "latency_ms": 0.0,
            })
            return recovered, total_usage, total_latency_ms, attempts
        if recovery_errors:
            attempts.append({
                "format": "raw_text_recovery", "repair_index": None,
                "valid": False, "errors": recovery_errors,
            })
    raise ValueError(
        "HCX 구조화 답변 생성 실패. attempts="
        + json.dumps(attempts, ensure_ascii=False, default=str)
    )


def _recover_basic_answer_from_raw(
    text: str,
    evidence_pack: Mapping[str, Any],
) -> dict[str, Any]:
    """본문과 명시적 [E#]가 있을 때만 기본 답변을 보수적으로 복구한다."""
    answer = _decode_raw_answer_text(text)
    if not answer:
        raise ValueError("복구 가능한 기본 답변 본문이 없습니다.")
    allowed = _allowed_evidence(evidence_pack)
    used_ids: list[str] = []
    for number in re.findall(r"\[E(\d+)\]", answer):
        evidence_id = f"E{int(number)}"
        if evidence_id in allowed and evidence_id not in used_ids:
            used_ids.append(evidence_id)
    if not used_ids:
        raise ValueError("본문에 Evidence Pack과 일치하는 [E#] 인용이 없습니다.")
    payload = {
        "answer": answer,
        "used_evidence_ids": used_ids,
        "used_chunk_ids": [allowed[evidence_id] for evidence_id in used_ids],
        "coverage_status": "PARTIAL",
        "missing_information": [],
    }
    return validate_basic_answer(payload, evidence_pack)


def generate_basic_answer_b_v2(
    *, client: Any, model: str, question: str, evidence_pack: Mapping[str, Any],
) -> dict[str, Any]:
    prompt = f"""
[사용자 질문]
{_clean(question)}

[Basic Evidence Pack JSON]
{json.dumps(evidence_pack, ensure_ascii=False, indent=2)}

[출력 JSON]
{{
  "answer": "사용자에게 바로 제시할 기본 답변. 근거 문장에 [E1] 표시",
  "used_evidence_ids": ["E1"],
  "used_chunk_ids": ["실제 대표 chunk_id"],
  "coverage_status": "SUFFICIENT | PARTIAL | INSUFFICIENT",
  "missing_information": ["근거만으로 확인할 수 없는 필수 정보"]
}}
""".strip()
    validated, usage, latency_ms, attempts = _call_structured(
        client=client, model=model, system_prompt=BASIC_ANSWER_SYSTEM_PROMPT,
        user_prompt=prompt, schema_name="kdic_basic_answer_b_v2",
        schema=BASIC_ANSWER_SCHEMA, max_tokens=1600,
        validator=lambda payload: validate_basic_answer(payload, evidence_pack),
        raw_recovery=lambda raw: _recover_basic_answer_from_raw(raw, evidence_pack),
    )
    return {
        **validated, "mode": "basic",
        "evidence_pack_sha256": evidence_pack_sha256(evidence_pack),
        "latency_ms": latency_ms, "usage": usage, "format_attempts": attempts,
    }


def generate_evidence_explanation_b_v2(
    *, client: Any, model: str, question: str, evidence_pack: Mapping[str, Any],
    basic_answer: Mapping[str, Any],
) -> dict[str, Any]:
    pack_hash = evidence_pack_sha256(evidence_pack)
    if str(basic_answer.get("evidence_pack_sha256")) != pack_hash:
        raise ValueError("기본 답변과 근거 상세설명의 Evidence Pack이 다릅니다.")
    basic_view = {
        key: basic_answer.get(key)
        for key in (
            "answer", "used_evidence_ids", "used_chunk_ids",
            "coverage_status", "missing_information",
        )
    }
    prompt = f"""
[사용자 질문]
{_clean(question)}

[이미 생성된 기본 답변]
{json.dumps(basic_view, ensure_ascii=False, indent=2)}

[동일 Basic Evidence Pack JSON]
{json.dumps(evidence_pack, ensure_ascii=False, indent=2)}

[출력 JSON]
{{
  "explanation_summary": "기본 답변이 어떤 문서 근거로 구성됐는지 요약",
  "claim_evidence_map": [
    {{
      "claim": "기본 답변의 핵심 주장",
      "evidence_ids": ["E1"],
      "chunk_ids": ["실제 대표 chunk_id"],
      "relevance_reason": "해당 Evidence가 질문과 주장에 관련되는 문서상 이유"
    }}
  ],
  "conditions": ["근거에 명시된 적용 조건"],
  "exceptions": ["근거에 명시된 예외"],
  "limitations": ["현재 근거로 단정할 수 없는 범위"],
  "additional_information_needed": ["개별 판단에 추가로 필요한 정보"]
}}
""".strip()
    validated, usage, latency_ms, attempts = _call_structured(
        client=client, model=model, system_prompt=EVIDENCE_EXPLANATION_SYSTEM_PROMPT,
        user_prompt=prompt, schema_name="kdic_evidence_explanation_b_v2",
        schema=EVIDENCE_EXPLANATION_SCHEMA, max_tokens=2200,
        validator=lambda payload: validate_evidence_explanation(payload, evidence_pack),
    )
    return {
        **validated, "mode": "evidence_explanation",
        "evidence_pack_sha256": pack_hash,
        "latency_ms": latency_ms, "usage": usage, "format_attempts": attempts,
    }


def build_used_sources(
    evidence_pack: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    used = set(payload.get("used_evidence_ids") or [])
    if not used:
        for item in payload.get("claim_evidence_map") or []:
            used.update(item.get("evidence_ids") or [])
    sources: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for row in evidence_pack.get("evidence") or []:
        if row.get("evidence_id") not in used:
            continue
        url = _clean(row.get("source_url"))
        if not url:
            continue
        target = sources.setdefault(url, {
            "url": url,
            "title": _clean(row.get("document_title")) or "공식 출처",
            "evidence_ids": [],
            "chunk_ids": [],
        })
        target["evidence_ids"].append(row["evidence_id"])
        target["chunk_ids"].append(row["chunk_id"])
    return list(sources.values())


def evidence_explanation_to_markdown(payload: Mapping[str, Any]) -> str:
    lines = ["#### 답변 근거 설명", "", _clean(payload.get("explanation_summary")), ""]
    lines.extend(["##### 답변 주장과 문서 근거", ""])
    for index, item in enumerate(payload.get("claim_evidence_map") or [], start=1):
        evidence = ", ".join(item.get("evidence_ids") or [])
        chunks = ", ".join(item.get("chunk_ids") or [])
        lines.extend([
            f"{index}. **{_clean(item.get('claim'))}**",
            f"   - 사용 근거: {evidence} · `{chunks}`",
            f"   - 관련 이유: {_clean(item.get('relevance_reason'))}",
            "",
        ])
    for title, key in (
        ("적용 조건", "conditions"),
        ("예외", "exceptions"),
        ("현재 근거의 한계", "limitations"),
        ("추가로 필요한 정보", "additional_information_needed"),
    ):
        values = _clean_list(payload.get(key))
        if values:
            lines.extend([f"##### {title}", ""])
            lines.extend(f"- {value}" for value in values)
            lines.append("")
    return "\n".join(lines).strip()
