from __future__ import annotations

import ast
import builtins
import contextvars
import copy
import dis
import hashlib
import importlib.util
import json
import math
import re
import sys
import time
import tempfile
import types
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


BASE_DIR = Path(__file__).resolve().parent
ENGINE_FILE = BASE_DIR / "kdic_pipeline_engine.py"
OVERLAY_FILE = BASE_DIR / "2026-08-25-kdic-production-overlay.py"
ADAPTER_FILE = BASE_DIR / "2026-08-23-kdic-colab-runtime-adapter.py"
FASTAPI_FILE = BASE_DIR / "2026-08-23-kdic-fastapi-service.py"
CHAT_UI_FILE = BASE_DIR / "2026-08-23-kdic-chat-ui.html"
ANSWER_CORE_FILE = BASE_DIR / "kdic_v15_answer_b_core.py"
PROMPT_MANAGER_FILE = BASE_DIR / "kdic_prompt_manager.py"
ADMIN_EXTENSION_FILE = BASE_DIR / "kdic_admin_extension_aws.py"
EXPECTED_SOURCE_SHA256 = (
    "F9A908D62A43EA3A3566A5D8DF0E982F214373FFF96470A749DC1EFE79E25083"
)


def _function_source(source: str, name: str) -> str:
    tree = ast.parse(source)
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if not nodes:
        raise AssertionError(f"missing function: {name}")
    value = ast.get_source_segment(source, nodes[-1])
    if not value:
        raise AssertionError(f"cannot extract function: {name}")
    return value


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_static_contracts() -> dict[str, Any]:
    engine = ENGINE_FILE.read_text(encoding="utf-8")
    overlay = OVERLAY_FILE.read_text(encoding="utf-8")
    adapter = ADAPTER_FILE.read_text(encoding="utf-8")
    api = FASTAPI_FILE.read_text(encoding="utf-8")
    answer_core = ANSWER_CORE_FILE.read_text(encoding="utf-8")
    prompt_manager = PROMPT_MANAGER_FILE.read_text(encoding="utf-8")
    admin_extension = ADMIN_EXTENSION_FILE.read_text(encoding="utf-8")
    for path, source in (
        (ENGINE_FILE, engine),
        (OVERLAY_FILE, overlay),
        (ADAPTER_FILE, adapter),
        (FASTAPI_FILE, api),
        (ANSWER_CORE_FILE, answer_core),
        (PROMPT_MANAGER_FILE, prompt_manager),
        (ADMIN_EXTENSION_FILE, admin_extension),
    ):
        ast.parse(source, filename=str(path))
    assert "2026-08-25-kdic-production-overlay.py" in engine
    assert "execute_production_variant_v1" in overlay
    assert "C_DEFAULT_DC2_COMPARE_ONLY_V1" in overlay
    assert "2026-08-26-declared-citation-canonicalization-v12" in overlay
    assert "반드시 유효한 단일 JSON 객체 하나만 출력" in overlay
    assert "C_STRUCTURED_SYSTEM_PROMPT_V3\n    +" not in overlay
    assert "answer_b_core._call_structured(" in overlay
    assert "_structured_failure_summary_v2" in overlay
    assert "cited_answer = answer" in overlay
    assert "audit_c_direct_references_v1(\n        cited_answer" in overlay
    assert "[허용 Evidence ID - 이 목록 밖의 ID 사용 금지]" in overlay
    assert "allowed_fact_claim_ids = sorted(_allowed_fact_claims_v3" in overlay
    assert "DC_1CALL is disabled" in overlay
    assert "import kdic_lightweight_router_v1 as light_router" in overlay
    assert 'name = "V1.5_C_DEFAULT_DC2_COMPARE_ONLY"' in adapter
    assert '"runtime_build": dict(RUNTIME_BUILD_INFO)' in api
    assert EXPECTED_SOURCE_SHA256 in overlay
    assert EXPECTED_SOURCE_SHA256 in adapter
    assert "2026-08-26-declared-citation-canonicalization-v12" in adapter
    assert "cache_compatible_overlay_revisions" in adapter
    assert "for repair_index in range(3):" in answer_core
    assert "[검증 실패 이유]" in answer_core
    assert 'system_prompt=_managed_prompt("C_CROSS_DIRECT_SYSTEM_PROMPT_V1", C_CROSS_DIRECT_SYSTEM_PROMPT_V1)' in overlay
    assert 'system_prompt=_managed_prompt("DC_SKELETON_SYSTEM_PROMPT_V1", DC_SKELETON_SYSTEM_PROMPT_V1)' in overlay
    assert 'system_prompt=_managed_prompt("DC_FINAL_SYSTEM_PROMPT_V1", DC_FINAL_SYSTEM_PROMPT_V1)' in overlay
    assert '"settings": ["C_CROSS_DIRECT_SYSTEM_PROMPT_V1"]' in admin_extension
    assert "BASIS_EXPLAINER_SYSTEM_PROMPT_V1" in overlay
    assert "내부 사고과정, Evidence·chunk·parent·need ID" in overlay
    assert "Reranker 같은 검색 기술정보를 공개하지 말고" in overlay
    assert "def generate_user_basis_explanation_v1(" in overlay
    assert 'schema_name="kdic_user_basis_explanation_v2"' in overlay
    assert "공식 근거에 없는 숫자가 근거 해설에 포함되어 있습니다." in overlay
    assert "근거 해설에 내부 검색 식별자가 포함되어 있습니다." in overlay
    return {
        "engine_overlay_loader": "passed",
        "overlay_syntax": "passed",
        "adapter_syntax": "passed",
        "fastapi_build_contract": "passed",
        "overlay_sha256": hashlib.sha256(OVERLAY_FILE.read_bytes()).hexdigest(),
    }


def test_managed_prompt_contract() -> dict[str, Any]:
    prompt_module = _load_module("kdic_prompt_manager_contract", PROMPT_MANAGER_FILE)
    expected_slots = (
        "C_CROSS_DIRECT_SYSTEM_PROMPT_V1",
        "DC_SKELETON_SYSTEM_PROMPT_V1",
        "DC_FINAL_SYSTEM_PROMPT_V1",
    )
    assert tuple(prompt_module.PROMPT_SLOTS) == expected_slots

    def long_prompt(label: str) -> str:
        return ((label + " 운영 프롬프트 ") * 20).strip()

    defaults = {slot: long_prompt("기본 " + slot) for slot in expected_slots}
    legacy_c = long_prompt("기존 C")
    active_skeleton = long_prompt("활성 D-C 골격")
    active_final = long_prompt("활성 D-C 최종")
    draft_skeleton = long_prompt("초안 D-C 골격")
    draft_final = long_prompt("초안 D-C 최종")
    legacy_values = {
        "C_STRUCTURED_SYSTEM_PROMPT_V3": legacy_c,
        "DC_SKELETON_SYSTEM_PROMPT_V1": active_skeleton,
        "DC_FINAL_SYSTEM_PROMPT_V1": active_final,
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        state_path = Path(temp_dir) / "admin_prompts.json"
        state_path.write_text(json.dumps({
            "active_version": "legacy-active-v7",
            "active_values": legacy_values,
            "draft_values": {
                **legacy_values,
                "DC_SKELETON_SYSTEM_PROMPT_V1": draft_skeleton,
                "DC_FINAL_SYSTEM_PROMPT_V1": draft_final,
            },
            "history": [{"version": "legacy-history-v6", "values": legacy_values}],
            "updated_at": 1.0,
        }, ensure_ascii=False), encoding="utf-8")
        manager = prompt_module.PromptManager(state_path, defaults)
        active = manager.active_values()
        draft = manager.draft_values()
        persisted = json.loads(state_path.read_text(encoding="utf-8"))
        assert tuple(active) == expected_slots
        assert active["C_CROSS_DIRECT_SYSTEM_PROMPT_V1"] == defaults["C_CROSS_DIRECT_SYSTEM_PROMPT_V1"]
        assert active["DC_SKELETON_SYSTEM_PROMPT_V1"] == active_skeleton
        assert active["DC_FINAL_SYSTEM_PROMPT_V1"] == active_final
        assert draft["DC_SKELETON_SYSTEM_PROMPT_V1"] == draft_skeleton
        assert draft["DC_FINAL_SYSTEM_PROMPT_V1"] == draft_final
        assert persisted["schema_version"] == prompt_module.PROMPT_SCHEMA_VERSION
        assert persisted["active_version"] == "legacy-active-v7"
        assert persisted["history"][0]["version"] == "legacy-history-v6"
        assert persisted["legacy_values"]["C_STRUCTURED_SYSTEM_PROMPT_V3"]["active"] == legacy_c
        assert [row["slot"] for row in manager.public()["slots"]] == list(expected_slots)
        restored = manager.activate(legacy_values, version="legacy-snapshot-restored", archive_current=False)
        restored_values = manager.active_values()
        assert restored["active_version"] == "legacy-snapshot-restored"
        assert restored_values["C_CROSS_DIRECT_SYSTEM_PROMPT_V1"] == defaults["C_CROSS_DIRECT_SYSTEM_PROMPT_V1"]
        assert restored_values["DC_SKELETON_SYSTEM_PROMPT_V1"] == active_skeleton
        assert restored_values["DC_FINAL_SYSTEM_PROMPT_V1"] == active_final

        engine = ENGINE_FILE.read_text(encoding="utf-8")
        overrides = contextvars.ContextVar("prompt_test_overrides", default=None)
        namespace = {"_KDIC_PROMPT_OVERRIDES": overrides, "KDIC_PROMPT_MANAGER": manager}
        exec(_function_source(engine, "_managed_prompt"), namespace)
        managed_prompt = namespace["_managed_prompt"]
        assert managed_prompt("DC_SKELETON_SYSTEM_PROMPT_V1", "fallback") == active_skeleton
        token = overrides.set({"DC_SKELETON_SYSTEM_PROMPT_V1": long_prompt("A/B override")})
        try:
            assert managed_prompt("DC_SKELETON_SYSTEM_PROMPT_V1", "fallback").startswith("A/B override")
        finally:
            overrides.reset(token)
        assert managed_prompt("UNKNOWN_PROMPT", "fallback") == "fallback"

    return {
        "slots": list(expected_slots),
        "legacy_state_migration": "passed",
        "legacy_snapshot_restore": "passed",
        "active_manager_lookup": "passed",
        "context_override_precedence": "passed",
        "default_fallback": "passed",
    }


def test_overlay_exec_contract() -> dict[str, Any]:
    """Execute overlay top-level wiring against the names exported by the engine."""

    engine = ENGINE_FILE.read_text(encoding="utf-8")
    overlay = OVERLAY_FILE.read_text(encoding="utf-8")
    namespace: dict[str, Any] = {
        "__file__": str(ENGINE_FILE),
        "__name__": "kdic_pipeline_engine_overlay_contract",
        "re": re,
        "json": json,
        "time": time,
        "copy": copy,
        "hashlib": hashlib,
        "math": math,
        "OrderedDict": OrderedDict,
        "defaultdict": defaultdict,
        "Any": Any,
        "Mapping": Mapping,
        "Sequence": Sequence,
    }

    def stub(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {}

    for node in ast.parse(engine).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            namespace[node.name] = stub
        elif isinstance(node, ast.Assign):
            try:
                value = ast.literal_eval(node.value)
            except Exception:
                value = None
            for target in node.targets:
                if isinstance(target, ast.Name):
                    namespace[target.id] = (
                        "" if value is None and target.id.isupper() else value
                    )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            try:
                value = ast.literal_eval(node.value)
            except Exception:
                value = None
            namespace[node.target.id] = (
                "" if value is None and node.target.id.isupper() else value
            )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                namespace.setdefault(alias.asname or alias.name.split(".")[0], types.SimpleNamespace())
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                namespace.setdefault(alias.asname or alias.name, types.SimpleNamespace())

    snapshot_names = (
        "_FUSE_QUERY_RESULTS_V4_1",
        "_VALIDATE_DC_SKELETON_GENERAL_V1",
        "_GENERATE_DC_TWOCALL_GENERAL_V1",
        "_EXECUTE_DC_GENERAL_V1",
        "_PARSE_ONECALL_STRICT_BEFORE_FALLBACK_V2",
        "_GENERATE_DC_ONECALL_BEFORE_SAFETY_V2",
        "_GENERATE_DC_TWOCALL_BEFORE_SAFETY_V2",
        "_EXECUTE_DC_BEFORE_SAFETY_AUDIT_V2",
        "_PARSE_ONECALL_BEFORE_TAGGED_FALLBACK_V3",
        "_GENERATE_DC_ONECALL_BEFORE_RELATION_GUARD_V3",
        "_GENERATE_DC_TWOCALL_BEFORE_RELATION_GUARD_V3",
        "_ACTION_LINKS_MARKDOWN_BEFORE_NOTICE_REMOVAL_V3",
        "_EXECUTE_DC_BEFORE_V3_AUDIT",
        "_EXECUTE_DC_BEFORE_C_THREEWAY_V1",
    )
    sentinels = {
        name: (lambda *args, _name=name, **kwargs: ({"baseline": _name}, []))
        for name in snapshot_names
    }
    namespace.update(sentinels)
    namespace["normalize_answer_markdown_v3"] = lambda text: str(text or "")
    sys.path.insert(0, str(BASE_DIR))
    exec(compile(overlay, str(OVERLAY_FILE), "exec"), namespace, namespace)
    required = (
        "execute_production_variant_v1",
        "audit_need_evidence_pack_v5",
        "is_cross_business_dc_v1",
        "generate_c_direct_threeway_v1",
        "order_businesses_by_question_p0_v1",
        "build_p0_cross_business_subqueries_v1",
        "_current_question_scoped_analysis_v1",
        "generate_user_basis_explanation_v1",
    )
    assert all(callable(namespace.get(name)) for name in required)
    unresolved: dict[str, list[str]] = {}
    for name, value in namespace.items():
        code = getattr(value, "__code__", None)
        if code is None or Path(code.co_filename).name != OVERLAY_FILE.name:
            continue
        missing = sorted({
            str(instruction.argval)
            for instruction in dis.get_instructions(value)
            if instruction.opname in {"LOAD_GLOBAL", "LOAD_NAME"}
            and str(instruction.argval) not in namespace
            and not hasattr(builtins, str(instruction.argval))
        })
        if missing:
            unresolved[name] = missing
    assert not unresolved, unresolved
    assert all(namespace[name] is sentinel for name, sentinel in sentinels.items())
    single_result = namespace["fuse_query_results"]([
        {"query": "고객 미수령금 조회", "weight": 1.0, "source": "ORIGINAL"}
    ])
    assert single_result[0]["baseline"] == "_FUSE_QUERY_RESULTS_V4_1"
    assert namespace["KDIC_PRODUCTION_OVERLAY_POLICY"] == "C_DEFAULT_DC2_COMPARE_ONLY_V1"
    normalized = namespace["normalize_answer_markdown_v3"](
        "고객센터 02-1588-0037 또는 02-758-1000"
    )
    assert normalized == "고객센터 1588-0037 또는 02-758-1000"

    class StructuredBasisStub:
        def __init__(self) -> None:
            self.parsed: dict[str, Any] = {}
            self.calls: list[dict[str, Any]] = []

        def _call_structured(self, **kwargs: Any):
            self.calls.append(kwargs)
            return copy.deepcopy(self.parsed), {}, 0.0, []

    base_basis = {
        "schema_version": "kdic-basis-explanation-v2",
        "summary": "착오송금 반환지원 공식 근거를 확인했습니다.",
        "items": [{
            "answer_point": "신청 대상을 확인했습니다.",
            "evidence_summary": "공식 안내에 신청 대상이 제시되어 있습니다.",
            "user_meaning": "본인이 대상에 해당하는지 확인할 수 있습니다.",
            "caveat": "개별 조건은 공식 안내에서 확인해 주세요.",
            "evidence_ids": ["E1"],
        }],
        "checkpoints": ["신청 전에 대상 조건을 확인해 주세요."],
        "sources": [{"title": "공식 안내", "url": "https://www.kdic.or.kr/example"}],
    }
    raw_result = {
        "answer": "공식 신청 대상에 해당하면 반환지원을 신청할 수 있습니다.",
        "payload": {"used_evidence_ids": ["E1"]},
    }
    namespace["_clean_text"] = lambda value: str(value or "").strip()
    namespace["_compact_json"] = lambda value: json.dumps(
        value, ensure_ascii=False, separators=(",", ":")
    )
    basis_stub = StructuredBasisStub()
    namespace["answer_b_core"] = basis_stub
    basis_stub.parsed = {
        "summary": "신청 가능 여부를 공식 대상 기준과 연결해 설명했습니다.",
        "items": [{
            "answer_point": "신청 대상 여부를 먼저 확인합니다.",
            "evidence_summary": "공식 안내에 신청 대상이 제시되어 있습니다.",
            "user_meaning": "본인의 조건을 공식 대상 기준과 대조하면 됩니다.",
            "caveat": "개별 조건은 공식 안내에서 확인해 주세요.",
            "evidence_indices": [1],
        }],
        "checkpoints": ["신청 전에 대상 조건을 확인해 주세요."],
    }
    explained = namespace["generate_user_basis_explanation_v1"](raw_result, base_basis)
    assert explained["items"][0]["evidence_ids"] == ["E1"]
    assert "본인의 조건" in explained["items"][0]["user_meaning"]
    assert basis_stub.calls[-1]["system_prompt"] == namespace["BASIS_EXPLAINER_SYSTEM_PROMPT_V1"]
    assert basis_stub.calls[-1]["schema"] == namespace["BASIS_EXPLANATION_SCHEMA_V1"]

    basis_stub.parsed["items"][0]["user_meaning"] = "999원 조건을 새로 적용합니다."
    unsupported_number = namespace["generate_user_basis_explanation_v1"](raw_result, base_basis)
    assert unsupported_number["summary"] == base_basis["summary"]
    assert "999원" not in json.dumps(unsupported_number, ensure_ascii=False)

    basis_stub.parsed["items"][0]["user_meaning"] = "자세한 내용은 https://example.com 에서 봅니다."
    unsupported_url = namespace["generate_user_basis_explanation_v1"](raw_result, base_basis)
    assert unsupported_url["summary"] == base_basis["summary"]
    assert "example.com" not in json.dumps(unsupported_url, ensure_ascii=False)

    basis_stub.parsed["items"][0]["user_meaning"] = "MT-005_chunk_000 근거를 사용했습니다."
    internal_id = namespace["generate_user_basis_explanation_v1"](raw_result, base_basis)
    assert internal_id["summary"] == base_basis["summary"]
    assert "chunk_000" not in json.dumps(internal_id, ensure_ascii=False)
    return {
        "top_level_wiring": "passed",
        "required_functions": list(required),
        "official_contact_guard": "passed",
        "basis_explanation": "validated_with_safe_fallback",
    }


def test_chat_ui_numbering_contract() -> dict[str, Any]:
    ui = CHAT_UI_FILE.read_text(encoding="utf-8")
    assert r"const ordered=line.match(/^(\d{1,2})[.)]\s+(.+)$/);" in ui
    assert 'html+=`<ol start="${number}">`' in ui
    assert 'html+=`<li value="${number}">${inline(ordered[2])}</li>`' in ui
    assert "inline(ordered[1])" not in ui
    assert "const PROGRESS_STAGE_DWELL_MS=280,CACHE_STAGE_DWELL_MS=140" in ui
    assert "async function completeProgress(row)" in ui
    assert "async function completeCachedProgress(row)" in ui
    assert "async function advanceProgress(row,p,stage" in ui
    assert "if(cacheHit)await completeCachedProgress(row);else await completeProgress(row)" in ui
    assert "키워드와 저장 질의를 확인하고 있어요" in ui
    assert "공식 근거와 조건을 검증하고 있어요" in ui
    assert "Math.max(previous" in ui
    assert "1/4단계" in ui
    assert "function summaryHtml(d)" in ui
    assert "async function loadSummary" in ui
    assert "function stripDocumentFormattingArtifacts(text='')" in ui
    assert "핵심 요약 보기" in ui
    assert "핵심 요약 접기" in ui
    assert "핵심만 정리했어요" in ui
    assert "summary-points" in ui
    assert "api('/api/summary'" in ui
    assert 'class="action-btn summary-btn"' in ui
    assert "if(summary)summary.onclick=()=>loadSummary(summary,row.querySelector('.summary-slot'),jobId)" in ui
    assert ".filter(Boolean).slice(0,3)" in ui
    assert "${esc(title)}" in ui
    assert "${esc(point)}" in ui
    assert "if(slot.dataset.loaded)" in ui
    assert "요약할 핵심 내용을 찾지 못했어요" in ui
    assert "api('/api/basis'" not in ui
    assert "function basisHtml(d)" not in ui
    assert "답변을 쉽게 이해하기" not in ui
    assert "근거가 된 공식 정보" not in ui
    assert "사용자에게 어떤 의미인가요?" not in ui
    assert "추가로 확인해 주세요" not in ui
    assert "function sourceLinksHtml(sources,visible=3)" in ui
    assert "function bindSourceToggles(root)" in ui
    assert "sourceLinksHtml(sources,3)" in ui
    assert "function scrollAnswerTop(node)" in ui
    assert "input.focus({preventScroll:true})" in ui
    assert "window.__KDIC_PROGRESS_PREVIEW=completeProgress(previewRow)" in ui
    assert "else if(preview==='cache-progress')" in ui
    assert "else if(preview==='summary'||preview==='basis')" in ui
    return {
        "ordered_list_start_preserved": "passed",
        "ordered_list_item_value_preserved": "passed",
        "visible_four_stage_progress": "passed",
        "user_answer_summary": "passed",
    }


def test_c_direct_json_validator() -> dict[str, Any]:
    overlay = OVERLAY_FILE.read_text(encoding="utf-8")
    answer_core = _load_module(
        "kdic_answer_core_validator_test", BASE_DIR / "kdic_v15_answer_b_core.py"
    )
    namespace: dict[str, Any] = {
        "Any": Any,
        "Mapping": Mapping,
        "re": re,
        "answer_b_core": answer_core,
        "_allowed_fact_claims_v3": lambda pack: {},
        "_clean_fact_claim_keys_v3": lambda values: list(values or []),
    }
    exec(_function_source(overlay, "_validate_c_direct_json_v2"), namespace)
    validate = namespace["_validate_c_direct_json_v2"]
    pack = {"evidence": [{"evidence_id": "E1", "chunk_id": "C1"}]}
    payload = {
        "response_mode": "SEPARATE",
        "answer": "공식 근거에 따른 안내입니다. [E1]",
        "used_evidence_ids": ["E1"],
        "used_fact_claim_ids": [],
        "coverage_status": "SUFFICIENT",
        "missing_information": [],
    }
    assert validate(payload, pack, "SEPARATE")["used_evidence_ids"] == ["E1"]
    declared_only = copy.deepcopy(payload)
    declared_only["answer"] = "선언된 공식 근거에 따른 답변입니다."
    declared_result = validate(declared_only, pack, "SEPARATE")
    assert declared_result["used_evidence_ids"] == ["E1"]
    assert declared_result["answer"].endswith("[E1]")
    invalid_cases = []
    missing_all = copy.deepcopy(payload)
    missing_all["answer"] = "근거 ID가 없는 답변"
    missing_all["used_evidence_ids"] = []
    invalid_cases.append(("missing_all_evidence", missing_all))
    wrong_mode = copy.deepcopy(payload)
    wrong_mode["response_mode"] = "COMPARE"
    invalid_cases.append(("response_mode", wrong_mode))
    invalid_id = copy.deepcopy(payload)
    invalid_id["answer"] = "허용되지 않은 근거를 선언한 답변"
    invalid_id["used_evidence_ids"] = ["E9"]
    invalid_cases.append(("invalid_evidence_id", invalid_id))
    for key, invalid in invalid_cases:
        try:
            validate(invalid, pack, "SEPARATE")
        except ValueError:
            continue
        raise AssertionError(f"invalid C direct payload was accepted: {key}")
    return {
        "valid_json_and_inline_evidence": "passed",
        "declared_ids_canonicalized_into_answer": "passed",
        "missing_inline_evidence": "blocked",
        "wrong_response_mode": "blocked",
    }


def _evidence_namespace(overlay: str) -> dict[str, Any]:
    namespace: dict[str, Any] = {
        "Any": Any,
        "Mapping": Mapping,
        "Sequence": Sequence,
        "defaultdict": defaultdict,
        "NEED_BATCH_MAX_BUSINESSES_V5": 3,
        "NEED_BATCH_REQUIRED_TOP_K_V5": 3,
        "NEED_BATCH_MIN_DISTINCT_PARENTS_V5": 2,
        "NEED_BATCH_MAX_EVIDENCE_PER_PARENT_V5": 2,
        "NEED_BATCH_SCORE_GATE_ENABLED_V5": False,
        "NEED_BATCH_MIN_RERANKER_SCORE_V5": 0.25,
        "_clean_text": lambda value: " ".join(str(value or "").split()).strip(),
    }
    for name in (
        "need_score_gate_passed_v5",
        "need_score_gate_metadata_v5",
        "audit_need_evidence_pack_v5",
        "_make_need_gate_fixture_v5",
    ):
        exec(_function_source(overlay, name), namespace)
    return namespace


def test_evidence_gate() -> dict[str, Any]:
    overlay = OVERLAY_FILE.read_text(encoding="utf-8")
    namespace = _evidence_namespace(overlay)
    fixture = namespace["_make_need_gate_fixture_v5"]
    audit = namespace["audit_need_evidence_pack_v5"]

    two_business = fixture(["업무A", "업무B"])
    assert audit(two_business)["passed"] is True
    assert len({row["source_url"] for row in two_business["evidence"]}) == 1

    low_score = fixture(["업무A", "업무B"], low_score=(1, 0, 0.249))
    assert audit(low_score)["passed"] is True
    namespace["NEED_BATCH_SCORE_GATE_ENABLED_V5"] = True
    assert audit(low_score)["passed"] is False
    namespace["NEED_BATCH_SCORE_GATE_ENABLED_V5"] = False

    assert audit(fixture(["업무A", "업무B"], counts=[2, 3]))["passed"] is False
    assert audit(fixture(["업무A", "업무B", "업무C"]))["passed"] is True
    four = audit(fixture(["업무A", "업무B", "업무C", "업무D"]))
    assert four["passed"] is False and four["capacity_exceeded"] is True
    assert audit(fixture(["업무A", ""]))["passed"] is False
    assert audit(fixture(["업무A", "업무A"]))["passed"] is False

    parent_fallback = fixture(["업무A", "업무B"])
    for business_index in (0, 1):
        rows = parent_fallback["evidence"][business_index * 3 : business_index * 3 + 3]
        rows[2]["parent_id"] = rows[0]["parent_id"]
        rows[2]["parent_fallback"] = True
        rows[2]["selection_reasons"] = ["PARENT_FALLBACK"]
    parent_audit = audit(parent_fallback)
    assert parent_audit["passed"] is True and parent_audit["parent_fallback_used"] is True

    missing_required = copy.deepcopy(two_business)
    missing_required["evidence"][0]["selection_types"] = []
    assert audit(missing_required)["passed"] is False
    return {
        "two_business_3x3": "passed",
        "three_business_3x3": "passed",
        "one_short": "blocked",
        "four_business_capacity": "blocked",
        "blank_duplicate_labels": "blocked",
        "same_url_different_parent": "passed",
        "parent_fallback": "passed",
        "score_gate_off_0_249": "passed",
        "score_gate_on_0_25": "blocked",
    }


def test_routing_and_cross_structure() -> dict[str, Any]:
    overlay = OVERLAY_FILE.read_text(encoding="utf-8")
    namespace = _evidence_namespace(overlay)
    fixture = namespace["_make_need_gate_fixture_v5"]
    audit = namespace["audit_need_evidence_pack_v5"]
    cross_namespace: dict[str, Any] = {
        "Any": Any,
        "Mapping": Mapping,
        "NEED_BATCH_MAX_BUSINESSES_V5": 3,
        "NEED_BATCH_REQUIRED_TOP_K_V5": 3,
        "_clean_text": lambda value: " ".join(str(value or "").split()).strip(),
        "audit_need_evidence_pack_v5": audit,
        "need_score_gate_metadata_v5": namespace["need_score_gate_metadata_v5"],
        "order_businesses_by_question_p0_v1": lambda question, values: list(values),
        "light_router": types.SimpleNamespace(find_businesses=lambda question: []),
    }
    exec(_function_source(overlay, "is_cross_business_dc_v1"), cross_namespace)
    is_cross = cross_namespace["is_cross_business_dc_v1"]

    valid_pack = fixture(["업무A", "업무B"])
    passed, cross_audit = is_cross({
        "route": "RETRIEVE",
        "analysis": {"businesses": ["업무A", "업무B"], "p0_cross_preserved": True},
        "evidence_pack": valid_pack,
        "evidence_quota_audit": audit(valid_pack),
    })
    assert passed is True and cross_audit["structure_valid"] is True
    for invalid_pack in (fixture(["업무A", ""]), fixture(["업무A", "업무A"])):
        passed, cross_audit = is_cross({
            "route": "RETRIEVE",
            "analysis": {"businesses": ["업무A", "업무B"], "p0_cross_preserved": True},
            "evidence_pack": invalid_pack,
            "evidence_quota_audit": audit(invalid_pack),
        })
        assert passed is False
        assert cross_audit["cross_business_candidate"] is True
        assert cross_audit["structure_valid"] is False

    route_namespace: dict[str, Any] = {"Any": Any}
    exec(_function_source(overlay, "select_production_variant_v1"), route_namespace)
    select = route_namespace["select_production_variant_v1"]
    cases = (
        ("SEPARATE", False, False, True, "C_1CALL"),
        ("RELATION", True, True, True, "C_1CALL"),
        ("SEQUENCE", True, True, True, "C_1CALL"),
        ("COMPARE", True, True, True, "DC_2CALL"),
        ("COMPARE", True, False, False, "C_1CALL"),
    )
    selected = [select(
        common_route="RETRIEVE",
        response_mode=mode,
        cross_business_candidate=candidate,
        cross_business_passed=gate,
        structure_valid=structure,
    )[0] for mode, candidate, gate, structure, expected in cases]
    assert selected == [expected for *_, expected in cases]
    assert "DC_1CALL" not in selected
    return {
        "cross_structure": "passed",
        "blank_duplicate_bypass": "blocked",
        "c_default": "passed",
        "dc2_compare_only": "passed",
        "dc1_enabled": False,
    }


def test_current_question_business_mapping() -> dict[str, Any]:
    overlay = OVERLAY_FILE.read_text(encoding="utf-8")
    router = _load_module("kdic_lightweight_router_v1", BASE_DIR / "kdic_lightweight_router_v1.py")
    namespace: dict[str, Any] = {
        "Any": Any,
        "Mapping": Mapping,
        "Sequence": Sequence,
        "light_router": router,
        "re": re,
        "_clean_text": lambda value: " ".join(str(value or "").split()).strip(),
        "NEED_BATCH_MAX_BUSINESSES_V5": 3,
        "V15_ORIGINAL_WEIGHT": 0.40,
        "V15_SUBQUERY_TOTAL_WEIGHT": 0.60,
        "HOW_COMPARE_PATTERN_V1": re.compile(r"어떻게\s*(?:다르|다른|달라|구분|비교)|어떤\s*차이", re.I),
    }
    exec(_function_source(overlay, "_need_business_v5"), namespace)
    exec(_function_source(overlay, "_cross_business_scope_count_v5"), namespace)
    exec(_function_source(overlay, "order_businesses_by_question_p0_v1"), namespace)
    exec(_function_source(overlay, "_business_local_segments_p0_v1"), namespace)
    exec(_function_source(overlay, "_business_need_topic_p0_v1"), namespace)
    exec(_function_source(overlay, "build_p0_cross_business_subqueries_v1"), namespace)
    exec(_function_source(overlay, "_current_question_scoped_analysis_v1"), namespace)
    need_business = namespace["_need_business_v5"]
    scope_count = namespace["_cross_business_scope_count_v5"]
    order_businesses = namespace["order_businesses_by_question_p0_v1"]
    scope_analysis = namespace["_current_question_scoped_analysis_v1"]

    atomic_queries = (
        "고객 미수령금 신청 방법",
        "착오송금 반환지원 신청 자격",
        "예금자보호 한도",
        "채무조정 신청 서류",
        "예금보험금 지급 조건",
        "은닉재산 신고 포상금",
    )
    mapped = [need_business(query) for query in atomic_queries]
    assert all(mapped), mapped
    reversed_values = ["은닉재산 신고", "예금보험금 안내"]
    assert order_businesses(
        "예금보험금 지급 조건과 은닉재산 신고 포상금",
        reversed_values,
    ) == ["예금보험금 안내", "은닉재산 신고"]

    two_business_question = "고객 미수령금 조회 방법과 채무조정 신청 서류를 함께 알려주세요."
    plans = [
        {"source": "ORIGINAL_ANCHOR", "query": two_business_question},
        {"source": "DECOMPOSED", "query": "고객 미수령금 조회 방법"},
        {"source": "DECOMPOSED", "query": "채무조정 신청 서류"},
    ]
    contaminated_analysis = {
        "route": "RETRIEVE",
        "resolved_question": "이전 네 업무가 합쳐진 질문",
        "businesses": [
            "예금자보호제도",
            "고객 미수령금 신청",
            "착오송금 반환 신청",
            "채무조정 안내",
        ]
    }
    count, businesses, decomposed = scope_count(
        contaminated_analysis,
        plans,
        two_business_question,
    )
    assert count == 2, (count, businesses)
    assert decomposed == 2
    assert businesses == ["고객 미수령금 신청", "채무조정 안내"], businesses
    repaired = scope_analysis(two_business_question, contaminated_analysis)
    assert repaired["resolved_question"] == two_business_question
    assert repaired["businesses"] == ["고객 미수령금 신청", "채무조정 안내"]
    assert repaired["context_used"] is False
    assert [row["source"] for row in repaired["plans"]] == [
        "ORIGINAL_ANCHOR",
        "P0_RULE_DECOMPOSED",
        "P0_RULE_DECOMPOSED",
    ]
    assert [router.find_businesses(row["query"]) for row in repaired["plans"][1:]] == [
        ["고객 미수령금 신청"],
        ["채무조정 안내"],
    ]

    single_question = "고객 미수령금은 어디에서 조회할 수 있나요?"
    repaired_single = scope_analysis(single_question, contaminated_analysis)
    assert repaired_single["businesses"] == ["고객 미수령금 신청"]
    assert repaired_single["resolved_question"] == single_question
    assert repaired_single["plans"] == [
        {"query": single_question, "weight": 1.0, "source": "ORIGINAL"}
    ]

    four_business_question = (
        "예금자보호 한도, 미수령금 신청, 착오송금 반환지원, 채무조정 서류를 알려주세요."
    )
    four_plans = [
        {"source": "ORIGINAL_ANCHOR", "query": four_business_question},
        *[
            {"source": "DECOMPOSED", "query": query}
            for query in (
                "예금자보호 한도",
                "미수령금 신청",
                "착오송금 반환지원",
                "채무조정 서류",
            )
        ],
    ]
    count, businesses, decomposed = scope_count({}, four_plans, four_business_question)
    assert count == 4, (count, businesses)
    assert decomposed == 4
    return {
        "router_import": "passed",
        "atomic_need_business_labels": "passed",
        "two_business_after_context_history": "passed",
        "explicit_current_scope_repair": "passed",
        "single_business_original_plan": "passed",
        "four_business_capacity": "blocked",
    }


def test_adapter() -> dict[str, Any]:
    module = _load_module("kdic_ec2_adapter_test", ADAPTER_FILE)
    calls: list[str] = []
    holders: list[dict[str, Any]] = []

    class PromptVersionStub:
        version = "prompt-active-v1"

        def public(self) -> dict[str, str]:
            return {"active_version": self.version}

    prompt_manager = PromptVersionStub()

    def holder() -> dict[str, Any]:
        return {"conversation": {"turns": []}, "answer_cache": {}, "events": []}

    def production(question: str, state: dict[str, Any]) -> dict[str, Any]:
        variant = "DC_2CALL" if "비교" in question else "C_1CALL"
        calls.append(variant)
        holders.append(state)
        return {
            "variant": variant,
            "route": "RETRIEVE",
            "payload": {"answer": "offline"},
            "production_routing": {"selected_variant": variant},
        }

    adapter = module.build_latest_kdic_pipeline({
        "new_dc_controller_state_v1": holder,
        "execute_production_variant_v1": production,
        "KDIC_PROMPT_MANAGER": prompt_manager,
    })
    first = adapter("단일질의", {})
    second = adapter("업무 비교", {})
    stale_state = {
        "_kdic_controller": {
            "_runtime_revision": "stale-revision",
            "current_question": "단일질의",
            "common": {"route": "EVIDENCE_INSUFFICIENT"},
        }
    }
    adapter("단일질의", stale_state)
    assert calls == ["C_1CALL", "DC_2CALL", "C_1CALL"]
    assert first["runtime_build"]["dc1_enabled"] is False
    assert second["routing_policy"] == "C_DEFAULT_DC2_COMPARE_ONLY_V1"
    refreshed = stale_state["_kdic_controller"]
    assert refreshed.get("_runtime_revision") == adapter.build_info["overlay_revision"]
    assert refreshed.get("common") != {"route": "EVIDENCE_INSUFFICIENT"}
    assert holders[-1] is refreshed
    assert adapter.answer_cache_revision == "prompt-active-v1"
    prompt_manager.version = "prompt-active-v2"
    assert adapter.answer_cache_revision == "prompt-active-v2"
    return {
        "c_default": "passed",
        "dc2_compare": "passed",
        "stale_controller_cache": "invalidated",
        "build_info": "passed",
        "prompt_cache_revision": "passed",
    }


def main() -> None:
    result = {
        "static": test_static_contracts(),
        "overlay_exec": test_overlay_exec_contract(),
        "managed_prompts": test_managed_prompt_contract(),
        "chat_ui": test_chat_ui_numbering_contract(),
        "c_direct_json": test_c_direct_json_validator(),
        "evidence": test_evidence_gate(),
        "routing": test_routing_and_cross_structure(),
        "business_mapping": test_current_question_business_mapping(),
        "adapter": test_adapter(),
    }
    print(json.dumps({"status": "passed", "result": result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
