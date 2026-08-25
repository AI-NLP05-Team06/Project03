from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import re
import sys
import time
import types
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


BASE_DIR = Path(__file__).resolve().parent
ENGINE_FILE = BASE_DIR / "kdic_pipeline_engine.py"
OVERLAY_FILE = BASE_DIR / "2026-08-25-kdic-production-overlay.py"
ADAPTER_FILE = BASE_DIR / "2026-08-23-kdic-colab-runtime-adapter.py"
FASTAPI_FILE = BASE_DIR / "2026-08-23-kdic-fastapi-service.py"
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
    for path, source in (
        (ENGINE_FILE, engine),
        (OVERLAY_FILE, overlay),
        (ADAPTER_FILE, adapter),
        (FASTAPI_FILE, api),
    ):
        ast.parse(source, filename=str(path))
    assert "2026-08-25-kdic-production-overlay.py" in engine
    assert "execute_production_variant_v1" in overlay
    assert "C_DEFAULT_DC2_COMPARE_ONLY_V1" in overlay
    assert "DC_1CALL is disabled" in overlay
    assert 'name = "V1.5_C_DEFAULT_DC2_COMPARE_ONLY"' in adapter
    assert '"runtime_build": dict(RUNTIME_BUILD_INFO)' in api
    assert EXPECTED_SOURCE_SHA256 in overlay
    assert EXPECTED_SOURCE_SHA256 in adapter
    return {
        "engine_overlay_loader": "passed",
        "overlay_syntax": "passed",
        "adapter_syntax": "passed",
        "fastapi_build_contract": "passed",
        "overlay_sha256": hashlib.sha256(OVERLAY_FILE.read_bytes()).hexdigest(),
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

    sys.path.insert(0, str(BASE_DIR))
    exec(compile(overlay, str(OVERLAY_FILE), "exec"), namespace, namespace)
    required = (
        "execute_production_variant_v1",
        "audit_need_evidence_pack_v5",
        "is_cross_business_dc_v1",
        "generate_c_direct_threeway_v1",
    )
    assert all(callable(namespace.get(name)) for name in required)
    assert namespace["KDIC_PRODUCTION_OVERLAY_POLICY"] == "C_DEFAULT_DC2_COMPARE_ONLY_V1"
    return {"top_level_wiring": "passed", "required_functions": list(required)}


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


def test_adapter() -> dict[str, Any]:
    module = _load_module("kdic_ec2_adapter_test", ADAPTER_FILE)
    calls: list[str] = []

    def holder() -> dict[str, Any]:
        return {"conversation": {"turns": []}, "answer_cache": {}, "events": []}

    def production(question: str, state: dict[str, Any]) -> dict[str, Any]:
        variant = "DC_2CALL" if "비교" in question else "C_1CALL"
        calls.append(variant)
        return {
            "variant": variant,
            "route": "RETRIEVE",
            "payload": {"answer": "offline"},
            "production_routing": {"selected_variant": variant},
        }

    adapter = module.build_latest_kdic_pipeline({
        "new_dc_controller_state_v1": holder,
        "execute_production_variant_v1": production,
    })
    first = adapter("단일질의", {})
    second = adapter("업무 비교", {})
    assert calls == ["C_1CALL", "DC_2CALL"]
    assert first["runtime_build"]["dc1_enabled"] is False
    assert second["routing_policy"] == "C_DEFAULT_DC2_COMPARE_ONLY_V1"
    return {"c_default": "passed", "dc2_compare": "passed", "build_info": "passed"}


def main() -> None:
    result = {
        "static": test_static_contracts(),
        "overlay_exec": test_overlay_exec_contract(),
        "evidence": test_evidence_gate(),
        "routing": test_routing_and_cross_structure(),
        "adapter": test_adapter(),
    }
    print(json.dumps({"status": "passed", "result": result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
