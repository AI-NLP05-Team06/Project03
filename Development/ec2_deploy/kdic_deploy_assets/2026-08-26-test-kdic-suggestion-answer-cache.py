from __future__ import annotations

import ast
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping


BASE_DIR = Path(__file__).resolve().parent
CORE_FILE = BASE_DIR / "2026-08-23-kdic-service-core.py"
API_FILE = BASE_DIR / "2026-08-23-kdic-fastapi-service.py"
UI_FILE = BASE_DIR / "2026-08-23-kdic-chat-ui.html"
ADAPTER_FILE = BASE_DIR / "2026-08-23-kdic-colab-runtime-adapter.py"
POSTGRES_FILE = BASE_DIR / "kdic_postgres_store.py"
PREWARM_FILE = BASE_DIR / "2026-08-26-prewarm-kdic-suggestion-answer-cache.py"
MIGRATION_FILE = BASE_DIR.parent / "2026-08-26-kdic-suggestion-answer-cache-migration.sql"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FakePipeline:
    name = "V1.5_C_DEFAULT_DC2_COMPARE_ONLY"

    def __init__(self, revision: str = "revision-a"):
        self.build_info = {
            "build_sha256": "build-sha",
            "overlay_revision": revision,
        }
        self.calls = 0
        self.cached_turns: list[tuple[str, str]] = []

    def __call__(self, question: str, state: dict[str, Any], progress=None):
        self.calls += 1
        return {
            "route": "RETRIEVE",
            "answer": "공식 근거를 사용한 검증 답변입니다.",
            "analysis": {"businesses": ["착오송금 반환 신청"]},
            "sources": [
                {
                    "title": "예금보험공사 공식 안내",
                    "url": "https://www.kdic.or.kr/example",
                }
            ],
            "payload": {
                "coverage_status": "SUFFICIENT",
                "validation_passed": True,
            },
        }

    def basis(self, result: Mapping[str, Any]) -> dict[str, Any]:
        return {"summary": "공식 근거 요약", "sources": result.get("sources") or []}

    def record_cached_turn(
        self,
        question: str,
        answer: str,
        state: dict[str, Any],
    ) -> None:
        self.cached_turns.append((question, answer))
        state.setdefault("turns", []).extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        )


def _wait_job(service, job_id: str, timeout: float = 3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        row = service.jobs.get(job_id)
        if row is not None and row.status in {"done", "error"}:
            return row
        time.sleep(0.01)
    raise TimeoutError(job_id)


def test_static_contracts() -> dict[str, str]:
    for path in (CORE_FILE, API_FILE, ADAPTER_FILE, POSTGRES_FILE, PREWARM_FILE):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    api = API_FILE.read_text(encoding="utf-8")
    ui = UI_FILE.read_text(encoding="utf-8")
    postgres = POSTGRES_FILE.read_text(encoding="utf-8")
    prewarm = PREWARM_FILE.read_text(encoding="utf-8")
    migration = MIGRATION_FILE.read_text(encoding="utf-8")
    assert "suggestion_id: str = Field" in api
    assert "suggestion_cache=SUGGESTION_ANSWER_CACHE" in api
    assert 'data-suggestion-id="${esc(x.suggestion_id||\'\')}"' in ui
    assert "suggestion_id:String(suggestionId||'')" in ui
    assert "⚡ 저장된 빠른 답변" not in ui
    assert "0.01초 미만" not in ui
    assert 'class="boot-screen" id="bootScreen"' in ui
    assert 'class="modal-backdrop hidden" id="keyModal"' in ui
    assert "async function bootstrapApp()" in ui
    assert "catch(e){showBootError()}" in ui
    assert "$('#bootRetry').onclick=bootstrapApp" in ui
    assert "catch(e){ openKey(); }" not in ui
    assert "class PostgresSuggestionAnswerCache" in postgres
    assert "CREATE TABLE IF NOT EXISTS suggestion_answer_cache" in migration
    assert "compatible_overlay_revisions" in prewarm
    return {
        "python_syntax": "passed",
        "api_ui_wiring": "passed",
        "postgres_migration": "passed",
    }


def test_registry(core) -> dict[str, Any]:
    catalog = core.suggestion_catalog()
    assert len(catalog) == 26
    assert len({row["suggestion_id"] for row in catalog}) == 26
    assert len({row["query"] for row in catalog}) == 26
    assert len({row["business_key"] for row in catalog}) == 6
    by_key = {(row["business_key"], row["label"]): row for row in catalog}
    assert by_key[("예금자보호", "제외 상품")]["query"].startswith("예금자보호제도에서")
    assert "본인 명의" in by_key[("미수령금", "조회 방법")]["query"]
    assert "본인 명의" in by_key[("미수령금", "필요 서류")]["query"]
    assert "1인당 최대 금액" in by_key[("예금보험금", "보호 한도")]["query"]
    for row in catalog:
        assert core.resolve_registered_suggestion(
            row["query"], row["suggestion_id"]
        ) == row
        assert core.resolve_registered_suggestion(
            row["query"] + " 변경", row["suggestion_id"]
        ) is None
        assert core.resolve_registered_suggestion(row["query"], "SQ-FORGED") is None
    return {
        "catalog_rows": len(catalog),
        "unique_ids": "passed",
        "exact_id_and_query": "passed",
        "forged_cache_access": "blocked",
    }


def test_memory_cache_flow(core) -> dict[str, str]:
    pipeline = FakePipeline()
    runtime = core.PipelineRuntime(pipeline)
    sessions = core.InMemorySessionStore()
    jobs = core.InMemoryJobStore()
    cache = core.InMemorySuggestionAnswerCache(ttl_seconds=3600, max_entries=100)
    service = core.KDICJobService(
        runtime=runtime,
        sessions=sessions,
        jobs=jobs,
        suggestion_cache=cache,
        max_workers=1,
    )
    first, second = core.suggestion_catalog()[:2]

    live_job = _wait_job(
        service,
        service.submit("live-session", first["query"], first["suggestion_id"]),
    )
    assert live_job.status == "done"
    assert live_job.result["suggestion_cache"] == {
        "eligible": True,
        "hit": False,
        "stored": True,
        "suggestion_id": first["suggestion_id"],
        "source": "LIVE_PIPELINE",
        "reason": "VALIDATED_STANDALONE_RETRIEVE",
        "skipped_stages": [],
    }
    assert pipeline.calls == 1

    hit_id = service.submit("hit-session", first["query"], first["suggestion_id"])
    hit_job = jobs.get(hit_id)
    assert hit_job is not None and hit_job.status == "done"
    assert hit_job.result["suggestion_cache"]["hit"] is True
    assert pipeline.calls == 1
    assert pipeline.cached_turns[-1][0] == first["query"]
    assert len(sessions.get("hit-session").state["turns"]) == 2
    assert service.basis(hit_id)["summary"] == "공식 근거 요약"

    forged_job = _wait_job(
        service,
        service.submit("forged-session", second["query"], first["suggestion_id"]),
    )
    assert forged_job.status == "done"
    assert "suggestion_cache" not in forged_job.result
    assert pipeline.calls == 2

    other_pipeline = FakePipeline(revision="revision-b")
    other_service = core.KDICJobService(
        runtime=core.PipelineRuntime(other_pipeline),
        suggestion_cache=cache,
        max_workers=1,
    )
    assert service._cache_key(first["suggestion_id"]) != other_service._cache_key(
        first["suggestion_id"]
    )
    service.shutdown()
    other_service.shutdown()
    return {
        "first_click_store": "passed",
        "next_click_hit_without_pipeline": "passed",
        "cached_turn_context": "passed",
        "cached_basis": "passed",
        "forged_id_query_pair": "live_fallback",
        "runtime_revision_invalidation": "passed",
    }


def test_adapter_cached_turn() -> dict[str, str]:
    adapter_module = _load_module("kdic_cache_adapter_test", ADAPTER_FILE)
    adapter = adapter_module.LatestKDICNotebookAdapter(
        {"execute_production_variant_v1": lambda question, holder: {}}
    )
    state: dict[str, Any] = {}
    adapter.record_cached_turn("질문", "답변", state)
    holder = state["_kdic_controller"]
    assert holder["conversation"]["turns"] == [
        {"role": "user", "content": "질문"},
        {"role": "assistant", "content": "답변"},
    ]
    assert holder["committed_variant"] == "SUGGESTION_ANSWER_CACHE"
    return {"adapter_context_commit": "passed"}


def main() -> None:
    core = _load_module("kdic_cache_core_test", CORE_FILE)
    result = {
        "static": test_static_contracts(),
        "registry": test_registry(core),
        "cache_flow": test_memory_cache_flow(core),
        "adapter": test_adapter_cached_turn(),
    }
    print(json.dumps({"status": "passed", "result": result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
