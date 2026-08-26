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

    def __init__(self, revision: str = "revision-a", prompt_revision: str = "prompt-v1"):
        self.build_info = {
            "build_sha256": "build-sha",
            "overlay_revision": revision,
        }
        self.calls = 0
        self.prompt_revision = prompt_revision
        self.cached_turns: list[tuple[str, str]] = []

    @property
    def answer_cache_revision(self) -> str:
        return self.prompt_revision

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
            "action_links": [
                {
                    "link_id": "MT-TEST-001",
                    "label": "착오송금 반환지원 신청",
                    "url": "https://www.kdic.or.kr/example/apply",
                    "description": "공식 신청 화면으로 이동합니다.",
                    "requires_auth": True,
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
    assert '"suggestion_cache_runtime_namespace": PIPELINE_RUNTIME.cache_namespace' in api
    assert 'health.get("suggestion_cache_runtime_namespace")' in prewarm
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
    assert "function officialUrl(raw)" in ui
    assert "function answerWithoutUrls(text='')" in ui
    assert "function stripDocumentFormattingArtifacts(text='')" in ui
    assert "function sourceLinksHtml(sources,visible=3)" in ui
    assert "function bindSourceToggles(root)" in ui
    assert "md(answerWithoutUrls(result.answer||'답변을 준비하지 못했습니다.'))" in ui
    assert "officialRows(result.sources)" in ui
    assert "officialRows(result.action_links,3)" in ui
    assert ui.index("officialRows(result.action_links,3)") < ui.index("officialRows(result.sources)")
    assert "관련 공식 서비스" in ui
    assert "본인인증 필요" in ui
    assert 'rel="noopener noreferrer"' in ui
    assert "host==='kdic.or.kr'||host.endsWith('.kdic.or.kr')" in ui
    assert "const PROGRESS_STAGE_DWELL_MS=280,CACHE_STAGE_DWELL_MS=140" in ui
    assert "async function completeProgress(row)" in ui
    assert "async function completeCachedProgress(row)" in ui
    assert "async function advanceProgress(row,p,stage" in ui
    assert "if(cacheHit)await completeCachedProgress(row);else await completeProgress(row)" in ui
    assert "else if(preview==='cache-progress')" in ui
    assert "function basisHtml(d)" in ui
    assert "쉬운 요약 보기" in ui
    assert "쉬운 요약 접기" in ui
    assert "공식 정보를 쉽게 요약하고 있어요" in ui
    assert "답변을 쉽게 이해하기" in ui
    assert "근거가 된 공식 정보" in ui
    assert "사용자에게 어떤 의미인가요?" in ui
    assert "sourceLinksHtml(sources,3)" in ui
    assert "scrollAnswerTop(row)" in ui
    assert "input.focus({preventScroll:true})" in ui
    assert "finally{state.busy=false;updateSendState();input.focus();scrollBottom()}" not in ui
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


def test_user_basis_contract(core) -> dict[str, str]:
    raw_result = {
        "answer": "착오송금 반환지원은 자진반환 및 지급명령으로 회수되면 신청 접수일로부터 2개월 내외가 예상됩니다.",
        "analysis": {"businesses": ["착오송금 반환 신청"]},
        "common": {"resolved_question": "착오송금 반환지원 신청부터 실제 반환까지 얼마나 걸리나요?"},
        "payload": {
            "used_evidence_ids": ["E1"],
            "missing_information": ["개별 거래 조건을 확인해 주세요."],
        },
        "evidence_pack": {
            "evidence": [
                {
                    "evidence_id": "E1",
                    "section_title": "1. 반환 기간은? · 2. 지원 대상은? · 3. 신청 방법은?",
                    "content": "[MT-005_chunk_000] 착오송금 반환지원 주요 FAQ / 1. 신청 접수부터 실제 반환까지 얼마나 걸리나요? ### Q. 1. 신청 접수부터 실제 반환까지 얼마나 걸리나요? 자진반환 및 지급명령을 통해 회수 가능한 경우 신청 접수일로부터 2개월 내외로 예상됩니다. [MT-005_chunk_001] 착오송금 반환지원 주요 FAQ / 2. 지원 대상이 아닌 경우는 무엇인가요?",
                },
                {
                    "evidence_id": "E2",
                    "section_title": "사용하지 않은 근거",
                    "content": "최종 답변에서 참조하지 않은 공식 정보입니다.",
                },
            ]
        },
        "sources": [
            {"title": "예금보험공사 공식 안내", "url": "https://www.kdic.or.kr/example"}
        ],
    }
    fallback = core.default_basis_from_result(raw_result)
    assert core.SUGGESTION_CACHE_SCHEMA_VERSION == "kdic-suggestion-answer-bundle-v5.1"
    assert fallback["schema_version"] == "kdic-basis-explanation-v2"
    assert len(fallback["items"]) == 1
    assert fallback["items"][0]["evidence_ids"] == ["E1"]
    assert "E2" not in json.dumps(fallback, ensure_ascii=False)
    assert fallback["items"][0]["user_meaning"]
    assert "예상 처리 기간" in fallback["items"][0]["user_meaning"]
    assert "2개월" in fallback["items"][0]["evidence_summary"]
    assert "chunk_" not in json.dumps(fallback, ensure_ascii=False)
    assert "2. 지원 대상" not in fallback["items"][0]["answer_point"]
    assert fallback["checkpoints"] == ["개별 거래 조건을 확인해 주세요."]
    assert fallback["mappings"] == fallback["items"]

    class BaseAwarePipeline:
        name = "BASIS_TEST"

        def __init__(self) -> None:
            self.received: dict[str, Any] | None = None

        def basis(self, result: Mapping[str, Any], *, base_basis=None) -> dict[str, Any]:
            self.received = dict(base_basis or {})
            output = dict(base_basis or {})
            output["summary"] = "사용자 관점으로 정리한 공식 근거입니다."
            return output

    pipeline = BaseAwarePipeline()
    explained = core.PipelineRuntime(pipeline).basis(raw_result)
    assert pipeline.received and pipeline.received["items"][0]["evidence_ids"] == ["E1"]
    assert explained["summary"] == "사용자 관점으로 정리한 공식 근거입니다."
    return {
        "verified_evidence_only": "passed",
        "backward_compatible_schema": "passed",
        "base_basis_handoff": "passed",
    }


def test_answer_normalization(core) -> dict[str, str]:
    first = "**예금자보호 대상 금융상품**\n\n1인당 보호 한도를 안내합니다."
    second = "**고객 미수령금 신청 방법**\n\n신청 경로를 안내합니다."
    expected = first + "\n\n" + second
    assert core._normalize_answer_text([first, second]) == expected
    assert core._normalize_answer_text(repr([first, second])) == expected
    assert core._answer_from_result({"answer": [first, second]}) == expected
    ordinary = "## 공식 안내\n\n일반 Markdown 답변입니다."
    assert core._normalize_answer_text(ordinary) == ordinary
    pandoc_empty = "금융안심포털([]{.underline})에 접속해 조회합니다."
    assert core._normalize_answer_text(pandoc_empty) == (
        "금융안심포털에 접속해 조회합니다."
    )
    pandoc_label = "[금융안심포털]{#portal .underline}에서 확인합니다."
    assert core._normalize_answer_text(pandoc_label) == (
        "금융안심포털에서 확인합니다."
    )
    assert core._normalize_answer_text("신청 조건은 {개별 상황}입니다.") == (
        "신청 조건은 {개별 상황}입니다."
    )
    unsafe = "[__import__('os').system('echo unsafe')]"
    assert core._normalize_answer_text(unsafe) == unsafe
    return {
        "list_answer_joined": "passed",
        "stringified_list_joined": "passed",
        "ordinary_markdown_preserved": "passed",
        "pandoc_underline_artifact_removed": "passed",
        "unrelated_braces_preserved": "passed",
        "literal_eval_only": "passed",
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
    assert live_job.result["sources"][0]["url"].startswith("https://www.kdic.or.kr/")
    assert live_job.result["action_links"][0]["label"] == "착오송금 반환지원 신청"
    assert live_job.result["action_links"][0]["requires_auth"] is True

    cached_bundle = cache.peek(service._cache_key(first["suggestion_id"]))
    assert cached_bundle is not None
    cached_bundle.public_result["answer"] = repr(
        [
            "**첫 번째 안내**\n\n금융안심포털([]{.underline})에서 확인합니다.",
            "**두 번째 안내**\n\n두 번째 내용",
        ]
    )
    cache.put(cached_bundle)

    hit_id = service.submit("hit-session", first["query"], first["suggestion_id"])
    hit_job = jobs.get(hit_id)
    assert hit_job is not None and hit_job.status == "done"
    assert hit_job.result["suggestion_cache"]["hit"] is True
    assert pipeline.calls == 1
    assert hit_job.result["sources"] == live_job.result["sources"]
    assert hit_job.result["action_links"] == live_job.result["action_links"]
    assert hit_job.result["answer"] == (
        "**첫 번째 안내**\n\n금융안심포털에서 확인합니다.\n\n"
        "**두 번째 안내**\n\n두 번째 내용"
    )
    assert ".underline" not in hit_job.result["answer"]
    assert pipeline.cached_turns[-1][0] == first["query"]
    assert pipeline.cached_turns[-1][1] == hit_job.result["answer"]
    assert len(sessions.get("hit-session").state["turns"]) == 2
    assert service.basis(hit_id)["summary"] == "공식 근거 요약"

    forged_job = _wait_job(
        service,
        service.submit("forged-session", second["query"], first["suggestion_id"]),
    )
    assert forged_job.status == "done"
    assert "suggestion_cache" not in forged_job.result
    assert pipeline.calls == 2

    before_prompt_change = service._cache_key(first["suggestion_id"])
    pipeline.prompt_revision = "prompt-v2"
    assert before_prompt_change != service._cache_key(first["suggestion_id"])
    prompt_changed_job = _wait_job(
        service,
        service.submit("prompt-version-session", first["query"], first["suggestion_id"]),
    )
    assert prompt_changed_job.status == "done"
    assert prompt_changed_job.result["suggestion_cache"]["hit"] is False
    assert pipeline.calls == 3

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
        "cached_sources_and_action_links": "passed",
        "cached_legacy_list_answer_normalized": "passed",
        "forged_id_query_pair": "live_fallback",
        "runtime_revision_invalidation": "passed",
        "prompt_revision_invalidation": "passed",
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
        "basis": test_user_basis_contract(core),
        "answer_normalization": test_answer_normalization(core),
        "cache_flow": test_memory_cache_flow(core),
        "adapter": test_adapter_cached_turn(),
    }
    print(json.dumps({"status": "passed", "result": result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
