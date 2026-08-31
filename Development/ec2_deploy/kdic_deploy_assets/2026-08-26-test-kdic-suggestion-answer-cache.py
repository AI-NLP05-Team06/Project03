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
MIGRATION_FILE = BASE_DIR.parent / "2026-08-28-kdic-persistent-suggestion-catalog-migration.sql"


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
        self.cached_turns: list[tuple[str, str, list[str]]] = []

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
        business_scope: list[str] | None = None,
    ) -> None:
        scope = list(business_scope or [])
        self.cached_turns.append((question, answer, scope))
        state.setdefault("turns", []).extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        )
        if business_scope is not None:
            state["active_businesses"] = scope
            state["excluded_businesses"] = []
            state["pending_clarification"] = None
            state["last_resolved_question"] = question


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
    assert 'suggestion["business"],' in prewarm
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
    assert '@app.post("/api/summary")' in api
    assert "return JOB_SERVICE.summary(payload.job_id)" in api
    assert "function summaryHtml(d)" in ui
    assert "async function loadSummary" in ui
    assert "api('/api/summary'" in ui
    assert "핵심 요약 보기" in ui
    assert "핵심 요약 접기" in ui
    assert "핵심 내용을 짧게 정리하고 있어요" in ui
    assert "핵심만 정리했어요" in ui
    assert "세부 조건과 공식 링크는 위 답변에서 확인해 주세요." in ui
    assert 'class="action-btn summary-btn"' in ui
    assert "const summary=row.querySelector('.summary-btn')" in ui
    assert "if(summary)summary.onclick=()=>loadSummary(summary,row.querySelector('.summary-slot'),jobId)" in ui
    assert ".filter(Boolean).slice(0,3)" in ui
    assert "${esc(title)}" in ui
    assert "${esc(point)}" in ui
    assert "if(slot.dataset.loaded)" in ui
    assert "요약할 핵심 내용을 찾지 못했어요" in ui
    assert "핵심 요약 다시 보기" in ui
    assert "function basisHtml(d)" not in ui
    assert "api('/api/basis'" not in ui
    assert "근거가 된 공식 정보" not in ui
    assert "사용자에게 어떤 의미인가요?" not in ui
    assert "sourceLinksHtml(sources,3)" in ui
    assert "scrollAnswerTop(row)" in ui
    assert "input.focus({preventScroll:true})" in ui
    assert "finally{state.busy=false;updateSendState();input.focus();scrollBottom()}" not in ui
    assert "class PostgresSuggestionAnswerCache" in postgres
    assert "CREATE TABLE IF NOT EXISTS suggestion_catalog" in migration
    assert "ALTER COLUMN expires_at DROP NOT NULL" in migration
    assert "SET expires_at = NULL" in migration
    assert "cache.peek_active" in prewarm
    assert "compatible_overlay_revisions" in prewarm
    assert "normalized_public = core.normalize_public_result(raw)" in prewarm
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
    assert {row["suggestion_id"] for row in catalog} == set(
        core.FOLLOWUP_SUGGESTION_IDS.values()
    )
    assert not any(
        bad in row["query"]
        for row in catalog
        for bad in ("서류을", "절차을", "시기을", "한도을", "자료을")
    )
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


def test_followup_keyword_policy(core) -> dict[str, Any]:
    business_key = "착오송금"
    original_rows = list(core._SUGGESTIONS_BY_BUSINESS_KEY[business_key])
    extra_row = {
        "suggestion_id": "SQ-TEST-UNLIMITED-0001",
        "business_key": business_key,
        "business": "착오송금 반환 신청",
        "label": "신청 자격",
        "query": "착오송금 반환 신청의 신청 자격을 알려주세요.",
    }
    core._SUGGESTIONS_BY_BUSINESS_KEY[business_key] = [*original_rows, extra_row]
    try:
        rows = core._followup_keywords(
            ["착오송금 반환 신청", "예금보험금 안내"]
        )
    finally:
        core._SUGGESTIONS_BY_BUSINESS_KEY[business_key] = original_rows

    assert len(rows) == 6
    assert rows[-1] == extra_row
    assert {row["business_key"] for row in rows} == {business_key}
    assert core._followup_keywords([]) == []
    return {
        "primary_business_only": "passed",
        "unlimited_keyword_count": len(rows),
        "empty_businesses": "passed",
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
    assert core.SUGGESTION_CACHE_SCHEMA_VERSION == "kdic-managed-suggestion-answer-v6"
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


def test_answer_summary_contract(core) -> dict[str, str]:
    question = "본인 명의 고객 미수령금 조회 방법을 알려주세요."
    contaminated_result = {
        "route": "RETRIEVE",
        "answer": """
**예금자보호 대상 금융상품**

원금과 소정이자를 합하여 1인당 1억원까지 보호됩니다.

보호대상 금융상품은 예·적금과 일부 보험계약 등을 포함합니다.

**본인 명의 고객 미수령금 조회 방법**

1. **금융안심포털 이용**

[금융안심포털](https://fins.kdic.or.kr "공식 안내")에서 본인 인증 후 고객 미수령금을 조회할 수 있습니다. fins.kdic.or.kr [E1] [MT-005_chunk_001] Evidence Pack parent_id=P1

2. **방문 확인**

온라인 이용이 어렵다면 안내된 방문 신청 방법을 확인할 수 있습니다.

3. **명의 변경**

명의가 변경된 경우 신청 전에 고객센터에 확인해야 합니다.
""",
        "businesses": ["예금자보호제도", "고객 미수령금 신청"],
    }
    summary = core.answer_summary_from_result(contaminated_result, question=question)
    points = summary["points"]
    public_text = " ".join(points)

    assert summary["schema_version"] == "kdic-answer-summary-v1"
    assert summary["source"] == "VALIDATED_FINAL_ANSWER"
    assert summary["extractive"] is True
    assert summary["point_count"] == len(points)
    assert 1 <= len(points) <= 3
    assert all(0 < len(point) <= 160 for point in points)
    assert "미수령금" in public_text or "금융안심포털" in public_text
    assert "예금자보호" not in public_text
    assert "1억원" not in public_text
    assert "보호대상 금융상품" not in public_text
    assert "http" not in public_text
    assert "Evidence Pack" not in public_text
    assert "chunk_" not in public_text
    assert "parent_id" not in public_text
    assert "fins.kdic.or.kr" not in public_text
    assert "[E1]" not in public_text
    assert summary == core.answer_summary_from_result(
        contaminated_result,
        question=question,
    )
    unrelated_only = core.answer_summary_from_result(
        {
            "route": "RETRIEVE",
            "answer": "**예금자보호 대상 금융상품**\n\n원금과 소정이자를 합하여 1인당 1억원까지 보호됩니다.",
        },
        question=question,
    )
    assert unrelated_only["points"] == []
    assert unrelated_only["point_count"] == 0
    unlabeled_unrelated = core.answer_summary_from_result(
        {
            "route": "RETRIEVE",
            "answer": "원금과 소정이자를 합하여 1인당 1억원까지 보호됩니다.",
        },
        question=question,
    )
    assert unlabeled_unrelated["points"] == []

    intent_scoped = core.answer_summary_from_result(
        {
            "route": "RETRIEVE",
            "answer": """
**고객 미수령금 조회 방법**

고객 미수령금은 부실화된 금융회사의 예금자 등이 찾아가지 않은 금액입니다.

- 금융안심포털에서 본인 인증 후 미수령금을 조회할 수 있습니다.
- 온라인 이용이 어렵다면 안내된 방문 신청 방법을 확인해 주세요.
- 추가 확인이 필요하면 안내된 고객센터에 전화해 주세요.
""",
            "businesses": ["고객 미수령금 신청"],
        },
        question=question,
    )
    intent_text = " ".join(intent_scoped["points"])
    assert len(intent_scoped["points"]) == 3
    assert "미수령금은 부실화된" not in intent_text
    assert "금융안심포털" in intent_text
    assert "방문" in intent_text
    assert "전화" in intent_text
    assert len(
        core.answer_summary_from_result(
            {"answer": intent_scoped["points"]},
            question=question,
            business_scope=["고객 미수령금 신청"],
            maximum_points=1,
        )["points"]
    ) == 1
    assert len(
        core.answer_summary_from_result(
            {
                "answer": """
**고객 미수령금 조회 방법**
- 금융안심포털에서 미수령금을 조회할 수 있습니다.
- 안내된 방문 신청 방법을 확인할 수 있습니다.
- 고객센터에 전화해 추가 조건을 확인할 수 있습니다.
- 본인 인증 수단을 미리 준비해 주세요.
""",
            },
            question=question,
            business_scope=["고객 미수령금 신청"],
            maximum_points=99,
        )["points"]
    ) == 3
    assert core.answer_summary_from_result(
        {"answer": "   "},
        question=question,
        business_scope=["고객 미수령금 신청"],
    )["points"] == []

    natural_language = core.answer_summary_from_result(
        contaminated_result,
        question="못 받은 돈은 어디에서 찾나요?",
    )
    natural_text = " ".join(natural_language["points"])
    assert natural_language["points"]
    assert "1억원" not in natural_text
    assert "금융안심포털" in natural_text

    mixed_heading = core.answer_summary_from_result(
        {
            "answer": """
**예금자보호제도·고객 미수령금 신청 답변**
원금과 소정이자를 합하여 1인당 1억원까지 보호됩니다.
고객 미수령금 조회는 금융안심포털에서 할 수 있습니다.
온라인 이용이 어렵다면 방문 신청 방법을 확인해 주세요.
""",
            "businesses": ["예금자보호제도", "고객 미수령금 신청"],
        },
        question=question,
    )
    mixed_text = " ".join(mixed_heading["points"])
    assert "금융안심포털" in mixed_text
    assert "방문" in mixed_text
    assert "1억원" not in mixed_text
    return {
        "validated_answer_only": "passed",
        "question_business_scoped": "passed",
        "unrelated_only_answer_suppressed": "passed",
        "unlabeled_contamination_suppressed": "passed",
        "same_business_intent_scoped": "passed",
        "natural_language_alias_scoped": "passed",
        "mixed_heading_scope_recovered": "passed",
        "public_artifacts_removed": "passed",
        "bounded_extracts": "passed",
        "deterministic": "passed",
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
    foreign = next(
        row
        for row in core.suggestion_catalog()
        if row["business"] != first["business"]
    )

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
    authoritative_cached_answer = repr(
        [
            f"**{first['business']} 첫 번째 안내**\n\n착오송금 반환 신청 대상은 금융안심포털([]{{.underline}})에서 확인합니다.",
            "**두 번째 안내**\n\n두 번째 내용",
        ]
    )
    cached_bundle.raw_result["answer"] = authoritative_cached_answer
    cached_bundle.public_result["answer"] = "채무조정에 관한 오염된 별도 저장 본문"
    cache.put(cached_bundle)

    hit_session = sessions.get("hit-session")
    hit_session.state["active_businesses"] = ["은닉재산 신고"]
    hit_id = service.submit("hit-session", first["query"], first["suggestion_id"])
    hit_job = jobs.get(hit_id)
    assert hit_job is not None and hit_job.status == "done"
    assert hit_job.result["suggestion_cache"]["hit"] is True
    assert pipeline.calls == 1
    assert hit_job.result["sources"] == live_job.result["sources"]
    assert hit_job.result["action_links"] == live_job.result["action_links"]
    assert hit_job.result["answer"] == (
        f"**{first['business']} 첫 번째 안내**\n\n"
        "착오송금 반환 신청 대상은 금융안심포털에서 확인합니다.\n\n"
        "**두 번째 안내**\n\n두 번째 내용"
    )
    assert ".underline" not in hit_job.result["answer"]
    assert pipeline.cached_turns[-1][0] == first["query"]
    assert pipeline.cached_turns[-1][1] == hit_job.result["answer"]
    assert pipeline.cached_turns[-1][2] == [first["business"]]
    assert len(sessions.get("hit-session").state["turns"]) == 2
    assert sessions.get("hit-session").state["active_businesses"] == [
        first["business"]
    ]
    assert sessions.get("hit-session").state["excluded_businesses"] == []
    assert service.basis(hit_id)["summary"] == "공식 근거 요약"
    cached_summary = service.summary(hit_id)
    assert cached_summary["schema_version"] == "kdic-answer-summary-v1"
    assert 1 <= cached_summary["point_count"] <= 3
    cached_summary_text = " ".join(cached_summary["points"])
    assert "금융안심포털" in cached_summary_text
    assert ".underline" not in cached_summary_text
    assert "공식 근거 요약" not in cached_summary_text
    assert pipeline.calls == 1

    try:
        service.summary("missing-job")
    except KeyError:
        pass
    else:
        raise AssertionError("unknown summary job must raise KeyError")
    queued = jobs.create("queued-session", "대기 중 질문")
    try:
        service.summary(queued.job_id)
    except RuntimeError:
        pass
    else:
        raise AssertionError("unfinished summary job must raise RuntimeError")

    stale_foreign_bundle = cache.peek(service._cache_key(first["suggestion_id"]))
    assert stale_foreign_bundle is not None
    stale_foreign_bundle.cache_key = service._cache_key(foreign["suggestion_id"])
    stale_foreign_bundle.suggestion_id = foreign["suggestion_id"]
    stale_foreign_bundle.business = foreign["business"]
    stale_foreign_bundle.keyword = foreign["label"]
    stale_foreign_bundle.question = foreign["query"]
    stale_foreign_bundle.basis_result = {
        "schema_version": core.BASIS_EXPLANATION_SCHEMA_VERSION,
        "summary": "거부되어야 하는 오래된 캐시 근거",
    }
    cache.put(stale_foreign_bundle)

    mismatch_job = _wait_job(
        service,
        service.submit(
            "mismatch-session",
            foreign["query"],
            foreign["suggestion_id"],
        ),
    )
    assert mismatch_job.status == "done"
    assert mismatch_job.result["suggestion_cache"] == {
        "eligible": False,
        "hit": False,
        "stored": False,
        "suggestion_id": foreign["suggestion_id"],
        "source": "LIVE_PIPELINE",
        "reason": "SUGGESTION_BUSINESS_MISMATCH",
        "skipped_stages": [],
    }
    assert cache.peek(service._cache_key(foreign["suggestion_id"])) is not None
    assert service.basis(mismatch_job.job_id)["summary"] == "공식 근거 요약"
    assert pipeline.calls == 2

    forged_job = _wait_job(
        service,
        service.submit("forged-session", second["query"], first["suggestion_id"]),
    )
    assert forged_job.status == "done"
    assert "suggestion_cache" not in forged_job.result
    assert pipeline.calls == 3

    before_prompt_change = service._cache_key(first["suggestion_id"])
    pipeline.prompt_revision = "prompt-v2"
    assert before_prompt_change != service._cache_key(first["suggestion_id"])
    prompt_changed_job = _wait_job(
        service,
        service.submit("prompt-version-session", first["query"], first["suggestion_id"]),
    )
    assert prompt_changed_job.status == "done"
    assert prompt_changed_job.result["suggestion_cache"]["hit"] is True
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
    other_hit_job = _wait_job(
        other_service,
        other_service.submit(
            "other-runtime-session", first["query"], first["suggestion_id"]
        ),
    )
    assert other_hit_job.status == "done"
    assert other_hit_job.result["suggestion_cache"]["hit"] is True
    assert other_pipeline.calls == 0
    fallback_runtime = core.PipelineRuntime(lambda question, state: {})
    fallback_state: dict[str, Any] = {}
    fallback_runtime.record_cached_turn(
        "채무조정에 필요한 서류를 알려주세요.",
        "채무조정 공식 답변",
        fallback_state,
        business_scope=["채무조정 안내"],
    )
    assert fallback_state["active_businesses"] == ["채무조정"], fallback_state
    service.shutdown()
    other_service.shutdown()
    return {
        "first_click_store": "passed",
        "next_click_hit_without_pipeline": "passed",
        "cached_turn_context": "passed",
        "cached_basis": "passed",
        "cached_summary_without_pipeline": "passed",
        "summary_job_state_guards": "passed",
        "cached_sources_and_action_links": "passed",
        "cached_legacy_list_answer_normalized": "passed",
        "cached_raw_answer_authoritative": "passed",
        "rejected_cache_basis_not_reused": "passed",
        "cache_hit_business_scope": "canonical_business_applied",
        "mismatched_business_bundle": "not_stored",
        "forged_id_query_pair": "live_fallback",
        "runtime_revision_persistence": "passed",
        "prompt_revision_persistence": "passed",
        "fallback_cached_scope_canonicalization": "passed",
    }


def test_adapter_cached_turn() -> dict[str, str]:
    adapter_module = _load_module("kdic_cache_adapter_test", ADAPTER_FILE)
    adapter = adapter_module.LatestKDICNotebookAdapter(
        {"execute_production_variant_v1": lambda question, holder: {}}
    )
    state: dict[str, Any] = {}
    adapter.record_cached_turn(
        "채무조정 필요 서류를 알려주세요.",
        "답변",
        state,
        business_scope=["채무조정 안내"],
    )
    holder = state["_kdic_controller"]
    assert holder["conversation"]["turns"] == [
        {"role": "user", "content": "채무조정 필요 서류를 알려주세요."},
        {"role": "assistant", "content": "답변"},
    ]
    assert holder["conversation"]["active_businesses"] == ["채무조정"]
    assert holder["conversation"]["excluded_businesses"] == []
    assert holder["conversation"]["pending_clarification"] is None
    assert holder["conversation"]["last_resolved_question"] == (
        "채무조정 필요 서류를 알려주세요."
    )
    assert holder["committed_variant"] == "SUGGESTION_ANSWER_CACHE"
    return {
        "adapter_context_commit": "passed",
        "adapter_cached_business_scope": "채무조정",
    }


def main() -> None:
    core = _load_module("kdic_cache_core_test", CORE_FILE)
    result = {
        "static": test_static_contracts(),
        "registry": test_registry(core),
        "followup_keywords": test_followup_keyword_policy(core),
        "basis": test_user_basis_contract(core),
        "answer_normalization": test_answer_normalization(core),
        "summary": test_answer_summary_contract(core),
        "cache_flow": test_memory_cache_flow(core),
        "adapter": test_adapter_cached_turn(),
    }
    print(json.dumps({"status": "passed", "result": result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
