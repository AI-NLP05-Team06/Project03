from __future__ import annotations

import hmac
import importlib.util
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any, Mapping

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
CORE_PATH = BASE_DIR / "2026-08-23-kdic-service-core.py"
HTML_PATH = BASE_DIR / "2026-08-23-kdic-chat-ui.html"
ADMIN_HTML_PATH = BASE_DIR / "kdic-admin-ui.html"


def _load_core():
    spec = importlib.util.spec_from_file_location("kdic_service_core", CORE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"KDIC 서비스 코어를 불러올 수 없습니다: {CORE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


core = _load_core()


def _load_local(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, BASE_DIR / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"{filename}을 불러올 수 없습니다: {BASE_DIR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# 요구사항 1~5 (관리자 데이터/파라미터/평가 기능) 지원 모듈들. kdic_final_pipeline은
# 크기가 커서(실전1 크롤링 엔진, 253KB) 실제로 신규 URL 추가/재수집 라우트를 쓸 때만
# 필요하지만, 기동 시점에 한 번에 확인하는 게 배포 문제를 더 빨리 드러내서 즉시 로드한다.
param_test = _load_local("kdic_param_test", "kdic_param_test.py")
eval_runner = _load_local("kdic_eval_runner", "kdic_eval_runner.py")
new_url_defaults = _load_local("kdic_new_url_defaults", "kdic_new_url_defaults.py")
preview_review = _load_local("kdic_preview_review", "kdic_preview_review.py")
reingest_trigger = _load_local("kdic_reingest_trigger", "kdic_reingest_trigger.py")
ingest_trigger = _load_local("kdic_ingest_trigger", "kdic_ingest_trigger.py")
kdic_final_pipeline = _load_local("kdic_final_pipeline", "kdic_final_pipeline.py")
es_publish = _load_local("kdic_es_publish", "kdic_es_publish.py")


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _pipeline_build_info(pipeline: Any) -> dict[str, Any]:
    value = getattr(pipeline, "build_info", None)
    return dict(value) if isinstance(value, Mapping) else {}


PIPELINE_RUNTIME = core.PipelineRuntime()
RUNTIME_BUILD_INFO: dict[str, Any] = {}

if os.getenv("KDIC_PIPELINE_ENTRYPOINT", "").strip():
    _initial_pipeline = core.load_entrypoint(os.environ["KDIC_PIPELINE_ENTRYPOINT"])
    PIPELINE_RUNTIME.set(_initial_pipeline)
    RUNTIME_BUILD_INFO = _pipeline_build_info(_initial_pipeline)
elif _env_bool("KDIC_DEMO_MODE"):
    _initial_pipeline = core.DemoKDICPipeline()
    PIPELINE_RUNTIME.set(_initial_pipeline)
    RUNTIME_BUILD_INFO = _pipeline_build_info(_initial_pipeline)

if os.getenv("KDIC_DATABASE_URL", "").strip():
    # Persistent, multi-instance-safe stores backed by the `jobs` /
    # `chat_sessions` RDS tables. Falls back to in-memory when unset so local
    # dev / Colab keeps working without a database.
    # Loaded the same way as `core` above (spec_from_file_location) rather than
    # a bare `import`, since this module is itself often loaded dynamically by
    # a launcher and its directory isn't guaranteed to be on sys.path.
    def _load_postgres_store():
        spec = importlib.util.spec_from_file_location(
            "kdic_postgres_store", BASE_DIR / "kdic_postgres_store.py"
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"kdic_postgres_store를 불러올 수 없습니다: {BASE_DIR}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    pg_store = _load_postgres_store()

    SESSION_STORE = pg_store.PostgresSessionStore(
        ttl_seconds=_env_int("KDIC_SESSION_TTL_SECONDS", 86_400, 60, 2_592_000),
        max_sessions=_env_int("KDIC_MAX_SESSIONS", 2_000, 10, 100_000),
    )
    JOB_STORE = pg_store.PostgresJobStore(
        ttl_seconds=_env_int("KDIC_JOB_TTL_SECONDS", 86_400, 300, 2_592_000),
        max_jobs=_env_int("KDIC_MAX_JOBS", 5_000, 50, 500_000),
    )
    SUGGESTION_ANSWER_CACHE = pg_store.PostgresSuggestionAnswerCache(
        ttl_seconds=_env_int(
            "KDIC_SUGGESTION_CACHE_TTL_SECONDS", 2_592_000, 60, 31_536_000
        ),
        max_entries=_env_int(
            "KDIC_SUGGESTION_CACHE_MAX_ENTRIES", 2_000, 10, 100_000
        ),
    )
else:
    SESSION_STORE = core.InMemorySessionStore(
        ttl_seconds=_env_int("KDIC_SESSION_TTL_SECONDS", 86_400, 60, 2_592_000),
        max_sessions=_env_int("KDIC_MAX_SESSIONS", 2_000, 10, 100_000),
    )
    JOB_STORE = core.InMemoryJobStore(
        ttl_seconds=_env_int("KDIC_JOB_TTL_SECONDS", 86_400, 300, 2_592_000),
        max_jobs=_env_int("KDIC_MAX_JOBS", 5_000, 50, 500_000),
    )
    SUGGESTION_ANSWER_CACHE = core.InMemorySuggestionAnswerCache(
        ttl_seconds=_env_int(
            "KDIC_SUGGESTION_CACHE_TTL_SECONDS", 2_592_000, 60, 31_536_000
        ),
        max_entries=_env_int(
            "KDIC_SUGGESTION_CACHE_MAX_ENTRIES", 200, 10, 10_000
        ),
    )
JOB_SERVICE = core.KDICJobService(
    runtime=PIPELINE_RUNTIME,
    sessions=SESSION_STORE,
    jobs=JOB_STORE,
    suggestion_cache=SUGGESTION_ANSWER_CACHE,
    # The current Colab pipeline still records several traces in module globals.
    # Keep one worker until those globals are made request-scoped and concurrency
    # regression tests pass. The service design itself does not require a global lock.
    max_workers=_env_int("KDIC_PIPELINE_WORKERS", 1, 1, 32),
)


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=200)
    question: str = Field(min_length=1, max_length=4_000)
    suggestion_id: str = Field(default="", max_length=100)


class ResetRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=200)


class BasisRequest(BaseModel):
    job_id: str = Field(min_length=1, max_length=100)


class ConfigureRequest(BaseModel):
    api_key: str = Field(default="", max_length=2_000)
    use_colab_secret: bool = False


class AdminSearchRequest(BaseModel):
    index: str = Field(min_length=1, max_length=255)
    query: str = Field(default="", max_length=2_000)
    size: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0, le=10_000)


# ---- 요구사항 3 (파라미터 테스트) ----
class ParamComboRequest(BaseModel):
    label: str = Field(min_length=1, max_length=100)
    dense_weight: float = Field(ge=0, le=1)
    bm25_weight: float = Field(ge=0, le=1)
    candidate_depth: int = Field(default=20, ge=1, le=200)
    final_top_k: int = Field(default=5, ge=1, le=50)
    rrf_k: int = Field(default=10, ge=1, le=200)


class ParamCompareRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2_000)
    combos: list[ParamComboRequest] = Field(min_length=1, max_length=10)


class ParamActivateRequest(ParamComboRequest):
    created_by: str = Field(default="", max_length=200)


# ---- 요구사항 5 (평가 연동) ----
class EvalRunRequest(BaseModel):
    search_params_id: str = Field(min_length=1, max_length=100)
    eval_query_ids: list[str] | None = None
    triggered_by: str = Field(default="", max_length=200)


# ---- 요구사항 4 (갱신 트리거링) ----
class ReingestRequest(BaseModel):
    url_ids: list[str] = Field(min_length=1, max_length=42)
    triggered_by: str = Field(default="", max_length=200)


# ---- 요구사항 1+2 (신규 URL 추가 + 미리보기) ----
class NewUrlRequest(BaseModel):
    source_url: str = Field(min_length=1, max_length=2_000)
    business_domain: str = Field(min_length=1, max_length=200)
    triggered_by: str = Field(default="", max_length=200)


class ReviewActionRequest(BaseModel):
    reviewed_by: str = Field(default="", max_length=200)


app = FastAPI(
    title="KDIC Chatbot and Read-only Admin API",
    version="2026.08.23",
    description=(
        "기존 KDIC HTML UI와 최신 질의분석·검색·답변 런타임을 연결하는 API입니다. "
        "관리자 Elasticsearch API는 현재 조회 전용입니다."
    ),
)


cors_origins = [
    value.strip()
    for value in os.getenv("KDIC_CORS_ORIGINS", "").split(",")
    if value.strip()
]
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )


def set_kdic_pipeline(pipeline: Any) -> None:
    """Attach the latest KDIC callable when this file is loaded in Colab."""

    global RUNTIME_BUILD_INFO
    PIPELINE_RUNTIME.set(pipeline)
    RUNTIME_BUILD_INFO = _pipeline_build_info(pipeline)


# PIPELINE_RUNTIME above wraps the *adapter* (route/answer selection only) --
# the admin param-test/eval/ingest routes need the raw kdic_pipeline_engine
# module itself (hybrid_minmax_search, DENSE_WEIGHT, ...), which the adapter
# doesn't expose. Kept as a separate global so chat and admin routes can't be
# confused for each other.
RAW_PIPELINE_MODULE: Any = None


def set_kdic_raw_pipeline_module(module: Any) -> None:
    """Attach the raw kdic_pipeline_engine module for admin param/eval routes."""

    global RAW_PIPELINE_MODULE
    RAW_PIPELINE_MODULE = module


def _raw_pipeline_or_503() -> Any:
    if RAW_PIPELINE_MODULE is None:
        raise HTTPException(
            status_code=503,
            detail="원본 파이프라인 모듈이 연결되지 않았습니다 (set_kdic_raw_pipeline_module 필요).",
        )
    return RAW_PIPELINE_MODULE


def _db_connect():
    database_url = os.getenv("KDIC_DATABASE_URL", "").strip()
    if not database_url:
        raise HTTPException(
            status_code=503, detail="KDIC_DATABASE_URL이 설정되지 않았습니다."
        )
    import psycopg2

    return psycopg2.connect(database_url)


def _dict_cursor(conn):
    """psycopg2는 KDIC_DATABASE_URL이 설정된 배포에서만 필요한 선택적 의존성이라
    (인메모리 모드로도 서비스가 뜰 수 있어야 함) 모듈 상단이 아니라 여기서
    지연 임포트한다."""
    from psycopg2.extras import RealDictCursor

    return conn.cursor(cursor_factory=RealDictCursor)


def _admin_token() -> str:
    return os.getenv("KDIC_ADMIN_TOKEN", "").strip()


def require_admin(authorization: str | None = Header(default=None)) -> None:
    expected = _admin_token()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="KDIC_ADMIN_TOKEN이 설정되지 않아 관리자 API가 잠겨 있습니다.",
        )
    supplied = ""
    if authorization and authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()
    if not supplied or not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="관리자 인증에 실패했습니다.")


def _job_or_404(job_id: str):
    record = JOB_STORE.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="작업을 찾지 못했습니다.")
    return record


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    if not HTML_PATH.is_file():
        return HTMLResponse(
            "<h1>KDIC API</h1><p>HTML UI 파일을 찾지 못했습니다.</p>",
            status_code=503,
        )
    return FileResponse(HTML_PATH, media_type="text/html; charset=utf-8")


@app.get("/admin", response_class=HTMLResponse, include_in_schema=False)
def admin_ui():
    # 페이지 자체는 인증 없이 서빙(브라우저 주소창 이동은 커스텀 헤더를 못 보냄).
    # 실제 보호는 페이지 안에서 입력받은 토큰으로 각 /api/admin/* 호출마다
    # Authorization 헤더를 실어 보내는 것으로 이뤄진다 -- 토큰 없이는 어떤
    # 관리자 API도 401/503으로 막힌다.
    if not ADMIN_HTML_PATH.is_file():
        return HTMLResponse(
            "<h1>KDIC Admin</h1><p>관리자 화면 파일을 찾지 못했습니다.</p>",
            status_code=503,
        )
    return FileResponse(ADMIN_HTML_PATH, media_type="text/html; charset=utf-8")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "pipeline_configured": PIPELINE_RUNTIME.configured,
        "pipeline": PIPELINE_RUNTIME.name,
        "runtime_build": dict(RUNTIME_BUILD_INFO),
        "sessions": SESSION_STORE.stats(),
        "jobs": JOB_STORE.stats(),
        "suggestion_answer_cache": SUGGESTION_ANSWER_CACHE.stats(),
        "suggestion_registry": core.suggestion_registry_stats(),
        "admin_mode": "STAGED_WRITE",
    }


@app.get("/api/config")
def config() -> dict[str, Any]:
    return {
        "configured": PIPELINE_RUNTIME.configured,
        "bootstrap_available": bool(os.getenv("HCX_API_KEY", "").strip()),
        "pipeline": PIPELINE_RUNTIME.name,
        "runtime_build": dict(RUNTIME_BUILD_INFO),
    }


@app.post("/api/configure", dependencies=[Depends(require_admin)])
def configure(payload: ConfigureRequest) -> dict[str, Any]:
    key = os.getenv("HCX_API_KEY", "").strip() if payload.use_colab_secret else payload.api_key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="HCX API 키가 필요합니다.")
    if key.lower().startswith("bearer ") or any(character.isspace() for character in key):
        raise HTTPException(
            status_code=400,
            detail="Bearer를 제외하고 공백 없는 API 키만 입력해 주세요.",
        )
    try:
        PIPELINE_RUNTIME.configure(key)
    except core.PipelineNotConfiguredError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {"ok": True, "configured": PIPELINE_RUNTIME.configured}


@app.post("/api/jobs", status_code=202)
@app.post("/api/v1/chat", status_code=202)
def create_job(payload: ChatRequest) -> dict[str, str]:
    if not PIPELINE_RUNTIME.configured:
        raise HTTPException(
            status_code=503,
            detail="KDIC 파이프라인이 연결되지 않았습니다.",
        )
    question = payload.question
    manager = getattr(RAW_PIPELINE_MODULE, "KDIC_GUARDRAIL_MANAGER", None) if RAW_PIPELINE_MODULE is not None else None
    if manager is not None:
        audit = manager.evaluate(question, "input")
        if audit["blocked"]:
            raise HTTPException(
                status_code=400,
                detail="민감정보 또는 관리자 차단 규칙이 감지되어 질문을 전송하지 않았습니다.",
            )
        question = audit["text"]
    try:
        job_id = JOB_SERVICE.submit(
            payload.session_id,
            question,
            suggestion_id=payload.suggestion_id,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
@app.get("/api/v1/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    return _job_or_404(job_id).public()


@app.post("/api/basis")
def answer_basis(payload: BasisRequest) -> dict[str, Any]:
    _job_or_404(payload.job_id)
    try:
        return JOB_SERVICE.basis(payload.job_id)
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/api/reset")
@app.post("/api/v1/sessions/reset")
def reset_session(payload: ResetRequest) -> dict[str, bool]:
    SESSION_STORE.reset(payload.session_id)
    return {"ok": True}


SAFE_INDEX_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,254}$")


def _safe_index(value: str) -> str:
    index = value.strip()
    if not SAFE_INDEX_PATTERN.fullmatch(index) or index.startswith("."):
        raise HTTPException(
            status_code=400,
            detail="관리자 조회는 와일드카드가 없는 정확한 일반 인덱스 이름만 허용합니다.",
        )
    return index


_ES_CLIENT: Any = None
_ES_LOCK = threading.RLock()


def _es_client():
    global _ES_CLIENT
    with _ES_LOCK:
        if _ES_CLIENT is not None:
            return _ES_CLIENT
        url = os.getenv("ELASTICSEARCH_URL", "").strip()
        if not url:
            raise HTTPException(
                status_code=503, detail="ELASTICSEARCH_URL이 설정되지 않았습니다."
            )
        try:
            from elasticsearch import Elasticsearch
        except ImportError as error:
            raise HTTPException(
                status_code=503, detail="elasticsearch Python 패키지가 필요합니다."
            ) from error
        kwargs: dict[str, Any] = {
            "request_timeout": _env_int("ELASTICSEARCH_TIMEOUT_SECONDS", 30, 1, 300),
            "verify_certs": _env_bool("ELASTICSEARCH_VERIFY_CERTS", True),
        }
        api_key = os.getenv("ELASTICSEARCH_API_KEY", "").strip()
        username = os.getenv("ELASTICSEARCH_USERNAME", "").strip()
        password = os.getenv("ELASTICSEARCH_PASSWORD", "")
        if api_key:
            kwargs["api_key"] = api_key
        elif username:
            kwargs["basic_auth"] = (username, password)
        _ES_CLIENT = Elasticsearch(url, **kwargs)
        return _ES_CLIENT


def _es_error(error: Exception) -> HTTPException:
    return HTTPException(
        status_code=502,
        detail=f"Elasticsearch 조회 실패: {type(error).__name__}: {error}",
    )


@app.get("/api/admin/capabilities", dependencies=[Depends(require_admin)])
def admin_capabilities() -> dict[str, Any]:
    return {
        "mode": "STAGED_WRITE",
        "features": [
            "runtime_summary",
            "job_audit",
            "elasticsearch_cluster_health",
            "elasticsearch_index_list",
            "elasticsearch_document_search",
            "elasticsearch_document_view",
            "parameter_compare",
            "parameter_activate",
            "evaluation_dataset_upload",
            "evaluation_run",
            "url_ingest_preview",
            "preview_review",
            "preview_publish",
            "chunk_staged_write",
            "rollback",
            "api_key_test",
            "api_key_runtime_rotation",
        ],
        "disabled_mutations": [
            "alias_switch",
            "index_delete",
        ],
    }


@app.get("/api/admin/summary", dependencies=[Depends(require_admin)])
def admin_summary() -> dict[str, Any]:
    es_payload: dict[str, Any]
    try:
        client = _es_client()
        info = client.info()
        health_payload = client.cluster.health()
        es_payload = {
            "connected": True,
            "cluster_name": info.get("cluster_name"),
            "version": (info.get("version") or {}).get("number"),
            "status": health_payload.get("status"),
            "number_of_nodes": health_payload.get("number_of_nodes"),
            "active_shards": health_payload.get("active_shards"),
        }
    except HTTPException as error:
        es_payload = {"connected": False, "error": error.detail}
    except Exception as error:
        es_payload = {"connected": False, "error": f"{type(error).__name__}: {error}"}
    return {
        "pipeline": {
            "configured": PIPELINE_RUNTIME.configured,
            "name": PIPELINE_RUNTIME.name,
        },
        "sessions": SESSION_STORE.stats(),
        "jobs": JOB_STORE.stats(),
        "elasticsearch": es_payload,
        "admin_mode": "STAGED_WRITE",
    }


@app.get("/api/admin/jobs", dependencies=[Depends(require_admin)])
def admin_jobs(limit: int = Query(default=100, ge=1, le=500)) -> dict[str, Any]:
    return {"items": JOB_STORE.list_public(limit), "stats": JOB_STORE.stats()}


@app.get("/api/admin/indices", dependencies=[Depends(require_admin)])
def admin_indices() -> dict[str, Any]:
    try:
        rows = _es_client().cat.indices(
            format="json", bytes="b", h="health,status,index,docs.count,store.size,pri,rep"
        )
        items = [
            dict(row)
            for row in rows
            if not str(row.get("index") or "").startswith(".")
        ]
        return {"items": items, "count": len(items)}
    except HTTPException:
        raise
    except Exception as error:
        raise _es_error(error) from error


@app.post("/api/admin/search", dependencies=[Depends(require_admin)])
def admin_search(payload: AdminSearchRequest) -> dict[str, Any]:
    index = _safe_index(payload.index)
    query_text = payload.query.strip()
    query_body: dict[str, Any]
    if query_text:
        # 현재 운영 인덱스는 원문을 ``search_text``에 적재한다. 과거의
        # title/content 필드만 검색하면 실제 데이터가 있어도 0건이 반환된다.
        # 사람이 쓰는 키워드 검색과 도메인 ID 접두사(HP, DA, MT 등) 검색을
        # 함께 지원해 관리자에서 어느 도메인의 청크인지 바로 확인할 수 있게 한다.
        should_queries: list[dict[str, Any]] = [
            {
                "simple_query_string": {
                    "query": query_text,
                    "fields": ["search_text^3", "chunk_id^2"],
                    "default_operator": "and",
                }
            }
        ]
        domain_prefix = query_text.upper()
        # HP 같은 두 글자 도메인뿐 아니라 ADMIN-TEST-001처럼 사람이 입력한
        # 전체/일부 청크 ID도 접두사로 찾는다. 한글 본문 검색은 위의
        # search_text 검색이 담당한다.
        if re.fullmatch(r"[A-Z0-9][A-Z0-9_-]{1,254}", domain_prefix):
            should_queries.append(
                {"prefix": {"chunk_id": {"value": f"{domain_prefix}-"}}}
            )
        query_body = {
            "bool": {
                "should": should_queries,
                "minimum_should_match": 1,
            }
        }
    else:
        query_body = {"match_all": {}}
    source_excludes = [
        value.strip()
        for value in os.getenv(
            "KDIC_ADMIN_SOURCE_EXCLUDES", "embedding,dense_vector,vector"
        ).split(",")
        if value.strip()
    ]
    try:
        response = _es_client().search(
            index=index,
            query=query_body,
            from_=payload.offset,
            size=payload.size,
            source_excludes=source_excludes,
            track_total_hits=True,
        )
        hits = response.get("hits") or {}
        total = hits.get("total") or {}
        items = [
            {
                "_id": row.get("_id"),
                "_index": row.get("_index"),
                "_score": row.get("_score"),
                "_source": row.get("_source") or {},
            }
            for row in hits.get("hits") or []
        ]
        return {
            "items": items,
            "total": int(total.get("value") or 0),
            "relation": total.get("relation"),
            "offset": payload.offset,
            "size": payload.size,
        }
    except HTTPException:
        raise
    except Exception as error:
        raise _es_error(error) from error


@app.get(
    "/api/admin/documents/{index}/{document_id:path}",
    dependencies=[Depends(require_admin)],
)
def admin_document(index: str, document_id: str) -> dict[str, Any]:
    safe_index = _safe_index(index)
    clean_id = document_id.strip()
    if not clean_id or len(clean_id) > 1_000:
        raise HTTPException(status_code=400, detail="유효한 문서 ID가 필요합니다.")
    try:
        response = _es_client().get(index=safe_index, id=clean_id)
        source = dict(response.get("_source") or {})
        for field_name in ("embedding", "dense_vector", "vector"):
            source.pop(field_name, None)
        return {
            "_id": response.get("_id"),
            "_index": response.get("_index"),
            "_version": response.get("_version"),
            "_source": source,
        }
    except HTTPException:
        raise
    except Exception as error:
        if getattr(error, "status_code", None) == 404:
            raise HTTPException(status_code=404, detail="문서를 찾지 못했습니다.") from error
        raise _es_error(error) from error


def _review_csv_path() -> Path:
    value = os.getenv("KDIC_REVIEW_CSV_PATH", "").strip()
    if not value:
        raise HTTPException(
            status_code=503, detail="KDIC_REVIEW_CSV_PATH이 설정되지 않았습니다."
        )
    return Path(value)


def _crawler_runtime_root() -> Path:
    base = os.getenv("KDIC_RUNTIME_DIR", "/opt/kdic/runtime")
    return Path(base) / "crawler"


# ============================================================
# 요구사항 3: 파라미터 테스트
# ============================================================
@app.post("/api/admin/params/compare", dependencies=[Depends(require_admin)])
def admin_params_compare(payload: ParamCompareRequest) -> dict[str, Any]:
    pipeline_module = _raw_pipeline_or_503()
    combos = [param_test.SearchParams(**combo.model_dump()) for combo in payload.combos]
    try:
        results = param_test.compare_params(pipeline_module, payload.question, combos)
    except Exception as error:
        raise HTTPException(
            status_code=502, detail=f"{type(error).__name__}: {error}"
        ) from error
    return {"question": payload.question, "results": results}


@app.post("/api/admin/params/activate", dependencies=[Depends(require_admin)])
def admin_params_activate(payload: ParamActivateRequest) -> dict[str, Any]:
    params = param_test.SearchParams(
        label=payload.label,
        dense_weight=payload.dense_weight,
        bm25_weight=payload.bm25_weight,
        candidate_depth=payload.candidate_depth,
        final_top_k=payload.final_top_k,
        rrf_k=payload.rrf_k,
    )
    conn = _db_connect()
    try:
        with _dict_cursor(conn) as cur:
            params_id = param_test.activate_params(
                cur, params, created_by=payload.created_by
            )
        conn.commit()
    finally:
        conn.close()
    return {"ok": True, "search_params_id": params_id}


@app.get("/api/admin/params", dependencies=[Depends(require_admin)])
def admin_params_list() -> dict[str, Any]:
    conn = _db_connect()
    try:
        with _dict_cursor(conn) as cur:
            cur.execute(
                "SELECT id, label, dense_weight, bm25_weight, candidate_depth, "
                "final_top_k, rrf_k, is_active, created_by, created_at "
                "FROM search_params ORDER BY created_at DESC LIMIT 100"
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return {"items": rows}


# ============================================================
# 요구사항 5: 평가 연동
# ============================================================
@app.post("/api/admin/eval-runs", dependencies=[Depends(require_admin)])
def admin_create_eval_run(payload: EvalRunRequest) -> dict[str, Any]:
    pipeline_module = _raw_pipeline_or_503()
    conn = _db_connect()
    try:
        with _dict_cursor(conn) as cur:
            try:
                eval_run_id = eval_runner.run_eval(
                    cur,
                    pipeline_module,
                    search_params_id=payload.search_params_id,
                    eval_query_ids=payload.eval_query_ids,
                    triggered_by=payload.triggered_by,
                )
                summary = eval_runner.summarize_eval_run(cur, eval_run_id)
            except ValueError as error:
                conn.rollback()
                raise HTTPException(status_code=400, detail=str(error)) from error
        conn.commit()
    finally:
        conn.close()
    return {"eval_run_id": eval_run_id, "summary": summary}


@app.get("/api/admin/eval-runs/{eval_run_id}", dependencies=[Depends(require_admin)])
def admin_get_eval_run(eval_run_id: str) -> dict[str, Any]:
    conn = _db_connect()
    try:
        with _dict_cursor(conn) as cur:
            summary = eval_runner.summarize_eval_run(cur, eval_run_id)
    finally:
        conn.close()
    if summary.get("query_count", 0) == 0:
        raise HTTPException(status_code=404, detail="평가 실행을 찾지 못했습니다.")
    return summary


# ============================================================
# 요구사항 4: 갱신 트리거링 (기존 URL 재수집)
# ============================================================
@app.post("/api/admin/reingest", dependencies=[Depends(require_admin)])
def admin_reingest(payload: ReingestRequest) -> dict[str, str]:
    job_id = reingest_trigger.trigger_reingest(
        _db_connect,
        kdic_final_pipeline,
        url_ids=payload.url_ids,
        review_csv_path=_review_csv_path(),
        runtime_root=_crawler_runtime_root(),
        triggered_by=payload.triggered_by,
        run_in_background=True,
    )
    return {"job_id": job_id}


# ============================================================
# 요구사항 1+2: 신규 URL 추가 + 미리보기
# ============================================================
@app.post("/api/admin/urls", dependencies=[Depends(require_admin)])
def admin_add_url(payload: NewUrlRequest) -> dict[str, str]:
    conn = _db_connect()
    try:
        outcome = ingest_trigger.trigger_new_url_ingest(
            conn,
            kdic_final_pipeline,
            source_url=payload.source_url,
            business_domain=payload.business_domain,
            review_csv_path=_review_csv_path(),
            runtime_root=_crawler_runtime_root(),
            triggered_by=payload.triggered_by,
        )
    finally:
        conn.close()
    return outcome


@app.get("/api/admin/previews", dependencies=[Depends(require_admin)])
def admin_list_previews() -> dict[str, Any]:
    conn = _db_connect()
    try:
        with _dict_cursor(conn) as cur:
            items = preview_review.list_pending_previews(cur)
    finally:
        conn.close()
    return {"items": items}


@app.get("/api/admin/previews/{preview_id}", dependencies=[Depends(require_admin)])
def admin_get_preview(preview_id: str) -> dict[str, Any]:
    conn = _db_connect()
    try:
        with _dict_cursor(conn) as cur:
            preview = preview_review.get_preview(cur, preview_id)
    finally:
        conn.close()
    if preview is None:
        raise HTTPException(status_code=404, detail="미리보기를 찾지 못했습니다.")
    return preview


@app.post(
    "/api/admin/previews/{preview_id}/approve", dependencies=[Depends(require_admin)]
)
def admin_approve_preview(preview_id: str, payload: ReviewActionRequest) -> dict[str, bool]:
    conn = _db_connect()
    try:
        with _dict_cursor(conn) as cur:
            try:
                preview_review.approve_preview(
                    cur, preview_id, reviewed_by=payload.reviewed_by
                )
            except ValueError as error:
                conn.rollback()
                raise HTTPException(status_code=409, detail=str(error)) from error
        conn.commit()
    finally:
        conn.close()
    return {"ok": True}


@app.post(
    "/api/admin/previews/{preview_id}/reject", dependencies=[Depends(require_admin)]
)
def admin_reject_preview(preview_id: str, payload: ReviewActionRequest) -> dict[str, bool]:
    conn = _db_connect()
    try:
        with _dict_cursor(conn) as cur:
            try:
                preview_review.reject_preview(
                    cur, preview_id, reviewed_by=payload.reviewed_by
                )
            except ValueError as error:
                conn.rollback()
                raise HTTPException(status_code=409, detail=str(error)) from error
        conn.commit()
    finally:
        conn.close()
    return {"ok": True}


@app.post(
    "/api/admin/previews/{preview_id}/publish", dependencies=[Depends(require_admin)]
)
def admin_publish_preview(preview_id: str) -> dict[str, Any]:
    """요구사항 1의 마지막 단계: 승인된 미리보기를 실제 검색 인덱스에 반영.
    새 청크 임베딩 생성을 위해 진짜 HCX_API_KEY가 필요하다."""
    pipeline_module = _raw_pipeline_or_503()
    conn = _db_connect()
    try:
        with _dict_cursor(conn) as cur:
            try:
                result = es_publish.publish_approved_preview(
                    cur, _es_client(), pipeline_module, preview_id
                )
            except ValueError as error:
                conn.rollback()
                raise HTTPException(status_code=409, detail=str(error)) from error
            except Exception as error:
                conn.rollback()
                raise HTTPException(
                    status_code=502, detail=f"{type(error).__name__}: {error}"
                ) from error
        conn.commit()
    finally:
        conn.close()
    # kdic_es_publish.publish_approved_preview()가 ES 반영과 동시에 지금
    # 떠있는 프로세스의 CHUNKS/DENSE_MATRIX 등에도 새 청크를 직접 append하므로,
    # 여기서 별도로 전체 리로드를 트리거할 필요가 없다 (그게 더 느리고,
    # CHUNKS의 출처인 로컬 ZIP 파일 자체는 안 바뀌어서 실효도 없었음).
    return result


@app.get("/api/admin/db-jobs/{job_id}", dependencies=[Depends(require_admin)])
def admin_get_db_job(job_id: str) -> dict[str, Any]:
    """/api/jobs/{id}는 채팅(job_type='chat')만 다루는 JOB_STORE 기준이라,
    ingest/reingest/eval_run 작업 상태는 이 라우트로 jobs 테이블을 직접 조회한다."""
    conn = _db_connect()
    try:
        with _dict_cursor(conn) as cur:
            cur.execute(
                "SELECT id, job_type, status, progress, stage, payload, result, "
                "error_message, created_at, started_at, finished_at "
                "FROM jobs WHERE id = %s",
                (job_id,),
            )
            row = cur.fetchone()
    finally:
        conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="작업을 찾지 못했습니다.")
    return row


_UVICORN_SERVER: Any = None
_UVICORN_THREAD: threading.Thread | None = None


def start_server_in_thread(
    host: str = "127.0.0.1", port: int = 8501, log_level: str = "info"
) -> dict[str, Any]:
    """Start the API inside a notebook kernel without blocking the next cell."""

    global _UVICORN_SERVER, _UVICORN_THREAD
    if _UVICORN_THREAD is not None and _UVICORN_THREAD.is_alive():
        return {"started": False, "reason": "ALREADY_RUNNING", "host": host, "port": port}
    import uvicorn

    config_value = uvicorn.Config(app, host=host, port=int(port), log_level=log_level)
    _UVICORN_SERVER = uvicorn.Server(config_value)
    _UVICORN_THREAD = threading.Thread(
        target=_UVICORN_SERVER.run, daemon=True, name="kdic-fastapi"
    )
    _UVICORN_THREAD.start()
    return {"started": True, "host": host, "port": int(port)}


def stop_server() -> None:
    global _UVICORN_SERVER
    if _UVICORN_SERVER is not None:
        _UVICORN_SERVER.should_exit = True


@app.on_event("shutdown")
def shutdown_job_service() -> None:
    JOB_SERVICE.shutdown()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("KDIC_API_HOST", "127.0.0.1"),
        port=_env_int("KDIC_API_PORT", 8501, 1, 65_535),
        log_level=os.getenv("KDIC_LOG_LEVEL", "info"),
    )
