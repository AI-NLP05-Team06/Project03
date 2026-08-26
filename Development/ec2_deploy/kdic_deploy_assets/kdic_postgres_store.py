from __future__ import annotations

"""Postgres-backed drop-in replacements for InMemorySessionStore / InMemoryJobStore.

Interface matches 2026-08-23-kdic-service-core.py exactly, so this can be swapped
in with zero changes to fastapi-service.py's route handlers.
"""

import copy
import json
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping

import psycopg2
import psycopg2.extras
from psycopg2.pool import ThreadedConnectionPool


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _valid_uuid(value: str) -> str | None:
    """Return the normalized UUID string, or None if value isn't a valid UUID.

    Callers must not pass malformed ids straight into a `WHERE id = %s`
    query -- Postgres rejects non-UUID text with InvalidTextRepresentation,
    which surfaces as an unhandled 500 instead of a clean 404/KeyError.
    """
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, AttributeError, TypeError):
        return None


# ============================================================
# Connection pool (module-level, shared across requests)
# ============================================================
_POOL: ThreadedConnectionPool | None = None
_POOL_LOCK = threading.Lock()


def _pool() -> ThreadedConnectionPool:
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            import os

            dsn = os.environ["KDIC_DATABASE_URL"]  # postgres://user:pass@host:5432/kdic
            _POOL = ThreadedConnectionPool(minconn=1, maxconn=10, dsn=dsn)
        return _POOL


class _cursor:
    """Context manager: borrow a connection, commit/rollback, return it."""

    def __enter__(self):
        self.conn = _pool().getconn()
        self.cur = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        return self.cur

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()
        finally:
            self.cur.close()
            _pool().putconn(self.conn)


# ============================================================
# Sessions -> chat_sessions table
# ============================================================
@dataclass
class SessionRecord:
    session_id: str
    state: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)


class PostgresSessionStore:
    """Same public interface as InMemorySessionStore: get(), reset(), stats().
    Caller MUST call save(record) after mutating record.state (KDICJobService._run
    needs one added line for this -- see integration notes)."""

    def __init__(self, ttl_seconds: int = 86_400, max_sessions: int = 2_000):
        self.ttl_seconds = max(60, int(ttl_seconds))
        self.max_sessions = max(10, int(max_sessions))
        self._local_locks: dict[str, threading.RLock] = {}
        self._locks_guard = threading.Lock()

    def _lock_for(self, session_id: str) -> threading.RLock:
        with self._locks_guard:
            lock = self._local_locks.setdefault(session_id, threading.RLock())
            return lock

    def get(self, session_id: str) -> SessionRecord:
        key = _clean_text(session_id)
        if not key or len(key) > 200:
            raise ValueError("유효한 session_id가 필요합니다.")
        with _cursor() as cur:
            cur.execute(
                "SELECT session_id, state, extract(epoch from created_at) as created_at, "
                "extract(epoch from updated_at) as updated_at FROM chat_sessions WHERE session_id = %s",
                (key,),
            )
            row = cur.fetchone()
            if row is None:
                cur.execute(
                    "INSERT INTO chat_sessions (session_id, state) VALUES (%s, '{}'::jsonb) "
                    "ON CONFLICT (session_id) DO NOTHING "
                    "RETURNING session_id, state, extract(epoch from created_at) as created_at, "
                    "extract(epoch from updated_at) as updated_at",
                    (key,),
                )
                row = cur.fetchone()
                if row is None:
                    # concurrent insert race - just re-select
                    cur.execute(
                        "SELECT session_id, state, extract(epoch from created_at) as created_at, "
                        "extract(epoch from updated_at) as updated_at FROM chat_sessions WHERE session_id = %s",
                        (key,),
                    )
                    row = cur.fetchone()
        return SessionRecord(
            session_id=row["session_id"],
            state=dict(row["state"] or {}),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            lock=self._lock_for(key),
        )

    def save(self, record: SessionRecord) -> None:
        """Persist record.state back to Postgres. Call this after mutating state."""
        with _cursor() as cur:
            cur.execute(
                "UPDATE chat_sessions SET state = %s, updated_at = now() WHERE session_id = %s",
                (json.dumps(record.state, ensure_ascii=False), record.session_id),
            )

    def reset(self, session_id: str) -> None:
        key = _clean_text(session_id)
        with _cursor() as cur:
            cur.execute(
                "UPDATE chat_sessions SET state = '{}'::jsonb, updated_at = now() WHERE session_id = %s",
                (key,),
            )

    def stats(self) -> dict[str, Any]:
        with _cursor() as cur:
            cur.execute("SELECT count(*) AS n FROM chat_sessions")
            count = cur.fetchone()["n"]
        return {
            "backend": "postgres",
            "session_count": int(count),
            "ttl_seconds": self.ttl_seconds,
            "max_sessions": self.max_sessions,
        }


# ============================================================
# Jobs -> jobs table
# ============================================================
@dataclass
class JobRecord:
    job_id: str
    session_id: str
    question: str
    status: str = "queued"
    progress: int = 2
    stage: str = "질문을 전달했습니다."
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    suggestion_id: str = ""
    cache_key: str = ""
    result: dict[str, Any] | None = None
    raw_result: dict[str, Any] | None = None
    error: str = ""

    def public(self) -> dict[str, Any]:
        payload = {
            "job_id": self.job_id,
            "session_id": self.session_id,
            "question": self.question,
            "status": self.status,
            "progress": self.progress,
            "stage": self.stage,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if self.result is not None:
            payload["result"] = copy.deepcopy(self.result)
        if self.error:
            payload["error"] = self.error
        return payload


_STATUS_MAP_TO_DB = {
    "queued": "pending",
    "running": "running",
    "done": "done",
    "error": "failed",
}
_STATUS_MAP_FROM_DB = {v: k for k, v in _STATUS_MAP_TO_DB.items()}


def _row_to_job(row: Mapping[str, Any]) -> JobRecord:
    payload = row["payload"] or {}
    return JobRecord(
        job_id=str(row["id"]),
        session_id=row["session_id"] or "",
        question=str(payload.get("question") or ""),
        status=_STATUS_MAP_FROM_DB.get(row["status"], row["status"]),
        progress=int(row["progress"] or 0),
        stage=row["stage"] or "",
        created_at=row["created_at"].timestamp() if row["created_at"] else time.time(),
        updated_at=row["created_at"].timestamp() if row["created_at"] else time.time(),
        suggestion_id=_clean_text(payload.get("suggestion_id")).upper(),
        cache_key=_clean_text(payload.get("cache_key")),
        result=row["result"],
        raw_result=row["raw_result"],
        error=row["error_message"] or "",
    )


class PostgresJobStore:
    """Same public interface as InMemoryJobStore: create(), get(), update(),
    list_public(), stats() -- backed by the `jobs` table (job_type='chat')."""

    def __init__(self, ttl_seconds: int = 86_400, max_jobs: int = 5_000):
        self.ttl_seconds = max(300, int(ttl_seconds))
        self.max_jobs = max(50, int(max_jobs))

    def create(
        self,
        session_id: str,
        question: str,
        suggestion_id: str = "",
        cache_key: str = "",
    ) -> JobRecord:
        job_id = str(uuid.uuid4())
        payload = {
            "question": _clean_text(question),
            "suggestion_id": _clean_text(suggestion_id).upper(),
            "cache_key": _clean_text(cache_key),
        }
        with _cursor() as cur:
            cur.execute(
                "INSERT INTO jobs (id, job_type, status, session_id, progress, stage, payload) "
                "VALUES (%s, 'chat', 'pending', %s, 2, %s, %s) "
                "RETURNING id, session_id, status, progress, stage, payload, result, "
                "raw_result, error_message, created_at",
                (
                    job_id,
                    _clean_text(session_id),
                    "질문을 전달했습니다.",
                    json.dumps(payload, ensure_ascii=False),
                ),
            )
            row = cur.fetchone()
        return _row_to_job(row)

    def get(self, job_id: str) -> JobRecord | None:
        clean_id = _valid_uuid(job_id)
        if clean_id is None:
            return None
        with _cursor() as cur:
            cur.execute(
                "SELECT id, session_id, status, progress, stage, payload, result, "
                "raw_result, error_message, created_at FROM jobs WHERE id = %s",
                (clean_id,),
            )
            row = cur.fetchone()
        return _row_to_job(row) if row else None

    def update(self, job_id: str, **values: Any) -> JobRecord:
        set_clauses = []
        params: list[Any] = []
        if "status" in values:
            set_clauses.append("status = %s")
            params.append(_STATUS_MAP_TO_DB.get(values["status"], values["status"]))
            if values["status"] == "running":
                set_clauses.append("started_at = COALESCE(started_at, now())")
            if values["status"] in ("done", "error"):
                set_clauses.append("finished_at = now()")
        if "progress" in values:
            set_clauses.append("progress = %s")
            params.append(int(values["progress"]))
        if "stage" in values:
            set_clauses.append("stage = %s")
            params.append(values["stage"])
        if "result" in values:
            set_clauses.append("result = %s")
            params.append(json.dumps(values["result"], ensure_ascii=False))
        if "raw_result" in values:
            set_clauses.append("raw_result = %s")
            params.append(json.dumps(values["raw_result"], ensure_ascii=False))
        if "error" in values:
            set_clauses.append("error_message = %s")
            params.append(values["error"])
        clean_id = _valid_uuid(job_id)
        if clean_id is None:
            raise KeyError(job_id)
        if not set_clauses:
            record = self.get(job_id)
            if record is None:
                raise KeyError(job_id)
            return record
        params.append(clean_id)
        with _cursor() as cur:
            cur.execute(
                f"UPDATE jobs SET {', '.join(set_clauses)} WHERE id = %s "
                "RETURNING id, session_id, status, progress, stage, payload, result, "
                "raw_result, error_message, created_at",
                params,
            )
            row = cur.fetchone()
            if row is None:
                raise KeyError(job_id)
        return _row_to_job(row)

    def list_public(self, limit: int = 100) -> list[dict[str, Any]]:
        with _cursor() as cur:
            cur.execute(
                "SELECT id, session_id, status, progress, stage, payload, result, "
                "raw_result, error_message, created_at FROM jobs "
                "WHERE job_type = 'chat' ORDER BY created_at DESC LIMIT %s",
                (max(1, min(int(limit), 500)),),
            )
            rows = cur.fetchall()
        return [_row_to_job(row).public() for row in rows]

    def stats(self) -> dict[str, Any]:
        with _cursor() as cur:
            cur.execute(
                "SELECT status, count(*) AS n FROM jobs WHERE job_type = 'chat' GROUP BY status"
            )
            rows = cur.fetchall()
        statuses = {
            _STATUS_MAP_FROM_DB.get(row["status"], row["status"]): int(row["n"])
            for row in rows
        }
        return {
            "backend": "postgres",
            "job_count": sum(statuses.values()),
            "statuses": statuses,
            "ttl_seconds": self.ttl_seconds,
            "max_jobs": self.max_jobs,
        }


# ============================================================
# Validated recommendation answer bundles -> suggestion_answer_cache
# ============================================================
@dataclass
class CachedAnswerBundle:
    cache_key: str
    suggestion_id: str
    business: str
    keyword: str
    question: str
    public_result: dict[str, Any]
    raw_result: dict[str, Any]
    basis_result: dict[str, Any]
    pipeline_name: str
    runtime_revision: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    hit_count: int = 0


def _row_to_cached_bundle(row: Mapping[str, Any]) -> CachedAnswerBundle:
    return CachedAnswerBundle(
        cache_key=str(row["cache_key"]),
        suggestion_id=str(row["suggestion_id"]),
        business=str(row["business"]),
        keyword=str(row["keyword"]),
        question=str(row["question"]),
        public_result=dict(row["public_result"] or {}),
        raw_result=dict(row["raw_result"] or {}),
        basis_result=dict(row["basis_result"] or {}),
        pipeline_name=str(row["pipeline_name"]),
        runtime_revision=str(row["runtime_revision"]),
        created_at=row["created_at"].timestamp(),
        updated_at=row["updated_at"].timestamp(),
        hit_count=int(row["hit_count"] or 0),
    )


class PostgresSuggestionAnswerCache:
    """Persistent cache for exact server-registered recommendation questions."""

    _SELECT_COLUMNS = (
        "cache_key, suggestion_id, business, keyword, question, public_result, "
        "raw_result, basis_result, pipeline_name, runtime_revision, created_at, "
        "updated_at, hit_count"
    )

    def __init__(self, ttl_seconds: int = 2_592_000, max_entries: int = 2_000):
        self.ttl_seconds = max(60, int(ttl_seconds))
        self.max_entries = max(10, int(max_entries))

    def get(self, cache_key: str) -> CachedAnswerBundle | None:
        with _cursor() as cur:
            cur.execute(
                "UPDATE suggestion_answer_cache SET hit_count = hit_count + 1, "
                "last_hit_at = now(), updated_at = now() "
                "WHERE cache_key = %s AND validation_status = 'VALIDATED' "
                "AND expires_at > now() RETURNING " + self._SELECT_COLUMNS,
                (_clean_text(cache_key),),
            )
            row = cur.fetchone()
        return _row_to_cached_bundle(row) if row else None

    def peek(self, cache_key: str) -> CachedAnswerBundle | None:
        with _cursor() as cur:
            cur.execute(
                "SELECT " + self._SELECT_COLUMNS + " FROM suggestion_answer_cache "
                "WHERE cache_key = %s AND validation_status = 'VALIDATED' "
                "AND expires_at > now()",
                (_clean_text(cache_key),),
            )
            row = cur.fetchone()
        return _row_to_cached_bundle(row) if row else None

    def put(self, bundle: Any) -> None:
        with _cursor() as cur:
            cur.execute(
                "INSERT INTO suggestion_answer_cache ("
                "cache_key, suggestion_id, business, keyword, question, public_result, "
                "raw_result, basis_result, pipeline_name, runtime_revision, "
                "validation_status, expires_at) "
                "VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, "
                "%s, %s, 'VALIDATED', now() + (%s * interval '1 second')) "
                "ON CONFLICT (cache_key) DO UPDATE SET "
                "suggestion_id = EXCLUDED.suggestion_id, business = EXCLUDED.business, "
                "keyword = EXCLUDED.keyword, question = EXCLUDED.question, "
                "public_result = EXCLUDED.public_result, raw_result = EXCLUDED.raw_result, "
                "basis_result = EXCLUDED.basis_result, pipeline_name = EXCLUDED.pipeline_name, "
                "runtime_revision = EXCLUDED.runtime_revision, validation_status = 'VALIDATED', "
                "expires_at = EXCLUDED.expires_at, created_at = now(), updated_at = now(), "
                "last_hit_at = NULL, hit_count = 0",
                (
                    _clean_text(bundle.cache_key),
                    _clean_text(bundle.suggestion_id).upper(),
                    _clean_text(bundle.business),
                    _clean_text(bundle.keyword),
                    _clean_text(bundle.question),
                    json.dumps(bundle.public_result, ensure_ascii=False),
                    json.dumps(bundle.raw_result, ensure_ascii=False),
                    json.dumps(bundle.basis_result, ensure_ascii=False),
                    _clean_text(bundle.pipeline_name),
                    _clean_text(bundle.runtime_revision),
                    self.ttl_seconds,
                ),
            )
            cur.execute("DELETE FROM suggestion_answer_cache WHERE expires_at <= now()")
            cur.execute("SELECT count(*) AS n FROM suggestion_answer_cache")
            overflow = max(0, int(cur.fetchone()["n"]) - self.max_entries)
            if overflow:
                cur.execute(
                    "DELETE FROM suggestion_answer_cache WHERE cache_key IN ("
                    "SELECT cache_key FROM suggestion_answer_cache "
                    "ORDER BY updated_at ASC LIMIT %s)",
                    (overflow,),
                )

    def stats(self) -> dict[str, Any]:
        with _cursor() as cur:
            cur.execute(
                "SELECT count(*) FILTER (WHERE validation_status = 'VALIDATED' "
                "AND expires_at > now()) AS active, "
                "coalesce(sum(hit_count), 0) AS hits, "
                "count(*) FILTER (WHERE expires_at <= now()) AS expired "
                "FROM suggestion_answer_cache"
            )
            row = cur.fetchone()
        return {
            "backend": "postgres",
            "schema_version": "kdic-suggestion-answer-bundle-v5.1",
            "entry_count": int(row["active"] or 0),
            "hits": int(row["hits"] or 0),
            "expired": int(row["expired"] or 0),
            "ttl_seconds": self.ttl_seconds,
            "max_entries": self.max_entries,
        }
