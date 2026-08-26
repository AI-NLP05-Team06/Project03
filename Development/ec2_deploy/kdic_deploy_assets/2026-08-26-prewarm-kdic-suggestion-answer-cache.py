from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Mapping

import psycopg2
import psycopg2.extras


BASE_DIR = Path(__file__).resolve().parent
CORE_FILE = BASE_DIR / "2026-08-23-kdic-service-core.py"
POSTGRES_FILE = BASE_DIR / "kdic_postgres_store.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _request_json(
    base_url: str,
    path: str,
    *,
    payload: Mapping[str, Any] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers=headers,
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return dict(json.load(response))


def _runtime_namespace(health: Mapping[str, Any]) -> str:
    service_namespace = str(
        health.get("suggestion_cache_runtime_namespace") or ""
    ).strip()
    if service_namespace:
        return service_namespace
    build = health.get("runtime_build")
    build = dict(build) if isinstance(build, Mapping) else {}
    return ":".join(
        value
        for value in (
            str(health.get("pipeline") or "").strip(),
            str(build.get("build_sha256") or "").strip(),
            str(build.get("overlay_revision") or "").strip(),
        )
        if value
    )


def _basis_for_job(base_url: str, job_id: str) -> dict[str, Any]:
    try:
        return _request_json(
            base_url,
            "/api/basis",
            payload={"job_id": job_id},
            timeout=30.0,
        )
    except Exception:
        return {}


def _latest_reusable_job(
    question: str,
    expected_build: Mapping[str, Any],
    compatible_overlay_revisions: set[str],
):
    connection = psycopg2.connect(os.environ["KDIC_DATABASE_URL"])
    try:
        cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cursor.execute(
            "SELECT id, result, raw_result FROM jobs "
            "WHERE job_type = 'chat' AND status = 'done' "
            "AND payload->>'question' = %s AND raw_result IS NOT NULL "
            "ORDER BY created_at DESC LIMIT 20",
            (question,),
        )
        for row in cursor.fetchall():
            raw = dict(row["raw_result"] or {})
            build = raw.get("runtime_build")
            build = dict(build) if isinstance(build, Mapping) else {}
            if expected_build.get("build_sha256") and (
                build.get("build_sha256") != expected_build.get("build_sha256")
            ):
                continue
            if expected_build.get("overlay_revision") and (
                build.get("overlay_revision") not in compatible_overlay_revisions
            ):
                continue
            yield str(row["id"]), dict(row["result"] or {}), raw
    finally:
        connection.close()


def _poll_job(base_url: str, job_id: str, timeout_seconds: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        row = _request_json(base_url, f"/api/jobs/{job_id}", timeout=30.0)
        if row.get("status") in {"done", "error"}:
            return row
        time.sleep(0.75)
    raise TimeoutError(f"job timeout: {job_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8501")
    parser.add_argument("--job-timeout", type=float, default=300.0)
    parser.add_argument("--skip-existing-job-seed", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("KDIC_DATABASE_URL", "").strip():
        raise RuntimeError("KDIC_DATABASE_URL is required")

    core = _load_module("kdic_cache_prewarm_core", CORE_FILE)
    postgres = _load_module("kdic_cache_prewarm_postgres", POSTGRES_FILE)
    health = _request_json(args.base_url, "/api/health")
    if not health.get("ok"):
        raise RuntimeError("KDIC API health check failed")
    runtime_namespace = _runtime_namespace(health)
    runtime_build = dict(health.get("runtime_build") or {})
    compatible_overlay_revisions = {
        str(value).strip()
        for value in runtime_build.get("cache_compatible_overlay_revisions") or []
        if str(value).strip()
    }
    compatible_overlay_revisions.add(str(runtime_build.get("overlay_revision") or ""))
    cache = postgres.PostgresSuggestionAnswerCache(
        ttl_seconds=int(os.getenv("KDIC_SUGGESTION_CACHE_TTL_SECONDS", "2592000")),
        max_entries=int(os.getenv("KDIC_SUGGESTION_CACHE_MAX_ENTRIES", "2000")),
    )

    summary = {
        "catalog_rows": 0,
        "already_cached": 0,
        "seeded_from_jobs": 0,
        "generated_live": 0,
        "failed": [],
    }
    for index, suggestion in enumerate(core.suggestion_catalog(), start=1):
        summary["catalog_rows"] += 1
        cache_key = ":".join(
            [
                core.SUGGESTION_CACHE_SCHEMA_VERSION,
                runtime_namespace,
                suggestion["suggestion_id"],
            ]
        )
        if cache.peek(cache_key) is not None:
            summary["already_cached"] += 1
            print(json.dumps({"index": index, "status": "already_cached"}))
            continue

        seeded = False
        if not args.skip_existing_job_seed:
            for job_id, public, raw in _latest_reusable_job(
                suggestion["query"], runtime_build, compatible_overlay_revisions
            ):
                normalized_public = core.normalize_public_result(raw)
                raw_answer = core._explicit_answer_from_result(raw)
                if not raw_answer:
                    continue
                normalized_public["answer"] = raw_answer
                eligible, _ = core.KDICJobService._cache_eligibility(
                    normalized_public,
                    suggestion["business"],
                )
                if not eligible:
                    continue
                cache.put(
                    core.CachedAnswerBundle(
                        cache_key=cache_key,
                        suggestion_id=suggestion["suggestion_id"],
                        business=suggestion["business"],
                        keyword=suggestion["label"],
                        question=suggestion["query"],
                        public_result=normalized_public,
                        raw_result=raw,
                        basis_result=_basis_for_job(args.base_url, job_id),
                        pipeline_name=str(health.get("pipeline") or ""),
                        runtime_revision=runtime_namespace,
                    )
                )
                summary["seeded_from_jobs"] += 1
                seeded = True
                print(json.dumps({"index": index, "status": "seeded_from_job"}))
                break
        if seeded:
            continue

        submitted = _request_json(
            args.base_url,
            "/api/jobs",
            payload={
                "session_id": "cache-prewarm-" + uuid.uuid4().hex,
                "question": suggestion["query"],
                "suggestion_id": suggestion["suggestion_id"],
            },
        )
        job = _poll_job(args.base_url, str(submitted["job_id"]), args.job_timeout)
        result = dict(job.get("result") or {})
        cache_meta = dict(result.get("suggestion_cache") or {})
        if job.get("status") == "done" and cache_meta.get("stored"):
            summary["generated_live"] += 1
            print(json.dumps({"index": index, "status": "generated_live"}))
        else:
            failure = {
                "index": index,
                "suggestion_id": suggestion["suggestion_id"],
                "keyword": suggestion["label"],
                "job_status": job.get("status"),
                "reason": cache_meta.get("reason") or job.get("error") or "NOT_STORED",
            }
            summary["failed"].append(failure)
            print(json.dumps({"index": index, "status": "failed", "reason": failure["reason"]}))

    catalog = core.suggestion_catalog()
    current_ready = 0
    for suggestion in catalog:
        current_key = ":".join(
            [
                core.SUGGESTION_CACHE_SCHEMA_VERSION,
                runtime_namespace,
                suggestion["suggestion_id"],
            ]
        )
        if cache.peek(current_key) is not None:
            current_ready += 1

    summary["current_runtime_ready"] = current_ready
    summary["cache_stats"] = cache.stats()
    print(json.dumps({"summary": summary}, ensure_ascii=False, indent=2))
    if summary["failed"] or current_ready != len(catalog):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
