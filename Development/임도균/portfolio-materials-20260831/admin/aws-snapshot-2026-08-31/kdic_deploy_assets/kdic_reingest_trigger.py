from __future__ import annotations

"""요구사항 4 (갱신 트리거링) 지원 모듈.

kdic_final_pipeline.py(실전 프로젝트 1)의 run_pipeline()은 동기/블로킹 함수라
관리자 화면에서 바로 부르면 응답이 몇 초~몇 분간 안 옴. 그래서 jobs 테이블에
job_type='reingest' 행을 만들고, 백그라운드 스레드에서 run_pipeline을 실행하며
상태(pending/running/done/failed)를 갱신한다.

주의: psycopg2 커넥션은 스레드 간 공유가 안전하지 않다. 그래서 conn을 직접
받지 않고 conn_factory(인자 없이 새 커넥션을 만드는 콜러블)를 받는다 --
호출한 쪽(FastAPI 요청 스레드)의 커넥션과 백그라운드 스레드의 커넥션이
완전히 분리되어, 요청이 먼저 끝나 커넥션을 닫아도 백그라운드 작업이
안전하게 계속된다.
"""

import threading
from pathlib import Path
from typing import Any, Callable

from psycopg2.extras import Json, RealDictCursor


def create_reingest_job(conn, *, url_ids: list[str], triggered_by: str = "") -> str:
    """jobs 행을 pending 상태로 만들고 job_id를 반환. (실제 실행은 별도)"""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            "INSERT INTO jobs (job_type, status, payload) "
            "VALUES ('reingest', 'pending', %s) RETURNING id",
            (Json({"url_ids": url_ids, "triggered_by": triggered_by}),),
        )
        job_id = str(cur.fetchone()["id"])
    conn.commit()
    return job_id


def _run_reingest_sync(
    conn_factory: Callable[[], Any],
    job_id: str,
    kdic_final_pipeline_module: Any,
    *,
    url_ids: list[str],
    review_csv_path: str | Path,
    runtime_root: str | Path,
) -> None:
    conn = conn_factory()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "UPDATE jobs SET status = 'running', started_at = now(), "
                "stage = %s WHERE id = %s",
                (f"{len(url_ids)}개 URL 재수집 중", job_id),
            )
        conn.commit()

        try:
            config = kdic_final_pipeline_module.PipelineConfig(run_only_url_ids=url_ids)
            result = kdic_final_pipeline_module.run_pipeline(
                review_csv_path=review_csv_path,
                runtime_root=runtime_root,
                config=config,
            )
            results_df = result["results_df"]
            summary = {
                "processed_count": int(len(results_df)),
                "status_counts": results_df["status"].value_counts().to_dict(),
                "rows": results_df[
                    ["url_id", "business_function", "page_title", "status", "content_chars"]
                ].to_dict(orient="records"),
            }
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "UPDATE jobs SET status = 'done', progress = 100, "
                    "stage = '재수집 완료', finished_at = now(), result = %s WHERE id = %s",
                    (Json(summary), job_id),
                )
            conn.commit()
        except Exception as error:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "UPDATE jobs SET status = 'failed', finished_at = now(), "
                    "error_message = %s WHERE id = %s",
                    (f"{type(error).__name__}: {error}", job_id),
                )
            conn.commit()
            raise
    finally:
        conn.close()


def trigger_reingest(
    conn_factory: Callable[[], Any],
    kdic_final_pipeline_module: Any,
    *,
    url_ids: list[str],
    review_csv_path: str | Path,
    runtime_root: str | Path,
    triggered_by: str = "",
    run_in_background: bool = True,
) -> str:
    """요구사항 4: 기존 URL(들)의 재수집·재적재를 트리거하고 job_id 반환.
    관리자 화면은 이 job_id로 상태(진행중/완료/실패)를 폴링하면 됨.

    conn_factory: 인자 없이 새 DB 커넥션을 반환하는 콜러블. (예:
    lambda: psycopg2.connect(DATABASE_URL)) 호출자의 커넥션과 백그라운드
    스레드의 커넥션을 분리하기 위해 conn이 아니라 factory를 받는다."""

    creation_conn = conn_factory()
    try:
        job_id = create_reingest_job(
            creation_conn, url_ids=url_ids, triggered_by=triggered_by
        )
    finally:
        creation_conn.close()

    kwargs = dict(
        url_ids=url_ids,
        review_csv_path=review_csv_path,
        runtime_root=runtime_root,
    )
    if run_in_background:
        thread = threading.Thread(
            target=_run_reingest_sync,
            args=(conn_factory, job_id, kdic_final_pipeline_module),
            kwargs=kwargs,
            daemon=True,
        )
        thread.start()
    else:
        _run_reingest_sync(conn_factory, job_id, kdic_final_pipeline_module, **kwargs)

    return job_id
