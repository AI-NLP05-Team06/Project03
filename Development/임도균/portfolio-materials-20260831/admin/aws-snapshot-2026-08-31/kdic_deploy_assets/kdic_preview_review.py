from __future__ import annotations

"""요구사항 2 (처리 결과 미리보기) 지원 모듈: 1)결과 저장 + 2)승인/반려.

신규 URL 크롤링 결과(파싱 텍스트+청크+메타데이터)를 staging_preview 테이블에
'검토 대기(pending)' 상태로 저장하고, 관리자가 승인/반려하면 상태를 바꾼다.
승인된 뒤 실제 Elasticsearch에 반영하는 건 별도 모듈(요구사항 1의 3단계, 아직 없음).
"""

from typing import Any

from psycopg2.extras import Json


def save_preview(
    cursor: Any,
    *,
    job_id: str,
    source_url: str,
    parsed_text: str,
    chunk_preview: list[dict[str, Any]],
) -> str:
    """요구사항 2의 1단계: 크롤링/청킹 결과를 검토 대기 상태로 저장."""
    cursor.execute(
        "INSERT INTO staging_preview (job_id, source_url, parsed_text, chunk_preview) "
        "VALUES (%s, %s, %s, %s) RETURNING id",
        (job_id, source_url, parsed_text, Json(chunk_preview)),
    )
    return str(cursor.fetchone()["id"])


def list_pending_previews(cursor: Any) -> list[dict[str, Any]]:
    cursor.execute(
        "SELECT id, job_id, source_url, chunk_preview, created_at "
        "FROM staging_preview WHERE review_status = 'pending' "
        "ORDER BY created_at"
    )
    return cursor.fetchall()


def get_preview(cursor: Any, preview_id: str) -> dict[str, Any] | None:
    cursor.execute(
        "SELECT id, job_id, source_url, parsed_text, chunk_preview, "
        "review_status, reviewed_by, reviewed_at, created_at "
        "FROM staging_preview WHERE id = %s",
        (preview_id,),
    )
    return cursor.fetchone()


def approve_preview(cursor: Any, preview_id: str, *, reviewed_by: str) -> None:
    """요구사항 2의 2단계: 관리자 승인. (ES 실반영은 별도 후속 단계)"""
    cursor.execute(
        "UPDATE staging_preview SET review_status = 'approved', "
        "reviewed_by = %s, reviewed_at = now() "
        "WHERE id = %s AND review_status = 'pending' RETURNING id",
        (reviewed_by, preview_id),
    )
    row = cursor.fetchone()
    if row is None:
        raise ValueError(f"승인 대기 중인 미리보기가 아닙니다: {preview_id}")


def reject_preview(cursor: Any, preview_id: str, *, reviewed_by: str) -> None:
    cursor.execute(
        "UPDATE staging_preview SET review_status = 'rejected', "
        "reviewed_by = %s, reviewed_at = now() "
        "WHERE id = %s AND review_status = 'pending' RETURNING id",
        (reviewed_by, preview_id),
    )
    row = cursor.fetchone()
    if row is None:
        raise ValueError(f"승인 대기 중인 미리보기가 아닙니다: {preview_id}")
