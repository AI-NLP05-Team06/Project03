from __future__ import annotations

"""요구사항 1(신규 URL 추가)+2(미리보기)를 잇는 모듈.

흐름: 기본값 행 생성 -> 기존 42개 매니페스트에 덧붙임 -> run_pipeline을 그
url_id 하나로 스코프 실행 -> 결과 chunks를 staging_preview에 저장(검토 대기).
ES에 실제로 반영하는 건 여기 포함 안 됨 (승인 후 별도 단계, 아직 미구현).
"""

from pathlib import Path
from typing import Any

import pandas as pd
from psycopg2.extras import Json

from kdic_new_url_defaults import build_default_manifest_row, append_new_url_to_manifest
from kdic_preview_review import save_preview


def trigger_new_url_ingest(
    conn,
    kdic_final_pipeline_module: Any,
    *,
    source_url: str,
    business_domain: str,
    review_csv_path: str | Path,
    runtime_root: str | Path,
    triggered_by: str = "",
) -> dict[str, str]:
    """요구사항 1+2: 신규 URL을 크롤링해서 staging_preview에 검토 대기로 저장.
    반환값: {"job_id":..., "preview_id":..., "url_id":...}"""

    import psycopg2.extras

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "INSERT INTO jobs (job_type, status, payload) "
            "VALUES ('ingest', 'pending', %s) RETURNING id",
            (Json({"source_url": source_url, "business_domain": business_domain}),),
        )
        job_id = str(cur.fetchone()["id"])
    conn.commit()

    # review_csv_path는 "지금까지 추가된 모든 URL을 포함한 최신 매니페스트"로
    # 취급한다 -- 매번 원본 42개짜리 파일로 되돌아가면 URL을 두 번째로 추가할 때
    # 첫 번째로 추가한 게 사라진다. 그래서 새 행을 덧붙인 뒤 같은 경로에 다시
    # 써서, 다음 번 신규 URL 추가가 이번 결과 위에 쌓이도록 한다.
    review_df = pd.read_csv(review_csv_path, encoding="utf-8-sig", dtype=str).fillna("")
    existing_ids = set(review_df["문서_ID"])
    new_row = build_default_manifest_row(
        source_url=source_url,
        business_domain=business_domain,
        existing_ids=existing_ids,
    )
    updated_manifest, url_id = append_new_url_to_manifest(review_df, new_row)

    Path(runtime_root).mkdir(parents=True, exist_ok=True)
    updated_manifest.to_csv(review_csv_path, index=False, encoding="utf-8-sig")
    updated_csv_path = review_csv_path

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "UPDATE jobs SET status = 'running', started_at = now(), "
            "stage = %s WHERE id = %s",
            (f"신규 URL 크롤링 중: {url_id}", job_id),
        )
    conn.commit()

    try:
        config = kdic_final_pipeline_module.PipelineConfig(run_only_url_ids=[url_id])
        result = kdic_final_pipeline_module.run_pipeline(
            review_csv_path=updated_csv_path,
            runtime_root=runtime_root,
            config=config,
        )

        doc = next(
            (d for d in result["documents"] if d.get("doc_id") == url_id
             or str(d.get("doc_id", "")).startswith(url_id)),
            None,
        )
        chunks = [
            c for c in result["chunks"]
            if str(c.get("document_id", "")).startswith(url_id)
        ]
        parsed_text = "\n\n".join(
            block.get("text", "") for block in (doc or {}).get("blocks", [])
            if isinstance(block, dict) and block.get("text")
        ) if doc else ""

        # 표시용(미리보기 화면)이면서 동시에 승인 후 ES 색인용 원본이기도 하므로,
        # build_dense_structured_v2_text()가 필요로 하는 title/section_title/content와
        # ES 문서에 필요한 business_function/source_url을 전부 보존해 둔다.
        chunk_preview = [
            {
                "chunk_id": chunk.get("chunk_id"),
                "document_id": chunk.get("document_id"),
                "parent_doc_id": chunk.get("parent_doc_id"),
                "chunk_type": chunk.get("chunk_type"),
                "title": chunk.get("title"),
                "section_title": chunk.get("section_title"),
                "content": chunk.get("content") or "",
                "char_count": len(chunk.get("content") or ""),
                "business_function": chunk.get("business_function"),
                "source_url": chunk.get("source_url"),
            }
            for chunk in chunks
        ]

        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            preview_id = save_preview(
                cur,
                job_id=job_id,
                source_url=source_url,
                parsed_text=parsed_text,
                chunk_preview=chunk_preview,
            )
            cur.execute(
                "UPDATE jobs SET status = 'done', progress = 100, "
                "stage = '미리보기 준비 완료 (승인 대기)', finished_at = now(), "
                "result = %s WHERE id = %s",
                (Json({"url_id": url_id, "preview_id": preview_id, "chunk_count": len(chunks)}), job_id),
            )
        conn.commit()

        return {"job_id": job_id, "preview_id": preview_id, "url_id": url_id}
    except Exception as error:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE jobs SET status = 'failed', finished_at = now(), "
                "error_message = %s WHERE id = %s",
                (f"{type(error).__name__}: {error}", job_id),
            )
        conn.commit()
        raise
