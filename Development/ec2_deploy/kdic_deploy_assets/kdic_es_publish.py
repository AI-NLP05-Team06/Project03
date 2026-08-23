from __future__ import annotations

"""요구사항 1의 마지막 단계: 승인된 미리보기를 실제 Elasticsearch에 반영.

staging_preview가 'approved' 상태인 건에 대해:
1. 각 청크를 HCX로 새로 임베딩 (진짜 API 키 필요)
2. pipeline_module의 ES_INDEX_NAME(운영 중인 그 인덱스)에 upsert
3. pages / chunks_index 테이블에 기록
4. staging_preview.published_at 기록 (중복 반영 방지)
5. 지금 떠있는 서버 프로세스의 메모리(CHUNKS/CHUNKS_BY_ID/DENSE_MATRIX 등)에도
   새 청크를 바로 끼워넣어서, 서버 재시작이나 전체 리로드 없이 즉시 검색되게 함.

   (처음엔 "파이프라인 모듈을 통째로 다시 실행"하는 방식(핫 리로드)을 시도했는데,
   CHUNKS는 애초에 로컬 ZIP 파일에서 읽어오는 거라 그 파일 자체가 안 바뀌면
   다시 실행해봤자 여전히 예전 개수만 나온다는 걸 실측으로 확인함. 그래서
   "이미 계산해둔 새 청크+임베딩을 살아있는 모듈 객체에 직접 append"하는
   방식으로 변경 -- 더 빠르고, 실제로 문제를 해결함.)
"""

import threading
from typing import Any

import numpy as np

# 동시에 두 번 발행이 들어와서 서로의 append를 덮어쓰는 것(lost update)만
# 막으면 됨 -- 검색(읽기) 쪽은 파이썬의 속성 재할당이 원자적이라 별도 락 없이도
# "완전히 이전 상태" 또는 "완전히 이후 상태" 둘 중 하나만 보인다.
_APPEND_LOCK = threading.Lock()


def _append_chunks_to_live_module(
    pipeline_module: Any,
    new_chunks: list[dict[str, Any]],
    new_vectors: list[np.ndarray],
) -> None:
    with _APPEND_LOCK:
        pipeline_module.CHUNKS = pipeline_module.CHUNKS + new_chunks
        pipeline_module.CHUNKS_BY_ID = {
            **pipeline_module.CHUNKS_BY_ID,
            **{str(chunk["chunk_id"]): chunk for chunk in new_chunks},
        }
        pipeline_module.DENSE_CHUNK_IDS = pipeline_module.DENSE_CHUNK_IDS + [
            str(chunk["chunk_id"]) for chunk in new_chunks
        ]
        pipeline_module.DENSE_MATRIX = np.vstack(
            [pipeline_module.DENSE_MATRIX, np.vstack(new_vectors)]
        )
        pipeline_module.DENSE_VECTOR_BY_ID = {
            **pipeline_module.DENSE_VECTOR_BY_ID,
            **{
                str(chunk["chunk_id"]): vector
                for chunk, vector in zip(new_chunks, new_vectors)
            },
        }


def publish_approved_preview(
    cursor: Any,
    es_client: Any,
    pipeline_module: Any,
    preview_id: str,
) -> dict[str, Any]:
    cursor.execute(
        "SELECT id, source_url, chunk_preview, review_status, published_at "
        "FROM staging_preview WHERE id = %s",
        (preview_id,),
    )
    preview = cursor.fetchone()
    if preview is None:
        raise ValueError(f"미리보기를 찾지 못했습니다: {preview_id}")
    if preview["review_status"] != "approved":
        raise ValueError(
            f"승인된 미리보기만 반영할 수 있습니다 (현재: {preview['review_status']})"
        )
    if preview["published_at"] is not None:
        raise ValueError("이미 반영된 미리보기입니다.")

    chunks = preview["chunk_preview"] or []
    if not chunks:
        raise ValueError("반영할 청크가 없습니다.")

    business_function = chunks[0].get("business_function") or ""
    document_id = chunks[0].get("document_id") or ""

    # 1) pages 테이블에 페이지 등록 (없으면 새로, 있으면 상태만 갱신)
    cursor.execute(
        "INSERT INTO pages (source_url, business_category, status, last_ingested_at) "
        "VALUES (%s, %s, 'active', now()) "
        "ON CONFLICT (source_url) DO UPDATE SET "
        "status = 'active', last_ingested_at = now() "
        "RETURNING id",
        (preview["source_url"], business_function),
    )
    page_id = cursor.fetchone()["id"]

    # 2) 각 청크 실제 임베딩 생성 + ES 색인 액션 준비 (진짜 HCX 호출 지점)
    from elasticsearch import helpers as es_helpers

    actions = []
    new_vectors: list[np.ndarray] = []
    for chunk in chunks:
        search_text = pipeline_module.build_dense_structured_v2_text(chunk)
        # ES 인덱스가 dot_product 유사도를 쓰므로(코사인 유사도 = 내적, 단위벡터
        # 전제) 반드시 정규화된 벡터를 넣어야 한다. 원본 색인 코드도
        # prepare_dense_embeddings()에서 vector/norm으로 정규화한 뒤에만
        # DENSE_MATRIX/색인에 사용한다 -- 여기서도 똑같이 맞춰준다.
        vector = pipeline_module._normalize_vector(
            pipeline_module.embed_hcx_single(search_text)
        )
        new_vectors.append(vector)
        chunk_id = str(chunk["chunk_id"])
        actions.append(
            {
                "_op_type": "index",
                "_index": pipeline_module.ES_INDEX_NAME,
                "_id": chunk_id,
                "_source": {
                    "chunk_id": chunk_id,
                    "search_text": search_text,
                    "embedding": vector.tolist(),
                },
            }
        )
        cursor.execute(
            "INSERT INTO chunks_index (chunk_id, page_id, parent_doc_id, status) "
            "VALUES (%s, %s, %s, 'active') "
            "ON CONFLICT (chunk_id) DO UPDATE SET "
            "page_id = EXCLUDED.page_id, status = 'active'",
            (chunk_id, page_id, chunk.get("parent_doc_id") or document_id),
        )

    bulk_client = es_client.options(request_timeout=120)
    success, errors = es_helpers.bulk(
        bulk_client,
        actions,
        chunk_size=100,
        raise_on_error=False,
        raise_on_exception=False,
    )
    if errors:
        raise RuntimeError(f"ES 색인 중 일부 실패: {errors}")
    es_client.indices.refresh(index=pipeline_module.ES_INDEX_NAME)

    cursor.execute(
        "UPDATE staging_preview SET published_at = now() WHERE id = %s", (preview_id,)
    )

    # ES 반영은 이미 끝났으니, 이제 지금 떠있는 프로세스도 즉시 이 청크들을
    # 검색에 쓸 수 있도록 메모리 구조를 맞춘다.
    _append_chunks_to_live_module(pipeline_module, list(chunks), new_vectors)

    return {
        "preview_id": preview_id,
        "page_id": str(page_id),
        "chunk_count": len(actions),
        "index": pipeline_module.ES_INDEX_NAME,
        "es_errors": errors,
        "live_chunk_count": len(pipeline_module.CHUNKS),
    }
