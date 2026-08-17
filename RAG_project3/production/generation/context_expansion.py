# [Phase3] 컨텍스트확장: top-5 근거 청크에 같은 문서의 인접 청크를 덧붙여 문맥을 넓힌다.
# 검색/재랭킹 순위는 그대로 두고, 답변생성 직전에 각 청크의 content만 확장한다.
from __future__ import annotations

from core.load_data import chunks

_CHUNKS_BY_DOC: dict[str, list[dict]] = {}
for _chunk in chunks:
    _CHUNKS_BY_DOC.setdefault(_chunk["parent_doc_id"], []).append(_chunk)
for _doc_chunks in _CHUNKS_BY_DOC.values():
    _doc_chunks.sort(key=lambda c: c["chunk_index"])


def expand_chunk_content(chunk: dict, *, window: int = 1) -> str:
    """같은 문서에서 chunk_index 기준 앞뒤 window개 청크의 content를 이어붙인다."""
    doc_chunks = _CHUNKS_BY_DOC.get(chunk.get("parent_doc_id"))
    if not doc_chunks:
        return chunk.get("content", "")

    index_in_doc = next(
        (i for i, c in enumerate(doc_chunks) if c["chunk_id"] == chunk.get("chunk_id")),
        None,
    )
    if index_in_doc is None:
        return chunk.get("content", "")

    start = max(0, index_in_doc - window)
    end = min(len(doc_chunks), index_in_doc + window + 1)
    return "\n\n".join(c.get("content", "") for c in doc_chunks[start:end])


def expand_search_results(search_results: list[dict], *, window: int = 1) -> list[dict]:
    """search_results의 각 chunk.content를 확장된 텍스트로 교체한 새 리스트를 반환한다.
    (원본 search_results는 건드리지 않음 — score·랭킹 등 나머지 필드는 그대로 유지)"""
    expanded = []
    for result in search_results:
        chunk = dict(result["chunk"])
        chunk["content"] = expand_chunk_content(chunk, window=window)
        expanded.append({**result, "chunk": chunk})
    return expanded
