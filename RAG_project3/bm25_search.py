# [BM25] kiwipiepy 형태소 분석 + rank_bm25(Okapi)로 청크를 인덱싱하고 검색합니다.
from __future__ import annotations

from kiwipiepy import Kiwi
from rank_bm25 import BM25Okapi

from load_data import RESULT

_kiwi = Kiwi()


def tokenize_korean(text: str) -> list[str]:
    if not text:
        return []
    return [token.form for token in _kiwi.tokenize(text)]


def _build_bm25_document(chunk: dict) -> str:
    parts = [
        chunk.get("title") or "",
        chunk.get("section_title") or "",
        chunk.get("content") or "",
    ]
    return "\n".join(part for part in parts if part)


bm25_chunks = RESULT["chunks"]
_bm25_corpus_tokens = [
    tokenize_korean(_build_bm25_document(chunk)) for chunk in bm25_chunks
]
bm25_index = BM25Okapi(_bm25_corpus_tokens)

print("BM25 인덱스 준비 완료:", len(bm25_chunks), "청크")


def bm25_search(
    question: str,
    *,
    top_k: int = 5,
    business_function: str | None = None,
) -> list[dict]:
    query_tokens = tokenize_korean(question)
    scores = bm25_index.get_scores(query_tokens)

    scored = []
    for chunk, score in zip(bm25_chunks, scores):
        if (
            business_function
            and chunk.get("business_function") != business_function
        ):
            continue
        scored.append({"score": float(score), "chunk": chunk})

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]
