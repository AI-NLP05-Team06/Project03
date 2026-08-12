# [Experiment/eval01] retrieval/bm25_search.py와 동일한 구조이되, 형태소 분석기만
# kiwipiepy 대신 pynori(Elasticsearch Nori의 순수 파이썬 포팅)로 교체합니다.
# decompound_mode='NONE': 복합명사를 원형 그대로 두고 쪼개지 않습니다(Nori 기본 옵션 중 하나).
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pynori.korean_analyzer import KoreanAnalyzer
from rank_bm25 import BM25Okapi

from core.load_data import RESULT

_nori = KoreanAnalyzer(
    decompound_mode="NONE",
    infl_decompound_mode="NONE",
    discard_punctuation=True,
    output_unknown_unigrams=False,
    pos_filter=False,
)


def tokenize_korean_nori(text: str) -> list[str]:
    if not text:
        return []
    return _nori.do_analysis(text)["termAtt"]


def _build_bm25_document(chunk: dict) -> str:
    parts = [
        chunk.get("title") or "",
        chunk.get("section_title") or "",
        chunk.get("content") or "",
    ]
    return "\n".join(part for part in parts if part)


bm25_nori_chunks = RESULT["chunks"]
_bm25_nori_corpus_tokens = [
    tokenize_korean_nori(_build_bm25_document(chunk)) for chunk in bm25_nori_chunks
]
bm25_nori_index = BM25Okapi(_bm25_nori_corpus_tokens)

print("BM25(Nori, decompound_mode=NONE) 인덱스 준비 완료:", len(bm25_nori_chunks), "청크")


def bm25_nori_search(
    question: str,
    *,
    top_k: int = 5,
    business_function: str | None = None,
) -> list[dict]:
    query_tokens = tokenize_korean_nori(question)
    scores = bm25_nori_index.get_scores(query_tokens)

    scored = []
    for chunk, score in zip(bm25_nori_chunks, scores):
        if (
            business_function
            and chunk.get("business_function") != business_function
        ):
            continue
        scored.append({"score": float(score), "chunk": chunk})

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]
