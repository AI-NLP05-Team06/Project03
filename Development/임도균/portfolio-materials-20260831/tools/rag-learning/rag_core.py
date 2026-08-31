from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


TOKEN_PATTERN = re.compile(r"[가-힣A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    """형태소 분석기 없이 이해하기 쉬운 검색 토큰을 만든다."""
    words = [word.lower() for word in TOKEN_PATTERN.findall(text)]
    korean_text = "".join(re.findall(r"[가-힣]", text))
    bigrams = [f"한글2:{korean_text[i:i + 2]}" for i in range(len(korean_text) - 1)]
    return words + bigrams


@dataclass(frozen=True)
class SearchResult:
    score: float
    chunk: dict
    matched_terms: list[str]


class ChunkSearchEngine:
    """chunks.jsonl을 읽어 BM25 방식으로 검색하는 학습용 검색기."""

    def __init__(self, chunks_path: Path):
        self.chunks_path = chunks_path
        self.chunks = self._load_chunks(chunks_path)
        self.term_frequencies: list[Counter[str]] = []
        self.document_frequencies: Counter[str] = Counter()
        self.document_lengths: list[int] = []
        self.business_indices: dict[str, list[int]] = defaultdict(list)

        for index, chunk in enumerate(self.chunks):
            searchable_text = " ".join(
                str(chunk.get(field, ""))
                for field in ("title", "section_title", "content")
            )
            frequencies = Counter(tokenize(searchable_text))
            self.term_frequencies.append(frequencies)
            self.document_frequencies.update(frequencies.keys())
            self.document_lengths.append(sum(frequencies.values()))
            self.business_indices[chunk["business_function"]].append(index)

        self.average_document_length = (
            sum(self.document_lengths) / len(self.document_lengths)
            if self.document_lengths
            else 0.0
        )

    @staticmethod
    def _load_chunks(path: Path) -> list[dict]:
        chunks: list[dict] = []
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{line_number}번째 JSONL 행을 읽을 수 없습니다.") from error

                required = ("chunk_id", "business_function", "content", "source_url")
                missing = [field for field in required if not chunk.get(field)]
                if missing:
                    raise ValueError(
                        f"{line_number}번째 청크에 필수 필드가 없습니다: {', '.join(missing)}"
                    )
                chunks.append(chunk)
        return chunks

    @property
    def business_functions(self) -> list[str]:
        return sorted(self.business_indices)

    def summary(self) -> dict:
        return {
            "chunk_count": len(self.chunks),
            "parent_document_count": len(
                {chunk.get("parent_doc_id") for chunk in self.chunks}
            ),
            "business_counts": {
                business: len(indices)
                for business, indices in sorted(self.business_indices.items())
            },
        }

    def search(
        self,
        question: str,
        business_function: str | None = None,
        top_k: int = 5,
    ) -> list[SearchResult]:
        query_terms = list(dict.fromkeys(tokenize(question)))
        if not query_terms:
            return []

        if business_function:
            candidate_indices = self.business_indices.get(business_function, [])
        else:
            candidate_indices = range(len(self.chunks))

        results: list[SearchResult] = []
        total_documents = len(self.chunks)
        k1 = 1.5
        b = 0.75

        for index in candidate_indices:
            frequencies = self.term_frequencies[index]
            document_length = self.document_lengths[index]
            score = 0.0
            matched_terms: list[str] = []

            for term in query_terms:
                term_frequency = frequencies.get(term, 0)
                if term_frequency == 0:
                    continue

                document_frequency = self.document_frequencies[term]
                inverse_document_frequency = math.log(
                    1 + (total_documents - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                length_adjustment = 1 - b + b * (
                    document_length / self.average_document_length
                )
                score += inverse_document_frequency * (
                    term_frequency * (k1 + 1)
                    / (term_frequency + k1 * length_adjustment)
                )
                matched_terms.append(term)

            if score > 0:
                results.append(
                    SearchResult(
                        score=round(score, 4),
                        chunk=self.chunks[index],
                        matched_terms=matched_terms,
                    )
                )

        results.sort(key=lambda result: result.score, reverse=True)
        return results[: max(1, min(top_k, 20))]

