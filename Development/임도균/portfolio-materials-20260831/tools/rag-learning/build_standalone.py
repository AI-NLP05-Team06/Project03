from __future__ import annotations

import json
from pathlib import Path

from app import HTML


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = PROJECT_DIR / "KDIC_RAG_학습용검색기_팀공유.html"
CHUNKS_PATH = PROJECT_DIR / "data" / "chunks.jsonl"


def load_compact_chunks() -> list[dict]:
    fields = (
        "chunk_id",
        "parent_doc_id",
        "business_function",
        "title",
        "chunk_type",
        "content",
        "source_url",
    )
    chunks: list[dict] = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                original = json.loads(line)
                chunks.append({field: original.get(field) for field in fields})
    return chunks


def build() -> Path:
    chunks_json = json.dumps(
        load_compact_chunks(),
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("<", "\\u003c")

    offline_search = r"""
<script>
// 이 HTML은 서버 없이 실행되므로 데이터와 검색 로직을 파일 안에 포함합니다.
const OFFLINE_CHUNKS = __CHUNKS_JSON__;
const TOKEN_RE = /[가-힣A-Za-z0-9]+/g;

function offlineTokenize(text) {
  const words = (String(text).match(TOKEN_RE) || []).map(word => word.toLowerCase());
  const korean = (String(text).match(/[가-힣]/g) || []).join('');
  const bigrams = [];
  for (let i = 0; i < korean.length - 1; i++) bigrams.push(`한글2:${korean.slice(i, i + 2)}`);
  return words.concat(bigrams);
}

const offlineTermFrequencies = [];
const offlineDocumentFrequencies = new Map();
const offlineDocumentLengths = [];
const offlineBusinessIndices = new Map();

for (let index = 0; index < OFFLINE_CHUNKS.length; index++) {
  const chunk = OFFLINE_CHUNKS[index];
  const tokens = offlineTokenize(`${chunk.title || ''} ${chunk.content || ''}`);
  const frequencies = new Map();
  for (const token of tokens) frequencies.set(token, (frequencies.get(token) || 0) + 1);
  offlineTermFrequencies.push(frequencies);
  offlineDocumentLengths.push(tokens.length);
  for (const token of frequencies.keys()) {
    offlineDocumentFrequencies.set(token, (offlineDocumentFrequencies.get(token) || 0) + 1);
  }
  if (!offlineBusinessIndices.has(chunk.business_function)) offlineBusinessIndices.set(chunk.business_function, []);
  offlineBusinessIndices.get(chunk.business_function).push(index);
}

const offlineAverageLength = offlineDocumentLengths.reduce((sum, value) => sum + value, 0) / OFFLINE_CHUNKS.length;
const offlineBusinesses = [...offlineBusinessIndices.keys()].sort((a, b) => a.localeCompare(b, 'ko'));

function offlineSearch(question, businessFunction, topK) {
  const queryTerms = [...new Set(offlineTokenize(question))];
  const candidates = businessFunction
    ? (offlineBusinessIndices.get(businessFunction) || [])
    : OFFLINE_CHUNKS.map((_, index) => index);
  const results = [];
  const totalDocuments = OFFLINE_CHUNKS.length;
  const k1 = 1.5;
  const b = 0.75;

  for (const index of candidates) {
    const frequencies = offlineTermFrequencies[index];
    const documentLength = offlineDocumentLengths[index];
    const matchedTerms = [];
    let score = 0;
    for (const term of queryTerms) {
      const termFrequency = frequencies.get(term) || 0;
      if (!termFrequency) continue;
      const documentFrequency = offlineDocumentFrequencies.get(term) || 0;
      const inverseDocumentFrequency = Math.log(
        1 + (totalDocuments - documentFrequency + 0.5) / (documentFrequency + 0.5)
      );
      const lengthAdjustment = 1 - b + b * documentLength / offlineAverageLength;
      score += inverseDocumentFrequency * (
        termFrequency * (k1 + 1) / (termFrequency + k1 * lengthAdjustment)
      );
      matchedTerms.push(term);
    }
    if (score > 0) results.push({
      score: Number(score.toFixed(4)),
      matched_terms: matchedTerms,
      ...OFFLINE_CHUNKS[index],
    });
  }
  results.sort((left, right) => right.score - left.score);
  return {candidates, results: results.slice(0, Math.max(1, Math.min(Number(topK) || 5, 20)))};
}

const nativeFetch = window.fetch.bind(window);
window.fetch = async function offlineFetch(url, options = {}) {
  if (url === '/api/summary') {
    const parentIds = new Set(OFFLINE_CHUNKS.map(chunk => chunk.parent_doc_id));
    return {json: async () => ({
      summary: {chunk_count: OFFLINE_CHUNKS.length, parent_document_count: parentIds.size},
      business_functions: offlineBusinesses,
    })};
  }
  if (url === '/api/search') {
    const request = JSON.parse(options.body || '{}');
    const found = offlineSearch(request.question || '', request.business_function || null, request.top_k || 5);
    return {json: async () => ({
      question: request.question,
      business_function: request.business_function || null,
      candidate_count: found.candidates.length,
      results: found.results,
    })};
  }
  return nativeFetch(url, options);
};
</script>
""".replace("__CHUNKS_JSON__", chunks_json)

    standalone = HTML.replace(
        "<script>",
        offline_search + "\n<script>",
        1,
    ).replace(
        "지금은 답변을 생성하지 않습니다.",
        "팀 공유용 단일 HTML입니다. 지금은 답변을 생성하지 않습니다.",
        1,
    )
    OUTPUT_PATH.write_text(standalone, encoding="utf-8")
    return OUTPUT_PATH


if __name__ == "__main__":
    result = build()
    print(result)

