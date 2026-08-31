import fs from "node:fs/promises";

const payload = JSON.parse(
  await fs.readFile("./KDIC_검색평가_AI분석데이터.json", "utf8"),
);
const chunkLines = (
  await fs.readFile("./corpus/KDIC_output/processed/chunks.jsonl", "utf8")
)
  .split(/\r?\n/)
  .filter(Boolean);
const chunks = new Map(
  chunkLines.map((line) => {
    const row = JSON.parse(line);
    return [row.chunk_id, row];
  }),
);
const methods = payload.methods.map((method) => ({
  name: method.name,
  byId: new Map(method.question_results.map((row) => [row.evaluation_id, row])),
}));
const ids = payload.methods[0].question_results.map((row) => row.evaluation_id);

function parseList(value) {
  if (Array.isArray(value)) return value.flat(Infinity);
  try {
    return JSON.parse(String(value ?? "[]")).flat(Infinity);
  } catch {
    return [];
  }
}

const failures = [];
for (const id of ids) {
  if (!methods.every((method) => Number(method.byId.get(id).mrr_at_10 ?? 0) === 0)) {
    continue;
  }
  const base = methods[0].byId.get(id);
  const goldIds = parseList(base.gold_chunk_ids);
  const rerankerIds = parseList(methods[4].byId.get(id).retrieved_chunk_ids).slice(0, 5);
  failures.push({
    evaluation_id: id,
    domain: base.domain,
    question: base.question,
    gold: goldIds.map((chunkId) => ({
      chunk_id: chunkId,
      title: chunks.get(chunkId)?.title,
      section_title: chunks.get(chunkId)?.section_title,
      content: chunks.get(chunkId)?.content?.slice(0, 320),
    })),
    reranker_top5: rerankerIds.map((chunkId) => ({
      chunk_id: chunkId,
      title: chunks.get(chunkId)?.title,
      section_title: chunks.get(chunkId)?.section_title,
      content: chunks.get(chunkId)?.content?.slice(0, 180),
    })),
  });
}

console.log(JSON.stringify(failures, null, 2));
