import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

function parseCsv(text) {
  const source = text.replace(/^\uFEFF/, "");
  const matrix = [];
  let row = [];
  let cell = "";
  let quoted = false;
  for (let i = 0; i < source.length; i += 1) {
    const char = source[i];
    if (quoted) {
      if (char === '"' && source[i + 1] === '"') {
        cell += '"';
        i += 1;
      } else if (char === '"') quoted = false;
      else cell += char;
    } else if (char === '"') quoted = true;
    else if (char === ",") {
      row.push(cell);
      cell = "";
    } else if (char === "\n") {
      row.push(cell.replace(/\r$/, ""));
      matrix.push(row);
      row = [];
      cell = "";
    } else cell += char;
  }
  if (cell.length || row.length) {
    row.push(cell.replace(/\r$/, ""));
    matrix.push(row);
  }
  const headers = matrix[0];
  return matrix.slice(1).filter((values) => values.some(Boolean)).map((values) =>
    Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""])));
}

test("bundled evaluation rows reference real chunks", async () => {
  const [csv, jsonl] = await Promise.all([
    readFile(new URL("../public/data/evaluation.csv", import.meta.url), "utf8"),
    readFile(new URL("../public/data/chunks.jsonl", import.meta.url), "utf8"),
  ]);
  const rows = parseCsv(csv);
  const chunks = jsonl.trim().split(/\r?\n/).map(JSON.parse);
  const chunkIds = new Set(chunks.map((chunk) => chunk.chunk_id));
  const goldIds = rows.flatMap((row) => JSON.parse(row.gold_chunk_ids || "[]"));

  assert.equal(rows.length, 128);
  assert.equal(chunks.length, 427);
  assert.equal(new Set(chunks.map((chunk) => chunk.business_function)).size, 6);
  assert.ok(goldIds.length > rows.length);
  assert.deepEqual(goldIds.filter((id) => !chunkIds.has(id)), []);
});

test("review UI includes the core workflow", async () => {
  const source = await readFile(new URL("../app/gold-review-app.tsx", import.meta.url), "utf8");
  assert.match(source, /Gold 청크 보기/);
  assert.match(source, /전체 427개 청크/);
  assert.match(source, /새 질문 추가/);
  assert.match(source, /CSV 내보내기/);
  assert.match(source, /Gold에서 제거/);
  assert.match(source, /수정 내용 보내기/);
  assert.match(source, /팀원 수정본 합치기/);
  assert.match(source, /kdic-gold-review-patch-v1/);
});
