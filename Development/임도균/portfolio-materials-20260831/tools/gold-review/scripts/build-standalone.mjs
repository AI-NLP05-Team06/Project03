import { build } from "esbuild";
import { mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const temp = resolve(root, ".standalone-build");
const output = resolve(root, "team-share", "KDIC_Gold_검수도구_팀공유.html");

await rm(temp, { recursive: true, force: true });
await mkdir(temp, { recursive: true });
await mkdir(dirname(output), { recursive: true });

await build({
  entryPoints: [resolve(root, "standalone-entry.tsx")],
  bundle: true,
  minify: true,
  format: "iife",
  target: ["chrome100", "edge100"],
  outfile: resolve(temp, "app.js"),
});

const [javascript, css, csv, jsonl] = await Promise.all([
  readFile(resolve(temp, "app.js"), "utf8"),
  readFile(resolve(temp, "app.css"), "utf8"),
  readFile(resolve(root, "public/data/evaluation.csv"), "utf8"),
  readFile(resolve(root, "public/data/chunks.jsonl"), "utf8"),
]);

function safeJson(value) {
  return JSON.stringify(value)
    .replaceAll("<", "\\u003c")
    .replaceAll("\u2028", "\\u2028")
    .replaceAll("\u2029", "\\u2029");
}

const html = `<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="KDIC RAG 평가데이터셋 Gold 청크 검수 도구">
  <title>KDIC Gold 검수 도구</title>
  <style>${css}</style>
</head>
<body>
  <div id="root"></div>
  <script>
    window.__KDIC_EVALUATION_CSV__ = ${safeJson(csv)};
    window.__KDIC_CHUNKS_JSONL__ = ${safeJson(jsonl)};
  </script>
  <script>${javascript}</script>
</body>
</html>`;

await writeFile(output, html, "utf8");
await rm(temp, { recursive: true, force: true });

console.log(output);
