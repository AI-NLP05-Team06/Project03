from __future__ import annotations

import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Timer
from urllib.parse import urlparse

from rag_core import ChunkSearchEngine


PROJECT_DIR = Path(__file__).resolve().parent
ENGINE = ChunkSearchEngine(PROJECT_DIR / "data" / "chunks.jsonl")
HOST = "127.0.0.1"
PORT = 8501


HTML = r"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>KDIC RAG 학습용 검색기</title>
  <style>
    :root { color-scheme: light; --ink:#17202a; --muted:#65717e; --line:#dce3e8; --blue:#1358a7; --soft:#f4f7fa; }
    * { box-sizing:border-box; }
    body { margin:0; background:#eef2f5; color:var(--ink); font-family:"Malgun Gothic",system-ui,sans-serif; }
    main { max-width:1080px; margin:0 auto; padding:36px 20px 64px; }
    h1 { margin:0 0 8px; font-size:30px; }
    .intro { color:var(--muted); line-height:1.65; margin-bottom:24px; }
    .panel,.result { background:white; border:1px solid var(--line); border-radius:14px; box-shadow:0 3px 14px rgba(23,32,42,.05); }
    .panel { padding:20px; margin-bottom:18px; }
    .flow { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:20px; }
    .flow span { background:#e9f2fc; color:#174f8e; padding:7px 10px; border-radius:999px; font-size:13px; }
    .flow b { align-self:center; color:#8592a0; }
    label { display:block; font-weight:700; margin:0 0 7px; }
    input,select,button { font:inherit; }
    input,select { width:100%; border:1px solid #bfcbd5; border-radius:9px; padding:11px 12px; background:white; }
    .grid { display:grid; grid-template-columns:1fr 230px 110px; gap:12px; align-items:end; }
    button { border:0; border-radius:9px; padding:12px 15px; background:var(--blue); color:white; font-weight:700; cursor:pointer; }
    button:hover { background:#0e477f; }
    .examples { display:flex; flex-wrap:wrap; gap:7px; margin-top:13px; }
    .examples button { padding:7px 10px; color:#314152; background:var(--soft); font-size:13px; font-weight:400; }
    #status { color:var(--muted); margin:16px 2px; }
    .result { padding:19px; margin:12px 0; }
    .rank { display:inline-block; min-width:34px; color:white; background:#263b50; border-radius:7px; padding:5px 7px; text-align:center; font-size:12px; }
    .score { margin-left:8px; font-size:13px; color:#4b5b6b; }
    .result h2 { font-size:19px; margin:13px 0 5px; }
    .meta { color:var(--muted); font-size:13px; line-height:1.7; }
    .content { white-space:pre-wrap; line-height:1.72; max-height:260px; overflow:auto; background:#fafbfc; padding:14px; border-radius:9px; margin:12px 0; }
    a { color:var(--blue); word-break:break-all; }
    details { margin-top:10px; }
    summary { cursor:pointer; color:#455767; }
    .notice { padding:12px 14px; background:#fff8db; border:1px solid #ecd987; border-radius:9px; line-height:1.6; }
    @media(max-width:760px) { .grid { grid-template-columns:1fr; } }
  </style>
</head>
<body>
<main>
  <h1>KDIC RAG 학습용 검색기</h1>
  <p class="intro">지금은 답변을 생성하지 않습니다. 질문에서 관련 청크를 찾고, 업무 필터가 검색 결과를 어떻게 바꾸는지 직접 확인하는 1단계입니다.</p>
  <div class="flow"><span>질문 입력</span><b>→</b><span>업무 범위 선택</span><b>→</b><span>관련 청크 검색</span><b>→</b><span>원문·출처 확인</span></div>
  <section class="panel">
    <div class="grid">
      <div><label for="question">질문</label><input id="question" value="예금은 얼마까지 보호되나요?"></div>
      <div><label for="business">업무 필터</label><select id="business"></select></div>
      <div><label for="topk">Top-K</label><select id="topk"><option>3</option><option selected>5</option><option>10</option></select></div>
    </div>
    <button id="search" style="margin-top:14px;width:100%">관련 자료 찾기</button>
    <div class="examples">
      <button data-q="예금은 얼마까지 보호되나요?" data-b="예금자보호제도">보호한도</button>
      <button data-q="착오송금 반환지원 신청 대상은 누구인가요?" data-b="착오송금 반환 신청">착오송금 신청대상</button>
      <button data-q="채무조정 신청에 필요한 서류가 무엇인가요?" data-b="채무조정 안내">채무조정 서류</button>
      <button data-q="미수령금은 어떻게 신청하나요?" data-b="고객 미수령금 신청">미수령금 신청</button>
    </div>
  </section>
  <div class="notice">비교 방법: 먼저 업무 필터를 ‘전체 업무’로 검색한 다음, 정확한 업무를 선택해 다시 검색해 보세요. 다른 업무의 비슷한 표현이 사라지는지 확인하면 됩니다.</div>
  <div id="status"></div>
  <section id="results"></section>
</main>
<script>
const q = document.querySelector('#question');
const business = document.querySelector('#business');
const topk = document.querySelector('#topk');
const statusBox = document.querySelector('#status');
const results = document.querySelector('#results');
function esc(value) { return String(value ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c])); }
async function init() {
  const response = await fetch('/api/summary');
  const data = await response.json();
  business.innerHTML = '<option value="">전체 업무 (필터 없음)</option>' + data.business_functions.map(x => `<option>${esc(x)}</option>`).join('');
  statusBox.textContent = `현재 데이터: ${data.summary.chunk_count}개 청크 · ${data.summary.parent_document_count}개 상위 문서 · ${data.business_functions.length}개 업무`;
}
async function search() {
  const question = q.value.trim();
  if (!question) return;
  statusBox.textContent = '검색 중…'; results.innerHTML = '';
  const response = await fetch('/api/search', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({question, business_function:business.value || null, top_k:Number(topk.value)})});
  const data = await response.json();
  statusBox.textContent = `${data.candidate_count}개 후보에서 상위 ${data.results.length}개를 찾았습니다. 업무 필터: ${data.business_function || '없음'}`;
  if (!data.results.length) { results.innerHTML = '<div class="result">일치하는 자료를 찾지 못했습니다.</div>'; return; }
  results.innerHTML = data.results.map((r,i) => `<article class="result">
    <span class="rank">${i+1}위</span><span class="score">검색 점수 ${r.score}</span>
    <h2>${esc(r.title || '(제목 없음)')}</h2>
    <div class="meta">업무: ${esc(r.business_function)} · 유형: ${esc(r.chunk_type)} · 청크 ID: ${esc(r.chunk_id)} · 상위 문서: ${esc(r.parent_doc_id)}</div>
    <details><summary>일치한 검색 단서 보기</summary><div class="meta">${r.matched_terms.map(esc).join(', ')}</div></details>
    <div class="content">${esc(r.content)}</div>
    <a href="${esc(r.source_url)}" target="_blank" rel="noopener">공식 원문 페이지 열기</a>
  </article>`).join('');
}
document.querySelector('#search').addEventListener('click', search);
q.addEventListener('keydown', event => { if(event.key === 'Enter') search(); });
document.querySelectorAll('.examples button').forEach(button => button.addEventListener('click', () => { q.value=button.dataset.q; business.value=button.dataset.b; search(); }));
init();
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/summary":
            self._send_json(
                {
                    "summary": ENGINE.summary(),
                    "business_functions": ENGINE.business_functions,
                }
            )
            return
        self._send_json({"error": "찾을 수 없는 주소입니다."}, 404)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/search":
            self._send_json({"error": "찾을 수 없는 주소입니다."}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length).decode("utf-8"))
            question = str(request.get("question", "")).strip()
            business_function = request.get("business_function") or None
            top_k = int(request.get("top_k", 5))
            if not question:
                raise ValueError("질문을 입력해 주세요.")
            if business_function and business_function not in ENGINE.business_functions:
                raise ValueError("존재하지 않는 업무 필터입니다.")

            found = ENGINE.search(question, business_function, top_k)
            candidate_count = (
                len(ENGINE.business_indices[business_function])
                if business_function
                else len(ENGINE.chunks)
            )
            self._send_json(
                {
                    "question": question,
                    "business_function": business_function,
                    "candidate_count": candidate_count,
                    "results": [
                        {
                            "score": result.score,
                            "matched_terms": result.matched_terms,
                            **{
                                field: result.chunk.get(field)
                                for field in (
                                    "chunk_id",
                                    "parent_doc_id",
                                    "business_function",
                                    "title",
                                    "chunk_type",
                                    "content",
                                    "source_url",
                                )
                            },
                        }
                        for result in found
                    ],
                }
            )
        except (ValueError, json.JSONDecodeError) as error:
            self._send_json({"error": str(error)}, 400)

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    url = f"http://{HOST}:{PORT}"
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"KDIC RAG 학습용 검색기를 시작했습니다: {url}")
    print("종료하려면 이 창에서 Ctrl+C를 누르세요.")
    Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n검색기를 종료합니다.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

