# [실험] Dense 임베딩 입력에 title(+content)을 포함시킨 버전을 만듭니다.
# 기존 chunk_embeddings_hcx.jsonl은 content만 임베딩했는데(해시로 확인됨),
# 여기서는 "제목\n내용"을 임베딩해서 별도 파일로 저장합니다 — 프로덕션 파일은
# 건드리지 않고, 성능이 더 좋을 때만 채택할 실험용 산출물입니다.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import json
import time

from core.config import *
from core.integrity_check import *
from core.hcx_api import hcx_embed_text

OUT_PATH = EXPERIMENTS_ROOT / "chunk_embeddings_dense_structured.jsonl"


def build_structured_text(chunk: dict) -> str:
    title = (chunk.get("title") or "").strip()
    content = chunk.get("content") or ""
    return f"{title}\n{content}" if title else content


done_ids: set = set()
if OUT_PATH.exists():
    with open(OUT_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                done_ids.add(json.loads(line)["chunk_id"])
    print(f"이어서 진행: 이미 완료된 청크 {len(done_ids)}건 건너뜀", flush=True)

chunks = RESULT["chunks"]
print("전체 청크 수:", len(chunks), flush=True)

start = time.perf_counter()
with open(OUT_PATH, "a", encoding="utf-8") as f:
    for i, chunk in enumerate(chunks, 1):
        if chunk["chunk_id"] in done_ids:
            continue
        text = build_structured_text(chunk)
        vector = hcx_embed_text(text)
        f.write(json.dumps({
            "chunk_id": chunk["chunk_id"],
            "embedding": vector,
        }, ensure_ascii=False) + "\n")
        f.flush()

        if i % 50 == 0:
            elapsed = time.perf_counter() - start
            print(f"  진행: {i}/{len(chunks)} (경과 {elapsed:.0f}초)", flush=True)

print("완료:", OUT_PATH, flush=True)
