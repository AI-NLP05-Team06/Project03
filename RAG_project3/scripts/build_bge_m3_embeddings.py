# [Data-prep] BGE-M3를 로컬로 로드해서 전체 청크를 Dense+Sparse+Multi-vector로 재임베딩합니다.
# HCX API 키 불필요(로컬 모델 추론만 사용). reranker(bge-reranker-v2-m3)와는 별개 환경입니다 —
# 이 스크립트를 돌리기 전에 transformers를 최신 버전으로 두세요(reranker용 4.46.3 고정 X).
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import os
os.environ.setdefault("USE_TF", "0")

import json
import time

from core.config import BGE_M3_DATA_ROOT
from core.load_data import RESULT
from FlagEmbedding import BGEM3FlagModel

OUT_PATH = BGE_M3_DATA_ROOT / "chunk_embeddings_bge_m3_structured.jsonl"
BATCH_SIZE = 12
MAX_LENGTH = 1024


def build_text(chunk: dict) -> str:
    parts = [
        chunk.get("title") or "",
        chunk.get("section_title") or "",
        chunk.get("content") or "",
    ]
    return "\n".join(part for part in parts if part)


chunks = RESULT["chunks"]
print("전체 청크 수:", len(chunks), flush=True)

texts = [build_text(chunk) for chunk in chunks]
avg_len = sum(len(t) for t in texts) / len(texts)
print("평균 텍스트 길이(자):", round(avg_len, 1), flush=True)

print("BGE-M3 모델 로드 중...", flush=True)
start = time.perf_counter()
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
print("모델 로드 완료:", round(time.perf_counter() - start, 1), "초", flush=True)

print("전체 재임베딩 시작...", flush=True)
start = time.perf_counter()
result = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    max_length=MAX_LENGTH,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
elapsed = time.perf_counter() - start
print(f"재임베딩 완료: {elapsed:.1f}초 ({elapsed/len(chunks):.2f}초/건)", flush=True)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    for i, chunk in enumerate(chunks):
        record = {
            "chunk_id": chunk["chunk_id"],
            "dense": result["dense_vecs"][i].tolist(),
            "sparse": {k: float(v) for k, v in result["lexical_weights"][i].items()},
            "colbert": result["colbert_vecs"][i].tolist(),
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print("저장 완료:", OUT_PATH, flush=True)
print("파일 크기(MB):", round(OUT_PATH.stat().st_size / 1024 / 1024, 1), flush=True)
