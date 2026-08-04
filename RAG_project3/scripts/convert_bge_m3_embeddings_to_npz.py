# [Data-prep] chunk_embeddings_bge_m3_structured.jsonl(2GB, 텍스트)을
# numpy 바이너리(.npz, dense+colbert)와 sparse용 json으로 변환합니다.
# ColBERT(multi-vector)는 청크마다 토큰 수가 달라서, 전부 이어붙인 배열 + 시작/끝 offset으로 저장합니다.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import json
import time

import numpy as np

from core.config import BGE_M3_DATA_ROOT

IN_PATH = BGE_M3_DATA_ROOT / "chunk_embeddings_bge_m3_structured.jsonl"
OUT_NPZ = BGE_M3_DATA_ROOT / "chunk_embeddings_bge_m3_structured.npz"
OUT_SPARSE_JSON = BGE_M3_DATA_ROOT / "chunk_embeddings_bge_m3_sparse.json"

print("입력 파일 크기(MB):", round(IN_PATH.stat().st_size / 1024 / 1024, 1), flush=True)

chunk_ids: list[str] = []
dense_list: list[list[float]] = []
colbert_offsets = [0]
colbert_chunks: list[np.ndarray] = []
sparse_by_id: dict[str, dict[str, float]] = {}

start = time.perf_counter()
with open(IN_PATH, encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        record = json.loads(line)
        chunk_ids.append(record["chunk_id"])
        dense_list.append(record["dense"])

        colbert = np.asarray(record["colbert"], dtype=np.float32)
        colbert_chunks.append(colbert)
        colbert_offsets.append(colbert_offsets[-1] + colbert.shape[0])

        sparse_by_id[record["chunk_id"]] = record["sparse"]

        if i % 100 == 0:
            print(f"  변환 진행: {i}건, 경과 {time.perf_counter()-start:.0f}초", flush=True)

print("전체", len(chunk_ids), "건 읽기 완료:", round(time.perf_counter() - start, 1), "초", flush=True)

dense_arr = np.asarray(dense_list, dtype=np.float32)
colbert_concat = np.concatenate(colbert_chunks, axis=0)
colbert_offsets_arr = np.asarray(colbert_offsets, dtype=np.int64)
chunk_ids_arr = np.asarray(chunk_ids)

print("dense shape:", dense_arr.shape, flush=True)
print("colbert 전체 shape(이어붙인 것):", colbert_concat.shape, flush=True)

np.savez_compressed(
    OUT_NPZ,
    chunk_ids=chunk_ids_arr,
    dense=dense_arr,
    colbert=colbert_concat,
    colbert_offsets=colbert_offsets_arr,
)
print("npz 저장 완료:", OUT_NPZ, flush=True)
print("npz 크기(MB):", round(OUT_NPZ.stat().st_size / 1024 / 1024, 1), flush=True)

with open(OUT_SPARSE_JSON, "w", encoding="utf-8") as f:
    json.dump(sparse_by_id, f, ensure_ascii=False)
print("sparse json 저장 완료:", OUT_SPARSE_JSON, flush=True)
print("sparse json 크기(MB):", round(OUT_SPARSE_JSON.stat().st_size / 1024 / 1024, 1), flush=True)
