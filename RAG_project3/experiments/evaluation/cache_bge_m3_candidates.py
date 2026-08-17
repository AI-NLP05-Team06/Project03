# [Step 1/2] BGE-M3 structured(확정 가중치 0.5:0:1.5)로 문항당 top-30 후보를 뽑아
# JSON으로 저장합니다. reranker(구버전 transformers 필요)와 같은 프로세스에서 못 써서
# 2단계로 나눈 것 중 1단계 — 이 스크립트는 "최신 transformers" 환경에서 실행하세요.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "production"))

import json
import time

from core.config import *
from core.integrity_check import *
from retrieval.bge_m3_structured_search import bge_m3_structured_search
from experiments.evaluation.eval_search import load_gold_set

POOL_SIZE = 30
OUT_PATH = BGE_M3_DATA_ROOT / "bge_m3_structured_candidates.json"

gold_records = load_gold_set()
print("평가 문항 수:", len(gold_records), flush=True)

start = time.perf_counter()
cache = []
for i, record in enumerate(gold_records, 1):
    results = bge_m3_structured_search(record["question"], top_k=POOL_SIZE)
    cache.append({
        "evaluation_id": record["evaluation_id"],
        "candidates": [
            {"score": r["score"], "chunk_id": r["chunk"]["chunk_id"]}
            for r in results
        ],
    })
    if i % 20 == 0:
        print(f"  진행: {i}/{len(gold_records)} (경과 {time.perf_counter()-start:.0f}초)", flush=True)

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(cache, f, ensure_ascii=False)

print("저장 완료:", OUT_PATH, flush=True)
print("총 소요:", round(time.perf_counter() - start, 1), "초", flush=True)
