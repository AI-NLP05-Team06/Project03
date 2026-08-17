# [Experiment/eval01] "dense v2 structured": evaluation/tune_dense_structured_hybrid.py의
# dense_search_structured()와 동일한 임베딩(title+content 포함, EXPERIMENTS_ROOT/
# chunk_embeddings_dense_structured.jsonl)을 쓰되, 그 파일은 import 시 평가까지
# 실행해버리는 스크립트라 순수 함수만 이 모듈로 옮겨왔습니다(기존 파일은 손대지 않음).
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "production"))

import numpy as np

from core.config import EXPERIMENTS_ROOT
from core.hcx_api import hcx_embed_text
from core.load_data import RESULT

STRUCTURED_PATH = EXPERIMENTS_ROOT / "chunk_embeddings_dense_structured.jsonl"

chunk_by_id = {c["chunk_id"]: c for c in RESULT["chunks"]}

structured_vectors: dict[str, np.ndarray] = {}
with open(STRUCTURED_PATH, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        structured_vectors[rec["chunk_id"]] = np.asarray(rec["embedding"], dtype=np.float32)

print("Dense v2 structured(title+content) 임베딩 로드 완료:", len(structured_vectors), "개")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def dense_structured_search(
    question: str,
    *,
    top_k: int = 5,
    business_function: str | None = None,
    min_score: float | None = None,
) -> list[dict]:
    query_vector = np.asarray(hcx_embed_text(question), dtype=np.float32)
    scored = []
    for chunk_id, vector in structured_vectors.items():
        chunk = chunk_by_id.get(chunk_id)
        if not chunk:
            continue
        if business_function and chunk.get("business_function") != business_function:
            continue
        score = cosine_similarity(query_vector, vector)
        if min_score is not None and score < min_score:
            continue
        scored.append({"score": score, "chunk": chunk})
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]
