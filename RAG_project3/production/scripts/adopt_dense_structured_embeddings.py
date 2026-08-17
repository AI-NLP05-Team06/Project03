# [채택] title+content 구조화 Dense 임베딩이 기존 content-only보다 성능이 좋다고
# 확인되어(evaluation/tune_dense_structured_hybrid.py 결과), 프로덕션 소스 데이터
# data/chunk_embeddings_hcx.jsonl을 이걸로 교체합니다. 되돌릴 수 있도록 원본은
# 백업해둡니다.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import hashlib
import json
import shutil
from datetime import datetime, timezone

from core.config import *
from core.integrity_check import *

SOURCE_PATH = Path("data/chunk_embeddings_hcx.jsonl")
BACKUP_PATH = EXPERIMENTS_ROOT / "chunk_embeddings_hcx_content_only_backup.jsonl"
STRUCTURED_PATH = EXPERIMENTS_ROOT / "chunk_embeddings_dense_structured.jsonl"

chunk_by_id = {c["chunk_id"]: c for c in RESULT["chunks"]}


def build_structured_text(chunk: dict) -> str:
    title = (chunk.get("title") or "").strip()
    content = chunk.get("content") or ""
    return f"{title}\n{content}" if title else content


structured_vectors: dict[str, list[float]] = {}
with open(STRUCTURED_PATH, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rec = json.loads(line)
            structured_vectors[rec["chunk_id"]] = rec["embedding"]

if not BACKUP_PATH.exists():
    shutil.copy(SOURCE_PATH, BACKUP_PATH)
    print("원본(content-only) 백업 완료:", BACKUP_PATH)
else:
    print("백업 파일이 이미 있어 건드리지 않음:", BACKUP_PATH)

now = datetime.now(timezone.utc).isoformat()
new_records = []
with open(SOURCE_PATH, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        chunk_id = rec["chunk_id"]
        chunk = chunk_by_id[chunk_id]
        structured_text = build_structured_text(chunk)

        rec["embedding"] = structured_vectors[chunk_id]
        rec["input_content_sha256"] = hashlib.sha256(
            structured_text.encode("utf-8")
        ).hexdigest()
        rec["cache_version"] = rec["cache_version"] + "-title-structured"
        rec["generated_at"] = now
        new_records.append(rec)

with open(SOURCE_PATH, "w", encoding="utf-8") as f:
    for rec in new_records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print("교체 완료:", SOURCE_PATH, f"({len(new_records)}건)")
print("되돌리려면: 이 파일을 백업 파일 내용으로 복원하면 됩니다 ->", BACKUP_PATH)
