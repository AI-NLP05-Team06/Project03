# [Eval-run] BGE-M3 Dense+Sparse+Multi-vector(구조화) 검색을 120문항 전체로 평가합니다.
# HCX API 불필요(로컬 모델 추론만 사용). reranker 노트북과는 별개 환경(최신 transformers)이어야 합니다.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "production"))

from core.config import *
from core.integrity_check import *
from retrieval.bge_m3_structured_search import bge_m3_structured_search
from experiments.evaluation.eval_search import *

COMBO_NAME = "bge_m3_structured_dense10_sparse03_colbert10"
SEARCH_TOP_K = 10


def bge_m3_search_fn(question: str) -> list[dict]:
    return bge_m3_structured_search(
        question,
        top_k=SEARCH_TOP_K,
        business_function=None,
    )


gold_records = load_gold_set()
print("평가 문항 수:", len(gold_records), flush=True)

eval_result = evaluate_search(bge_m3_search_fn, gold_records)
print("\n=== BGE-M3 Structured(Dense+Sparse+ColBERT) 결과 ===")
print(json.dumps(eval_result["summary"], ensure_ascii=False, indent=2))

log_result(COMBO_NAME, eval_result["summary"])

detail_path = DETAIL_ROOT / f"{COMBO_NAME}.csv"
eval_result["detail"].to_csv(detail_path, index=False, encoding="utf-8-sig")
print("문항별 상세 결과 저장:", detail_path)
