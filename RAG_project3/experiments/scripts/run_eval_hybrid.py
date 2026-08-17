# [Eval-run] 9개 조합 실험: Hybrid(Weighted-sum Min-Max, Reranking 미적용) baseline을 평가합니다.

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "production"))
from core.config import *
from core.integrity_check import *
from retrieval.hybrid_search import *
from experiments.evaluation.eval_search import *

COMBO_NAME = "hybrid_wsum_minmax_w70_30_pool10_norerank_noexp"
SEARCH_TOP_K = 10


def hybrid_search_fn(question: str) -> list[dict]:
    return hybrid_search(
        question,
        top_k=SEARCH_TOP_K,
        business_function=None,
    )


gold_records = load_gold_set()
print("평가 문항 수:", len(gold_records))

eval_result = evaluate_search(hybrid_search_fn, gold_records)
print("\n=== Hybrid(RRF) baseline 결과 ===")
print(json.dumps(eval_result["summary"], ensure_ascii=False, indent=2))

log_result(COMBO_NAME, eval_result["summary"])

detail_path = DETAIL_ROOT / f"{COMBO_NAME}.csv"
eval_result["detail"].to_csv(detail_path, index=False, encoding="utf-8-sig")
print("문항별 상세 결과 저장:", detail_path)
