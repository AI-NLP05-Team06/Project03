# [Eval-run] Hybrid(pool=50) + Reranker(BGE-Reranker-v2-m3, N=25) 최종 확정값을 평가합니다.
# GPU(Colab 등)에서 돌리는 걸 권장합니다 — CPU에서는 문항당 수십 초가 걸릴 수 있습니다.

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "production"))
from core.config import *
from core.integrity_check import *
from retrieval.reranker import hybrid_rerank_search
from experiments.evaluation.eval_search import *

COMBO_NAME = "hybrid_rerank_pool50_top25_norerank_noexp"
SEARCH_TOP_K = 10


def rerank_search_fn(question: str) -> list[dict]:
    return hybrid_rerank_search(
        question,
        top_k=SEARCH_TOP_K,
        business_function=None,
    )


gold_records = load_gold_set()
print("평가 문항 수:", len(gold_records))

eval_result = evaluate_search(rerank_search_fn, gold_records)
print("\n=== Hybrid + Reranker(N=25) 최종 결과 ===")
print(json.dumps(eval_result["summary"], ensure_ascii=False, indent=2))

log_result(COMBO_NAME, eval_result["summary"])

detail_path = DETAIL_ROOT / f"{COMBO_NAME}.csv"
eval_result["detail"].to_csv(detail_path, index=False, encoding="utf-8-sig")
print("문항별 상세 결과 저장:", detail_path)
