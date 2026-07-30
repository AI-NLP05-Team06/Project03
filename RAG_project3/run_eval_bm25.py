# [Eval-run] 9개 조합 실험: BM25 단독(Reranking 미적용) baseline을 평가합니다.
from config import *
from integrity_check import *
from bm25_search import *
from eval_search import *

COMBO_NAME = "bm25_norerank_noexp"
SEARCH_TOP_K = 10


def bm25_search_fn(question: str) -> list[dict]:
    return bm25_search(
        question,
        top_k=SEARCH_TOP_K,
        business_function=None,
    )


gold_records = load_gold_set()
print("평가 문항 수:", len(gold_records))

eval_result = evaluate_search(bm25_search_fn, gold_records)
print("\n=== BM25 baseline 결과 ===")
print(json.dumps(eval_result["summary"], ensure_ascii=False, indent=2))

log_result(COMBO_NAME, eval_result["summary"])

detail_path = RESULT_ROOT / f"detail_{COMBO_NAME}.csv"
eval_result["detail"].to_csv(detail_path, index=False, encoding="utf-8-sig")
print("문항별 상세 결과 저장:", detail_path)
