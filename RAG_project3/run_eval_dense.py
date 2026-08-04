# [Eval-run] 9개 조합 실험 1단계: Dense 단독(Reranking 미적용) baseline을 평가합니다.
from config import *
from integrity_check import *
from search_answer import *
from eval_search import *  # pyright: ignore[reportMissingImports]

COMBO_NAME = "dense_norerank_noexp"
SEARCH_TOP_K = 10


def dense_search(question: str) -> list[dict]:
    return semantic_search_hcx(
        question,
        top_k=SEARCH_TOP_K,
        business_function=None,
        min_score=None,
    )


gold_records = load_gold_set()
print("평가 문항 수:", len(gold_records))

eval_result = evaluate_search(dense_search, gold_records)
print("\n=== Dense baseline 결과 ===")
print(json.dumps(eval_result["summary"], ensure_ascii=False, indent=2))

log_result(COMBO_NAME, eval_result["summary"])

detail_path = RESULT_ROOT / f"detail_{COMBO_NAME}.csv"
eval_result["detail"].to_csv(detail_path, index=False, encoding="utf-8-sig")
print("문항별 상세 결과 저장:", detail_path)
