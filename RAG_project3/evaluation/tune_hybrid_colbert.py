# [Tune] 확정 Hybrid(Dense+BM25)에 ColBERT(Multi-vector)를 세 번째 축으로 추가해서
# 3-way 가중치(dense:bm25:colbert)를 스윕합니다. Dense/BM25는 기존 확정 파이프라인 그대로
# (HCX API dense, kiwipiepy+rank_bm25) 가져오고, ColBERT만 로컬 BGE-M3에서 추가로 뽑습니다.
# reranker(구버전 transformers)와는 안 섞이니 최신 transformers 환경에서 실행하세요.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import time

from core.config import *
from core.integrity_check import *
from retrieval.search_answer import semantic_search_hcx
from retrieval.bm25_search import bm25_search
from retrieval.bge_m3_structured_search import (
    CHUNK_IDS,
    CHUNK_BY_ID,
    compute_colbert_scores,
)
from evaluation.eval_search import (
    load_gold_set,
    hit_at_k,
    recall_at_k,
    mrr_at_k,
    average_precision_at_k,
    complete_at_k,
    ndcg_at_k,
    precision_at_k,
    f1_at_k,
    log_result,
)

TOP_K_FINAL = 5
N_CHUNKS = len(CHUNK_IDS)
CHUNK_ID_TO_IDX = {cid: i for i, cid in enumerate(CHUNK_IDS)}

DENSE_OPTIONS = [0.5, 1.0, 1.5, 2.0]
BM25_OPTIONS = [0.15, 0.3, 0.5, 0.7]
COLBERT_OPTIONS = [0.0, 0.5, 1.0, 1.5]  # 0.0 포함 = "ColBERT 안 쓴 원래 Hybrid"도 그리드 안에서 같이 비교

gold_records = load_gold_set()
print("평가 문항 수:", len(gold_records), flush=True)


def _minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return np.ones_like(arr)
    return (arr - lo) / (hi - lo)


def _results_to_array(results: list[dict]) -> np.ndarray:
    arr = np.zeros(N_CHUNKS, dtype=np.float32)
    for r in results:
        idx = CHUNK_ID_TO_IDX.get(r["chunk"]["chunk_id"])
        if idx is not None:
            arr[idx] = r["score"]
    return arr


# ============================================================
# 1) 질문당 한 번만: Dense/BM25/ColBERT 점수를 전체 청크에 대해 계산 + min-max 정규화 캐싱
# ============================================================
start = time.perf_counter()
cache: list[tuple[dict, np.ndarray, np.ndarray, np.ndarray]] = []
for i, record in enumerate(gold_records, 1):
    dense_results = semantic_search_hcx(
        record["question"], top_k=N_CHUNKS, business_function=None, min_score=None,
    )
    bm25_results = bm25_search(record["question"], top_k=N_CHUNKS, business_function=None)
    colbert_arr = compute_colbert_scores(record["question"])

    dense_norm = _minmax(_results_to_array(dense_results))
    bm25_norm = _minmax(_results_to_array(bm25_results))
    colbert_norm = _minmax(colbert_arr)

    cache.append((record, dense_norm, bm25_norm, colbert_norm))
    if i % 20 == 0:
        print(f"  진행: {i}/{len(gold_records)} (경과 {time.perf_counter()-start:.0f}초)", flush=True)

print("캐시 완료:", len(cache), "건,", round(time.perf_counter() - start, 1), "초", flush=True)


# ============================================================
# 2) 평가 함수
# ============================================================
def evaluate_weights(weights: tuple[float, float, float]) -> dict:
    rows = []
    for record, dense_norm, bm25_norm, colbert_norm in cache:
        combined = weights[0] * dense_norm + weights[1] * bm25_norm + weights[2] * colbert_norm
        order = np.argsort(-combined)[:TOP_K_FINAL]
        ranked_ids = [CHUNK_IDS[idx] for idx in order]

        gold_ids = record["gold_chunk_ids"]
        primary_ids = record["primary_chunk_ids"]
        supporting_ids = record["supporting_chunk_ids"]

        precision5 = precision_at_k(ranked_ids, gold_ids, 5)
        recall5 = recall_at_k(ranked_ids, gold_ids, 5)

        row = {
            "hit@3": hit_at_k(ranked_ids, gold_ids, 3),
            "recall@5": recall5,
            "mrr@10": mrr_at_k(ranked_ids, gold_ids, 10),
            "map@10": average_precision_at_k(ranked_ids, gold_ids, 10),
            "precision@5": precision5,
            "f1@5": f1_at_k(precision5, recall5),
            "ndcg@5": ndcg_at_k(ranked_ids, primary_ids, supporting_ids, 5),
        }
        complete_applicable = (
            record["multi_chunk_required"] and 0 < len(primary_ids) <= 5
        )
        row["complete@5"] = (
            complete_at_k(ranked_ids, primary_ids, 5)
            if complete_applicable else None
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    summary: dict = {"n_questions": len(df)}
    for metric in ("hit@3", "recall@5", "mrr@10", "map@10", "precision@5", "f1@5", "ndcg@5"):
        summary[metric] = round(float(df[metric].mean()), 4)
    mask = df["complete@5"].notna()
    summary["complete@5"] = round(float(df.loc[mask, "complete@5"].mean()), 4) if mask.any() else None
    summary["complete@5_n"] = int(mask.sum())
    return summary


# ============================================================
# 3) 3-way 가중치 그리드 스윕 (dense x bm25 x colbert = 4x4x4 = 64개)
# ============================================================
summaries: dict[str, dict] = {}
for dense_w in DENSE_OPTIONS:
    for bm25_w in BM25_OPTIONS:
        for colbert_w in COLBERT_OPTIONS:
            weights = (dense_w, bm25_w, colbert_w)
            combo_name = f"hybrid_colbert_d{dense_w}_b{bm25_w}_c{colbert_w}"
            summary = evaluate_weights(weights)
            summaries[combo_name] = summary
            log_result(combo_name, summary)

result_df = pd.DataFrame(summaries).T.sort_values("mrr@10", ascending=False)
print("\n=== 3-way 가중치 스윕 결과 (상위 15개, mrr@10 내림차순) ===")
print(result_df[["mrr@10", "recall@5", "ndcg@5", "hit@3", "map@10"]].head(15))

best_name = result_df.index[0]
print(f"\n>>> 최고 조합: {best_name} (mrr@10={summaries[best_name]['mrr@10']})")
print("성능:", json.dumps(summaries[best_name], ensure_ascii=False, indent=2))

# 참고용: colbert=0 (=원래 Hybrid) 중 제일 좋았던 조합과 직접 비교
no_colbert = result_df[result_df.index.str.endswith("_c0.0")]
if not no_colbert.empty:
    best_no_colbert = no_colbert.iloc[0]
    print(f"\n>>> (참고) ColBERT 없는 조합 중 최고: {no_colbert.index[0]} (mrr@10={best_no_colbert['mrr@10']})")
