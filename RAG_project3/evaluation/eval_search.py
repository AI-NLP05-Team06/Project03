# [Eval] Evaluation_DataSet_v3.xlsx 기준으로 '지표·칼럼 안내'에 정의된 지표를 계산합니다.
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from core.config import RESULT_ROOT

GOLD_PATH = Path("data/Evaluation_DataSet_v3.xlsx")
GOLD_SHEET = "평가데이터셋 V4_1"
EVAL_LOG_PATH = RESULT_ROOT / "search_eval_log.csv"


def _parse_id_list(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    value = str(value).strip()
    if not value:
        return []

    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise TypeError(f"gold_chunk_ids가 리스트가 아닙니다: {parsed!r}")

    # 이중 대괄호(예: [["X"]]) 오타를 한 겹 펴서 복구합니다.
    flattened: list[str] = []
    for item in parsed:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)
    return flattened


def load_gold_set(
    path: Path = GOLD_PATH,
    sheet_name: str = GOLD_SHEET,
) -> list[dict[str, Any]]:
    df = pd.read_excel(path, sheet_name=sheet_name)

    records: list[dict[str, Any]] = []
    skipped: list[str] = []
    for _, row in df.iterrows():
        if "evaluation_target" in row and row["evaluation_target"] != "Y":
            continue

        try:
            primary_ids = _parse_id_list(row["gold_primary_chunk_ids"])
            supporting_ids = _parse_id_list(row["gold_supporting_chunk_ids"])
            gold_ids = _parse_id_list(row["gold_chunk_ids"])
        except (json.JSONDecodeError, TypeError):
            skipped.append(str(row["evaluation_id"]))
            continue

        if not gold_ids:
            continue

        records.append({
            "evaluation_id": row["evaluation_id"],
            "question": row["question"],
            "business_function": row["gold_business_function"],
            "complexity": row["complexity"],
            "multi_chunk_required": row["multi_chunk_required"] == "Y",
            "primary_chunk_ids": primary_ids,
            "supporting_chunk_ids": supporting_ids,
            "gold_chunk_ids": gold_ids,
        })

    if not records:
        raise RuntimeError(f"검색평가대상=Y 문항이 없습니다: {path}")

    if skipped:
        print(
            f"[경고] gold_chunk_ids 형식이 깨져서 건너뛴 문항 {len(skipped)}건:",
            ", ".join(skipped),
        )

    return records


# ============================================================
# 지표 함수 ('지표·칼럼 안내' 시트 정의 그대로 구현)
# ============================================================

def hit_at_k(ranked_ids: list[str], gold_ids: list[str], k: int) -> float:
    gold_set = set(gold_ids)
    return 1.0 if any(cid in gold_set for cid in ranked_ids[:k]) else 0.0


def recall_at_k(ranked_ids: list[str], gold_ids: list[str], k: int) -> float:
    if not gold_ids:
        return 1.0
    top_k_ids = set(ranked_ids[:k])
    unique_gold = set(gold_ids)
    hits = sum(1 for gid in unique_gold if gid in top_k_ids)
    return hits / len(unique_gold)


def mrr_at_k(ranked_ids: list[str], gold_ids: list[str], k: int) -> float:
    gold_set = set(gold_ids)
    for rank, cid in enumerate(ranked_ids[:k], start=1):
        if cid in gold_set:
            return 1.0 / rank
    return 0.0


def average_precision_at_k(
    ranked_ids: list[str],
    gold_ids: list[str],
    k: int,
) -> float:
    gold_set = set(gold_ids)
    hits = 0
    precisions: list[float] = []

    for rank, cid in enumerate(ranked_ids[:k], start=1):
        if cid in gold_set:
            hits += 1
            precisions.append(hits / rank)

    return sum(precisions) / len(precisions) if precisions else 0.0


def complete_at_k(
    ranked_ids: list[str],
    primary_ids: list[str],
    k: int,
) -> float:
    top_k_ids = set(ranked_ids[:k])
    return 1.0 if all(pid in top_k_ids for pid in primary_ids) else 0.0


def ndcg_at_k(
    ranked_ids: list[str],
    primary_ids: list[str],
    supporting_ids: list[str],
    k: int,
) -> float:
    primary_set = set(primary_ids)
    supporting_set = set(supporting_ids)

    def relevance(chunk_id: str) -> float:
        if chunk_id in primary_set:
            return 2.0
        if chunk_id in supporting_set:
            return 1.0
        return 0.0

    dcg = sum(
        relevance(cid) / math.log2(rank + 1)
        for rank, cid in enumerate(ranked_ids[:k], start=1)
    )

    ideal_gains = sorted(
        [2.0] * len(primary_set) + [1.0] * len(supporting_set),
        reverse=True,
    )[:k]
    idcg = sum(
        gain / math.log2(rank + 1)
        for rank, gain in enumerate(ideal_gains, start=1)
    )

    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(ranked_ids: list[str], gold_ids: list[str], k: int) -> float:
    gold_set = set(gold_ids)
    hits = sum(1 for cid in ranked_ids[:k] if cid in gold_set)
    return hits / k


def f1_at_k(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ============================================================
# 평가 실행
# ============================================================

def evaluate_search(
    search_fn: Callable[[str], list[dict]],
    gold_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """search_fn(question) -> semantic_search_hcx와 같은 형태의 결과 리스트를 반환해야 합니다."""
    rows: list[dict[str, Any]] = []

    for record in gold_records:
        start = time.perf_counter()
        results = search_fn(record["question"])
        elapsed_sec = time.perf_counter() - start

        ranked_ids = [result["chunk"]["chunk_id"] for result in results]
        gold_ids = record["gold_chunk_ids"]
        primary_ids = record["primary_chunk_ids"]
        supporting_ids = record["supporting_chunk_ids"]

        precision5 = precision_at_k(ranked_ids, gold_ids, 5)
        recall5 = recall_at_k(ranked_ids, gold_ids, 5)

        row: dict[str, Any] = {
            "evaluation_id": record["evaluation_id"],
            "business_function": record["business_function"],
            "complexity": record["complexity"],
            "latency_sec": elapsed_sec,
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

    detail_df = pd.DataFrame(rows)

    summary: dict[str, Any] = {"n_questions": len(detail_df)}
    for metric in (
        "hit@3", "recall@5", "mrr@10", "map@10",
        "precision@5", "f1@5", "ndcg@5",
    ):
        summary[metric] = round(float(detail_df[metric].mean()), 4)

    complete_mask = detail_df["complete@5"].notna()
    summary["complete@5"] = (
        round(float(detail_df.loc[complete_mask, "complete@5"].mean()), 4)
        if complete_mask.any() else None
    )
    summary["complete@5_n"] = int(complete_mask.sum())

    summary["latency_p50_sec"] = round(
        float(detail_df["latency_sec"].median()), 4
    )
    summary["latency_p95_sec"] = round(
        float(detail_df["latency_sec"].quantile(0.95)), 4
    )

    return {"summary": summary, "detail": detail_df}


def log_result(combo_name: str, summary: dict[str, Any]) -> None:
    row = {"combo": combo_name, **summary}
    row_df = pd.DataFrame([row])

    if EVAL_LOG_PATH.exists():
        existing = pd.read_csv(EVAL_LOG_PATH)
        existing = existing[existing["combo"] != combo_name]
        row_df = pd.concat([existing, row_df], ignore_index=True)

    row_df.to_csv(EVAL_LOG_PATH, index=False, encoding="utf-8-sig")
    print("평가 로그 저장:", EVAL_LOG_PATH)


print("평가 하니스 준비 완료")
