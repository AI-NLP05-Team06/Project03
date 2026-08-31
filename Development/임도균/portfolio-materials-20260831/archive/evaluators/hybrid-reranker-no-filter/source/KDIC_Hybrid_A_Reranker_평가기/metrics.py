"""검색 결과와 Gold 청크를 비교하는 순위 평가 지표."""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def hit_at_k(ranked_ids: Sequence[str], gold_ids: Iterable[str], k: int) -> float:
    gold = set(_unique(gold_ids))
    return float(any(chunk_id in gold for chunk_id in _unique(ranked_ids)[:k]))


def recall_at_k(ranked_ids: Sequence[str], gold_ids: Iterable[str], k: int) -> float:
    gold = set(_unique(gold_ids))
    if not gold:
        return 0.0
    found = gold.intersection(_unique(ranked_ids)[:k])
    return len(found) / len(gold)


def precision_at_k(ranked_ids: Sequence[str], gold_ids: Iterable[str], k: int) -> float:
    if k < 1:
        raise ValueError("k는 1 이상이어야 합니다.")
    gold = set(_unique(gold_ids))
    found = gold.intersection(_unique(ranked_ids)[:k])
    return len(found) / k


def reciprocal_rank_at_k(
    ranked_ids: Sequence[str],
    gold_ids: Iterable[str],
    k: int,
) -> float:
    gold = set(_unique(gold_ids))
    for rank, chunk_id in enumerate(_unique(ranked_ids)[:k], start=1):
        if chunk_id in gold:
            return 1.0 / rank
    return 0.0


def average_precision_at_k(
    ranked_ids: Sequence[str],
    gold_ids: Iterable[str],
    k: int,
) -> float:
    """AP@K. 분모는 min(전체 Gold 수, K)로 정의합니다."""
    gold = set(_unique(gold_ids))
    if not gold:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for rank, chunk_id in enumerate(_unique(ranked_ids)[:k], start=1):
        if chunk_id in gold:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / min(len(gold), k)


def complete_at_k(
    ranked_ids: Sequence[str],
    primary_gold_ids: Iterable[str],
    k: int,
    *,
    multi_chunk_required: bool,
) -> float | None:
    """적용 대상이 아니면 None, 모든 주요 Gold가 Top-K에 있으면 1."""
    primary = set(_unique(primary_gold_ids))
    if not multi_chunk_required or len(primary) < 2 or len(primary) > k:
        return None
    return float(primary.issubset(set(_unique(ranked_ids)[:k])))


def ndcg_at_k(
    ranked_ids: Sequence[str],
    primary_gold_ids: Iterable[str],
    supporting_gold_ids: Iterable[str],
    k: int,
) -> float:
    """Primary 관련도 2, Supporting 관련도 1로 계산합니다."""
    primary = set(_unique(primary_gold_ids))
    supporting = set(_unique(supporting_gold_ids)) - primary

    def grade(chunk_id: str) -> int:
        if chunk_id in primary:
            return 2
        if chunk_id in supporting:
            return 1
        return 0

    ranked = _unique(ranked_ids)[:k]
    dcg = sum(
        ((2**grade(chunk_id)) - 1) / math.log2(rank + 1)
        for rank, chunk_id in enumerate(ranked, start=1)
        if grade(chunk_id) > 0
    )
    ideal_grades = sorted(
        [2] * len(primary) + [1] * len(supporting),
        reverse=True,
    )[:k]
    idcg = sum(
        ((2**relevance) - 1) / math.log2(rank + 1)
        for rank, relevance in enumerate(ideal_grades, start=1)
    )
    return dcg / idcg if idcg else 0.0


def f1_score(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def evaluate_ranking(
    ranked_ids: Sequence[str],
    *,
    gold_ids: Iterable[str],
    primary_gold_ids: Iterable[str],
    supporting_gold_ids: Iterable[str],
    multi_chunk_required: bool,
) -> dict[str, float | None]:
    recall5 = recall_at_k(ranked_ids, gold_ids, 5)
    precision5 = precision_at_k(ranked_ids, gold_ids, 5)
    return {
        "hit_at_3": hit_at_k(ranked_ids, gold_ids, 3),
        "recall_at_5": recall5,
        "mrr_at_10": reciprocal_rank_at_k(ranked_ids, gold_ids, 10),
        "ap_at_10": average_precision_at_k(ranked_ids, gold_ids, 10),
        "complete_at_5": complete_at_k(
            ranked_ids,
            primary_gold_ids,
            5,
            multi_chunk_required=multi_chunk_required,
        ),
        "ndcg_at_5": ndcg_at_k(
            ranked_ids,
            primary_gold_ids,
            supporting_gold_ids,
            5,
        ),
        "precision_at_5": precision5,
        "f1_at_5": f1_score(precision5, recall5),
    }
