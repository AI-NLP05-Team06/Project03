from __future__ import annotations

"""요구사항 5 (평가 연동) 지원 모듈.

eval_queries(정답 chunk_id 포함) 세트를 주어진 search_params 조합으로 전부
검색해보고, 결과를 eval_run_results에 기록한다. 파라미터 변경 전/후로 이 함수를
두 번 돌려서 eval_runs 두 개를 비교하면 "품질이 좋아졌는지"를 정량적으로 볼 수 있다.
"""

import time
from typing import Any

from psycopg2.extras import Json

from kdic_param_test import SearchParams, run_search_with_params


def _params_from_row(row: dict[str, Any]) -> SearchParams:
    return SearchParams(
        label=row["label"],
        dense_weight=float(row["dense_weight"]),
        bm25_weight=float(row["bm25_weight"]),
        candidate_depth=int(row["candidate_depth"]),
        final_top_k=int(row["final_top_k"]),
        rrf_k=int(row["rrf_k"]) if row.get("rrf_k") is not None else 10,
    )


def run_eval(
    cursor: Any,
    pipeline_module: Any,
    *,
    search_params_id: str,
    eval_query_ids: list[str] | None = None,
    triggered_by: str = "",
) -> str:
    """요구사항 5: eval_query 세트를 한 파라미터 조합으로 전부 실행하고
    eval_run_results에 기록. 반환값은 eval_runs.id (조회용)."""

    cursor.execute("SELECT * FROM search_params WHERE id = %s", (search_params_id,))
    params_row = cursor.fetchone()
    if params_row is None:
        raise ValueError(f"search_params를 찾을 수 없습니다: {search_params_id}")
    params = _params_from_row(params_row)

    if eval_query_ids:
        cursor.execute(
            "SELECT id, question, expected_chunk_ids FROM eval_queries "
            "WHERE id = ANY(%s) AND is_active = true",
            (eval_query_ids,),
        )
    else:
        cursor.execute(
            "SELECT id, question, expected_chunk_ids FROM eval_queries WHERE is_active = true"
        )
    queries = cursor.fetchall()
    if not queries:
        raise ValueError("실행할 평가 질의가 없습니다.")

    cursor.execute(
        "INSERT INTO eval_runs (search_params_id, status, triggered_by, started_at) "
        "VALUES (%s, 'running', %s, now()) RETURNING id",
        (search_params_id, triggered_by),
    )
    eval_run_id = str(cursor.fetchone()["id"])

    try:
        for query_row in queries:
            expected = set(query_row["expected_chunk_ids"] or [])
            started = time.perf_counter()
            hits = run_search_with_params(
                pipeline_module, query_row["question"], params
            )["hits"]
            latency_ms = (time.perf_counter() - started) * 1000

            retrieved_ids = [hit["chunk_id"] for hit in hits]
            rank_of_expected = next(
                (
                    index
                    for index, chunk_id in enumerate(retrieved_ids, start=1)
                    if chunk_id in expected
                ),
                None,
            )
            hit_found = rank_of_expected is not None

            cursor.execute(
                "INSERT INTO eval_run_results "
                "(eval_run_id, eval_query_id, retrieved_chunk_ids, rank_of_expected, "
                " hit, latency_ms) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (
                    eval_run_id,
                    query_row["id"],
                    Json(retrieved_ids),
                    rank_of_expected,
                    hit_found,
                    latency_ms,
                ),
            )

        cursor.execute(
            "UPDATE eval_runs SET status = 'done', finished_at = now() WHERE id = %s",
            (eval_run_id,),
        )
    except Exception:
        cursor.execute(
            "UPDATE eval_runs SET status = 'failed', finished_at = now() WHERE id = %s",
            (eval_run_id,),
        )
        raise

    return eval_run_id


def summarize_eval_run(cursor: Any, eval_run_id: str) -> dict[str, Any]:
    """Hit@K, MRR, 평균 지연시간 집계."""
    cursor.execute(
        "SELECT rank_of_expected, hit, latency_ms FROM eval_run_results "
        "WHERE eval_run_id = %s",
        (eval_run_id,),
    )
    rows = cursor.fetchall()
    if not rows:
        return {"eval_run_id": eval_run_id, "query_count": 0}

    hits = sum(1 for row in rows if row["hit"])
    reciprocal_ranks = [
        1.0 / row["rank_of_expected"] for row in rows if row["rank_of_expected"]
    ]
    latencies = [float(row["latency_ms"]) for row in rows if row["latency_ms"] is not None]

    return {
        "eval_run_id": eval_run_id,
        "query_count": len(rows),
        "hit_at_k": hits / len(rows),
        "mrr": sum(reciprocal_ranks) / len(rows) if rows else 0.0,
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else None,
    }
