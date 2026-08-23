from __future__ import annotations

"""요구사항 3 (파라미터 테스트) 지원 모듈.

kdic_pipeline_engine.py의 검색 함수들은 DENSE_WEIGHT/BM25_WEIGHT/CANDIDATE_DEPTH
같은 값을 모듈 전역변수로 읽는다 (요청마다 인자로 받는 구조가 아님). 그래서
"파라미터 조합 A vs B를 나란히 비교"하려면, 실행 직전에 전역값을 잠깐 바꿔치기하고
실행 후 반드시 원복해야 한다. 여러 요청이 동시에 이 작업을 하면 서로의 값을
덮어써서 결과가 뒤섞이므로, 락으로 전체를 직렬화한다.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SearchParams:
    label: str
    dense_weight: float = 0.7
    bm25_weight: float = 0.3
    candidate_depth: int = 20
    final_top_k: int = 5
    rrf_k: int = 10

    def __post_init__(self) -> None:
        if abs((self.dense_weight + self.bm25_weight) - 1.0) > 1e-9:
            raise ValueError("dense_weight + bm25_weight는 1이어야 합니다.")
        if self.candidate_depth < self.final_top_k:
            raise ValueError("candidate_depth는 final_top_k 이상이어야 합니다.")


_OVERRIDE_LOCK = threading.Lock()
_OVERRIDE_KEYS = (
    "DENSE_WEIGHT",
    "BM25_WEIGHT",
    "CANDIDATE_DEPTH",
    "FINAL_TOP_K",
    "QUERY_FUSION_RRF_K",
)


def run_search_with_params(
    pipeline_module: Any, question: str, params: SearchParams
) -> dict[str, Any]:
    """한 가지 파라미터 조합으로 검색 1회 실행. (리랭커는 적용하지 않는
    hybrid_minmax_search 기준 -- fuse_query_results 단계의 리랭킹은 별도.)"""
    with _OVERRIDE_LOCK:
        originals = {key: getattr(pipeline_module, key) for key in _OVERRIDE_KEYS}
        try:
            pipeline_module.DENSE_WEIGHT = params.dense_weight
            pipeline_module.BM25_WEIGHT = params.bm25_weight
            pipeline_module.CANDIDATE_DEPTH = params.candidate_depth
            pipeline_module.FINAL_TOP_K = params.final_top_k
            pipeline_module.QUERY_FUSION_RRF_K = params.rrf_k

            started = time.perf_counter()
            hits = pipeline_module.hybrid_minmax_search(
                question, top_k=params.final_top_k
            )
            latency_ms = (time.perf_counter() - started) * 1000
        finally:
            for key, value in originals.items():
                setattr(pipeline_module, key, value)

    return {
        "label": params.label,
        "params": params,
        "latency_ms": latency_ms,
        "hits": [
            {
                "chunk_id": row["chunk_id"],
                "rank": row["rank"],
                "minmax_score": row.get("minmax_score"),
                "dense_rank": row.get("dense_rank"),
                "bm25_rank": row.get("bm25_rank"),
            }
            for row in hits
        ],
    }


def compare_params(
    pipeline_module: Any, question: str, params_list: list[SearchParams]
) -> list[dict[str, Any]]:
    """요구사항 3의 1단계: 같은 질문을 파라미터 조합별로 나란히 비교 (DB 미기록)."""
    return [
        run_search_with_params(pipeline_module, question, params)
        for params in params_list
    ]


def activate_params(cursor: Any, params: SearchParams, *, created_by: str = "") -> str:
    """요구사항 3의 2단계: 비교 후 확정된 조합을 운영값으로 DB에 반영.

    search_params.is_active는 partial unique index로 한 행만 true를 허용하므로,
    이 INSERT가 성공하면 자동으로 이전 운영값은 비활성 상태가 된다 (먼저 UPDATE로
    기존 활성 행을 꺼둬야 함 -- unique index는 "동시에 두 개"만 막지, 자동으로
    기존 값을 꺼주지는 않는다).
    """
    cursor.execute("UPDATE search_params SET is_active = false WHERE is_active = true")
    cursor.execute(
        "INSERT INTO search_params "
        "(label, dense_weight, bm25_weight, candidate_depth, final_top_k, rrf_k, "
        " reranker_model, is_active, created_by) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, true, %s) "
        "RETURNING id",
        (
            params.label,
            params.dense_weight,
            params.bm25_weight,
            params.candidate_depth,
            params.final_top_k,
            params.rrf_k,
            "BAAI/bge-reranker-v2-m3",
            created_by,
        ),
    )
    return str(cursor.fetchone()["id"])
