from __future__ import annotations

import ast
import base64
import copy
import hashlib
import hmac
import inspect
import io
import json
import math
import os
import secrets
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd
from fastapi import Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field, SecretStr


COOKIE_NAME = "kdic_admin_session"
SESSION_TTL_SECONDS = 8 * 60 * 60
MAX_DATASET_BYTES = 25 * 1024 * 1024
ALLOWED_CONFIG = {
    "dense_weight", "bm25_weight", "candidate_depth", "final_top_k",
    "query_fusion_rrf_k", "parent_child", "parent_context_max_chars",
}
DEFAULT_METRIC_KS = {
    "hit": 3,
    "recall": 5,
    "mrr": 10,
    "map": 10,
    "complete": 5,
    "ndcg": 5,
    "precision": 5,
    "f1": 5,
}
METRIC_LABELS = {
    "hit": "Hit",
    "recall": "Recall",
    "mrr": "MRR",
    "map": "MAP",
    "complete": "Complete",
    "ndcg": "nDCG",
    "precision": "Precision",
    "f1": "F1",
}


class AdminSearchPayload(BaseModel):
    index: str = Field(min_length=1, max_length=255)
    query: str = Field(default="", max_length=2_000)
    size: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0, le=10_000)


class DatasetUploadPayload(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    content_base64: str = Field(min_length=1)
    sheet_name: str | None = Field(default=None, max_length=255)


class EvaluationRunPayload(BaseModel):
    dataset_id: str = Field(min_length=8, max_length=100)
    candidate: dict[str, Any] = Field(default_factory=dict)
    max_questions: int | None = Field(default=None, ge=1, le=5_000)
    evaluation_depth: int = Field(default=20, ge=1, le=200)
    metric_ks: dict[str, int] = Field(default_factory=lambda: dict(DEFAULT_METRIC_KS))
    curve_ks: list[int] = Field(default_factory=lambda: [1, 3, 5, 10, 20])


class ConfigDraftPayload(BaseModel):
    values: dict[str, Any]


class ChunkAddPayload(BaseModel):
    chunk: dict[str, Any]


class ApplyPayload(BaseModel):
    confirmation: str


class ApiKeyPayload(BaseModel):
    api_key: SecretStr


class ApiKeyRotatePayload(ApiKeyPayload):
    confirmation: str


class AdminLoginPayload(BaseModel):
    admin_token: SecretStr


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _safe(value: Any) -> Any:
    return value if value is None or isinstance(value, (str, int, float, bool)) else str(value)


DEFAULT_PIPELINE_GRAPH_SPEC: list[dict[str, Any]] = [
    {
        "id": "question",
        "label": "사용자 질문",
        "stage": "INPUT",
        "description": "자유 질문, FAQ 선택, 후속 질문을 동일한 진입점으로 받습니다.",
        "virtual": True,
    },
    {
        "id": "controller",
        "label": "대화·업무 제어",
        "stage": "ROUTING",
        "description": "현재 대화 상태를 확인하고 단일 업무와 교차 업무 처리 경로를 제어합니다.",
        "symbols": ["execute_dc_variant_v1"],
    },
    {
        "id": "context",
        "label": "문맥·모호성 판단",
        "stage": "ANALYSIS",
        "description": "이전 대화 문맥, 업무 생략, 추가 확인 필요 여부를 판단합니다.",
        "symbols": ["resolve_context_v2", "resolve_context_v21", "resolve_context_v22"],
    },
    {
        "id": "analysis",
        "label": "질의 분석·검색 계획",
        "stage": "ANALYSIS",
        "description": "업무와 질문 유형을 분석하고 검색용 원질문·재작성 질문을 구성합니다.",
        "symbols": ["analyze_v15_improved", "analyze_v31_cross_only", "analyze_v15_chat_query"],
    },
    {
        "id": "route",
        "label": "응답 경로 결정",
        "stage": "ROUTING",
        "description": "DIRECT, CLARIFY, OOS, RETRIEVE 중 처리 경로를 결정합니다.",
        "virtual": True,
    },
    {
        "id": "direct",
        "label": "직접·추가질문 응답",
        "stage": "ANSWER",
        "description": "검색이 필요 없는 안내, 범위 밖 안내, 버튼형 추가 질문을 반환합니다.",
        "symbols": ["_route_response"],
    },
    {
        "id": "hybrid",
        "label": "Structured Hybrid 검색",
        "stage": "RETRIEVAL",
        "description": "Structured Dense와 BM25-Nori 후보를 결합하고 다중 질의 결과를 융합합니다.",
        "symbols": ["fuse_query_results", "hybrid_minmax_search", "weighted_minmax"],
    },
    {
        "id": "reranker",
        "label": "BGE Reranker",
        "stage": "RETRIEVAL",
        "description": "Hybrid 후보에 질문-청크 관련성 점수를 다시 매겨 최종 순위를 정합니다.",
        "symbols": ["rerank_candidates"],
    },
    {
        "id": "parent_child",
        "label": "Parent-Child 확장",
        "stage": "EVIDENCE",
        "description": "상위 Child 청크를 유지하면서 Parent와 인접 문맥을 답변 근거로 확장합니다.",
        "symbols": ["expand_parent_context"],
        "enabled_flag": "PARENT_CHILD_ENABLED",
    },
    {
        "id": "evidence",
        "label": "Evidence Pack 구성",
        "stage": "EVIDENCE",
        "description": "검색 근거, 출처 URL, 업무별 Fact를 답변 모델이 사용할 구조로 묶습니다.",
        "symbols": ["build_compact_parent_evidence_pack_v1", "build_parent_basic_evidence_pack"],
    },
    {
        "id": "answer_c",
        "label": "단일·동일 업무 C안",
        "stage": "ANSWER",
        "description": "한 업무 안에서 근거 기반 기본 답변과 신청·처리 링크를 생성합니다.",
        "symbols": ["execute_bcd_variant_v3", "generate_answer_c_v3", "generate_basic_answer_b_v2"],
    },
    {
        "id": "answer_dc",
        "label": "교차 업무 D-C 2Call",
        "stage": "ANSWER",
        "description": "서로 다른 업무의 근거를 분리해 구성한 뒤 교차 업무 답변을 생성합니다.",
        "symbols": ["generate_dc_twocall_v1", "execute_dc_variant_v1"],
    },
    {
        "id": "validation",
        "label": "출처·주장 안전성 검증",
        "stage": "VALIDATION",
        "description": "숫자, 적용 대상, 근거 참조, 허용된 행동 링크를 검증하고 제한 답변을 적용합니다.",
        "symbols": ["audit_dc_final_references_v1", "audit_numeric_support_dc_v1", "validate_basic_answer"],
    },
    {
        "id": "response",
        "label": "최종 답변·출처",
        "stage": "OUTPUT",
        "description": "검증된 답변, 공식 출처, 신청 링크와 후속 질문을 사용자에게 표시합니다.",
        "virtual": True,
    },
]

DEFAULT_PIPELINE_GRAPH_EDGES: list[tuple[str, str, str]] = [
    ("question", "controller", "질문 전달"),
    ("controller", "context", "대화 상태"),
    ("context", "analysis", "확정·보완된 질문"),
    ("analysis", "route", "분석 결과"),
    ("route", "direct", "DIRECT · CLARIFY · OOS"),
    ("direct", "response", "안내 응답"),
    ("route", "hybrid", "RETRIEVE"),
    ("hybrid", "reranker", "후보 청크"),
    ("reranker", "parent_child", "상위 Child"),
    ("parent_child", "evidence", "확장 문맥"),
    ("evidence", "answer_c", "단일·동일 업무"),
    ("evidence", "answer_dc", "교차 업무"),
    ("answer_c", "validation", "생성 답변"),
    ("answer_dc", "validation", "생성 답변"),
    ("validation", "response", "검증 통과·제한 응답"),
]


def _pipeline_source_descriptor(
    runtime: Mapping[str, Any], spec: Mapping[str, Any], source_root: Path
) -> dict[str, Any] | None:
    for symbol in spec.get("symbols") or []:
        target = runtime.get(str(symbol))
        if not callable(target):
            continue
        try:
            source_file = Path(inspect.getsourcefile(target) or "").resolve()
            if not source_file.is_file() or not source_file.is_relative_to(source_root):
                continue
            source_lines, line_start = inspect.getsourcelines(target)
            source = "".join(source_lines)
            return {
                "symbol": getattr(target, "__name__", str(symbol)),
                "module": getattr(target, "__module__", ""),
                "file": source_file.relative_to(source_root).as_posix(),
                "line_start": int(line_start),
                "line_end": int(line_start + len(source_lines) - 1),
                "signature": str(inspect.signature(target)),
                "source_hash": hashlib.sha256(source.encode("utf-8")).hexdigest()[:12],
                "source": source,
            }
        except (OSError, TypeError, ValueError):
            continue
    return None


def _build_pipeline_graph(runtime: Mapping[str, Any], source_root: Path) -> dict[str, Any]:
    custom = runtime.get("KDIC_PIPELINE_GRAPH_SPEC")
    specs = custom.get("nodes") if isinstance(custom, Mapping) else None
    edges = custom.get("edges") if isinstance(custom, Mapping) else None
    specs = list(specs) if isinstance(specs, Sequence) and not isinstance(specs, (str, bytes)) else DEFAULT_PIPELINE_GRAPH_SPEC
    edges = list(edges) if isinstance(edges, Sequence) and not isinstance(edges, (str, bytes)) else DEFAULT_PIPELINE_GRAPH_EDGES
    nodes: list[dict[str, Any]] = []
    source_index: dict[str, dict[str, Any]] = {}
    for raw in specs:
        spec = dict(raw)
        node_id = _clean(spec.get("id"))
        if not node_id:
            continue
        descriptor = None if spec.get("virtual") else _pipeline_source_descriptor(runtime, spec, source_root)
        enabled_flag = _clean(spec.get("enabled_flag"))
        enabled = bool(runtime.get(enabled_flag)) if enabled_flag else True
        node = {
            "id": node_id,
            "label": _clean(spec.get("label")) or node_id,
            "stage": _clean(spec.get("stage")) or "PIPELINE",
            "description": _clean(spec.get("description")),
            "enabled": enabled,
            "code_available": descriptor is not None,
            "source": ({key: descriptor[key] for key in descriptor if key != "source"} if descriptor else None),
        }
        nodes.append(node)
        if descriptor:
            source_index[node_id] = descriptor
    valid_ids = {node["id"] for node in nodes}
    public_edges = []
    for raw in edges:
        if isinstance(raw, Mapping):
            source, target, label = _clean(raw.get("source")), _clean(raw.get("target")), _clean(raw.get("label"))
        else:
            values = list(raw)
            source, target = _clean(values[0] if values else ""), _clean(values[1] if len(values) > 1 else "")
            label = _clean(values[2] if len(values) > 2 else "")
        if source in valid_ids and target in valid_ids:
            public_edges.append({"source": source, "target": target, "label": label})
    fingerprint_text = json.dumps(
        [{"id": node["id"], "hash": (node.get("source") or {}).get("source_hash"), "enabled": node["enabled"]} for node in nodes],
        ensure_ascii=False,
        sort_keys=True,
    )
    return {
        "pipeline_name": _clean(runtime.get("PIPELINE_VERSION")) or "KDIC Final Runtime",
        "generated_at": time.time(),
        "fingerprint": hashlib.sha256(fingerprint_text.encode("utf-8")).hexdigest()[:16],
        "mode": "RUNTIME_INTROSPECTION" if custom is None else "RUNTIME_REGISTRY",
        "read_only": True,
        "nodes": nodes,
        "edges": public_edges,
        "_source_index": source_index,
    }


def _api_key_fingerprint(value: str) -> str:
    cleaned = str(value or "").strip()
    if not cleaned:
        return ""
    import hashlib
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:12]


def _validate_api_key_text(value: str) -> str:
    key = str(value or "").strip()
    if not key:
        raise ValueError("API 키가 비어 있습니다.")
    if key.lower().startswith("bearer "):
        raise ValueError("Bearer 접두사를 제외하고 키 값만 입력해 주세요.")
    if any(character.isspace() for character in key):
        raise ValueError("API 키에 공백 또는 줄바꿈이 포함되어 있습니다.")
    if len(key) < 20 or len(key) > 500:
        raise ValueError("API 키 길이가 올바르지 않습니다.")
    return key


def _persist_hcx_key_to_env_file(key: str) -> str:
    """Persist the rotated key for the next systemd restart without exposing it."""
    env_path = Path(os.getenv("KDIC_ENV_FILE", "/opt/kdic/kdic.env")).resolve()
    if not env_path.is_file():
        raise FileNotFoundError(f"AWS 환경파일을 찾지 못했습니다: {env_path}")
    if not os.access(env_path, os.W_OK):
        raise PermissionError(f"AWS 환경파일 쓰기 권한이 없습니다: {env_path}")
    rows = env_path.read_text(encoding="utf-8").splitlines()
    replacement = f"HCX_API_KEY={key}"
    replaced = False
    output: list[str] = []
    for row in rows:
        if row.startswith("HCX_API_KEY="):
            output.append(replacement)
            replaced = True
        else:
            output.append(row)
    if not replaced:
        output.append(replacement)
    temp_path = env_path.with_suffix(env_path.suffix + ".tmp")
    temp_path.write_text("\n".join(output) + "\n", encoding="utf-8")
    os.chmod(temp_path, 0o600)
    os.replace(temp_path, env_path)
    return str(env_path)


def _runtime_config(runtime: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "version": "2026-08-24-final-admin-eval-ab-v2",
        "search": {
            "dense_model": _safe(runtime.get("HCX_EMBEDDING_MODEL")),
            "dense_backend": _safe(runtime.get("DENSE_BACKEND")),
            "dense_knn_num_candidates": _safe(runtime.get("DENSE_KNN_NUM_CANDIDATES")),
            "sparse": "Elasticsearch BM25 + Nori-none",
            "fusion": "Weighted Min-Max + multi-query RRF",
            "dense_weight": _safe(runtime.get("DENSE_WEIGHT")),
            "bm25_weight": _safe(runtime.get("BM25_WEIGHT")),
            "query_fusion_rrf_k": _safe(runtime.get("QUERY_FUSION_RRF_K")),
            "candidate_depth": _safe(runtime.get("CANDIDATE_DEPTH")),
            "final_top_k": _safe(runtime.get("FINAL_TOP_K")),
            "reranker_model": _safe(runtime.get("RERANKER_MODEL_NAME")),
            "reranker_candidate_depth": _safe(runtime.get("RERANKER_CANDIDATE_DEPTH")),
            "parent_child": bool(runtime.get("PARENT_CHILD_ENABLED")),
            "parent_context_max_chars": _safe(runtime.get("PARENT_CONTEXT_MAX_CHARS")),
            "answer_pack_max_chars": 14_000,
            "cross_business_top_k": 6,
        },
        "query_analysis": {
            "model": _safe(runtime.get("HCX_DECOMPOSITION_MODEL")),
            "max_subqueries": _safe(runtime.get("V15_MAX_SUBQUERIES")),
            "min_confidence": _safe(runtime.get("V15_MIN_CONFIDENCE")),
            "routing_policy": "DIRECT / OOS / CLARIFY / RETRIEVE",
        },
        "answer_system": {
            "model": _safe(runtime.get("HCX_CHAT_MODEL")),
            "single_or_same_business": "C안", "cross_business": "D-C 2Call",
            "fact_index": "27개 검증 레코드 조건부 매칭", "action_links": "승인 Registry만 표시",
            "user_friendly_basis": "답변 연결 근거만 프로그램 후처리",
        },
        "secrets": {"hcx_api_key_exposed_to_browser": False, "admin_session_cookie": "HttpOnly + Secure + SameSite=Lax"},
    }


def _parse_list(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        raw = list(value)
    else:
        text = str(value).strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except Exception:
            try:
                parsed = ast.literal_eval(text)
            except Exception:
                parsed = [part.strip() for part in text.replace(";", ",").split(",")]
        raw = parsed if isinstance(parsed, (list, tuple, set)) else [parsed]
    return list(dict.fromkeys(_clean(item) for item in raw if _clean(item)))


def _column(columns: Sequence[str], *aliases: str) -> str | None:
    normalized = {_clean(col).lower().replace(" ", "").replace("_", ""): col for col in columns}
    for alias in aliases:
        key = alias.lower().replace(" ", "").replace("_", "")
        if key in normalized:
            return normalized[key]
    return None


def _validate_config(values: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(values) - ALLOWED_CONFIG)
    if unknown:
        raise ValueError(f"지원하지 않는 파라미터입니다: {unknown}")
    merged = {**current, **values}
    dense, bm25 = float(merged["dense_weight"]), float(merged["bm25_weight"])
    if dense < 0 or bm25 < 0 or not math.isclose(dense + bm25, 1.0, abs_tol=1e-6):
        raise ValueError("Dense/BM25 가중치는 0 이상이며 합이 1이어야 합니다.")
    depth, top_k = int(merged["candidate_depth"]), int(merged["final_top_k"])
    if not 5 <= depth <= 200:
        raise ValueError("후보 깊이는 5~200이어야 합니다.")
    if not 1 <= top_k <= min(20, depth):
        raise ValueError("최종 Top-K는 1~20이며 후보 깊이 이하여야 합니다.")
    if not 1 <= int(merged["query_fusion_rrf_k"]) <= 200:
        raise ValueError("RRF K는 1~200이어야 합니다.")
    if not 500 <= int(merged["parent_context_max_chars"]) <= 30_000:
        raise ValueError("Parent 문맥 한도는 500~30,000자여야 합니다.")
    return {"dense_weight": dense, "bm25_weight": bm25, "candidate_depth": depth, "final_top_k": top_k,
            "query_fusion_rrf_k": int(merged["query_fusion_rrf_k"]), "parent_child": bool(merged["parent_child"]),
            "parent_context_max_chars": int(merged["parent_context_max_chars"])}


def _average_precision(retrieved: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    hits, total = 0, 0.0
    for rank, cid in enumerate(retrieved[:k], 1):
        if cid in gold:
            hits += 1
            total += hits / rank
    return total / min(len(gold), k)


def _validate_evaluation_policy(
    evaluation_depth: int,
    metric_ks: Mapping[str, Any],
    curve_ks: Sequence[Any],
    *configs: Mapping[str, Any],
) -> dict[str, Any]:
    unknown = sorted(set(metric_ks) - set(DEFAULT_METRIC_KS))
    if unknown:
        raise ValueError(f"지원하지 않는 평가지표입니다: {unknown}")
    normalized = {name: int(metric_ks.get(name, default)) for name, default in DEFAULT_METRIC_KS.items()}
    if any(value < 1 or value > 200 for value in normalized.values()):
        raise ValueError("평가지표 K는 1~200이어야 합니다.")
    curve = sorted(set(int(value) for value in curve_ks))
    if not curve or any(value < 1 or value > 200 for value in curve):
        raise ValueError("구간 차트 K는 1~200의 값을 하나 이상 지정해야 합니다.")
    depth = int(evaluation_depth)
    required_depth = max([*normalized.values(), *curve])
    if depth < required_depth:
        raise ValueError(f"평가 검색 깊이는 모든 지표 K 이상이어야 합니다. 최소 {required_depth}이 필요합니다.")
    candidate_limit = min(int(config["candidate_depth"]) for config in configs) if configs else 200
    if depth > candidate_limit:
        raise ValueError(f"평가 검색 깊이 {depth}는 A/B 후보 깊이의 최솟값 {candidate_limit} 이하여야 합니다.")
    return {
        "evaluation_depth": depth,
        "metric_ks": normalized,
        "curve_ks": curve,
        "required_depth": required_depth,
    }


def _metric_value(name: str, retrieved: list[str], gold: set[str], k: int, applicable: bool) -> float | None:
    selected = retrieved[:k]
    relevance = [1 if chunk_id in gold else 0 for chunk_id in selected]
    hits = len(set(selected) & gold)
    recall = hits / len(gold) if gold else 0.0
    precision = sum(relevance) / k
    if name == "hit":
        return float(bool(hits))
    if name == "recall":
        return recall
    if name == "mrr":
        first = next((rank for rank, chunk_id in enumerate(selected, 1) if chunk_id in gold), None)
        return 1 / first if first else 0.0
    if name == "map":
        return _average_precision(retrieved, gold, k)
    if name == "complete":
        return float(gold.issubset(set(selected))) if applicable and gold else None
    if name == "ndcg":
        dcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(relevance, 1))
        ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(gold), k) + 1))
        return dcg / ideal if ideal else 0.0
    if name == "precision":
        return precision
    if name == "f1":
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    raise ValueError(f"지원하지 않는 평가지표입니다: {name}")


def _metrics(
    retrieved: list[str],
    gold_ids: list[str],
    multi_required: bool,
    metric_ks: Mapping[str, int] | None = None,
    curve_ks: Sequence[int] | None = None,
) -> dict[str, Any]:
    gold = set(gold_ids)
    applicable = bool(multi_required or len(gold) > 1)
    ks = {**DEFAULT_METRIC_KS, **dict(metric_ks or {})}
    out = {f"{name}_at_{k}": _metric_value(name, retrieved, gold, int(k), applicable) for name, k in ks.items()}
    out["curve"] = {
        str(k): {
            "hit": _metric_value("hit", retrieved, gold, int(k), applicable),
            "recall": _metric_value("recall", retrieved, gold, int(k), applicable),
            "precision": _metric_value("precision", retrieved, gold, int(k), applicable),
        }
        for k in (curve_ks or [1, 3, 5, 10, 20])
    }
    return out


def install_admin_routes(service_module: Any, html_path: str | Path, runtime_globals: MutableMapping[str, Any]) -> dict[str, Any]:
    """Cookie-authenticated admin UI with isolated A/B eval and staged apply/rollback."""
    app, page = service_module.app, Path(html_path).resolve()
    sessions: dict[str, float] = {}
    auth_lock, mutation_lock, eval_lock = threading.RLock(), threading.RLock(), threading.RLock()
    bootstrap_state = {"used": False}
    datasets: dict[str, dict[str, Any]] = {}
    eval_jobs: dict[str, dict[str, Any]] = {}
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kdic-admin-eval")
    drafts: dict[str, Any] = {"config": {}, "add": {}, "remove": set()}
    history: list[dict[str, Any]] = []
    initial_key = str(runtime_globals.get("HCX_API_KEY") or os.getenv("HCX_API_KEY") or "").strip()
    api_key_state: dict[str, Any] = {
        "configured": bool(initial_key),
        "fingerprint": _api_key_fingerprint(initial_key),
        "last_rotated_at": None,
        "source": "AWS_ENV_FILE_OR_STARTUP",
        "scope": "AWS_RUNTIME_AND_ENV_FILE",
    }

    def active_config() -> dict[str, Any]:
        return {"dense_weight": float(runtime_globals.get("DENSE_WEIGHT", .7)), "bm25_weight": float(runtime_globals.get("BM25_WEIGHT", .3)),
                "candidate_depth": int(runtime_globals.get("CANDIDATE_DEPTH", 20)), "final_top_k": int(runtime_globals.get("FINAL_TOP_K", 5)),
                "query_fusion_rrf_k": int(runtime_globals.get("QUERY_FUSION_RRF_K", 10)),
                "parent_child": bool(runtime_globals.get("PARENT_CHILD_ENABLED", True)),
                "parent_context_max_chars": int(runtime_globals.get("PARENT_CONTEXT_MAX_CHARS", 8192))}

    def require_admin_session(request: Request) -> None:
        supplied = _clean(request.cookies.get(COOKIE_NAME))
        with auth_lock:
            now = time.time()
            for token, expires in list(sessions.items()):
                if expires <= now:
                    sessions.pop(token, None)
            expires_at = sessions.get(supplied)
        if not supplied or not expires_at or expires_at <= time.time():
            raise HTTPException(status_code=401, detail="관리자 세션이 없거나 만료되었습니다.")

    def dataset_summary(row: Mapping[str, Any]) -> dict[str, Any]:
        return {key: row[key] for key in ("dataset_id", "filename", "sheet_name", "row_count", "usable_count", "warnings", "created_at")}

    def draft_public() -> dict[str, Any]:
        return {"active_config": active_config(), "candidate_config": _validate_config(drafts["config"], active_config()),
                "add": list(drafts["add"].values()), "remove": sorted(drafts["remove"]),
                "has_changes": bool(drafts["config"] or drafts["add"] or drafts["remove"]),
                "history": [{k: v for k, v in row.items() if k != "snapshot"} for row in history[-10:]][::-1]}

    def normalize_dataset(data: bytes, filename: str, requested_sheet: str | None) -> dict[str, Any]:
        suffix = Path(filename).suffix.lower()
        if suffix == ".csv":
            for encoding in ("utf-8-sig", "cp949", "utf-8"):
                try:
                    df, sheet = pd.read_csv(io.BytesIO(data), encoding=encoding), "CSV"
                    break
                except Exception:
                    pass
            else:
                raise ValueError("CSV 인코딩을 확인해 주세요.")
        elif suffix == ".xlsx":
            book = pd.ExcelFile(io.BytesIO(data))
            sheet = requested_sheet if requested_sheet in book.sheet_names else book.sheet_names[0]
            df = pd.read_excel(book, sheet_name=sheet)
        else:
            raise ValueError("XLSX 또는 CSV 파일만 지원합니다.")
        df.columns = [_clean(col) for col in df.columns]
        qid_col = _column(df.columns, "question_id", "질문ID", "Q번호", "질의ID")
        question_col = _column(df.columns, "question", "예상질문", "질문", "query")
        gold_col = _column(df.columns, "gold_chunk_ids", "정답청크", "골드청크")
        multi_col = _column(df.columns, "multi_chunk_required", "다중청크필요")
        domain_col = _column(df.columns, "gold_business_function", "business_function", "도메인(원본)", "업무라벨", "도메인")
        if not question_col or not gold_col:
            raise ValueError("필수 칼럼(질문/예상질문, gold_chunk_ids)이 없습니다.")
        rows, warnings = [], []
        corpus_ids = set(map(str, runtime_globals.get("CHUNKS_BY_ID", {})))
        for index, row in df.iterrows():
            question, gold = _clean(row.get(question_col)), _parse_list(row.get(gold_col))
            if not question:
                continue
            missing = [cid for cid in gold if cid not in corpus_ids]
            if not gold:
                warnings.append(f"{index + 2}행: Gold가 비어 있어 제외")
            elif missing:
                warnings.append(f"{index + 2}행: corpus에 없는 Gold {', '.join(missing[:4])}")
            multi = _clean(row.get(multi_col)).lower() if multi_col else ""
            rows.append({"question_id": _clean(row.get(qid_col)) if qid_col else f"ROW-{index + 2}", "question": question,
                         "gold_chunk_ids": gold, "missing_gold_ids": missing, "domain": _clean(row.get(domain_col)) if domain_col else "",
                         "multi_chunk_required": multi in {"y", "yes", "true", "1", "필요"}, "source_row": int(index + 2)})
        usable = [row for row in rows if row["gold_chunk_ids"] and not row["missing_gold_ids"]]
        if not usable:
            raise ValueError("현재 corpus 기준으로 평가 가능한 질문이 없습니다.")
        return {"sheet_name": sheet, "rows": rows, "usable": usable, "warnings": warnings[:200], "row_count": len(rows), "usable_count": len(usable)}

    def search_one(question: str, config: Mapping[str, Any], evaluation_depth: int) -> tuple[list[str], float]:
        started, depth = time.perf_counter(), int(config["candidate_depth"])
        vector = runtime_globals["_normalize_vector"](runtime_globals["embed_hcx_single"](question))
        dense = runtime_globals["dense_search_from_vector"](vector, depth)
        bm25 = runtime_globals["bm25_search"](question, depth)
        fused = runtime_globals["weighted_minmax"](dense, bm25, dense_weight=float(config["dense_weight"]),
                                                    bm25_weight=float(config["bm25_weight"]), top_k=depth)
        metric_depth = min(depth, int(evaluation_depth))
        if runtime_globals.get("RERANKER_MODEL") is not None and callable(runtime_globals.get("rerank_candidates")):
            fused, _ = runtime_globals["rerank_candidates"](
                question, fused, chunks_by_id=runtime_globals["CHUNKS_BY_ID"], model=runtime_globals["RERANKER_MODEL"],
                text_builder=runtime_globals["_reranker_passage"], candidate_depth=depth, final_top_k=metric_depth,
                batch_size=int(runtime_globals.get("RERANKER_BATCH_SIZE", 8)))
        return [str(row["chunk_id"]) for row in fused[:metric_depth]], (time.perf_counter() - started) * 1000

    def aggregate(rows: list[dict[str, Any]], prefix: str, policy: Mapping[str, Any]) -> dict[str, Any]:
        metric_keys = [f"{name}_at_{k}" for name, k in policy["metric_ks"].items()]
        out: dict[str, Any] = {}
        for key in metric_keys:
            values = [row[prefix][key] for row in rows if row[prefix][key] is not None]
            out[key] = sum(float(value) for value in values) / len(values) if values else None
        complete_key = f"complete_at_{policy['metric_ks']['complete']}"
        complete = [row[prefix][complete_key] for row in rows if row[prefix][complete_key] is not None]
        out.update(
            complete_question_count=len(complete),
            latency_avg_ms=sum(float(row[f"{prefix}_latency_ms"]) for row in rows) / len(rows),
            question_count=len(rows),
            curve={
                str(k): {
                    name: sum(float(row[prefix]["curve"][str(k)][name]) for row in rows) / len(rows)
                    for name in ("hit", "recall", "precision")
                }
                for k in policy["curve_ks"]
            },
        )
        return out

    def run_evaluation(
        job_id: str,
        dataset_id: str,
        values: Mapping[str, Any],
        max_questions: int | None,
        evaluation_depth: int,
        metric_ks: Mapping[str, int],
        curve_ks: Sequence[int],
    ) -> None:
        try:
            record, baseline = datasets[dataset_id], active_config()
            candidate = _validate_config(values, baseline)
            policy = _validate_evaluation_policy(evaluation_depth, metric_ks, curve_ks, baseline, candidate)
            source = record["usable"][:max_questions] if max_questions else record["usable"]
            details = []
            for index, row in enumerate(source, 1):
                base_ids, base_ms = search_one(row["question"], baseline, policy["evaluation_depth"])
                cand_ids, cand_ms = search_one(row["question"], candidate, policy["evaluation_depth"])
                details.append({**row, "baseline_retrieved": base_ids, "candidate_retrieved": cand_ids,
                                "baseline": _metrics(base_ids, row["gold_chunk_ids"], row["multi_chunk_required"], policy["metric_ks"], policy["curve_ks"]),
                                "candidate": _metrics(cand_ids, row["gold_chunk_ids"], row["multi_chunk_required"], policy["metric_ks"], policy["curve_ks"]),
                                "baseline_latency_ms": base_ms, "candidate_latency_ms": cand_ms})
                with eval_lock:
                    eval_jobs[job_id].update(progress=round(index / len(source) * 100), processed=index, updated_at=time.time())
            base_metrics, cand_metrics = aggregate(details, "baseline", policy), aggregate(details, "candidate", policy)
            delta = {key: None if base_metrics.get(key) is None or cand_metrics.get(key) is None else cand_metrics[key] - base_metrics[key]
                     for key in base_metrics if key not in {"question_count", "complete_question_count", "curve"}}
            with eval_lock:
                eval_jobs[job_id].update(status="done", progress=100, updated_at=time.time(), result={
                    "dataset": dataset_summary(record),
                    "evaluation_scope": "검색 계층 A/B: Structured Dense + BM25-Nori + Min-Max + BGE Reranker (질의분해/답변생성 제외)",
                    "evaluation_policy": policy,
                    "metric_definitions": [
                        {"name": name, "label": METRIC_LABELS[name], "k": k, "key": f"{name}_at_{k}"}
                        for name, k in policy["metric_ks"].items()
                    ],
                    "baseline_config": baseline, "candidate_config": candidate, "baseline": base_metrics,
                    "candidate": cand_metrics, "delta": delta, "details": details})
        except Exception as error:
            with eval_lock:
                eval_jobs[job_id].update(status="error", error=f"{type(error).__name__}: {error}", updated_at=time.time())

    def no_chat_jobs_running() -> None:
        statuses = service_module.JOB_STORE.stats().get("statuses", {})
        active = int(statuses.get("queued", 0)) + int(statuses.get("running", 0))
        if active:
            raise ValueError(f"챗봇 작업 {active}건이 실행 중입니다. 완료 후 반영해 주세요.")

    def test_hcx_key(key: str) -> dict[str, Any]:
        started = time.perf_counter()
        client = runtime_globals["OpenAI"](
            api_key=key,
            base_url=runtime_globals["HCX_BASE_URL"],
            timeout=runtime_globals["HCX_REQUEST_TIMEOUT"],
            max_retries=0,
        )
        embedding_started = time.perf_counter()
        embedding = client.embeddings.create(
            model=runtime_globals["HCX_EMBEDDING_MODEL"],
            input="예금보험공사 API 연결 확인",
            encoding_format=runtime_globals["HCX_ENCODING_FORMAT"],
        )
        if len(embedding.data) != 1 or not getattr(embedding.data[0], "embedding", None):
            raise RuntimeError("임베딩 연결 검증 결과가 올바르지 않습니다.")
        embedding_ms = (time.perf_counter() - embedding_started) * 1000
        chat_started = time.perf_counter()
        response = client.chat.completions.create(
            model=runtime_globals["HCX_CHAT_MODEL"],
            messages=[{"role": "user", "content": "연결 확인이라고만 답하세요."}],
            temperature=0.0,
            max_tokens=16,
        )
        if not response.choices:
            raise RuntimeError("답변 모델 연결 검증 결과가 비어 있습니다.")
        return {
            "valid": True,
            "fingerprint": _api_key_fingerprint(key),
            "embedding_model": str(runtime_globals["HCX_EMBEDDING_MODEL"]),
            "chat_model": str(runtime_globals["HCX_CHAT_MODEL"]),
            "embedding_latency_ms": embedding_ms,
            "chat_latency_ms": (time.perf_counter() - chat_started) * 1000,
            "total_latency_ms": (time.perf_counter() - started) * 1000,
        }

    def install_hcx_key(key: str) -> None:
        new_client = runtime_globals["OpenAI"](
            api_key=key,
            base_url=runtime_globals["HCX_BASE_URL"],
            timeout=runtime_globals["HCX_REQUEST_TIMEOUT"],
            max_retries=runtime_globals["HCX_MAX_RETRIES"],
        )
        raw_client = new_client.with_options(max_retries=0) if hasattr(new_client, "with_options") else new_client
        runtime_globals["HCX_API_KEY"] = key
        runtime_globals["HCX_CLIENT"] = new_client
        runtime_globals["_HCX_RAW_CLIENT_V3"] = raw_client
        runtime_globals["_ANSWER_BASE_CLIENT"] = raw_client
        answer_class = runtime_globals.get("_SharedGateAnswerClientV3")
        if not callable(answer_class):
            raise RuntimeError("최종 답변 클라이언트 클래스를 찾지 못했습니다.")
        runtime_globals["ANSWER_HCX_CLIENT"] = answer_class(raw_client)

        v31 = runtime_globals.get("v31")
        if v31 is None:
            raise RuntimeError("질의분석 모듈 v31을 찾지 못했습니다.")
        v31_client = v31.HCX007AtomicNeedClientV3(key, runtime_globals["V31_CONFIG"])
        v31_base = v31.KDICLightweightRAGAnalyzerV31(v31_client, runtime_globals["V31_CONFIG"])
        runtime_globals["V31_CLIENT"] = v31_client
        runtime_globals["V31_BASE_ANALYZER"] = v31_base
        runtime_globals["V31_CROSS_ANALYZER"] = runtime_globals["KDICV31V15CrossRewriteAnalyzer"](
            v31_base, v31, runtime_globals["V31_CROSS_POLICY"]
        )
        os.environ["HCX_API_KEY"] = key
        for cache_name in ("QUERY_EMBEDDING_CACHE_V3", "V31_ANALYSIS_CACHE", "_CONTEXT_CLASSIFIER_CACHE"):
            cache = runtime_globals.get(cache_name)
            if hasattr(cache, "clear"):
                cache.clear()

    def rebuild_maps(chunks: list[dict[str, Any]], vectors: Mapping[str, np.ndarray]) -> None:
        chunks.sort(key=lambda row: str(row.get("chunk_id") or ""))
        ids = [str(row["chunk_id"]) for row in chunks]
        runtime_globals["CHUNKS"], runtime_globals["CHUNKS_BY_ID"] = chunks, {str(row["chunk_id"]): row for row in chunks}
        runtime_globals["DENSE_CHUNK_IDS"] = ids
        runtime_globals["DENSE_VECTOR_BY_ID"] = {cid: np.asarray(vectors[cid], dtype=np.float32) for cid in ids}
        runtime_globals["DENSE_MATRIX"] = np.vstack([runtime_globals["DENSE_VECTOR_BY_ID"][cid] for cid in ids])
        parents, parent_ids = {}, {}
        for chunk in chunks:
            cid = str(chunk["chunk_id"]); parent = _clean(chunk.get("parent_doc_id")) or _clean(chunk.get("document_id")) or cid
            parent_ids[cid] = parent; parents.setdefault(parent, []).append(chunk)
        for children in parents.values():
            children.sort(key=lambda row: (int(row.get("chunk_index") or 0), str(row.get("chunk_id") or "")))
        runtime_globals["PARENT_CHILDREN_BY_ID"], runtime_globals["CHUNK_PARENT_ID"] = parents, parent_ids

    def snapshot() -> dict[str, Any]:
        return {"config": active_config(), "chunks": copy.deepcopy(list(runtime_globals["CHUNKS"])),
                "vectors": {cid: np.asarray(vec, dtype=np.float32).copy() for cid, vec in runtime_globals["DENSE_VECTOR_BY_ID"].items()}}

    def set_config(config: Mapping[str, Any]) -> None:
        mapping = {"DENSE_WEIGHT": "dense_weight", "BM25_WEIGHT": "bm25_weight", "CANDIDATE_DEPTH": "candidate_depth",
                   "FINAL_TOP_K": "final_top_k", "QUERY_FUSION_RRF_K": "query_fusion_rrf_k",
                   "PARENT_CHILD_ENABLED": "parent_child", "PARENT_CONTEXT_MAX_CHARS": "parent_context_max_chars"}
        for target, source in mapping.items():
            runtime_globals[target] = config[source]
        runtime_globals["RERANKER_CANDIDATE_DEPTH"] = int(config["candidate_depth"])
        runtime_globals["DENSE_KNN_NUM_CANDIDATES"] = max(int(runtime_globals.get("DENSE_KNN_NUM_CANDIDATES", 100)), int(config["candidate_depth"]))
        fuse = runtime_globals.get("fuse_query_results")
        if callable(fuse):
            kw = dict(getattr(fuse, "__kwdefaults__", {}) or {})
            kw.update(top_k=int(config["final_top_k"]), rrf_k=int(config["query_fusion_rrf_k"]))
            fuse.__kwdefaults__ = kw

    def apply_snapshot(target: Mapping[str, Any]) -> None:
        es, index = runtime_globals["ES"], str(runtime_globals["ES_INDEX_NAME"])
        current, desired = set(map(str, runtime_globals["CHUNKS_BY_ID"])), set(map(str, target["vectors"]))
        for cid in current - desired:
            if es.exists(index=index, id=cid):
                es.delete(index=index, id=cid, refresh=False)
        for chunk in target["chunks"]:
            cid = str(chunk["chunk_id"])
            es.index(index=index, id=cid, document={"chunk_id": cid,
                     "search_text": runtime_globals["build_dense_structured_v2_text"](chunk),
                     "embedding": np.asarray(target["vectors"][cid]).tolist()}, refresh=False)
        es.indices.refresh(index=index)
        rebuild_maps(copy.deepcopy(target["chunks"]), target["vectors"])
        set_config(target["config"])

    @app.get("/admin/bootstrap", include_in_schema=False)
    def admin_bootstrap(admin_token: str = Query(min_length=20, max_length=300)):
        expected = _clean(os.getenv("KDIC_ADMIN_BOOTSTRAP_TOKEN"))
        if not expected:
            raise HTTPException(status_code=503, detail="관리자 부트스트랩 토큰이 설정되지 않았습니다.")
        with auth_lock:
            if bootstrap_state["used"]:
                raise HTTPException(status_code=410, detail="관리자 일회용 접속 링크가 이미 사용되었습니다.")
            if not hmac.compare_digest(admin_token, expected):
                raise HTTPException(status_code=401, detail="관리자 일회용 접속 토큰이 올바르지 않습니다.")
            token = secrets.token_urlsafe(48); sessions[token] = time.time() + SESSION_TTL_SECONDS; bootstrap_state["used"] = True
        response = RedirectResponse(url="/admin", status_code=303)
        response.set_cookie(COOKIE_NAME, token, max_age=SESSION_TTL_SECONDS, httponly=True, secure=True, samesite="lax", path="/")
        return response

    @app.post("/api/admin-ui/login", include_in_schema=False)
    def admin_login(payload: AdminLoginPayload):
        expected = str(os.getenv("KDIC_ADMIN_TOKEN") or "").strip()
        supplied = payload.admin_token.get_secret_value().strip()
        if not expected:
            raise HTTPException(status_code=503, detail="KDIC_ADMIN_TOKEN이 설정되지 않아 관리자 로그인이 잠겨 있습니다.")
        if not supplied or not hmac.compare_digest(supplied, expected):
            raise HTTPException(status_code=401, detail="관리자 토큰이 올바르지 않습니다.")
        token = secrets.token_urlsafe(48)
        with auth_lock:
            sessions[token] = time.time() + SESSION_TTL_SECONDS
        response = JSONResponse({"authenticated": True, "expires_in_seconds": SESSION_TTL_SECONDS})
        response.set_cookie(COOKIE_NAME, token, max_age=SESSION_TTL_SECONDS, httponly=True, secure=True, samesite="strict", path="/")
        return response

    @app.get("/admin", include_in_schema=False)
    def admin_page(_: None = Depends(require_admin_session)):
        if not page.is_file(): raise HTTPException(status_code=503, detail="관리자 HTML 파일을 찾지 못했습니다.")
        return FileResponse(page, media_type="text/html; charset=utf-8")

    @app.post("/api/admin-ui/logout", include_in_schema=False)
    def logout(request: Request, _: None = Depends(require_admin_session)):
        with auth_lock: sessions.pop(_clean(request.cookies.get(COOKIE_NAME)), None)
        response = RedirectResponse(url="/", status_code=303)
        response.delete_cookie(COOKIE_NAME, path="/", secure=True, httponly=True, samesite="lax")
        return response

    @app.get("/api/admin-ui/runtime-config")
    def runtime_config(_: None = Depends(require_admin_session)): return _runtime_config(runtime_globals)

    @app.get("/api/admin-ui/summary")
    def summary(_: None = Depends(require_admin_session)):
        return {**service_module.admin_summary(), "admin_mode": "STAGED_WRITE", "draft": draft_public(),
                "datasets": [dataset_summary(row) for row in datasets.values()]}

    @app.get("/api/admin-ui/jobs")
    def jobs(limit: int = Query(default=100, ge=1, le=500), _: None = Depends(require_admin_session)): return service_module.admin_jobs(limit)

    @app.get("/api/admin-ui/indices")
    def indices(_: None = Depends(require_admin_session)): return service_module.admin_indices()

    @app.post("/api/admin-ui/search")
    def search(payload: AdminSearchPayload, _: None = Depends(require_admin_session)):
        return service_module.admin_search(service_module.AdminSearchRequest(**payload.model_dump()))

    @app.get("/api/admin-ui/documents/{index}/{document_id:path}")
    def document(index: str, document_id: str, _: None = Depends(require_admin_session)):
        base = service_module.admin_document(index, document_id)
        return {**base, "chunk_metadata": copy.deepcopy(runtime_globals.get("CHUNKS_BY_ID", {}).get(str(document_id)))}

    @app.post("/api/admin-ui/evaluations/upload")
    def upload(payload: DatasetUploadPayload, _: None = Depends(require_admin_session)):
        try:
            data = base64.b64decode(payload.content_base64, validate=True)
            if len(data) > MAX_DATASET_BYTES: raise ValueError("평가데이터셋은 25MB 이하여야 합니다.")
            parsed = normalize_dataset(data, payload.filename, payload.sheet_name)
        except Exception as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        dataset_id = f"ds-{uuid.uuid4().hex[:12]}"
        record = {"dataset_id": dataset_id, "filename": payload.filename, "created_at": time.time(), **parsed}
        datasets[dataset_id] = record
        return dataset_summary(record)

    @app.get("/api/admin-ui/evaluations/datasets")
    def list_datasets(_: None = Depends(require_admin_session)): return {"items": [dataset_summary(row) for row in datasets.values()]}

    @app.delete("/api/admin-ui/evaluations/datasets/{dataset_id}")
    def delete_dataset(dataset_id: str, _: None = Depends(require_admin_session)):
        with eval_lock:
            if dataset_id not in datasets:
                raise HTTPException(status_code=404, detail="평가데이터셋을 찾지 못했습니다.")
            if any(job.get("dataset_id") == dataset_id and job.get("status") == "running" for job in eval_jobs.values()):
                raise HTTPException(status_code=409, detail="이 데이터셋을 사용하는 평가가 실행 중입니다.")
            datasets.pop(dataset_id, None)
            removed_jobs = [job_id for job_id, job in eval_jobs.items() if job.get("dataset_id") == dataset_id]
            for job_id in removed_jobs:
                eval_jobs.pop(job_id, None)
        return {"deleted": True, "dataset_id": dataset_id, "removed_jobs": len(removed_jobs)}

    @app.delete("/api/admin-ui/evaluations/datasets")
    def clear_datasets(_: None = Depends(require_admin_session)):
        with eval_lock:
            if any(job.get("status") == "running" for job in eval_jobs.values()):
                raise HTTPException(status_code=409, detail="실행 중인 평가가 있어 데이터셋을 비울 수 없습니다.")
            count = len(datasets)
            datasets.clear()
            eval_jobs.clear()
        return {"cleared": True, "dataset_count": count}

    @app.post("/api/admin-ui/evaluations/run")
    def start_eval(payload: EvaluationRunPayload, _: None = Depends(require_admin_session)):
        if payload.dataset_id not in datasets: raise HTTPException(status_code=404, detail="평가데이터셋을 찾지 못했습니다.")
        try:
            baseline = active_config()
            candidate = _validate_config(payload.candidate, baseline)
            policy = _validate_evaluation_policy(payload.evaluation_depth, payload.metric_ks, payload.curve_ks, baseline, candidate)
        except Exception as error: raise HTTPException(status_code=422, detail=str(error)) from error
        job_id = f"eval-{uuid.uuid4().hex[:12]}"
        eval_jobs[job_id] = {"job_id": job_id, "status": "running", "progress": 0, "processed": 0,
                             "dataset_id": payload.dataset_id, "dataset_filename": datasets[payload.dataset_id]["filename"],
                             "evaluation_policy": policy, "created_at": time.time(), "updated_at": time.time(),
                             "error": None, "result": None}
        executor.submit(run_evaluation, job_id, payload.dataset_id, payload.candidate, payload.max_questions,
                        payload.evaluation_depth, payload.metric_ks, payload.curve_ks)
        return {k: v for k, v in eval_jobs[job_id].items() if k != "result"}

    @app.get("/api/admin-ui/evaluations/jobs")
    def list_eval_jobs(_: None = Depends(require_admin_session)):
        with eval_lock:
            items = [
                {key: value for key, value in copy.deepcopy(job).items() if key != "result"}
                for job in eval_jobs.values()
            ]
        return {"items": sorted(items, key=lambda row: row["created_at"], reverse=True)}

    @app.get("/api/admin-ui/evaluations/jobs/{job_id}")
    def eval_job(job_id: str, _: None = Depends(require_admin_session)):
        with eval_lock: job = copy.deepcopy(eval_jobs.get(job_id))
        if not job: raise HTTPException(status_code=404, detail="평가 작업을 찾지 못했습니다.")
        return job

    @app.delete("/api/admin-ui/evaluations/jobs/{job_id}")
    def delete_eval_job(job_id: str, _: None = Depends(require_admin_session)):
        with eval_lock:
            job = eval_jobs.get(job_id)
            if not job:
                raise HTTPException(status_code=404, detail="평가 작업을 찾지 못했습니다.")
            if job.get("status") == "running":
                raise HTTPException(status_code=409, detail="실행 중인 평가는 삭제할 수 없습니다.")
            eval_jobs.pop(job_id, None)
        return {"deleted": True, "job_id": job_id}

    @app.delete("/api/admin-ui/evaluations/jobs")
    def clear_eval_jobs(_: None = Depends(require_admin_session)):
        with eval_lock:
            removable = [job_id for job_id, job in eval_jobs.items() if job.get("status") != "running"]
            for job_id in removable:
                eval_jobs.pop(job_id, None)
        return {"cleared": True, "removed_count": len(removable)}

    @app.get("/api/admin-ui/draft")
    def get_draft(_: None = Depends(require_admin_session)): return draft_public()

    @app.get("/api/admin-ui/api-keys/status")
    def api_key_status(_: None = Depends(require_admin_session)):
        return {**api_key_state, "secret_returned_to_browser": False,
                "persistence_notice": "교체한 키는 현재 AWS 런타임과 다음 systemd 재시작에 모두 유지됩니다."}

    @app.post("/api/admin-ui/api-keys/test")
    def api_key_test(payload: ApiKeyPayload, _: None = Depends(require_admin_session)):
        try:
            key = _validate_api_key_text(payload.api_key.get_secret_value())
            return test_hcx_key(key)
        except Exception as error:
            raise HTTPException(status_code=422, detail=f"키 검증 실패: {type(error).__name__}: {error}") from error

    @app.post("/api/admin-ui/api-keys/rotate")
    def api_key_rotate(payload: ApiKeyRotatePayload, _: None = Depends(require_admin_session)):
        if payload.confirmation != "API 키 교체":
            raise HTTPException(status_code=422, detail="확인 문구로 'API 키 교체'를 입력해 주세요.")
        key = _validate_api_key_text(payload.api_key.get_secret_value())
        with mutation_lock:
            old_key = str(runtime_globals.get("HCX_API_KEY") or "")
            try:
                no_chat_jobs_running()
                check = test_hcx_key(key)
                env_file = _persist_hcx_key_to_env_file(key)
                install_hcx_key(key)
                api_key_state.update(configured=True, fingerprint=check["fingerprint"],
                                     last_rotated_at=time.time(), source="ADMIN_AWS_ENV_ROTATION")
                return {**api_key_state, "rotated": True, "test": check,
                        "persisted": True, "env_file": env_file, "secret_returned_to_browser": False}
            except HTTPException:
                raise
            except Exception as error:
                try:
                    if old_key:
                        _persist_hcx_key_to_env_file(old_key)
                        install_hcx_key(old_key)
                except Exception:
                    pass
                raise HTTPException(status_code=422, detail=f"키 교체 실패, 기존 키 유지: {type(error).__name__}: {error}") from error

    @app.put("/api/admin-ui/draft/config")
    def put_config(payload: ConfigDraftPayload, _: None = Depends(require_admin_session)):
        try: candidate = _validate_config(payload.values, active_config())
        except Exception as error: raise HTTPException(status_code=422, detail=str(error)) from error
        drafts["config"] = {key: candidate[key] for key in payload.values}
        return draft_public()

    @app.delete("/api/admin-ui/draft/config")
    def clear_config_draft(_: None = Depends(require_admin_session)):
        drafts["config"].clear()
        return draft_public()

    @app.post("/api/admin-ui/draft/chunks")
    def add_chunk(payload: ChunkAddPayload, _: None = Depends(require_admin_session)):
        chunk, cid = copy.deepcopy(payload.chunk), _clean(payload.chunk.get("chunk_id"))
        if not cid or not _clean(chunk.get("content")): raise HTTPException(status_code=422, detail="chunk_id와 content가 필요합니다.")
        if cid in runtime_globals.get("CHUNKS_BY_ID", {}) and cid not in drafts["remove"]: raise HTTPException(status_code=409, detail="이미 존재하는 chunk_id입니다.")
        chunk["chunk_id"] = cid; chunk.setdefault("parent_doc_id", _clean(chunk.get("document_id")) or cid)
        chunk.setdefault("document_id", _clean(chunk.get("parent_doc_id")) or cid); chunk.setdefault("chunk_index", 0)
        drafts["add"][cid] = chunk; drafts["remove"].discard(cid)
        return draft_public()

    @app.delete("/api/admin-ui/draft/chunks/{chunk_id:path}")
    def remove_chunk(chunk_id: str, _: None = Depends(require_admin_session)):
        cid = _clean(chunk_id)
        if cid in drafts["add"]: drafts["add"].pop(cid, None)
        elif cid in runtime_globals.get("CHUNKS_BY_ID", {}): drafts["remove"].add(cid)
        else: raise HTTPException(status_code=404, detail="청크를 찾지 못했습니다.")
        return draft_public()

    @app.delete("/api/admin-ui/draft/additions/{chunk_id:path}")
    def undo_chunk_addition(chunk_id: str, _: None = Depends(require_admin_session)):
        cid = _clean(chunk_id)
        if cid not in drafts["add"]:
            raise HTTPException(status_code=404, detail="추가 초안에서 청크를 찾지 못했습니다.")
        drafts["add"].pop(cid, None)
        return draft_public()

    @app.delete("/api/admin-ui/draft/removals/{chunk_id:path}")
    def undo_chunk_removal(chunk_id: str, _: None = Depends(require_admin_session)):
        cid = _clean(chunk_id)
        if cid not in drafts["remove"]:
            raise HTTPException(status_code=404, detail="제거 초안에서 청크를 찾지 못했습니다.")
        drafts["remove"].discard(cid)
        return draft_public()

    @app.delete("/api/admin-ui/draft")
    def clear_draft(_: None = Depends(require_admin_session)):
        drafts["config"].clear(); drafts["add"].clear(); drafts["remove"].clear(); return draft_public()

    @app.post("/api/admin-ui/apply")
    def apply(payload: ApplyPayload, _: None = Depends(require_admin_session)):
        if payload.confirmation != "운영 반영": raise HTTPException(status_code=422, detail="확인 문구로 '운영 반영'을 입력해 주세요.")
        if not draft_public()["has_changes"]: raise HTTPException(status_code=409, detail="반영할 초안이 없습니다.")
        with mutation_lock:
            try:
                no_chat_jobs_running(); before = snapshot()
                target_chunks = [copy.deepcopy(row) for row in runtime_globals["CHUNKS"] if str(row["chunk_id"]) not in drafts["remove"]]
                target_vectors = {cid: np.asarray(vec).copy() for cid, vec in runtime_globals["DENSE_VECTOR_BY_ID"].items() if cid not in drafts["remove"]}
                for cid, chunk in drafts["add"].items():
                    vector = runtime_globals["_normalize_vector"](runtime_globals["embed_hcx_single"](runtime_globals["build_dense_structured_v2_text"](chunk)))
                    if vector.shape != (int(runtime_globals["DENSE_DIMENSION"]),): raise ValueError(f"{cid} 임베딩 차원 오류: {vector.shape}")
                    target_chunks = [row for row in target_chunks if str(row["chunk_id"]) != cid] + [copy.deepcopy(chunk)]
                    target_vectors[cid] = vector
                target = {"config": _validate_config(drafts["config"], active_config()), "chunks": target_chunks, "vectors": target_vectors}
                apply_snapshot(target)
                version_id = f"ver-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
                history.append({"version_id": version_id, "created_at": time.time(), "snapshot": before, "active": True,
                                "summary": {"config": copy.deepcopy(drafts["config"]), "added": sorted(drafts["add"]), "removed": sorted(drafts["remove"])}})
                for row in history[:-1]: row["active"] = False
                drafts["config"].clear(); drafts["add"].clear(); drafts["remove"].clear()
                return {"applied": True, "version_id": version_id, "chunk_count": len(target_chunks), "active_config": active_config()}
            except Exception as error:
                try: apply_snapshot(before)
                except Exception: pass
                if isinstance(error, HTTPException): raise
                raise HTTPException(status_code=500, detail=f"반영 실패, 기존 상태 복구 시도 완료: {type(error).__name__}: {error}") from error

    @app.post("/api/admin-ui/rollback/{version_id}")
    def rollback(version_id: str, _: None = Depends(require_admin_session)):
        with mutation_lock:
            try:
                no_chat_jobs_running(); row = next((item for item in history if item["version_id"] == version_id), None)
                if not row: raise HTTPException(status_code=404, detail="롤백 버전을 찾지 못했습니다.")
                apply_snapshot(row["snapshot"])
                for item in history: item["active"] = False
                return {"rolled_back": True, "version_id": version_id, "chunk_count": len(runtime_globals["CHUNKS"]), "active_config": active_config()}
            except HTTPException: raise
            except Exception as error: raise HTTPException(status_code=500, detail=f"롤백 실패: {type(error).__name__}: {error}") from error

    @app.delete("/api/admin-ui/history/{version_id}")
    def delete_history(version_id: str, _: None = Depends(require_admin_session)):
        index = next((idx for idx, item in enumerate(history) if item["version_id"] == version_id), None)
        if index is None:
            raise HTTPException(status_code=404, detail="반영 이력을 찾지 못했습니다.")
        history.pop(index)
        return {"deleted": True, "version_id": version_id, "remaining": len(history)}

    @app.delete("/api/admin-ui/history")
    def clear_history(_: None = Depends(require_admin_session)):
        count = len(history)
        history.clear()
        return {"cleared": True, "removed_count": count}

    @app.get("/api/admin-ui/pipeline/graph")
    def pipeline_graph(_: None = Depends(require_admin_session)):
        graph = _build_pipeline_graph(runtime_globals, page.parent)
        return {key: value for key, value in graph.items() if not key.startswith("_")}

    @app.get("/api/admin-ui/pipeline/nodes/{node_id}/source")
    def pipeline_node_source(node_id: str, _: None = Depends(require_admin_session)):
        graph = _build_pipeline_graph(runtime_globals, page.parent)
        descriptor = graph["_source_index"].get(_clean(node_id))
        if descriptor is None:
            raise HTTPException(status_code=404, detail="이 노드에 연결된 조회 가능 코드가 없습니다.")
        return {
            "node_id": _clean(node_id),
            "read_only": True,
            **descriptor,
        }

    @app.get("/api/admin-ui/capabilities")
    def capabilities(_: None = Depends(require_admin_session)):
        base = service_module.admin_capabilities(); features = list(base.get("features") or [])
        for value in ("chat_pipeline_test", "runtime_config", "evaluation_dataset_upload", "evaluation_run", "evaluation_k_curve", "evaluation_record_delete", "parameter_apply", "chunk_staged_write", "draft_partial_clear", "rollback", "history_delete", "api_key_test", "api_key_runtime_rotation", "pipeline_runtime_graph", "pipeline_source_read"):
            if value not in features: features.append(value)
        blocked = [v for v in (base.get("disabled_mutations") or []) if v not in {"document_write", "document_delete", "evaluation_run", "parameter_apply"}]
        return {**base, "admin_mode": "STAGED_WRITE", "features": features, "disabled_mutations": blocked}

    return {"installed": True, "page": str(page), "auth": "one_time_bootstrap_to_httponly_cookie",
            "session_ttl_seconds": SESSION_TTL_SECONDS, "mode": "STAGED_WRITE", "evaluation": "isolated_ab",
            "mutations": "draft_validate_apply_rollback", "pipeline_studio": "dynamic_graph_read_only_source"}
