from __future__ import annotations

import ast
import base64
import copy
import hashlib
import hmac
import io
import json
import math
import os
import pickle
import re
import secrets
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from fastapi import Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field, SecretStr


COOKIE_NAME = "kdic_admin_session"
SESSION_TTL_SECONDS = 8 * 60 * 60
MAX_DATASET_BYTES = 25 * 1024 * 1024
MAX_PARAMETER_PRESETS = 30
MAX_PROMPT_PRESETS = 30
ALLOWED_CONFIG = {
    "dense_weight", "bm25_weight", "candidate_depth", "final_top_k",
    "query_fusion_rrf_k", "min_relevance_score", "parent_child", "parent_context_max_chars",
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
    baseline_metric_ks: dict[str, int] | None = None
    candidate_metric_ks: dict[str, int] | None = None
    curve_ks: list[int] = Field(default_factory=lambda: [1, 3, 5, 10, 20])


class ConfigDraftPayload(BaseModel):
    values: dict[str, Any]


class ParameterPresetPayload(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    values: dict[str, Any]


class PromptPresetPayload(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    values: dict[str, str]


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


class GuardrailDraftPayload(BaseModel):
    rules: list[dict[str, Any]] = Field(default_factory=list, max_length=100)


class GuardrailTestPayload(BaseModel):
    text: str = Field(min_length=1, max_length=10_000)
    scope: str = Field(default="input", pattern="^(input|output)$")


class GuardrailApplyPayload(BaseModel):
    confirmation: str = Field(min_length=1, max_length=100)


class PromptDraftPayload(BaseModel):
    values: dict[str, str]


class PromptComparePayload(BaseModel):
    question: str = Field(min_length=2, max_length=4_000)
    preset_id: str | None = Field(default=None, max_length=100)


class PromptApplyPayload(BaseModel):
    confirmation: str = Field(min_length=1, max_length=100)


class PipelineLabelsPayload(BaseModel):
    labels: dict[str, str] = Field(default_factory=dict)


class RechunkPreviewPayload(BaseModel):
    """Request a safe preview for one already-crawled source URL."""

    source_url: str = Field(min_length=8, max_length=4_000)


class SuggestionCreatePayload(BaseModel):
    business_key: str = Field(min_length=1, max_length=100)
    business: str = Field(min_length=1, max_length=200)
    keyword: str = Field(min_length=1, max_length=100)
    canonical_question: str = Field(min_length=2, max_length=4_000)
    display_order: int | None = Field(default=None, ge=1, le=32_767)


class SuggestionUpdatePayload(SuggestionCreatePayload):
    display_order: int = Field(ge=1, le=32_767)
    row_version: int = Field(ge=1)


class SuggestionAnswerPayload(BaseModel):
    answer: str = Field(min_length=2, max_length=30_000)
    sources: list[dict[str, str]] = Field(default_factory=list, max_length=20)


class SuggestionVersionPayload(BaseModel):
    cache_key: str = Field(min_length=1, max_length=500)


class NewUrlCrawlPayload(BaseModel):
    """Create an isolated crawl/chunk preview for one new KDIC page."""

    source_url: str = Field(min_length=8, max_length=4_000)
    document_id: str = Field(min_length=3, max_length=80)
    business_function: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=500)
    target_business_function: str | None = Field(default=None, max_length=200)
    page_type: str = Field(default="static_page", min_length=1, max_length=100)


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _safe(value: Any) -> Any:
    return value if value is None or isinstance(value, (str, int, float, bool)) else str(value)


def _is_public_kdic_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
        host = (parsed.hostname or "").lower().rstrip(".")
        return parsed.scheme.lower() in {"http", "https"} and (host == "kdic.or.kr" or host.endswith(".kdic.or.kr"))
    except Exception:
        return False


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
        "setting_view": "runtime",
        "settings": ["dense_weight", "bm25_weight", "candidate_depth", "min_relevance_score"],
    },
    {
        "id": "reranker",
        "label": "BGE Reranker",
        "stage": "RETRIEVAL",
        "description": "Hybrid 후보에 질문-청크 관련성 점수를 다시 매겨 최종 순위를 정합니다.",
        "symbols": ["rerank_candidates"],
        "setting_view": "runtime",
        "settings": ["candidate_depth", "final_top_k", "min_relevance_score"],
    },
    {
        "id": "parent_child",
        "label": "Parent-Child 확장",
        "stage": "EVIDENCE",
        "description": "상위 Child 청크를 유지하면서 Parent와 인접 문맥을 답변 근거로 확장합니다.",
        "symbols": ["expand_parent_context"],
        "enabled_flag": "PARENT_CHILD_ENABLED",
        "setting_view": "runtime",
        "settings": ["parent_child", "parent_context_max_chars"],
    },
    {
        "id": "evidence",
        "label": "Evidence Pack 구성",
        "stage": "EVIDENCE",
        "description": "검색 근거, 출처 URL, 업무별 Fact를 답변 모델이 사용할 구조로 묶습니다.",
        "symbols": ["build_compact_parent_evidence_pack_v1", "build_parent_basic_evidence_pack"],
        "setting_view": "runtime",
        "settings": ["final_top_k", "parent_context_max_chars"],
    },
    {
        "id": "answer_c",
        "label": "기본 답변 C 1Call",
        "stage": "ANSWER",
        "description": "최신 C안 Evidence Pack으로 기본 답변과 비교용 직접 답변을 생성합니다.",
        "symbols": ["generate_c_direct_threeway_v1", "_execute_c_threeway_v1", "execute_production_variant_v1"],
        "setting_view": "prompts",
        "settings": ["C_CROSS_DIRECT_SYSTEM_PROMPT_V1"],
    },
    {
        "id": "answer_dc",
        "label": "교차 업무 D-C 2Call",
        "stage": "ANSWER",
        "description": "서로 다른 업무의 근거를 분리해 구성한 뒤 교차 업무 답변을 생성합니다.",
        "symbols": ["generate_dc_twocall_v1", "execute_dc_variant_v1"],
        "setting_view": "prompts",
        "settings": ["DC_SKELETON_SYSTEM_PROMPT_V1", "DC_FINAL_SYSTEM_PROMPT_V1"],
    },
    {
        "id": "validation",
        "label": "출처·주장 안전성 검증",
        "stage": "VALIDATION",
        "description": "숫자, 적용 대상, 근거 참조, 허용된 행동 링크를 검증하고 제한 답변을 적용합니다.",
        "symbols": ["audit_dc_final_references_v1", "audit_numeric_support_dc_v1", "validate_basic_answer"],
        "setting_view": "guardrails",
        "settings": ["PII_MASKING", "FORBIDDEN_WORDS"],
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


def _build_pipeline_graph(runtime: Mapping[str, Any], labels: Mapping[str, str] | None = None) -> dict[str, Any]:
    custom = runtime.get("KDIC_PIPELINE_GRAPH_SPEC")
    specs = custom.get("nodes") if isinstance(custom, Mapping) else None
    edges = custom.get("edges") if isinstance(custom, Mapping) else None
    specs = list(specs) if isinstance(specs, Sequence) and not isinstance(specs, (str, bytes)) else DEFAULT_PIPELINE_GRAPH_SPEC
    edges = list(edges) if isinstance(edges, Sequence) and not isinstance(edges, (str, bytes)) else DEFAULT_PIPELINE_GRAPH_EDGES
    nodes: list[dict[str, Any]] = []
    custom_labels = dict(labels or {})
    for raw in specs:
        spec = dict(raw)
        node_id = _clean(spec.get("id"))
        if not node_id:
            continue
        enabled_flag = _clean(spec.get("enabled_flag"))
        enabled = bool(runtime.get(enabled_flag)) if enabled_flag else True
        settings = [str(value) for value in spec.get("settings") or [] if str(value)]
        setting_view = _clean(spec.get("setting_view"))
        node = {
            "id": node_id,
            "label": _clean(custom_labels.get(node_id)) or _clean(spec.get("label")) or node_id,
            "default_label": _clean(spec.get("label")) or node_id,
            "custom_label": node_id in custom_labels,
            "stage": _clean(spec.get("stage")) or "PIPELINE",
            "description": _clean(spec.get("description")),
            "enabled": enabled,
            "configurable": bool(settings and setting_view),
            "setting_view": setting_view or None,
            "settings": settings,
        }
        nodes.append(node)
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
        [{"id": node["id"], "settings": node["settings"], "enabled": node["enabled"]} for node in nodes],
        ensure_ascii=False,
        sort_keys=True,
    )
    return {
        "pipeline_name": _clean(runtime.get("PIPELINE_VERSION")) or "KDIC Final Runtime",
        "generated_at": time.time(),
        "fingerprint": hashlib.sha256(fingerprint_text.encode("utf-8")).hexdigest()[:16],
        "mode": "RUNTIME_INTROSPECTION" if custom is None else "RUNTIME_REGISTRY",
        "read_only": False,
        "code_access": False,
        "nodes": nodes,
        "edges": public_edges,
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
            "min_relevance_score": _safe(runtime.get("MIN_RELEVANCE_SCORE")),
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
    if not 0.0 <= float(merged["min_relevance_score"]) <= 1.0:
        raise ValueError("최소 관련성 점수는 0~1이어야 합니다.")
    if not 500 <= int(merged["parent_context_max_chars"]) <= 30_000:
        raise ValueError("Parent 문맥 한도는 500~30,000자여야 합니다.")
    return {"dense_weight": dense, "bm25_weight": bm25, "candidate_depth": depth, "final_top_k": top_k,
            "query_fusion_rrf_k": int(merged["query_fusion_rrf_k"]),
            "min_relevance_score": float(merged["min_relevance_score"]), "parent_child": bool(merged["parent_child"]),
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
    baseline_metric_ks: Mapping[str, Any],
    candidate_metric_ks: Mapping[str, Any],
    curve_ks: Sequence[Any],
    *configs: Mapping[str, Any],
) -> dict[str, Any]:
    def normalize(values: Mapping[str, Any], label: str) -> dict[str, int]:
        unknown = sorted(set(values) - set(DEFAULT_METRIC_KS))
        if unknown:
            raise ValueError(f"{label}에서 지원하지 않는 평가지표입니다: {unknown}")
        normalized = {name: int(values.get(name, default)) for name, default in DEFAULT_METRIC_KS.items()}
        if any(value < 1 or value > 200 for value in normalized.values()):
            raise ValueError(f"{label} 평가지표 K는 1~200이어야 합니다.")
        return normalized

    baseline_ks = normalize(baseline_metric_ks, "현재안")
    candidate_ks = normalize(candidate_metric_ks, "개선안")
    curve = sorted(set(int(value) for value in curve_ks))
    if any(value < 1 or value > 200 for value in curve):
        raise ValueError("구간 차트 K는 1~200 범위여야 합니다.")
    depth = int(evaluation_depth)
    required_depth = max([*baseline_ks.values(), *candidate_ks.values()])
    if depth < required_depth:
        raise ValueError(f"평가 검색 깊이는 모든 지표 K 이상이어야 합니다. 최소 {required_depth}이 필요합니다.")
    candidate_limit = min(int(config["candidate_depth"]) for config in configs) if configs else 200
    if depth > candidate_limit:
        raise ValueError(f"평가 검색 깊이 {depth}는 A/B 후보 깊이의 최솟값 {candidate_limit} 이하여야 합니다.")
    return {
        "evaluation_depth": depth,
        "baseline_metric_ks": baseline_ks,
        "candidate_metric_ks": candidate_ks,
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
    auth_lock, mutation_lock, eval_lock = threading.RLock(), threading.RLock(), threading.RLock()
    rechunk_lock = threading.RLock()
    prompt_compare_lock = threading.Lock()
    report_lock = threading.Lock()
    bootstrap_state = {"used": False}
    datasets: dict[str, dict[str, Any]] = {}
    eval_jobs: dict[str, dict[str, Any]] = {}
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kdic-admin-eval")
    rechunk_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kdic-admin-rechunk")
    rechunk_jobs: dict[str, dict[str, Any]] = {}
    suggestion_cache = getattr(service_module, "SUGGESTION_ANSWER_CACHE", None)
    drafts: dict[str, Any] = {
        "config": {},
        "add": {},
        "remove": set(),
        "rechunk": [],
        "suggestion_cache": {},
    }
    history_path = Path(os.getenv("KDIC_ADMIN_HISTORY_PATH", "/opt/kdic/runtime/admin_change_history.pkl")).resolve()
    default_snapshot_path = Path(os.getenv("KDIC_ADMIN_DEFAULT_SNAPSHOT_PATH", "/opt/kdic/runtime/admin_default_snapshot.pkl")).resolve()
    pipeline_labels_path = Path(os.getenv("KDIC_PIPELINE_LABELS_PATH", "/opt/kdic/runtime/admin_pipeline_labels.json")).resolve()
    parameter_presets_path = Path(os.getenv("KDIC_PARAMETER_PRESETS_PATH", "/opt/kdic/runtime/admin_parameter_presets.json")).resolve()
    prompt_presets_path = Path(os.getenv("KDIC_PROMPT_PRESETS_PATH", "/opt/kdic/runtime/admin_prompt_presets.json")).resolve()
    admin_sessions_path = Path(os.getenv("KDIC_ADMIN_SESSIONS_PATH", "/opt/kdic/runtime/admin_sessions.json")).resolve()
    evaluation_state_path = Path(os.getenv("KDIC_ADMIN_EVALUATIONS_PATH", "/opt/kdic/runtime/admin_evaluations.pkl")).resolve()

    def require_suggestion_cache() -> Any:
        required = (
            "list_catalog",
            "create_suggestion",
            "update_suggestion",
            "set_enabled",
            "clear_active",
            "versions",
            "save_manual_answer",
            "activate",
            "snapshot_catalog",
            "restore_catalog",
        )
        if suggestion_cache is None or any(
            not callable(getattr(suggestion_cache, name, None)) for name in required
        ):
            raise HTTPException(
                status_code=503,
                detail="현재 챗봇의 영속 질의 캐시 관리 기능이 연결되지 않았습니다.",
            )
        return suggestion_cache

    def suggestion_public() -> dict[str, Any]:
        manager = require_suggestion_cache()
        rows = manager.list_catalog(include_disabled=True)
        for row in rows:
            public_result = dict(row.pop("public_result", None) or {})
            row["answer"] = str(public_result.get("answer") or "").strip()
            row["sources"] = copy.deepcopy(
                public_result.get("sources")
                or public_result.get("official_sources")
                or []
            )
            row["has_active_answer"] = bool(row.get("active_cache_key")) and bool(
                row["answer"]
            )
            row["versions"] = manager.versions(row["suggestion_id"])
            for version in row["versions"]:
                result = dict(version.pop("public_result", None) or {})
                version["answer_preview"] = _clean(result.get("answer"))[:240]
        by_id = {row["suggestion_id"]: row for row in rows}
        for suggestion_id, change in drafts["suggestion_cache"].items():
            row = by_id.get(suggestion_id)
            if row is None and change.get("create"):
                metadata = dict(change.get("metadata") or {})
                row = {
                    "suggestion_id": suggestion_id,
                    **metadata,
                    "is_enabled": bool(change.get("enabled", True)),
                    "is_locked": False,
                    "active_cache_key": None,
                    "row_version": 1,
                    "hit_count": 0,
                    "last_hit_at": None,
                    "answer": "",
                    "sources": [],
                    "has_active_answer": False,
                    "versions": [],
                    "created_by": "ADMIN_UI_DRAFT",
                    "updated_by": "ADMIN_UI_DRAFT",
                }
                rows.append(row)
                by_id[suggestion_id] = row
            if row is None:
                continue
            row["operating_state"] = {
                key: copy.deepcopy(row.get(key))
                for key in (
                    "business_key", "business", "keyword", "canonical_question",
                    "display_order", "is_enabled", "active_cache_key", "answer", "sources",
                )
            }
            if change.get("metadata"):
                row.update(copy.deepcopy(change["metadata"]))
            if "enabled" in change:
                row["is_enabled"] = bool(change["enabled"])
            answer_change = dict(change.get("answer_change") or {})
            if answer_change.get("action") == "save":
                row["answer"] = answer_change.get("answer", "")
                row["sources"] = copy.deepcopy(answer_change.get("sources") or [])
                row["has_active_answer"] = True
            elif answer_change.get("action") == "clear":
                row["answer"] = ""
                row["sources"] = []
                row["has_active_answer"] = False
            elif answer_change.get("action") == "activate":
                selected = next(
                    (
                        version for version in row.get("versions") or []
                        if version.get("cache_key") == answer_change.get("cache_key")
                    ),
                    None,
                )
                if selected:
                    row["answer"] = selected.get("answer_preview", "")
                    row["has_active_answer"] = True
            row["draft_change"] = copy.deepcopy(change)
        rows.sort(key=lambda row: (int(row.get("display_order") or 0), row["suggestion_id"]))
        enabled = [row for row in rows if row.get("is_enabled")]
        ready = [row for row in enabled if row.get("has_active_answer")]
        return {
            "items": rows,
            "summary": {
                "registered": len(rows),
                "enabled": len(enabled),
                "ready": len(ready),
                "missing": len(enabled) - len(ready),
                "hits": sum(int(row.get("hit_count") or 0) for row in rows),
                "pending": len(drafts["suggestion_cache"]),
            },
            "draft_changes": copy.deepcopy(list(drafts["suggestion_cache"].values())),
            "policy": {
                "exact_id_and_question": True,
                "edit_invalidates_answer": True,
                "delete_is_reversible": True,
                "answer_history_retained": True,
            },
        }

    def stage_suggestion_change(suggestion_id: str, **values: Any) -> dict[str, Any]:
        clean_id = _clean(suggestion_id).upper()
        current = copy.deepcopy(drafts["suggestion_cache"].get(clean_id) or {})
        current.update({key: copy.deepcopy(value) for key, value in values.items()})
        current["suggestion_id"] = clean_id
        current["staged_at"] = time.time()
        drafts["suggestion_cache"][clean_id] = current
        return current

    def require_known_suggestion(suggestion_id: str) -> str:
        clean_id = _clean(suggestion_id).upper()
        known = {
            row["suggestion_id"]
            for row in require_suggestion_cache().list_catalog(include_disabled=True)
        }
        if clean_id not in known and not drafts["suggestion_cache"].get(clean_id, {}).get("create"):
            raise KeyError(suggestion_id)
        return clean_id

    def suggestion_id_for(payload: SuggestionCreatePayload) -> str:
        return "SQ-" + uuid.uuid5(
            uuid.NAMESPACE_URL,
            "kdic://managed-suggestion/"
            + "/".join(
                [
                    _clean(payload.business),
                    _clean(payload.keyword),
                    _clean(payload.canonical_question),
                ]
            ),
        ).hex[:16].upper()

    def load_admin_sessions() -> dict[str, float]:
        try:
            loaded = json.loads(admin_sessions_path.read_text(encoding="utf-8"))
            now = time.time()
            return {
                str(token): float(expires_at)
                for token, expires_at in dict(loaded or {}).items()
                if str(token) and float(expires_at) > now
            }
        except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError, ValueError):
            return {}

    def save_admin_sessions() -> None:
        admin_sessions_path.parent.mkdir(parents=True, exist_ok=True)
        temp = admin_sessions_path.with_suffix(admin_sessions_path.suffix + ".tmp")
        temp.write_text(json.dumps(sessions, ensure_ascii=False), encoding="utf-8")
        try:
            os.chmod(temp, 0o600)
        except OSError:
            pass
        os.replace(temp, admin_sessions_path)

    def load_evaluation_state() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        try:
            loaded = pickle.loads(evaluation_state_path.read_bytes())
            raw_datasets = loaded.get("datasets") if isinstance(loaded, Mapping) else {}
            raw_jobs = loaded.get("eval_jobs") if isinstance(loaded, Mapping) else {}
            saved_datasets = {str(key): dict(value) for key, value in dict(raw_datasets or {}).items() if isinstance(value, Mapping)}
            saved_jobs = {str(key): dict(value) for key, value in dict(raw_jobs or {}).items() if isinstance(value, Mapping)}
            for job in saved_jobs.values():
                if job.get("status") == "running":
                    job.update(status="error", error="서버 재시작으로 진행 중이던 평가가 중단되었습니다. 다시 실행해 주세요.", updated_at=time.time())
            return saved_datasets, saved_jobs
        except (FileNotFoundError, OSError, TypeError, ValueError, pickle.PickleError):
            return {}, {}

    def save_evaluation_state() -> None:
        evaluation_state_path.parent.mkdir(parents=True, exist_ok=True)
        temp = evaluation_state_path.with_suffix(evaluation_state_path.suffix + ".tmp")
        temp.write_bytes(pickle.dumps({"datasets": datasets, "eval_jobs": eval_jobs}, protocol=pickle.HIGHEST_PROTOCOL))
        os.replace(temp, evaluation_state_path)

    sessions: dict[str, float] = load_admin_sessions()
    datasets, eval_jobs = load_evaluation_state()

    def load_pipeline_labels() -> dict[str, str]:
        try:
            loaded = json.loads(pipeline_labels_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, Mapping):
                return {}
            valid_ids = {str(row["id"]) for row in DEFAULT_PIPELINE_GRAPH_SPEC}
            return {
                str(key): _clean(value)[:80]
                for key, value in loaded.items()
                if str(key) in valid_ids and _clean(value)
            }
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    def save_pipeline_labels() -> None:
        pipeline_labels_path.parent.mkdir(parents=True, exist_ok=True)
        temp = pipeline_labels_path.with_suffix(pipeline_labels_path.suffix + ".tmp")
        temp.write_text(json.dumps(pipeline_labels, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, pipeline_labels_path)

    def load_parameter_presets() -> dict[str, dict[str, Any]]:
        try:
            loaded = json.loads(parameter_presets_path.read_text(encoding="utf-8"))
            rows = loaded.get("items") if isinstance(loaded, Mapping) else loaded
            if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
                return {}
            records: dict[str, dict[str, Any]] = {}
            baseline = active_config()
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                preset_id, name = _clean(row.get("preset_id")), _clean(row.get("name"))
                if not preset_id or not name or len(name) > 60:
                    continue
                try:
                    values = _validate_config(dict(row.get("values") or {}), baseline)
                except Exception:
                    continue
                records[preset_id] = {
                    "preset_id": preset_id,
                    "name": name,
                    "values": values,
                    "created_at": float(row.get("created_at") or time.time()),
                    "updated_at": float(row.get("updated_at") or row.get("created_at") or time.time()),
                }
            return records
        except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError, ValueError):
            return {}

    def save_parameter_presets() -> None:
        parameter_presets_path.parent.mkdir(parents=True, exist_ok=True)
        ordered = sorted(parameter_presets.values(), key=lambda row: float(row.get("updated_at") or 0), reverse=True)
        temp = parameter_presets_path.with_suffix(parameter_presets_path.suffix + ".tmp")
        temp.write_text(json.dumps({"items": ordered[:MAX_PARAMETER_PRESETS]}, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, parameter_presets_path)

    def parameter_preset_public(row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "preset_id": str(row["preset_id"]),
            "name": str(row["name"]),
            "values": copy.deepcopy(dict(row["values"])),
            "created_at": float(row.get("created_at") or 0),
            "updated_at": float(row.get("updated_at") or 0),
        }

    def validate_prompt_preset_values(raw: Mapping[str, Any]) -> dict[str, str]:
        manager = require_prompt_manager()
        active = manager.active_values()
        values = dict(raw or {})
        unknown = set(values) - set(active)
        missing = set(active) - set(values)
        if unknown:
            raise ValueError("관리 대상이 아닌 프롬프트가 포함되어 있습니다: " + ", ".join(sorted(unknown)))
        if missing:
            raise ValueError("필수 프롬프트가 누락되었습니다: " + ", ".join(sorted(missing)))
        validated: dict[str, str] = {}
        secret_pattern = re.compile(r"(?i)(?:nv-|sk-)[A-Za-z0-9_-]{16,}")
        for slot in active:
            value = str(values.get(slot) or "").strip()
            if len(value) < 100:
                raise ValueError(f"{slot}: 프롬프트는 100자 이상이어야 합니다.")
            if len(value) > 30_000:
                raise ValueError(f"{slot}: 프롬프트는 30,000자 이하여야 합니다.")
            if secret_pattern.search(value):
                raise ValueError(f"{slot}: API 키로 보이는 값이 포함되어 있습니다.")
            validated[slot] = value
        return validated

    def load_prompt_presets() -> dict[str, dict[str, Any]]:
        try:
            loaded = json.loads(prompt_presets_path.read_text(encoding="utf-8"))
            rows = loaded.get("items") if isinstance(loaded, Mapping) else loaded
            if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
                return {}
            records: dict[str, dict[str, Any]] = {}
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                preset_id, name = _clean(row.get("preset_id")), _clean(row.get("name"))
                if not preset_id or not name or len(name) > 60:
                    continue
                try:
                    values = validate_prompt_preset_values(dict(row.get("values") or {}))
                except Exception:
                    continue
                records[preset_id] = {
                    "preset_id": preset_id,
                    "name": name,
                    "values": values,
                    "created_at": float(row.get("created_at") or time.time()),
                    "updated_at": float(row.get("updated_at") or row.get("created_at") or time.time()),
                }
            return records
        except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError, ValueError):
            return {}

    def save_prompt_presets() -> None:
        prompt_presets_path.parent.mkdir(parents=True, exist_ok=True)
        ordered = sorted(prompt_presets.values(), key=lambda row: float(row.get("updated_at") or 0), reverse=True)
        temp = prompt_presets_path.with_suffix(prompt_presets_path.suffix + ".tmp")
        temp.write_text(json.dumps({"items": ordered[:MAX_PROMPT_PRESETS]}, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, prompt_presets_path)

    def prompt_preset_public(row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "preset_id": str(row["preset_id"]),
            "name": str(row["name"]),
            "values": copy.deepcopy(dict(row["values"])),
            "created_at": float(row.get("created_at") or 0),
            "updated_at": float(row.get("updated_at") or 0),
        }

    def load_history() -> list[dict[str, Any]]:
        try:
            if not history_path.is_file():
                return []
            loaded = pickle.loads(history_path.read_bytes())
            if not isinstance(loaded, list):
                return []
            return [row for row in loaded if isinstance(row, dict) and isinstance(row.get("snapshot"), Mapping)][-10:]
        except Exception:
            return []

    def save_history() -> None:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        temp = history_path.with_suffix(history_path.suffix + ".tmp")
        temp.write_bytes(pickle.dumps(history[-10:], protocol=pickle.HIGHEST_PROTOCOL))
        os.replace(temp, history_path)

    history: list[dict[str, Any]] = load_history()
    pipeline_labels: dict[str, str] = load_pipeline_labels()
    initial_key = str(runtime_globals.get("HCX_API_KEY") or os.getenv("HCX_API_KEY") or "").strip()
    api_key_state: dict[str, Any] = {
        "configured": bool(initial_key),
        "fingerprint": _api_key_fingerprint(initial_key),
        "last_rotated_at": None,
        "source": "AWS_ENV_FILE_OR_STARTUP",
        "scope": "AWS_RUNTIME_AND_ENV_FILE",
    }
    guardrail_manager = runtime_globals.get("KDIC_GUARDRAIL_MANAGER")
    prompt_manager = runtime_globals.get("KDIC_PROMPT_MANAGER")
    execution_lock = runtime_globals.get("KDIC_RUNTIME_EXECUTION_LOCK")
    if execution_lock is None:
        execution_lock = threading.RLock()
        runtime_globals["KDIC_RUNTIME_EXECUTION_LOCK"] = execution_lock

    def require_guardrail_manager() -> Any:
        if guardrail_manager is None:
            raise HTTPException(status_code=503, detail="가드레일 관리자가 현재 챗봇 런타임에 연결되지 않았습니다.")
        return guardrail_manager

    def require_prompt_manager() -> Any:
        if prompt_manager is None:
            raise HTTPException(status_code=503, detail="프롬프트 관리자가 현재 챗봇 런타임에 연결되지 않았습니다.")
        return prompt_manager

    def run_prompt_variant(question: str, values: Mapping[str, str]) -> dict[str, Any]:
        override = runtime_globals.get("kdic_prompt_overrides")
        if not callable(override):
            raise HTTPException(status_code=503, detail="프롬프트 격리 실행기가 연결되지 않았습니다.")
        started = time.perf_counter()
        with override(dict(values)):
            raw = service_module.PIPELINE_RUNTIME.run(question, {}, lambda *_: None)
        public = service_module.core.normalize_public_result(raw)
        return {
            "route": public.get("route"),
            "answer": public.get("answer"),
            "answer_system": public.get("answer_system"),
            "businesses": public.get("businesses") or [],
            "sources": public.get("sources") or [],
            "coverage_status": public.get("coverage_status"),
            "validation_passed": public.get("validation_passed"),
            "latency_seconds": public.get("latency_seconds") or {},
            "wall_seconds": round(time.perf_counter() - started, 3),
        }

    def active_config() -> dict[str, Any]:
        return {"dense_weight": float(runtime_globals.get("DENSE_WEIGHT", .7)), "bm25_weight": float(runtime_globals.get("BM25_WEIGHT", .3)),
                "candidate_depth": int(runtime_globals.get("CANDIDATE_DEPTH", 20)), "final_top_k": int(runtime_globals.get("FINAL_TOP_K", 5)),
                "query_fusion_rrf_k": int(runtime_globals.get("QUERY_FUSION_RRF_K", 10)),
                "min_relevance_score": float(runtime_globals.get("MIN_RELEVANCE_SCORE", 0.0)),
                "parent_child": bool(runtime_globals.get("PARENT_CHILD_ENABLED", True)),
                "parent_context_max_chars": int(runtime_globals.get("PARENT_CONTEXT_MAX_CHARS", 8192))}

    def chunk_source_url(chunk: Mapping[str, Any]) -> str:
        return _clean(chunk.get("source_url"))

    def chunk_source_catalog() -> list[dict[str, Any]]:
        """Group only existing runtime chunks that retain their source URL.

        A title/document ID is never treated as a URL.  That prevents a broad
        replacement from accidentally affecting another source page.
        """
        grouped: dict[str, dict[str, Any]] = {}
        for chunk in runtime_globals.get("CHUNKS", []):
            if not isinstance(chunk, Mapping):
                continue
            source_url = chunk_source_url(chunk)
            if not source_url:
                continue
            row = grouped.setdefault(source_url, {
                "source_url": source_url,
                "document_ids": set(),
                "business_functions": set(),
                "chunk_count": 0,
                "sample_title": "",
            })
            row["chunk_count"] += 1
            document_id = _clean(chunk.get("document_id"))
            if document_id:
                row["document_ids"].add(document_id)
            business = _clean(chunk.get("business_function"))
            if business:
                row["business_functions"].add(business)
            if not row["sample_title"]:
                row["sample_title"] = _clean(chunk.get("title")) or _clean(chunk.get("section_title"))
        return [{
            "source_url": row["source_url"],
            "document_ids": sorted(row["document_ids"]),
            "business_functions": sorted(row["business_functions"]),
            "chunk_count": row["chunk_count"],
            "sample_title": row["sample_title"],
        } for row in sorted(grouped.values(), key=lambda item: (str(item["sample_title"]), str(item["source_url"])))]

    def source_chunks(source_url: str) -> list[dict[str, Any]]:
        return [copy.deepcopy(dict(chunk)) for chunk in runtime_globals.get("CHUNKS", [])
                if isinstance(chunk, Mapping) and chunk_source_url(chunk) == source_url]

    def new_url_business_catalog() -> list[dict[str, Any]]:
        """Recommend the next domain document ID from current runtime data."""
        grouped: dict[str, dict[str, Any]] = {}
        all_document_ids: set[str] = set()
        for chunk in runtime_globals.get("CHUNKS", []):
            if not isinstance(chunk, Mapping):
                continue
            business = _clean(chunk.get("business_function")) or "미분류"
            document_id = _clean(chunk.get("document_id"))
            if document_id:
                all_document_ids.add(document_id)
            match = re.fullmatch(r"([A-Z][A-Z0-9]{1,7})-(\d{3,6})", document_id)
            if not match:
                continue
            row = grouped.setdefault(business, {"business_function": business, "prefix_counts": {}, "ids": []})
            prefix, number = match.group(1), int(match.group(2))
            row["prefix_counts"][prefix] = row["prefix_counts"].get(prefix, 0) + 1
            row["ids"].append((prefix, number, len(match.group(2))))
        items: list[dict[str, Any]] = []
        for business, row in grouped.items():
            prefix = sorted(row["prefix_counts"], key=lambda value: (-row["prefix_counts"][value], value))[0]
            matching = [(number, width) for current_prefix, number, width in row["ids"] if current_prefix == prefix]
            width = max([value[1] for value in matching] or [3])
            next_number = max([value[0] for value in matching] or [0]) + 1
            recommendation = f"{prefix}-{next_number:0{width}d}"
            while recommendation in all_document_ids:
                next_number += 1
                recommendation = f"{prefix}-{next_number:0{width}d}"
            items.append({
                "business_function": business,
                "document_prefix": prefix,
                "recommended_document_id": recommendation,
            })
        return sorted(items, key=lambda row: row["business_function"])

    # A URL refresh is deliberately reviewed before it is staged.  These are
    # the fields needed to preserve document identity, provenance and search
    # context after replacement.  Missing values are a blocking error, not a
    # best-effort warning.
    rechunk_required_fields = (
        "chunk_id", "document_id", "parent_doc_id", "title",
        "business_function", "source_url", "content",
    )
    rechunk_preview_text_limit = 120_000

    def rechunk_json_value(value: Any) -> Any:
        """JSONResponse rejects NaN values emitted by pandas quality reports."""
        if isinstance(value, (float, np.floating)):
            try:
                return None if not math.isfinite(float(value)) else float(value)
            except (TypeError, ValueError):
                return None
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)):
            return int(value)
        return _safe(value)

    def rechunk_chunk_review(chunk: Mapping[str, Any]) -> dict[str, Any]:
        """Return an admin-safe, human-readable review record for one chunk."""
        content = str(chunk.get("content") or "").strip()
        metadata = {
            key: _safe(chunk.get(key)) for key in (
                "chunk_id", "document_id", "parent_doc_id", "chunk_index",
                "title", "section_title", "business_function",
                "target_business_function", "source_url", "page_type",
                "chunk_type",
            )
        }
        missing = [key for key in rechunk_required_fields if not _clean(chunk.get(key))]
        length_flags: list[str] = []
        if content and len(content) < 80:
            length_flags.append("본문이 80자 미만입니다. 제목·표·FAQ 단위 청크인지 확인하세요.")
        if len(content) > 2_000:
            length_flags.append("본문이 2,000자를 초과합니다. 청크 분할 지점을 확인하세요.")
        return {
            "chunk_id": str(chunk.get("chunk_id") or ""),
            "content": content,
            "content_chars": len(content),
            "metadata": metadata,
            "missing_required": missing,
            "length_flags": length_flags,
        }

    def rechunk_document_review(document: Mapping[str, Any]) -> dict[str, Any]:
        """Expose parsed text and parsing quality without exposing raw HTML."""
        markdown = str(document.get("content_markdown") or document.get("body_markdown") or "").strip()
        text = str(document.get("content_text") or "").strip()
        shown = markdown or text
        truncated = len(shown) > rechunk_preview_text_limit
        quality = document.get("quality") if isinstance(document.get("quality"), Mapping) else {}
        return {
            "document_id": _clean(document.get("doc_id")) or _clean(document.get("document_id")),
            "title": _clean(document.get("title")),
            "source_url": _clean(document.get("source_url")),
            "parsed_markdown": shown[:rechunk_preview_text_limit],
            "parsed_text_chars": len(text) or len(shown),
            "preview_truncated": truncated,
            "metadata": {
                key: _safe(document.get(key)) for key in (
                    "doc_id", "parent_doc_id", "title", "business_function",
                    "target_business_function", "source_url", "final_url", "page_type",
                    "parser_profile", "chunk_strategy", "parsed_at",
                )
            },
            "quality": {
                "status": _safe(quality.get("status")),
                "reasons": [_safe(value) for value in (quality.get("reasons") or [])],
                "source_text_chars": _safe(quality.get("source_text_chars")),
                "parsed_text_chars": _safe(quality.get("parsed_text_chars")),
                "retention_ratio": _safe(quality.get("retention_ratio")),
                "faq_count": _safe(quality.get("faq_count")),
                "table_count": _safe(quality.get("table_count")),
            },
        }

    def rechunk_review_bundle(*, old_chunks: Sequence[Mapping[str, Any]],
                              new_chunks: Sequence[Mapping[str, Any]],
                              documents: Sequence[Mapping[str, Any]],
                              quality_rows: Sequence[Mapping[str, Any]],
                              validation_rows: Sequence[Mapping[str, Any]],
                              document_ids: set[str]) -> dict[str, Any]:
        """Build comparison, warnings and hard staging checks for a refresh."""
        old = [rechunk_chunk_review(chunk) for chunk in old_chunks]
        new = [rechunk_chunk_review(chunk) for chunk in new_chunks]
        old_by_id = {row["chunk_id"]: row for row in old if row["chunk_id"]}
        new_by_id = {row["chunk_id"]: row for row in new if row["chunk_id"]}
        comparisons: list[dict[str, Any]] = []
        counts = {"same": 0, "changed": 0, "removed": 0, "added": 0}
        for chunk_id in sorted(set(old_by_id) | set(new_by_id)):
            before, after = old_by_id.get(chunk_id), new_by_id.get(chunk_id)
            if before is None:
                status = "added"
            elif after is None:
                status = "removed"
            elif before["content"] == after["content"] and before["metadata"] == after["metadata"]:
                status = "same"
            else:
                status = "changed"
            counts[status] += 1
            comparisons.append({
                "chunk_id": chunk_id,
                "status": status,
                "old_chars": before["content_chars"] if before else 0,
                "new_chars": after["content_chars"] if after else 0,
                "old_section_title": (before or {}).get("metadata", {}).get("section_title", ""),
                "new_section_title": (after or {}).get("metadata", {}).get("section_title", ""),
            })

        selected_documents = [rechunk_document_review(document) for document in documents
                              if (_clean(document.get("doc_id")) in document_ids
                                  and _clean(document.get("source_url")))]
        selected_quality = [dict(row) for row in quality_rows if _clean(row.get("url_id")) in document_ids]
        selected_validations = [dict(row) for row in validation_rows
                                if any(str(document_id) in str(row.get("validation") or "")
                                       for document_id in document_ids)]
        errors: list[str] = []
        warnings: list[str] = []
        if not new:
            errors.append("새 청크가 0개입니다. 기존 청크를 제거하거나 교체 초안을 만들 수 없습니다.")
        if not selected_documents:
            errors.append("재수집한 파싱 문서를 찾지 못했습니다. 파싱 결과를 검토할 수 없어 반영을 차단했습니다.")
        for row in new:
            if row["missing_required"]:
                errors.append(f"{row['chunk_id'] or '청크 ID 없음'}: 필수 메타데이터 누락 ({', '.join(row['missing_required'])})")
            for flag in row["length_flags"]:
                warnings.append(f"{row['chunk_id']}: {flag}")
        for row in selected_documents:
            if not row["parsed_markdown"]:
                errors.append(f"{row['document_id'] or '문서 ID 없음'}: 파싱된 본문이 비어 있습니다.")
            if row["quality"].get("status") == "fail":
                reasons = ", ".join(map(str, row["quality"].get("reasons") or [])) or "품질 검사 실패"
                errors.append(f"{row['document_id']}: 파싱 품질 검사 실패 ({reasons})")
            try:
                ratio = float(row["quality"].get("retention_ratio") or 0)
                if ratio and ratio < .98:
                    errors.append(f"{row['document_id']}: 본문 보존율 {ratio:.2%}로 기준 98% 미만입니다.")
            except (TypeError, ValueError):
                warnings.append(f"{row['document_id']}: 본문 보존율을 숫자로 확인하지 못했습니다.")
        for row in selected_quality:
            if str(row.get("quality_status") or "").lower() == "fail":
                errors.append(f"{row.get('url_id')}: 청킹 품질 검사 실패 ({row.get('quality_reasons') or '사유 없음'})")
        for row in selected_validations:
            passed = row.get("passed")
            if isinstance(passed, str):
                passed = passed.strip().lower() in {"true", "1", "yes", "y"}
            if passed is False:
                errors.append(f"회귀 검사 실패: {row.get('validation')}")
        # Preserve first occurrence: one failed quality check can be represented in
        # both document and quality reports.
        errors = list(dict.fromkeys(errors))
        warnings = list(dict.fromkeys(warnings))
        return {
            "parsed_documents": selected_documents,
            "old_chunks": old,
            "new_chunks": new,
            "comparisons": comparisons,
            "comparison_counts": counts,
            "quality_rows": [{key: rechunk_json_value(value) for key, value in row.items()} for row in selected_quality],
            "validation_rows": [{key: rechunk_json_value(value) for key, value in row.items()} for row in selected_validations],
            "errors": errors,
            "warnings": warnings,
            "can_stage": not errors,
        }

    def normalized_rechunk_preview(chunks: Sequence[Mapping[str, Any]], *, source_url: str,
                                   document_ids: set[str]) -> list[dict[str, Any]]:
        """Keep the source document's ID convention while issuing fresh chunk IDs."""
        candidates: list[dict[str, Any]] = []
        for raw in chunks:
            chunk = copy.deepcopy(dict(raw))
            document_id = _clean(chunk.get("document_id"))
            if document_id not in document_ids:
                continue
            content = str(chunk.get("content") or "").strip()
            if not content:
                continue
            chunk.update(document_id=document_id,
                         parent_doc_id=_clean(chunk.get("parent_doc_id")) or document_id,
                         source_url=source_url,
                         content=content,
                         title=_clean(chunk.get("title")),
                         section_title=_clean(chunk.get("section_title")),
                         business_function=_clean(chunk.get("business_function")))
            candidates.append(chunk)
        if not candidates:
            raise ValueError("재수집 결과에서 선택한 URL의 본문 청크를 찾지 못했습니다. 문서 ID와 URL 연결을 확인해 주세요.")
        def index_of(row: Mapping[str, Any]) -> int:
            try:
                return int(row.get("chunk_index") or 0)
            except (TypeError, ValueError):
                return 0
        candidates.sort(key=lambda row: (_clean(row.get("document_id")), index_of(row), _clean(row.get("chunk_id"))))
        counters: dict[str, int] = {}
        normalized: list[dict[str, Any]] = []
        for chunk in candidates:
            document_id = str(chunk["document_id"])
            index = counters.get(document_id, 0)
            counters[document_id] = index + 1
            chunk["chunk_id"] = f"{document_id}_chunk_{index:03d}"
            chunk["chunk_index"] = index
            normalized.append(chunk)
        return normalized

    def rechunk_public(row: Mapping[str, Any], *, include_preview: bool = False) -> dict[str, Any]:
        out = {key: copy.deepcopy(row.get(key)) for key in (
            "job_id", "mode", "source_url", "status", "stage", "stage_detail", "progress",
            "error", "created_at", "started_at", "heartbeat_at", "updated_at",
            "old_chunk_count", "new_chunk_count", "document_ids", "business_functions", "staged",
        )}
        if include_preview:
            out["preview_chunks"] = copy.deepcopy(list(row.get("preview_chunks") or []))
            out["review"] = copy.deepcopy(dict(row.get("review") or {}))
        return out

    def build_rechunk_review_csv(job_id: str, *, source_url: str, document_ids: set[str],
                                 new_document: Mapping[str, Any] | None = None) -> Path:
        """Create the minimum review manifest for one already-approved URL.

        Older EC2 deployments keep the published chunks but not the original
        crawling review CSV.  Recreating a single-row (or document-ID scoped)
        manifest from the current runtime metadata lets this admin operation
        refresh only its selected URL without fabricating a whole policy table.
        """
        by_document: dict[str, dict[str, Any]] = {}
        if new_document is not None:
            document_id = _clean(new_document.get("document_id"))
            if document_id:
                by_document[document_id] = dict(new_document)
        else:
            for chunk in source_chunks(source_url):
                document_id = _clean(chunk.get("document_id"))
                if document_id in document_ids and document_id not in by_document:
                    by_document[document_id] = chunk
        missing = sorted(document_ids - set(by_document))
        if missing:
            raise ValueError(f"선택 URL의 문서 메타데이터를 찾지 못했습니다: {', '.join(missing)}")
        common = {
            "권장_최종결정": "include_full",
            "RAG_본문_인덱싱": "O",
            "인덱싱_범위": "본문 전체",
            "Action_링크": "",
            "권장_Action_Type": "",
            "Action_인증": "",
            "다중페이지_수집정책": "",
            "검토_의견": "관리자 URL 재수집·재청킹",
            "검토_근거": "기존 운영 청크의 source_url과 document_id를 사용",
            "첨부파일_원본수집": "",
            "첨부파일_RAG_정책": "",
            "첨부파일_사용자제공정책": "",
            "영상_처리정책": "",
            "웹툰_처리정책": "",
            "일반이미지_처리정책": "",
            "보조콘텐츠_표시조건": "",
            "보조콘텐츠_링크라벨": "",
        }
        rows: list[dict[str, str]] = []
        for document_id in sorted(document_ids):
            chunk = by_document[document_id]
            business = _clean(chunk.get("business_function")) or "미분류"
            rows.append({
                **common,
                "문서_ID": document_id,
                "업무_도메인": business,
                "목표_도메인": _clean(chunk.get("target_business_function")) or business,
                "문서명": _clean(chunk.get("title")) or document_id,
                "페이지_유형": _clean(chunk.get("page_type")) or "static_page",
                "출처_URL": source_url,
            })
        path = service_module._crawler_runtime_root() / "admin_rechunk_manifests" / f"{job_id}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
        return path

    def run_rechunk_preview(job_id: str) -> None:
        def update_progress(percent: int, stage: str, detail: str = "") -> None:
            with rechunk_lock:
                current = rechunk_jobs.get(job_id)
                if current is not None:
                    current.update(
                        progress=max(0, min(100, int(percent))),
                        stage=stage,
                        stage_detail=str(detail or ""),
                        heartbeat_at=time.time(),
                        updated_at=time.time(),
                    )

        with rechunk_lock:
            job = rechunk_jobs.get(job_id)
            if job is None:
                return
            job.update(status="running", stage="수집 준비", stage_detail="정책과 URL을 확인하고 있습니다.",
                       progress=1, started_at=time.time(), heartbeat_at=time.time(), updated_at=time.time())
            source_url, document_ids = str(job["source_url"]), set(job["document_ids"])
            old_chunks = copy.deepcopy(list(job.get("old_chunks_snapshot") or []))
            new_document = copy.deepcopy(job.get("new_document")) if isinstance(job.get("new_document"), Mapping) else None
        try:
            review_csv_path = build_rechunk_review_csv(
                job_id, source_url=source_url, document_ids=document_ids, new_document=new_document,
            )
            config = service_module.kdic_final_pipeline.PipelineConfig(run_only_url_ids=sorted(document_ids))
            result = service_module.kdic_final_pipeline.run_pipeline(
                review_csv_path=review_csv_path,
                runtime_root=service_module._crawler_runtime_root(),
                config=config,
                progress_callback=update_progress,
            )
            if new_document is not None:
                redirected = [
                    _clean(document.get("final_url")) or _clean(document.get("source_url"))
                    for document in list(result.get("page_documents") or [])
                ]
                invalid_redirects = [url for url in redirected if url and not _is_public_kdic_url(url)]
                if invalid_redirects:
                    raise ValueError("신규 URL이 예금보험공사 공개 도메인 밖으로 이동해 처리를 차단했습니다.")
            preview = normalized_rechunk_preview(result.get("chunks") or [], source_url=source_url,
                                                 document_ids=document_ids)
            quality_df = result.get("quality_df")
            validation_df = result.get("validation_df")
            quality_rows = quality_df.to_dict("records") if isinstance(quality_df, pd.DataFrame) else []
            validation_rows = validation_df.to_dict("records") if isinstance(validation_df, pd.DataFrame) else []
            review = rechunk_review_bundle(
                old_chunks=old_chunks,
                new_chunks=preview,
                documents=list(result.get("page_documents") or result.get("documents") or []),
                quality_rows=quality_rows,
                validation_rows=validation_rows,
                document_ids=document_ids,
            )
            with rechunk_lock:
                job = rechunk_jobs[job_id]
                stage = "재청킹 미리보기 준비 완료"
                if not review["can_stage"]:
                    stage = "검수 오류 발견 · 교체 초안 생성 차단"
                job.update(status="ready", stage=stage, preview_chunks=preview, review=review,
                           new_chunk_count=len(preview), progress=100, stage_detail="검수 화면을 확인해 주세요.",
                           heartbeat_at=time.time(), updated_at=time.time())
        except Exception as error:
            with rechunk_lock:
                job = rechunk_jobs.get(job_id)
                if job is not None:
                    job.update(status="failed", stage="재청킹 실패",
                               stage_detail="오류 내용을 확인하고 다시 실행해 주세요.", progress=100,
                               error=f"{type(error).__name__}: {error}", heartbeat_at=time.time(),
                               updated_at=time.time())

    parameter_presets: dict[str, dict[str, Any]] = load_parameter_presets()
    prompt_presets: dict[str, dict[str, Any]] = load_prompt_presets()

    def job_telemetry(public: Mapping[str, Any], stored: Any) -> dict[str, Any]:
        """Return request-time metrics without confusing a cache hit with its origin run."""

        result = public.get("result") if isinstance(public.get("result"), Mapping) else {}
        raw = getattr(stored, "raw_result", None)
        raw = raw if isinstance(raw, Mapping) else {}
        cache = result.get("suggestion_cache") if isinstance(result.get("suggestion_cache"), Mapping) else {}
        usage = raw.get("usage") if isinstance(raw.get("usage"), Mapping) else {}
        event = raw.get("event") if isinstance(raw.get("event"), Mapping) else {}
        latency = raw.get("latency") if isinstance(raw.get("latency"), Mapping) else {}

        def number(value: Any) -> float:
            try:
                return float(value or 0)
            except (TypeError, ValueError):
                return 0.0

        origin_prompt = int(number(usage.get("prompt_tokens") or event.get("prompt_tokens")))
        origin_completion = int(number(usage.get("completion_tokens") or event.get("completion_tokens")))
        origin_total = int(number(usage.get("total_tokens") or event.get("total_tokens")))
        origin_latency_ms = number(event.get("click_wall_ms") or latency.get("click_wall_ms"))
        cache_hit = bool(cache.get("hit"))

        if cache_hit:
            request_latency_ms = number(cache.get("lookup_ms"))
            if request_latency_ms <= 0:
                current_latency = result.get("latency_seconds")
                current_latency = current_latency if isinstance(current_latency, Mapping) else {}
                request_latency_ms = sum(
                    number(value) * 1000.0
                    for value in current_latency.values()
                )
            prompt_tokens = completion_tokens = total_tokens = 0
        else:
            request_latency_ms = origin_latency_ms
            prompt_tokens, completion_tokens, total_tokens = (
                origin_prompt,
                origin_completion,
                origin_total,
            )

        return {
            "cache_hit": cache_hit,
            "cache_source": str(cache.get("source") or "").strip(),
            "request_latency_ms": round(request_latency_ms, 3),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "origin_latency_ms": round(origin_latency_ms, 3),
            "origin_prompt_tokens": origin_prompt,
            "origin_completion_tokens": origin_completion,
            "origin_total_tokens": origin_total,
            "saved_latency_ms": round(max(0.0, origin_latency_ms - request_latency_ms), 3) if cache_hit else 0.0,
            "saved_tokens": origin_total if cache_hit else 0,
        }

    def job_public_with_telemetry(public: Mapping[str, Any]) -> dict[str, Any]:
        item = dict(public)
        stored = service_module.JOB_STORE.get(str(item.get("job_id") or ""))
        item["telemetry"] = job_telemetry(item, stored)
        return item

    def monitoring_snapshot(hours: float, limit: int, offset: int = 0) -> dict[str, Any]:
        now = time.time()
        cutoff = now - (hours * 3600)
        try:
            rows = service_module.JOB_STORE.list_public(5_000, since=cutoff)
            total_available = int(service_module.JOB_STORE.count_public(since=cutoff))
        except TypeError:
            # 이전 로컬 개발용 저장소와도 호환되도록 유지한다.
            rows = service_module.JOB_STORE.list_public(5_000)
            total_available = None
        records: list[dict[str, Any]] = []
        for public in rows:
            created_at = float(public.get("created_at") or 0)
            if created_at < cutoff:
                continue
            stored = service_module.JOB_STORE.get(str(public.get("job_id") or ""))
            raw = getattr(stored, "raw_result", None)
            raw = raw if isinstance(raw, Mapping) else {}
            result = public.get("result") if isinstance(public.get("result"), Mapping) else {}
            telemetry = job_telemetry(public, stored)
            businesses = result.get("businesses") if isinstance(result.get("businesses"), Sequence) and not isinstance(result.get("businesses"), (str, bytes)) else []
            records.append({
                "job_id": str(public.get("job_id") or ""),
                "created_at": created_at,
                "question": _clean(public.get("question"))[:160],
                "status": _clean(public.get("status")) or "unknown",
                "route": _clean(result.get("route") or raw.get("route")) or "UNKNOWN",
                "businesses": [_clean(value) for value in businesses if _clean(value)],
                **telemetry,
                "latency_ms": telemetry["request_latency_ms"],
                "validation_passed": result.get("validation_passed"),
                "coverage_status": _clean(result.get("coverage_status")),
            })

        completed = [row for row in records if row["status"] == "done"]
        failed = [row for row in records if row["status"] == "error"]
        latency_values = sorted(row["latency_ms"] for row in completed if row["latency_ms"] > 0)
        token_rows = [row for row in completed if row["total_tokens"] > 0]
        p95_index = max(0, math.ceil(len(latency_values) * .95) - 1) if latency_values else 0
        route_counts: dict[str, int] = {}
        business_counts: dict[str, int] = {}
        for row in records:
            route_counts[row["route"]] = route_counts.get(row["route"], 0) + 1
            for business in row["businesses"] or ["미분류"]:
                business_counts[business] = business_counts.get(business, 0) + 1

        bucket_seconds = 3600 if hours <= 48 else 86400
        bucket_count = min(31, math.ceil(hours * 3600 / bucket_seconds))
        buckets = []
        start = math.floor(cutoff / bucket_seconds) * bucket_seconds
        for index in range(bucket_count + 1):
            bucket_start = start + (index * bucket_seconds)
            if bucket_start > now:
                break
            selected = [row for row in records if bucket_start <= row["created_at"] < bucket_start + bucket_seconds]
            buckets.append({
                "timestamp": bucket_start,
                "questions": len(selected),
                "tokens": sum(row["total_tokens"] for row in selected),
                "failures": sum(row["status"] == "error" for row in selected),
                "avg_latency_ms": round(sum(row["latency_ms"] for row in selected) / len(selected), 3) if selected else 0,
            })

        total_tokens = sum(row["total_tokens"] for row in completed)
        cache_hits = sum(bool(row["cache_hit"]) for row in completed)
        return {
            "generated_at": now,
            "hours": hours,
            "retention_notice": "현재 작업 저장소의 보존 범위 안에서 집계합니다.",
            "summary": {
                "questions": total_available if total_available is not None else len(records),
                "completed": len(completed),
                "failed": len(failed),
                "success_rate": len(completed) / len(records) if records else 0,
                "total_tokens": total_tokens,
                "prompt_tokens": sum(row["prompt_tokens"] for row in completed),
                "completion_tokens": sum(row["completion_tokens"] for row in completed),
                "avg_tokens": total_tokens / len(token_rows) if token_rows else 0,
                "token_coverage_rate": len(token_rows) / len(completed) if completed else 0,
                "avg_latency_ms": sum(latency_values) / len(latency_values) if latency_values else 0,
                "p95_latency_ms": latency_values[p95_index] if latency_values else 0,
                "cache_hits": cache_hits,
                "cache_misses": len(completed) - cache_hits,
                "cache_hit_rate": cache_hits / len(completed) if completed else 0,
                "cache_saved_tokens": sum(row["saved_tokens"] for row in completed),
                "cache_saved_latency_ms": sum(row["saved_latency_ms"] for row in completed),
            },
            "timeseries": buckets,
            "routes": [{"name": key, "count": value} for key, value in sorted(route_counts.items(), key=lambda item: (-item[1], item[0]))],
            "businesses": [{"name": key, "count": value} for key, value in sorted(business_counts.items(), key=lambda item: (-item[1], item[0]))],
            "recent": sorted(records, key=lambda row: row["created_at"], reverse=True)[max(0, offset):max(0, offset) + limit],
            "recent_total": total_available if total_available is not None else len(records),
            "recent_offset": max(0, offset),
            "recent_limit": limit,
        }

    def require_admin_session(request: Request) -> None:
        supplied = _clean(request.cookies.get(COOKIE_NAME))
        with auth_lock:
            now = time.time()
            expired = False
            for token, expires in list(sessions.items()):
                if expires <= now:
                    sessions.pop(token, None)
                    expired = True
            expires_at = sessions.get(supplied)
            if expired:
                save_admin_sessions()
        if not supplied or not expires_at or expires_at <= time.time():
            raise HTTPException(status_code=401, detail="관리자 세션이 없거나 만료되었습니다.")

    def dataset_summary(row: Mapping[str, Any]) -> dict[str, Any]:
        return {key: row[key] for key in ("dataset_id", "filename", "sheet_name", "row_count", "usable_count", "warnings", "created_at")}

    def draft_public() -> dict[str, Any]:
        prompt_state = prompt_manager.public() if prompt_manager is not None else {"has_changes": False, "slots": []}
        return {"active_config": active_config(), "candidate_config": _validate_config(drafts["config"], active_config()),
                "add": list(drafts["add"].values()), "remove": sorted(drafts["remove"]),
                "rechunk": copy.deepcopy(drafts["rechunk"]),
                "suggestion_cache": copy.deepcopy(list(drafts["suggestion_cache"].values())),
                "prompt_changes": [row["slot"] for row in prompt_state.get("slots") or [] if row.get("changed")],
                "has_changes": bool(drafts["config"] or drafts["add"] or drafts["remove"] or drafts["suggestion_cache"] or prompt_state.get("has_changes")),
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
        threshold = float(config.get("min_relevance_score", 0.0))
        score_fn = runtime_globals.get("retrieval_relevance_score")
        if callable(score_fn):
            fused = [row for row in fused if float(score_fn(row)) >= threshold]
        return [str(row["chunk_id"]) for row in fused[:metric_depth]], (time.perf_counter() - started) * 1000

    def aggregate(
        rows: list[dict[str, Any]],
        prefix: str,
        metric_ks: Mapping[str, int],
        curve_ks: Sequence[int],
    ) -> dict[str, Any]:
        metric_keys = [f"{name}_at_{k}" for name, k in metric_ks.items()]
        out: dict[str, Any] = {}
        for key in metric_keys:
            values = [row[prefix][key] for row in rows if row[prefix][key] is not None]
            out[key] = sum(float(value) for value in values) / len(values) if values else None
        complete_key = f"complete_at_{metric_ks['complete']}"
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
                for k in curve_ks
            },
        )
        return out

    def run_evaluation(
        job_id: str,
        dataset_id: str,
        values: Mapping[str, Any],
        max_questions: int | None,
        evaluation_depth: int,
        baseline_metric_ks: Mapping[str, int],
        candidate_metric_ks: Mapping[str, int],
        curve_ks: Sequence[int],
    ) -> None:
        try:
            record, baseline = datasets[dataset_id], active_config()
            candidate = _validate_config(values, baseline)
            policy = _validate_evaluation_policy(
                evaluation_depth, baseline_metric_ks, candidate_metric_ks, curve_ks, baseline, candidate
            )
            baseline_ks = policy["baseline_metric_ks"]
            candidate_ks = policy["candidate_metric_ks"]
            source = record["usable"][:max_questions] if max_questions else record["usable"]
            details = []
            for index, row in enumerate(source, 1):
                base_ids, base_ms = search_one(row["question"], baseline, policy["evaluation_depth"])
                cand_ids, cand_ms = search_one(row["question"], candidate, policy["evaluation_depth"])
                details.append({**row, "baseline_retrieved": base_ids, "candidate_retrieved": cand_ids,
                                "baseline": _metrics(base_ids, row["gold_chunk_ids"], row["multi_chunk_required"], baseline_ks, policy["curve_ks"]),
                                "candidate": _metrics(cand_ids, row["gold_chunk_ids"], row["multi_chunk_required"], candidate_ks, policy["curve_ks"]),
                                "baseline_latency_ms": base_ms, "candidate_latency_ms": cand_ms})
                with eval_lock:
                    eval_jobs[job_id].update(progress=round(index / len(source) * 100), processed=index, updated_at=time.time())
            base_metrics = aggregate(details, "baseline", baseline_ks, policy["curve_ks"])
            cand_metrics = aggregate(details, "candidate", candidate_ks, policy["curve_ks"])
            definitions = [
                {
                    "name": name,
                    "label": METRIC_LABELS[name],
                    "baseline_k": baseline_ks[name],
                    "candidate_k": candidate_ks[name],
                    "baseline_key": f"{name}_at_{baseline_ks[name]}",
                    "candidate_key": f"{name}_at_{candidate_ks[name]}",
                }
                for name in DEFAULT_METRIC_KS
            ]
            delta = {
                definition["name"]: (
                    None
                    if base_metrics.get(definition["baseline_key"]) is None
                    or cand_metrics.get(definition["candidate_key"]) is None
                    else cand_metrics[definition["candidate_key"]] - base_metrics[definition["baseline_key"]]
                )
                for definition in definitions
            }
            delta["latency_avg_ms"] = cand_metrics["latency_avg_ms"] - base_metrics["latency_avg_ms"]
            with eval_lock:
                eval_jobs[job_id].update(status="done", progress=100, updated_at=time.time(), result={
                    "dataset": dataset_summary(record),
                    "evaluation_scope": "검색 계층 A/B: Structured Dense + BM25-Nori + Min-Max + BGE Reranker (질의분해/답변생성 제외)",
                    "evaluation_policy": policy,
                    "metric_definitions": definitions,
                    "baseline_config": baseline, "candidate_config": candidate, "baseline": base_metrics,
                    "candidate": cand_metrics, "delta": delta, "details": details})
                save_evaluation_state()
        except Exception as error:
            with eval_lock:
                eval_jobs[job_id].update(status="error", error=f"{type(error).__name__}: {error}", updated_at=time.time())
                save_evaluation_state()

    def evaluation_change_summary(result: Mapping[str, Any]) -> dict[str, Any]:
        """Calculate whole-dataset changes before asking the LLM to interpret them."""
        definitions = list(result.get("metric_definitions") or [])
        rows: list[dict[str, Any]] = []
        domain_counts: dict[str, dict[str, int]] = {}
        tolerance = 0.005
        for detail in result.get("details") or []:
            current_values, improvement_values = [], []
            metric_deltas = []
            for definition in definitions:
                baseline_key = str(definition.get("baseline_key") or definition.get("key") or "")
                candidate_key = str(definition.get("candidate_key") or definition.get("key") or "")
                current = (detail.get("baseline") or {}).get(baseline_key)
                improvement = (detail.get("candidate") or {}).get(candidate_key)
                if current is None or improvement is None:
                    continue
                current_value, improvement_value = float(current), float(improvement)
                current_values.append(current_value)
                improvement_values.append(improvement_value)
                metric_deltas.append({
                    "name": str(definition.get("name") or baseline_key),
                    "metric": str(definition.get("label") or definition.get("name") or baseline_key),
                    "current": round(current_value, 6),
                    "improvement": round(improvement_value, 6),
                    "delta": round(improvement_value - current_value, 6),
                    "baseline_k": definition.get("baseline_k", definition.get("k")),
                    "candidate_k": definition.get("candidate_k", definition.get("k")),
                })
            current_score = sum(current_values) / len(current_values) if current_values else 0.0
            improvement_score = sum(improvement_values) / len(improvement_values) if improvement_values else 0.0
            delta = improvement_score - current_score
            verdict = "improved" if delta > tolerance else "degraded" if delta < -tolerance else "unchanged"
            domain = _clean(detail.get("domain")) or "미분류"
            counts = domain_counts.setdefault(domain, {"total": 0, "improved": 0, "degraded": 0, "unchanged": 0})
            counts["total"] += 1
            counts[verdict] += 1
            rows.append({
                "question_id": _clean(detail.get("question_id")),
                "question": _clean(detail.get("question")),
                "domain": domain,
                "verdict": verdict,
                "score_delta": round(delta, 6),
                "current_score": round(current_score, 6),
                "improvement_score": round(improvement_score, 6),
                "metric_deltas": metric_deltas,
                "gold_chunk_ids": list(detail.get("gold_chunk_ids") or []),
                "current_top": list(detail.get("baseline_retrieved") or [])[:5],
                "improvement_top": list(detail.get("candidate_retrieved") or [])[:5],
            })
        total = len(rows)
        counts = {name: sum(row["verdict"] == name for row in rows) for name in ("improved", "degraded", "unchanged")}
        metric_changes = []
        for definition in definitions:
            baseline_key = str(definition.get("baseline_key") or definition.get("key") or "")
            candidate_key = str(definition.get("candidate_key") or definition.get("key") or "")
            baseline_k = definition.get("baseline_k", definition.get("k"))
            candidate_k = definition.get("candidate_k", definition.get("k"))
            metric_changes.append({
                "metric": f"{definition.get('label')} (현재 @{baseline_k} / 개선 @{candidate_k})",
                "current": (result.get("baseline") or {}).get(baseline_key),
                "improvement": (result.get("candidate") or {}).get(candidate_key),
                "delta": (result.get("delta") or {}).get(definition.get("name")),
            })
        latency_delta = (result.get("delta") or {}).get("latency_avg_ms")
        current_overall_score = sum(float(row["current_score"]) for row in rows) / total if total else 0.0
        improvement_overall_score = sum(float(row["improvement_score"]) for row in rows) / total if total else 0.0
        adjustment_hints = {
            "hit": {
                "group": "정답 포함 여부",
                "down": ["후보 깊이를 늘려 Gold 후보 누락 여부 확인", "최종 Top-K를 단계적으로 늘려 재평가"],
                "parameters": ["candidate_depth", "final_top_k", "min_relevance_score"],
            },
            "recall": {
                "group": "정답 범위 회수",
                "down": ["후보 깊이 또는 최종 Top-K를 늘려 누락 Gold 청크 확인", "임계치가 높다면 소폭 낮춰 재평가"],
                "parameters": ["candidate_depth", "final_top_k", "min_relevance_score"],
            },
            "mrr": {
                "group": "첫 정답 순위",
                "down": ["Dense·BM25 가중치를 한 번에 하나씩 조정", "Reranker 후보 깊이를 늘려 정답 재정렬 여부 확인"],
                "parameters": ["dense_weight", "bm25_weight", "candidate_depth"],
            },
            "map": {
                "group": "복수 정답 순위",
                "down": ["Dense·BM25 가중치를 재조정", "Reranker 후보 깊이와 정답 청크 순서를 질문별로 확인"],
                "parameters": ["dense_weight", "bm25_weight", "candidate_depth"],
            },
            "complete": {
                "group": "다중 청크 완전 회수",
                "down": ["다중 청크 문항에서 후보 깊이·최종 Top-K 확대", "Parent-Child 확장이 검색 Child 순위를 가리지 않는지 확인"],
                "parameters": ["candidate_depth", "final_top_k", "parent_child"],
            },
            "ndcg": {
                "group": "상위 순위 품질",
                "down": ["Reranker 입력 후보와 Dense·BM25 가중치를 함께 점검", "Gold 청크가 상위권에서 밀린 대표 문항 확인"],
                "parameters": ["dense_weight", "bm25_weight", "candidate_depth"],
            },
            "precision": {
                "group": "검색 결과 순도",
                "down": ["최종 Top-K를 낮추거나 최소 관련성 점수를 높여 불필요 청크 축소", "Recall 손실이 없는 범위에서 한 단계씩 조정"],
                "parameters": ["final_top_k", "min_relevance_score"],
            },
            "f1": {
                "group": "정밀도·재현율 균형",
                "down": ["Top-K와 최소 관련성 점수를 함께 보지 말고 한 항목씩 조정", "Precision과 Recall 중 어느 쪽이 F1을 끌어내렸는지 먼저 확인"],
                "parameters": ["final_top_k", "min_relevance_score", "candidate_depth"],
            },
        }
        metric_contributions = []
        for definition in definitions:
            name = str(definition.get("name") or "")
            per_question = [
                metric
                for row in rows
                for metric in row["metric_deltas"]
                if metric.get("name") == name
            ]
            if not per_question:
                continue
            # 질문마다 적용 가능한 지표 수가 다를 수 있으므로 delta/n을 먼저 계산한다.
            # 이 기여도를 전체 질문에 대해 더하면 아래 overall delta와 정확히 일치한다.
            score_contribution = sum(
                next(
                    (
                        float(metric["delta"]) / len(row["metric_deltas"])
                        for metric in row["metric_deltas"]
                        if metric.get("name") == name
                    ),
                    0.0,
                )
                for row in rows
            ) / total if total else 0.0
            mean_delta = sum(float(metric["delta"]) for metric in per_question) / len(per_question)
            hint = adjustment_hints.get(name, {"group": "검색 품질", "down": [], "parameters": []})
            metric_contributions.append({
                "name": name,
                "metric": str(definition.get("label") or name),
                "group": hint["group"],
                "applicable_questions": len(per_question),
                "mean_delta": round(mean_delta, 6),
                "score_contribution": round(score_contribution, 6),
                "positive_questions": sum(float(metric["delta"]) > tolerance for metric in per_question),
                "negative_questions": sum(float(metric["delta"]) < -tolerance for metric in per_question),
                "unchanged_questions": sum(abs(float(metric["delta"])) <= tolerance for metric in per_question),
                "adjustment_hints": list(hint["down"]),
                "target_parameters": list(hint["parameters"]),
            })
        return {
            "question_count": total,
            "classification_rule": "질문별 적용 가능 검색지표를 동일 가중치로 평균하고, 현재안 대비 ±0.5%p를 넘으면 개선/악화, 이하는 동일",
            "score_method": {
                "name": "질문별 검색 품질 종합점수",
                "formula": "질문별 종합점수 = 적용 가능한 (Hit + Recall + MRR + MAP + Complete + nDCG + Precision + F1) / 적용 지표 수",
                "weighting": "모든 적용 지표 동일 가중치(1/n). Complete는 다중 청크 평가 대상 문항에만 포함",
                "threshold": tolerance,
                "current": round(current_overall_score, 6),
                "improvement": round(improvement_overall_score, 6),
                "delta": round(improvement_overall_score - current_overall_score, 6),
                "contribution_check": round(sum(float(row["score_contribution"]) for row in metric_contributions), 6),
            },
            "metric_contributions": metric_contributions,
            "counts": counts,
            "rates": {name: (counts[name] / total if total else 0.0) for name in counts},
            "metric_changes": metric_changes,
            "latency": {
                "current_ms": (result.get("baseline") or {}).get("latency_avg_ms"),
                "improvement_ms": (result.get("candidate") or {}).get("latency_avg_ms"),
                "delta_ms": latency_delta,
            },
            "config_changes": {
                key: {"current": value, "improvement": (result.get("candidate_config") or {}).get(key)}
                for key, value in (result.get("baseline_config") or {}).items()
                if (result.get("candidate_config") or {}).get(key) != value
            },
            "domains": domain_counts,
            "representative_improvements": sorted(
                (row for row in rows if row["verdict"] == "improved"),
                key=lambda row: row["score_delta"], reverse=True,
            )[:5],
            "representative_degradations": sorted(
                (row for row in rows if row["verdict"] == "degraded"),
                key=lambda row: row["score_delta"],
            )[:5],
        }

    def generate_llm_evaluation_report(job_id: str, result: Mapping[str, Any]) -> dict[str, Any]:
        summary = evaluation_change_summary(result)
        client = (
            runtime_globals.get("ANSWER_HCX_CLIENT")
            or runtime_globals.get("HCX_CLIENT")
            or runtime_globals.get("_HCX_RAW_CLIENT_V3")
        )
        if client is None:
            raise RuntimeError("HCX 답변 클라이언트가 연결되지 않았습니다.")
        prompt_data = {
            "evaluation_scope": result.get("evaluation_scope"),
            "dataset": result.get("dataset"),
            "summary": summary,
        }
        system_prompt = (
            "당신은 공공기관 RAG 검색 품질 검토자입니다. 제공된 수치만 사용하고 추측하지 마세요. "
            "전체 테스트셋 집계는 이미 코드로 계산되었으므로 수치를 다시 만들지 말고 해석하세요. "
            "score_method의 산식과 metric_contributions를 근거로 어떤 지표가 종합점수를 얼마나 올리거나 내렸는지 설명하세요. "
            "config_changes와 지표별 adjustment_hints를 연결하되 인과관계가 확정되지 않았으면 '가설'이라고 표시하세요. "
            "반드시 한국어로 작성하며, 1) 결론 2) 종합점수 계산 근거 3) 점수를 올린 지표와 내린 지표 "
            "4) 대표 개선·악화 사례 5) 다음 A/B 테스트에서 한 번에 하나씩 바꿀 파라미터 6) 적용 전 확인사항 순서로 작성하세요. "
            "권장안에는 대상 파라미터, 조정 방향, 확인할 지표를 함께 적고 성능과 지연시간의 교환관계도 판단하세요. "
            "근거가 부족하면 명시하고, 제공되지 않은 수치나 원인을 만들지 마세요."
        )
        started = time.perf_counter()
        response = client.chat.completions.create(
            model=runtime_globals["HCX_CHAT_MODEL"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(prompt_data, ensure_ascii=False)},
            ],
            temperature=0.1,
            max_tokens=1800,
        )
        if not response.choices:
            raise RuntimeError("LLM 리포트 응답이 비어 있습니다.")
        usage = getattr(response, "usage", None)
        report = {
            "job_id": job_id,
            "generated_at": time.time(),
            "model": str(runtime_globals["HCX_CHAT_MODEL"]),
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "usage": {
                "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
            },
            "summary": summary,
            "report_markdown": str(response.choices[0].message.content or "").strip(),
        }
        if not report["report_markdown"]:
            raise RuntimeError("LLM 리포트 본문이 비어 있습니다.")
        return report

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
        prompt_snapshot = None
        if prompt_manager is not None:
            prompt_public = prompt_manager.public()
            prompt_snapshot = {"version": prompt_public["active_version"], "values": prompt_manager.active_values()}
        cache_snapshot = None
        if suggestion_cache is not None and callable(getattr(suggestion_cache, "snapshot_catalog", None)):
            cache_snapshot = suggestion_cache.snapshot_catalog()
        return {"config": active_config(), "chunks": copy.deepcopy(list(runtime_globals["CHUNKS"])),
                "prompts": prompt_snapshot,
                "suggestion_catalog": cache_snapshot,
                "vectors": {cid: np.asarray(vec, dtype=np.float32).copy() for cid, vec in runtime_globals["DENSE_VECTOR_BY_ID"].items()}}

    def load_default_snapshot() -> dict[str, Any]:
        """Load the immutable operating baseline, falling back to the first pre-apply snapshot."""
        try:
            loaded = pickle.loads(default_snapshot_path.read_bytes())
            if isinstance(loaded, Mapping) and isinstance(loaded.get("chunks"), Sequence) and isinstance(loaded.get("vectors"), Mapping):
                return copy.deepcopy(dict(loaded))
        except (FileNotFoundError, OSError, pickle.PickleError, EOFError):
            pass
        # Existing installations already have a first history snapshot. It is
        # the true baseline before their first administrator change, so retain
        # it instead of capturing a possibly modified current runtime.
        fallback = history[0]["snapshot"] if history else snapshot()
        default_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        temp = default_snapshot_path.with_suffix(default_snapshot_path.suffix + ".tmp")
        temp.write_bytes(pickle.dumps(fallback, protocol=pickle.HIGHEST_PROTOCOL))
        os.replace(temp, default_snapshot_path)
        return copy.deepcopy(fallback)

    default_snapshot = load_default_snapshot()

    def set_config(config: Mapping[str, Any]) -> None:
        mapping = {"DENSE_WEIGHT": "dense_weight", "BM25_WEIGHT": "bm25_weight", "CANDIDATE_DEPTH": "candidate_depth",
                   "FINAL_TOP_K": "final_top_k", "QUERY_FUSION_RRF_K": "query_fusion_rrf_k",
                   "MIN_RELEVANCE_SCORE": "min_relevance_score",
                   "PARENT_CHILD_ENABLED": "parent_child", "PARENT_CONTEXT_MAX_CHARS": "parent_context_max_chars"}
        fallback = active_config()
        for target, source in mapping.items():
            runtime_globals[target] = config.get(source, fallback[source])
        runtime_globals["RERANKER_CANDIDATE_DEPTH"] = int(config["candidate_depth"])
        runtime_globals["DENSE_KNN_NUM_CANDIDATES"] = max(int(runtime_globals.get("DENSE_KNN_NUM_CANDIDATES", 100)), int(config["candidate_depth"]))
        fuse = runtime_globals.get("fuse_query_results")
        if callable(fuse):
            kw = dict(getattr(fuse, "__kwdefaults__", {}) or {})
            kw.update(top_k=int(config["final_top_k"]), rrf_k=int(config["query_fusion_rrf_k"]))
            fuse.__kwdefaults__ = kw

    def apply_snapshot(target: Mapping[str, Any], *, archive_prompts: bool = False) -> None:
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
        prompt_target = target.get("prompts")
        if prompt_manager is not None and isinstance(prompt_target, Mapping):
            prompt_manager.activate(
                prompt_target.get("values") or {},
                version=str(prompt_target.get("version") or "prompt-restored"),
                archive_current=archive_prompts,
            )
        cache_target = target.get("suggestion_catalog")
        if cache_target is not None:
            manager = require_suggestion_cache()
            restore = getattr(manager, "restore_catalog", None)
            if not callable(restore):
                raise RuntimeError("질의 캐시 롤백 기능이 연결되지 않았습니다.")
            restore(copy.deepcopy(list(cache_target)), actor="ADMIN_ROLLBACK")

    def apply_suggestion_cache_drafts() -> None:
        manager = require_suggestion_cache()
        for suggestion_id, change in copy.deepcopy(drafts["suggestion_cache"]).items():
            rows = {
                row["suggestion_id"]: row
                for row in manager.list_catalog(include_disabled=True)
            }
            metadata = dict(change.get("metadata") or {})
            if change.get("create"):
                if suggestion_id in rows:
                    raise ValueError(f"이미 존재하는 질의 캐시 ID입니다: {suggestion_id}")
                created = manager.create_suggestion(
                    suggestion_id=suggestion_id,
                    business_key=metadata["business_key"],
                    business=metadata["business"],
                    keyword=metadata["keyword"],
                    canonical_question=metadata["canonical_question"],
                    display_order=None,
                    actor="ADMIN_APPLY",
                )
                requested_order = int(metadata.get("display_order") or created["display_order"])
                if requested_order != int(created["display_order"]):
                    manager.update_suggestion(
                        suggestion_id,
                        business_key=metadata["business_key"],
                        business=metadata["business"],
                        keyword=metadata["keyword"],
                        canonical_question=metadata["canonical_question"],
                        display_order=requested_order,
                        row_version=int(created["row_version"]),
                        actor="ADMIN_APPLY",
                    )
                rows = {
                    row["suggestion_id"]: row
                    for row in manager.list_catalog(include_disabled=True)
                }
            elif metadata:
                row = rows.get(suggestion_id)
                if row is None:
                    raise KeyError(suggestion_id)
                expected = change.get("expected_row_version")
                if expected is not None and int(row["row_version"]) != int(expected):
                    raise RuntimeError(
                        f"{suggestion_id} 항목이 초안 저장 후 변경되었습니다. "
                        "질의 캐시 목록을 다시 읽고 초안을 확인해 주세요."
                    )
                manager.update_suggestion(
                    suggestion_id,
                    business_key=metadata["business_key"],
                    business=metadata["business"],
                    keyword=metadata["keyword"],
                    canonical_question=metadata["canonical_question"],
                    display_order=int(metadata["display_order"]),
                    row_version=int(row["row_version"]),
                    actor="ADMIN_APPLY",
                )
            if "enabled" in change:
                manager.set_enabled(
                    suggestion_id,
                    bool(change["enabled"]),
                    actor="ADMIN_APPLY",
                )
            answer_change = dict(change.get("answer_change") or {})
            action = answer_change.get("action")
            if action == "save":
                manager.save_manual_answer(
                    suggestion_id,
                    answer=answer_change["answer"],
                    sources=answer_change["sources"],
                    actor="ADMIN_APPLY",
                )
            elif action == "clear":
                manager.clear_active(suggestion_id, actor="ADMIN_APPLY")
            elif action == "activate":
                manager.activate(
                    suggestion_id,
                    answer_change["cache_key"],
                    actor="ADMIN_APPLY",
                )

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
            token = secrets.token_urlsafe(48); sessions[token] = time.time() + SESSION_TTL_SECONDS; save_admin_sessions(); bootstrap_state["used"] = True
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
            save_admin_sessions()
        response = JSONResponse({"authenticated": True, "expires_in_seconds": SESSION_TTL_SECONDS})
        response.set_cookie(COOKIE_NAME, token, max_age=SESSION_TTL_SECONDS, httponly=True, secure=True, samesite="strict", path="/")
        return response

    @app.get("/admin", include_in_schema=False)
    def admin_page(_: None = Depends(require_admin_session)):
        if not page.is_file(): raise HTTPException(status_code=503, detail="관리자 HTML 파일을 찾지 못했습니다.")
        # 페이지에 inline JavaScript가 포함되어 있어 구버전 문서 캐시는 UI와
        # API 응답의 스키마를 어긋나게 할 수 있습니다. 관리자 화면은 항상 최신
        # 배포본을 받게 합니다.
        return FileResponse(
            page,
            media_type="text/html; charset=utf-8",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    @app.post("/api/admin-ui/logout", include_in_schema=False)
    def logout(request: Request, _: None = Depends(require_admin_session)):
        with auth_lock:
            sessions.pop(_clean(request.cookies.get(COOKIE_NAME)), None)
            save_admin_sessions()
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
    def jobs(limit: int = Query(default=25, ge=1, le=100), offset: int = Query(default=0, ge=0),
             hours: float = Query(default=24, ge=.25, le=720), _: None = Depends(require_admin_session)):
        cutoff = time.time() - (hours * 3600)
        try:
            items = service_module.JOB_STORE.list_public(limit, offset=offset, since=cutoff)
            total = int(service_module.JOB_STORE.count_public(since=cutoff))
        except TypeError:
            all_items = [row for row in service_module.JOB_STORE.list_public(500) if float(row.get("created_at") or 0) >= cutoff]
            items, total = all_items[offset:offset + limit], len(all_items)
        return {
            "stats": service_module.JOB_STORE.stats(),
            "items": [job_public_with_telemetry(item) for item in items if isinstance(item, Mapping)],
            "total": total,
            "offset": offset,
            "limit": limit,
            "hours": hours,
        }

    @app.get("/api/admin-ui/indices")
    def indices(_: None = Depends(require_admin_session)): return service_module.admin_indices()

    @app.post("/api/admin-ui/search")
    def search(payload: AdminSearchPayload, _: None = Depends(require_admin_session)):
        return service_module.admin_search(service_module.AdminSearchRequest(**payload.model_dump()))

    @app.get("/api/admin-ui/documents/{index}/{document_id:path}")
    def document(index: str, document_id: str, _: None = Depends(require_admin_session)):
        base = service_module.admin_document(index, document_id)
        return {**base, "chunk_metadata": copy.deepcopy(runtime_globals.get("CHUNKS_BY_ID", {}).get(str(document_id)))}

    @app.get("/api/admin-ui/chunk-sources")
    def list_chunk_sources(_: None = Depends(require_admin_session)):
        items = chunk_source_catalog()
        return {"items": items, "count": len(items), "business_catalog": new_url_business_catalog()}

    @app.post("/api/admin-ui/chunk-sources/rechunk")
    def start_rechunk_preview(payload: RechunkPreviewPayload, _: None = Depends(require_admin_session)):
        source_url = _clean(payload.source_url)
        source = next((row for row in chunk_source_catalog() if row["source_url"] == source_url), None)
        if source is None:
            raise HTTPException(status_code=404, detail="현재 운영 청크에서 해당 URL을 찾지 못했습니다. URL이 없는 기존 청크는 재청킹할 수 없습니다.")
        if not source["document_ids"]:
            raise HTTPException(status_code=422, detail="해당 URL의 document_id가 없어 도메인별 청크 ID를 재발급할 수 없습니다.")
        with rechunk_lock:
            if any(row.get("source_url") == source_url and row.get("status") in {"queued", "running"}
                   for row in rechunk_jobs.values()):
                raise HTTPException(status_code=409, detail="이 URL은 이미 재수집·재청킹 중입니다.")
            job_id = f"rechunk-{uuid.uuid4().hex[:12]}"
            rechunk_jobs[job_id] = {
                "job_id": job_id, "mode": "refresh", "source_url": source_url,
                "status": "queued", "stage": "재수집 대기", "stage_detail": "작업 실행을 기다리고 있습니다.",
                "progress": 0,
                "error": None, "created_at": time.time(), "updated_at": time.time(),
                "old_chunk_count": int(source["chunk_count"]), "new_chunk_count": 0,
                "document_ids": list(source["document_ids"]),
                "business_functions": list(source["business_functions"]),
                "preview_chunks": [], "review": {}, "staged": False,
                # This fingerprint protects against removing a newer runtime
                # version while an administrator is reviewing the preview.
                "old_chunk_ids": sorted(str(chunk.get("chunk_id")) for chunk in source_chunks(source_url)
                                        if chunk.get("chunk_id")),
                "old_chunks_snapshot": source_chunks(source_url),
            }
        rechunk_executor.submit(run_rechunk_preview, job_id)
        return rechunk_public(rechunk_jobs[job_id])

    @app.post("/api/admin-ui/chunk-sources/crawl")
    def start_new_url_preview(payload: NewUrlCrawlPayload, _: None = Depends(require_admin_session)):
        source_url = _clean(payload.source_url)
        if not _is_public_kdic_url(source_url):
            raise HTTPException(status_code=422, detail="신규 URL은 예금보험공사 공개 도메인(kdic.or.kr) 주소만 등록할 수 있습니다.")
        document_id = _clean(payload.document_id).upper()
        if not re.fullmatch(r"[A-Z][A-Z0-9]{1,7}-\d{3,6}", document_id):
            raise HTTPException(status_code=422, detail="문서 ID는 도메인 코드-숫자 형식으로 입력해 주세요. 예: DP-007")
        if any(row["source_url"] == source_url for row in chunk_source_catalog()):
            raise HTTPException(status_code=409, detail="이미 운영 중인 URL입니다. ‘기존 URL 갱신’에서 재수집해 주세요.")
        existing_document_ids = {
            _clean(chunk.get("document_id")) for chunk in runtime_globals.get("CHUNKS", [])
            if isinstance(chunk, Mapping) and _clean(chunk.get("document_id"))
        }
        if document_id in existing_document_ids:
            raise HTTPException(status_code=409, detail=f"이미 사용 중인 문서 ID입니다: {document_id}")
        business = _clean(payload.business_function)
        title = _clean(payload.title)
        new_document = {
            "document_id": document_id,
            "parent_doc_id": document_id,
            "title": title,
            "business_function": business,
            "target_business_function": _clean(payload.target_business_function) or business,
            "page_type": _clean(payload.page_type) or "static_page",
            "source_url": source_url,
        }
        with rechunk_lock:
            if any(row.get("source_url") == source_url and row.get("status") in {"queued", "running"}
                   for row in rechunk_jobs.values()):
                raise HTTPException(status_code=409, detail="이 URL은 이미 신규 수집 중입니다.")
            job_id = f"crawl-{uuid.uuid4().hex[:12]}"
            rechunk_jobs[job_id] = {
                "job_id": job_id, "mode": "new", "source_url": source_url,
                "status": "queued", "stage": "신규 수집 대기", "stage_detail": "작업 실행을 기다리고 있습니다.",
                "progress": 0, "error": None, "created_at": time.time(), "updated_at": time.time(),
                "old_chunk_count": 0, "new_chunk_count": 0,
                "document_ids": [document_id], "business_functions": [business],
                "preview_chunks": [], "review": {}, "staged": False,
                "old_chunk_ids": [], "old_chunks_snapshot": [], "new_document": new_document,
            }
        rechunk_executor.submit(run_rechunk_preview, job_id)
        return rechunk_public(rechunk_jobs[job_id])

    @app.get("/api/admin-ui/chunk-sources/rechunk/{job_id}")
    def get_rechunk_preview(job_id: str, _: None = Depends(require_admin_session)):
        with rechunk_lock:
            job = rechunk_jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="재청킹 작업을 찾지 못했습니다. 서버 재시작 후에는 새로 재수집해 주세요.")
            return rechunk_public(job, include_preview=job.get("status") == "ready")

    @app.post("/api/admin-ui/chunk-sources/rechunk/{job_id}/stage")
    def stage_rechunk_preview(job_id: str, _: None = Depends(require_admin_session)):
        with rechunk_lock:
            job = copy.deepcopy(rechunk_jobs.get(job_id))
        if job is None:
            raise HTTPException(status_code=404, detail="재청킹 작업을 찾지 못했습니다.")
        if job.get("status") != "ready":
            raise HTTPException(status_code=409, detail="재청킹 미리보기가 준비된 뒤에만 교체 초안을 만들 수 있습니다.")
        new_chunks = list(job.get("preview_chunks") or [])
        if not new_chunks:
            raise HTTPException(status_code=422, detail="교체할 새 청크가 없습니다.")
        review = job.get("review") if isinstance(job.get("review"), Mapping) else {}
        if not review.get("can_stage"):
            errors = "; ".join(map(str, list(review.get("errors") or [])[:3]))
            raise HTTPException(status_code=422, detail=f"검수 오류가 있어 교체 초안을 만들 수 없습니다. {errors}")
        source_url = str(job["source_url"])
        mode = str(job.get("mode") or "refresh")
        with mutation_lock:
            current_ids = {str(row.get("chunk_id")) for row in source_chunks(source_url) if row.get("chunk_id")}
            if mode == "refresh" and not current_ids:
                raise HTTPException(status_code=409, detail="현재 운영 청크가 바뀌었습니다. URL 목록을 새로고침하고 다시 재청킹해 주세요.")
            if mode == "refresh" and current_ids != set(map(str, job.get("old_chunk_ids") or [])):
                raise HTTPException(status_code=409, detail="미리보기 이후 운영 청크가 변경되었습니다. 이전 결과를 적용하지 않고 다시 재청킹해 주세요.")
            if mode == "new" and current_ids:
                raise HTTPException(status_code=409, detail="미리보기 이후 같은 URL이 운영에 추가되었습니다. 신규 수집을 다시 시작해 주세요.")
            new_ids = {str(row.get("chunk_id")) for row in new_chunks}
            all_ids = set(map(str, runtime_globals.get("CHUNKS_BY_ID", {})))
            conflicts = sorted((new_ids & (all_ids - current_ids)) | (new_ids & (set(drafts["add"]) - current_ids)))
            if conflicts:
                raise HTTPException(status_code=409, detail=f"다른 운영 문서와 청크 ID가 충돌합니다: {', '.join(conflicts[:5])}")
            if mode == "refresh":
                drafts["remove"].update(current_ids)
            for chunk in new_chunks:
                drafts["add"][str(chunk["chunk_id"])] = copy.deepcopy(chunk)
            drafts["rechunk"] = [row for row in drafts["rechunk"] if row.get("source_url") != source_url] + [{
                "job_id": job_id, "source_url": source_url, "document_ids": list(job["document_ids"]),
                "mode": mode, "old_chunk_count": len(current_ids), "new_chunk_count": len(new_chunks), "status": "staged",
            }]
        with rechunk_lock:
            if job_id in rechunk_jobs:
                rechunk_jobs[job_id].update(staged=True, updated_at=time.time())
        return {"staged": True, "job_id": job_id, "old_chunk_count": len(current_ids),
                "new_chunk_count": len(new_chunks), "draft": draft_public()}

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
        with eval_lock:
            datasets[dataset_id] = record
            save_evaluation_state()
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
            save_evaluation_state()
        return {"deleted": True, "dataset_id": dataset_id, "removed_jobs": len(removed_jobs)}

    @app.delete("/api/admin-ui/evaluations/datasets")
    def clear_datasets(_: None = Depends(require_admin_session)):
        with eval_lock:
            if any(job.get("status") == "running" for job in eval_jobs.values()):
                raise HTTPException(status_code=409, detail="실행 중인 평가가 있어 데이터셋을 비울 수 없습니다.")
            count = len(datasets)
            datasets.clear()
            eval_jobs.clear()
            save_evaluation_state()
        return {"cleared": True, "dataset_count": count}

    @app.post("/api/admin-ui/evaluations/run")
    def start_eval(payload: EvaluationRunPayload, _: None = Depends(require_admin_session)):
        if payload.dataset_id not in datasets: raise HTTPException(status_code=404, detail="평가데이터셋을 찾지 못했습니다.")
        try:
            baseline = active_config()
            candidate = _validate_config(payload.candidate, baseline)
            baseline_metric_ks = payload.baseline_metric_ks or payload.metric_ks
            candidate_metric_ks = payload.candidate_metric_ks or payload.metric_ks
            policy = _validate_evaluation_policy(
                payload.evaluation_depth,
                baseline_metric_ks,
                candidate_metric_ks,
                payload.curve_ks,
                baseline,
                candidate,
            )
        except Exception as error: raise HTTPException(status_code=422, detail=str(error)) from error
        job_id = f"eval-{uuid.uuid4().hex[:12]}"
        with eval_lock:
            eval_jobs[job_id] = {"job_id": job_id, "status": "running", "progress": 0, "processed": 0,
                                 "dataset_id": payload.dataset_id, "dataset_filename": datasets[payload.dataset_id]["filename"],
                                 "evaluation_policy": policy, "created_at": time.time(), "updated_at": time.time(),
                                 "error": None, "result": None}
            save_evaluation_state()
        executor.submit(run_evaluation, job_id, payload.dataset_id, payload.candidate, payload.max_questions,
                        payload.evaluation_depth, baseline_metric_ks, candidate_metric_ks, payload.curve_ks)
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

    @app.post("/api/admin-ui/evaluations/jobs/{job_id}/llm-report")
    def create_eval_llm_report(job_id: str, _: None = Depends(require_admin_session)):
        if not report_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="다른 LLM 품질 리포트를 생성하고 있습니다. 잠시 후 다시 시도해 주세요.")
        try:
            with eval_lock:
                job = copy.deepcopy(eval_jobs.get(job_id))
            if not job:
                raise HTTPException(status_code=404, detail="평가 작업을 찾지 못했습니다.")
            if job.get("status") != "done" or not job.get("result"):
                raise HTTPException(status_code=409, detail="완료된 검색 평가에서만 LLM 품질 리포트를 만들 수 있습니다.")
            try:
                report = generate_llm_evaluation_report(job_id, job["result"])
            except HTTPException:
                raise
            except Exception as error:
                raise HTTPException(status_code=502, detail=f"LLM 리포트 생성 실패: {type(error).__name__}: {error}") from error
            with eval_lock:
                current = eval_jobs.get(job_id)
                if current and current.get("result"):
                    current["result"]["llm_report"] = copy.deepcopy(report)
                    current["updated_at"] = time.time()
                    save_evaluation_state()
            return report
        finally:
            report_lock.release()

    @app.delete("/api/admin-ui/evaluations/jobs/{job_id}")
    def delete_eval_job(job_id: str, _: None = Depends(require_admin_session)):
        with eval_lock:
            job = eval_jobs.get(job_id)
            if not job:
                raise HTTPException(status_code=404, detail="평가 작업을 찾지 못했습니다.")
            if job.get("status") == "running":
                raise HTTPException(status_code=409, detail="실행 중인 평가는 삭제할 수 없습니다.")
            eval_jobs.pop(job_id, None)
            save_evaluation_state()
        return {"deleted": True, "job_id": job_id}

    @app.delete("/api/admin-ui/evaluations/jobs")
    def clear_eval_jobs(_: None = Depends(require_admin_session)):
        with eval_lock:
            removable = [job_id for job_id, job in eval_jobs.items() if job.get("status") != "running"]
            for job_id in removable:
                eval_jobs.pop(job_id, None)
            save_evaluation_state()
        return {"cleared": True, "removed_count": len(removable)}

    @app.get("/api/admin-ui/parameter-presets")
    def list_parameter_presets(_: None = Depends(require_admin_session)):
        items = sorted(parameter_presets.values(), key=lambda row: float(row.get("updated_at") or 0), reverse=True)
        return {"items": [parameter_preset_public(row) for row in items], "limit": MAX_PARAMETER_PRESETS}

    @app.post("/api/admin-ui/parameter-presets")
    def create_parameter_preset(payload: ParameterPresetPayload, _: None = Depends(require_admin_session)):
        name = _clean(payload.name)
        if not name:
            raise HTTPException(status_code=422, detail="파라미터 설정 이름을 입력해 주세요.")
        with mutation_lock:
            duplicate = next((row for row in parameter_presets.values() if _clean(row.get("name")) == name), None)
            if duplicate is not None:
                raise HTTPException(status_code=409, detail="같은 이름의 파라미터 설정이 이미 있습니다. 다른 이름을 사용해 주세요.")
            if len(parameter_presets) >= MAX_PARAMETER_PRESETS:
                raise HTTPException(status_code=409, detail=f"파라미터 설정은 최대 {MAX_PARAMETER_PRESETS}개까지 저장할 수 있습니다.")
            try:
                values = _validate_config(payload.values, active_config())
            except Exception as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
            now = time.time()
            row = {"preset_id": f"preset-{uuid.uuid4().hex[:12]}", "name": name, "values": values,
                   "created_at": now, "updated_at": now}
            parameter_presets[row["preset_id"]] = row
            save_parameter_presets()
        return parameter_preset_public(row)

    @app.delete("/api/admin-ui/parameter-presets/{preset_id}")
    def delete_parameter_preset(preset_id: str, _: None = Depends(require_admin_session)):
        with mutation_lock:
            row = parameter_presets.pop(_clean(preset_id), None)
            if row is None:
                raise HTTPException(status_code=404, detail="파라미터 설정을 찾지 못했습니다.")
            save_parameter_presets()
        return {"deleted": True, "preset_id": str(row["preset_id"])}

    @app.post("/api/admin-ui/parameter-presets/{preset_id}/load-draft")
    def load_parameter_preset_to_draft(preset_id: str, _: None = Depends(require_admin_session)):
        row = parameter_presets.get(_clean(preset_id))
        if row is None:
            raise HTTPException(status_code=404, detail="파라미터 설정을 찾지 못했습니다.")
        with mutation_lock:
            drafts["config"] = copy.deepcopy(dict(row["values"]))
        return {"preset": parameter_preset_public(row), "draft": draft_public()}

    @app.get("/api/admin-ui/prompt-presets")
    def list_prompt_presets(_: None = Depends(require_admin_session)):
        items = sorted(prompt_presets.values(), key=lambda row: float(row.get("updated_at") or 0), reverse=True)
        return {"items": [prompt_preset_public(row) for row in items], "limit": MAX_PROMPT_PRESETS}

    @app.post("/api/admin-ui/prompt-presets")
    def create_prompt_preset(payload: PromptPresetPayload, _: None = Depends(require_admin_session)):
        name = _clean(payload.name)
        if not name:
            raise HTTPException(status_code=422, detail="프롬프트 설정 이름을 입력해 주세요.")
        with mutation_lock:
            duplicate = next((row for row in prompt_presets.values() if _clean(row.get("name")) == name), None)
            if duplicate is not None:
                raise HTTPException(status_code=409, detail="같은 이름의 프롬프트 설정이 이미 있습니다. 다른 이름을 사용해 주세요.")
            if len(prompt_presets) >= MAX_PROMPT_PRESETS:
                raise HTTPException(status_code=409, detail=f"프롬프트 설정은 최대 {MAX_PROMPT_PRESETS}개까지 저장할 수 있습니다.")
            try:
                values = validate_prompt_preset_values(payload.values)
            except ValueError as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
            now = time.time()
            row = {
                "preset_id": f"prompt-preset-{uuid.uuid4().hex[:12]}",
                "name": name,
                "values": values,
                "created_at": now,
                "updated_at": now,
            }
            prompt_presets[row["preset_id"]] = row
            save_prompt_presets()
        return prompt_preset_public(row)

    @app.delete("/api/admin-ui/prompt-presets/{preset_id}")
    def delete_prompt_preset(preset_id: str, _: None = Depends(require_admin_session)):
        with mutation_lock:
            row = prompt_presets.pop(_clean(preset_id), None)
            if row is None:
                raise HTTPException(status_code=404, detail="프롬프트 설정을 찾지 못했습니다.")
            save_prompt_presets()
        return {"deleted": True, "preset_id": str(row["preset_id"])}

    @app.post("/api/admin-ui/prompt-presets/{preset_id}/load-draft")
    def load_prompt_preset_to_draft(preset_id: str, _: None = Depends(require_admin_session)):
        row = prompt_presets.get(_clean(preset_id))
        if row is None:
            raise HTTPException(status_code=404, detail="프롬프트 설정을 찾지 못했습니다.")
        try:
            prompt_state = require_prompt_manager().save_draft(row["values"])
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return {"preset": prompt_preset_public(row), "prompts": prompt_state, "draft": draft_public()}

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
        with mutation_lock, execution_lock:
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
        merged = {**dict(drafts["config"]), **dict(payload.values)}
        try: candidate = _validate_config(merged, active_config())
        except Exception as error: raise HTTPException(status_code=422, detail=str(error)) from error
        drafts["config"] = {key: candidate[key] for key in merged}
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
        drafts["config"].clear(); drafts["add"].clear(); drafts["remove"].clear(); drafts["rechunk"].clear(); drafts["suggestion_cache"].clear()
        if prompt_manager is not None:
            prompt_manager.clear_draft()
        return draft_public()

    @app.delete("/api/admin-ui/draft/suggestion-cache/{suggestion_id}")
    def clear_suggestion_cache_draft(suggestion_id: str, _: None = Depends(require_admin_session)):
        clean_id = _clean(suggestion_id).upper()
        if clean_id not in drafts["suggestion_cache"]:
            raise HTTPException(status_code=404, detail="질의 캐시 초안을 찾지 못했습니다.")
        drafts["suggestion_cache"].pop(clean_id, None)
        return {"cleared": True, "draft": draft_public(), "catalog": suggestion_public()}

    @app.post("/api/admin-ui/apply")
    def apply(payload: ApplyPayload, _: None = Depends(require_admin_session)):
        if payload.confirmation != "운영 반영": raise HTTPException(status_code=422, detail="확인 문구로 '운영 반영'을 입력해 주세요.")
        if not draft_public()["has_changes"]: raise HTTPException(status_code=409, detail="반영할 초안이 없습니다.")
        with mutation_lock, execution_lock:
            before = None
            try:
                no_chat_jobs_running(); before = snapshot()
                target_chunks = [copy.deepcopy(row) for row in runtime_globals["CHUNKS"] if str(row["chunk_id"]) not in drafts["remove"]]
                target_vectors = {cid: np.asarray(vec).copy() for cid, vec in runtime_globals["DENSE_VECTOR_BY_ID"].items() if cid not in drafts["remove"]}
                for cid, chunk in drafts["add"].items():
                    vector = runtime_globals["_normalize_vector"](runtime_globals["embed_hcx_single"](runtime_globals["build_dense_structured_v2_text"](chunk)))
                    if vector.shape != (int(runtime_globals["DENSE_DIMENSION"]),): raise ValueError(f"{cid} 임베딩 차원 오류: {vector.shape}")
                    target_chunks = [row for row in target_chunks if str(row["chunk_id"]) != cid] + [copy.deepcopy(chunk)]
                    target_vectors[cid] = vector
                version_id = f"ver-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
                prompt_changes = list(draft_public().get("prompt_changes") or [])
                prompt_target = None
                if prompt_manager is not None:
                    prompt_target = {"version": f"{version_id}-prompt", "values": prompt_manager.draft_values()}
                target = {"config": _validate_config(drafts["config"], active_config()), "chunks": target_chunks,
                          "vectors": target_vectors, "prompts": prompt_target}
                runtime_changes = bool(
                    drafts["config"] or drafts["add"] or drafts["remove"] or prompt_changes
                )
                if runtime_changes:
                    apply_snapshot(target, archive_prompts=bool(prompt_changes))
                cache_changes = copy.deepcopy(list(drafts["suggestion_cache"].values()))
                apply_suggestion_cache_drafts()
                history.append({"version_id": version_id, "created_at": time.time(), "snapshot": before, "active": True,
                                "summary": {"config": copy.deepcopy(drafts["config"]), "added": sorted(drafts["add"]),
                                            "removed": sorted(drafts["remove"]), "rechunk": copy.deepcopy(drafts["rechunk"]), "prompts": prompt_changes,
                                            "suggestion_cache": cache_changes,
                                            "label": "초기 운영 상태" if not history else "이전 운영 상태"}})
                history[:] = history[-10:]
                for row in history[:-1]: row["active"] = False
                drafts["config"].clear(); drafts["add"].clear(); drafts["remove"].clear(); drafts["rechunk"].clear(); drafts["suggestion_cache"].clear()
                save_history()
                return {"applied": True, "version_id": version_id, "chunk_count": len(target_chunks), "active_config": active_config()}
            except Exception as error:
                if before is not None:
                    try: apply_snapshot(before, archive_prompts=False)
                    except Exception: pass
                if isinstance(error, HTTPException): raise
                raise HTTPException(status_code=500, detail=f"반영 실패, 기존 상태 복구 시도 완료: {type(error).__name__}: {error}") from error

    @app.post("/api/admin-ui/reset-defaults")
    def reset_defaults(payload: ApplyPayload, _: None = Depends(require_admin_session)):
        if payload.confirmation != "운영 기본값 초기화":
            raise HTTPException(status_code=422, detail="확인 문구로 '운영 기본값 초기화'를 입력해 주세요.")
        with mutation_lock, execution_lock:
            before = None
            try:
                no_chat_jobs_running()
                before = snapshot()
                apply_snapshot(default_snapshot, archive_prompts=True)
                version_id = f"reset-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
                history.append({
                    "version_id": version_id,
                    "created_at": time.time(),
                    "snapshot": before,
                    "active": True,
                    "summary": {
                        "config": {}, "added": [], "removed": [], "rechunk": [], "prompts": [],
                        "label": "기본값 초기화 직전 상태",
                    },
                })
                history[:] = history[-10:]
                for row in history[:-1]:
                    row["active"] = False
                drafts["config"].clear(); drafts["add"].clear(); drafts["remove"].clear(); drafts["rechunk"].clear(); drafts["suggestion_cache"].clear()
                if prompt_manager is not None:
                    prompt_manager.clear_draft()
                save_history()
                return {
                    "reset": True,
                    "version_id": version_id,
                    "chunk_count": len(runtime_globals["CHUNKS"]),
                    "active_config": active_config(),
                }
            except Exception as error:
                if before is not None:
                    try: apply_snapshot(before, archive_prompts=False)
                    except Exception: pass
                if isinstance(error, HTTPException):
                    raise
                raise HTTPException(status_code=500, detail=f"운영 기본값 초기화 실패: {type(error).__name__}: {error}") from error

    @app.post("/api/admin-ui/rollback/{version_id}")
    def rollback(version_id: str, _: None = Depends(require_admin_session)):
        with mutation_lock, execution_lock:
            try:
                no_chat_jobs_running(); row = next((item for item in history if item["version_id"] == version_id), None)
                if not row: raise HTTPException(status_code=404, detail="롤백 버전을 찾지 못했습니다.")
                apply_snapshot(row["snapshot"], archive_prompts=True)
                for item in history: item["active"] = False
                save_history()
                return {"rolled_back": True, "version_id": version_id, "chunk_count": len(runtime_globals["CHUNKS"]), "active_config": active_config()}
            except HTTPException: raise
            except Exception as error: raise HTTPException(status_code=500, detail=f"롤백 실패: {type(error).__name__}: {error}") from error

    @app.delete("/api/admin-ui/history/{version_id}")
    def delete_history(version_id: str, _: None = Depends(require_admin_session)):
        index = next((idx for idx, item in enumerate(history) if item["version_id"] == version_id), None)
        if index is None:
            raise HTTPException(status_code=404, detail="반영 이력을 찾지 못했습니다.")
        history.pop(index)
        save_history()
        return {"deleted": True, "version_id": version_id, "remaining": len(history)}

    @app.delete("/api/admin-ui/history")
    def clear_history(_: None = Depends(require_admin_session)):
        count = len(history)
        history.clear()
        save_history()
        return {"cleared": True, "removed_count": count}

    @app.get("/api/admin-ui/pipeline/graph")
    def pipeline_graph(_: None = Depends(require_admin_session)):
        return _build_pipeline_graph(runtime_globals, pipeline_labels)

    @app.put("/api/admin-ui/pipeline/labels")
    def pipeline_label_update(payload: PipelineLabelsPayload, _: None = Depends(require_admin_session)):
        valid_ids = {str(row["id"]) for row in DEFAULT_PIPELINE_GRAPH_SPEC}
        updated: dict[str, str] = {}
        for node_id, value in payload.labels.items():
            if node_id not in valid_ids:
                raise HTTPException(status_code=422, detail=f"알 수 없는 파이프라인 블록입니다: {node_id}")
            label = _clean(value)
            if not label:
                pipeline_labels.pop(node_id, None)
                continue
            if len(label) > 80:
                raise HTTPException(status_code=422, detail="표시명은 80자 이하로 입력해 주세요.")
            pipeline_labels[node_id] = label
            updated[node_id] = label
        save_pipeline_labels()
        return {"updated": updated, "graph": _build_pipeline_graph(runtime_globals, pipeline_labels)}

    @app.delete("/api/admin-ui/pipeline/labels/{node_id}")
    def pipeline_label_reset(node_id: str, _: None = Depends(require_admin_session)):
        pipeline_labels.pop(_clean(node_id), None)
        save_pipeline_labels()
        return {"reset": True, "graph": _build_pipeline_graph(runtime_globals, pipeline_labels)}

    @app.get("/api/admin-ui/monitoring")
    def monitoring(hours: float = Query(default=24, ge=.25, le=720), limit: int = Query(default=30, ge=1, le=500),
                   offset: int = Query(default=0, ge=0),
                   _: None = Depends(require_admin_session)):
        return monitoring_snapshot(hours, limit, offset)

    @app.get("/api/admin-ui/suggestion-cache")
    def suggestion_cache_list(_: None = Depends(require_admin_session)):
        return suggestion_public()

    @app.post("/api/admin-ui/suggestion-cache")
    def suggestion_cache_create(payload: SuggestionCreatePayload, _: None = Depends(require_admin_session)):
        manager = require_suggestion_cache()
        try:
            suggestion_id = suggestion_id_for(payload)
            existing = {
                row["suggestion_id"]
                for row in manager.list_catalog(include_disabled=True)
            }
            if suggestion_id in existing or suggestion_id in drafts["suggestion_cache"]:
                raise ValueError("같은 업무·키워드·표준 질문의 항목이 이미 존재합니다.")
            stage_suggestion_change(
                suggestion_id,
                create=True,
                metadata={
                    "business_key": payload.business_key,
                    "business": payload.business,
                    "keyword": payload.keyword,
                    "canonical_question": payload.canonical_question,
                    "display_order": payload.display_order
                    or len(existing) + 1,
                },
                enabled=True,
            )
            catalog = suggestion_public()
            row = next(item for item in catalog["items"] if item["suggestion_id"] == suggestion_id)
            return {"staged": True, "created": True, "item": row, "catalog": catalog,
                    "draft": draft_public()}
        except Exception as error:
            raise HTTPException(status_code=422, detail=f"질의 캐시 추가 초안 실패: {error}") from error

    @app.put("/api/admin-ui/suggestion-cache/{suggestion_id}")
    def suggestion_cache_update(suggestion_id: str, payload: SuggestionUpdatePayload,
                                _: None = Depends(require_admin_session)):
        try:
            clean_id = require_known_suggestion(suggestion_id)
            stage_suggestion_change(
                clean_id,
                metadata={
                    "business_key": payload.business_key,
                    "business": payload.business,
                    "keyword": payload.keyword,
                    "canonical_question": payload.canonical_question,
                    "display_order": payload.display_order,
                },
                expected_row_version=payload.row_version,
            )
            catalog = suggestion_public()
            row = next(item for item in catalog["items"] if item["suggestion_id"] == clean_id)
            return {"staged": True, "updated": True, "item": row, "catalog": catalog,
                    "draft": draft_public()}
        except KeyError as error:
            raise HTTPException(status_code=404, detail="질의 캐시 항목을 찾지 못했습니다.") from error
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(status_code=422, detail=f"질의 캐시 수정 실패: {error}") from error

    @app.delete("/api/admin-ui/suggestion-cache/{suggestion_id}")
    def suggestion_cache_disable(suggestion_id: str, _: None = Depends(require_admin_session)):
        try:
            clean_id = require_known_suggestion(suggestion_id)
            stage_suggestion_change(clean_id, enabled=False)
            return {"staged": True, "removed_from_chatbot": False,
                    "catalog": suggestion_public(), "draft": draft_public()}
        except KeyError as error:
            raise HTTPException(status_code=404, detail="질의 캐시 항목을 찾지 못했습니다.") from error

    @app.post("/api/admin-ui/suggestion-cache/{suggestion_id}/restore")
    def suggestion_cache_restore(suggestion_id: str, _: None = Depends(require_admin_session)):
        try:
            clean_id = require_known_suggestion(suggestion_id)
            stage_suggestion_change(clean_id, enabled=True)
            return {"staged": True, "restored": False,
                    "catalog": suggestion_public(), "draft": draft_public()}
        except KeyError as error:
            raise HTTPException(status_code=404, detail="질의 캐시 항목을 찾지 못했습니다.") from error

    @app.put("/api/admin-ui/suggestion-cache/{suggestion_id}/answer")
    def suggestion_cache_answer_save(suggestion_id: str, payload: SuggestionAnswerPayload,
                                     _: None = Depends(require_admin_session)):
        sources = []
        for row in payload.sources:
            title = _clean(row.get("title") or "공식 안내")
            url = _clean(row.get("url"))
            if not url:
                continue
            if not _is_public_kdic_url(url):
                raise HTTPException(
                    status_code=422,
                    detail="검증 답변의 출처는 예금보험공사 공식 URL(kdic.or.kr)만 사용할 수 있습니다.",
                )
            sources.append({"title": title, "url": url})
        if not sources:
            raise HTTPException(
                status_code=422,
                detail="검증 답변을 저장하려면 예금보험공사 공식 출처 URL을 1개 이상 입력해 주세요.",
            )
        try:
            clean_id = require_known_suggestion(suggestion_id)
            stage_suggestion_change(
                clean_id,
                answer_change={"action": "save", "answer": payload.answer, "sources": sources},
            )
            return {"staged": True, "saved": False, "catalog": suggestion_public(),
                    "draft": draft_public()}
        except KeyError as error:
            raise HTTPException(status_code=404, detail="질의 캐시 항목을 찾지 못했습니다.") from error
        except Exception as error:
            raise HTTPException(status_code=422, detail=f"검증 답변 저장 실패: {error}") from error

    @app.delete("/api/admin-ui/suggestion-cache/{suggestion_id}/answer")
    def suggestion_cache_answer_clear(suggestion_id: str, _: None = Depends(require_admin_session)):
        try:
            stage_suggestion_change(
                require_known_suggestion(suggestion_id),
                answer_change={"action": "clear"},
            )
            return {"staged": True, "cleared": False, "catalog": suggestion_public(),
                    "draft": draft_public()}
        except KeyError as error:
            raise HTTPException(status_code=404, detail="질의 캐시 항목을 찾지 못했습니다.") from error

    @app.post("/api/admin-ui/suggestion-cache/{suggestion_id}/activate")
    def suggestion_cache_activate(suggestion_id: str, payload: SuggestionVersionPayload,
                                  _: None = Depends(require_admin_session)):
        try:
            clean_id = require_known_suggestion(suggestion_id)
            versions = require_suggestion_cache().versions(clean_id)
            if not any(row.get("cache_key") == payload.cache_key for row in versions):
                raise ValueError("선택한 검증 답변 버전을 찾지 못했습니다.")
            stage_suggestion_change(
                clean_id,
                answer_change={"action": "activate", "cache_key": payload.cache_key},
            )
            return {"staged": True, "activated": False, "catalog": suggestion_public(),
                    "draft": draft_public()}
        except (KeyError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/api/admin-ui/suggestion-cache/{suggestion_id}/generate", status_code=202)
    def suggestion_cache_generate(suggestion_id: str, _: None = Depends(require_admin_session)):
        manager = require_suggestion_cache()
        row = next((item for item in manager.list_catalog(include_disabled=False)
                    if item["suggestion_id"] == _clean(suggestion_id).upper()), None)
        if row is None:
            raise HTTPException(status_code=409, detail="신규 키워드는 먼저 운영 반영한 뒤 챗봇 답변을 생성해 주세요.")
        try:
            job_id = service_module.JOB_SERVICE.submit(
                f"admin-cache-{uuid.uuid4().hex[:12]}", row["canonical_question"],
                suggestion_id="",
            )
        except Exception as error:
            raise HTTPException(status_code=422, detail=f"검증 답변 생성 시작 실패: {error}") from error
        return {"job_id": job_id, "suggestion_id": row["suggestion_id"], "status": "running"}

    @app.get("/api/admin-ui/guardrails")
    def guardrails(_: None = Depends(require_admin_session)):
        return require_guardrail_manager().public()

    @app.put("/api/admin-ui/guardrails/draft")
    def guardrails_save_draft(payload: GuardrailDraftPayload, _: None = Depends(require_admin_session)):
        try:
            return require_guardrail_manager().save_draft(payload.rules)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.delete("/api/admin-ui/guardrails/draft")
    def guardrails_clear_draft(_: None = Depends(require_admin_session)):
        return require_guardrail_manager().clear_draft()

    @app.post("/api/admin-ui/guardrails/test")
    def guardrails_test(payload: GuardrailTestPayload, _: None = Depends(require_admin_session)):
        return require_guardrail_manager().evaluate(payload.text, payload.scope, use_draft=True)

    @app.post("/api/admin-ui/guardrails/apply")
    def guardrails_apply(payload: GuardrailApplyPayload, _: None = Depends(require_admin_session)):
        if payload.confirmation != "가드레일 반영":
            raise HTTPException(status_code=422, detail="확인 문구 '가드레일 반영'을 정확히 입력해 주세요.")
        return require_guardrail_manager().apply_draft()

    @app.post("/api/admin-ui/guardrails/rollback/{version}")
    def guardrails_rollback(version: str, _: None = Depends(require_admin_session)):
        try:
            return require_guardrail_manager().rollback(version)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="가드레일 버전 이력을 찾지 못했습니다.") from error

    @app.get("/api/admin-ui/prompts")
    def prompts(_: None = Depends(require_admin_session)):
        return require_prompt_manager().public()

    @app.put("/api/admin-ui/prompts/draft")
    def prompts_save_draft(payload: PromptDraftPayload, _: None = Depends(require_admin_session)):
        try:
            return require_prompt_manager().save_draft(payload.values)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.delete("/api/admin-ui/prompts/draft")
    def prompts_clear_draft(_: None = Depends(require_admin_session)):
        return require_prompt_manager().clear_draft()

    @app.post("/api/admin-ui/prompts/compare")
    def prompts_compare(payload: PromptComparePayload, _: None = Depends(require_admin_session)):
        manager = require_prompt_manager()
        guard = require_guardrail_manager().evaluate(payload.question, "input")
        if guard["blocked"]:
            raise HTTPException(status_code=400, detail="가드레일 차단 규칙에 해당하는 질문입니다.")
        if not prompt_compare_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="다른 프롬프트 A/B 비교가 실행 중입니다.")
        try:
            question = str(guard["text"])
            active = manager.active_values()
            selected_preset = None
            if payload.preset_id:
                selected_preset = prompt_presets.get(_clean(payload.preset_id))
                if selected_preset is None:
                    raise HTTPException(status_code=404, detail="비교할 프롬프트 설정을 찾지 못했습니다.")
                candidate_values = copy.deepcopy(dict(selected_preset["values"]))
            else:
                candidate_values = manager.draft_values()
            baseline = run_prompt_variant(question, active)
            candidate = run_prompt_variant(question, candidate_values)
            return {
                "question": question,
                "guardrail_hits": guard["hits"],
                "active_version": manager.public()["active_version"],
                "candidate_preset": prompt_preset_public(selected_preset) if selected_preset else None,
                "baseline": baseline,
                "candidate": candidate,
                "changed_slots": [slot for slot in active if active[slot] != candidate_values[slot]],
            }
        finally:
            prompt_compare_lock.release()

    @app.post("/api/admin-ui/chat/compare")
    def chat_compare(payload: PromptComparePayload, _: None = Depends(require_admin_session)):
        manager = require_prompt_manager()
        guard = require_guardrail_manager().evaluate(payload.question, "input")
        if guard["blocked"]:
            raise HTTPException(status_code=400, detail="가드레일 차단 규칙에 해당하는 질문입니다.")
        if not prompt_compare_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="다른 최종 답변 A/B 비교가 실행 중입니다.")
        try:
            before_config = active_config()
            candidate_config = _validate_config(drafts["config"], before_config)
            question = str(guard["text"])
            with execution_lock:
                set_config(before_config)
                baseline = run_prompt_variant(question, manager.active_values())
                set_config(candidate_config)
                candidate = run_prompt_variant(question, manager.draft_values())
            return {
                "question": question,
                "guardrail_hits": guard["hits"],
                "baseline_config": before_config,
                "candidate_config": candidate_config,
                "changed_parameters": [key for key in before_config if before_config[key] != candidate_config[key]],
                "changed_prompts": draft_public().get("prompt_changes") or [],
                "baseline": baseline,
                "candidate": candidate,
            }
        finally:
            if "before_config" in locals():
                with execution_lock:
                    set_config(before_config)
            prompt_compare_lock.release()

    @app.get("/api/admin-ui/capabilities")
    def capabilities(_: None = Depends(require_admin_session)):
        base = service_module.admin_capabilities(); features = list(base.get("features") or [])
        for value in ("chat_pipeline_test", "chat_pipeline_compare", "runtime_config", "evaluation_dataset_upload", "evaluation_run", "evaluation_k_curve", "evaluation_record_delete", "parameter_apply", "chunk_staged_write", "draft_partial_clear", "rollback", "history_delete", "api_key_test", "api_key_runtime_rotation", "pipeline_runtime_graph", "pipeline_setting_links", "pipeline_label_customization", "operations_monitoring", "guardrail_draft", "guardrail_test", "guardrail_apply", "guardrail_rollback", "prompt_draft", "prompt_compare", "prompt_presets"):
            if value not in features: features.append(value)
        blocked = [v for v in (base.get("disabled_mutations") or []) if v not in {"document_write", "document_delete", "evaluation_run", "parameter_apply"}]
        return {**base, "admin_mode": "STAGED_WRITE", "features": features, "disabled_mutations": blocked}

    return {"installed": True, "page": str(page), "auth": "one_time_bootstrap_to_httponly_cookie",
            "session_ttl_seconds": SESSION_TTL_SECONDS, "mode": "STAGED_WRITE", "evaluation": "isolated_ab",
            "mutations": "draft_validate_apply_rollback", "pipeline_studio": "dynamic_graph_safe_settings"}
