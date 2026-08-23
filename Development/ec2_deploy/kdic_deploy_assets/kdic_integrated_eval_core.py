from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QueryPlan:
    need_id: str
    variant_id: str
    dense_query: str
    bm25_query: str
    filter_mode: str = "NONE"
    business_filters: list[str] = field(default_factory=list)
    soft_business_hints: list[str] = field(default_factory=list)
    query_weight: float = 1.0
    query_source: str = "ORIGINAL"


@dataclass
class AnalyzerCase:
    evaluation_id: str
    analyzer: str
    original_question: str
    route: str
    analysis_latency_ms: float
    plans: list[QueryPlan]
    raw_result: dict[str, Any]


def normalize_route(value: Any) -> str:
    text = str(value or "").strip().upper()
    mapping = {
        "SIMPLE_RETRIEVE": "RETRIEVE",
        "MULTI_RETRIEVE": "RETRIEVE",
        "OUT_OF_SCOPE": "OUT_OF_SCOPE",
        "OOS": "OUT_OF_SCOPE",
        "DIRECT": "DIRECT_RESPONSE",
        "DIRECT_RESPONSE": "DIRECT_RESPONSE",
        "CLARIFY": "CLARIFY",
        "RETRIEVE": "RETRIEVE",
    }
    return mapping.get(text, text)