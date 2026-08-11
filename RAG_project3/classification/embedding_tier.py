# [7/9] 임베딩 티어: rule이 못 정한 업무·의도를 bge-m3 유사도(1등-2등 gap)로 판단한다.
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.hcx_api import hcx_embed_text

CACHE_PATH = Path("data/classification_embedding_cache.json")

# classification/calibrate_embedding_gap.py로 rule 미해결 문항(업무 49건/의도 86건)
# 대상 gap-정답률 상관관계를 실측해서 잡은 값. gap 0.08 이상 구간은 두 분류 모두
# 정답률 92~100%였고, 그 아래는 50~77%로 뚝 떨어져서 여기를 경계로 잡았다.
BUSINESS_FUNCTION_GAP_THRESHOLD = 0.08
INTENT_GAP_THRESHOLD = 0.08

# data/chunks.jsonl의 title·section_title을 참고해 업무별 대표 문장을 뽑았다.
BUSINESS_FUNCTION_REPRESENTATIVES: dict[str, list[str]] = {
    "예금자보호제도": [
        "예금자보호제도의 보호 대상과 보호 한도에 대한 질문",
        "예금이 예금자보호법에 따라 얼마까지, 어떤 조건으로 보호되는지",
        "부보금융회사와 보호 대상 금융상품 여부",
    ],
    "예금보험금 안내": [
        "금융회사 파산 시 예금보험금 지급 대상과 신청 절차",
        "가지급금과 예금보험금 신청 방법, 필요 서류",
        "예금보험금 지급대상 금융회사 확인과 지급 공고",
    ],
    "고객 미수령금 신청": [
        "고객 미수령금 조회와 통합신청 방법",
        "상속인 금융거래조회, 지급대행점 방문 신청",
        "파산 금융회사에 남은 미수령금과 파산배당금",
    ],
    "착오송금 반환 신청": [
        "착오송금 반환지원 신청 대상과 절차",
        "잘못 보낸 송금, 간편송금 반환지원",
        "착오송금 반환지원 제외 대상과 지급명령",
    ],
    "채무조정 안내": [
        "개인회생, 개인파산, 신용회복지원 등 채무조정 제도",
        "채무조정 신청 자격과 절차, 변제기간",
        "면책, 워크아웃, 신용회복위원회 상담",
    ],
    "은닉재산 신고": [
        "부실관련자의 은닉재산 신고 방법과 포상금",
        "차명재산, 은닉재산 신고센터와 신고 절차",
        "은닉재산 신고자 신원 보호",
    ],
}

INTENT_REPRESENTATIVES: dict[str, list[str]] = {
    "정보": [
        "제도나 금액, 대상 여부에 대한 정보를 묻는 질문",
        "얼마인지, 대상인지, 기한이 언제인지 등 사실 확인 질문",
    ],
    "민원처리": [
        "신청 방법, 필요 서류, 접수 절차를 묻는 질문",
        "신청 결과나 처리 상태를 조회하는 방법을 묻는 질문",
    ],
}


def _load_cache() -> dict[str, list[float]]:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict[str, list[float]]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


_cache = _load_cache()


def embed_cached(text: str) -> np.ndarray:
    if text not in _cache:
        _cache[text] = hcx_embed_text(text)
        _save_cache(_cache)
    return np.asarray(_cache[text])


def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if denominator == 0:
        return 0.0
    return float(np.dot(vector_a, vector_b) / denominator)


def _category_vectors(representatives: dict[str, list[str]]) -> dict[str, np.ndarray]:
    vectors = {}
    for category, sentences in representatives.items():
        embeddings = [embed_cached(sentence) for sentence in sentences]
        vectors[category] = np.mean(embeddings, axis=0)
    return vectors


_BUSINESS_FUNCTION_VECTORS = _category_vectors(BUSINESS_FUNCTION_REPRESENTATIVES)
_INTENT_VECTORS = _category_vectors(INTENT_REPRESENTATIVES)


def _rank_categories(
    question: str,
    category_vectors: dict[str, np.ndarray],
) -> list[tuple[str, float]]:
    query_vector = embed_cached(question)
    scored = [
        (category, cosine_similarity(query_vector, vector))
        for category, vector in category_vectors.items()
    ]
    return sorted(scored, key=lambda item: item[1], reverse=True)


def classify_business_function_embedding(
    question: str,
    *,
    gap_threshold: float = BUSINESS_FUNCTION_GAP_THRESHOLD,
) -> tuple[str | None, float, float]:
    """1등-2등 gap이 임계값 이상이면 1등을 확정, 아니면 None(LLM 티어로 escalate)을 반환한다."""
    ranked = _rank_categories(question, _BUSINESS_FUNCTION_VECTORS)
    top1_category, top1_score = ranked[0]
    top2_score = ranked[1][1]
    gap = top1_score - top2_score
    return (top1_category if gap >= gap_threshold else None), top1_score, gap


def classify_intent_embedding(
    question: str,
    *,
    gap_threshold: float = INTENT_GAP_THRESHOLD,
) -> tuple[str | None, float, float]:
    ranked = _rank_categories(question, _INTENT_VECTORS)
    top1_category, top1_score = ranked[0]
    top2_score = ranked[1][1]
    gap = top1_score - top2_score
    return (top1_category if gap >= gap_threshold else None), top1_score, gap
