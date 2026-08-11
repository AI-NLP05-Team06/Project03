# [8/9] LLM 폴백 티어: rule+임베딩도 실패한 문항만 마지막으로 LLM이 판단한다.
# 업무 불명확 케이스를 "우리 업무인데 애매함(CLARIFY)"과 "6개 업무 전부 무관(OUT_OF_SCOPE)"
# 으로 나누는 3분기를 여기서 처리한다 (세션에서 rule 티어 테스트 중 발견한 설계 갭).
from __future__ import annotations

import json
import re

from core.hcx_api import hcx_chat_text

BUSINESS_FUNCTIONS = [
    "예금자보호제도",
    "예금보험금 안내",
    "고객 미수령금 신청",
    "착오송금 반환 신청",
    "채무조정 안내",
    "은닉재산 신고",
]

_CLASSIFY_SYSTEM_PROMPT = """당신은 예금보험공사(KDIC) RAG 챗봇의 질의 분류기입니다.
사용자 질문을 아래 6개 업무 중 하나로 분류하세요.

- 예금자보호제도: 보호 대상, 보호 한도 등 제도 자체에 대한 질문
- 예금보험금 안내: 금융회사 파산 시 예금보험금 신청·수령
- 고객 미수령금 신청: 미수령금 조회, 상속인 금융거래조회, 통합신청
- 착오송금 반환 신청: 잘못 보낸 송금의 반환지원
- 채무조정 안내: 개인회생·개인파산·신용회복지원
- 은닉재산 신고: 부실관련자 은닉재산 신고

이 6개 중 하나로 특정 가능하면 business_function에 그 이름을 쓰세요.
6개 중 하나인 것은 맞는데 질문이 너무 짧거나 모호해서 특정이 안 되면
business_function을 "CLARIFY"로 쓰세요.
6개 업무 어디에도 해당하지 않는 완전히 무관한 질문(예: 일반 금융상품 추천,
타 기관 업무, 잡담)이면 business_function을 "OUT_OF_SCOPE"로 쓰세요.

business_function이 6개 중 하나로 특정된 경우에만 intent도 판단하세요.
intent는 "정보"(개념·대상·금액·연락처·기한 등 사실 확인) 또는
"민원처리"(신청 방법·필요 서류·처리 상태 조회 등 절차) 중 하나입니다.
CLARIFY나 OUT_OF_SCOPE면 intent는 null로 쓰세요.

반드시 아래 JSON 형식만 출력하세요:
{"business_function": "<6개 업무명 또는 CLARIFY 또는 OUT_OF_SCOPE>", "intent": "정보" 또는 "민원처리" 또는 null}"""

_INTENT_ONLY_SYSTEM_PROMPT = """당신은 예금보험공사(KDIC) RAG 챗봇의 질의 분류기입니다.
사용자 질문의 의도를 "정보"(개념·대상·금액·연락처·기한 등 사실 확인) 또는
"민원처리"(신청 방법·필요 서류·처리 상태 조회 등 절차) 중 하나로 분류하세요.
반드시 아래 JSON 형식만 출력하세요:
{"intent": "정보" 또는 "민원처리"}"""


def _extract_json(raw: str) -> dict | None:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def classify_business_and_intent_llm(question: str) -> dict:
    """rule+임베딩이 업무를 못 정한 문항용. business_function은 6개 업무명 / "CLARIFY" / "OUT_OF_SCOPE" 중 하나."""
    raw = hcx_chat_text(
        system_prompt=_CLASSIFY_SYSTEM_PROMPT,
        user_prompt=question,
        max_tokens=200,
        temperature=0.0,
    )
    parsed = _extract_json(raw)
    if parsed is None or "business_function" not in parsed:
        # 파싱 실패는 "판단 불가"와 같은 취급 -> 안전하게 되묻기로 보낸다.
        return {"business_function": "CLARIFY", "intent": None, "llm_raw": raw}

    business_function = parsed.get("business_function")
    if business_function not in BUSINESS_FUNCTIONS + ["CLARIFY", "OUT_OF_SCOPE"]:
        business_function = "CLARIFY"

    return {
        "business_function": business_function,
        "intent": parsed.get("intent") if business_function in BUSINESS_FUNCTIONS else None,
    }


def classify_intent_llm(question: str) -> str:
    """업무는 이미 rule/임베딩으로 정해졌는데 의도만 못 정한 문항용."""
    raw = hcx_chat_text(
        system_prompt=_INTENT_ONLY_SYSTEM_PROMPT,
        user_prompt=question,
        max_tokens=50,
        temperature=0.0,
    )
    parsed = _extract_json(raw)
    intent = parsed.get("intent") if parsed else None
    return intent if intent in ("정보", "민원처리") else "정보"  # 파싱 실패 시 더 가벼운 쪽(정보)으로 폴백
