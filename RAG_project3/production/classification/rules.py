# [2-5/6] rule 티어: 업무(6개) 키워드 사전, 의도(정보/민원처리) 패턴, DIRECT_RESPONSE 게이트, 되묻기 규칙
from __future__ import annotations

import re

# ============================================================
# 2. 업무(도메인) 키워드 사전
# ============================================================
# data/chunks.jsonl의 title·section_title을 business_function별로 모아서
# 서로 구별되는 표현만 추렸다 (예: "가지급금"처럼 여러 업무에 겹치는 단어는 제외).
BUSINESS_FUNCTION_KEYWORDS: dict[str, list[str]] = {
    "예금자보호제도": ["예금자보호", "보호한도", "보호대상", "부보금융회사", "예금보호", "영업정지"],
    "예금보험금 안내": ["예금보험금", "보험사고", "지급대상 금융회사", "지급공고", "가지급금"],
    "고객 미수령금 신청": ["미수령금", "상속인 금융거래조회", "지급대행점", "통합신청", "파산배당금"],
    "착오송금 반환 신청": ["착오송금", "반환지원", "간편송금", "잘못 송금", "잘못 보낸"],
    "채무조정 안내": ["채무조정", "개인회생", "개인파산", "신용회복", "면책", "워크아웃", "변제유예"],
    "은닉재산 신고": ["은닉재산", "부실관련자", "차명재산", "포상금"],
}


def classify_business_function_rule(text: str) -> str | None:
    """키워드가 정확히 한 업무에서만 매칭되면 그 업무를, 매칭 0개나 2개 이상(충돌)이면 None을 반환한다."""
    matched = [
        business_function
        for business_function, keywords in BUSINESS_FUNCTION_KEYWORDS.items()
        if any(keyword in text for keyword in keywords)
    ]
    if len(matched) == 1:
        return matched[0]
    return None


# ============================================================
# 3. 의도(정보/민원처리) 키워드 패턴
# ============================================================
# 세션에서 확정한 매핑: 정보질문(OVERVIEW/ELIGIBILITY/AMOUNT/CONTACT/TIME) vs
# 민원처리질문(APPLICATION/DOCUMENTS/STATUS). 어미·동사 위주 규칙이라 완벽하진
# 않고, 애매하면 None을 반환해 임베딩/LLM 티어로 넘긴다.
COMPLAINT_INTENT_PATTERNS = [
    re.compile(p) for p in [
        r"하려면", r"신청\s*(방법|절차|하는)?", r"서류", r"제출", r"접수",
        r"구비", r"확인\s*방법", r"조회\s*방법", r"어디서\s*확인",
    ]
]
INFO_INTENT_PATTERNS = [
    re.compile(p) for p in [
        r"(무엇|뭔가요|뭐예요|뭐가)", r"인가요\??$", r"얼마", r"기한",
        r"며칠", r"포함되나요", r"대상인가요", r"차이", r"얼마나",
    ]
]


def classify_intent_rule(text: str) -> str | None:
    is_complaint = any(p.search(text) for p in COMPLAINT_INTENT_PATTERNS)
    is_info = any(p.search(text) for p in INFO_INTENT_PATTERNS)
    if is_complaint and not is_info:
        return "민원처리"
    if is_info and not is_complaint:
        return "정보"
    return None  # 둘 다 걸리거나 둘 다 안 걸리면 애매 → 임베딩/LLM 티어로


# ============================================================
# 4. DIRECT_RESPONSE 게이트 (인사말·감사·메타질문·스타일 지시)
# ============================================================
DIRECT_RESPONSE_KEYWORDS = [
    "안녕", "반갑",
    "고마워", "감사", "도움이 됐", "도움이 되었",
    "챗봇", "무슨 질문", "어떻게 사용", "지원하는 업무", "목록으로",
    "쉽게 설명", "전문가 수준", "핵심만", "핵심 내용", "자세히 설명", "간단히",
    "잘못 입력", "다시 물어",
]


def is_direct_response(text: str) -> bool:
    return any(keyword in text for keyword in DIRECT_RESPONSE_KEYWORDS)


# ============================================================
# 5. 되묻기(CLARIFY) 규칙
# ============================================================
def needs_clarification_rule(text: str) -> bool:
    """업무가 뭔지 rule 티어로 특정이 안 되면(0개 또는 2개 이상 매칭) 되묻기 대상으로 본다."""
    return classify_business_function_rule(text) is None
