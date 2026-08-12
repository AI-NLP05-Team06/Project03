# [Phase7] 복합질의 판별 + Query Decomposition (Type A/B만 — Type C 순차의존은
# 세션4에서 스코프 아웃함, 실제 데이터에서 거의 안 보였음). 판별과 분리를
# LLM 한 번의 호출로 같이 처리한다: 단일 질의면 원문 그대로 1개 리스트,
# 복합 질의면 하위질문으로 분리하되 공유 맥락(대상·조건 등)을 각 하위질문에
# 주입해서 독립적으로 이해 가능하게 만든다(Type B 처리).
from __future__ import annotations

import re

from core.hcx_api import hcx_chat_text

_LEADING_NUMBER_PATTERN = re.compile(r"^\d+[.)]\s*")

# 규칙 기반 사전필터: 복합질의 신호(체언+와/과, 쉼표, "~하고" 식 동사 연결형)가
# 하나도 없으면 LLM 호출 없이 바로 단일질의로 처리한다(지연시간 절감).
# `검색평가데이터셋.xlsx`의 RETRIEVE 180건(단일 105/복합 75)으로 검증: 이 신호가
# 하나라도 있으면 LLM 판별을 그대로 유지(복합질의의 96%=72/75가 이 신호를 가짐),
# 신호가 전혀 없으면 스킵(단일질의의 83%=87/105에서 LLM 호출을 절약).
# 트레이드오프: 신호 없이 실제로는 복합인 질문 3/75(4%)은 분해 없이 그대로
# 단일 처리된다 — LLM 판별 자체도 recall 96%로 완벽하지 않았으므로 감내 가능한
# 수준으로 판단(evaluate_decomposition.py로 재검증함).
_NOUN_CONJUNCTION_PATTERN = re.compile(r"[가-힣]+[와과](?:\s|$)")
_VERB_CONJUNCTION_PATTERN = re.compile(r"[가-힣]{2,}고(?:\s|,)")


def _looks_possibly_compound(question: str) -> bool:
    return bool(
        _NOUN_CONJUNCTION_PATTERN.search(question)
        or _VERB_CONJUNCTION_PATTERN.search(question)
    ) or "," in question


# 실측 확인된 오류: 아주 드물게(180건 중 2건) LLM이 하위질문을 나누는 대신
# 답변을 통째로 생성해버림(예: "네이버의 CLOVA X가 답변드립니다..." 같은
# 문장까지 섞여 나옴). 정상적인 질문은 이 길이를 넘지 않으므로, 넘으면
# 분해 실패로 간주하고 원문 그대로 되돌린다.
MAX_SUB_QUESTION_LENGTH = 150

_DECOMPOSITION_SYSTEM_PROMPT = """당신은 사용자 질문을 검색하기 좋은 단위로
정리하는 질의 분해기입니다.

1. 질문이 하나의 요청만 담고 있으면, 원래 질문을 그대로 한 줄로 출력하세요.
2. 질문이 여러 개의 독립적인 요청을 담고 있으면(예: "A와 B를 알려주세요",
   "A는 얼마고 B는 어떻게 되나요"), 각각을 별도의 하위질문으로 나누세요.
   이때 원래 질문에 있는 공유 맥락(특정 상품·조건·대상 등)을 모든 하위질문에
   포함시켜서, 각 하위질문만 봐도 무슨 뜻인지 알 수 있게 만드세요.

- 원래 질문에 없는 새로운 사실이나 조건을 추가하지 마세요.
- 한 줄에 질문 하나씩만 출력하고, 번호·설명·따옴표는 붙이지 마세요."""


def decompose_query(question: str) -> list[str]:
    if not _looks_possibly_compound(question):
        return [question]

    raw = hcx_chat_text(
        system_prompt=_DECOMPOSITION_SYSTEM_PROMPT,
        user_prompt=question,
        max_tokens=300,
        temperature=0.0,
    )
    sub_questions = [
        _LEADING_NUMBER_PATTERN.sub("", line.strip())
        for line in raw.splitlines() if line.strip()
    ]
    if not sub_questions:
        return [question]
    if any(len(sq) > MAX_SUB_QUESTION_LENGTH for sq in sub_questions):
        return [question]
    return sub_questions
