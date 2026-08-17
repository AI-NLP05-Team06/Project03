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

# 실측 확인된 오류: "어떤 순서로 진행돼요?" 같은 절차 순서를 묻는 질문을 LLM이
# 하위질문이 아니라 "대상자 확인"/"조사"/"회수"처럼 명사 조각으로 쪼개버리는
# 사고가 관찰됨(예: eval01 challenge_v3의 CHALLENGE_064). 실제 정상 하위질문은
# challenge_v3+routing_v5 254문항 전체에서 가장 짧은 것도 13자("가지급금이란
# 무엇인가요?")였으므로, 이보다 훨씬 여유를 둔 8자 미만 하위질문이 하나라도
# 있으면 분해 자체가 깨진 것으로 보고 전체를 원문으로 되돌린다.
MIN_SUB_QUESTION_LENGTH = 8

_NUMBER_PATTERN = re.compile(r"\d+")

# temperature=0.0이어도 CHALLENGE_018을 다시 호출했을 때 숫자를 지어내는
# 대신 "다음과 같습니다"로 답변을 여는 문장이 나온 적이 있어(재현마다 살짝
# 다름), 숫자 체크만으로는 다 못 걸러낼 수 있다고 판단해 이 문구 체크를
# 추가함. 254문항 중 실제 분해된 99건 전수 검사에서 이 4개 문구는 전부
# CHALLENGE_018(이미 위 숫자 체크로도 잡히는 동일 문항)에서만 등장했고
# 다른 정상 하위질문 97건에는 하나도 없었다(오탐 0건).
_ANSWER_OPENING_PHRASES = ("다음과 같습니다", "다음과 같이", "따라서", "결론적으로")


def _has_fabricated_number(question: str, sub_questions: list[str]) -> bool:
    """실측 확인된 오류(eval01 CHALLENGE_017/018): LLM이 하위질문을 나누는
    대신 답변을 계산해서 지어내는데(예: 옛날 예금자보호 한도 "5천만원"을
    끌어와 "8천만원"이라는 합산액까지 만들어냄), 각 줄이 150자 미만이라
    MAX_SUB_QUESTION_LENGTH 체크를 통과해버림. 하위질문은 "묻는" 역할만
    해야 하므로 원본 질문에 없던 숫자가 새로 등장할 이유가 없다 — 실제
    분해된 254문항 중 99건을 전수 검사해서 원본에 없는 숫자가 등장한 건
    이 두 환각 사례뿐이었고(오탐 0건), 정상 하위질문은 전부 원본 숫자만
    재사용했다. 하위질문 중 하나라도 원본에 없는 숫자를 담고 있으면
    분해 자체가 답변 지어내기로 오염된 것으로 보고 원문으로 되돌린다."""
    original_numbers = set(_NUMBER_PATTERN.findall(question))
    has_new_number = any(
        set(_NUMBER_PATTERN.findall(sq)) - original_numbers
        for sq in sub_questions
    )
    has_answer_opening = any(
        phrase in sq for sq in sub_questions for phrase in _ANSWER_OPENING_PHRASES
    )
    return has_new_number or has_answer_opening


_DECOMPOSITION_SYSTEM_PROMPT = """당신은 사용자 질문을 검색하기 좋은 단위로
정리하는 질의 분해기입니다.

1. 질문이 하나의 요청만 담고 있으면, 원래 질문을 그대로 한 줄로 출력하세요.
2. 질문이 여러 개의 독립적인 요청을 담고 있으면(예: "A와 B를 알려주세요",
   "A는 얼마고 B는 어떻게 되나요"), 각각을 별도의 하위질문으로 나누세요.
   이때 원래 질문에 있는 공유 맥락(특정 상품·조건·대상 등)을 모든 하위질문에
   포함시켜서, 각 하위질문만 봐도 무슨 뜻인지 알 수 있게 만드세요.

- 원래 질문에 없는 새로운 사실이나 조건을 추가하지 마세요.
- 한 줄에 질문 하나씩만 출력하고, 번호·설명·따옴표는 붙이지 마세요."""

# 실측 확인된 실패 시도(되돌림, 기록만 남김): 같은 프롬프트에 "나열형 질문은
# 항목 개수만큼 쪼개지 말라"는 규칙+예시를 추가해서 CHALLENGE_043류(미수령금에
# 예금보험금/파산배당금/정산금이 다 포함되나요? 각각 뭐에요? -> 4분할)의
# 과도분해는 의도대로 고쳐졌으나(4분할->2분할), challenge_v3/routing_v5 두
# 데이터셋 전체 평균(hit@3/mrr@10/map@10)이 오히려 나빠짐(5개 검색기법 전부).
# 온도 0이라도 공유 프롬프트에 규칙을 추가하면 의도한 케이스 밖의 다른 정상
# 복합질의 분해 방식까지 같이 흔들린다는 뜻 — 프롬프트 레벨 땜질이 아니라
# 분기 자체를 프롬프트 밖(규칙 기반)에서 하지 않는 한 이 문제는 재발할
# 가능성이 높다고 판단, 시도하지 않기로 함.


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
    if any(len(sq) < MIN_SUB_QUESTION_LENGTH for sq in sub_questions):
        return [question]
    if _has_fabricated_number(question, sub_questions):
        return [question]

    # 실측 확인된 문제: 프롬프트가 "단일 요청이면 원문을 그대로 출력하라"고
    # 지시해도 LLM이 종종 말을 안 듣고 장황하게 다시 씀(예: "일반예금, IRP,
    # 연금저축이 같이 있으면 전부 합쳐 1억원인가요?" -> "일반예금, IRP, 연금저축
    # 계좌가 있을 때 이 세 가지 금융상품의 금액이 모두 합산되어..."). 원본보다
    # 장황한 재작성은 검색 임베딩의 구체성만 희석시키므로(세션7의 Multi-Query/
    # HyDE 기각 사유와 동일), 출력이 1줄(=분해 안 함)이면 LLM 출력을 버리고
    # 원문을 그대로 쓴다. 실제로 여러 개로 쪼갠 경우(len>1)만 LLM 출력을 신뢰한다.
    if len(sub_questions) == 1:
        return [question]
    return sub_questions
