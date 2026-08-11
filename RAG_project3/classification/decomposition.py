# [Phase7] 복합질의 판별 + Query Decomposition (Type A/B만 — Type C 순차의존은
# 세션4에서 스코프 아웃함, 실제 데이터에서 거의 안 보였음). 판별과 분리를
# LLM 한 번의 호출로 같이 처리한다: 단일 질의면 원문 그대로 1개 리스트,
# 복합 질의면 하위질문으로 분리하되 공유 맥락(대상·조건 등)을 각 하위질문에
# 주입해서 독립적으로 이해 가능하게 만든다(Type B 처리).
from __future__ import annotations

from core.hcx_api import hcx_chat_text

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
    raw = hcx_chat_text(
        system_prompt=_DECOMPOSITION_SYSTEM_PROMPT,
        user_prompt=question,
        max_tokens=300,
        temperature=0.0,
    )
    sub_questions = [line.strip() for line in raw.splitlines() if line.strip()]
    return sub_questions or [question]
