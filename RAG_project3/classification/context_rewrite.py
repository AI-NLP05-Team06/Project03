# [Phase5] 맥락반영 재작성: 직전 턴을 참고해서 지시어·생략 표현이 있는 후속
# 질문을 맥락 없이도 이해 가능한 완전한 질문으로 다시 쓴다.
from __future__ import annotations

from core.hcx_api import hcx_chat_text

_SYSTEM_PROMPT = """당신은 대화형 챗봇의 질의 재작성기입니다.
직전 질문(이전 턴)과 현재 질문(후속 턴)이 주어지면, 현재 질문에 있는
지시어("그거", "그럼", "그 경우" 등)나 생략된 내용을 이전 질문의 맥락으로
채워서, 이전 턴 없이도 단독으로 이해 가능한 완전한 질문 하나로 다시 쓰세요.

- 이전 질문에 없는 새로운 사실을 지어내지 마세요.
- 현재 질문의 의도(무엇을 묻는지)는 그대로 유지하세요.
- 재작성한 질문 문장만 출력하고, 다른 설명은 붙이지 마세요."""


def rewrite_with_context(current_question: str, previous_question: str) -> str:
    rewritten = hcx_chat_text(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=f"""
[이전 질문]
{previous_question}

[현재 질문]
{current_question}

위 현재 질문을 이전 질문의 맥락을 반영해서 완전한 질문으로 다시 쓰세요.
""".strip(),
        max_tokens=150,
        temperature=0.0,
    )
    return rewritten.strip()
