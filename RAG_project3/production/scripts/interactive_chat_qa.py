# [9-2/10] 질의분석(구어체 정규화/맥락재작성/복합질의 분해+분류) 포함 전체
# 파이프라인(generation/compound_answer.py::answer_query)을 반복 입력으로
# 테스트하는 대화형 루프입니다. (실행 시 즉시 입력창이 뜹니다)
# interactive_chat.py와 차이: 그쪽은 run_kdic_rag()만 써서 검색+답변만
# 보여주고, 이쪽은 질의분석 8단계 전체를 거친 실제 프로덕션 답변을 보여줍니다.

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from core.config import *
from generation.compound_answer import answer_query

EXIT_COMMANDS = {"종료", "끝", "exit", "quit", "q"}
RESET_COMMANDS = {"초기화", "reset"}
chat_history: list[dict[str, Any]] = []
previous_question: str | None = None


def run_kdic_chat_qa() -> list[dict[str, Any]]:
    """입력창에서 질문을 반복해서 받아 질의분석 포함 전체 파이프라인으로
    실행합니다. 직전 질문을 이어서 previous_question으로 넘기므로
    후속질문(그럼/그거 등)도 맥락반영 재작성이 자동으로 적용됩니다."""
    global previous_question

    print("=" * 72)
    print("예금보험공사 문서 기반 RAG 질의응답 (질의분석 전체 파이프라인)")
    print("질문을 직접 입력하세요. 직전 질문을 맥락으로 이어서 사용합니다.")
    print("종료 명령어: 종료 / 끝 / exit / quit / q")
    print("맥락 초기화: 초기화 / reset")
    print("=" * 72)

    while True:
        try:
            question = input("\n사용자 질문 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n입력을 종료합니다.")
            break

        if not question:
            print("질문이 비어 있습니다. 내용을 입력해 주세요.")
            continue

        if question.lower() in EXIT_COMMANDS:
            print("대화형 질의응답을 종료합니다.")
            break

        if question.lower() in RESET_COMMANDS:
            previous_question = None
            print("맥락을 초기화했습니다. 새 대화로 시작합니다.")
            continue

        print("\n" + "-" * 72)

        try:
            answer = answer_query(question, previous_question=previous_question)
        except Exception as error:
            print("\n질문 처리 중 오류가 발생했습니다.")
            print(f"- 오류 유형: {type(error).__name__}")
            print(f"- 오류 내용: {error}")
            print("다른 질문을 입력하거나 설정과 API 연결 상태를 확인해 주세요.")
            continue

        print(answer)
        print("-" * 72)

        chat_history.append({"question": question, "previous_question": previous_question, "answer": answer})
        previous_question = question
        print(f"현재 세션 처리 질문 수: {len(chat_history)}")

    print(f"\n세션 종료 — 총 {len(chat_history)}개 질문 처리")
    return chat_history


chat_history = run_kdic_chat_qa()
