# [9/10] 사용자 입력을 받아 반복적으로 run_kdic_rag()를 실행하는 대화형 질의응답 루프입니다. (실행 시 즉시 입력창이 뜹니다)

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from core.config import *
from scripts.run_rag import *

# None이면 6개 업무 전체에서 검색합니다.
# 특정 업무만 검색하려면 아래 값을 6개 업무 중 하나로 변경할 수 있습니다.
BUSINESS_FUNCTION_FILTER = None

# 기존 V4.6 Baseline 검색 설정을 그대로 사용합니다.
TOP_K = 5
MIN_SCORE = 0.30

EXIT_COMMANDS = {"종료", "끝", "exit", "quit", "q"}
chat_history: list[dict[str, Any]] = []
rag_output: dict[str, Any] | None = None


def run_kdic_chat() -> list[dict[str, Any]]:
    """입력창에서 질문을 반복해서 받아 기존 RAG 출력 형식으로 실행합니다."""
    print("=" * 72)
    print("예금보험공사 문서 기반 RAG 질의응답")
    print("질문을 직접 입력하세요.")
    print("종료 명령어: 종료 / 끝 / exit / quit / q")
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

        print("\n" + "-" * 72)

        try:
            output = run_kdic_rag(
                question,
                top_k=TOP_K,
                business_function=BUSINESS_FUNCTION_FILTER,
                min_score=MIN_SCORE,
            )
        except Exception as error:
            print("\n질문 처리 중 오류가 발생했습니다.")
            print(f"- 오류 유형: {type(error).__name__}")
            print(f"- 오류 내용: {error}")
            print("다른 질문을 입력하거나 설정과 API 연결 상태를 확인해 주세요.")
            continue

        chat_history.append(output)

        # 직전 결과를 노트북 전역 변수에 저장해
        # 다음 다운로드 셀에서 그대로 사용할 수 있게 합니다.
        globals()["rag_output"] = output

        print("-" * 72)
        print(f"현재 세션 처리 질문 수: {len(chat_history)}")

    print(f"\n세션 종료 — 총 {len(chat_history)}개 질문 처리")
    return chat_history


chat_history = run_kdic_chat()
