# [Phase5] 맥락반영 재작성 검증용 합성 멀티턴 테스트셋 로더.
# 데이터 자체는 data/context_rewrite_testset.json에 있음(실제 대화 로그가 없어서
# 직접 만든 것 — 골드셋 L4_multi_turn 10건은 전부 독립된 단일 질문이라 이전 턴
# 참조가 필요 없어서 검증에 못 씀). 각 쌍의 turn2는 turn1 없이는 무슨 뜻인지
# 알 수 없는 지시어/생략 표현을 담고 있다.
from __future__ import annotations

import json
from pathlib import Path

CONTEXT_REWRITE_TESTSET_PATH = Path("data/context_rewrite_testset.json")


def load_context_rewrite_testset(path: Path = CONTEXT_REWRITE_TESTSET_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))
