# [문서화용] 파이프라인 실제 지연시간을 케이스별로 재서 기록한다.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "production"))

import time

from core.config import *
from core.integrity_check import *
from generation.compound_answer import answer_query

CASES = [
    ("단일질의 (rule 티어로 즉시 분류)", {"question": "은닉재산 신고는 어떻게 하나요?"}),
    ("단일질의 (구어체, 정규화 필요)", {"question": "예보에 미수령금 있는지 어떻게 확인해요?"}),
    ("멀티턴 (맥락재작성 필요)", {"question": "그럼 이자도 포함되나요?", "previous_question": "예금자보호 한도가 얼마예요?"}),
    ("복합질의 (하위질문 2개)", {"question": "예금자보호 한도와 그 한도에 이자가 포함되는지 함께 알려주세요."}),
    ("인사말 (DIRECT_RESPONSE 게이트)", {"question": "안녕하세요"}),
]

print(f"{'케이스':40s} | 소요시간(초)")
print("-" * 60)
for label, kwargs in CASES:
    start = time.time()
    answer_query(**kwargs)
    elapsed = time.time() - start
    print(f"{label:40s} | {elapsed:.2f}")
