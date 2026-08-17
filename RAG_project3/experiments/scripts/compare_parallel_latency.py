# [검증용] 복합질의 순차 처리 vs 병렬 처리를 같은 환경/같은 질문으로 직접 비교한다.
# answer_query()의 병렬화(ThreadPoolExecutor)가 실제로 순차보다 빠른지 확인하기 위한 임시 스크립트.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "production"))

import time
from concurrent.futures import ThreadPoolExecutor

from core.config import *
from core.integrity_check import *
from classification.decomposition import decompose_query
from generation.compound_answer import _answer_single

QUESTION = "예금자보호 한도와 그 한도에 이자가 포함되는지 함께 알려주세요."

sub_questions = decompose_query(QUESTION)
print("하위질문:", sub_questions)

# 재정렬 모델을 미리 한 번 워밍업해서(첫 호출의 모델 로딩 시간이 결과를 왜곡하지 않도록)
# 순차/병렬 비교 전에 워밍업 1회를 실행한다.
print("\n워밍업 중...")
_answer_single(sub_questions[0])
print("워밍업 완료\n")

start = time.time()
sequential_answers = [_answer_single(q) for q in sub_questions]
sequential_elapsed = time.time() - start
print(f"순차 처리: {sequential_elapsed:.2f}초")

start = time.time()
with ThreadPoolExecutor(max_workers=len(sub_questions)) as executor:
    parallel_answers = list(executor.map(_answer_single, sub_questions))
parallel_elapsed = time.time() - start
print(f"병렬 처리: {parallel_elapsed:.2f}초")

print(f"\n순차/병렬 비율: {sequential_elapsed / parallel_elapsed:.2f}배")
