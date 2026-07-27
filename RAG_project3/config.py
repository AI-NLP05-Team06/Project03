from __future__ import annotations

import json
import math
import os
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from IPython.display import display
from openai import OpenAI


# ============================================================
# 모델·검색 Baseline 설정
# ============================================================

HCX_BASE_URL = "https://clovastudio.stream.ntruss.com/v1/openai"
HCX_CHAT_MODEL = "HCX-005"
HCX_EMBEDDING_MODEL = "bge-m3"
HCX_EMBEDDING_ENCODING_FORMAT = "float"

HCX_REQUEST_TIMEOUT_SECONDS = 120
HCX_MAX_RETRIES = 4

HCX_RAG_TOP_K = 5
HCX_RAG_MIN_SCORE = 0.30

# None이면 6개 업무 전체에서 검색합니다.
# 예: "착오송금 반환 신청"
HCX_RAG_BUSINESS_FUNCTION: str | None = None

if Path("/content").exists():
    # Colab 환경: 기존 경로를 그대로 사용합니다.
    WORK_ROOT = Path("/content/kdic_rag_baseline")
else:
    # 로컬 환경: KDIC_WORK_ROOT 환경변수로 덮어쓸 수 있고,
    # 지정하지 않으면 현재 폴더 아래에 생성합니다.
    WORK_ROOT = Path(
        os.environ.get("KDIC_WORK_ROOT", "./kdic_rag_baseline")
    ).resolve()

EXTRACT_ROOT = WORK_ROOT / "uploaded_output"
RESULT_ROOT = WORK_ROOT / "results"

WORK_ROOT.mkdir(parents=True, exist_ok=True)
RESULT_ROOT.mkdir(parents=True, exist_ok=True)

print("설정 완료")
print("- Chat 모델:", HCX_CHAT_MODEL)
print("- 질문 임베딩 모델:", HCX_EMBEDDING_MODEL)
print("- Top-K:", HCX_RAG_TOP_K)
print("- 최소 유사도:", HCX_RAG_MIN_SCORE)
print("- 업무 필터:", HCX_RAG_BUSINESS_FUNCTION)
