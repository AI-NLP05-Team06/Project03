# [1/10] 공통 설정: HCX 모델명, Top-K·최소 유사도 등 검색 기본값, 작업 폴더 경로를 정의합니다.
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

# Dense 코사인 유사도(0~1) 전용 임계값입니다. semantic_search_hcx를 단독으로
# 쓸 때만 의미가 있고, 프로덕션 답변 생성(hybrid_rerank_search 기반)에는
# 쓰지 않습니다 — cross-encoder reranker 점수는 스케일이 전혀 다릅니다.
HCX_RAG_MIN_SCORE = 0.30

# cross-encoder reranker(normalize=True, sigmoid 0~1 출력) 전용 임계값입니다.
# 실측 보정: 실제 관련 있는 질문-근거 쌍은 0.98~0.99, 관련 없는 쌍(다른 주제,
# 또는 금융처럼 들리지만 범위 밖인 질문)은 0.001 미만으로 확 갈립니다.
# 0.1은 그 사이에서 충분한 안전 마진을 두고 잡은 값입니다.
HCX_RAG_MIN_SCORE_RERANK = 0.10

# None이면 6개 업무 전체에서 검색합니다.
# 예: "착오송금 반환 신청"
HCX_RAG_BUSINESS_FUNCTION: str | None = None

if Path("/content").exists():
    # Colab 환경: KDIC_WORK_ROOT 환경변수가 있으면 우선 사용(예: zip을 풀어둔 프로젝트 폴더
    # 안의 rag_baseline/를 그대로 가리키게), 없으면 기존 고정 경로로 폴백합니다.
    WORK_ROOT = Path(
        os.environ.get("KDIC_WORK_ROOT", "/content/kdic_rag_baseline")
    ).resolve()
else:
    # 로컬 환경: KDIC_WORK_ROOT 환경변수로 덮어쓸 수 있고,
    # 지정하지 않으면 현재 폴더 아래에 생성합니다.
    WORK_ROOT = Path(
        os.environ.get("KDIC_WORK_ROOT", "./rag_baseline")
    ).resolve()

EXTRACT_ROOT = WORK_ROOT / "uploaded_output"
RESULT_ROOT = WORK_ROOT / "results"

# results/ 아래 성격별 하위 폴더 (로그 / 문항별 상세 / 도메인별 breakdown / BGE-M3 캐시 데이터 / 일회성 실험)
LOG_ROOT = RESULT_ROOT / "logs"
DETAIL_ROOT = RESULT_ROOT / "detail"
DOMAIN_BREAKDOWN_ROOT = RESULT_ROOT / "domain_breakdown"
BGE_M3_DATA_ROOT = RESULT_ROOT / "bge_m3_embeddings"
EXPERIMENTS_ROOT = RESULT_ROOT / "experiments"

WORK_ROOT.mkdir(parents=True, exist_ok=True)
RESULT_ROOT.mkdir(parents=True, exist_ok=True)
LOG_ROOT.mkdir(parents=True, exist_ok=True)
DETAIL_ROOT.mkdir(parents=True, exist_ok=True)
DOMAIN_BREAKDOWN_ROOT.mkdir(parents=True, exist_ok=True)
BGE_M3_DATA_ROOT.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_ROOT.mkdir(parents=True, exist_ok=True)

print("설정 완료")
print("- Chat 모델:", HCX_CHAT_MODEL)
print("- 질문 임베딩 모델:", HCX_EMBEDDING_MODEL)
print("- Top-K:", HCX_RAG_TOP_K)
print("- 최소 유사도:", HCX_RAG_MIN_SCORE)
print("- 업무 필터:", HCX_RAG_BUSINESS_FUNCTION)
