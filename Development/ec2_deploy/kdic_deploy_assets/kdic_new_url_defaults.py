from __future__ import annotations

"""요구사항 1 (신규 URL 추가) 지원 모듈.

kdic_final_pipeline.py가 필요로 하는 23개 필수 정책 컬럼(REVIEW_COLUMNS)을
관리자가 다 입력하게 하는 대신, 무난한 기본값으로 채운 한 줄을 만들어서
기존 42개 매니페스트에 덧붙인다. "검토_근거" 컬럼에 "자동 기본값, 검수 필요"라고
명시해 두어, 기존 42개(사람이 실제로 검토한 행)와 구분되게 한다.
"""

import re
from typing import Any

import pandas as pd


def _slugify_url_id(source_url: str, existing_ids: set[str]) -> str:
    """새 문서_ID를 만든다. 기존 접두어(DP-, MT-, BI-...)와 안 겹치게 NEW- 접두어 사용."""
    base = re.sub(r"[^0-9]", "", source_url) or "0"
    candidate = f"NEW-{len(existing_ids) + 1:03d}"
    counter = 1
    while candidate in existing_ids:
        counter += 1
        candidate = f"NEW-{len(existing_ids) + counter:03d}"
    return candidate


def build_default_manifest_row(
    *,
    source_url: str,
    business_domain: str,
    page_title: str = "",
    existing_ids: set[str] | None = None,
) -> dict[str, str]:
    """요구사항 1: 관리자가 URL + 업무분류만 입력하면, 나머지 정책 컬럼은
    무난한 기본값으로 채운 한 줄(dict)을 만든다."""
    if not source_url.strip():
        raise ValueError("source_url이 비어 있습니다.")
    if not business_domain.strip():
        raise ValueError("business_domain(업무_도메인)이 비어 있습니다.")

    url_id = _slugify_url_id(source_url, existing_ids or set())
    return {
        "문서_ID": url_id,
        "업무_도메인": business_domain,
        "목표_도메인": business_domain,
        "문서명": page_title or "(자동 크롤링 후 채워짐)",
        "페이지_유형": "static_page",
        "권장_최종결정": "include_full",
        "RAG_본문_인덱싱": "O",
        "인덱싱_범위": "본문 전체 (자동 기본값 - 검수 필요)",
        "Action_링크": "X",
        "권장_Action_Type": "해당 없음",
        "Action_인증": "불필요",
        "다중페이지_수집정책": "불필요",
        "검토_의견": "",
        "검토_근거": "관리자 신규 추가 - 자동 기본값 적용, 적재 전 미리보기 검수 필요",
        "출처_URL": source_url,
        "첨부파일_원본수집": "X",
        "첨부파일_RAG_정책": "none",
        "첨부파일_사용자제공정책": "none",
        "영상_처리정책": "none",
        "웹툰_처리정책": "none",
        "일반이미지_처리정책": "none",
        "보조콘텐츠_표시조건": "metadata_only_nondecorative",
        "보조콘텐츠_링크라벨": "",
    }


def append_new_url_to_manifest(
    existing_manifest_df: pd.DataFrame, new_row: dict[str, str]
) -> tuple[pd.DataFrame, str]:
    """기존 review CSV(DataFrame)에 새 행을 덧붙인 새 DataFrame과 url_id를 반환."""
    if new_row["문서_ID"] in set(existing_manifest_df["문서_ID"]):
        raise ValueError(f"이미 존재하는 문서_ID: {new_row['문서_ID']}")
    updated = pd.concat(
        [existing_manifest_df, pd.DataFrame([new_row])], ignore_index=True
    )
    return updated, new_row["문서_ID"]
