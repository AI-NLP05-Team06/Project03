from __future__ import annotations

import hashlib
import json
import mimetypes
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment, NavigableString, Tag
from requests.adapters import HTTPAdapter
from tqdm.auto import tqdm
from urllib3.util.retry import Retry

KST = timezone(timedelta(hours=9))


# ============================================================
# 설정
# ============================================================

@dataclass
class PipelineConfig:
    select_business_functions: list[str] | None = None
    run_only_url_ids: list[str] | None = None
    max_urls: int | None = None

    request_delay_seconds: float = 1.2
    request_timeout_seconds: int = 30
    max_retries: int = 3
    max_asset_bytes: int = 50 * 1024 * 1024

    respect_robots_txt: bool = True
    enable_generic_pagination: bool = True
    pagination_max_pages: int = 100
    pagination_page_size: int = 10

    download_direct_attachments: bool = True
    download_images: bool = False
    download_js_attachments_with_playwright: bool = True
    playwright_download_timeout_ms: int = 60_000

    # 이전 실행과 첨부파일 변경 비교. 경로를 지정하지 않으면 같은 runtime_root의
    # 가장 최근 kdic_crawl_output_* manifest를 자동 탐색합니다.
    previous_attachment_manifest_path: str | None = None
    auto_detect_previous_attachment_manifest: bool = True

    # 영상·웹툰은 원본 파일을 내려받지 않고 공식 게시 페이지 링크만 제공합니다.
    collect_supplementary_media_links: bool = True
    download_videos: bool = False
    download_webtoons: bool = False

    # 첨부파일 하이브리드 정책
    extract_attachment_text: bool = True
    create_attachment_documents: bool = True
    attachment_index_min_chars: int = 120
    attachment_max_text_chars: int = 1_000_000
    attachment_pdf_max_pages: int = 300
    attachment_keep_failed_metadata: bool = True
    attachment_mark_scanned_pdf_for_ocr: bool = True
    attachment_unknown_long_text_auto_index: bool = False

    # 설명·목록 첨부파일 전용 청킹. 웹 본문보다 보수적으로 자릅니다.
    attachment_chunk_max_chars: int = 1200
    attachment_list_rows_per_chunk: int = 8

    enable_playwright_snapshot: bool = False
    playwright_wait_ms: int = 1500

    create_chunks: bool = True
    chunk_max_chars: int = 1600
    min_chunk_chars: int = 80

    user_agent: str = (
        "KDIC-RAG-Academic-Crawler/1.0 "
        "(low-rate public-document collection; no authentication automation)"
    )

    def __post_init__(self) -> None:
        self.select_business_functions = self.select_business_functions or []
        self.run_only_url_ids = self.run_only_url_ids or []


@dataclass
class OutputPaths:
    root: Path
    raw_html: Path
    raw_assets: Path
    response_meta: Path
    parsed_markdown: Path
    parsed_json: Path
    parsed_attachments: Path
    pagination: Path
    dynamic_diagnostics: Path
    screenshots: Path
    logs: Path
    processed: Path
    manifest: Path
    quality: Path


def create_output_paths(runtime_root: Path) -> OutputPaths:
    run_id = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    root = runtime_root / f"kdic_crawl_output_{run_id}"
    paths = OutputPaths(
        root=root,
        raw_html=root / "raw" / "html",
        raw_assets=root / "raw" / "assets",
        response_meta=root / "raw" / "response_meta",
        parsed_markdown=root / "parsed" / "markdown",
        parsed_json=root / "parsed" / "json",
        parsed_attachments=root / "parsed" / "attachments",
        pagination=root / "diagnostics" / "pagination",
        dynamic_diagnostics=root / "diagnostics" / "dynamic",
        screenshots=root / "diagnostics" / "screenshots",
        logs=root / "logs",
        processed=root / "processed",
        manifest=root / "manifest",
        quality=root / "quality",
    )
    for value in asdict(paths).values():
        Path(value).mkdir(parents=True, exist_ok=True)
    return paths


# ============================================================
# 공통 유틸리티
# ============================================================

ATTACHMENT_EXTENSIONS = {
    ".pdf", ".hwp", ".hwpx", ".doc", ".docx", ".xls", ".xlsx",
    ".ppt", ".pptx", ".zip", ".csv", ".txt",
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".avi", ".mkv"}

NOISE_SELECTORS = [
    "script", "style", "noscript", "template", ".floatTop", ".floatBottom",
    ".paging", ".pagination", ".pageNavi", ".page_navi", ".sr-only",
    ".skip", ".skipnav", ".skip-navigation", ".loading", ".waitPage",
    "#waitPage", ".pdfViewerGuide", ".adobe-reader", ".tabMenu.clone",
]

NOISE_EXACT_TEXTS = {
    "퀵메뉴", "글자 크기", "글자확대", "글자축소", "KOR", "ENG",
    "상단으로 이동", "검색 초기화", "좌우로 움직여보세요", "표 더보기",
    "계산하기", "닫기", "Adobe Reader 바로가기",
    "영상내용입니다.", "영상 내용입니다.", "제도 소개 안내영상입니다.",
}

NOISE_CONTAINS_TEXTS = {
    "Adobe Reader", "영상내용입니다", "제도 소개 안내영상입니다",
}

NOISE_REGEXES = [
    re.compile(r"PDF\s*파일\s*내용이\s*보이지\s*않으시면.*?설치하시기\s*바랍니다[.。]?", re.I),
    re.compile(r"PDF파일\s*내용이\s*보이지\s*않으시면.*?설치하시기\s*바랍니다[.。]?", re.I),
]

ALLOWED_ACTION_DOMAIN_SUFFIXES = {
    "kdic.or.kr", "fins.kdic.or.kr", "ccrs.or.kr", "scourt.go.kr",
    "klac.or.kr", "law.go.kr", "easylaw.go.kr", "gov.kr",
}

ACTION_KEYWORDS = {
    "신청", "바로가기", "조회", "검색", "자가진단", "상담", "진행상황", "접수",
    "신고", "발급", "환급", "위치", "찾아오시는 길", "법령", "규정",
    "다운로드", "전화문의", "문의",
}

DOWNLOAD_CALL_RE = re.compile(
    r"gfn_downloadFile\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"]([^'\"]+)['\"]\s*\)"
)
URL_IN_JS_PATTERNS = [
    re.compile(r"gfn_openUrl\(\s*['\"]([^'\"]+)['\"]\s*\)"),
    re.compile(r"gfn_moveUrl\(\s*['\"]([^'\"]+)['\"]\s*\)"),
    re.compile(r"window\.open\(\s*['\"]([^'\"]+)['\"]"),
    re.compile(r"(?:window\.)?location(?:\.href)?\s*=\s*['\"]([^'\"]+)['\"]"),
    re.compile(r"location\.replace\(\s*['\"]([^'\"]+)['\"]\s*\)"),
]

BLOCK_TAGS = {
    "div", "section", "article", "aside", "header", "footer", "main",
    "p", "ul", "ol", "dl", "table", "figure", "h1", "h2", "h3",
    "h4", "h5", "h6",
}


def now_kst_iso() -> str:
    return datetime.now(KST).isoformat(timespec="seconds")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def norm(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def safe_name(value: str, max_length: int = 100) -> str:
    value = re.sub(r'[\\/:*?"<>|]+', "_", norm(value))
    value = re.sub(r"_+", "_", value).strip("._ ")
    return (value or "untitled")[:max_length]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(data, ensure_ascii=False) + "\n")


def absolute_url(base_url: str, candidate: str | None, allow_contact: bool = False) -> str | None:
    if not candidate:
        return None
    candidate = candidate.strip()
    if not candidate or candidate.startswith(("#", "javascript:")):
        return None
    if candidate.startswith(("tel:", "mailto:")):
        return candidate if allow_contact else None
    return urljoin(base_url, candidate)


def is_noise_text(text: str) -> bool:
    value = norm(text)
    if not value:
        return True
    if value in NOISE_EXACT_TEXTS:
        return True
    if any(pattern.search(value) for pattern in NOISE_REGEXES):
        return True
    return any(fragment in value for fragment in NOISE_CONTAINS_TEXTS)


def clean_visible_text(text: str) -> str:
    value = norm(text)
    for pattern in NOISE_REGEXES:
        value = norm(pattern.sub(" ", value))
    for phrase in ["검색 초기화", "Adobe Reader 바로가기", "계산하기", "닫기"]:
        value = norm(value.replace(phrase, " "))
    for fragment in NOISE_CONTAINS_TEXTS:
        if fragment in value:
            value = norm(value.replace(fragment, " "))
    return value


def row_to_dict(row: pd.Series) -> dict[str, str]:
    return {str(k): "" if pd.isna(v) else str(v) for k, v in row.to_dict().items()}


def unique_dicts(items: Iterable[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for item in items:
        key = tuple(item.get(name) for name in keys)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


# ============================================================
# 42개 검토 CSV → 실행 Manifest
# ============================================================

REVIEW_COLUMNS = {
    "문서_ID", "업무_도메인", "목표_도메인", "문서명", "페이지_유형",
    "권장_최종결정", "RAG_본문_인덱싱", "인덱싱_범위", "Action_링크",
    "권장_Action_Type", "Action_인증", "다중페이지_수집정책", "검토_의견",
    "검토_근거", "출처_URL",
    "첨부파일_원본수집", "첨부파일_RAG_정책", "첨부파일_사용자제공정책",
    "영상_처리정책", "웹툰_처리정책", "일반이미지_처리정책",
    "보조콘텐츠_표시조건", "보조콘텐츠_링크라벨",
}

RUN_DECISIONS = {
    "include_full", "include_reference", "include_full_public", "include_partial",
    "include_partial_action", "include_full_filtered", "include_separate_domain",
    "link_only",
}


def _site_name(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    return "금융안심포털" if host.startswith("fins.") else "예금보험공사"


def _page_type(row: pd.Series) -> str:
    doc_id = row["문서_ID"]
    title = row["문서명"]
    policy = row["다중페이지_수집정책"]
    decision = row["권장_최종결정"]
    raw = row.get("페이지_유형", "")
    if decision == "link_only":
        return "link_only"
    if "FAQ" in title.upper() or "FAQ" in policy.upper():
        return "faq"
    if doc_id == "BI-004":
        return "dynamic_lookup"
    if "상호작용" in policy or "인증 서비스" in policy or "신청 서비스" in policy:
        return "interactive_service"
    return raw or "static_page"


def _fetch_strategy(row: pd.Series) -> str:
    decision = row["권장_최종결정"]
    policy = row["다중페이지_수집정책"]
    action = row["Action_링크"]
    if decision == "link_only":
        return "link_only"
    if "필수" in policy or "자동탐지" in policy:
        return "paginated_public"
    if "다운로드" in action:
        return "requests_html_assets"
    return "requests_html"


def _index_mode(value: str) -> str:
    if value == "O":
        return "full"
    if "공개 설명만" in value:
        return "public_only"
    if "보조 인덱스" in value:
        return "reference"
    if "일부 제외" in value:
        return "filtered"
    if "별도 도메인" in value:
        return "separate_domain"
    return "none"


def normalize_review_manifest(review_df: pd.DataFrame) -> pd.DataFrame:
    review_df = review_df.fillna("").astype(str)
    missing = sorted(REVIEW_COLUMNS - set(review_df.columns))
    if missing:
        raise ValueError(f"42개 검토 CSV 필수 열 누락: {missing}")

    records: list[dict[str, str]] = []
    for _, row in review_df.iterrows():
        url = row["출처_URL"]
        decision = row["권장_최종결정"]
        if decision not in RUN_DECISIONS:
            raise ValueError(f"지원하지 않는 결정: {row['문서_ID']}={decision}")
        page_type = _page_type(row)
        action_auth = row["Action_인증"]
        records.append({
            "url_id": row["문서_ID"],
            "business_function": row["업무_도메인"],
            "target_business_function": row["목표_도메인"] or row["업무_도메인"],
            "page_title": row["문서명"],
            "source_url": url,
            "site_name": _site_name(url),
            "normalized_decision": decision,
            "decision_reason": row["검토_의견"],
            "review_basis": row["검토_근거"],
            "page_type": page_type,
            "fetch_strategy": _fetch_strategy(row),
            "parser_profile": "kdic_final_integrated_v4_5",
            "auth_required": action_auth,
            "asset_policy": row["Action_링크"],
            "content_scope": row["인덱싱_범위"],
            "crawl_wave": (
                "META" if decision == "link_only" else
                "W2_DYNAMIC" if ("필수" in row["다중페이지_수집정책"] or row["문서_ID"] == "BI-004") else
                "W1_ASSET" if "다운로드" in row["Action_링크"] else
                "W1_STATIC"
            ),
            "rag_index_mode": _index_mode(row["RAG_본문_인덱싱"]),
            "rag_index_label": row["RAG_본문_인덱싱"],
            "action_policy": row["Action_링크"],
            "expected_action_types": row["권장_Action_Type"],
            "action_auth": action_auth,
            "pagination_policy": row["다중페이지_수집정책"],
            "attachment_download_policy": row["첨부파일_원본수집"],
            "attachment_rag_policy": row["첨부파일_RAG_정책"],
            "attachment_user_delivery_policy": row["첨부파일_사용자제공정책"],
            "video_policy": row["영상_처리정책"],
            "webtoon_policy": row["웹툰_처리정책"],
            "image_policy": row["일반이미지_처리정책"],
            "supplementary_display_condition": row["보조콘텐츠_표시조건"],
            "supplementary_link_label": row["보조콘텐츠_링크라벨"],
        })

    manifest = pd.DataFrame(records)
    if manifest["url_id"].duplicated().any():
        raise ValueError("Manifest url_id 중복")
    if len(manifest) < 1:
        raise ValueError("검토표에 최소 1개 이상의 URL이 있어야 합니다.")
    # 원래는 "정확히 42개"만 허용했음(공식 정책표 42건 고정 가정). 관리자 화면에서
    # 신규 URL을 42건 매니페스트에 추가해 개별 실행(run_only_url_ids)할 수 있도록
    # 개수 제한을 최소 1건으로 완화함 -- url_id 중복 방지는 위에서 이미 검증됨.
    return manifest



# ============================================================
# 공통 표시 제목 및 Action 라벨 정규화
# ============================================================

GENERIC_PAGE_HEADINGS = {
    "top 10", "top10", "개요", "안내", "유의사항", "구비서류안내",
    "구비서류 안내", "신고센터", "faq", "자주 묻는 질문",
}

GENERIC_BREADCRUMB_PARTS = {
    "고객센터", "faq", "소개와 방법안내", "소개와 신청방법 안내",
    "반환지원 신청하기", "지원자금관리", "부실책임조사", "채무조정 재기지원",
}

DOMAIN_DISPLAY_NAMES = {
    "예금자보호제도": "예금자보호제도",
    "예금보험금 안내": "예금보험금",
    "고객 미수령금 신청": "고객 미수령금",
    "착오송금 반환 신청": "착오송금 반환지원",
    "채무조정 안내": "채무조정",
    "은닉재산 신고": "은닉재산 신고",
}


def normalize_title_phrase(value: str) -> str:
    value = norm(value)
    value = re.sub(r"\([^)]*사이트 분류상 명칭[^)]*\)", "", value)
    replacements = {
        "착오송금반환지원": "착오송금 반환지원",
        "착오송금수취인": "착오송금 수취인",
        "은닉재산신고": "은닉재산 신고",
        "구비서류안내": "구비서류 안내",
        "채무정보조회": "채무정보조회",
        "미수령금통합신청": "미수령금 통합신청",
    }
    for source, target in replacements.items():
        value = value.replace(source, target)
    return norm(value)


def split_manifest_breadcrumb(title: str) -> list[str]:
    return [normalize_title_phrase(part) for part in re.split(r"\s*>\s*", title or "") if normalize_title_phrase(part)]


def is_generic_heading(value: str) -> bool:
    return normalize_title_phrase(value).lower() in GENERIC_PAGE_HEADINGS


def resolve_display_title(manifest_row: dict[str, str], page_heading: str = "") -> str:
    """TOP 10·개요·유의사항처럼 일반적인 제목을 업무 맥락이 있는 제목으로 바꿉니다."""
    manifest_title = manifest_row.get("page_title", "")
    parts = split_manifest_breadcrumb(manifest_title)
    heading = normalize_title_phrase(page_heading)
    business = DOMAIN_DISPLAY_NAMES.get(
        manifest_row.get("business_function", ""),
        normalize_title_phrase(manifest_row.get("business_function", "")),
    )
    page_type = manifest_row.get("page_type", "")

    if manifest_row.get("normalized_decision") == "link_only" and "self_check" in manifest_row.get("expected_action_types", ""):
        return norm(f"{business} 신청대상 자가진단")

    if page_type == "faq":
        leaf = parts[-1] if parts else ""
        leaf_lower = leaf.lower()
        if "사이트 분류상 명칭" in manifest_title:
            subject = business
            suffix = "FAQ"
        elif leaf_lower in {"top 10", "top10"}:
            subject = business
            suffix = "주요 FAQ"
        else:
            subject = leaf
            subject = re.sub(r"FAQ$", "", subject, flags=re.I).strip()
            if subject.endswith("신청") and "반환지원" in subject:
                subject = subject[:-2].strip()
            if not subject or is_generic_heading(subject):
                subject = business
            suffix = "FAQ"
        title = f"{subject} {suffix}"
        return norm(title)

    if heading and not is_generic_heading(heading):
        return heading[:100]

    meaningful = [part for part in parts if part.lower() not in GENERIC_BREADCRUMB_PARTS]
    generic = heading or (meaningful[-1] if meaningful else "")

    if generic in {"개요"} and len(meaningful) >= 2:
        return norm(" ".join(meaningful[-3:]))[:100]
    if generic in {"유의사항"}:
        subject = meaningful[-2] if len(meaningful) >= 2 else business
        return norm(f"{subject} 유의사항")[:100]
    if generic in {"구비서류 안내", "구비서류안내"}:
        # 제목 뒤에 대상(착오송금인/수취인)이 있는 구조까지 공통 처리
        subject_candidates = [p for p in meaningful if p not in {"구비서류 안내", "구비서류안내"}]
        subject = subject_candidates[-1] if subject_candidates else business
        return norm(f"{subject} 구비서류 안내")[:100]
    if generic == "신고센터":
        subject = meaningful[-2] if len(meaningful) >= 2 else business
        return norm(f"{subject} 신고센터")[:100]
    if heading and heading not in {"안내"}:
        return heading[:100]
    if meaningful:
        return norm(" ".join(meaningful[-3:]))[:100]
    return business or normalize_title_phrase(manifest_title)


def compact_action_label(raw_label: str, action_type: str, manifest_row: dict[str, str]) -> tuple[str, str]:
    """버튼 카드 전체 문구가 라벨로 붙는 경우 사용자용 라벨을 짧게 만듭니다."""
    raw = norm(raw_label)
    cleaned = clean_visible_text(raw.replace("예금보험공사 핵심가치", " "))
    business = DOMAIN_DISPLAY_NAMES.get(
        manifest_row.get("business_function", ""),
        normalize_title_phrase(manifest_row.get("business_function", "")),
    )
    patterns = [
        ("온라인 신청", f"{business} 온라인 신청"),
        ("방문 신청", f"{business} 방문 신청 위치"),
        ("자가진단", f"{resolve_display_title(manifest_row)} 자가진단"),
        ("진행현황", f"{resolve_display_title(manifest_row)} 진행현황 조회"),
    ]
    for token, replacement in patterns:
        if token in cleaned:
            return norm(replacement), raw
    if len(cleaned) > 80:
        first = re.split(r"\s+(?:접속방법|준비물|연락처|주소)\s*:", cleaned, maxsplit=1)[0]
        first = re.sub(r"\s*\([^)]*\)", "", first)
        cleaned = norm(first)
    if not cleaned:
        cleaned = f"{resolve_display_title(manifest_row)} 바로가기"
    return cleaned[:80], raw

# ============================================================
# HTTP 수집
# ============================================================

@dataclass
class FetchResult:
    requested_url: str
    final_url: str
    status_code: int
    ok: bool
    content_type: str
    encoding: str
    elapsed_seconds: float
    collected_at: str
    content_length: int
    raw_sha256: str
    redirect_history: list[dict[str, Any]]
    selected_headers: dict[str, str]
    body: bytes


def create_session(config: PipelineConfig) -> requests.Session:
    retry = Retry(
        total=config.max_retries,
        connect=config.max_retries,
        read=config.max_retries,
        status=config.max_retries,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "POST", "HEAD"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=2, pool_maxsize=2)
    session = requests.Session()
    session.headers.update({
        "User-Agent": config.user_agent,
        "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.5",
    })
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def choose_encoding(response: requests.Response) -> str:
    encoding = response.encoding
    if not encoding or encoding.lower() in {"iso-8859-1", "ascii"}:
        encoding = response.apparent_encoding or "utf-8"
    return encoding


def robots_allows(url: str, config: PipelineConfig) -> tuple[bool, str]:
    if not config.respect_robots_txt:
        return True, "disabled"
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    try:
        parser = RobotFileParser()
        parser.set_url(robots_url)
        parser.read()
        allowed = parser.can_fetch(config.user_agent, url)
        return allowed, "allowed" if allowed else "disallowed"
    except Exception as error:
        return True, f"unavailable:{type(error).__name__}"


def response_to_fetch_result(response: requests.Response, requested_url: str, elapsed: float) -> FetchResult:
    encoding = choose_encoding(response)
    response.encoding = encoding
    selected = {
        key: value for key, value in response.headers.items()
        if key in {"Content-Type", "Content-Length", "Last-Modified", "ETag", "Cache-Control", "Content-Disposition"}
    }
    return FetchResult(
        requested_url=requested_url,
        final_url=response.url,
        status_code=response.status_code,
        ok=response.ok,
        content_type=response.headers.get("Content-Type", ""),
        encoding=encoding,
        elapsed_seconds=round(elapsed, 3),
        collected_at=now_kst_iso(),
        content_length=len(response.content),
        raw_sha256=sha256_bytes(response.content),
        redirect_history=[
            {"status_code": item.status_code, "url": item.url, "location": item.headers.get("Location")}
            for item in response.history
        ],
        selected_headers=selected,
        body=response.content,
    )


def fetch_url(session: requests.Session, url: str, config: PipelineConfig) -> FetchResult:
    allowed, status = robots_allows(url, config)
    if not allowed:
        raise PermissionError(f"robots.txt에서 허용하지 않음: {url}")
    started = time.perf_counter()
    response = session.get(url, timeout=config.request_timeout_seconds, allow_redirects=True)
    result = response_to_fetch_result(response, url, time.perf_counter() - started)
    result.selected_headers["Robots-Check"] = status
    return result


def save_fetch_result(paths: OutputPaths, row: dict[str, str], result: FetchResult) -> tuple[Path, Path]:
    folder = safe_name(row["business_function"])
    html_path = paths.raw_html / folder / f"{row['url_id']}.html"
    meta_path = paths.response_meta / folder / f"{row['url_id']}.json"
    ensure_parent(html_path)
    html_path.write_bytes(result.body)
    meta = asdict(result)
    meta.pop("body", None)
    meta.update({
        "url_id": row["url_id"],
        "business_function": row["business_function"],
        "page_title_manifest": row["page_title"],
        "fetch_strategy": row["fetch_strategy"],
        "parser_profile": row["parser_profile"],
        "raw_html_path": str(html_path.relative_to(paths.root)),
    })
    write_json(meta_path, meta)
    return html_path, meta_path


# ============================================================
# 결정론적 HTML 파싱
# ============================================================

def safe_text(node: Tag) -> str:
    fragment = BeautifulSoup(str(node), "lxml")
    root = fragment.body.find() if fragment.body else fragment.find()
    if not root:
        return ""
    for child in root.select(
        ".sr-only,.hide,script,style,noscript,template,.floatTop,.floatBottom,.paging,.pagination"
    ):
        child.decompose()
    for image in root.find_all("img"):
        image.replace_with(norm(image.get("alt", "")))
    for br in root.find_all("br"):
        br.replace_with(" ")
    return clean_visible_text(root.get_text(" ", strip=True))


def infer_file_format(node: Tag) -> str:
    text = (" ".join(node.get("class", [])).lower() + " " + norm(node.get_text(" ", strip=True)).lower())
    for keyword, fmt in [
        ("icohwp", "HWP"), ("hwp", "HWP"), ("icopdf", "PDF"), ("pdf", "PDF"),
        ("xlsx", "XLSX"), ("excel", "XLSX"), ("docx", "DOCX"), ("doc", "DOC"),
    ]:
        if keyword in text:
            return fmt
    return "FILE"


def cell_text(cell: Tag) -> str:
    fragment = BeautifulSoup(str(cell), "lxml")
    root = fragment.body.find() if fragment.body else fragment.find()
    if not root:
        return ""
    for child in root.select("script,style,noscript,template,.sr-only"):
        child.decompose()
    for button in root.find_all("button"):
        fmt = infer_file_format(button)
        label = re.sub(r"다운로드$", "", norm(button.get_text(" ", strip=True))).strip()
        button.replace_with(f"{label} {fmt} 다운로드".strip())
    for image in root.find_all("img"):
        image.replace_with(norm(image.get("alt", "")))
    for br in root.find_all("br"):
        br.replace_with(" ")
    return norm(root.get_text(" ", strip=True))


def expand_table(table: Tag) -> tuple[list[str], list[list[str]]]:
    grid: list[list[str]] = []
    active: dict[int, list[Any]] = {}
    header_flags: list[bool] = []
    max_columns = 0

    for tr in table.find_all("tr"):
        row: list[str] = []
        column = 0
        cells = tr.find_all(["th", "td"], recursive=False)

        def fill_active() -> None:
            nonlocal column
            while column in active:
                remaining, value = active[column]
                while len(row) <= column:
                    row.append("")
                row[column] = value
                remaining -= 1
                if remaining <= 0:
                    del active[column]
                else:
                    active[column] = [remaining, value]
                column += 1

        fill_active()
        header_flags.append(any(cell.name == "th" for cell in cells))
        for cell in cells:
            fill_active()
            text = cell_text(cell)
            rowspan = max(1, int(cell.get("rowspan", 1) or 1))
            colspan = max(1, int(cell.get("colspan", 1) or 1))
            for offset in range(colspan):
                target = column + offset
                while len(row) <= target:
                    row.append("")
                row[target] = text
                if rowspan > 1:
                    active[target] = [rowspan - 1, text]
            column += colspan
        if active:
            final_col = max(active)
            while column <= final_col:
                if column in active:
                    fill_active()
                else:
                    while len(row) <= column:
                        row.append("")
                    column += 1
        max_columns = max(max_columns, len(row))
        if row and any(row):
            grid.append(row)

    if not grid:
        return [], []
    for row in grid:
        row.extend([""] * (max_columns - len(row)))
    if table.thead:
        header_count = len(table.thead.find_all("tr", recursive=False))
    else:
        header_count = 1 if header_flags and header_flags[0] else 0
    header_count = max(1, header_count)
    header_rows = grid[:header_count]
    rows = grid[header_count:]
    headers: list[str] = []
    for col in range(max_columns):
        values: list[str] = []
        for hrow in header_rows:
            value = norm(hrow[col])
            if value and value not in values:
                values.append(value)
        headers.append(" / ".join(values) if values else f"열{col + 1}")
    counts: dict[str, int] = {}
    unique_headers: list[str] = []
    for header in headers:
        counts[header] = counts.get(header, 0) + 1
        unique_headers.append(header if counts[header] == 1 else f"{header} {counts[header]}")
    return unique_headers, rows




def classify_action_type(label: str, expected_types: str = "") -> str:
    text = norm(label)
    tests = [
        ("자가진단", "self_check"), ("진행상황", "progress"), ("신고", "report"),
        ("상담", "consult"), ("부채증명", "issue"), ("발급", "issue"),
        ("환급", "refund"), ("법령", "legal_reference"), ("규정", "legal_reference"),
        ("위치", "location"), ("찾아오시는", "location"), ("전화", "contact"),
        ("문의", "contact"), ("조회", "lookup"), ("검색", "lookup"),
        ("다운로드", "download"),
        ("신청", "apply"), ("접수", "apply"),
    ]
    for keyword, action_type in tests:
        if keyword in text:
            return action_type
    return "related_service"


def allowed_action_domain(url: str) -> bool:
    if url.startswith(("tel:", "mailto:")):
        return True
    host = (urlparse(url).hostname or "").lower()
    return any(host == suffix or host.endswith("." + suffix) for suffix in ALLOWED_ACTION_DOMAIN_SUFFIXES)


def _extract_js_url(onclick: str) -> str | None:
    for pattern in URL_IN_JS_PATTERNS:
        match = pattern.search(onclick or "")
        if match:
            return match.group(1)
    direct = re.search(r"https?://[^'\"\s)]+", onclick or "")
    return direct.group(0) if direct else None


def action_label_with_context(node: Tag, label: str) -> str:
    label = norm(label)
    if label not in {"바로가기", "신청", "조회", "다운로드"}:
        return label
    parent = node.find_parent(["div", "li", "td", "section", "article"])
    if parent:
        context_node = parent.find(["strong", "h2", "h3", "h4", "dt"], recursive=True)
        context = norm(context_node.get_text(" ", strip=True)) if context_node else ""
        if context and context != label:
            return f"{context} {label}"
    previous = node.find_previous(["strong", "h2", "h3", "h4", "dt"])
    context = norm(previous.get_text(" ", strip=True)) if previous else ""
    return f"{context} {label}".strip() if context else label


def extract_actions(
    root: Tag,
    base_url: str,
    manifest_row: dict[str, str],
    attachments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    expected = manifest_row.get("expected_action_types", "")
    expected_set = {value.strip() for value in expected.split("|") if value.strip()}
    action_policy = manifest_row.get("action_policy", "")
    if action_policy.startswith("X"):
        return []
    seen: set[tuple[str, str]] = set()

    def add(label: str, url: str, source_tag: str, js_function: str = "") -> None:
        raw_label = norm(label)
        if not raw_label or is_noise_text(raw_label):
            return
        if any(noise in raw_label for noise in ["디지털역사관", "예솜이", "챗봇", "앱 설치", "공식 홈페이지"]):
            return
        action_like = any(keyword in raw_label for keyword in ACTION_KEYWORDS)
        if not action_like:
            return
        action_type = classify_action_type(raw_label, expected)
        label, raw_label = compact_action_label(raw_label, action_type, manifest_row)
        host = (urlparse(url).hostname or "").lower()
        if host == "law.go.kr" or host.endswith(".law.go.kr"):
            action_type = "legal_reference"
        if expected_set and action_type not in expected_set and action_type not in {"download", "legal_reference", "location", "contact"}:
            return
        key = (url, label)
        if key in seen:
            return
        seen.add(key)
        actions.append({
            "action_id": sha256_text(f"{manifest_row['url_id']}|{url}|{label}")[:16],
            "label": label,
            "raw_label": raw_label if raw_label != label else "",
            "url": url,
            "action_type": action_type,
            "source_tag": source_tag,
            "javascript_function": js_function,
            "auth_required": manifest_row.get("action_auth", manifest_row.get("auth_required", "")),
            "official_domain": allowed_action_domain(url),
            "requires_review": not allowed_action_domain(url),
        })

    for anchor in root.find_all("a", href=True):
        href = anchor.get("href", "").strip()
        url = absolute_url(base_url, href, allow_contact=True)
        if not url:
            continue
        label = norm(anchor.get_text(" ", strip=True)) or norm(anchor.get("title", ""))
        label = action_label_with_context(anchor, label)
        if any(keyword in label for keyword in ACTION_KEYWORDS):
            add(label, url, "a")

    for node in root.find_all(["button", "a", "input"], onclick=True):
        onclick = node.get("onclick", "")
        candidate = _extract_js_url(onclick)
        if not candidate:
            continue
        url = absolute_url(base_url, candidate, allow_contact=True)
        if not url:
            continue
        label = norm(node.get_text(" ", strip=True)) or norm(node.get("value", "")) or norm(node.get("title", ""))
        label = action_label_with_context(node, label)
        function_name = onclick.split("(", 1)[0].strip()
        add(label, url, node.name, function_name)

    # 다운로드도 챗봇 Action으로 제공할 수 있도록 연결합니다.
    for attachment in attachments:
        url = attachment.get("url") or base_url
        label = attachment.get("display_name", "첨부파일 다운로드")
        key = (url, label)
        if key in seen:
            continue
        seen.add(key)
        actions.append({
            "action_id": attachment["attachment_id"],
            "label": label,
            "url": url,
            "action_type": "download",
            "source_tag": "attachment",
            "attachment_id": attachment["attachment_id"],
            "download_method": attachment["download_method"],
            "auth_required": "불필요",
            "official_domain": True,
            "requires_review": False,
        })

    return actions


GLOBAL_UI_LINK_TEXTS = {
    "창립 30주년 예금보험공사 디지털역사관 바로가기",
    "KDIC(예금보험공사) 공식 홈페이지",
    "상단으로 이동",
}


def is_global_ui_link(anchor: Tag, url: str, text: str) -> bool:
    """본문 아래 고정 배너·퀵메뉴·공통 이동 링크를 업무 링크에서 제외합니다."""
    if text in GLOBAL_UI_LINK_TEXTS:
        return True
    lowered = f"{url} {text}".lower()
    if any(token in lowered for token in ["kdic30th.kr", "디지털역사관", "예솜이", "챗봇"]):
        return True
    for parent in [anchor, *list(anchor.parents)[:6]]:
        if not isinstance(parent, Tag):
            continue
        tokens = " ".join(parent.get("class", [])).lower() + " " + norm(parent.get("id", "")).lower()
        if any(token in tokens for token in [
            "floattop", "floatbottom", "quickmenu", "quick-menu", "floating",
            "snslist", "share", "site-move", "sitemove", "thirty",
        ]):
            return True
    return False


def extract_links(root: Tag, base_url: str) -> list[dict[str, Any]]:
    links: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for anchor in root.find_all("a", href=True):
        url = absolute_url(base_url, anchor.get("href"))
        text = norm(anchor.get_text(" ", strip=True))
        if not url or not text or is_noise_text(text) or is_global_ui_link(anchor, url, text):
            continue
        key = (url, text)
        if key in seen:
            continue
        seen.add(key)
        links.append({
            "url": url,
            "anchor_text": text,
            "link_role": "content",
            "source_tag": "a",
        })
    return links


def merge_links_with_actions(
    content_links: list[dict[str, Any]],
    actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """JSON의 links에서도 버튼형 Action URL을 한 번에 확인할 수 있게 합칩니다."""
    merged = list(content_links)
    seen = {(item.get("url"), item.get("anchor_text")) for item in merged}
    for action in actions:
        url = action.get("url")
        label = action.get("label")
        if not url or not label:
            continue
        key = (url, label)
        if key in seen:
            continue
        seen.add(key)
        merged.append({
            "url": url,
            "anchor_text": label,
            "link_role": "action",
            "action_type": action.get("action_type", "related_service"),
            "source_tag": action.get("source_tag", ""),
            "action_id": action.get("action_id", ""),
        })
    return merged


def _looks_decorative_image(image: Tag, filename: str, alt: str) -> bool:
    lowered = " ".join([
        filename.lower(), alt.lower(),
        " ".join(image.get("class", [])).lower(),
        norm(image.get("id", "")).lower(),
    ])
    decorative_tokens = {
        "logo", "icon", "ico_", "bullet", "btn_", "button", "banner",
        "bg_", "background", "qr", "character", "mascot", "예솜이",
    }
    return any(token in lowered for token in decorative_tokens)


def extract_videos(root: Tag, base_url: str, manifest_row: dict[str, str]) -> list[dict[str, Any]]:
    """영상 URL은 관리용 메타데이터로만 저장하고 다운로드·인덱싱하지 않습니다."""
    videos: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(media_url: str | None, poster_url: str | None, source_tag: str, label: str) -> None:
        url = absolute_url(base_url, media_url)
        if not url or url in seen:
            return
        seen.add(url)
        videos.append({
            "video_id": sha256_text(f"{manifest_row['url_id']}|{url}")[:16],
            "label": label or f"{manifest_row['page_title']} 영상",
            "media_url": url,
            "poster_url": absolute_url(base_url, poster_url),
            "source_page_url": manifest_row["source_url"],
            "source_tag": source_tag,
            "indexable": False,
            "download": False,
            "content_role": "supplementary",
            "delivery_mode": "official_page",
        })

    for video in root.find_all("video"):
        label = norm(video.get("title", "")) or norm(video.get("aria-label", ""))
        add(video.get("src"), video.get("poster"), "video", label)
        for source_node in video.find_all("source", src=True):
            add(source_node.get("src"), video.get("poster"), "video/source", label)

    for iframe in root.find_all("iframe", src=True):
        src = iframe.get("src", "")
        lowered = src.lower()
        if any(token in lowered for token in ["youtube", "youtu.be", "vimeo", ".mp4", ".webm"]):
            label = norm(iframe.get("title", ""))
            add(src, None, "iframe", label)

    return videos


def _has_webtoon(root: Tag) -> bool:
    for image in root.find_all("img"):
        value = " ".join([
            image.get("src", ""), image.get("data-src", ""),
            image.get("alt", ""), image.get("title", ""),
        ]).lower()
        if "webtoon" in value or "웹툰" in value:
            return True
    return "웹툰" in norm(root.get_text(" ", strip=True))


def extract_supplementary_links(
    root: Tag,
    base_url: str,
    manifest_row: dict[str, str],
    videos: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """영상·웹툰은 Action이 아닌 참고 링크로 공식 게시 페이지만 제공합니다."""
    links: list[dict[str, Any]] = []
    label_override = norm(manifest_row.get("supplementary_link_label", ""))
    display_condition = norm(manifest_row.get("supplementary_display_condition", ""))

    if videos and manifest_row.get("video_policy") == "reference_only_official_page":
        links.append({
            "supplementary_id": sha256_text(f"{manifest_row['url_id']}|video|{manifest_row['source_url']}")[:16],
            "label": label_override or f"{manifest_row['page_title']} 영상 보기",
            "url": manifest_row["source_url"],
            "link_type": "video",
            "indexable": False,
            "content_status": "reference_only",
            "display_condition": display_condition or "user_requests_video",
            "media_urls": [item["media_url"] for item in videos],
            "notice": "영상은 이해를 돕는 참고 자료이며 최신 제도 답변의 직접 근거로 사용하지 않습니다.",
        })

    if _has_webtoon(root) and manifest_row.get("webtoon_policy") == "reference_only_official_page":
        links.append({
            "supplementary_id": sha256_text(f"{manifest_row['url_id']}|webtoon|{manifest_row['source_url']}")[:16],
            "label": label_override or f"{manifest_row['page_title']} 웹툰 보기",
            "url": manifest_row["source_url"],
            "link_type": "webtoon",
            "indexable": False,
            "content_status": "reference_only",
            "display_condition": display_condition or "user_requests_visual_guide",
            "notice": "웹툰은 참고용 시각 자료이며 금액·신청 조건은 현재 공식 본문을 우선합니다.",
        })

    return links


def extract_images(
    root: Tag,
    base_url: str,
    manifest_row: dict[str, str],
    video_poster_urls: set[str] | None = None,
) -> list[dict[str, Any]]:
    """웹툰·영상 포스터·장식 이미지는 제외하고 핵심 이미지 메타데이터만 저장합니다."""
    images: list[dict[str, Any]] = []
    seen: set[str] = set()
    poster_urls = video_poster_urls or set()
    for image in root.find_all("img"):
        source = image.get("src") or image.get("data-src") or image.get("data-original")
        url = absolute_url(base_url, source)
        if not url or url in seen or url in poster_urls:
            continue
        alt = norm(image.get("alt", ""))
        filename = Path(urlparse(url).path).name
        lowered = f"{filename} {alt}".lower()

        # 웹툰은 이미지 목록·다운로드·RAG에서 제외하고 공식 페이지 참고 링크만 제공합니다.
        if "webtoon" in lowered or "웹툰" in lowered:
            continue
        if _looks_decorative_image(image, filename, alt):
            continue

        role = "content_image"
        if any(keyword in lowered for keyword in ["proc", "process", "step", "flow", "절차", "과정"]):
            role = "process_diagram"
        elif not alt:
            # 파일명만으로 의미를 판단할 수 없는 이미지는 저장하지 않습니다.
            continue

        seen.add(url)
        images.append({
            "url": url,
            "alt": alt,
            "filename": filename,
            "image_role": role,
            "indexable": False,
            "download_policy": manifest_row.get("image_policy", "metadata_only_nondecorative"),
        })
    return images


class StructureParser:
    def __init__(self, page_type: str = "static_page"):
        self.blocks: list[dict[str, Any]] = []
        self.page_type = page_type

    def add(self, block: dict[str, Any] | None) -> None:
        if not block:
            return
        if self.blocks and block.get("type") in {"paragraph", "heading"}:
            if self.blocks[-1].get("type") == block.get("type") and self.blocks[-1].get("text") == block.get("text"):
                return
        self.blocks.append(block)

    def parse(self, root: Tag) -> list[dict[str, Any]]:
        faq_dls = root.select(".accoWrap .accodian > dl, .accodian.con.ico > dl, .accordion > dl")
        if self.page_type == "faq" or faq_dls:
            if not faq_dls:
                faq_dls = [dl for dl in root.find_all("dl") if dl.find("dt", recursive=False) and dl.find("dd", recursive=False)]
            faq_ids = {id(node) for node in faq_dls}
            for child in root.find_all(recursive=False):
                if isinstance(child, Tag) and any(id(dl) in faq_ids for dl in child.select("dl")):
                    continue
                self.process_node(child)
            for dl in faq_dls:
                self.parse_faq(dl)
        else:
            for child in root.find_all(recursive=False):
                self.process_node(child)
        return self.blocks

    def parse_faq(self, dl: Tag) -> None:
        dt = dl.find("dt", recursive=False)
        dd = dl.find("dd", recursive=False)
        if not dt or not dd:
            return
        question = re.sub(r"^\s*질문\s*", "", safe_text(dt))
        question = re.sub(r"\s*열기\s*$", "", question)
        answer_parser = StructureParser("static_page")
        answer_parser.process_node(dd)
        cleaned_answer_blocks: list[dict[str, Any]] = []
        for block in answer_parser.blocks:
            if block.get("type") == "paragraph":
                value = re.sub(r"^\s*답변\s*[:：]?\s*", "", block.get("text", ""))
                if not norm(value):
                    continue
                block = {**block, "text": norm(value)}
            cleaned_answer_blocks.append(block)
        answer_parser.blocks = cleaned_answer_blocks
        if not answer_parser.blocks:
            fallback_answer = re.sub(r"^\s*답변\s*[:：]?\s*", "", safe_text(dd))
            answer_parser.add({"type": "paragraph", "text": norm(fallback_answer)})
        answer_text = blocks_text(answer_parser.blocks)
        if question and answer_text:
            self.add({
                "type": "faq", "question": norm(question),
                "answer_blocks": answer_parser.blocks, "answer_text": answer_text,
            })

    def parse_definition_list(self, dl: Tag) -> None:
        children = [child for child in dl.find_all(recursive=False) if isinstance(child, Tag)]
        index = 0
        while index < len(children):
            if children[index].name != "dt":
                self.process_node(children[index])
                index += 1
                continue
            dt = children[index]
            dd = children[index + 1] if index + 1 < len(children) and children[index + 1].name == "dd" else None
            term = re.sub(r"\s*열기\s*$", "", safe_text(dt))
            term = re.sub(r"^\s*\d{1,2}\s+(?=\D)", "", term)
            if dd:
                parser = StructureParser("static_page")
                parser.process_node(dd)
                if not parser.blocks:
                    parser.add({"type": "paragraph", "text": safe_text(dd)})
                self.add({
                    "type": "definition", "term": norm(term),
                    "definition_blocks": parser.blocks,
                    "definition_text": blocks_text(parser.blocks),
                })
                index += 2
            else:
                self.add({"type": "heading", "level": 3, "text": norm(term)})
                index += 1

    def parse_resource_card(self, node: Tag) -> bool:
        """제목·설명/목록·바로가기 버튼으로 구성된 카드 한 개를 의미 단위로 묶습니다."""
        title_node = node.find(["strong", "h2", "h3", "h4", "dt"], recursive=False)
        if not title_node:
            return False
        has_body = bool(node.find(["ul", "ol", "p", "dl", "table"], recursive=False))
        has_action = bool(node.find(["a", "button"], recursive=False))
        if not (has_body or has_action):
            return False

        title = safe_text(title_node)
        if not title or is_noise_text(title):
            return False

        fragment = BeautifulSoup(str(node), "lxml")
        copied = fragment.body.find(node.name) if fragment.body else None
        if not copied:
            return False

        copied_title = copied.find(["strong", "h2", "h3", "h4", "dt"], recursive=False)
        if copied_title:
            copied_title.decompose()

        # 이동 URL은 actions/links에서 구조화하므로 본문에는 '바로가기' 글자를 남기지 않습니다.
        for action_node in copied.find_all(["a", "button", "input"]):
            action_node.decompose()

        child_parser = StructureParser("static_page")
        for child in copied.find_all(recursive=False):
            child_parser.process_node(child)

        content_blocks = child_parser.blocks
        self.add({
            "type": "resource_group",
            "title": title,
            "content_blocks": content_blocks,
            "content_text": blocks_text(content_blocks),
        })
        return True

    def process_node(self, node: Any) -> None:
        if isinstance(node, Comment):
            return
        if isinstance(node, NavigableString):
            text = norm(str(node))
            if text and not is_noise_text(text):
                self.add({"type": "paragraph", "text": text})
            return
        if not isinstance(node, Tag):
            return
        name = node.name.lower()
        classes = set(node.get("class", []))
        if name in {"script", "style", "noscript", "template"} or classes & {"floatTop", "floatBottom", "paging", "pagination", "sr-only"}:
            return
        if "item" in classes and self.parse_resource_card(node):
            return
        if name in {"button", "input"}:
            return
        if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            text = safe_text(node)
            if text and not is_noise_text(text):
                self.add({"type": "heading", "level": int(name[1]), "text": text})
            return
        if name == "p":
            text = safe_text(node)
            if text and not is_noise_text(text):
                self.add({"type": "paragraph", "text": text})
            return
        if name == "table":
            headers, rows = expand_table(node)
            if headers or rows:
                self.add({"type": "table", "headers": headers, "rows": rows, "row_count": len(rows)})
            return
        if name in {"ul", "ol"}:
            items: list[str] = []
            for li in node.find_all("li", recursive=False):
                fragment = BeautifulSoup(str(li), "lxml")
                copied = fragment.body.find("li") if fragment.body else None
                if not copied:
                    continue
                for nested in copied.find_all(["ul", "ol"]):
                    nested.decompose()

                # 사이트가 CSS용 번호 span과 실제 텍스트 번호를 동시에 두는 경우가 있습니다.
                # 예: <li><span>1</span> 1. 첫 번째 절차</li>
                # 번호 전용 직계 자식 태그를 먼저 제거한 뒤, 텍스트 번호도 반복 제거합니다.
                if name == "ol":
                    for marker in copied.find_all(
                        ["span", "em", "strong", "i", "b"],
                        recursive=False,
                    ):
                        marker_text = norm(marker.get_text(" ", strip=True))
                        marker_classes = " ".join(marker.get("class", [])).lower()
                        if (
                            re.fullmatch(r"(?:\d{1,3}|[①-⑳])(?:[.)])?", marker_text)
                            or any(token in marker_classes for token in ["num", "number", "step"])
                            and re.fullmatch(r"\d{1,3}", marker_text)
                        ):
                            marker.decompose()

                text = safe_text(copied)
                if name == "ol":
                    prefix_pattern = re.compile(
                        r"^\s*(?:\(?\d{1,3}\)?[.)]|[①-⑳]|\d{1,3}(?=\s+\D))\s*"
                    )
                    # 최대 3회 반복 제거해 '1 1. 내용', '1. 1. 내용'을 모두 정리합니다.
                    for _ in range(3):
                        cleaned = norm(prefix_pattern.sub("", text, count=1))
                        if cleaned == text:
                            break
                        text = cleaned
                if text and not is_noise_text(text):
                    items.append(text)
            # 단순 탭 메뉴는 제거합니다.
            if set(items).issubset({"착오송금인", "착오송금수취인"}):
                items = []
            if items:
                self.add({"type": "list", "ordered": name == "ol", "items": items})
            for li in node.find_all("li", recursive=False):
                for nested in li.find_all(["ul", "ol"], recursive=False):
                    self.process_node(nested)
            return
        if name == "dl":
            self.parse_definition_list(node)
            return
        if name == "figure":
            caption = node.find("figcaption")
            if caption:
                text = safe_text(caption)
                if text and not is_noise_text(text):
                    self.add({"type": "paragraph", "text": text})
            return

        inline: list[str] = []
        def flush() -> None:
            nonlocal inline
            text = norm(" ".join(inline))
            inline = []
            if text and not is_noise_text(text):
                self.add({"type": "paragraph", "text": text})

        for child in node.children:
            if isinstance(child, Comment):
                continue
            if isinstance(child, NavigableString):
                value = norm(str(child))
                if value:
                    inline.append(value)
            elif isinstance(child, Tag):
                if child.name.lower() == "br":
                    inline.append(" ")
                elif child.name.lower() in BLOCK_TAGS:
                    flush()
                    self.process_node(child)
                elif child.name.lower() in {"button", "input"}:
                    continue
                else:
                    value = safe_text(child)
                    if value:
                        inline.append(value)
        flush()


def render_blocks(blocks: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for block in blocks:
        kind = block.get("type")
        if kind == "heading":
            level = min(max(int(block.get("level", 2)), 2), 6)
            lines.append(f"{'#' * level} {block.get('text', '')}")
        elif kind == "paragraph":
            lines.append(block.get("text", ""))
        elif kind == "list":
            for index, item in enumerate(block.get("items", []), start=1):
                prefix = f"{index}. " if block.get("ordered") else "- "
                lines.append(prefix + item)
        elif kind == "resource_group":
            lines.append("### " + block.get("title", ""))
            content = render_blocks(block.get("content_blocks", []))
            if content:
                lines.append(content)
        elif kind == "definition":
            lines.append("### " + block.get("term", ""))
            lines.append(render_blocks(block.get("definition_blocks", [])))
        elif kind == "faq":
            lines.append("### Q. " + block.get("question", ""))
            lines.append(render_blocks(block.get("answer_blocks", [])))
        elif kind == "table":
            headers = [norm(value).replace("|", "\\|") for value in block.get("headers", [])]
            if headers:
                lines.append("| " + " | ".join(headers) + " |")
                lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                for row in block.get("rows", []):
                    values = [norm(value).replace("|", "\\|") for value in row]
                    values.extend([""] * (len(headers) - len(values)))
                    lines.append("| " + " | ".join(values[:len(headers)]) + " |")
        lines.append("")
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def render_document_markdown(document: dict[str, Any]) -> str:
    """검수용 MD에는 페이지 큰 제목과 공식 버튼·링크를 명시적으로 표시합니다."""
    lines: list[str] = []
    title = norm(document.get("display_title") or document.get("page_heading") or document.get("title"))
    if title:
        lines.extend([f"# {title}", ""])
    body = document.get("content_markdown", "").strip()
    if body:
        lines.extend([body, ""])

    actions = [item for item in document.get("actions", []) if item.get("url") and item.get("label")]
    if actions:
        lines.extend(["## 공식 바로가기", ""])
        seen: set[tuple[str, str]] = set()
        for action in actions:
            key = (action["url"], action["label"])
            if key in seen:
                continue
            seen.add(key)
            suffix = f" — {action.get('action_type')}" if action.get("action_type") else ""
            lines.append(f"- [{action['label']}]({action['url']}){suffix}")
        lines.append("")

    action_urls = {item.get("url") for item in actions}
    related = [
        item for item in document.get("links", [])
        if item.get("link_role") == "content" and item.get("url") not in action_urls
        and item.get("url") and item.get("anchor_text")
    ]
    if related:
        lines.extend(["## 관련 링크", ""])
        seen_urls: set[tuple[str, str]] = set()
        for item in related:
            key = (item["url"], item["anchor_text"])
            if key in seen_urls:
                continue
            seen_urls.add(key)
            lines.append(f"- [{item['anchor_text']}]({item['url']})")
        lines.append("")

    supplementary = [item for item in document.get("supplementary_links", []) if item.get("url")]
    if supplementary:
        lines.extend(["## 참고 자료", ""])
        for item in supplementary:
            lines.append(f"- [{item.get('label', '참고 자료')}]({item['url']})")
        lines.append("")

    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def blocks_text(blocks: list[dict[str, Any]]) -> str:
    values: list[str] = []
    for block in blocks:
        kind = block.get("type")
        if kind in {"heading", "paragraph"}:
            values.append(block.get("text", ""))
        elif kind == "list":
            values.extend(block.get("items", []))
        elif kind == "resource_group":
            values.extend([block.get("title", ""), block.get("content_text", "")])
        elif kind == "definition":
            values.extend([block.get("term", ""), block.get("definition_text", "")])
        elif kind == "faq":
            values.extend([block.get("question", ""), block.get("answer_text", "")])
        elif kind == "table":
            values.extend(block.get("headers", []))
            for row in block.get("rows", []):
                values.extend(row)
    return norm(" ".join(values))


def iter_blocks_recursive(blocks: list[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for block in blocks:
        yield block
        for key in ("definition_blocks", "answer_blocks", "content_blocks"):
            children = block.get(key, [])
            if isinstance(children, list):
                yield from iter_blocks_recursive(children)


def count_block_type(blocks: list[dict[str, Any]], target: str) -> int:
    return sum(1 for block in iter_blocks_recursive(blocks) if block.get("type") == target)


def block_fingerprint(block: dict[str, Any]) -> str:
    return sha256_text(json.dumps(block, ensure_ascii=False, sort_keys=True))


def apply_document_filters(document: dict[str, Any]) -> None:
    excluded: list[dict[str, Any]] = []
    cleaned: list[dict[str, Any]] = []
    for block in document.get("blocks", []):
        text = blocks_text([block])
        if document.get("policy", {}).get("webtoon_policy") == "reference_only_official_page" and any(
            token in text for token in ["예튼이", "예솜이", "예툰이", "웹툰"]
        ):
            excluded.append({"reason": "웹툰 참고 링크 전용 정책", "content": text})
            continue
        if document["doc_id"] == "MT-009" and any(token in text for token in ["5천만 원", "5천만원", "1천만원 이하", "1천만원 초과"]):
            excluded.append({"reason": "구버전 지원금액이 포함된 웹툰 설명", "content": text})
            continue
        if document["doc_id"] == "HP-001" and block.get("type") == "table":
            headers = block.get("headers", [])
            rows = block.get("rows", [])
            if "회수기여도" in headers and rows and any("예상실회수금액" in norm(cell) for row in rows for cell in row):
                excluded.append({"reason": "포상금 자동계산기 실행 전 초기값 표", "content": text})
                continue
        if text in {".", "-", "·", "|"}:
            continue
        if is_noise_text(text):
            continue
        cleaned.append(block)
    document["blocks"] = cleaned
    if excluded:
        document["excluded_content"] = excluded
        document["content_filter"] = "legacy_amount_removed"


def refresh_document(document: dict[str, Any]) -> None:
    document["content_markdown"] = render_blocks(document.get("blocks", []))
    document["body_markdown"] = document["content_markdown"]
    document["content_text"] = blocks_text(document.get("blocks", []))
    document["markdown_export"] = render_document_markdown(document)
    document["export_markdown"] = document["markdown_export"]
    parsed_hash = sha256_text(document["content_markdown"])
    document["parsed_content_sha256"] = parsed_hash
    document["content_hash"] = parsed_hash


def select_content_root(soup: BeautifulSoup) -> Tag:
    root = (
        soup.select_one(".contents") or soup.select_one("#contents") or
        soup.select_one("#content") or soup.select_one("main") or soup.body
    )
    if not root:
        raise ValueError("본문 루트를 찾지 못했습니다.")
    return root


def extract_page_heading(soup: BeautifulSoup, fallback: str) -> str:
    selectors = [
        ".pageTit h1", ".pageTit h2", ".page-title h1", ".page-title h2",
        ".subTitle h1", ".subTitle h2", "main h1",
    ]
    for selector in selectors:
        node = soup.select_one(selector)
        if node:
            value = safe_text(node)
            if value and not is_noise_text(value):
                return value
    parts = [norm(part) for part in re.split(r"\s*>\s*", fallback or "") if norm(part)]
    return parts[-1] if parts else norm(fallback)


def clone_clean_content_root(root: Tag) -> Tag:
    fragment = BeautifulSoup(str(root), "lxml")
    cloned = fragment.body.find() if fragment.body else fragment.find()
    if not cloned:
        raise ValueError("본문 복제에 실패했습니다.")
    for selector in NOISE_SELECTORS:
        for node in cloned.select(selector):
            node.decompose()
    return cloned



def remove_reference_only_webtoon(root: Tag, manifest_row: dict[str, str]) -> list[dict[str, str]]:
    """웹툰은 공식 페이지 링크만 남기고 DOM 본문·청킹에서는 완전히 제외합니다."""
    if manifest_row.get("webtoon_policy") != "reference_only_official_page":
        return []
    removed: list[dict[str, str]] = []
    targets: list[Tag] = []
    for image in root.find_all("img"):
        value = " ".join([
            image.get("src", ""), image.get("data-src", ""),
            image.get("alt", ""), image.get("title", ""),
        ]).lower()
        if "webtoon" not in value and "웹툰" not in value:
            continue
        target: Tag = image
        for parent in image.parents:
            if not isinstance(parent, Tag) or parent is root:
                break
            tokens = " ".join(parent.get("class", [])).lower() + " " + norm(parent.get("id", "")).lower()
            target = parent
            if any(token in tokens for token in ["sliderwrap", "swiper", "webtoon"]):
                if "sliderwrap" in tokens:
                    break
        targets.append(target)
    for heading in root.find_all(["h2", "h3", "h4"]):
        text = safe_text(heading)
        if any(token in text for token in ["예툰이와 예솜이", "예튼이와 예솜이"]):
            targets.append(heading.find_parent(class_=re.compile(r"topTit", re.I)) or heading)
    seen: set[int] = set()
    for target in targets:
        if id(target) in seen or target.parent is None:
            continue
        seen.add(id(target))
        text = norm(target.get_text(" ", strip=True))
        removed.append({"reason": "웹툰 참고 링크 전용 정책", "content": text[:1000]})
        target.decompose()
    return removed


def synchronize_document_navigation(document: dict[str, Any]) -> None:
    """모든 Action을 links와 검수용 Markdown에 같은 순서로 반영합니다."""
    content_links = [item for item in document.get("links", []) if item.get("link_role") == "content"]
    document["links"] = merge_links_with_actions(content_links, document.get("actions", []))
    document["markdown_export"] = render_document_markdown(document)
    document["export_markdown"] = document["markdown_export"]


def parse_html_document(
    html_bytes: bytes,
    final_url: str,
    manifest_row: dict[str, str],
    encoding: str,
) -> dict[str, Any]:
    soup = BeautifulSoup(html_bytes.decode(encoding, errors="replace"), "lxml")
    original_root = select_content_root(soup)
    page_heading = extract_page_heading(soup, manifest_row["page_title"])
    display_title = resolve_display_title(manifest_row, page_heading)
    root = clone_clean_content_root(original_root)

    attachments = extract_attachments(root, final_url, manifest_row["url_id"])
    actions = extract_actions(root, final_url, manifest_row, attachments)
    if not actions and manifest_row.get("page_type") == "dynamic_lookup":
        actions = [{
            "action_id": sha256_text(f"{manifest_row['url_id']}|{manifest_row['source_url']}|lookup")[:16],
            "label": f"{manifest_row['page_title']} 조회 바로가기",
            "url": manifest_row["source_url"],
            "action_type": "lookup",
            "source_tag": "source_page",
            "auth_required": manifest_row.get("action_auth", "불필요"),
            "official_domain": True,
            "requires_review": False,
        }]
    videos = extract_videos(root, final_url, manifest_row)
    supplementary_links = extract_supplementary_links(root, final_url, manifest_row, videos)
    poster_urls = {item.get("poster_url") for item in videos if item.get("poster_url")}
    images = extract_images(root, final_url, manifest_row, poster_urls)
    content_links = extract_links(root, final_url)
    links = merge_links_with_actions(content_links, actions)

    source_fragment = BeautifulSoup(str(root), "lxml")
    source_root = source_fragment.body.find() if source_fragment.body else source_fragment.find()
    if source_root:
        for selector in NOISE_SELECTORS:
            for node in source_root.select(selector):
                node.decompose()
        source_text = norm(source_root.get_text(" ", strip=True))
    else:
        source_text = ""

    for media_node in root.find_all(["video", "source", "iframe", "embed", "object"]):
        media_node.decompose()
    removed_webtoon = remove_reference_only_webtoon(root, manifest_row)

    parser = StructureParser(manifest_row.get("page_type", "static_page"))
    blocks = parser.parse(root)
    document = {
        "doc_id": manifest_row["url_id"],
        "parent_doc_id": manifest_row["url_id"],
        "record_type": "page",
        "indexable": manifest_row.get("rag_index_mode") != "none",
        "rag_index_mode": manifest_row.get("rag_index_mode", "full"),
        "title": display_title,
        "manifest_title": manifest_row["page_title"],
        "page_heading": page_heading,
        "display_title": display_title,
        "html_title": norm(soup.title.get_text(" ", strip=True)) if soup.title else "",
        "source_url": manifest_row["source_url"],
        "final_url": final_url,
        "site_name": manifest_row["site_name"],
        "business_function": manifest_row["business_function"],
        "target_business_function": manifest_row.get("target_business_function", manifest_row["business_function"]),
        "page_type": manifest_row["page_type"],
        "parser_profile": manifest_row["parser_profile"],
        "breadcrumb": [manifest_row["business_function"], manifest_row["page_title"]],
        "blocks": blocks,
        "links": links,
        "actions": actions,
        "attachments": attachments,
        "images": images,
        "videos": videos,
        "supplementary_links": supplementary_links,
        "policy": {
            "normalized_decision": manifest_row["normalized_decision"],
            "content_scope": manifest_row["content_scope"],
            "rag_index_label": manifest_row["rag_index_label"],
            "action_policy": manifest_row["action_policy"],
            "expected_action_types": manifest_row["expected_action_types"],
            "pagination_policy": manifest_row["pagination_policy"],
            "auth_required": manifest_row["auth_required"],
            "attachment_download_policy": manifest_row["attachment_download_policy"],
            "attachment_rag_policy": manifest_row["attachment_rag_policy"],
            "attachment_user_delivery_policy": manifest_row["attachment_user_delivery_policy"],
            "video_policy": manifest_row["video_policy"],
            "webtoon_policy": manifest_row["webtoon_policy"],
            "image_policy": manifest_row["image_policy"],
        },
        "parsed_at": now_kst_iso(),
    }
    if removed_webtoon:
        document.setdefault("excluded_content", []).extend(removed_webtoon)
    apply_document_filters(document)
    refresh_document(document)
    synchronize_document_navigation(document)
    document["quality"] = build_quality(document, source_text, manifest_row)
    return document


def build_quality(document: dict[str, Any], source_text: str, row: dict[str, str]) -> dict[str, Any]:
    blocks = document.get("blocks", [])
    content = document.get("content_text", "")
    faq_count = count_block_type(blocks, "faq")
    table_count = count_block_type(blocks, "table")
    markdown_lines = {norm(line.lstrip("#-0123456789. ")) for line in document.get("content_markdown", "").splitlines()}
    noise_hits = [value for value in NOISE_EXACT_TEXTS if value in markdown_lines]
    reasons: list[str] = []
    if document.get("indexable", True) and len(content) < 80:
        reasons.append("본문이 80자 미만")
    retention = round(len(content) / max(1, len(source_text)), 3)
    if document.get("indexable", True) and retention < 0.55:
        reasons.append("본문 보존율 55% 미만")
    if noise_hits:
        reasons.append("공통 UI 문구 잔존")
    if row.get("page_type") == "faq" and faq_count == 0:
        reasons.append("FAQ 질문-답변 추출 실패")
    if row.get("action_policy", "").startswith("O") and not document.get("actions"):
        reasons.append("예상 Action 링크 미검출")
    status = "fail" if any(value in reasons for value in ["공통 UI 문구 잔존", "FAQ 질문-답변 추출 실패"]) else ("review" if reasons else "pass")
    return {
        "status": status,
        "reasons": reasons,
        "source_text_chars": len(source_text),
        "parsed_text_chars": len(content),
        "retention_ratio": retention,
        "faq_count": faq_count,
        "table_count": table_count,
        "attachment_button_count": len(document.get("attachments", [])),
        "action_count": len(document.get("actions", [])),
        "video_count": len(document.get("videos", [])),
        "supplementary_link_count": len(document.get("supplementary_links", [])),
        "noise_hits": noise_hits,
    }


# ============================================================
# 공통 페이지네이션 수집
# ============================================================

@dataclass
class PaginationPlan:
    method: str
    endpoint: str
    page_param: str
    page_size_param: str | None
    page_size: int
    last_internal_index: int
    index_base: int
    base_payload: dict[str, str]
    detection_method: str


def extract_form_payload(form: Tag | None) -> dict[str, str]:
    payload: dict[str, str] = {}
    if not form:
        return payload
    for field in form.select("input[name], select[name], textarea[name]"):
        name = norm(field.get("name", ""))
        if not name:
            continue
        if field.name == "select":
            selected = field.find("option", selected=True) or field.find("option")
            value = selected.get("value", "") if selected else ""
        elif field.name == "textarea":
            value = field.get_text("", strip=False)
        else:
            field_type = norm(field.get("type", "text")).lower()
            if field_type in {"checkbox", "radio"} and not field.has_attr("checked"):
                continue
            value = field.get("value", "")
        payload[name] = str(value or "")
    return payload


def extract_script_text(soup: BeautifulSoup) -> str:
    return "\n".join(script.get_text("\n", strip=True) for script in soup.find_all("script") if not script.get("src"))


def detect_pagination_plan(
    html_bytes: bytes,
    encoding: str,
    final_url: str,
    row: dict[str, str],
    config: PipelineConfig,
) -> PaginationPlan | None:
    if not config.enable_generic_pagination:
        return None
    policy = row.get("pagination_policy", "")
    if "제외" in policy:
        return None

    soup = BeautifulSoup(html_bytes.decode(encoding, errors="replace"), "lxml")
    onclick_indexes: list[int] = []
    for node in soup.select(".paging [onclick*='chgPagingNo'], .pagination [onclick*='chgPagingNo']"):
        match = re.search(r"chgPagingNo\(\s*(\d+)\s*\)", node.get("onclick", ""))
        if match:
            onclick_indexes.append(int(match.group(1)))

    if onclick_indexes:
        last_index = max(onclick_indexes)
        if last_index <= 0:
            return None
        form = soup.select_one("form#srchForm") or soup.select_one("form[name='srchForm']")
        payload = extract_form_payload(form)
        scripts = extract_script_text(soup)
        action = norm(form.get("action", "")) if form else ""
        if not action:
            match = re.search(r"[\"']#srchForm[\"']\)\.attr\(\s*[\"']action[\"']\s*,\s*[\"']([^\"']+)[\"']", scripts)
            if not match:
                match = re.search(r"attr\(\s*[\"']action[\"']\s*,\s*[\"']([^\"']+)[\"']", scripts)
            action = match.group(1) if match else ""
        endpoint = urljoin(final_url, action) if action else final_url
        page_param = "curPage"
        for candidate in ["curPage", "pageIndex", "pageNo", "currentPage", "page"]:
            if candidate in payload or re.search(rf"name\s*[:=]\s*['\"]{candidate}['\"]", scripts):
                page_param = candidate
                break
        page_size_param = "pageSize" if "pageSize" in payload or "pageSize" in scripts else None
        return PaginationPlan(
            method="POST", endpoint=endpoint, page_param=page_param,
            page_size_param=page_size_param, page_size=config.pagination_page_size,
            last_internal_index=last_index, index_base=0, base_payload=payload,
            detection_method="chgPagingNo",
        )

    # query string 기반 페이지 링크도 지원합니다.
    page_links: list[tuple[str, str, int]] = []
    for anchor in soup.select(".paging a[href], .pagination a[href]"):
        href = anchor.get("href", "")
        if not href or href.startswith("javascript:"):
            continue
        absolute = urljoin(final_url, href)
        query = parse_qs(urlparse(absolute).query)
        for param in ["curPage", "pageIndex", "pageNo", "currentPage", "page"]:
            if param in query and query[param] and str(query[param][0]).isdigit():
                page_links.append((absolute, param, int(query[param][0])))
    if page_links:
        _, param, max_page = max(page_links, key=lambda item: item[2])
        if max_page <= 1:
            return None
        return PaginationPlan(
            method="GET", endpoint=final_url, page_param=param, page_size_param=None,
            page_size=config.pagination_page_size, last_internal_index=max_page,
            index_base=1, base_payload={}, detection_method="query_link",
        )
    return None


def displayed_page_number(html_bytes: bytes, encoding: str) -> int | None:
    soup = BeautifulSoup(html_bytes.decode(encoding, errors="replace"), "lxml")
    current = soup.select_one(".paging strong[title*='현재 페이지'] span") or soup.select_one(".pagination .active")
    if not current:
        return None
    text = norm(current.get_text(" ", strip=True))
    return int(text) if text.isdigit() else None


def repeatable_signature(document: dict[str, Any]) -> str:
    selected = [
        block for block in document.get("blocks", [])
        if block.get("type") in {"faq", "table", "list", "definition"}
    ]
    if not selected:
        selected = document.get("blocks", [])
    return sha256_text(json.dumps(selected, ensure_ascii=False, sort_keys=True))


def merge_table_blocks(target: dict[str, Any], incoming: dict[str, Any]) -> None:
    existing = {tuple(row) for row in target.get("rows", [])}
    for row in incoming.get("rows", []):
        key = tuple(row)
        if key not in existing:
            existing.add(key)
            target.setdefault("rows", []).append(row)
    target["row_count"] = len(target.get("rows", []))


def merge_paginated_documents(documents: list[dict[str, Any]], row: dict[str, str]) -> dict[str, Any]:
    merged = documents[0]
    page_type = row.get("page_type", "")

    if page_type == "faq":
        base_non_faq = [block for block in merged["blocks"] if block.get("type") != "faq"]
        faq_blocks: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for document in documents:
            for block in document.get("blocks", []):
                if block.get("type") != "faq":
                    continue
                key = (norm(block.get("question", "")), norm(block.get("answer_text", "")))
                if key in seen:
                    continue
                seen.add(key)
                faq_blocks.append(block)
        merged["blocks"] = base_non_faq + faq_blocks
    else:
        # 같은 헤더를 가진 표는 행을 합치고, 나머지 새 블록은 중복 없이 추가합니다.
        table_by_headers: dict[tuple[str, ...], dict[str, Any]] = {
            tuple(block.get("headers", [])): block
            for block in merged.get("blocks", []) if block.get("type") == "table"
        }
        fingerprints = {block_fingerprint(block) for block in merged.get("blocks", [])}
        for document in documents[1:]:
            for block in document.get("blocks", []):
                if block.get("type") == "table":
                    key = tuple(block.get("headers", []))
                    if key in table_by_headers:
                        merge_table_blocks(table_by_headers[key], block)
                        continue
                fingerprint = block_fingerprint(block)
                if fingerprint not in fingerprints:
                    fingerprints.add(fingerprint)
                    merged["blocks"].append(block)

    for field, keys in [
        ("actions", ("url", "label")), ("attachments", ("attachment_id",)),
        ("images", ("url",)), ("videos", ("media_url",)),
        ("supplementary_links", ("url", "link_type")),
        ("links", ("url", "anchor_text")),
    ]:
        combined: list[dict[str, Any]] = []
        for document in documents:
            combined.extend(document.get(field, []))
        merged[field] = unique_dicts(combined, keys)

    apply_document_filters(merged)
    refresh_document(merged)
    return merged


def collect_paginated_document(
    session: requests.Session,
    first_result: FetchResult,
    first_document: dict[str, Any],
    row: dict[str, str],
    paths: OutputPaths,
    config: PipelineConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = detect_pagination_plan(first_result.body, first_result.encoding, first_result.final_url, row, config)
    if plan is None:
        collection = {
            "collection_scope": "single_public_page",
            "is_complete": True,
            "pagination_detected": False,
            "expected_page_count": 1,
            "fetched_page_count": 1,
            "failures": [],
        }
        first_document["pagination_collection"] = collection
        return first_document, collection

    if plan.last_internal_index + 1 > config.pagination_max_pages:
        raise ValueError(
            f"{row['url_id']} 페이지 수가 안전 한도 초과: {plan.last_internal_index + 1}"
        )

    page_dir = paths.pagination / row["url_id"]
    page_dir.mkdir(parents=True, exist_ok=True)
    first_page_path = page_dir / "page_001.html"
    first_page_path.write_bytes(first_result.body)

    page_documents = [first_document]
    page_records: list[dict[str, Any]] = [{
        "internal_page_index": 0 if plan.index_base == 0 else 1,
        "displayed_page_number": displayed_page_number(first_result.body, first_result.encoding),
        "status_code": first_result.status_code,
        "request_method": "GET",
        "final_url": first_result.final_url,
        "signature": repeatable_signature(first_document),
        "raw_sha256": first_result.raw_sha256,
        "raw_html_path": str(first_page_path.relative_to(paths.root)),
    }]
    failures: list[dict[str, Any]] = []
    seen_signatures = {page_records[0]["signature"]: page_records[0]["internal_page_index"]}

    if plan.index_base == 0:
        page_indexes = range(1, plan.last_internal_index + 1)
    else:
        page_indexes = range(2, plan.last_internal_index + 1)

    for page_index in page_indexes:
        try:
            started = time.perf_counter()
            if plan.method == "POST":
                payload = dict(plan.base_payload)
                payload[plan.page_param] = str(page_index)
                if plan.page_size_param:
                    payload[plan.page_size_param] = str(plan.page_size)
                response = session.post(
                    plan.endpoint, data=payload, headers={"Referer": first_result.final_url},
                    timeout=config.request_timeout_seconds, allow_redirects=True,
                )
                request_payload: dict[str, Any] = payload
            else:
                parsed = urlparse(plan.endpoint)
                query = parse_qs(parsed.query)
                query[plan.page_param] = [str(page_index)]
                query_string = urlencode(query, doseq=True)
                url = urlunparse(parsed._replace(query=query_string))
                response = session.get(url, timeout=config.request_timeout_seconds, allow_redirects=True)
                request_payload = {plan.page_param: page_index}
            response.raise_for_status()
            result = response_to_fetch_result(response, plan.endpoint, time.perf_counter() - started)
            page_document = parse_html_document(result.body, result.final_url, row, result.encoding)
            signature = repeatable_signature(page_document)
            page_number = page_index + 1 if plan.index_base == 0 else page_index
            page_path = page_dir / f"page_{page_number:03d}.html"
            page_path.write_bytes(result.body)
            record = {
                "internal_page_index": page_index,
                "displayed_page_number": displayed_page_number(result.body, result.encoding),
                "status_code": result.status_code,
                "request_method": plan.method,
                "request_payload": request_payload,
                "final_url": result.final_url,
                "signature": signature,
                "raw_sha256": result.raw_sha256,
                "raw_html_path": str(page_path.relative_to(paths.root)),
            }
            if signature in seen_signatures:
                failures.append({
                    "internal_page_index": page_index,
                    "reason": "다른 페이지와 같은 반복 콘텐츠",
                    "same_as": seen_signatures[signature],
                })
            else:
                seen_signatures[signature] = page_index
            expected_display = page_number
            actual_display = record["displayed_page_number"]
            if actual_display is not None and actual_display != expected_display:
                failures.append({
                    "internal_page_index": page_index,
                    "reason": "요청 페이지와 표시 페이지 불일치",
                    "expected": expected_display,
                    "actual": actual_display,
                })
            page_records.append(record)
            page_documents.append(page_document)
        except Exception as error:
            failures.append({
                "internal_page_index": page_index,
                "reason": "페이지 요청 또는 파싱 실패",
                "error_type": type(error).__name__,
                "error": str(error),
            })
        time.sleep(config.request_delay_seconds)

    expected_count = plan.last_internal_index + 1 if plan.index_base == 0 else plan.last_internal_index
    is_complete = not failures and len(page_documents) == expected_count
    merged = merge_paginated_documents(page_documents, row)
    collection = {
        "collection_scope": "all_public_pages" if is_complete else "partial_public_pages",
        "is_complete": is_complete,
        "pagination_detected": True,
        "detection_method": plan.detection_method,
        "method": plan.method,
        "endpoint": plan.endpoint,
        "page_param": plan.page_param,
        "index_base": plan.index_base,
        "expected_page_count": expected_count,
        "fetched_page_count": len(page_documents),
        "pages": page_records,
        "failures": failures,
        "collected_at": now_kst_iso(),
    }
    merged["pagination_collection"] = collection
    merged["dynamic_collection"] = collection if row["url_id"] == "BI-004" else None
    return merged, collection


# ============================================================
# 자산 다운로드
# ============================================================



def ensure_playwright() -> None:
    try:
        import playwright  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "playwright"], check=True)
    subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)




# ============================================================
# 문서 저장과 청킹
# ============================================================

def save_document(paths: OutputPaths, document: dict[str, Any], row: dict[str, str]) -> tuple[Path, Path]:
    folder = safe_name(row["business_function"])
    md_path = paths.parsed_markdown / folder / f"{row['url_id']}.md"
    json_path = paths.parsed_json / folder / f"{row['url_id']}.json"
    ensure_parent(md_path)
    markdown = document.get("markdown_export") or render_document_markdown(document)
    md_path.write_text(markdown, encoding="utf-8")
    write_json(json_path, document)
    return md_path, json_path


def render_table_chunk(headers: list[str], rows: list[list[str]]) -> str:
    return render_blocks([{"type": "table", "headers": headers, "rows": rows}])


def split_table(block: dict[str, Any], max_chars: int, heading_prefix: str = "") -> list[str]:
    headers, rows = block.get("headers", []), block.get("rows", [])
    prefix = f"### {heading_prefix}\n\n" if heading_prefix else ""
    complete = prefix + render_table_chunk(headers, rows)
    if len(complete) <= max_chars:
        return [complete]
    chunks: list[str] = []
    current: list[list[str]] = []
    for row in rows:
        candidate = prefix + render_table_chunk(headers, current + [row])
        if current and len(candidate) > max_chars:
            chunks.append(prefix + render_table_chunk(headers, current))
            current = [row]
        else:
            current.append(row)
    if current:
        chunks.append(prefix + render_table_chunk(headers, current))
    return chunks


def meaningful_chunk(text: str, min_chars: int) -> bool:
    plain = re.sub(r"[#|\-\s]", "", text)
    if len(plain) < min_chars:
        return False
    if any(fragment in text for fragment in NOISE_CONTAINS_TEXTS):
        return False
    return True




def build_link_only_document(row: dict[str, str]) -> dict[str, Any]:
    expected = [value.strip() for value in row.get("expected_action_types", "").split("|") if value.strip()]
    action_type = expected[0] if expected else "related_service"
    display_title = resolve_display_title(row, "")
    label_suffix = {
        "self_check": "자가진단", "lookup": "조회", "apply": "신청",
        "progress": "진행현황 조회", "consult": "상담",
    }.get(action_type, "바로가기")
    label = display_title if label_suffix in display_title else f"{display_title} {label_suffix}"
    action = {
        "action_id": sha256_text(f"{row['url_id']}|{row['source_url']}")[:16],
        "label": label,
        "raw_label": row["page_title"],
        "url": row["source_url"],
        "action_type": action_type,
        "source_tag": "source_page",
        "auth_required": row.get("action_auth", ""),
        "official_domain": allowed_action_domain(row["source_url"]),
        "requires_review": False,
    }
    document = {
        "doc_id": row["url_id"], "parent_doc_id": row["url_id"],
        "record_type": "link_only", "indexable": False, "rag_index_mode": "none",
        "title": display_title, "manifest_title": row["page_title"],
        "page_heading": "", "display_title": display_title,
        "source_url": row["source_url"],
        "site_name": row["site_name"], "business_function": row["business_function"],
        "target_business_function": row["target_business_function"],
        "page_type": row["page_type"], "content_markdown": "", "body_markdown": "",
        "content_text": "", "blocks": [], "attachments": [], "images": [], "videos": [],
        "supplementary_links": [], "links": [], "actions": [action],
        "policy": {
            "normalized_decision": row["normalized_decision"],
            "content_scope": row["content_scope"], "action_policy": row["action_policy"],
            "auth_required": row["auth_required"],
            "attachment_download_policy": row.get("attachment_download_policy", "X"),
            "attachment_rag_policy": row.get("attachment_rag_policy", "none"),
            "attachment_user_delivery_policy": row.get("attachment_user_delivery_policy", "none"),
            "video_policy": row.get("video_policy", "none"),
            "webtoon_policy": row.get("webtoon_policy", "none"),
            "image_policy": row.get("image_policy", "metadata_only_nondecorative"),
        },
        "quality": {"status": "pass", "reasons": [], "action_count": 1},
        "parsed_at": now_kst_iso(),
    }
    synchronize_document_navigation(document)
    return document


# ============================================================
# 전체 실행
# ============================================================

# ============================================================
# 첨부파일 하이브리드 처리 및 무손실 청킹 확장
# ============================================================

import csv
import io
import struct
import xml.etree.ElementTree as ET
import zipfile
import zlib
from email.message import Message

FORM_NO_INDEX_KEYWORDS = {
    "신청서", "청구서", "지급청구서", "위임장", "동의서", "개인정보",
    "서약서", "확인서", "철회서", "신고서", "계좌신고서", "통지서",
    "양식", "서식", "작성용", "채권양도", "인감증명",
}
ATTACHMENT_INDEX_KEYWORDS = {
    "안내", "작성요령", "작성 방법", "설명", "매뉴얼", "가이드", "리플릿",
    "절차", "기준", "목록", "현황", "대상 금융회사", "구비서류 안내",
    "주요내용", "업무편람", "FAQ", "자주 묻는 질문",
}
SUPPORTED_ATTACHMENT_TEXT_FORMATS = {
    "PDF", "HWP", "HWPX", "DOCX", "XLSX", "PPTX", "CSV", "TXT",
}
OLE_MAGIC = bytes.fromhex("D0CF11E0A1B11AE1")


def extract_attachments(root: Tag, base_url: str, parent_doc_id: str) -> list[dict[str, Any]]:
    """다운로드 방식과 공식 링크를 함께 보존합니다."""
    attachments: list[dict[str, Any]] = []
    seen: set[Any] = set()

    for button in root.find_all(["button", "a"], onclick=True):
        match = DOWNLOAD_CALL_RE.search(button.get("onclick", ""))
        if not match:
            continue
        key = (match.group(1), match.group(2))
        if key in seen:
            continue
        seen.add(key)
        fmt = infer_file_format(button)
        label = re.sub(r"다운로드$", "", norm(button.get_text(" ", strip=True))).strip()
        row_context = ""
        parent_row = button.find_parent("tr")
        if parent_row:
            values = []
            for cell in parent_row.find_all(["th", "td"], recursive=False):
                if button in cell.descendants:
                    continue
                value = cell_text(cell)
                if value and "다운로드" not in value and value not in values:
                    values.append(value)
            row_context = " | ".join(values[:4])
        display_name = label or row_context or f"{fmt} 첨부파일"
        if fmt not in display_name.upper():
            display_name = f"{display_name} ({fmt})"
        attachments.append({
            "attachment_id": sha256_text(
                f"{parent_doc_id}|{base_url}|{match.group(1)}|{match.group(2)}|{display_name}|{fmt}"
            )[:16],
            "display_name": display_name,
            "file_format": fmt,
            "download_method": "gfn_downloadFile",
            "token1": match.group(1),
            "token2": match.group(2),
            "row_context": row_context,
            "button_text": norm(button.get_text(" ", strip=True)),
            "source_page_url": base_url,
            "source_url": base_url,
            "official_download_url": None,
            "user_action_url": base_url,
            "delivery_mode": "official_download_page",
            "download_status": "metadata_only",
            "processing_policy": "metadata_only",
            "indexable": False,
            "needs_review": False,
        })

    for anchor in root.find_all("a", href=True):
        url = absolute_url(base_url, anchor.get("href"))
        if not url:
            continue
        ext = Path(urlparse(url).path).suffix.lower()
        if ext not in ATTACHMENT_EXTENSIONS:
            continue
        key = ("href", url)
        if key in seen:
            continue
        seen.add(key)
        attachments.append({
            "attachment_id": sha256_text(f"{parent_doc_id}|{url}")[:16],
            "display_name": norm(anchor.get_text(" ", strip=True)) or Path(urlparse(url).path).name,
            "file_format": ext.lstrip(".").upper(),
            "download_method": "href",
            "url": url,
            "source_page_url": base_url,
            "source_url": base_url,
            "official_download_url": url,
            "user_action_url": url,
            "delivery_mode": "direct_official_download",
            "download_status": "metadata_only",
            "processing_policy": "metadata_only",
            "indexable": False,
            "needs_review": False,
        })
    # PC·모바일 DOM에 같은 파일 버튼이 중복된 경우 표시명·형식 기준으로 하나만 유지합니다.
    deduped: list[dict[str, Any]] = []
    semantic_seen: set[tuple[str, str, str]] = set()
    for item in attachments:
        semantic_name = re.sub(r"\s*\([A-Z0-9]+\)\s*$", "", norm(item.get("display_name", ""))).lower()
        semantic_key = (semantic_name, item.get("file_format", ""), item.get("download_method", ""))
        if semantic_key in semantic_seen:
            continue
        semantic_seen.add(semantic_key)
        deduped.append(item)
    return deduped


def _filename_from_content_disposition(value: str) -> str | None:
    if not value:
        return None
    message = Message()
    message["content-disposition"] = value
    filename = message.get_filename()
    return filename.strip() if filename else None


def _detect_ole_attachment_format(path: Path, declared: str = "") -> str:
    """OLE Compound File 내부 스트림을 이용해 확장자 없는 FILE도 구분합니다."""
    try:
        import olefile
        if not olefile.isOleFile(str(path)):
            return declared or "OLE"
        ole = olefile.OleFileIO(str(path))
        try:
            stream_names = {"/".join(parts) for parts in ole.listdir()}
        finally:
            ole.close()
    except Exception:
        return declared if declared not in {"", "FILE", "OLE"} else "OLE"

    if "FileHeader" in stream_names and any(name.startswith("BodyText/Section") for name in stream_names):
        return "HWP"
    if "WordDocument" in stream_names:
        return "DOC"
    if "Workbook" in stream_names or "Book" in stream_names:
        return "XLS"
    if "PowerPoint Document" in stream_names:
        return "PPT"
    return declared if declared not in {"", "FILE", "OLE"} else "OLE"


def detect_attachment_format(path: Path, declared: str = "") -> str:
    data = path.read_bytes()[:8192]
    declared = (declared or path.suffix.lstrip(".")).upper()
    if data.startswith(b"%PDF-"):
        return "PDF"
    if data.startswith(OLE_MAGIC):
        return _detect_ole_attachment_format(path, declared)
    if data.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(path) as archive:
                names = set(archive.namelist())
                lower_names = {name.lower() for name in names}
                if "mimetype" in names:
                    mime = archive.read("mimetype").decode("utf-8", errors="ignore")
                    if "hwp+zip" in mime:
                        return "HWPX"
                if any(name.startswith("word/") for name in lower_names):
                    return "DOCX"
                if any(name.startswith("xl/") for name in lower_names):
                    return "XLSX"
                if any(name.startswith("ppt/") for name in lower_names):
                    return "PPTX"
                if any(name.startswith("contents/section") for name in lower_names):
                    return "HWPX"
        except Exception:
            pass
        return declared if declared not in {"", "FILE"} else "ZIP"
    if declared in {"CSV", "TXT"}:
        return declared
    return declared if declared not in {"", "FILE"} else "UNKNOWN"


def inspect_downloaded_attachment(path: Path, declared: str = "") -> dict[str, Any]:
    data = path.read_bytes()[:8192]
    lower = data.lstrip().lower()
    html_like = lower.startswith((b"<!doctype html", b"<html", b"<script"))
    detected = detect_attachment_format(path, declared)
    return {
        "detected_format": detected,
        "signature_valid": not html_like and path.stat().st_size > 0,
        "html_error_response": html_like,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_bytes(path.read_bytes()),
    }


def _clean_extracted_text(text: str, max_chars: int) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f]", " ", text)
    lines = []
    for line in text.splitlines():
        value = norm(line)
        if value and (not lines or value != lines[-1]):
            lines.append(value)
    return "\n".join(lines)[:max_chars].strip()


def extract_pdf_text(path: Path, config: PipelineConfig) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    texts = []
    for page in reader.pages[: config.attachment_pdf_max_pages]:
        texts.append(page.extract_text() or "")
    return _clean_extracted_text("\n".join(texts), config.attachment_max_text_chars)


def _word_text_from_node(node: ET.Element) -> str:
    return norm(" ".join(value.strip() for value in node.itertext() if value.strip()))


def _format_docx_table_rows(rows: list[list[str]]) -> list[str]:
    """DOCX 표를 행 경계가 살아 있는 결정론적 텍스트로 변환합니다."""
    cleaned_rows = [
        [norm(cell) for cell in row]
        for row in rows
        if any(norm(cell) for cell in row)
    ]
    if not cleaned_rows:
        return []

    max_columns = max(len(row) for row in cleaned_rows)
    seven_column_rows = sum(1 for row in cleaned_rows if len(row) == 7)
    protected_institution_table = (
        max_columns == 7
        and seven_column_rows >= max(3, len(cleaned_rows) // 2)
    )

    if protected_institution_table:
        headers = [
            "금융권역", "금융회사명", "주소", "대표전화",
            "팩스", "대표자", "홈페이지",
        ]
    else:
        headers = [f"열{index}" for index in range(1, max_columns + 1)]
        first = cleaned_rows[0]
        if (
            len(first) == max_columns
            and len(cleaned_rows) >= 2
            and all(first)
            and all(len(value) <= 30 for value in first)
        ):
            headers = first
            cleaned_rows = cleaned_rows[1:]

    lines: list[str] = []
    for row in cleaned_rows:
        # 문서 제목처럼 병합된 1셀 행은 그대로 보존합니다.
        nonempty = [value for value in row if value]
        if len(nonempty) == 1 and len(row) < max_columns:
            lines.append(nonempty[0])
            continue

        padded = row + [""] * (len(headers) - len(row))
        fields = [
            f"{header}: {value}"
            for header, value in zip(headers, padded)
            if value
        ]
        if fields:
            lines.append(" | ".join(fields))
    return lines


def extract_docx_or_pptx_text(path: Path, config: PipelineConfig) -> str:
    values: list[str] = []
    word_ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())

        if "word/document.xml" in names:
            root = ET.fromstring(archive.read("word/document.xml"))
            body = root.find(f".//{{{word_ns}}}body")
            if body is not None:
                for child in list(body):
                    local_name = child.tag.rsplit("}", 1)[-1]
                    if local_name == "p":
                        text = _word_text_from_node(child)
                        if text:
                            values.append(text)
                    elif local_name == "tbl":
                        rows: list[list[str]] = []
                        for row_node in child.findall(f"./{{{word_ns}}}tr"):
                            row = [
                                _word_text_from_node(cell)
                                for cell in row_node.findall(f"./{{{word_ns}}}tc")
                            ]
                            rows.append(row)
                        values.extend(_format_docx_table_rows(rows))
        else:
            targets = [
                name for name in names
                if name.startswith("ppt/slides/") and name.endswith(".xml")
            ]
            for name in sorted(targets):
                try:
                    root = ET.fromstring(archive.read(name))
                    text = norm(" ".join(
                        value.strip() for value in root.itertext() if value.strip()
                    ))
                    if text:
                        values.append(text)
                except Exception:
                    continue

    return _clean_extracted_text(
        "\n".join(values),
        config.attachment_max_text_chars,
    )


def extract_hwpx_text(path: Path, config: PipelineConfig) -> str:
    values: list[str] = []
    with zipfile.ZipFile(path) as archive:
        targets = [
            name for name in archive.namelist()
            if name.lower().startswith("contents/section") and name.lower().endswith(".xml")
        ]
        for name in sorted(targets):
            try:
                root = ET.fromstring(archive.read(name))
                text = " ".join(value.strip() for value in root.itertext() if value.strip())
                if text:
                    values.append(text)
            except Exception:
                continue
    return _clean_extracted_text("\n".join(values), config.attachment_max_text_chars)


def _excel_column_name(cell_ref: str) -> str:
    match = re.match(r"([A-Z]+)", cell_ref or "")
    return match.group(1) if match else "?"


def extract_xlsx_text(path: Path, config: PipelineConfig) -> str:
    """실제 시트명·행 번호·셀 주소를 보존하여 XLSX를 결정론적으로 변환합니다."""
    main_ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rel_ns = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}
    doc_rel = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
    values: list[str] = []

    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        shared: list[str] = []
        if "xl/sharedStrings.xml" in names:
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for si in root.findall("m:si", main_ns):
                shared.append("".join(si.itertext()).strip())

        relationships: dict[str, str] = {}
        if "xl/_rels/workbook.xml.rels" in names:
            rel_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
            for rel in rel_root.findall("r:Relationship", rel_ns):
                target = rel.get("Target", "")
                if target.startswith("/"):
                    target = target.lstrip("/")
                elif not target.startswith("xl/"):
                    target = "xl/" + target.lstrip("./")
                relationships[rel.get("Id", "")] = target

        sheet_targets: list[tuple[str, str]] = []
        if "xl/workbook.xml" in names:
            workbook = ET.fromstring(archive.read("xl/workbook.xml"))
            for sheet in workbook.findall(".//m:sheet", main_ns):
                sheet_name = sheet.get("name", "시트")
                target = relationships.get(sheet.get(doc_rel, ""), "")
                if target in names:
                    sheet_targets.append((sheet_name, target))

        if not sheet_targets:
            sheet_targets = [
                (f"시트 {index}", name)
                for index, name in enumerate(sorted(
                    name for name in names
                    if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
                ), start=1)
            ]

        for sheet_name, target in sheet_targets:
            root = ET.fromstring(archive.read(target))
            values.append(f"[시트: {sheet_name}]")
            for row in root.findall(".//m:row", main_ns):
                row_no = row.get("r", "?")
                row_values: list[str] = []
                for cell in row.findall("m:c", main_ns):
                    cell_ref = cell.get("r", "")
                    cell_type = cell.get("t", "")
                    value_node = cell.find("m:v", main_ns)
                    inline = cell.find("m:is", main_ns)
                    value = ""
                    if cell_type == "s" and value_node is not None:
                        try:
                            value = shared[int(value_node.text or "0")]
                        except Exception:
                            value = value_node.text or ""
                    elif cell_type == "inlineStr" and inline is not None:
                        value = "".join(inline.itertext())
                    elif value_node is not None:
                        value = value_node.text or ""
                    value = norm(value)
                    if value:
                        row_values.append(f"{cell_ref or _excel_column_name(cell_ref)}={value}")
                if row_values:
                    values.append(f"행 {row_no}: " + " | ".join(row_values))

    return _clean_extracted_text("\n".join(values), config.attachment_max_text_chars)


def extract_hwp_text(path: Path, config: PipelineConfig) -> str:
    import olefile
    ole = olefile.OleFileIO(str(path))
    try:
        header = ole.openstream("FileHeader").read()
        compressed = len(header) > 36 and bool(header[36] & 1)
        sections = []
        for entry in ole.listdir():
            joined = "/".join(entry)
            if joined.startswith("BodyText/Section"):
                try:
                    number = int(re.search(r"Section(\d+)", joined).group(1))
                except Exception:
                    number = 0
                sections.append((number, joined))
        texts: list[str] = []
        for _, stream_name in sorted(sections):
            data = ole.openstream(stream_name).read()
            if compressed:
                try:
                    data = zlib.decompress(data, -15)
                except Exception:
                    continue
            offset = 0
            while offset + 4 <= len(data):
                header_value = struct.unpack_from("<I", data, offset)[0]
                offset += 4
                tag_id = header_value & 0x3FF
                size = (header_value >> 20) & 0xFFF
                if size == 0xFFF:
                    if offset + 4 > len(data):
                        break
                    size = struct.unpack_from("<I", data, offset)[0]
                    offset += 4
                payload = data[offset: offset + size]
                offset += size
                if tag_id == 67:
                    text = payload.decode("utf-16le", errors="ignore")
                    text = re.sub(r"[\x00-\x1f]", " ", text)
                    if norm(text):
                        texts.append(norm(text))
        return _clean_extracted_text("\n".join(texts), config.attachment_max_text_chars)
    finally:
        ole.close()


def extract_attachment_text(path: Path, detected_format: str, config: PipelineConfig) -> tuple[str, str, str]:
    if not config.extract_attachment_text:
        return "", "disabled", ""
    try:
        fmt = detected_format.upper()
        if fmt == "PDF":
            text = extract_pdf_text(path, config)
        elif fmt in {"DOCX", "PPTX"}:
            text = extract_docx_or_pptx_text(path, config)
        elif fmt == "HWPX":
            text = extract_hwpx_text(path, config)
        elif fmt == "XLSX":
            text = extract_xlsx_text(path, config)
        elif fmt == "HWP":
            text = extract_hwp_text(path, config)
        elif fmt in {"CSV", "TXT"}:
            raw = path.read_bytes()
            for encoding in ("utf-8-sig", "cp949", "euc-kr", "utf-8"):
                try:
                    text = raw.decode(encoding)
                    break
                except Exception:
                    text = ""
            text = _clean_extracted_text(text, config.attachment_max_text_chars)
        else:
            return "", "unsupported", f"지원하지 않는 텍스트 추출 형식: {fmt}"
        if not text:
            return "", "empty_or_scanned", "텍스트가 없거나 스캔 문서일 수 있음"
        return text, "success", ""
    except Exception as error:
        return "", "failed", f"{type(error).__name__}: {error}"


def infer_attachment_role(attachment: dict[str, Any]) -> dict[str, Any]:
    name = norm(" ".join([
        attachment.get("display_name", ""),
        attachment.get("row_context", ""),
        attachment.get("filename", ""),
    ]))
    text = attachment.get("extracted_text", "")

    form = any(keyword in name for keyword in FORM_NO_INDEX_KEYWORDS)
    guide = any(keyword in name for keyword in {"안내", "작성요령", "작성 방법", "매뉴얼", "가이드", "리플릿", "절차", "FAQ"})
    list_data = any(keyword in name for keyword in {"목록", "현황", "대상 금융회사", "금융회사"})
    regulation = any(keyword in name for keyword in {"법령", "규정", "기준", "업무편람"})
    sample = "샘플" in name or "예시" in name

    if form:
        role = "form"
    elif sample:
        role = "sample"
    elif list_data:
        role = "list"
    elif regulation:
        role = "regulation"
    elif guide:
        role = "guide"
    else:
        role = "unknown"

    fillable_markers = ["성명", "서명", "날인", "신청인", "위임인", "생년월일", "계좌번호"]
    instruction_markers = ["신청 방법", "제출", "구비서류", "절차", "유의사항", "작성 방법"]
    return {
        "attachment_role": role,
        "contains_fillable_fields": form or sum(marker in text for marker in fillable_markers) >= 2,
        "contains_instructions": guide or sum(marker in text for marker in instruction_markers) >= 2,
        "has_unique_information": role in {"guide", "list", "regulation"},
        "duplicates_parent_content": False,
    }


def classify_attachment_policy(attachment: dict[str, Any], config: PipelineConfig) -> dict[str, Any]:
    text = attachment.get("extracted_text", "")
    status = attachment.get("text_extraction_status", "")
    downloaded = attachment.get("download_status") == "downloaded"
    detected = attachment.get("detected_format", attachment.get("file_format", "")).upper()
    role_info = infer_attachment_role(attachment)
    role = role_info["attachment_role"]

    if not downloaded:
        return {
            **role_info,
            "processing_policy": "metadata_only",
            "indexable": False,
            "classification_reason": "원본 파일 미다운로드",
            "needs_review": attachment.get("download_status") == "failed",
        }
    if role in {"form", "sample"}:
        return {
            **role_info,
            "processing_policy": "download_no_index",
            "indexable": False,
            "classification_reason": "신청서·동의서·위임장·샘플 등 작성용 자료",
            "needs_review": False,
        }
    if status != "success":
        return {
            **role_info,
            "processing_policy": "download_no_index" if detected in SUPPORTED_ATTACHMENT_TEXT_FORMATS else "metadata_only",
            "indexable": False,
            "classification_reason": "텍스트 추출 실패·미지원 또는 스캔 문서",
            "needs_review": True,
        }
    if role in {"guide", "list", "regulation"}:
        return {
            **role_info,
            "processing_policy": "download_and_index",
            "indexable": True,
            "classification_reason": f"{role} 성격의 설명·목록 자료",
            "needs_review": False,
        }
    if len(text) >= max(config.attachment_index_min_chars, 400):
        return {
            **role_info,
            "processing_policy": "download_and_index" if config.attachment_unknown_long_text_auto_index else "download_no_index",
            "indexable": bool(config.attachment_unknown_long_text_auto_index),
            "classification_reason": "설명 텍스트는 충분하지만 문서 역할이 불명확하여 사람 검토 필요",
            "needs_review": True,
        }
    return {
        **role_info,
        "processing_policy": "download_no_index",
        "indexable": False,
        "classification_reason": "짧거나 설명 가치가 낮은 파일",
        "needs_review": len(text) >= config.attachment_index_min_chars,
    }


def _write_stream_to_file(response: requests.Response, output: Path, config: PipelineConfig) -> int:
    ensure_parent(output)
    written = 0
    with output.open("wb") as file:
        for chunk in response.iter_content(64 * 1024):
            if not chunk:
                continue
            written += len(chunk)
            if written > config.max_asset_bytes:
                raise ValueError("첨부파일 제한 용량 초과")
            file.write(chunk)
    return written


def download_direct_assets(
    session: requests.Session,
    document: dict[str, Any],
    row: dict[str, str],
    paths: OutputPaths,
    config: PipelineConfig,
) -> dict[str, list[dict[str, Any]]]:
    result = {"attachments": [], "images": [], "failures": []}
    base_dir = paths.raw_assets / safe_name(row["business_function"]) / row["url_id"]

    attachment_download_enabled = (
        document.get("policy", {}).get("attachment_download_policy") == "O"
    )
    if config.download_direct_attachments and attachment_download_enabled:
        for attachment in document.get("attachments", []):
            if attachment.get("download_method") != "href" or not attachment.get("url"):
                continue
            try:
                response = session.get(
                    attachment["url"], timeout=config.request_timeout_seconds,
                    stream=True, allow_redirects=True,
                )
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "")
                filename = (
                    _filename_from_content_disposition(response.headers.get("Content-Disposition", ""))
                    or Path(urlparse(response.url).path).name
                    or attachment["display_name"]
                )
                filename = f"{attachment['attachment_id']}_{safe_name(filename)}"
                output = base_dir / "attachments" / filename
                _write_stream_to_file(response, output, config)
                inspection = inspect_downloaded_attachment(output, attachment.get("file_format", ""))
                if not inspection["signature_valid"]:
                    raise ValueError("파일 대신 HTML 오류 페이지 또는 빈 응답을 받음")
                attachment.update({
                    "download_status": "downloaded",
                    "filename": filename,
                    "local_path": str(output.relative_to(paths.root)),
                    "mime_type": content_type.split(";", 1)[0].strip(),
                    "official_download_url": response.url,
                    "download_url_final": response.url,
                    "user_action_url": response.url,
                    "delivery_mode": "direct_official_download",
                    **inspection,
                })
                result["attachments"].append(attachment)
            except Exception as error:
                attachment.update({
                    "download_status": "failed",
                    "processing_policy": "metadata_only",
                    "indexable": False,
                    "needs_review": True,
                    "error": f"{type(error).__name__}: {error}",
                })
                result["failures"].append({
                    "type": "attachment", "name": attachment.get("display_name", ""),
                    "error": attachment["error"],
                })

    if config.download_images:
        for image in document.get("images", []):
            if image.get("image_role") == "decorative":
                continue
            try:
                response = session.get(image["url"], timeout=config.request_timeout_seconds, stream=True)
                response.raise_for_status()
                filename = image.get("filename") or Path(urlparse(response.url).path).name
                output = base_dir / "images" / safe_name(filename)
                _write_stream_to_file(response, output, config)
                image.update({
                    "download_status": "downloaded",
                    "download_policy": "download_original_no_index",
                    "local_path": str(output.relative_to(paths.root)),
                    "size_bytes": output.stat().st_size,
                    "sha256": sha256_bytes(output.read_bytes()),
                })
                result["images"].append(dict(image))
            except Exception as error:
                result["failures"].append({
                    "type": "image", "url": image.get("url", ""),
                    "error": f"{type(error).__name__}: {error}",
                })
    return result


def _download_js_attachments_sync(
    records: list[tuple[dict[str, Any], dict[str, Any]]],
    paths: OutputPaths,
    config: PipelineConfig,
) -> list[dict[str, Any]]:
    """Playwright Sync API를 전용 작업 스레드에서 실행합니다.

    Colab/Jupyter 커널은 메인 스레드에서 asyncio 이벤트 루프를 이미 실행하므로
    같은 스레드에서 sync_playwright()를 호출하면 오류가 발생합니다. 이 함수는
    download_js_attachments()가 만든 별도 스레드 안에서만 호출됩니다.
    """
    ensure_playwright()
    from playwright.sync_api import sync_playwright

    results: list[dict[str, Any]] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(
            accept_downloads=True,
            user_agent=config.user_agent,
            locale="ko-KR",
        )
        try:
            grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
            for document, attachment in records:
                grouped.setdefault(document["source_url"], []).append((document, attachment))

            for source_url, items in grouped.items():
                page = context.new_page()
                try:
                    page.goto(
                        source_url,
                        wait_until="domcontentloaded",
                        timeout=config.request_timeout_seconds * 1000,
                    )
                    page.wait_for_function(
                        "typeof gfn_downloadFile === 'function'",
                        timeout=config.request_timeout_seconds * 1000,
                    )
                    for document, attachment in items:
                        try:
                            with page.expect_download(
                                timeout=config.playwright_download_timeout_ms
                            ) as info:
                                page.evaluate(
                                    "([a,b]) => gfn_downloadFile(a,b)",
                                    [attachment["token1"], attachment["token2"]],
                                )
                            download = info.value
                            observed_download_url = getattr(download, "url", "") or ""
                            filename = (
                                f"{attachment['attachment_id']}_"
                                f"{safe_name(download.suggested_filename)}"
                            )
                            output = (
                                paths.raw_assets
                                / safe_name(document["business_function"])
                                / document["doc_id"]
                                / "attachments"
                                / filename
                            )
                            ensure_parent(output)
                            download.save_as(str(output))
                            inspection = inspect_downloaded_attachment(
                                output, attachment.get("file_format", "")
                            )
                            if not inspection["signature_valid"]:
                                raise ValueError(
                                    "파일 대신 HTML 오류 페이지 또는 빈 응답을 받음"
                                )
                            attachment.update({
                                "download_status": "downloaded",
                                "filename": filename,
                                "local_path": str(output.relative_to(paths.root)),
                                "source_page_url": source_url,
                                "official_download_url": None,
                                "user_action_url": source_url,
                                "delivery_mode": "official_download_page",
                                "download_url_observed": observed_download_url,
                                "mime_type": mimetypes.guess_type(filename)[0] or "",
                                **inspection,
                            })
                            results.append(attachment)
                        except Exception as error:
                            attachment.update({
                                "download_status": "failed",
                                "processing_policy": "metadata_only",
                                "indexable": False,
                                "needs_review": True,
                                "error": f"{type(error).__name__}: {error}",
                            })
                finally:
                    page.close()
        finally:
            context.close()
            browser.close()
    return results


def download_js_attachments(
    documents: list[dict[str, Any]], paths: OutputPaths, config: PipelineConfig
) -> list[dict[str, Any]]:
    records: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for document in documents:
        if document.get("policy", {}).get("attachment_download_policy") != "O":
            continue
        for attachment in document.get("attachments", []):
            if attachment.get("download_method") == "gfn_downloadFile":
                records.append((document, attachment))

    if not records or not config.download_js_attachments_with_playwright:
        return []

    # Colab/Jupyter는 메인 스레드에서 asyncio 루프를 실행합니다.
    # Playwright Sync API는 실행 중인 루프와 같은 스레드에서 사용할 수 없으므로
    # 별도 작업 스레드에서 브라우저 수명주기 전체를 처리합니다.
    from concurrent.futures import ThreadPoolExecutor

    try:
        with ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="kdic-playwright"
        ) as executor:
            return executor.submit(
                _download_js_attachments_sync, records, paths, config
            ).result()
    except Exception as error:
        # 브라우저 시작 실패가 전체 HTML 파이프라인을 중단시키지 않도록
        # 대상 첨부파일을 실패/검토 상태로 남기고 이후 처리를 계속합니다.
        error_text = f"{type(error).__name__}: {error}"
        print(f"[첨부파일 경고] Playwright 실행 실패: {error_text}")
        for _, attachment in records:
            if attachment.get("download_status") != "downloaded":
                attachment.update({
                    "download_status": "failed",
                    "processing_policy": "metadata_only",
                    "indexable": False,
                    "needs_review": True,
                    "error": error_text,
                })
        return []


def ensure_source_page_action(document: dict[str, Any], row: dict[str, str]) -> None:
    if document.get("actions") or not row.get("action_policy", "").startswith("O"):
        return
    expected = [value for value in row.get("expected_action_types", "").split("|") if value]
    action_type = expected[0] if expected else "related_service"
    label_suffix = {
        "self_check": "자가진단", "lookup": "조회", "apply": "신청",
        "progress": "진행상황 조회", "consult": "상담",
    }.get(action_type, "바로가기")
    display_title = document.get("display_title") or resolve_display_title(row, "")
    label = display_title if label_suffix in display_title else f"{display_title} {label_suffix}"
    document["actions"] = [{
        "action_id": sha256_text(f"{row['url_id']}|source-page|{action_type}")[:16],
        "label": label,
        "raw_label": row["page_title"],
        "url": row["source_url"],
        "action_type": action_type,
        "source_tag": "source_page",
        "auth_required": row.get("action_auth", row.get("auth_required", "")),
        "official_domain": allowed_action_domain(row["source_url"]),
        "requires_review": False,
    }]


def sync_attachment_actions(document: dict[str, Any]) -> None:
    actions = [action for action in document.get("actions", []) if action.get("source_tag") != "attachment"]
    if document.get("policy", {}).get("attachment_user_delivery_policy") == "none":
        document["actions"] = actions
        return
    seen = {(action.get("url"), action.get("label")) for action in actions}
    for attachment in document.get("attachments", []):
        direct = bool(attachment.get("official_download_url"))
        url = attachment.get("official_download_url") or attachment.get("source_page_url") or document["source_url"]
        # 안정적인 공식 파일 URL이 있으면 직접 다운로드, JS 토큰 방식이면 공식 게시 페이지를 제공합니다.
        suffix = "공식 파일 다운로드" if direct else "공식 다운로드 페이지"
        label = f"{attachment.get('display_name', '첨부파일')} {suffix}"
        key = (url, label)
        if key in seen:
            continue
        seen.add(key)
        actions.append({
            "action_id": attachment["attachment_id"],
            "label": label,
            "url": url,
            "action_type": "download",
            "source_tag": "attachment",
            "attachment_id": attachment["attachment_id"],
            "download_method": attachment.get("download_method"),
            "delivery_mode": "direct_official_download" if direct else "official_download_page",
            "auth_required": "불필요",
            "official_domain": allowed_action_domain(url),
            "requires_review": attachment.get("needs_review", False),
        })
    document["actions"] = actions


def attachment_to_document(
    parent: dict[str, Any], attachment: dict[str, Any], paths: OutputPaths, config: PipelineConfig
) -> dict[str, Any]:
    local_path = paths.root / attachment["local_path"]
    detected = attachment.get("detected_format") or detect_attachment_format(local_path, attachment.get("file_format", ""))
    text, extraction_status, extraction_error = extract_attachment_text(local_path, detected, config)
    is_scanned_candidate = bool(
        detected == "PDF"
        and extraction_status == "empty_or_scanned"
        and attachment.get("size_bytes", 0) > 0
    )
    attachment.update({
        "detected_format": detected,
        "extracted_text": text,
        "extracted_text_chars": len(text),
        "text_extraction_status": extraction_status,
        "text_extraction_error": extraction_error,
        "is_scanned_candidate": is_scanned_candidate,
        "ocr_required": bool(config.attachment_mark_scanned_pdf_for_ocr and is_scanned_candidate),
        "extraction_quality": (
            "good" if extraction_status == "success" and len(text) >= config.attachment_index_min_chars
            else "partial" if extraction_status == "success"
            else "empty" if extraction_status == "empty_or_scanned"
            else "failed"
        ),
    })
    attachment.update(classify_attachment_policy(attachment, config))

    display_name = attachment.get("display_name") or ""
    raw_filename = attachment.get("filename") or ""
    generic_button_title = bool(re.search(
        r"(?:다운받기|다운로드)(?:\s*\([A-Z0-9]+\))?$",
        display_name,
        re.I,
    ))
    if generic_button_title and raw_filename:
        title = Path(raw_filename).stem
        title = re.sub(r"^[0-9a-fA-F]{8,32}_", "", title)
    else:
        title = display_name or Path(raw_filename).stem or "첨부파일"
    markdown = f"# {title}\n\n{text}".strip() if text else f"# {title}"
    doc_id = f"{parent['doc_id']}-ATT-{attachment['attachment_id'][:8]}"
    document = {
        "doc_id": doc_id,
        "parent_doc_id": parent["doc_id"],
        "attachment_id": attachment["attachment_id"],
        "record_type": "attachment",
        "indexable": attachment["indexable"],
        "rag_index_mode": "attachment" if attachment["indexable"] else "none",
        "title": title,
        "source_url": attachment.get("source_page_url") or parent["source_url"],
        "official_download_url": attachment.get("official_download_url"),
        "user_action_url": attachment.get("user_action_url") or attachment.get("source_page_url"),
        "site_name": parent.get("site_name", ""),
        "business_function": parent["business_function"],
        "target_business_function": parent.get("target_business_function", parent["business_function"]),
        "page_type": "attachment",
        "file_format": detected,
        "mime_type": attachment.get("mime_type", ""),
        "filename": attachment.get("filename", ""),
        "local_path": attachment.get("local_path", ""),
        "size_bytes": attachment.get("size_bytes", 0),
        "sha256": attachment.get("sha256", ""),
        "processing_policy": attachment["processing_policy"],
        "classification_reason": attachment["classification_reason"],
        "needs_review": attachment["needs_review"],
        "attachment_role": attachment.get("attachment_role", "unknown"),
        "contains_fillable_fields": attachment.get("contains_fillable_fields", False),
        "contains_instructions": attachment.get("contains_instructions", False),
        "has_unique_information": attachment.get("has_unique_information", False),
        "duplicates_parent_content": attachment.get("duplicates_parent_content", False),
        "is_scanned_candidate": attachment.get("is_scanned_candidate", False),
        "ocr_required": attachment.get("ocr_required", False),
        "extraction_quality": attachment.get("extraction_quality", ""),
        "text_extraction_status": extraction_status,
        "content_text": text,
        "content_markdown": markdown,
        "blocks": [{"type": "paragraph", "text": text}] if text else [],
        "actions": [],
        "attachments": [],
        "images": [],
        "videos": [],
        "supplementary_links": [],
        "links": [],
        "quality": {
            "status": "review" if attachment["needs_review"] else "pass",
            "reasons": [attachment["classification_reason"]] if attachment["needs_review"] else [],
            "parsed_text_chars": len(text),
            "faq_count": 0,
            "table_count": 0,
        },
        "parsed_at": now_kst_iso(),
    }
    folder = paths.parsed_attachments / safe_name(parent["business_function"]) / parent["doc_id"]
    md_path = folder / f"{doc_id}.md"
    json_path = folder / f"{doc_id}.json"
    ensure_parent(md_path)
    md_path.write_text(markdown, encoding="utf-8")
    write_json(json_path, document)
    attachment["parsed_markdown_path"] = str(md_path.relative_to(paths.root))
    attachment["parsed_json_path"] = str(json_path.relative_to(paths.root))
    return document


def _normalized_attachment_identity(row: dict[str, Any]) -> str:
    name = re.sub(r"\s*\([A-Z0-9]+\)\s*$", "", norm(str(row.get("display_name", "")))).lower()
    declared = str(row.get("file_format") or "").upper()
    identity_format = declared if declared not in {"", "FILE", "UNKNOWN", "OLE"} else "GENERIC"
    return "|".join([
        str(row.get("parent_doc_id", "")),
        name,
        identity_format,
    ])


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                records.append(value)
        except json.JSONDecodeError:
            continue
    return records


def resolve_previous_attachment_manifest(paths: OutputPaths, config: PipelineConfig) -> Path | None:
    if config.previous_attachment_manifest_path:
        candidate = Path(config.previous_attachment_manifest_path)
        return candidate if candidate.exists() else None
    if not config.auto_detect_previous_attachment_manifest:
        return None
    candidates = [
        path for path in paths.root.parent.glob("kdic_crawl_output_*/processed/attachment_manifest.jsonl")
        if paths.root not in path.parents
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def build_attachment_change_report(
    current_rows: list[dict[str, Any]],
    previous_path: Path | None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    previous_rows = _read_jsonl_records(previous_path) if previous_path else []
    previous_by_key = {_normalized_attachment_identity(row): row for row in previous_rows}
    current_by_key = {_normalized_attachment_identity(row): row for row in current_rows}
    changes: list[dict[str, Any]] = []
    current_change_meta: dict[str, dict[str, Any]] = {}

    for key, current in current_by_key.items():
        previous = previous_by_key.get(key)
        current_hash = str(current.get("sha256", "") or "")
        previous_hash = str((previous or {}).get("sha256", "") or "")
        if previous is None:
            status = "baseline" if not previous_rows else "new"
        elif current_hash and previous_hash:
            status = "unchanged" if current_hash == previous_hash else "updated"
        else:
            status = "unverified"
        meta = {
            "change_status": status,
            "previous_sha256": previous_hash,
            "previous_manifest_path": str(previous_path) if previous_path else "",
        }
        current_change_meta[key] = meta
        changes.append({**current, **meta})

    for key, previous in previous_by_key.items():
        if key in current_by_key:
            continue
        changes.append({
            **previous,
            "change_status": "deleted",
            "previous_sha256": previous.get("sha256", ""),
            "previous_manifest_path": str(previous_path) if previous_path else "",
        })

    return changes, current_change_meta


def process_attachment_documents(
    page_documents: list[dict[str, Any]], paths: OutputPaths, config: PipelineConfig
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    attachment_documents: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for parent in page_documents:
        parent_rag_policy = parent.get("policy", {}).get("attachment_rag_policy", "none")
        for attachment in parent.get("attachments", []):
            if (
                attachment.get("download_status") == "downloaded"
                and config.create_attachment_documents
                and parent_rag_policy == "auto_classify_guide_only"
            ):
                try:
                    attachment_document = attachment_to_document(parent, attachment, paths, config)
                    attachment_documents.append(attachment_document)
                except Exception as error:
                    attachment.update({
                        "processing_policy": "metadata_only", "indexable": False,
                        "needs_review": True,
                        "text_extraction_status": "failed",
                        "text_extraction_error": f"{type(error).__name__}: {error}",
                    })
            else:
                attachment.update(classify_attachment_policy(attachment, config))

            if parent_rag_policy != "auto_classify_guide_only":
                attachment.update({
                    "processing_policy": "download_no_index" if attachment.get("download_status") == "downloaded" else "metadata_only",
                    "indexable": False,
                    "classification_reason": "URL 정책표에서 첨부파일 RAG 인덱싱을 사용하지 않음",
                })
            manifest_rows.append({
                "parent_doc_id": parent["doc_id"],
                "business_function": parent["business_function"],
                "attachment_download_policy": parent.get("policy", {}).get("attachment_download_policy", "X"),
                "attachment_rag_policy": parent.get("policy", {}).get("attachment_rag_policy", "none"),
                "attachment_user_delivery_policy": parent.get("policy", {}).get("attachment_user_delivery_policy", "none"),
                "attachment_id": attachment.get("attachment_id", ""),
                "display_name": attachment.get("display_name", ""),
                "file_format": attachment.get("file_format", ""),
                "detected_format": attachment.get("detected_format", ""),
                "download_method": attachment.get("download_method", ""),
                "download_status": attachment.get("download_status", ""),
                "processing_policy": attachment.get("processing_policy", ""),
                "indexable": attachment.get("indexable", False),
                "needs_review": attachment.get("needs_review", False),
                "classification_reason": attachment.get("classification_reason", ""),
                "attachment_role": attachment.get("attachment_role", "unknown"),
                "contains_fillable_fields": attachment.get("contains_fillable_fields", False),
                "contains_instructions": attachment.get("contains_instructions", False),
                "has_unique_information": attachment.get("has_unique_information", False),
                "duplicates_parent_content": attachment.get("duplicates_parent_content", False),
                "is_scanned_candidate": attachment.get("is_scanned_candidate", False),
                "ocr_required": attachment.get("ocr_required", False),
                "extraction_quality": attachment.get("extraction_quality", ""),
                "text_extraction_status": attachment.get("text_extraction_status", ""),
                "extracted_text_chars": attachment.get("extracted_text_chars", 0),
                "source_page_url": attachment.get("source_page_url", parent["source_url"]),
                "official_download_url": attachment.get("official_download_url"),
                "download_url_observed": attachment.get("download_url_observed", ""),
                "user_action_url": attachment.get("user_action_url", parent["source_url"]),
                "local_path": attachment.get("local_path", ""),
                "size_bytes": attachment.get("size_bytes", 0),
                "sha256": attachment.get("sha256", ""),
                "error": attachment.get("error") or attachment.get("text_extraction_error", ""),
            })
        sync_attachment_actions(parent)
    previous_manifest_path = resolve_previous_attachment_manifest(paths, config)
    change_rows, change_meta_by_key = build_attachment_change_report(
        manifest_rows,
        previous_manifest_path,
    )
    for row in manifest_rows:
        row.update(change_meta_by_key.get(_normalized_attachment_identity(row), {
            "change_status": "baseline",
            "previous_sha256": "",
            "previous_manifest_path": "",
        }))

    manifest_df = pd.DataFrame(manifest_rows)
    changes_df = pd.DataFrame(change_rows)
    manifest_df.to_csv(paths.processed / "attachment_manifest.csv", index=False, encoding="utf-8-sig")
    changes_df.to_csv(paths.processed / "attachment_changes.csv", index=False, encoding="utf-8-sig")

    with (paths.processed / "attachment_manifest.jsonl").open("w", encoding="utf-8") as file:
        for row in manifest_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (paths.processed / "attachment_changes.jsonl").open("w", encoding="utf-8") as file:
        for row in change_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    if not manifest_df.empty:
        manifest_df[manifest_df["needs_review"].fillna(False)].to_csv(
            paths.quality / "attachment_review.csv", index=False, encoding="utf-8-sig"
        )
    return attachment_documents, manifest_df, changes_df


def _chunk_plain_len(text: str) -> int:
    return len(re.sub(r"[#|\-\s]", "", text))


def _noise_chunk(text: str) -> bool:
    return any(fragment in text for fragment in NOISE_CONTAINS_TEXTS)


def _split_long_fragment_lossless(fragment: str, max_chars: int) -> list[str]:
    """긴 한 줄을 공백 경계 우선으로 나누며 어떤 경우에도 max_chars를 넘기지 않습니다."""
    fragment = fragment.strip()
    if not fragment:
        return []
    if len(fragment) <= max_chars:
        return [fragment]

    parts: list[str] = []
    remaining = fragment
    while len(remaining) > max_chars:
        cut = remaining.rfind(" ", 0, max_chars + 1)
        if cut < max_chars // 2:
            cut = max_chars
        parts.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        parts.append(remaining)
    return parts


def split_text_lossless(text: str, max_chars: int) -> list[str]:
    """줄·문장·공백 경계를 순서대로 사용하고 긴 문장을 반드시 재분할합니다."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    # 단일 줄도 의미 단위로 취급하여 DOCX 표 행이 서로 섞이지 않게 합니다.
    raw_units = [
        value.strip()
        for value in re.split(r"\n+", text)
        if value.strip()
    ]
    units: list[str] = []
    for raw in raw_units:
        if len(raw) <= max_chars:
            units.append(raw)
            continue
        sentences = [
            value.strip()
            for value in re.split(r"(?<=[.!?다요])\s+", raw)
            if value.strip()
        ]
        if not sentences:
            sentences = [raw]
        for sentence in sentences:
            units.extend(_split_long_fragment_lossless(sentence, max_chars))

    chunks: list[str] = []
    current = ""
    for unit in units:
        # 방어 코드: 앞 단계의 결과가 잘못되어도 초대형 청크를 만들지 않습니다.
        safe_units = (
            [unit] if len(unit) <= max_chars
            else _split_long_fragment_lossless(unit, max_chars)
        )
        for safe_unit in safe_units:
            candidate = f"{current}\n{safe_unit}".strip() if current else safe_unit
            if current and len(candidate) > max_chars:
                chunks.append(current)
                current = safe_unit
            else:
                current = candidate
    if current:
        chunks.append(current)

    if any(len(chunk) > max_chars for chunk in chunks):
        raise AssertionError("split_text_lossless가 최대 길이를 초과한 청크를 생성했습니다.")
    return chunks


def structure_aware_chunks(document: dict[str, Any], config: PipelineConfig) -> list[dict[str, Any]]:
    """짧은 FAQ와 짧은 본문을 삭제하지 않는 무손실 청킹입니다."""
    if not document.get("indexable", True):
        return []

    intermediate: list[dict[str, str]] = []
    current_heading = ""
    current_parts: list[str] = []

    def add_section(content: str, title: str) -> None:
        content = content.strip()
        if not content or _noise_chunk(content):
            return
        for part in split_text_lossless(content, config.chunk_max_chars):
            intermediate.append({"section_title": title, "chunk_type": "section", "content": part})

    def current_is_heading_only() -> bool:
        content = "\n\n".join(part for part in current_parts if part).strip()
        return bool(current_heading and content == f"### {current_heading}")

    def flush() -> None:
        nonlocal current_parts
        content = "\n\n".join(part for part in current_parts if part).strip()
        current_parts = []
        add_section(content, current_heading)

    for block in document.get("blocks", []):
        kind = block.get("type")
        if kind == "heading":
            flush()
            current_heading = block.get("text", "")
            current_parts = [render_blocks([block])]
            continue
        if kind == "faq":
            if current_is_heading_only():
                current_parts = []
            else:
                flush()
            content = render_blocks([block]).strip()
            if content and not _noise_chunk(content):
                intermediate.append({
                    "section_title": block.get("question", ""),
                    "chunk_type": "faq", "content": content,
                })
            continue
        if kind == "table":
            pending_heading = current_heading
            # 표 앞의 설명·목록은 먼저 보존하되, 제목만 있는 경우에는
            # 제목을 별도 청크로 만들지 않고 표 청크에 붙입니다.
            if current_is_heading_only():
                current_parts = []
            else:
                flush()
            for content in split_table(block, config.chunk_max_chars, pending_heading):
                if content.strip() and not _noise_chunk(content):
                    intermediate.append({
                        "section_title": pending_heading,
                        "chunk_type": "table", "content": content.strip(),
                    })
            continue
        block_text = render_blocks([block])
        candidate = "\n\n".join(current_parts + [block_text]).strip()
        if current_parts and len(candidate) > config.chunk_max_chars:
            flush()
            if current_heading:
                current_parts = [f"### {current_heading}"]
        current_parts.append(block_text)
    flush()

    # 짧은 일반 섹션은 삭제하지 않고 앞 섹션과 결합하거나 그대로 유지합니다.
    merged: list[dict[str, str]] = []
    for item in intermediate:
        if (
            item["chunk_type"] == "section"
            and _chunk_plain_len(item["content"]) < config.min_chunk_chars
            and merged
            and merged[-1]["chunk_type"] == "section"
            and len(merged[-1]["content"] + "\n\n" + item["content"]) <= config.chunk_max_chars
        ):
            merged[-1]["content"] += "\n\n" + item["content"]
        else:
            merged.append(item)

    # 문서 맨 앞의 짧은 제목 청크는 다음 일반 섹션과 결합합니다.
    if (
        len(merged) >= 2
        and merged[0]["chunk_type"] == "section"
        and _chunk_plain_len(merged[0]["content"]) < config.min_chunk_chars
        and merged[1]["chunk_type"] == "section"
        and len(merged[0]["content"] + "\n\n" + merged[1]["content"]) <= config.chunk_max_chars
    ):
        merged[1]["content"] = merged[0]["content"] + "\n\n" + merged[1]["content"]
        merged.pop(0)

    # 인덱싱 문서가 내용이 있는데 청크 0개가 되는 것을 방지합니다.
    if not merged and document.get("content_markdown", "").strip():
        for part in split_text_lossless(document["content_markdown"], config.chunk_max_chars):
            merged.append({"section_title": "", "chunk_type": "section", "content": part})

    records: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    action_types = sorted({
        action.get("action_type", "") for action in document.get("actions", [])
        if action.get("action_type")
    })
    for item in merged:
        content = item["content"].strip()
        content_hash = sha256_text(content)
        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)
        records.append({
            "chunk_id": f"{document['doc_id']}_chunk_{len(records):03d}",
            "parent_doc_id": document.get("parent_doc_id", document["doc_id"]),
            "document_id": document["doc_id"],
            "attachment_id": document.get("attachment_id"),
            "record_type": document.get("record_type", "page"),
            "chunk_index": len(records),
            "title": document["title"],
            "section_title": item["section_title"],
            "chunk_type": item["chunk_type"],
            "business_function": document["business_function"],
            "target_business_function": document["target_business_function"],
            "page_type": document["page_type"],
            "rag_index_mode": document.get("rag_index_mode", "full"),
            "source_url": document["source_url"],
            "official_download_url": document.get("official_download_url"),
            "available_action_types": action_types,
            "content": content,
            "content_hash": content_hash,
        })
    return records


def _normalized_coverage_text(value: str) -> str:
    return re.sub(r"\s+", "", re.sub(r"[#|`*\-]", "", value or ""))


def atomic_index_units(document: dict[str, Any]) -> list[str]:
    units: list[str] = []
    for block in iter_blocks_recursive(document.get("blocks", [])):
        kind = block.get("type")
        if kind == "heading":
            continue
        if kind == "faq":
            value = render_blocks([block])
            if value:
                units.append(value)
        elif kind == "table":
            headers = block.get("headers", [])
            for row in block.get("rows", []):
                units.append(" ".join(headers + row))
        else:
            value = blocks_text([block])
            if value:
                units.append(value)
    if not units and document.get("content_text"):
        units.append(document["content_text"])
    return units


def _coverage_tokens(value: str) -> set[str]:
    return set(re.findall(r"[가-힣A-Za-z]{2,}|[0-9]+", (value or "").lower()))


def _unit_covered_by_chunks(unit: str, chunk_text: str) -> bool:
    normalized_unit = _normalized_coverage_text(unit)
    normalized_chunks = _normalized_coverage_text(chunk_text)
    if not normalized_unit:
        return True
    if normalized_unit in normalized_chunks:
        return True
    source_tokens = _coverage_tokens(unit)
    if not source_tokens:
        return True
    chunk_tokens = _coverage_tokens(chunk_text)
    return len(source_tokens & chunk_tokens) / len(source_tokens) >= 0.98


def calculate_chunk_coverage(document: dict[str, Any], chunks: list[dict[str, Any]]) -> float:
    """문단·FAQ·표 행 단위가 청크에 보존됐는지를 가중 평균으로 계산합니다."""
    units = [value for value in atomic_index_units(document) if _normalized_coverage_text(value)]
    if not units:
        return 1.0
    chunk_text = "\n".join(chunk.get("content", "") for chunk in chunks)
    weights = [max(1, len(_normalized_coverage_text(unit))) for unit in units]
    covered_weight = sum(
        weight for unit, weight in zip(units, weights)
        if _unit_covered_by_chunks(unit, chunk_text)
    )
    return round(covered_weight / sum(weights), 4)


def run_pipeline(
    review_csv_path: str | Path,
    runtime_root: str | Path,
    config: PipelineConfig | None = None,
    progress_callback: Callable[[int, str, str], None] | None = None,
) -> dict[str, Any]:
    def report_progress(percent: int, stage: str, detail: str = "") -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(max(0, min(100, int(percent))), stage, detail)
        except Exception:
            # Progress reporting is observational and must never break crawling.
            pass

    config = config or PipelineConfig()
    validate_media_policy(config)
    runtime_root = Path(runtime_root)
    paths = create_output_paths(runtime_root)
    report_progress(2, "작업 폴더 준비", paths.root.name)
    session = create_session(config)

    review_df = pd.read_csv(review_csv_path, encoding="utf-8-sig", dtype=str).fillna("")
    manifest_df = normalize_review_manifest(review_df)
    manifest_df.to_csv(paths.manifest / "manifest_full_42.csv", index=False, encoding="utf-8-sig")
    review_df.to_csv(paths.manifest / "review_policy_42.csv", index=False, encoding="utf-8-sig")

    target_df = manifest_df.copy()
    if config.select_business_functions:
        target_df = target_df[target_df["business_function"].isin(config.select_business_functions)]
    if config.run_only_url_ids:
        unknown = sorted(set(config.run_only_url_ids) - set(manifest_df["url_id"]))
        if unknown:
            raise ValueError(f"Manifest에 없는 URL ID: {unknown}")
        target_df = target_df[target_df["url_id"].isin(config.run_only_url_ids)]
    if config.max_urls is not None:
        target_df = target_df.head(config.max_urls)
    target_df.to_csv(paths.manifest / "manifest_run.csv", index=False, encoding="utf-8-sig")
    report_progress(8, "수집 정책 확인", f"대상 URL {len(target_df)}건")

    page_documents: list[dict[str, Any]] = []
    embedded_documents: list[dict[str, Any]] = []
    run_results: list[dict[str, Any]] = []
    crawl_jsonl = paths.logs / "crawl_results.jsonl"

    target_count = max(1, len(target_df))
    for target_index, (_, series) in enumerate(
        tqdm(target_df.iterrows(), total=len(target_df), desc="KDIC 42개 정책 기반 수집"),
        start=1,
    ):
        row = row_to_dict(series)
        report_progress(
            10 + int((target_index - 1) / target_count * 42),
            "URL 수집·파싱",
            f"{target_index}/{len(target_df)} · {row.get('page_title') or row.get('url_id')}",
        )
        started = now_kst_iso()
        try:
            if row["normalized_decision"] == "link_only":
                document = build_link_only_document(row)
                save_document(paths, document, row)
                page_documents.append(document)
                result = {
                    "url_id": row["url_id"], "business_function": row["business_function"],
                    "page_title": row["page_title"], "status": "link_metadata_created",
                    "status_code": None, "content_chars": 0, "faq_count": 0,
                    "table_count": 0, "action_count": 1, "attachment_count": 0,
                    "downloaded_attachment_count": 0,
                    "pagination_detected": False, "pagination_is_complete": True,
                    "pagination_page_count": 0, "error_type": "", "error": "",
                }
            else:
                first = fetch_url(session, row["source_url"], config)
                html_path, meta_path = save_fetch_result(paths, row, first)
                if not first.ok:
                    raise requests.HTTPError(f"HTTP {first.status_code}: {first.final_url}")
                if "html" not in first.content_type.lower():
                    raise ValueError(f"HTML 응답이 아님: {first.content_type}")
                first_document = parse_html_document(first.body, first.final_url, row, first.encoding)
                document, pagination = collect_paginated_document(
                    session, first, first_document, row, paths, config
                )
                child_documents = document.pop("_embedded_documents", [])
                for child_document in child_documents:
                    save_embedded_document(paths, child_document, row)
                embedded_documents.extend(child_documents)
                ensure_source_page_action(document, row)
                synchronize_document_navigation(document)
                document["quality"] = build_quality(document, document.get("content_text", ""), row)
                assets = download_direct_assets(session, document, row, paths, config)
                document["downloaded_assets"] = assets
                md_path, json_path = save_document(paths, document, row)
                page_documents.append(document)
                result = {
                    "url_id": row["url_id"], "business_function": row["business_function"],
                    "page_title": document["title"], "status": (
                        "paginated_full_collection_created" if pagination.get("pagination_detected") and pagination.get("is_complete")
                        else "pagination_collection_incomplete" if pagination.get("pagination_detected")
                        else "parse_success"
                    ),
                    "status_code": first.status_code,
                    "content_chars": len(document.get("content_text", "")),
                    "quality_status": document.get("quality", {}).get("status", ""),
                    "quality_reasons": " | ".join(document.get("quality", {}).get("reasons", [])),
                    "retention_ratio": document.get("quality", {}).get("retention_ratio", 0),
                    "faq_count": document.get("quality", {}).get("faq_count", 0),
                    "table_count": document.get("quality", {}).get("table_count", 0),
                    "action_count": len(document.get("actions", [])),
                    "attachment_count": len(document.get("attachments", [])),
                    "downloaded_attachment_count": len(assets.get("attachments", [])),
                    "image_count": len(document.get("images", [])),
                    "video_count": len(document.get("videos", [])),
                    "supplementary_link_count": len(document.get("supplementary_links", [])),
                    "asset_failure_count": len(assets.get("failures", [])),
                    "pagination_detected": pagination.get("pagination_detected", False),
                    "pagination_is_complete": pagination.get("is_complete", True),
                    "pagination_page_count": pagination.get("fetched_page_count", 1),
                    "pagination_failure_count": len(pagination.get("failures", [])),
                    "raw_html_path": str(html_path.relative_to(paths.root)),
                    "response_meta_path": str(meta_path.relative_to(paths.root)),
                    "parsed_markdown_path": str(md_path.relative_to(paths.root)),
                    "parsed_json_path": str(json_path.relative_to(paths.root)),
                    "error_type": "", "error": "",
                }
        except Exception as error:
            result = {
                "url_id": row["url_id"], "business_function": row["business_function"],
                "page_title": row["page_title"], "status": "failed", "status_code": None,
                "content_chars": 0, "faq_count": 0, "table_count": 0, "action_count": 0,
                "attachment_count": 0, "downloaded_attachment_count": 0,
                "pagination_detected": False, "pagination_is_complete": False,
                "pagination_page_count": 0,
                "error_type": type(error).__name__, "error": str(error),
            }
        result["started_at"] = started
        result["finished_at"] = now_kst_iso()
        run_results.append(result)
        append_jsonl(crawl_jsonl, result)
        report_progress(
            10 + int(target_index / target_count * 42),
            "URL 수집·파싱",
            f"{target_index}/{len(target_df)} · {result.get('status')}",
        )
        if row["normalized_decision"] != "link_only":
            time.sleep(config.request_delay_seconds)

    # JavaScript 기반 공개 첨부파일도 다운로드합니다.
    report_progress(55, "문서 후처리", "첨부파일·페이지 구조 확인")
    download_js_attachments(page_documents, paths, config)

    # 모든 첨부파일을 검증·텍스트 추출·정책 분류합니다.
    attachment_documents, attachment_manifest_df, attachment_changes_df = process_attachment_documents(
        page_documents, paths, config
    )
    documents = page_documents + embedded_documents + attachment_documents
    report_progress(62, "문서 구조화 완료", f"문서 {len(documents)}건")

    # 영상·웹툰은 다운로드하지 않고 공식 게시 페이지 참고 링크 목록으로 저장합니다.
    supplementary_rows: list[dict[str, Any]] = []
    for parent in page_documents:
        for link in parent.get("supplementary_links", []):
            supplementary_rows.append({
                "parent_doc_id": parent["doc_id"],
                "business_function": parent["business_function"],
                "title": parent["title"],
                **link,
            })
    supplementary_df = pd.DataFrame(supplementary_rows)
    supplementary_df.to_csv(
        paths.processed / "supplementary_links.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with (paths.processed / "supplementary_links.jsonl").open("w", encoding="utf-8") as file:
        for row in supplementary_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 첨부파일 상태와 Action이 갱신되었으므로 links·Markdown까지 다시 동기화합니다.
    by_id = {row["url_id"]: row_to_dict(row) for _, row in target_df.iterrows()}
    for document in page_documents:
        synchronize_document_navigation(document)
        save_document(paths, document, by_id[document["doc_id"]])

    results_df = pd.DataFrame(run_results)
    if not results_df.empty:
        downloaded_by_parent = {}
        for parent in page_documents:
            downloaded_by_parent[parent["doc_id"]] = sum(
                1 for item in parent.get("attachments", [])
                if item.get("download_status") == "downloaded"
            )
        results_df["downloaded_attachment_count"] = results_df["url_id"].map(downloaded_by_parent).fillna(0).astype(int)
    results_df.to_csv(paths.logs / "crawl_results.csv", index=False, encoding="utf-8-sig")

    documents_path = paths.processed / "documents.jsonl"
    with documents_path.open("w", encoding="utf-8") as file:
        for document in documents:
            file.write(json.dumps(document, ensure_ascii=False) + "\n")

    chunks: list[dict[str, Any]] = []
    chunks_by_doc: dict[str, list[dict[str, Any]]] = {}
    if config.create_chunks:
        document_count = max(1, len(documents))
        for document_index, document in enumerate(documents, start=1):
            report_progress(
                65 + int((document_index - 1) / document_count * 14),
                "구조 기반 청킹",
                f"{document_index}/{len(documents)} · {document.get('title') or document.get('doc_id')}",
            )
            doc_chunks = structure_aware_chunks(document, config)
            chunks.extend(doc_chunks)
            chunks_by_doc[document["doc_id"]] = doc_chunks
        report_progress(79, "구조 기반 청킹 완료", f"청크 {len(chunks)}개")
    chunks_path = paths.processed / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    quality_rows: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    for document in documents:
        quality = document.get("quality", {})
        pagination = document.get("pagination_collection", {}) or {}
        doc_chunks = chunks_by_doc.get(document["doc_id"], [])
        faq_count = count_block_type(document.get("blocks", []), "faq")
        faq_chunk_count = sum(1 for chunk in doc_chunks if chunk["chunk_type"] == "faq")
        coverage = calculate_chunk_coverage(document, doc_chunks) if document.get("indexable", True) else 1.0
        chunk_complete = (
            (not document.get("indexable", True))
            or (
                len(doc_chunks) >= 1
                and faq_count == faq_chunk_count
                and coverage >= 0.98
            )
        )
        reasons = list(quality.get("reasons", []))
        if not chunk_complete:
            reasons.append("청킹 데이터 손실 가능성")
        quality_status = "fail" if not chunk_complete else quality.get("status", "pass")
        quality_rows.append({
            "url_id": document["doc_id"],
            "parent_doc_id": document.get("parent_doc_id", document["doc_id"]),
            "record_type": document.get("record_type", "page"),
            "business_function": document["business_function"],
            "title": document["title"],
            "indexable": document.get("indexable", True),
            "rag_index_mode": document.get("rag_index_mode", ""),
            "quality_status": quality_status,
            "quality_reasons": " | ".join(dict.fromkeys(reasons)),
            "content_chars": len(document.get("content_text", "")),
            "chunk_count": len(doc_chunks),
            "chunk_coverage_ratio": coverage,
            "faq_count": faq_count,
            "faq_chunk_count": faq_chunk_count,
            "table_count": count_block_type(document.get("blocks", []), "table"),
            "action_count": len(document.get("actions", [])),
            "attachment_count": len(document.get("attachments", [])),
            "video_count": len(document.get("videos", [])),
            "supplementary_link_count": len(document.get("supplementary_links", [])),
            "pagination_detected": pagination.get("pagination_detected", False),
            "pagination_is_complete": pagination.get("is_complete", True),
            "pagination_page_count": pagination.get("fetched_page_count", 1),
            "processing_policy": document.get("processing_policy", ""),
            "needs_review": document.get("needs_review", False),
        })
        if document.get("indexable", True):
            validations.extend([
                {
                    "validation": f"{document['doc_id']} 인덱싱 문서 청크 존재",
                    "passed": len(doc_chunks) >= 1,
                },
                {
                    "validation": f"{document['doc_id']} FAQ 청크 무손실",
                    "passed": faq_count == faq_chunk_count,
                },
                {
                    "validation": f"{document['doc_id']} 청크 본문 보존율",
                    "passed": coverage >= 0.98,
                    "value": coverage,
                },
            ])

    quality_df = pd.DataFrame(quality_rows)
    quality_df.to_csv(paths.quality / "quality_report.csv", index=False, encoding="utf-8-sig")

    by_id = {document["doc_id"]: document for document in page_documents}
    if "DP-001" in by_id:
        validations.append({"validation": "DP-001 1억원", "passed": "1억원" in by_id["DP-001"].get("content_text", "")})
    if "UN-003" in by_id:
        text = by_id["UN-003"].get("content_text", "")
        validations.append({"validation": "UN-003 미수령금 종류", "passed": all(k in text for k in ["예금보험금", "개산지급금", "파산배당금"])})
    if "BI-004" in by_id:
        pc = by_id["BI-004"].get("pagination_collection", {})
        validations.append({"validation": "BI-004 전체 페이지", "passed": pc.get("is_complete", False)})
    if "MT-001" in by_id:
        validations.append({
            "validation": "MT-001 영상 공식 페이지 참고 링크",
            "passed": any(item.get("link_type") == "video" for item in by_id["MT-001"].get("supplementary_links", []))
            and all(not item.get("download", False) and not item.get("indexable", False) for item in by_id["MT-001"].get("videos", [])),
        })
    if "MT-009" in by_id:
        text = by_id["MT-009"].get("content_text", "")
        validations.append({"validation": "MT-009 구버전 금액 제외", "passed": "5천만원" not in text and "5천만 원" not in text})
        validations.append({
            "validation": "MT-009 웹툰 공식 페이지 참고 링크",
            "passed": any(item.get("link_type") == "webtoon" for item in by_id["MT-009"].get("supplementary_links", []))
            and not any("webtoon" in image.get("url", "").lower() for image in by_id["MT-009"].get("images", []))
            and not any(token in by_id["MT-009"].get("content_text", "") for token in ["예튼이", "예솜이", "예툰이"]),
        })

    # 구조적 회귀 검사: 링크·제목·노이즈·웹툰·Action 라벨
    generic_titles = {value.lower() for value in GENERIC_PAGE_HEADINGS}
    for document in page_documents:
        action_urls = {item.get("url") for item in document.get("actions", []) if item.get("url")}
        link_urls = {item.get("url") for item in document.get("links", []) if item.get("url")}
        export_md = document.get("markdown_export", "")
        display_title = norm(document.get("display_title", ""))
        all_text = document.get("content_text", "") + " " + export_md
        validations.extend([
            {
                "validation": f"{document['doc_id']} Action URL links 동기화",
                "passed": action_urls.issubset(link_urls),
            },
            {
                "validation": f"{document['doc_id']} Action URL Markdown 동기화",
                "passed": all(url in export_md for url in action_urls),
            },
            {
                "validation": f"{document['doc_id']} 맥락 포함 표시 제목",
                "passed": bool(display_title) and display_title.lower() not in generic_titles,
                "value": display_title,
            },
            {
                "validation": f"{document['doc_id']} Adobe Reader 노이즈 제거",
                "passed": not any(pattern.search(all_text) for pattern in NOISE_REGEXES),
            },
            {
                "validation": f"{document['doc_id']} Action 라벨 길이",
                "passed": all(len(norm(action.get('label', ''))) <= 80 for action in document.get('actions', [])),
            },
            {
                "validation": f"{document['doc_id']} 목록 번호 중복 제거",
                "passed": not bool(re.search(r"(?:^|\\n)\\s*\\d+[.]\\s+\\d+[.]\\s+", export_md)),
            },
        ])
    if "MT-009" in by_id:
        mt009_text = by_id["MT-009"].get("content_text", "")
        mt009_chunks = " ".join(chunk.get("content", "") for chunk in chunks_by_doc.get("MT-009", []))
        validations.append({
            "validation": "MT-009 웹툰 본문·청크 제외",
            "passed": not any(token in mt009_text + " " + mt009_chunks for token in ["예튼이", "예솜이", "예툰이"]),
        })

    validation_df = pd.DataFrame(validations)
    validation_df.to_csv(paths.quality / "regression_tests.csv", index=False, encoding="utf-8-sig")
    report_progress(92, "품질·회귀 검사 완료", f"검사 {len(validations)}건")

    service_action_count = sum(
        1 for document in page_documents for action in document.get("actions", [])
        if action.get("action_type") != "download"
    )
    download_action_count = sum(
        1 for document in page_documents for action in document.get("actions", [])
        if action.get("action_type") == "download"
    )
    pagination_detected_count = int(quality_df["pagination_detected"].fillna(False).sum()) if not quality_df.empty else 0
    pagination_complete_count = int(
        quality_df.loc[quality_df["pagination_detected"].fillna(False), "pagination_is_complete"].fillna(False).sum()
    ) if not quality_df.empty else 0

    summary = {
        "run_id": paths.root.name,
        "manifest_count": len(manifest_df),
        "target_count": len(target_df),
        "page_document_count": len(page_documents),
        "embedded_document_count": len(embedded_documents),
        "attachment_document_count": len(attachment_documents),
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "status_counts": results_df["status"].value_counts().to_dict() if not results_df.empty else {},
        "quality_counts": quality_df["quality_status"].value_counts().to_dict() if not quality_df.empty else {},
        "service_action_count": service_action_count,
        "download_action_count": download_action_count,
        "attachment_count": len(attachment_manifest_df),
        "downloaded_attachment_count": int((attachment_manifest_df["download_status"] == "downloaded").sum()) if not attachment_manifest_df.empty else 0,
        "indexed_attachment_count": int(attachment_manifest_df["indexable"].fillna(False).sum()) if not attachment_manifest_df.empty else 0,
        "video_reference_count": sum(len(document.get("videos", [])) for document in page_documents),
        "supplementary_link_count": sum(len(document.get("supplementary_links", [])) for document in page_documents),
        "attachment_review_count": int(attachment_manifest_df["needs_review"].fillna(False).sum()) if not attachment_manifest_df.empty else 0,
        "attachment_change_counts": attachment_changes_df["change_status"].value_counts().to_dict() if not attachment_changes_df.empty else {},
        "scanned_pdf_candidate_count": int(attachment_manifest_df["is_scanned_candidate"].fillna(False).sum()) if (not attachment_manifest_df.empty and "is_scanned_candidate" in attachment_manifest_df) else 0,
        "pagination_detected_count": pagination_detected_count,
        "pagination_complete_count": pagination_complete_count,
        "non_paginated_count": len(page_documents) - pagination_detected_count,
        "regression_failed_count": int((~validation_df["passed"].fillna(False)).sum()) if not validation_df.empty else 0,
    }
    write_json(paths.quality / "quality_summary.json", summary)

    report_progress(97, "결과 패키징", "미리보기 데이터 저장")
    archive_path = Path(shutil.make_archive(
        base_name=str(paths.root), format="zip",
        root_dir=paths.root.parent, base_dir=paths.root.name,
    ))
    report_progress(100, "처리 완료", f"문서 {len(documents)}건 · 청크 {len(chunks)}개")
    return {
        "paths": paths,
        "manifest_df": manifest_df,
        "target_df": target_df,
        "results_df": results_df,
        "quality_df": quality_df,
        "validation_df": validation_df,
        "attachment_manifest_df": attachment_manifest_df,
        "attachment_changes_df": attachment_changes_df,
        "supplementary_df": supplementary_df,
        "documents": documents,
        "page_documents": page_documents,
        "embedded_documents": embedded_documents,
        "attachment_documents": attachment_documents,
        "chunks": chunks,
        "summary": summary,
        "archive_path": archive_path,
    }


# 정책 불변식: 영상과 웹툰은 공식 페이지 참고 링크 전용입니다.
def validate_media_policy(config: PipelineConfig) -> None:
    if config.download_videos or config.download_webtoons:
        raise ValueError("영상·웹툰 다운로드는 이 파이프라인 정책에서 허용하지 않습니다.")

# ============================================================
# V4.2: 안전한 의미 그룹화·모달 문서·검색 단위 보강
# ============================================================

_normalize_review_manifest_v41 = normalize_review_manifest
_extract_actions_v41 = extract_actions
_StructureParserV41 = StructureParser


def normalize_review_manifest(review_df: pd.DataFrame) -> pd.DataFrame:
    """V4 정책표를 유지하면서 구조형 페이지의 명시적 선택자를 보강합니다."""
    manifest = _normalize_review_manifest_v41(review_df)
    defaults = {
        "resource_group_selector": "",
        "resource_group_chunk_policy": "page",
        "modal_content_policy": "none",
    }
    for column, default in defaults.items():
        if column not in manifest.columns:
            manifest[column] = default

    # .item 같은 범용 클래스가 아니라 확인된 부모-자식 경계를 명시합니다.
    mask = manifest["url_id"] == "MT-010"
    manifest.loc[mask, "resource_group_selector"] = ".ruleBox > .item"
    manifest.loc[mask, "resource_group_chunk_policy"] = "resource_group"
    manifest.loc[mask, "modal_content_policy"] = "extract_referenced_layers"
    manifest.loc[mask, "parser_profile"] = "resource_card_with_modal_v1"
    return manifest


def resource_group_selector_for(manifest_row: dict[str, str]) -> str:
    selector = norm(manifest_row.get("resource_group_selector", ""))
    if selector:
        return selector
    # 안전한 자동 탐지는 부모가 ruleBox이고 직계 제목·본문/동작을 가진 경우로 제한합니다.
    return ".ruleBox > .item"


def _fn_layer_target(onclick: str) -> str:
    match = re.search(
        r"\bfn_layer\s*\(\s*['\"]([^'\"]+)['\"]",
        onclick or "",
        flags=re.I,
    )
    return norm(match.group(1)) if match else ""


def _safe_fragment_id(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z가-힣_-]+", "_", value or "").strip("_")
    return cleaned or sha256_text(value or "fragment")[:12]


def _resource_card_descriptors(
    root: Tag,
    base_url: str,
    manifest_row: dict[str, str],
) -> list[dict[str, Any]]:
    selector = resource_group_selector_for(manifest_row)
    descriptors: list[dict[str, Any]] = []
    for index, card in enumerate(root.select(selector), start=1):
        title_node = card.find(["strong", "h2", "h3", "h4", "dt"], recursive=False)
        title = safe_text(title_node) if title_node else ""
        if not title:
            continue
        group_id = f"{manifest_row['url_id']}_resource_{index:03d}"
        descriptor: dict[str, Any] = {
            "resource_group_id": group_id,
            "title": title,
            "index": index,
            "external_url": "",
            "target_layer_id": "",
            "target_document_id": "",
        }
        action_node = card.find(["a", "button"], recursive=False)
        if action_node:
            onclick = action_node.get("onclick", "")
            target_layer_id = _fn_layer_target(onclick)
            if target_layer_id:
                descriptor["target_layer_id"] = target_layer_id
                descriptor["target_document_id"] = (
                    f"{manifest_row['url_id']}_{_safe_fragment_id(target_layer_id)}"
                )
            else:
                href = action_node.get("href", "")
                descriptor["external_url"] = absolute_url(base_url, href) or ""
        descriptors.append(descriptor)
    return descriptors


def merge_links_with_actions(
    content_links: list[dict[str, Any]],
    actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """동일 URL의 일반 링크·Action 중복을 제거하되 서로 다른 모달은 보존합니다."""
    action_external_urls = {
        item.get("url")
        for item in actions
        if item.get("url") and not item.get("target_layer_id")
    }

    merged: list[dict[str, Any]] = []
    seen_content: set[tuple[str, str]] = set()
    for item in content_links:
        url = item.get("url")
        text = item.get("anchor_text")
        if not url or not text:
            continue
        # 같은 URL이 Action으로 승격되면 의미 없는 '바로가기' content link는 제거합니다.
        if url in action_external_urls:
            continue
        key = (url, text)
        if key in seen_content:
            continue
        seen_content.add(key)
        merged.append(item)

    seen_actions: set[tuple[str, str, str]] = set()
    for action in actions:
        url = action.get("url")
        label = action.get("label")
        target_layer_id = action.get("target_layer_id", "")
        if not url or not label:
            continue
        key = (url, target_layer_id, label)
        if key in seen_actions:
            continue
        seen_actions.add(key)
        merged.append({
            "url": url,
            "anchor_text": label,
            "link_role": "action",
            "action_type": action.get("action_type", "related_service"),
            "source_tag": action.get("source_tag", ""),
            "action_id": action.get("action_id", ""),
            "resource_group_id": action.get("resource_group_id", ""),
            "interaction_type": action.get("interaction_type", "navigate"),
            "target_layer_id": target_layer_id,
            "target_document_id": action.get("target_document_id", ""),
        })
    return merged


def extract_actions(
    root: Tag,
    base_url: str,
    manifest_row: dict[str, str],
    attachments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    actions = _extract_actions_v41(root, base_url, manifest_row, attachments)
    descriptors = _resource_card_descriptors(root, base_url, manifest_row)

    # 외부 URL Action에 그룹 참조를 붙입니다.
    for descriptor in descriptors:
        external_url = descriptor.get("external_url")
        if not external_url:
            continue
        for action in actions:
            if action.get("url") == external_url:
                action["resource_group_id"] = descriptor["resource_group_id"]
                action["interaction_type"] = "navigate"

    # fn_layer는 외부 URL이 아니라 같은 공식 페이지 안의 규정 전문을 여는 동작입니다.
    for descriptor in descriptors:
        target_layer_id = descriptor.get("target_layer_id")
        if not target_layer_id:
            continue
        title = descriptor["title"]
        label = f"{title} 내용 보기"
        action_id = sha256_text(
            f"{manifest_row['url_id']}|{target_layer_id}|{label}"
        )[:16]
        actions.append({
            "action_id": action_id,
            "label": label,
            "raw_label": "바로가기",
            "url": manifest_row["source_url"],
            "action_type": "legal_reference",
            "source_tag": "a",
            "javascript_function": "fn_layer",
            "interaction_type": "open_modal",
            "delivery_mode": "official_page_modal",
            "target_layer_id": target_layer_id,
            "target_document_id": descriptor["target_document_id"],
            "resource_group_id": descriptor["resource_group_id"],
            "auth_required": "불필요",
            "official_domain": True,
            "requires_review": False,
        })

    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for action in actions:
        key = (
            action.get("url", ""),
            action.get("target_layer_id", ""),
            action.get("label", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(action)
    return unique


class StructureParser(_StructureParserV41):
    """명시적으로 확인된 카드 경계에서만 resource_group을 만듭니다."""

    def __init__(
        self,
        page_type: str = "static_page",
        document_id: str = "",
        resource_group_selector: str = "",
    ):
        super().__init__(page_type)
        self.document_id = document_id
        self.resource_group_selector = resource_group_selector
        self._resource_node_ids: set[int] = set()
        self._resource_index = 0

    def parse(self, root: Tag) -> list[dict[str, Any]]:
        selector = self.resource_group_selector
        if selector:
            self._resource_node_ids = {id(node) for node in root.select(selector)}
        else:
            self._resource_node_ids = set()
        return super().parse(root)

    def parse_resource_card(self, node: Tag) -> bool:
        title_node = node.find(["strong", "h2", "h3", "h4", "dt"], recursive=False)
        if not title_node:
            return False
        title = safe_text(title_node)
        if not title or is_noise_text(title):
            return False

        self._resource_index += 1
        group_id = (
            f"{self.document_id}_resource_{self._resource_index:03d}"
            if self.document_id
            else f"resource_{self._resource_index:03d}"
        )

        fragment = BeautifulSoup(str(node), "lxml")
        copied = fragment.body.find(node.name) if fragment.body else None
        if not copied:
            return False
        copied_title = copied.find(["strong", "h2", "h3", "h4", "dt"], recursive=False)
        if copied_title:
            copied_title.decompose()

        target_layer_id = ""
        external_url = ""
        action_node = node.find(["a", "button"], recursive=False)
        if action_node:
            target_layer_id = _fn_layer_target(action_node.get("onclick", ""))
            if not target_layer_id:
                external_url = action_node.get("href", "")

        for action_node_copy in copied.find_all(["a", "button", "input"]):
            action_node_copy.decompose()

        child_parser = StructureParser("static_page")
        for child in copied.find_all(recursive=False):
            child_parser.process_node(child)

        block: dict[str, Any] = {
            "type": "resource_group",
            "resource_group_id": group_id,
            "title": title,
            "content_blocks": child_parser.blocks,
            "content_text": blocks_text(child_parser.blocks),
            "interaction_type": "open_modal" if target_layer_id else "navigate",
        }
        if target_layer_id:
            block["target_layer_id"] = target_layer_id
            block["child_document_id"] = (
                f"{self.document_id}_{_safe_fragment_id(target_layer_id)}"
            )
        if external_url:
            block["external_url_raw"] = external_url
        self.add(block)
        return True

    def process_node(self, node: Any) -> None:
        if isinstance(node, Tag) and id(node) in self._resource_node_ids:
            if self.parse_resource_card(node):
                return

        # 기존 V4.1의 범용 '.item' 자동 그룹화를 차단합니다.
        if isinstance(node, Tag) and "item" in set(node.get("class", [])):
            original_classes = list(node.get("class", []))
            node["class"] = [value for value in original_classes if value != "item"]
            try:
                return super().process_node(node)
            finally:
                node["class"] = original_classes
        return super().process_node(node)


def _modal_content_root(layer: Tag) -> Tag:
    return (
        layer.select_one(".ruleArea")
        or layer.select_one(".cont")
        or layer
    )


def extract_modal_child_documents(
    soup: BeautifulSoup,
    final_url: str,
    manifest_row: dict[str, str],
    actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """실제로 fn_layer가 참조한 레이어만 별도 검색 문서로 만듭니다."""
    documents: list[dict[str, Any]] = []
    seen_targets: set[str] = set()
    for action in actions:
        target_layer_id = norm(action.get("target_layer_id", ""))
        if not target_layer_id or target_layer_id in seen_targets:
            continue
        seen_targets.add(target_layer_id)
        layer = soup.find(id=target_layer_id)
        if not isinstance(layer, Tag):
            action["requires_review"] = True
            action["modal_resolution_status"] = "target_not_found"
            continue

        title_node = layer.select_one(":scope > .inner > .tit") or layer.select_one(".tit")
        title = safe_text(title_node) if title_node else ""
        if not title:
            title = re.sub(r"\s*내용\s*보기$", "", action.get("label", ""))

        content_node = _modal_content_root(layer)
        fragment = BeautifulSoup(str(content_node), "lxml")
        cloned = fragment.body.find() if fragment.body else fragment.find()
        if not cloned:
            action["requires_review"] = True
            action["modal_resolution_status"] = "content_not_found"
            continue

        for selector in [
            ".btnBottomArea", ".popClose", ".close", "script", "style", "noscript",
        ]:
            for node in cloned.select(selector):
                node.decompose()
        for node in cloned.find_all(["button", "input"]):
            node.decompose()
        for anchor in cloned.find_all("a"):
            if safe_text(anchor) in {"닫기", "바로가기"}:
                anchor.decompose()

        parser = StructureParser(
            "modal_document",
            document_id=action["target_document_id"],
            resource_group_selector="",
        )
        blocks = parser.parse(cloned)
        # 모달 제목이 본문 첫 문단에 중복되면 한 번만 유지합니다.
        while blocks and blocks[0].get("type") in {"paragraph", "heading"}:
            first_text = blocks[0].get("text", "")
            if norm(first_text) == norm(title):
                blocks.pop(0)
            else:
                break

        child = {
            "doc_id": action["target_document_id"],
            "parent_doc_id": manifest_row["url_id"],
            "resource_group_id": action.get("resource_group_id", ""),
            "record_type": "modal_document",
            "indexable": True,
            "rag_index_mode": "full",
            "title": title,
            "manifest_title": manifest_row["page_title"],
            "page_heading": title,
            "display_title": title,
            "html_title": "",
            "source_url": manifest_row["source_url"],
            "final_url": final_url,
            "source_fragment": f"#{target_layer_id}",
            "target_layer_id": target_layer_id,
            "site_name": manifest_row["site_name"],
            "business_function": manifest_row["business_function"],
            "target_business_function": manifest_row.get(
                "target_business_function", manifest_row["business_function"]
            ),
            "page_type": "modal_document",
            "parser_profile": "referenced_modal_document_v1",
            "breadcrumb": [
                manifest_row["business_function"],
                manifest_row["page_title"],
                title,
            ],
            "blocks": blocks,
            "links": [],
            "actions": [],
            "attachments": [],
            "images": [],
            "videos": [],
            "supplementary_links": [],
            "policy": {
                "normalized_decision": "embedded_public_document",
                "content_scope": "fn_layer로 참조된 공개 규정 전문",
                "rag_index_label": "O",
                "action_policy": "source_page_modal",
                "expected_action_types": "legal_reference",
                "pagination_policy": "none",
                "video_policy": "none",
                "webtoon_policy": "none",
                "image_policy": "none",
            },
            "parsed_at": now_kst_iso(),
        }
        refresh_document(child)
        child["quality"] = {
            "status": "pass" if child.get("content_text") else "fail",
            "reasons": [] if child.get("content_text") else ["모달 본문 없음"],
            "parsed_text_chars": len(child.get("content_text", "")),
            "faq_count": count_block_type(blocks, "faq"),
            "table_count": count_block_type(blocks, "table"),
            "action_count": 0,
        }
        action["modal_resolution_status"] = "resolved"
        documents.append(child)
    return documents


def parse_html_document(
    html_bytes: bytes,
    final_url: str,
    manifest_row: dict[str, str],
    encoding: str,
) -> dict[str, Any]:
    soup = BeautifulSoup(html_bytes.decode(encoding, errors="replace"), "lxml")
    original_root = select_content_root(soup)
    page_heading = extract_page_heading(soup, manifest_row["page_title"])
    display_title = resolve_display_title(manifest_row, page_heading)
    root = clone_clean_content_root(original_root)

    attachments = extract_attachments(root, final_url, manifest_row["url_id"])
    actions = extract_actions(root, final_url, manifest_row, attachments)
    if not actions and manifest_row.get("page_type") == "dynamic_lookup":
        actions = [{
            "action_id": sha256_text(
                f"{manifest_row['url_id']}|{manifest_row['source_url']}|lookup"
            )[:16],
            "label": f"{manifest_row['page_title']} 조회 바로가기",
            "raw_label": "",
            "url": manifest_row["source_url"],
            "action_type": "lookup",
            "source_tag": "source_page",
            "interaction_type": "navigate",
            "auth_required": manifest_row.get("action_auth", "불필요"),
            "official_domain": True,
            "requires_review": False,
        }]

    embedded_documents = extract_modal_child_documents(
        soup, final_url, manifest_row, actions
    )

    videos = extract_videos(root, final_url, manifest_row)
    supplementary_links = extract_supplementary_links(
        root, final_url, manifest_row, videos
    )
    poster_urls = {
        item.get("poster_url") for item in videos if item.get("poster_url")
    }
    images = extract_images(root, final_url, manifest_row, poster_urls)
    content_links = extract_links(root, final_url)
    links = merge_links_with_actions(content_links, actions)

    source_fragment = BeautifulSoup(str(root), "lxml")
    source_root = source_fragment.body.find() if source_fragment.body else source_fragment.find()
    if source_root:
        for selector in NOISE_SELECTORS:
            for node in source_root.select(selector):
                node.decompose()
        source_text = norm(source_root.get_text(" ", strip=True))
    else:
        source_text = ""

    for media_node in root.find_all(["video", "source", "iframe", "embed", "object"]):
        media_node.decompose()
    removed_webtoon = remove_reference_only_webtoon(root, manifest_row)

    parser = StructureParser(
        manifest_row.get("page_type", "static_page"),
        document_id=manifest_row["url_id"],
        resource_group_selector=resource_group_selector_for(manifest_row),
    )
    blocks = parser.parse(root)
    document = {
        "doc_id": manifest_row["url_id"],
        "parent_doc_id": manifest_row["url_id"],
        "record_type": "page",
        "indexable": manifest_row.get("rag_index_mode") != "none",
        "rag_index_mode": manifest_row.get("rag_index_mode", "full"),
        "title": display_title,
        "manifest_title": manifest_row["page_title"],
        "page_heading": page_heading,
        "display_title": display_title,
        "html_title": norm(soup.title.get_text(" ", strip=True)) if soup.title else "",
        "source_url": manifest_row["source_url"],
        "final_url": final_url,
        "site_name": manifest_row["site_name"],
        "business_function": manifest_row["business_function"],
        "target_business_function": manifest_row.get(
            "target_business_function", manifest_row["business_function"]
        ),
        "page_type": manifest_row["page_type"],
        "parser_profile": manifest_row["parser_profile"],
        "resource_group_selector": manifest_row.get("resource_group_selector", ""),
        "resource_group_chunk_policy": manifest_row.get(
            "resource_group_chunk_policy", "page"
        ),
        "breadcrumb": [manifest_row["business_function"], manifest_row["page_title"]],
        "blocks": blocks,
        "links": links,
        "actions": actions,
        "attachments": attachments,
        "images": images,
        "videos": videos,
        "supplementary_links": supplementary_links,
        "embedded_document_ids": [item["doc_id"] for item in embedded_documents],
        "modal_resources": [{
            "target_layer_id": item["target_layer_id"],
            "target_document_id": item["doc_id"],
            "title": item["title"],
            "resource_group_id": item.get("resource_group_id", ""),
            "status": "resolved",
        } for item in embedded_documents],
        "policy": {
            "normalized_decision": manifest_row["normalized_decision"],
            "content_scope": manifest_row["content_scope"],
            "rag_index_label": manifest_row["rag_index_label"],
            "action_policy": manifest_row["action_policy"],
            "expected_action_types": manifest_row["expected_action_types"],
            "pagination_policy": manifest_row["pagination_policy"],
            "auth_required": manifest_row["auth_required"],
            "attachment_download_policy": manifest_row["attachment_download_policy"],
            "attachment_rag_policy": manifest_row["attachment_rag_policy"],
            "attachment_user_delivery_policy": manifest_row[
                "attachment_user_delivery_policy"
            ],
            "video_policy": manifest_row["video_policy"],
            "webtoon_policy": manifest_row["webtoon_policy"],
            "image_policy": manifest_row["image_policy"],
            "modal_content_policy": manifest_row.get("modal_content_policy", "none"),
        },
        "parsed_at": now_kst_iso(),
        "_embedded_documents": embedded_documents,
    }
    if removed_webtoon:
        document.setdefault("excluded_content", []).extend(removed_webtoon)
    apply_document_filters(document)
    refresh_document(document)
    synchronize_document_navigation(document)
    document["quality"] = build_quality(document, source_text, manifest_row)
    return document


def save_embedded_document(
    paths: OutputPaths,
    document: dict[str, Any],
    row: dict[str, str],
) -> tuple[Path, Path]:
    folder = safe_name(row["business_function"])
    md_path = (
        paths.parsed_markdown / folder / "embedded" / f"{document['doc_id']}.md"
    )
    json_path = (
        paths.parsed_json / folder / "embedded" / f"{document['doc_id']}.json"
    )
    ensure_parent(md_path)
    md_path.write_text(
        document.get("markdown_export") or render_document_markdown(document),
        encoding="utf-8",
    )
    write_json(json_path, document)
    return md_path, json_path


def structure_aware_chunks(
    document: dict[str, Any],
    config: PipelineConfig,
) -> list[dict[str, Any]]:
    """resource_group·법령 조문 경계를 검색 단위로 보존하는 무손실 청킹입니다."""
    if not document.get("indexable", True):
        return []

    intermediate: list[dict[str, Any]] = []
    current_heading = ""
    current_parts: list[str] = []

    def add_unit(
        content: str,
        title: str,
        chunk_type: str,
        resource_group_id: str = "",
    ) -> None:
        content = content.strip()
        if not content or _noise_chunk(content):
            return
        for part in split_text_lossless(content, config.chunk_max_chars):
            intermediate.append({
                "section_title": title,
                "chunk_type": chunk_type,
                "content": part,
                "resource_group_id": resource_group_id,
            })

    def current_is_heading_only() -> bool:
        content = "\n\n".join(
            part for part in current_parts if part
        ).strip()
        return bool(current_heading and content == f"### {current_heading}")

    def flush() -> None:
        nonlocal current_parts
        content = "\n\n".join(
            part for part in current_parts if part
        ).strip()
        current_parts = []
        add_unit(content, current_heading, "section")

    for block in document.get("blocks", []):
        kind = block.get("type")
        if kind == "heading":
            flush()
            current_heading = block.get("text", "")
            current_parts = [render_blocks([block])]
            continue
        if kind == "resource_group":
            flush()
            add_unit(
                render_blocks([block]),
                block.get("title", ""),
                "resource_group",
                block.get("resource_group_id", ""),
            )
            continue
        if kind == "definition":
            flush()
            definition_type = (
                "legal_article"
                if document.get("record_type") == "modal_document"
                else "definition"
            )
            add_unit(
                render_blocks([block]),
                block.get("term", ""),
                definition_type,
                document.get("resource_group_id", ""),
            )
            continue
        if kind == "faq":
            if current_is_heading_only():
                current_parts = []
            else:
                flush()
            add_unit(
                render_blocks([block]),
                block.get("question", ""),
                "faq",
            )
            continue
        if kind == "table":
            pending_heading = current_heading
            if current_is_heading_only():
                current_parts = []
            else:
                flush()
            for content in split_table(
                block, config.chunk_max_chars, pending_heading
            ):
                add_unit(content, pending_heading, "table")
            continue

        block_text = render_blocks([block])
        candidate = "\n\n".join(
            current_parts + [block_text]
        ).strip()
        if current_parts and len(candidate) > config.chunk_max_chars:
            flush()
            if current_heading:
                current_parts = [f"### {current_heading}"]
        current_parts.append(block_text)
    flush()

    merged: list[dict[str, Any]] = []
    for item in intermediate:
        if (
            item["chunk_type"] == "section"
            and _chunk_plain_len(item["content"]) < config.min_chunk_chars
            and merged
            and merged[-1]["chunk_type"] == "section"
            and len(
                merged[-1]["content"] + "\n\n" + item["content"]
            ) <= config.chunk_max_chars
        ):
            merged[-1]["content"] += "\n\n" + item["content"]
        else:
            merged.append(item)

    if not merged and document.get("content_markdown", "").strip():
        for part in split_text_lossless(
            document["content_markdown"], config.chunk_max_chars
        ):
            merged.append({
                "section_title": "",
                "chunk_type": "section",
                "content": part,
                "resource_group_id": "",
            })

    records: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    action_types = sorted({
        action.get("action_type", "")
        for action in document.get("actions", [])
        if action.get("action_type")
    })
    for item in merged:
        content = item["content"].strip()
        content_hash = sha256_text(content)
        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)
        records.append({
            "chunk_id": f"{document['doc_id']}_chunk_{len(records):03d}",
            "parent_doc_id": document.get("parent_doc_id", document["doc_id"]),
            "document_id": document["doc_id"],
            "attachment_id": document.get("attachment_id"),
            "record_type": document.get("record_type", "page"),
            "chunk_index": len(records),
            "title": document["title"],
            "section_title": item["section_title"],
            "chunk_type": item["chunk_type"],
            "resource_group_id": item.get(
                "resource_group_id", document.get("resource_group_id", "")
            ),
            "source_fragment": document.get("source_fragment", ""),
            "business_function": document["business_function"],
            "target_business_function": document["target_business_function"],
            "page_type": document["page_type"],
            "rag_index_mode": document.get("rag_index_mode", "full"),
            "source_url": document["source_url"],
            "official_download_url": document.get("official_download_url"),
            "available_action_types": action_types,
            "content": content,
            "content_hash": content_hash,
        })
    return records

# ============================================================
# V4.2.1: 모달 법령의 장·절 문맥을 다음 조문에 결합
# ============================================================

def structure_aware_chunks(
    document: dict[str, Any],
    config: PipelineConfig,
) -> list[dict[str, Any]]:
    """카드·FAQ·표·법령 조문 경계를 유지하면서 짧은 문맥도 버리지 않습니다."""
    if not document.get("indexable", True):
        return []

    intermediate: list[dict[str, Any]] = []
    current_heading = ""
    current_parts: list[str] = []
    modal_context: list[str] = []
    is_modal = document.get("record_type") == "modal_document"

    def add_unit(
        content: str,
        title: str,
        chunk_type: str,
        resource_group_id: str = "",
    ) -> None:
        content = content.strip()
        if not content or _noise_chunk(content):
            return
        for part in split_text_lossless(content, config.chunk_max_chars):
            intermediate.append({
                "section_title": title,
                "chunk_type": chunk_type,
                "content": part,
                "resource_group_id": resource_group_id,
            })

    def current_is_heading_only() -> bool:
        content = "\n\n".join(
            part for part in current_parts if part
        ).strip()
        return bool(current_heading and content == f"### {current_heading}")

    def flush() -> None:
        nonlocal current_parts
        content = "\n\n".join(
            part for part in current_parts if part
        ).strip()
        current_parts = []
        add_unit(content, current_heading, "section")

    for block in document.get("blocks", []):
        kind = block.get("type")

        # 모달 법령의 제정일·개정일·장·절 제목은 단독 짧은 청크가 아니라
        # 바로 다음 조문의 문맥으로 결합합니다.
        if is_modal and kind in {"paragraph", "heading"}:
            text = render_blocks([block]).strip()
            if text:
                modal_context.append(text)
            continue

        if kind == "heading":
            flush()
            current_heading = block.get("text", "")
            current_parts = [render_blocks([block])]
            continue

        if kind == "resource_group":
            flush()
            add_unit(
                render_blocks([block]),
                block.get("title", ""),
                "resource_group",
                block.get("resource_group_id", ""),
            )
            continue

        if kind == "definition":
            flush()
            rendered = render_blocks([block]).strip()
            if is_modal and modal_context:
                rendered = "\n\n".join(modal_context + [rendered])
                modal_context = []
            add_unit(
                rendered,
                block.get("term", ""),
                "legal_article" if is_modal else "definition",
                document.get("resource_group_id", ""),
            )
            continue

        if kind == "faq":
            if current_is_heading_only():
                current_parts = []
            else:
                flush()
            add_unit(
                render_blocks([block]),
                block.get("question", ""),
                "faq",
            )
            continue

        if kind == "table":
            pending_heading = current_heading
            if current_is_heading_only():
                current_parts = []
            else:
                flush()
            for content in split_table(
                block, config.chunk_max_chars, pending_heading
            ):
                add_unit(content, pending_heading, "table")
            continue

        block_text = render_blocks([block])
        candidate = "\n\n".join(
            current_parts + [block_text]
        ).strip()
        if current_parts and len(candidate) > config.chunk_max_chars:
            flush()
            if current_heading:
                current_parts = [f"### {current_heading}"]
        current_parts.append(block_text)

    flush()
    if modal_context:
        add_unit(
            "\n\n".join(modal_context),
            document.get("title", ""),
            "legal_context",
            document.get("resource_group_id", ""),
        )

    merged: list[dict[str, Any]] = []
    for item in intermediate:
        if (
            item["chunk_type"] == "section"
            and _chunk_plain_len(item["content"]) < config.min_chunk_chars
            and merged
            and merged[-1]["chunk_type"] == "section"
            and len(
                merged[-1]["content"] + "\n\n" + item["content"]
            ) <= config.chunk_max_chars
        ):
            merged[-1]["content"] += "\n\n" + item["content"]
        else:
            merged.append(item)

    if not merged and document.get("content_markdown", "").strip():
        for part in split_text_lossless(
            document["content_markdown"], config.chunk_max_chars
        ):
            merged.append({
                "section_title": "",
                "chunk_type": "section",
                "content": part,
                "resource_group_id": "",
            })

    records: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    action_types = sorted({
        action.get("action_type", "")
        for action in document.get("actions", [])
        if action.get("action_type")
    })
    for item in merged:
        content = item["content"].strip()
        content_hash = sha256_text(content)
        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)
        records.append({
            "chunk_id": f"{document['doc_id']}_chunk_{len(records):03d}",
            "parent_doc_id": document.get("parent_doc_id", document["doc_id"]),
            "document_id": document["doc_id"],
            "attachment_id": document.get("attachment_id"),
            "record_type": document.get("record_type", "page"),
            "chunk_index": len(records),
            "title": document["title"],
            "section_title": item["section_title"],
            "chunk_type": item["chunk_type"],
            "resource_group_id": item.get(
                "resource_group_id", document.get("resource_group_id", "")
            ),
            "source_fragment": document.get("source_fragment", ""),
            "business_function": document["business_function"],
            "target_business_function": document["target_business_function"],
            "page_type": document["page_type"],
            "rag_index_mode": document.get("rag_index_mode", "full"),
            "source_url": document["source_url"],
            "official_download_url": document.get("official_download_url"),
            "available_action_types": action_types,
            "content": content,
            "content_hash": content_hash,
        })
    return records

# V4.2 정책 CSV에 구조 처리 열이 있으면 URL ID 하드코딩보다 우선 적용합니다.
_normalize_review_manifest_v42_default = normalize_review_manifest

def normalize_review_manifest(review_df: pd.DataFrame) -> pd.DataFrame:
    manifest = _normalize_review_manifest_v42_default(review_df)
    lookup = review_df.fillna("").astype(str).set_index("문서_ID")
    column_map = {
        "구조그룹_선택자": "resource_group_selector",
        "구조그룹_청킹정책": "resource_group_chunk_policy",
        "모달콘텐츠_처리정책": "modal_content_policy",
    }
    for source_column, target_column in column_map.items():
        if source_column not in lookup.columns:
            continue
        values = manifest["url_id"].map(lookup[source_column]).fillna("")
        mask = values.str.strip() != ""
        manifest.loc[mask, target_column] = values[mask]
    return manifest


# ============================================================
# V4.4: 검색 단위 의미 보존·표 정책·제목 정규화·HCX 사전 품질 보강
# ============================================================

_refresh_document_v43 = refresh_document
_parse_html_document_v43 = parse_html_document
_normalize_review_manifest_v43 = normalize_review_manifest


def _clean_v44_text(value: str) -> str:
    text = norm(value)
    for fragment in [
        "예금보험공사 핵심가치",
    ]:
        text = norm(text.replace(fragment, " "))
    return text


def _looks_like_body_sentence_heading(text: str) -> bool:
    value = norm(text)
    if len(value) < 60:
        return False
    sentence_pattern = re.compile(
        r"(?:합니다|됩니다|있습니다|없습니다|입니다|말합니다|"
        r"필요합니다|수 있습니다|하여야 합니다|아니합니다|"
        r"않습니다|바랍니다|같습니다)"
    )
    if value.endswith("다.") or re.search(
        sentence_pattern.pattern + r"\.?$",
        value,
    ):
        return True
    # 긴 dt/strong 설명문은 마지막에 괄호가 붙어도 본문으로 처리합니다.
    return len(value) >= 100 and bool(sentence_pattern.search(value))


def _normalize_blocks_v44(
    blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    for raw_block in blocks:
        block = dict(raw_block)
        kind = block.get("type")

        if kind in {"paragraph", "heading"}:
            text = _clean_v44_text(block.get("text", ""))
            if not text or is_noise_text(text):
                continue
            if kind == "heading" and _looks_like_body_sentence_heading(text):
                block = {
                    "type": "paragraph",
                    "text": text,
                    "source_semantic_fix": "sentence_heading_to_paragraph",
                }
            else:
                block["text"] = text
            result.append(block)
            continue

        if kind == "list":
            items = []
            seen = set()
            for item in block.get("items", []):
                text = _clean_v44_text(item)
                if not text or is_noise_text(text) or text in seen:
                    continue
                seen.add(text)
                items.append(text)
            if items:
                block["items"] = items
                result.append(block)
            continue

        if kind == "table":
            headers = [_clean_v44_text(value) for value in block.get("headers", [])]
            rows = []
            for row in block.get("rows", []):
                cleaned_row = [_clean_v44_text(value) for value in row]
                if any(cleaned_row):
                    rows.append(cleaned_row)
            if headers or rows:
                block["headers"] = headers
                block["rows"] = rows
                block["row_count"] = len(rows)
                result.append(block)
            continue

        if kind == "definition":
            term = _clean_v44_text(block.get("term", ""))
            children = _normalize_blocks_v44(
                block.get("definition_blocks", [])
            )
            definition_text = blocks_text(children)
            # 빈 검색 입력 필드는 문서에는 남길 필요가 없습니다.
            if not term or not definition_text:
                continue
            # 공공기관 페이지는 긴 설명 문장을 dt/strong으로 감싸는 경우가 있습니다.
            # 이러한 문장은 소제목이 아니라 본문 도입문으로 되돌립니다.
            if _looks_like_body_sentence_heading(term):
                result.append({
                    "type": "paragraph",
                    "text": term,
                    "source_semantic_fix": "sentence_term_to_paragraph",
                })
                result.extend(children)
                continue
            block["term"] = term
            block["definition_blocks"] = children
            block["definition_text"] = definition_text
            result.append(block)
            continue

        if kind == "faq":
            question = _clean_v44_text(block.get("question", ""))
            children = _normalize_blocks_v44(
                block.get("answer_blocks", [])
            )
            answer_text = blocks_text(children)
            if question and answer_text:
                block["question"] = question
                block["answer_blocks"] = children
                block["answer_text"] = answer_text
                result.append(block)
            continue

        if kind == "resource_group":
            title = _clean_v44_text(block.get("title", ""))
            children = _normalize_blocks_v44(
                block.get("content_blocks", [])
            )
            if not title:
                continue
            block["title"] = title
            block["content_blocks"] = children
            block["content_text"] = blocks_text(children)
            result.append(block)
            continue

        result.append(block)

    # 동일한 짧은 제목이 연속으로 반복되는 경우 한 번만 유지합니다.
    deduped: list[dict[str, Any]] = []
    for block in result:
        if (
            deduped
            and block.get("type") == "heading"
            and deduped[-1].get("type") == "heading"
            and norm(block.get("text")) == norm(deduped[-1].get("text"))
        ):
            continue
        deduped.append(block)
    return deduped


def refresh_document(document: dict[str, Any]) -> None:
    document["blocks"] = _normalize_blocks_v44(
        document.get("blocks", [])
    )
    _refresh_document_v43(document)


def normalize_review_manifest(review_df: pd.DataFrame) -> pd.DataFrame:
    manifest = _normalize_review_manifest_v43(review_df)

    defaults = {
        "chunk_strategy": "auto_structure",
        "table_group_key": "",
        "table_entity_column": "",
        "heading_context_policy": "inherit",
    }
    for column, default in defaults.items():
        if column not in manifest.columns:
            manifest[column] = default

    strategy_by_id = {
        "DP-004": "product_protection_by_sector",
        "BI-004": "table_row_entity",
        "MT-008": "application_document_hierarchy",
    }
    for url_id, strategy in strategy_by_id.items():
        manifest.loc[
            manifest["url_id"] == url_id,
            "chunk_strategy",
        ] = strategy

    manifest.loc[
        manifest["url_id"] == "DP-004",
        "table_group_key",
    ] = "구분"
    manifest.loc[
        manifest["url_id"] == "BI-004",
        "table_entity_column",
    ] = "금융회사명"

    # 정책 CSV에 열이 있는 경우 CSV 값을 우선합니다.
    if "문서_ID" in review_df.columns:
        lookup = review_df.fillna("").astype(str).set_index("문서_ID")
        column_map = {
            "청킹_전략": "chunk_strategy",
            "표_그룹키": "table_group_key",
            "표_엔터티열": "table_entity_column",
            "제목문맥_정책": "heading_context_policy",
        }
        for source_column, target_column in column_map.items():
            if source_column not in lookup.columns:
                continue
            values = manifest["url_id"].map(
                lookup[source_column]
            ).fillna("")
            mask = values.str.strip() != ""
            manifest.loc[mask, target_column] = values[mask]

    return manifest


def parse_html_document(
    html_bytes: bytes,
    final_url: str,
    manifest_row: dict[str, str],
    encoding: str,
) -> dict[str, Any]:
    document = _parse_html_document_v43(
        html_bytes,
        final_url,
        manifest_row,
        encoding,
    )
    document["chunk_strategy"] = manifest_row.get(
        "chunk_strategy",
        "auto_structure",
    )
    document["table_group_key"] = manifest_row.get(
        "table_group_key",
        "",
    )
    document["table_entity_column"] = manifest_row.get(
        "table_entity_column",
        "",
    )
    document["heading_context_policy"] = manifest_row.get(
        "heading_context_policy",
        "inherit",
    )

    embedded_documents = document.get(
        "_embedded_documents",
        [],
    )
    for child in embedded_documents:
        child["chunk_strategy"] = "legal_article_semantic"
        child["heading_context_policy"] = "inherit"
        if child.get("doc_id") == "MT-010_operation_layer01":
            child["related_documents"] = [{
                "document_id": "MT-010_rule_layer01",
                "relation_type": "definition_reference",
                "description": (
                    "시행세칙의 정의 조항이 상위 규정의 정의를 참조함"
                ),
            }]
            refresh_document(child)

    refresh_document(document)
    synchronize_document_navigation(document)
    return document


def _heading_stack_update(
    stack: list[tuple[int, str]],
    level: int,
    text: str,
) -> list[tuple[int, str]]:
    stack = [
        item for item in stack
        if item[0] < level
    ]
    stack.append((level, text))
    return stack


def _heading_prefix(
    stack: list[tuple[int, str]],
) -> str:
    lines = []
    for index, (_, text) in enumerate(stack):
        level = min(3 + index, 6)
        lines.append(f"{'#' * level} {text}")
    return "\n\n".join(lines)


def _split_prefixed_text(
    *,
    prefix: str,
    body: str,
    max_chars: int,
) -> list[str]:
    prefix = prefix.strip()
    body = body.strip()
    if not body:
        return [prefix] if prefix else []
    combined = "\n\n".join(
        part for part in [prefix, body] if part
    )
    if len(combined) <= max_chars:
        return [combined]

    allowance = max(
        200,
        max_chars - len(prefix) - 2,
    )
    parts = split_text_lossless(
        body,
        allowance,
    )
    return [
        "\n\n".join(
            part for part in [prefix, item] if part
        ).strip()
        for item in parts
        if item.strip()
    ]


def _definition_markdown(
    block: dict[str, Any],
    heading_stack: list[tuple[int, str]],
) -> tuple[str, str]:
    parent_prefix = _heading_prefix(heading_stack)
    term = block.get("term", "")
    term_level = min(3 + len(heading_stack), 6)
    term_heading = f"{'#' * term_level} {term}"
    body = render_blocks(
        block.get("definition_blocks", [])
    )
    prefix = "\n\n".join(
        part for part in [parent_prefix, term_heading]
        if part
    )
    return prefix, body


def _faq_markdown(
    block: dict[str, Any],
    heading_stack: list[tuple[int, str]],
) -> tuple[str, str]:
    parent_prefix = _heading_prefix(heading_stack)
    question_level = min(3 + len(heading_stack), 6)
    question = block.get("question", "")
    question_heading = (
        f"{'#' * question_level} Q. {question}"
    )
    body = render_blocks(
        block.get("answer_blocks", [])
    )
    prefix = "\n\n".join(
        part for part in [
            parent_prefix,
            question_heading,
        ]
        if part
    )
    return prefix, body


def _unique_nonempty(values: Iterable[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        text = norm(value)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _render_labeled_values(
    title: str,
    values: Iterable[str],
) -> str:
    cleaned = _unique_nonempty(values)
    if not cleaned:
        return ""
    lines = [f"#### {title}", ""]
    lines.extend(f"- {value}" for value in cleaned)
    return "\n".join(lines).strip()


def _find_primary_table(
    document: dict[str, Any],
) -> dict[str, Any] | None:
    return next(
        (
            block for block in document.get("blocks", [])
            if block.get("type") == "table"
        ),
        None,
    )


def _generic_units_v44(
    document: dict[str, Any],
    config: PipelineConfig,
) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    heading_stack: list[tuple[int, str]] = []
    paragraph_blocks: list[dict[str, Any]] = []

    def add_unit(
        content: str,
        section_title: str,
        chunk_type: str,
        **extra: Any,
    ) -> None:
        if not content.strip() or _noise_chunk(content):
            return
        units.append({
            "content": content.strip(),
            "section_title": section_title,
            "chunk_type": chunk_type,
            **extra,
        })

    def flush_paragraphs() -> None:
        nonlocal paragraph_blocks
        if not paragraph_blocks:
            return
        prefix = _heading_prefix(heading_stack)
        body = render_blocks(paragraph_blocks)
        for part in _split_prefixed_text(
            prefix=prefix,
            body=body,
            max_chars=config.chunk_max_chars,
        ):
            add_unit(
                part,
                heading_stack[-1][1] if heading_stack else "",
                "section",
                heading_path=[
                    item[1] for item in heading_stack
                ],
            )
        paragraph_blocks = []

    for block in document.get("blocks", []):
        kind = block.get("type")

        if kind == "heading":
            flush_paragraphs()
            heading_text = block.get("text", "")
            if (
                heading_stack
                and norm(heading_text)
                in {"안내", "상세 안내", "내용"}
            ):
                continue
            heading_stack = _heading_stack_update(
                heading_stack,
                int(block.get("level", 3)),
                heading_text,
            )
            continue

        if kind in {"paragraph", "list"}:
            paragraph_blocks.append(block)
            # 문단 버퍼가 너무 길어지면 제목 문맥을 반복하여 분할합니다.
            candidate = render_blocks(paragraph_blocks)
            prefix = _heading_prefix(heading_stack)
            if len(prefix) + len(candidate) > config.chunk_max_chars:
                last = paragraph_blocks.pop()
                flush_paragraphs()
                paragraph_blocks = [last]
            continue

        flush_paragraphs()

        if kind == "definition":
            prefix, body = _definition_markdown(
                block,
                heading_stack,
            )
            for part in _split_prefixed_text(
                prefix=prefix,
                body=body,
                max_chars=config.chunk_max_chars,
            ):
                add_unit(
                    part,
                    block.get("term", ""),
                    "definition",
                    heading_path=[
                        item[1] for item in heading_stack
                    ] + [block.get("term", "")],
                )
            continue

        if kind == "faq":
            prefix, body = _faq_markdown(
                block,
                heading_stack,
            )
            for part in _split_prefixed_text(
                prefix=prefix,
                body=body,
                max_chars=config.chunk_max_chars,
            ):
                add_unit(
                    part,
                    block.get("question", ""),
                    "faq",
                    heading_path=[
                        item[1] for item in heading_stack
                    ] + [block.get("question", "")],
                )
            continue

        if kind == "table":
            prefix = _heading_prefix(heading_stack)
            heading_title = (
                heading_stack[-1][1]
                if heading_stack
                else ""
            )
            for part in split_table(
                block,
                max(
                    200,
                    config.chunk_max_chars - len(prefix) - 2,
                ),
                "",
            ):
                content = "\n\n".join(
                    item for item in [prefix, part]
                    if item
                )
                add_unit(
                    content,
                    heading_title,
                    "table",
                    heading_path=[
                        item[1] for item in heading_stack
                    ],
                )
            continue

        if kind == "resource_group":
            content = render_blocks([block])
            add_unit(
                content,
                block.get("title", ""),
                "resource_group",
                resource_group_id=block.get(
                    "resource_group_id",
                    "",
                ),
                heading_path=[block.get("title", "")],
            )
            continue

        rendered = render_blocks([block])
        if rendered:
            paragraph_blocks.append(block)

    flush_paragraphs()
    return units


def _product_protection_units(
    document: dict[str, Any],
    config: PipelineConfig,
) -> list[dict[str, Any]]:
    units = []
    table = _find_primary_table(document)
    if not table:
        return _generic_units_v44(document, config)

    # 표 이외의 설명·각주는 일반 구조 청킹으로 유지합니다.
    non_table_document = {
        **document,
        "blocks": [
            block
            for block in document.get("blocks", [])
            if block is not table
        ],
    }
    units.extend(
        _generic_units_v44(
            non_table_document,
            config,
        )
    )

    headers = table.get("headers", [])
    rows = table.get("rows", [])
    if not rows:
        return units

    group_index = 0
    protected_index = (
        headers.index("보호")
        if "보호" in headers else 1
    )
    unprotected_index = (
        headers.index("비보호")
        if "비보호" in headers else 2
    )

    grouped: dict[str, dict[str, list[str]]] = {}
    for row in rows:
        if not row:
            continue
        sector = norm(row[group_index])
        if not sector:
            continue
        group = grouped.setdefault(
            sector,
            {"보호": [], "비보호": []},
        )
        if protected_index < len(row):
            group["보호"].append(
                row[protected_index]
            )
        if unprotected_index < len(row):
            group["비보호"].append(
                row[unprotected_index]
            )

    for sector, values in grouped.items():
        for protection_type in ["보호", "비보호"]:
            title = (
                f"{sector} {protection_type} 금융상품"
            )
            body_values = _unique_nonempty(
                values[protection_type]
            )
            body = "\n".join(
                f"- {value}"
                for value in body_values
            )
            prefix = (
                "### 보호대상 금융상품\n\n"
                f"#### {title}"
            )
            for part in _split_prefixed_text(
                prefix=prefix,
                body=body,
                max_chars=config.chunk_max_chars,
            ):
                units.append({
                    "content": part,
                    "section_title": title,
                    "chunk_type": "table_group",
                    "heading_path": [
                        "보호대상 금융상품",
                        title,
                    ],
                    "table_group_key": sector,
                    "protection_type": protection_type,
                })
    return units


def _table_row_entity_units(
    document: dict[str, Any],
    config: PipelineConfig,
) -> list[dict[str, Any]]:
    units = []
    table = _find_primary_table(document)
    if not table:
        return _generic_units_v44(document, config)

    non_table_blocks = []
    for block in document.get("blocks", []):
        if block is table:
            continue
        if (
            block.get("type") == "definition"
            and not block.get("definition_text")
        ):
            continue
        non_table_blocks.append(block)

    units.extend(
        _generic_units_v44(
            {
                **document,
                "blocks": non_table_blocks,
            },
            config,
        )
    )

    headers = table.get("headers", [])
    entity_column = document.get(
        "table_entity_column",
        "금융회사명",
    )
    try:
        entity_index = headers.index(
            entity_column
        )
    except ValueError:
        entity_index = 1 if len(headers) > 1 else 0

    for row_index, row in enumerate(
        table.get("rows", []),
        start=1,
    ):
        entity_name = (
            norm(row[entity_index])
            if entity_index < len(row)
            else f"행 {row_index}"
        )
        row_block = {
            "type": "table",
            "headers": headers,
            "rows": [row],
        }
        content = "\n\n".join([
            "### 보험금 지급대상 금융회사",
            f"#### {entity_name}",
            render_blocks([row_block]),
        ])
        units.append({
            "content": content,
            "section_title": entity_name,
            "chunk_type": "table_row",
            "heading_path": [
                "보험금 지급대상 금융회사",
                entity_name,
            ],
            "entity_name": entity_name,
            "table_row_index": row_index,
            "table_headers": headers,
            "table_row": row,
        })
    return units


def _application_document_units(
    document: dict[str, Any],
    config: PipelineConfig,
) -> list[dict[str, Any]]:
    units = []
    heading = ""
    for block in document.get("blocks", []):
        kind = block.get("type")
        if kind == "heading":
            heading = block.get("text", "")
            continue
        if kind != "table":
            continue

        headers = block.get("headers", [])
        rows = block.get("rows", [])
        if not rows:
            continue

        # 착오송금 신청의 다중 행 표만 신청자·서류 유형별로 재구성합니다.
        if (
            heading == "착오송금 신청"
            and {"구분", "구분 2"}.issubset(set(headers))
        ):
            grouped: dict[tuple[str, str, str], list[list[str]]] = {}
            for row in rows:
                applicant = norm(row[0]) if len(row) > 0 else ""
                category = norm(row[1]) if len(row) > 1 else ""
                scenario = ""

                if category == "추가 서류":
                    source = " ".join(row[2:4])
                    match = re.search(
                        r"([^。.!?]{1,30}?(?:을|를) 대리하는 경우)",
                        source,
                    )
                    scenario = (
                        norm(match.group(1))
                        if match
                        else "추가 서류"
                    )

                key = (
                    applicant,
                    category,
                    scenario,
                )
                grouped.setdefault(key, []).append(row)

            for (
                applicant,
                category,
                scenario,
            ), group_rows in grouped.items():
                group_title = " > ".join(
                    value
                    for value in [
                        applicant,
                        category,
                        scenario,
                    ]
                    if value
                )

                column_map: dict[str, list[str]] = {
                    header: []
                    for header in headers[2:]
                }
                for row in group_rows:
                    for index, header in enumerate(
                        headers[2:],
                        start=2,
                    ):
                        if index < len(row):
                            column_map[header].append(
                                row[index]
                            )

                sections = []
                for header in headers[2:]:
                    section = _render_labeled_values(
                        header,
                        column_map[header],
                    )
                    if section:
                        sections.append(section)

                prefix = (
                    "### 착오송금 신청\n\n"
                    f"#### {group_title}"
                )
                body = "\n\n".join(sections)
                for part in _split_prefixed_text(
                    prefix=prefix,
                    body=body,
                    max_chars=config.chunk_max_chars,
                ):
                    units.append({
                        "content": part,
                        "section_title": group_title,
                        "chunk_type": "application_documents",
                        "heading_path": [
                            "착오송금 신청",
                            group_title,
                        ],
                        "applicant_type": applicant,
                        "document_category": category,
                        "scenario": scenario,
                    })
            continue

        # 철회·계약해제·계좌변경·대리인 변경은 행별 완결 단위로 유지합니다.
        for row_index, row in enumerate(rows, start=1):
            row_title = (
                norm(row[0])
                if row else heading
            )
            row_block = {
                "type": "table",
                "headers": headers,
                "rows": [row],
            }
            content = "\n\n".join(
                value for value in [
                    f"### {heading}" if heading else "",
                    f"#### {row_title}" if row_title else "",
                    render_blocks([row_block]),
                ]
                if value
            )
            units.append({
                "content": content,
                "section_title": row_title or heading,
                "chunk_type": "application_documents",
                "heading_path": [
                    value
                    for value in [heading, row_title]
                    if value
                ],
                "table_row_index": row_index,
                "table_headers": headers,
                "table_row": row,
            })
    return units


def _legal_article_units(
    document: dict[str, Any],
    config: PipelineConfig,
) -> list[dict[str, Any]]:
    units = []
    chapter_context = ""
    revision_context: list[str] = []
    pending_appendix = False

    for block in document.get("blocks", []):
        kind = block.get("type")

        if kind == "paragraph":
            text = norm(block.get("text", ""))
            compact = re.sub(r"\s+", "", text)

            if re.match(r"^제\d+장", compact) or re.match(
                r"^제\d+절",
                compact,
            ):
                chapter_context = text
                continue

            if compact in {"부칙", "부칙"}:
                pending_appendix = True
                chapter_context = "부칙"
                continue

            if pending_appendix:
                prefix = "### 부칙"
                body = text
                units.append({
                    "content": f"{prefix}\n\n{body}",
                    "section_title": "부칙",
                    "chunk_type": "legal_effective_date",
                    "heading_path": ["부칙"],
                })
                pending_appendix = False
                continue

            if re.match(r"^(제정|개정)\s*\d", text):
                revision_context.append(text)
                continue

            # 기타 법령 문맥은 다음 조문에 붙입니다.
            revision_context.append(text)
            continue

        if kind != "definition":
            continue

        term = norm(block.get("term", ""))
        prefix_parts = []

        if revision_context:
            prefix_parts.extend(revision_context)
            revision_context = []

        if chapter_context:
            prefix_parts.append(
                f"### {chapter_context}"
            )

        prefix_parts.append(
            f"#### {term}"
            if chapter_context
            else f"### {term}"
        )
        prefix = "\n\n".join(prefix_parts)

        child_blocks = block.get(
            "definition_blocks",
            [],
        )

        lead_blocks = []
        remaining_blocks = []
        for child in child_blocks:
            if (
                child.get("type") == "paragraph"
                and not remaining_blocks
            ):
                lead_blocks.append(child)
            else:
                remaining_blocks.append(child)

        lead_text = render_blocks(lead_blocks)

        if not remaining_blocks:
            body = lead_text
            for part in _split_prefixed_text(
                prefix=prefix,
                body=body,
                max_chars=config.chunk_max_chars,
            ):
                units.append({
                    "content": part,
                    "section_title": term,
                    "chunk_type": "legal_article",
                    "heading_path": [
                        value for value in [
                            chapter_context,
                            term,
                        ] if value
                    ],
                })
            continue

        current_blocks = []
        for child in remaining_blocks:
            if child.get("type") == "list":
                items = child.get("items", [])
                current_items = []
                for item in items:
                    candidate_block = {
                        "type": "list",
                        "ordered": child.get(
                            "ordered",
                            False,
                        ),
                        "items": current_items + [item],
                    }
                    body_parts = []
                    if lead_text:
                        body_parts.append(lead_text)
                    body_parts.append(
                        render_blocks([candidate_block])
                    )
                    candidate_body = "\n\n".join(body_parts)

                    if (
                        current_items
                        and len(prefix) + len(candidate_body) + 2
                        > config.chunk_max_chars
                    ):
                        final_block = {
                            "type": "list",
                            "ordered": child.get(
                                "ordered",
                                False,
                            ),
                            "items": current_items,
                        }
                        body_parts = []
                        if lead_text:
                            body_parts.append(lead_text)
                        body_parts.append(
                            render_blocks([final_block])
                        )
                        units.append({
                            "content": (
                                prefix
                                + "\n\n"
                                + "\n\n".join(body_parts)
                            ),
                            "section_title": term,
                            "chunk_type": "legal_article",
                            "heading_path": [
                                value for value in [
                                    chapter_context,
                                    term,
                                ] if value
                            ],
                        })
                        current_items = [item]
                    else:
                        current_items.append(item)

                if current_items:
                    final_block = {
                        "type": "list",
                        "ordered": child.get(
                            "ordered",
                            False,
                        ),
                        "items": current_items,
                    }
                    body_parts = []
                    if lead_text:
                        body_parts.append(lead_text)
                    body_parts.append(
                        render_blocks([final_block])
                    )
                    units.append({
                        "content": (
                            prefix
                            + "\n\n"
                            + "\n\n".join(body_parts)
                        ),
                        "section_title": term,
                        "chunk_type": "legal_article",
                        "heading_path": [
                            value for value in [
                                chapter_context,
                                term,
                            ] if value
                        ],
                    })
                continue

            body = "\n\n".join(
                value for value in [
                    lead_text,
                    render_blocks([child]),
                ] if value
            )
            for part in _split_prefixed_text(
                prefix=prefix,
                body=body,
                max_chars=config.chunk_max_chars,
            ):
                units.append({
                    "content": part,
                    "section_title": term,
                    "chunk_type": "legal_article",
                    "heading_path": [
                        value for value in [
                            chapter_context,
                            term,
                        ] if value
                    ],
                })

    if revision_context:
        units.append({
            "content": "\n\n".join(revision_context),
            "section_title": document.get("title", ""),
            "chunk_type": "legal_context",
            "heading_path": [document.get("title", "")],
        })
    return units


def _related_links_for_chunk(
    document: dict[str, Any],
    content: str,
) -> list[dict[str, Any]]:
    related = []
    links = document.get("links", [])
    for link in links:
        anchor = norm(link.get("anchor_text", ""))
        url = link.get("url", "")
        if not url:
            continue
        if anchor and anchor in content:
            related.append(link)
        elif (
            anchor in {"(링크)", "링크"}
            and anchor in content
        ):
            related.append(link)
    return unique_dicts(
        related,
        ("url", "anchor_text", "link_role"),
    )


def _finalize_chunk_units_v44(
    document: dict[str, Any],
    units: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records = []
    seen_hashes: set[str] = set()
    action_types = sorted({
        action.get("action_type", "")
        for action in document.get("actions", [])
        if action.get("action_type")
    })

    for item in units:
        content = item.get("content", "").strip()
        if not content or _noise_chunk(content):
            continue

        # 제목만 있는 청크는 생성하지 않습니다.
        plain_lines = [
            re.sub(r"^#{1,6}\s*", "", line).strip()
            for line in content.splitlines()
            if line.strip()
        ]
        body_lines = [
            line
            for line in content.splitlines()
            if line.strip()
            and not re.match(r"^#{1,6}\s+\S", line)
        ]
        if plain_lines and not body_lines:
            continue

        content_hash = sha256_text(content)
        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)

        extra = {
            key: value
            for key, value in item.items()
            if key not in {
                "content",
                "section_title",
                "chunk_type",
                "resource_group_id",
            }
        }

        phone_numbers = sorted(set(
            re.findall(
                r"(?:0\d{1,2}[-)]\s*)?\d{3,4}-\d{4}",
                content,
            )
        ))
        related_links = _related_links_for_chunk(
            document,
            content,
        )

        record = {
            "chunk_id": (
                f"{document['doc_id']}_chunk_{len(records):03d}"
            ),
            "parent_doc_id": document.get(
                "parent_doc_id",
                document["doc_id"],
            ),
            "document_id": document["doc_id"],
            "attachment_id": document.get(
                "attachment_id",
            ),
            "record_type": document.get(
                "record_type",
                "page",
            ),
            "chunk_index": len(records),
            "title": document["title"],
            "section_title": item.get(
                "section_title",
                "",
            ),
            "chunk_type": item.get(
                "chunk_type",
                "section",
            ),
            "resource_group_id": item.get(
                "resource_group_id",
                document.get("resource_group_id", ""),
            ),
            "source_fragment": document.get(
                "source_fragment",
                "",
            ),
            "business_function": document[
                "business_function"
            ],
            "target_business_function": document[
                "target_business_function"
            ],
            "page_type": document["page_type"],
            "rag_index_mode": document.get(
                "rag_index_mode",
                "full",
            ),
            "source_url": document["source_url"],
            "official_download_url": document.get(
                "official_download_url",
            ),
            "available_action_types": action_types,
            "detected_phone_numbers": phone_numbers,
            "related_links": related_links,
            "related_documents": document.get(
                "related_documents",
                [],
            ),
            "content": content,
            "content_hash": content_hash,
            **extra,
        }
        records.append(record)

    return records


def _recover_missing_atomic_units_v45(
    document: dict[str, Any],
    units: list[dict[str, Any]],
    config: PipelineConfig,
) -> list[dict[str, Any]]:
    """구조별 청킹 과정에서 빠진 원문 문단·FAQ·표 행을 마지막 안전망으로 복구합니다."""
    combined = "\n".join(item.get("content", "") for item in units)
    recovered = 0
    for atomic_unit in atomic_index_units(document):
        if _unit_covered_by_chunks(atomic_unit, combined):
            continue
        for part in split_text_lossless(atomic_unit, config.chunk_max_chars):
            units.append({
                "content": part,
                "section_title": document.get("title", ""),
                "chunk_type": "recovered_atomic_unit",
                "heading_path": [document.get("title", "")],
                "recovery_reason": "구조 청킹 결과에서 원문 atomic unit 누락 감지",
            })
            combined += "\n" + part
            recovered += 1
    document["chunk_recovery_count"] = recovered
    return units


def structure_aware_chunks(
    document: dict[str, Any],
    config: PipelineConfig,
) -> list[dict[str, Any]]:
    if not document.get("indexable", True):
        return []

    # 이전 결과 문서를 재처리하는 경우에도 정규화 규칙을 적용합니다.
    refresh_document(document)

    strategy = document.get(
        "chunk_strategy",
        "",
    )
    if not strategy:
        strategy = {
            "DP-004": "product_protection_by_sector",
            "BI-004": "table_row_entity",
            "MT-008": "application_document_hierarchy",
        }.get(
            document.get("doc_id"),
            "legal_article_semantic"
            if document.get("record_type")
            == "modal_document"
            else "auto_structure",
        )

    if strategy == "product_protection_by_sector":
        units = _product_protection_units(
            document,
            config,
        )
    elif strategy == "table_row_entity":
        units = _table_row_entity_units(
            document,
            config,
        )
    elif strategy == "application_document_hierarchy":
        units = _application_document_units(
            document,
            config,
        )
    elif strategy == "legal_article_semantic":
        units = _legal_article_units(
            document,
            config,
        )
    else:
        units = _generic_units_v44(
            document,
            config,
        )

    units = _recover_missing_atomic_units_v45(
        document,
        units,
        config,
    )
    return _finalize_chunk_units_v44(
        document,
        units,
    )


def chunk_quality_findings(
    chunks: list[dict[str, Any]],
    *,
    large_warning_chars: int = 1200,
    short_warning_chars: int = 60,
) -> pd.DataFrame:
    findings = []
    for chunk in chunks:
        content = chunk.get("content", "")
        plain_len = _chunk_plain_len(content)
        body_lines = [
            line for line in content.splitlines()
            if line.strip()
            and not re.match(r"^#{1,6}\s+\S", line)
        ]

        if len(content) > large_warning_chars:
            findings.append({
                "severity": "warning",
                "finding_type": "large_chunk",
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "chars": len(content),
                "message": (
                    f"{large_warning_chars}자 초과 청크"
                ),
            })

        if not body_lines:
            findings.append({
                "severity": "error",
                "finding_type": "heading_only_chunk",
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "chars": len(content),
                "message": "제목만 있고 본문이 없는 청크",
            })
        elif plain_len < short_warning_chars:
            findings.append({
                "severity": "review",
                "finding_type": "short_chunk",
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "chars": len(content),
                "message": (
                    "짧지만 독립 사실일 수 있어 검토 필요"
                ),
            })

        if (
            chunk.get("chunk_type") == "legal_article"
            and not re.search(
                r"제\d+조(?:의\d+)?\s*\(",
                content,
            )
        ):
            findings.append({
                "severity": "error",
                "finding_type": "legal_article_title_missing",
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "chars": len(content),
                "message": "분할된 법령 청크에 조문 제목이 없음",
            })

    return pd.DataFrame(findings)


# ============================================================
# V4.5: 첨부파일 실수집·변경 감지·청크 무손실 안전망
# ============================================================


# ============================================================
# V4.6: 설명·목록 첨부파일 전용 행 기반 청킹
# ============================================================

_structure_aware_chunks_v451 = structure_aware_chunks


def _attachment_list_units_v46(
    document: dict[str, Any],
    config: PipelineConfig,
) -> list[dict[str, Any]]:
    text = str(document.get("content_text", "") or "").strip()
    if not text:
        return []

    max_chars = min(
        config.chunk_max_chars,
        max(400, config.attachment_chunk_max_chars),
    )
    max_rows = max(1, config.attachment_list_rows_per_chunk)
    source_lines = [norm(line) for line in text.splitlines() if norm(line)]

    # 과거 추출 결과처럼 줄바꿈이 사라진 경우에도 일반 안전 분할을 적용합니다.
    if len(source_lines) <= 1:
        source_lines = split_text_lossless(text, max_chars)

    units: list[dict[str, Any]] = []
    current_rows: list[str] = []
    current_chars = 0
    row_start = 1

    def flush() -> None:
        nonlocal current_rows, current_chars, row_start
        if not current_rows:
            return
        row_end = row_start + len(current_rows) - 1
        content = "\n".join(current_rows)
        units.append({
            "content": content,
            "section_title": document.get("title", ""),
            "chunk_type": "attachment_list_rows",
            "heading_path": [document.get("title", "")],
            "attachment_file_format": document.get("file_format", ""),
            "attachment_row_start": row_start,
            "attachment_row_end": row_end,
            "attachment_row_count": len(current_rows),
        })
        row_start = row_end + 1
        current_rows = []
        current_chars = 0

    for source_line in source_lines:
        line_parts = (
            [source_line]
            if len(source_line) <= max_chars
            else _split_long_fragment_lossless(source_line, max_chars)
        )
        for line in line_parts:
            projected = current_chars + len(line) + (1 if current_rows else 0)
            if current_rows and (
                projected > max_chars
                or len(current_rows) >= max_rows
            ):
                flush()
            current_rows.append(line)
            current_chars += len(line) + (1 if len(current_rows) > 1 else 0)
    flush()

    if any(len(item["content"]) > max_chars for item in units):
        raise AssertionError("첨부파일 목록 청크가 설정 길이를 초과했습니다.")
    return units


def structure_aware_chunks(
    document: dict[str, Any],
    config: PipelineConfig,
) -> list[dict[str, Any]]:
    if not document.get("indexable", True):
        return []

    if (
        document.get("record_type") == "attachment"
        and document.get("attachment_role") == "list"
    ):
        # DOCX/XLSX 행 경계를 보존하기 위해 refresh_document의 공백 정규화 전에 처리합니다.
        units = _attachment_list_units_v46(document, config)
        units = _recover_missing_atomic_units_v45(document, units, config)
        records = _finalize_chunk_units_v44(document, units)
        max_allowed = min(
            config.chunk_max_chars,
            max(400, config.attachment_chunk_max_chars),
        )
        oversized = [
            record["chunk_id"]
            for record in records
            if len(record.get("content", "")) > max_allowed
        ]
        if oversized:
            raise AssertionError(
                "첨부파일 전용 청킹 최대 길이 초과: " + ", ".join(oversized)
            )
        return records

    return _structure_aware_chunks_v451(document, config)
