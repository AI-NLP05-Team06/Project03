"""KDIC V1.5 pipeline engine -- EC2/non-Colab build.
Auto-extracted from the 2026-08-23 notebook (cells 0-93), with Colab-only
interactive code replaced by environment-variable-driven config.
"""
from __future__ import annotations

import os
import contextvars
from contextlib import contextmanager

# ==== cell 7 ====

import getpass
import hashlib
import json
import math
import os
import re
import shutil
import zipfile
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

import ipywidgets as widgets
import numpy as np
from elasticsearch import Elasticsearch, helpers
from IPython.display import JSON, Markdown, clear_output, display
from openai import BadRequestError, OpenAI
from tqdm.auto import tqdm


# 관리자 프롬프트 A/B 비교는 운영 프롬프트 전역값을 덮어쓰지 않는다.
# ContextVar로 현재 요청 스레드에만 후보 프롬프트를 주입하고, 일반 챗봇
# 요청은 KDIC_PROMPT_MANAGER의 운영 버전을 매 호출 시 읽는다.
_KDIC_PROMPT_OVERRIDES: contextvars.ContextVar[dict[str, str] | None] = (
    contextvars.ContextVar("kdic_prompt_overrides", default=None)
)


def _managed_prompt(name: str, default: str) -> str:
    overrides = _KDIC_PROMPT_OVERRIDES.get()
    if isinstance(overrides, dict) and name in overrides:
        return str(overrides[name])
    manager = globals().get("KDIC_PROMPT_MANAGER")
    if manager is not None:
        values = manager.active_values()
        if name in values:
            return str(values[name])
    return str(default)


@contextmanager
def kdic_prompt_overrides(values: dict[str, str] | None):
    """Temporarily apply prompt values to this execution context only."""
    token = _KDIC_PROMPT_OVERRIDES.set(dict(values or {}))
    try:
        yield
    finally:
        _KDIC_PROMPT_OVERRIDES.reset(token)

try:
    from google.colab import output as colab_output
    colab_output.enable_custom_widget_manager()
except Exception:
    pass


# ---------- 데이터 / 캐시 ----------
# 경로를 직접 지정하지 않으면 ZIP 업로드 창이 열립니다.
DATA_SOURCE: str | None = os.environ.get("KDIC_DATA_SOURCE")
DENSE_CACHE_FILENAME = "kdic_dense_structured_v2_embeddings.jsonl"
DENSE_CACHE_PATH = (
    Path(os.environ.get("KDIC_RUNTIME_DIR", "/opt/kdic/runtime")) / DENSE_CACHE_FILENAME
)

# ---------- HCX ----------
HCX_BASE_URL = "https://clovastudio.stream.ntruss.com/v1/openai"
HCX_EMBEDDING_MODEL = "bge-m3"
HCX_CHAT_MODEL = "HCX-005"
HCX_ENCODING_FORMAT = "float"
HCX_REQUEST_TIMEOUT = 120.0
HCX_MAX_RETRIES = 4

# ---------- 확정 검색 조건 ----------
# 문서 Dense 벡터는 Elasticsearch dense_vector에 저장하고 kNN으로 검색합니다.
# NUMPY_EXACT는 Elasticsearch kNN 비교 및 장애 fallback에만 사용합니다.
DENSE_BACKEND: Literal["ELASTICSEARCH_KNN", "NUMPY_EXACT"] = "ELASTICSEARCH_KNN"
DENSE_KNN_NUM_CANDIDATES = 200
ALLOW_NUMPY_DENSE_FALLBACK = True
DENSE_WEIGHT = 0.7
BM25_WEIGHT = 0.3
QUERY_FUSION_RRF_K = 10
CANDIDATE_DEPTH = 20
FINAL_TOP_K = 5
# BGE Reranker raw logit을 sigmoid로 0~1 변환한 뒤 적용합니다.
# 0.0은 기존 동작을 그대로 유지하며, 운영 반영 전 평가데이터셋 비교가 필요합니다.
MIN_RELEVANCE_SCORE = 0.0

# ---------- Reranker ----------
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
RERANKER_CANDIDATE_DEPTH = 20
RERANKER_BATCH_SIZE = 8
RERANKER_MAX_LENGTH = 512

# ---------- Parent-Child ----------
# 검색 순위는 Child 청크 기준으로 유지하고, Reranker Top-K 확정 뒤
# parent_doc_id가 같은 형제 청크를 Evidence Context로 확장합니다.
PARENT_CHILD_ENABLED = True
# None이면 전체 Parent를 사용합니다. 숫자를 넣으면 문자 수 기준으로
# matched child 우선 + 가까운 sibling 순서로 선택합니다.
PARENT_CONTEXT_MAX_CHARS: int | None = 8192

# ---------- 버전 / Elasticsearch ----------
DENSE_INPUT_VERSION = "kdic-dense-structured-v2-title-section-content-newline"
DENSE_CACHE_VERSION = "kdic-hcx-dense-structured-v2-cache-v1"
ES_EXPECTED_VERSION = "8.15.3"
ES_URL = os.environ.get("ELASTICSEARCH_URL", "http://127.0.0.1:9220")
ES_ANALYZER_NAME = "kdic_nori_none"
ES_INDEX_SCHEMA_VERSION = "kdic-hybrid-bm25-dense-v3"
FORCE_REBUILD_HYBRID_INDEX = False

assert math.isclose(DENSE_WEIGHT + BM25_WEIGHT, 1.0)
assert QUERY_FUSION_RRF_K > 0 and CANDIDATE_DEPTH > 0 and FINAL_TOP_K > 0
assert RERANKER_CANDIDATE_DEPTH == CANDIDATE_DEPTH
assert RERANKER_CANDIDATE_DEPTH >= FINAL_TOP_K
assert DENSE_KNN_NUM_CANDIDATES >= CANDIDATE_DEPTH
assert 0.0 <= MIN_RELEVANCE_SCORE <= 1.0


def retrieval_relevance_score(row: Mapping[str, Any]) -> float:
    """Return a comparable 0..1 relevance value for threshold filtering."""
    if row.get("reranker_score") is not None:
        raw = max(-60.0, min(60.0, float(row["reranker_score"])))
        return 1.0 / (1.0 + math.exp(-raw))
    return max(0.0, min(1.0, float(row.get("minmax_score") or 0.0)))


def filter_search_results_by_relevance(
    rows: Sequence[Mapping[str, Any]],
    threshold: float | None = None,
) -> list[dict[str, Any]]:
    minimum = MIN_RELEVANCE_SCORE if threshold is None else float(threshold)
    if not 0.0 <= minimum <= 1.0:
        raise ValueError("최소 관련성 점수는 0~1이어야 합니다.")
    filtered = []
    for row in rows:
        score = retrieval_relevance_score(row)
        if score >= minimum:
            filtered.append({**dict(row), "relevance_score": score})
    return [{**row, "rank": rank} for rank, row in enumerate(filtered, start=1)]

print({
    "answer_method": "B_BASIC_EVIDENCE_PACK",
    "dense": "HCX bge-m3 Dense-structured-v2 + Elasticsearch kNN",
    "dense_backend": DENSE_BACKEND,
    "dense_knn_num_candidates": DENSE_KNN_NUM_CANDIDATES,
    "sparse": "Elasticsearch BM25 + Nori-none",
    "weights": [DENSE_WEIGHT, BM25_WEIGHT],
    "fusion": "MINMAX",
    "query_fusion_rrf_k": QUERY_FUSION_RRF_K,
    "candidate_depth": CANDIDATE_DEPTH,
    "final_top_k": FINAL_TOP_K,
    "min_relevance_score": MIN_RELEVANCE_SCORE,
    "reranker": RERANKER_MODEL_NAME,
    "parent_child": PARENT_CHILD_ENABLED,
    "parent_context_max_chars": PARENT_CONTEXT_MAX_CHARS,
    "evidence_pack": True,
    "answer_skeleton": False,
    "fact_index": False,
    "fact_sheet": False,
})


# ---------- V1.5 질의분석 ----------
HCX_DECOMPOSITION_MODEL = "HCX-007"
V15_ORIGINAL_WEIGHT = 0.40
V15_SUBQUERY_TOTAL_WEIGHT = 0.60
V15_MIN_CONFIDENCE = 0.80
V15_MAX_SUBQUERIES = 4
V15_CACHE_PATH = Path(
    os.environ.get("KDIC_RUNTIME_DIR", "/opt/kdic/runtime")
) / "kdic_v15_chat_decomposition_cache.jsonl"

assert math.isclose(V15_ORIGINAL_WEIGHT + V15_SUBQUERY_TOTAL_WEIGHT, 1.0)

# ==== cell 9 ====
def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"JSONL 파싱 실패: {path}, line={line_number}") from error
            if not isinstance(record, dict):
                raise TypeError(f"JSONL 레코드가 객체가 아닙니다: {path}, line={line_number}")
            records.append(record)
    return records


def _safe_extract_zip(zip_path: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(zip_path, "r") as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"손상된 ZIP 항목: {bad_member}")
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"안전하지 않은 ZIP 경로: {member.filename}")
        archive.extractall(destination)
    return destination


def _find_unique_file(root: Path, filename: str) -> Path:
    matches = list(root.rglob(filename))
    processed = [path for path in matches if path.parent.name == "processed"]
    candidates = processed or matches
    if not candidates:
        raise FileNotFoundError(f"{filename}을 찾지 못했습니다: {root}")
    if len(candidates) != 1:
        raise RuntimeError(f"{filename} 후보가 여러 개입니다: {candidates}")
    return candidates[0]


def resolve_data_source(configured_path: str | None) -> Path:
    if configured_path:
        path = Path(configured_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"DATA_SOURCE 경로가 없습니다: {path}")

    try:
        from google.colab import files
    except ImportError as error:
        raise FileNotFoundError(
            "DATA_SOURCE에 KDIC_output ZIP 또는 압축 해제 폴더 경로를 지정하세요."
        ) from error

    print("KDIC_output ZIP 파일을 업로드하세요.")
    uploaded = files.upload()
    zip_names = [name for name in uploaded if name.lower().endswith(".zip")]
    if len(zip_names) != 1:
        raise RuntimeError(f"ZIP 파일을 정확히 1개 업로드해야 합니다: {list(uploaded)}")
    return Path("/content") / zip_names[0]


def prepare_data_root(source: Path) -> Path:
    if source.is_dir():
        return source
    if not zipfile.is_zipfile(source):
        raise ValueError(f"ZIP 파일이 아닙니다: {source}")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    destination = (
    Path(os.environ.get("KDIC_RUNTIME_DIR", "/opt/kdic/runtime")) / "kdic_data_a" / digest
)
    marker = destination / ".ready"
    if marker.exists():
        return destination
    if destination.exists():
        shutil.rmtree(destination)
    _safe_extract_zip(source, destination)
    marker.write_text("ready", encoding="utf-8")
    return destination


def _clean_text(value: Any) -> str:
    text = str(value or "").replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_chunks(data_root: Path) -> list[dict[str, Any]]:
    chunks_path = _find_unique_file(data_root, "chunks.jsonl")
    chunks = _read_jsonl(chunks_path)
    if not chunks:
        raise RuntimeError("chunks.jsonl이 비어 있습니다.")

    chunk_ids = [str(chunk.get("chunk_id") or "").strip() for chunk in chunks]
    if any(not chunk_id for chunk_id in chunk_ids):
        raise RuntimeError("빈 chunk_id가 있습니다.")
    if len(chunk_ids) != len(set(chunk_ids)):
        raise RuntimeError("중복 chunk_id가 있습니다.")
    if any(not _clean_text(chunk.get("content")) for chunk in chunks):
        raise RuntimeError("본문이 비어 있는 청크가 있습니다.")
    return chunks


def build_dense_structured_v2_text(chunk: dict[str, Any]) -> str:
    parts = [
        _clean_text(chunk.get("title")),
        _clean_text(chunk.get("section_title")),
        _clean_text(chunk.get("content")),
    ]
    text = "\n".join(part for part in parts if part)
    if not text:
        raise ValueError(f"Dense 입력이 비었습니다: {chunk.get('chunk_id')}")
    return text


DATA_PATH = resolve_data_source(DATA_SOURCE)
DATA_ROOT = prepare_data_root(DATA_PATH)
CHUNKS = load_chunks(DATA_ROOT)
CHUNKS_BY_ID = {str(chunk["chunk_id"]): chunk for chunk in CHUNKS}

# Parent-Child 인덱스: parent_doc_id가 없으면 document_id, 그것도 없으면 chunk_id를 사용합니다.
PARENT_CHILDREN_BY_ID: dict[str, list[dict[str, Any]]] = {}
CHUNK_PARENT_ID: dict[str, str] = {}
for chunk in CHUNKS:
    chunk_id = str(chunk["chunk_id"])
    parent_id = (
        _clean_text(chunk.get("parent_doc_id"))
        or _clean_text(chunk.get("document_id"))
        or chunk_id
    )
    CHUNK_PARENT_ID[chunk_id] = parent_id
    PARENT_CHILDREN_BY_ID.setdefault(parent_id, []).append(chunk)

for parent_id, children in PARENT_CHILDREN_BY_ID.items():
    children.sort(key=lambda row: (
        int(row.get("chunk_index") or 0),
        str(row.get("chunk_id") or ""),
    ))

dataset_hash = hashlib.sha256()
for chunk in CHUNKS:
    dataset_hash.update(str(chunk["chunk_id"]).encode("utf-8"))
    dataset_hash.update(b"\0")
    dataset_hash.update(build_dense_structured_v2_text(chunk).encode("utf-8"))
    dataset_hash.update(b"\0")
DATASET_FINGERPRINT = dataset_hash.hexdigest()
ES_INDEX_NAME = f"kdic-hybrid-nori-none-dense-v3-{DATASET_FINGERPRINT[:12]}"

print("데이터 경로:", DATA_ROOT)
print("청크 수:", len(CHUNKS))
print("Parent 문서 수:", len(PARENT_CHILDREN_BY_ID))
print("데이터 지문:", DATASET_FINGERPRINT[:16])
print("업무:", sorted({_clean_text(chunk.get("business_function")) for chunk in CHUNKS}))

# ==== cell 11 ====
def upload_dense_cache_from_browser(
    target_path: Path = DENSE_CACHE_PATH,
    *,
    reuse_existing: bool = True,
) -> Path:
    if reuse_existing and target_path.is_file() and target_path.stat().st_size > 0:
        print("이미 런타임에 있는 캐시를 재사용합니다:", target_path)
        return target_path

    try:
        from google.colab import files
    except ImportError as error:
        raise RuntimeError(
            f"Colab이 아니면 {target_path} 경로에 {DENSE_CACHE_FILENAME}을 직접 복사하세요."
        ) from error

    print(f"{DENSE_CACHE_FILENAME} 파일 하나를 업로드하세요.")
    uploaded = files.upload()
    if set(uploaded) != {DENSE_CACHE_FILENAME}:
        raise RuntimeError(
            "업로드 파일명이 정확하지 않습니다. "
            f"expected={DENSE_CACHE_FILENAME}, uploaded={list(uploaded)}"
        )

    payload = uploaded[DENSE_CACHE_FILENAME]
    if not payload:
        raise RuntimeError("업로드한 Dense 캐시 파일이 비어 있습니다.")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(payload)
    print(f"Dense 캐시 업로드 완료: {target_path} ({target_path.stat().st_size:,} bytes)")
    return target_path


# 두 번째 런타임부터 이 셀을 실행합니다. 최초 생성 시에는 셀 자체를 건너뛰세요.
UPLOADED_DENSE_CACHE_PATH = upload_dense_cache_from_browser()

# ==== cell 13 ====
def load_hcx_api_key() -> str:
    key: str | None = None
    try:
        from google.colab import userdata
        key = userdata.get("HCX_API_KEY")
    except Exception:
        key = None
    if not key:
        key = os.environ.get("HCX_API_KEY")
    if not key:
        raise RuntimeError(
            "HCX_API_KEY 환경변수가 설정되지 않았습니다. "
            "EC2에서는 대화형 입력(getpass) 대신 환경변수/시크릿으로만 주입해야 합니다."
        )

    key = str(key or "").strip()
    if not key:
        raise ValueError("HCX_API_KEY가 비어 있습니다.")
    if key.lower().startswith("bearer "):
        raise ValueError("HCX_API_KEY 앞에 'Bearer '를 붙이지 마세요.")
    if any(character.isspace() for character in key):
        raise ValueError("HCX_API_KEY 안에 공백 또는 줄바꿈이 있습니다.")
    return key


HCX_API_KEY = load_hcx_api_key()
HCX_CLIENT = OpenAI(
    api_key=HCX_API_KEY,
    base_url=HCX_BASE_URL,
    timeout=HCX_REQUEST_TIMEOUT,
    max_retries=HCX_MAX_RETRIES,
)


def embed_hcx_single(text: str) -> np.ndarray:
    cleaned = _clean_text(text)
    if not cleaned:
        raise ValueError("임베딩 입력이 비어 있습니다.")
    response = HCX_CLIENT.embeddings.create(
        model=HCX_EMBEDDING_MODEL,
        input=cleaned,
        encoding_format=HCX_ENCODING_FORMAT,
    )
    if len(response.data) != 1:
        raise RuntimeError(f"단일 임베딩 응답 개수가 1이 아닙니다: {len(response.data)}")
    vector = np.asarray(response.data[0].embedding, dtype=np.float32)
    if vector.ndim != 1 or vector.size == 0:
        raise RuntimeError(f"잘못된 임베딩 shape: {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise RuntimeError("임베딩에 NaN 또는 무한대가 있습니다.")
    return vector


print("HCX 클라이언트 준비 완료")
print("Dense 입력 예시:\n", build_dense_structured_v2_text(CHUNKS[0])[:500])

# ==== cell 14 ====
def structured_input_sha256(chunk: dict[str, Any]) -> str:
    text = build_dense_structured_v2_text(chunk)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_valid_dense_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    valid: dict[str, dict[str, Any]] = {}
    for record in _read_jsonl(path):
        chunk_id = str(record.get("chunk_id") or "")
        chunk = CHUNKS_BY_ID.get(chunk_id)
        if chunk is None:
            continue
        if record.get("model") != HCX_EMBEDDING_MODEL:
            continue
        if record.get("input_version") != DENSE_INPUT_VERSION:
            continue
        if record.get("cache_version") != DENSE_CACHE_VERSION:
            continue
        if record.get("input_sha256") != structured_input_sha256(chunk):
            continue
        vector = np.asarray(record.get("embedding"), dtype=np.float32)
        if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
            continue
        if int(record.get("dimensions") or 0) != vector.size:
            continue
        valid[chunk_id] = record
    return valid


def _write_dense_cache_atomic(path: Path, records_by_id: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        for chunk in CHUNKS:
            record = records_by_id.get(str(chunk["chunk_id"]))
            if record is not None:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temp_path, path)


def prepare_dense_embeddings(
    cache_path: Path = DENSE_CACHE_PATH,
    checkpoint_every: int = 5,
    allow_generate_missing: bool = False,
) -> tuple[np.ndarray, list[str]]:
    cache = _load_valid_dense_cache(cache_path)
    missing = [chunk for chunk in CHUNKS if str(chunk["chunk_id"]) not in cache]
    print(f"Dense cache: valid={len(cache)}, missing={len(missing)}")

    if missing and not allow_generate_missing:
        raise RuntimeError(
            "Dense 캐시에 유효한 문서 임베딩이 부족하므로 자동 생성을 중단했습니다. "
            f"valid={len(cache)}, missing={len(missing)}. "
            "기존 캐시 파일을 올바르게 업로드하거나, 최초 생성일 때만 "
            "CREATE_DENSE_CACHE_ONCE=True로 바꾼 뒤 이 셀을 다시 실행하세요."
        )

    try:
        for index, chunk in enumerate(
            tqdm(missing, desc="Dense-structured-v2 embedding"),
            start=1,
        ):
            chunk_id = str(chunk["chunk_id"])
            input_text = build_dense_structured_v2_text(chunk)
            vector = embed_hcx_single(input_text)
            cache[chunk_id] = {
                "chunk_id": chunk_id,
                "model": HCX_EMBEDDING_MODEL,
                "encoding_format": HCX_ENCODING_FORMAT,
                "input_version": DENSE_INPUT_VERSION,
                "input_sha256": hashlib.sha256(input_text.encode("utf-8")).hexdigest(),
                "cache_version": DENSE_CACHE_VERSION,
                "dimensions": int(vector.size),
                "embedding": vector.tolist(),
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            if index % checkpoint_every == 0:
                _write_dense_cache_atomic(cache_path, cache)
    finally:
        if cache:
            _write_dense_cache_atomic(cache_path, cache)

    ordered_vectors: list[np.ndarray] = []
    dimensions: set[int] = set()
    chunk_ids: list[str] = []
    for chunk in CHUNKS:
        chunk_id = str(chunk["chunk_id"])
        record = cache.get(chunk_id)
        if record is None:
            raise RuntimeError(f"Dense 캐시 누락: {chunk_id}")
        vector = np.asarray(record["embedding"], dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            raise RuntimeError(f"영벡터 임베딩: {chunk_id}")
        ordered_vectors.append(vector / norm)
        dimensions.add(int(vector.size))
        chunk_ids.append(chunk_id)

    if len(dimensions) != 1:
        raise RuntimeError(f"임베딩 차원 불일치: {dimensions}")
    return np.vstack(ordered_vectors), chunk_ids


def download_dense_cache_to_browser(
    cache_path: Path = DENSE_CACHE_PATH,
) -> None:
    if not cache_path.is_file() or cache_path.stat().st_size == 0:
        raise FileNotFoundError(f"다운로드할 Dense 캐시가 없습니다: {cache_path}")
    try:
        from google.colab import files
    except ImportError as error:
        raise RuntimeError(f"Colab 외부에서는 이 파일을 직접 가져가세요: {cache_path}") from error
    print(f"Dense 캐시 다운로드를 시작합니다: {cache_path.name}")
    files.download(str(cache_path))


# 최초 캐시를 만드는 단 한 번만 True로 바꾸세요.
# 이후 A~E 노트북에서는 False를 유지하고 캐시 업로드 셀을 실행합니다.
CREATE_DENSE_CACHE_ONCE = False

DENSE_MATRIX, DENSE_CHUNK_IDS = prepare_dense_embeddings(
    allow_generate_missing=CREATE_DENSE_CACHE_ONCE,
)
DENSE_DIMENSION = int(DENSE_MATRIX.shape[1])
DENSE_VECTOR_BY_ID = dict(zip(DENSE_CHUNK_IDS, DENSE_MATRIX))
if len(DENSE_VECTOR_BY_ID) != len(CHUNKS):
    raise RuntimeError(
        f"Dense 벡터 매핑 건수 불일치: vectors={len(DENSE_VECTOR_BY_ID)}, chunks={len(CHUNKS)}"
    )
missing_dense_ids = sorted(set(CHUNKS_BY_ID) - set(DENSE_VECTOR_BY_ID))
if missing_dense_ids:
    raise RuntimeError(f"Dense 벡터가 없는 청크가 있습니다: {missing_dense_ids[:10]}")

print("Dense matrix:", DENSE_MATRIX.shape)
print("Dense vector map:", len(DENSE_VECTOR_BY_ID))
print("Dense cache:", DENSE_CACHE_PATH)
if CREATE_DENSE_CACHE_ONCE:
    download_dense_cache_to_browser()
    print("다운로드한 파일을 보관하고 A~E 실험에서 공통으로 업로드해 사용하세요.")
else:
    print("문서 임베딩 API 호출 없이 업로드된 Dense 캐시를 사용했습니다.")

# ==== cell 16 ====
def connect_elasticsearch() -> Elasticsearch:
    client = Elasticsearch(
        ES_URL,
        request_timeout=120,
        max_retries=5,
        retry_on_timeout=True,
    )
    try:
        info = client.info()
    except Exception as error:
        raise RuntimeError(
            "Elasticsearch 연결 실패입니다. 2번 준비 셀의 마지막 로그를 확인하세요. "
            f"원인={type(error).__name__}: {error}"
        ) from error

    running_version = str(info["version"]["number"])
    if running_version != ES_EXPECTED_VERSION:
        raise RuntimeError(
            f"Elasticsearch 버전 불일치: running={running_version}, expected={ES_EXPECTED_VERSION}"
        )

    nodes = client.nodes.info(metric="plugins")
    plugin_names = {
        str(plugin.get("name") or "")
        for node in nodes["nodes"].values()
        for plugin in node.get("plugins", [])
    }
    if "analysis-nori" not in plugin_names:
        raise RuntimeError(f"analysis-nori 플러그인이 없습니다: {sorted(plugin_names)}")
    return client


def _hybrid_index_is_reusable(client: Elasticsearch) -> bool:
    if FORCE_REBUILD_HYBRID_INDEX:
        return False
    if not client.indices.exists(index=ES_INDEX_NAME):
        return False
    count = int(client.count(index=ES_INDEX_NAME)["count"])
    mapping = client.indices.get_mapping(index=ES_INDEX_NAME)
    mappings = mapping[ES_INDEX_NAME]["mappings"]
    metadata = mappings.get("_meta", {})
    properties = mappings.get("properties", {})
    embedding = properties.get("embedding", {})
    return (
        count == len(CHUNKS)
        and metadata.get("schema_version") == ES_INDEX_SCHEMA_VERSION
        and metadata.get("dataset_fingerprint") == DATASET_FINGERPRINT
        and metadata.get("dense_input_version") == DENSE_INPUT_VERSION
        and metadata.get("dense_model") == HCX_EMBEDDING_MODEL
        and int(metadata.get("dense_dimension") or 0) == DENSE_DIMENSION
        and embedding.get("type") == "dense_vector"
        and int(embedding.get("dims") or 0) == DENSE_DIMENSION
    )


def prepare_hybrid_nori_dense_index(client: Elasticsearch) -> None:
    if _hybrid_index_is_reusable(client):
        print(f"기존 BM25 + Dense 통합 인덱스 재사용: {ES_INDEX_NAME}")
        return

    # 이름에 schema v3와 데이터 지문이 포함된 전용 인덱스만 재생성합니다.
    if client.indices.exists(index=ES_INDEX_NAME):
        client.indices.delete(index=ES_INDEX_NAME)

    client.indices.create(
        index=ES_INDEX_NAME,
        settings={
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "similarity": {
                "kdic_bm25": {
                    "type": "BM25",
                    "k1": 1.2,
                    "b": 0.75,
                }
            },
            "analysis": {
                "tokenizer": {
                    "kdic_nori_none_tokenizer": {
                        "type": "nori_tokenizer",
                        "decompound_mode": "none",
                    }
                },
                "analyzer": {
                    ES_ANALYZER_NAME: {
                        "type": "custom",
                        "tokenizer": "kdic_nori_none_tokenizer",
                    }
                },
            },
        },
        mappings={
            "_meta": {
                "schema_version": ES_INDEX_SCHEMA_VERSION,
                "dataset_fingerprint": DATASET_FINGERPRINT,
                "dense_input_version": DENSE_INPUT_VERSION,
                "dense_model": HCX_EMBEDDING_MODEL,
                "dense_dimension": DENSE_DIMENSION,
            },
            "properties": {
                "chunk_id": {"type": "keyword"},
                "search_text": {
                    "type": "text",
                    "analyzer": ES_ANALYZER_NAME,
                    "search_analyzer": ES_ANALYZER_NAME,
                    "similarity": "kdic_bm25",
                },
                "embedding": {
                    "type": "dense_vector",
                    "dims": DENSE_DIMENSION,
                    "index": True,
                    "similarity": "dot_product",
                },
            },
        },
    )

    actions = (
        {
            "_op_type": "index",
            "_index": ES_INDEX_NAME,
            "_id": str(chunk["chunk_id"]),
            "_source": {
                "chunk_id": str(chunk["chunk_id"]),
                "search_text": build_dense_structured_v2_text(chunk),
                "embedding": DENSE_VECTOR_BY_ID[str(chunk["chunk_id"])].tolist(),
            },
        }
        for chunk in CHUNKS
    )
    bulk_client = client.options(request_timeout=120)
    success, errors = helpers.bulk(
        bulk_client,
        actions,
        chunk_size=100,
        max_retries=4,
        initial_backoff=1,
        max_backoff=8,
        raise_on_error=False,
        raise_on_exception=False,
    )
    client.indices.refresh(index=ES_INDEX_NAME)

    if errors:
        preview = json.dumps(errors[:3], ensure_ascii=False, default=str)[:3000]
        raise RuntimeError(f"Hybrid 인덱싱 실패 {len(errors)}건: {preview}")
    if int(success) != len(CHUNKS):
        raise RuntimeError(f"Hybrid 인덱싱 건수 불일치: success={success}, chunks={len(CHUNKS)}")

    actual_count = int(client.count(index=ES_INDEX_NAME)["count"])
    dense_count = int(
        client.count(
            index=ES_INDEX_NAME,
            query={"exists": {"field": "embedding"}},
        )["count"]
    )
    if actual_count != len(CHUNKS) or dense_count != len(CHUNKS):
        raise RuntimeError(
            "Hybrid 저장 건수 불일치: "
            f"documents={actual_count}, dense_vectors={dense_count}, chunks={len(CHUNKS)}"
        )

    analysis = client.indices.analyze(
        index=ES_INDEX_NAME,
        analyzer=ES_ANALYZER_NAME,
        text="예금자보호제도",
    )
    if not analysis.get("tokens"):
        raise RuntimeError("Nori 분석 결과가 비어 있습니다.")


ES = connect_elasticsearch()
prepare_hybrid_nori_dense_index(ES)

print("Elasticsearch:", ES.info()["version"]["number"])
print("BM25 + Dense 인덱스:", ES_INDEX_NAME)
print("통합 문서 수:", ES.count(index=ES_INDEX_NAME)["count"])
print("Dense 벡터 수:", ES.count(
    index=ES_INDEX_NAME,
    query={"exists": {"field": "embedding"}},
)["count"])
print("Nori-none 토큰:", [
    token["token"]
    for token in ES.indices.analyze(
        index=ES_INDEX_NAME,
        analyzer=ES_ANALYZER_NAME,
        text="예금자보호제도",
    )["tokens"]
])

# ==== cell 24 ====
def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise RuntimeError("질문 임베딩이 영벡터입니다.")
    return vector / norm


def dense_search_numpy_by_vector(
    query_vector: np.ndarray,
    depth: int = CANDIDATE_DEPTH,
) -> list[dict[str, Any]]:
    """현재 corpus 전체를 대상으로 한 NumPy exact dot-product 기준 검색."""
    if depth < 1:
        raise ValueError("depth는 1 이상이어야 합니다.")
    vector = _normalize_vector(query_vector)
    if vector.shape != (DENSE_DIMENSION,):
        raise RuntimeError(
            f"질문 임베딩 차원 불일치: query={vector.shape}, stored={DENSE_DIMENSION}"
        )
    scores = DENSE_MATRIX @ vector
    limit = min(depth, len(DENSE_CHUNK_IDS))
    candidate_indices = np.argpartition(-scores, limit - 1)[:limit]
    ordered_indices = sorted(
        candidate_indices.tolist(),
        key=lambda index: (-float(scores[index]), DENSE_CHUNK_IDS[index]),
    )
    return [
        {
            "chunk_id": DENSE_CHUNK_IDS[index],
            "score": float(scores[index]),
            "rank": rank,
        }
        for rank, index in enumerate(ordered_indices, start=1)
    ]


def dense_search_by_vector(
    query_vector: np.ndarray,
    depth: int = CANDIDATE_DEPTH,
) -> list[dict[str, Any]]:
    """Elasticsearch dense_vector kNN 검색."""
    if depth < 1:
        raise ValueError("depth는 1 이상이어야 합니다.")
    vector = _normalize_vector(query_vector)
    if vector.shape != (DENSE_DIMENSION,):
        raise RuntimeError(
            f"질문 임베딩 차원 불일치: query={vector.shape}, stored={DENSE_DIMENSION}"
        )
    response = ES.search(
        index=ES_INDEX_NAME,
        knn={
            "field": "embedding",
            "query_vector": vector.tolist(),
            "k": min(depth, len(CHUNKS)),
            "num_candidates": min(
                len(CHUNKS),
                max(depth, DENSE_KNN_NUM_CANDIDATES),
            ),
        },
        size=min(depth, len(CHUNKS)),
        source=["chunk_id"],
    )
    results = []
    seen: set[str] = set()
    for hit in response["hits"]["hits"]:
        chunk_id = str(hit["_source"]["chunk_id"])
        if chunk_id in seen:
            continue
        if chunk_id not in CHUNKS_BY_ID:
            raise RuntimeError(f"Dense kNN 결과의 chunk_id가 원본에 없습니다: {chunk_id}")
        seen.add(chunk_id)
        results.append({
            "chunk_id": chunk_id,
            "score": float(hit["_score"]),
            "rank": len(results) + 1,
        })
    if not results:
        raise RuntimeError("Elasticsearch Dense kNN 결과가 없습니다.")
    return results


_LAST_DENSE_BACKEND_USED = ""
_LAST_DENSE_FALLBACK_ERROR = ""


def dense_search_from_vector(
    query_vector: np.ndarray,
    depth: int = CANDIDATE_DEPTH,
) -> list[dict[str, Any]]:
    global _LAST_DENSE_BACKEND_USED, _LAST_DENSE_FALLBACK_ERROR
    _LAST_DENSE_FALLBACK_ERROR = ""
    if DENSE_BACKEND == "NUMPY_EXACT":
        _LAST_DENSE_BACKEND_USED = "NUMPY_EXACT"
        return dense_search_numpy_by_vector(query_vector, depth)
    if DENSE_BACKEND != "ELASTICSEARCH_KNN":
        raise ValueError(f"지원하지 않는 DENSE_BACKEND입니다: {DENSE_BACKEND}")
    try:
        results = dense_search_by_vector(query_vector, depth)
        _LAST_DENSE_BACKEND_USED = "ELASTICSEARCH_KNN"
        return results
    except Exception as error:
        if not ALLOW_NUMPY_DENSE_FALLBACK:
            raise
        _LAST_DENSE_BACKEND_USED = "NUMPY_EXACT_FALLBACK"
        _LAST_DENSE_FALLBACK_ERROR = f"{type(error).__name__}: {error}"
        return dense_search_numpy_by_vector(query_vector, depth)


def dense_search(question: str, depth: int = CANDIDATE_DEPTH) -> list[dict[str, Any]]:
    query_vector = _normalize_vector(embed_hcx_single(question))
    return dense_search_from_vector(query_vector, depth)

def bm25_search(question: str, depth: int = CANDIDATE_DEPTH) -> list[dict[str, Any]]:
    if depth < 1:
        raise ValueError("depth는 1 이상이어야 합니다.")
    response = ES.search(
        index=ES_INDEX_NAME,
        size=depth,
        query={"match": {"search_text": {"query": question}}},
    )
    results = []
    for rank, hit in enumerate(response["hits"]["hits"], start=1):
        chunk_id = str(hit["_source"]["chunk_id"])
        if chunk_id not in CHUNKS_BY_ID:
            raise RuntimeError(f"BM25 결과의 chunk_id가 원본에 없습니다: {chunk_id}")
        results.append({
            "chunk_id": chunk_id,
            "score": float(hit["_score"]),
            "rank": rank,
        })
    return results


def _minmax_by_chunk(results: list[dict[str, Any]]) -> dict[str, float]:
    if not results:
        return {}
    scores = np.asarray([float(row["score"]) for row in results], dtype=np.float64)
    low, high = float(scores.min()), float(scores.max())
    if abs(high - low) <= 1e-12:
        normalized = np.ones_like(scores)
    else:
        normalized = (scores - low) / (high - low)
    return {
        str(row["chunk_id"]): float(score)
        for row, score in zip(results, normalized)
    }


def weighted_minmax(
    dense_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    *,
    dense_weight: float = DENSE_WEIGHT,
    bm25_weight: float = BM25_WEIGHT,
    top_k: int = FINAL_TOP_K,
) -> list[dict[str, Any]]:
    if not math.isclose(dense_weight + bm25_weight, 1.0):
        raise ValueError("Dense/BM25 가중치 합은 1이어야 합니다.")
    dense_norm = _minmax_by_chunk(dense_results)
    bm25_norm = _minmax_by_chunk(bm25_results)
    dense_by_id = {str(row["chunk_id"]): row for row in dense_results}
    bm25_by_id = {str(row["chunk_id"]): row for row in bm25_results}
    candidates = []
    for chunk_id in sorted(set(dense_norm) | set(bm25_norm)):
        dense_row = dense_by_id.get(chunk_id)
        bm25_row = bm25_by_id.get(chunk_id)
        score = dense_weight * dense_norm.get(chunk_id, 0.0) + bm25_weight * bm25_norm.get(chunk_id, 0.0)
        candidates.append({
            "chunk_id": chunk_id,
            "minmax_score": float(score),
            "dense_rank": dense_row.get("rank") if dense_row else None,
            "dense_score": dense_row.get("score") if dense_row else None,
            "bm25_rank": bm25_row.get("rank") if bm25_row else None,
            "bm25_score": bm25_row.get("score") if bm25_row else None,
        })
    infinity = float("inf")
    ordered = sorted(candidates, key=lambda row: (
        -row["minmax_score"],
        row["dense_rank"] or infinity,
        row["bm25_rank"] or infinity,
        row["chunk_id"],
    ))
    return [
        {**row, "rank": rank, "chunk": CHUNKS_BY_ID[row["chunk_id"]]}
        for rank, row in enumerate(ordered[:top_k], start=1)
    ]


def hybrid_minmax_search(question: str, *, top_k: int = FINAL_TOP_K) -> list[dict[str, Any]]:
    cleaned = _clean_text(question)
    if not cleaned:
        raise ValueError("검색 질문이 비어 있습니다.")
    dense_results = dense_search(cleaned, CANDIDATE_DEPTH)
    bm25_results = bm25_search(cleaned, CANDIDATE_DEPTH)
    results = weighted_minmax(
        dense_results,
        bm25_results,
        dense_weight=DENSE_WEIGHT,
        bm25_weight=BM25_WEIGHT,
        top_k=top_k,
    )
    if not results:
        raise RuntimeError("Hybrid Min-Max 검색 결과가 없습니다.")
    return results


def fuse_query_results(
    plans: list[dict[str, Any]],
    *,
    top_k: int = FINAL_TOP_K,
    rrf_k: int = QUERY_FUSION_RRF_K,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not plans:
        raise ValueError("검색 계획이 없습니다.")
    if not math.isclose(sum(float(plan["weight"]) for plan in plans), 1.0, abs_tol=1e-9):
        raise ValueError("검색 계획 가중치 합은 1이어야 합니다.")
    per_query = []
    fused: dict[str, dict[str, Any]] = {}
    for plan_index, plan in enumerate(plans, start=1):
        started = time.perf_counter()
        hits = hybrid_minmax_search(str(plan["query"]), top_k=CANDIDATE_DEPTH)
        elapsed_ms = (time.perf_counter() - started) * 1000
        per_query.append({**plan, "latency_ms": elapsed_ms, "hits": hits})
        for hit in hits:
            chunk_id = str(hit["chunk_id"])
            row = fused.setdefault(chunk_id, {
                "chunk_id": chunk_id,
                "query_fusion_score": 0.0,
                "best_minmax_score": 0.0,
                "dense_rank": None,
                "bm25_rank": None,
                "matched_queries": [],
                "chunk": hit["chunk"],
            })
            row["query_fusion_score"] += float(plan["weight"]) / (rrf_k + int(hit["rank"]))
            row["best_minmax_score"] = max(row["best_minmax_score"], float(hit["minmax_score"]))
            if row["dense_rank"] is None or (hit["dense_rank"] is not None and hit["dense_rank"] < row["dense_rank"]):
                row["dense_rank"] = hit["dense_rank"]
            if row["bm25_rank"] is None or (hit["bm25_rank"] is not None and hit["bm25_rank"] < row["bm25_rank"]):
                row["bm25_rank"] = hit["bm25_rank"]
            row["matched_queries"].append({
                "plan_index": plan_index,
                "query": plan["query"],
                "source": plan["source"],
                "rank": hit["rank"],
                "weight": plan["weight"],
            })
    ordered = sorted(fused.values(), key=lambda row: (
        -row["query_fusion_score"], -row["best_minmax_score"], row["chunk_id"]
    ))
    final = []
    for rank, row in enumerate(ordered[:top_k], start=1):
        final.append({
            **row,
            "rank": rank,
            "minmax_score": row["best_minmax_score"],
        })
    return final, per_query


print("M3 Hybrid 7:3 Min-Max + Reranker 검색기 준비 완료")

# ==== cell 26 ====
# 상세 검색 레이턴시 버전으로 기존 함수를 재정의합니다.
_V15_LAST_QUERY_TRACE: dict[str, Any] = {}


def hybrid_minmax_search(question: str, *, top_k: int = FINAL_TOP_K) -> list[dict[str, Any]]:
    global _V15_LAST_QUERY_TRACE
    cleaned = _clean_text(question)
    if not cleaned:
        raise ValueError("검색 질문이 비어 있습니다.")
    total_started = time.perf_counter()

    embedding_started = time.perf_counter()
    query_vector = _normalize_vector(embed_hcx_single(cleaned))
    embedding_latency_ms = (time.perf_counter() - embedding_started) * 1000
    if query_vector.shape != (DENSE_DIMENSION,):
        raise RuntimeError(
            f"질문 임베딩 차원 불일치: query={query_vector.shape}, stored={DENSE_DIMENSION}"
        )

    dense_started = time.perf_counter()
    dense_results = dense_search_from_vector(query_vector, CANDIDATE_DEPTH)
    dense_compute_latency_ms = (time.perf_counter() - dense_started) * 1000

    bm25_started = time.perf_counter()
    bm25_results = bm25_search(cleaned, CANDIDATE_DEPTH)
    bm25_latency_ms = (time.perf_counter() - bm25_started) * 1000

    minmax_started = time.perf_counter()
    results = weighted_minmax(
        dense_results,
        bm25_results,
        dense_weight=DENSE_WEIGHT,
        bm25_weight=BM25_WEIGHT,
        top_k=top_k,
    )
    minmax_latency_ms = (time.perf_counter() - minmax_started) * 1000
    total_latency_ms = (time.perf_counter() - total_started) * 1000

    _V15_LAST_QUERY_TRACE = {
        "question": cleaned,
        "embedding_latency_ms": embedding_latency_ms,
        "dense_compute_latency_ms": dense_compute_latency_ms,
        "dense_backend_requested": DENSE_BACKEND,
        "dense_backend_used": _LAST_DENSE_BACKEND_USED,
        "dense_fallback_error": _LAST_DENSE_FALLBACK_ERROR,
        "dense_knn_num_candidates": DENSE_KNN_NUM_CANDIDATES,
        "bm25_latency_ms": bm25_latency_ms,
        "minmax_latency_ms": minmax_latency_ms,
        "query_total_latency_ms": total_latency_ms,
        "dense_candidate_count": len(dense_results),
        "bm25_candidate_count": len(bm25_results),
        "combined_candidate_count": len(results),
    }
    if not results:
        raise RuntimeError("Hybrid Min-Max 검색 결과가 없습니다.")
    return results


def fuse_query_results(
    plans: list[dict[str, Any]],
    *,
    top_k: int = FINAL_TOP_K,
    rrf_k: int = QUERY_FUSION_RRF_K,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not plans:
        raise ValueError("검색 계획이 없습니다.")
    if not math.isclose(sum(float(plan["weight"]) for plan in plans), 1.0, abs_tol=1e-9):
        raise ValueError("검색 계획 가중치 합은 1이어야 합니다.")

    per_query = []
    all_hits = []
    for plan_index, plan in enumerate(plans, start=1):
        hits = hybrid_minmax_search(str(plan["query"]), top_k=CANDIDATE_DEPTH)
        trace = dict(_V15_LAST_QUERY_TRACE)
        per_query.append({
            **plan,
            "plan_index": plan_index,
            "latency_ms": trace["query_total_latency_ms"],
            "latency_breakdown_ms": trace,
            "hits": hits,
        })
        all_hits.append((plan_index, plan, hits))

    fusion_started = time.perf_counter()
    fused: dict[str, dict[str, Any]] = {}
    for plan_index, plan, hits in all_hits:
        for hit in hits:
            chunk_id = str(hit["chunk_id"])
            row = fused.setdefault(chunk_id, {
                "chunk_id": chunk_id,
                "query_fusion_score": 0.0,
                "best_minmax_score": 0.0,
                "dense_rank": None,
                "bm25_rank": None,
                "matched_queries": [],
                "chunk": hit["chunk"],
            })
            row["query_fusion_score"] += float(plan["weight"]) / (rrf_k + int(hit["rank"]))
            row["best_minmax_score"] = max(row["best_minmax_score"], float(hit["minmax_score"]))
            if row["dense_rank"] is None or (hit["dense_rank"] is not None and hit["dense_rank"] < row["dense_rank"]):
                row["dense_rank"] = hit["dense_rank"]
            if row["bm25_rank"] is None or (hit["bm25_rank"] is not None and hit["bm25_rank"] < row["bm25_rank"]):
                row["bm25_rank"] = hit["bm25_rank"]
            row["matched_queries"].append({
                "plan_index": plan_index,
                "query": plan["query"],
                "source": plan["source"],
                "rank": hit["rank"],
                "weight": plan["weight"],
            })

    ordered = sorted(fused.values(), key=lambda row: (
        -row["query_fusion_score"], -row["best_minmax_score"], row["chunk_id"]
    ))
    final = [
        {
            **row,
            "rank": rank,
            "minmax_score": row["best_minmax_score"],
        }
        for rank, row in enumerate(ordered[:top_k], start=1)
    ]
    fusion_latency_ms = (time.perf_counter() - fusion_started) * 1000
    for row in per_query:
        row["query_fusion_latency_ms"] = fusion_latency_ms
    return final, per_query


print({
    "retrieval": "HYBRID_7_3_MINMAX",
    "dense_backend": DENSE_BACKEND,
    "dense_fallback": ALLOW_NUMPY_DENSE_FALLBACK,
    "dense_knn_num_candidates": DENSE_KNN_NUM_CANDIDATES,
    "query_fusion_rrf_k": QUERY_FUSION_RRF_K,
})

# ==== cell 28 ====
def compare_dense_backends_by_vector(
    query_vector: np.ndarray,
    *,
    depth: int = CANDIDATE_DEPTH,
) -> dict[str, Any]:
    exact = dense_search_numpy_by_vector(query_vector, depth)
    knn = dense_search_by_vector(query_vector, depth)
    exact_ids = [row["chunk_id"] for row in exact]
    knn_ids = [row["chunk_id"] for row in knn]
    overlap_ids = sorted(set(exact_ids) & set(knn_ids))
    return {
        "depth": depth,
        "exact_count": len(exact_ids),
        "knn_count": len(knn_ids),
        "overlap_count": len(overlap_ids),
        "overlap_rate": len(overlap_ids) / max(1, len(exact_ids)),
        "exact_top5": exact_ids[:5],
        "knn_top5": knn_ids[:5],
    }


probe_chunk_id = DENSE_CHUNK_IDS[0]
probe_report = compare_dense_backends_by_vector(
    DENSE_VECTOR_BY_ID[probe_chunk_id],
    depth=min(CANDIDATE_DEPTH, len(CHUNKS)),
)
if probe_report["knn_count"] != min(CANDIDATE_DEPTH, len(CHUNKS)):
    raise RuntimeError(f"Dense kNN 후보 수가 부족합니다: {probe_report}")
if probe_chunk_id not in probe_report["knn_top5"]:
    raise RuntimeError(f"자기 문서 벡터가 ES kNN Top-5에 없습니다: {probe_report}")
if probe_report["overlap_rate"] < 0.80:
    raise RuntimeError(f"NumPy Exact와 ES kNN Top-20 겹침이 지나치게 낮습니다: {probe_report}")

display(probe_report)
print("Elasticsearch Dense kNN 자체 검증 통과")

# ==== cell 31 ====
import torch
from sentence_transformers import CrossEncoder

from kdic_v15_context_rerank_core import rerank_candidates

RERANKER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RERANKER_MODEL = CrossEncoder(
    RERANKER_MODEL_NAME,
    device=RERANKER_DEVICE,
    max_length=RERANKER_MAX_LENGTH,
)

_FUSE_QUERY_RESULTS_BEFORE_RERANKER = fuse_query_results
_LAST_RERANK_TRACE: dict[str, Any] = {}


def _reranker_passage(chunk: dict[str, Any]) -> str:
    return build_dense_structured_v2_text(chunk)[:4_000]


def fuse_query_results(
    plans: list[dict[str, Any]],
    *,
    top_k: int = FINAL_TOP_K,
    rrf_k: int = QUERY_FUSION_RRF_K,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    global _LAST_RERANK_TRACE
    # 먼저 Hybrid/다중질의 결합 상위 20개를 확보합니다.
    candidates, per_query = _FUSE_QUERY_RESULTS_BEFORE_RERANKER(
        plans,
        top_k=RERANKER_CANDIDATE_DEPTH,
        rrf_k=rrf_k,
    )
    original_plan = next(
        (plan for plan in plans if "ORIGINAL" in str(plan.get("source") or "")),
        plans[0],
    )
    rerank_question = str(original_plan.get("query") or "").strip()
    reranked, trace = rerank_candidates(
        rerank_question,
        candidates,
        chunks_by_id=CHUNKS_BY_ID,
        model=RERANKER_MODEL,
        text_builder=_reranker_passage,
        candidate_depth=RERANKER_CANDIDATE_DEPTH,
        final_top_k=top_k,
        batch_size=RERANKER_BATCH_SIZE,
    )
    _LAST_RERANK_TRACE = {
        **trace,
        "model": RERANKER_MODEL_NAME,
        "device": RERANKER_DEVICE,
    }
    for row in per_query:
        row["reranker_latency_ms"] = float(trace["latency_ms"])
        row["reranker_candidate_count"] = int(trace["candidate_count"])
    return reranked, per_query


print({
    "reranker": RERANKER_MODEL_NAME,
    "device": RERANKER_DEVICE,
    "candidate_depth": RERANKER_CANDIDATE_DEPTH,
    "final_top_k": FINAL_TOP_K,
})

# ==== cell 33 ====
import time

_LAST_PARENT_CHILD_TRACE: dict[str, Any] = {}


def _parent_id_for_chunk(chunk: dict[str, Any]) -> str:
    chunk_id = _clean_text(chunk.get("chunk_id"))
    return (
        _clean_text(chunk.get("parent_doc_id"))
        or _clean_text(chunk.get("document_id"))
        or chunk_id
    )


def _select_parent_context_chunks(
    parent_id: str,
    matched_child_ids: list[str],
    *,
    max_chars: int | None = PARENT_CONTEXT_MAX_CHARS,
) -> list[dict[str, Any]]:
    """Parent의 sibling 청크를 문서 순서대로 반환합니다.

    max_chars=None이면 전체 Parent를 사용합니다.
    숫자이면 matched child를 반드시 우선 포함하고, 가장 가까운 sibling부터
    예산 안에서 추가한 뒤 최종 출력은 원문 chunk_index 순서로 정렬합니다.
    """
    children = list(PARENT_CHILDREN_BY_ID.get(parent_id) or [])
    if not children:
        return []

    if max_chars is None:
        return children

    if max_chars <= 0:
        raise ValueError("PARENT_CONTEXT_MAX_CHARS는 None 또는 양수여야 합니다.")

    positions = {
        str(chunk.get("chunk_id") or ""): index
        for index, chunk in enumerate(children)
    }
    matched_positions = [
        positions[chunk_id]
        for chunk_id in matched_child_ids
        if chunk_id in positions
    ]
    if not matched_positions:
        matched_positions = [0]

    def distance(index: int) -> tuple[int, int]:
        return (min(abs(index - anchor) for anchor in matched_positions), index)

    priority = sorted(range(len(children)), key=distance)
    selected_indices: list[int] = []
    used_chars = 0

    # matched child는 예산보다 길더라도 최소 1개는 보존합니다.
    matched_set = set(matched_child_ids)
    for index in priority:
        child = children[index]
        chunk_id = str(child.get("chunk_id") or "")
        content_chars = len(_clean_text(child.get("content")))
        must_include = chunk_id in matched_set or not selected_indices
        if must_include or used_chars + content_chars <= max_chars:
            selected_indices.append(index)
            used_chars += content_chars

    selected_indices.sort()
    return [children[index] for index in selected_indices]


def expand_parent_context(
    search_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reranker Top-K Child를 유지하면서 Parent Evidence 단위만 연결합니다."""
    global _LAST_PARENT_CHILD_TRACE
    started = time.perf_counter()

    if not PARENT_CHILD_ENABLED:
        output = []
        for result in search_results:
            row = dict(result)
            row["parent_doc_id"] = _parent_id_for_chunk(result["chunk"])
            row["parent_evidence_ref"] = f"C{int(result['rank'])}"
            row["parent_context_chunk_ids"] = [str(result["chunk_id"])]
            row["parent_context_chunk_count"] = 1
            row["parent_context_char_count"] = len(_clean_text(result["chunk"].get("content")))
            output.append(row)
        _LAST_PARENT_CHILD_TRACE = {
            "enabled": False,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "matched_child_count": len(search_results),
            "unique_parent_count": len(search_results),
            "expanded_chunk_count": len(search_results),
            "expanded_char_count": sum(row["parent_context_char_count"] for row in output),
        }
        return output

    # Top-K 안에서 같은 Parent가 여러 번 검색되면 하나의 Evidence ref를 공유합니다.
    parent_order: list[str] = []
    matched_by_parent: dict[str, list[str]] = {}
    for result in search_results:
        chunk = result["chunk"]
        parent_id = _parent_id_for_chunk(chunk)
        if parent_id not in matched_by_parent:
            parent_order.append(parent_id)
            matched_by_parent[parent_id] = []
        matched_by_parent[parent_id].append(str(result["chunk_id"]))

    evidence_ref_by_parent = {
        parent_id: f"C{index}"
        for index, parent_id in enumerate(parent_order, start=1)
    }
    selected_by_parent: dict[str, list[dict[str, Any]]] = {
        parent_id: _select_parent_context_chunks(
            parent_id,
            matched_by_parent[parent_id],
            max_chars=PARENT_CONTEXT_MAX_CHARS,
        )
        for parent_id in parent_order
    }

    output: list[dict[str, Any]] = []
    for result in search_results:
        row = dict(result)
        parent_id = _parent_id_for_chunk(result["chunk"])
        selected = selected_by_parent[parent_id]
        row["parent_doc_id"] = parent_id
        row["parent_evidence_ref"] = evidence_ref_by_parent[parent_id]
        row["parent_context_chunk_ids"] = [
            str(chunk.get("chunk_id") or "")
            for chunk in selected
        ]
        row["parent_context_chunk_count"] = len(selected)
        row["parent_context_char_count"] = sum(
            len(_clean_text(chunk.get("content")))
            for chunk in selected
        )
        output.append(row)

    unique_selected = {
        (parent_id, str(chunk.get("chunk_id") or ""))
        for parent_id, chunks in selected_by_parent.items()
        for chunk in chunks
    }
    _LAST_PARENT_CHILD_TRACE = {
        "enabled": True,
        "latency_ms": (time.perf_counter() - started) * 1000,
        "matched_child_count": len(search_results),
        "unique_parent_count": len(parent_order),
        "expanded_chunk_count": len(unique_selected),
        "expanded_char_count": sum(
            len(_clean_text(chunk.get("content")))
            for chunks in selected_by_parent.values()
            for chunk in chunks
        ),
        "parent_context_max_chars": PARENT_CONTEXT_MAX_CHARS,
        "parent_refs": evidence_ref_by_parent,
    }
    return output


def parent_child_markdown(search_results: list[dict[str, Any]]) -> str:
    if not search_results:
        return "### Parent-Child 확장\n\n검색 결과가 없습니다."
    seen: set[str] = set()
    lines = [
        "### Parent-Child 확장",
        "",
        "|Evidence|Parent|매칭 Child|확장 청크 수|확장 문자 수|",
        "|---|---|---|---:|---:|",
    ]
    for result in search_results:
        parent_id = str(result.get("parent_doc_id") or "")
        if parent_id in seen:
            continue
        seen.add(parent_id)
        ref = str(result.get("parent_evidence_ref") or "-")
        matched = [
            str(row["chunk_id"])
            for row in search_results
            if str(row.get("parent_doc_id") or "") == parent_id
        ]
        lines.append(
            f"|{ref}|{parent_id}|{', '.join(matched)}|"
            f"{int(result.get('parent_context_chunk_count') or 0)}|"
            f"{int(result.get('parent_context_char_count') or 0):,}|"
        )
    return "\n".join(lines)


print({
    "parent_child_enabled": PARENT_CHILD_ENABLED,
    "parent_context_max_chars": PARENT_CONTEXT_MAX_CHARS,
    "parent_count": len(PARENT_CHILDREN_BY_ID),
})

# ==== cell 36 ====


import random
from collections import OrderedDict
from types import SimpleNamespace

import kdic_v15_answer_b_core as answer_b_core
from kdic_v15_answer_b_core import (
    build_used_sources,
    evidence_explanation_to_markdown,
    evidence_pack_sha256,
    generate_basic_answer_b_v2,
    generate_evidence_explanation_b_v2,
)


# D안 최종답변 프롬프트의 근거 제한과 설명 원칙을 B안 입력 구조에 맞게 이전합니다.
# Answer Skeleton은 생성하지 않으며, Basic Evidence Pack에서 기본답변을 직접 만듭니다.
answer_b_core.BASIC_ANSWER_SYSTEM_PROMPT = """
당신은 예금보험공사 공식 문서 기반 질의응답 시스템입니다.

반드시 지킬 규칙:
1. 사용자 질문과 Basic Evidence Pack에 적힌 사실만 사용하여 한국어로 답하세요.
2. 질문에 먼저 직접 답한 뒤 필요한 대상·조건·예외·금액·기간·절차를 설명하세요.
3. Basic Evidence Pack에 없는 사실을 추정하거나 일반상식으로 보완하지 마세요.
4. 서로 다른 대상이나 제도의 내용을 임의로 결합하지 마세요.
5. 청크 사이에 차이가 있으면 한쪽을 임의로 선택하지 말고 확인 가능한 차이를 설명하세요.
6. 근거가 부족한 필수 내용은 추측하지 말고 missing_information에 기록하세요.
7. URL, 전화번호, 추천 질문, 추천 키워드를 답변 본문에 작성하지 마세요.
8. 답변을 짧게 줄이는 것보다 사용자가 이해할 수 있게 충분히 설명하는 것을 우선하세요.
9. 전문용어는 공식 용어를 사용하되 같은 문장이나 다음 문장에서 쉽게 풀어 설명하세요.
10. 일반 조건과 예외를 분리하고, 절차는 Evidence에 순서가 있을 때 번호로 설명하세요.
11. 근거를 사용한 문장 끝에 [E1] 형식으로 실제 evidence_id를 표시하세요.
12. 검색 점수, Basic Evidence Pack, JSON, 내부 구현을 답변 본문에서 언급하지 마세요.
13. 지정된 JSON 객체 하나만 출력하고 문자열 내부 줄바꿈은 \\n으로 이스케이프하세요.
""".strip()

answer_b_core.EVIDENCE_EXPLANATION_SYSTEM_PROMPT = """
당신은 예금보험공사 기본답변의 공식 문서 근거를 설명하는 시스템입니다.

반드시 지킬 규칙:
1. 기본답변 생성에 사용한 것과 SHA-256이 동일한 Basic Evidence Pack만 사용하세요.
2. 기본답변의 핵심 주장과 Evidence ID·Chunk ID 연결을 문서 내용 기준으로 설명하세요.
3. 왜 해당 Evidence가 질문과 주장에 관련되는지 사용자가 확인 가능한 문장으로 설명하세요.
4. 적용 조건·예외·근거 한계·추가 필요 정보를 구분하세요.
5. 기본답변과 모순되는 새 결론이나 Evidence에 없는 사실을 추가하지 마세요.
6. 모델의 숨겨진 사고과정이나 내부 추론을 출력하지 마세요.
7. URL과 전화번호를 생성하지 마세요. 출처는 프로그램이 별도로 표시합니다.
8. 검색 점수나 내부 구현을 설명하지 마세요.
9. 지정된 JSON 객체 하나만 출력하고 문자열 내부 줄바꿈은 \\n으로 이스케이프하세요.
""".strip()


ANSWER_API_MIN_INTERVAL_SECONDS = 2.0
ANSWER_API_MAX_ATTEMPTS = 5
_LAST_ANSWER_API_TRACE: dict[str, Any] = {}


def _header_seconds(value: Any) -> float | None:
    text = str(value or "").strip().lower()
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
    return float(match.group(1)) if match else None


def _answer_retry_delay(error: Exception, attempt: int) -> float:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None) or {}
    values = []
    for key in (
        "retry-after", "Retry-After",
        "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens",
    ):
        seconds = _header_seconds(headers.get(key))
        if seconds is not None:
            values.append(seconds)
    if values:
        return min(120.0, max(1.0, max(values)) + random.uniform(0.1, 0.9))
    return min(60.0, 5.0 * (2 ** max(0, attempt - 1)) + random.uniform(0.1, 1.0))


class _RateLimitedAnswerCompletions:
    def __init__(self, base_completions: Any):
        self.base_completions = base_completions
        self.last_call_at = 0.0

    def create(self, **kwargs: Any) -> Any:
        global _LAST_ANSWER_API_TRACE
        started = time.perf_counter()
        attempts = []
        total_wait_seconds = 0.0
        for attempt in range(1, ANSWER_API_MAX_ATTEMPTS + 1):
            elapsed = time.perf_counter() - self.last_call_at
            pacing_wait = max(0.0, ANSWER_API_MIN_INTERVAL_SECONDS - elapsed)
            if pacing_wait:
                time.sleep(pacing_wait)
                total_wait_seconds += pacing_wait
            self.last_call_at = time.perf_counter()
            try:
                response = self.base_completions.create(**kwargs)
                attempts.append({"attempt": attempt, "status": "SUCCESS"})
                _LAST_ANSWER_API_TRACE = {
                    "attempts": attempts,
                    "total_wait_ms": total_wait_seconds * 1000,
                    "wall_latency_ms": (time.perf_counter() - started) * 1000,
                }
                return response
            except Exception as error:
                if type(error).__name__ != "RateLimitError":
                    raise
                delay = _answer_retry_delay(error, attempt)
                attempts.append({
                    "attempt": attempt, "status": "RATE_LIMIT_429",
                    "delay_seconds": delay,
                })
                if attempt >= ANSWER_API_MAX_ATTEMPTS:
                    _LAST_ANSWER_API_TRACE = {
                        "attempts": attempts,
                        "total_wait_ms": total_wait_seconds * 1000,
                        "wall_latency_ms": (time.perf_counter() - started) * 1000,
                    }
                    raise RuntimeError(
                        "HCX-005 답변 생성 단계에서 429 재시도를 모두 소진했습니다. "
                        "잠시 후 다시 시도하세요."
                    ) from error
                time.sleep(delay)
                total_wait_seconds += delay
        raise AssertionError("도달할 수 없는 답변 재시도 상태")


class _RateLimitedAnswerClient:
    def __init__(self, base_client: Any):
        self.chat = SimpleNamespace(
            completions=_RateLimitedAnswerCompletions(base_client.chat.completions)
        )
        # 현재 HCX OpenAI 호환 API에서 두 response_format이 거부된 것이 확인됐으므로
        # 첫 질문부터 prompt JSON 방식만 사용합니다.
        self._kdic_structured_output_capability = {HCX_CHAT_MODEL: False}


_ANSWER_BASE_CLIENT = (
    HCX_CLIENT.with_options(max_retries=0)
    if hasattr(HCX_CLIENT, "with_options")
    else HCX_CLIENT
)
ANSWER_HCX_CLIENT = _RateLimitedAnswerClient(_ANSWER_BASE_CLIENT)


def build_parent_basic_evidence_pack(
    question: str,
    search_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Reranker Top-5를 동일 Parent별로 묶어 결정적인 B안 Pack을 만든다."""
    by_parent: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for result in search_results:
        child = result["chunk"]
        parent_id = str(result.get("parent_doc_id") or _parent_id_for_chunk(child))
        target = by_parent.setdefault(parent_id, {
            "rank": int(result["rank"]),
            "parent_id": parent_id,
            "representative_chunk_id": str(result["chunk_id"]),
            "matched_child_ids": [],
            "matched_child_ranks": [],
            "context_chunk_ids": list(result.get("parent_context_chunk_ids") or [str(result["chunk_id"])]),
            "document_title": _clean_text(child.get("title") or child.get("document_title")),
            "source_url": _clean_text(child.get("source_url")),
        })
        target["matched_child_ids"].append(str(result["chunk_id"]))
        target["matched_child_ranks"].append(int(result["rank"]))

    evidence = []
    sources: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for index, row in enumerate(by_parent.values(), start=1):
        context_parts = []
        section_titles = []
        valid_context_ids = []
        for chunk_id in row["context_chunk_ids"]:
            chunk = CHUNKS_BY_ID.get(str(chunk_id))
            if chunk is None:
                raise KeyError(f"Parent context 청크가 corpus에 없습니다: {chunk_id}")
            valid_context_ids.append(str(chunk_id))
            title = _clean_text(chunk.get("title"))
            section = _clean_text(chunk.get("section_title"))
            if section and section not in section_titles:
                section_titles.append(section)
            label = " / ".join(value for value in (title, section) if value)
            content = _clean_text(chunk.get("content"))
            context_parts.append(f"[{chunk_id}] {label}\n{content}".strip())

        joined_content = "\n\n".join(context_parts)
        context_truncated = len(joined_content) > PARENT_CONTEXT_MAX_CHARS
        if context_truncated:
            limited = joined_content[:PARENT_CONTEXT_MAX_CHARS]
            boundary = max(limited.rfind("\n"), limited.rfind(" "))
            if boundary >= int(PARENT_CONTEXT_MAX_CHARS * 0.8):
                limited = limited[:boundary]
            joined_content = limited.rstrip()

        evidence_id = f"E{index}"
        evidence.append({
            "evidence_id": evidence_id,
            "rank": int(row["rank"]),
            "chunk_id": row["representative_chunk_id"],
            "parent_id": row["parent_id"],
            "context_chunk_ids": valid_context_ids,
            "matched_child_ids": list(dict.fromkeys(row["matched_child_ids"])),
            "matched_child_ranks": sorted(set(row["matched_child_ranks"])),
            "document_title": row["document_title"],
            "section_title": " · ".join(section_titles),
            "content": joined_content,
            "context_char_count": len(joined_content),
            "context_truncated": context_truncated,
            "source_url": row["source_url"],
        })
        url = row["source_url"]
        if url:
            source = sources.setdefault(url, {
                "source_id": f"S{len(sources) + 1}",
                "title": row["document_title"] or "공식 출처",
                "source_url": url,
                "evidence_ids": [],
            })
            source["evidence_ids"].append(evidence_id)

    if not evidence:
        raise ValueError("Basic Evidence Pack을 만들 검색 결과가 없습니다.")
    return {
        "question": _clean_text(question),
        "parent_context_max_chars": PARENT_CONTEXT_MAX_CHARS,
        "evidence": evidence,
        "sources": list(sources.values()),
    }


def format_basic_evidence_pack(pack: dict[str, Any]) -> str:
    return json.dumps(pack, ensure_ascii=False, indent=2, default=str)


def sources_to_markdown_b(sources: list[dict[str, Any]]) -> str:
    if not sources:
        return ""
    lines = ["### 출처", ""]
    for source in sources:
        title = str(source.get("title") or "공식 출처")
        url = str(source.get("url") or source.get("source_url") or "")
        evidence_ids = ", ".join(source.get("evidence_ids") or [])
        if url:
            lines.append(f"- [{title}]({url}) — {evidence_ids}")
    return "\n".join(lines) if len(lines) > 2 else ""


print({
    "answer_system": "B_BASIC_EVIDENCE_PACK",
    "initial_hcx005_calls": 1,
    "detail_policy": "SAME_EVIDENCE_PACK_ON_DEMAND",
    "answer_skeleton": False,
    "parent_context_max_chars": PARENT_CONTEXT_MAX_CHARS,
})

# 사용자에게 보이는 답변은 구조화하되, 내부 Evidence ID는 검증용으로만 유지합니다.
answer_b_core.BASIC_ANSWER_SYSTEM_PROMPT = """
당신은 예금보험공사 공식 문서 기반 답변 시스템입니다.

근거 제한:
1. 사용자 질문과 Basic Evidence Pack에 포함된 사실만 사용하세요.
2. Evidence에 없는 사실·URL·전화번호·조건을 추정하거나 일반상식으로 보완하지 마세요.
3. 서로 다른 업무·대상·신청인의 조건을 임의로 결합하지 마세요.
4. 근거가 부족한 필수 내용은 추측하지 말고 missing_information에 기록하세요.

답변 구조:
5. 첫 문단에서 질문에 대한 결론을 직접 제시하세요.
6. 질문에 독립 요구가 둘 이상이면 요구별 Markdown 소제목으로 나누세요.
7. 절차 질문은 Evidence에 순서가 있을 때 번호 목록으로 작성하세요.
8. 비교 질문은 비교 기준이 둘 이상이면 간결한 Markdown 표를 사용하세요.
9. 조건·기한·금액·필요서류·예외는 질문과 관련된 항목만 별도로 구분하세요.
10. 해당 내용이 없는데 형식만 맞추기 위한 빈 소제목을 만들지 마세요.
11. 전문용어는 공식 명칭을 사용하고 바로 이해할 수 있게 풀어 설명하세요.

출력·내부 검증:
12. answer 문자열에는 근거 문장 끝에 [E1] 형식의 실제 evidence_id를 표시하세요.
    이 표시는 프로그램이 검증 후 사용자 화면에서 숨깁니다.
13. used_evidence_ids와 used_chunk_ids에는 실제 사용한 값만 넣으세요.
14. 검색 점수, JSON, Evidence Pack, 내부 구현은 answer 본문에서 언급하지 마세요.
15. 지정된 JSON 객체 하나만 출력하고 문자열 내부 줄바꿈은 \\n으로 이스케이프하세요.
""".strip()

answer_b_core.EVIDENCE_EXPLANATION_SYSTEM_PROMPT = """
당신은 예금보험공사 기본답변이 공식 문서의 어떤 내용에 근거했는지 설명하는 시스템입니다.

반드시 지킬 규칙:
1. 기본답변 생성에 사용한 것과 SHA-256이 동일한 Basic Evidence Pack만 사용하세요.
2. claim에는 기본답변의 핵심 안내 내용을 적으세요.
3. relevance_reason에는 반드시 다음 두 내용을 한 문단으로 적으세요.
   - 공식 문서에서 확인되는 근거 내용을 구체적으로 요약
   - 그 문서 내용 때문에 기본답변의 해당 안내를 할 수 있었던 연결 이유
4. relevance_reason을 'E1이 근거다'처럼 ID만 나열하는 문장으로 작성하지 마세요.
5. 적용 조건·예외·근거 한계·추가 필요 정보를 구분하세요.
6. 기본답변과 모순되는 새 결론이나 Evidence에 없는 사실을 추가하지 마세요.
7. 숨겨진 사고과정이나 내부 추론은 출력하지 말고, 사용자가 문서에서 확인할 수 있는 근거 관계만 설명하세요.
8. URL과 전화번호는 생성하지 마세요. 공식 출처는 프로그램이 별도로 표시합니다.
9. 지정된 JSON 객체 하나만 출력하고 문자열 내부 줄바꿈은 \\n으로 이스케이프하세요.
""".strip()

# 비교 실행에서 연속 HCX 호출로 429가 발생할 때를 위한 공통 답변 호출 간격입니다.
ANSWER_API_MIN_INTERVAL_SECONDS = 3.0
ANSWER_API_MAX_ATTEMPTS = 6


def user_visible_answer(text: Any) -> str:
    """내부 검증용 [E#] 표지만 사용자 화면에서 제거합니다."""
    value = str(text or "")
    value = re.sub(r"\s*\[(?:E\d+)(?:\s*,\s*E\d+)*\]", "", value)
    value = re.sub(r"\bE\d+\b", "해당 공식 문서", value)
    value = re.sub(
        r"`?[A-Z]{2,}-[A-Za-z0-9_-]+_chunk_[A-Za-z0-9_-]+`?",
        "관련 문서 구간",
        value,
    )
    value = re.sub(r"[ \t]+\n", "\n", value)
    return value.strip()


def sources_to_markdown_user(sources: list[dict[str, Any]]) -> str:
    lines = ["### 공식 출처", ""]
    seen: set[str] = set()
    for source in sources:
        title = str(source.get("title") or "공식 출처").strip()
        url = str(source.get("url") or source.get("source_url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        lines.append(f"- [{title}]({url})")
    return "\n".join(lines) if len(lines) > 2 else ""


def evidence_explanation_to_user_markdown(payload: Mapping[str, Any]) -> str:
    """Evidence/Chunk ID 대신 문서 내용과 답변의 연결을 사용자에게 설명합니다."""
    lines = [
        "### 이 답변을 낸 근거", "",
        user_visible_answer(answer_b_core._clean(payload.get("explanation_summary"))), "",
    ]
    for index, item in enumerate(payload.get("claim_evidence_map") or [], start=1):
        claim = user_visible_answer(answer_b_core._clean(item.get("claim")))
        reason = user_visible_answer(answer_b_core._clean(item.get("relevance_reason")))
        reason = re.sub(r"\bE\d+\b", "해당 공식 문서", reason)
        lines.extend([
            f"#### {index}. 답변에서 안내한 내용",
            "",
            claim,
            "",
            "**문서에서 확인한 내용과 답변의 연결**",
            "",
            reason,
            "",
        ])
    for title, key in (
        ("적용 조건", "conditions"),
        ("예외", "exceptions"),
        ("현재 문서 근거의 한계", "limitations"),
        ("정확한 판단에 추가로 필요한 정보", "additional_information_needed"),
    ):
        values = answer_b_core._clean_list(payload.get(key))
        if values:
            lines.extend([f"#### {title}", ""])
            lines.extend(f"- {user_visible_answer(value)}" for value in values)
            lines.append("")
    return "\n".join(lines).strip()


print({
    "answer_system": "B_STRUCTURED_BASIC_EVIDENCE_PACK",
    "user_visible_evidence_ids": False,
    "evidence_detail": "DOCUMENT_CONTENT_TO_ANSWER_CONNECTION",
})

# ==== cell 42 ====

from typing import Mapping, Sequence

import pandas as pd


LEGACY_HARD_GATES = {
    "execution_success_rate": {"operator": ">=", "threshold": 0.995},
    "retrieve_query_valid_rate": {"operator": ">=", "threshold": 0.99},
    "wrong_oos_count": {"operator": "==", "threshold": 0},
    "wrong_direct_count": {"operator": "==", "threshold": 0},
    "wrong_hard_filter_count": {"operator": "==", "threshold": 0},
    "clarify_precision": {"operator": ">=", "threshold": 0.95},
}

ANALYZER_LABELS = {
    "v15": "V1.5 추가질의 개선",
    "v31": "V3.1 교차업무만 재작성",
}
COMPARISON_BRANCH_INTERVAL_SECONDS = 3.0


def _comparison_plans_valid(route: str, plans: Sequence[Mapping[str, Any]]) -> bool:
    if route != "RETRIEVE":
        return len(plans) == 0
    if not plans:
        return False
    total = sum(float(plan.get("weight") or 0) for plan in plans)
    if not math.isclose(total, 1.0, abs_tol=1e-9):
        return False
    for plan in plans:
        if not str(plan.get("query") or "").strip():
            return False
        if float(plan.get("weight") or 0) <= 0:
            return False
        if str((plan.get("business_filter") or {}).get("mode") or "").upper() == "HARD":
            return False
    return True

# ==== cell 43 ====
import time
import uuid

from kdic_decomposition_quality_core import BASELINE
from kdic_hcx007_resumable_decomposition_core import (
    HCX_DECOMPOSITION_ENDPOINT,
    ResumableBaselineDecomposer,
    TransportPolicy,
    _condition_record,
    _normalize_baseline_record,
)
from kdic_lightweight_query_ablation_core import AblationConfig, analyze_common


v15_transport_policy = TransportPolicy(
    request_delay_seconds=0.0,
    max_transport_retries=4,
    base_backoff_seconds=5.0,
    max_backoff_seconds=120.0,
    jitter_seconds=1.0,
    consecutive_429_cooldown_threshold=1,
    cooldown_seconds=65.0,
    timeout_seconds=HCX_REQUEST_TIMEOUT,
)
v15_decomposition_config = AblationConfig(
    llm_model=HCX_DECOMPOSITION_MODEL,
    llm_endpoint=HCX_DECOMPOSITION_ENDPOINT,
    llm_timeout_seconds=HCX_REQUEST_TIMEOUT,
    llm_min_confidence=V15_MIN_CONFIDENCE,
    max_subqueries=V15_MAX_SUBQUERIES,
)
V15_DECOMPOSER = ResumableBaselineDecomposer(
    HCX_API_KEY,
    cache_path=V15_CACHE_PATH,
    seed_cache_paths=[],
    config=v15_decomposition_config,
    transport_policy=v15_transport_policy,
)


def _route_response(route: str, common: dict[str, Any]) -> str:
    if route == "DIRECT_RESPONSE":
        return "안녕하세요. 예금보험공사 관련 제도와 신청 절차에 관해 질문해 주세요."
    if route == "OUT_OF_SCOPE":
        return "이 챗봇은 예금자보호, 예금보험금, 고객 미수령금, 착오송금 반환지원, 채무조정, 은닉재산 신고 관련 질문에 답변합니다."
    missing = common.get("missing_information") or []
    detail = " / ".join(str(value) for value in missing if str(value).strip())
    if detail:
        return f"정확한 안내를 위해 정보가 더 필요합니다: {detail}"
    return "어떤 업무에 관한 질문인지 선택해 주세요: 예금자보호, 예금보험금, 고객 미수령금, 착오송금 반환지원, 채무조정, 은닉재산 신고."


def analyze_v15_chat_query(question: str, previous_turns: Any = None) -> dict[str, Any]:
    started = time.perf_counter()
    common = analyze_common(
        f"CHAT_{uuid.uuid4().hex[:12]}",
        question,
        previous_turns=previous_turns,
    )
    route = str(common["route"])
    businesses = list(dict.fromkeys((common.get("complexity") or {}).get("businesses") or []))
    cross_candidate = bool(
        route == "RETRIEVE"
        and common.get("complex_candidate")
        and len(businesses) >= 2
    )
    record = None
    decomposition_latency_ms = 0.0
    if cross_candidate:
        decomposition_started = time.perf_counter()
        first = _normalize_baseline_record(
            common["normalized_question"],
            businesses,
            V15_DECOMPOSER.decompose(common["normalized_question"], businesses),
        )
        record = _condition_record(BASELINE, first, first, retry_called=False)
        decomposition_latency_ms = (time.perf_counter() - decomposition_started) * 1000

    accepted = bool((record or {}).get("final_accepted"))
    subqueries = list((record or {}).get("final_subqueries") or []) if accepted else []
    if route != "RETRIEVE":
        plans = []
    elif subqueries:
        sub_weight = V15_SUBQUERY_TOTAL_WEIGHT / len(subqueries)
        plans = [{
            "query": common["original_question"],
            "weight": V15_ORIGINAL_WEIGHT,
            "source": "ORIGINAL_ANCHOR",
        }]
        plans.extend({
            "query": query,
            "weight": sub_weight,
            "source": "DECOMPOSED",
        } for query in subqueries)
    else:
        plans = [{
            "query": common["original_question"],
            "weight": 1.0,
            "source": "ORIGINAL",
        }]

    return {
        "route": route,
        "route_reasons": common.get("route_reasons") or [],
        "businesses": businesses,
        "complexity": (common.get("complexity") or {}).get("question_type", "NONE"),
        "cross_business_candidate": cross_candidate,
        "decomposition_called": cross_candidate,
        "decomposition_accepted": accepted,
        "decomposition_status": (record or {}).get("final_status", "NOT_CALLED"),
        "decomposition_issues": (record or {}).get("final_issues") or [],
        "decomposition_cache_hit": bool((record or {}).get("first_cache_hit")),
        "subqueries": subqueries,
        "fallback_to_original": bool(cross_candidate and not accepted),
        "plans": plans,
        "route_response": _route_response(route, common) if route != "RETRIEVE" else "",
        "routing_latency_ms": float(common.get("common_latency_ms") or 0.0),
        "rule_latency_ms": float(common.get("rule_latency_ms") or 0.0),
        "decomposition_latency_ms": decomposition_latency_ms,
        "analysis_wall_latency_ms": (time.perf_counter() - started) * 1000,
    }


def retrieval_table_markdown(results: list[dict[str, Any]]) -> str:
    lines = [
        "### 검색 결과",
        "",
        "|순위|청크 ID|Dense 순위|BM25 순위|Min-Max 최고점|질의결합점수|제목 / 소제목|",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for result in results:
        chunk = result["chunk"]
        dense_rank = result.get("dense_rank") or "-"
        bm25_rank = result.get("bm25_rank") or "-"
        title = " / ".join(
            part for part in [
                str(chunk.get("title") or "").replace("|", "\\|"),
                str(chunk.get("section_title") or "").replace("|", "\\|"),
            ] if part
        )
        lines.append(
            f"|{result['rank']}|{result['chunk_id']}|{dense_rank}|{bm25_rank}|"
            f"{float(result.get('minmax_score') or 0):.6f}|"
            f"{float(result.get('query_fusion_score') or 0):.6f}|{title}|"
        )
    return "\n".join(lines)


def analysis_markdown(analysis: dict[str, Any]) -> str:
    plans = analysis.get("plans") or []
    plan_text = "<br>".join(
        f"{index}. {plan['source']} · {plan['weight']:.3f} · {plan['query']}"
        for index, plan in enumerate(plans, start=1)
    ) or "검색 계획 없음"
    return (
        "### V1.5 질의분석\n\n"
        "|항목|결과|\n|---|---|\n"
        f"|최종 경로|{analysis['route']}|\n"
        f"|탐지 업무|{', '.join(analysis['businesses']) or '-'}|\n"
        f"|복합 후보|{analysis['complexity']}|\n"
        f"|교차업무 분해 호출|{analysis['decomposition_called']}|\n"
        f"|분해 승인|{analysis['decomposition_accepted']}|\n"
        f"|원문 fallback|{analysis['fallback_to_original']}|\n"
        f"|검색 계획|{plan_text}|"
    )


def latency_markdown(latency: dict[str, float]) -> str:
    return (
        "### 단계별 지연시간\n\n"
        "|단계|지연시간|\n|---|---:|\n"
        + "\n".join(f"|{key}|{value:,.1f}ms|" for key, value in latency.items())
    )

# ==== cell 44 ====


import copy
import hashlib
import html
import time
from typing import Mapping, Sequence

import kdic_query_analyzer_v31 as v31
from kdic_context_policy_v2 import new_context_state, resolve_context_v2
from kdic_v31_v15_cross_rewrite import CrossRewritePolicy, KDICV31V15CrossRewriteAnalyzer


# ---------- V3.1: 단일·동일업무는 원문, 교차업무만 Need 재작성 ----------
V31_CONFIG = v31.PipelineConfig(
    model="HCX-007",
    max_completion_tokens=700,
    temperature=0.1,
    top_p=0.8,
    request_interval_seconds=1.05,
)
V31_CLIENT = v31.HCX007AtomicNeedClientV3(HCX_API_KEY, V31_CONFIG)
V31_BASE_ANALYZER = v31.KDICLightweightRAGAnalyzerV31(V31_CLIENT, V31_CONFIG)
V31_CROSS_POLICY = CrossRewritePolicy(
    original_weight=0.40,
    rewritten_total_weight=0.60,
    max_rewritten_queries=4,
    minimum_business_confidence=0.70,
    minimum_token_overlap=0.25,
    allow_soft_business_hint=True,
)
V31_CROSS_ANALYZER = KDICV31V15CrossRewriteAnalyzer(
    V31_BASE_ANALYZER,
    v31,
    V31_CROSS_POLICY,
)
V31_ANALYSIS_CACHE: dict[str, dict[str, Any]] = {}


# ---------- 두 분석기에 공통 적용하는 개선 문맥 게이트 ----------
CONTEXT_CLASSIFIER_SYSTEM_PROMPT = """
당신은 대화 문맥 적용 여부만 판정하는 구조화 분류기입니다.
질의를 재작성하거나 사용자 질문에 답하지 마세요.
현재 질문이 독립적으로 완결되면 이전 대화 상태를 사용하지 마세요.
현재 질문에 명시된 업무는 이전 업무보다 우선합니다.
확신할 수 없으면 AMBIGUOUS로 판정하세요.
JSON 객체 하나만 출력하세요.
""".strip()

_CONTEXT_CLASSIFIER_CACHE: dict[str, dict[str, Any]] = {}


def _extract_context_json(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(text or "").strip(), flags=re.I)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        if start < 0:
            raise ValueError("문맥 판단 JSON 객체가 없습니다.")
        value, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    if not isinstance(value, dict):
        raise TypeError("문맥 판단 결과의 최상위 값은 객체여야 합니다.")
    return value


def classify_ambiguous_context(question: str, state: Mapping[str, Any]) -> dict[str, Any]:
    cache_payload = {
        "question": question,
        "active_businesses": state.get("active_businesses") or [],
        "excluded_businesses": state.get("excluded_businesses") or [],
        "actor_role": state.get("actor_role"),
        "pending_clarification": state.get("pending_clarification"),
        "last_resolved_question": state.get("last_resolved_question"),
        "turns": (state.get("turns") or [])[-4:],
    }
    cache_key = hashlib.sha256(
        json.dumps(cache_payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    cached = _CONTEXT_CLASSIFIER_CACHE.get(cache_key)
    if cached is not None:
        return {**copy.deepcopy(cached), "_cache_hit": True, "_latency_ms": 0.0}

    prompt = f"""
[현재 질문]
{question}

[현재 대화 상태]
{json.dumps(cache_payload, ensure_ascii=False, indent=2)}

[출력 JSON]
{{
  "dialogue_act": "NEW_TOPIC | FOLLOW_UP | CORRECTION | EXCLUSION | AMBIGUOUS",
  "current_question_complete": true,
  "context_required": false,
  "selected_businesses": [],
  "excluded_businesses": [],
  "actor_role": "",
  "missing_slots": [],
  "confidence": 0.0,
  "reason_code": ""
}}
""".strip()
    started = time.perf_counter()
    response = ANSWER_HCX_CLIENT.chat.completions.create(
        model="HCX-007",
        messages=[
            {"role": "system", "content": CONTEXT_CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=500,
    )
    value = _extract_context_json(response.choices[0].message.content)
    value["_cache_hit"] = False
    value["_latency_ms"] = (time.perf_counter() - started) * 1000
    _CONTEXT_CLASSIFIER_CACHE[cache_key] = copy.deepcopy(value)
    return value


BUSINESS_TO_CONTEXT = {
    "예금자보호제도": "예금자보호",
    "예금자보호": "예금자보호",
    "예금보험금 안내": "예금보험금",
    "예금보험금": "예금보험금",
    "고객 미수령금 신청": "고객 미수령금",
    "고객 미수령금": "고객 미수령금",
    "착오송금 반환 신청": "착오송금 반환지원",
    "착오송금 반환지원": "착오송금 반환지원",
    "채무조정 안내": "채무조정",
    "채무조정": "채무조정",
    "은닉재산 신고": "은닉재산 신고",
}


def _context_businesses(values: Sequence[Any]) -> list[str]:
    output = []
    for value in values:
        mapped = BUSINESS_TO_CONTEXT.get(str(value).strip(), str(value).strip())
        if mapped and mapped not in output:
            output.append(mapped)
    return output


def new_analyzer_state() -> dict[str, Any]:
    return new_context_state()


def _route_only_analysis(
    analyzer_key: str,
    question: str,
    resolution: Mapping[str, Any],
    started: float,
) -> dict[str, Any]:
    route = str(resolution.get("route") or "CLARIFY")
    return {
        "analyzer": analyzer_key,
        "route": route,
        "original_question": question,
        "resolved_question": "",
        "businesses": list(resolution.get("active_businesses") or []),
        "query_type": "NO_RETRIEVAL",
        "plans": [],
        "context_resolution": dict(resolution),
        "context_used": bool(resolution.get("context_used")),
        "context_reason": resolution.get("reason"),
        "route_response": resolution.get("clarification_message") or resolution.get("direct_response") or "추가 정보가 필요합니다.",
        "decomposition_or_rewrite_called": False,
        "decomposition_or_rewrite_accepted": False,
        "fallback_to_original": False,
        "issues": [],
        "query_plan_valid": route != "RETRIEVE",
        "hard_filter_count": 0,
        "context_latency_ms": float(resolution.get("latency_ms") or 0),
        "core_analysis_latency_ms": 0.0,
        "analysis_wall_latency_ms": (time.perf_counter() - started) * 1000,
        "raw_analysis": {},
    }


def analyze_v15_improved(question: str, state: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    resolution = resolve_context_v2(
        question,
        state=state,
        llm_classifier=classify_ambiguous_context,
    )
    if resolution["route"] != "CONTINUE":
        return _route_only_analysis("V1.5_IMPROVED", question, resolution, started)

    resolved = str(resolution["resolved_question"])
    core_started = time.perf_counter()
    base = analyze_v15_chat_query(resolved, previous_turns=[])
    core_latency_ms = (time.perf_counter() - core_started) * 1000
    businesses = list(base.get("businesses") or [])
    if base.get("cross_business_candidate"):
        query_type = "CROSS_BUSINESS"
    elif str(base.get("complexity") or "") == "MULTI":
        query_type = "SAME_BUSINESS_MULTI"
    else:
        query_type = "SINGLE"
    plans = [
        {
            "query": str(plan.get("query") or "").strip(),
            "weight": float(plan.get("weight") or 0),
            "source": str(plan.get("source") or "V15"),
            "business_filter": {"mode": "NONE"},
        }
        for plan in base.get("plans") or []
    ]
    if base.get("route") == "RETRIEVE" and businesses:
        state["active_businesses"] = _context_businesses(businesses)
    return {
        "analyzer": "V1.5_IMPROVED",
        "route": str(base.get("route") or ""),
        "original_question": question,
        "resolved_question": resolved,
        "businesses": businesses,
        "query_type": query_type,
        "plans": plans,
        "context_resolution": dict(resolution),
        "context_used": bool(resolution.get("context_used")),
        "context_reason": resolution.get("reason"),
        "route_response": str(base.get("route_response") or ""),
        "decomposition_or_rewrite_called": bool(base.get("decomposition_called")),
        "decomposition_or_rewrite_accepted": bool(base.get("decomposition_accepted")),
        "fallback_to_original": bool(base.get("fallback_to_original")),
        "issues": list(base.get("decomposition_issues") or []),
        "query_plan_valid": _comparison_plans_valid(str(base.get("route") or ""), plans),
        "hard_filter_count": 0,
        "context_latency_ms": float(resolution.get("latency_ms") or 0),
        "core_analysis_latency_ms": core_latency_ms,
        "analysis_wall_latency_ms": (time.perf_counter() - started) * 1000,
        "analysis_cache_hit": bool(base.get("decomposition_cache_hit")),
        "raw_analysis": base,
    }


def _v31_cache_key(question: str) -> str:
    return hashlib.sha256(question.strip().encode("utf-8")).hexdigest()


def analyze_v31_cross_only(question: str, state: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    resolution = resolve_context_v2(
        question,
        state=state,
        llm_classifier=classify_ambiguous_context,
    )
    if resolution["route"] != "CONTINUE":
        return _route_only_analysis("V3.1_CROSS_ONLY", question, resolution, started)

    resolved = str(resolution["resolved_question"])
    cache_key = _v31_cache_key(resolved)
    cached = V31_ANALYSIS_CACHE.get(cache_key)
    core_started = time.perf_counter()
    if cached is None:
        base = V31_CROSS_ANALYZER.run(resolved, conversation_state=None)
        V31_ANALYSIS_CACHE[cache_key] = copy.deepcopy(base)
        cache_hit = False
    else:
        base = copy.deepcopy(cached)
        cache_hit = True
    core_latency_ms = (time.perf_counter() - core_started) * 1000

    raw_plans = list(base.get("search_plans") or [])
    plans = [
        {
            "query": str(plan.get("semantic_query") or "").strip(),
            "weight": float(plan.get("query_weight") or 0),
            "source": str(plan.get("query_source") or "V31"),
            "business_filter": dict(plan.get("business_filter") or {"mode": "NONE"}),
        }
        for plan in raw_plans
    ]
    cross = dict(base.get("cross_business") or {})
    needs = list((base.get("v31_analysis") or {}).get("needs") or [])
    businesses = list(cross.get("businesses") or [])
    if not businesses:
        businesses = list(dict.fromkeys(
            str(need.get("business_function") or "").strip()
            for need in needs if str(need.get("business_function") or "").strip()
        ))
    if base.get("route") == "RETRIEVE" and businesses:
        state["active_businesses"] = _context_businesses(businesses)
    return {
        "analyzer": "V3.1_CROSS_ONLY",
        "route": str(base.get("route") or ""),
        "original_question": question,
        "resolved_question": resolved,
        "businesses": businesses,
        "query_type": str(base.get("query_type") or "SINGLE"),
        "plans": plans,
        "context_resolution": dict(resolution),
        "context_used": bool(resolution.get("context_used")),
        "context_reason": resolution.get("reason"),
        "route_response": "",
        "decomposition_or_rewrite_called": bool(base.get("rewrite_called")),
        "decomposition_or_rewrite_accepted": bool(base.get("rewrite_accepted")),
        "fallback_to_original": bool(base.get("fallback_to_original")),
        "issues": list((base.get("rewrite_validation") or {}).get("issues") or []),
        "query_plan_valid": bool(base.get("query_plan_valid")) and _comparison_plans_valid(str(base.get("route") or ""), plans),
        "hard_filter_count": int(base.get("hard_filter_count") or 0),
        "context_latency_ms": float(resolution.get("latency_ms") or 0),
        "core_analysis_latency_ms": core_latency_ms,
        "analysis_wall_latency_ms": (time.perf_counter() - started) * 1000,
        "analysis_cache_hit": cache_hit,
        "raw_analysis": base,
    }


print({
    "analyzers": ["V1.5_IMPROVED", "V3.1_CROSS_ONLY"],
    "shared_context_policy": "CONTEXT_POLICY_V2",
    "single_and_same_business_policy": "ORIGINAL_1.0",
    "cross_business_policy": "ORIGINAL_0.4_REWRITTEN_0.6",
})

# ==== cell 46 ====
def new_comparison_state() -> dict[str, Any]:
    return {
        "v15": new_analyzer_state(),
        "v31": new_analyzer_state(),
        "results": {"v15": [], "v31": []},
        "compare_run_count": 0,
        "busy": False,
    }


ANALYZER_FUNCTIONS = {
    "v15": analyze_v15_improved,
    "v31": analyze_v31_cross_only,
}


def _route_message(analysis: Mapping[str, Any]) -> str:
    if analysis.get("route_response"):
        return str(analysis["route_response"])
    route = str(analysis.get("route") or "")
    if route == "DIRECT_RESPONSE":
        return "안녕하세요. 예금보험공사 관련 제도와 신청 절차에 관해 질문해 주세요."
    if route == "OUT_OF_SCOPE":
        return "예금보험공사의 예금자보호·예금보험금·미수령금·착오송금 반환지원·채무조정·은닉재산 신고 범위에서 질문해 주세요."
    if route == "CLARIFY":
        return "정확한 안내를 위해 어떤 업무와 대상에 관한 질문인지 조금 더 알려주세요."
    return "답변할 수 없는 경로입니다."


def _query_latency_rows(per_query: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in per_query:
        trace = dict(row.get("latency_breakdown_ms") or {})
        rows.append({
            "plan_index": row.get("plan_index"),
            "source": row.get("source"),
            "weight": float(row.get("weight") or 0),
            "query": row.get("query"),
            "embedding_latency_ms": float(trace.get("embedding_latency_ms") or 0),
            "dense_compute_latency_ms": float(trace.get("dense_compute_latency_ms") or 0),
            "bm25_latency_ms": float(trace.get("bm25_latency_ms") or 0),
            "minmax_latency_ms": float(trace.get("minmax_latency_ms") or 0),
            "query_total_latency_ms": float(trace.get("query_total_latency_ms") or 0),
        })
    return rows


def run_fixed_pipeline(
    analyzer_key: str,
    question: str,
    *,
    state: dict[str, Any],
) -> dict[str, Any]:
    global _LAST_RERANK_TRACE, _LAST_PARENT_CHILD_TRACE
    if analyzer_key not in ANALYZER_FUNCTIONS:
        raise KeyError(f"알 수 없는 분석기: {analyzer_key}")
    total_started = time.perf_counter()
    _LAST_RERANK_TRACE = {}
    _LAST_PARENT_CHILD_TRACE = {}

    analysis = ANALYZER_FUNCTIONS[analyzer_key](question, state)
    if analysis["route"] != "RETRIEVE":
        result = {
            "analyzer_key": analyzer_key,
            "analyzer_label": ANALYZER_LABELS[analyzer_key],
            "question": question,
            "resolved_question": analysis.get("resolved_question") or "",
            "route": analysis["route"],
            "analysis": analysis,
            "display_answer": _route_message(analysis),
            "basic_answer": _route_message(analysis),
            "basic_answer_payload": None,
            "evidence_explanation_payload": None,
            "search_results": [],
            "evidence_pack": None,
            "sources": [],
            "latency_ms": {
                "문맥 처리": float(analysis.get("context_latency_ms") or 0),
                "질의분석": float(analysis.get("core_analysis_latency_ms") or 0),
                "전체": (time.perf_counter() - total_started) * 1000,
            },
            "success": True,
        }
        return result

    plans = list(analysis.get("plans") or [])
    if not analysis.get("query_plan_valid") or not _comparison_plans_valid("RETRIEVE", plans):
        raise RuntimeError("RETRIEVE 검색 계획이 유효하지 않습니다.")

    search_started = time.perf_counter()
    search_results, per_query = fuse_query_results(plans)
    before_threshold_count = len(search_results)
    search_results = filter_search_results_by_relevance(search_results)
    child_search_ms = (time.perf_counter() - search_started) * 1000
    reranker_trace = dict(_LAST_RERANK_TRACE)
    reranker_trace.update({
        "min_relevance_score": float(MIN_RELEVANCE_SCORE),
        "before_threshold_count": before_threshold_count,
        "after_threshold_count": len(search_results),
    })

    if not search_results:
        message = (
            "공식 근거에서 충분히 관련된 내용을 찾지 못했습니다. "
            "질문에 업무명이나 신청 상황을 조금 더 구체적으로 적어 주세요."
        )
        return {
            "analyzer_key": analyzer_key,
            "analyzer_label": ANALYZER_LABELS[analyzer_key],
            "question": question,
            "resolved_question": analysis.get("resolved_question") or question,
            "route": "DIRECT",
            "analysis": analysis,
            "display_answer": message,
            "basic_answer": message,
            "basic_answer_payload": None,
            "evidence_explanation_payload": None,
            "search_plans": plans,
            "search_results": [],
            "per_query_search": per_query,
            "reranker": reranker_trace,
            "evidence_pack": None,
            "sources": [],
            "latency_ms": {
                "문맥 처리": float(analysis.get("context_latency_ms") or 0),
                "질의분석": float(analysis.get("core_analysis_latency_ms") or 0),
                "검색": child_search_ms,
                "전체": (time.perf_counter() - total_started) * 1000,
            },
            "success": True,
        }

    parent_started = time.perf_counter()
    search_results = expand_parent_context(search_results)
    parent_ms = (time.perf_counter() - parent_started) * 1000
    parent_trace = dict(_LAST_PARENT_CHILD_TRACE)

    answer_question = str(analysis.get("resolved_question") or question)
    evidence_started = time.perf_counter()
    evidence_pack = build_parent_basic_evidence_pack(answer_question, search_results)
    evidence_ms = (time.perf_counter() - evidence_started) * 1000

    answer_started = time.perf_counter()
    basic_payload = generate_basic_answer_b_v2(
        client=ANSWER_HCX_CLIENT,
        model=HCX_CHAT_MODEL,
        question=answer_question,
        evidence_pack=evidence_pack,
    )
    answer_ms = (time.perf_counter() - answer_started) * 1000
    display_answer = user_visible_answer(basic_payload["answer"])
    sources = build_used_sources(evidence_pack, basic_payload)
    per_query_latency = _query_latency_rows(per_query)
    search_wall_ms = child_search_ms + parent_ms
    result = {
        "analyzer_key": analyzer_key,
        "analyzer_label": ANALYZER_LABELS[analyzer_key],
        "question": question,
        "resolved_question": answer_question,
        "route": analysis["route"],
        "analysis": analysis,
        "display_answer": display_answer,
        "basic_answer": basic_payload["answer"],
        "basic_answer_payload": basic_payload,
        "evidence_explanation_payload": None,
        "search_plans": plans,
        "per_query_search": per_query,
        "per_query_latency": per_query_latency,
        "search_results": search_results,
        "reranker": reranker_trace,
        "parent_child": parent_trace,
        "evidence_pack": evidence_pack,
        "evidence_pack_sha256": evidence_pack_sha256(evidence_pack),
        "sources": sources,
        "detail_sources": [],
        "answer_api_trace": dict(_LAST_ANSWER_API_TRACE),
        "latency_ms": {
            "문맥 처리": float(analysis.get("context_latency_ms") or 0),
            "질의분석": float(analysis.get("core_analysis_latency_ms") or 0),
            "검색": search_wall_ms,
            "질문 임베딩": sum(row["embedding_latency_ms"] for row in per_query_latency),
            "Dense 계산": sum(row["dense_compute_latency_ms"] for row in per_query_latency),
            "BM25": sum(row["bm25_latency_ms"] for row in per_query_latency),
            "BAAI Reranker": float(reranker_trace.get("latency_ms") or 0),
            "Parent-Child8192": parent_ms,
            "Evidence Pack": evidence_ms,
            "구조화 기본답변": answer_ms,
            "전체": (time.perf_counter() - total_started) * 1000,
        },
        "success": True,
    }
    state.setdefault("turns", []).extend([
        {"role": "user", "content": question},
        {"role": "assistant", "content": display_answer},
    ])
    return result


def generate_answer_basis(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("route") != "RETRIEVE" or not result.get("evidence_pack"):
        raise ValueError("답변 근거를 생성할 RETRIEVE 결과가 없습니다.")
    if result.get("evidence_explanation_payload"):
        return result["evidence_explanation_payload"]
    started = time.perf_counter()
    payload = generate_evidence_explanation_b_v2(
        client=ANSWER_HCX_CLIENT,
        model=HCX_CHAT_MODEL,
        question=result["resolved_question"],
        evidence_pack=result["evidence_pack"],
        basic_answer=result["basic_answer_payload"],
    )
    latency_ms = (time.perf_counter() - started) * 1000
    result["evidence_explanation_payload"] = payload
    result["detail_sources"] = build_used_sources(result["evidence_pack"], payload)
    result["latency_ms"]["답변 근거 설명"] = latency_ms
    result["latency_ms"]["기본답변+근거 누적"] = float(result["latency_ms"]["전체"]) + latency_ms
    return payload


def analysis_markdown_compare(result: Mapping[str, Any]) -> str:
    analysis = dict(result.get("analysis") or {})
    plans = analysis.get("plans") or []
    plan_text = "<br>".join(
        f"{index}. {plan.get('source')} · {float(plan.get('weight') or 0):.3f} · {str(plan.get('query') or '').replace('|', chr(92) + '|')}"
        for index, plan in enumerate(plans, start=1)
    ) or "검색 계획 없음"
    businesses = ", ".join(analysis.get("businesses") or []) or "-"
    issues = ", ".join(analysis.get("issues") or []) or "-"
    return (
        "### 질의분석 결과\n\n|항목|값|\n|---|---|\n"
        f"|분석기|{result.get('analyzer_label')}|\n"
        f"|최종 경로|{analysis.get('route')}|\n"
        f"|검색용 독립질의|{analysis.get('resolved_question') or '-'}|\n"
        f"|문맥 판단|{analysis.get('context_reason') or '-'}|\n"
        f"|문맥 사용|{bool(analysis.get('context_used'))}|\n"
        f"|탐지 업무|{businesses}|\n"
        f"|질문 유형|{analysis.get('query_type')}|\n"
        f"|분해·재작성 호출|{bool(analysis.get('decomposition_or_rewrite_called'))}|\n"
        f"|분해·재작성 승인|{bool(analysis.get('decomposition_or_rewrite_accepted'))}|\n"
        f"|원문 fallback|{bool(analysis.get('fallback_to_original'))}|\n"
        f"|분석 캐시 적중|{bool(analysis.get('analysis_cache_hit'))}|\n"
        f"|검증 이슈|{issues}|\n"
        f"|검색 계획|{plan_text}|"
    )


def latency_markdown_compare(result: Mapping[str, Any]) -> str:
    lines = ["### 단계별 레이턴시", "", "|단계|지연시간|", "|---|---:|"]
    for key, value in (result.get("latency_ms") or {}).items():
        lines.append(f"|{key}|{float(value):,.1f}ms|")
    return "\n".join(lines)


def retrieval_markdown_compare(result: Mapping[str, Any]) -> str:
    rows = result.get("search_results") or []
    if not rows:
        return "### 검색 결과\n\n검색을 실행하지 않았습니다."
    lines = [
        "### 공통 BAAI Reranker + Parent-Child8192 검색 결과", "",
        "|순위|Child|Parent|Reranker|제목 / 섹션|",
        "|---:|---|---|---:|---|",
    ]
    for row in rows:
        chunk = row["chunk"]
        title = " / ".join(str(value).replace("|", "\\|") for value in (chunk.get("title"), chunk.get("section_title")) if value)
        lines.append(
            f"|{row.get('rank')}|{row.get('chunk_id')}|{row.get('parent_doc_id', '-')}|"
            f"{float(row.get('reranker_score') or 0):.6f}|{title}|"
        )
    return "\n".join(lines)


def render_result(result: dict[str, Any], *, show_pack: bool = False) -> None:
    display(Markdown(f"## {result['analyzer_label']}\n\n### 기본답변\n\n{result['display_answer']}"))
    source_text = sources_to_markdown_user(result.get("sources") or [])
    if source_text:
        display(Markdown(source_text))
    display(Markdown(analysis_markdown_compare(result)))
    display(Markdown(retrieval_markdown_compare(result)))
    if show_pack and result.get("evidence_pack"):
        display(JSON(result["evidence_pack"], expanded=False))
    display(Markdown(latency_markdown_compare(result)))


def comparison_summary_frame(results: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for key in ("v15", "v31"):
        result = results.get(key)
        if not result:
            continue
        analysis = dict(result.get("analysis") or {})
        rows.append({
            "질의분석기": result.get("analyzer_label"),
            "최종 경로": result.get("route"),
            "문맥 판단": analysis.get("context_reason"),
            "질문 유형": analysis.get("query_type"),
            "분해·재작성 승인": bool(analysis.get("decomposition_or_rewrite_accepted")),
            "원문 fallback": bool(analysis.get("fallback_to_original")),
            "검색계획 수": len(analysis.get("plans") or []),
            "Top-5": ", ".join(str(row.get("chunk_id")) for row in result.get("search_results") or []),
            "질의분석(ms)": float((result.get("latency_ms") or {}).get("질의분석") or 0),
            "검색(ms)": float((result.get("latency_ms") or {}).get("검색") or 0),
            "답변(ms)": float((result.get("latency_ms") or {}).get("구조화 기본답변") or 0),
            "전체(ms)": float((result.get("latency_ms") or {}).get("전체") or 0),
        })
    return pd.DataFrame(rows)


print({
    "retrieval": "HYBRID_7_3_MINMAX",
    "reranker": RERANKER_MODEL_NAME,
    "parent_child": PARENT_CHILD_ENABLED,
    "parent_context_max_chars": PARENT_CONTEXT_MAX_CHARS,
    "answer": "B_STRUCTURED_BASIC_AND_DOCUMENT_BASIS",
    "legacy_hard_gates": LEGACY_HARD_GATES,
})

# ==== cell 48 ====

assert DENSE_WEIGHT == 0.7
assert BM25_WEIGHT == 0.3
assert CANDIDATE_DEPTH == 20
assert FINAL_TOP_K == 5
assert RERANKER_MODEL_NAME == "BAAI/bge-reranker-v2-m3"
assert PARENT_CHILD_ENABLED is True
assert PARENT_CONTEXT_MAX_CHARS == 8192
assert math.isclose(V15_ORIGINAL_WEIGHT, 0.40)
assert math.isclose(V15_SUBQUERY_TOTAL_WEIGHT, 0.60)
assert math.isclose(V31_CROSS_POLICY.original_weight, 0.40)
assert math.isclose(V31_CROSS_POLICY.rewritten_total_weight, 0.60)

# 네트워크 없이 확인 가능한 검색 계획 안전성 검사
assert _comparison_plans_valid("RETRIEVE", [{
    "query": "예금자보호 한도는 얼마인가요?",
    "weight": 1.0,
    "source": "ORIGINAL",
    "business_filter": {"mode": "NONE"},
}])
assert not _comparison_plans_valid("RETRIEVE", [{
    "query": "예금자보호 한도는 얼마인가요?",
    "weight": 1.0,
    "source": "INVALID_HARD",
    "business_filter": {"mode": "HARD"},
}])

print("공통 검색·답변 조건과 질의계획 안전성 검사 통과")

# ==== cell 50 ====
def elasticsearch_diagnostics() -> dict[str, Any]:
    info = ES.info()
    nodes = ES.nodes.info(metric="plugins")
    plugins = sorted({
        str(plugin.get("name") or "")
        for node in nodes["nodes"].values()
        for plugin in node.get("plugins", [])
    })
    index_exists = bool(ES.indices.exists(index=ES_INDEX_NAME))
    document_count = int(ES.count(index=ES_INDEX_NAME)["count"]) if index_exists else None
    tokens = []
    if index_exists:
        tokens = [
            token["token"]
            for token in ES.indices.analyze(
                index=ES_INDEX_NAME,
                analyzer=ES_ANALYZER_NAME,
                text="착오송금 반환지원",
            )["tokens"]
        ]
    return {
        "connected": True,
        "url": ES_URL,
        "version": info["version"]["number"],
        "cluster_name": info["cluster_name"],
        "analysis_nori_installed": "analysis-nori" in plugins,
        "plugins": plugins,
        "index_name": ES_INDEX_NAME,
        "index_exists": index_exists,
        "document_count": document_count,
        "expected_document_count": len(CHUNKS),
        "nori_none_tokens": tokens,
    }


display(JSON(elasticsearch_diagnostics(), expanded=True))

# ==== cell 52 ====

import copy
import re
import time
from typing import Any, Callable, Mapping, Sequence

from kdic_context_policy_v2 import (
    AMBIGUOUS_REFERENCE_PATTERN,
    BUSINESS_PATTERNS,
    CANCEL_PATTERN,
    CORRECTION_PATTERN,
    EXCLUSION_PATTERN,
    _clean,
    _clarify,
    _selected_pending,
    detect_businesses,
)

FOLLOWUP_INTENT_RULES_V21: dict[str, tuple[str, ...]] = {
    "APPLICATION": ("신청", "접수", "신청하려", "접수하려"),
    "DOCUMENTS": ("서류", "필요서류", "필요 서류", "구비서류", "구비 서류", "준비물", "뭘 준비", "무엇을 준비"),
    "ELIGIBILITY": ("자격", "대상", "해당", "신청 가능", "가능한 사람"),
    "PROCEDURE": ("절차", "방법", "순서", "과정", "어떻게"),
    "TIME": ("기간", "기한", "언제", "얼마나 걸", "며칠", "몇 일"),
    "COST": ("비용", "수수료", "돈이 드", "얼마가 드"),
    "LOOKUP": ("조회", "확인", "진행상태", "진행 상태"),
    "LIMIT": ("금액", "한도", "얼마까지"),
    "EXCEPTION": ("예외", "제외", "안 되는", "불가능"),
    "CHANGE_CANCEL": ("취소", "철회", "변경", "수정"),
    "ACTOR": ("본인", "대리인", "송금인", "수취인", "상속인"),
    "REASON": ("왜", "이유"),
}

EXPLICIT_OOS_CONTEXT_BLOCK_V21 = re.compile(
    r"(?:비트코인|가상자산|코스피|주식\s*(?:투자|매수|매도|포트폴리오)|"
    r"날씨|기온|미세먼지|환율|환전|주택담보대출|전세대출|신용카드|상속세|세금\s*환급)",
    re.I,
)


def detect_followup_intents_v21(question: str) -> list[str]:
    text = _clean(question).lower()
    return [
        intent
        for intent, terms in FOLLOWUP_INTENT_RULES_V21.items()
        if any(term.lower() in text for term in terms)
    ]


def _explicit_oos_before_context_v21(question: str) -> bool:
    if EXPLICIT_OOS_CONTEXT_BLOCK_V21.search(question):
        return True
    try:
        return bool(light_router.OOS_PATTERN.search(question))
    except Exception:
        return False


def _context_resolution_payload_v21(
    *,
    original: str,
    resolved: str,
    state: dict[str, Any],
    intents: Sequence[str],
    started: float,
) -> dict[str, Any]:
    state["pending_clarification"] = None
    state["last_resolved_question"] = resolved
    return {
        "route": "CONTINUE",
        "dialogue_act": "FOLLOW_UP",
        "original_question": original,
        "resolved_question": resolved,
        "current_question_complete": False,
        "context_used": True,
        "reason": "INTENT_BASED_UNIQUE_ACTIVE_BUSINESS",
        "clarification_message": "",
        "active_businesses": list(state.get("active_businesses") or []),
        "excluded_businesses": list(state.get("excluded_businesses") or []),
        "actor_role": state.get("actor_role"),
        "missing_slots": [],
        "followup_intents": list(intents),
        "pending_clarification": None,
        "llm_judgment": {"called": False, "reason": "RULE_INTENT_FOLLOWUP"},
        "latency_ms": (time.perf_counter() - started) * 1000,
    }


if "_ORIGINAL_RESOLVE_CONTEXT_V2" not in globals():
    _ORIGINAL_RESOLVE_CONTEXT_V2 = resolve_context_v2


def resolve_context_v21(
    question: str,
    *,
    state: dict[str, Any],
    llm_classifier: Callable[[str, Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    original = _clean(question)
    if not original:
        raise ValueError("질문이 비어 있습니다.")
    for key, default in new_context_state().items():
        state.setdefault(key, copy.deepcopy(default))

    explicit_businesses = detect_businesses(original)
    active_businesses = list(state.get("active_businesses") or [])
    intents = detect_followup_intents_v21(original)
    pending = state.get("pending_clarification") or {}
    pending_selection = _selected_pending(original, pending) if pending else None

    must_use_existing_policy = bool(
        CANCEL_PATTERN.fullmatch(original)
        or pending_selection
        or explicit_businesses
        or EXCLUSION_PATTERN.search(original)
        or CORRECTION_PATTERN.search(original)
        or AMBIGUOUS_REFERENCE_PATTERN.search(original)
        or _explicit_oos_before_context_v21(original)
    )
    if must_use_existing_policy or not intents:
        return _ORIGINAL_RESOLVE_CONTEXT_V2(
            original,
            state=state,
            llm_classifier=llm_classifier,
        )

    if len(active_businesses) == 1:
        business = active_businesses[0]
        if business == "착오송금 반환지원" and "TIME" in intents:
            return _clarify(
                question=original,
                state=state,
                reason="MISTAKEN_TRANSFER_TIME_SCOPE_AMBIGUOUS",
                message=(
                    "착오송금의 어느 기간을 묻는지 확인이 필요합니다. "
                    "송금인의 반환지원 처리기간과 수취인의 자진반환 관련 기간 중 선택해 주세요."
                ),
                options=["송금인", "수취인"],
                missing_slots=["actor_role", "process_stage"],
                started=started,
            )
        resolved = f"{business} 관련 {original}"
        return _context_resolution_payload_v21(
            original=original,
            resolved=resolved,
            state=state,
            intents=intents,
            started=started,
        )

    if len(active_businesses) > 1:
        return _clarify(
            question=original,
            state=state,
            reason="INTENT_FOLLOWUP_MULTIPLE_ACTIVE_BUSINESSES",
            message="어느 업무에 관한 후속 질문인지 선택해 주세요.",
            options=active_businesses,
            missing_slots=["business_function"],
            started=started,
        )

    return _clarify(
        question=original,
        state=state,
        reason="INTENT_FOLLOWUP_WITHOUT_ACTIVE_BUSINESS",
        message="어떤 업무에 관한 질문인지 알려주세요.",
        options=list(BUSINESS_PATTERNS),
        missing_slots=["business_function"],
        started=started,
    )


resolve_context_v2 = resolve_context_v21


if "_ORIGINAL_ANALYZE_V15_IMPROVED" not in globals():
    _ORIGINAL_ANALYZE_V15_IMPROVED = analyze_v15_improved


def analyze_v15_improved_v21(question: str, state: dict[str, Any]) -> dict[str, Any]:
    result = _ORIGINAL_ANALYZE_V15_IMPROVED(question, state)
    if result.get("route") != "RETRIEVE" or not result.get("context_used"):
        result["context_business_preserved"] = True
        return result

    expected = set(_context_businesses((result.get("context_resolution") or {}).get("active_businesses") or []))
    detected = set(_context_businesses(result.get("businesses") or []))
    preserved = not expected or bool(expected & detected)
    result["context_business_preserved"] = preserved
    if preserved:
        return result

    options = sorted(expected) or list(BUSINESS_PATTERNS)
    state["pending_clarification"] = {
        "original_question": question,
        "reason": "CONTEXT_BUSINESS_NOT_PRESERVED",
        "options": options,
        "missing_slots": ["business_function"],
    }
    result.update({
        "route": "CLARIFY",
        "query_type": "NO_RETRIEVAL",
        "plans": [],
        "query_plan_valid": True,
        "route_response": "검색 질의에 이전 업무가 안전하게 반영되지 않았습니다. 어떤 업무인지 다시 선택해 주세요.",
        "issues": list(result.get("issues") or []) + ["CONTEXT_BUSINESS_NOT_PRESERVED"],
    })
    return result


analyze_v15_improved = analyze_v15_improved_v21
ANALYZER_FUNCTIONS["v15"] = analyze_v15_improved_v21
ANALYZER_LABELS["v15"] = "V1.5 멀티턴 의도결합 개선"

print({
    "analyzer": ANALYZER_LABELS["v15"],
    "followup_policy": "UNIQUE_ACTIVE_BUSINESS_PLUS_INTENT",
    "missing_business_policy": "CLARIFY",
    "context_business_guard": True,
    "cross_business_decomposer": "HCX-007_STRUCTURED_OUTPUT",
})

# ==== cell 54 ====
def run_multiturn_regression_tests_v21() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def run_case(
        name: str,
        question: str,
        active: Sequence[str],
        expected_route: str,
        *,
        expected_context: bool | None = None,
        resolved_contains: str | None = None,
        expected_reason: str | None = None,
    ) -> None:
        state = new_context_state()
        state["active_businesses"] = list(active)
        result = resolve_context_v2(question, state=state, llm_classifier=None)
        passed = result["route"] == expected_route
        if expected_context is not None:
            passed = passed and bool(result.get("context_used")) is expected_context
        if resolved_contains is not None:
            passed = passed and resolved_contains in str(result.get("resolved_question") or "")
        if expected_reason is not None:
            passed = passed and result.get("reason") == expected_reason
        rows.append({
            "case": name,
            "question": question,
            "active_businesses": " | ".join(active) or "-",
            "route": result.get("route"),
            "context_used": bool(result.get("context_used")),
            "reason": result.get("reason"),
            "resolved_question": result.get("resolved_question"),
            "passed": passed,
        })

    run_case(
        "자연어 신청서류 후속질의",
        "신청 서류는 어떻게 돼?",
        ["채무조정"],
        "CONTINUE",
        expected_context=True,
        resolved_contains="채무조정",
        expected_reason="INTENT_BASED_UNIQUE_ACTIVE_BUSINESS",
    )
    run_case(
        "준비물 표현 후속질의",
        "필요한 준비물이 뭐야?",
        ["채무조정"],
        "CONTINUE",
        expected_context=True,
        resolved_contains="채무조정",
    )
    run_case(
        "활성업무 없는 신청서류",
        "신청 서류는 어떻게 돼?",
        [],
        "CLARIFY",
        expected_context=False,
        expected_reason="INTENT_FOLLOWUP_WITHOUT_ACTIVE_BUSINESS",
    )
    run_case(
        "활성업무 둘인 후속질의",
        "신청 서류는 어떻게 돼?",
        ["고객 미수령금", "착오송금 반환지원"],
        "CLARIFY",
        expected_context=False,
        expected_reason="INTENT_FOLLOWUP_MULTIPLE_ACTIVE_BUSINESSES",
    )
    run_case(
        "명시적 새 업무 우선",
        "착오송금 신청 서류는 무엇인가요?",
        ["채무조정"],
        "CONTINUE",
        expected_context=False,
        resolved_contains="착오송금",
    )
    run_case(
        "착오송금 기간 역할 확인",
        "얼마나 걸리나요?",
        ["착오송금 반환지원"],
        "CLARIFY",
        expected_context=False,
        expected_reason="MISTAKEN_TRANSFER_TIME_SCOPE_AMBIGUOUS",
    )
    run_case(
        "명시적 OOS는 문맥결합 금지",
        "비트코인 투자 방법은 어떻게 돼?",
        ["채무조정"],
        "CONTINUE",
        expected_context=False,
    )

    frame = pd.DataFrame(rows)
    failed = frame.loc[~frame["passed"]]
    if not failed.empty:
        raise AssertionError("멀티턴 회귀 테스트 실패:\n" + failed.to_string(index=False))
    return frame


multiturn_regression_v21 = run_multiturn_regression_tests_v21()
display(multiturn_regression_v21)
print(f"멀티턴 회귀 테스트: {len(multiturn_regression_v21)}/{len(multiturn_regression_v21)} 통과")

# ==== cell 56 ====

import copy
import re
import time
from typing import Any, Mapping, Sequence


RELATIONAL_CROSS_BUSINESS_PATTERN_V22 = re.compile(
    r"(?:도\s*(?:같이|함께|동시에)|같이\s*(?:신청|이용|진행)|"
    r"함께|동시에|동시\s*신청|병행|둘\s*다|두\s*(?:제도|업무)\s*모두)",
    re.I,
)


def _is_relational_cross_business_followup_v22(
    question: str,
    explicit_businesses: Sequence[str],
    active_businesses: Sequence[str],
) -> bool:
    if not explicit_businesses or not active_businesses:
        return False
    if EXCLUSION_PATTERN.search(question) or CORRECTION_PATTERN.search(question):
        return False
    if not RELATIONAL_CROSS_BUSINESS_PATTERN_V22.search(question):
        return False
    return bool(set(explicit_businesses) - set(active_businesses))


_RESOLVE_CONTEXT_V21_BEFORE_RELATIONAL = resolve_context_v2


def resolve_context_v22(
    question: str,
    *,
    state: dict[str, Any],
    llm_classifier=None,
) -> dict[str, Any]:
    started = time.perf_counter()
    original = _clean(question)
    if not original:
        raise ValueError("질문이 비어 있습니다.")
    for key, default in new_context_state().items():
        state.setdefault(key, copy.deepcopy(default))

    explicit = detect_businesses(original)
    active = list(state.get("active_businesses") or [])
    if _is_relational_cross_business_followup_v22(original, explicit, active):
        combined = list(dict.fromkeys(active + explicit))
        if len(combined) < 2:
            return _RESOLVE_CONTEXT_V21_BEFORE_RELATIONAL(
                original, state=state, llm_classifier=llm_classifier
            )
        relation_subject = "과 ".join(combined)
        resolved = f"{relation_subject}의 동시·병행 신청 가능 여부에 관한 질문: {original}"
        state["active_businesses"] = combined
        state["pending_clarification"] = None
        state["last_resolved_question"] = resolved
        return {
            "route": "CONTINUE",
            "dialogue_act": "RELATIONAL_FOLLOW_UP",
            "original_question": original,
            "resolved_question": resolved,
            "current_question_complete": False,
            "context_used": True,
            "reason": "RELATIONAL_CROSS_BUSINESS_FOLLOWUP",
            "clarification_message": "",
            "active_businesses": combined,
            "excluded_businesses": list(state.get("excluded_businesses") or []),
            "actor_role": state.get("actor_role"),
            "missing_slots": [],
            "followup_intents": detect_followup_intents_v21(original),
            "pending_clarification": None,
            "llm_judgment": {"called": False, "reason": "RULE_RELATIONAL_CROSS_BUSINESS"},
            "latency_ms": (time.perf_counter() - started) * 1000,
        }

    return _RESOLVE_CONTEXT_V21_BEFORE_RELATIONAL(
        original, state=state, llm_classifier=llm_classifier
    )


resolve_context_v2 = resolve_context_v22


def run_relational_multiturn_regression_v22() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def check(
        name: str,
        question: str,
        active: Sequence[str],
        expected_reason: str,
        expected_context: bool,
        expected_businesses: Sequence[str],
    ) -> None:
        state = new_context_state()
        state["active_businesses"] = list(active)
        result = resolve_context_v2(question, state=state, llm_classifier=None)
        passed = (
            result.get("reason") == expected_reason
            and bool(result.get("context_used")) is expected_context
            and set(result.get("active_businesses") or []) == set(expected_businesses)
        )
        rows.append({
            "case": name,
            "question": question,
            "reason": result.get("reason"),
            "context_used": bool(result.get("context_used")),
            "active_businesses": " | ".join(result.get("active_businesses") or []),
            "resolved_question": result.get("resolved_question"),
            "passed": passed,
        })

    check(
        "관계형 교차업무 보존",
        "그럼 채무조정도 같이 신청할 수 있어요?",
        ["착오송금 반환지원"],
        "RELATIONAL_CROSS_BUSINESS_FOLLOWUP",
        True,
        ["착오송금 반환지원", "채무조정"],
    )
    check(
        "명시적 독립 새 질문",
        "채무조정 신청 자격은 무엇인가요?",
        ["착오송금 반환지원"],
        "CURRENT_QUESTION_COMPLETE",
        False,
        ["채무조정"],
    )
    check(
        "제외 표현은 기존 정책 우선",
        "착오송금 말고 채무조정 신청을 알려주세요",
        ["착오송금 반환지원"],
        "EXPLICIT_EXCLUSION_WITH_REPLACEMENT",
        False,
        ["채무조정"],
    )
    frame = pd.DataFrame(rows)
    if not bool(frame["passed"].all()):
        raise AssertionError("관계형 멀티턴 회귀 테스트 실패\n" + frame.to_string(index=False))
    return frame


relational_multiturn_regression_v22 = run_relational_multiturn_regression_v22()
display(relational_multiturn_regression_v22)
print("관계형 멀티턴 회귀 테스트 통과")

# ==== cell 58 ====

import json
import random
import re
import time
from collections import OrderedDict
from typing import Any, Mapping, Sequence


ANSWER_EVIDENCE_TOTAL_MAX_CHARS = 14_000
ANSWER_EVIDENCE_RANK_BUDGETS = (4_000, 3_500, 3_000, 2_000, 1_500)
ANSWER_CACHE_ENABLED_FOR_COMPARISON = False
ANSWER_PROMPT_VERSION = "bd-low-latency-v1"


def _truncate_at_boundary_v1(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    limited = text[:limit]
    boundary = max(limited.rfind("\n"), limited.rfind(" "))
    if boundary >= int(limit * 0.75):
        limited = limited[:boundary]
    return limited.rstrip()


def _proximity_order_v1(context_ids: Sequence[str], matched_ids: Sequence[str]) -> list[str]:
    context = list(dict.fromkeys(str(value) for value in context_ids if str(value)))
    matched = list(dict.fromkeys(str(value) for value in matched_ids if str(value)))
    output = [value for value in matched if value in context]
    for matched_id in matched:
        if matched_id not in context:
            output.append(matched_id)
            continue
        center = context.index(matched_id)
        for distance in range(1, len(context) + 1):
            for index in (center - distance, center + distance):
                if 0 <= index < len(context) and context[index] not in output:
                    output.append(context[index])
    output.extend(value for value in context if value not in output)
    return output


def build_compact_parent_evidence_pack_v1(
    question: str,
    search_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_parent: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for result in search_results:
        child = dict(result.get("chunk") or {})
        child_id = str(result.get("chunk_id") or child.get("chunk_id") or "")
        parent_id = str(result.get("parent_doc_id") or _parent_id_for_chunk(child))
        row = by_parent.setdefault(parent_id, {
            "rank": int(result.get("rank") or len(by_parent) + 1),
            "parent_id": parent_id,
            "representative_chunk_id": child_id,
            "matched_child_ids": [],
            "matched_child_ranks": [],
            "context_chunk_ids": list(result.get("parent_context_chunk_ids") or [child_id]),
            "document_title": _clean_text(child.get("title") or child.get("document_title")),
            "source_url": _clean_text(child.get("source_url")),
        })
        row["matched_child_ids"].append(child_id)
        row["matched_child_ranks"].append(int(result.get("rank") or 0))

    evidence: list[dict[str, Any]] = []
    sources: OrderedDict[str, dict[str, Any]] = OrderedDict()
    total_remaining = ANSWER_EVIDENCE_TOTAL_MAX_CHARS
    for parent_index, row in enumerate(by_parent.values()):
        if total_remaining <= 0 or parent_index >= len(ANSWER_EVIDENCE_RANK_BUDGETS):
            break
        parent_budget = min(ANSWER_EVIDENCE_RANK_BUDGETS[parent_index], total_remaining)
        ordered_ids = _proximity_order_v1(
            row["context_chunk_ids"], row["matched_child_ids"]
        )
        parts: list[str] = []
        included_ids: list[str] = []
        section_titles: list[str] = []
        remaining = parent_budget
        for chunk_id in ordered_ids:
            chunk = CHUNKS_BY_ID.get(str(chunk_id))
            if chunk is None:
                continue
            title = _clean_text(chunk.get("title"))
            section = _clean_text(chunk.get("section_title"))
            if section and section not in section_titles:
                section_titles.append(section)
            label = " / ".join(value for value in (title, section) if value)
            part = f"[{chunk_id}] {label}\n{_clean_text(chunk.get('content'))}".strip()
            separator_cost = 2 if parts else 0
            if remaining <= separator_cost:
                break
            part = _truncate_at_boundary_v1(part, remaining - separator_cost)
            if not part:
                break
            parts.append(part)
            included_ids.append(str(chunk_id))
            remaining -= len(part) + separator_cost
            if remaining < 120:
                break

        content = "\n\n".join(parts)
        if not content:
            continue
        evidence_id = f"E{len(evidence) + 1}"
        evidence.append({
            "evidence_id": evidence_id,
            "rank": int(row["rank"]),
            "chunk_id": row["representative_chunk_id"],
            "parent_id": row["parent_id"],
            "context_chunk_ids": included_ids,
            "matched_child_ids": list(dict.fromkeys(row["matched_child_ids"])),
            "matched_child_ranks": sorted(set(row["matched_child_ranks"])),
            "document_title": row["document_title"],
            "section_title": " · ".join(section_titles),
            "content": content,
            "context_char_count": len(content),
            "context_truncated": len(included_ids) < len(ordered_ids),
            "source_url": row["source_url"],
        })
        total_remaining -= len(content)
        url = row["source_url"]
        if url:
            source = sources.setdefault(url, {
                "source_id": f"S{len(sources) + 1}",
                "title": row["document_title"] or "공식 출처",
                "source_url": url,
                "evidence_ids": [],
            })
            source["evidence_ids"].append(evidence_id)

    if not evidence:
        raise ValueError("저지연 Evidence Pack을 만들 근거가 없습니다.")
    return {
        "question": _clean_text(question),
        "search_parent_context_max_chars": PARENT_CONTEXT_MAX_CHARS,
        "answer_evidence_total_max_chars": ANSWER_EVIDENCE_TOTAL_MAX_CHARS,
        "answer_prompt_version": ANSWER_PROMPT_VERSION,
        "evidence": evidence,
        "sources": list(sources.values()),
    }


RELATION_QUESTION_PATTERN_V1 = re.compile(
    r"(?:같이|함께|동시에|동시\s*신청|병행|둘\s*다|두\s*(?:제도|업무)\s*모두)", re.I
)
RELATION_POSITIVE_ANSWER_PATTERN_V1 = re.compile(
    r"(?:같이|함께|동시에|병행).{0,12}(?:신청|이용).{0,8}(?:가능|할\s*수\s*있)", re.I
)
RELATION_EVIDENCE_PATTERN_V1 = re.compile(
    r"(?:동시\s*신청|함께\s*신청|같이\s*신청|병행\s*(?:신청|이용)|중복\s*신청)", re.I
)


def relation_constraint_v1(question: str, pack: Mapping[str, Any]) -> dict[str, Any]:
    relation_question = bool(RELATION_QUESTION_PATTERN_V1.search(question))
    direct_rows = [
        str(row.get("evidence_id"))
        for row in pack.get("evidence") or []
        if RELATION_EVIDENCE_PATTERN_V1.search(str(row.get("content") or ""))
    ]
    return {
        "relation_question": relation_question,
        "direct_relation_evidence_ids": direct_rows,
        "may_affirm_joint_application": bool(direct_rows),
        "rule": (
            "두 제도의 동시·병행 가능성을 직접 명시한 동일 Evidence가 없으면 "
            "가능하다고 단정하지 않고 확인되지 않는다고 답한다."
        ),
    }


B_LOW_LATENCY_SYSTEM_PROMPT = """
당신은 예금보험공사 공식 문서 기반 답변 시스템입니다.

1. 사용자 질문과 제공된 Evidence Pack의 사실만 사용하세요.
2. 질문에 직접 답하고 필요한 대상·조건·예외·금액·기간·절차를 설명하세요.
3. Evidence에 없는 사실이나 서로 다른 제도의 조건을 임의로 결합하지 마세요.
4. 동시·병행 신청 질문은 같은 Evidence가 그 관계를 직접 명시할 때만 가능하다고 답하세요.
5. 별도 문서가 각 제도의 자격을 각각 설명한다는 사실만으로 동시 신청 가능성을 추론하지 마세요.
6. 직접 관계 근거가 없으면 확인되지 않는다고 답하고 coverage_status를 PARTIAL로 두세요.
7. 문장 수는 제한하지 않되 질문하지 않은 배경 설명과 중복은 넣지 마세요.
8. 근거 문장 끝에 [E1] 형식으로 실제 Evidence ID를 표시하세요.
9. 지정된 JSON 객체 하나만 출력하고 문자열 내부 줄바꿈은 \\n으로 이스케이프하세요.
""".strip()


D_SKELETON_SYSTEM_PROMPT = """
당신은 예금보험공사 공식 문서에서 답변에 필요한 사실 구조만 추출하는 분석기입니다.

1. 사용자 질문과 동일 Evidence Pack에 명시된 사실만 사용하세요.
2. 최종 사용자 문장이 아니라 Answer Skeleton JSON만 작성하세요.
3. 각 answer_item에는 질문이 요구한 항목 하나와 실제 evidence_ids를 연결하세요.
4. 서로 다른 제도의 조건을 임의로 결합하지 마세요.
5. 동시·병행 신청 관계는 같은 Evidence가 그 관계를 직접 명시할 때만 claim으로 채택하세요.
6. 직접 관계 근거가 없으면 uncertainties에 기록하고 가능하다고 추론하지 마세요.
7. 문서 충돌은 conflicts, 확인 불가는 uncertainties에 기록하세요.
8. JSON 객체 하나만 출력하고 문자열 내부 줄바꿈은 \\n으로 이스케이프하세요.
""".strip()


D_FINAL_SYSTEM_PROMPT = """
당신은 예금보험공사 공식 문서 기반 질의응답 시스템입니다.

1. 제공된 Answer Skeleton과 동일 Evidence Pack의 사실만 사용하세요.
2. 질문에 직접 답하고 Skeleton의 항목에 필요한 조건·예외·금액·기간·절차를 설명하세요.
3. Skeleton 또는 Evidence에 없는 사실을 추정하지 마세요.
4. 동시·병행 가능성에 직접 근거가 없으면 가능하다고 단정하지 마세요.
5. 문장 수는 제한하지 않되 질문하지 않은 배경 설명과 중복은 넣지 마세요.
6. 각 주장에는 Skeleton이 허용한 [E1] 형식의 Evidence ID만 표시하세요.
7. JSON, Skeleton, 내부 구현, 검색 점수는 답변에 언급하지 마세요.
""".strip()


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _usage_dict(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    return {
        key: int(getattr(usage, key, 0) or 0)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def _call_answer_api_v1(
    *, system_prompt: str, user_prompt: str, max_tokens: int
) -> tuple[str, dict[str, int], float, dict[str, Any]]:
    global _LAST_ANSWER_API_TRACE
    started = time.perf_counter()
    response = ANSWER_HCX_CLIENT.chat.completions.create(
        model=HCX_CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    wall_ms = (time.perf_counter() - started) * 1000
    content = response.choices[0].message.content
    if not content or not str(content).strip():
        raise RuntimeError("HCX 답변 출력이 비어 있습니다.")
    return str(content), _usage_dict(response), wall_ms, dict(_LAST_ANSWER_API_TRACE)


def _merge_usage_v1(*values: Mapping[str, Any]) -> dict[str, int]:
    return {
        key: sum(int(value.get(key) or 0) for value in values)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def _relation_safe_answer_v1(
    answer: str,
    constraint: Mapping[str, Any],
) -> tuple[str, bool]:
    if (
        constraint.get("relation_question")
        and not constraint.get("may_affirm_joint_application")
        and RELATION_POSITIVE_ANSWER_PATTERN_V1.search(answer)
    ):
        return (
            "현재 검색된 공식 문서 근거만으로 두 제도를 동시에 또는 병행하여 "
            "신청할 수 있는지는 확인되지 않습니다. 각 제도의 개별 신청 요건은 "
            "확인할 수 있지만, 그것만으로 동시 신청 가능성을 단정할 수는 없습니다.",
            True,
        )
    return answer, False


def generate_answer_b_low_latency_v1(
    question: str,
    pack: Mapping[str, Any],
) -> dict[str, Any]:
    constraint = relation_constraint_v1(question, pack)
    prompt = f"""[사용자 질문]\n{_clean_text(question)}\n\n[관계 주장 안전조건]\n{_compact_json(constraint)}\n\n[Basic Evidence Pack]\n{_compact_json(pack)}\n\n[출력 JSON]\n{{\"answer\":\"근거 문장에 [E1] 표시\",\"used_evidence_ids\":[\"E1\"],\"coverage_status\":\"SUFFICIENT|PARTIAL|INSUFFICIENT\",\"missing_information\":[]}}"""
    raw, usage, first_ms, first_trace = _call_answer_api_v1(
        system_prompt=B_LOW_LATENCY_SYSTEM_PROMPT,
        user_prompt=prompt,
        max_tokens=1600,
    )
    attempts = [{"stage": "initial", "latency_ms": first_ms, "trace": first_trace}]
    total_usage = dict(usage)
    total_ms = first_ms
    try:
        raw_payload = answer_b_core._extract_json_object(raw)
        requested_ids = answer_b_core._clean_list(raw_payload.get("used_evidence_ids"))
        allowed = answer_b_core._allowed_evidence(pack)
        valid_ids = [value for value in requested_ids if value in allowed]
        answer = answer_b_core._strip_model_urls(answer_b_core._clean(raw_payload.get("answer")))
        if not answer:
            raise ValueError("answer가 비어 있습니다.")
        if not valid_ids:
            valid_ids = [f"E{n}" for n in re.findall(r"\[E(\d+)\]", answer) if f"E{n}" in allowed]
        if not valid_ids:
            raise ValueError("유효 Evidence ID가 없습니다.")
        payload = {
            "answer": answer,
            "used_evidence_ids": list(dict.fromkeys(valid_ids)),
            "used_chunk_ids": [allowed[value] for value in dict.fromkeys(valid_ids)],
            "coverage_status": str(raw_payload.get("coverage_status") or "PARTIAL").upper(),
            "missing_information": answer_b_core._clean_list(raw_payload.get("missing_information")),
        }
        payload = answer_b_core.validate_basic_answer(payload, pack)
    except (ValueError, TypeError):
        try:
            payload = answer_b_core._recover_basic_answer_from_raw(raw, pack)
            attempts.append({"stage": "local_raw_recovery", "latency_ms": 0.0})
        except (ValueError, TypeError):
            repair_prompt = f"""다음 출력을 사실 변경 없이 올바른 JSON 객체로만 고치세요. Evidence Pack 밖의 ID를 만들지 마세요.\n\n[원래 요청]\n{prompt}\n\n[교정 대상]\n{raw[:6000]}"""
            repaired, repair_usage, repair_ms, repair_trace = _call_answer_api_v1(
                system_prompt=B_LOW_LATENCY_SYSTEM_PROMPT,
                user_prompt=repair_prompt,
                max_tokens=1600,
            )
            total_usage = _merge_usage_v1(total_usage, repair_usage)
            total_ms += repair_ms
            attempts.append({"stage": "repair", "latency_ms": repair_ms, "trace": repair_trace})
            parsed = answer_b_core._extract_json_object(repaired)
            allowed = answer_b_core._allowed_evidence(pack)
            ids = [value for value in answer_b_core._clean_list(parsed.get("used_evidence_ids")) if value in allowed]
            payload = answer_b_core.validate_basic_answer({
                "answer": parsed.get("answer"),
                "used_evidence_ids": ids,
                "used_chunk_ids": [allowed[value] for value in ids],
                "coverage_status": parsed.get("coverage_status") or "PARTIAL",
                "missing_information": parsed.get("missing_information") or [],
            }, pack)

    safe_answer, guard_applied = _relation_safe_answer_v1(payload["answer"], constraint)
    payload["answer"] = safe_answer
    if guard_applied:
        payload["coverage_status"] = "PARTIAL"
        payload["missing_information"] = list(dict.fromkeys(
            list(payload.get("missing_information") or [])
            + ["두 제도의 동시·병행 신청 가능 여부를 직접 명시한 공식 근거"]
        ))
    return {
        **payload,
        "system": "B",
        "latency_ms": total_ms,
        "usage": total_usage,
        "api_calls": sum(1 for row in attempts if row["stage"] in {"initial", "repair"}),
        "attempts": attempts,
        "relation_constraint": constraint,
        "relation_guard_applied": guard_applied,
    }


def _validate_d_skeleton_v1(
    raw: Mapping[str, Any], pack: Mapping[str, Any]
) -> dict[str, Any]:
    allowed = answer_b_core._allowed_evidence(pack)
    items: list[dict[str, Any]] = []
    for index, item in enumerate(raw.get("answer_items") or [], start=1):
        if not isinstance(item, Mapping):
            continue
        claim = answer_b_core._clean(item.get("claim"))
        ids = [value for value in answer_b_core._clean_list(item.get("evidence_ids")) if value in allowed]
        if claim and ids:
            items.append({
                "item_id": f"A{len(items) + 1}",
                "topic": answer_b_core._clean(item.get("topic")) or f"답변 항목 {index}",
                "claim": claim,
                "conditions": answer_b_core._clean_list(item.get("conditions")),
                "details": answer_b_core._clean_list(item.get("details")),
                "evidence_ids": list(dict.fromkeys(ids)),
            })
    core = answer_b_core._clean(raw.get("core_answer"))
    if not core or not items:
        raise ValueError("D안 Skeleton의 핵심 답변 또는 유효 항목이 없습니다.")
    coverage = answer_b_core._clean(raw.get("coverage_status")).upper()
    if coverage not in answer_b_core.ALLOWED_COVERAGE_STATUS:
        coverage = "PARTIAL"
    return {
        "core_answer": core,
        "answer_items": items,
        "uncertainties": answer_b_core._clean_list(raw.get("uncertainties")),
        "conflicts": answer_b_core._clean_list(raw.get("conflicts")),
        "coverage_status": coverage,
    }


def generate_answer_d_v1(
    question: str,
    pack: Mapping[str, Any],
) -> dict[str, Any]:
    constraint = relation_constraint_v1(question, pack)
    skeleton_prompt = f"""[사용자 질문]\n{_clean_text(question)}\n\n[관계 주장 안전조건]\n{_compact_json(constraint)}\n\n[Evidence Pack]\n{_compact_json(pack)}\n\n[출력 JSON]\n{{\"core_answer\":\"핵심 결론\",\"answer_items\":[{{\"topic\":\"항목\",\"claim\":\"근거 사실\",\"conditions\":[],\"details\":[],\"evidence_ids\":[\"E1\"]}}],\"uncertainties\":[],\"conflicts\":[],\"coverage_status\":\"SUFFICIENT|PARTIAL|INSUFFICIENT\"}}"""
    raw_skeleton, usage1, skeleton_ms, trace1 = _call_answer_api_v1(
        system_prompt=D_SKELETON_SYSTEM_PROMPT,
        user_prompt=skeleton_prompt,
        max_tokens=2000,
    )
    skeleton = _validate_d_skeleton_v1(
        answer_b_core._extract_json_object(raw_skeleton), pack
    )
    final_prompt = f"""[사용자 질문]\n{_clean_text(question)}\n\n[관계 주장 안전조건]\n{_compact_json(constraint)}\n\n[Answer Skeleton]\n{_compact_json(skeleton)}\n\n[동일 Evidence Pack]\n{_compact_json(pack)}\n\n위 근거 범위에서 사용자용 최종 답변을 작성하세요."""
    raw_answer, usage2, final_ms, trace2 = _call_answer_api_v1(
        system_prompt=D_FINAL_SYSTEM_PROMPT,
        user_prompt=final_prompt,
        max_tokens=1600,
    )
    answer = answer_b_core._strip_model_urls(str(raw_answer).strip())
    if not answer:
        raise ValueError("D안 최종 답변이 비어 있습니다.")
    safe_answer, guard_applied = _relation_safe_answer_v1(answer, constraint)
    return {
        "system": "D",
        "answer": safe_answer,
        "skeleton": skeleton,
        "coverage_status": "PARTIAL" if guard_applied else skeleton["coverage_status"],
        "latency_ms": skeleton_ms + final_ms,
        "skeleton_latency_ms": skeleton_ms,
        "final_latency_ms": final_ms,
        "usage": _merge_usage_v1(usage1, usage2),
        "api_calls": 2,
        "attempts": [
            {"stage": "skeleton", "latency_ms": skeleton_ms, "trace": trace1},
            {"stage": "final", "latency_ms": final_ms, "trace": trace2},
        ],
        "relation_constraint": constraint,
        "relation_guard_applied": guard_applied,
    }


print({
    "search_parent_context_max_chars": PARENT_CONTEXT_MAX_CHARS,
    "answer_evidence_total_max_chars": ANSWER_EVIDENCE_TOTAL_MAX_CHARS,
    "answer_evidence_rank_budgets": ANSWER_EVIDENCE_RANK_BUDGETS,
    "sentence_limit": None,
    "comparison_cache": ANSWER_CACHE_ENABLED_FOR_COMPARISON,
})

# ==== cell 60 ====

import copy
import html
import time
from typing import Any, Mapping, Sequence


def new_bd_comparison_state() -> dict[str, Any]:
    return new_context_state()


def prepare_common_retrieval_v1(
    question: str,
    *,
    state: dict[str, Any],
) -> dict[str, Any]:
    global _LAST_RERANK_TRACE, _LAST_PARENT_CHILD_TRACE
    total_started = time.perf_counter()
    _LAST_RERANK_TRACE = {}
    _LAST_PARENT_CHILD_TRACE = {}

    analysis = ANALYZER_FUNCTIONS["v15"](question, state)
    analysis_ms = (time.perf_counter() - total_started) * 1000
    if analysis["route"] != "RETRIEVE":
        return {
            "question": question,
            "resolved_question": analysis.get("resolved_question") or "",
            "route": analysis["route"],
            "analysis": analysis,
            "route_message": _route_message(analysis),
            "latency_ms": {
                "질의분석": analysis_ms,
                "공통 준비 전체": (time.perf_counter() - total_started) * 1000,
            },
        }

    plans = list(analysis.get("plans") or [])
    if not analysis.get("query_plan_valid") or not _comparison_plans_valid("RETRIEVE", plans):
        raise RuntimeError("RETRIEVE 검색 계획이 유효하지 않습니다.")

    search_started = time.perf_counter()
    search_results, per_query = fuse_query_results(plans)
    before_threshold_count = len(search_results)
    search_results = filter_search_results_by_relevance(search_results)
    child_search_ms = (time.perf_counter() - search_started) * 1000
    reranker_trace = dict(_LAST_RERANK_TRACE)
    reranker_trace.update({
        "min_relevance_score": float(MIN_RELEVANCE_SCORE),
        "before_threshold_count": before_threshold_count,
        "after_threshold_count": len(search_results),
    })

    if not search_results:
        return {
            "question": question,
            "resolved_question": analysis.get("resolved_question") or question,
            "route": "DIRECT",
            "analysis": analysis,
            "route_message": (
                "공식 근거에서 충분히 관련된 내용을 찾지 못했습니다. "
                "질문에 업무명이나 신청 상황을 조금 더 구체적으로 적어 주세요."
            ),
            "plans": plans,
            "search_results": [],
            "per_query": per_query,
            "reranker": reranker_trace,
            "latency_ms": {
                "질의분석": analysis_ms,
                "검색": child_search_ms,
                "공통 준비 전체": (time.perf_counter() - total_started) * 1000,
            },
        }

    parent_started = time.perf_counter()
    search_results = expand_parent_context(search_results)
    parent_ms = (time.perf_counter() - parent_started) * 1000
    parent_trace = dict(_LAST_PARENT_CHILD_TRACE)

    answer_question = str(analysis.get("resolved_question") or question)
    pack_started = time.perf_counter()
    pack = build_compact_parent_evidence_pack_v1(answer_question, search_results)
    pack_ms = (time.perf_counter() - pack_started) * 1000
    per_query_latency = _query_latency_rows(per_query)
    common_ms = (time.perf_counter() - total_started) * 1000
    return {
        "question": question,
        "resolved_question": answer_question,
        "route": analysis["route"],
        "analysis": analysis,
        "plans": plans,
        "search_results": search_results,
        "per_query": per_query,
        "per_query_latency": per_query_latency,
        "reranker": reranker_trace,
        "parent_child": parent_trace,
        "evidence_pack": pack,
        "evidence_pack_sha256": evidence_pack_sha256(pack),
        "evidence_chars": sum(int(row.get("context_char_count") or 0) for row in pack["evidence"]),
        "latency_ms": {
            "문맥 처리": float(analysis.get("context_latency_ms") or 0),
            "질의분석": analysis_ms,
            "검색": child_search_ms + parent_ms,
            "질문 임베딩": sum(row["embedding_latency_ms"] for row in per_query_latency),
            "Dense 계산": sum(row["dense_compute_latency_ms"] for row in per_query_latency),
            "BM25": sum(row["bm25_latency_ms"] for row in per_query_latency),
            "BAAI Reranker": float(reranker_trace.get("latency_ms") or 0),
            "Parent-Child8192": parent_ms,
            "저지연 Evidence Pack": pack_ms,
            "공통 준비 전체": common_ms,
        },
    }


def _trace_totals_v1(payload: Mapping[str, Any]) -> dict[str, float | int]:
    waits = 0.0
    rate_limits = 0
    for stage in payload.get("attempts") or []:
        trace = stage.get("trace") or {}
        waits += float(trace.get("total_wait_ms") or 0)
        rate_limits += sum(
            1 for attempt in trace.get("attempts") or []
            if attempt.get("status") == "RATE_LIMIT_429"
        )
    return {"wait_ms": waits, "rate_limit_429_count": rate_limits}


_BD_ORDER_COUNTER = 0


def compare_answer_systems_v1(
    question: str,
    *,
    state: dict[str, Any],
    order: str = "ALTERNATE",
) -> dict[str, Any]:
    global _BD_ORDER_COUNTER
    common = prepare_common_retrieval_v1(question, state=state)
    if common["route"] != "RETRIEVE":
        return {"common": common, "answers": {}, "summary": pd.DataFrame()}

    normalized_order = str(order or "ALTERNATE").upper()
    if normalized_order == "ALTERNATE":
        normalized_order = "B_FIRST" if _BD_ORDER_COUNTER % 2 == 0 else "D_FIRST"
        _BD_ORDER_COUNTER += 1
    sequence = ("B", "D") if normalized_order == "B_FIRST" else ("D", "B")
    answers: dict[str, dict[str, Any]] = {}
    for system in sequence:
        if system == "B":
            answers[system] = generate_answer_b_low_latency_v1(
                common["resolved_question"], common["evidence_pack"]
            )
        else:
            answers[system] = generate_answer_d_v1(
                common["resolved_question"], common["evidence_pack"]
            )

    common_ms = float(common["latency_ms"]["공통 준비 전체"])
    rows = []
    for system in ("B", "D"):
        payload = answers[system]
        trace = _trace_totals_v1(payload)
        usage = payload.get("usage") or {}
        rows.append({
            "답변안": system,
            "실행순서": sequence.index(system) + 1,
            "Evidence문자": common["evidence_chars"],
            "API호출": int(payload.get("api_calls") or 0),
            "429횟수": int(trace["rate_limit_429_count"]),
            "호출간격·429대기(ms)": float(trace["wait_ms"]),
            "입력토큰": int(usage.get("prompt_tokens") or 0),
            "출력토큰": int(usage.get("completion_tokens") or 0),
            "답변지연(ms)": float(payload.get("latency_ms") or 0),
            "가상E2E(ms)": common_ms + float(payload.get("latency_ms") or 0),
            "답변글자": len(str(payload.get("answer") or "")),
            "Coverage": payload.get("coverage_status"),
            "관계안전가드": bool(payload.get("relation_guard_applied")),
        })
    summary = pd.DataFrame(rows)
    state.setdefault("turns", []).extend([
        {"role": "user", "content": question},
        {"role": "assistant", "content": user_visible_answer(answers["B"]["answer"])},
    ])
    return {
        "common": common,
        "answers": answers,
        "execution_order": list(sequence),
        "summary": summary,
    }


def _sources_from_pack_v1(pack: Mapping[str, Any]) -> str:
    lines = ["### 공통 공식 출처", ""]
    for source in pack.get("sources") or []:
        title = str(source.get("title") or "공식 출처")
        url = str(source.get("source_url") or "")
        if url:
            lines.append(f"- [{title}]({url})")
    return "\n".join(lines) if len(lines) > 2 else ""


def render_bd_comparison_v1(result: Mapping[str, Any], *, show_pack: bool = False) -> None:
    common = result["common"]
    if common["route"] != "RETRIEVE":
        display(Markdown(f"### {common['route']}\n\n{common.get('route_message', '')}"))
        return
    answers = result["answers"]
    display(Markdown("## 답변 B안\n\n" + user_visible_answer(answers["B"]["answer"])))
    display(Markdown("## 답변 D안\n\n" + user_visible_answer(answers["D"]["answer"])))
    source_md = _sources_from_pack_v1(common["evidence_pack"])
    if source_md:
        display(Markdown(source_md))
    display(Markdown("### B/D 레이턴시·토큰 비교"))
    display(result["summary"])
    display(Markdown(analysis_markdown_compare({
        "analyzer_label": "V1.5 관계형 멀티턴 개선",
        "analysis": common["analysis"],
    })))
    display(Markdown(retrieval_markdown_compare({"search_results": common["search_results"]})))
    display(Markdown(latency_markdown_compare({"latency_ms": common["latency_ms"]})))
    if show_pack:
        display(JSON(common["evidence_pack"], expanded=False))


print("공통 검색 1회 기반 B/D 비교 실행기 준비 완료")

# ==== cell 62 ====

import json
import re
from typing import Any, Mapping, Sequence


ANSWER_NEED_RULES_V2: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("ACTOR", "신청·신고 주체", re.compile(r"(?:누가|누구|어떤\s*사람|신청자|신고자)")),
    ("ELIGIBILITY", "신청·지원 자격", re.compile(r"(?:자격|지원\s*대상|신청\s*대상|신고\s*대상|대상자|가능\s*여부)")),
    ("AMOUNT", "금액·한도", re.compile(r"(?:얼마|금액|한도|포상금|보호액|비율|퍼센트|%)")),
    ("DOCUMENTS", "필요서류", re.compile(r"(?:서류|구비서류|필요\s*서류|준비물|증빙)")),
    ("PROCEDURE", "신청·처리 절차", re.compile(r"(?:어떻게|절차|방법|순서|과정|신청하려면|신고하려면|접수하려면)")),
    ("TIME", "기간·기한", re.compile(r"(?:기간|기한|언제|얼마나\s*걸|며칠|몇\s*일)")),
    ("COST", "비용·수수료", re.compile(r"(?:비용|수수료|돈이\s*드|차감)")),
    ("EXCEPTION", "예외·제외", re.compile(r"(?:예외|제외|안\s*되는|불가능|받지\s*못|해당하지\s*않)")),
    ("COMPARISON", "차이·비교", re.compile(r"(?:차이|다른가|비교|무엇이\s*다|뭐가\s*다)")),
)

NEED_STATUS_VALUES_V2 = {"ANSWERED", "PARTIAL", "UNSUPPORTED"}


def _question_clauses_v2(question: str) -> list[str]:
    cleaned = _clean_text(question)
    parts = re.split(
        r"\s*(?:,|;|\?|그리고|또한|또|및|그러면|그럼)\s*",
        cleaned,
    )
    return [part.strip(" .?!") for part in parts if part.strip(" .?!")]


def extract_answer_needs_v2(question: str) -> list[dict[str, Any]]:
    original = _clean_text(question)
    clauses = _question_clauses_v2(original) or [original]
    needs: list[dict[str, Any]] = []
    for need_type, label, pattern in ANSWER_NEED_RULES_V2:
        if not pattern.search(original):
            continue
        matching = [clause for clause in clauses if pattern.search(clause)]
        needs.append({
            "need_id": f"N{len(needs) + 1}",
            "need_type": need_type,
            "label": label,
            "question_part": matching[0] if matching else original,
        })
    if not needs:
        needs.append({
            "need_id": "N1",
            "need_type": "GENERAL",
            "label": "질문의 핵심 요청",
            "question_part": original,
        })
    return needs


def normalize_need_coverage_v2(
    raw_rows: Any,
    answer_needs: Sequence[Mapping[str, Any]],
    allowed_evidence_ids: set[str],
) -> list[dict[str, Any]]:
    rows = raw_rows if isinstance(raw_rows, list) else []
    by_id = {
        str(row.get("need_id") or ""): row
        for row in rows
        if isinstance(row, Mapping)
    }
    normalized: list[dict[str, Any]] = []
    for need in answer_needs:
        need_id = str(need["need_id"])
        row = by_id.get(need_id) or {}
        status = str(row.get("status") or "UNSUPPORTED").upper()
        evidence_ids = [
            value
            for value in answer_b_core._clean_list(row.get("evidence_ids"))
            if value in allowed_evidence_ids
        ]
        if status not in NEED_STATUS_VALUES_V2:
            status = "UNSUPPORTED"
        if status == "ANSWERED" and not evidence_ids:
            status = "PARTIAL"
        normalized.append({
            "need_id": need_id,
            "need_type": str(need.get("need_type") or "GENERAL"),
            "label": str(need.get("label") or need_id),
            "status": status,
            "evidence_ids": list(dict.fromkeys(evidence_ids)),
            "missing_reason": answer_b_core._clean(row.get("missing_reason")),
        })
    return normalized


def calculate_program_coverage_v2(
    need_coverage: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    statuses = [str(row.get("status") or "UNSUPPORTED") for row in need_coverage]
    total = len(statuses)
    answered = statuses.count("ANSWERED")
    partial = statuses.count("PARTIAL")
    unsupported = statuses.count("UNSUPPORTED")
    if total and answered == total:
        overall = "SUFFICIENT"
    elif answered or partial:
        overall = "PARTIAL"
    else:
        overall = "INSUFFICIENT"
    return {
        "coverage_status": overall,
        "need_count": total,
        "answered_need_count": answered,
        "partial_need_count": partial,
        "unsupported_need_count": unsupported,
        "strict_need_coverage_rate": answered / total if total else 0.0,
        "answerable_need_coverage_rate": (answered + partial) / total if total else 0.0,
    }


NUMERIC_FACT_PATTERN_V2 = re.compile(
    r"\d+(?:[.,]\d+)?\s*(?:%|퍼센트|억원|백만원|만원|원|년|개월|일|시간)",
    re.I,
)


def _normalize_numeric_fact_v2(value: str) -> str:
    return re.sub(r"[\s,]", "", str(value or "")).lower()


def audit_numeric_support_v2(
    answer: str,
    pack: Mapping[str, Any],
) -> dict[str, Any]:
    evidence_text = " ".join(str(row.get("content") or "") for row in pack.get("evidence") or [])
    evidence_numbers = {
        _normalize_numeric_fact_v2(value)
        for value in NUMERIC_FACT_PATTERN_V2.findall(evidence_text)
    }
    answer_numbers = list(dict.fromkeys(
        _normalize_numeric_fact_v2(value)
        for value in NUMERIC_FACT_PATTERN_V2.findall(str(answer or ""))
    ))
    unsupported = [value for value in answer_numbers if value not in evidence_numbers]
    return {
        "answer_numeric_facts": answer_numbers,
        "unsupported_numeric_facts": unsupported,
        "numeric_support_passed": not unsupported,
    }


answer_need_regression_v2 = pd.DataFrame([
    {
        "question": question,
        "types": ",".join(row["need_type"] for row in extract_answer_needs_v2(question)),
        "expected": expected,
    }
    for question, expected in (
        ("은닉재산 신고는 누가 할 수 있고, 포상금은 얼마나 받나요?", "ACTOR,AMOUNT"),
        ("채무조정 신청 자격과 필요서류를 알려주세요", "ELIGIBILITY,DOCUMENTS"),
        ("착오송금 반환 신청은 어떻게 하나요?", "PROCEDURE"),
        ("미수령금과 착오송금은 무엇이 다른가요?", "COMPARISON"),
    )
])
answer_need_regression_v2["passed"] = (
    answer_need_regression_v2["types"] == answer_need_regression_v2["expected"]
)
display(answer_need_regression_v2)
if not bool(answer_need_regression_v2["passed"].all()):
    raise AssertionError("Answer Need 규칙 회귀 테스트 실패")
print("Answer Need 규칙 회귀 테스트 통과")

# ==== cell 64 ====
B2_SYSTEM_PROMPT = """
당신은 예금보험공사 공식 문서 기반 답변 시스템입니다.

근거 규칙:
1. 사용자 질문, Answer Need, Evidence Pack에 포함된 사실만 사용하세요.
2. Evidence에 없는 사실이나 서로 다른 제도의 조건을 임의로 결합하지 마세요.
3. 동시·병행 가능성은 같은 Evidence가 그 관계를 직접 명시할 때만 단정하세요.

완전성 규칙:
4. 모든 need_id를 한 번씩 처리하세요.
5. ANSWERED에는 실제로 해당 요구를 뒷받침하는 evidence_ids가 있어야 합니다.
6. 일부만 확인되면 PARTIAL, 확인할 수 없으면 UNSUPPORTED로 표시하세요.
7. 금액 구간·최고 한도·선행 조건이 질문과 직접 관련되고 Evidence에 있으면 생략하지 마세요.
8. 질문하지 않은 배경·회수 이후 절차·중복 설명은 추가하지 마세요.

출력 규칙:
9. 문장 수는 제한하지 않습니다. 결론을 먼저 쓰고 질문 유형에 맞게 목록·표·소제목을 사용하세요.
10. 근거 문장 끝에는 [E1] 형식의 실제 Evidence ID를 표시하세요.
11. JSON 객체 하나만 출력하고 문자열 내부 줄바꿈은 \\n으로 이스케이프하세요.
""".strip()


def _validate_b2_payload_v2(
    raw: Mapping[str, Any],
    answer_needs: Sequence[Mapping[str, Any]],
    pack: Mapping[str, Any],
) -> dict[str, Any]:
    answer = answer_b_core._strip_model_urls(answer_b_core._clean(raw.get("answer")))
    if not answer:
        raise ValueError("B2 answer가 비어 있습니다.")
    allowed = set(answer_b_core._allowed_evidence(pack))
    need_coverage = normalize_need_coverage_v2(
        raw.get("need_coverage"), answer_needs, allowed
    )
    used_ids = list(dict.fromkeys(
        evidence_id
        for row in need_coverage
        for evidence_id in row["evidence_ids"]
    ))
    for number in re.findall(r"\[E(\d+)\]", answer):
        evidence_id = f"E{number}"
        if evidence_id in allowed and evidence_id not in used_ids:
            used_ids.append(evidence_id)
    if not used_ids:
        raise ValueError("B2 출력에 유효한 Evidence ID가 없습니다.")
    program = calculate_program_coverage_v2(need_coverage)
    numeric = audit_numeric_support_v2(answer, pack)
    return {
        "answer": answer,
        "answer_needs": list(answer_needs),
        "need_coverage": need_coverage,
        **program,
        "used_evidence_ids": used_ids,
        "used_chunk_ids": [answer_b_core._allowed_evidence(pack)[value] for value in used_ids],
        "missing_information": answer_b_core._clean_list(raw.get("missing_information")),
        "numeric_audit": numeric,
    }


def generate_answer_b2_v2(
    question: str,
    pack: Mapping[str, Any],
) -> dict[str, Any]:
    answer_needs = extract_answer_needs_v2(question)
    relation_constraint = relation_constraint_v1(question, pack)
    prompt = f"""[사용자 질문]\n{_clean_text(question)}\n\n[Answer Needs]\n{_compact_json(answer_needs)}\n\n[관계 주장 안전조건]\n{_compact_json(relation_constraint)}\n\n[Evidence Pack]\n{_compact_json(pack)}\n\n[출력 JSON]\n{{\"answer\":\"근거 문장에 [E1] 표시\",\"need_coverage\":[{{\"need_id\":\"N1\",\"status\":\"ANSWERED|PARTIAL|UNSUPPORTED\",\"evidence_ids\":[\"E1\"],\"missing_reason\":\"\"}}],\"missing_information\":[]}}"""
    raw, usage, first_ms, first_trace = _call_answer_api_v1(
        system_prompt=B2_SYSTEM_PROMPT,
        user_prompt=prompt,
        max_tokens=1600,
    )
    attempts = [{"stage": "initial", "latency_ms": first_ms, "trace": first_trace}]
    total_usage = dict(usage)
    total_ms = first_ms
    try:
        payload = _validate_b2_payload_v2(
            answer_b_core._extract_json_object(raw), answer_needs, pack
        )
    except (ValueError, TypeError):
        try:
            recovered = answer_b_core._recover_basic_answer_from_raw(raw, pack)
            allowed_ids = list(recovered["used_evidence_ids"])
            fallback_rows = [
                {
                    "need_id": need["need_id"],
                    "status": "PARTIAL",
                    "evidence_ids": allowed_ids,
                    "missing_reason": "구조화 Need 메타데이터를 로컬 복구함",
                }
                for need in answer_needs
            ]
            payload = _validate_b2_payload_v2({
                "answer": recovered["answer"],
                "need_coverage": fallback_rows,
                "missing_information": ["Need별 구조화 메타데이터 로컬 복구"],
            }, answer_needs, pack)
            attempts.append({"stage": "local_raw_recovery", "latency_ms": 0.0})
        except (ValueError, TypeError):
            repair_prompt = f"""직전 출력을 사실 변경 없이 요청한 JSON 객체로만 교정하세요. 모든 need_id를 보존하세요.\n\n[원래 요청]\n{prompt}\n\n[직전 출력]\n{raw[:6000]}"""
            repaired, repair_usage, repair_ms, repair_trace = _call_answer_api_v1(
                system_prompt=B2_SYSTEM_PROMPT,
                user_prompt=repair_prompt,
                max_tokens=1600,
            )
            total_usage = _merge_usage_v1(total_usage, repair_usage)
            total_ms += repair_ms
            attempts.append({"stage": "repair", "latency_ms": repair_ms, "trace": repair_trace})
            payload = _validate_b2_payload_v2(
                answer_b_core._extract_json_object(repaired), answer_needs, pack
            )

    safe_answer, guard_applied = _relation_safe_answer_v1(
        payload["answer"], relation_constraint
    )
    payload["answer"] = safe_answer
    if guard_applied:
        payload["coverage_status"] = "PARTIAL"
    return {
        **payload,
        "system": "B2",
        "latency_ms": total_ms,
        "usage": total_usage,
        "api_calls": sum(1 for row in attempts if row["stage"] in {"initial", "repair"}),
        "attempts": attempts,
        "relation_constraint": relation_constraint,
        "relation_guard_applied": guard_applied,
    }


print("B2 Answer Need + 프로그램 Coverage 생성기 준비 완료")

# ==== cell 66 ====
D2_SKELETON_SYSTEM_PROMPT = """
당신은 예금보험공사 공식 문서에서 답변에 필요한 사실 구조를 추출하는 분석기입니다.

1. 사용자 질문, Answer Need, Evidence Pack에 명시된 사실만 사용하세요.
2. 모든 need_id에 정확히 하나의 answer_item을 작성하세요.
3. Evidence로 충분히 답하면 ANSWERED, 일부만 확인되면 PARTIAL, 없으면 UNSUPPORTED로 표시하세요.
4. ANSWERED와 PARTIAL에는 실제 evidence_ids를 연결하세요.
5. 신청 주체 질문에서도 선행 자격 조건이 Evidence에 있으면 conditions에 보존하세요.
6. 금액 질문에서는 구간, 최고 한도, 산정 기준을 Evidence 범위에서 details에 보존하세요.
7. 서로 다른 제도의 조건을 결합하지 말고, 직접 관계 근거가 없는 동시 신청은 uncertainties에 기록하세요.
8. 최종 사용자 문장이 아니라 JSON Skeleton 하나만 출력하세요.
""".strip()

D2_FINAL_SYSTEM_PROMPT = """
당신은 예금보험공사 공식 문서 기반 질의응답 시스템입니다.

1. Answer Needs, Answer Skeleton, 선택 Evidence에 있는 사실만 사용하세요.
2. 모든 answer_item을 최종답변에 반영하세요.
3. ANSWERED의 claim·conditions·details에서 질문 판단에 필요한 내용을 생략하지 마세요.
4. PARTIAL과 UNSUPPORTED는 확인 가능한 범위와 부족한 정보를 구분하세요.
5. core_answer 한 문장만 복사하고 종료하지 마세요.
6. 문장 수는 제한하지 않되 질문하지 않은 배경과 반복은 추가하지 마세요.
7. 근거 문장에는 Skeleton이 허용한 [E1] 형식의 Evidence ID만 표시하세요.
8. Skeleton, JSON, 내부 구현, 검색 점수는 답변에 언급하지 마세요.
""".strip()


def validate_d2_skeleton_v2(
    raw: Mapping[str, Any],
    answer_needs: Sequence[Mapping[str, Any]],
    pack: Mapping[str, Any],
) -> dict[str, Any]:
    allowed = set(answer_b_core._allowed_evidence(pack))
    raw_items = raw.get("answer_items") if isinstance(raw.get("answer_items"), list) else []
    by_need = {
        str(item.get("need_id") or ""): item
        for item in raw_items
        if isinstance(item, Mapping)
    }
    items: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    for need in answer_needs:
        need_id = str(need["need_id"])
        source = by_need.get(need_id) or {}
        status = str(source.get("status") or "UNSUPPORTED").upper()
        if status not in NEED_STATUS_VALUES_V2:
            status = "UNSUPPORTED"
        evidence_ids = [
            value
            for value in answer_b_core._clean_list(source.get("evidence_ids"))
            if value in allowed
        ]
        if status == "ANSWERED" and not evidence_ids:
            status = "PARTIAL" if answer_b_core._clean(source.get("claim")) else "UNSUPPORTED"
        claim = answer_b_core._clean(source.get("claim"))
        if status == "UNSUPPORTED" and not claim:
            claim = f"{need['label']}은 현재 Evidence로 확인되지 않습니다."
        items.append({
            "need_id": need_id,
            "need_type": need["need_type"],
            "topic": answer_b_core._clean(source.get("topic")) or need["label"],
            "status": status,
            "claim": claim,
            "conditions": answer_b_core._clean_list(source.get("conditions")),
            "details": answer_b_core._clean_list(source.get("details")),
            "evidence_ids": list(dict.fromkeys(evidence_ids)),
            "missing_reason": answer_b_core._clean(source.get("missing_reason")),
        })
        coverage_rows.append({
            "need_id": need_id,
            "need_type": need["need_type"],
            "label": need["label"],
            "status": status,
            "evidence_ids": list(dict.fromkeys(evidence_ids)),
            "missing_reason": answer_b_core._clean(source.get("missing_reason")),
        })
    program = calculate_program_coverage_v2(coverage_rows)
    return {
        "core_answer": answer_b_core._clean(raw.get("core_answer")),
        "answer_items": items,
        "need_coverage": coverage_rows,
        "uncertainties": answer_b_core._clean_list(raw.get("uncertainties")),
        "conflicts": answer_b_core._clean_list(raw.get("conflicts")),
        **program,
    }


def filter_pack_for_d2_v2(
    pack: Mapping[str, Any],
    skeleton: Mapping[str, Any],
) -> dict[str, Any]:
    used_ids = {
        evidence_id
        for item in skeleton.get("answer_items") or []
        for evidence_id in item.get("evidence_ids") or []
    }
    evidence = [
        dict(row)
        for row in pack.get("evidence") or []
        if row.get("evidence_id") in used_ids
    ]
    source_urls = {str(row.get("source_url") or "") for row in evidence}
    return {
        **dict(pack),
        "evidence": evidence,
        "sources": [
            dict(row)
            for row in pack.get("sources") or []
            if str(row.get("source_url") or "") in source_urls
        ],
        "filtered_for_d2": True,
        "original_evidence_count": len(pack.get("evidence") or []),
        "selected_evidence_count": len(evidence),
    }


def generate_answer_d2_v2(
    question: str,
    pack: Mapping[str, Any],
) -> dict[str, Any]:
    answer_needs = extract_answer_needs_v2(question)
    relation_constraint = relation_constraint_v1(question, pack)
    skeleton_prompt = f"""[사용자 질문]\n{_clean_text(question)}\n\n[Answer Needs]\n{_compact_json(answer_needs)}\n\n[관계 주장 안전조건]\n{_compact_json(relation_constraint)}\n\n[Evidence Pack]\n{_compact_json(pack)}\n\n[출력 JSON]\n{{\"core_answer\":\"핵심 결론\",\"answer_items\":[{{\"need_id\":\"N1\",\"topic\":\"항목\",\"status\":\"ANSWERED|PARTIAL|UNSUPPORTED\",\"claim\":\"근거 사실\",\"conditions\":[],\"details\":[],\"evidence_ids\":[\"E1\"],\"missing_reason\":\"\"}}],\"uncertainties\":[],\"conflicts\":[]}}"""
    raw_skeleton, usage1, skeleton_ms, trace1 = _call_answer_api_v1(
        system_prompt=D2_SKELETON_SYSTEM_PROMPT,
        user_prompt=skeleton_prompt,
        max_tokens=2000,
    )
    skeleton = validate_d2_skeleton_v2(
        answer_b_core._extract_json_object(raw_skeleton), answer_needs, pack
    )
    selected_pack = filter_pack_for_d2_v2(pack, skeleton)
    final_prompt = f"""[사용자 질문]\n{_clean_text(question)}\n\n[Answer Needs]\n{_compact_json(answer_needs)}\n\n[관계 주장 안전조건]\n{_compact_json(relation_constraint)}\n\n[Answer Skeleton]\n{_compact_json(skeleton)}\n\n[Skeleton 참조 Evidence]\n{_compact_json(selected_pack)}\n\n위 범위에서 최종 사용자 답변을 작성하세요."""
    raw_answer, usage2, final_ms, trace2 = _call_answer_api_v1(
        system_prompt=D2_FINAL_SYSTEM_PROMPT,
        user_prompt=final_prompt,
        max_tokens=1600,
    )
    answer = answer_b_core._strip_model_urls(str(raw_answer).strip())
    if not answer:
        raise ValueError("D2 최종 답변이 비어 있습니다.")
    safe_answer, guard_applied = _relation_safe_answer_v1(answer, relation_constraint)
    numeric = audit_numeric_support_v2(safe_answer, selected_pack)
    return {
        "system": "D2",
        "answer": safe_answer,
        "answer_needs": answer_needs,
        "skeleton": skeleton,
        "need_coverage": skeleton["need_coverage"],
        "coverage_status": "PARTIAL" if guard_applied else skeleton["coverage_status"],
        "strict_need_coverage_rate": skeleton["strict_need_coverage_rate"],
        "answerable_need_coverage_rate": skeleton["answerable_need_coverage_rate"],
        "numeric_audit": numeric,
        "selected_evidence_pack": selected_pack,
        "selected_evidence_count": len(selected_pack["evidence"]),
        "latency_ms": skeleton_ms + final_ms,
        "skeleton_latency_ms": skeleton_ms,
        "final_latency_ms": final_ms,
        "usage": _merge_usage_v1(usage1, usage2),
        "api_calls": 2,
        "attempts": [
            {"stage": "skeleton", "latency_ms": skeleton_ms, "trace": trace1},
            {"stage": "final", "latency_ms": final_ms, "trace": trace2},
        ],
        "relation_constraint": relation_constraint,
        "relation_guard_applied": guard_applied,
    }


print("D2 Need Skeleton + 선택 Evidence 생성기 준비 완료")

# ==== cell 68 ====

import copy
import hashlib
import random
import re
import threading
import time
from collections import OrderedDict
from types import SimpleNamespace
from typing import Any, Callable, Mapping


HCX_GLOBAL_MIN_INTERVAL_SECONDS_V3 = 3.0
HCX_QUERY_EMBEDDING_MIN_INTERVAL_SECONDS_V4_1 = 2.0
HCX_GLOBAL_MAX_ATTEMPTS_V3 = 6
HCX_GLOBAL_BACKOFF_BASE_SECONDS_V3 = 2.0
HCX_GLOBAL_BACKOFF_MAX_SECONDS_V3 = 60.0
QUERY_EMBEDDING_CACHE_ENABLED_V3 = True
QUERY_EMBEDDING_CACHE_MAX_SIZE_V3 = 512


def _rate_limit_headers_v3(error: Exception) -> dict[str, str]:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None) or {}
    return {
        str(key): str(value)
        for key, value in dict(headers).items()
        if any(marker in str(key).lower() for marker in (
            "rate", "limit", "remaining", "reset", "retry"
        ))
    }


def _header_delay_seconds_v3(error: Exception) -> float | None:
    headers = _rate_limit_headers_v3(error)
    values: list[float] = []
    for key, value in headers.items():
        if key.lower() not in {
            "retry-after",
            "x-ratelimit-reset-requests",
            "x-ratelimit-reset-tokens",
        }:
            continue
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)", value)
        if match:
            values.append(float(match.group(1)))
    return min(120.0, max(values)) if values else None


class HCXSharedRequestGateV3:
    def __init__(
        self,
        *,
        min_interval_seconds: float,
        max_attempts: int,
    ) -> None:
        self.min_interval_seconds = float(min_interval_seconds)
        self.max_attempts = int(max_attempts)
        self.lock = threading.RLock()
        self.last_call_started_at = 0.0
        self.last_trace: dict[str, Any] = {}
        self.history: list[dict[str, Any]] = []

    def _backoff(self, error: Exception, attempt: int) -> float:
        header_delay = _header_delay_seconds_v3(error)
        if header_delay is not None:
            return min(120.0, max(1.0, header_delay) + random.uniform(0.1, 0.8))
        exponential = HCX_GLOBAL_BACKOFF_BASE_SECONDS_V3 * (2 ** max(0, attempt - 1))
        return min(HCX_GLOBAL_BACKOFF_MAX_SECONDS_V3, exponential) + random.uniform(0.1, 0.8)

    def call(
        self,
        stage: str,
        operation: Callable[[], Any],
        *,
        max_attempts: int | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        attempts_limit = int(max_attempts or self.max_attempts)
        started = time.perf_counter()
        attempts: list[dict[str, Any]] = []
        pacing_wait_seconds = 0.0
        retry_wait_seconds = 0.0
        with self.lock:
            for attempt in range(1, attempts_limit + 1):
                elapsed = time.perf_counter() - self.last_call_started_at
                stage_min_interval_seconds = (
                    HCX_QUERY_EMBEDDING_MIN_INTERVAL_SECONDS_V4_1
                    if stage == "query_embedding"
                    else self.min_interval_seconds
                )
                pacing = max(0.0, stage_min_interval_seconds - elapsed)
                if pacing:
                    time.sleep(pacing)
                    pacing_wait_seconds += pacing
                self.last_call_started_at = time.perf_counter()
                try:
                    result = operation()
                except Exception as error:
                    if type(error).__name__ != "RateLimitError":
                        trace = {
                            "stage": stage,
                            "success": False,
                            "attempts": attempts + [{
                                "attempt": attempt,
                                "status": type(error).__name__,
                                "message": str(error)[:1000],
                            }],
                            "pacing_wait_ms": pacing_wait_seconds * 1000,
                            "retry_wait_ms": retry_wait_seconds * 1000,
                            "wall_latency_ms": (time.perf_counter() - started) * 1000,
                        }
                        self.last_trace = trace
                        self.history.append(copy.deepcopy(trace))
                        raise
                    delay = self._backoff(error, attempt)
                    attempts.append({
                        "attempt": attempt,
                        "status": "RATE_LIMIT_429",
                        "delay_seconds": delay,
                        "headers": _rate_limit_headers_v3(error),
                    })
                    if attempt >= attempts_limit:
                        trace = {
                            "stage": stage,
                            "success": False,
                            "attempts": attempts,
                            "pacing_wait_ms": pacing_wait_seconds * 1000,
                            "retry_wait_ms": retry_wait_seconds * 1000,
                            "wall_latency_ms": (time.perf_counter() - started) * 1000,
                        }
                        self.last_trace = trace
                        self.history.append(copy.deepcopy(trace))
                        raise RuntimeError(
                            f"{stage} 단계에서 429 재시도 {attempts_limit}회를 모두 소진했습니다. "
                            "Trace의 Retry-After와 Rate Limit 헤더를 확인하세요."
                        ) from error
                    time.sleep(delay)
                    retry_wait_seconds += delay
                    continue
                attempts.append({"attempt": attempt, "status": "SUCCESS"})
                trace = {
                    "stage": stage,
                    "success": True,
                    "attempts": attempts,
                    "pacing_wait_ms": pacing_wait_seconds * 1000,
                    "retry_wait_ms": retry_wait_seconds * 1000,
                    "wall_latency_ms": (time.perf_counter() - started) * 1000,
                }
                self.last_trace = trace
                self.history.append(copy.deepcopy(trace))
                return result, trace
        raise AssertionError("도달할 수 없는 HCX 공통 게이트 상태")


HCX_SHARED_GATE_V3 = HCXSharedRequestGateV3(
    min_interval_seconds=HCX_GLOBAL_MIN_INTERVAL_SECONDS_V3,
    max_attempts=HCX_GLOBAL_MAX_ATTEMPTS_V3,
)

_HCX_RAW_CLIENT_V3 = (
    HCX_CLIENT.with_options(max_retries=0)
    if hasattr(HCX_CLIENT, "with_options")
    else HCX_CLIENT
)

QUERY_EMBEDDING_CACHE_V3: OrderedDict[str, np.ndarray] = OrderedDict()
_LAST_QUERY_EMBEDDING_TRACE_V3: dict[str, Any] = {}


def _query_embedding_cache_key_v3(text: str) -> str:
    raw = f"{HCX_EMBEDDING_MODEL}\n{HCX_ENCODING_FORMAT}\n{_clean_text(text)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def clear_query_embedding_cache_v3() -> None:
    QUERY_EMBEDDING_CACHE_V3.clear()


def embed_hcx_single(text: str) -> np.ndarray:
    global _LAST_QUERY_EMBEDDING_TRACE_V3
    cleaned = _clean_text(text)
    if not cleaned:
        raise ValueError("임베딩 입력이 비어 있습니다.")
    cache_key = _query_embedding_cache_key_v3(cleaned)
    if QUERY_EMBEDDING_CACHE_ENABLED_V3 and cache_key in QUERY_EMBEDDING_CACHE_V3:
        vector = QUERY_EMBEDDING_CACHE_V3.pop(cache_key)
        QUERY_EMBEDDING_CACHE_V3[cache_key] = vector
        _LAST_QUERY_EMBEDDING_TRACE_V3 = {
            "stage": "query_embedding",
            "success": True,
            "cache_hit": True,
            "attempts": [],
            "pacing_wait_ms": 0.0,
            "retry_wait_ms": 0.0,
            "wall_latency_ms": 0.0,
        }
        return vector.copy()

    def operation() -> Any:
        return _HCX_RAW_CLIENT_V3.embeddings.create(
            model=HCX_EMBEDDING_MODEL,
            input=cleaned,
            encoding_format=HCX_ENCODING_FORMAT,
        )

    response, trace = HCX_SHARED_GATE_V3.call("query_embedding", operation)
    if len(response.data) != 1:
        raise RuntimeError(f"단일 임베딩 응답 개수가 1이 아닙니다: {len(response.data)}")
    vector = np.asarray(response.data[0].embedding, dtype=np.float32)
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise RuntimeError(f"잘못된 질문 임베딩 벡터입니다: shape={vector.shape}")
    _LAST_QUERY_EMBEDDING_TRACE_V3 = {**trace, "cache_hit": False}
    if QUERY_EMBEDDING_CACHE_ENABLED_V3:
        QUERY_EMBEDDING_CACHE_V3[cache_key] = vector.copy()
        while len(QUERY_EMBEDDING_CACHE_V3) > QUERY_EMBEDDING_CACHE_MAX_SIZE_V3:
            QUERY_EMBEDDING_CACHE_V3.popitem(last=False)
    return vector


class _SharedGateAnswerCompletionsV3:
    def __init__(self, raw_completions: Any) -> None:
        self.raw_completions = raw_completions

    def create(self, **kwargs: Any) -> Any:
        global _LAST_ANSWER_API_TRACE
        response, trace = HCX_SHARED_GATE_V3.call(
            "answer_generation",
            lambda: self.raw_completions.create(**kwargs),
        )
        _LAST_ANSWER_API_TRACE = {
            "attempts": trace.get("attempts") or [],
            "total_wait_ms": float(trace.get("pacing_wait_ms") or 0)
            + float(trace.get("retry_wait_ms") or 0),
            "pacing_wait_ms": trace.get("pacing_wait_ms"),
            "retry_wait_ms": trace.get("retry_wait_ms"),
            "wall_latency_ms": trace.get("wall_latency_ms"),
            "stage": trace.get("stage"),
        }
        return response


class _SharedGateAnswerClientV3:
    def __init__(self, raw_client: Any) -> None:
        self.chat = SimpleNamespace(
            completions=_SharedGateAnswerCompletionsV3(raw_client.chat.completions)
        )
        self._kdic_structured_output_capability = {HCX_CHAT_MODEL: False}


ANSWER_HCX_CLIENT = _SharedGateAnswerClientV3(_HCX_RAW_CLIENT_V3)


def hcx_gate_diagnostics_v3() -> pd.DataFrame:
    rows = []
    for index, trace in enumerate(HCX_SHARED_GATE_V3.history, start=1):
        attempts = trace.get("attempts") or []
        rows.append({
            "call": index,
            "stage": trace.get("stage"),
            "success": bool(trace.get("success")),
            "attempt_count": len(attempts),
            "rate_limit_429_count": sum(
                1 for row in attempts if row.get("status") == "RATE_LIMIT_429"
            ),
            "pacing_wait_ms": float(trace.get("pacing_wait_ms") or 0),
            "retry_wait_ms": float(trace.get("retry_wait_ms") or 0),
            "wall_latency_ms": float(trace.get("wall_latency_ms") or 0),
        })
    return pd.DataFrame(rows)


print({
    "global_hcx_min_interval_seconds": HCX_GLOBAL_MIN_INTERVAL_SECONDS_V3,
    "query_embedding_min_interval_seconds": HCX_QUERY_EMBEDDING_MIN_INTERVAL_SECONDS_V4_1,
    "global_hcx_max_attempts": HCX_GLOBAL_MAX_ATTEMPTS_V3,
    "query_embedding_cache": QUERY_EMBEDDING_CACHE_ENABLED_V3,
    "query_embedding_cache_max_size": QUERY_EMBEDDING_CACHE_MAX_SIZE_V3,
})

# ==== cell 70 ====

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


FACT_INDEX_UPLOAD_NAME_C1 = "KDIC_Fact_Index_C_Ver3_Reviewable.json"
FACT_INDEX_EXPECTED_SCHEMA_C1 = "kdic-fact-index-c-v3-reviewable"
FACT_INDEX_ALLOWED_REVIEW_STATUS_C1 = {
    "HUMAN_APPROVED_EVAL",
    "CANDIDATE_EVIDENCE_REVIEWED",
}


def locate_or_upload_fact_index_c1() -> Path:
    candidates = [
        Path("/content") / FACT_INDEX_UPLOAD_NAME_C1,
        Path.cwd() / FACT_INDEX_UPLOAD_NAME_C1,
    ]
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate

    try:
        from google.colab import files
    except ImportError as error:
        raise FileNotFoundError(
            f"{FACT_INDEX_UPLOAD_NAME_C1} 파일을 현재 작업 폴더에 두세요."
        ) from error

    print(f"Fact Index 파일을 업로드하세요: {FACT_INDEX_UPLOAD_NAME_C1}")
    uploaded = files.upload()
    if FACT_INDEX_UPLOAD_NAME_C1 not in uploaded:
        raise FileNotFoundError(
            f"업로드 파일명이 다릅니다. 필요한 파일: {FACT_INDEX_UPLOAD_NAME_C1}"
        )
    path = Path("/content") / FACT_INDEX_UPLOAD_NAME_C1
    if not path.is_file():
        path = Path(FACT_INDEX_UPLOAD_NAME_C1)
    return path


def load_fact_index_c1(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    document = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(document, dict):
        raise TypeError("Fact Index 최상위 값은 JSON 객체여야 합니다.")
    if document.get("schema_version") != FACT_INDEX_EXPECTED_SCHEMA_C1:
        raise ValueError(
            "Fact Index schema_version 불일치: "
            f"{document.get('schema_version')}"
        )
    records = document.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Fact Index records가 비어 있습니다.")

    trigger_owner: dict[str, str] = {}
    errors: list[str] = []
    for record in records:
        fact_id = str(record.get("fact_index_id") or "")
        triggers = [str(value) for value in record.get("trigger_chunk_ids") or []]
        if not fact_id or not triggers:
            errors.append(f"식별자 또는 trigger 누락: {fact_id or '<empty>'}")
            continue
        if record.get("activation_policy") != "TRIGGER_CHUNK_AND_BUSINESS_AND_KEYWORD":
            errors.append(f"지원하지 않는 activation_policy: {fact_id}")
        if record.get("review_status") not in FACT_INDEX_ALLOWED_REVIEW_STATUS_C1:
            errors.append(f"평가 허용 review_status 아님: {fact_id}")
        for chunk_id in triggers:
            previous = trigger_owner.get(chunk_id)
            if previous and previous != fact_id:
                errors.append(f"trigger 중복: {chunk_id} -> {previous}, {fact_id}")
            trigger_owner[chunk_id] = fact_id
    if errors:
        raise ValueError("Fact Index 검증 실패: " + " | ".join(errors[:10]))

    return {
        **document,
        "_path": str(path),
        "_sha256": hashlib.sha256(raw).hexdigest(),
        "_record_count": len(records),
    }


FACT_INDEX_PATH_C1 = locate_or_upload_fact_index_c1()
FACT_INDEX_DOCUMENT_C1 = load_fact_index_c1(FACT_INDEX_PATH_C1)
FACT_INDEX_RECORDS_C1 = tuple(FACT_INDEX_DOCUMENT_C1["records"])

print({
    "fact_index_path": str(FACT_INDEX_PATH_C1),
    "schema_version": FACT_INDEX_DOCUMENT_C1["schema_version"],
    "status": FACT_INDEX_DOCUMENT_C1.get("status"),
    "prototype_use_allowed": FACT_INDEX_DOCUMENT_C1.get("prototype_use_allowed"),
    "record_count": len(FACT_INDEX_RECORDS_C1),
    "sha256": FACT_INDEX_DOCUMENT_C1["_sha256"],
})

# ==== cell 72 ====
BUSINESS_CODE_BY_LABEL_C1 = {
    "예금자보호제도": "deposit_protection",
    "예금자보호": "deposit_protection",
    "예금보험금 안내": "deposit_insurance_payout",
    "예금보험금": "deposit_insurance_payout",
    "고객 미수령금 신청": "unclaimed_funds",
    "고객 미수령금": "unclaimed_funds",
    "착오송금 반환 신청": "mistaken_transfer",
    "착오송금 반환지원": "mistaken_transfer",
    "채무조정 안내": "debt_adjustment",
    "채무조정": "debt_adjustment",
    "은닉재산 신고": "hidden_assets_report",
}
FACT_PRIORITY_ORDER_C1 = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


def _detected_business_codes_c1(common: Mapping[str, Any]) -> set[str]:
    analysis = common.get("analysis") or {}
    values = list(analysis.get("businesses") or [])
    values.extend(analysis.get("detected_businesses") or [])
    output: set[str] = set()
    for value in values:
        label = str(value).strip()
        code = BUSINESS_CODE_BY_LABEL_C1.get(label)
        if code:
            output.add(code)
    return output


def match_fact_index_c1(
    common: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if common.get("route") != "RETRIEVE":
        return [], []

    top5_ids = {
        str(row.get("chunk_id") or (row.get("chunk") or {}).get("chunk_id") or "")
        for row in list(common.get("search_results") or [])[:5]
    }
    top5_ids.discard("")
    detected_codes = _detected_business_codes_c1(common)
    question = _clean_text(
        " ".join([
            str(common.get("question") or ""),
            str(common.get("resolved_question") or ""),
        ])
    ).lower()

    matched: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for record in FACT_INDEX_RECORDS_C1:
        triggers = {str(value) for value in record.get("trigger_chunk_ids") or []}
        trigger_hits = sorted(top5_ids.intersection(triggers))
        record_code = str(record.get("business_function_code") or "")
        business_match = bool(record_code and record_code in detected_codes)
        keyword_hits = [
            str(keyword)
            for keyword in record.get("activation_keywords") or []
            if str(keyword).strip() and str(keyword).strip().lower() in question
        ]
        accepted = bool(trigger_hits and business_match and keyword_hits)
        row = {
            "fact_index_id": str(record.get("fact_index_id") or ""),
            "accepted": accepted,
            "trigger_hits": trigger_hits,
            "business_function": record.get("business_function"),
            "business_function_code": record_code,
            "business_match": business_match,
            "keyword_hits": list(dict.fromkeys(keyword_hits)),
            "confusion_type": record.get("confusion_type"),
            "priority": record.get("priority"),
            "review_status": record.get("review_status"),
            "reason": (
                "TRIGGER_BUSINESS_KEYWORD_MATCH"
                if accepted
                else "NO_TRIGGER" if not trigger_hits
                else "BUSINESS_MISMATCH" if not business_match
                else "NO_ACTIVATION_KEYWORD"
            ),
        }
        audit.append(row)
        if accepted:
            matched.append({**copy.deepcopy(record), "_match": row})

    matched.sort(key=lambda record: (
        FACT_PRIORITY_ORDER_C1.get(str(record.get("priority") or ""), 9),
        str(record.get("fact_index_id") or ""),
    ))
    return matched, audit


def _claim_source_ids_c1(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return [item for item in str(value or "").split() if item]


def build_fact_augmented_pack_c1(
    base_pack: Mapping[str, Any],
    matched_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    supplements: list[dict[str, Any]] = []
    for record in matched_records:
        verified_claims = []
        for claim in record.get("verified_claims") or []:
            verified_claims.append({
                "claim_id": str(claim.get("claim_id") or ""),
                "statement": _clean_text(claim.get("statement")),
                "source_chunk_ids": _claim_source_ids_c1(claim.get("source_chunk_ids")),
                "origin": str(claim.get("origin") or ""),
            })
        supplements.append({
            "fact_index_id": str(record.get("fact_index_id") or ""),
            "review_status": str(record.get("review_status") or ""),
            "business_function": str(record.get("business_function") or ""),
            "business_function_code": str(record.get("business_function_code") or ""),
            "priority": str(record.get("priority") or ""),
            "confusion_type": _clean_text(record.get("confusion_type")),
            "confusion_point": _clean_text(record.get("confusion_point")),
            "verified_claims": verified_claims,
            "forbidden_claims": [
                _clean_text(value) for value in record.get("forbidden_claims") or []
                if _clean_text(value)
            ],
            "related_chunk_ids": [
                str(value) for value in record.get("related_chunk_ids") or [] if str(value)
            ],
            "source_chunk_ids": [
                str(value) for value in record.get("source_chunk_ids") or [] if str(value)
            ],
            "source_urls": [
                str(value) for value in record.get("source_urls") or [] if str(value)
            ],
            "activation_audit": copy.deepcopy(record.get("_match") or {}),
        })

    return {
        **copy.deepcopy(dict(base_pack)),
        "fact_index": {
            "schema_version": FACT_INDEX_DOCUMENT_C1["schema_version"],
            "source_sha256": FACT_INDEX_DOCUMENT_C1["_sha256"],
            "matching_policy": FACT_INDEX_DOCUMENT_C1.get("matching_policy"),
            "review_required": FACT_INDEX_DOCUMENT_C1.get("status") == "REVIEW_REQUIRED",
            "supplement_count": len(supplements),
            "supplements": supplements,
            "precedence_rule": (
                "원본 Evidence를 보존한다. Fact Index는 검증 보조정보이며 원본을 덮어쓰지 않는다. "
                "충돌이 의심되면 단정하지 않고 검토 필요로 표시한다."
            ),
        },
    }


print({
    "matching_policy": "trigger_chunk_ids AND business_function_code AND activation_keywords",
    "semantic_similarity_matching": False,
    "fact_record_count": len(FACT_INDEX_RECORDS_C1),
})

# ==== cell 74 ====

import json
import re
import time
from typing import Any, Mapping


MARKDOWN_FORMAT_RULES_V3 = """
[답변 표시 형식]
- 먼저 질문에 대한 결론을 한 문단으로 작성하세요.
- 절차는 각 단계가 한 줄에 하나씩 보이도록 Markdown 번호 목록을 사용하세요.
- 서류·조건·예외·대상은 각 항목이 한 줄에 하나씩 보이도록 Markdown 글머리표를 사용하세요.
- 목록 앞뒤에는 빈 줄을 넣으세요.
- `1. 내용 2. 내용 3. 내용`처럼 번호를 한 문단에 이어 쓰지 마세요.
- 짧은 단답형 질문에는 불필요한 제목이나 목록을 만들지 마세요.
- JSON answer 문자열 내부의 줄바꿈은 \\n으로 이스케이프하세요.
""".strip()

B_STRUCTURED_SYSTEM_PROMPT_V3 = (
    B_LOW_LATENCY_SYSTEM_PROMPT + "\n\n" + MARKDOWN_FORMAT_RULES_V3
)
C_BASE_SYSTEM_PROMPT_V3 = """
당신은 예금보험공사 공식 문서 기반 답변 시스템입니다.

1. 사용자 질문, 원본 Evidence, 활성화된 Fact Index 보강정보만 사용하세요.
2. 원본 Evidence를 기본 근거로 사용하고 Fact Index는 혼동 방지와 사실 검증에 사용하세요.
3. verified_claims는 해당 Fact Index의 source_chunk_ids와 source_urls 범위에서 검수된 주장입니다.
4. forbidden_claims에 해당하는 내용은 답변에서 주장하지 마세요.
5. Fact Index와 원본 Evidence가 충돌하면 어느 쪽도 임의로 선택하지 말고 확인이 필요하다고 답하세요.
6. Fact Index가 연결되지 않은 질문은 B안과 같은 원본 Evidence 범위에서 답하세요.
7. 서로 다른 대상·제도·금액·기간·조건을 임의로 결합하지 마세요.
8. 동시·병행 신청 관계는 동일 근거가 직접 명시한 경우에만 가능하다고 답하세요.
9. 근거 문장에는 원본 Evidence를 [E1], Fact claim을 [FI-CAND-001:F1] 형식으로 표시하세요.
10. 지정된 JSON 객체 하나만 출력하세요.
""".strip()
C_STRUCTURED_SYSTEM_PROMPT_V3 = (
    C_BASE_SYSTEM_PROMPT_V3 + "\n\n" + MARKDOWN_FORMAT_RULES_V3
)
D_STRUCTURED_FINAL_SYSTEM_PROMPT_V3 = (
    D2_FINAL_SYSTEM_PROMPT + "\n\n" + MARKDOWN_FORMAT_RULES_V3
)


def normalize_answer_markdown_v3(text: Any) -> str:
    value = user_visible_answer(text).replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"\s*\[FI-CAND-\d+:F\d+\]", "", value)
    value = re.sub(r"\(\s*\)", "", value)
    numbered_markers = re.findall(r"(?<!\d)(?:[1-9]|1[0-9])\.\s+\S", value)
    if len(numbered_markers) >= 2:
        value = re.sub(
            r"\s+(?=(?:[1-9]|1[0-9])\.\s+\S)",
            "\n",
            value,
        )
        value = re.sub(r"([^\n])\n(?=1\.\s+)", r"\1\n\n", value)
    value = re.sub(r"[ \t]+\n", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    value = re.sub(r"\s+([,.!?])", r"\1", value)
    return value.strip()


def _trace_parts_v3(trace: Mapping[str, Any]) -> dict[str, float]:
    wall = float(trace.get("wall_latency_ms") or 0)
    pacing = float(trace.get("pacing_wait_ms") or 0)
    retry = float(trace.get("retry_wait_ms") or 0)
    return {
        "api_wall_ms": wall,
        "pacing_wait_ms": pacing,
        "retry_wait_ms": retry,
        "estimated_service_ms": max(0.0, wall - pacing - retry),
    }


def _allowed_fact_claims_v3(pack: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for supplement in (pack.get("fact_index") or {}).get("supplements") or []:
        fact_id = str(supplement.get("fact_index_id") or "")
        for claim in supplement.get("verified_claims") or []:
            claim_id = str(claim.get("claim_id") or "")
            if fact_id and claim_id:
                output[f"{fact_id}:{claim_id}"] = dict(claim)
    return output


def _clean_fact_claim_keys_v3(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(
        str(item).strip().strip("[]") for item in value if str(item).strip()
    ))


def _direct_answer_payload_v3(
    raw: str,
    pack: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    allowed = answer_b_core._allowed_evidence(pack)
    local_recovery = False
    try:
        parsed = answer_b_core._extract_json_object(raw)
        answer = answer_b_core._strip_model_urls(answer_b_core._clean(parsed.get("answer")))
        if not answer:
            raise ValueError("answer가 비어 있습니다.")
        used_ids = [
            value for value in answer_b_core._clean_list(parsed.get("used_evidence_ids"))
            if value in allowed
        ]
        if not used_ids:
            used_ids = [
                f"E{number}" for number in re.findall(r"\[E(\d+)\]", answer)
                if f"E{number}" in allowed
            ]
        coverage = str(parsed.get("coverage_status") or "PARTIAL").upper()
        if coverage not in answer_b_core.ALLOWED_COVERAGE_STATUS:
            coverage = "PARTIAL"
        missing = answer_b_core._clean_list(parsed.get("missing_information"))
    except (ValueError, TypeError, json.JSONDecodeError):
        local_recovery = True
        recovered = answer_b_core._recover_basic_answer_from_raw(raw, pack)
        answer = recovered["answer"]
        used_ids = list(recovered.get("used_evidence_ids") or [])
        coverage = "PARTIAL"
        missing = ["구조화 메타데이터를 로컬 복구함"]
    if not used_ids and allowed:
        used_ids = [next(iter(allowed))]
        coverage = "PARTIAL"
        missing = list(dict.fromkeys(missing + ["근거 ID를 로컬 보완함"]))
    return {
        "answer": answer,
        "used_evidence_ids": list(dict.fromkeys(used_ids)),
        "used_chunk_ids": [allowed[value] for value in dict.fromkeys(used_ids)],
        "coverage_status": coverage,
        "missing_information": missing,
    }, local_recovery


def generate_answer_b_v3(question: str, pack: Mapping[str, Any]) -> dict[str, Any]:
    total_started = time.perf_counter()
    prompt_started = time.perf_counter()
    constraint = relation_constraint_v1(question, pack)
    prompt = f"""[사용자 질문]\n{_clean_text(question)}\n\n[관계 주장 안전조건]\n{_compact_json(constraint)}\n\n[Basic Evidence Pack]\n{_compact_json(pack)}\n\n[출력 JSON]\n{{\"answer\":\"Markdown 형식 답변과 [E1] 근거표기\",\"used_evidence_ids\":[\"E1\"],\"coverage_status\":\"SUFFICIENT|PARTIAL|INSUFFICIENT\",\"missing_information\":[]}}"""
    prompt_ms = (time.perf_counter() - prompt_started) * 1000

    raw, usage, api_ms, trace = _call_answer_api_v1(
        system_prompt=B_STRUCTURED_SYSTEM_PROMPT_V3,
        user_prompt=prompt,
        max_tokens=1600,
    )
    parse_started = time.perf_counter()
    payload, local_recovery = _direct_answer_payload_v3(raw, pack)
    parse_ms = (time.perf_counter() - parse_started) * 1000

    post_started = time.perf_counter()
    safe_answer, guard_applied = _relation_safe_answer_v1(payload["answer"], constraint)
    payload["answer"] = normalize_answer_markdown_v3(safe_answer)
    if guard_applied:
        payload["coverage_status"] = "PARTIAL"
    numeric = audit_numeric_support_v2(payload["answer"], pack)
    post_ms = (time.perf_counter() - post_started) * 1000
    trace_parts = _trace_parts_v3(trace)
    total_ms = (time.perf_counter() - total_started) * 1000
    return {
        **payload,
        "system": "B",
        "latency_ms": total_ms,
        "usage": usage,
        "api_calls": 1,
        "attempts": [{"stage": "answer_b", "latency_ms": api_ms, "trace": trace}],
        "local_recovery": local_recovery,
        "relation_constraint": constraint,
        "relation_guard_applied": guard_applied,
        "numeric_audit": numeric,
        "stage_latency_ms": {
            "prompt_build_ms": prompt_ms,
            **trace_parts,
            "parse_validation_ms": parse_ms,
            "postprocess_ms": post_ms,
            "answer_total_ms": total_ms,
        },
    }


def generate_answer_c_v3(
    question: str,
    augmented_pack: Mapping[str, Any],
) -> dict[str, Any]:
    total_started = time.perf_counter()
    prompt_started = time.perf_counter()
    constraint = relation_constraint_v1(question, augmented_pack)
    prompt = f"""[사용자 질문]\n{_clean_text(question)}\n\n[관계 주장 안전조건]\n{_compact_json(constraint)}\n\n[Fact Index 보강 Evidence Pack]\n{_compact_json(augmented_pack)}\n\n[출력 JSON]\n{{\"answer\":\"Markdown 형식 답변과 [E1] 또는 [FI-CAND-001:F1] 근거표기\",\"used_evidence_ids\":[\"E1\"],\"used_fact_claim_ids\":[\"FI-CAND-001:F1\"],\"coverage_status\":\"SUFFICIENT|PARTIAL|INSUFFICIENT\",\"missing_information\":[]}}"""
    prompt_ms = (time.perf_counter() - prompt_started) * 1000

    raw, usage, api_ms, trace = _call_answer_api_v1(
        system_prompt=_managed_prompt("C_STRUCTURED_SYSTEM_PROMPT_V3", C_STRUCTURED_SYSTEM_PROMPT_V3),
        user_prompt=prompt,
        max_tokens=1600,
    )
    parse_started = time.perf_counter()
    allowed_evidence = answer_b_core._allowed_evidence(augmented_pack)
    allowed_facts = _allowed_fact_claims_v3(augmented_pack)
    local_recovery = False
    try:
        parsed = answer_b_core._extract_json_object(raw)
        answer = answer_b_core._strip_model_urls(answer_b_core._clean(parsed.get("answer")))
        if not answer:
            raise ValueError("C안 answer가 비어 있습니다.")
        used_evidence = [
            value for value in answer_b_core._clean_list(parsed.get("used_evidence_ids"))
            if value in allowed_evidence
        ]
        used_facts = [
            value for value in _clean_fact_claim_keys_v3(parsed.get("used_fact_claim_ids"))
            if value in allowed_facts
        ]
        coverage = str(parsed.get("coverage_status") or "PARTIAL").upper()
        if coverage not in answer_b_core.ALLOWED_COVERAGE_STATUS:
            coverage = "PARTIAL"
        missing = answer_b_core._clean_list(parsed.get("missing_information"))
    except (ValueError, TypeError, json.JSONDecodeError):
        local_recovery = True
        answer = answer_b_core._strip_model_urls(str(raw).strip())
        if not answer:
            raise ValueError("C안 출력이 비어 있습니다.")
        used_evidence = [
            f"E{number}" for number in re.findall(r"\[E(\d+)\]", answer)
            if f"E{number}" in allowed_evidence
        ]
        used_facts = [
            value for value in re.findall(r"\[(FI-CAND-\d+:F\d+)\]", answer)
            if value in allowed_facts
        ]
        coverage = "PARTIAL"
        missing = ["구조화 메타데이터를 로컬 복구함"]
    parse_ms = (time.perf_counter() - parse_started) * 1000

    post_started = time.perf_counter()
    safe_answer, guard_applied = _relation_safe_answer_v1(answer, constraint)
    answer = normalize_answer_markdown_v3(safe_answer)
    if guard_applied:
        coverage = "PARTIAL"
    post_ms = (time.perf_counter() - post_started) * 1000
    trace_parts = _trace_parts_v3(trace)
    total_ms = (time.perf_counter() - total_started) * 1000
    return {
        "system": "C",
        "answer": answer,
        "used_evidence_ids": list(dict.fromkeys(used_evidence)),
        "used_chunk_ids": [
            allowed_evidence[value] for value in dict.fromkeys(used_evidence)
        ],
        "used_fact_claim_ids": list(dict.fromkeys(used_facts)),
        "used_fact_index_ids": list(dict.fromkeys(
            value.split(":", 1)[0] for value in used_facts
        )),
        "coverage_status": coverage,
        "missing_information": missing,
        "latency_ms": total_ms,
        "usage": usage,
        "api_calls": 1,
        "attempts": [{"stage": "answer_c", "latency_ms": api_ms, "trace": trace}],
        "local_recovery": local_recovery,
        "relation_constraint": constraint,
        "relation_guard_applied": guard_applied,
        "stage_latency_ms": {
            "prompt_build_ms": prompt_ms,
            **trace_parts,
            "parse_validation_ms": parse_ms,
            "postprocess_ms": post_ms,
            "answer_total_ms": total_ms,
        },
    }


def generate_answer_d_v3(question: str, pack: Mapping[str, Any]) -> dict[str, Any]:
    total_started = time.perf_counter()
    need_started = time.perf_counter()
    answer_needs = extract_answer_needs_v2(question)
    relation_constraint = relation_constraint_v1(question, pack)
    need_ms = (time.perf_counter() - need_started) * 1000

    skeleton_prompt_started = time.perf_counter()
    skeleton_prompt = f"""[사용자 질문]\n{_clean_text(question)}\n\n[Answer Needs]\n{_compact_json(answer_needs)}\n\n[관계 주장 안전조건]\n{_compact_json(relation_constraint)}\n\n[Evidence Pack]\n{_compact_json(pack)}\n\n[출력 JSON]\n{{\"core_answer\":\"핵심 결론\",\"answer_items\":[{{\"need_id\":\"N1\",\"topic\":\"항목\",\"status\":\"ANSWERED|PARTIAL|UNSUPPORTED\",\"claim\":\"근거 사실\",\"conditions\":[],\"details\":[],\"evidence_ids\":[\"E1\"],\"missing_reason\":\"\"}}],\"uncertainties\":[],\"conflicts\":[]}}"""
    skeleton_prompt_ms = (time.perf_counter() - skeleton_prompt_started) * 1000
    raw_skeleton, usage1, skeleton_api_ms, trace1 = _call_answer_api_v1(
        system_prompt=D2_SKELETON_SYSTEM_PROMPT,
        user_prompt=skeleton_prompt,
        max_tokens=2000,
    )

    validation_started = time.perf_counter()
    skeleton = validate_d2_skeleton_v2(
        answer_b_core._extract_json_object(raw_skeleton), answer_needs, pack
    )
    validation_ms = (time.perf_counter() - validation_started) * 1000

    selection_started = time.perf_counter()
    selected_pack = filter_pack_for_d2_v2(pack, skeleton)
    selection_ms = (time.perf_counter() - selection_started) * 1000

    final_prompt_started = time.perf_counter()
    final_prompt = f"""[사용자 질문]\n{_clean_text(question)}\n\n[Answer Needs]\n{_compact_json(answer_needs)}\n\n[관계 주장 안전조건]\n{_compact_json(relation_constraint)}\n\n[Answer Skeleton]\n{_compact_json(skeleton)}\n\n[Skeleton 참조 Evidence]\n{_compact_json(selected_pack)}\n\n위 범위에서 Markdown 구조를 지켜 최종 사용자 답변을 작성하세요."""
    final_prompt_ms = (time.perf_counter() - final_prompt_started) * 1000
    raw_answer, usage2, final_api_ms, trace2 = _call_answer_api_v1(
        system_prompt=D_STRUCTURED_FINAL_SYSTEM_PROMPT_V3,
        user_prompt=final_prompt,
        max_tokens=1600,
    )

    post_started = time.perf_counter()
    answer = answer_b_core._strip_model_urls(str(raw_answer).strip())
    if not answer:
        raise ValueError("D안 최종 답변이 비어 있습니다.")
    safe_answer, guard_applied = _relation_safe_answer_v1(answer, relation_constraint)
    safe_answer = normalize_answer_markdown_v3(safe_answer)
    numeric = audit_numeric_support_v2(safe_answer, selected_pack)
    post_ms = (time.perf_counter() - post_started) * 1000
    skeleton_trace = _trace_parts_v3(trace1)
    final_trace = _trace_parts_v3(trace2)
    total_ms = (time.perf_counter() - total_started) * 1000
    return {
        "system": "D",
        "answer": safe_answer,
        "answer_needs": answer_needs,
        "skeleton": skeleton,
        "need_coverage": skeleton["need_coverage"],
        "coverage_status": "PARTIAL" if guard_applied else skeleton["coverage_status"],
        "strict_need_coverage_rate": skeleton["strict_need_coverage_rate"],
        "answerable_need_coverage_rate": skeleton["answerable_need_coverage_rate"],
        "numeric_audit": numeric,
        "selected_evidence_pack": selected_pack,
        "selected_evidence_count": len(selected_pack["evidence"]),
        "latency_ms": total_ms,
        "skeleton_latency_ms": skeleton_api_ms,
        "final_latency_ms": final_api_ms,
        "usage": _merge_usage_v1(usage1, usage2),
        "api_calls": 2,
        "attempts": [
            {"stage": "skeleton", "latency_ms": skeleton_api_ms, "trace": trace1},
            {"stage": "final", "latency_ms": final_api_ms, "trace": trace2},
        ],
        "relation_constraint": relation_constraint,
        "relation_guard_applied": guard_applied,
        "stage_latency_ms": {
            "need_extraction_ms": need_ms,
            "skeleton_prompt_build_ms": skeleton_prompt_ms,
            "skeleton_api_wall_ms": skeleton_trace["api_wall_ms"],
            "skeleton_pacing_wait_ms": skeleton_trace["pacing_wait_ms"],
            "skeleton_retry_wait_ms": skeleton_trace["retry_wait_ms"],
            "skeleton_estimated_service_ms": skeleton_trace["estimated_service_ms"],
            "skeleton_validation_ms": validation_ms,
            "evidence_selection_ms": selection_ms,
            "final_prompt_build_ms": final_prompt_ms,
            "final_api_wall_ms": final_trace["api_wall_ms"],
            "final_pacing_wait_ms": final_trace["pacing_wait_ms"],
            "final_retry_wait_ms": final_trace["retry_wait_ms"],
            "final_estimated_service_ms": final_trace["estimated_service_ms"],
            "postprocess_ms": post_ms,
            "answer_total_ms": total_ms,
        },
    }


print({
    "answer_b": "1-call + Markdown + detailed latency",
    "answer_c": "1-call + Fact Index + Markdown + detailed latency",
    "answer_d": "Skeleton + final + detailed latency",
    "additional_llm_calls_for_formatting": 0,
})

# ==== cell 76 ====

import copy
import html
import json
import re
import time
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit


ACTION_LINK_REGISTRY_DOCUMENT_V1 = json.loads('{"schema_version":"kdic-action-link-registry-v1.0","registry_version":"2026-08-20","policy":{"allowed_schemes":["https"],"allowed_hosts":["www.kdic.or.kr","fins.kdic.or.kr","mkcs.kdic.or.kr"],"max_buttons":3,"selection_mode":"ROUTE_AND_BUSINESS_AND_ACTION_AND_ROLE","network_check_at_runtime":false,"llm_may_generate_or_select_urls":false,"source_links_and_action_links_are_separate":true},"records":[{"link_id":"DP-INSTITUTION-SEARCH-001","business_function_code":"deposit_protection","action_type":"INSTITUTION_SEARCH","actor_roles":["ANY"],"channel":"WEB","button_label":"보호대상 금융회사 검색","description":"금융회사명이 예금자보호 대상인지 공식 검색 화면에서 확인합니다.","url":"https://www.kdic.or.kr/sp/dpstrprot/selectProtSystProtSrch.do","activation_keywords":["금융회사","은행","저축은행","보호대상","보호되나요","가입기관"],"exclusion_keywords":[],"source_chunk_ids":[],"priority":10,"approved_for_display":true,"verification_status":"OFFICIAL_PAGE_VERIFIED","last_verified_date":"2026-08-20"},{"link_id":"DP-PRODUCT-SEARCH-001","business_function_code":"deposit_protection","action_type":"PRODUCT_SEARCH","actor_roles":["ANY"],"channel":"WEB","button_label":"보호대상 금융상품 검색","description":"예금·적금·금융상품이 보호대상인지 공식 검색 화면에서 확인합니다.","url":"https://www.kdic.or.kr/sp/dpstrprot/selectProtSystProtTrgtPrdctSrchList.do","activation_keywords":["금융상품","예금","적금","상품","보호대상","보호되나요"],"exclusion_keywords":[],"source_chunk_ids":[],"priority":10,"approved_for_display":true,"verification_status":"OFFICIAL_PAGE_VERIFIED","last_verified_date":"2026-08-20"},{"link_id":"DI-APPLICATION-PROCEDURE-001","business_function_code":"deposit_insurance_payout","action_type":"APPLY_GUIDE","actor_roles":["SELF","PROXY","HEIR","ANY"],"channel":"WEB","button_label":"예금보험금 신청 절차","description":"방문·인터넷 신청 절차와 지급 흐름을 확인합니다.","url":"https://www.kdic.or.kr/sp/dpstrprot/DpsmIbamtAplyProc/selectScrn.do","activation_keywords":["신청","지급","받으려면","절차","방법","수령"],"exclusion_keywords":["신청할 수 없는"],"source_chunk_ids":["BI-002_chunk_002"],"priority":20,"approved_for_display":true,"verification_status":"OFFICIAL_PAGE_VERIFIED","last_verified_date":"2026-08-20"},{"link_id":"DI-DOCUMENT-GUIDE-001","business_function_code":"deposit_insurance_payout","action_type":"DOCUMENT_GUIDE","actor_roles":["SELF","PROXY","HEIR","ANY"],"channel":"WEB","button_label":"예금보험금 구비서류·양식","description":"본인·대리인·상속인 등 신청 상황별 서류와 공식 양식을 확인합니다.","url":"https://www.kdic.or.kr/sp/dpstrprot/DpsmIbamtAplyPossDcmnt/selectScrn.do","activation_keywords":["서류","구비","준비물","위임장","양식","다운로드"],"exclusion_keywords":[],"source_chunk_ids":["BI-001_chunk_000","BI-001_chunk_001","BI-001_chunk_004","BI-001_chunk_006"],"priority":5,"approved_for_display":true,"verification_status":"OFFICIAL_PAGE_VERIFIED","last_verified_date":"2026-08-20"},{"link_id":"DI-PAYMENT-AGENT-SEARCH-001","business_function_code":"deposit_insurance_payout","action_type":"OFFLINE_LOCATION_SEARCH","actor_roles":["ANY"],"channel":"WEB","button_label":"예금보험금 지급대행점 조회","description":"방문 신청이 가능한 지급대행점을 조회합니다.","url":"https://www.kdic.or.kr/sp/dpstrprot/selectProtSystBamtGiveInq.do","activation_keywords":["방문","지급대행점","어디","지점","오프라인"],"exclusion_keywords":[],"source_chunk_ids":["BI-002_chunk_002"],"priority":8,"approved_for_display":true,"verification_status":"OFFICIAL_PAGE_VERIFIED","last_verified_date":"2026-08-20"},{"link_id":"UN-INTEGRATED-APPLICATION-001","business_function_code":"unclaimed_funds","action_type":"APPLY","actor_roles":["SELF","PROXY","HEIR","ANY"],"channel":"WEB_AUTH","button_label":"미수령금 통합신청","description":"본인인증 후 미수령금 통합 확인·신청을 진행합니다.","url":"https://fins.kdic.or.kr/ua/itgraply/selectItgrInqDsctn.do","activation_keywords":["신청","찾기","받기","수령","통합신청"],"exclusion_keywords":["신청할 수 없는","제외"],"source_chunk_ids":["UN-001_chunk_000"],"priority":5,"approved_for_display":true,"verification_status":"OFFICIAL_AUTH_REDIRECT_VERIFIED","last_verified_date":"2026-08-20"},{"link_id":"UN-APPLICATION-STATUS-001","business_function_code":"unclaimed_funds","action_type":"STATUS_CHECK","actor_roles":["SELF","PROXY","HEIR","ANY"],"channel":"WEB_AUTH","button_label":"미수령금 진행·지급내역 조회","description":"본인인증 후 신청 진행상태와 지급내역을 확인합니다.","url":"https://fins.kdic.or.kr/ua/dsctninq/selectItgrInq.do","activation_keywords":["조회","확인","진행","상태","지급내역","신청내역"],"exclusion_keywords":[],"source_chunk_ids":["UN-001_chunk_000"],"priority":4,"approved_for_display":true,"verification_status":"OFFICIAL_AUTH_PAGE_VERIFIED","last_verified_date":"2026-08-20"},{"link_id":"UN-APPLICATION-GUIDE-001","business_function_code":"unclaimed_funds","action_type":"APPLY_GUIDE","actor_roles":["SELF","PROXY","HEIR","ANY"],"channel":"WEB","button_label":"미수령금 통합신청 안내","description":"미수령금 종류와 온라인·오프라인 신청방법을 먼저 확인합니다.","url":"https://fins.kdic.or.kr/ua/aplygudn/NramtItgrAplyItrdMthdGudn/selectScrn.do","activation_keywords":["미수령금","신청","방법","절차","어떻게"],"exclusion_keywords":[],"source_chunk_ids":["UN-003_chunk_000"],"priority":15,"approved_for_display":true,"verification_status":"OFFICIAL_PAGE_VERIFIED","last_verified_date":"2026-08-20"},{"link_id":"UN-HEIR-INQUIRY-GUIDE-001","business_function_code":"unclaimed_funds","action_type":"HEIR_INQUIRY","actor_roles":["HEIR"],"channel":"WEB","button_label":"상속인 금융거래조회 안내","description":"상속인의 금융거래·미수령금 조회 절차를 확인합니다.","url":"https://www.kdic.or.kr/sp/dpstrprot/ProtSystHrpeHistInq/selectScrn.do","activation_keywords":["상속","상속인","사망","피상속인"],"exclusion_keywords":[],"source_chunk_ids":["UN-004_chunk_000"],"priority":3,"approved_for_display":true,"verification_status":"OFFICIAL_PAGE_VERIFIED","last_verified_date":"2026-08-20"},{"link_id":"MT-SITUATION-SELECT-001","business_function_code":"mistaken_transfer","action_type":"SITUATION_SELECT","actor_roles":["ANY"],"channel":"WEB","button_label":"착오송금 상황 선택","description":"송금인인지 수취인인지에 따라 이용할 절차를 선택합니다.","url":"https://fins.kdic.or.kr/ir/aplygudn/MtrsStutChc/selectScrn.do","activation_keywords":["착오송금","잘못 보냈","모르는 돈","송금인","수취인","받았"],"exclusion_keywords":[],"source_chunk_ids":["MT-002_chunk_000"],"priority":30,"approved_for_display":true,"verification_status":"OFFICIAL_PAGE_VERIFIED","last_verified_date":"2026-08-20"},{"link_id":"MT-SENDER-ELIGIBILITY-001","business_function_code":"mistaken_transfer","action_type":"ELIGIBILITY_CHECK","actor_roles":["SENDER"],"channel":"WEB","button_label":"착오송금 신청자격 확인","description":"반환지원 신청 전에 공식 자가진단 항목으로 대상 여부를 확인합니다.","url":"https://fins.kdic.or.kr/ir/msdrpr/selectAplyQlfcIdntyRslt.do","activation_keywords":["자격","대상","신청할 수","가능","조건","제외","누가"],"exclusion_keywords":[],"source_chunk_ids":["MT-004_chunk_000","MT-013_chunk_000"],"priority":3,"approved_for_display":true,"verification_status":"OFFICIAL_INTERACTIVE_PAGE_VERIFIED","last_verified_date":"2026-08-20"},{"link_id":"MT-SENDER-APPLICATION-001","business_function_code":"mistaken_transfer","action_type":"APPLY","actor_roles":["SENDER"],"channel":"WEB_AUTH","button_label":"착오송금 반환지원 신청","description":"신청자격 확인 후 본인인증을 거쳐 반환지원을 신청합니다.","url":"https://fins.kdic.or.kr/ir/msdrpr/selectAplyQlfcIdntyChc.do","activation_keywords":["신청","접수","반환지원","어떻게","방법"],"exclusion_keywords":["신청할 수 없는","제외","수취인"],"source_chunk_ids":["MT-002_chunk_000","MT-013_chunk_006"],"priority":5,"approved_for_display":true,"verification_status":"OFFICIAL_AUTH_REDIRECT_VERIFIED","last_verified_date":"2026-08-20"},{"link_id":"MT-SENDER-STATUS-001","business_function_code":"mistaken_transfer","action_type":"STATUS_CHECK","actor_roles":["SENDER"],"channel":"WEB_AUTH","button_label":"착오송금 신청내역 확인","description":"본인인증 후 반환지원 신청 진행·지급내역을 확인합니다.","url":"https://fins.kdic.or.kr/ir/msdrpr/selectAplyDsctnInqList.do","activation_keywords":["신청내역","진행","상태","조회","지급내역","확인"],"exclusion_keywords":[],"source_chunk_ids":["MT-013_chunk_006"],"priority":3,"approved_for_display":true,"verification_status":"OFFICIAL_AUTH_REDIRECT_VERIFIED","last_verified_date":"2026-08-20"},{"link_id":"MT-SENDER-DOCUMENTS-001","business_function_code":"mistaken_transfer","action_type":"DOCUMENT_GUIDE","actor_roles":["SENDER","PROXY"],"channel":"WEB","button_label":"착오송금인 구비서류","description":"착오송금인 본인·대리 신청에 필요한 공식 서류와 양식을 확인합니다.","url":"https://fins.kdic.or.kr/ir/aplygudn/MsdrprPossDcmntGudn/selectScrn.do","activation_keywords":["서류","구비","준비물","위임장","양식","대리인"],"exclusion_keywords":["수취인"],"source_chunk_ids":["MT-010_operation_layer01_chunk_002"],"priority":3,"approved_for_display":true,"verification_status":"OFFICIAL_PAGE_VERIFIED","last_verified_date":"2026-08-20"},{"link_id":"MT-RECIPIENT-DOCUMENTS-001","business_function_code":"mistaken_transfer","action_type":"DOCUMENT_GUIDE","actor_roles":["RECIPIENT","PROXY"],"channel":"WEB","button_label":"착오송금 수취인 구비서류","description":"착오송금 수취인 관련 반환·이의 절차의 공식 서류를 확인합니다.","url":"https://fins.kdic.or.kr/ir/aplygudn/MsdrAddrsePossDcmntGudn/selectScrn.do","activation_keywords":["서류","구비","준비물","위임장","양식","수취인","받은 사람"],"exclusion_keywords":["송금인"],"source_chunk_ids":["MT-010_operation_layer01_chunk_002"],"priority":3,"approved_for_display":true,"verification_status":"OFFICIAL_PAGE_VERIFIED","last_verified_date":"2026-08-20"},{"link_id":"MT-RECIPIENT-BALANCE-001","business_function_code":"mistaken_transfer","action_type":"BALANCE_CHECK","actor_roles":["RECIPIENT"],"channel":"WEB_AUTH","button_label":"수취인 채무잔액 확인","description":"본인인증 후 착오송금 수취인의 채무잔액 관련 내역을 확인합니다.","url":"https://fins.kdic.or.kr/ir/addrse/selectLbltBlncIdntyList.do","activation_keywords":["채무잔액","잔액","수취인","받은 사람"],"exclusion_keywords":["송금인"],"source_chunk_ids":[],"priority":2,"approved_for_display":true,"verification_status":"OFFICIAL_AUTH_REDIRECT_VERIFIED","last_verified_date":"2026-08-20"},{"link_id":"MT-RECIPIENT-RETURN-CHECK-001","business_function_code":"mistaken_transfer","action_type":"RETURN_CHECK","actor_roles":["RECIPIENT"],"channel":"WEB_AUTH","button_label":"수취인 반환 확인","description":"본인인증 후 착오송금 수취인의 반환 처리 결과를 확인합니다.","url":"https://fins.kdic.or.kr/ir/addrse/selectGvbkIdntyList.do","activation_keywords":["반환 확인","반환했","처리 결과","수취인","받은 사람"],"exclusion_keywords":["송금인"],"source_chunk_ids":[],"priority":2,"approved_for_display":true,"verification_status":"OFFICIAL_AUTH_REDIRECT_VERIFIED","last_verified_date":"2026-08-20"},{"link_id":"DA-DEBT-INQUIRY-001","business_function_code":"debt_adjustment","action_type":"DEBT_INQUIRY","actor_roles":["DEBTOR","SELF","ANY"],"channel":"WEB_AUTH","button_label":"채무정보 조회·상담 신청","description":"본인인증 후 채무정보를 조회하고 대상이면 채무조정 상담을 신청합니다.","url":"https://fins.kdic.or.kr/lb/lbltinfo/selectLbltInfoInq.do","activation_keywords":["채무정보","조회","상담","신청","채무조정","확인"],"exclusion_keywords":[],"source_chunk_ids":["DA-002_chunk_000","DA-002_chunk_002"],"priority":5,"approved_for_display":true,"verification_status":"OFFICIAL_ACTION_LINK_FROM_CORPUS","last_verified_date":"2026-08-20"},{"link_id":"DA-ELIGIBILITY-DOCUMENTS-001","business_function_code":"debt_adjustment","action_type":"DOCUMENT_GUIDE","actor_roles":["DEBTOR","SELF","ANY"],"channel":"WEB","button_label":"채무조정 자격·구비서류","description":"채무조정 신청자격과 필요한 서류를 공식 안내에서 확인합니다.","url":"https://www.kdic.or.kr/rb/lbltajmt/LbltAjmtSprtLbltAjmtSyst/selectScrn.do","activation_keywords":["자격","대상","서류","구비","준비물","조건"],"exclusion_keywords":[],"source_chunk_ids":["DA-001_chunk_002","DA-001_chunk_003"],"priority":3,"approved_for_display":true,"verification_status":"OFFICIAL_PAGE_VERIFIED","last_verified_date":"2026-08-20"},{"link_id":"HP-REPORT-GUIDE-001","business_function_code":"hidden_assets_report","action_type":"REPORT_GUIDE","actor_roles":["REPORTER","ANY"],"channel":"WEB","button_label":"은닉재산 신고 안내","description":"신고 대상, 포상금, 신고방법과 보호조치를 공식 안내에서 확인합니다.","url":"https://www.kdic.or.kr/sp/sprtfund/SprtFndCncmDclrGudn/selectScrn.do","activation_keywords":["신고","제보","포상금","은닉재산","방법"],"exclusion_keywords":[],"source_chunk_ids":["HP-001_chunk_000","HP-001_chunk_002","HP-001_chunk_005"],"priority":10,"approved_for_display":true,"verification_status":"OFFICIAL_PAGE_VERIFIED","last_verified_date":"2026-08-20"},{"link_id":"HP-REPORT-AND-INQUIRY-001","business_function_code":"hidden_assets_report","action_type":"REPORT","actor_roles":["REPORTER","ANY"],"channel":"WEB","button_label":"은닉재산 신고·조회","description":"은닉재산을 신고하거나 기존 신고 관련 조회 화면으로 이동합니다.","url":"https://www.kdic.or.kr/sp/sprtfund/SprtCncmDclrInqGudn/selectScrn.do","activation_keywords":["신고","제보","접수","조회","진행","신고내역"],"exclusion_keywords":[],"source_chunk_ids":["HP-001_chunk_005"],"priority":5,"approved_for_display":true,"verification_status":"OFFICIAL_PAGE_VERIFIED","last_verified_date":"2026-08-20"}]}')
ACTION_LINK_RECORDS_V1 = copy.deepcopy(
    ACTION_LINK_REGISTRY_DOCUMENT_V1.get("records") or []
)
ACTION_LINK_POLICY_V1 = copy.deepcopy(
    ACTION_LINK_REGISTRY_DOCUMENT_V1.get("policy") or {}
)
ACTION_LINK_ALLOWED_HOSTS_V1 = {
    str(value).lower() for value in ACTION_LINK_POLICY_V1.get("allowed_hosts") or []
}
ACTION_LINK_ALLOWED_SCHEMES_V1 = {
    str(value).lower() for value in ACTION_LINK_POLICY_V1.get("allowed_schemes") or []
}
ACTION_LINK_MAX_BUTTONS_V1 = int(ACTION_LINK_POLICY_V1.get("max_buttons") or 3)


ACTION_LINK_BUSINESS_BY_LABEL_V1 = {
    "예금자보호제도": "deposit_protection",
    "예금자보호": "deposit_protection",
    "예금보험금 안내": "deposit_insurance_payout",
    "예금보험금": "deposit_insurance_payout",
    "고객 미수령금 신청": "unclaimed_funds",
    "고객 미수령금": "unclaimed_funds",
    "미수령금": "unclaimed_funds",
    "착오송금 반환 신청": "mistaken_transfer",
    "착오송금 반환지원": "mistaken_transfer",
    "착오송금": "mistaken_transfer",
    "채무조정 안내": "debt_adjustment",
    "채무조정": "debt_adjustment",
    "은닉재산 신고": "hidden_assets_report",
}


def validate_action_url_v1(url: Any) -> tuple[bool, str]:
    value = str(url or "").strip()
    if not value:
        return False, "EMPTY_URL"
    try:
        parsed = urlsplit(value)
    except Exception:
        return False, "URL_PARSE_ERROR"
    if parsed.scheme.lower() not in ACTION_LINK_ALLOWED_SCHEMES_V1:
        return False, "SCHEME_NOT_ALLOWED"
    if (parsed.hostname or "").lower() not in ACTION_LINK_ALLOWED_HOSTS_V1:
        return False, "HOST_NOT_ALLOWED"
    if parsed.username or parsed.password:
        return False, "URL_CREDENTIALS_NOT_ALLOWED"
    if not parsed.path or parsed.path == "/":
        return False, "EMPTY_ACTION_PATH"
    return True, "VALID"


def validate_action_link_registry_v1() -> list[dict[str, Any]]:
    required = {
        "link_id", "business_function_code", "action_type", "actor_roles",
        "button_label", "description", "url", "activation_keywords",
        "priority", "approved_for_display", "verification_status",
        "last_verified_date",
    }
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for record in ACTION_LINK_RECORDS_V1:
        link_id = str(record.get("link_id") or "")
        missing = sorted(required.difference(record))
        duplicate = bool(link_id and link_id in seen)
        seen.add(link_id)
        url_valid, url_reason = validate_action_url_v1(record.get("url"))
        roles_valid = bool(record.get("actor_roles"))
        approved = record.get("approved_for_display") is True
        passed = bool(
            link_id and not missing and not duplicate and url_valid and roles_valid and approved
        )
        rows.append({
            "link_id": link_id,
            "passed": passed,
            "missing_fields": ", ".join(missing),
            "duplicate": duplicate,
            "url_valid": url_valid,
            "url_reason": url_reason,
            "actor_roles_valid": roles_valid,
            "approved_for_display": approved,
        })
    return rows


def _action_clean_v1(value: Any) -> str:
    cleaner = globals().get("_clean_text") or globals().get("_clean")
    if callable(cleaner):
        try:
            return str(cleaner(value))
        except Exception:
            pass
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _action_business_codes_v1(common: Mapping[str, Any]) -> set[str]:
    existing = globals().get("_detected_business_codes_c1")
    if callable(existing):
        try:
            values = set(existing(common))
            if values:
                return values
        except Exception:
            pass
    analysis = common.get("analysis") or {}
    labels = list(analysis.get("businesses") or [])
    labels.extend(analysis.get("detected_businesses") or [])
    labels.extend(analysis.get("active_businesses") or [])
    return {
        ACTION_LINK_BUSINESS_BY_LABEL_V1[str(label).strip()]
        for label in labels
        if str(label).strip() in ACTION_LINK_BUSINESS_BY_LABEL_V1
    }


def detect_action_actor_roles_v1(
    common: Mapping[str, Any],
    question_text: str,
    business_codes: set[str],
) -> set[str]:
    roles: set[str] = set()
    analysis = common.get("analysis") or {}
    raw_context = analysis.get("context")
    context = raw_context if isinstance(raw_context, Mapping) else {}
    raw_roles = [
        analysis.get("actor_role"),
        context.get("actor_role"),
        (analysis.get("context_result") or {}).get("actor_role")
        if isinstance(analysis.get("context_result"), Mapping) else None,
    ]
    role_aliases = {
        "SENDER": "SENDER", "송금인": "SENDER", "착오송금인": "SENDER",
        "RECIPIENT": "RECIPIENT", "수취인": "RECIPIENT", "착오송금수취인": "RECIPIENT",
        "PROXY": "PROXY", "대리인": "PROXY",
        "HEIR": "HEIR", "상속인": "HEIR",
        "DEBTOR": "DEBTOR", "채무자": "DEBTOR",
        "REPORTER": "REPORTER", "신고자": "REPORTER", "제보자": "REPORTER",
        "SELF": "SELF", "본인": "SELF",
    }
    for raw in raw_roles:
        mapped = role_aliases.get(str(raw or "").strip())
        if mapped:
            roles.add(mapped)

    text = question_text.lower()
    if re.search(r"송금인|착오송금인|잘못\s*보냈|돈을\s*보낸|보낸\s*사람", text):
        roles.add("SENDER")
    if re.search(r"수취인|받은\s*사람|모르는\s*돈|잘못\s*받았|입금\s*받", text):
        roles.add("RECIPIENT")
    if re.search(r"대리인|대리\s*신청|위임", text):
        roles.add("PROXY")
    if re.search(r"상속인|피상속인|사망한|사망자", text):
        roles.add("HEIR")
    if re.search(r"채무자|내\s*채무|본인\s*채무", text):
        roles.add("DEBTOR")
    if re.search(r"신고자|제보자|신고하려|제보하려", text):
        roles.add("REPORTER")
    if re.search(r"본인|제가|내가|저는", text):
        roles.add("SELF")

    if "mistaken_transfer" in business_codes and not {"SENDER", "RECIPIENT"}.intersection(roles):
        if re.search(r"착오송금.*(?:반환지원\s*)?신청|반환지원.*신청", text):
            roles.add("SENDER")
    if "debt_adjustment" in business_codes and re.search(r"채무조정|채무정보|상담", text):
        roles.add("DEBTOR")
    if "hidden_assets_report" in business_codes and re.search(r"신고|제보", text):
        roles.add("REPORTER")
    if not roles:
        roles.add("ANY")
    return roles


def detect_action_types_v1(
    question_text: str,
    business_codes: set[str],
    actor_roles: set[str],
) -> set[str]:
    text = question_text.lower()
    output: set[str] = set()

    has_application = bool(re.search(r"신청|접수|신고|제보|받으려|수령", text))
    has_status = bool(re.search(r"신청\s*내역|진행\s*(?:상태|상황)?|지급\s*내역|처리\s*결과|조회", text))
    has_documents = bool(re.search(r"서류|구비|준비물|위임장|양식|다운로드|첨부", text))
    has_eligibility = bool(re.search(r"자격|대상|누가|신청할\s*수|가능한가|가능해|조건|제외|안\s*되", text))
    has_method = bool(re.search(r"어떻게|방법|절차|하려면", text))

    if "deposit_protection" in business_codes:
        protection_check_intent = bool(
            re.search(
                r"검색|조회|확인|보호\s*대상|보호되|가입\s*(?:여부|기관)|"
                r"금융회사\s*(?:인가|인지|여부)|상품\s*(?:인가|인지|여부)|"
                r"(?:은행|금융회사|상품|적금).*보호",
                text,
            )
        )
        if protection_check_intent:
            if re.search(r"금융회사|은행|저축은행|가입기관", text):
                output.add("INSTITUTION_SEARCH")
            if re.search(r"금융상품|적금|상품|예금\s*(?:상품|계좌|통장)", text):
                output.add("PRODUCT_SEARCH")

    if "deposit_insurance_payout" in business_codes:
        if has_documents:
            output.add("DOCUMENT_GUIDE")
        elif re.search(r"방문|지급대행점|지점|오프라인", text):
            output.add("OFFLINE_LOCATION_SEARCH")
        elif has_application or has_method:
            output.add("APPLY_GUIDE")

    if "unclaimed_funds" in business_codes:
        if re.search(r"상속|상속인|사망", text):
            output.add("HEIR_INQUIRY")
        elif has_status:
            output.add("STATUS_CHECK")
        elif has_application or has_method:
            output.update({"APPLY", "APPLY_GUIDE"})

    if "mistaken_transfer" in business_codes:
        if re.search(r"채무\s*잔액|잔액", text) and "RECIPIENT" in actor_roles:
            output.add("BALANCE_CHECK")
        elif re.search(r"반환\s*확인|반환했|처리\s*결과", text) and "RECIPIENT" in actor_roles:
            output.add("RETURN_CHECK")
        elif has_documents:
            output.add("DOCUMENT_GUIDE")
        elif has_status and "SENDER" in actor_roles:
            output.add("STATUS_CHECK")
        elif has_eligibility and "SENDER" in actor_roles:
            output.add("ELIGIBILITY_CHECK")
        elif (has_application or has_method) and "SENDER" in actor_roles:
            output.add("APPLY")
        elif not {"SENDER", "RECIPIENT"}.intersection(actor_roles) and (
            has_application or has_status or has_documents or has_eligibility or has_method
        ):
            output.add("SITUATION_SELECT")

    if "debt_adjustment" in business_codes:
        if has_documents or has_eligibility:
            output.add("DOCUMENT_GUIDE")
        elif has_application or has_status or has_method or re.search(r"상담|문의|채무정보", text):
            output.add("DEBT_INQUIRY")

    if "hidden_assets_report" in business_codes:
        if re.search(r"신고|제보|접수|조회|신고내역", text):
            output.update({"REPORT", "REPORT_GUIDE"})
        elif re.search(r"포상금.*(?:어디|방법)|어디.*포상금", text):
            output.add("REPORT_GUIDE")

    if re.search(r"철회|취소", text):
        return {"WITHDRAW"}
    if re.search(r"정보\s*변경|신청\s*변경|수정", text):
        return {"MODIFY"}
    return output


def resolve_action_links_v1(
    common: Mapping[str, Any],
    *,
    max_buttons: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if str(common.get("route") or "") != "RETRIEVE":
        return [], [{"accepted": False, "reason": "NON_RETRIEVE_ROUTE"}]

    question_text = _action_clean_v1(" ".join([
        str(common.get("question") or ""),
        str(common.get("resolved_question") or ""),
    ]))
    text_lower = question_text.lower()
    business_codes = _action_business_codes_v1(common)
    actor_roles = detect_action_actor_roles_v1(common, question_text, business_codes)
    action_types = detect_action_types_v1(question_text, business_codes, actor_roles)
    limit = max(0, int(max_buttons or ACTION_LINK_MAX_BUTTONS_V1))
    if not business_codes or not action_types or limit == 0:
        return [], [{
            "accepted": False,
            "reason": "NO_BUSINESS" if not business_codes else "NO_ACTION_INTENT",
            "business_codes": sorted(business_codes),
            "actor_roles": sorted(actor_roles),
            "action_types": sorted(action_types),
        }]

    accepted: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for record in ACTION_LINK_RECORDS_V1:
        link_id = str(record.get("link_id") or "")
        business_match = str(record.get("business_function_code") or "") in business_codes
        action_match = str(record.get("action_type") or "") in action_types
        record_roles = {str(value) for value in record.get("actor_roles") or []}
        role_match = "ANY" in record_roles or bool(record_roles.intersection(actor_roles))
        keyword_hits = [
            str(value) for value in record.get("activation_keywords") or []
            if str(value).strip() and str(value).strip().lower() in text_lower
        ]
        exclusion_hits = [
            str(value) for value in record.get("exclusion_keywords") or []
            if str(value).strip() and str(value).strip().lower() in text_lower
        ]
        url_valid, url_reason = validate_action_url_v1(record.get("url"))
        approved = record.get("approved_for_display") is True
        is_accepted = bool(
            business_match and action_match and role_match and keyword_hits
            and not exclusion_hits and url_valid and approved
        )
        score = (
            100 * int(action_match)
            + 30 * int(role_match and "ANY" not in record_roles)
            + 10 * len(set(keyword_hits))
            - int(record.get("priority") or 99)
        )
        reason = (
            "ACCEPTED"
            if is_accepted else
            "BUSINESS_MISMATCH" if not business_match else
            "ACTION_MISMATCH" if not action_match else
            "ROLE_MISMATCH" if not role_match else
            "NO_ACTIVATION_KEYWORD" if not keyword_hits else
            "EXCLUSION_KEYWORD" if exclusion_hits else
            url_reason if not url_valid else
            "NOT_APPROVED"
        )
        row = {
            "link_id": link_id,
            "accepted": is_accepted,
            "reason": reason,
            "business_match": business_match,
            "action_match": action_match,
            "role_match": role_match,
            "keyword_hits": list(dict.fromkeys(keyword_hits)),
            "exclusion_hits": list(dict.fromkeys(exclusion_hits)),
            "url_valid": url_valid,
            "url_reason": url_reason,
            "score": score,
            "detected_business_codes": sorted(business_codes),
            "detected_actor_roles": sorted(actor_roles),
            "detected_action_types": sorted(action_types),
        }
        audit.append(row)
        if is_accepted:
            accepted.append({**copy.deepcopy(record), "_selection_audit": row})

    accepted.sort(key=lambda value: (
        -int((value.get("_selection_audit") or {}).get("score") or 0),
        int(value.get("priority") or 99),
        str(value.get("link_id") or ""),
    ))
    selected: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for record in accepted:
        url = str(record.get("url") or "")
        if url in seen_urls:
            continue
        seen_urls.add(url)
        selected.append(record)
        if len(selected) >= limit:
            break
    selected_ids = {str(value.get("link_id") or "") for value in selected}
    for row in audit:
        if row.get("accepted") and row.get("link_id") not in selected_ids:
            row["accepted"] = False
            row["reason"] = "LOWER_PRIORITY_OR_BUTTON_LIMIT"
    return selected, audit


def sanitize_answer_urls_v1(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"<a\b[^>]*>(.*?)</a>", r"\1", text, flags=re.I | re.S)
    text = re.sub(r"\[([^\]]+)\]\(https?://[^)]+\)", r"\1", text, flags=re.I)
    text = re.sub(r"https?://[^\s<>)\]]+", "", text, flags=re.I)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def action_links_markdown_v1(action_links: Sequence[Mapping[str, Any]]) -> str:
    lines = ["### 관련 공식 서비스", ""]
    for record in action_links:
        url_valid, _ = validate_action_url_v1(record.get("url"))
        if not url_valid or record.get("approved_for_display") is not True:
            continue
        label = str(record.get("button_label") or "공식 서비스 열기").replace("|", "\\|")
        description = _action_clean_v1(record.get("description")).replace("|", "\\|")
        url = str(record.get("url") or "")
        auth_note = " · 본인인증 필요" if str(record.get("channel") or "") == "WEB_AUTH" else ""
        lines.append(
            f"- [{label}]({url}){auth_note}  \n  {description}"
        )
    if len(lines) == 2:
        return ""
    lines.extend([
        "",
        "> 위 링크는 답변 모델이 생성한 주소가 아니라 Action Link Registry에서 검증된 예금보험공사 공식 페이지입니다.",
    ])
    return "\n".join(lines)


def render_action_links_colab_v1(action_links: Sequence[Mapping[str, Any]]) -> None:
    markdown = action_links_markdown_v1(action_links)
    if not markdown:
        return
    from IPython.display import Markdown, display
    display(Markdown(markdown))


def action_links_for_streamlit_v1(
    action_links: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """추후 Streamlit의 st.link_button()에 바로 전달할 안전한 표시 데이터입니다."""
    output: list[dict[str, Any]] = []
    for record in action_links:
        url_valid, _ = validate_action_url_v1(record.get("url"))
        if url_valid and record.get("approved_for_display") is True:
            output.append({
                "link_id": str(record.get("link_id") or ""),
                "business_function_code": str(record.get("business_function_code") or ""),
                "action_type": str(record.get("action_type") or ""),
                "label": str(record.get("button_label") or "공식 서비스 열기"),
                "url": str(record.get("url") or ""),
                "description": str(record.get("description") or ""),
                "requires_auth": str(record.get("channel") or "") == "WEB_AUTH",
            })
    return output


ACTION_LINK_PROMPT_RULE_V1 = """
[후속 행동 링크 안전 규칙]
- 답변 본문에 URL, 링크 주소, Markdown 링크를 직접 만들지 마세요.
- 신청·조회·서류·상담 페이지는 프로그램의 검증된 Action Link Registry가 답변 뒤에 별도로 제공합니다.
- Evidence에 URL이 있어도 답변 문장 안에 복사하지 마세요.
""".strip()
for _prompt_name in (
    "B_STRUCTURED_SYSTEM_PROMPT_V3",
    "C_STRUCTURED_SYSTEM_PROMPT_V3",
    "D_STRUCTURED_FINAL_SYSTEM_PROMPT_V3",
):
    if _prompt_name in globals():
        globals()[_prompt_name] = globals()[_prompt_name] + "\n\n" + ACTION_LINK_PROMPT_RULE_V1


def run_action_link_regression_v1() -> list[dict[str, Any]]:
    cases = [
        ("보호 금융회사", "이 은행은 예금자보호 금융회사인가요?", ["예금자보호제도"], {"DP-INSTITUTION-SEARCH-001"}, set()),
        ("보호 금융상품", "이 적금 상품도 보호대상인가요?", ["예금자보호제도"], {"DP-PRODUCT-SEARCH-001"}, set()),
        ("보험금 서류", "예금보험금 신청 서류와 양식은 어디서 받나요?", ["예금보험금 안내"], {"DI-DOCUMENT-GUIDE-001"}, set()),
        ("보험금 방문", "예금보험금을 방문 신청할 지급대행점은 어디인가요?", ["예금보험금 안내"], {"DI-PAYMENT-AGENT-SEARCH-001"}, set()),
        ("미수령금 신청", "미수령금 통합신청은 어떻게 하나요?", ["고객 미수령금 신청"], {"UN-APPLICATION-GUIDE-001", "UN-INTEGRATED-APPLICATION-001"}, set()),
        ("미수령금 상태", "미수령금 신청 진행상태를 조회하고 싶어요", ["고객 미수령금 신청"], {"UN-APPLICATION-STATUS-001"}, set()),
        ("상속인 조회", "사망한 가족의 미수령금을 상속인이 조회하려면?", ["고객 미수령금 신청"], {"UN-HEIR-INQUIRY-GUIDE-001"}, set()),
        ("착오송금 신청", "착오송금 반환지원 신청은 어떻게 하나요?", ["착오송금 반환 신청"], {"MT-SENDER-APPLICATION-001"}, set()),
        ("착오송금 자격", "착오송금인은 누가 반환지원을 신청할 수 있나요?", ["착오송금 반환 신청"], {"MT-SENDER-ELIGIBILITY-001"}, set()),
        ("송금인 서류", "착오송금인 본인 신청 서류는 무엇인가요?", ["착오송금 반환 신청"], {"MT-SENDER-DOCUMENTS-001"}, set()),
        ("수취인 서류", "착오송금 수취인이 준비할 서류는 무엇인가요?", ["착오송금 반환 신청"], {"MT-RECIPIENT-DOCUMENTS-001"}, set()),
        ("수취인 잔액", "착오송금 수취인이 채무잔액을 확인하려면?", ["착오송금 반환 신청"], {"MT-RECIPIENT-BALANCE-001"}, set()),
        ("채무 조회", "채무조정 신청 전에 채무정보를 조회하고 상담받고 싶어요", ["채무조정 안내"], {"DA-DEBT-INQUIRY-001"}, set()),
        ("채무 서류", "채무조정 신청 자격과 구비서류가 궁금합니다", ["채무조정 안내"], {"DA-ELIGIBILITY-DOCUMENTS-001"}, set()),
        ("은닉재산 신고", "은닉재산을 신고하려면 어디에서 접수하나요?", ["은닉재산 신고"], {"HP-REPORT-AND-INQUIRY-001", "HP-REPORT-GUIDE-001"}, set()),
        ("정보형 무버튼", "예금자보호 한도는 얼마인가요?", ["예금자보호제도"], set(), set()),
        ("비검색 무버튼", "채무조정 신청 방법", ["채무조정 안내"], set(), set()),
    ]
    rows = []
    for name, question, businesses, required, forbidden in cases:
        route = "OUT_OF_SCOPE" if name == "비검색 무버튼" else "RETRIEVE"
        common = {
            "route": route,
            "question": question,
            "resolved_question": question,
            "analysis": {"businesses": businesses},
        }
        selected, _ = resolve_action_links_v1(common)
        selected_ids = {str(value.get("link_id") or "") for value in selected}
        passed = selected_ids == required and not forbidden.intersection(selected_ids)
        rows.append({
            "case": name,
            "question": question,
            "route": route,
            "businesses": " | ".join(businesses),
            "selected_link_ids": " | ".join(sorted(selected_ids)),
            "required_link_ids": " | ".join(sorted(required)),
            "forbidden_link_ids": " | ".join(sorted(forbidden)),
            "passed": bool(passed),
        })
    return rows


ACTION_LINK_REGISTRY_GATE_ROWS_V1 = validate_action_link_registry_v1()
ACTION_LINK_REGRESSION_ROWS_V1 = run_action_link_regression_v1()
ACTION_LINK_HARD_GATE_V1 = {
    "registry_record_count": len(ACTION_LINK_RECORDS_V1),
    "invalid_registry_record_count": sum(
        1 for row in ACTION_LINK_REGISTRY_GATE_ROWS_V1 if not row.get("passed")
    ),
    "regression_case_count": len(ACTION_LINK_REGRESSION_ROWS_V1),
    "regression_failure_count": sum(
        1 for row in ACTION_LINK_REGRESSION_ROWS_V1 if not row.get("passed")
    ),
    "llm_url_selection_enabled": False,
    "runtime_network_check_enabled": False,
}
if ACTION_LINK_HARD_GATE_V1["invalid_registry_record_count"]:
    raise RuntimeError("Action Link Registry 구조·도메인 하드 게이트 실패")
if ACTION_LINK_HARD_GATE_V1["regression_failure_count"]:
    failed = [row for row in ACTION_LINK_REGRESSION_ROWS_V1 if not row.get("passed")]
    raise RuntimeError("Action Link 회귀 테스트 실패: " + json.dumps(failed, ensure_ascii=False))

display(pd.DataFrame(ACTION_LINK_REGISTRY_GATE_ROWS_V1))
display(pd.DataFrame(ACTION_LINK_REGRESSION_ROWS_V1))
print(ACTION_LINK_HARD_GATE_V1)

# ==== cell 78 ====

import copy
import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Mapping


HCX_CIRCUIT_MAX_ATTEMPTS_C1 = 2
HCX_CIRCUIT_COOLDOWN_SECONDS_C1 = 60.0


class HCXCircuitOpenErrorC1(RuntimeError):
    pass


class HCXCircuitBreakerGateC1(HCXSharedRequestGateV3):
    def __init__(self, *, min_interval_seconds: float) -> None:
        super().__init__(
            min_interval_seconds=min_interval_seconds,
            max_attempts=HCX_CIRCUIT_MAX_ATTEMPTS_C1,
        )
        self.cooldown_until_monotonic = 0.0
        self.circuit_lock = threading.RLock()

    def cooldown_remaining_seconds(self) -> float:
        return max(0.0, self.cooldown_until_monotonic - time.monotonic())

    def call(
        self,
        stage: str,
        operation: Callable[[], Any],
        *,
        max_attempts: int | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        with self.circuit_lock:
            remaining = self.cooldown_remaining_seconds()
            if remaining > 0:
                trace = {
                    "stage": stage,
                    "success": False,
                    "circuit_open": True,
                    "cooldown_remaining_seconds": remaining,
                    "attempts": [],
                    "pacing_wait_ms": 0.0,
                    "retry_wait_ms": 0.0,
                    "wall_latency_ms": 0.0,
                }
                self.last_trace = trace
                self.history.append(copy.deepcopy(trace))
                raise HCXCircuitOpenErrorC1(
                    f"HCX 429 보호 대기 중입니다. 약 {remaining:.1f}초 후 다시 시도하세요."
                )

        try:
            return super().call(
                stage,
                operation,
                max_attempts=HCX_CIRCUIT_MAX_ATTEMPTS_C1,
            )
        except Exception as error:
            cause = getattr(error, "__cause__", None)
            is_rate_limit = (
                type(error).__name__ == "RateLimitError"
                or type(cause).__name__ == "RateLimitError"
                or "429" in str(error)
            )
            if is_rate_limit:
                with self.circuit_lock:
                    self.cooldown_until_monotonic = max(
                        self.cooldown_until_monotonic,
                        time.monotonic() + HCX_CIRCUIT_COOLDOWN_SECONDS_C1,
                    )
                raise HCXCircuitOpenErrorC1(
                    "HCX 429가 반복되어 이번 실행을 중단했습니다. "
                    f"{HCX_CIRCUIT_COOLDOWN_SECONDS_C1:.0f}초 후 자동으로 다시 허용됩니다."
                ) from error
            raise


# 기존 질문 임베딩·문맥 판정·답변 래퍼가 참조하는 전역 게이트만 교체합니다.
HCX_SHARED_GATE_V3 = HCXCircuitBreakerGateC1(
    min_interval_seconds=HCX_GLOBAL_MIN_INTERVAL_SECONDS_V3,
)


# HCX-007 분해기는 별도 requests 전송기를 사용하므로 재시도 횟수와 429 상태를
# 공통 Circuit Breaker에 연결합니다. 정상 분해·캐시 동작은 그대로 유지합니다.
if hasattr(V15_DECOMPOSER, "transport") and hasattr(V15_DECOMPOSER.transport, "policy"):
    _v15_policy_c1 = V15_DECOMPOSER.transport.policy
    V15_DECOMPOSER.transport.policy = TransportPolicy(
        request_delay_seconds=float(_v15_policy_c1.request_delay_seconds),
        max_transport_retries=1,
        base_backoff_seconds=float(_v15_policy_c1.base_backoff_seconds),
        max_backoff_seconds=float(_v15_policy_c1.max_backoff_seconds),
        jitter_seconds=float(_v15_policy_c1.jitter_seconds),
        consecutive_429_cooldown_threshold=1,
        cooldown_seconds=HCX_CIRCUIT_COOLDOWN_SECONDS_C1,
        timeout_seconds=float(_v15_policy_c1.timeout_seconds),
    )

_V15_DECOMPOSE_RAW_C1 = V15_DECOMPOSER.decompose


def _v15_decompose_circuit_guard_c1(
    question: str,
    expected_businesses: Sequence[str],
) -> dict[str, Any]:
    remaining = HCX_SHARED_GATE_V3.cooldown_remaining_seconds()
    if remaining > 0:
        raise HCXCircuitOpenErrorC1(
            f"HCX 429 보호 대기 중이므로 구조화 분해를 호출하지 않습니다. 약 {remaining:.1f}초 남았습니다."
        )
    row = _V15_DECOMPOSE_RAW_C1(question, expected_businesses)
    if int(row.get("http_status") or 0) == 429:
        HCX_SHARED_GATE_V3.cooldown_until_monotonic = max(
            HCX_SHARED_GATE_V3.cooldown_until_monotonic,
            time.monotonic() + HCX_CIRCUIT_COOLDOWN_SECONDS_C1,
        )
        raise HCXCircuitOpenErrorC1(
            "HCX-007 구조화 분해에서 429가 반복되어 이번 실행을 중단했습니다. "
            f"{HCX_CIRCUIT_COOLDOWN_SECONDS_C1:.0f}초 후 자동 해제됩니다."
        )
    return row


V15_DECOMPOSER.decompose = _v15_decompose_circuit_guard_c1


def hcx_circuit_status_c1() -> dict[str, Any]:
    remaining = HCX_SHARED_GATE_V3.cooldown_remaining_seconds()
    return {
        "open": remaining > 0,
        "cooldown_remaining_seconds": remaining,
        "max_attempts_per_logical_call": HCX_CIRCUIT_MAX_ATTEMPTS_C1,
        "cooldown_seconds": HCX_CIRCUIT_COOLDOWN_SECONDS_C1,
    }


def _stable_json_hash_c1(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _common_request_key_c1(question: str, state: Mapping[str, Any]) -> str:
    payload = {
        "question": _clean_text(question),
        "state": copy.deepcopy(dict(state)),
        "dense_weight": DENSE_WEIGHT,
        "bm25_weight": BM25_WEIGHT,
        "candidate_depth": CANDIDATE_DEPTH,
        "min_relevance_score": MIN_RELEVANCE_SCORE,
        "reranker_model": RERANKER_MODEL_NAME,
        "parent_context_max_chars": PARENT_CONTEXT_MAX_CHARS,
        "dense_cache_version": DENSE_CACHE_VERSION,
        "fact_index_sha256": FACT_INDEX_DOCUMENT_C1["_sha256"],
    }
    return _stable_json_hash_c1(payload)


def _answer_cache_key_c1(
    variant: str,
    common: Mapping[str, Any],
    augmented_pack: Mapping[str, Any] | None,
) -> str:
    return _stable_json_hash_c1({
        "variant": variant,
        "resolved_question": common.get("resolved_question"),
        "base_pack_sha256": common.get("evidence_pack_sha256"),
        "augmented_pack_sha256": (
            _stable_json_hash_c1(augmented_pack) if augmented_pack is not None else None
        ),
        "prompt_version": "b-c-d-individual-c1",
    })


def new_bcd_controller_state_c1() -> dict[str, Any]:
    return {
        "conversation": new_bd_comparison_state(),
        "current_question": "",
        "common_request_key": "",
        "common": None,
        "common_created_at": None,
        "answer_cache": {},
        "committed": False,
        "committed_variant": None,
        "events": [],
        "running": False,
        "ignored_duplicate_events": 0,
    }


print({
    "hcx_circuit_breaker": True,
    "max_attempts": HCX_CIRCUIT_MAX_ATTEMPTS_C1,
    "cooldown_seconds": HCX_CIRCUIT_COOLDOWN_SECONDS_C1,
    "automatic_release": True,
    "hcx007_transport_max_retries": 1,
})

# ==== cell 80 ====
def _variant_usage_v3(payload: Mapping[str, Any]) -> dict[str, int]:
    usage = payload.get("usage") or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }


def _answer_cache_key_v3(
    variant: str,
    common: Mapping[str, Any],
    augmented_pack: Mapping[str, Any] | None,
) -> str:
    return _stable_json_hash_c1({
        "variant": variant,
        "resolved_question": common.get("resolved_question"),
        "base_pack_sha256": common.get("evidence_pack_sha256"),
        "augmented_pack_sha256": (
            _stable_json_hash_c1(augmented_pack) if augmented_pack is not None else None
        ),
        "prompt_version": "b-c-d-detailed-latency-markdown-v3",
    })


def _trace_summary_since_v3(start_index: int) -> dict[str, Any]:
    traces = HCX_SHARED_GATE_V3.history[start_index:]
    attempts = [attempt for trace in traces for attempt in trace.get("attempts") or []]
    return {
        "logical_api_calls": len(traces),
        "physical_http_attempts": len(attempts),
        "rate_limit_429_count": sum(
            1 for attempt in attempts if attempt.get("status") == "RATE_LIMIT_429"
        ),
        "pacing_wait_ms": sum(float(trace.get("pacing_wait_ms") or 0) for trace in traces),
        "retry_wait_ms": sum(float(trace.get("retry_wait_ms") or 0) for trace in traces),
        "traces": copy.deepcopy(traces),
    }


def _prepare_or_reuse_common_v3(
    question: str,
    holder: dict[str, Any],
) -> tuple[dict[str, Any], bool, float]:
    cleaned = _clean_text(question)
    if holder.get("common") is not None and cleaned == holder.get("current_question"):
        return holder["common"], True, 0.0

    request_key = _common_request_key_c1(cleaned, holder["conversation"])
    gate_start = len(HCX_SHARED_GATE_V3.history)
    started = time.perf_counter()
    common = prepare_common_retrieval_v1(cleaned, state=holder["conversation"])
    wall_ms = (time.perf_counter() - started) * 1000
    common["common_hcx_trace_v3"] = _trace_summary_since_v3(gate_start)
    holder.update({
        "current_question": cleaned,
        "common_request_key": request_key,
        "common": common,
        "common_created_at": time.time(),
        "answer_cache": {},
        "committed": False,
        "committed_variant": None,
    })
    return common, False, wall_ms


def execute_bcd_variant_v3(
    variant: str,
    question: str,
    holder: dict[str, Any],
    *,
    force_answer_regeneration: bool = False,
) -> dict[str, Any]:
    variant = str(variant).upper()
    if variant not in {"B", "C", "D"}:
        raise ValueError(f"지원하지 않는 답변안: {variant}")
    click_started = time.perf_counter()
    gate_start_index = len(HCX_SHARED_GATE_V3.history)
    common, common_cache_hit, common_this_click_ms = _prepare_or_reuse_common_v3(
        question, holder
    )
    if common.get("route") != "RETRIEVE":
        return {
            "variant": variant,
            "common": common,
            "route": common.get("route"),
            "route_message": common.get("route_message"),
            "common_cache_hit": common_cache_hit,
            "answer_cache_hit": False,
            "latency": {
                "common_this_click_ms": common_this_click_ms,
                "answer_ms": 0.0,
                "click_wall_ms": (time.perf_counter() - click_started) * 1000,
            },
            "api_trace": _trace_summary_since_v3(gate_start_index),
        }

    fact_started = time.perf_counter()
    matched_records, fact_audit = match_fact_index_c1(common)
    augmented_pack = build_fact_augmented_pack_c1(common["evidence_pack"], matched_records)
    fact_ms = (time.perf_counter() - fact_started) * 1000
    cache_key = _answer_cache_key_v3(
        variant,
        common,
        augmented_pack if variant == "C" else None,
    )
    cached = holder["answer_cache"].get(cache_key)
    if cached is not None and not force_answer_regeneration:
        payload = copy.deepcopy(cached)
        answer_cache_hit = True
        answer_ms = 0.0
    else:
        answer_cache_hit = False
        answer_started = time.perf_counter()
        if variant == "B":
            payload = generate_answer_b_v3(
                common["resolved_question"], common["evidence_pack"]
            )
        elif variant == "C":
            payload = generate_answer_c_v3(common["resolved_question"], augmented_pack)
        else:
            payload = generate_answer_d_v3(
                common["resolved_question"], common["evidence_pack"]
            )
        answer_ms = (time.perf_counter() - answer_started) * 1000
        holder["answer_cache"][cache_key] = copy.deepcopy(payload)

    if not holder.get("committed"):
        holder["conversation"].setdefault("turns", []).extend([
            {"role": "user", "content": _clean_text(question)},
            {"role": "assistant", "content": normalize_answer_markdown_v3(payload.get("answer"))},
        ])
        holder["committed"] = True
        holder["committed_variant"] = variant

    trace = _trace_summary_since_v3(gate_start_index)
    usage = _variant_usage_v3(payload)
    stage = payload.get("stage_latency_ms") or {}
    result = {
        "variant": variant,
        "route": "RETRIEVE",
        "common": common,
        "payload": payload,
        "matched_fact_records": matched_records,
        "fact_audit": fact_audit,
        "augmented_pack": augmented_pack if variant == "C" else None,
        "common_cache_hit": common_cache_hit,
        "answer_cache_hit": answer_cache_hit,
        "committed_variant": holder.get("committed_variant"),
        "latency": {
            "stored_common_pipeline_ms": float(
                (common.get("latency_ms") or {}).get("공통 준비 전체") or 0
            ),
            "common_this_click_ms": common_this_click_ms,
            "fact_index_match_ms": fact_ms if variant == "C" else 0.0,
            "answer_ms": answer_ms,
            "click_wall_ms": (time.perf_counter() - click_started) * 1000,
        },
        "api_trace": trace,
        "usage": usage,
        "circuit": hcx_circuit_status_c1(),
    }
    event = {
        "question": _clean_text(question),
        "variant": variant,
        "common_cache_hit": common_cache_hit,
        "answer_cache_hit": answer_cache_hit,
        "fact_index_count": len(matched_records) if variant == "C" else 0,
        "fact_index_ids": ", ".join(
            str(row.get("fact_index_id") or "") for row in matched_records
        ) if variant == "C" else "",
        "stored_common_pipeline_ms": result["latency"]["stored_common_pipeline_ms"],
        "common_this_click_ms": common_this_click_ms,
        "fact_index_match_ms": result["latency"]["fact_index_match_ms"],
        "answer_ms": answer_ms,
        "skeleton_api_ms": float(stage.get("skeleton_api_wall_ms") or 0),
        "final_api_ms": float(stage.get("final_api_wall_ms") or 0),
        "click_wall_ms": result["latency"]["click_wall_ms"],
        **{key: value for key, value in trace.items() if key != "traces"},
        **usage,
        "coverage_status": payload.get("coverage_status"),
        "answer_chars": len(str(payload.get("answer") or "")),
    }
    holder["events"].append(event)
    result["event"] = event
    return result


print("B/C/D v3 상세 레이턴시 실행기 준비 완료")

# ==== cell 82 ====
_PREPARE_COMMON_BEFORE_ACTION_LINK_V4 = _prepare_or_reuse_common_v3
_NORMALIZE_MARKDOWN_BEFORE_ACTION_LINK_V4 = normalize_answer_markdown_v3
_EXECUTE_BCD_BEFORE_ACTION_LINK_V4 = execute_bcd_variant_v3


def normalize_answer_markdown_v3(text: Any) -> str:
    return sanitize_answer_urls_v1(
        _NORMALIZE_MARKDOWN_BEFORE_ACTION_LINK_V4(text)
    )


def _prepare_or_reuse_common_v3(
    question: str,
    holder: dict[str, Any],
) -> tuple[dict[str, Any], bool, float]:
    common, cache_hit, common_this_click_ms = _PREPARE_COMMON_BEFORE_ACTION_LINK_V4(
        question, holder
    )
    if "action_links" not in common:
        started = time.perf_counter()
        selected, audit = resolve_action_links_v1(common)
        action_ms = (time.perf_counter() - started) * 1000
        common["action_links"] = selected
        common["action_link_audit"] = audit
        common["action_link_registry_version"] = ACTION_LINK_REGISTRY_DOCUMENT_V1.get(
            "registry_version"
        )
        common.setdefault("latency_ms", {})["Action Link Registry"] = action_ms
        if "공통 준비 전체" in common.get("latency_ms", {}):
            common["latency_ms"]["공통 준비 전체"] = (
                float(common["latency_ms"].get("공통 준비 전체") or 0) + action_ms
            )
        common_this_click_ms += action_ms
    return common, cache_hit, common_this_click_ms


def execute_bcd_variant_v3(
    variant: str,
    question: str,
    holder: dict[str, Any],
    *,
    force_answer_regeneration: bool = False,
) -> dict[str, Any]:
    result = _EXECUTE_BCD_BEFORE_ACTION_LINK_V4(
        variant,
        question,
        holder,
        force_answer_regeneration=force_answer_regeneration,
    )
    common = result.get("common") or {}
    payload = result.get("payload") or {}
    result["answer"] = normalize_answer_markdown_v3(
        payload.get("answer") or result.get("route_message") or ""
    )
    official_sources = []
    seen_source_urls = set()
    for source in (common.get("evidence_pack") or {}).get("sources") or []:
        url = str(source.get("source_url") or "").strip()
        if not url or url in seen_source_urls:
            continue
        seen_source_urls.add(url)
        official_sources.append({
            "title": str(source.get("title") or "공식 출처"),
            "url": url,
        })
    result["official_sources"] = official_sources
    result["action_links"] = action_links_for_streamlit_v1(
        common.get("action_links") or []
    )
    result["action_link_audit"] = copy.deepcopy(
        common.get("action_link_audit") or []
    )
    result["action_link_registry_version"] = common.get(
        "action_link_registry_version"
    )
    event = result.get("event")
    if isinstance(event, dict):
        event["action_link_count"] = len(result["action_links"])
        event["action_link_ids"] = " | ".join(
            str(row.get("link_id") or "") for row in result["action_links"]
        )
    return result


print({
    "action_link_integration": "enabled",
    "answer_url_policy": "LLM URLs stripped; registry links only",
    "colab_renderer": "Markdown official action links",
    "streamlit_adapter": "action_links_for_streamlit_v1",
})

# ==== cell 84 ====
# V5: 교차업무 Need-aware Batch Reranking
from collections import OrderedDict, defaultdict
from typing import Any, Mapping, Sequence

NEED_BATCH_MAX_EVIDENCE_V5 = 6
NEED_BATCH_REQUIRED_TOP_K_V5 = 2
NEED_BATCH_GLOBAL_RERANK_WEIGHT_V5 = 0.70
NEED_BATCH_GLOBAL_RRF_WEIGHT_V5 = 0.30
ANSWER_EVIDENCE_RANK_BUDGETS_V5 = (3_000, 2_800, 2_400, 2_200, 2_000, 1_600)
ANSWER_PROMPT_VERSION = "need-batch-rerank-v5"

_FUSE_QUERY_RESULTS_V4_1 = fuse_query_results
_LAST_NEED_BATCH_CONTEXT_V5: dict[str, Any] = {}


def _source_is_decomposed_v5(plan: Mapping[str, Any]) -> bool:
    return "DECOMPOSED" in str(plan.get("source") or "").upper()


def _source_is_original_v5(plan: Mapping[str, Any]) -> bool:
    return "ORIGINAL" in str(plan.get("source") or "").upper()


def _need_business_v5(question: str) -> str:
    try:
        values = list(light_router.find_businesses(question) or [])
    except Exception:
        values = []
    return str(values[0]) if len(values) == 1 else ""


def _row_parent_key_v5(row: Mapping[str, Any]) -> str:
    chunk = dict(row.get("chunk") or {})
    return str(row.get("parent_doc_id") or _parent_id_for_chunk(chunk) or row.get("chunk_id") or "")


def _row_url_key_v5(row: Mapping[str, Any]) -> str:
    chunk = dict(row.get("chunk") or {})
    return _clean_text(chunk.get("source_url"))


def _minmax_values_v5(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    numbers = np.asarray(values, dtype=np.float64)
    low, high = float(numbers.min()), float(numbers.max())
    if abs(high - low) <= 1e-12:
        return [1.0 for _ in values]
    return [float(value) for value in ((numbers - low) / (high - low))]


def _fused_candidates_v5(
    all_hits: Sequence[tuple[int, Mapping[str, Any], Sequence[Mapping[str, Any]]]],
    rrf_k: int,
) -> list[dict[str, Any]]:
    fused: dict[str, dict[str, Any]] = {}
    for plan_index, plan, hits in all_hits:
        for hit in hits:
            chunk_id = str(hit["chunk_id"])
            row = fused.setdefault(chunk_id, {
                "chunk_id": chunk_id,
                "query_fusion_score": 0.0,
                "best_minmax_score": 0.0,
                "dense_rank": None,
                "bm25_rank": None,
                "matched_queries": [],
                "chunk": hit["chunk"],
            })
            row["query_fusion_score"] += float(plan["weight"]) / (rrf_k + int(hit["rank"]))
            row["best_minmax_score"] = max(
                float(row["best_minmax_score"]), float(hit.get("minmax_score") or 0.0)
            )
            if row["dense_rank"] is None or (
                hit.get("dense_rank") is not None and int(hit["dense_rank"]) < int(row["dense_rank"])
            ):
                row["dense_rank"] = hit.get("dense_rank")
            if row["bm25_rank"] is None or (
                hit.get("bm25_rank") is not None and int(hit["bm25_rank"]) < int(row["bm25_rank"])
            ):
                row["bm25_rank"] = hit.get("bm25_rank")
            row["matched_queries"].append({
                "plan_index": plan_index,
                "query": str(plan.get("query") or ""),
                "source": str(plan.get("source") or ""),
                "rank": int(hit["rank"]),
                "weight": float(plan["weight"]),
            })
    return sorted(
        fused.values(),
        key=lambda row: (
            -float(row["query_fusion_score"]),
            -float(row["best_minmax_score"]),
            str(row["chunk_id"]),
        ),
    )


def _can_select_need_row_v5(
    row: Mapping[str, Any],
    *,
    selected_chunks: set[str],
    selected_parents: set[str],
    selected_urls: set[str],
) -> bool:
    chunk_id = str(row.get("chunk_id") or "")
    parent_id = _row_parent_key_v5(row)
    source_url = _row_url_key_v5(row)
    if not chunk_id or chunk_id in selected_chunks:
        return False
    if parent_id and parent_id in selected_parents:
        return False
    if source_url and source_url in selected_urls:
        return False
    return True


def _mark_selected_need_row_v5(
    row: Mapping[str, Any],
    *,
    selected_chunks: set[str],
    selected_parents: set[str],
    selected_urls: set[str],
) -> None:
    selected_chunks.add(str(row.get("chunk_id") or ""))
    parent_id = _row_parent_key_v5(row)
    source_url = _row_url_key_v5(row)
    if parent_id:
        selected_parents.add(parent_id)
    if source_url:
        selected_urls.add(source_url)


def fuse_query_results(
    plans: list[dict[str, Any]],
    *,
    top_k: int = FINAL_TOP_K,
    rrf_k: int = QUERY_FUSION_RRF_K,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """교차업무만 Need-aware batch rerank, 나머지는 검증된 V4.1 검색을 유지합니다."""
    global _LAST_RERANK_TRACE, _LAST_NEED_BATCH_CONTEXT_V5
    if not plans:
        raise ValueError("검색 계획이 없습니다.")
    if not math.isclose(sum(float(plan["weight"]) for plan in plans), 1.0, abs_tol=1e-9):
        raise ValueError("검색 계획 가중치 합은 1이어야 합니다.")

    decomposed_plans = [plan for plan in plans if _source_is_decomposed_v5(plan)]
    if len(decomposed_plans) < 2:
        _LAST_NEED_BATCH_CONTEXT_V5 = {
            "strategy": "V4_1_STANDARD_RERANK",
            "needs": [],
            "max_evidence": int(top_k),
        }
        return _FUSE_QUERY_RESULTS_V4_1(plans, top_k=top_k, rrf_k=rrf_k)

    original_plan = next((plan for plan in plans if _source_is_original_v5(plan)), plans[0])
    original_question = _clean_text(original_plan.get("query"))
    needs = [
        {
            "need_id": f"N{index}",
            "plan_index": plans.index(plan) + 1,
            "question": _clean_text(plan.get("query")),
            "business_function": _need_business_v5(str(plan.get("query") or "")),
        }
        for index, plan in enumerate(decomposed_plans, start=1)
    ]
    need_by_plan_index = {int(row["plan_index"]): row for row in needs}

    per_query: list[dict[str, Any]] = []
    all_hits: list[tuple[int, Mapping[str, Any], Sequence[Mapping[str, Any]]]] = []
    hits_by_plan_index: dict[int, list[dict[str, Any]]] = {}
    for plan_index, plan in enumerate(plans, start=1):
        hits = hybrid_minmax_search(str(plan["query"]), top_k=CANDIDATE_DEPTH)
        trace = dict(_V15_LAST_QUERY_TRACE)
        hits_by_plan_index[plan_index] = hits
        all_hits.append((plan_index, plan, hits))
        per_query.append({
            **plan,
            "plan_index": plan_index,
            "latency_ms": float(trace.get("query_total_latency_ms") or 0.0),
            "latency_breakdown_ms": trace,
            "hits": hits,
        })

    fusion_started = time.perf_counter()
    fused = _fused_candidates_v5(all_hits, rrf_k)[:RERANKER_CANDIDATE_DEPTH]
    fusion_latency_ms = (time.perf_counter() - fusion_started) * 1000

    pair_texts: list[list[str]] = []
    pair_refs: list[dict[str, Any]] = []
    for need in needs:
        for hit in hits_by_plan_index[int(need["plan_index"])]:
            pair_texts.append([str(need["question"]), _reranker_passage(hit["chunk"])])
            pair_refs.append({"kind": "need", "need_id": need["need_id"], "row": hit})
    for row in fused:
        pair_texts.append([original_question, _reranker_passage(row["chunk"])])
        pair_refs.append({"kind": "global", "row": row})
    if not pair_texts:
        raise RuntimeError("Need-aware Reranker 입력 쌍이 없습니다.")

    rerank_started = time.perf_counter()
    pair_scores = np.asarray(
        RERANKER_MODEL.predict(pair_texts, batch_size=RERANKER_BATCH_SIZE),
        dtype=np.float64,
    ).reshape(-1)
    rerank_latency_ms = (time.perf_counter() - rerank_started) * 1000
    if len(pair_scores) != len(pair_refs):
        raise RuntimeError("BAAI Reranker 점수 개수가 후보 쌍 개수와 다릅니다.")

    need_scored: dict[str, list[dict[str, Any]]] = defaultdict(list)
    global_scored: list[dict[str, Any]] = []
    for ref, score in zip(pair_refs, pair_scores.tolist()):
        if ref["kind"] == "need":
            need = next(value for value in needs if value["need_id"] == ref["need_id"])
            row = {**dict(ref["row"]), "need_reranker_score": float(score), **need}
            need_scored[str(need["need_id"])].append(row)
        else:
            global_scored.append({**dict(ref["row"]), "global_reranker_score": float(score)})
    for need_id in need_scored:
        need_scored[need_id].sort(
            key=lambda row: (-float(row["need_reranker_score"]), int(row.get("rank") or 10**9), str(row["chunk_id"]))
        )

    # 4개 이상의 need에서는 총 6개 제한을 지키기 위해 필수 보장치를 1개로 낮춥니다.
    required_top_k = NEED_BATCH_REQUIRED_TOP_K_V5 if len(needs) <= 3 else 1
    selected: list[dict[str, Any]] = []
    selected_chunks: set[str] = set()
    selected_parents: set[str] = set()
    selected_urls: set[str] = set()
    coverage_rows: list[dict[str, Any]] = []

    for need in needs:
        candidates = list(need_scored.get(str(need["need_id"]), []))
        business = str(need.get("business_function") or "")
        domain_candidates = [
            row for row in candidates
            if not business or _clean_text((row.get("chunk") or {}).get("business_function")) == business
        ]
        selected_for_need = 0
        # 업무가 검출됐을 때는 다른 업무 청크로 필수 자리를 채우지 않습니다.
        for row in domain_candidates:
            if selected_for_need >= required_top_k:
                break
            if not _can_select_need_row_v5(
                row,
                selected_chunks=selected_chunks,
                selected_parents=selected_parents,
                selected_urls=selected_urls,
            ):
                continue
            chosen = {
                **row,
                "selection_type": "NEED_REQUIRED",
                "need_ids": [str(need["need_id"])],
                "need_queries": [str(need["question"])],
                "need_businesses": [business] if business else [],
                "reranker_score": float(row["need_reranker_score"]),
            }
            fused_row = next((value for value in fused if value["chunk_id"] == row["chunk_id"]), None)
            if fused_row:
                chosen["query_fusion_score"] = float(fused_row["query_fusion_score"])
                chosen["matched_queries"] = list(fused_row["matched_queries"])
            selected.append(chosen)
            _mark_selected_need_row_v5(
                chosen,
                selected_chunks=selected_chunks,
                selected_parents=selected_parents,
                selected_urls=selected_urls,
            )
            selected_for_need += 1
        coverage_rows.append({
            **need,
            "candidate_count": len(candidates),
            "domain_candidate_count": len(domain_candidates),
            "required_count": required_top_k,
            "selected_count": selected_for_need,
            "sufficient": selected_for_need >= required_top_k,
        })

    rerank_norm = _minmax_values_v5([float(row["global_reranker_score"]) for row in global_scored])
    rrf_norm = _minmax_values_v5([float(row["query_fusion_score"]) for row in global_scored])
    optional_rows: list[dict[str, Any]] = []
    expected_businesses = {str(row["business_function"]) for row in needs if row.get("business_function")}
    for row, rerank_value, rrf_value in zip(global_scored, rerank_norm, rrf_norm):
        chunk_business = _clean_text((row.get("chunk") or {}).get("business_function"))
        if expected_businesses and chunk_business not in expected_businesses:
            continue
        matched_need_ids = [
            str(need_by_plan_index[int(match["plan_index"])]["need_id"])
            for match in row.get("matched_queries") or []
            if int(match.get("plan_index") or 0) in need_by_plan_index
        ]
        optional_rows.append({
            **row,
            "selection_type": "GLOBAL_OPTIONAL",
            "need_ids": list(dict.fromkeys(matched_need_ids)),
            "need_queries": [
                str(next(value for value in needs if value["need_id"] == need_id)["question"])
                for need_id in dict.fromkeys(matched_need_ids)
            ],
            "need_businesses": [chunk_business] if chunk_business else [],
            "global_reranker_norm": float(rerank_value),
            "query_fusion_norm": float(rrf_value),
            "composite_score": (
                NEED_BATCH_GLOBAL_RERANK_WEIGHT_V5 * float(rerank_value)
                + NEED_BATCH_GLOBAL_RRF_WEIGHT_V5 * float(rrf_value)
            ),
            "reranker_score": float(row["global_reranker_score"]),
        })
    optional_rows.sort(
        key=lambda row: (-float(row["composite_score"]), -float(row["global_reranker_score"]), str(row["chunk_id"]))
    )
    for row in optional_rows:
        if len(selected) >= NEED_BATCH_MAX_EVIDENCE_V5:
            break
        if not _can_select_need_row_v5(
            row,
            selected_chunks=selected_chunks,
            selected_parents=selected_parents,
            selected_urls=selected_urls,
        ):
            continue
        selected.append(row)
        _mark_selected_need_row_v5(
            row,
            selected_chunks=selected_chunks,
            selected_parents=selected_parents,
            selected_urls=selected_urls,
        )

    # 필수 선택이 부족해도 검색을 중단하지 않고, 안전하게 확보된 결과로 답변을 만들며 상태를 기록합니다.
    final = []
    for rank, row in enumerate(selected[:NEED_BATCH_MAX_EVIDENCE_V5], start=1):
        final.append({
            **row,
            "rank": rank,
            "minmax_score": float(row.get("best_minmax_score") or row.get("minmax_score") or 0.0),
        })
    if not final:
        raise RuntimeError("Need-aware 선택 후 남은 검색 근거가 없습니다.")

    _LAST_NEED_BATCH_CONTEXT_V5 = {
        "strategy": "NEED_BATCH_RERANK_V5",
        "original_question": original_question,
        "needs": coverage_rows,
        "required_top_k": required_top_k,
        "max_evidence": NEED_BATCH_MAX_EVIDENCE_V5,
        "selected_count": len(final),
    }
    _LAST_RERANK_TRACE = {
        "latency_ms": rerank_latency_ms,
        "model": RERANKER_MODEL_NAME,
        "device": RERANKER_DEVICE,
        "strategy": "NEED_BATCH_RERANK_V5",
        "candidate_count": len(pair_texts),
        "returned_count": len(final),
        "batch_size": RERANKER_BATCH_SIZE,
        "need_count": len(needs),
        "need_pair_count": sum(1 for ref in pair_refs if ref["kind"] == "need"),
        "global_pair_count": sum(1 for ref in pair_refs if ref["kind"] == "global"),
        "required_top_k": required_top_k,
        "max_evidence": NEED_BATCH_MAX_EVIDENCE_V5,
        "need_coverage": coverage_rows,
        "all_needs_sufficient": all(bool(row["sufficient"]) for row in coverage_rows),
        "global_composite_weights": {
            "original_reranker": NEED_BATCH_GLOBAL_RERANK_WEIGHT_V5,
            "weighted_rrf": NEED_BATCH_GLOBAL_RRF_WEIGHT_V5,
        },
    }
    for row in per_query:
        row["query_fusion_latency_ms"] = fusion_latency_ms
        row["reranker_latency_ms"] = rerank_latency_ms
        row["reranker_candidate_count"] = len(pair_texts)
    return final, per_query


def build_compact_parent_evidence_pack_v1(
    question: str,
    search_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """최대 6개 Parent 근거와 need-근거 매핑을 보존하는 V5 Evidence Pack."""
    by_parent: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for result in search_results:
        child = dict(result.get("chunk") or {})
        child_id = str(result.get("chunk_id") or child.get("chunk_id") or "")
        parent_id = str(result.get("parent_doc_id") or _parent_id_for_chunk(child))
        row = by_parent.setdefault(parent_id, {
            "rank": int(result.get("rank") or len(by_parent) + 1),
            "parent_id": parent_id,
            "representative_chunk_id": child_id,
            "matched_child_ids": [],
            "matched_child_ranks": [],
            "context_chunk_ids": list(result.get("parent_context_chunk_ids") or [child_id]),
            "document_title": _clean_text(child.get("title") or child.get("document_title")),
            "source_url": _clean_text(child.get("source_url")),
            "need_ids": [],
            "need_queries": [],
            "need_businesses": [],
            "selection_types": [],
        })
        row["matched_child_ids"].append(child_id)
        row["matched_child_ranks"].append(int(result.get("rank") or 0))
        row["need_ids"].extend(str(value) for value in result.get("need_ids") or [])
        row["need_queries"].extend(str(value) for value in result.get("need_queries") or [])
        row["need_businesses"].extend(str(value) for value in result.get("need_businesses") or [])
        row["selection_types"].append(str(result.get("selection_type") or "STANDARD"))

    evidence: list[dict[str, Any]] = []
    sources: OrderedDict[str, dict[str, Any]] = OrderedDict()
    total_remaining = ANSWER_EVIDENCE_TOTAL_MAX_CHARS
    for parent_index, row in enumerate(by_parent.values()):
        if total_remaining <= 0 or parent_index >= len(ANSWER_EVIDENCE_RANK_BUDGETS_V5):
            break
        parent_budget = min(ANSWER_EVIDENCE_RANK_BUDGETS_V5[parent_index], total_remaining)
        ordered_ids = _proximity_order_v1(row["context_chunk_ids"], row["matched_child_ids"])
        parts: list[str] = []
        included_ids: list[str] = []
        section_titles: list[str] = []
        remaining = parent_budget
        for chunk_id in ordered_ids:
            chunk = CHUNKS_BY_ID.get(str(chunk_id))
            if chunk is None:
                continue
            title = _clean_text(chunk.get("title"))
            section = _clean_text(chunk.get("section_title"))
            if section and section not in section_titles:
                section_titles.append(section)
            label = " / ".join(value for value in (title, section) if value)
            part = f"[{chunk_id}] {label}\n{_clean_text(chunk.get('content'))}".strip()
            separator_cost = 2 if parts else 0
            if remaining <= separator_cost:
                break
            part = _truncate_at_boundary_v1(part, remaining - separator_cost)
            if not part:
                break
            parts.append(part)
            included_ids.append(str(chunk_id))
            remaining -= len(part) + separator_cost
            if remaining < 120:
                break
        content = "\n\n".join(parts)
        if not content:
            continue
        evidence_id = f"E{len(evidence) + 1}"
        evidence.append({
            "evidence_id": evidence_id,
            "rank": int(row["rank"]),
            "chunk_id": row["representative_chunk_id"],
            "parent_id": row["parent_id"],
            "context_chunk_ids": included_ids,
            "matched_child_ids": list(dict.fromkeys(row["matched_child_ids"])),
            "matched_child_ranks": sorted(set(row["matched_child_ranks"])),
            "document_title": row["document_title"],
            "section_title": " · ".join(section_titles),
            "content": content,
            "context_char_count": len(content),
            "context_truncated": len(included_ids) < len(ordered_ids),
            "source_url": row["source_url"],
            "need_ids": list(dict.fromkeys(row["need_ids"])),
            "need_queries": list(dict.fromkeys(row["need_queries"])),
            "need_businesses": list(dict.fromkeys(row["need_businesses"])),
            "selection_types": list(dict.fromkeys(row["selection_types"])),
        })
        total_remaining -= len(content)
        if row["source_url"]:
            source = sources.setdefault(row["source_url"], {
                "source_id": f"S{len(sources) + 1}",
                "title": row["document_title"] or "공식 출처",
                "source_url": row["source_url"],
                "evidence_ids": [],
            })
            source["evidence_ids"].append(evidence_id)

    if not evidence:
        raise ValueError("V5 Evidence Pack을 만들 근거가 없습니다.")
    evidence_ids = {row["evidence_id"] for row in evidence}
    need_rows = []
    for need in _LAST_NEED_BATCH_CONTEXT_V5.get("needs") or []:
        linked = [
            row["evidence_id"] for row in evidence
            if str(need.get("need_id")) in set(row.get("need_ids") or [])
        ]
        need_rows.append({
            **dict(need),
            "evidence_ids": [value for value in linked if value in evidence_ids],
        })
    shared_evidence_ids = [
        row["evidence_id"] for row in evidence
        if not row.get("need_ids") or "GLOBAL_OPTIONAL" in set(row.get("selection_types") or [])
    ]
    return {
        "question": _clean_text(question),
        "retrieval_strategy": str(_LAST_NEED_BATCH_CONTEXT_V5.get("strategy") or "V4_1_STANDARD_RERANK"),
        "search_parent_context_max_chars": PARENT_CONTEXT_MAX_CHARS,
        "answer_evidence_total_max_chars": ANSWER_EVIDENCE_TOTAL_MAX_CHARS,
        "answer_prompt_version": ANSWER_PROMPT_VERSION,
        "needs": need_rows,
        "shared_evidence_ids": shared_evidence_ids,
        "evidence": evidence,
        "sources": list(sources.values()),
    }


NEED_COVERAGE_PROMPT_V5 = """
[교차업무 need coverage 규칙]
- Evidence Pack의 needs가 비어 있지 않으면 각 need_id를 독립된 요구로 처리하세요.
- 각 need를 Markdown 소제목으로 구분하고, 해당 need의 evidence_ids에 있는 근거만 우선 사용하세요.
- 어떤 need도 다른 업무의 근거로 대신 답하지 마세요.
- 특정 need의 evidence_ids가 없거나 답할 근거가 부족하면 추측하지 말고 부족한 항목을 명시하세요.
- 하나의 통합 답변을 생성하되 모든 need의 결론이 포함되었는지 확인하세요.
""".strip()
for _prompt_name in (
    "B_STRUCTURED_SYSTEM_PROMPT_V3",
    "C_STRUCTURED_SYSTEM_PROMPT_V3",
    "D_STRUCTURED_FINAL_SYSTEM_PROMPT_V3",
    "D2_SKELETON_SYSTEM_PROMPT",
):
    if _prompt_name in globals() and NEED_COVERAGE_PROMPT_V5 not in str(globals()[_prompt_name]):
        globals()[_prompt_name] = str(globals()[_prompt_name]) + "\n\n" + NEED_COVERAGE_PROMPT_V5


print({
    "cross_business_retrieval": "NEED_BATCH_RERANK_V5",
    "hybrid_per_query_top_k": CANDIDATE_DEPTH,
    "need_required_top_k": NEED_BATCH_REQUIRED_TOP_K_V5,
    "max_evidence": NEED_BATCH_MAX_EVIDENCE_V5,
    "global_composite": {
        "original_reranker": NEED_BATCH_GLOBAL_RERANK_WEIGHT_V5,
        "weighted_rrf": NEED_BATCH_GLOBAL_RRF_WEIGHT_V5,
    },
    "standard_query_retrieval": "V4_1_UNCHANGED",
})

# ==== cell 85 ====
# V5: B안 [E#] 누락 시 로컬 Evidence 귀속 복구
_B_CITATION_TOKEN_PATTERN_V5 = re.compile(r"[가-힣A-Za-z0-9]{2,}")
_B_CITATION_NUMBER_PATTERN_V5 = re.compile(r"\d+(?:[.,]\d+)?")
_B_CITATION_STOPWORDS_V5 = {
    "있습니다", "합니다", "됩니다", "대한", "경우", "위해", "통해", "관련", "다음",
    "사용자", "질문", "답변", "그리고", "또한", "있는", "하는", "해당", "정도",
}


def _recover_answer_text_without_json_v5(raw: str) -> str:
    try:
        text = answer_b_core._decode_raw_answer_text(raw)
    except Exception:
        text = ""
    text = answer_b_core._strip_model_urls(answer_b_core._clean(text))
    if text:
        return text
    match = re.search(
        r'"answer"\s*:\s*"(.*?)(?<!\\)"\s*,\s*"(?:used_evidence_ids|coverage_status|missing_information)"',
        str(raw or ""),
        flags=re.S,
    )
    if match:
        candidate = match.group(1)
        try:
            candidate = json.loads('"' + candidate + '"')
        except Exception:
            candidate = candidate.replace("\\n", "\n").replace('\\"', '"')
        candidate = answer_b_core._strip_model_urls(answer_b_core._clean(candidate))
        if candidate:
            return candidate
    plain = re.sub(r"^```(?:json)?|```$", "", str(raw or "").strip(), flags=re.I).strip()
    if plain and not plain.startswith("{"):
        return answer_b_core._strip_model_urls(answer_b_core._clean(plain))
    raise ValueError("B안 답변 본문을 로컬 복구하지 못했습니다.")


def _citation_tokens_v5(value: Any) -> set[str]:
    return {
        token.lower() for token in _B_CITATION_TOKEN_PATTERN_V5.findall(str(value or ""))
        if token.lower() not in _B_CITATION_STOPWORDS_V5
    }


def _evidence_attribution_score_v5(answer: str, evidence: Mapping[str, Any]) -> float:
    answer_tokens = _citation_tokens_v5(answer)
    title_text = " ".join([
        str(evidence.get("document_title") or ""),
        str(evidence.get("section_title") or ""),
    ])
    content_text = str(evidence.get("content") or "")
    title_overlap = len(answer_tokens.intersection(_citation_tokens_v5(title_text)))
    content_overlap = len(answer_tokens.intersection(_citation_tokens_v5(content_text)))
    answer_numbers = set(_B_CITATION_NUMBER_PATTERN_V5.findall(answer))
    evidence_numbers = set(_B_CITATION_NUMBER_PATTERN_V5.findall(content_text))
    numeric_overlap = len(answer_numbers.intersection(evidence_numbers))
    return 3.0 * title_overlap + 1.0 * content_overlap + 5.0 * numeric_overlap


def _infer_used_evidence_ids_v5(answer: str, pack: Mapping[str, Any]) -> list[str]:
    evidence_rows = [dict(row) for row in pack.get("evidence") or []]
    allowed_ids = {str(row.get("evidence_id") or "") for row in evidence_rows}
    inferred: list[str] = []
    # 교차업무에서는 need별로 최소 한 근거를 귀속해 한쪽 업무만 남는 것을 막습니다.
    for need in pack.get("needs") or []:
        candidate_ids = [str(value) for value in need.get("evidence_ids") or [] if str(value) in allowed_ids]
        candidates = [row for row in evidence_rows if str(row.get("evidence_id")) in candidate_ids]
        if candidates:
            best = max(candidates, key=lambda row: (
                _evidence_attribution_score_v5(answer, row),
                -int(row.get("rank") or 10**9),
            ))
            inferred.append(str(best["evidence_id"]))
    scored = sorted(
        (
            (_evidence_attribution_score_v5(answer, row), int(row.get("rank") or 10**9), str(row.get("evidence_id") or ""))
            for row in evidence_rows
        ),
        key=lambda value: (-value[0], value[1], value[2]),
    )
    for score, _, evidence_id in scored:
        if score <= 0 or not evidence_id or evidence_id in inferred:
            continue
        inferred.append(evidence_id)
        if len(inferred) >= 3:
            break
    if not inferred and evidence_rows:
        inferred = [str(evidence_rows[0].get("evidence_id") or "E1")]
    return list(dict.fromkeys(value for value in inferred if value in allowed_ids))


def _direct_answer_payload_v3(
    raw: str,
    pack: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    allowed = answer_b_core._allowed_evidence(pack)
    local_recovery = False
    recovery_mode = "MODEL_STRUCTURED_CITATION"
    missing: list[str] = []
    try:
        parsed = answer_b_core._extract_json_object(raw)
        answer = answer_b_core._strip_model_urls(answer_b_core._clean(parsed.get("answer")))
        if not answer:
            raise ValueError("answer가 비어 있습니다.")
        used_ids = [
            value for value in answer_b_core._clean_list(parsed.get("used_evidence_ids"))
            if value in allowed
        ]
        if not used_ids:
            used_ids = [
                f"E{number}" for number in re.findall(r"\[E(\d+)\]", answer)
                if f"E{number}" in allowed
            ]
        coverage = str(parsed.get("coverage_status") or "PARTIAL").upper()
        if coverage not in answer_b_core.ALLOWED_COVERAGE_STATUS:
            coverage = "PARTIAL"
        missing = answer_b_core._clean_list(parsed.get("missing_information"))
    except (ValueError, TypeError, json.JSONDecodeError):
        local_recovery = True
        recovery_mode = "LOCAL_BODY_RECOVERY"
        answer = _recover_answer_text_without_json_v5(raw)
        used_ids = [
            f"E{number}" for number in re.findall(r"\[E(\d+)\]", answer)
            if f"E{number}" in allowed
        ]
        coverage = "PARTIAL"
        missing = ["구조화 JSON 메타데이터를 로컬 복구함"]

    if not used_ids and allowed:
        local_recovery = True
        recovery_mode = "LOCAL_EVIDENCE_ATTRIBUTION"
        used_ids = _infer_used_evidence_ids_v5(answer, pack)
        coverage = "PARTIAL"
        missing = list(dict.fromkeys(missing + [
            "모델의 [E#] 인용 누락을 프로그램이 Evidence 유사도로 보완함"
        ]))
    used_ids = list(dict.fromkeys(value for value in used_ids if value in allowed))
    if not used_ids:
        raise ValueError("B안 답변은 생성됐지만 연결할 유효 Evidence가 없습니다.")
    return {
        "answer": answer,
        "used_evidence_ids": used_ids,
        "used_chunk_ids": [allowed[value] for value in used_ids],
        "coverage_status": coverage,
        "missing_information": missing,
        "citation_recovery_mode": recovery_mode,
    }, local_recovery


def run_b_citation_recovery_regression_v5() -> list[dict[str, Any]]:
    pack = {
        "question": "예금보험금 조건과 은닉재산 포상금은?",
        "needs": [
            {"need_id": "N1", "evidence_ids": ["E1"]},
            {"need_id": "N2", "evidence_ids": ["E2"]},
        ],
        "evidence": [
            {
                "evidence_id": "E1", "chunk_id": "DI-test", "rank": 1,
                "document_title": "예금보험금", "section_title": "지급 조건",
                "content": "예금보험금 지급 조건과 신청 절차에 관한 공식 근거",
            },
            {
                "evidence_id": "E2", "chunk_id": "HP-test", "rank": 2,
                "document_title": "은닉재산 신고", "section_title": "포상금",
                "content": "은닉재산 신고 포상금은 회수기여금액을 기준으로 산정한다.",
            },
        ],
    }
    cases = [
        (
            "structured_with_ids",
            json.dumps({
                "answer": "지급 조건은 다음과 같습니다.[E1] 포상금은 회수기여금액 기준입니다.[E2]",
                "used_evidence_ids": ["E1", "E2"],
                "coverage_status": "SUFFICIENT",
                "missing_information": [],
            }, ensure_ascii=False),
            {"E1", "E2"},
            False,
        ),
        (
            "structured_without_e_markers",
            json.dumps({
                "answer": "예금보험금 지급 조건을 확인하고, 은닉재산 신고 포상금은 회수기여금액을 기준으로 봅니다.",
                "used_evidence_ids": [],
                "coverage_status": "SUFFICIENT",
                "missing_information": [],
            }, ensure_ascii=False),
            {"E1", "E2"},
            True,
        ),
        (
            "plain_text_without_e_markers",
            "예금보험금 지급 조건을 확인해야 합니다. 은닉재산 신고 포상금은 회수기여금액 기준입니다.",
            {"E1", "E2"},
            True,
        ),
    ]
    rows = []
    for name, raw, expected, expected_local in cases:
        payload, local = _direct_answer_payload_v3(raw, pack)
        actual = set(payload.get("used_evidence_ids") or [])
        passed = expected.issubset(actual) and local is expected_local
        rows.append({
            "case": name,
            "expected_ids": sorted(expected),
            "actual_ids": sorted(actual),
            "local_recovery": local,
            "recovery_mode": payload.get("citation_recovery_mode"),
            "passed": passed,
        })
    return rows


B_CITATION_RECOVERY_REGRESSION_V5 = run_b_citation_recovery_regression_v5()
if not all(bool(row["passed"]) for row in B_CITATION_RECOVERY_REGRESSION_V5):
    raise RuntimeError("B안 [E#] 누락 복구 회귀 테스트 실패")
display(pd.DataFrame(B_CITATION_RECOVERY_REGRESSION_V5))
print("B안 [E#] 누락 복구 회귀 테스트 통과")

# ==== cell 87 ====
# D-C 1Call vs 2Call 공통 구조와 검증기

import copy
import json
import re
import time
from typing import Any, Mapping, Sequence


DC_PROMPT_VERSION_V1 = "dc-fact-pack-one-vs-two-call-tagged-v2"
DC_ONECALL_MAX_TOKENS_V1 = 2400
DC_SKELETON_MAX_TOKENS_V1 = 1800
DC_FINAL_MAX_TOKENS_V1 = 1600

DC_FACT_SAFETY_RULES_V1 = """
[Fact Index 검증 규칙]
1. 원본 Evidence를 기본 근거로 사용하고 활성화된 Fact Index는 혼동 방지와 검증에만 사용하세요.
2. verified_claims는 해당 claim_id와 source_chunk_ids 범위에서만 사용하세요.
3. forbidden_claims에 해당하는 내용을 주장하지 마세요.
4. Fact Index와 원본 Evidence가 충돌하면 임의로 선택하지 말고 불확실성 또는 충돌로 표시하세요.
5. Fact Index가 없으면 원본 Evidence 범위에서만 답하세요.
6. Fact claim을 사용한 항목에는 [FI-CAND-001:F1] 형태의 실제 ID를 연결하세요.
""".strip()

DC_SKELETON_SYSTEM_PROMPT_V1 = (
    D2_SKELETON_SYSTEM_PROMPT
    + "\n\n"
    + DC_FACT_SAFETY_RULES_V1
    + "\n\n"
    + "ANSWERED와 PARTIAL에는 evidence_ids 또는 fact_claim_ids 중 하나 이상의 실제 근거를 연결하세요."
)

DC_FINAL_SYSTEM_PROMPT_V1 = (
    D_STRUCTURED_FINAL_SYSTEM_PROMPT_V3
    + "\n\n"
    + DC_FACT_SAFETY_RULES_V1
    + "\n최종답변은 Skeleton이 허용한 Evidence ID와 Fact Claim ID만 사용하세요."
)

DC_ONECALL_SYSTEM_PROMPT_V1 = (
    "당신은 예금보험공사 공식 문서 기반 Answer Skeleton 및 최종답변 생성기입니다.\n\n"
    "1. 모든 Answer Need에 정확히 하나의 answer_item을 만드세요.\n"
    "2. 각 item은 ANSWERED, PARTIAL, UNSUPPORTED 중 하나이며 실제 evidence_ids 또는 fact_claim_ids를 연결하세요.\n"
    "3. 동일한 호출에서 Answer Skeleton과 사용자용 Markdown 답변을 함께 생성하세요.\n"
    "4. 최종 answer는 answer_skeleton의 claim·conditions·details와 허용 근거만 사용하세요.\n"
    "5. 한 업무의 조건을 다른 업무에 적용하지 말고, 근거 없는 동시·병행 가능성을 추론하지 마세요.\n"
    "6. 전체를 하나의 JSON으로 감싸지 마세요. 아래 SKELETON_JSON과 FINAL_ANSWER 태그 두 개를 정확히 출력하세요.\n"
    "7. SKELETON_JSON 내부만 유효한 JSON 객체로 작성하고 FINAL_ANSWER에는 일반 Markdown을 작성하세요.\n"
    "8. 코드 펜스와 두 태그 밖의 설명은 출력하지 마세요.\n\n"
    + DC_FACT_SAFETY_RULES_V1
    + "\n\n"
    + MARKDOWN_FORMAT_RULES_V3
    + "\n\n"
    + ACTION_LINK_PROMPT_RULE_V1
    + "\n\n"
    + NEED_COVERAGE_PROMPT_V5
)


def extract_dc_answer_needs_v1(
    question: str,
    pack: Mapping[str, Any],
) -> list[dict[str, Any]]:
    retrieval_needs = [
        dict(row) for row in pack.get("needs") or []
        if str(row.get("need_id") or "") and str(row.get("question") or "")
    ]
    if len(retrieval_needs) >= 2:
        return [
            {
                "need_id": str(row["need_id"]),
                "need_type": "CROSS_BUSINESS",
                "label": str(row.get("business_function") or row.get("question") or row["need_id"]),
                "question_part": str(row.get("question") or ""),
                "retrieval_evidence_ids": list(row.get("evidence_ids") or []),
            }
            for row in retrieval_needs
        ]
    return extract_answer_needs_v2(question)


def validate_dc_skeleton_v1(
    raw: Mapping[str, Any],
    answer_needs: Sequence[Mapping[str, Any]],
    pack: Mapping[str, Any],
) -> dict[str, Any]:
    allowed_evidence = set(answer_b_core._allowed_evidence(pack))
    allowed_facts = set(_allowed_fact_claims_v3(pack))
    raw_items = raw.get("answer_items") if isinstance(raw.get("answer_items"), list) else []
    by_need = {
        str(item.get("need_id") or ""): item
        for item in raw_items
        if isinstance(item, Mapping)
    }
    items: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    invalid_evidence_ids: list[str] = []
    invalid_fact_claim_ids: list[str] = []
    for need in answer_needs:
        need_id = str(need["need_id"])
        source = by_need.get(need_id) or {}
        status = str(source.get("status") or "UNSUPPORTED").upper()
        if status not in NEED_STATUS_VALUES_V2:
            status = "UNSUPPORTED"
        requested_evidence = answer_b_core._clean_list(source.get("evidence_ids"))
        requested_facts = _clean_fact_claim_keys_v3(source.get("fact_claim_ids"))
        invalid_evidence_ids.extend(value for value in requested_evidence if value not in allowed_evidence)
        invalid_fact_claim_ids.extend(value for value in requested_facts if value not in allowed_facts)
        evidence_ids = [value for value in requested_evidence if value in allowed_evidence]
        fact_claim_ids = [value for value in requested_facts if value in allowed_facts]
        claim = answer_b_core._clean(source.get("claim"))
        if status == "ANSWERED" and not evidence_ids and not fact_claim_ids:
            status = "PARTIAL" if claim else "UNSUPPORTED"
        if status == "UNSUPPORTED" and not claim:
            claim = f"{need['label']}은 현재 근거로 확인되지 않습니다."
        item = {
            "need_id": need_id,
            "need_type": str(need.get("need_type") or "GENERAL"),
            "topic": answer_b_core._clean(source.get("topic")) or str(need["label"]),
            "status": status,
            "claim": claim,
            "conditions": answer_b_core._clean_list(source.get("conditions")),
            "details": answer_b_core._clean_list(source.get("details")),
            "evidence_ids": list(dict.fromkeys(evidence_ids)),
            "fact_claim_ids": list(dict.fromkeys(fact_claim_ids)),
            "missing_reason": answer_b_core._clean(source.get("missing_reason")),
        }
        items.append(item)
        coverage_rows.append({
            "need_id": need_id,
            "need_type": item["need_type"],
            "label": str(need["label"]),
            "status": status,
            "evidence_ids": item["evidence_ids"],
            "fact_claim_ids": item["fact_claim_ids"],
            "missing_reason": item["missing_reason"],
        })
    program = calculate_program_coverage_v2(coverage_rows)
    return {
        "core_answer": answer_b_core._clean(raw.get("core_answer")),
        "answer_items": items,
        "need_coverage": coverage_rows,
        "uncertainties": answer_b_core._clean_list(raw.get("uncertainties")),
        "conflicts": answer_b_core._clean_list(raw.get("conflicts")),
        "invalid_evidence_ids": list(dict.fromkeys(invalid_evidence_ids)),
        "invalid_fact_claim_ids": list(dict.fromkeys(invalid_fact_claim_ids)),
        "reference_validation_passed": not invalid_evidence_ids and not invalid_fact_claim_ids,
        **program,
    }


def filter_augmented_pack_for_dc_v1(
    pack: Mapping[str, Any],
    skeleton: Mapping[str, Any],
) -> dict[str, Any]:
    used_evidence_ids = {
        value
        for item in skeleton.get("answer_items") or []
        for value in item.get("evidence_ids") or []
    }
    used_fact_ids = {
        value
        for item in skeleton.get("answer_items") or []
        for value in item.get("fact_claim_ids") or []
    }
    evidence = [
        copy.deepcopy(dict(row))
        for row in pack.get("evidence") or []
        if str(row.get("evidence_id") or "") in used_evidence_ids
    ]
    source_urls = {str(row.get("source_url") or "") for row in evidence}
    fact_index = copy.deepcopy(dict(pack.get("fact_index") or {}))
    filtered_supplements = []
    for supplement in fact_index.get("supplements") or []:
        fact_id = str(supplement.get("fact_index_id") or "")
        selected_claims = [
            copy.deepcopy(dict(claim))
            for claim in supplement.get("verified_claims") or []
            if f"{fact_id}:{claim.get('claim_id')}" in used_fact_ids
        ]
        if selected_claims or supplement.get("forbidden_claims"):
            filtered_supplements.append({
                **copy.deepcopy(dict(supplement)),
                "verified_claims": selected_claims,
            })
    fact_index["supplements"] = filtered_supplements
    fact_index["supplement_count"] = len(filtered_supplements)
    return {
        **copy.deepcopy(dict(pack)),
        "evidence": evidence,
        "sources": [
            copy.deepcopy(dict(row))
            for row in pack.get("sources") or []
            if str(row.get("source_url") or "") in source_urls
        ],
        "fact_index": fact_index,
        "filtered_for_dc": True,
        "original_evidence_count": len(pack.get("evidence") or []),
        "selected_evidence_count": len(evidence),
        "selected_fact_claim_count": len(used_fact_ids),
    }


def _extract_json_relaxed_dc_v1(raw: str) -> dict[str, Any]:
    try:
        value = answer_b_core._extract_json_object(raw)
        if isinstance(value, Mapping):
            return dict(value)
    except Exception:
        pass
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(raw or "").strip(), flags=re.I | re.S)
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start:end + 1]
        try:
            value = json.loads(candidate, strict=False)
        except Exception as error:
            raise ValueError(f"D-C JSON 로컬 복구 실패: {error}") from error
        if isinstance(value, Mapping):
            return dict(value)
    raise ValueError("D-C 구조화 JSON 객체를 찾지 못했습니다.")


def _onecall_raw_preview_dc_v2(raw: Any, limit: int = 900) -> str:
    value = str(raw or "").replace("\x00", "").strip()
    return value[:limit].replace("\r", "\\r").replace("\n", "\\n")


def _parse_onecall_output_dc_v2(raw: str) -> dict[str, Any]:
    """태그 형식을 우선 처리하고 기존 중첩 JSON도 하위 호환으로 허용합니다."""
    text = str(raw or "").strip()
    if not text:
        raise ValueError("ONECALL_OUTPUT_CONTRACT_FAILED: HCX 응답이 비어 있습니다.")

    # v1.1 형식과 이미 정상적으로 중첩 JSON을 반환하는 응답을 계속 지원합니다.
    try:
        legacy = _extract_json_relaxed_dc_v1(text)
        legacy_skeleton = legacy.get("answer_skeleton") or legacy.get("skeleton")
        if isinstance(legacy_skeleton, Mapping) and str(legacy.get("answer") or "").strip():
            return {
                **legacy,
                "answer_skeleton": dict(legacy_skeleton),
                "answer": str(legacy.get("answer") or "").strip(),
                "output_contract": "LEGACY_NESTED_JSON",
            }
    except Exception:
        pass

    skeleton_match = re.search(
        r"<SKELETON_JSON>\s*(.*?)\s*</SKELETON_JSON>",
        text,
        flags=re.I | re.S,
    )
    answer_match = re.search(
        r"<FINAL_ANSWER>\s*(.*?)\s*</FINAL_ANSWER>",
        text,
        flags=re.I | re.S,
    )
    missing_tags = []
    if skeleton_match is None:
        missing_tags.append("SKELETON_JSON")
    if answer_match is None:
        missing_tags.append("FINAL_ANSWER")
    if missing_tags:
        raise ValueError(
            "ONECALL_OUTPUT_CONTRACT_FAILED: 필수 태그 누락="
            + ",".join(missing_tags)
            + "; raw_preview="
            + _onecall_raw_preview_dc_v2(text)
        )

    skeleton_text = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        skeleton_match.group(1).strip(),
        flags=re.I | re.S,
    ).strip()
    try:
        skeleton = _extract_json_relaxed_dc_v1(skeleton_text)
    except Exception as error:
        raise ValueError(
            "ONECALL_OUTPUT_CONTRACT_FAILED: SKELETON_JSON 파싱 실패="
            + str(error)
            + "; raw_preview="
            + _onecall_raw_preview_dc_v2(text)
        ) from error
    answer = answer_match.group(1).strip()
    if not answer:
        raise ValueError(
            "ONECALL_OUTPUT_CONTRACT_FAILED: FINAL_ANSWER가 비어 있습니다.; raw_preview="
            + _onecall_raw_preview_dc_v2(text)
        )
    return {
        "answer_skeleton": skeleton,
        "answer": answer,
        "used_evidence_ids": _DC_EVIDENCE_REF_PATTERN_V1.findall(answer),
        "used_fact_claim_ids": _DC_FACT_REF_PATTERN_V1.findall(answer),
        "output_contract": "TAGGED_SKELETON_AND_MARKDOWN_V2",
    }


_DC_EVIDENCE_REF_PATTERN_V1 = re.compile(r"\[(E\d+)\]")
_DC_FACT_REF_PATTERN_V1 = re.compile(r"\[((?:FI-[A-Z0-9-]+):F\d+)\]", re.I)


def audit_dc_final_references_v1(
    answer: str,
    skeleton: Mapping[str, Any],
    pack: Mapping[str, Any],
    *,
    explicit_evidence_ids: Sequence[str] = (),
    explicit_fact_claim_ids: Sequence[str] = (),
) -> dict[str, Any]:
    allowed_evidence = set(answer_b_core._allowed_evidence(pack))
    allowed_facts = set(_allowed_fact_claims_v3(pack))
    skeleton_evidence = {
        value for item in skeleton.get("answer_items") or [] for value in item.get("evidence_ids") or []
    }
    skeleton_facts = {
        value for item in skeleton.get("answer_items") or [] for value in item.get("fact_claim_ids") or []
    }
    requested_evidence = list(explicit_evidence_ids) + _DC_EVIDENCE_REF_PATTERN_V1.findall(str(answer or ""))
    requested_facts = list(explicit_fact_claim_ids) + _DC_FACT_REF_PATTERN_V1.findall(str(answer or ""))
    requested_evidence = list(dict.fromkeys(str(value) for value in requested_evidence))
    requested_facts = list(dict.fromkeys(str(value) for value in requested_facts))
    local_recovery = False
    if not requested_evidence and skeleton_evidence:
        requested_evidence = sorted(skeleton_evidence)
        local_recovery = True
    if not requested_facts and skeleton_facts:
        requested_facts = sorted(skeleton_facts)
        local_recovery = True
    invalid_evidence = [value for value in requested_evidence if value not in allowed_evidence]
    invalid_facts = [value for value in requested_facts if value not in allowed_facts]
    outside_skeleton_evidence = [value for value in requested_evidence if value in allowed_evidence and value not in skeleton_evidence]
    outside_skeleton_facts = [value for value in requested_facts if value in allowed_facts and value not in skeleton_facts]
    return {
        "used_evidence_ids": [value for value in requested_evidence if value in allowed_evidence and value in skeleton_evidence],
        "used_fact_claim_ids": [value for value in requested_facts if value in allowed_facts and value in skeleton_facts],
        "invalid_evidence_ids": invalid_evidence,
        "invalid_fact_claim_ids": invalid_facts,
        "outside_skeleton_evidence_ids": outside_skeleton_evidence,
        "outside_skeleton_fact_claim_ids": outside_skeleton_facts,
        "local_reference_recovery": local_recovery,
        "reference_consistency_passed": not (
            invalid_evidence or invalid_facts or outside_skeleton_evidence or outside_skeleton_facts
        ),
    }


def audit_numeric_support_dc_v1(answer: str, pack: Mapping[str, Any]) -> dict[str, Any]:
    augmented = copy.deepcopy(dict(pack))
    evidence = list(augmented.get("evidence") or [])
    fact_statements = [
        str(claim.get("statement") or "")
        for supplement in (augmented.get("fact_index") or {}).get("supplements") or []
        for claim in supplement.get("verified_claims") or []
    ]
    if fact_statements:
        evidence.append({"evidence_id": "FACT-INDEX", "content": " ".join(fact_statements)})
    augmented["evidence"] = evidence
    return audit_numeric_support_v2(answer, augmented)


def audit_forbidden_claims_dc_v1(answer: str, pack: Mapping[str, Any]) -> dict[str, Any]:
    normalized_answer = re.sub(r"\s+", "", str(answer or "")).lower()
    hits = []
    for supplement in (pack.get("fact_index") or {}).get("supplements") or []:
        fact_id = str(supplement.get("fact_index_id") or "")
        for claim in supplement.get("forbidden_claims") or []:
            normalized_claim = re.sub(r"\s+", "", str(claim or "")).lower()
            if len(normalized_claim) >= 8 and normalized_claim in normalized_answer:
                hits.append({"fact_index_id": fact_id, "forbidden_claim": str(claim)})
    return {"forbidden_claim_hits": hits, "forbidden_claim_check_passed": not hits}


def _dc_skeleton_prompt_v1(
    question: str,
    answer_needs: Sequence[Mapping[str, Any]],
    relation_constraint: Mapping[str, Any],
    augmented_pack: Mapping[str, Any],
) -> str:
    return f"""[사용자 질문]\n{_clean_text(question)}\n\n[Answer Needs]\n{_compact_json(answer_needs)}\n\n[관계 주장 안전조건]\n{_compact_json(relation_constraint)}\n\n[C안 Fact Index 보강 Evidence Pack]\n{_compact_json(augmented_pack)}\n\n[출력 JSON]\n{{\"core_answer\":\"핵심 결론\",\"answer_items\":[{{\"need_id\":\"N1\",\"topic\":\"항목\",\"status\":\"ANSWERED|PARTIAL|UNSUPPORTED\",\"claim\":\"근거 사실\",\"conditions\":[],\"details\":[],\"evidence_ids\":[\"E1\"],\"fact_claim_ids\":[\"FI-CAND-001:F1\"],\"missing_reason\":\"\"}}],\"uncertainties\":[],\"conflicts\":[]}}"""


def generate_dc_twocall_v1(
    question: str,
    augmented_pack: Mapping[str, Any],
) -> dict[str, Any]:
    total_started = time.perf_counter()
    need_started = time.perf_counter()
    answer_needs = extract_dc_answer_needs_v1(question, augmented_pack)
    relation_constraint = relation_constraint_v1(question, augmented_pack)
    need_ms = (time.perf_counter() - need_started) * 1000

    skeleton_prompt_started = time.perf_counter()
    skeleton_prompt = _dc_skeleton_prompt_v1(question, answer_needs, relation_constraint, augmented_pack)
    skeleton_prompt_ms = (time.perf_counter() - skeleton_prompt_started) * 1000
    raw_skeleton, usage1, skeleton_api_ms, trace1 = _call_answer_api_v1(
        system_prompt=_managed_prompt("DC_SKELETON_SYSTEM_PROMPT_V1", DC_SKELETON_SYSTEM_PROMPT_V1),
        user_prompt=skeleton_prompt,
        max_tokens=DC_SKELETON_MAX_TOKENS_V1,
    )

    validation_started = time.perf_counter()
    skeleton = validate_dc_skeleton_v1(
        _extract_json_relaxed_dc_v1(raw_skeleton), answer_needs, augmented_pack
    )
    validation_ms = (time.perf_counter() - validation_started) * 1000
    selection_started = time.perf_counter()
    selected_pack = filter_augmented_pack_for_dc_v1(augmented_pack, skeleton)
    selection_ms = (time.perf_counter() - selection_started) * 1000

    final_prompt_started = time.perf_counter()
    final_prompt = f"""[사용자 질문]\n{_clean_text(question)}\n\n[Answer Needs]\n{_compact_json(answer_needs)}\n\n[관계 주장 안전조건]\n{_compact_json(relation_constraint)}\n\n[검증된 Answer Skeleton]\n{_compact_json(skeleton)}\n\n[Skeleton 선택 C안 Evidence Pack]\n{_compact_json(selected_pack)}\n\n모든 Answer Need를 반영한 최종 Markdown 답변만 작성하세요."""
    final_prompt_ms = (time.perf_counter() - final_prompt_started) * 1000
    raw_answer, usage2, final_api_ms, trace2 = _call_answer_api_v1(
        system_prompt=_managed_prompt("DC_FINAL_SYSTEM_PROMPT_V1", DC_FINAL_SYSTEM_PROMPT_V1),
        user_prompt=final_prompt,
        max_tokens=DC_FINAL_MAX_TOKENS_V1,
    )

    post_started = time.perf_counter()
    raw_answer = answer_b_core._strip_model_urls(str(raw_answer).strip())
    if not raw_answer:
        raise ValueError("D-C 2Call 최종 답변이 비어 있습니다.")
    reference_audit = audit_dc_final_references_v1(raw_answer, skeleton, augmented_pack)
    safe_answer, guard_applied = _relation_safe_answer_v1(raw_answer, relation_constraint)
    safe_answer = normalize_answer_markdown_v3(safe_answer)
    numeric_audit = audit_numeric_support_dc_v1(safe_answer, selected_pack)
    forbidden_audit = audit_forbidden_claims_dc_v1(safe_answer, selected_pack)
    post_ms = (time.perf_counter() - post_started) * 1000

    skeleton_trace = _trace_parts_v3(trace1)
    final_trace = _trace_parts_v3(trace2)
    total_ms = (time.perf_counter() - total_started) * 1000
    validation_passed = bool(
        skeleton.get("reference_validation_passed")
        and reference_audit["reference_consistency_passed"]
        and numeric_audit["numeric_support_passed"]
        and forbidden_audit["forbidden_claim_check_passed"]
        and not guard_applied
    )
    coverage = skeleton["coverage_status"]
    if guard_applied or not validation_passed:
        coverage = "PARTIAL" if coverage != "INSUFFICIENT" else coverage
    return {
        "system": "D-C 2Call",
        "answer": safe_answer,
        "answer_needs": answer_needs,
        "skeleton": skeleton,
        "need_coverage": skeleton["need_coverage"],
        "coverage_status": coverage,
        "strict_need_coverage_rate": skeleton["strict_need_coverage_rate"],
        "answerable_need_coverage_rate": skeleton["answerable_need_coverage_rate"],
        "selected_evidence_pack": selected_pack,
        "selected_evidence_count": len(selected_pack.get("evidence") or []),
        "used_evidence_ids": reference_audit["used_evidence_ids"],
        "used_fact_claim_ids": reference_audit["used_fact_claim_ids"],
        "reference_audit": reference_audit,
        "numeric_audit": numeric_audit,
        "forbidden_claim_audit": forbidden_audit,
        "validation_passed": validation_passed,
        "relation_constraint": relation_constraint,
        "relation_guard_applied": guard_applied,
        "latency_ms": total_ms,
        "skeleton_latency_ms": skeleton_api_ms,
        "final_latency_ms": final_api_ms,
        "usage": _merge_usage_v1(usage1, usage2),
        "api_calls": 2,
        "attempts": [
            {"stage": "dc_skeleton", "latency_ms": skeleton_api_ms, "trace": trace1},
            {"stage": "dc_final", "latency_ms": final_api_ms, "trace": trace2},
        ],
        "stage_latency_ms": {
            "need_extraction_ms": need_ms,
            "skeleton_prompt_build_ms": skeleton_prompt_ms,
            "skeleton_api_wall_ms": skeleton_trace["api_wall_ms"],
            "skeleton_pacing_wait_ms": skeleton_trace["pacing_wait_ms"],
            "skeleton_retry_wait_ms": skeleton_trace["retry_wait_ms"],
            "skeleton_estimated_service_ms": skeleton_trace["estimated_service_ms"],
            "skeleton_validation_ms": validation_ms,
            "evidence_selection_ms": selection_ms,
            "final_prompt_build_ms": final_prompt_ms,
            "final_api_wall_ms": final_trace["api_wall_ms"],
            "final_pacing_wait_ms": final_trace["pacing_wait_ms"],
            "final_retry_wait_ms": final_trace["retry_wait_ms"],
            "final_estimated_service_ms": final_trace["estimated_service_ms"],
            "postprocess_ms": post_ms,
            "answer_total_ms": total_ms,
        },
    }


def generate_dc_onecall_v1(
    question: str,
    augmented_pack: Mapping[str, Any],
) -> dict[str, Any]:
    total_started = time.perf_counter()
    need_started = time.perf_counter()
    answer_needs = extract_dc_answer_needs_v1(question, augmented_pack)
    relation_constraint = relation_constraint_v1(question, augmented_pack)
    need_ms = (time.perf_counter() - need_started) * 1000

    prompt_started = time.perf_counter()
    prompt = f"""[사용자 질문]\n{_clean_text(question)}\n\n[Answer Needs]\n{_compact_json(answer_needs)}\n\n[관계 주장 안전조건]\n{_compact_json(relation_constraint)}\n\n[C안 Fact Index 보강 Evidence Pack]\n{_compact_json(augmented_pack)}\n\n[반드시 지킬 출력 형식]\n<SKELETON_JSON>\n{{\"core_answer\":\"핵심 결론\",\"answer_items\":[{{\"need_id\":\"N1\",\"topic\":\"항목\",\"status\":\"ANSWERED|PARTIAL|UNSUPPORTED\",\"claim\":\"근거 사실\",\"conditions\":[],\"details\":[],\"evidence_ids\":[\"E1\"],\"fact_claim_ids\":[\"FI-CAND-001:F1\"],\"missing_reason\":\"\"}}],\"uncertainties\":[],\"conflicts\":[]}}\n</SKELETON_JSON>\n<FINAL_ANSWER>\n결론을 먼저 작성한 최종 Markdown 답변. 근거 문장에는 [E1] 또는 [FI-CAND-001:F1]을 표시합니다.\n</FINAL_ANSWER>"""
    prompt_ms = (time.perf_counter() - prompt_started) * 1000
    raw, usage, api_ms, trace = _call_answer_api_v1(
        system_prompt=DC_ONECALL_SYSTEM_PROMPT_V1,
        user_prompt=prompt,
        max_tokens=DC_ONECALL_MAX_TOKENS_V1,
    )

    validation_started = time.perf_counter()
    parsed = _parse_onecall_output_dc_v2(raw)
    raw_skeleton = parsed.get("answer_skeleton") or {}
    if not isinstance(raw_skeleton, Mapping):
        raise ValueError("D-C 1Call answer_skeleton이 JSON 객체가 아닙니다.")
    skeleton = validate_dc_skeleton_v1(raw_skeleton, answer_needs, augmented_pack)
    # JSON answer 문자열의 Markdown 줄바꿈을 보존합니다. _clean()은 모든 공백을
    # 한 칸으로 합치므로 번호 목록과 글머리표가 한 문단이 될 수 있습니다.
    raw_answer = answer_b_core._strip_model_urls(str(parsed.get("answer") or "").strip())
    if not raw_answer:
        raise ValueError("D-C 1Call 최종 답변이 비어 있습니다.")
    reference_audit = audit_dc_final_references_v1(
        raw_answer,
        skeleton,
        augmented_pack,
        explicit_evidence_ids=answer_b_core._clean_list(parsed.get("used_evidence_ids")),
        explicit_fact_claim_ids=_clean_fact_claim_keys_v3(parsed.get("used_fact_claim_ids")),
    )
    selected_pack = filter_augmented_pack_for_dc_v1(augmented_pack, skeleton)
    validation_ms = (time.perf_counter() - validation_started) * 1000

    post_started = time.perf_counter()
    safe_answer, guard_applied = _relation_safe_answer_v1(raw_answer, relation_constraint)
    safe_answer = normalize_answer_markdown_v3(safe_answer)
    numeric_audit = audit_numeric_support_dc_v1(safe_answer, selected_pack)
    forbidden_audit = audit_forbidden_claims_dc_v1(safe_answer, selected_pack)
    post_ms = (time.perf_counter() - post_started) * 1000
    trace_parts = _trace_parts_v3(trace)
    total_ms = (time.perf_counter() - total_started) * 1000
    validation_passed = bool(
        skeleton.get("reference_validation_passed")
        and reference_audit["reference_consistency_passed"]
        and numeric_audit["numeric_support_passed"]
        and forbidden_audit["forbidden_claim_check_passed"]
        and not guard_applied
    )
    requested_coverage = str(parsed.get("coverage_status") or skeleton["coverage_status"]).upper()
    coverage = skeleton["coverage_status"]
    if requested_coverage == "INSUFFICIENT" and coverage == "SUFFICIENT":
        coverage = "PARTIAL"
    if guard_applied or not validation_passed:
        coverage = "PARTIAL" if coverage != "INSUFFICIENT" else coverage
    return {
        "system": "D-C 1Call",
        "answer": safe_answer,
        "answer_needs": answer_needs,
        "skeleton": skeleton,
        "need_coverage": skeleton["need_coverage"],
        "coverage_status": coverage,
        "strict_need_coverage_rate": skeleton["strict_need_coverage_rate"],
        "answerable_need_coverage_rate": skeleton["answerable_need_coverage_rate"],
        "selected_evidence_pack": selected_pack,
        "selected_evidence_count": len(selected_pack.get("evidence") or []),
        "used_evidence_ids": reference_audit["used_evidence_ids"],
        "used_fact_claim_ids": reference_audit["used_fact_claim_ids"],
        "reference_audit": reference_audit,
        "numeric_audit": numeric_audit,
        "forbidden_claim_audit": forbidden_audit,
        "validation_passed": validation_passed,
        "output_contract": parsed.get("output_contract"),
        "relation_constraint": relation_constraint,
        "relation_guard_applied": guard_applied,
        "latency_ms": total_ms,
        "skeleton_latency_ms": api_ms,
        "final_latency_ms": 0.0,
        "usage": usage,
        "api_calls": 1,
        "attempts": [{"stage": "dc_skeleton_and_final", "latency_ms": api_ms, "trace": trace}],
        "stage_latency_ms": {
            "need_extraction_ms": need_ms,
            "combined_prompt_build_ms": prompt_ms,
            "combined_api_wall_ms": trace_parts["api_wall_ms"],
            "combined_pacing_wait_ms": trace_parts["pacing_wait_ms"],
            "combined_retry_wait_ms": trace_parts["retry_wait_ms"],
            "combined_estimated_service_ms": trace_parts["estimated_service_ms"],
            "combined_json_skeleton_validation_ms": validation_ms,
            "postprocess_ms": post_ms,
            "answer_total_ms": total_ms,
        },
    }


print({
    "comparison": ["D-C 2Call", "D-C 1Call"],
    "shared_input": "C Fact Index augmented Evidence Pack",
    "cross_business_needs": "V5 retrieval needs",
    "onecall_max_tokens": DC_ONECALL_MAX_TOKENS_V1,
    "twocall_max_tokens": DC_SKELETON_MAX_TOKENS_V1 + DC_FINAL_MAX_TOKENS_V1,
})

# ==== cell 88 ====
# 공통 검색·Fact Index 1회 캐시와 D-C 개별 실행기
def new_dc_controller_state_v1() -> dict[str, Any]:
    state = new_bcd_controller_state_c1()
    state["events"] = []
    return state


def _prepare_dc_common_v1(
    question: str,
    holder: dict[str, Any],
) -> tuple[dict[str, Any], bool, float, float]:
    common, cache_hit, common_this_click_ms = _prepare_or_reuse_common_v3(question, holder)
    fact_this_click_ms = 0.0
    if common.get("route") == "RETRIEVE" and "dc_augmented_pack" not in common:
        started = time.perf_counter()
        matched_records, fact_audit = match_fact_index_c1(common)
        augmented_pack = build_fact_augmented_pack_c1(common["evidence_pack"], matched_records)
        fact_this_click_ms = (time.perf_counter() - started) * 1000
        common["dc_matched_fact_records"] = matched_records
        common["dc_fact_audit"] = fact_audit
        common["dc_augmented_pack"] = augmented_pack
        common["dc_augmented_pack_sha256"] = _stable_json_hash_c1(augmented_pack)
        common.setdefault("latency_ms", {})["Fact Index 보강"] = fact_this_click_ms
        common["latency_ms"]["공통 준비 전체"] = (
            float(common["latency_ms"].get("공통 준비 전체") or 0) + fact_this_click_ms
        )
        common_this_click_ms += fact_this_click_ms
    return common, cache_hit, common_this_click_ms, fact_this_click_ms


def _dc_answer_cache_key_v1(
    variant: str,
    common: Mapping[str, Any],
) -> str:
    return _stable_json_hash_c1({
        "variant": variant,
        "resolved_question": common.get("resolved_question"),
        "augmented_pack_sha256": common.get("dc_augmented_pack_sha256"),
        "prompt_version": DC_PROMPT_VERSION_V1,
    })


def _official_sources_dc_v1(common: Mapping[str, Any]) -> list[dict[str, str]]:
    sources = []
    seen = set()
    for row in (common.get("evidence_pack") or {}).get("sources") or []:
        url = str(row.get("source_url") or "")
        if url and url not in seen:
            seen.add(url)
            sources.append({"title": str(row.get("title") or "공식 출처"), "url": url})
    for record in common.get("dc_matched_fact_records") or []:
        title = str(record.get("document_title") or record.get("fact_index_id") or "Fact Index 공식 근거")
        for url in record.get("source_urls") or []:
            url = str(url)
            if url and url not in seen:
                seen.add(url)
                sources.append({"title": title, "url": url})
    return sources


def execute_dc_variant_v1(
    variant: str,
    question: str,
    holder: dict[str, Any],
    *,
    force_answer_regeneration: bool = False,
) -> dict[str, Any]:
    variant = str(variant).upper()
    if variant not in {"DC_2CALL", "DC_1CALL"}:
        raise ValueError(f"지원하지 않는 D-C 답변안: {variant}")
    click_started = time.perf_counter()
    gate_start = len(HCX_SHARED_GATE_V3.history)
    common, common_cache_hit, common_this_click_ms, fact_this_click_ms = _prepare_dc_common_v1(
        question, holder
    )
    if common.get("route") != "RETRIEVE":
        return {
            "variant": variant,
            "route": common.get("route"),
            "route_message": common.get("route_message"),
            "common": common,
            "common_cache_hit": common_cache_hit,
            "answer_cache_hit": False,
            "latency": {
                "common_this_click_ms": common_this_click_ms,
                "fact_index_match_ms": fact_this_click_ms,
                "answer_ms": 0.0,
                "click_wall_ms": (time.perf_counter() - click_started) * 1000,
            },
            "api_trace": _trace_summary_since_v3(gate_start),
        }

    cache_key = _dc_answer_cache_key_v1(variant, common)
    cached = holder["answer_cache"].get(cache_key)
    if cached is not None and not force_answer_regeneration:
        payload = copy.deepcopy(cached)
        answer_cache_hit = True
        answer_ms = 0.0
    else:
        answer_cache_hit = False
        answer_started = time.perf_counter()
        if variant == "DC_2CALL":
            payload = generate_dc_twocall_v1(
                common["resolved_question"], common["dc_augmented_pack"]
            )
        else:
            payload = generate_dc_onecall_v1(
                common["resolved_question"], common["dc_augmented_pack"]
            )
        answer_ms = (time.perf_counter() - answer_started) * 1000
        holder["answer_cache"][cache_key] = copy.deepcopy(payload)

    if not holder.get("committed"):
        holder["conversation"].setdefault("turns", []).extend([
            {"role": "user", "content": _clean_text(question)},
            {"role": "assistant", "content": normalize_answer_markdown_v3(payload.get("answer"))},
        ])
        holder["committed"] = True
        holder["committed_variant"] = variant

    trace = _trace_summary_since_v3(gate_start)
    usage = _variant_usage_v3(payload)
    stage = payload.get("stage_latency_ms") or {}
    result = {
        "variant": variant,
        "route": "RETRIEVE",
        "common": common,
        "payload": payload,
        "augmented_pack": common["dc_augmented_pack"],
        "matched_fact_records": common.get("dc_matched_fact_records") or [],
        "fact_audit": common.get("dc_fact_audit") or [],
        "common_cache_hit": common_cache_hit,
        "answer_cache_hit": answer_cache_hit,
        "committed_variant": holder.get("committed_variant"),
        "official_sources": _official_sources_dc_v1(common),
        "action_links": action_links_for_streamlit_v1(common.get("action_links") or []),
        "latency": {
            "stored_common_pipeline_ms": float((common.get("latency_ms") or {}).get("공통 준비 전체") or 0),
            "common_this_click_ms": common_this_click_ms,
            "fact_index_match_ms": fact_this_click_ms,
            "answer_ms": answer_ms,
            "click_wall_ms": (time.perf_counter() - click_started) * 1000,
        },
        "api_trace": trace,
        "usage": usage,
        "circuit": hcx_circuit_status_c1(),
    }
    event = {
        "question": _clean_text(question),
        "variant": variant,
        "answer_api_calls": int(payload.get("api_calls") or 0),
        "common_cache_hit": common_cache_hit,
        "answer_cache_hit": answer_cache_hit,
        "fact_index_count": len(result["matched_fact_records"]),
        "fact_index_ids": " | ".join(str(row.get("fact_index_id") or "") for row in result["matched_fact_records"]),
        "stored_common_pipeline_ms": result["latency"]["stored_common_pipeline_ms"],
        "common_this_click_ms": common_this_click_ms,
        "fact_index_match_ms": fact_this_click_ms,
        "answer_total_ms": float(stage.get("answer_total_ms") or 0),
        "skeleton_api_wall_ms": float(stage.get("skeleton_api_wall_ms") or 0),
        "final_api_wall_ms": float(stage.get("final_api_wall_ms") or 0),
        "combined_api_wall_ms": float(stage.get("combined_api_wall_ms") or 0),
        "answer_pacing_wait_ms": sum(float(value) for key, value in stage.items() if key.endswith("pacing_wait_ms")),
        "answer_retry_wait_ms": sum(float(value) for key, value in stage.items() if key.endswith("retry_wait_ms")),
        "answer_estimated_service_ms": sum(float(value) for key, value in stage.items() if key.endswith("estimated_service_ms")),
        "click_wall_ms": result["latency"]["click_wall_ms"],
        **usage,
        "coverage_status": payload.get("coverage_status"),
        "strict_need_coverage_rate": float(payload.get("strict_need_coverage_rate") or 0),
        "answerable_need_coverage_rate": float(payload.get("answerable_need_coverage_rate") or 0),
        "reference_consistency_passed": bool((payload.get("reference_audit") or {}).get("reference_consistency_passed")),
        "numeric_support_passed": bool((payload.get("numeric_audit") or {}).get("numeric_support_passed")),
        "forbidden_claim_check_passed": bool((payload.get("forbidden_claim_audit") or {}).get("forbidden_claim_check_passed")),
        "relation_guard_applied": bool(payload.get("relation_guard_applied")),
        "validation_passed": bool(payload.get("validation_passed")),
        "output_contract": str(payload.get("output_contract") or "TWO_CALL_SEPARATE_OUTPUT"),
        "answer_chars": len(str(payload.get("answer") or "")),
    }
    holder["events"].append(event)
    result["event"] = event
    return result


print("D-C 공통 검색·Fact Index 캐시 및 개별 실행기 준비 완료")

# ==== cell 91 ====
# D안 교차업무 전용 프롬프트·검증·호출 게이트
DC_CROSS_PROMPT_VERSION_V1 = "dc-cross-business-specialized-tagged-v1"
DC_RESPONSE_MODES_V1 = {"SEPARATE", "COMPARE", "RELATION", "SEQUENCE"}


def classify_dc_response_mode_v1(question: str) -> str:
    text = _clean_text(question)
    if re.search(r"동시에|같이|함께|한\s*번에|둘\s*다|모두\s*신청|연계|병행|받을\s*수\s*있", text):
        return "RELATION"
    if re.search(r"차이|다른가|어떻게\s*다르|비교|같은\s*(?:건|것)|구분", text):
        return "COMPARE"
    if re.search(r"먼저|다음|그\s*후|이후|뒤에|하고\s*나서|한\s*뒤|순서", text):
        return "SEQUENCE"
    return "SEPARATE"


def is_cross_business_dc_v1(common: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    analysis = dict(common.get("analysis") or {})
    pack = dict(common.get("evidence_pack") or {})
    businesses = list(dict.fromkeys(str(value) for value in analysis.get("businesses") or [] if str(value)))
    needs = [dict(row) for row in pack.get("needs") or []]
    strategy = str(pack.get("retrieval_strategy") or "")
    passed = bool(
        common.get("route") == "RETRIEVE"
        and len(businesses) >= 2
        and len(needs) >= 2
        and strategy == "NEED_BATCH_RERANK_V5"
    )
    return passed, {
        "passed": passed,
        "business_count": len(businesses),
        "businesses": businesses,
        "need_count": len(needs),
        "need_ids": [str(row.get("need_id") or "") for row in needs],
        "retrieval_strategy": strategy,
        "query_type": str(analysis.get("query_type") or ""),
        "decomposition_accepted": bool(analysis.get("decomposition_or_rewrite_accepted")),
    }


DC_CROSS_SKELETON_RULES_V1 = """
[교차업무 Answer Skeleton 규칙]
1. response_mode은 제공된 expected_response_mode과 동일하게 작성하세요.
2. 모든 업무별 need_id에 정확히 하나의 answer_item을 작성하세요.
3. 각 item에 business_function을 명시하고 그 Need에 연결된 Evidence를 우선 사용하세요.
4. N1의 대상·조건·금액·기간·서류·절차를 N2에 적용하지 마세요.
5. 특정 Need의 근거가 부족하면 다른 Need의 근거로 채우지 말고 PARTIAL 또는 UNSUPPORTED로 표시하세요.
6. SEPARATE는 각 업무를 독립적으로 안내하고 요구하지 않은 비교를 만들지 마세요.
7. COMPARE는 각 업무의 독립 설명을 먼저 확보하고 근거가 있는 차이만 cross_need_relation에 작성하세요.
8. RELATION은 두 업무의 개별 자격만으로 동시·병행·인과 관계를 추론하지 마세요.
9. SEQUENCE는 사용자가 제시한 업무 순서를 보존하고 각 단계의 업무명을 명시하세요.
10. cross_need_relation의 supported=true는 관계를 직접 뒷받침하는 Evidence 또는 Fact Claim이 있을 때만 허용합니다.
""".strip()

DC_CROSS_FINAL_RULES_V1 = """
[교차업무 최종답변 규칙]
- 모든 Need를 업무명 소제목으로 구분하세요.
- 수치·기간·대상·조건·서류 앞에는 적용 업무를 분명히 표시하세요.
- SEPARATE: 업무별 답변만 제공하고 불필요한 공통점·차이점을 만들지 마세요.
- COMPARE: 각 업무 설명 뒤에 질문과 관련된 주요 차이를 정리하세요.
- RELATION: 관계 판단을 결론에서 먼저 말하고, 직접 근거가 없으면 확인되지 않는다고 답하세요.
- SEQUENCE: 사용자가 요청한 순서대로 단계와 Action을 구분하세요.
- 한 업무의 Evidence를 모든 업무의 공통 근거처럼 사용하지 마세요.
""".strip()

DC_SKELETON_SYSTEM_PROMPT_V1 = (
    D2_SKELETON_SYSTEM_PROMPT
    + "\n\n" + DC_FACT_SAFETY_RULES_V1
    + "\n\n" + DC_CROSS_SKELETON_RULES_V1
)
DC_FINAL_SYSTEM_PROMPT_V1 = (
    D_STRUCTURED_FINAL_SYSTEM_PROMPT_V3
    + "\n\n" + DC_FACT_SAFETY_RULES_V1
    + "\n\n" + DC_CROSS_FINAL_RULES_V1
)
DC_ONECALL_SYSTEM_PROMPT_V1 = (
    "당신은 예금보험공사 교차업무 복합질의 전용 Answer Skeleton 및 최종답변 생성기입니다.\n"
    "반드시 SKELETON_JSON과 FINAL_ANSWER 두 태그만 출력하세요.\n"
    "SKELETON_JSON 내부만 JSON이며 FINAL_ANSWER는 Markdown입니다.\n\n"
    + DC_FACT_SAFETY_RULES_V1
    + "\n\n" + DC_CROSS_SKELETON_RULES_V1
    + "\n\n" + DC_CROSS_FINAL_RULES_V1
    + "\n\n" + ACTION_LINK_PROMPT_RULE_V1
)


_VALIDATE_DC_SKELETON_GENERAL_V1 = validate_dc_skeleton_v1


def _fact_claim_business_map_dc_v1(pack: Mapping[str, Any]) -> dict[str, str]:
    output = {}
    for supplement in (pack.get("fact_index") or {}).get("supplements") or []:
        fact_id = str(supplement.get("fact_index_id") or "")
        business = str(supplement.get("business_function") or "")
        for claim in supplement.get("verified_claims") or []:
            output[f"{fact_id}:{claim.get('claim_id')}"] = business
    return output


def validate_dc_skeleton_v1(
    raw: Mapping[str, Any],
    answer_needs: Sequence[Mapping[str, Any]],
    pack: Mapping[str, Any],
) -> dict[str, Any]:
    result = _VALIDATE_DC_SKELETON_GENERAL_V1(raw, answer_needs, pack)
    need_map = {str(row.get("need_id") or ""): dict(row) for row in answer_needs}
    fact_business = _fact_claim_business_map_dc_v1(pack)
    evidence_violations = []
    fact_violations = []
    coverage_rows = []
    for item in result.get("answer_items") or []:
        need = need_map.get(str(item.get("need_id") or ""), {})
        business = str(need.get("label") or "")
        allowed_need_evidence = set(need.get("retrieval_evidence_ids") or [])
        original_evidence = list(item.get("evidence_ids") or [])
        if allowed_need_evidence:
            outside = [value for value in original_evidence if value not in allowed_need_evidence]
            evidence_violations.extend({"need_id": item["need_id"], "evidence_id": value} for value in outside)
            item["evidence_ids"] = [value for value in original_evidence if value in allowed_need_evidence]
        original_facts = list(item.get("fact_claim_ids") or [])
        outside_facts = [
            value for value in original_facts
            if fact_business.get(value) and business and fact_business.get(value) != business
        ]
        fact_violations.extend({"need_id": item["need_id"], "fact_claim_id": value} for value in outside_facts)
        item["fact_claim_ids"] = [value for value in original_facts if value not in outside_facts]
        item["business_function"] = business
        if item["status"] == "ANSWERED" and not item["evidence_ids"] and not item["fact_claim_ids"]:
            item["status"] = "PARTIAL" if item.get("claim") else "UNSUPPORTED"
        coverage_rows.append({
            "need_id": item["need_id"],
            "need_type": item.get("need_type"),
            "label": business,
            "business_function": business,
            "status": item["status"],
            "evidence_ids": item["evidence_ids"],
            "fact_claim_ids": item["fact_claim_ids"],
            "missing_reason": item.get("missing_reason"),
        })
    relation_raw = raw.get("cross_need_relation") if isinstance(raw.get("cross_need_relation"), Mapping) else {}
    allowed_evidence = set(answer_b_core._allowed_evidence(pack))
    allowed_facts = set(_allowed_fact_claims_v3(pack))
    relation_evidence = [
        value for value in answer_b_core._clean_list(relation_raw.get("evidence_ids"))
        if value in allowed_evidence
    ]
    relation_facts = [
        value for value in _clean_fact_claim_keys_v3(relation_raw.get("fact_claim_ids"))
        if value in allowed_facts
    ]
    relation_supported = bool(relation_raw.get("supported") and (relation_evidence or relation_facts))
    response_mode = str(raw.get("response_mode") or "SEPARATE").upper()
    if response_mode not in DC_RESPONSE_MODES_V1:
        response_mode = "SEPARATE"
    program = calculate_program_coverage_v2(coverage_rows)
    scope_passed = not evidence_violations and not fact_violations
    result.update({
        "response_mode": response_mode,
        "answer_items": result["answer_items"],
        "need_coverage": coverage_rows,
        "cross_need_relation": {
            "requested": bool(relation_raw.get("requested")),
            "supported": relation_supported,
            "claim": answer_b_core._clean(relation_raw.get("claim")),
            "evidence_ids": list(dict.fromkeys(relation_evidence)),
            "fact_claim_ids": list(dict.fromkeys(relation_facts)),
            "missing_reason": answer_b_core._clean(relation_raw.get("missing_reason")),
        },
        "cross_need_evidence_violations": evidence_violations,
        "cross_need_fact_violations": fact_violations,
        "cross_need_scope_passed": scope_passed,
        "reference_validation_passed": bool(result.get("reference_validation_passed") and scope_passed),
        **program,
    })
    return result


def _dc_skeleton_prompt_v1(
    question: str,
    answer_needs: Sequence[Mapping[str, Any]],
    relation_constraint: Mapping[str, Any],
    augmented_pack: Mapping[str, Any],
) -> str:
    expected_mode = classify_dc_response_mode_v1(question)
    return f"""[사용자 질문]\n{_clean_text(question)}\n\n[expected_response_mode]\n{expected_mode}\n\n[업무별 Answer Needs]\n{_compact_json(answer_needs)}\n\n[관계 주장 안전조건]\n{_compact_json(relation_constraint)}\n\n[C안 Fact Index 보강 Evidence Pack]\n{_compact_json(augmented_pack)}\n\n[출력 JSON]\n{{\"response_mode\":\"{expected_mode}\",\"core_answer\":\"핵심 결론\",\"answer_items\":[{{\"need_id\":\"N1\",\"business_function\":\"업무명\",\"topic\":\"항목\",\"status\":\"ANSWERED|PARTIAL|UNSUPPORTED\",\"claim\":\"근거 사실\",\"conditions\":[],\"details\":[],\"evidence_ids\":[\"E1\"],\"fact_claim_ids\":[\"FI-CAND-001:F1\"],\"missing_reason\":\"\"}}],\"cross_need_relation\":{{\"requested\":false,\"supported\":false,\"claim\":\"\",\"evidence_ids\":[],\"fact_claim_ids\":[],\"missing_reason\":\"\"}},\"uncertainties\":[],\"conflicts\":[]}}"""


_GENERATE_DC_TWOCALL_GENERAL_V1 = generate_dc_twocall_v1


def generate_dc_twocall_v1(
    question: str,
    augmented_pack: Mapping[str, Any],
) -> dict[str, Any]:
    if len(augmented_pack.get("needs") or []) < 2:
        raise RuntimeError("D_CROSS_ONLY_POLICY: 단일·동일업무 질문은 C안 대상입니다.")
    result = _GENERATE_DC_TWOCALL_GENERAL_V1(question, augmented_pack)
    expected = classify_dc_response_mode_v1(question)
    actual = str((result.get("skeleton") or {}).get("response_mode") or "SEPARATE")
    mode_passed = actual == expected
    result.update({
        "response_mode_expected": expected,
        "response_mode_actual": actual,
        "response_mode_validation_passed": mode_passed,
        "cross_business_prompt": True,
        "validation_passed": bool(result.get("validation_passed") and mode_passed),
    })
    if not mode_passed and result.get("coverage_status") == "SUFFICIENT":
        result["coverage_status"] = "PARTIAL"
    return result


def generate_dc_onecall_v1(
    question: str,
    augmented_pack: Mapping[str, Any],
) -> dict[str, Any]:
    if len(augmented_pack.get("needs") or []) < 2:
        raise RuntimeError("D_CROSS_ONLY_POLICY: 단일·동일업무 질문은 C안 대상입니다.")
    total_started = time.perf_counter()
    need_started = time.perf_counter()
    answer_needs = extract_dc_answer_needs_v1(question, augmented_pack)
    relation_constraint = relation_constraint_v1(question, augmented_pack)
    expected_mode = classify_dc_response_mode_v1(question)
    need_ms = (time.perf_counter() - need_started) * 1000
    prompt_started = time.perf_counter()
    prompt = f"""[사용자 질문]\n{_clean_text(question)}\n\n[expected_response_mode]\n{expected_mode}\n\n[업무별 Answer Needs]\n{_compact_json(answer_needs)}\n\n[관계 주장 안전조건]\n{_compact_json(relation_constraint)}\n\n[C안 Fact Index 보강 Evidence Pack]\n{_compact_json(augmented_pack)}\n\n[반드시 지킬 출력 형식]\n<SKELETON_JSON>\n{{\"response_mode\":\"{expected_mode}\",\"core_answer\":\"핵심 결론\",\"answer_items\":[{{\"need_id\":\"N1\",\"business_function\":\"업무명\",\"topic\":\"항목\",\"status\":\"ANSWERED|PARTIAL|UNSUPPORTED\",\"claim\":\"근거 사실\",\"conditions\":[],\"details\":[],\"evidence_ids\":[\"E1\"],\"fact_claim_ids\":[\"FI-CAND-001:F1\"],\"missing_reason\":\"\"}}],\"cross_need_relation\":{{\"requested\":false,\"supported\":false,\"claim\":\"\",\"evidence_ids\":[],\"fact_claim_ids\":[],\"missing_reason\":\"\"}},\"uncertainties\":[],\"conflicts\":[]}}\n</SKELETON_JSON>\n<FINAL_ANSWER>\nresponse_mode에 맞춰 모든 업무 Need를 구분한 Markdown 답변\n</FINAL_ANSWER>"""
    prompt_ms = (time.perf_counter() - prompt_started) * 1000
    raw, usage, api_ms, trace = _call_answer_api_v1(
        system_prompt=DC_ONECALL_SYSTEM_PROMPT_V1,
        user_prompt=prompt,
        max_tokens=DC_ONECALL_MAX_TOKENS_V1,
    )
    validation_started = time.perf_counter()
    parsed = _parse_onecall_output_dc_v2(raw)
    raw_skeleton = parsed.get("answer_skeleton") or {}
    if not isinstance(raw_skeleton, Mapping):
        raise ValueError("D-C 교차업무 1Call answer_skeleton이 JSON 객체가 아닙니다.")
    skeleton = validate_dc_skeleton_v1(raw_skeleton, answer_needs, augmented_pack)
    raw_answer = answer_b_core._strip_model_urls(str(parsed.get("answer") or "").strip())
    if not raw_answer:
        raise ValueError("D-C 교차업무 1Call 최종답변이 비어 있습니다.")
    reference_audit = audit_dc_final_references_v1(
        raw_answer,
        skeleton,
        augmented_pack,
        explicit_evidence_ids=answer_b_core._clean_list(parsed.get("used_evidence_ids")),
        explicit_fact_claim_ids=_clean_fact_claim_keys_v3(parsed.get("used_fact_claim_ids")),
    )
    selected_pack = filter_augmented_pack_for_dc_v1(augmented_pack, skeleton)
    validation_ms = (time.perf_counter() - validation_started) * 1000
    post_started = time.perf_counter()
    safe_answer, guard_applied = _relation_safe_answer_v1(raw_answer, relation_constraint)
    safe_answer = normalize_answer_markdown_v3(safe_answer)
    numeric_audit = audit_numeric_support_dc_v1(safe_answer, selected_pack)
    forbidden_audit = audit_forbidden_claims_dc_v1(safe_answer, selected_pack)
    post_ms = (time.perf_counter() - post_started) * 1000
    trace_parts = _trace_parts_v3(trace)
    total_ms = (time.perf_counter() - total_started) * 1000
    actual_mode = str(skeleton.get("response_mode") or "SEPARATE")
    mode_passed = actual_mode == expected_mode
    validation_passed = bool(
        skeleton.get("reference_validation_passed")
        and skeleton.get("cross_need_scope_passed")
        and reference_audit["reference_consistency_passed"]
        and numeric_audit["numeric_support_passed"]
        and forbidden_audit["forbidden_claim_check_passed"]
        and not guard_applied
        and mode_passed
    )
    coverage = skeleton["coverage_status"]
    if not validation_passed and coverage == "SUFFICIENT":
        coverage = "PARTIAL"
    return {
        "system": "D-C 1Call · 교차업무 전용",
        "answer": safe_answer,
        "answer_needs": answer_needs,
        "skeleton": skeleton,
        "need_coverage": skeleton["need_coverage"],
        "coverage_status": coverage,
        "strict_need_coverage_rate": skeleton["strict_need_coverage_rate"],
        "answerable_need_coverage_rate": skeleton["answerable_need_coverage_rate"],
        "selected_evidence_pack": selected_pack,
        "selected_evidence_count": len(selected_pack.get("evidence") or []),
        "used_evidence_ids": reference_audit["used_evidence_ids"],
        "used_fact_claim_ids": reference_audit["used_fact_claim_ids"],
        "reference_audit": reference_audit,
        "numeric_audit": numeric_audit,
        "forbidden_claim_audit": forbidden_audit,
        "validation_passed": validation_passed,
        "relation_constraint": relation_constraint,
        "relation_guard_applied": guard_applied,
        "response_mode_expected": expected_mode,
        "response_mode_actual": actual_mode,
        "response_mode_validation_passed": mode_passed,
        "cross_business_prompt": True,
        "output_contract": parsed.get("output_contract"),
        "latency_ms": total_ms,
        "skeleton_latency_ms": api_ms,
        "final_latency_ms": 0.0,
        "usage": usage,
        "api_calls": 1,
        "attempts": [{"stage": "dc_cross_skeleton_and_final", "latency_ms": api_ms, "trace": trace}],
        "stage_latency_ms": {
            "need_extraction_ms": need_ms,
            "combined_prompt_build_ms": prompt_ms,
            "combined_api_wall_ms": trace_parts["api_wall_ms"],
            "combined_pacing_wait_ms": trace_parts["pacing_wait_ms"],
            "combined_retry_wait_ms": trace_parts["retry_wait_ms"],
            "combined_estimated_service_ms": trace_parts["estimated_service_ms"],
            "combined_json_skeleton_validation_ms": validation_ms,
            "postprocess_ms": post_ms,
            "answer_total_ms": total_ms,
        },
    }


_EXECUTE_DC_GENERAL_V1 = execute_dc_variant_v1


def execute_dc_variant_v1(
    variant: str,
    question: str,
    holder: dict[str, Any],
    *,
    force_answer_regeneration: bool = False,
) -> dict[str, Any]:
    total_started = time.perf_counter()
    common, original_cache_hit, original_common_ms, fact_ms = _prepare_dc_common_v1(question, holder)
    if common.get("route") != "RETRIEVE":
        return {
            "variant": str(variant).upper(),
            "route": common.get("route"),
            "route_message": common.get("route_message"),
            "common": common,
            "common_cache_hit": original_cache_hit,
            "answer_cache_hit": False,
            "latency": {
                "common_this_click_ms": original_common_ms,
                "fact_index_match_ms": fact_ms,
                "answer_ms": 0.0,
                "click_wall_ms": (time.perf_counter() - total_started) * 1000,
            },
            "api_trace": {"logical_api_calls": 0, "physical_http_attempts": 0, "traces": []},
        }
    cross_passed, cross_audit = is_cross_business_dc_v1(common)
    common["dc_cross_business_gate"] = cross_audit
    if not cross_passed:
        event = {
            "question": _clean_text(question),
            "variant": str(variant).upper(),
            "answer_api_calls": 0,
            "route": "C_POLICY_TARGET",
            "cross_business_gate_passed": False,
            "business_count": cross_audit["business_count"],
            "need_count": cross_audit["need_count"],
            "common_this_click_ms": original_common_ms,
            "click_wall_ms": (time.perf_counter() - total_started) * 1000,
        }
        holder["events"].append(event)
        return {
            "variant": str(variant).upper(),
            "route": "C_POLICY_TARGET",
            "route_message": (
                "이 질문은 교차업무 복합질의가 아니므로 D안을 호출하지 않았습니다. "
                "최종 운영 정책에서는 C안으로 답변합니다."
            ),
            "common": common,
            "common_cache_hit": original_cache_hit,
            "answer_cache_hit": False,
            "latency": {
                "common_this_click_ms": original_common_ms,
                "fact_index_match_ms": fact_ms,
                "answer_ms": 0.0,
                "click_wall_ms": event["click_wall_ms"],
            },
            "api_trace": {"logical_api_calls": 0, "physical_http_attempts": 0, "traces": []},
            "cross_business_gate": cross_audit,
            "event": event,
        }
    result = _EXECUTE_DC_GENERAL_V1(
        variant,
        question,
        holder,
        force_answer_regeneration=force_answer_regeneration,
    )
    result["common_cache_hit"] = original_cache_hit
    result["latency"]["common_this_click_ms"] = original_common_ms
    result["latency"]["fact_index_match_ms"] = fact_ms
    result["latency"]["click_wall_ms"] = (time.perf_counter() - total_started) * 1000
    result["cross_business_gate"] = cross_audit
    payload = result.get("payload") or {}
    event = result.get("event") or {}
    event.update({
        "common_cache_hit": original_cache_hit,
        "common_this_click_ms": original_common_ms,
        "fact_index_match_ms": fact_ms,
        "click_wall_ms": result["latency"]["click_wall_ms"],
        "cross_business_gate_passed": True,
        "response_mode_expected": payload.get("response_mode_expected"),
        "response_mode_actual": payload.get("response_mode_actual"),
        "response_mode_validation_passed": payload.get("response_mode_validation_passed"),
        "cross_need_scope_passed": (payload.get("skeleton") or {}).get("cross_need_scope_passed"),
    })
    return result


DC_TEST_QUESTIONS_V1 = {
    "SEP01 · 각각 안내": "예금보험금 지급 조건과 은닉재산 신고 포상금을 각각 알려주세요.",
    "SEP02 · 금액과 기간": "예금자보호 한도와 착오송금 반환지원 신청기한을 각각 알려주세요.",
    "SEP05 · 세 업무": "예금자보호 한도, 착오송금 신청기한, 은닉재산 포상금 최고 한도를 각각 알려주세요.",
    "CMP01 · 제도 차이": "미수령금과 착오송금은 무엇이 다른가요?",
    "CMP03 · 신청대상 비교": "채무조정과 착오송금 반환지원의 신청 대상은 어떻게 다른가요?",
    "REL01 · 동시 신청": "착오송금 반환지원과 채무조정을 동시에 신청할 수 있나요?",
    "REL02 · 한 번에 신청": "예금보험금과 미수령금을 한 번에 신청할 수 있나요?",
    "SEQ01 · 순차 처리": "미수령금을 먼저 조회한 다음 착오송금 반환지원도 신청하려면 어떻게 해야 하나요?",
    "C01 · D 차단 단일질의": "예금자보호 한도는 얼마인가요?",
    "C02 · D 차단 동일업무": "채무조정 신청자격과 필요서류를 알려주세요.",
}


print({
    "d_policy": "CROSS_BUSINESS_ONLY",
    "response_modes": sorted(DC_RESPONSE_MODES_V1),
    "non_cross_policy": "C_POLICY_TARGET_WITHOUT_D_API_CALL",
    "prompt_version": DC_CROSS_PROMPT_VERSION_V1,
})

# ==== cell 92 ====
# v1.2: 1Call 일반답변 fallback + 적용 대상 특수성 안전가드
DC_APPLICABILITY_SCOPE_RULES_V2 = """
[적용 대상·상황 특수성 규칙]
- 질문에 상속인·사망·피상속인이 없으면 상속인 금융거래조회를 일반적인 미수령금 조회 방법으로 설명하지 마세요.
- 질문이 일반적이면 일반 안내 근거를 우선 사용하고, 상속인·대리인·법인·미성년자 근거는 반드시 '해당 경우'로 한정하세요.
- 특정 역할이나 상황의 Evidence를 사용하면 answer_item의 applicability_scope를 SPECIAL_CASE로 표시하고 applies_to를 작성하세요.
- GENERAL 질문에 SPECIAL_CASE를 제도의 정의·대표 절차·유일한 방법처럼 확대하지 마세요.
- 미수령금 전체를 상속인이 받을 돈 또는 주로 상속 절차로 처리되는 돈이라고 설명하지 마세요.
""".strip()

DC_SKELETON_SYSTEM_PROMPT_V1 = DC_SKELETON_SYSTEM_PROMPT_V1 + "\n\n" + DC_APPLICABILITY_SCOPE_RULES_V2
DC_FINAL_SYSTEM_PROMPT_V1 = DC_FINAL_SYSTEM_PROMPT_V1 + "\n\n" + DC_APPLICABILITY_SCOPE_RULES_V2
DC_ONECALL_SYSTEM_PROMPT_V1 = DC_ONECALL_SYSTEM_PROMPT_V1 + "\n\n" + DC_APPLICABILITY_SCOPE_RULES_V2


_PARSE_ONECALL_STRICT_BEFORE_FALLBACK_V2 = _parse_onecall_output_dc_v2


def _parse_onecall_output_dc_v2(raw: str) -> dict[str, Any]:
    """정상 태그·기존 JSON을 우선 사용하고, 일반 Markdown은 감사 가능한 fallback으로 보존합니다."""
    try:
        return _PARSE_ONECALL_STRICT_BEFORE_FALLBACK_V2(raw)
    except ValueError as error:
        answer = str(raw or "").strip()
        if not answer:
            raise
        return {
            "answer_skeleton": {
                "response_mode": "SEPARATE",
                "core_answer": "",
                "answer_items": [],
                "cross_need_relation": {
                    "requested": False,
                    "supported": False,
                    "claim": "",
                    "evidence_ids": [],
                    "fact_claim_ids": [],
                    "missing_reason": "모델이 Answer Skeleton 출력 계약을 따르지 않음",
                },
                "uncertainties": ["Answer Skeleton이 모델 출력에서 누락됨"],
                "conflicts": [],
            },
            "answer": answer,
            "used_evidence_ids": [],
            "used_fact_claim_ids": [],
            "output_contract": "PLAIN_ANSWER_CONTRACT_FALLBACK",
            "output_contract_passed": False,
            "output_contract_error": str(error),
        }


def build_program_fallback_skeleton_dc_v2(
    question: str,
    answer_needs: Sequence[Mapping[str, Any]],
    pack: Mapping[str, Any],
) -> dict[str, Any]:
    expected_mode = classify_dc_response_mode_v1(question)
    items = []
    for need in answer_needs:
        evidence_ids = list(dict.fromkeys(
            str(value) for value in need.get("retrieval_evidence_ids") or [] if str(value)
        ))[:2]
        items.append({
            "need_id": str(need.get("need_id") or ""),
            "business_function": str(need.get("label") or ""),
            "topic": str(need.get("question_part") or need.get("label") or ""),
            "status": "PARTIAL",
            "claim": "모델이 Answer Skeleton을 반환하지 않아 최종답변만 로컬 검증합니다.",
            "conditions": [],
            "details": [],
            "evidence_ids": evidence_ids,
            "fact_claim_ids": [],
            "missing_reason": "ONECALL_OUTPUT_CONTRACT_FAILED",
            "applicability_scope": "UNVERIFIED",
            "applies_to": [],
        })
    raw = {
        "response_mode": expected_mode,
        "core_answer": "",
        "answer_items": items,
        "cross_need_relation": {
            "requested": expected_mode in {"COMPARE", "RELATION"},
            "supported": False,
            "claim": "",
            "evidence_ids": [],
            "fact_claim_ids": [],
            "missing_reason": "모델 Skeleton 누락으로 관계 근거 구조를 검증할 수 없음",
        },
        "uncertainties": ["Answer Skeleton이 프로그램 fallback으로 생성됨"],
        "conflicts": [],
    }
    return validate_dc_skeleton_v1(raw, answer_needs, pack)


DC_SCOPE_ROLE_PATTERNS_V2 = {
    "INHERITANCE": re.compile(r"상속|상속인|피상속인|사망|유족"),
    "PROXY": re.compile(r"대리인|위임|대리\s*신청"),
    "SENDER": re.compile(r"송금인|착오송금인|돈을\s*보낸"),
    "RECIPIENT": re.compile(r"수취인|돈을\s*받은|잘못\s*받"),
    "MINOR": re.compile(r"미성년자|친권자"),
    "CORPORATION": re.compile(r"법인|사업자|대표자"),
}

DC_HIGH_RISK_GENERALIZATION_PATTERNS_V2 = [
    (
        "UNCLAIMED_FUNDS_INHERITANCE_GENERALIZATION",
        re.compile(r"미수령금(?:은|이란|의 경우).{0,45}(?:주로\s*)?(?:상속\s*절차|상속인이\s*받아야|상속을\s*통해).{0,30}(?:처리|수령|돈)", re.I),
    ),
    (
        "UNCLAIMED_FUNDS_INHERITANCE_ONLY_METHOD",
        re.compile(r"미수령금.{0,35}(?:상속인\s*금융거래조회).{0,25}(?:로만|유일|통해서만)", re.I),
    ),
]


def audit_applicability_scope_dc_v2(
    question: str,
    answer: str,
) -> dict[str, Any]:
    question_roles = {
        role for role, pattern in DC_SCOPE_ROLE_PATTERNS_V2.items() if pattern.search(str(question or ""))
    }
    answer_roles = {
        role for role, pattern in DC_SCOPE_ROLE_PATTERNS_V2.items() if pattern.search(str(answer or ""))
    }
    hits = []
    if "INHERITANCE" not in question_roles:
        for rule_id, pattern in DC_HIGH_RISK_GENERALIZATION_PATTERNS_V2:
            match = pattern.search(str(answer or ""))
            if match:
                hits.append({"rule_id": rule_id, "matched_text": match.group(0)})
    return {
        "question_roles": sorted(question_roles),
        "answer_roles": sorted(answer_roles),
        "high_risk_generalization_hits": hits,
        "applicability_scope_passed": not hits,
    }


def apply_applicability_scope_guard_dc_v2(
    question: str,
    answer: str,
) -> tuple[str, dict[str, Any], bool]:
    audit = audit_applicability_scope_dc_v2(question, answer)
    if audit["applicability_scope_passed"]:
        return answer, audit, False
    corrected = str(answer or "")
    corrected = re.sub(
        r"미수령금은\s*주로\s*상속\s*절차를\s*통해\s*처리됩니다\.?",
        "미수령금의 일반 조회·신청 절차와 상속인에게 적용되는 별도 조회 절차는 구분해야 합니다.",
        corrected,
        flags=re.I,
    )
    corrected = re.sub(
        r"미수령금은\s*상속인이\s*받아야\s*할\s*돈(?:을\s*의미합니다)?\.?",
        "미수령금은 예금자 등이 찾아가지 않은 금액이며, 상속인 조회는 사망한 예금자와 관련된 특수한 경우입니다.",
        corrected,
        flags=re.I,
    )
    notice = (
        "**적용 대상 안내:** 질문에 상속 상황이 명시되지 않았으므로, 상속인 금융거래조회 절차를 "
        "일반적인 미수령금 조회 방법으로 단정하지 않습니다. 상속인의 경우에만 별도 절차가 적용될 수 있습니다."
    )
    corrected = notice + "\n\n" + corrected
    audit["guard_notice_added"] = True
    return corrected, audit, True


_GENERATE_DC_ONECALL_BEFORE_SAFETY_V2 = generate_dc_onecall_v1
_GENERATE_DC_TWOCALL_BEFORE_SAFETY_V2 = generate_dc_twocall_v1


def _rebuild_plain_fallback_payload_dc_v2(
    result: dict[str, Any],
    question: str,
    augmented_pack: Mapping[str, Any],
) -> dict[str, Any]:
    answer_needs = list(result.get("answer_needs") or extract_dc_answer_needs_v1(question, augmented_pack))
    skeleton = build_program_fallback_skeleton_dc_v2(question, answer_needs, augmented_pack)
    selected_pack = filter_augmented_pack_for_dc_v1(augmented_pack, skeleton)
    answer = str(result.get("answer") or "").strip()
    reference_audit = audit_dc_final_references_v1(answer, skeleton, augmented_pack)
    numeric_audit = audit_numeric_support_dc_v1(answer, selected_pack)
    forbidden_audit = audit_forbidden_claims_dc_v1(answer, selected_pack)
    expected_mode = classify_dc_response_mode_v1(question)
    result.update({
        "skeleton": skeleton,
        "need_coverage": skeleton["need_coverage"],
        "coverage_status": "PARTIAL",
        "strict_need_coverage_rate": 0.0,
        "answerable_need_coverage_rate": 1.0 if answer_needs else 0.0,
        "selected_evidence_pack": selected_pack,
        "selected_evidence_count": len(selected_pack.get("evidence") or []),
        "used_evidence_ids": reference_audit["used_evidence_ids"],
        "used_fact_claim_ids": reference_audit["used_fact_claim_ids"],
        "reference_audit": reference_audit,
        "numeric_audit": numeric_audit,
        "forbidden_claim_audit": forbidden_audit,
        "validation_passed": False,
        "response_mode_expected": expected_mode,
        "response_mode_actual": expected_mode,
        "response_mode_validation_passed": True,
        "output_contract": "PLAIN_ANSWER_CONTRACT_FALLBACK",
        "output_contract_passed": False,
        "skeleton_source": "PROGRAM_FALLBACK",
        "plain_answer_fallback": True,
    })
    return result


def _apply_scope_safety_to_result_dc_v2(
    result: dict[str, Any],
    question: str,
) -> dict[str, Any]:
    guarded_answer, scope_audit, guard_applied = apply_applicability_scope_guard_dc_v2(
        question, str(result.get("answer") or "")
    )
    result["answer"] = normalize_answer_markdown_v3(guarded_answer)
    result["applicability_scope_audit"] = scope_audit
    result["applicability_scope_guard_applied"] = guard_applied
    if guard_applied:
        result["validation_passed"] = False
        if result.get("coverage_status") == "SUFFICIENT":
            result["coverage_status"] = "PARTIAL"
    result.setdefault("output_contract_passed", True)
    result.setdefault("skeleton_source", "MODEL")
    result.setdefault("plain_answer_fallback", False)
    return result


def generate_dc_onecall_v1(
    question: str,
    augmented_pack: Mapping[str, Any],
) -> dict[str, Any]:
    result = _GENERATE_DC_ONECALL_BEFORE_SAFETY_V2(question, augmented_pack)
    if str(result.get("output_contract") or "") == "PLAIN_ANSWER_CONTRACT_FALLBACK":
        result = _rebuild_plain_fallback_payload_dc_v2(result, question, augmented_pack)
    return _apply_scope_safety_to_result_dc_v2(result, question)


def generate_dc_twocall_v1(
    question: str,
    augmented_pack: Mapping[str, Any],
) -> dict[str, Any]:
    result = _GENERATE_DC_TWOCALL_BEFORE_SAFETY_V2(question, augmented_pack)
    result["output_contract_passed"] = True
    result["skeleton_source"] = "MODEL"
    result["plain_answer_fallback"] = False
    return _apply_scope_safety_to_result_dc_v2(result, question)


_EXECUTE_DC_BEFORE_SAFETY_AUDIT_V2 = execute_dc_variant_v1


def execute_dc_variant_v1(
    variant: str,
    question: str,
    holder: dict[str, Any],
    *,
    force_answer_regeneration: bool = False,
) -> dict[str, Any]:
    result = _EXECUTE_DC_BEFORE_SAFETY_AUDIT_V2(
        variant,
        question,
        holder,
        force_answer_regeneration=force_answer_regeneration,
    )
    payload = result.get("payload") or {}
    event = result.get("event")
    if isinstance(event, dict) and payload:
        event.update({
            "output_contract_passed": bool(payload.get("output_contract_passed")),
            "skeleton_source": str(payload.get("skeleton_source") or ""),
            "plain_answer_fallback": bool(payload.get("plain_answer_fallback")),
            "applicability_scope_passed": bool(
                (payload.get("applicability_scope_audit") or {}).get("applicability_scope_passed")
            ),
            "applicability_scope_guard_applied": bool(payload.get("applicability_scope_guard_applied")),
        })
    return result


print({
    "plain_answer_fallback": "enabled_without_additional_llm_call",
    "fallback_coverage": "PARTIAL",
    "fallback_validation_passed": False,
    "audience_scope_guard": "enabled",
    "inheritance_generalization_guard": "enabled",
})

# ==== cell 93 ====
# v1.3: 태그 포함 fallback 정리 + 미지원 관계 추론 차단 + Registry 내부 안내 미노출
_PARSE_ONECALL_BEFORE_TAGGED_FALLBACK_V3 = _parse_onecall_output_dc_v2


def _extract_final_answer_from_tagged_raw_dc_v3(raw: str) -> str:
    """구조화 Skeleton이 실패해도 FINAL_ANSWER 사용자 답변만 안전하게 분리합니다."""
    text = str(raw or "").strip()
    if not text:
        return ""
    patterns = [
        re.compile(
            r"<\s*FINAL_ANSWER\s*>\s*(.*?)(?:<\s*/\s*FINAL_ANSWER\s*>|\Z)",
            re.I | re.S,
        ),
        re.compile(
            r"\[\s*FINAL_ANSWER\s*\]\s*(.*?)(?=\n\s*\[\s*(?:SKELETON_JSON|FINAL_ANSWER)\s*\]|\Z)",
            re.I | re.S,
        ),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            answer = match.group(1).strip()
            answer = re.sub(r"<\s*/\s*FINAL_ANSWER\s*>\s*$", "", answer, flags=re.I).strip()
            if answer:
                return answer
    return ""


def _parse_onecall_output_dc_v2(raw: str) -> dict[str, Any]:
    """정상 구조화 결과를 우선 사용하고, 실패 시 사용자용 최종답변만 보존합니다."""
    parsed = _PARSE_ONECALL_BEFORE_TAGGED_FALLBACK_V3(raw)
    if str(parsed.get("output_contract") or "") != "PLAIN_ANSWER_CONTRACT_FALLBACK":
        return parsed

    final_answer = _extract_final_answer_from_tagged_raw_dc_v3(raw)
    if final_answer:
        parsed["answer"] = final_answer
        parsed["fallback_kind"] = "TAGGED_INVALID_SKELETON_FINAL_ONLY"
        parsed["tagged_final_answer_extracted"] = True
    else:
        parsed["fallback_kind"] = "PLAIN_MARKDOWN"
        parsed["tagged_final_answer_extracted"] = False
    return parsed


DC_RELATION_TERM_PATTERN_V3 = re.compile(r"(?:동시|한\s*번에|같이|함께|따로|각각)", re.I)
DC_RELATION_SPECULATION_PATTERN_V3 = re.compile(
    r"(?:가능성(?:이)?\s*(?:높|있)|것으로\s*(?:보|추정)|것\s*같|추측)",
    re.I,
)
DC_RELATION_UNSUPPORTED_NOTICE_V3 = (
    "제공된 공식 근거만으로는 두 항목의 동시 처리 또는 동시 신청 가능 여부를 확인할 수 없습니다."
)


def _remove_unsupported_relation_speculation_dc_v3(
    question: str,
    answer: str,
    skeleton: Mapping[str, Any],
) -> tuple[str, dict[str, Any], bool]:
    expected_mode = classify_dc_response_mode_v1(question)
    relation = skeleton.get("cross_need_relation") or {}
    relation_requested = expected_mode in {"RELATION", "COMPARE"} or bool(relation.get("requested"))
    relation_supported = bool(relation.get("supported"))
    audit = {
        "response_mode": expected_mode,
        "relation_requested": relation_requested,
        "relation_supported": relation_supported,
        "removed_sentences": [],
        "unsupported_relation_speculation_passed": True,
    }
    if not relation_requested or relation_supported:
        return str(answer or ""), audit, False

    text = str(answer or "").strip()
    if not text:
        return text, audit, False

    parts = re.split(r"(?<=[.!?])\s+", text)
    kept = []
    for part in parts:
        sentence = part.strip()
        if not sentence:
            continue
        if (
            DC_RELATION_TERM_PATTERN_V3.search(sentence)
            and DC_RELATION_SPECULATION_PATTERN_V3.search(sentence)
        ):
            audit["removed_sentences"].append(sentence)
            continue
        kept.append(sentence)

    if not audit["removed_sentences"]:
        return text, audit, False

    corrected = "\n\n".join(kept).strip()
    if not re.search(r"(?:공식\s*근거|직접적인\s*(?:언급|근거)).{0,30}(?:확인|명시).{0,10}(?:않|없)", corrected):
        corrected = (corrected + "\n\n" + DC_RELATION_UNSUPPORTED_NOTICE_V3).strip()
    audit["unsupported_relation_speculation_passed"] = False
    return corrected, audit, True


_GENERATE_DC_ONECALL_BEFORE_RELATION_GUARD_V3 = generate_dc_onecall_v1
_GENERATE_DC_TWOCALL_BEFORE_RELATION_GUARD_V3 = generate_dc_twocall_v1


def _apply_relation_speculation_guard_to_result_dc_v3(
    result: dict[str, Any],
    question: str,
    augmented_pack: Mapping[str, Any],
) -> dict[str, Any]:
    corrected, audit, applied = _remove_unsupported_relation_speculation_dc_v3(
        question,
        str(result.get("answer") or ""),
        result.get("skeleton") or {},
    )
    result["answer"] = normalize_answer_markdown_v3(corrected)
    result["relation_speculation_audit"] = audit
    result["relation_speculation_guard_applied"] = applied
    result["fallback_answer_sanitized"] = bool(
        result.get("plain_answer_fallback")
        and "SKELETON_JSON" not in str(result.get("answer") or "")
        and "FINAL_ANSWER" not in str(result.get("answer") or "")
    )
    if applied:
        result["validation_passed"] = False
        if result.get("coverage_status") == "SUFFICIENT":
            result["coverage_status"] = "PARTIAL"
        selected_pack = result.get("selected_evidence_pack") or augmented_pack
        result["reference_audit"] = audit_dc_final_references_v1(
            result["answer"], result.get("skeleton") or {}, augmented_pack
        )
        result["numeric_audit"] = audit_numeric_support_dc_v1(result["answer"], selected_pack)
        result["forbidden_claim_audit"] = audit_forbidden_claims_dc_v1(result["answer"], selected_pack)
    return result


def generate_dc_onecall_v1(
    question: str,
    augmented_pack: Mapping[str, Any],
) -> dict[str, Any]:
    result = _GENERATE_DC_ONECALL_BEFORE_RELATION_GUARD_V3(question, augmented_pack)
    return _apply_relation_speculation_guard_to_result_dc_v3(result, question, augmented_pack)


def generate_dc_twocall_v1(
    question: str,
    augmented_pack: Mapping[str, Any],
) -> dict[str, Any]:
    result = _GENERATE_DC_TWOCALL_BEFORE_RELATION_GUARD_V3(question, augmented_pack)
    return _apply_relation_speculation_guard_to_result_dc_v3(result, question, augmented_pack)


_ACTION_LINKS_MARKDOWN_BEFORE_NOTICE_REMOVAL_V3 = action_links_markdown_v1


def action_links_markdown_v1(action_links: Sequence[Mapping[str, Any]]) -> str:
    """Registry 검증은 유지하고 내부 구현 설명 문구만 사용자 화면에서 제외합니다."""
    rendered = _ACTION_LINKS_MARKDOWN_BEFORE_NOTICE_REMOVAL_V3(action_links)
    visible_lines = [
        line
        for line in str(rendered or "").splitlines()
        if "Action Link Registry" not in line
    ]
    return "\n".join(visible_lines).rstrip()


_EXECUTE_DC_BEFORE_V3_AUDIT = execute_dc_variant_v1


def execute_dc_variant_v1(
    variant: str,
    question: str,
    holder: dict[str, Any],
    *,
    force_answer_regeneration: bool = False,
) -> dict[str, Any]:
    result = _EXECUTE_DC_BEFORE_V3_AUDIT(
        variant,
        question,
        holder,
        force_answer_regeneration=force_answer_regeneration,
    )
    payload = result.get("payload") or {}
    event = result.get("event")
    if isinstance(event, dict) and payload:
        event.update({
            "fallback_answer_sanitized": bool(payload.get("fallback_answer_sanitized")),
            "relation_speculation_guard_applied": bool(payload.get("relation_speculation_guard_applied")),
            "unsupported_relation_speculation_passed": bool(
                (payload.get("relation_speculation_audit") or {}).get(
                    "unsupported_relation_speculation_passed", True
                )
            ),
        })
    return result


print({
    "tagged_invalid_skeleton_fallback": "FINAL_ANSWER_ONLY",
    "fallback_additional_llm_calls": 0,
    "unsupported_relation_speculation_guard": "enabled",
    "action_link_registry_notice_visible": False,
    "action_link_registry_validation": "preserved",
})
