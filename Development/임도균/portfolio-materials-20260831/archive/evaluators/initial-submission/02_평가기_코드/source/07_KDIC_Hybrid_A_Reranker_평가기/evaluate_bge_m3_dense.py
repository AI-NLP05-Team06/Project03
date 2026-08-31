"""KDIC BGE-M3 Dense 검색 일괄 평가기.

LLM 답변은 생성하지 않고 질문별 Top-10 청크를 Gold와 비교합니다.
"""

from __future__ import annotations

import argparse
import csv
import getpass
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from metrics import evaluate_ranking


DEFAULT_BASE_URL = "https://clovastudio.stream.ntruss.com/v1/openai"
DEFAULT_MODEL = "bge-m3"
DEFAULT_SHEET = "검색평가용"
TOP_K = 10

BUSINESS_FUNCTION_MAP = {
    "deposit_protection": "예금자보호제도",
    "deposit_insurance_payout": "예금보험금 안내",
    "unclaimed_funds": "고객 미수령금 신청",
    "mistaken_transfer": "착오송금 반환 신청",
    "debt_adjustment": "채무조정 안내",
    "hidden_assets_report": "은닉재산 신고",
}

REQUIRED_COLUMNS = {
    "검색평가대상",
    "evaluation_id",
    "예상질문",
    "도메인",
    "gold_business_function",
    "gold_primary_chunk_ids",
    "gold_supporting_chunk_ids",
    "gold_chunk_ids",
    "multi_chunk_required",
    "gold_review_status",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_json_array(value: Any, field_name: str, evaluation_id: str) -> list[str]:
    if isinstance(value, list):
        parsed = value
    else:
        text = "" if pd.isna(value) else str(value).strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{evaluation_id}: {field_name}이 JSON 배열이 아닙니다: {text}"
            ) from error
    if not isinstance(parsed, list):
        raise ValueError(f"{evaluation_id}: {field_name}이 배열이 아닙니다.")
    return list(dict.fromkeys(str(item).strip() for item in parsed if str(item).strip()))


def load_jsonl_from_zip(archive: zipfile.ZipFile, suffix: str) -> list[dict[str, Any]]:
    candidates = [
        name
        for name in archive.namelist()
        if name.replace("\\", "/").endswith(suffix)
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"ZIP에서 {suffix} 파일을 하나로 결정할 수 없습니다: {candidates}")
    records: list[dict[str, Any]] = []
    with archive.open(candidates[0], "r") as raw:
        for line_number, raw_line in enumerate(raw, start=1):
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{candidates[0]} {line_number}행 JSON 파싱 실패"
                ) from error
            records.append(record)
    return records


@dataclass
class EvaluationQuestion:
    evaluation_id: str
    original_question_id: str
    question: str
    domain_display: str
    business_function_code: str
    business_function_label: str
    complexity: str
    importance: str
    primary_gold: list[str]
    supporting_gold: list[str]
    all_gold: list[str]
    multi_chunk_required: bool
    review_status: str


def load_questions(
    dataset_path: Path,
    *,
    sheet_name: str,
    approved_only: bool,
    limit: int | None,
) -> list[EvaluationQuestion]:
    frame = pd.read_excel(dataset_path, sheet_name=sheet_name, dtype=str).fillna("")
    missing_columns = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing_columns:
        raise ValueError(f"평가데이터셋 필수 칼럼 누락: {missing_columns}")

    frame = frame[frame["검색평가대상"].str.upper().eq("Y")].copy()
    if approved_only:
        frame = frame[frame["gold_review_status"].eq("auto_approved")].copy()
    if limit is not None:
        frame = frame.head(limit)

    questions: list[EvaluationQuestion] = []
    for _, row in frame.iterrows():
        evaluation_id = row["evaluation_id"].strip()
        business_code = row["gold_business_function"].strip()
        business_label = BUSINESS_FUNCTION_MAP.get(business_code)
        if not business_label:
            raise ValueError(
                f"{evaluation_id}: 지원하지 않는 gold_business_function={business_code}"
            )
        primary = parse_json_array(
            row["gold_primary_chunk_ids"],
            "gold_primary_chunk_ids",
            evaluation_id,
        )
        supporting = parse_json_array(
            row["gold_supporting_chunk_ids"],
            "gold_supporting_chunk_ids",
            evaluation_id,
        )
        all_gold = parse_json_array(row["gold_chunk_ids"], "gold_chunk_ids", evaluation_id)
        if not all_gold:
            raise ValueError(f"{evaluation_id}: 검색평가대상 Y인데 Gold 청크가 없습니다.")

        questions.append(
            EvaluationQuestion(
                evaluation_id=evaluation_id,
                original_question_id=row.get("질문ID(원본)", "").strip(),
                question=row["예상질문"].strip(),
                domain_display=row["도메인"].strip(),
                business_function_code=business_code,
                business_function_label=business_label,
                complexity=row.get("질문 복잡도", "").strip(),
                importance=row.get("중요도", "").strip(),
                primary_gold=primary or all_gold,
                supporting_gold=[
                    chunk_id for chunk_id in supporting if chunk_id not in set(primary)
                ],
                all_gold=all_gold,
                multi_chunk_required=row["multi_chunk_required"].strip().upper() == "Y",
                review_status=row["gold_review_status"].strip(),
            )
        )

    ids = [question.evaluation_id for question in questions]
    duplicates = sorted({value for value in ids if ids.count(value) > 1})
    if duplicates:
        raise ValueError(f"evaluation_id 중복: {duplicates}")
    return questions


class DenseIndex:
    def __init__(
        self,
        *,
        chunks: list[dict[str, Any]],
        embeddings: list[dict[str, Any]],
    ) -> None:
        chunks_by_id = {str(chunk["chunk_id"]): chunk for chunk in chunks}
        if len(chunks_by_id) != len(chunks):
            raise RuntimeError("chunks.jsonl에 중복 chunk_id가 있습니다.")

        ids: list[str] = []
        labels: list[str] = []
        vectors: list[list[float]] = []
        models: set[str] = set()
        dimensions: set[int] = set()

        for record in embeddings:
            chunk_id = str(record.get("chunk_id", "")).strip()
            vector = record.get("embedding")
            if chunk_id not in chunks_by_id:
                raise RuntimeError(f"임베딩에만 존재하는 청크: {chunk_id}")
            if not isinstance(vector, list) or not vector:
                raise RuntimeError(f"임베딩 벡터가 비어 있습니다: {chunk_id}")
            ids.append(chunk_id)
            labels.append(str(record.get("business_function", "")).strip())
            vectors.append([float(value) for value in vector])
            models.add(str(record.get("model", "")).strip())
            dimensions.add(len(vector))

        if len(set(ids)) != len(ids):
            raise RuntimeError("임베딩 파일에 중복 chunk_id가 있습니다.")
        missing_embeddings = sorted(set(chunks_by_id) - set(ids))
        if missing_embeddings:
            raise RuntimeError(f"임베딩이 없는 청크: {missing_embeddings[:10]}")
        if len(dimensions) != 1:
            raise RuntimeError(f"임베딩 차원이 일치하지 않습니다: {dimensions}")

        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise RuntimeError("길이가 0인 임베딩 벡터가 있습니다.")
        self.matrix = matrix / norms
        self.chunk_ids = np.asarray(ids, dtype=object)
        self.business_labels = np.asarray(labels, dtype=object)
        self.chunks_by_id = chunks_by_id
        self.models = sorted(models)
        self.dimension = next(iter(dimensions))

    def search(
        self,
        query_vector: Iterable[float],
        *,
        top_k: int,
        business_function_label: str | None,
    ) -> list[dict[str, Any]]:
        query = np.asarray(list(query_vector), dtype=np.float32)
        if query.ndim != 1 or len(query) != self.dimension:
            raise ValueError(
                f"질문 임베딩 차원 불일치: query={len(query)}, index={self.dimension}"
            )
        norm = np.linalg.norm(query)
        if norm == 0:
            raise ValueError("질문 임베딩의 길이가 0입니다.")
        query = query / norm

        if business_function_label:
            candidate_indices = np.flatnonzero(
                self.business_labels == business_function_label
            )
        else:
            candidate_indices = np.arange(len(self.chunk_ids))
        if len(candidate_indices) == 0:
            raise RuntimeError(
                f"업무 필터에 해당하는 청크가 없습니다: {business_function_label}"
            )

        scores = self.matrix[candidate_indices] @ query
        count = min(top_k, len(candidate_indices))
        local_top = np.argpartition(-scores, count - 1)[:count]
        local_top = local_top[np.argsort(-scores[local_top], kind="stable")]

        results = []
        for local_index in local_top:
            global_index = int(candidate_indices[local_index])
            chunk_id = str(self.chunk_ids[global_index])
            results.append(
                {
                    "chunk_id": chunk_id,
                    "score": float(scores[local_index]),
                    "chunk": self.chunks_by_id[chunk_id],
                }
            )
        return results


class EmbeddingClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout_seconds: int,
        max_retries: int,
        cache_path: Path,
    ) -> None:
        self.api_key = api_key
        self.endpoint = base_url.rstrip("/") + "/embeddings"
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.cache_path = cache_path
        self.cache = self._load_cache()

    def _load_cache(self) -> dict[str, list[float]]:
        cache: dict[str, list[float]] = {}
        if not self.cache_path.exists():
            return cache
        with self.cache_path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                record = json.loads(line)
                cache[str(record["cache_key"])] = [
                    float(value) for value in record["embedding"]
                ]
        return cache

    def embed(self, text: str) -> tuple[list[float], bool]:
        cleaned = str(text).replace("\x00", "").strip()
        if not cleaned:
            raise ValueError("질문이 비어 있습니다.")
        cache_key = hashlib.sha256(
            f"{self.model}\n{cleaned}".encode("utf-8")
        ).hexdigest()
        if cache_key in self.cache:
            return self.cache[cache_key], True

        payload = json.dumps(
            {
                "model": self.model,
                "input": cleaned,
                "encoding_format": "float",
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(
                    request,
                    timeout=self.timeout_seconds,
                ) as response:
                    body = json.loads(response.read().decode("utf-8"))
                vector = [float(value) for value in body["data"][0]["embedding"]]
                self.cache[cache_key] = vector
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with self.cache_path.open("a", encoding="utf-8") as file:
                    file.write(
                        json.dumps(
                            {
                                "cache_key": cache_key,
                                "model": self.model,
                                "text": cleaned,
                                "embedding": vector,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                return vector, False
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
                last_error = error
                if attempt >= self.max_retries:
                    break
                time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"임베딩 API 호출 실패: {last_error}") from last_error


def get_api_key() -> str:
    key = os.environ.get("HCX_API_KEY", "").strip()
    if not key and sys.stdin.isatty():
        key = getpass.getpass("HCX_API_KEY를 입력하세요(화면에 표시되지 않음): ").strip()
    if not key:
        raise RuntimeError(
            "HCX_API_KEY가 없습니다. 환경 변수로 등록하거나 실행 시 입력하세요."
        )
    if key.lower().startswith("bearer "):
        raise RuntimeError("API 키 앞에 'Bearer '를 붙이지 마세요.")
    return key


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float(row[key])
        for row in rows
        if row.get(key) is not None and row.get(key) != ""
    ]
    return sum(values) / len(values) if values else None


METRIC_KEYS = [
    "hit_at_3",
    "recall_at_5",
    "mrr_at_10",
    "ap_at_10",
    "complete_at_5",
    "ndcg_at_5",
    "precision_at_5",
    "f1_at_5",
]


def summarize(rows: list[dict[str, Any]], group_name: str, group_value: str) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "group_name": group_name,
        "group_value": group_value,
        "question_count": len(rows),
        "complete_applicable_count": sum(
            row.get("complete_at_5") is not None for row in rows
        ),
    }
    for key in METRIC_KEYS:
        summary["map_at_10" if key == "ap_at_10" else key] = mean_metric(rows, key)
    summary["latency_ms_mean"] = mean_metric(rows, "latency_ms")
    return summary


def validate_gold_chunks(
    questions: list[EvaluationQuestion],
    available_chunk_ids: set[str],
) -> list[dict[str, str]]:
    issues = []
    for question in questions:
        for chunk_id in question.all_gold:
            if chunk_id not in available_chunk_ids:
                issues.append(
                    {
                        "evaluation_id": question.evaluation_id,
                        "question": question.question,
                        "missing_gold_chunk_id": chunk_id,
                    }
                )
    return issues


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="평가 XLSX 경로")
    parser.add_argument("--kdic-zip", type=Path, required=True, help="KDIC_output.zip 경로")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--sheet-name", default=DEFAULT_SHEET)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--approved-only", action="store_true")
    parser.add_argument("--no-domain-filter", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-missing-gold", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.top_k < 10:
        raise ValueError("MRR@10과 MAP@10 계산을 위해 --top-k는 10 이상이어야 합니다.")

    questions = load_questions(
        args.dataset,
        sheet_name=args.sheet_name,
        approved_only=args.approved_only,
        limit=args.limit,
    )
    with zipfile.ZipFile(args.kdic_zip) as archive:
        chunks = load_jsonl_from_zip(archive, "/processed/chunks.jsonl")
        embeddings = load_jsonl_from_zip(
            archive,
            "/processed/chunk_embeddings_hcx.jsonl",
        )
    dense_index = DenseIndex(chunks=chunks, embeddings=embeddings)
    missing_gold = validate_gold_chunks(questions, set(dense_index.chunk_ids.tolist()))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dry_report = {
        "status": "ready" if not missing_gold else "gold_validation_failed",
        "dataset": str(args.dataset.resolve()),
        "kdic_zip": str(args.kdic_zip.resolve()),
        "question_count": len(questions),
        "chunk_count": len(chunks),
        "embedding_count": len(embeddings),
        "embedding_models": dense_index.models,
        "embedding_dimension": dense_index.dimension,
        "domain_filter": not args.no_domain_filter,
        "missing_gold_count": len(missing_gold),
        "generated_at": utc_now_iso(),
    }
    (args.output_dir / "dry_run_report.json").write_text(
        json.dumps(dry_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(args.output_dir / "missing_gold.csv", missing_gold)
    print(json.dumps(dry_report, ensure_ascii=False, indent=2))

    if missing_gold and not args.allow_missing_gold:
        raise RuntimeError(
            "평가데이터셋의 Gold 청크 일부가 chunks.jsonl에 없습니다. "
            "missing_gold.csv를 확인하세요."
        )
    if args.dry_run:
        print("Dry-run 완료: API를 호출하지 않았습니다.")
        return 0

    client = EmbeddingClient(
        api_key=get_api_key(),
        base_url=args.base_url,
        model=args.model,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        cache_path=args.output_dir / "query_embedding_cache.jsonl",
    )

    details: list[dict[str, Any]] = []
    run_started = time.perf_counter()
    for number, question in enumerate(questions, start=1):
        query_started = time.perf_counter()
        query_vector, cache_hit = client.embed(question.question)
        results = dense_index.search(
            query_vector,
            top_k=args.top_k,
            business_function_label=(
                None if args.no_domain_filter else question.business_function_label
            ),
        )
        latency_ms = (time.perf_counter() - query_started) * 1000
        ranked_ids = [result["chunk_id"] for result in results]
        metrics = evaluate_ranking(
            ranked_ids,
            gold_ids=question.all_gold,
            primary_gold_ids=question.primary_gold,
            supporting_gold_ids=question.supporting_gold,
            multi_chunk_required=question.multi_chunk_required,
        )
        details.append(
            {
                "evaluation_id": question.evaluation_id,
                "question_id_original": question.original_question_id,
                "question": question.question,
                "domain": question.domain_display,
                "gold_business_function": question.business_function_code,
                "question_complexity": question.complexity,
                "importance": question.importance,
                "gold_review_status": question.review_status,
                "gold_primary_chunk_ids": json.dumps(
                    question.primary_gold, ensure_ascii=False
                ),
                "gold_supporting_chunk_ids": json.dumps(
                    question.supporting_gold, ensure_ascii=False
                ),
                "gold_chunk_ids": json.dumps(question.all_gold, ensure_ascii=False),
                "retrieved_chunk_ids": json.dumps(ranked_ids, ensure_ascii=False),
                "retrieved_scores": json.dumps(
                    [round(result["score"], 8) for result in results]
                ),
                "hit_at_3": metrics["hit_at_3"],
                "recall_at_5": metrics["recall_at_5"],
                "mrr_at_10": metrics["mrr_at_10"],
                "ap_at_10": metrics["ap_at_10"],
                "complete_at_5": metrics["complete_at_5"],
                "ndcg_at_5": metrics["ndcg_at_5"],
                "precision_at_5": metrics["precision_at_5"],
                "f1_at_5": metrics["f1_at_5"],
                "latency_ms": round(latency_ms, 3),
                "query_embedding_cache_hit": cache_hit,
            }
        )
        print(
            f"[{number:03d}/{len(questions):03d}] "
            f"{question.evaluation_id} Hit@3={metrics['hit_at_3']:.0f} "
            f"Recall@5={metrics['recall_at_5']:.3f}"
        )

    write_csv(args.output_dir / "question_results.csv", details)
    overall = summarize(details, "overall", "all")
    domain_summaries = [
        summarize(
            [row for row in details if row["gold_business_function"] == domain],
            "gold_business_function",
            domain,
        )
        for domain in sorted({row["gold_business_function"] for row in details})
    ]
    write_csv(args.output_dir / "summary_by_domain.csv", domain_summaries)

    result_summary = {
        "retriever": "BGE-M3 Dense cosine",
        "model": args.model,
        "top_k": args.top_k,
        "domain_filter": not args.no_domain_filter,
        "approved_only": args.approved_only,
        "overall": overall,
        "by_domain": domain_summaries,
        "total_runtime_seconds": round(time.perf_counter() - run_started, 3),
        "generated_at": utc_now_iso(),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "dataset": str(args.dataset.resolve()),
                "kdic_zip": str(args.kdic_zip.resolve()),
                "base_url": args.base_url,
                "model": args.model,
                "top_k": args.top_k,
                "domain_filter": not args.no_domain_filter,
                "approved_only": args.approved_only,
                "question_count": len(questions),
                "chunk_count": len(chunks),
                "embedding_dimension": dense_index.dimension,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\n평가 완료")
    print(json.dumps(overall, ensure_ascii=False, indent=2))
    print("결과 폴더:", args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
