"""KDIC BGE-M3 Sparse 검색 일괄 평가기.

답변은 생성하지 않습니다. BGE-M3의 lexical_weights로 427개 청크를 검색하고
질문별 Top-10을 Gold 청크와 비교합니다.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from metrics import evaluate_ranking


DEFAULT_MODEL = "BAAI/bge-m3"
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
    flattened: list[str] = []
    stack = list(parsed)
    while stack:
        item = stack.pop(0)
        if isinstance(item, list):
            stack = list(item) + stack
            continue
        normalized = str(item).strip()
        if normalized and normalized not in flattened:
            flattened.append(normalized)
    return flattened


def load_jsonl_from_zip(archive: zipfile.ZipFile, suffix: str) -> list[dict[str, Any]]:
    candidates = [
        name for name in archive.namelist()
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
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{candidates[0]} {line_number}행 JSON 파싱 실패"
                ) from error
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
    # Evaluation_DataSet_v3.5의 영문 칼럼을 기존 평가기 내부 이름으로만 매핑합니다.
    # 원본 XLSX는 수정하거나 다시 저장하지 않습니다.
    frame.columns = [str(column).strip() for column in frame.columns]
    alias_map = {
        "question": "예상질문",
        "question_id": "질문ID(원본)",
        "domain": "도메인",
        "complexity": "질문 복잡도",
        "importance": "중요도",
    }
    frame = frame.rename(columns={
        source: target
        for source, target in alias_map.items()
        if source in frame.columns and target not in frame.columns
    })
    if "검색평가대상" not in frame.columns:
        if "evaluation_id" not in frame.columns:
            raise ValueError("평가데이터셋에 evaluation_id 칼럼이 없습니다.")
        frame["검색평가대상"] = frame["evaluation_id"].astype(str).str.strip().map(
            lambda value: "Y" if value else ""
        )
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
            row["gold_primary_chunk_ids"], "gold_primary_chunk_ids", evaluation_id
        )
        supporting = parse_json_array(
            row["gold_supporting_chunk_ids"], "gold_supporting_chunk_ids", evaluation_id
        )
        all_gold = parse_json_array(row["gold_chunk_ids"], "gold_chunk_ids", evaluation_id)
        if not all_gold:
            raise ValueError(f"{evaluation_id}: 검색평가대상 Y인데 Gold 청크가 없습니다.")
        primary_set = set(primary)
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
                supporting_gold=[x for x in supporting if x not in primary_set],
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


def clean_text(value: Any) -> str:
    return str(value or "").replace("\x00", "").strip()


def normalize_lexical_weights(weights: Any) -> dict[str, float]:
    if not isinstance(weights, dict):
        raise ValueError("BGE-M3 lexical_weights가 사전 형태가 아닙니다.")
    return {
        str(token_id): float(weight)
        for token_id, weight in weights.items()
        if float(weight) > 0
    }


class SparseIndex:
    def __init__(
        self,
        *,
        chunks: list[dict[str, Any]],
        model: Any,
        model_name: str,
        cache_path: Path,
        batch_size: int,
        max_passage_length: int,
    ) -> None:
        self.model = model
        self.model_name = model_name
        self.cache_path = cache_path
        self.max_passage_length = max_passage_length

        chunks_by_id = {str(chunk["chunk_id"]): chunk for chunk in chunks}
        if len(chunks_by_id) != len(chunks):
            raise RuntimeError("chunks.jsonl에 중복 chunk_id가 있습니다.")

        cached = self._load_cache()
        missing: list[dict[str, Any]] = []
        lexical_by_id: dict[str, dict[str, float]] = {}
        for chunk in chunks:
            chunk_id = str(chunk["chunk_id"])
            record = cached.get(chunk_id)
            if (
                record
                and record.get("model") == model_name
                and int(record.get("max_length", 0)) == max_passage_length
                and record.get("content_hash") == chunk.get("content_hash")
            ):
                lexical_by_id[chunk_id] = normalize_lexical_weights(
                    record["lexical_weights"]
                )
            else:
                missing.append(chunk)

        if missing:
            print(f"Sparse 청크 벡터 생성: {len(missing)}개")
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            for start in range(0, len(missing), batch_size):
                batch = missing[start:start + batch_size]
                texts = [clean_text(chunk.get("content")) for chunk in batch]
                output = model.encode(
                    texts,
                    batch_size=batch_size,
                    max_length=max_passage_length,
                    return_dense=False,
                    return_sparse=True,
                    return_colbert_vecs=False,
                )
                weights_batch = output["lexical_weights"]
                with self.cache_path.open("a", encoding="utf-8") as file:
                    for chunk, weights in zip(batch, weights_batch):
                        chunk_id = str(chunk["chunk_id"])
                        normalized = normalize_lexical_weights(weights)
                        lexical_by_id[chunk_id] = normalized
                        file.write(json.dumps({
                            "chunk_id": chunk_id,
                            "content_hash": chunk.get("content_hash"),
                            "model": model_name,
                            "max_length": max_passage_length,
                            "lexical_weights": normalized,
                        }, ensure_ascii=False) + "\n")
                print(f"  {min(start + batch_size, len(missing))}/{len(missing)} 완료")
        else:
            print("청크 Sparse 벡터 캐시 사용")

        self.chunk_ids = [str(chunk["chunk_id"]) for chunk in chunks]
        self.business_labels = [
            str(chunk.get("business_function", "")).strip() for chunk in chunks
        ]
        self.weights = [lexical_by_id[chunk_id] for chunk_id in self.chunk_ids]
        self.chunks_by_id = chunks_by_id

    def _load_cache(self) -> dict[str, dict[str, Any]]:
        records: dict[str, dict[str, Any]] = {}
        if not self.cache_path.exists():
            return records
        with self.cache_path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                record = json.loads(line)
                records[str(record["chunk_id"])] = record
        return records

    @staticmethod
    def lexical_score(
        query_weights: dict[str, float],
        document_weights: dict[str, float],
    ) -> float:
        if len(query_weights) > len(document_weights):
            query_weights, document_weights = document_weights, query_weights
        return sum(
            weight * document_weights.get(token_id, 0.0)
            for token_id, weight in query_weights.items()
        )

    def search(
        self,
        query_weights: dict[str, float],
        *,
        top_k: int,
        business_function_label: str | None,
    ) -> list[dict[str, Any]]:
        candidate_indices = [
            index for index, label in enumerate(self.business_labels)
            if business_function_label is None or label == business_function_label
        ]
        if not candidate_indices:
            raise RuntimeError(
                f"업무 필터에 해당하는 청크가 없습니다: {business_function_label}"
            )
        scored = [
            (
                self.lexical_score(query_weights, self.weights[index]),
                index,
            )
            for index in candidate_indices
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        results = []
        for score, index in scored[:min(top_k, len(scored))]:
            chunk_id = self.chunk_ids[index]
            results.append({
                "chunk_id": chunk_id,
                "score": float(score),
                "chunk": self.chunks_by_id[chunk_id],
            })
        return results


class SparseQueryEncoder:
    def __init__(
        self,
        *,
        model: Any,
        model_name: str,
        max_query_length: int,
        cache_path: Path,
    ) -> None:
        self.model = model
        self.model_name = model_name
        self.max_query_length = max_query_length
        self.cache_path = cache_path
        self.cache = self._load_cache()

    def _load_cache(self) -> dict[str, dict[str, float]]:
        cache: dict[str, dict[str, float]] = {}
        if not self.cache_path.exists():
            return cache
        with self.cache_path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                record = json.loads(line)
                if (
                    record.get("model") == self.model_name
                    and int(record.get("max_length", 0)) == self.max_query_length
                ):
                    cache[str(record["text"])] = normalize_lexical_weights(
                        record["lexical_weights"]
                    )
        return cache

    def encode(self, text: str) -> tuple[dict[str, float], bool]:
        cleaned = clean_text(text)
        if not cleaned:
            raise ValueError("질문이 비어 있습니다.")
        if cleaned in self.cache:
            return self.cache[cleaned], True
        output = self.model.encode(
            [cleaned],
            batch_size=1,
            max_length=self.max_query_length,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        weights = normalize_lexical_weights(output["lexical_weights"][0])
        self.cache[cleaned] = weights
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps({
                "text": cleaned,
                "model": self.model_name,
                "max_length": self.max_query_length,
                "lexical_weights": weights,
            }, ensure_ascii=False) + "\n")
        return weights, False


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
        float(row[key]) for row in rows
        if row.get(key) is not None and row.get(key) != ""
    ]
    return sum(values) / len(values) if values else None


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
                issues.append({
                    "evaluation_id": question.evaluation_id,
                    "question": question.question,
                    "missing_gold_chunk_id": chunk_id,
                })
    return issues


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--kdic-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--sheet-name", default=DEFAULT_SHEET)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-passage-length", type=int, default=2048)
    parser.add_argument("--max-query-length", type=int, default=512)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--approved-only", action="store_true")
    parser.add_argument("--no-domain-filter", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-missing-gold", action="store_true")
    parser.add_argument("--no-fp16", action="store_true")
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
    chunk_ids = {str(chunk["chunk_id"]) for chunk in chunks}
    missing_gold = validate_gold_chunks(questions, chunk_ids)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dry_report = {
        "status": "ready" if not missing_gold else "gold_validation_failed",
        "retriever": "BGE-M3 Sparse lexical matching",
        "dataset": str(args.dataset.resolve()),
        "kdic_zip": str(args.kdic_zip.resolve()),
        "question_count": len(questions),
        "chunk_count": len(chunks),
        "model": args.model,
        "max_passage_length": args.max_passage_length,
        "max_query_length": args.max_query_length,
        "domain_filter": not args.no_domain_filter,
        "missing_gold_count": len(missing_gold),
        "generated_at": utc_now_iso(),
    }
    (args.output_dir / "dry_run_report.json").write_text(
        json.dumps(dry_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(args.output_dir / "missing_gold.csv", missing_gold)
    print(json.dumps(dry_report, ensure_ascii=False, indent=2))

    if missing_gold and not args.allow_missing_gold:
        raise RuntimeError(
            "평가데이터셋의 Gold 청크 일부가 chunks.jsonl에 없습니다. "
            "missing_gold.csv를 확인하세요."
        )
    if args.dry_run:
        print("Dry-run 완료: 모델을 내려받거나 벡터를 생성하지 않았습니다.")
        return 0

    try:
        from FlagEmbedding import BGEM3FlagModel
    except ImportError as error:
        raise RuntimeError(
            "FlagEmbedding이 없습니다. pip install -U FlagEmbedding을 실행하세요."
        ) from error

    print(f"모델 로딩: {args.model}")
    model = BGEM3FlagModel(args.model, use_fp16=not args.no_fp16)
    index = SparseIndex(
        chunks=chunks,
        model=model,
        model_name=args.model,
        cache_path=args.output_dir / "chunk_sparse_vectors.jsonl",
        batch_size=args.batch_size,
        max_passage_length=args.max_passage_length,
    )
    query_encoder = SparseQueryEncoder(
        model=model,
        model_name=args.model,
        max_query_length=args.max_query_length,
        cache_path=args.output_dir / "query_sparse_cache.jsonl",
    )

    details: list[dict[str, Any]] = []
    run_started = time.perf_counter()
    for number, question in enumerate(questions, start=1):
        query_started = time.perf_counter()
        query_weights, cache_hit = query_encoder.encode(question.question)
        results = index.search(
            query_weights,
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
        details.append({
            "evaluation_id": question.evaluation_id,
            "question_id_original": question.original_question_id,
            "question": question.question,
            "domain": question.domain_display,
            "gold_business_function": question.business_function_code,
            "question_complexity": question.complexity,
            "importance": question.importance,
            "gold_review_status": question.review_status,
            "gold_primary_chunk_ids": json.dumps(question.primary_gold, ensure_ascii=False),
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
            "query_sparse_cache_hit": cache_hit,
            "query_sparse_token_count": len(query_weights),
        })
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
        "retriever": "BGE-M3 Sparse lexical matching",
        "model": args.model,
        "top_k": args.top_k,
        "domain_filter": not args.no_domain_filter,
        "approved_only": args.approved_only,
        "max_passage_length": args.max_passage_length,
        "max_query_length": args.max_query_length,
        "overall": overall,
        "by_domain": domain_summaries,
        "total_runtime_seconds": round(time.perf_counter() - run_started, 3),
        "generated_at": utc_now_iso(),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps({
            "dataset": str(args.dataset.resolve()),
            "kdic_zip": str(args.kdic_zip.resolve()),
            "retriever": result_summary["retriever"],
            "model": args.model,
            "top_k": args.top_k,
            "domain_filter": not args.no_domain_filter,
            "approved_only": args.approved_only,
            "question_count": len(questions),
            "chunk_count": len(chunks),
            "batch_size": args.batch_size,
            "max_passage_length": args.max_passage_length,
            "max_query_length": args.max_query_length,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n평가 완료")
    print(json.dumps(overall, ensure_ascii=False, indent=2))
    print("결과 폴더:", args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
