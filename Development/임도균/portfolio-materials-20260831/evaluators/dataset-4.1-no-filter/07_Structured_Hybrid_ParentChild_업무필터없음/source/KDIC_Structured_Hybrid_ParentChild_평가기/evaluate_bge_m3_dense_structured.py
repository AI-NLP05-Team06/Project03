"""KDIC BGE-M3 Dense structured 검색 평가기.

제목·소제목·브레드크럼·본문을 하나의 문서 표현으로 결합하여 BGE-M3로
임베딩한 뒤 코사인 유사도로 검색합니다. 답변 생성은 수행하지 않습니다.
"""

from __future__ import annotations

import argparse
import json
import time
import zipfile
from pathlib import Path
from typing import Any

from metrics import evaluate_ranking
from evaluate_bge_m3_dense import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_SHEET,
    TOP_K,
    DenseIndex,
    EmbeddingClient,
    get_api_key,
    load_jsonl_from_zip,
    load_questions,
    summarize,
    utc_now_iso,
    validate_gold_chunks,
    write_csv,
)


def clean_text(value: Any) -> str:
    return str(value or "").replace("\x00", "").strip()


def structured_text(chunk: dict[str, Any]) -> str:
    heading_path = chunk.get("heading_path") or []
    if isinstance(heading_path, list):
        breadcrumb = " > ".join(clean_text(item) for item in heading_path if clean_text(item))
    else:
        breadcrumb = clean_text(heading_path)

    parts = []
    for label, value in (
        ("제목", chunk.get("title")),
        ("소제목", chunk.get("section_title")),
        ("경로", breadcrumb),
        ("본문", chunk.get("content")),
    ):
        text = clean_text(value)
        if text:
            parts.append(f"{label}: {text}")
    if not parts:
        raise ValueError(f"구조화할 텍스트가 없는 청크: {chunk.get('chunk_id')}")
    return "\n".join(parts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--kdic-zip", type=Path, required=True)
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

    chunk_ids = {str(chunk["chunk_id"]) for chunk in chunks}
    missing_gold = validate_gold_chunks(questions, chunk_ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dry_report = {
        "status": "ready" if not missing_gold else "gold_validation_failed",
        "retriever": "BGE-M3 Dense structured cosine",
        "dataset": str(args.dataset.resolve()),
        "kdic_zip": str(args.kdic_zip.resolve()),
        "sheet_name": args.sheet_name,
        "question_count": len(questions),
        "chunk_count": len(chunks),
        "model": args.model,
        "document_fields": ["title", "section_title", "heading_path", "content"],
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
        print("Dry-run 완료: API를 호출하지 않았습니다.")
        return 0

    client = EmbeddingClient(
        api_key=get_api_key(),
        base_url=args.base_url,
        model=args.model,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        cache_path=args.output_dir / "structured_embedding_cache.jsonl",
    )

    embeddings = []
    print(f"Structured 문서 임베딩 생성/캐시 확인: {len(chunks)}개")
    for number, chunk in enumerate(chunks, start=1):
        vector, cache_hit = client.embed(structured_text(chunk))
        embeddings.append({
            "chunk_id": str(chunk["chunk_id"]),
            "business_function": str(chunk.get("business_function", "")).strip(),
            "model": args.model,
            "embedding": vector,
        })
        if number % 25 == 0 or number == len(chunks):
            print(f"  {number}/{len(chunks)} 완료 (마지막 캐시={cache_hit})")

    dense_index = DenseIndex(chunks=chunks, embeddings=embeddings)
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
            "gold_supporting_chunk_ids": json.dumps(question.supporting_gold, ensure_ascii=False),
            "gold_chunk_ids": json.dumps(question.all_gold, ensure_ascii=False),
            "retrieved_chunk_ids": json.dumps(ranked_ids, ensure_ascii=False),
            "retrieved_scores": json.dumps([round(result["score"], 8) for result in results]),
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
        })
        print(
            f"[{number:03d}/{len(questions):03d}] {question.evaluation_id} "
            f"Hit@3={metrics['hit_at_3']:.0f} Recall@5={metrics['recall_at_5']:.3f}"
        )

    write_csv(args.output_dir / "question_results.csv", details)
    overall = summarize(details, "overall", "all")
    domains = sorted({row["gold_business_function"] for row in details})
    by_domain = [
        summarize(
            [row for row in details if row["gold_business_function"] == domain],
            "gold_business_function",
            domain,
        )
        for domain in domains
    ]
    write_csv(args.output_dir / "summary_by_domain.csv", by_domain)

    result_summary = {
        "retriever": "BGE-M3 Dense structured cosine",
        "model": args.model,
        "top_k": args.top_k,
        "domain_filter": not args.no_domain_filter,
        "document_fields": ["title", "section_title", "heading_path", "content"],
        "overall": overall,
        "by_domain": by_domain,
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
            "sheet_name": args.sheet_name,
            "retriever": result_summary["retriever"],
            "model": args.model,
            "top_k": args.top_k,
            "domain_filter": not args.no_domain_filter,
            "question_count": len(questions),
            "chunk_count": len(chunks),
            "document_fields": result_summary["document_fields"],
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n평가 완료")
    print(json.dumps(overall, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
