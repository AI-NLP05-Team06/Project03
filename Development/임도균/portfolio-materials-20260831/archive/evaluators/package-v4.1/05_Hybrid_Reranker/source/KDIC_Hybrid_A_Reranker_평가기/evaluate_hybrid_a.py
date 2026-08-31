"""KDIC Hybrid A 검색 평가기.

BGE-M3 Dense와 BM25-Nori Discard의 Top-10 순위를 가중 RRF로 결합합니다.
답변 생성은 하지 않으며, 최종 검색 순위만 Gold 청크와 비교합니다.
"""

from __future__ import annotations

import argparse
import json
import time
import zipfile
from pathlib import Path
from typing import Any

from evaluate_bge_m3_dense import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_SHEET,
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
from evaluate_bm25_nori import BM25Index, NoriTokenizer
from metrics import evaluate_ranking


DEFAULT_DENSE_WEIGHT = 0.85
DEFAULT_NORI_WEIGHT = 0.15
DEFAULT_RRF_CONSTANT = 10
DEFAULT_CANDIDATE_K = 10
DEFAULT_FINAL_K = 10
DEFAULT_BM25_K1 = 1.5
DEFAULT_BM25_B = 0.75
DEFAULT_LUCENE_VERSION = "9.12.2"


def weighted_rrf(
    dense_results: list[dict[str, Any]],
    nori_results: list[dict[str, Any]],
    *,
    dense_weight: float,
    nori_weight: float,
    rrf_constant: int,
    final_k: int,
) -> list[dict[str, Any]]:
    """두 순위 목록을 Chunk ID 기준으로 합치고 최종 순위를 반환합니다."""
    scores: dict[str, float] = {}
    evidence: dict[str, dict[str, Any]] = {}

    for source, weight, results in (
        ("dense", dense_weight, dense_results),
        ("nori", nori_weight, nori_results),
    ):
        for rank, result in enumerate(results, start=1):
            chunk_id = str(result["chunk_id"])
            scores[chunk_id] = scores.get(chunk_id, 0.0) + (
                weight / (rrf_constant + rank)
            )
            record = evidence.setdefault(
                chunk_id,
                {
                    "chunk_id": chunk_id,
                    "dense_rank": None,
                    "dense_score": None,
                    "nori_rank": None,
                    "nori_score": None,
                },
            )
            record[f"{source}_rank"] = rank
            record[f"{source}_score"] = float(result["score"])

    def sort_key(chunk_id: str) -> tuple[float, int, int, str]:
        record = evidence[chunk_id]
        dense_rank = record["dense_rank"] or 10**9
        nori_rank = record["nori_rank"] or 10**9
        return (-scores[chunk_id], dense_rank, nori_rank, chunk_id)

    ranked_ids = sorted(scores, key=sort_key)[:final_k]
    return [
        {
            **evidence[chunk_id],
            "rrf_score": scores[chunk_id],
        }
        for chunk_id in ranked_ids
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--kdic-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--sheet-name", default=DEFAULT_SHEET)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K)
    parser.add_argument("--final-k", type=int, default=DEFAULT_FINAL_K)
    parser.add_argument("--dense-weight", type=float, default=DEFAULT_DENSE_WEIGHT)
    parser.add_argument("--nori-weight", type=float, default=DEFAULT_NORI_WEIGHT)
    parser.add_argument("--rrf-constant", type=int, default=DEFAULT_RRF_CONSTANT)
    parser.add_argument("--bm25-k1", type=float, default=DEFAULT_BM25_K1)
    parser.add_argument("--bm25-b", type=float, default=DEFAULT_BM25_B)
    parser.add_argument("--lucene-version", default=DEFAULT_LUCENE_VERSION)
    parser.add_argument("--lucene-cache-dir", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--approved-only", action="store_true")
    parser.add_argument("--no-domain-filter", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-missing-gold", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.candidate_k < 10:
        raise ValueError("--candidate-k는 10 이상이어야 합니다.")
    if args.final_k < 10:
        raise ValueError("MRR@10과 MAP@10 계산을 위해 --final-k는 10 이상이어야 합니다.")
    if args.rrf_constant < 0:
        raise ValueError("--rrf-constant는 0 이상이어야 합니다.")
    if args.dense_weight < 0 or args.nori_weight < 0:
        raise ValueError("검색 가중치는 0 이상이어야 합니다.")
    weight_sum = args.dense_weight + args.nori_weight
    if abs(weight_sum - 1.0) > 1e-9:
        raise ValueError(
            "Dense와 Nori 가중치의 합은 1이어야 합니다: "
            f"{args.dense_weight} + {args.nori_weight} = {weight_sum}"
        )


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)

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
    available_chunk_ids = set(dense_index.chunk_ids.tolist())
    missing_gold = validate_gold_chunks(questions, available_chunk_ids)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dry_report = {
        "status": "ready" if not missing_gold else "gold_validation_failed",
        "retriever": (
            "Hybrid A: BGE-M3 Dense 0.85 + "
            "BM25-Nori Discard 0.15 (Weighted RRF)"
        ),
        "dataset": str(args.dataset.resolve()),
        "kdic_zip": str(args.kdic_zip.resolve()),
        "question_count": len(questions),
        "chunk_count": len(chunks),
        "embedding_count": len(embeddings),
        "embedding_models": dense_index.models,
        "embedding_dimension": dense_index.dimension,
        "candidate_k": args.candidate_k,
        "final_k": args.final_k,
        "dense_weight": args.dense_weight,
        "nori_weight": args.nori_weight,
        "rrf_constant": args.rrf_constant,
        "nori_decompound_mode": "discard",
        "bm25_k1": args.bm25_k1,
        "bm25_b": args.bm25_b,
        "lucene_version": args.lucene_version,
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
        print("Dry-run 완료: HCX API와 Lucene Nori 검색은 실행하지 않았습니다.")
        return 0

    lucene_cache_dir = (
        args.lucene_cache_dir
        if args.lucene_cache_dir
        else args.output_dir / "lucene_jars"
    )
    print("Lucene Nori Discard 로딩 및 BM25 인덱스 생성")
    tokenizer = NoriTokenizer(
        lucene_cache_dir,
        args.lucene_version,
        "discard",
    )
    nori_index = BM25Index(
        chunks=chunks,
        tokenizer=tokenizer,
        k1=args.bm25_k1,
        b=args.bm25_b,
    )
    print(
        "Nori 토큰화 확인:",
        tokenizer.tokenize("착오송금 반환지원 신청 기간과 필요한 서류"),
    )

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
        business_label = (
            None if args.no_domain_filter else question.business_function_label
        )
        query_vector, cache_hit = client.embed(question.question)
        dense_results = dense_index.search(
            query_vector,
            top_k=args.candidate_k,
            business_function_label=business_label,
        )
        nori_results = nori_index.search(
            question.question,
            top_k=args.candidate_k,
            business_function_label=business_label,
        )
        fused_results = weighted_rrf(
            dense_results,
            nori_results,
            dense_weight=args.dense_weight,
            nori_weight=args.nori_weight,
            rrf_constant=args.rrf_constant,
            final_k=args.final_k,
        )
        latency_ms = (time.perf_counter() - query_started) * 1000
        ranked_ids = [result["chunk_id"] for result in fused_results]
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
                "gold_chunk_ids": json.dumps(
                    question.all_gold, ensure_ascii=False
                ),
                "retrieved_chunk_ids": json.dumps(
                    ranked_ids, ensure_ascii=False
                ),
                "retrieved_scores": json.dumps(
                    [
                        round(result["rrf_score"], 10)
                        for result in fused_results
                    ]
                ),
                "dense_candidate_ids": json.dumps(
                    [result["chunk_id"] for result in dense_results],
                    ensure_ascii=False,
                ),
                "dense_candidate_scores": json.dumps(
                    [round(result["score"], 8) for result in dense_results]
                ),
                "nori_candidate_ids": json.dumps(
                    [result["chunk_id"] for result in nori_results],
                    ensure_ascii=False,
                ),
                "nori_candidate_scores": json.dumps(
                    [round(result["score"], 8) for result in nori_results]
                ),
                "rrf_rank_evidence": json.dumps(
                    [
                        {
                            "chunk_id": result["chunk_id"],
                            "dense_rank": result["dense_rank"],
                            "nori_rank": result["nori_rank"],
                        }
                        for result in fused_results
                    ],
                    ensure_ascii=False,
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
            [
                row
                for row in details
                if row["gold_business_function"] == domain
            ],
            "gold_business_function",
            domain,
        )
        for domain in sorted(
            {row["gold_business_function"] for row in details}
        )
    ]
    write_csv(args.output_dir / "summary_by_domain.csv", domain_summaries)

    retriever_name = (
        f"Hybrid A: BGE-M3 Dense {args.dense_weight:.2f} + "
        f"BM25-Nori Discard {args.nori_weight:.2f} "
        f"(Weighted RRF, c={args.rrf_constant})"
    )
    result_summary = {
        "retriever": retriever_name,
        "model": args.model,
        "candidate_k": args.candidate_k,
        "top_k": args.final_k,
        "dense_weight": args.dense_weight,
        "nori_weight": args.nori_weight,
        "rrf_constant": args.rrf_constant,
        "nori_decompound_mode": "discard",
        "bm25_k1": args.bm25_k1,
        "bm25_b": args.bm25_b,
        "domain_filter": not args.no_domain_filter,
        "approved_only": args.approved_only,
        "overall": overall,
        "by_domain": domain_summaries,
        "total_runtime_seconds": round(
            time.perf_counter() - run_started, 3
        ),
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
                "candidate_k": args.candidate_k,
                "final_k": args.final_k,
                "dense_weight": args.dense_weight,
                "nori_weight": args.nori_weight,
                "rrf_constant": args.rrf_constant,
                "nori_decompound_mode": "discard",
                "bm25_k1": args.bm25_k1,
                "bm25_b": args.bm25_b,
                "lucene_version": args.lucene_version,
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
    print("\nHybrid A 평가 완료")
    print(json.dumps(overall, ensure_ascii=False, indent=2))
    print("결과 폴더:", args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
