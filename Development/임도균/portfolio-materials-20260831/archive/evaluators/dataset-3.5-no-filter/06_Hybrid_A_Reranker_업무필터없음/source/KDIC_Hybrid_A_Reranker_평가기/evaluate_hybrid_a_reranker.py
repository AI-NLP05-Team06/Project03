"""KDIC Hybrid A + BGE Reranker 검색 평가기.

Hybrid A가 만든 후보를 BAAI/bge-reranker-v2-m3 Cross-Encoder로 재정렬하고,
최종 Top-10을 Gold 청크와 비교합니다. 답변 생성은 수행하지 않습니다.
"""

from __future__ import annotations

import argparse
import hashlib
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
from evaluate_hybrid_a import weighted_rrf
from metrics import evaluate_ranking


DEFAULT_DENSE_WEIGHT = 0.85
DEFAULT_NORI_WEIGHT = 0.15
DEFAULT_RRF_CONSTANT = 10
DEFAULT_SOURCE_K = 10
DEFAULT_RERANK_K = 20
DEFAULT_FINAL_K = 10
DEFAULT_BM25_K1 = 1.5
DEFAULT_BM25_B = 0.75
DEFAULT_LUCENE_VERSION = "9.12.2"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_RERANKER_BATCH_SIZE = 8
DEFAULT_RERANKER_MAX_LENGTH = 512


def build_passage(chunk: dict[str, Any]) -> str:
    """제목·부제목·본문을 중복 없이 결합해 Reranker 입력을 만듭니다."""
    parts: list[str] = []
    for key in ("title", "section_title"):
        value = str(chunk.get(key) or "").strip()
        if value and value not in parts:
            parts.append(value)
    heading_path = chunk.get("heading_path")
    if isinstance(heading_path, list):
        heading = " > ".join(
            str(value).strip() for value in heading_path if str(value).strip()
        )
        if heading and heading not in parts:
            parts.append(heading)
    content = str(chunk.get("content") or "").strip()
    if content:
        parts.append(content)
    passage = "\n\n".join(parts).strip()
    if not passage:
        raise ValueError(f"Reranker 본문이 비어 있습니다: {chunk.get('chunk_id')}")
    return passage


class LocalBGEReranker:
    def __init__(
        self,
        *,
        model_name: str,
        batch_size: int,
        max_length: int,
        cache_path: Path,
        allow_cpu: bool,
    ) -> None:
        try:
            import torch
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as error:
            raise RuntimeError(
                "torch 또는 transformers가 없습니다. requirements를 설치하세요."
            ) from error

        self.torch = torch
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.cache_path = cache_path
        self.cache = self._load_cache()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu" and not allow_cpu:
            raise RuntimeError(
                "GPU를 찾지 못했습니다. Colab 런타임을 T4 GPU로 변경하세요. "
                "CPU 실행을 감수하려면 --allow-cpu를 사용하세요."
            )

        print(f"Reranker 로딩: {model_name}, device={self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_kwargs: dict[str, Any] = {}
        if self.device == "cuda":
            model_kwargs["dtype"] = torch.float16
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            **model_kwargs,
        )
        self.model.to(self.device)
        self.model.eval()

    def _load_cache(self) -> dict[str, float]:
        cache: dict[str, float] = {}
        if not self.cache_path.exists():
            return cache
        with self.cache_path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                record = json.loads(line)
                cache[str(record["cache_key"])] = float(record["score"])
        return cache

    def _cache_key(self, query: str, passage: str) -> str:
        payload = (
            f"{self.model_name}\n{self.max_length}\n{query}\n{passage}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def score_pairs(
        self,
        query: str,
        passages: list[str],
    ) -> tuple[list[float], int]:
        keys = [self._cache_key(query, passage) for passage in passages]
        scores: list[float | None] = [
            self.cache.get(key) for key in keys
        ]
        missing_indices = [
            index for index, score in enumerate(scores) if score is None
        ]
        cache_hit_count = len(scores) - len(missing_indices)

        for start in range(0, len(missing_indices), self.batch_size):
            batch_indices = missing_indices[start : start + self.batch_size]
            pairs = [[query, passages[index]] for index in batch_indices]
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length,
            )
            inputs = {
                key: value.to(self.device) for key, value in inputs.items()
            }
            with self.torch.inference_mode():
                logits = self.model(
                    **inputs,
                    return_dict=True,
                ).logits.view(-1).float().cpu().tolist()
            if not isinstance(logits, list):
                logits = [float(logits)]
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as file:
                for index, raw_score in zip(batch_indices, logits):
                    score = float(raw_score)
                    scores[index] = score
                    self.cache[keys[index]] = score
                    file.write(
                        json.dumps(
                            {
                                "cache_key": keys[index],
                                "model": self.model_name,
                                "max_length": self.max_length,
                                "score": score,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        return [float(score) for score in scores], cache_hit_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--kdic-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--sheet-name", default=DEFAULT_SHEET)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--source-k", type=int, default=DEFAULT_SOURCE_K)
    parser.add_argument("--rerank-k", type=int, default=DEFAULT_RERANK_K)
    parser.add_argument("--final-k", type=int, default=DEFAULT_FINAL_K)
    parser.add_argument("--dense-weight", type=float, default=DEFAULT_DENSE_WEIGHT)
    parser.add_argument("--nori-weight", type=float, default=DEFAULT_NORI_WEIGHT)
    parser.add_argument("--rrf-constant", type=int, default=DEFAULT_RRF_CONSTANT)
    parser.add_argument("--bm25-k1", type=float, default=DEFAULT_BM25_K1)
    parser.add_argument("--bm25-b", type=float, default=DEFAULT_BM25_B)
    parser.add_argument("--lucene-version", default=DEFAULT_LUCENE_VERSION)
    parser.add_argument("--lucene-cache-dir", type=Path)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument(
        "--reranker-batch-size",
        type=int,
        default=DEFAULT_RERANKER_BATCH_SIZE,
    )
    parser.add_argument(
        "--reranker-max-length",
        type=int,
        default=DEFAULT_RERANKER_MAX_LENGTH,
    )
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--approved-only", action="store_true")
    parser.add_argument("--no-domain-filter", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-missing-gold", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.source_k < 10:
        raise ValueError("--source-k는 10 이상이어야 합니다.")
    if args.rerank_k < args.final_k:
        raise ValueError("--rerank-k는 --final-k 이상이어야 합니다.")
    if args.final_k < 10:
        raise ValueError("MRR@10과 MAP@10을 위해 --final-k는 10 이상이어야 합니다.")
    if args.rrf_constant < 0:
        raise ValueError("--rrf-constant는 0 이상이어야 합니다.")
    if args.reranker_batch_size < 1:
        raise ValueError("--reranker-batch-size는 1 이상이어야 합니다.")
    if args.reranker_max_length < 64:
        raise ValueError("--reranker-max-length는 64 이상이어야 합니다.")
    weight_sum = args.dense_weight + args.nori_weight
    if args.dense_weight < 0 or args.nori_weight < 0:
        raise ValueError("검색 가중치는 0 이상이어야 합니다.")
    if abs(weight_sum - 1.0) > 1e-9:
        raise ValueError("Dense와 Nori 가중치의 합은 1이어야 합니다.")


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
    missing_gold = validate_gold_chunks(
        questions,
        set(dense_index.chunk_ids.tolist()),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    legacy_before_csv = (
        args.output_dir / "question_results_before_rerank.csv"
    )
    if legacy_before_csv.exists():
        legacy_before_csv.unlink()
    dry_report = {
        "status": "ready" if not missing_gold else "gold_validation_failed",
        "retriever": (
            "Hybrid A + BGE Reranker: Dense 0.85 + "
            "Nori Discard 0.15, Weighted RRF"
        ),
        "dataset": str(args.dataset.resolve()),
        "kdic_zip": str(args.kdic_zip.resolve()),
        "question_count": len(questions),
        "chunk_count": len(chunks),
        "embedding_count": len(embeddings),
        "embedding_models": dense_index.models,
        "embedding_dimension": dense_index.dimension,
        "source_k": args.source_k,
        "rerank_k": args.rerank_k,
        "final_k": args.final_k,
        "dense_weight": args.dense_weight,
        "nori_weight": args.nori_weight,
        "rrf_constant": args.rrf_constant,
        "nori_decompound_mode": "discard",
        "reranker_model": args.reranker_model,
        "reranker_batch_size": args.reranker_batch_size,
        "reranker_max_length": args.reranker_max_length,
        "reranker_text_fields": [
            "title",
            "section_title",
            "heading_path",
            "content",
        ],
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
            "Gold 청크 일부가 chunks.jsonl에 없습니다. "
            "missing_gold.csv를 확인하세요."
        )
    if args.dry_run:
        print(
            "Dry-run 완료: HCX API, Lucene Nori, Reranker 모델은 "
            "실행하지 않았습니다."
        )
        return 0

    lucene_cache_dir = (
        args.lucene_cache_dir
        if args.lucene_cache_dir
        else args.output_dir / "lucene_jars"
    )
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
    client = EmbeddingClient(
        api_key=get_api_key(),
        base_url=args.base_url,
        model=args.model,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        cache_path=args.output_dir / "query_embedding_cache.jsonl",
    )
    reranker = LocalBGEReranker(
        model_name=args.reranker_model,
        batch_size=args.reranker_batch_size,
        max_length=args.reranker_max_length,
        cache_path=args.output_dir / "reranker_score_cache.jsonl",
        allow_cpu=args.allow_cpu,
    )

    details: list[dict[str, Any]] = []
    before_details: list[dict[str, Any]] = []
    run_started = time.perf_counter()
    for number, question in enumerate(questions, start=1):
        query_started = time.perf_counter()
        business_label = (
            None if args.no_domain_filter else question.business_function_label
        )
        query_vector, embedding_cache_hit = client.embed(question.question)
        dense_results = dense_index.search(
            query_vector,
            top_k=args.source_k,
            business_function_label=business_label,
        )
        nori_results = nori_index.search(
            question.question,
            top_k=args.source_k,
            business_function_label=business_label,
        )
        hybrid_pool = weighted_rrf(
            dense_results,
            nori_results,
            dense_weight=args.dense_weight,
            nori_weight=args.nori_weight,
            rrf_constant=args.rrf_constant,
            final_k=args.rerank_k,
        )
        retrieval_latency_ms = (
            time.perf_counter() - query_started
        ) * 1000

        before_ids = [
            result["chunk_id"] for result in hybrid_pool[: args.final_k]
        ]
        before_metrics = evaluate_ranking(
            before_ids,
            gold_ids=question.all_gold,
            primary_gold_ids=question.primary_gold,
            supporting_gold_ids=question.supporting_gold,
            multi_chunk_required=question.multi_chunk_required,
        )

        rerank_started = time.perf_counter()
        passages = [
            build_passage(dense_index.chunks_by_id[result["chunk_id"]])
            for result in hybrid_pool
        ]
        reranker_scores, reranker_cache_hits = reranker.score_pairs(
            question.question,
            passages,
        )
        reranked = sorted(
            [
                {
                    **result,
                    "reranker_score": float(score),
                    "hybrid_rank": index + 1,
                }
                for index, (result, score) in enumerate(
                    zip(hybrid_pool, reranker_scores)
                )
            ],
            key=lambda result: (
                -result["reranker_score"],
                result["hybrid_rank"],
                result["chunk_id"],
            ),
        )
        rerank_latency_ms = (
            time.perf_counter() - rerank_started
        ) * 1000
        final_results = reranked[: args.final_k]
        ranked_ids = [result["chunk_id"] for result in final_results]
        metrics = evaluate_ranking(
            ranked_ids,
            gold_ids=question.all_gold,
            primary_gold_ids=question.primary_gold,
            supporting_gold_ids=question.supporting_gold,
            multi_chunk_required=question.multi_chunk_required,
        )
        common = {
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
        }
        before_details.append(
            {
                **common,
                "retrieved_chunk_ids": json.dumps(
                    before_ids, ensure_ascii=False
                ),
                **before_metrics,
                "latency_ms": round(retrieval_latency_ms, 3),
            }
        )
        details.append(
            {
                **common,
                "retrieved_chunk_ids": json.dumps(
                    ranked_ids, ensure_ascii=False
                ),
                "retrieved_scores": json.dumps(
                    [
                        round(result["reranker_score"], 8)
                        for result in final_results
                    ]
                ),
                "hybrid_before_rerank_ids": json.dumps(
                    before_ids, ensure_ascii=False
                ),
                "hybrid_candidate_ids": json.dumps(
                    [result["chunk_id"] for result in hybrid_pool],
                    ensure_ascii=False,
                ),
                "hybrid_candidate_rrf_scores": json.dumps(
                    [
                        round(result["rrf_score"], 10)
                        for result in hybrid_pool
                    ]
                ),
                "rerank_evidence": json.dumps(
                    [
                        {
                            "chunk_id": result["chunk_id"],
                            "hybrid_rank": result["hybrid_rank"],
                            "reranker_score": round(
                                result["reranker_score"], 8
                            ),
                        }
                        for result in final_results
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
                "before_hit_at_3": before_metrics["hit_at_3"],
                "before_recall_at_5": before_metrics["recall_at_5"],
                "before_mrr_at_10": before_metrics["mrr_at_10"],
                "before_ap_at_10": before_metrics["ap_at_10"],
                "before_complete_at_5": before_metrics["complete_at_5"],
                "before_ndcg_at_5": before_metrics["ndcg_at_5"],
                "before_precision_at_5": before_metrics["precision_at_5"],
                "before_f1_at_5": before_metrics["f1_at_5"],
                "retrieval_latency_ms": round(retrieval_latency_ms, 3),
                "rerank_latency_ms": round(rerank_latency_ms, 3),
                "latency_ms": round(
                    retrieval_latency_ms + rerank_latency_ms, 3
                ),
                "query_embedding_cache_hit": embedding_cache_hit,
                "reranker_cache_hit_count": reranker_cache_hits,
            }
        )
        print(
            f"[{number:03d}/{len(questions):03d}] "
            f"{question.evaluation_id} "
            f"Hit@3 {before_metrics['hit_at_3']:.0f}"
            f"→{metrics['hit_at_3']:.0f}, "
            f"MRR {before_metrics['mrr_at_10']:.3f}"
            f"→{metrics['mrr_at_10']:.3f}"
        )

    write_csv(args.output_dir / "question_results.csv", details)
    overall = summarize(details, "overall", "all")
    before_overall = summarize(before_details, "overall", "all")
    domain_summaries = [
        summarize(
            [
                row for row in details
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
    comparison_rows: list[dict[str, Any]] = []
    for key in (
        "hit_at_3",
        "recall_at_5",
        "mrr_at_10",
        "map_at_10",
        "complete_at_5",
        "ndcg_at_5",
        "precision_at_5",
        "f1_at_5",
        "latency_ms_mean",
    ):
        before_value = before_overall.get(key)
        after_value = overall.get(key)
        comparison_rows.append(
            {
                "metric": key,
                "hybrid_a_before": before_value,
                "after_reranker": after_value,
                "delta": (
                    None
                    if before_value is None or after_value is None
                    else after_value - before_value
                ),
            }
        )
    write_csv(
        args.output_dir / "comparison_before_after.csv",
        comparison_rows,
    )
    retriever_name = (
        "Hybrid A + BGE Reranker v2-m3 "
        f"(Dense {args.dense_weight:.2f}, "
        f"Nori {args.nori_weight:.2f}, "
        f"RRF c={args.rrf_constant})"
    )
    result_summary = {
        "retriever": retriever_name,
        "model": args.model,
        "reranker_model": args.reranker_model,
        "source_k": args.source_k,
        "rerank_k": args.rerank_k,
        "top_k": args.final_k,
        "dense_weight": args.dense_weight,
        "nori_weight": args.nori_weight,
        "rrf_constant": args.rrf_constant,
        "domain_filter": not args.no_domain_filter,
        "overall": overall,
        "before_rerank_overall": before_overall,
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
    (args.output_dir / "summary_before_rerank.json").write_text(
        json.dumps(
            {
                "retriever": "Hybrid A before BGE Reranker",
                "overall": before_overall,
                "generated_at": utc_now_iso(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                **dry_report,
                "approved_only": args.approved_only,
                "allow_cpu": args.allow_cpu,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nHybrid A + Reranker 평가 완료")
    print("Reranker 적용 전:", json.dumps(
        before_overall, ensure_ascii=False, indent=2
    ))
    print("Reranker 적용 후:", json.dumps(
        overall, ensure_ascii=False, indent=2
    ))
    print("결과 폴더:", args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
