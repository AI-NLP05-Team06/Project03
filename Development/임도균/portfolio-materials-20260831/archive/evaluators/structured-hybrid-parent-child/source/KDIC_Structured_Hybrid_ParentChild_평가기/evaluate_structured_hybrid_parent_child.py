"""KDIC Structured Dense + BM25-Nori Hybrid + Parent-Child 평가기.

업무 라벨로 검색 범위를 제한하지 않는다. 제목·소제목·브레드크럼·본문을
BGE-M3로 임베딩한 Structured Dense 순위와 BM25-Nori Discard 순위를
가중 RRF로 결합한다. 최종 Child 검색 순위는 기존 검색 지표로 평가하고,
상위 Child를 같은 parent_doc_id의 인접 청크로 확장한 답변 컨텍스트는
별도의 Parent-Child 지표로 평가한다.
"""

from __future__ import annotations

import argparse
import json
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

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
from evaluate_bge_m3_dense_structured import structured_text
from evaluate_bm25_nori import BM25Index, NoriTokenizer
from evaluate_hybrid_a import weighted_rrf
from metrics import evaluate_ranking


DEFAULT_DENSE_WEIGHT = 0.85
DEFAULT_NORI_WEIGHT = 0.15
DEFAULT_RRF_CONSTANT = 10
DEFAULT_CANDIDATE_K = 20
DEFAULT_FINAL_K = 10
DEFAULT_SEED_CHILD_K = 5
DEFAULT_NEIGHBOR_WINDOW = 1
DEFAULT_MAX_PARENTS = 3
DEFAULT_MAX_CHUNKS_PER_PARENT = 3
DEFAULT_MAX_CONTEXT_CHUNKS = 10
DEFAULT_BM25_K1 = 1.5
DEFAULT_BM25_B = 0.75
DEFAULT_LUCENE_VERSION = "9.12.2"


def clean_id(value: Any) -> str:
    return str(value or "").strip()


def mean_value(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(clean_id(value) for value in values if clean_id(value)))


def recall(ids: Iterable[str], gold_ids: Iterable[str]) -> float:
    gold = set(unique(gold_ids))
    if not gold:
        return 0.0
    return len(set(unique(ids)).intersection(gold)) / len(gold)


class ParentChildExpander:
    """검색된 Child를 같은 Parent의 인접 Child로 확장한다."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.by_id: dict[str, dict[str, Any]] = {}
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for chunk in chunks:
            chunk_id = clean_id(chunk.get("chunk_id"))
            if not chunk_id:
                continue
            parent_id = (
                clean_id(chunk.get("parent_doc_id"))
                or clean_id(chunk.get("document_id"))
                or chunk_id
            )
            normalized = dict(chunk)
            normalized["chunk_id"] = chunk_id
            normalized["_parent_id"] = parent_id
            try:
                normalized["_chunk_index"] = int(chunk.get("chunk_index", 0))
            except (TypeError, ValueError):
                normalized["_chunk_index"] = 0
            self.by_id[chunk_id] = normalized
            grouped[parent_id].append(normalized)

        self.by_parent: dict[str, list[dict[str, Any]]] = {}
        self.position: dict[str, tuple[str, int]] = {}
        for parent_id, members in grouped.items():
            ordered = sorted(
                members,
                key=lambda item: (item["_chunk_index"], item["chunk_id"]),
            )
            self.by_parent[parent_id] = ordered
            for position, member in enumerate(ordered):
                self.position[member["chunk_id"]] = (parent_id, position)

    def parent_id(self, chunk_id: str) -> str:
        chunk = self.by_id.get(clean_id(chunk_id))
        return clean_id(chunk.get("_parent_id")) if chunk else ""

    def parent_ids(self, chunk_ids: Iterable[str]) -> list[str]:
        return unique(self.parent_id(chunk_id) for chunk_id in chunk_ids)

    def expand(
        self,
        ranked_child_ids: list[str],
        *,
        seed_child_k: int,
        neighbor_window: int,
        max_parents: int,
        max_chunks_per_parent: int,
        max_context_chunks: int,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        seeds = unique(ranked_child_ids)[:seed_child_k]
        selected_parents: list[str] = []
        for seed_id in seeds:
            parent_id = self.parent_id(seed_id)
            if parent_id and parent_id not in selected_parents:
                if len(selected_parents) >= max_parents:
                    continue
                selected_parents.append(parent_id)

        expanded: list[str] = []
        evidence: list[dict[str, Any]] = []
        per_parent_count: dict[str, int] = defaultdict(int)

        def add(chunk_id: str, seed_id: str, distance: int) -> None:
            if len(expanded) >= max_context_chunks or chunk_id in expanded:
                return
            parent_id = self.parent_id(chunk_id)
            if not parent_id or parent_id not in selected_parents:
                return
            if per_parent_count[parent_id] >= max_chunks_per_parent:
                return
            chunk = self.by_id[chunk_id]
            expanded.append(chunk_id)
            per_parent_count[parent_id] += 1
            evidence.append(
                {
                    "chunk_id": chunk_id,
                    "parent_doc_id": parent_id,
                    "chunk_index": chunk.get("_chunk_index"),
                    "seed_chunk_id": seed_id,
                    "distance": distance,
                    "is_seed": chunk_id == seed_id,
                }
            )

        # 먼저 검색된 seed를 최대한 보존한 뒤 남는 예산으로 인접 청크를 추가한다.
        for seed_id in seeds:
            if len(expanded) >= max_context_chunks:
                break
            position_info = self.position.get(seed_id)
            if not position_info:
                continue
            parent_id, position = position_info
            if parent_id not in selected_parents:
                continue
            add(seed_id, seed_id, 0)

        for seed_id in seeds:
            if len(expanded) >= max_context_chunks:
                break
            position_info = self.position.get(seed_id)
            if not position_info:
                continue
            parent_id, position = position_info
            if parent_id not in selected_parents:
                continue
            members = self.by_parent[parent_id]
            for distance in range(1, neighbor_window + 1):
                for candidate_position in (position - distance, position + distance):
                    if 0 <= candidate_position < len(members):
                        add(
                            members[candidate_position]["chunk_id"],
                            seed_id,
                            candidate_position - position,
                        )

        return expanded, evidence

    def context_char_count(self, chunk_ids: Iterable[str]) -> int:
        return sum(
            len(str(self.by_id[chunk_id].get("content", "")))
            for chunk_id in unique(chunk_ids)
            if chunk_id in self.by_id
        )


def parent_child_metrics(
    *,
    expander: ParentChildExpander,
    ranked_child_ids: list[str],
    expanded_ids: list[str],
    gold_ids: list[str],
    primary_gold_ids: list[str],
    supporting_gold_ids: list[str],
    seed_child_k: int,
) -> dict[str, float | int | None]:
    seed_ids = ranked_child_ids[:seed_child_k]
    gold_parents = expander.parent_ids(gold_ids)
    seed_parents = expander.parent_ids(seed_ids)
    child_seed_recall = recall(seed_ids, gold_ids)
    expanded_gold_recall = recall(expanded_ids, gold_ids)
    expanded_primary_recall = recall(expanded_ids, primary_gold_ids)
    expanded_supporting_recall = (
        recall(expanded_ids, supporting_gold_ids)
        if supporting_gold_ids
        else None
    )
    expanded_set = set(expanded_ids)
    gold_set = set(gold_ids)
    return {
        "parent_hit_at_3": float(
            bool(set(seed_parents[:3]).intersection(gold_parents))
        ),
        "parent_recall_at_3": recall(seed_parents[:3], gold_parents),
        "child_seed_recall": child_seed_recall,
        "expanded_gold_recall": expanded_gold_recall,
        "expanded_primary_recall": expanded_primary_recall,
        "expanded_supporting_recall": expanded_supporting_recall,
        "expansion_recall_gain": expanded_gold_recall - child_seed_recall,
        "expanded_gold_count": len(expanded_set.intersection(gold_set)),
        "expanded_context_chunk_count": len(expanded_ids),
        "expansion_added_chunk_count": len(set(expanded_ids) - set(seed_ids)),
        "expanded_non_gold_ratio": (
            len(expanded_set - gold_set) / len(expanded_set)
            if expanded_set
            else 0.0
        ),
    }


PARENT_METRIC_KEYS = [
    "parent_hit_at_3",
    "parent_recall_at_3",
    "child_seed_recall",
    "expanded_gold_recall",
    "expanded_primary_recall",
    "expanded_supporting_recall",
    "expansion_recall_gain",
    "expanded_gold_count",
    "expanded_context_chunk_count",
    "expansion_added_chunk_count",
    "expanded_non_gold_ratio",
    "expanded_context_char_count",
]


def extended_summary(
    rows: list[dict[str, Any]], group_name: str, group_value: str
) -> dict[str, Any]:
    summary = summarize(rows, group_name, group_value)
    for key in PARENT_METRIC_KEYS:
        summary[f"{key}_mean"] = mean_value(rows, key)
    summary["expansion_improved_question_count"] = sum(
        float(row.get("expansion_recall_gain") or 0) > 0 for row in rows
    )
    summary["expansion_worsened_question_count"] = sum(
        float(row.get("expansion_recall_gain") or 0) < 0 for row in rows
    )
    return summary


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
    parser.add_argument("--seed-child-k", type=int, default=DEFAULT_SEED_CHILD_K)
    parser.add_argument("--neighbor-window", type=int, default=DEFAULT_NEIGHBOR_WINDOW)
    parser.add_argument("--max-parents", type=int, default=DEFAULT_MAX_PARENTS)
    parser.add_argument(
        "--max-chunks-per-parent",
        type=int,
        default=DEFAULT_MAX_CHUNKS_PER_PARENT,
    )
    parser.add_argument(
        "--max-context-chunks", type=int, default=DEFAULT_MAX_CONTEXT_CHUNKS
    )
    parser.add_argument("--bm25-k1", type=float, default=DEFAULT_BM25_K1)
    parser.add_argument("--bm25-b", type=float, default=DEFAULT_BM25_B)
    parser.add_argument("--lucene-version", default=DEFAULT_LUCENE_VERSION)
    parser.add_argument("--lucene-cache-dir", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--approved-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-missing-gold", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.candidate_k < 10 or args.final_k < 10:
        raise ValueError("MRR@10과 MAP@10을 위해 candidate-k와 final-k는 10 이상이어야 합니다.")
    if args.dense_weight < 0 or args.nori_weight < 0:
        raise ValueError("검색 가중치는 0 이상이어야 합니다.")
    if abs(args.dense_weight + args.nori_weight - 1.0) > 1e-9:
        raise ValueError("Dense와 Nori 가중치 합은 1이어야 합니다.")
    for name in (
        "seed_child_k",
        "max_parents",
        "max_chunks_per_parent",
        "max_context_chunks",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')}는 1 이상이어야 합니다.")
    if args.neighbor_window < 0:
        raise ValueError("--neighbor-window는 0 이상이어야 합니다.")


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

    available_chunk_ids = {clean_id(chunk.get("chunk_id")) for chunk in chunks}
    missing_gold = validate_gold_chunks(questions, available_chunk_ids)
    expander = ParentChildExpander(chunks)
    missing_parent_count = sum(
        not clean_id(chunk.get("parent_doc_id")) for chunk in chunks
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    retriever_name = (
        "BGE-M3 Structured Dense + BM25-Nori Discard Hybrid "
        "+ Parent-Child Neighbor Expansion (no domain filter)"
    )
    run_config = {
        "retriever": retriever_name,
        "dataset": str(args.dataset.resolve()),
        "kdic_zip": str(args.kdic_zip.resolve()),
        "sheet_name": args.sheet_name,
        "question_count": len(questions),
        "chunk_count": len(chunks),
        "model": args.model,
        "structured_fields": ["title", "section_title", "heading_path", "content"],
        "domain_filter": False,
        "candidate_k_per_retriever": args.candidate_k,
        "final_child_k": args.final_k,
        "dense_weight": args.dense_weight,
        "nori_weight": args.nori_weight,
        "rrf_constant": args.rrf_constant,
        "nori_decompound_mode": "discard",
        "seed_child_k": args.seed_child_k,
        "neighbor_window": args.neighbor_window,
        "max_parents": args.max_parents,
        "max_chunks_per_parent": args.max_chunks_per_parent,
        "max_context_chunks": args.max_context_chunks,
        "bm25_k1": args.bm25_k1,
        "bm25_b": args.bm25_b,
        "lucene_version": args.lucene_version,
        "missing_parent_doc_id_count": missing_parent_count,
        "missing_gold_count": len(missing_gold),
    }
    dry_report = {
        **run_config,
        "status": "ready" if not missing_gold else "gold_validation_failed",
        "generated_at": utc_now_iso(),
    }
    (args.output_dir / "dry_run_report.json").write_text(
        json.dumps(dry_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(args.output_dir / "missing_gold.csv", missing_gold)
    print(json.dumps(dry_report, ensure_ascii=False, indent=2))
    if missing_gold and not args.allow_missing_gold:
        raise RuntimeError("Gold 청크 일부가 corpus에 없습니다. missing_gold.csv를 확인하세요.")
    if args.dry_run:
        print("Dry-run 완료: API 및 Lucene 검색은 실행하지 않았습니다.")
        return 0

    lucene_cache_dir = args.lucene_cache_dir or args.output_dir / "lucene_jars"
    tokenizer = NoriTokenizer(lucene_cache_dir, args.lucene_version, "discard")
    nori_index = BM25Index(
        chunks=chunks, tokenizer=tokenizer, k1=args.bm25_k1, b=args.bm25_b
    )
    print("Nori 확인:", tokenizer.tokenize("착오송금 반환지원 신청 기간과 필요한 서류"))

    client = EmbeddingClient(
        api_key=get_api_key(),
        base_url=args.base_url,
        model=args.model,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        cache_path=args.output_dir / "structured_embedding_cache.jsonl",
    )
    embeddings: list[dict[str, Any]] = []
    print(f"Structured Child 임베딩 생성/캐시 확인: {len(chunks)}개")
    for number, chunk in enumerate(chunks, start=1):
        vector, cache_hit = client.embed(structured_text(chunk))
        embeddings.append(
            {
                "chunk_id": clean_id(chunk.get("chunk_id")),
                "business_function": clean_id(chunk.get("business_function")),
                "model": args.model,
                "embedding": vector,
            }
        )
        if number % 25 == 0 or number == len(chunks):
            print(f"  {number}/{len(chunks)} 완료 (마지막 캐시={cache_hit})")
    dense_index = DenseIndex(chunks=chunks, embeddings=embeddings)

    details: list[dict[str, Any]] = []
    run_started = time.perf_counter()
    for number, question in enumerate(questions, start=1):
        query_started = time.perf_counter()
        query_vector, cache_hit = client.embed(question.question)
        # 업무 분류 및 업무 필터를 적용하지 않고 427개 전체 청크를 검색한다.
        dense_results = dense_index.search(
            query_vector, top_k=args.candidate_k, business_function_label=None
        )
        nori_results = nori_index.search(
            question.question, top_k=args.candidate_k, business_function_label=None
        )
        fused_results = weighted_rrf(
            dense_results,
            nori_results,
            dense_weight=args.dense_weight,
            nori_weight=args.nori_weight,
            rrf_constant=args.rrf_constant,
            final_k=args.final_k,
        )
        ranked_ids = [result["chunk_id"] for result in fused_results]
        child_metrics = evaluate_ranking(
            ranked_ids,
            gold_ids=question.all_gold,
            primary_gold_ids=question.primary_gold,
            supporting_gold_ids=question.supporting_gold,
            multi_chunk_required=question.multi_chunk_required,
        )
        expanded_ids, expansion_evidence = expander.expand(
            ranked_ids,
            seed_child_k=args.seed_child_k,
            neighbor_window=args.neighbor_window,
            max_parents=args.max_parents,
            max_chunks_per_parent=args.max_chunks_per_parent,
            max_context_chunks=args.max_context_chunks,
        )
        pc_metrics = parent_child_metrics(
            expander=expander,
            ranked_child_ids=ranked_ids,
            expanded_ids=expanded_ids,
            gold_ids=question.all_gold,
            primary_gold_ids=question.primary_gold,
            supporting_gold_ids=question.supporting_gold,
            seed_child_k=args.seed_child_k,
        )
        pc_metrics["expanded_context_char_count"] = expander.context_char_count(expanded_ids)
        latency_ms = (time.perf_counter() - query_started) * 1000

        row = {
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
            "retrieved_scores": json.dumps(
                [round(result["rrf_score"], 10) for result in fused_results]
            ),
            "dense_candidate_ids": json.dumps(
                [result["chunk_id"] for result in dense_results], ensure_ascii=False
            ),
            "nori_candidate_ids": json.dumps(
                [result["chunk_id"] for result in nori_results], ensure_ascii=False
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
            "seed_child_ids": json.dumps(ranked_ids[: args.seed_child_k], ensure_ascii=False),
            "seed_parent_ids": json.dumps(
                expander.parent_ids(ranked_ids[: args.seed_child_k]), ensure_ascii=False
            ),
            "gold_parent_ids": json.dumps(
                expander.parent_ids(question.all_gold), ensure_ascii=False
            ),
            "expanded_context_chunk_ids": json.dumps(expanded_ids, ensure_ascii=False),
            "parent_child_evidence": json.dumps(expansion_evidence, ensure_ascii=False),
            **child_metrics,
            **pc_metrics,
            "latency_ms": round(latency_ms, 3),
            "query_embedding_cache_hit": cache_hit,
        }
        details.append(row)
        print(
            f"[{number:03d}/{len(questions):03d}] {question.evaluation_id} "
            f"Hit@3={child_metrics['hit_at_3']:.0f} "
            f"ChildRecall={pc_metrics['child_seed_recall']:.3f} "
            f"ExpandedRecall={pc_metrics['expanded_gold_recall']:.3f}"
        )

    write_csv(args.output_dir / "question_results.csv", details)
    overall = extended_summary(details, "overall", "all")
    by_domain = [
        extended_summary(
            [row for row in details if row["gold_business_function"] == domain],
            "gold_business_function",
            domain,
        )
        for domain in sorted({row["gold_business_function"] for row in details})
    ]
    write_csv(args.output_dir / "summary_by_domain.csv", by_domain)

    summary = {
        **run_config,
        "metric_note": {
            "standard_metrics": "확장 전 Child RRF 순위를 평가",
            "parent_child_metrics": "상위 Child를 Parent 내 인접 청크로 확장한 답변 컨텍스트를 평가",
            "expanded_non_gold_ratio": "Gold 미표기 청크 비율이며 곧바로 무관 청크라는 뜻은 아님",
        },
        "overall": overall,
        "by_domain": by_domain,
        "total_runtime_seconds": round(time.perf_counter() - run_started, 3),
        "generated_at": utc_now_iso(),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n평가 완료")
    print(json.dumps(overall, ensure_ascii=False, indent=2))
    print("결과 폴더:", args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
