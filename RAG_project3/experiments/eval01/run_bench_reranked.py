# [Experiment/eval01 후속] run_bench.py는 재정렬(Cross-Encoder) 없이 사전(pre-rerank)
# dense+bm25 단계만으로 질의분석 O/X를 비교했는데, 실제 프로덕션(compound_answer.py)은
# 항상 retrieval/reranker.py::hybrid_rerank_search(_hedged)로 재정렬까지 거칩니다.
# 스모크테스트에서 재정렬이 pre-rerank 단계의 손해를 스스로 상당히 복구하는 걸
# 확인했기 때문에, 여기서는 재정렬까지 포함한 "실제 프로덕션 함수 그대로"
# raw(질의분석 X) / qa(질의분석 O, hedged 없음) / hedged(질의분석 O, 원문 안전망 포함)
# 세 갈래를 다시 비교합니다.
#
# run_bench.py를 import하지 않습니다 — run_bench.py의 다른 최상단 import들
# (dense_structured_search.py, bm25_nori_search.py)이 이 실험엔 필요없는
# pynori 기반 BM25 인덱스를 새로 빌드해버려서(수 분~수십 분) 낭비이기 때문에,
# 필요한 골드셋 로딩 로직만 이 파일에 다시 작성했습니다(load_gold_set_v2와
# 동일 로직).
#
# 채점 방식: 프로덕션은 하위질문마다 독립적으로 검색+답변하고 결과를 병합하지
# 않으므로(compound_answer.py의 _answer_single이 하위질문마다 따로 검색), "하나로
# 합친 top5"가 아니라 "하위질문 각자의 top-K 안에서 각 gold 청크가 도달한 가장
# 좋은 순위"로 합성 랭킹을 만들어 채점합니다(점수가 아니라 순위 기반 병합이라
# eval01에서 확인된 "서로 다른 정규화 기준끼리 점수 비교" 문제가 생기지 않음).
# 단일질의(하위질문 1개)면 이 병합이 사실상 그 결과 그대로입니다.
#
# 5가지 검색기법 비교는 여기선 안 합니다 — 프로덕션은 확정된 Hybrid(0.7:0.3,
# MinMax weighted-sum)+Cross-Encoder 재정렬 조합 하나만 쓰므로 그대로 씁니다.
from __future__ import annotations

import json
import math
import pickle
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "production"))

import pandas as pd

from classification.context_rewrite import rewrite_with_context
from classification.decomposition import decompose_query
from classification.pipeline import classify
from experiments.evaluation.eval_search import (
    hit_at_k,
    recall_at_k,
    precision_at_k,
    mrr_at_k,
    average_precision_at_k,
    f1_at_k,
)
from retrieval.reranker import hybrid_rerank_search, hybrid_rerank_search_hedged

DATASETS = [
    {"label": "challenge_v3", "path": Path("data/Evaluation_Dataset03.xlsx"), "sheet": "도전평가셋"},
    {"label": "routing_v5", "path": Path("data/Evaluation_Dataset04.xlsx"), "sheet": "데이터셋_v5"},
]

FINAL_K = 10  # 합성 랭킹 길이(hit@3/recall@5엔 top5까지만 쓰지만 mrr@10/map@10 계산엔 10까지 필요)

RESULTS_DIR = Path(__file__).resolve().parent / "results_reranked"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = RESULTS_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_MULTITURN_PATTERN = re.compile(r"^\[이전 대화\]\s*(.+?)\s*\[현재 질문\]\s*(.+)$", re.DOTALL)


def _resolve_multiturn(question: str) -> str:
    m = _MULTITURN_PATTERN.match(question.strip())
    if not m:
        return question
    previous_question, current_question = m.group(1), m.group(2)
    return rewrite_with_context(current_question, previous_question)


def _parse_id_list(value) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    value = str(value).strip()
    if not value:
        return []
    parsed = json.loads(value)
    flattened: list[str] = []
    for item in parsed:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)
    return flattened


def load_gold_set_v2(path: Path, sheet_name: str, *, label: str) -> list[dict]:
    df = pd.read_excel(path, sheet_name=sheet_name)
    records: list[dict] = []
    for _, row in df.iterrows():
        if str(row.get("route_type")) != "RETRIEVE":
            continue

        primary_ids = _parse_id_list(row.get("gold_primary_chunk_ids"))
        supporting_ids = _parse_id_list(row.get("gold_supporting_chunk_ids"))

        raw_gold = row.get("gold_chunk_ids") if "gold_chunk_ids" in row.index else None
        has_gold_col = not (raw_gold is None or (isinstance(raw_gold, float) and math.isnan(raw_gold)))
        if has_gold_col:
            gold_ids = _parse_id_list(raw_gold)
        else:
            seen: set[str] = set()
            gold_ids = []
            for cid in primary_ids + supporting_ids:
                if cid not in seen:
                    seen.add(cid)
                    gold_ids.append(cid)

        if not gold_ids:
            continue

        records.append({
            "evaluation_id": row["evaluation_id"],
            "question": row["question"],
            "dataset": label,
            "gold_chunk_ids": gold_ids,
        })

    return records


def _rank_merge(per_subq_ranked: list[list[dict]], final_k: int) -> list[str]:
    """점수가 아니라 '각 리스트 안에서의 순위'로 병합합니다(서로 다른 하위질문의
    정규화 기준이 달라 점수를 직접 비교하면 안 된다는 게 eval01에서 확인된
    문제라, 순위는 리스트 내부적으로만 의미가 있으므로 최선의(가장 작은) 순위를
    그대로 비교합니다)."""
    best_rank: dict[str, int] = {}
    for ranked in per_subq_ranked:
        for rank, item in enumerate(ranked, start=1):
            cid = item["chunk"]["chunk_id"]
            if cid not in best_rank or rank < best_rank[cid]:
                best_rank[cid] = rank
    ordered = sorted(best_rank.items(), key=lambda kv: kv[1])
    return [cid for cid, _ in ordered[:final_k]]


def _score(ranked_ids: list[str], gold_ids: list[str], suffix: str) -> dict:
    precision5 = precision_at_k(ranked_ids, gold_ids, 5)
    recall5 = recall_at_k(ranked_ids, gold_ids, 5)
    return {
        f"hit@3_{suffix}": hit_at_k(ranked_ids, gold_ids, 3),
        f"recall@5_{suffix}": recall5,
        f"precision@5_{suffix}": precision5,
        f"f1@5_{suffix}": f1_at_k(precision5, recall5),
        f"mrr@10_{suffix}": mrr_at_k(ranked_ids, gold_ids, 10),
        f"map@10_{suffix}": average_precision_at_k(ranked_ids, gold_ids, 10),
    }


def _process_record(record: dict) -> dict:
    question = record["question"]

    # raw: 질의분석 없이 원문 그대로(대괄호 포함) 검색 — run_bench.py와 동일한 비교기준
    t0 = time.perf_counter()
    raw_ranked = hybrid_rerank_search(question, top_k=FINAL_K, business_function=None)
    raw_latency = time.perf_counter() - t0
    raw_ids = [it["chunk"]["chunk_id"] for it in raw_ranked]

    # 질의분석 경로: 후속질문 형식이면 맥락 먼저 해소(프로덕션 순서와 동일)
    t0 = time.perf_counter()
    resolved = _resolve_multiturn(question)
    subqs = decompose_query(resolved)
    analysis_latency = time.perf_counter() - t0

    qa_ranked_lists: list[list[dict]] = []
    hedged_ranked_lists: list[list[dict]] = []
    search_latency_qa = 0.0
    search_latency_hedged = 0.0

    for subq in subqs:
        t0 = time.perf_counter()
        route_result = classify(subq)
        analysis_latency += time.perf_counter() - t0
        if route_result["route"] != "RETRIEVE":
            continue  # 프로덕션에서도 이 하위질문은 검색을 안 함(고정 안내문구로 대체)

        t0 = time.perf_counter()
        qa_one = hybrid_rerank_search(subq, top_k=FINAL_K, business_function=None)
        search_latency_qa += time.perf_counter() - t0
        qa_ranked_lists.append(qa_one)

        t0 = time.perf_counter()
        if subq != resolved:
            hedged_one = hybrid_rerank_search_hedged(subq, resolved, top_k=FINAL_K, business_function=None)
        else:
            hedged_one = qa_one  # 프로덕션과 동일: 단일질의는 hedged 스킵
        search_latency_hedged += time.perf_counter() - t0
        hedged_ranked_lists.append(hedged_one)

    qa_ids = _rank_merge(qa_ranked_lists, FINAL_K) if qa_ranked_lists else []
    hedged_ids = _rank_merge(hedged_ranked_lists, FINAL_K) if hedged_ranked_lists else []

    gold_ids = record["gold_chunk_ids"]
    row = {
        "evaluation_id": record["evaluation_id"],
        "dataset": record["dataset"],
        "n_sub_questions": len(subqs),
        "n_retrieve_subqs": len(qa_ranked_lists),
        "raw_latency_sec": raw_latency,
        "analysis_latency_sec": analysis_latency,
        "search_latency_sec_qa": search_latency_qa,
        "search_latency_sec_hedged": search_latency_hedged,
        "total_latency_sec_raw": raw_latency,
        "total_latency_sec_qa": analysis_latency + search_latency_qa,
        "total_latency_sec_hedged": analysis_latency + search_latency_hedged,
    }
    row.update(_score(raw_ids, gold_ids, "raw"))
    row.update(_score(qa_ids, gold_ids, "qa"))
    row.update(_score(hedged_ids, gold_ids, "hedged"))
    return row


def evaluate(*, limit: int | None = None) -> pd.DataFrame:
    all_rows = []

    for ds in DATASETS:
        label = ds["label"]
        print(f"\n{'=' * 70}\n데이터셋: {label} ({ds['path'].name} / {ds['sheet']})\n{'=' * 70}")

        gold_records = load_gold_set_v2(ds["path"], ds["sheet"], label=label)
        records = gold_records[:limit] if limit else gold_records
        print(f"평가 문항 수(route_type=RETRIEVE): {len(records)}")

        cache_path = CACHE_DIR / f"detail_{label}.pkl"
        cache: dict[str, dict] = {}
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            print(f"  이어서 진행: 기존 {len(cache)}건 로드됨")

        todo = [r for r in records if r["evaluation_id"] not in cache]
        failed: list[str] = []
        for i, record in enumerate(todo, 1):
            # 일시적 네트워크/API 오류(APIConnectionError 등) 하나 때문에 254문항
            # 전체가 죽지 않도록, 문항 하나당 최대 3번 재시도하고 그래도 안 되면
            # 이 문항만 건너뜁니다(캐시에 안 남기므로 다음 실행 시 자동으로 재시도됨).
            for attempt in range(3):
                try:
                    cache[record["evaluation_id"]] = _process_record(record)
                    break
                except Exception as exc:
                    print(f"  [경고] {record['evaluation_id']} 처리 실패(시도 {attempt + 1}/3): {exc}")
                    if attempt < 2:
                        time.sleep(5)
            else:
                failed.append(record["evaluation_id"])
                continue

            with open(cache_path, "wb") as f:
                pickle.dump(cache, f)
            if i % 10 == 0:
                print(f"  진행: {i}/{len(todo)}(신규) [총 {len(cache)}/{len(records)}]")

        if failed:
            print(f"  [경고] 3번 재시도 후에도 실패한 문항 {len(failed)}건(다음 실행 시 자동 재시도): {failed}")

        print(f"  완료: {len(records)}문항 (신규 {len(todo)}건 처리)")

        rows = [cache[r["evaluation_id"]] for r in records if r["evaluation_id"] in cache]
        detail_df = pd.DataFrame(rows)
        detail_path = RESULTS_DIR / f"detail_{label}.csv"
        detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")

        summary = {"dataset": label, "n_questions": len(detail_df)}
        metric_cols = [
            "raw_latency_sec", "analysis_latency_sec",
            "search_latency_sec_qa", "search_latency_sec_hedged",
            "total_latency_sec_raw", "total_latency_sec_qa", "total_latency_sec_hedged",
        ] + [f"{m}_{suffix}" for suffix in ("raw", "qa", "hedged") for m in
             ("hit@3", "recall@5", "precision@5", "f1@5", "mrr@10", "map@10")]
        for metric in metric_cols:
            summary[metric] = round(float(detail_df[metric].mean()), 4)

        print(f"\n=== [{label}] 요약 ===")
        print(summary)
        all_rows.append(summary)

    summary_df = pd.DataFrame(all_rows)
    summary_path = RESULTS_DIR / "summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print("\n요약 저장:", summary_path)
    print(summary_df.to_string(index=False))
    return summary_df


if __name__ == "__main__":
    evaluate()
