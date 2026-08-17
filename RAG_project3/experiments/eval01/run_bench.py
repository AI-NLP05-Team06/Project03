# [Experiment/eval01] Evaluation_Dataset03.xlsx("도전평가셋", 질의분석 난이도를 겨냥해
# 새로 만든 콜로퀴얼/복합/대명사/후속질문 등 챌린지셋)과 Evaluation_Dataset04.xlsx
# ("데이터셋_v5", 일반 라우팅 골드셋) 두 개를 각각 따로 돌려서 "질의분석(decompose_query
# +classify)을 거친 뒤 검색"까지 파이프라인 그대로 태워서 5가지 검색기법을 비교합니다.
# 두 데이터셋을 따로 채점하는 이유: 03은 질의분석이 실제로 필요한 어려운 케이스만
# 모아둔 셋이라 "질의분석이 어려운 질문에서 정말 이득을 주는지"를 보여주고, 04는 넓은
# 범위의 일반 질문 셋이라 "쉬운 질문에서 질의분석이 오히려 오버헤드만 되는지"를 같이
# 보여줍니다. 합쳐서 채점하면 이 대비가 평균에 묻혀 안 보입니다.
#
# 두 파일 다 route_type 컬럼이 있고 RETRIEVE인 행만 검색 정답(gold_chunk_ids)이 의미
# 있어서 그 행만 사용합니다(CLARIFY/OUT_OF_SCOPE/DIRECT_RESPONSE는 제외). Dataset03은
# gold_chunk_ids 컬럼이 그대로 있지만, Dataset04는 그 컬럼이 없어서
# gold_primary_chunk_ids ∪ gold_supporting_chunk_ids로 파생합니다(load_gold_set_v2).
#   1. dense v2 structured 단독 (title+content 임베딩)
#   2. hybrid 7:3 RRF   (dense v2 structured + BM25-Nori, rrf k=10)
#   3. hybrid 7:3 MinMax
#   4. hybrid 9:1 RRF
#   5. hybrid 9:1 MinMax
# (dense/bm25 후보 풀 depth=20)
#
# 기존 코드는 전혀 수정하지 않고, 필요한 순수 함수만 이 experiment/ 폴더 안으로
# 옮겨와서 조립했습니다 (dense_structured_search.py, bm25_nori_search.py, fusion.py).
#
# 하위질문 병합 규칙: decompose_query()가 하위질문을 여러 개로 쪼갠 경우(전체 문항의
# 소수), 하위질문별로 검색한 뒤 같은 청크는 최고 점수로 병합하고 점수 내림차순으로
# 최종 랭킹을 만듭니다. 단일질의(하위질문 1개, 대다수)는 사실상 그 결과 그대로입니다.
#
# latency는 세 갈래로 나눠서 잽니다(실제 프로덕션 흐름 generation/compound_answer.py의
# 순서 그대로: decompose_query(질문 1회) -> 하위질문마다 classify(라우팅+업무+의도)
# -> 검색):
#   - analysis_latency_sec = decompose_query 1회(방법 5개가 공유하는 단계라 한 번만
#     측정해 그대로 더함) + 하위질문마다 classify() 호출 시간의 합
#   - search_latency_sec   = 하위질문마다 dense/bm25 후보 풀 조회 시간의 합
#     (융합(RRF/MinMax) 자체는 이미 가져온 후보 리스트에 대한 순수 연산이라
#     무시할 수준(<1ms)이라 별도로 재지 않습니다)
#   - total_latency_sec    = 위 둘의 합
#
# hit@3/recall@5(그 외 precision@5·f1@5·mrr@10·map@10도 동일)는 "질의분석을 거친
# 경우"(_qa 접미사)와 "원문 질문을 그대로 검색한 경우"(_raw 접미사, decompose/classify
# 없이 record["question"]을 바로 검색) 두 갈래로 같이 뽑습니다. 질의분석이 정확도에
# 실제로 얼마나 기여하는지(혹은 하위질문 병합 때문에 오히려 떨어뜨리는지) 보기 위함.
# search_latency_sec_raw/total_latency_sec_raw도 같이 남겨서 "질의분석 생략 시
# 얼마나 더 빠른가"도 바로 비교할 수 있게 합니다(total_latency_sec_raw는 analysis
# 단계가 없으므로 search_latency_sec_raw와 같습니다).
#
# 세 번째 갈래(_hedged 접미사): 하위질문별 병합(_qa)이 "병합손해"(각 하위질문이
# 자기 후보군 안에서만 Min-Max 정규화되다 보니 후보군이 약한 하위질문의 1등이
# 실제로 더 관련있는 다른 하위질문의 후보를 밀어내는 문제, eval01에서 실측 진단됨)
# 로 raw보다 못한 경우가 있어서, 하위질문 결과 + 원문(raw) 검색결과를 같은 풀에
# 넣고 병합합니다. 원문 결과가 항상 안전망으로 포함되므로 이론상 raw보다 나빠질
# 수 없습니다. search_latency_sec_hedged = search_latency_sec + search_latency_sec_raw
# (원문 검색 1회를 추가로 더 하는 비용).
#
# 캐시(decompose/classify/dense/bm25 호출 결과)는 데이터셋별로 문항 처리할 때마다
# results/cache/subq_cache_{label}.pkl에 즉시 저장하고, 실행 중 끊기면(Colab 세션
# 끊김, API rate limit 등) 다음 실행 때 이미 처리된 문항은 건너뛰고 이어서 진행합니다.
from __future__ import annotations

import json
import math
import pickle
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "production"))

import pandas as pd

from classification.context_rewrite import rewrite_with_context
from classification.decomposition import decompose_query
from classification.pipeline import classify
from experiments.evaluation.eval_search import (
    _parse_id_list,
    hit_at_k,
    recall_at_k,
    precision_at_k,
    mrr_at_k,
    average_precision_at_k,
    f1_at_k,
)

from experiments.eval01.dense_structured_search import dense_structured_search
from experiments.eval01.bm25_nori_search import bm25_nori_search
from experiments.eval01.fusion import weighted_sum_fuse_minmax

DEPTH = 50    # dense/bm25 후보 풀 크기. dense_structured_search/bm25_nori_search
# 둘 다 top_k와 무관하게 항상 코퍼스 전체(427개)를 브루트포스로 계산한 뒤 마지막에
# 상위 N개만 자르는 구조라, DEPTH를 키워도 계산량은 그대로라 latency 영향이 거의
# 없다(직접 실측 확인, 병합 단계 후보 수 증가에 따른 오버헤드는 무시 가능한 수준).
# 프로덕션의 재정렬 전 후보 풀 크기(retrieval/reranker.py::HYBRID_POOL_FOR_RERANK)와
# 맞춰서 20->50으로 올림 — CHALLENGE_052처럼 원문 검색조차 20위 안에 정답이 없던
# 케이스를 hedged가 구제할 여지를 넓히기 위함.
FINAL_K = 10  # 병합 후 최종 랭킹 길이(hit@3/recall@5 계산에 충분한 여유)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = RESULTS_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 검색기법 확정됨(사용자 결정, 2026-08-18): hybrid_minmax_7_3. dense_v2_structured/
# hybrid_rrf_7_3/hybrid_rrf_9_1/hybrid_minmax_9_1과의 비교는 더 이상 필요 없음 —
# eval01 실험 목적이 "검색기법 선정"에서 "질의분석 O/X/hedged 비교"로 좁혀짐.
METHODS = [
    ("hybrid_minmax_7_3", True, lambda d, b: weighted_sum_fuse_minmax(d, b, weights=(0.7, 0.3))),
]


def load_gold_set_v2(path: Path, sheet_name: str, *, label: str) -> list[dict]:
    """Evaluation_Dataset03/04.xlsx 전용 로더. evaluation/eval_search.py의 load_gold_set과
    스키마가 달라서(gold_business_function/complexity 컬럼 없음, multi_chunk_required가
    "Y"/"N" 문자열이 아니라 bool/float, Dataset04는 gold_chunk_ids 컬럼 자체가 없음)
    별도로 구현합니다. route_type이 RETRIEVE가 아닌 행(CLARIFY/OUT_OF_SCOPE/
    DIRECT_RESPONSE)은 검색 정답이 없는 게 정상이라 제외합니다."""
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
            # Dataset04는 gold_chunk_ids 컬럼이 없어서 primary+supporting 합집합으로 파생
            seen: set[str] = set()
            gold_ids = []
            for cid in primary_ids + supporting_ids:
                if cid not in seen:
                    seen.add(cid)
                    gold_ids.append(cid)

        if not gold_ids:
            continue

        mcr = row.get("multi_chunk_required")
        multi_chunk_required = bool(mcr) if not (isinstance(mcr, float) and math.isnan(mcr)) else False

        records.append({
            "evaluation_id": row["evaluation_id"],
            "question": row["question"],
            "dataset": label,
            "primary_chunk_ids": primary_ids,
            "supporting_chunk_ids": supporting_ids,
            "gold_chunk_ids": gold_ids,
            "multi_chunk_required": multi_chunk_required,
        })

    return records


def load_gold_set_challenge_v5(path: Path, sheet_name: str, *, label: str) -> list[dict]:
    """Evaluation_Dataset05.xlsx(시트 V6_Review) 전용 로더. 이 시트는 두 origin이
    섞여 있음: LEGACY_V5(270행) = Dataset04(routing_v5)와 동일한 기존 180 RETRIEVE
    문항 + CLARIFY/OUT_OF_SCOPE/DIRECT_RESPONSE, V6_DERIVED_CHALLENGE(130행, 내부
    라벨링 스킴 이름이 V6일 뿐 파일은 Dataset05) = Revision_Log에서 신규로 값이
    채워진 챌린지 문항(CONTEXT_RESOLVABLE/QUERY_ROBUSTNESS/MULTI_POLICY_CHALLENGE/
    CROSS_BUSINESS_MULTI). LEGACY_V5는 routing_v5로 이미 테스트한 것과 같은
    문항이라 여기서는 제외하고 V6_DERIVED_CHALLENGE만 사용합니다(중복 API
    호출/비용 절감).

    load_gold_set_v2와 다른 점: route 컬럼명이 route_type이 아니라 gold_route_v6(시트
    내부 컬럼명, 이 데이터셋 자체는 Dataset05/challenge_v5로 부름)이고
    RETRIEVE가 SIMPLE_RETRIEVE/MULTI_RETRIEVE 둘로 세분화됨. previous_turns가
    Dataset03의 "[이전 대화]...[현재 질문]..." 대괄호 형식이 아니라 별도 JSON
    컬럼(`[{"user":...,"assistant":...}]`)이라, 있으면 마지막 turn의 user 발화를
    꺼내 같은 대괄호 형식으로 합성해서 기존 _resolve_multiturn이 그대로 인식하게
    만듭니다(코드 재사용, decompose_query/_resolve_multiturn 쪽은 수정 안 함).

    주의: 이 130행은 route_label_status="AI_REVISED_REQUIRES_HUMAN_APPROVAL"로,
    gold 라벨이 아직 사람 검수를 거치지 않은 상태입니다 — 결과 해석 시 감안할 것."""
    df = pd.read_excel(path, sheet_name=sheet_name)
    df = df[df["origin"] == "V6_DERIVED_CHALLENGE"]

    records: list[dict] = []
    for _, row in df.iterrows():
        if row.get("gold_route_v6") not in ("SIMPLE_RETRIEVE", "MULTI_RETRIEVE"):
            continue

        gold_ids = _parse_id_list(row.get("gold_chunk_ids"))
        if not gold_ids:
            continue

        question = str(row["question"])
        turns = json.loads(row.get("previous_turns") or "[]")
        if turns:
            prev_question = turns[-1]["user"]
            question = f"[이전 대화] {prev_question} [현재 질문] {question}"

        gold_need_count = row.get("gold_need_count")
        multi_chunk_required = bool(gold_need_count and gold_need_count > 1)

        records.append({
            "evaluation_id": row["evaluation_id"],
            "question": question,
            "dataset": label,
            "primary_chunk_ids": _parse_id_list(row.get("gold_primary_chunk_ids")),
            "supporting_chunk_ids": _parse_id_list(row.get("gold_supporting_chunk_ids")),
            "gold_chunk_ids": gold_ids,
            "multi_chunk_required": multi_chunk_required,
        })

    return records


DATASETS = [
    {"label": "challenge_v3", "path": Path("data/Evaluation_Dataset03.xlsx"), "sheet": "도전평가셋", "loader": load_gold_set_v2},
    {"label": "routing_v5", "path": Path("data/Evaluation_Dataset04.xlsx"), "sheet": "데이터셋_v5", "loader": load_gold_set_v2},
    {"label": "challenge_v5", "path": Path("data/Evaluation_Dataset05.xlsx"), "sheet": "V6_Review", "loader": load_gold_set_challenge_v5},
]


# Dataset03(도전평가셋)의 FOLLOW_UP류 문항은 "[이전 대화] {이전 턴} [현재 질문]
# {후속 질문}" 형식으로 한 필드에 합쳐져 있습니다. 프로덕션(generation/compound_answer.py
# ::answer_query)에서는 이전 턴이 있으면 decompose_query 전에 반드시
# rewrite_with_context를 먼저 태워서 완전한 단일 질문으로 만드는데, decompose_query
# 자체는 이 대괄호 형식을 처리하도록 설계된 적이 없습니다(실측 결과 "[이전 대화]"
# 부분을 통째로 버리고 "[현재 질문]"만 남기는 사고가 확인됨). 그래서 질의분석 O
# 경로로 넘기기 전에 이 형식을 감지해서 먼저 맥락반영 재작성을 거칩니다 — 이게
# 프로덕션 순서와 맞는 처리이고, 질의분석 X(raw) 경로는 원문(대괄호 포함) 그대로
# 검색해서 비교 기준을 그대로 유지합니다.
_MULTITURN_PATTERN = re.compile(r"^\[이전 대화\]\s*(.+?)\s*\[현재 질문\]\s*(.+)$", re.DOTALL)


def _resolve_multiturn(question: str) -> tuple[str, float]:
    m = _MULTITURN_PATTERN.match(question.strip())
    if not m:
        return question, 0.0
    previous_question, current_question = m.group(1), m.group(2)
    t0 = time.perf_counter()
    resolved = rewrite_with_context(current_question, previous_question)
    return resolved, time.perf_counter() - t0


def _fetch_pools(question: str) -> dict:
    t0 = time.perf_counter()
    dense_pool = dense_structured_search(question, top_k=DEPTH, business_function=None)
    dense_latency = time.perf_counter() - t0

    t0 = time.perf_counter()
    bm25_pool = bm25_nori_search(question, top_k=DEPTH, business_function=None)
    bm25_latency = time.perf_counter() - t0

    return {
        "dense_pool": dense_pool,
        "bm25_pool": bm25_pool,
        "dense_latency": dense_latency,
        "bm25_latency": bm25_latency,
    }


def build_subq_cache(
    gold_records: list[dict],
    *,
    limit: int | None = None,
    cache_path: Path | None = None,
) -> dict:
    """문항마다 decompose_query 1회 + 하위질문마다 classify(라우팅+업무+의도) 1회 +
    dense/bm25 후보 풀을 한 번만 가져와서 캐싱합니다(질의분석 O 경로, "_qa").
    여기에 더해 원문 질문 그대로(질의분석 없이) 검색한 dense/bm25 후보 풀도 한 번만
    가져와서 같이 캐싱합니다(질의분석 X 경로, "_raw") — 두 경로 모두 5가지 방법이
    이 캐시를 그대로 재사용하므로 API 호출이 문항 수 x 방법 수가 아니라 문항 수 x
    (하위질문 수 + 1) 만큼만 발생합니다.

    cache_path를 주면 문항 하나 처리할 때마다 즉시 pickle로 저장하고, 기존 파일이
    있으면 이미 처리된 evaluation_id는 건너뜁니다(세션 끊김/rate limit 대비 resume)."""
    records = gold_records[:limit] if limit else gold_records

    cache: dict[str, dict] = {}
    if cache_path and cache_path.exists():
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        print(f"  캐시 이어서 진행: 기존 {len(cache)}건 로드됨 ({cache_path.name})")

    def _build_one(record: dict) -> dict:
        question = record["question"]

        # 프로덕션 순서와 맞춤: "[이전 대화]...[현재 질문]..." 형식이면 decompose_query
        # 전에 맥락반영 재작성부터 거친다(질의분석 O 경로에만 적용, raw는 원문 그대로).
        resolved_question, rewrite_latency = _resolve_multiturn(question)

        t0 = time.perf_counter()
        subqs = decompose_query(resolved_question)
        decompose_latency = rewrite_latency + (time.perf_counter() - t0)

        def _process_subq(subq: str) -> tuple[str, dict]:
            t0 = time.perf_counter()
            classify_result = classify(subq)
            classify_latency = time.perf_counter() - t0
            return subq, {
                "route": classify_result["route"],
                "business_function": classify_result.get("business_function"),
                "intent": classify_result.get("intent"),
                "classify_latency": classify_latency,
                **_fetch_pools(subq),
            }

        # 비복합질의(대다수)면서 맥락반영 재작성도 없었던 경우(resolved_question ==
        # question) subqs == [question]이 돼서, 아래서 question에 대해 이미 계산해둔
        # per_subq 결과를 raw로 재사용한다(중복 API 호출 방지). 복합질의거나 맥락반영
        # 재작성이 있었던 경우(subqs != [question])는 raw가 실제로 다른 검색이라
        # 새로 계산해야 한다.
        raw_is_duplicate = subqs == [question]
        n_tasks = len(subqs) + (0 if raw_is_duplicate else 1)

        # 하위질문(들) + 필요하면 raw까지 독립적인 작업들을 병렬로 돌린다. 이 실험은
        # (프로덕션의 answer_query와 달리) Cross-Encoder 재정렬이 없고 순수 API
        # 호출(네트워크 I/O)만 있어서, 세션7에서 확인된 "CPU에서 병렬화하면 오히려
        # 손해"(재정렬이 CPU 연산이라 GIL 경합) 결론이 여기엔 적용되지 않는다고 판단해
        # 시도함 — I/O 대기 중엔 GIL이 풀리므로 실제 동시성 이득을 기대할 수 있음.
        # 작업이 1개뿐이면(대다수인 완전 단일질의) 스레드 오버헤드 없이 그대로 순차 실행.
        t0 = time.perf_counter()
        if n_tasks > 1:
            with ThreadPoolExecutor(max_workers=n_tasks) as executor:
                subq_futures = [executor.submit(_process_subq, subq) for subq in subqs]
                raw_future = None if raw_is_duplicate else executor.submit(_fetch_pools, question)
                per_subq = dict(f.result() for f in subq_futures)
                raw = per_subq[question] if raw_is_duplicate else raw_future.result()
        else:
            per_subq = dict(_process_subq(subq) for subq in subqs)
            raw = per_subq[question]
        subq_wall_latency = time.perf_counter() - t0

        return {
            "subqs": subqs,
            "decompose_latency": decompose_latency,
            "per_subq": per_subq,
            "raw": raw,
            "raw_is_duplicate": raw_is_duplicate,
            # 병렬 실행 덕에 sum(개별 classify/dense/bm25 latency)보다 짧아진 실제
            # 체감 latency. run_method에서 total_latency_sec_actual 계산에 사용.
            "subq_wall_latency": subq_wall_latency,
        }

    todo = [r for r in records if r["evaluation_id"] not in cache]
    failed: list[str] = []
    for i, record in enumerate(todo, 1):
        eid = record["evaluation_id"]

        # 429 rate limit 등 일시적 API 오류 하나 때문에 전체 실행이 죽지 않도록,
        # 문항 하나당 최대 5번 재시도하고 그래도 안 되면 이 문항만 건너뜁니다
        # (캐시에 안 남기므로 다음 실행 시 자동으로 재시도됨). 고정 5초 간격은
        # rate limit 윈도우(분 단위로 추정)를 못 뚫는 경우가 실측으로 확인돼서,
        # 지수 백오프(10/20/40/60초)로 늘리고 API가 Retry-After 헤더를 주면
        # 그 값을 우선 사용하도록 함.
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                cache[eid] = _build_one(record)
                break
            except Exception as exc:
                wait = min(60, 10 * (2 ** attempt))
                retry_after = getattr(getattr(exc, "response", None), "headers", {}).get("retry-after")
                if retry_after:
                    try:
                        wait = max(wait, float(retry_after))
                    except ValueError:
                        pass
                print(f"  [경고] {eid} 처리 실패(시도 {attempt + 1}/{max_attempts}): {exc}")
                if attempt < max_attempts - 1:
                    print(f"    {wait:.0f}초 대기 후 재시도")
                    time.sleep(wait)
        else:
            failed.append(eid)
            continue

        if cache_path:
            with open(cache_path, "wb") as f:
                pickle.dump(cache, f)

        if i % 20 == 0:
            print(f"  캐시 구축: {i}/{len(todo)}(신규) [총 {len(cache)}/{len(records)}]")

    print(f"캐시 구축 완료: {len(records)}문항 (신규 {len(todo) - len(failed)}건 처리)")
    if failed:
        print(f"  [경고] 3번 재시도 후에도 실패한 문항 {len(failed)}건(다음 실행 시 자동 재시도): {failed}")
    return cache


def _merge_subquery_results(per_subq_ranked: list[list[dict]], final_k: int) -> list[str]:
    best_score: dict[str, float] = {}
    for ranked in per_subq_ranked:
        for item in ranked:
            cid = item["chunk"]["chunk_id"]
            if cid not in best_score or item["score"] > best_score[cid]:
                best_score[cid] = item["score"]
    ordered = sorted(best_score.items(), key=lambda kv: kv[1], reverse=True)
    return [cid for cid, _ in ordered[:final_k]]


def _score_ranked(ranked_ids: list[str], gold_ids: list[str], suffix: str) -> dict:
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


def run_method(
    name: str,
    uses_bm25: bool,
    fuse_fn,
    gold_records: list[dict],
    cache: dict,
) -> tuple[dict, pd.DataFrame]:
    rows = []
    for record in gold_records:
        eid = record["evaluation_id"]
        entry = cache[eid]
        gold_ids = record["gold_chunk_ids"]

        # 질의분석 O: 하위질문마다 검색 후 병합
        per_subq_ranked = []
        analysis_latency = entry["decompose_latency"]
        search_latency = 0.0
        for subq in entry["subqs"]:
            pools = entry["per_subq"][subq]
            analysis_latency += pools["classify_latency"]
            search_latency += pools["dense_latency"]
            if uses_bm25:
                search_latency += pools["bm25_latency"]
                ranked = fuse_fn(pools["dense_pool"], pools["bm25_pool"])
            else:
                ranked = pools["dense_pool"]
            per_subq_ranked.append(ranked)
        ranked_ids_qa = _merge_subquery_results(per_subq_ranked, FINAL_K)

        # 질의분석 X: 원문 질문 그대로 검색(decompose/classify 없음)
        raw_pools = entry["raw"]
        search_latency_raw = raw_pools["dense_latency"]
        if uses_bm25:
            search_latency_raw += raw_pools["bm25_latency"]
            ranked_raw = fuse_fn(raw_pools["dense_pool"], raw_pools["bm25_pool"])
        else:
            ranked_raw = raw_pools["dense_pool"]
        ranked_ids_raw = [item["chunk"]["chunk_id"] for item in ranked_raw[:FINAL_K]]

        # 질의분석 O + 안전망: 하위질문 결과에 원문(raw) 검색결과까지 같이 넣고
        # 병합한다. 하위질문별 병합(_merge_subquery_results)은 각 하위질문이
        # "자기 후보군 안에서만" Min-Max 정규화되기 때문에, 후보군이 약한
        # 하위질문의 1등이 실제로는 더 관련있는 다른 하위질문의 후보를 밀어내는
        # 문제가 있었다(eval01 진단, "병합손해" C 이슈). 원문 검색결과를 같은
        # 풀에 추가하면 이 문제가 생겨도 원문이 찾은 정답 청크가 살아남는다.
        # 단일질의(n_sub==1)면 subq==원문이라 raw와 완전히 같은 리스트를 한 번
        # 더 병합하는 것뿐이라 결과에 영향이 없다 — 복합질의에서만 의미 있음.
        # latency 측면에서도: entry["raw_is_duplicate"]가 True면 raw를 다시 조회하지
        # 않고 per_subq 결과를 재사용한 것이므로(위 build_subq_cache의 중복제거 최적화),
        # 실제로 추가 비용이 들지 않았다 — search_latency_raw를 또 더하면 안 쓴 시간을
        # 쓴 것처럼 이중 계상하게 된다. 프로덕션(compound_answer.py)도 단일질의면
        # hedged 검색 자체를 안 하므로 이 처리가 실제 동작과 일치한다.
        analysis_latency_hedged = analysis_latency
        search_latency_hedged = search_latency if entry.get("raw_is_duplicate") else search_latency + search_latency_raw
        ranked_ids_hedged = _merge_subquery_results(per_subq_ranked + [ranked_raw], FINAL_K)

        # total_latency_sec/_hedged는 하위질문별 classify+dense+bm25 시간을 그대로
        # 합산한 값이라 "순차로 돌렸다면"을 가정한 상한선이다. 실제로는 하위질문(+raw)을
        # ThreadPoolExecutor로 병렬 처리하므로(build_subq_cache._build_one), 체감
        # latency는 entry["decompose_latency"] + entry["subq_wall_latency"](병렬 블록의
        # 실측 wall-clock)가 더 정확하다. raw가 그 병렬 블록에 같이 들어가 있어서(단일
        # 질의처럼 raw_is_duplicate인 경우 제외) qa/hedged 둘 다 이 값을 그대로 쓴다.
        total_latency_actual = entry["decompose_latency"] + entry["subq_wall_latency"]

        row = {
            "evaluation_id": eid,
            "n_sub_questions": len(entry["subqs"]),
            "analysis_latency_sec": analysis_latency,
            "search_latency_sec": search_latency,
            "total_latency_sec": analysis_latency + search_latency,
            "search_latency_sec_raw": search_latency_raw,
            "total_latency_sec_raw": search_latency_raw,
            "search_latency_sec_hedged": search_latency_hedged,
            "total_latency_sec_hedged": analysis_latency_hedged + search_latency_hedged,
            "total_latency_sec_actual": total_latency_actual,
        }
        row.update(_score_ranked(ranked_ids_qa, gold_ids, "qa"))
        row.update(_score_ranked(ranked_ids_raw, gold_ids, "raw"))
        row.update(_score_ranked(ranked_ids_hedged, gold_ids, "hedged"))
        rows.append(row)

    detail_df = pd.DataFrame(rows)
    summary = {"method": name, "n_questions": len(detail_df)}
    metric_cols = [
        "analysis_latency_sec", "search_latency_sec", "total_latency_sec",
        "search_latency_sec_raw", "total_latency_sec_raw",
        "search_latency_sec_hedged", "total_latency_sec_hedged",
        "total_latency_sec_actual",
    ] + [f"{m}_{suffix}" for suffix in ("qa", "raw", "hedged") for m in
         ("hit@3", "recall@5", "precision@5", "f1@5", "mrr@10", "map@10")]
    for metric in metric_cols:
        summary[metric] = round(float(detail_df[metric].mean()), 4)

    return summary, detail_df


def evaluate(*, limit: int | None = None) -> pd.DataFrame:
    all_summaries = []

    for ds in DATASETS:
        label = ds["label"]
        print(f"\n{'=' * 70}\n데이터셋: {label} ({ds['path'].name} / {ds['sheet']})\n{'=' * 70}")

        gold_records = ds["loader"](ds["path"], ds["sheet"], label=label)
        print(
            f"평가 문항 수: {len(gold_records)}"
            + (f" (상위 {limit}건만 시험 실행)" if limit else "")
        )

        cache_path = CACHE_DIR / f"subq_cache_{label}.pkl"
        cache = build_subq_cache(gold_records, limit=limit, cache_path=cache_path)
        records = gold_records[:limit] if limit else gold_records

        for name, uses_bm25, fuse_fn in METHODS:
            summary, detail_df = run_method(name, uses_bm25, fuse_fn, records, cache)
            summary = {"dataset": label, **summary}
            all_summaries.append(summary)
            detail_path = RESULTS_DIR / f"detail_{label}_{name}.csv"
            detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
            print(f"\n=== [{label}] {name} ===")
            print(summary)
            print("문항별 상세 저장:", detail_path)

    summary_df = pd.DataFrame(all_summaries)
    summary_path = RESULTS_DIR / "summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print("\n요약 저장:", summary_path)
    print(summary_df.to_string(index=False))
    return summary_df


if __name__ == "__main__":
    evaluate()
