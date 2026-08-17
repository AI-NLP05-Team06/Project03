# [Phase7] Decomposition 품질: requests[].request_text(정답 하위질문 요지)가
# decompose_query()의 실제 출력에 얼마나 커버되는지 LLM-judge로 채점한다.
from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pandas as pd

from core.hcx_api import hcx_chat_text
from classification.data_prep import load_routing_eval_set
from classification.decomposition import decompose_query

SAMPLE_SIZE = 20
RANDOM_SEED = 42
DETAIL_PATH = Path("data/evaluate_decomposition_quality_detail.csv")

_JUDGE_SYSTEM_PROMPT = """당신은 질의 분해(query decomposition) 품질을 채점하는
평가자입니다. "정답 요지 목록"은 원래 질문이 실제로 담고 있는 요청들을 요약한
것이고, "생성된 하위질문"은 어떤 시스템이 원래 질문을 분해한 결과입니다.

정답 요지 각각에 대해, 생성된 하위질문 중 그 요지를 실제로 다루는 것이
있는지 판단하세요. 표현이 달라도 같은 내용을 묻고 있으면 커버된 것으로
인정하세요.

반드시 아래 JSON 형식만 출력하세요:
{"covered": ["커버된 정답 요지", ...], "missing": ["커버 안 된 정답 요지", ...]}"""


def _judge_coverage(request_texts: list[str], sub_questions: list[str]) -> dict:
    raw = hcx_chat_text(
        system_prompt=_JUDGE_SYSTEM_PROMPT,
        user_prompt=f"""
[정답 요지 목록]
{json.dumps(request_texts, ensure_ascii=False)}

[생성된 하위질문]
{json.dumps(sub_questions, ensure_ascii=False)}
""".strip(),
        max_tokens=400,
        temperature=0.0,
    )
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {"covered": [], "missing": request_texts}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"covered": [], "missing": request_texts}
    return {
        "covered": parsed.get("covered", []),
        "missing": parsed.get("missing", []),
    }


def evaluate() -> None:
    records = load_routing_eval_set()
    targets = [
        r for r in records
        if r["route_type"] == "RETRIEVE" and r["request_count"] >= 2 and r["request_texts"]
    ]
    random.seed(RANDOM_SEED)
    sampled = random.sample(targets, min(SAMPLE_SIZE, len(targets)))
    print(f"평가 대상 {len(sampled)}건 (복합질의 전체 {len(targets)}건 중 표본)")

    rows = []
    done_ids: set = set()
    if DETAIL_PATH.exists():
        existing_df = pd.read_csv(DETAIL_PATH)
        rows = existing_df.to_dict("records")
        done_ids = set(existing_df["evaluation_id"])
        print(f"이어서 진행: 이미 완료된 문항 {len(done_ids)}건 건너뜀")

    for i, r in enumerate(sampled, 1):
        if r["evaluation_id"] in done_ids:
            continue
        sub_questions = decompose_query(r["question"])
        verdict = _judge_coverage(r["request_texts"], sub_questions)
        coverage = len(verdict["covered"]) / len(r["request_texts"]) if r["request_texts"] else None

        rows.append({
            "evaluation_id": r["evaluation_id"],
            "question": r["question"],
            "request_texts": " | ".join(r["request_texts"]),
            "sub_questions": " | ".join(sub_questions),
            "covered": " | ".join(verdict["covered"]),
            "missing": " | ".join(verdict["missing"]),
            "coverage": coverage,
        })
        print(f"  {i}/{len(sampled)} | coverage={coverage:.2f}" if coverage is not None else f"  {i}/{len(sampled)} | coverage=N/A")
        pd.DataFrame(rows).to_csv(DETAIL_PATH, index=False, encoding="utf-8-sig")

    df = pd.DataFrame(rows)
    print(f"\n=== 평균 coverage: {df['coverage'].mean():.1%} ===")
    print(f"상세 저장: {DETAIL_PATH}")


if __name__ == "__main__":
    evaluate()
