# [Phase3] 컨텍스트확장(같은 문서 인접 청크 결합) 적용 전/후로 같은 문항에 대해
# 답변을 각각 생성하고, LLM-judge로 fully_grounded 비율 delta를 측정한다.
# (evaluation/judge_answer_quality.py와 동일한 채점 방식, 표본만 작게 잡음)
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import json
import random
import re

from core.config import *
from core.integrity_check import *
from core.hcx_api import hcx_chat_text
from retrieval.reranker import hybrid_rerank_search
from generation.answer_generation import generate_grounded_hcx_answer
from generation.context_expansion import expand_search_results
from evaluation.eval_search import load_gold_set

SAMPLE_PER_DOMAIN = 2
RANDOM_SEED = 42
CONTEXT_WINDOW = 1
REFUSAL_TEXT = (
    "검색된 공식 근거의 관련도가 충분하지 않아 "
    "현재 수집 데이터만으로는 정확하게 답할 수 없습니다."
)

random.seed(RANDOM_SEED)
gold_records = load_gold_set()
by_domain: dict[str, list[dict]] = {}
for record in gold_records:
    by_domain.setdefault(record["business_function"], []).append(record)

sampled_records: list[dict] = []
for domain, records in by_domain.items():
    sampled_records.extend(
        random.sample(records, min(SAMPLE_PER_DOMAIN, len(records)))
    )

print(f"채점 대상 문항 수: {len(sampled_records)} (도메인 {len(by_domain)}개 x 최대 {SAMPLE_PER_DOMAIN}개)")


def _build_evidence_text(search_results: list[dict]) -> str:
    blocks = []
    for rank, result in enumerate(search_results, start=1):
        chunk = result["chunk"]
        blocks.append("\n".join([
            f"[근거 {rank}]",
            f"청크 ID: {chunk.get('chunk_id')}",
            "내용:",
            chunk.get("content", ""),
        ]))
    return "\n\n".join(blocks)


def _judge(question: str, evidence_text: str, answer: str) -> dict:
    raw = hcx_chat_text(
        system_prompt=(
            "당신은 RAG 시스템이 생성한 답변을 두 가지 축으로 검증하는 "
            "평가자입니다.\n"
            "1) grounded: 답변에 등장하는 모든 구체적 사실(금액, 한도, 기한, "
            "절차, 서류, 조건 등)을 근거 원문과 하나씩 대조하세요. 근거 "
            "원문에 없는 내용이 하나라도 있으면 fully_grounded를 false로 "
            "판정하세요.\n"
            "2) completeness: 근거 원문에 있고 사용자 질문에 답하는 데 "
            "필요한 정보인데 답변에서 빠진 내용이 있는지 확인하세요. "
            "근거에 있지만 질문과 무관한 내용까지 다 담으라는 게 아니라, "
            "'이 질문에 답하려면 꼭 필요한데 빠진 것'만 missing_points에 "
            "적으세요. 빠진 게 없으면 빈 리스트로 두세요.\n"
            "반드시 아래 JSON 형식만 출력하세요: "
            '{"fully_grounded": true 또는 false, '
            '"unsupported_claims": ["문제된 문장 또는 사실", ...], '
            '"missing_points": ["질문에 필요한데 답변에서 빠진 내용", ...]}'
        ),
        user_prompt=f"""
[사용자 질문]
{question}

[근거 원문]
{evidence_text}

[검증할 답변]
{answer}
""".strip(),
        max_tokens=600,
        temperature=0.0,
    )
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {"fully_grounded": None, "unsupported_claims": [], "missing_points": []}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"fully_grounded": None, "unsupported_claims": [], "missing_points": []}
    return {
        "fully_grounded": parsed.get("fully_grounded"),
        "unsupported_claims": parsed.get("unsupported_claims", []),
        "missing_points": parsed.get("missing_points", []),
    }


def _run_variant(question: str, search_results: list[dict]) -> dict:
    answer = generate_grounded_hcx_answer(question, search_results)
    is_refusal = answer.strip() == REFUSAL_TEXT
    result = {
        "answer": answer,
        "is_refusal": is_refusal,
        "fully_grounded": None,
        "unsupported_claims": [],
        "missing_points": [],
    }
    if not is_refusal:
        evidence_text = _build_evidence_text(search_results)
        verdict = _judge(question, evidence_text, answer)
        result["fully_grounded"] = verdict["fully_grounded"]
        result["unsupported_claims"] = verdict["unsupported_claims"]
        result["missing_points"] = verdict["missing_points"]
    return result


detail_path = EXPERIMENTS_ROOT / "context_expansion_comparison.csv"
rows = []

for i, record in enumerate(sampled_records, 1):
    question = record["question"]
    search_results = hybrid_rerank_search(question, top_k=5, business_function=None)

    baseline = _run_variant(question, search_results)
    expanded_results = expand_search_results(search_results, window=CONTEXT_WINDOW)
    expanded = _run_variant(question, expanded_results)

    rows.append({
        "evaluation_id": record["evaluation_id"],
        "business_function": record["business_function"],
        "question": question,
        "baseline_refusal": baseline["is_refusal"],
        "baseline_fully_grounded": baseline["fully_grounded"],
        "baseline_missing_points": json.dumps(baseline["missing_points"], ensure_ascii=False),
        "baseline_answer": baseline["answer"],
        "expanded_refusal": expanded["is_refusal"],
        "expanded_fully_grounded": expanded["fully_grounded"],
        "expanded_unsupported_claims": json.dumps(expanded["unsupported_claims"], ensure_ascii=False),
        "expanded_missing_points": json.dumps(expanded["missing_points"], ensure_ascii=False),
        "expanded_answer": expanded["answer"],
    })
    print(f"  진행: {i}/{len(sampled_records)} | baseline_missing={len(baseline['missing_points'])} | "
          f"expanded_missing={len(expanded['missing_points'])}", flush=True)
    pd.DataFrame(rows).to_csv(detail_path, index=False, encoding="utf-8-sig")

df = pd.DataFrame(rows)
judged = df[~df["baseline_refusal"] & ~df["expanded_refusal"]]

baseline_rate = (judged["baseline_fully_grounded"] == True).mean()  # noqa: E712
expanded_rate = (judged["expanded_fully_grounded"] == True).mean()  # noqa: E712
baseline_complete_rate = (judged["baseline_missing_points"] == "[]").mean()
expanded_complete_rate = (judged["expanded_missing_points"] == "[]").mean()

print("\n=== 컨텍스트확장 전/후 fully_grounded 비율 ===")
print(f"채점 가능 문항(양쪽 다 거부 아님): {len(judged)}/{len(df)}")
print(f"적용 전: {baseline_rate:.1%}")
print(f"적용 후: {expanded_rate:.1%}")

print("\n=== 컨텍스트확장 전/후 completeness(빠진 내용 없음) 비율 ===")
print(f"적용 전: {baseline_complete_rate:.1%}")
print(f"적용 후: {expanded_complete_rate:.1%}")

improved = judged[(judged["baseline_missing_points"] != "[]") & (judged["expanded_missing_points"] == "[]")]
if len(improved):
    print(f"\n확장 후 누락이 해소된 문항 {len(improved)}건:")
    for _, r in improved.iterrows():
        print(f"  - {r['evaluation_id']}: 이전 누락 {r['baseline_missing_points']}")

print(f"\n상세 저장: {detail_path}")
