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


def _judge(evidence_text: str, answer: str) -> dict:
    raw = hcx_chat_text(
        system_prompt=(
            "당신은 RAG 시스템이 생성한 답변이 근거에 실제로 기반하는지 "
            "엄격하게 검증하는 평가자입니다. 답변에 등장하는 모든 구체적 "
            "사실(금액, 한도, 기한, 절차, 서류, 조건 등)을 근거 원문과 "
            "하나씩 대조하세요. 근거 원문에 없는 내용이 하나라도 있으면 "
            "fully_grounded를 false로 판정하세요. 반드시 아래 JSON 형식만 "
            '출력하세요: {"fully_grounded": true 또는 false, '
            '"unsupported_claims": ["문제된 문장 또는 사실", ...]}'
        ),
        user_prompt=f"""
[근거 원문]
{evidence_text}

[검증할 답변]
{answer}
""".strip(),
        max_tokens=500,
        temperature=0.0,
    )
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {"fully_grounded": None, "unsupported_claims": []}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"fully_grounded": None, "unsupported_claims": []}
    return {
        "fully_grounded": parsed.get("fully_grounded"),
        "unsupported_claims": parsed.get("unsupported_claims", []),
    }


def _run_variant(question: str, search_results: list[dict]) -> dict:
    answer = generate_grounded_hcx_answer(question, search_results)
    is_refusal = answer.strip() == REFUSAL_TEXT
    result = {"answer": answer, "is_refusal": is_refusal, "fully_grounded": None, "unsupported_claims": []}
    if not is_refusal:
        evidence_text = _build_evidence_text(search_results)
        verdict = _judge(evidence_text, answer)
        result["fully_grounded"] = verdict["fully_grounded"]
        result["unsupported_claims"] = verdict["unsupported_claims"]
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
        "expanded_refusal": expanded["is_refusal"],
        "expanded_fully_grounded": expanded["fully_grounded"],
        "expanded_unsupported_claims": json.dumps(expanded["unsupported_claims"], ensure_ascii=False),
    })
    print(f"  진행: {i}/{len(sampled_records)} | baseline={baseline['fully_grounded']} | "
          f"expanded={expanded['fully_grounded']}", flush=True)
    pd.DataFrame(rows).to_csv(detail_path, index=False, encoding="utf-8-sig")

df = pd.DataFrame(rows)
judged = df[~df["baseline_refusal"] & ~df["expanded_refusal"]]

baseline_rate = (judged["baseline_fully_grounded"] == True).mean()  # noqa: E712
expanded_rate = (judged["expanded_fully_grounded"] == True).mean()  # noqa: E712

print("\n=== 컨텍스트확장 전/후 fully_grounded 비율 ===")
print(f"채점 가능 문항(양쪽 다 거부 아님): {len(judged)}/{len(df)}")
print(f"적용 전: {baseline_rate:.1%}")
print(f"적용 후: {expanded_rate:.1%}")
print(f"\n상세 저장: {detail_path}")
