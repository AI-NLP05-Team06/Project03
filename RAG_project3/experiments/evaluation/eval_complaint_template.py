# [Phase4] 민원처리 3단계 템플릿 평가: 3요소(절차·서류·페이지링크) 포함율(룰체크) +
# LLM-judge 정확성(fully_grounded) 점수.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "production"))

import json
import random
import re

from core.config import *
from core.integrity_check import *
from core.hcx_api import hcx_chat_text
from retrieval.reranker import hybrid_rerank_search
from generation.complaint_template import generate_complaint_template_answer
from classification.data_prep import load_routing_eval_set

SAMPLE_PER_DOMAIN = 2
RANDOM_SEED = 42
REFUSAL_TEXT = (
    "검색된 공식 근거의 관련도가 충분하지 않아 "
    "현재 수집 데이터만으로는 정확하게 답할 수 없습니다."
)

records = load_routing_eval_set()
targets = [
    r for r in records
    if r["route_type"] == "RETRIEVE" and r["gold_intent"] == "민원처리" and r["business_functions"]
]

by_domain: dict[str, list[dict]] = {}
for r in targets:
    by_domain.setdefault(r["business_functions"][0], []).append(r)

random.seed(RANDOM_SEED)
sampled_records: list[dict] = []
for domain, records_in_domain in by_domain.items():
    sampled_records.extend(
        random.sample(records_in_domain, min(SAMPLE_PER_DOMAIN, len(records_in_domain)))
    )

print(f"채점 대상 문항 수: {len(sampled_records)} (도메인 {len(by_domain)}개 x 최대 {SAMPLE_PER_DOMAIN}개)")


def _check_three_elements(answer: str) -> dict:
    return {
        "has_procedure": bool(re.search(r"절차", answer)),
        "has_documents": bool(re.search(r"서류", answer)),
        "has_page_link": bool(re.search(r"https?://", answer)) or bool(re.search(r"페이지", answer)),
    }


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
            "당신은 RAG 시스템이 생성한 답변이 근거에 실제로 기반하는지 "
            "엄격하게 검증하는 평가자입니다. 답변에 등장하는 모든 구체적 "
            "사실(금액, 한도, 기한, 절차, 서류, 조건 등)을 근거 원문과 "
            "하나씩 대조하세요. 근거 원문에 없는 내용이 하나라도 있으면 "
            "fully_grounded를 false로 판정하세요. 반드시 아래 JSON 형식만 "
            '출력하세요: {"fully_grounded": true 또는 false, '
            '"unsupported_claims": ["문제된 문장 또는 사실", ...]}'
        ),
        user_prompt=f"""
[사용자 질문]
{question}

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


detail_path = EXPERIMENTS_ROOT / "complaint_template_eval.csv"
rows = []

for i, record in enumerate(sampled_records, 1):
    question = record["question"]
    search_results = hybrid_rerank_search(question, top_k=5, business_function=None)
    answer = generate_complaint_template_answer(question, search_results)
    is_refusal = answer.strip() == REFUSAL_TEXT

    elements = _check_three_elements(answer) if not is_refusal else {
        "has_procedure": None, "has_documents": None, "has_page_link": None,
    }
    fully_grounded = None
    unsupported_claims = []
    if not is_refusal:
        evidence_text = _build_evidence_text(search_results)
        verdict = _judge(question, evidence_text, answer)
        fully_grounded = verdict["fully_grounded"]
        unsupported_claims = verdict["unsupported_claims"]

    rows.append({
        "evaluation_id": record["evaluation_id"],
        "business_function": record["business_functions"][0],
        "question": question,
        "is_refusal": is_refusal,
        "has_procedure": elements["has_procedure"],
        "has_documents": elements["has_documents"],
        "has_page_link": elements["has_page_link"],
        "fully_grounded": fully_grounded,
        "unsupported_claims": json.dumps(unsupported_claims, ensure_ascii=False),
        "answer": answer,
    })
    print(f"  진행: {i}/{len(sampled_records)} | refusal={is_refusal} | "
          f"elements={elements} | fully_grounded={fully_grounded}", flush=True)
    pd.DataFrame(rows).to_csv(detail_path, index=False, encoding="utf-8-sig")

df = pd.DataFrame(rows)
scored = df[~df["is_refusal"]]

print("\n=== 3요소 포함율 (거부 응답 제외) ===")
print(f"채점 대상: {len(scored)}/{len(df)}")
print(f"절차 포함: {scored['has_procedure'].mean():.1%}")
print(f"서류 포함: {scored['has_documents'].mean():.1%}")
print(f"페이지링크 포함: {scored['has_page_link'].mean():.1%}")

grounded_rate = (scored["fully_grounded"] == True).mean()  # noqa: E712
print(f"\n=== LLM-judge fully_grounded 비율 ===")
print(f"{grounded_rate:.1%} ({int((scored['fully_grounded']==True).sum())}/{len(scored)})")  # noqa: E712

missing = scored[~(scored["has_procedure"] & scored["has_documents"] & scored["has_page_link"])]
if len(missing):
    print(f"\n3요소 중 하나라도 빠진 문항 {len(missing)}건:")
    for _, r in missing.iterrows():
        print(f"  - {r['evaluation_id']} ({r['business_function']}): "
              f"절차={r['has_procedure']} 서류={r['has_documents']} 링크={r['has_page_link']}")

print(f"\n상세 저장: {detail_path}")
