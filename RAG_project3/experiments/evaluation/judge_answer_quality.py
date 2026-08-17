# [LLM-judge] 확정 프로덕션 파이프라인(Hybrid+Reranker → generate_grounded_hcx_answer,
# 수치 검증 2차 패스 포함)으로 실제 답변을 생성한 뒤, 그 답변의 모든 구체적 사실이
# 근거 원문에 실제로 있는지 별도 HCX 호출로 채점합니다. 검색 지표(hit@3 등)와 달리
# "생성된 답변 자체가 근거에 얼마나 충실한지"를 재는 답변 품질 축입니다.
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
from generation.answer_generation import generate_grounded_hcx_answer
from experiments.evaluation.eval_search import load_gold_set

SAMPLE_PER_DOMAIN = 4
RANDOM_SEED = 42
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
        return {"fully_grounded": None, "unsupported_claims": [], "judge_raw": raw}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"fully_grounded": None, "unsupported_claims": [], "judge_raw": raw}
    return {
        "fully_grounded": parsed.get("fully_grounded"),
        "unsupported_claims": parsed.get("unsupported_claims", []),
        "judge_raw": raw,
    }


detail_path = DETAIL_ROOT / "answer_quality_judge.csv"

rows = []
done_ids: set = set()
if detail_path.exists():
    existing_df = pd.read_csv(detail_path)
    rows = existing_df.to_dict("records")
    done_ids = set(existing_df["evaluation_id"])
    print(f"이어서 진행: 이미 완료된 문항 {len(done_ids)}건 건너뜀", flush=True)

for i, record in enumerate(sampled_records, 1):
    if record["evaluation_id"] in done_ids:
        continue
    question = record["question"]
    search_results = hybrid_rerank_search(question, top_k=5, business_function=None)
    answer = generate_grounded_hcx_answer(question, search_results)

    is_refusal = answer.strip() == REFUSAL_TEXT
    row = {
        "evaluation_id": record["evaluation_id"],
        "business_function": record["business_function"],
        "question": question,
        "top1_score": search_results[0]["score"] if search_results else None,
        "is_refusal": is_refusal,
        "fully_grounded": None,
        "unsupported_claims": None,
    }

    if not is_refusal:
        evidence_text = _build_evidence_text(search_results)
        verdict = _judge(evidence_text, answer)
        row["fully_grounded"] = verdict["fully_grounded"]
        row["unsupported_claims"] = json.dumps(
            verdict["unsupported_claims"], ensure_ascii=False
        )

    rows.append(row)
    print(f"  진행: {i}/{len(sampled_records)} | refusal={is_refusal} | fully_grounded={row['fully_grounded']}", flush=True)

    # 문항마다 즉시 저장 — 중간에 죽어도 여기까지 진행된 결과는 남습니다.
    pd.DataFrame(rows).to_csv(detail_path, index=False, encoding="utf-8-sig")

detail_df = pd.DataFrame(rows)

judged = detail_df[~detail_df["is_refusal"]]
n_judged = len(judged)
n_grounded = int((judged["fully_grounded"] == True).sum())  # noqa: E712
n_ungrounded = int((judged["fully_grounded"] == False).sum())  # noqa: E712
n_unparsed = n_judged - n_grounded - n_ungrounded

summary = {
    "n_sampled": len(sampled_records),
    "n_refusal": int(detail_df["is_refusal"].sum()),
    "n_judged": n_judged,
    "n_fully_grounded": n_grounded,
    "n_unsupported": n_ungrounded,
    "n_judge_unparsed": n_unparsed,
    "grounded_rate": round(n_grounded / n_judged, 4) if n_judged else None,
}

print("\n=== 답변 품질(LLM-judge) 결과 ===")
print(json.dumps(summary, ensure_ascii=False, indent=2))
print("\n문항별 상세:", detail_path)

if n_ungrounded:
    print("\n근거 불충족으로 판정된 문항:")
    for _, r in judged[judged["fully_grounded"] == False].iterrows():  # noqa: E712
        print(f"  - {r['evaluation_id']} ({r['business_function']}): {r['unsupported_claims']}")
