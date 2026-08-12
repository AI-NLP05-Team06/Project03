from __future__ import annotations

import ast
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
V3_RAW = WORKSPACE / "2026-08-12-kdic-lightweight-rag-v3-evaluation" / "kdic_lightweight_rag_v3_eval_raw.jsonl"
V3_CSV = WORKSPACE / "2026-08-12-kdic-lightweight-rag-v3-evaluation" / "kdic_lightweight_rag_v3_eval_per_question.csv"
V31_NOTEBOOK = ROOT / "2026-08-12-KDIC-경량-RAG-질의분석-Colab-v3-1.ipynb"

BUSINESS_KEYWORDS = {
    "예금자보호제도": ["예금자보호", "보호한도", "보호대상", "예금 보호", "보호 한도"],
    "예금보험금 안내": ["예금보험금", "보험금 지급", "보험사고", "가지급금", "개산지급금", "1종 보험사고", "2종 보험사고"],
    "고객 미수령금 신청": ["미수령금", "파산배당금", "개산지급금 정산금", "지급대행점", "상속인 금융거래 조회"],
    "착오송금 반환 신청": ["착오송금", "잘못 보낸 돈", "잘못 송금", "착오 송금", "반환지원", "매입계약", "지급명령", "강제집행"],
    "채무조정 안내": ["채무조정", "신용회복지원", "파산선고", "면책", "채무감면", "개인회생", "개인파산", "워크아웃", "부채증명원"],
    "은닉재산 신고": ["은닉재산", "은닉 재산", "금융부실관련자", "부실관련자", "차명 재산", "차명재산", "신고 포상금"],
}
STRONG_KEYWORDS = {
    "예금자보호제도": ["예금자보호", "보호한도", "예금 보호", "보호 한도"],
    "예금보험금 안내": ["예금보험금", "보험금 지급", "1종 보험사고", "2종 보험사고"],
    "고객 미수령금 신청": ["미수령금", "파산배당금", "지급대행점", "상속인 금융거래 조회"],
    "착오송금 반환 신청": ["착오송금", "착오 송금", "반환지원", "매입계약", "지급명령", "강제집행"],
    "채무조정 안내": ["채무조정", "신용회복지원", "개인회생", "개인파산", "워크아웃", "부채증명원"],
    "은닉재산 신고": ["은닉재산", "은닉 재산", "금융부실관련자", "차명재산", "차명 재산", "신고 포상금"],
}
MULTI_SIGNAL = re.compile(
    r"각각|뿐만\s*아니라|함께\s*알려|동시에\s*알려|(?:와|과).{0,30}(?:관계|차이|비교)", re.I
)


def find_businesses(text: str, mapping: dict[str, list[str]] = BUSINESS_KEYWORDS) -> list[str]:
    lowered = (text or "").lower()
    return [business for business, terms in mapping.items() if any(term.lower() in lowered for term in terms)]


def replay_filter_modes(item: dict) -> list[dict]:
    gold = item["gold"]
    result = item["result"]
    needs = result.get("analysis", {}).get("needs", [])
    original = gold["question"]
    decomposition = "PARTIAL" if len(needs) == 1 and MULTI_SIGNAL.search(original) else "COMPLETE"
    output = []
    for need in needs:
        query = need.get("query") or original
        business = need.get("business_function")
        candidates = find_businesses(query) or (find_businesses(original) if len(needs) == 1 else [])
        candidates = list(dict.fromkeys(candidates))
        strong = find_businesses(query, STRONG_KEYWORDS)
        if not strong and len(needs) == 1:
            strong = find_businesses(original, STRONG_KEYWORDS)
        conflict = bool(need.get("model_business_function") and need.get("model_business_function") != business)
        confidence = 0.99 if business in strong and len(candidates) == 1 else (0.90 if business in candidates and len(candidates) == 1 else 0.70)
        eligible = bool(
            decomposition == "COMPLETE"
            and business
            and len(candidates) == 1
            and not conflict
            and business in strong
            and confidence >= 0.98
        )
        output.append({
            "need_id": need.get("need_id"), "business": business,
            "mode": "HARD" if eligible else ("SOFT" if business else "NONE"),
            "candidates": candidates, "strong": strong, "decomposition": decomposition,
            "confidence": confidence, "conflict": conflict,
        })
    return output


def validate_notebook_syntax() -> None:
    notebook = json.loads(V31_NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        source = "".join(source) if isinstance(source, list) else source
        # `%pip`, `%cd`, `!ls` 등은 Colab/IPython 명령이며 일반 Python AST 대상이 아니다.
        source = "\n".join(
            line for line in source.splitlines()
            if not line.lstrip().startswith(("%", "!"))
        )
        try:
            ast.parse(source)
        except SyntaxError as exc:
            raise AssertionError(f"코드 셀 {index} 문법 오류: {exc}") from exc


def main() -> int:
    if not V3_RAW.exists() or not V3_CSV.exists() or not V31_NOTEBOOK.exists():
        raise FileNotFoundError("V3 평가 원본 또는 생성된 V3.1 노트북이 없습니다.")
    validate_notebook_syntax()

    items = []
    with V3_RAW.open("r", encoding="utf-8") as handle:
        for line in handle:
            items.append(json.loads(line))
    by_id = {item["gold"]["evaluation_id"]: item for item in items}

    with V3_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    wrong_filters = []
    mode_counts = Counter()
    for item in items:
        gold_businesses = {pair[0] for pair in item["gold"].get("gold_request_pairs", [])}
        for plan in replay_filter_modes(item):
            mode_counts[plan["mode"]] += 1
            if plan["mode"] == "HARD" and plan["business"] not in gold_businesses:
                wrong_filters.append({
                    "evaluation_id": item["gold"]["evaluation_id"],
                    "question": item["gold"]["question"],
                    "plan": plan,
                    "gold_businesses": sorted(gold_businesses),
                })

    critical_ids = ["Q026", "Q074", "Q083"]
    critical = {
        evaluation_id: replay_filter_modes(by_id[evaluation_id])
        for evaluation_id in critical_ids
    }
    critical_hard = [
        (evaluation_id, plan)
        for evaluation_id, plans in critical.items()
        for plan in plans
        if plan["mode"] == "HARD"
    ]

    execution_success = sum(row["analysis_status"] in {"OK", "FAST_PATH"} for row in rows)
    result = {
        "notebook_code_cells_syntax_valid": True,
        "offline_replay_scope": "V3 raw final needs; no external API calls",
        "evaluation_count": len(rows),
        "execution_success_rate_from_v3": execution_success / len(rows),
        "v31_replayed_filter_mode_counts": dict(mode_counts),
        "v31_replayed_wrong_hard_filter_count": len(wrong_filters),
        "v31_replayed_wrong_hard_filters": wrong_filters,
        "critical_case_modes": critical,
        "critical_case_hard_count": len(critical_hard),
        "hard_gate_policy_replay_pass": len(wrong_filters) == 0 and len(critical_hard) == 0,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["hard_gate_policy_replay_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
