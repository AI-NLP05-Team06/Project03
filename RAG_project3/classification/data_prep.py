# [1/6] 검색평가데이터셋.xlsx 정제: requests 컬럼 malformed JSON 복구 + 의도(정보/민원처리) 파생 라벨 부여
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

ROUTING_EVAL_PATH = Path("data/검색평가데이터셋.xlsx")
ROUTING_EVAL_SHEET = "데이터셋_v5"
CLEAN_OUTPUT_PATH = Path("data/routing_eval_clean.jsonl")

# requests 컬럼에 콤마·콜론 누락, "?ELIGIBILITY" 같은 깨진 토큰이 섞여 있어
# json.loads가 실패한다. business_function 키는 모든 샘플에서 깨지지 않았으므로
# "정상적인 information_need 값 개수"와 "business_function 키 등장 횟수"를 각각
# 세어 파싱 신뢰도를 판단한다.
INFO_NEED_PATTERN = re.compile(
    r'"information_need":"(OVERVIEW|ELIGIBILITY|APPLICATION|DOCUMENTS|TIME|AMOUNT|STATUS|CONTACT)"'
)
BUSINESS_FUNCTION_KEY_PATTERN = re.compile(r'"business_function":"')

# 세션에서 실제 문항 샘플로 확정한 매핑 (정보질문: grounded 답변 / 민원처리질문: 절차·서류·페이지연결 템플릿)
INFO_INTENT_NEEDS = {"OVERVIEW", "ELIGIBILITY", "AMOUNT", "CONTACT", "TIME"}
COMPLAINT_INTENT_NEEDS = {"APPLICATION", "DOCUMENTS", "STATUS"}


def _extract_information_needs(raw: Any) -> list[str]:
    if not isinstance(raw, str) or not raw.strip():
        return []
    return INFO_NEED_PATTERN.findall(raw)


def _expected_request_count(raw: Any) -> int:
    if not isinstance(raw, str):
        return 0
    return len(BUSINESS_FUNCTION_KEY_PATTERN.findall(raw))


def _derive_intent(information_needs: list[str]) -> str | None:
    # 한 질문에 정보/민원처리 요청이 섞여 있으면 민원처리 템플릿(절차 포함)이
    # 정보 템플릿보다 상위 호환이므로 민원처리를 우선한다.
    needs = set(information_needs)
    if needs & COMPLAINT_INTENT_NEEDS:
        return "민원처리"
    if needs & INFO_INTENT_NEEDS:
        return "정보"
    return None


def _parse_str_list(value: Any) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    parsed = json.loads(value)
    return parsed if isinstance(parsed, list) else []


def load_routing_eval_set(
    path: Path = ROUTING_EVAL_PATH,
    sheet_name: str = ROUTING_EVAL_SHEET,
) -> list[dict[str, Any]]:
    df = pd.read_excel(path, sheet_name=sheet_name)

    records: list[dict[str, Any]] = []
    parse_warnings: list[str] = []
    for _, row in df.iterrows():
        information_needs = _extract_information_needs(row["requests"])
        expected_count = _expected_request_count(row["requests"])
        if expected_count != len(information_needs):
            parse_warnings.append(str(row["evaluation_id"]))

        missing_slots = _parse_str_list(row["missing_slots"])

        records.append({
            "evaluation_id": row["evaluation_id"],
            "question": row["question"],
            "route_type": row["route_type"],
            "business_functions": _parse_str_list(row["business_functions"]),
            "user_roles": _parse_str_list(row["user_roles"]),
            "missing_slots": missing_slots,
            "information_needs": information_needs,
            "request_count": len(information_needs),
            "gold_intent": _derive_intent(information_needs),
            # 5번 되묻기는 "업무가 뭔지 불명확할 때" 조건 하나로만 트리거하기로
            # 스코프를 좁혔으므로, CLARIFY 중에서도 missing_slots에
            # business_function이 포함된 경우만 우리 시스템의 되묻기 정답으로 삼는다.
            "should_clarify": row["route_type"] == "CLARIFY" and "business_function" in missing_slots,
        })

    if parse_warnings:
        print(
            f"[경고] requests 파싱 개수가 예상과 다른 문항 {len(parse_warnings)}건 "
            "(business_function 키 등장 횟수 vs 정상 추출된 information_need 개수 불일치, 수동 검수 권장):",
            ", ".join(parse_warnings),
        )

    return records


def save_clean_dataset(
    records: list[dict[str, Any]],
    path: Path = CLEAN_OUTPUT_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"정제된 데이터셋 저장: {path} ({len(records)}건)")


if __name__ == "__main__":
    records = load_routing_eval_set()
    save_clean_dataset(records)

    print("route_type 분포:", Counter(r["route_type"] for r in records))
    print("gold_intent 분포:", Counter(r["gold_intent"] for r in records))
    print("should_clarify=True 건수:", sum(r["should_clarify"] for r in records))
