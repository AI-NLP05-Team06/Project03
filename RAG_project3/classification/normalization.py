# [Phase2] 규칙기반 질의보정: 구어체/약칭 표현을 공식 용어로 치환한다.
# data/chunks.jsonl의 실제 FAQ 원문(page_type=="faq", 96건)에서 관찰된 표현만 담았다.
from __future__ import annotations

# 매칭 우선순위(더 긴/구체적인 표현 먼저)를 위해 dict 순서를 유지한다.
COLLOQUIAL_NORMALIZATION: dict[str, str] = {
    "예보": "예금보험공사",
    "부도나면": "영업정지되면",
    "부도난": "영업정지된",
    "부도": "영업정지",
    "망하면": "영업정지되면",
    "망한": "영업정지된",
    "떼인 돈": "미수령금",
    "떼인돈": "미수령금",
    "못 받은 돈": "미수령금",
    "못받은 돈": "미수령금",
}


def normalize_colloquial(text: str) -> str:
    normalized = text
    for colloquial, formal in COLLOQUIAL_NORMALIZATION.items():
        normalized = normalized.replace(colloquial, formal)
    return normalized
