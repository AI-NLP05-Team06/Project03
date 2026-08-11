# [Phase2] 실제 FAQ 원문(chunks.jsonl, page_type=="faq", 96건, gold=business_function)을
# 별도 검증셋으로 써서 구어체 정규화 전/후 rule 티어 커버리지·정답률을 비교한다.
from __future__ import annotations

import json

from classification.normalization import normalize_colloquial
from classification.rules import classify_business_function_rule


def load_faq_gold() -> list[tuple[str, str]]:
    faqs = []
    with open("data/chunks.jsonl", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record.get("page_type") == "faq" and record.get("section_title"):
                faqs.append((record["section_title"], record["business_function"]))
    return faqs


def _score(faqs: list[tuple[str, str]], apply_normalization: bool) -> None:
    covered = correct = 0
    for question, gold in faqs:
        text = normalize_colloquial(question) if apply_normalization else question
        pred = classify_business_function_rule(text)
        if pred is not None:
            covered += 1
            if pred == gold:
                correct += 1
    label = "정규화 후" if apply_normalization else "정규화 전"
    print(f"[{label}] {len(faqs)}건 중 커버리지 {covered}건 ({covered/len(faqs):.1%}), "
          f"정답률 {correct}/{covered} ({correct/covered:.1%} of covered)")


if __name__ == "__main__":
    faqs = load_faq_gold()
    _score(faqs, apply_normalization=False)
    _score(faqs, apply_normalization=True)
