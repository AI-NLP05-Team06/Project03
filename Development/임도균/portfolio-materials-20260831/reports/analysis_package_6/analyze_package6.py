from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT / "KDIC_검색평가_AI분석데이터.json").read_text(encoding="utf-8"))

print("top", list(data))
print("active_filter", json.dumps(data.get("active_question_filter"), ensure_ascii=False, indent=2))
print("protocol", json.dumps(data.get("evaluation_protocol"), ensure_ascii=False, indent=2))

for method in data["methods"]:
    print("\nMETHOD", method["name"])
    print("keys", list(method))
    print("overall", json.dumps(method.get("overall"), ensure_ascii=False, indent=2))
    diagnostics = method.get("dataset_diagnostics")
    if diagnostics:
        print("diagnostics", json.dumps(diagnostics, ensure_ascii=False, indent=2)[:8000])

metric7 = [
    "hit_at_3", "recall_at_5", "mrr_at_10", "map_at_10",
    "ndcg_at_5", "precision_at_5", "f1_at_5",
]
print("\nSEVEN METRIC MEAN")
for method in data["methods"]:
    overall = method["overall"]
    score = sum(float(overall[key]) for key in metric7) / len(metric7)
    print(method["name"], round(score, 6))

print("\nDOMAIN TABLE")
for method in data["methods"]:
    print("\n", method["name"])
    for row in method["by_domain"]:
        print(
            row["group_value"], row["question_count"],
            "hit", round(float(row["hit_at_3"]), 4),
            "recall", round(float(row["recall_at_5"]), 4),
            "mrr", round(float(row["mrr_at_10"]), 4),
            "ndcg", round(float(row["ndcg_at_5"]), 4),
        )

parent = next(method for method in data["methods"] if "Parent-Child" in method["name"])
print("\nPARENT FIELDS")
print([key for key in parent["question_results"][0] if any(word in key for word in ("parent", "expanded", "expansion", "seed"))])

parent_keys = [
    "parent_hit_at_3", "parent_recall_at_3", "child_seed_recall",
    "expanded_gold_recall", "expanded_primary_recall",
    "expanded_supporting_recall", "expansion_recall_gain",
    "expanded_gold_count", "expanded_context_chunk_count",
    "expansion_added_chunk_count", "expanded_non_gold_ratio",
    "expanded_context_char_count",
]
for key in parent_keys:
    values = []
    for row in parent["question_results"]:
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            pass
    if values:
        print(key, round(sum(values) / len(values), 6), "n", len(values))

gains = []
for row in parent["question_results"]:
    try:
        gain = float(row.get("expansion_recall_gain") or 0)
    except (TypeError, ValueError):
        gain = 0
    gains.append((gain, row["evaluation_id"], row["question"]))
print("improved", sum(g > 0 for g, *_ in gains), "worsened", sum(g < 0 for g, *_ in gains), "same", sum(g == 0 for g, *_ in gains))
print("top gains", sorted(gains, reverse=True)[:20])
print("top losses", sorted(gains)[:20])

def by_id(method):
    return {row["evaluation_id"]: row for row in method["question_results"]}

structured = next(method for method in data["methods"] if method["name"] == "BGE-M3 Dense structured")
reranker = next(method for method in data["methods"] if "Reranker" in method["name"])
for left, right, label in [
    (parent, structured, "parenthybrid-minus-structured"),
    (reranker, parent, "reranker-minus-parenthybrid"),
]:
    lmap, rmap = by_id(left), by_id(right)
    print("\nPAIR", label)
    for metric in ("hit_at_3", "recall_at_5", "mrr_at_10", "map_at_10", "ndcg_at_5", "f1_at_5"):
        diffs = []
        for eid in sorted(set(lmap) & set(rmap)):
            lv = float(lmap[eid].get(metric) or 0)
            rv = float(rmap[eid].get(metric) or 0)
            diffs.append((lv-rv, eid, lmap[eid]["question"]))
        print(metric, "mean", round(sum(x[0] for x in diffs)/len(diffs), 6), "wins", sum(x[0]>0 for x in diffs), "loss", sum(x[0]<0 for x in diffs))
    hitdiff = []
    for eid in sorted(set(lmap) & set(rmap)):
        diff = float(lmap[eid].get("hit_at_3") or 0)-float(rmap[eid].get("hit_at_3") or 0)
        if diff:
            hitdiff.append((diff, eid, lmap[eid]["question"]))
    print("hit changed", hitdiff)

print("\nCOMMON FAIL HIT@3")
maps = [by_id(method) for method in data["methods"]]
common = []
for eid in sorted(set.intersection(*(set(mapping) for mapping in maps))):
    if all(float(mapping[eid].get("hit_at_3") or 0) == 0 for mapping in maps):
        common.append((eid, maps[0][eid]["question"], maps[0][eid]["gold_business_function"]))
print(len(common), common)

# Complete@5는 AI 패키지 변환 과정에서 빈값이 0으로 바뀐 흔적이 있어 원본 XLSX로 재계산한다.
dataset_path = Path(
    r"C:\Users\임도균\Desktop\2026-08-04 (검색방식)"
    r"\KDIC_검색평가_v3_5_업무필터없음_Colab_패키지"
    r"\06_Hybrid_A_Reranker_업무필터없음\test_dataset_4.1.xlsx"
)
frame = pd.read_excel(dataset_path, sheet_name="Sheet1", dtype=str).fillna("")
frame = frame[frame["evaluation_target"].str.upper().eq("Y")]
applicable = {}
for _, row in frame.iterrows():
    primary = json.loads(row["gold_primary_chunk_ids"] or "[]")
    if str(row["multi_chunk_required"]).upper() == "Y" and 2 <= len(primary) <= 5:
        applicable[row["evaluation_id"]] = set(primary)
print("\nCORRECT COMPLETE@5 applicable", len(applicable), sorted(applicable))
for method in data["methods"]:
    hits = []
    rows = by_id(method)
    for eid, primary in applicable.items():
        retrieved = json.loads(rows[eid]["retrieved_chunk_ids"] or "[]")[:5]
        hits.append(float(primary.issubset(set(retrieved))))
    print(method["name"], sum(hits) / len(hits) if hits else None, sum(hits), "/", len(hits))
