from __future__ import annotations

import json
from pathlib import Path

import numpy as np


data = json.loads((Path(__file__).parent / "KDIC_검색평가_AI분석데이터.json").read_text(encoding="utf-8"))
methods = {method["name"]: method for method in data["methods"]}


def find(fragment: str):
    return next(method for name, method in methods.items() if fragment in name)


dense = methods["BGE-M3 Dense"]
hybrid = next(method for name, method in methods.items() if name.startswith("Hybrid A:") and "Reranker" not in name)
structured = methods["BGE-M3 Dense structured"]
parent = find("Parent-Child")
reranker = find("Reranker")


def rows(method):
    return {row["evaluation_id"]: row for row in method["question_results"]}


def bootstrap(left, right, metric, samples=20000):
    lmap, rmap = rows(left), rows(right)
    ids = sorted(set(lmap) & set(rmap))
    diffs = np.array([
        float(lmap[eid].get(metric) or 0) - float(rmap[eid].get(metric) or 0)
        for eid in ids
    ])
    rng = np.random.default_rng(20260804)
    indices = rng.integers(0, len(diffs), size=(samples, len(diffs)))
    means = diffs[indices].mean(axis=1)
    return float(diffs.mean()), [float(x) for x in np.quantile(means, [0.025, 0.975])]


for left, right, label in [
    (hybrid, dense, "content hybrid - dense"),
    (parent, structured, "structured hybrid - structured"),
    (reranker, parent, "reranker - structured hybrid"),
]:
    print(label)
    for metric in ("hit_at_3", "recall_at_5", "mrr_at_10", "map_at_10", "ndcg_at_5", "f1_at_5"):
        mean, ci = bootstrap(left, right, metric)
        print(metric, round(mean, 6), [round(x, 6) for x in ci])
