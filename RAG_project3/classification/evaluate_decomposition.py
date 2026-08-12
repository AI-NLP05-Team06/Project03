# [Phase7] 복합질의 판별 정확도: decompose_query()가 반환한 하위질문 개수가
# 실제 복합 여부(검색평가데이터셋의 request_count>=2)와 맞는지 측정한다.
# 문항마다 중간저장 + 재실행 시 이어서 진행(rate limit 대비, 세션 교훈 반영).
from __future__ import annotations

from pathlib import Path

import pandas as pd

from classification.data_prep import load_routing_eval_set
from classification.decomposition import decompose_query

DETAIL_PATH = Path("data/evaluate_decomposition_detail.csv")


def evaluate() -> None:
    records = load_routing_eval_set()
    targets = [r for r in records if r["route_type"] == "RETRIEVE"]
    print(f"평가 대상 {len(targets)}건 (RETRIEVE 문항만)")

    rows = []
    done_ids: set = set()
    if DETAIL_PATH.exists():
        existing_df = pd.read_csv(DETAIL_PATH)
        rows = existing_df.to_dict("records")
        done_ids = set(existing_df["evaluation_id"])
        print(f"이어서 진행: 이미 완료된 문항 {len(done_ids)}건 건너뜀")

    for i, r in enumerate(targets, 1):
        if r["evaluation_id"] in done_ids:
            continue
        gold_compound = r["request_count"] >= 2
        sub_questions = decompose_query(r["question"])
        pred_compound = len(sub_questions) >= 2

        rows.append({
            "evaluation_id": r["evaluation_id"],
            "question": r["question"],
            "gold_request_count": r["request_count"],
            "gold_compound": gold_compound,
            "pred_sub_question_count": len(sub_questions),
            "pred_compound": pred_compound,
            "correct": gold_compound == pred_compound,
            "sub_questions": " | ".join(sub_questions),
        })
        if i % 20 == 0:
            print(f"  진행: {i}/{len(targets)}")
        pd.DataFrame(rows).to_csv(DETAIL_PATH, index=False, encoding="utf-8-sig")

    df = pd.DataFrame(rows)
    print(f"\n=== 복합질의 판별 정확도: {df['correct'].sum()}/{len(df)} ({df['correct'].mean():.1%}) ===")

    tp = ((df["gold_compound"]) & (df["pred_compound"])).sum()
    fp = ((~df["gold_compound"]) & (df["pred_compound"])).sum()
    fn = ((df["gold_compound"]) & (~df["pred_compound"])).sum()
    tn = ((~df["gold_compound"]) & (~df["pred_compound"])).sum()
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    print(f"TP={tp} FP={fp} FN={fn} TN={tn} | Precision={precision:.2f} Recall={recall:.2f}")

    print(f"\n상세 저장: {DETAIL_PATH}")


if __name__ == "__main__":
    evaluate()
