# [Calibrate] Hybrid 검색 결과의 "1등-2등 점수 격차(gap)"가 정답 여부와
# 상관관계가 있는지 확인해서, reranker를 조건부로만 켤 임계값을 데이터 기반으로 잡습니다.
# (reranker 없이 hybrid_search만 돌리므로 빠릅니다.)
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from core.config import *
from core.integrity_check import *
from retrieval.hybrid_search import hybrid_search
from evaluation.eval_search import load_gold_set, hit_at_k

gold_records = load_gold_set()
print("평가 문항 수:", len(gold_records), flush=True)

rows = []
for i, record in enumerate(gold_records, 1):
    results = hybrid_search(
        record["question"],
        top_k=5,
        business_function=None,
    )
    scores = [r["score"] for r in results]
    gap = scores[0] - scores[1] if len(scores) >= 2 else None

    ranked_ids = [r["chunk"]["chunk_id"] for r in results]
    hit3 = hit_at_k(ranked_ids, record["gold_chunk_ids"], 3)

    rows.append({
        "evaluation_id": record["evaluation_id"],
        "business_function": record["business_function"],
        "top1_score": scores[0] if scores else None,
        "gap": gap,
        "hit@3": hit3,
    })
    if i % 20 == 0:
        print(f"  진행: {i}/{len(gold_records)}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(EXPERIMENTS_ROOT / "gap_calibration.csv", index=False, encoding="utf-8-sig")

print("\n=== hit@3=1(정답) vs hit@3=0(오답) 그룹별 gap 분포 ===")
print(df.groupby("hit@3")["gap"].describe())

print("\n=== gap 구간별 hit@3 비율 (구간을 촘촘히 나눠서 어디서 갈리는지 확인) ===")
bins = [0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 1.0]
df["gap_bin"] = pd.cut(df["gap"], bins=bins)
print(df.groupby("gap_bin")["hit@3"].agg(["mean", "count"]))

print("\ngap_calibration.csv 저장 완료:", EXPERIMENTS_ROOT / "gap_calibration.csv")
