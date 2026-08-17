# [Phase5] 맥락반영 재작성 검증: turn2 원문만으로 검색했을 때 vs 재작성 후
# 검색했을 때, (1) top-1 business_function이 의도한 도메인과 맞는지(거친 지표),
# (2) gold_chunk_ids가 있는 문항에 한해 Hit@3(진짜 정답 청크가 top-3 안에 있는지)를 비교한다.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from core.config import *
from core.integrity_check import *
from retrieval.reranker import hybrid_rerank_search
from classification.context_rewrite import rewrite_with_context
from classification.context_rewrite_testset import load_context_rewrite_testset
from evaluation.eval_search import recall_at_k

TOP_K = 3

testset = load_context_rewrite_testset()
print(f"검증 대상 {len(testset)}건 (gold_chunk_ids 있는 문항: "
      f"{sum(1 for p in testset if p.get('gold_chunk_ids'))}건)")

raw_domain_correct = rewritten_domain_correct = 0
raw_hit3 = rewritten_hit3 = 0
raw_recall3_sum = rewritten_recall3_sum = 0.0
hit3_targets = 0
rows = []

for i, pair in enumerate(testset, 1):
    raw_results = hybrid_rerank_search(pair["turn2"], top_k=TOP_K, business_function=None)
    raw_top1_bf = raw_results[0]["chunk"].get("business_function")
    raw_ids = [r["chunk"]["chunk_id"] for r in raw_results]

    rewritten_question = rewrite_with_context(pair["turn2"], pair["turn1"])
    rewritten_results = hybrid_rerank_search(rewritten_question, top_k=TOP_K, business_function=None)
    rewritten_top1_bf = rewritten_results[0]["chunk"].get("business_function")
    rewritten_ids = [r["chunk"]["chunk_id"] for r in rewritten_results]

    raw_domain_hit = raw_top1_bf == pair["domain"]
    rewritten_domain_hit = rewritten_top1_bf == pair["domain"]
    raw_domain_correct += raw_domain_hit
    rewritten_domain_correct += rewritten_domain_hit

    gold_ids = pair.get("gold_chunk_ids")
    raw_hit3_flag = rewritten_hit3_flag = None
    raw_recall3 = rewritten_recall3 = None
    if gold_ids:
        hit3_targets += 1
        raw_hit3_flag = any(gid in raw_ids for gid in gold_ids)
        rewritten_hit3_flag = any(gid in rewritten_ids for gid in gold_ids)
        raw_hit3 += raw_hit3_flag
        rewritten_hit3 += rewritten_hit3_flag

        raw_recall3 = recall_at_k(raw_ids, gold_ids, TOP_K)
        rewritten_recall3 = recall_at_k(rewritten_ids, gold_ids, TOP_K)
        raw_recall3_sum += raw_recall3
        rewritten_recall3_sum += rewritten_recall3

    rows.append({
        "domain": pair["domain"], "turn2": pair["turn2"], "rewritten": rewritten_question,
        "raw_domain_hit": raw_domain_hit, "rewritten_domain_hit": rewritten_domain_hit,
        "gold_chunk_ids": gold_ids, "raw_hit3": raw_hit3_flag, "rewritten_hit3": rewritten_hit3_flag,
        "raw_recall3": raw_recall3, "rewritten_recall3": rewritten_recall3,
    })
    print(f"  {i}/{len(testset)} | domain: raw={raw_domain_hit} rewritten={rewritten_domain_hit} | "
          f"hit@3: raw={raw_hit3_flag} rewritten={rewritten_hit3_flag} | "
          f"recall@3: raw={raw_recall3} rewritten={rewritten_recall3} | "
          f"{pair['turn2']!r} -> {rewritten_question!r}")

print()
print(f"[도메인 일치율] raw: {raw_domain_correct}/{len(testset)} ({raw_domain_correct/len(testset):.1%}) | "
      f"재작성: {rewritten_domain_correct}/{len(testset)} ({rewritten_domain_correct/len(testset):.1%})")

if hit3_targets:
    print(f"[Hit@3, gold_chunk_ids 있는 {hit3_targets}건만] raw: {raw_hit3}/{hit3_targets} "
          f"({raw_hit3/hit3_targets:.1%}) | 재작성: {rewritten_hit3}/{hit3_targets} ({rewritten_hit3/hit3_targets:.1%})")
    print(f"[Recall@3, 같은 {hit3_targets}건] raw: {raw_recall3_sum/hit3_targets:.1%} | "
          f"재작성: {rewritten_recall3_sum/hit3_targets:.1%}")

no_gold = [r for r in rows if r["gold_chunk_ids"] is None]
if no_gold:
    print(f"\ngold_chunk_ids 없음(코퍼스에 명확한 답 없음, Hit@3 계산 제외) {len(no_gold)}건:")
    for r in no_gold:
        print(f"  - {r['turn2']}")
