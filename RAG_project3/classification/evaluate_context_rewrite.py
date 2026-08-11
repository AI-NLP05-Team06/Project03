# [Phase5] 맥락반영 재작성 검증: turn2 원문만으로 검색했을 때 vs 재작성 후
# 검색했을 때, top-1 결과의 business_function이 의도한 도메인과 맞는지 비교한다.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from core.config import *
from core.integrity_check import *
from retrieval.reranker import hybrid_rerank_search
from classification.context_rewrite import rewrite_with_context
from classification.context_rewrite_testset import load_context_rewrite_testset

testset = load_context_rewrite_testset()
print(f"검증 대상 {len(testset)}건")

raw_correct = rewritten_correct = 0
rows = []

for i, pair in enumerate(testset, 1):
    raw_top1 = hybrid_rerank_search(pair["turn2"], top_k=1, business_function=None)[0]
    raw_business_function = raw_top1["chunk"].get("business_function")

    rewritten_question = rewrite_with_context(pair["turn2"], pair["turn1"])
    rewritten_top1 = hybrid_rerank_search(rewritten_question, top_k=1, business_function=None)[0]
    rewritten_business_function = rewritten_top1["chunk"].get("business_function")

    raw_hit = raw_business_function == pair["domain"]
    rewritten_hit = rewritten_business_function == pair["domain"]
    raw_correct += raw_hit
    rewritten_correct += rewritten_hit

    rows.append({
        "domain": pair["domain"],
        "turn1": pair["turn1"],
        "turn2": pair["turn2"],
        "rewritten": rewritten_question,
        "raw_hit": raw_hit,
        "rewritten_hit": rewritten_hit,
    })
    print(f"  {i}/{len(testset)} | raw={raw_hit}({raw_business_function}) | "
          f"rewritten={rewritten_hit}({rewritten_business_function}) | {pair['turn2']!r} -> {rewritten_question!r}")

print()
print(f"raw turn2만 검색: {raw_correct}/{len(testset)} ({raw_correct/len(testset):.1%})")
print(f"재작성 후 검색:   {rewritten_correct}/{len(testset)} ({rewritten_correct/len(testset):.1%})")

regressions = [r for r in rows if r["raw_hit"] and not r["rewritten_hit"]]
if regressions:
    print(f"\n재작성이 오히려 악화시킨 문항 {len(regressions)}건:")
    for r in regressions:
        print(f"  - {r['turn2']} -> {r['rewritten']}")
