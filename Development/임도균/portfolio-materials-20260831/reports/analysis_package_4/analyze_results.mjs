import fs from "node:fs/promises";

const payload = JSON.parse(
  await fs.readFile(
    "C:/Users/임도균/Documents/Codex/2026-07-22/d/work/analysis_package_4/KDIC_검색평가_AI분석데이터.json",
    "utf8",
  ),
);

const shortNames = ["Nori", "Dense", "Sparse", "Hybrid", "Reranker"];
const methods = payload.methods.map((method, index) => ({
  ...method,
  short: shortNames[index],
  byId: new Map(method.question_results.map((row) => [row.evaluation_id, row])),
}));
const ids = methods[0].question_results.map((row) => row.evaluation_id);

function number(value) {
  return Number(value ?? 0);
}

function parseList(value) {
  if (Array.isArray(value)) return value.flat(Infinity);
  try {
    const parsed = JSON.parse(String(value ?? "[]"));
    return Array.isArray(parsed) ? parsed.flat(Infinity) : [];
  } catch {
    return [];
  }
}

const overall = methods.map((method) => ({
  method: method.short,
  ...method.overall,
}));

const domainRows = [];
for (const method of methods) {
  for (const row of method.by_domain) {
    domainRows.push({ method: method.short, ...row });
  }
}

const commonFailures = ids
  .filter((id) => methods.every((method) => number(method.byId.get(id).hit_at_3) === 0))
  .map((id) => {
    const row = methods[0].byId.get(id);
    return {
      evaluation_id: id,
      domain: row.domain,
      question: row.question,
      gold: row.gold_chunk_ids,
      retrieved_top3: Object.fromEntries(
        methods.map((method) => [
          method.short,
          parseList(method.byId.get(id).retrieved_chunk_ids).slice(0, 3),
        ]),
      ),
    };
  });

const allSuccess = ids.filter((id) =>
  methods.every((method) => number(method.byId.get(id).hit_at_3) === 1),
);

function compare(leftIndex, rightIndex) {
  const left = methods[leftIndex];
  const right = methods[rightIndex];
  const changes = ids.map((id) => {
    const a = left.byId.get(id);
    const b = right.byId.get(id);
    return {
      evaluation_id: id,
      domain: a.domain,
      question: a.question,
      hit_delta: number(b.hit_at_3) - number(a.hit_at_3),
      recall_delta: number(b.recall_at_5) - number(a.recall_at_5),
      mrr_delta: number(b.mrr_at_10) - number(a.mrr_at_10),
      ndcg_delta: number(b.ndcg_at_5) - number(a.ndcg_at_5),
      left_top5: parseList(a.retrieved_chunk_ids).slice(0, 5),
      right_top5: parseList(b.retrieved_chunk_ids).slice(0, 5),
      gold: parseList(a.gold_chunk_ids),
    };
  });
  const improved = changes.filter(
    (row) => row.hit_delta > 0 || row.recall_delta > 1e-12 || row.mrr_delta > 1e-12,
  );
  const degraded = changes.filter(
    (row) => row.hit_delta < 0 || row.recall_delta < -1e-12 || row.mrr_delta < -1e-12,
  );
  return {
    pair: `${left.short}->${right.short}`,
    hit_gained: changes.filter((row) => row.hit_delta > 0).length,
    hit_lost: changes.filter((row) => row.hit_delta < 0).length,
    improved_count: improved.length,
    degraded_count: degraded.length,
    top_improvements: improved
      .sort(
        (a, b) =>
          b.hit_delta - a.hit_delta ||
          b.recall_delta - a.recall_delta ||
          b.mrr_delta - a.mrr_delta,
      )
      .slice(0, 12),
    top_degradations: degraded
      .sort(
        (a, b) =>
          a.hit_delta - b.hit_delta ||
          a.recall_delta - b.recall_delta ||
          a.mrr_delta - b.mrr_delta,
      )
      .slice(0, 12),
  };
}

const onlySuccess = Object.fromEntries(
  methods.map((method) => [
    method.short,
    ids
      .filter((id) => {
        const success = number(method.byId.get(id).hit_at_3) === 1;
        const othersFail = methods
          .filter((candidate) => candidate !== method)
          .every((candidate) => number(candidate.byId.get(id).hit_at_3) === 0);
        return success && othersFail;
      })
      .map((id) => {
        const row = method.byId.get(id);
        return { evaluation_id: id, domain: row.domain, question: row.question };
      }),
  ]),
);

const multiProxyIds = ids.filter((id) => {
  const count = parseList(methods[0].byId.get(id).gold_primary_chunk_ids).length;
  return count >= 2 && count <= 5;
});
const correctedCompleteProxy = methods.map((method) => ({
  method: method.short,
  applicable_count: multiProxyIds.length,
  complete_at_5:
    multiProxyIds.reduce(
      (sum, id) => sum + number(method.byId.get(id).complete_at_5),
      0,
    ) / multiProxyIds.length,
}));

const output = {
  metadata: {
    active_question_filter: payload.active_question_filter,
    question_count: ids.length,
    method_count: methods.length,
    complete_applicable_reported: methods.map((m) => ({
      method: m.short,
      count: m.overall.complete_applicable_count,
    })),
    multi_primary_proxy_count: multiProxyIds.length,
  },
  overall,
  domainRows,
  common_failure_count: commonFailures.length,
  commonFailures,
  all_success_count: allSuccess.length,
  onlySuccess,
  comparisons: [compare(1, 3), compare(3, 4), compare(1, 4)],
  correctedCompleteProxy,
};

await fs.writeFile(
  "C:/Users/임도균/Documents/Codex/2026-07-22/d/work/analysis_package_4/analysis_output.json",
  `${JSON.stringify(output, null, 2)}\n`,
  "utf8",
);

console.log(
  JSON.stringify(
    {
      metadata: output.metadata,
      overall,
      correctedCompleteProxy,
      common_failure_count: output.common_failure_count,
      all_success_count: output.all_success_count,
      onlySuccessCounts: Object.fromEntries(
        Object.entries(onlySuccess).map(([key, rows]) => [key, rows.length]),
      ),
      comparisons: output.comparisons.map((comparison) => ({
        pair: comparison.pair,
        hit_gained: comparison.hit_gained,
        hit_lost: comparison.hit_lost,
        improved_count: comparison.improved_count,
        degraded_count: comparison.degraded_count,
      })),
    },
    null,
    2,
  ),
);
