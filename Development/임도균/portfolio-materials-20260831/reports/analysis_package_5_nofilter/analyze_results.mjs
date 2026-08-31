import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const csvPath = "./KDIC_검색방식_평균비교.csv";
const jsonPath = "./KDIC_검색평가_AI분석데이터.json";
const csvText = await fs.readFile(csvPath, "utf8");
const workbook = await Workbook.fromCSV(csvText, { sheetName: "평균비교" });
const csvInspect = await workbook.inspect({
  kind: "workbook,sheet,table",
  maxChars: 10000,
  tableMaxRows: 10,
  tableMaxCols: 15,
  tableMaxCellChars: 100,
});
console.log("--- ARTIFACT CSV INSPECT ---");
console.log(csvInspect.ndjson);

const data = JSON.parse(await fs.readFile(jsonPath, "utf8"));
const methods = data.methods;
const metricKeys = [
  "hit_at_3", "recall_at_5", "mrr_at_10", "map_at_10",
  "ndcg_at_5", "precision_at_5", "f1_at_5",
];
const domains = [...new Set(methods.flatMap((m) => m.by_domain.map((d) => d.group_value)))];

function parseArray(value) {
  if (Array.isArray(value)) return value.flat(Infinity).map(String);
  if (value == null || value === "") return [];
  try {
    const parsed = JSON.parse(value);
    return (Array.isArray(parsed) ? parsed.flat(Infinity) : [parsed]).map(String);
  } catch {
    return [];
  }
}

function mean(values) {
  const valid = values.filter((value) => Number.isFinite(value));
  return valid.length ? valid.reduce((a, b) => a + b, 0) / valid.length : null;
}

function round(value, digits = 4) {
  return value == null ? null : Number(value.toFixed(digits));
}

let randomState = 20260804;
function random() {
  randomState = (1664525 * randomState + 1013904223) >>> 0;
  return randomState / 2 ** 32;
}

function pairedBootstrapCI(values, iterations = 10000) {
  if (!values.length) return [null, null];
  const samples = [];
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    let total = 0;
    for (let index = 0; index < values.length; index += 1) {
      total += values[Math.floor(random() * values.length)];
    }
    samples.push(total / values.length);
  }
  samples.sort((a, b) => a - b);
  return [
    round(samples[Math.floor(iterations * 0.025)]),
    round(samples[Math.floor(iterations * 0.975)]),
  ];
}

const domainSummary = [];
for (const domain of domains) {
  const rows = methods.map((method) => {
    const stats = method.by_domain.find((item) => item.group_value === domain);
    return {
      method: method.name,
      question_count: stats.question_count,
      composite7: mean(metricKeys.map((key) => Number(stats[key]))),
      ...Object.fromEntries(metricKeys.map((key) => [key, Number(stats[key])])),
    };
  });
  const metricWinners = {};
  for (const key of [...metricKeys, "composite7"]) {
    const best = Math.max(...rows.map((row) => row[key]));
    metricWinners[key] = rows
      .filter((row) => Math.abs(row[key] - best) < 1e-12)
      .map((row) => row.method);
  }
  domainSummary.push({
    domain,
    question_count: rows[0].question_count,
    ranking_by_composite7: rows
      .sort((a, b) => b.composite7 - a.composite7)
      .map((row) => ({ method: row.method, composite7: round(row.composite7), ndcg_at_5: round(row.ndcg_at_5), hit_at_3: round(row.hit_at_3), recall_at_5: round(row.recall_at_5), mrr_at_10: round(row.mrr_at_10) })),
    metric_winners: metricWinners,
  });
}

const methodByName = Object.fromEntries(methods.map((method) => [method.name, method]));
const rerankerName = methods.find((m) => m.name.includes("Reranker"))?.name;
const structuredName = methods.find((m) => m.name.includes("structured"))?.name;
const denseName = methods.find((m) => m.name === "BGE-M3 Dense")?.name;

function questionMap(methodName) {
  return new Map(methodByName[methodName].question_results.map((row) => [row.evaluation_id, row]));
}

const rerankMap = questionMap(rerankerName);
const structuredMap = questionMap(structuredName);
const denseMap = questionMap(denseName);
const ids = [...rerankMap.keys()];

const pairwise = [];
for (const domain of domains) {
  const domainIds = ids.filter((id) => rerankMap.get(id).gold_business_function === domain);
  for (const [leftName, leftMap, rightName, rightMap] of [
    [rerankerName, rerankMap, structuredName, structuredMap],
    [structuredName, structuredMap, denseName, denseMap],
  ]) {
    const ndcgDiffs = domainIds.map((id) => Number(leftMap.get(id).ndcg_at_5) - Number(rightMap.get(id).ndcg_at_5));
    const hitDiffs = domainIds.map((id) => Number(leftMap.get(id).hit_at_3) - Number(rightMap.get(id).hit_at_3));
    const ndcgCI = pairedBootstrapCI(ndcgDiffs);
    const hitCI = pairedBootstrapCI(hitDiffs);
    pairwise.push({
      domain,
      left: leftName,
      right: rightName,
      count: domainIds.length,
      ndcg_mean_delta: round(mean(ndcgDiffs)),
      ndcg_delta_ci95_low: ndcgCI[0],
      ndcg_delta_ci95_high: ndcgCI[1],
      ndcg_better: ndcgDiffs.filter((x) => x > 1e-12).length,
      ndcg_worse: ndcgDiffs.filter((x) => x < -1e-12).length,
      ndcg_tied: ndcgDiffs.filter((x) => Math.abs(x) <= 1e-12).length,
      hit_gained: hitDiffs.filter((x) => x > 0).length,
      hit_lost: hitDiffs.filter((x) => x < 0).length,
      hit_mean_delta: round(mean(hitDiffs)),
      hit_delta_ci95_low: hitCI[0],
      hit_delta_ci95_high: hitCI[1],
    });
  }
}

const allMaps = methods.map((method) => ({ name: method.name, map: questionMap(method.name) }));
const commonFailures = ids.filter((id) => allMaps.every(({ map }) => Number(map.get(id).hit_at_3) === 0));
const commonSuccesses = ids.filter((id) => allMaps.every(({ map }) => Number(map.get(id).hit_at_3) === 1));
const rerankerOnlyRescues = ids.filter((id) => Number(rerankMap.get(id).hit_at_3) === 1 && allMaps.filter(({name}) => name !== rerankerName).every(({map}) => Number(map.get(id).hit_at_3) === 0));
const structuredBeatsRerankerHit = ids.filter((id) => Number(structuredMap.get(id).hit_at_3) === 1 && Number(rerankMap.get(id).hit_at_3) === 0);
const rerankerBeatsStructuredHit = ids.filter((id) => Number(rerankMap.get(id).hit_at_3) === 1 && Number(structuredMap.get(id).hit_at_3) === 0);

function questionInfo(id) {
  const row = rerankMap.get(id);
  return { id, domain: row.gold_business_function, complexity: row.question_complexity, question: row.question };
}

const requiredCounts = ids.map((id) => {
  const row = rerankMap.get(id);
  const primary = parseArray(row.gold_primary_chunk_ids);
  const supporting = parseArray(row.gold_supporting_chunk_ids);
  const required = [...new Set([...primary, ...supporting])];
  return { id, count: required.length, required };
});
const eligible = requiredCounts.filter((item) => item.count >= 2 && item.count <= 5);
const impossible = requiredCounts.filter((item) => item.count > 5);
const recomputedComplete = {};
for (const method of methods) {
  const map = questionMap(method.name);
  recomputedComplete[method.name] = round(mean(eligible.map(({id, required}) => {
    const retrieved = parseArray(map.get(id).retrieved_chunk_ids).slice(0, 5);
    return required.every((chunk) => retrieved.includes(chunk)) ? 1 : 0;
  })));
}

const evaluationWorkbook = await SpreadsheetFile.importXlsx(
  await FileBlob.load("C:/Users/임도균/Downloads/Evaluation_DataSet_v3.5.xlsx"),
);
const evaluationSheet = evaluationWorkbook.worksheets.getItem("평가데이터셋 v3");
const evaluationValues = evaluationSheet.getRange("A1:N140").values;
const evaluationHeaders = evaluationValues[0].map(String);
const evaluationIndex = Object.fromEntries(
  evaluationHeaders.map((header, index) => [header, index]),
);
const exactRequirements = new Map();
for (const row of evaluationValues.slice(1)) {
  const id = String(row[evaluationIndex.evaluation_id] ?? "").trim();
  if (!id) continue;
  const primary = parseArray(row[evaluationIndex.gold_primary_chunk_ids]);
  const supporting = parseArray(row[evaluationIndex.gold_supporting_chunk_ids]);
  const gold = parseArray(row[evaluationIndex.gold_chunk_ids]);
  const requirement = String(row[evaluationIndex.gold_evidence_requirement] ?? "").trim().toUpperCase();
  const required = requirement === "PRIMARY_ONLY"
    ? primary
    : requirement === "PRIMARY_PLUS_SUPPORT"
      ? [...new Set([...primary, ...supporting])]
      : gold;
  const multiFlag = String(row[evaluationIndex.multi_chunk_required] ?? "").trim().toUpperCase() === "Y";
  exactRequirements.set(id, { required, requirement, multiFlag });
}
const exactEligible = ids.filter((id) => {
  const item = exactRequirements.get(id);
  return item && (item.multiFlag || item.required.length > 1) && item.required.length >= 1 && item.required.length <= 5;
});
const exactImpossible = ids.filter((id) => {
  const item = exactRequirements.get(id);
  return item && (item.multiFlag || item.required.length > 1) && item.required.length > 5;
});
const exactComplete = {};
for (const method of methods) {
  const map = questionMap(method.name);
  exactComplete[method.name] = round(mean(exactEligible.map((id) => {
    const retrieved = parseArray(map.get(id).retrieved_chunk_ids).slice(0, 5);
    return exactRequirements.get(id).required.every((chunk) => retrieved.includes(chunk)) ? 1 : 0;
  })));
}

const chunksText = await fs.readFile(
  "./kdic_output_extracted/KDIC_output/processed/chunks.jsonl",
  "utf8",
);
const chunks = chunksText.split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line));
const chunkById = new Map(chunks.map((chunk) => [String(chunk.chunk_id), chunk]));

function allowedBusinessLabels(row) {
  const goldIds = parseArray(row.gold_chunk_ids);
  return new Set(
    goldIds
      .map((id) => chunkById.get(id)?.business_function)
      .filter(Boolean)
      .map(String),
  );
}

function contaminationForRows(rows) {
  const top1Off = [];
  const top5Fractions = [];
  const anyTop5Off = [];
  for (const row of rows) {
    const allowed = allowedBusinessLabels(row);
    const retrieved = parseArray(row.retrieved_chunk_ids).slice(0, 5);
    const labels = retrieved.map((id) => String(chunkById.get(id)?.business_function ?? "UNKNOWN"));
    const off = labels.map((label) => !allowed.has(label));
    top1Off.push(off[0] ? 1 : 0);
    top5Fractions.push(off.length ? off.filter(Boolean).length / off.length : 0);
    anyTop5Off.push(off.some(Boolean) ? 1 : 0);
  }
  return {
    question_count: rows.length,
    top1_off_gold_business_rate: round(mean(top1Off)),
    top5_off_gold_business_chunk_fraction: round(mean(top5Fractions)),
    questions_with_any_off_business_in_top5_rate: round(mean(anyTop5Off)),
  };
}

const contamination = methods.map((method) => ({
  method: method.name,
  overall: contaminationForRows(method.question_results),
  by_domain: domains.map((domain) => ({
    domain,
    ...contaminationForRows(
      method.question_results.filter((row) => row.gold_business_function === domain),
    ),
  })),
}));

const commonFailureDetails = commonFailures.map((id) => {
  const row = rerankMap.get(id);
  const allowed = [...allowedBusinessLabels(row)];
  const top3 = parseArray(row.retrieved_chunk_ids).slice(0, 3).map((chunkId) => ({
    chunk_id: chunkId,
    business_function: chunkById.get(chunkId)?.business_function ?? "UNKNOWN",
    title: chunkById.get(chunkId)?.title ?? "",
    section_title: chunkById.get(chunkId)?.section_title ?? "",
  }));
  return { ...questionInfo(id), allowed_business_functions: allowed, reranker_top3: top3 };
});
const rerankerFailures = ids.filter((id) => Number(rerankMap.get(id).hit_at_3) === 0);
const rerankerFailureAudit = {
  total: rerankerFailures.length,
  by_domain: Object.fromEntries(domains.map((domain) => [
    domain,
    rerankerFailures.filter((id) => rerankMap.get(id).gold_business_function === domain).length,
  ])),
  by_complexity: Object.fromEntries(
    [...new Set(rerankerFailures.map((id) => String(rerankMap.get(id).question_complexity || "(blank)")))]
      .sort()
      .map((complexity) => [
        complexity,
        rerankerFailures.filter((id) => String(rerankMap.get(id).question_complexity || "(blank)") === complexity).length,
      ]),
  ),
  top1_outside_allowed_business_count: rerankerFailures.filter((id) => {
    const row = rerankMap.get(id);
    const allowed = allowedBusinessLabels(row);
    const top1 = parseArray(row.retrieved_chunk_ids)[0];
    return !allowed.has(String(chunkById.get(top1)?.business_function ?? "UNKNOWN"));
  }).length,
};

const result = {
  methods: methods.map((m) => ({
    name: m.name,
    overall: m.overall,
    composite7_excluding_complete: round(mean(metricKeys.map((key) => Number(m.overall[key])))),
  })),
  domain_summary: domainSummary,
  pairwise,
  question_level: {
    common_failure_count: commonFailures.length,
    common_failures: commonFailures.map(questionInfo),
    common_success_count: commonSuccesses.length,
    reranker_only_rescue_count: rerankerOnlyRescues.length,
    reranker_only_rescues: rerankerOnlyRescues.map(questionInfo),
    structured_hit_reranker_miss_count: structuredBeatsRerankerHit.length,
    structured_hit_reranker_miss: structuredBeatsRerankerHit.map(questionInfo),
    reranker_hit_structured_miss_count: rerankerBeatsStructuredHit.length,
    reranker_hit_structured_miss: rerankerBeatsStructuredHit.map(questionInfo),
  },
  complete_metric_audit: {
    reported_applicable_count: methods[0].overall.complete_applicable_count,
    inferred_multi_evidence_2_to_5_count: eligible.length,
    impossible_over_5_count: impossible.length,
    impossible_ids: impossible.map((item) => ({id: item.id, required_count: item.count})),
    recomputed_complete_at_5_on_inferred_eligible: recomputedComplete,
    exact_applicable_count_from_v3_5: exactEligible.length,
    exact_impossible_count_from_v3_5: exactImpossible.length,
    exact_impossible_ids: exactImpossible.map((id) => ({
      id,
      requirement: exactRequirements.get(id).requirement,
      required_count: exactRequirements.get(id).required.length,
    })),
    exact_recomputed_complete_at_5: exactComplete,
  },
  business_contamination: contamination,
  common_failure_details: commonFailureDetails,
  reranker_failure_audit: rerankerFailureAudit,
  diagnostics: methods[0].dataset_diagnostics,
};

await fs.writeFile("./analysis_computed.json", JSON.stringify(result, null, 2), "utf8");
console.log("--- COMPUTED SUMMARY ---");
console.log(JSON.stringify(result, null, 2));
