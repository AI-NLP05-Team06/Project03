import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const inputPath = process.env.KDIC_INPUT_XLSX;
const outputPath = process.env.KDIC_OUTPUT_XLSX;
const chunksPath = process.env.KDIC_CHUNKS_JSONL;
const outputDir = path.dirname(outputPath);

if (!inputPath || !outputPath || !chunksPath) {
  throw new Error("KDIC_INPUT_XLSX, KDIC_OUTPUT_XLSX, KDIC_CHUNKS_JSONL 환경변수가 필요합니다.");
}

await fs.mkdir(outputDir, { recursive: true });

const chunkLines = (await fs.readFile(chunksPath, "utf8"))
  .replace(/^\uFEFF/, "")
  .split(/\r?\n/)
  .filter(Boolean);
const chunks = new Map(chunkLines.map((line) => {
  const item = JSON.parse(line);
  return [item.chunk_id, item];
}));

const inputBlob = await FileBlob.load(inputPath);
const workbook = await SpreadsheetFile.importXlsx(inputBlob);
const main = workbook.worksheets.getItem("검색평가용");

const originalPreview = await workbook.render({
  sheetName: "검색평가용",
  range: "A1:O24",
  scale: 1.2,
  format: "png",
});
await fs.writeFile(
  path.join(outputDir, "original_preview.png"),
  new Uint8Array(await originalPreview.arrayBuffer()),
);

const originalInspect = await workbook.inspect({
  kind: "computedStyle",
  sheetId: "검색평가용",
  range: "A1:O4",
  maxChars: 4000,
});
console.log("ORIGINAL_STYLE", originalInspect.ndjson);

const used = main.getUsedRange(true);
const values = used.values;
const headers = values[0].map((v) => String(v ?? "").trim());
const index = Object.fromEntries(headers.map((h, i) => [h, i]));

const requiredHeaders = [
  "검색평가대상",
  "evaluation_id",
  "예상질문",
  "도메인",
  "gold_business_function",
  "gold_source_urls",
  "gold_primary_chunk_ids",
  "gold_supporting_chunk_ids",
  "gold_chunk_ids",
  "gold_evidence_requirement",
  "multi_chunk_required",
  "gold_review_status",
];
for (const header of requiredHeaders) {
  if (!(header in index)) throw new Error(`필수 칼럼 누락: ${header}`);
}

const newHeaders = [
  "retrieval_evaluation_applicable",
  "gold_answerable",
  "gold_ambiguity_status",
  "gold_expected_action",
  "gold_clarification_question",
  "gold_clarification_options",
  "allowed_business_functions",
  "gold_alternative_chunk_ids",
  "related_chunk_ids",
  "gold_document_ids",
  "gold_section_titles",
  "gold_review_note",
  "gold_reviewed_by",
  "gold_reviewed_at",
  "dataset_version",
];

const startCol = headers.length;
main.getRangeByIndexes(0, startCol, 1, newHeaders.length).values = [newHeaders];
newHeaders.forEach((header, offset) => {
  index[header] = startCol + offset;
});

const jsonList = (items) => JSON.stringify([...new Set(items ?? [])]);
const parseList = (value) => {
  if (Array.isArray(value)) return value;
  const text = String(value ?? "").trim();
  if (!text) return [];
  try {
    const parsed = JSON.parse(text.replaceAll("'", '"'));
    return Array.isArray(parsed) ? parsed : [String(parsed)];
  } catch {
    return [];
  }
};

const domainLabels = {
  deposit_protection: "예금자보호제도",
  deposit_insurance_payout: "예금보험금",
  unclaimed_funds: "고객 미수령금",
  mistaken_transfer: "착오송금 반환지원",
  debt_adjustment: "채무조정",
  hidden_assets_report: "은닉재산 신고",
};

const excluded = new Map([
  ["DP-Q010", {
    ambiguity: "needs_clarification",
    action: "ask_clarification",
    question: "어느 금융권역 또는 어떤 금융상품의 보호 여부를 확인하고 싶으신가요?",
    options: ["은행 예금", "보험상품", "증권·투자상품", "퇴직연금"],
    note: "질문 범위가 전체 금융상품으로 넓고 기존 Gold는 보험회사 일부만 포함하여 검색평가에서 제외.",
  }],
  ["DP-Q012", {
    ambiguity: "needs_clarification",
    action: "ask_clarification",
    question: "보호 여부를 확인할 금융회사 종류나 상품명을 알려주시겠어요?",
    options: ["은행", "보험회사", "증권사", "저축은행"],
    note: "은행·보험 청크만으로 전체 보호·비보호 상품 구분을 대표할 수 없어 검색평가에서 제외.",
  }],
  ["BI-Q020", {
    ambiguity: "needs_clarification",
    action: "ask_clarification",
    question: "누가 예금보험금을 신청하나요?",
    options: ["본인", "대리인", "미성년자 친권자·후견인", "상속인"],
    note: "신청자 역할에 따라 서류가 달라 한 번에 모든 청크를 정답으로 요구하지 않도록 제외.",
  }],
  ["UN-Q133", {
    ambiguity: "needs_clarification",
    action: "ask_clarification",
    question: "어떤 종류의 미수령금에 대한 세금인지 알려주시겠어요?",
    options: ["예금보험금", "가지급금", "개산지급금 정산금", "파산배당금"],
    note: "예금보험금 세금 근거를 일반 미수령금 전체에 적용할 수 없어 직접 답변 불가.",
  }],
  ["UN-Q134", {
    ambiguity: "needs_clarification",
    action: "ask_clarification",
    question: "신청하려는 미수령금 종류와 본인인증 가능 여부를 알려주시겠어요?",
    options: ["예금보험금", "가지급금", "파산배당금", "종류를 모름"],
    note: "재외국민 관련 근거가 다른 업무 FAQ에 있고 모든 미수령금에 공통 적용되는지 불명확.",
  }],
  ["UN-Q135", {
    ambiguity: "clear",
    action: "answer_with_limitation",
    question: "",
    options: [],
    note: "법인 명의 미수령금 조회·신청 가능 여부를 직접 설명하는 청크가 없어 문의 경로 안내 필요.",
  }],
  ["UN-Q136", {
    ambiguity: "needs_clarification",
    action: "ask_clarification",
    question: "어떤 종류의 미수령금 이자 계산을 확인하고 싶으신가요?",
    options: ["예금보험금", "가지급금", "개산지급금 정산금", "파산배당금"],
    note: "기존 근거는 예금보험금·가지급금에 한정되어 일반 미수령금 질문에 바로 적용할 수 없음.",
  }],
  ["UN-Q137", {
    ambiguity: "needs_clarification",
    action: "ask_clarification",
    question: "외화로 보유했던 금액이 예금보험금인지 다른 미수령금인지 알려주시겠어요?",
    options: ["예금보험금", "파산배당금", "기타 미수령금", "종류를 모름"],
    note: "환율 청크는 예금보험금에 한정되어 미수령금 종류 확인이 필요.",
  }],
  ["UN-Q138", {
    ambiguity: "needs_clarification",
    action: "ask_clarification",
    question: "확인하려는 금액이 예금보험금, 파산배당금, 가지급금 중 무엇인가요?",
    options: ["예금보험금", "파산배당금", "가지급금", "종류를 모름"],
    note: "금액 종류에 따라 지급 주체와 절차가 달라 기존 6개 Gold를 일괄 정답으로 사용할 수 없음.",
  }],
  ["UN-Q139", {
    ambiguity: "clear",
    action: "answer_with_limitation",
    question: "",
    options: [],
    note: "공동명의자 단독 신청 가능 여부를 직접 설명하는 청크가 없어 공식 문의 안내가 필요.",
  }],
  ["UN-Q141", {
    ambiguity: "needs_clarification",
    action: "ask_clarification",
    question: "조회된 미수령금의 종류가 무엇인가요?",
    options: ["예금보험금", "파산배당금", "가지급금", "개산지급금 정산금"],
    note: "예금보험금 입금기간을 모든 미수령금에 적용할 수 없어 종류 확인 필요.",
  }],
  ["UN-Q142", {
    ambiguity: "needs_clarification",
    action: "ask_clarification",
    question: "부모님 명의의 어떤 미수령금을 대신 신청하려는지와 위임 가능 여부를 알려주시겠어요?",
    options: ["예금보험금", "파산배당금", "상속인 조회", "종류를 모름"],
    note: "다른 업무의 대리인 서류 청크 6개가 혼입되어 기존 Gold를 제거.",
  }],
  ["MT-Q043", {
    ambiguity: "clear",
    action: "answer_with_limitation",
    question: "",
    options: [],
    note: "온라인과 방문 신청의 처리속도를 비교하는 근거가 없으며 전체 처리기간 청크만 존재.",
  }],
  ["MT-Q047", {
    ambiguity: "clear",
    action: "show_faq_list",
    question: "",
    options: [],
    answerable: true,
    keepGold: true,
    note: "FAQ 10개 목록 제공 기능 문항. Top-5 검색평가가 아닌 UI·기능 테스트로 분리.",
  }],
  ["MT-Q048", {
    ambiguity: "clear",
    action: "show_faq_list",
    question: "",
    options: [],
    answerable: true,
    keepGold: true,
    note: "FAQ 10개 목록 제공 기능 문항. Top-5 검색평가가 아닌 UI·기능 테스트로 분리.",
  }],
  ["MT-Q108", {
    ambiguity: "needs_clarification",
    action: "answer_with_limitation",
    question: "예금보호가 아니라 잘못 보낸 돈을 돌려받는 절차를 안내해 드릴까요?",
    options: ["착오송금 반환지원 안내", "예금자보호제도 설명"],
    note: "예금보호와 착오송금을 혼동한 복수 업무 문항으로 단일 업무 검색평가에서 제외.",
  }],
]);

const directEdits = new Map([
  ["DP-Q007", {
    primary: ["DP-003_chunk_000"],
    support: ["DP-003_chunk_002"],
    evidence: "PRIMARY_ONLY",
    multi: "N",
    alternatives: ["DP-003_chunk_002"],
    note: "금융회사 종류를 직접 설명하는 개요 청크를 Primary로 변경.",
  }],
  ["DP-Q008", {
    primary: ["DP-003_chunk_000"],
    support: ["DP-003_chunk_002"],
    evidence: "PRIMARY_ONLY",
    multi: "N",
    alternatives: ["DP-003_chunk_002"],
    note: "상호저축은행 포함 여부를 직접 확인할 수 있는 개요 청크로 정리.",
  }],
  ["DP-Q009", {
    primary: ["DP-003_chunk_002"],
    support: [],
    evidence: "PRIMARY_ONLY",
    multi: "N",
    alternatives: ["DP-003_chunk_000"],
    note: "증권사 포함 여부는 표와 개요 청크 모두 정답 가능하여 대체 Gold 추가.",
  }],
  ["DP-Q011", {
    primary: ["DP-004_chunk_005"],
    support: [],
    evidence: "PRIMARY_ONLY",
    multi: "N",
    note: "무관한 보험회사 비보호 상품 청크 제거.",
  }],
  ["DP-Q014", {
    primary: ["DP-005_chunk_009"],
    support: [],
    evidence: "PRIMARY_ONLY",
    multi: "N",
    note: "질문과 무관한 후순위채권 청크 제거.",
  }],
  ["BI-Q024", {
    questionText: "예금보험금 신청 후 입금까지 얼마나 걸리나요?",
    primary: ["BI-002_chunk_002"],
    support: [],
    evidence: "PRIMARY_ONLY",
    multi: "N",
    note: "Gold가 설명하는 범위에 맞게 질문을 '신청 후 입금기간'으로 구체화.",
  }],
  ["BI-Q036", {
    primary: ["BI-002_chunk_002"],
    support: ["BI-003_chunk_008"],
    evidence: "PRIMARY_ONLY",
    multi: "N",
    note: "온라인·방문 신청을 직접 설명하는 청크를 Primary로 단순화.",
  }],
  ["MT-Q049", {
    primary: ["MT-006_chunk_000"],
    support: ["MT-007_chunk_000"],
    evidence: "PRIMARY_ONLY",
    multi: "N",
    note: "기한은 유의사항 청크만 필수이며 서류 청크는 보조 근거로 정리.",
  }],
  ["MT-Q064", {
    primary: ["MT-013_chunk_000"],
    support: [],
    evidence: "PRIMARY_ONLY",
    multi: "N",
    note: "금액 질문에서 제외 대상 청크를 제거.",
  }],
  ["MT-Q066", {
    primary: ["MT-013_chunk_000"],
    support: [],
    evidence: "PRIMARY_ONLY",
    multi: "N",
    note: "비용률 예시가 실제로 포함된 신청 대상 청크로 교체.",
  }],
  ["DA-Q075", {
    primary: ["DA-002_chunk_002"],
    support: ["DA-002_chunk_001"],
    evidence: "PRIMARY_ONLY",
    multi: "N",
    note: "상담 신청 방법 청크를 Primary로, 제도 설명을 Supporting으로 정리.",
  }],
  ["DA-Q082", {
    primary: ["DA-005_chunk_000", "DA-005_chunk_005"],
    support: ["DA-005_chunk_006"],
    evidence: "PRIMARY_ONLY",
    multi: "Y",
    note: "면책의 의미와 자동 면책이 아님을 설명하는 두 청크를 필수 근거로 정리.",
  }],
  ["HA-Q094", {
    primary: ["HP-001_chunk_000", "HP-003_chunk_000"],
    support: ["HP-003_chunk_001"],
    evidence: "PRIMARY_ONLY",
    multi: "Y",
    note: "은닉재산 신고와 불법행위 신고 양쪽 정의 청크를 모두 연결.",
  }],
]);

const rowById = new Map();
for (let r = 1; r < values.length; r += 1) {
  const id = String(values[r][index.evaluation_id] ?? "").trim();
  if (id) rowById.set(id, r);
}

const getCellValue = (r, header) => main.getCell(r, index[header]).values?.[0]?.[0];
const setCellValue = (r, header, value) => {
  main.getCell(r, index[header]).values = [[value]];
};

const selectedIds = [];
for (let r = 1; r < values.length; r += 1) {
  if (String(values[r][index["검색평가대상"]] ?? "").trim().toUpperCase() !== "Y") continue;
  const id = String(values[r][index.evaluation_id] ?? "").trim();
  selectedIds.push(id);
  const business = String(values[r][index.gold_business_function] ?? "").trim();
  setCellValue(r, "도메인", domainLabels[business] ?? values[r][index["도메인"]]);
  setCellValue(r, "retrieval_evaluation_applicable", "Y");
  setCellValue(r, "gold_answerable", true);
  setCellValue(r, "gold_ambiguity_status", "clear");
  setCellValue(r, "gold_expected_action", "answer");
  setCellValue(r, "gold_clarification_question", "");
  setCellValue(r, "gold_clarification_options", "[]");
  setCellValue(r, "allowed_business_functions", jsonList(business ? [business] : []));
  setCellValue(r, "gold_alternative_chunk_ids", "[]");
  setCellValue(r, "related_chunk_ids", "[]");
  setCellValue(r, "gold_review_note", "");
  setCellValue(r, "gold_reviewed_by", "Codex 1차 검수");
  setCellValue(r, "gold_reviewed_at", "2026-07-30");
  setCellValue(r, "dataset_version", "v1.1_gold_review");
}

for (const [id, edit] of directEdits) {
  const r = rowById.get(id);
  if (r == null) throw new Error(`수정 대상 ID 없음: ${id}`);
  if (edit.questionText) setCellValue(r, "예상질문", edit.questionText);
  setCellValue(r, "gold_primary_chunk_ids", jsonList(edit.primary));
  setCellValue(r, "gold_supporting_chunk_ids", jsonList(edit.support));
  setCellValue(r, "gold_chunk_ids", jsonList([...edit.primary, ...edit.support]));
  setCellValue(r, "gold_evidence_requirement", edit.evidence);
  setCellValue(r, "multi_chunk_required", edit.multi);
  setCellValue(r, "gold_alternative_chunk_ids", jsonList(edit.alternatives ?? []));
  setCellValue(r, "gold_review_note", edit.note);
  setCellValue(r, "gold_review_status", "codex_reviewed_pending_human");
}

for (const [id, rule] of excluded) {
  const r = rowById.get(id);
  if (r == null) throw new Error(`제외 대상 ID 없음: ${id}`);
  const originalGold = parseList(getCellValue(r, "gold_chunk_ids"));
  setCellValue(r, "검색평가대상", "N");
  setCellValue(r, "retrieval_evaluation_applicable", "N");
  setCellValue(r, "gold_answerable", rule.answerable ?? false);
  setCellValue(r, "gold_ambiguity_status", rule.ambiguity);
  setCellValue(r, "gold_expected_action", rule.action);
  setCellValue(r, "gold_clarification_question", rule.question);
  setCellValue(r, "gold_clarification_options", jsonList(rule.options));
  setCellValue(r, "related_chunk_ids", jsonList(originalGold));
  setCellValue(r, "gold_review_note", rule.note);
  setCellValue(r, "gold_review_status", "codex_reviewed_pending_human");
  if (id === "MT-Q108") {
    setCellValue(r, "allowed_business_functions", jsonList(["mistaken_transfer", "deposit_protection"]));
  }
  if (!rule.keepGold) {
    setCellValue(r, "gold_source_urls", "[]");
    setCellValue(r, "gold_primary_chunk_ids", "[]");
    setCellValue(r, "gold_supporting_chunk_ids", "[]");
    setCellValue(r, "gold_chunk_ids", "[]");
    setCellValue(r, "gold_evidence_requirement", "NOT_APPLICABLE");
    setCellValue(r, "multi_chunk_required", "N");
  }
}

// Gold에서 URL·문서·부제목을 자동 파생해 수기 불일치를 제거한다.
for (const id of selectedIds) {
  const r = rowById.get(id);
  const goldIds = parseList(getCellValue(r, "gold_chunk_ids"));
  const goldChunks = goldIds.map((chunkId) => chunks.get(chunkId)).filter(Boolean);
  const urls = goldChunks.map((c) => c.source_url).filter(Boolean);
  const docs = goldChunks.map((c) => c.parent_doc_id || c.document_id).filter(Boolean);
  const sections = goldChunks.map((c) => c.section_title || c.title).filter(Boolean);
  setCellValue(r, "gold_source_urls", jsonList(urls));
  setCellValue(r, "gold_document_ids", jsonList(docs));
  setCellValue(r, "gold_section_titles", jsonList(sections));
}

// 새 칼럼 서식과 검수용 입력 규칙.
const lastRow = values.length;
const newHeaderRange = main.getRangeByIndexes(0, startCol, 1, newHeaders.length);
newHeaderRange.format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF", size: 10 },
  wrapText: true,
  verticalAlignment: "center",
  horizontalAlignment: "center",
  borders: { preset: "outside", style: "thin", color: "#AAB7C4" },
};
newHeaderRange.format.rowHeight = 36;

const newDataRange = main.getRangeByIndexes(1, startCol, lastRow - 1, newHeaders.length);
newDataRange.format = {
  font: { size: 9, color: "#1F2937" },
  verticalAlignment: "top",
  wrapText: true,
};
newDataRange.format.borders = {
  insideHorizontal: { style: "thin", color: "#E5E7EB" },
};

const widths = {
  retrieval_evaluation_applicable: 14,
  gold_answerable: 12,
  gold_ambiguity_status: 16,
  gold_expected_action: 18,
  gold_clarification_question: 34,
  gold_clarification_options: 32,
  allowed_business_functions: 24,
  gold_alternative_chunk_ids: 28,
  related_chunk_ids: 30,
  gold_document_ids: 24,
  gold_section_titles: 30,
  gold_review_note: 46,
  gold_reviewed_by: 16,
  gold_reviewed_at: 14,
  dataset_version: 18,
};
for (const [header, width] of Object.entries(widths)) {
  main.getRangeByIndexes(0, index[header], lastRow, 1).format.columnWidth = width;
}

main.getRangeByIndexes(1, index.retrieval_evaluation_applicable, lastRow - 1, 1).dataValidation = {
  rule: { type: "list", values: ["Y", "N"] },
};
main.getRangeByIndexes(1, index.gold_ambiguity_status, lastRow - 1, 1).dataValidation = {
  rule: { type: "list", values: ["clear", "needs_clarification", "out_of_scope", "no_intent"] },
};
main.getRangeByIndexes(1, index.gold_expected_action, lastRow - 1, 1).dataValidation = {
  rule: {
    type: "list",
    values: ["answer", "ask_clarification", "answer_with_limitation", "redirect_out_of_scope", "request_intent", "show_faq_list"],
  },
};
main.freezePanes.freezeRows(1);
main.freezePanes.freezeColumns(4);

// 검수 요약 시트.
let summary;
try {
  summary = workbook.worksheets.getItem("Gold 검수 요약");
  summary.getUsedRange()?.clear({ applyTo: "all" });
} catch {
  summary = workbook.worksheets.add("Gold 검수 요약");
}
summary.showGridLines = false;
summary.getRange("A1:H1").merge();
summary.getRange("A1").values = [["평가데이터셋 Gold 1차 검수 결과"]];
summary.getRange("A1:H1").format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF", size: 18 },
  horizontalAlignment: "left",
  verticalAlignment: "center",
};
summary.getRange("A1:H1").format.rowHeight = 34;

const totalSelected = selectedIds.length;
const excludedCount = excluded.size;
const modifiedCount = new Set([...excluded.keys(), ...directEdits.keys()]).size;
summary.getRange("A3:F5").values = [
  ["원래 검색평가 문항", totalSelected, "수정 후 검색평가 문항", totalSelected - excludedCount, "검색평가 제외", excludedCount],
  ["Gold 직접 수정", directEdits.size, "수정·정책검토 전체", modifiedCount, "원본 보존", "예"],
  ["검수 버전", "v1.1_gold_review", "검수일", "2026-07-30", "상태", "사람 최종 검수 필요"],
];
summary.getRange("A3:F5").format = {
  fill: "#EEF4FB",
  font: { bold: true, color: "#17365D", size: 11 },
  wrapText: true,
  verticalAlignment: "center",
  borders: { preset: "outside", style: "thin", color: "#9FBAD0" },
};

const detailHeaders = ["evaluation_id", "처리구분", "검색평가", "답변가능", "모호성", "예상행동", "검수 내용", "사람 최종검수"];
summary.getRange("A7:H7").values = [detailHeaders];
summary.getRange("A7:H7").format = {
  fill: "#2F75B5",
  font: { bold: true, color: "#FFFFFF", size: 10 },
  horizontalAlignment: "center",
  verticalAlignment: "center",
};

const detailRows = [];
for (const id of [...new Set([...excluded.keys(), ...directEdits.keys()])].sort()) {
  const r = rowById.get(id);
  detailRows.push([
    id,
    excluded.has(id) ? "정책·평가분리" : "Gold 수정",
    String(getCellValue(r, "retrieval_evaluation_applicable") ?? ""),
    String(getCellValue(r, "gold_answerable") ?? ""),
    String(getCellValue(r, "gold_ambiguity_status") ?? ""),
    String(getCellValue(r, "gold_expected_action") ?? ""),
    String(getCellValue(r, "gold_review_note") ?? ""),
    "필요",
  ]);
}
summary.getRangeByIndexes(7, 0, detailRows.length, detailHeaders.length).values = detailRows;
summary.getRangeByIndexes(7, 0, detailRows.length, detailHeaders.length).format = {
  font: { size: 10, color: "#1F2937" },
  wrapText: true,
  verticalAlignment: "top",
};
summary.getRangeByIndexes(7, 0, detailRows.length, detailHeaders.length).format.borders = {
  insideHorizontal: { style: "thin", color: "#D9E2F3" },
};
summary.getRange("A:A").format.columnWidth = 16;
summary.getRange("B:B").format.columnWidth = 16;
summary.getRange("C:F").format.columnWidth = 15;
summary.getRange("G:G").format.columnWidth = 58;
summary.getRange("H:H").format.columnWidth = 16;
summary.freezePanes.freezeRows(7);

// 지표·칼럼 안내 시트에 신규 칼럼 사전을 덧붙인다.
const guide = workbook.worksheets.getItem("지표·칼럼 안내");
const guideUsed = guide.getUsedRange(true);
const guideStartRow = guideUsed.values.length + 2;
guide.getRangeByIndexes(guideStartRow, 0, 1, 5).values = [[
  "추가 검수 칼럼", "의미", "값 예시", "평가 사용", "운영 규칙",
]];
guide.getRangeByIndexes(guideStartRow, 0, 1, 5).format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF" },
};
const guideRows = [
  ["retrieval_evaluation_applicable", "검색 지표 계산 대상 여부", "Y / N", "필수", "N인 문항은 검색 성능 평균에서 제외"],
  ["gold_answerable", "현재 질문과 코퍼스로 직접 답변 가능한지", "true / false", "정책", "false이면 Gold를 억지로 지정하지 않음"],
  ["gold_ambiguity_status", "질문의 추가 확인 필요 상태", "clear / needs_clarification", "정책", "모호성과 질문 복잡도를 분리"],
  ["gold_expected_action", "챗봇이 수행해야 할 행동", "answer / ask_clarification", "정책", "검색평가와 대화정책 평가를 분리"],
  ["gold_clarification_question", "추가로 물어볼 질문", "어떤 종류의 미수령금인가요?", "정책", "명확한 질문은 빈칸"],
  ["gold_clarification_options", "사용자에게 제공할 선택지", "[\"예금보험금\",\"파산배당금\"]", "정책", "JSON 배열"],
  ["allowed_business_functions", "검색을 허용할 업무 범위", "[\"unclaimed_funds\"]", "필터", "복수 업무 문항은 배열에 함께 기록"],
  ["gold_alternative_chunk_ids", "동등한 정답으로 인정할 대체 청크", "[\"DP-003_chunk_000\"]", "보조지표", "기존 평가기가 지원하면 정답 집합에 포함"],
  ["related_chunk_ids", "직접 정답은 아니지만 정책 판단에 참고한 청크", "[\"DP-006_chunk_013\"]", "검수", "Gold와 구분"],
  ["gold_document_ids", "Gold 문서 ID", "[\"DP-003\"]", "검수", "Gold 청크에서 자동 생성"],
  ["gold_section_titles", "Gold 부제목", "[\"보호대상금융회사\"]", "검수", "Gold 청크에서 자동 생성"],
  ["gold_review_note", "수정 및 제외 사유", "직접 근거 없음", "검수", "사람 최종 검수 시 확인"],
];
guide.getRangeByIndexes(guideStartRow + 1, 0, guideRows.length, 5).values = guideRows;
guide.getRangeByIndexes(guideStartRow + 1, 0, guideRows.length, 5).format = {
  wrapText: true,
  verticalAlignment: "top",
  font: { size: 10 },
};

// 원본 안내 시트의 한글 시트명 참조 수식을 Excel 규칙에 맞게 보정한다.
guide.getRange("B37").formulas = [["=COUNTA('검색평가용'!$B$2:$B$152)"]];
guide.getRange("B38").formulas = [["=COUNTIF('검색평가용'!$A$2:$A$152,\"Y\")"]];

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);

const finalMainPreview = await workbook.render({
  sheetName: "검색평가용",
  range: "A1:AD22",
  scale: 0.8,
  format: "png",
});
await fs.writeFile(
  path.join(outputDir, "reviewed_main_preview.png"),
  new Uint8Array(await finalMainPreview.arrayBuffer()),
);
const finalSummaryPreview = await workbook.render({
  sheetName: "Gold 검수 요약",
  range: `A1:H${Math.min(40, 8 + detailRows.length)}`,
  scale: 1.1,
  format: "png",
});
await fs.writeFile(
  path.join(outputDir, "review_summary_preview.png"),
  new Uint8Array(await finalSummaryPreview.arrayBuffer()),
);

const finalInspect = await workbook.inspect({
  kind: "table",
  range: "Gold 검수 요약!A1:H20",
  include: "values,formulas",
  tableMaxRows: 20,
  tableMaxCols: 8,
  maxChars: 7000,
});
console.log("FINAL_SUMMARY", finalInspect.ndjson);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
console.log("FORMULA_ERRORS", errors.ndjson);
console.log(JSON.stringify({
  outputPath,
  totalSelected,
  retrievalApplicable: totalSelected - excludedCount,
  excludedCount,
  directGoldEdits: directEdits.size,
  modifiedCount,
  selectedRowsWithMissingGold: selectedIds.filter((id) => {
    const r = rowById.get(id);
    return String(getCellValue(r, "retrieval_evaluation_applicable")) === "Y"
      && parseList(getCellValue(r, "gold_chunk_ids")).length === 0;
  }),
}, null, 2));
