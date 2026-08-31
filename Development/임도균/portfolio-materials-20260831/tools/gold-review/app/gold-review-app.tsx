"use client";

import {
  AlertTriangle,
  ArrowLeft,
  ArrowRight,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Clipboard,
  Copy,
  Download,
  FileJson,
  FileSpreadsheet,
  Filter,
  FolderOpen,
  GitMerge,
  Link2,
  ListPlus,
  Plus,
  RotateCcw,
  Save,
  Search,
  Send,
  Trash2,
  Users,
  X,
} from "lucide-react";
import { ChangeEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";

type Row = Record<string, string>;
type Chunk = {
  chunk_id: string;
  parent_doc_id?: string;
  document_id?: string;
  title?: string;
  section_title?: string;
  business_function?: string;
  target_business_function?: string;
  source_url?: string;
  content?: string;
  chunk_type?: string;
  chunk_index?: number;
};

type TeamPatchChange = {
  questionId: string;
  isNew: boolean;
  base: Row;
  values: Row;
  snapshot: Row;
};

type TeamPatch = {
  format: "kdic-gold-review-patch-v1";
  createdAt: string;
  createdBy: string;
  changes: TeamPatchChange[];
};

declare global {
  interface Window {
    __KDIC_EVALUATION_CSV__?: string;
    __KDIC_CHUNKS_JSONL__?: string;
  }
}

const FIELD = {
  id: "질문ID",
  question: "예상질문",
  business: "업무라벨",
  intentGroup: "의도그룹",
  intentMain: "상세의도_주",
  intentSub: "상세의도_부",
  responsePolicy: "응답정책",
  importance: "중요도",
  note: "비고/기대동작",
  complexity: "question_complexity",
  goldDocs: "gold_document_ids",
  goldSections: "gold_section_titles",
  goldChunks: "gold_chunk_ids",
  goldUrls: "gold_source_urls",
  mapping: "gold_mapping_status",
  reviewRequired: "gold_review_required",
  mappingNote: "gold_mapping_note",
  answerable: "gold_answerable_status",
  reviewStatus: "gold_review_status",
  reviewer: "gold_reviewed_by(검수자 이름)",
  reviewedAt: "gold_reviewed_at(검수 날짜)",
  answerSummary: "gold_answer_summary(정답 요약 텍스트)",
} as const;

const REVIEW_LABELS: Record<string, string> = {
  auto_approved: "자동 승인",
  approved: "검수 완료",
  pending: "검수 대기",
  needs_revision: "수정 필요",
  "": "미검수",
};

function parseCsv(text: string): { headers: string[]; rows: Row[] } {
  const source = text.replace(/^\uFEFF/, "");
  const matrix: string[][] = [];
  let row: string[] = [];
  let cell = "";
  let quoted = false;

  for (let i = 0; i < source.length; i += 1) {
    const char = source[i];
    if (quoted) {
      if (char === '"' && source[i + 1] === '"') {
        cell += '"';
        i += 1;
      } else if (char === '"') {
        quoted = false;
      } else {
        cell += char;
      }
    } else if (char === '"') {
      quoted = true;
    } else if (char === ",") {
      row.push(cell);
      cell = "";
    } else if (char === "\n") {
      row.push(cell.replace(/\r$/, ""));
      matrix.push(row);
      row = [];
      cell = "";
    } else {
      cell += char;
    }
  }
  if (cell.length || row.length) {
    row.push(cell.replace(/\r$/, ""));
    matrix.push(row);
  }
  const headers = matrix[0] ?? [];
  return {
    headers,
    rows: matrix.slice(1).filter((values) => values.some(Boolean)).map((values) =>
      Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""])),
    ),
  };
}

function parseJsonList(value = ""): string[] {
  if (!value.trim()) return [];
  try {
    const parsed = JSON.parse(value);
    return Array.isArray(parsed) ? parsed.map(String) : [String(parsed)];
  } catch {
    return value.split(/[,\n]/).map((item) => item.trim()).filter(Boolean);
  }
}

function jsonList(items: string[]) {
  return JSON.stringify([...new Set(items.filter(Boolean))]);
}

function csvEscape(value: string) {
  return `"${String(value ?? "").replaceAll('"', '""')}"`;
}

function exportCsv(headers: string[], rows: Row[]) {
  return "\uFEFF" + [headers, ...rows.map((row) => headers.map((header) => row[header] ?? ""))]
    .map((line) => line.map(csvEscape).join(","))
    .join("\r\n");
}

function download(name: string, contents: string, mime: string) {
  const blob = new Blob([contents], { type: mime });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  URL.revokeObjectURL(url);
}

function today() {
  return new Date().toISOString().slice(0, 10);
}

function nextQuestionId(rows: Row[]) {
  const max = Math.max(0, ...rows.map((row) => Number((row[FIELD.id] || "").match(/\d+/)?.[0] ?? 0)));
  return `Q${String(max + 1).padStart(3, "0")}`;
}

function statusTone(status: string) {
  if (status === "approved" || status === "auto_approved") return "good";
  if (status === "needs_revision") return "bad";
  return "warn";
}

export default function GoldReviewApp() {
  const [headers, setHeaders] = useState<string[]>([]);
  const [rows, setRows] = useState<Row[]>([]);
  const [baselineRows, setBaselineRows] = useState<Row[]>([]);
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [query, setQuery] = useState("");
  const [businessFilter, setBusinessFilter] = useState("전체");
  const [statusFilter, setStatusFilter] = useState("전체");
  const [onlyIssues, setOnlyIssues] = useState(false);
  const [expandedGold, setExpandedGold] = useState<Set<string>>(new Set());
  const [goldOpen, setGoldOpen] = useState(true);
  const [compareOpen, setCompareOpen] = useState(true);
  const [editOpen, setEditOpen] = useState(false);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [compareQuery, setCompareQuery] = useState("");
  const [chunkScope, setChunkScope] = useState<"document" | "all">("document");
  const [chunkBusinessFilter, setChunkBusinessFilter] = useState("전체");
  const [pickedChunks, setPickedChunks] = useState<Set<string>>(new Set());
  const [addOpen, setAddOpen] = useState(false);
  const [newQuestion, setNewQuestion] = useState("");
  const [newBusiness, setNewBusiness] = useState("");
  const [newIntent, setNewIntent] = useState("INFORMATION");
  const [newComplexity, setNewComplexity] = useState("simple");
  const [dirty, setDirty] = useState(false);
  const [teamName, setTeamName] = useState("");
  const [mergeReport, setMergeReport] = useState("");
  const [loading, setLoading] = useState(true);
  const csvInput = useRef<HTMLInputElement>(null);
  const jsonInput = useRef<HTMLInputElement>(null);
  const patchInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const embedded = window.__KDIC_EVALUATION_CSV__ && window.__KDIC_CHUNKS_JSONL__;
    const sources = embedded
      ? Promise.resolve([window.__KDIC_EVALUATION_CSV__!, window.__KDIC_CHUNKS_JSONL__!])
      : Promise.all([
          fetch("/data/evaluation.csv").then((res) => res.text()),
          fetch("/data/chunks.jsonl").then((res) => res.text()),
        ]);
    sources.then(([csv, jsonl]) => {
      const parsed = parseCsv(csv);
      const loadedChunks = jsonl.split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line) as Chunk);
      setHeaders(parsed.headers);
      setRows(parsed.rows);
      setBaselineRows(parsed.rows.map((row) => ({ ...row })));
      setChunks(loadedChunks);
      setSelectedId(parsed.rows[0]?.[FIELD.id] ?? "");
      setLoading(false);
    });
  }, []);

  useEffect(() => {
    const guard = (event: BeforeUnloadEvent) => {
      if (dirty) event.preventDefault();
    };
    window.addEventListener("beforeunload", guard);
    return () => window.removeEventListener("beforeunload", guard);
  }, [dirty]);

  const chunkMap = useMemo(() => new Map(chunks.map((chunk) => [chunk.chunk_id, chunk])), [chunks]);
  const selected = rows.find((row) => row[FIELD.id] === selectedId) ?? rows[0];
  const selectedGoldIds = useMemo(() => parseJsonList(selected?.[FIELD.goldChunks]), [selected]);
  const missingGold = selectedGoldIds.filter((id) => !chunkMap.has(id));
  const businessOptions = useMemo(
    () => [...new Set(rows.map((row) => row[FIELD.business]).filter(Boolean))].sort(),
    [rows],
  );
  const dataBusinesses = useMemo(
    () => [...new Set(chunks.map((chunk) => chunk.business_function).filter(Boolean))],
    [chunks],
  );

  const hasIssue = useCallback((row: Row) => {
    const ids = parseJsonList(row[FIELD.goldChunks]);
    return ids.length === 0 || ids.some((id) => !chunkMap.has(id)) ||
      row[FIELD.reviewRequired]?.toUpperCase() === "Y" ||
      row[FIELD.reviewStatus] === "needs_revision";
  }, [chunkMap]);

  const filteredRows = useMemo(() => rows.filter((row) => {
    const term = query.toLowerCase().trim();
    const searchable = `${row[FIELD.id]} ${row[FIELD.question]} ${row[FIELD.goldChunks]}`.toLowerCase();
    return (!term || searchable.includes(term)) &&
      (businessFilter === "전체" || row[FIELD.business] === businessFilter) &&
      (statusFilter === "전체" || (row[FIELD.reviewStatus] || "") === statusFilter) &&
      (!onlyIssues || hasIssue(row));
  }), [rows, query, businessFilter, statusFilter, onlyIssues, hasIssue]);

  const documentIds = useMemo(() => {
    const fromGold = selectedGoldIds.map((id) => chunkMap.get(id)?.parent_doc_id).filter(Boolean) as string[];
    return new Set([...parseJsonList(selected?.[FIELD.goldDocs]), ...fromGold]);
  }, [selected, selectedGoldIds, chunkMap]);

  const comparableChunks = useMemo(() => {
    const term = compareQuery.toLowerCase().trim();
    return chunks
      .filter((chunk) => chunkScope === "all" || documentIds.has(chunk.parent_doc_id || chunk.document_id || ""))
      .filter((chunk) => chunkBusinessFilter === "전체" || chunk.business_function === chunkBusinessFilter)
      .filter((chunk) => !term || `${chunk.chunk_id} ${chunk.title} ${chunk.section_title} ${chunk.content}`.toLowerCase().includes(term));
  }, [chunks, documentIds, compareQuery, chunkScope, chunkBusinessFilter]);

  const currentIndex = filteredRows.findIndex((row) => row[FIELD.id] === selectedId);
  const reviewedCount = rows.filter((row) => ["approved", "auto_approved"].includes(row[FIELD.reviewStatus])).length;
  const issueCount = rows.filter(hasIssue).length;

  function updateSelected(field: string, value: string) {
    setRows((prev) => prev.map((row) => row[FIELD.id] === selectedId ? { ...row, [field]: value } : row));
    setDirty(true);
  }

  function chooseRow(id: string) {
    setSelectedId(id);
    setPickedChunks(new Set());
    setExpandedGold(new Set());
    setCompareQuery("");
  }

  function navigate(delta: number) {
    if (!filteredRows.length) return;
    const index = currentIndex < 0 ? 0 : currentIndex;
    const target = filteredRows[Math.min(filteredRows.length - 1, Math.max(0, index + delta))];
    if (target) chooseRow(target[FIELD.id]);
  }

  function addPickedToGold() {
    if (!selected || !pickedChunks.size) return;
    const ids = [...new Set([...selectedGoldIds, ...pickedChunks])];
    const found = ids.map((id) => chunkMap.get(id)).filter(Boolean) as Chunk[];
    updateSelected(FIELD.goldChunks, jsonList(ids));
    updateSelected(FIELD.goldDocs, jsonList(found.map((c) => c.parent_doc_id || c.document_id || "")));
    updateSelected(FIELD.goldSections, jsonList(found.map((c) => c.section_title || c.title || "")));
    updateSelected(FIELD.goldUrls, jsonList(found.map((c) => c.source_url || "")));
    updateSelected(FIELD.mapping, ids.length > 1 ? "multi_chunk" : "single_chunk");
    setPickedChunks(new Set());
  }

  function removeGold(id: string) {
    const ids = selectedGoldIds.filter((chunkId) => chunkId !== id);
    const found = ids.map((chunkId) => chunkMap.get(chunkId)).filter(Boolean) as Chunk[];
    updateSelected(FIELD.goldChunks, jsonList(ids));
    updateSelected(FIELD.goldDocs, jsonList(found.map((c) => c.parent_doc_id || c.document_id || "")));
    updateSelected(FIELD.goldSections, jsonList(found.map((c) => c.section_title || c.title || "")));
    updateSelected(FIELD.goldUrls, jsonList(found.map((c) => c.source_url || "")));
  }

  function mark(status: "approved" | "pending" | "needs_revision") {
    updateSelected(FIELD.reviewStatus, status);
    updateSelected(FIELD.reviewRequired, status === "approved" ? "N" : "Y");
    updateSelected(FIELD.reviewedAt, today());
  }

  function addQuestion(duplicate = false) {
    if (!newQuestion.trim() && !duplicate) return;
    const base = duplicate && selected ? { ...selected } : Object.fromEntries(headers.map((header) => [header, ""]));
    const id = nextQuestionId(rows);
    const added: Row = {
      ...base,
      [FIELD.id]: id,
      [FIELD.question]: duplicate ? `${selected[FIELD.question]} (복사본)` : newQuestion.trim(),
      [FIELD.business]: duplicate ? selected[FIELD.business] : newBusiness,
      [FIELD.intentGroup]: duplicate ? selected[FIELD.intentGroup] : newIntent,
      [FIELD.complexity]: duplicate ? selected[FIELD.complexity] : newComplexity,
      [FIELD.reviewStatus]: "pending",
      [FIELD.reviewRequired]: "Y",
      [FIELD.reviewedAt]: "",
      [FIELD.reviewer]: "",
    };
    setRows((prev) => [...prev, added]);
    setSelectedId(id);
    setDirty(true);
    setAddOpen(false);
    setNewQuestion("");
  }

  function resetFilters() {
    setQuery("");
    setBusinessFilter("전체");
    setStatusFilter("전체");
    setOnlyIssues(false);
  }

  function buildTeamPatch(): TeamPatch | null {
    const reviewer = teamName.trim();
    if (!reviewer) {
      setMergeReport("먼저 협업 표시줄에 본인 이름을 입력해 주세요.");
      return null;
    }
    const baseline = new Map(baselineRows.map((row) => [row[FIELD.id], row]));
    const changes: TeamPatchChange[] = [];
    for (const row of rows) {
      const before = baseline.get(row[FIELD.id]);
      if (!before) {
        changes.push({
          questionId: row[FIELD.id],
          isNew: true,
          base: {},
          values: { ...row },
          snapshot: { ...row },
        });
        continue;
      }
      const changedBase: Row = {};
      const changedValues: Row = {};
      for (const header of headers) {
        if ((before[header] ?? "") !== (row[header] ?? "")) {
          changedBase[header] = before[header] ?? "";
          changedValues[header] = row[header] ?? "";
        }
      }
      if (Object.keys(changedValues).length) {
        changes.push({
          questionId: row[FIELD.id],
          isNew: false,
          base: changedBase,
          values: changedValues,
          snapshot: { ...row },
        });
      }
    }
    return {
      format: "kdic-gold-review-patch-v1",
      createdAt: new Date().toISOString(),
      createdBy: reviewer,
      changes,
    };
  }

  function sendTeamUpdates() {
    const patch = buildTeamPatch();
    if (!patch) return;
    if (!patch.changes.length) {
      setMergeReport("보낼 수정 내용이 없습니다.");
      return;
    }
    download(
      `KDIC_Gold_수정내용_${patch.createdBy}_${today()}.json`,
      JSON.stringify(patch, null, 2),
      "application/json;charset=utf-8",
    );
    setMergeReport(`${patch.createdBy}님의 수정 질문 ${patch.changes.length}개를 전송용 파일로 만들었습니다.`);
  }

  async function mergeTeamPatches(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files ?? []);
    if (!files.length) return;
    const nextRows = rows.map((row) => ({ ...row }));
    let appliedFields = 0;
    let addedRows = 0;
    let renamedRows = 0;
    const conflicts: string[] = [];
    const senders: string[] = [];

    for (const file of files) {
      try {
        const patch = JSON.parse(await file.text()) as TeamPatch;
        if (patch.format !== "kdic-gold-review-patch-v1" || !Array.isArray(patch.changes)) {
          conflicts.push(`${file.name}: 지원하지 않는 파일`);
          continue;
        }
        senders.push(patch.createdBy || file.name);
        for (const change of patch.changes) {
          const index = nextRows.findIndex((row) => row[FIELD.id] === change.questionId);
          if (change.isNew) {
            if (index < 0) {
              nextRows.push({ ...change.snapshot });
              addedRows += 1;
            } else if (JSON.stringify(nextRows[index]) !== JSON.stringify(change.snapshot)) {
              const newId = nextQuestionId(nextRows);
              nextRows.push({ ...change.snapshot, [FIELD.id]: newId });
              renamedRows += 1;
            }
            continue;
          }
          if (index < 0) {
            nextRows.push({ ...change.snapshot });
            addedRows += 1;
            continue;
          }
          const updated = { ...nextRows[index] };
          for (const [field, value] of Object.entries(change.values)) {
            const current = updated[field] ?? "";
            const base = change.base[field] ?? "";
            if (current === base || current === value) {
              updated[field] = value;
              appliedFields += 1;
            } else {
              conflicts.push(`${change.questionId} · ${field} (${patch.createdBy})`);
            }
          }
          nextRows[index] = updated;
        }
      } catch {
        conflicts.push(`${file.name}: 파일을 읽을 수 없음`);
      }
    }
    setRows(nextRows);
    setDirty(true);
    const conflictText = conflicts.length
      ? ` 충돌 ${conflicts.length}건은 기존 값을 유지했습니다: ${conflicts.slice(0, 3).join(", ")}${conflicts.length > 3 ? "…" : ""}`
      : " 충돌 없이 합쳐졌습니다.";
    setMergeReport(`${senders.join(", ")} 수정본 적용: 필드 ${appliedFields}개, 새 질문 ${addedRows}개${renamedRows ? `, ID 중복 자동변경 ${renamedRows}개` : ""}.${conflictText}`);
    event.target.value = "";
  }

  async function importCsvFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    const bytes = await file.arrayBuffer();
    let text = new TextDecoder("utf-8").decode(bytes);
    if ((text.match(/�/g) || []).length > 2) text = new TextDecoder("euc-kr").decode(bytes);
    const parsed = parseCsv(text);
    setHeaders(parsed.headers);
    setRows(parsed.rows);
    setBaselineRows(parsed.rows.map((row) => ({ ...row })));
    setSelectedId(parsed.rows[0]?.[FIELD.id] ?? "");
    setDirty(false);
  }

  async function importJsonFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    const text = await file.text();
    setChunks(text.split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line)));
  }

  if (loading) {
    return <main className="loading"><div className="loader" /><p>질문과 청크를 불러오고 있습니다.</p></main>;
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <div className="eyebrow">KDIC RAG · EVALUATION WORKSPACE</div>
          <h1>Gold 검수 도구</h1>
          <p>질문별 정답 청크를 읽고, 비교하고, 바로 수정하세요.</p>
        </div>
        <div className="top-actions">
          <div className="dataset-stats">
            <span><b>{rows.length}</b> 질문</span>
            <span><b>{chunks.length}</b> 청크</span>
            <span><b>{dataBusinesses.length}</b> 업무</span>
          </div>
          <button className="button ghost" onClick={() => csvInput.current?.click()} title="다른 평가 CSV 열기">
            <FileSpreadsheet size={17} /> CSV 열기
          </button>
          <button className="button ghost" onClick={() => jsonInput.current?.click()} title="다른 청크 JSONL 열기">
            <FileJson size={17} /> 청크 열기
          </button>
          <button className="button primary" onClick={() => {
            download(`평가데이터셋_Gold검수_${today()}.csv`, exportCsv(headers, rows), "text/csv;charset=utf-8");
            setDirty(false);
          }}>
            <Download size={17} /> CSV 내보내기
          </button>
          <input ref={csvInput} hidden type="file" accept=".csv" onChange={importCsvFile} />
          <input ref={jsonInput} hidden type="file" accept=".jsonl,.json" onChange={importJsonFile} />
          <input ref={patchInput} hidden multiple type="file" accept=".json" onChange={mergeTeamPatches} />
        </div>
      </header>

      <section className="progress-strip">
        <div><span>검수 완료</span><strong>{reviewedCount} / {rows.length}</strong></div>
        <div className="progress-track"><span style={{ width: `${rows.length ? reviewedCount / rows.length * 100 : 0}%` }} /></div>
        <div className={issueCount ? "issue-count" : "issue-count clear"}>
          {issueCount ? <AlertTriangle size={15} /> : <CheckCircle2 size={15} />}
          확인 필요 {issueCount}건
        </div>
        {dirty && <div className="dirty"><span /> 내보내지 않은 변경사항이 있습니다</div>}
      </section>

      <section className="collab-strip">
        <div className="collab-title"><Users size={18} /><span><b>4인 협업 모드</b><small>서버 없이 수정 파일을 주고받아 안전하게 합칩니다.</small></span></div>
        <label className="reviewer-input"><span>내 이름</span><input value={teamName} onChange={(e) => setTeamName(e.target.value)} placeholder="예: 임도균" /></label>
        <button className="button collab-send" onClick={sendTeamUpdates}><Send size={16} /> 수정 내용 보내기</button>
        <button className="button collab-merge" onClick={() => patchInput.current?.click()}><GitMerge size={16} /> 팀원 수정본 합치기</button>
        {mergeReport && <div className="merge-report"><CheckCircle2 size={15} /><span>{mergeReport}</span><button onClick={() => setMergeReport("")}><X size={14} /></button></div>}
      </section>

      <div className="workspace">
        <aside className="sidebar">
          <div className="sidebar-head">
            <div>
              <span className="section-label">질문 목록</span>
              <strong>{filteredRows.length}개</strong>
            </div>
            <button className="icon-button accent" onClick={() => setAddOpen(true)} title="새 질문 추가">
              <Plus size={19} />
            </button>
          </div>
          <label className="search-box">
            <Search size={17} />
            <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="질문, ID, 청크 ID 검색" />
            {query && <button onClick={() => setQuery("")} title="검색어 지우기"><X size={15} /></button>}
          </label>
          <div className="filters">
            <label><span>업무</span><select value={businessFilter} onChange={(e) => setBusinessFilter(e.target.value)}>
              <option>전체</option>{businessOptions.map((option) => <option key={option}>{option}</option>)}
            </select></label>
            <label><span>상태</span><select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
              <option>전체</option>
              <option value="approved">검수 완료</option>
              <option value="auto_approved">자동 승인</option>
              <option value="pending">검수 대기</option>
              <option value="needs_revision">수정 필요</option>
              <option value="">미검수</option>
            </select></label>
          </div>
          <div className="filter-row">
            <label className="check-label">
              <input type="checkbox" checked={onlyIssues} onChange={(e) => setOnlyIssues(e.target.checked)} />
              문제 있는 질문만
            </label>
            <button onClick={resetFilters}><RotateCcw size={14} /> 초기화</button>
          </div>
          <button className="filtered-export" onClick={() => {
            const scope = businessFilter === "전체" ? "현재목록" : businessFilter.replaceAll(/[\\/:*?"<>|]/g, "_");
            download(`평가데이터셋_${scope}_${today()}.csv`, exportCsv(headers, filteredRows), "text/csv;charset=utf-8");
          }} disabled={!filteredRows.length}>
            <Download size={15} /> 현재 목록 {filteredRows.length}개만 내보내기
          </button>
          <div className="question-list">
            {filteredRows.map((row) => {
              const ids = parseJsonList(row[FIELD.goldChunks]);
              const issue = hasIssue(row);
              const status = row[FIELD.reviewStatus] || "";
              return (
                <button key={row[FIELD.id]} className={`question-card ${selectedId === row[FIELD.id] ? "selected" : ""}`} onClick={() => chooseRow(row[FIELD.id])}>
                  <div className="question-meta">
                    <span className="qid">{row[FIELD.id]}</span>
                    <span className="business">{row[FIELD.business] || "업무 미지정"}</span>
                    <span className={`status-dot ${statusTone(status)}`} title={REVIEW_LABELS[status] || status} />
                  </div>
                  <p>{row[FIELD.question] || "질문 내용 없음"}</p>
                  <div className="question-foot">
                    <span>{ids.length}개 Gold 청크</span>
                    {issue && <span className="needs-check"><AlertTriangle size={12} /> 확인 필요</span>}
                  </div>
                </button>
              );
            })}
            {!filteredRows.length && <div className="empty-mini"><Filter size={24} /><p>조건에 맞는 질문이 없습니다.</p></div>}
          </div>
          <button className="add-question-wide" onClick={() => setAddOpen(true)}><ListPlus size={17} /> 새 질문 추가</button>
        </aside>

        {selected ? (
          <section className="content">
            <nav className="question-nav">
              <button onClick={() => navigate(-1)} disabled={currentIndex <= 0} title="이전 질문"><ArrowLeft size={18} /></button>
              <span>{Math.max(1, currentIndex + 1)} / {filteredRows.length || rows.length}</span>
              <button onClick={() => navigate(1)} disabled={currentIndex < 0 || currentIndex >= filteredRows.length - 1} title="다음 질문"><ArrowRight size={18} /></button>
            </nav>

            <article className="question-hero">
              <div className="hero-topline">
                <div className="id-group"><span>{selected[FIELD.id]}</span><span>{selected[FIELD.importance] ? `중요도 ${selected[FIELD.importance]}` : "중요도 미지정"}</span></div>
                <span className={`status-pill ${statusTone(selected[FIELD.reviewStatus])}`}>
                  {REVIEW_LABELS[selected[FIELD.reviewStatus]] || selected[FIELD.reviewStatus]}
                </span>
              </div>
              <h2>{selected[FIELD.question]}</h2>
              <div className="chips">
                {[selected[FIELD.business], selected[FIELD.intentGroup], selected[FIELD.intentMain], selected[FIELD.complexity]]
                  .filter(Boolean).map((value, index) => <span key={`${value}-${index}`}>{value}</span>)}
              </div>
            </article>

            <section className={`validation ${missingGold.length ? "invalid" : selectedGoldIds.length ? "valid" : "empty"}`}>
              {missingGold.length ? <AlertTriangle size={21} /> : selectedGoldIds.length ? <CheckCircle2 size={21} /> : <AlertTriangle size={21} />}
              <div>
                <strong>{missingGold.length ? "Gold 청크 ID를 확인해 주세요" : selectedGoldIds.length ? "Gold 매핑이 정상입니다" : "Gold 청크가 비어 있습니다"}</strong>
                <p>{selectedGoldIds.length}개 지정 · {selectedGoldIds.length - missingGold.length}개 존재
                  {missingGold.length > 0 && ` · 누락: ${missingGold.join(", ")}`}</p>
              </div>
            </section>

            <section className="panel">
              <button className="panel-title" onClick={() => setGoldOpen(!goldOpen)}>
                <span className="title-icon"><Clipboard size={19} /></span>
                <span><b>Gold 청크 보기</b><small>긴 본문을 세로형 카드로 편하게 읽습니다.</small></span>
                <span className="count-badge">{selectedGoldIds.length}</span>
                {goldOpen ? <ChevronDown size={20} /> : <ChevronRight size={20} />}
              </button>
              {goldOpen && <div className="panel-body">
                <div className="panel-tools">
                  <span>청크 ID를 누르면 본문이 펼쳐집니다.</span>
                  <button onClick={() => setExpandedGold(
                    expandedGold.size === selectedGoldIds.length ? new Set() : new Set(selectedGoldIds),
                  )}>
                    {expandedGold.size === selectedGoldIds.length ? "모두 접기" : "모두 펼치기"}
                  </button>
                </div>
                <div className="gold-list">
                  {selectedGoldIds.map((id, index) => {
                    const chunk = chunkMap.get(id);
                    const open = expandedGold.has(id);
                    return (
                      <article className={`chunk-card ${!chunk ? "missing" : ""}`} key={id}>
                        <button className="chunk-summary" onClick={() => {
                          setExpandedGold((prev) => {
                            const next = new Set(prev);
                            if (next.has(id)) next.delete(id);
                            else next.add(id);
                            return next;
                          });
                        }}>
                          <span className="chunk-order">{index + 1}</span>
                          <span className="chunk-main">
                            <b>{id}</b>
                            <small>{chunk ? `${chunk.title || "제목 없음"}${chunk.section_title ? ` · ${chunk.section_title}` : ""}` : "chunks.jsonl에서 찾을 수 없음"}</small>
                          </span>
                          {open ? <ChevronDown size={19} /> : <ChevronRight size={19} />}
                        </button>
                        {open && <div className="chunk-detail">
                          {chunk ? <>
                            <dl>
                              <div><dt>상위 문서</dt><dd>{chunk.parent_doc_id || chunk.document_id}</dd></div>
                              <div><dt>업무</dt><dd>{chunk.business_function}</dd></div>
                              <div><dt>섹션</dt><dd>{chunk.section_title || chunk.title || "-"}</dd></div>
                            </dl>
                            <div className="chunk-actions">
                              <button onClick={() => navigator.clipboard.writeText(id)}><Copy size={14} /> ID 복사</button>
                              <button onClick={() => navigator.clipboard.writeText(chunk.content || "")}><Copy size={14} /> 본문 복사</button>
                              {chunk.source_url && <a href={chunk.source_url} target="_blank" rel="noreferrer"><Link2 size={14} /> 원문 열기</a>}
                              <button className="danger-link" onClick={() => removeGold(id)}><Trash2 size={14} /> Gold에서 제거</button>
                            </div>
                            <div className="chunk-content">{chunk.content || "본문 없음"}</div>
                          </> : <div className="missing-message">이 ID와 일치하는 청크가 없습니다. 오타이거나 청크 파일 버전이 다른지 확인하세요.</div>}
                        </div>}
                      </article>
                    );
                  })}
                  {!selectedGoldIds.length && <div className="empty-state">
                    <FolderOpen size={34} />
                    <h3>아직 Gold 청크가 없습니다</h3>
                    <p>아래 ‘청크 찾아 Gold에 추가’에서 전체 청크를 검색해 선택하세요.</p>
                  </div>}
                </div>
              </div>}
            </section>

            <section className="panel">
              <button className="panel-title" onClick={() => setCompareOpen(!compareOpen)}>
                <span className="title-icon blue"><Search size={19} /></span>
                <span><b>청크 찾아 Gold에 추가</b><small>같은 문서 또는 전체 427개 청크에서 정답 근거를 선택합니다.</small></span>
                <span className="count-badge">{comparableChunks.length}</span>
                {compareOpen ? <ChevronDown size={20} /> : <ChevronRight size={20} />}
              </button>
              {compareOpen && <div className="panel-body">
                <div className="scope-tabs" role="group" aria-label="청크 검색 범위">
                  <button className={chunkScope === "document" ? "active" : ""} onClick={() => setChunkScope("document")}>같은 문서 청크</button>
                  <button className={chunkScope === "all" ? "active" : ""} onClick={() => setChunkScope("all")}>전체 427개 청크</button>
                </div>
                {(chunkScope === "all" || documentIds.size) ? <>
                  <div className="compare-toolbar">
                    <label className="search-box compact"><Search size={16} /><input value={compareQuery} onChange={(e) => setCompareQuery(e.target.value)} placeholder={chunkScope === "all" ? "전체 청크에서 ID·제목·본문 검색" : "현재 문서 안에서 검색"} /></label>
                    <select className="chunk-business-select" value={chunkBusinessFilter} onChange={(e) => setChunkBusinessFilter(e.target.value)} aria-label="청크 업무 필터">
                      <option>전체</option>
                      {dataBusinesses.map((business) => <option key={business}>{business}</option>)}
                    </select>
                    <span>{chunkScope === "all" ? `전체 ${chunks.length}개 중 ${comparableChunks.length}개` : `문서 ${Array.from(documentIds).join(", ")}`}</span>
                    <button className="button secondary" disabled={!pickedChunks.size} onClick={addPickedToGold}>
                      <Plus size={16} /> 선택 {pickedChunks.size || ""}개 Gold에 추가
                    </button>
                  </div>
                  <div className="candidate-list">
                    {comparableChunks.map((chunk) => {
                      const isGold = selectedGoldIds.includes(chunk.chunk_id);
                      const checked = pickedChunks.has(chunk.chunk_id);
                      return (
                        <label className={`candidate ${isGold ? "is-gold" : ""}`} key={chunk.chunk_id}>
                          <input type="checkbox" disabled={isGold} checked={checked || isGold} onChange={(e) => {
                            setPickedChunks((prev) => {
                              const next = new Set(prev);
                              if (e.target.checked) next.add(chunk.chunk_id);
                              else next.delete(chunk.chunk_id);
                              return next;
                            });
                          }} />
                          <span className="candidate-text">
                            <span><b>{chunk.chunk_id}</b>{isGold && <em>현재 Gold</em>}</span>
                            <strong>{chunk.section_title || chunk.title || "제목 없음"}</strong>
                            <small>{(chunk.content || "").replace(/\s+/g, " ").slice(0, 180)}{(chunk.content?.length || 0) > 180 ? "…" : ""}</small>
                          </span>
                        </label>
                      );
                    })}
                  </div>
                </> : <div className="empty-state small">
                  <Search size={30} /><h3>비교할 상위 문서가 없습니다</h3>
                  <p>위에서 ‘전체 427개 청크’를 선택하면 모든 청크에서 찾을 수 있습니다.</p>
                </div>}
              </div>}
            </section>

            <section className="panel">
              <button className="panel-title" onClick={() => setEditOpen(!editOpen)}>
                <span className="title-icon amber"><Save size={19} /></span>
                <span><b>라벨 · 검수 정보 편집</b><small>정답 요약, 검수 상태, 메모와 원본 라벨을 수정합니다.</small></span>
                {editOpen ? <ChevronDown size={20} /> : <ChevronRight size={20} />}
              </button>
              {editOpen && <div className="panel-body form-body">
                <div className="form-grid">
                  <label className="span-2"><span>예상 질문</span><textarea value={selected[FIELD.question]} onChange={(e) => updateSelected(FIELD.question, e.target.value)} rows={2} /></label>
                  <label><span>업무 라벨</span><input value={selected[FIELD.business]} onChange={(e) => updateSelected(FIELD.business, e.target.value)} /></label>
                  <label><span>질문 복잡도</span><select value={selected[FIELD.complexity]} onChange={(e) => updateSelected(FIELD.complexity, e.target.value)}>
                    {["simple", "conditional", "multi_condition", "multi_intent", "ambiguous"].map((item) => <option key={item}>{item}</option>)}
                  </select></label>
                  <label><span>검수 상태</span><select value={selected[FIELD.reviewStatus]} onChange={(e) => updateSelected(FIELD.reviewStatus, e.target.value)}>
                    <option value="">미검수</option><option value="pending">검수 대기</option><option value="approved">검수 완료</option><option value="needs_revision">수정 필요</option><option value="auto_approved">자동 승인</option>
                  </select></label>
                  <label><span>검수자</span><input value={selected[FIELD.reviewer]} onChange={(e) => updateSelected(FIELD.reviewer, e.target.value)} placeholder="이름" /></label>
                  <label className="span-2"><span>정답 요약</span><textarea value={selected[FIELD.answerSummary]} onChange={(e) => updateSelected(FIELD.answerSummary, e.target.value)} rows={4} placeholder="이 질문에 반드시 포함돼야 할 정답을 간결하게 적으세요." /></label>
                  <label className="span-2"><span>Gold 매핑 메모</span><textarea value={selected[FIELD.mappingNote]} onChange={(e) => updateSelected(FIELD.mappingNote, e.target.value)} rows={2} placeholder="왜 이 청크를 Gold로 선택했는지, 추가 확인할 점 등을 적으세요." /></label>
                </div>
                <button className="advanced-toggle" onClick={() => setAdvancedOpen(!advancedOpen)}>
                  {advancedOpen ? <ChevronDown size={16} /> : <ChevronRight size={16} />} 고급 라벨 펼치기
                </button>
                {advancedOpen && <div className="form-grid advanced">
                  {[FIELD.intentGroup, FIELD.intentMain, FIELD.intentSub, FIELD.responsePolicy, FIELD.answerable, FIELD.mapping, FIELD.note].map((field) =>
                    <label key={field} className={field === FIELD.note ? "span-2" : ""}><span>{field}</span><input value={selected[field]} onChange={(e) => updateSelected(field, e.target.value)} /></label>,
                  )}
                  <label className="span-2"><span>gold_document_ids</span><input value={selected[FIELD.goldDocs]} onChange={(e) => updateSelected(FIELD.goldDocs, e.target.value)} /></label>
                  <label className="span-2"><span>gold_chunk_ids</span><input value={selected[FIELD.goldChunks]} onChange={(e) => updateSelected(FIELD.goldChunks, e.target.value)} /></label>
                </div>}
              </div>}
            </section>

            <footer className="sticky-actions">
              <div><span>{selected[FIELD.id]}</span><p>{dirty ? "변경사항은 CSV 내보내기로 저장됩니다." : "현재 데이터와 동일합니다."}</p></div>
              <button className="button ghost" onClick={() => mark("needs_revision")}><AlertTriangle size={16} /> 수정 필요</button>
              <button className="button secondary" onClick={() => mark("pending")}><Save size={16} /> 검수 대기</button>
              <button className="button success" onClick={() => mark("approved")}><Check size={17} /> 검수 완료</button>
            </footer>
          </section>
        ) : <section className="content empty-state"><h2>질문을 선택하세요.</h2></section>}
      </div>

      {addOpen && <div className="modal-backdrop" role="presentation" onMouseDown={(e) => e.target === e.currentTarget && setAddOpen(false)}>
        <section className="modal" role="dialog" aria-modal="true" aria-label="새 질문 추가">
          <header><div><span className="section-label">평가데이터 확장</span><h2>새 질문 추가</h2></div><button className="icon-button" onClick={() => setAddOpen(false)}><X size={20} /></button></header>
          <div className="modal-body">
            <label><span>예상 질문 <b>*</b></span><textarea rows={4} autoFocus value={newQuestion} onChange={(e) => setNewQuestion(e.target.value)} placeholder="사용자가 실제로 물을 법한 문장으로 입력하세요." /></label>
            <div className="form-grid">
              <label><span>업무 라벨</span><select value={newBusiness} onChange={(e) => setNewBusiness(e.target.value)}>
                <option value="">선택</option>{businessOptions.map((item) => <option key={item}>{item}</option>)}
              </select></label>
              <label><span>의도 그룹</span><select value={newIntent} onChange={(e) => setNewIntent(e.target.value)}>
                {["INFORMATION", "ACTION", "STATUS", "CLARIFICATION", "OUT_OF_SCOPE"].map((item) => <option key={item}>{item}</option>)}
              </select></label>
              <label><span>질문 복잡도</span><select value={newComplexity} onChange={(e) => setNewComplexity(e.target.value)}>
                {["simple", "conditional", "multi_condition", "multi_intent", "ambiguous"].map((item) => <option key={item}>{item}</option>)}
              </select></label>
              <label><span>생성될 ID</span><input readOnly value={nextQuestionId(rows)} /></label>
            </div>
            <div className="add-tip"><CheckCircle2 size={18} /><p>먼저 질문만 추가해도 됩니다. 추가 후 본문 화면에서 Gold 청크와 세부 라벨을 지정하세요.</p></div>
          </div>
          <footer>
            <button className="button ghost" onClick={() => addQuestion(true)} disabled={!selected}><Copy size={16} /> 현재 질문 복제</button>
            <span />
            <button className="button ghost" onClick={() => setAddOpen(false)}>취소</button>
            <button className="button primary" onClick={() => addQuestion(false)} disabled={!newQuestion.trim()}><Plus size={17} /> 질문 추가</button>
          </footer>
        </section>
      </div>}
    </main>
  );
}
