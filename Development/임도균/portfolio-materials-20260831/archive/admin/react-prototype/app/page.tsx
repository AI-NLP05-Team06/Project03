'use client';

import { useEffect, useMemo, useState } from 'react';

type View = 'dashboard' | 'data' | 'new' | 'parameters' | 'pipeline' | 'evaluation';
type DocumentRow = { doc_id: string; title: string; business_function: string; source_url: string; page_type?: string; parsed_at?: string; content_text?: string; quality?: { status?: string; reasons?: string[] } };
type ChunkRow = { chunk_id: string; parent_doc_id: string; title: string; section_title?: string; content: string; business_function: string; source_url?: string };

const navItems: { id: View; label: string; icon: string; caption: string }[] = [
  { id: 'dashboard', label: '대시보드', icon: '▦', caption: '운영 현황' },
  { id: 'data', label: '데이터 관리', icon: '▤', caption: '페이지·청크' },
  { id: 'new', label: '신규 데이터', icon: '＋', caption: '파싱·청킹 검수' },
  { id: 'parameters', label: '파라미터 테스트', icon: '⇄', caption: '설정 A/B 비교' },
  { id: 'pipeline', label: '파이프라인', icon: '↻', caption: '재수집·재적재' },
  { id: 'evaluation', label: '평가 관리', icon: '◎', caption: '테스트셋 평가' },
];
const domains = ['전체', '예금자보호제도', '예금보험금 안내', '고객 미수령금 신청', '착오송금 반환 신청', '채무조정 안내', '은닉재산 신고'];
const mockResults = [
  { id: 'MT-013_chunk_001', title: '반환지원 비용', dense: 1, bm25: 2, score: .0458, gold: true },
  { id: 'MT-013_chunk_003', title: '회수금액별 비용률', dense: 4, bm25: 1, score: .0412, gold: true },
  { id: 'MT-004_chunk_002', title: '반환지원 신청 대상', dense: 2, bm25: 6, score: .0365, gold: false },
  { id: 'MT-005_chunk_001', title: '신청 가능 기간', dense: 7, bm25: 3, score: .0317, gold: false },
  { id: 'BI-002_chunk_004', title: '예금보험금 지급절차', dense: 5, bm25: 8, score: .0248, gold: false },
];
const initialJobs = [
  { id: 'JOB-20260824-003', type: '선택 페이지 재수집', target: 'DP-001 외 2건', status: 'SUCCEEDED', progress: 100, step: '완료', started: '오늘 09:42' },
  { id: 'JOB-20260824-002', type: '테스트 인덱스 생성', target: '427개 청크', status: 'RUNNING', progress: 68, step: '임베딩', started: '오늘 09:18' },
  { id: 'JOB-20260823-008', type: '평가 실행', target: 'v5 · 270문항', status: 'FAILED', progress: 74, step: '평가', started: '어제 17:35' },
];

function parseJsonl<T>(text: string): T[] { return text.split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line)); }
function StatusBadge({ status }: { status: string }) {
  const labels: Record<string, string> = { ACTIVE: '운영 중', REVIEW_REQUIRED: '검수 필요', RUNNING: '처리 중', SUCCEEDED: '완료', FAILED: '실패', INACTIVE: '비활성', QUEUED: '대기', APPROVED: '승인', DRAFT: '초안' };
  return <span className={`status status-${status.toLowerCase()}`}><i />{labels[status] ?? status}</span>;
}
function Metric({ label, value, detail, tone = 'blue' }: { label: string; value: string; detail: string; tone?: string }) {
  return <article className={`metric metric-${tone}`}><div className="metric-top"><span>{label}</span><b>↗</b></div><strong>{value}</strong><small>{detail}</small></article>;
}

export default function Home() {
  const [view, setView] = useState<View>('dashboard');
  const [dark, setDark] = useState(false);
  const [documents, setDocuments] = useState<DocumentRow[]>([]);
  const [chunks, setChunks] = useState<ChunkRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedDoc, setSelectedDoc] = useState<string>('DP-001');
  const [query, setQuery] = useState('');
  const [domain, setDomain] = useState('전체');
  const [inactive, setInactive] = useState<string[]>([]);
  const [toast, setToast] = useState('');
  const [previewed, setPreviewed] = useState(false);
  const [approved, setApproved] = useState(false);
  const [url, setUrl] = useState('https://www.kdic.or.kr/example');
  const [chunkSize, setChunkSize] = useState(500);
  const [overlap, setOverlap] = useState(50);
  const [topK, setTopK] = useState(5);
  const [denseWeight, setDenseWeight] = useState(70);
  const [candidateDepth, setCandidateDepth] = useState(20);
  const [rrfK, setRrfK] = useState(10);
  const [compared, setCompared] = useState(false);
  const [jobs, setJobs] = useState(initialJobs);
  const [evalProgress, setEvalProgress] = useState(0);
  const [evalDone, setEvalDone] = useState(false);

  useEffect(() => {
    Promise.all([fetch('/data/documents.jsonl').then((r) => r.text()), fetch('/data/chunks.jsonl').then((r) => r.text())])
      .then(([docText, chunkText]) => { setDocuments(parseJsonl<DocumentRow>(docText)); setChunks(parseJsonl<ChunkRow>(chunkText)); })
      .catch(() => setToast('로컬 데이터 파일을 불러오지 못해 예시 데이터로 표시합니다.'))
      .finally(() => setLoading(false));
  }, []);
  useEffect(() => {
    if (evalProgress <= 0 || evalProgress >= 100) return;
    const timer = window.setInterval(() => setEvalProgress((value) => Math.min(100, value + 8)), 260);
    return () => window.clearInterval(timer);
  }, [evalProgress]);
  useEffect(() => { if (evalProgress === 100) setEvalDone(true); }, [evalProgress]);

  const chunkCounts = useMemo(() => { const counts: Record<string, number> = {}; chunks.forEach((c) => { counts[c.parent_doc_id] = (counts[c.parent_doc_id] ?? 0) + 1; }); return counts; }, [chunks]);
  const filteredDocs = useMemo(() => documents.filter((doc) => `${doc.doc_id} ${doc.title} ${doc.source_url}`.toLowerCase().includes(query.toLowerCase()) && (domain === '전체' || doc.business_function === domain)), [documents, query, domain]);
  const currentDoc = documents.find((doc) => doc.doc_id === selectedDoc) ?? documents[0];
  const currentChunks = chunks.filter((chunk) => chunk.parent_doc_id === currentDoc?.doc_id);
  const showToast = (message: string) => { setToast(message); window.setTimeout(() => setToast(''), 2800); };
  const triggerJob = (type: string) => {
    const id = `JOB-20260824-${String(jobs.length + 4).padStart(3, '0')}`;
    setJobs((rows) => [{ id, type, target: type.includes('전체') ? `${documents.length || 87}개 페이지` : '선택 대상', status: 'QUEUED', progress: 0, step: '대기', started: '방금 전' }, ...rows]);
    showToast(`${type} 작업을 생성했습니다. 실제 실행은 관리자 API 연결 후 활성화됩니다.`);
  };

  return <div className={dark ? 'admin-app dark' : 'admin-app'}>
    <aside className="sidebar">
      <div className="brand"><div className="brand-mark">K</div><div><strong>KDIC RAG</strong><span>ADMIN CONSOLE</span></div></div>
      <nav aria-label="관리자 메뉴"><p className="nav-label">WORKSPACE</p>{navItems.map((item) => <button key={item.id} className={view === item.id ? 'nav-item active' : 'nav-item'} onClick={() => setView(item.id)}><span className="nav-icon">{item.icon}</span><span><b>{item.label}</b><small>{item.caption}</small></span></button>)}</nav>
      <div className="sidebar-config"><p>운영 검색 설정</p><strong>Structured Hybrid</strong><span>Dense 0.7 · BM25 0.3</span><span>RRF K=10 · Top-K 5</span><StatusBadge status="ACTIVE" /></div>
      <div className="profile"><div className="avatar">관</div><div><strong>관리자</strong><span>RAG 운영팀</span></div><button>⋮</button></div>
    </aside>
    <main className="main">
      <header className="topbar"><div><span className="crumb">KDIC RAG Admin /</span><strong>{navItems.find((item) => item.id === view)?.label}</strong></div><div className="system-pills"><span><i className="green-dot" /> Elasticsearch 정상</span><span><i className="green-dot" /> Vector 1024D</span><button onClick={() => setDark(!dark)} aria-label="테마 변경">{dark ? '☀' : '☾'}</button></div></header>
      <div className="content">
        {view === 'dashboard' && <Dashboard documents={documents.length || 87} chunks={chunks.length || 427} jobs={jobs} loading={loading} onNavigate={setView} />}
        {view === 'data' && <DataManagement loading={loading} documents={filteredDocs} chunks={currentChunks} selectedDoc={selectedDoc} currentDoc={currentDoc} chunkCounts={chunkCounts} query={query} domain={domain} inactive={inactive} onQuery={setQuery} onDomain={setDomain} onSelect={setSelectedDoc} onToggleInactive={(id: string) => { setInactive((rows) => rows.includes(id) ? rows.filter((row) => row !== id) : [...rows, id]); showToast(inactive.includes(id) ? '페이지를 다시 활성화했습니다.' : '페이지를 비활성화했습니다. 원본 데이터는 삭제하지 않았습니다.'); }} onJob={() => triggerJob('선택 페이지 재수집')} />}
        {view === 'new' && <NewData url={url} setUrl={setUrl} previewed={previewed} approved={approved} chunkSize={chunkSize} overlap={overlap} setChunkSize={setChunkSize} setOverlap={setOverlap} onPreview={() => { setPreviewed(true); setApproved(false); showToast('파싱·청킹 미리보기를 생성했습니다. 현재는 안전한 시연 데이터입니다.'); }} onApprove={() => { setApproved(true); triggerJob('신규 URL 테스트 적재'); }} />}
        {view === 'parameters' && <ParameterTest topK={topK} setTopK={setTopK} denseWeight={denseWeight} setDenseWeight={setDenseWeight} candidateDepth={candidateDepth} setCandidateDepth={setCandidateDepth} rrfK={rrfK} setRrfK={setRrfK} compared={compared} onCompare={() => setCompared(true)} />}
        {view === 'pipeline' && <Pipeline jobs={jobs} onTrigger={triggerJob} />}
        {view === 'evaluation' && <Evaluation progress={evalProgress} done={evalDone} onRun={() => { setEvalDone(false); setEvalProgress(1); }} />}
      </div>
      <footer><span>활성 인덱스 <b>kdic-hybrid-c-v3</b></span><span>페이지 {documents.length || 87} · 청크 {chunks.length || 427}</span><span>최근 갱신 2026.08.24 09:42</span></footer>
    </main>
    {toast && <div className="toast"><b>✓</b>{toast}</div>}
  </div>;
}

function PageHeading({ eyebrow, title, description, action }: { eyebrow: string; title: string; description: string; action?: React.ReactNode }) {
  return <div className="page-heading"><div><span>{eyebrow}</span><h1>{title}</h1><p>{description}</p></div>{action}</div>;
}

function Dashboard({ documents, chunks, jobs, loading, onNavigate }: { documents: number; chunks: number; jobs: typeof initialJobs; loading: boolean; onNavigate: (view: View) => void }) {
  return <><PageHeading eyebrow="OVERVIEW" title="관리자 대시보드" description="데이터 상태와 검색 품질, 진행 중인 파이프라인을 한눈에 확인합니다." action={<button className="primary-btn" onClick={() => onNavigate('new')}>＋ 신규 URL 추가</button>} />
    <section className="metric-grid"><Metric label="적재 페이지" value={loading ? '—' : `${documents}`} detail="6개 업무 도메인" /><Metric label="검색 청크" value={loading ? '—' : `${chunks}`} detail="Structured 임베딩 완료" tone="purple" /><Metric label="평가 데이터셋" value="270" detail="Gold Labels v5" tone="cyan" /><Metric label="실패 Job" value="1" detail="검토가 필요합니다" tone="red" /></section>
    <section className="dashboard-grid"><article className="panel quality-panel"><div className="panel-head"><div><span className="section-kicker">RETRIEVAL QUALITY</span><h2>검색 품질 기준선</h2></div><button onClick={() => onNavigate('evaluation')}>평가 관리 →</button></div><div className="quality-row"><div><span>Hit@3</span><strong>91.2%</strong><small>+2.8%p</small></div><div><span>Recall@5</span><strong>78.4%</strong><small>+3.1%p</small></div><div><span>nDCG@5</span><strong>82.1%</strong><small>+1.6%p</small></div><div><span>평균 검색시간</span><strong>384ms</strong><small className="neutral">+24ms</small></div></div><div className="spark-bars">{[42,58,49,66,61,74,68,83,77,92,86,96].map((h, i) => <i key={i} style={{height: `${h}%`}} />)}</div></article><article className="panel config-panel"><div className="panel-head"><div><span className="section-kicker">ACTIVE CONFIG</span><h2>현재 운영 설정</h2></div><StatusBadge status="ACTIVE" /></div><dl><div><dt>검색 방식</dt><dd>Structured Hybrid</dd></div><div><dt>Dense / BM25</dt><dd>0.7 / 0.3</dd></div><div><dt>RRF K</dt><dd>10</dd></div><div><dt>Candidate Depth</dt><dd>각 20</dd></div><div><dt>최종 Top-K</dt><dd>5</dd></div><div><dt>Reranker</dt><dd>미사용</dd></div></dl></article></section>
    <section className="panel jobs-panel"><div className="panel-head"><div><span className="section-kicker">RECENT JOBS</span><h2>최근 파이프라인</h2></div><button onClick={() => onNavigate('pipeline')}>전체 작업 보기 →</button></div><JobTable jobs={jobs.slice(0,3)} /></section></>;
}

function JobTable({ jobs, detailed = false }: { jobs: typeof initialJobs; detailed?: boolean }) {
  return <div className={detailed ? 'job-table detailed' : 'job-table'}><div className="job-row job-header"><span>Job ID</span><span>작업</span><span>대상</span><span>상태</span>{detailed && <span>현재 단계</span>}<span>진행률</span><span>시작</span></div>{jobs.map((job) => <div className="job-row" key={job.id}><b>{job.id}</b><span>{job.type}</span><span>{job.target}</span><StatusBadge status={job.status} />{detailed && <span>{job.step}</span>}<div className="progress-cell"><i><em style={{width: `${job.progress}%`}} /></i><small>{job.progress}%</small></div><span>{job.started}</span></div>)}</div>;
}

function DataManagement({ loading, documents, chunks, selectedDoc, currentDoc, chunkCounts, query, domain, inactive, onQuery, onDomain, onSelect, onToggleInactive, onJob }: any) {
  const [expanded, setExpanded] = useState('');
  return <><PageHeading eyebrow="DATA CATALOG" title="적재 데이터 관리" description="운영 인덱스의 페이지·청크·메타데이터를 조회하고 안전하게 비활성화합니다." action={<button className="secondary-btn" onClick={onJob}>↻ 선택 페이지 재수집</button>} />
    <section className="data-layout"><article className="panel document-list"><div className="filter-row"><input value={query} onChange={(e) => onQuery(e.target.value)} placeholder="제목·ID·URL 검색" /><select value={domain} onChange={(e) => onDomain(e.target.value)}>{domains.map((item) => <option key={item}>{item}</option>)}</select></div><div className="list-summary"><b>{loading ? '불러오는 중' : `${documents.length}개 페이지`}</b><span>원본은 변경되지 않습니다</span></div><div className="doc-scroll">{documents.slice(0,40).map((doc: DocumentRow) => <button className={selectedDoc === doc.doc_id ? 'doc-row selected' : 'doc-row'} key={doc.doc_id} onClick={() => onSelect(doc.doc_id)}><span className="doc-id">{doc.doc_id}</span><span className="doc-main"><b>{doc.title}</b><small>{doc.business_function}</small></span><span className="chunk-count">{chunkCounts[doc.doc_id] ?? 0}</span><StatusBadge status={inactive.includes(doc.doc_id) ? 'INACTIVE' : 'ACTIVE'} /></button>)}</div></article>
      <article className="panel doc-detail">{currentDoc ? <><div className="detail-head"><div><span className="doc-id">{currentDoc.doc_id}</span><h2>{currentDoc.title}</h2><p>{currentDoc.business_function} · {currentDoc.page_type ?? 'page'}</p></div><div className="detail-actions"><a href={currentDoc.source_url} target="_blank" rel="noreferrer">원문 열기 ↗</a><button className="danger-btn" onClick={() => onToggleInactive(currentDoc.doc_id)}>{inactive.includes(currentDoc.doc_id) ? '다시 활성화' : '비활성화'}</button></div></div><div className="meta-strip"><span><small>연결 청크</small><b>{chunks.length}개</b></span><span><small>품질 상태</small><b>{currentDoc.quality?.status === 'pass' ? 'PASS' : '검토'}</b></span><span><small>최종 수집</small><b>{currentDoc.parsed_at?.slice(0,10) ?? '2026-07-21'}</b></span></div><h3 className="sub-title">연결된 청크</h3><div className="chunk-stack">{chunks.map((chunk: ChunkRow) => <div className={expanded === chunk.chunk_id ? 'chunk-card open' : 'chunk-card'} key={chunk.chunk_id}><button onClick={() => setExpanded(expanded === chunk.chunk_id ? '' : chunk.chunk_id)}><span><b>{chunk.chunk_id}</b><small>{chunk.section_title || chunk.title}</small></span><span>{chunk.content.length}자</span><i>{expanded === chunk.chunk_id ? '−' : '+'}</i></button>{expanded === chunk.chunk_id && <div className="chunk-body"><p>{chunk.content}</p><dl><div><dt>Parent</dt><dd>{chunk.parent_doc_id}</dd></div><div><dt>업무</dt><dd>{chunk.business_function}</dd></div></dl></div>}</div>)}</div></> : <div className="empty-state">페이지를 선택해 주세요.</div>}</article></section></>;
}

function NewData({ url, setUrl, previewed, approved, chunkSize, overlap, setChunkSize, setOverlap, onPreview, onApprove }: any) {
  const previewChunks = [{ id: 'TEMP_chunk_000', section: '지원 대상', chars: 486, state: '정상', text: '착오송금 반환지원 신청 대상과 기본 요건에 관한 파싱 결과입니다.' },{ id: 'TEMP_chunk_001', section: '신청 절차', chars: 522, state: '경고', text: '온라인 신청과 방문 신청 절차, 신청에 필요한 확인사항을 포함합니다.' },{ id: 'TEMP_chunk_002', section: '구비 서류', chars: 394, state: '정상', text: '본인 및 대리인 신청 시 필요한 서류와 양식 안내입니다.' }];
  return <><PageHeading eyebrow="INGESTION REVIEW" title="신규 URL 추가" description="적재하기 전에 파싱 원문과 청킹 결과를 건별로 검수합니다." /><div className="stepper"><span className="done"><i>1</i>URL 입력</span><b /><span className={previewed ? 'done' : ''}><i>2</i>파싱 결과</span><b /><span className={previewed ? 'active' : ''}><i>3</i>청크 검수</span><b /><span className={approved ? 'done' : ''}><i>4</i>승인·적재</span></div><section className="panel ingest-form"><div className="form-grid"><label className="wide"><span>신규 URL</span><input value={url} onChange={(e) => setUrl(e.target.value)} /></label><label><span>업무 카테고리</span><select><option>착오송금 반환 신청</option><option>예금자보호제도</option><option>예금보험금 안내</option></select></label><label><span>문서 유형</span><select><option>일반 안내 페이지</option><option>FAQ</option><option>신청·처리 페이지</option></select></label><label><span>Chunk Size</span><input type="number" value={chunkSize} onChange={(e) => setChunkSize(Number(e.target.value))} /></label><label><span>Overlap</span><input type="number" value={overlap} onChange={(e) => setOverlap(Number(e.target.value))} /></label></div><div className="form-actions"><p><b>테스트 환경</b>에만 미리보기와 적재가 수행됩니다.</p><button className="primary-btn" onClick={onPreview}>미리보기 생성</button></div></section>
    {previewed && <section className="preview-grid"><article className="panel parsed-preview"><div className="panel-head"><div><span className="section-kicker">PARSED CONTENT</span><h2>파싱 원문</h2></div><StatusBadge status="APPROVED" /></div><div className="parse-stats"><span>HTTP <b>200</b></span><span>본문 <b>1,402자</b></span><span>제거 태그 <b>34개</b></span></div><div className="paper"><h3>착오송금 반환지원 신청 안내</h3><p>착오송금인이 잘못 송금한 금전을 돌려받지 못한 경우 예금보험공사의 반환지원 제도를 신청할 수 있습니다.</p><h4>지원 대상</h4><p>송금일로부터 정해진 기간 내에 금융회사를 통한 반환 요청을 먼저 진행한 경우 신청할 수 있습니다.</p><h4>신청 절차</h4><p>온라인 또는 방문을 통해 신청하며 사실관계 확인에 필요한 서류를 제출합니다.</p></div></article><article className="panel chunk-preview"><div className="panel-head"><div><span className="section-kicker">CHUNK PREVIEW</span><h2>청킹 결과</h2></div><span className="warning-count">경고 1건</span></div>{previewChunks.map((chunk) => <div className="preview-chunk" key={chunk.id}><div><b>{chunk.id}</b><span className={chunk.state === '경고' ? 'mini-warning' : 'mini-ok'}>{chunk.state}</span></div><h3>{chunk.section}</h3><p>{chunk.text}</p><small>{chunk.chars}자 · Parent TEMP-001</small>{chunk.state === '경고' && <em>문장 경계와 목표 길이를 다시 확인해 주세요.</em>}</div>)}<div className="approve-bar"><button className="secondary-btn">반려</button><button className="secondary-btn" onClick={onPreview}>설정 변경 후 재처리</button><button className="primary-btn" disabled={approved} onClick={onApprove}>{approved ? '테스트 적재 요청 완료' : '승인 후 테스트 적재'}</button></div></article></section>}</>;
}

function ParameterTest({ topK, setTopK, denseWeight, setDenseWeight, candidateDepth, setCandidateDepth, rrfK, setRrfK, compared, onCompare }: any) {
  return <><PageHeading eyebrow="SEARCH LAB" title="파라미터 변경 전후 비교" description="확정된 Structured Hybrid 검색 방식 안에서 운영 설정과 후보 설정을 비교합니다." action={<button className="primary-btn" onClick={onCompare}>비교 실행</button>} /><section className="panel query-box"><label><span>대표 질문</span><input defaultValue="착오송금 반환지원 신청 시 비용이 발생하나요?" /></label><span className="env-tag">TEST INDEX</span></section><section className="config-compare"><article className="panel config-card baseline"><div className="config-title"><span>A</span><div><h2>현재 운영 설정</h2><p>kdic-hybrid-c-v3</p></div><StatusBadge status="ACTIVE" /></div><dl><div><dt>최종 Top-K</dt><dd>5</dd></div><div><dt>Dense / BM25</dt><dd>0.7 / 0.3</dd></div><div><dt>Candidate Depth</dt><dd>20</dd></div><div><dt>RRF K</dt><dd>10</dd></div><div><dt>Reranker</dt><dd>미사용</dd></div></dl></article><article className="panel config-card candidate"><div className="config-title"><span>B</span><div><h2>후보 설정</h2><p>저장 전 임시 설정</p></div><span className="test-badge">TEST</span></div><label><span>최종 Top-K <b>{topK}</b></span><input type="range" min="3" max="10" value={topK} onChange={(e) => setTopK(Number(e.target.value))} /></label><label><span>Dense 가중치 <b>{denseWeight / 100}</b></span><input type="range" min="50" max="90" step="5" value={denseWeight} onChange={(e) => setDenseWeight(Number(e.target.value))} /><small>BM25 {1 - denseWeight / 100}</small></label><label><span>Candidate Depth <b>{candidateDepth}</b></span><input type="range" min="10" max="50" step="5" value={candidateDepth} onChange={(e) => setCandidateDepth(Number(e.target.value))} /></label><label><span>RRF K <b>{rrfK}</b></span><input type="range" min="5" max="60" step="5" value={rrfK} onChange={(e) => setRrfK(Number(e.target.value))} /></label></article></section>
    {compared && <section className="panel result-compare"><div className="panel-head"><div><span className="section-kicker">RANK COMPARISON</span><h2>검색 결과 차이</h2></div><div className="latency"><span>A <b>384ms</b></span><span>B <b>{410 + candidateDepth * 3}ms</b></span></div></div><div className="result-table"><div className="result-row result-header"><span>순위</span><span>Chunk ID · 제목</span><span>Dense</span><span>BM25</span><span>Weighted RRF</span><span>변화</span><span>Gold</span></div>{mockResults.slice(0,topK).map((row, index) => <div className="result-row" key={row.id}><b>{index + 1}</b><span><strong>{row.id}</strong><small>{row.title}</small></span><span>{row.dense}</span><span>{row.bm25}</span><span>{(row.score + (denseWeight - 70) / 10000).toFixed(4)}</span><span className={index < 2 ? 'rank-up' : index === 4 ? 'rank-down' : 'rank-same'}>{index < 2 ? '↑ 1' : index === 4 ? '↓ 2' : '—'}</span><span>{row.gold ? <b className="gold-mark">GOLD</b> : '—'}</span></div>)}</div><div className="comparison-summary"><span><small>Recall@5</small><b>66.7% → 100%</b></span><span><small>Gold 회수</small><b>1개 → 2개</b></span><span><small>레이턴시</small><b className="cost-up">+{26 + candidateDepth * 3}ms</b></span><button className="secondary-btn">후보 설정 저장</button></div></section>}</>;
}

function Pipeline({ jobs, onTrigger }: { jobs: typeof initialJobs; onTrigger: (type: string) => void }) {
  const actions = [{title:'선택 페이지 재수집',desc:'선택한 URL의 최신 내용을 다시 수집합니다.',icon:'↻'},{title:'선택 데이터 재청킹',desc:'지정한 문서만 새로운 설정으로 청킹합니다.',icon:'▥'},{title:'선택 데이터 재임베딩',desc:'Structured 입력을 다시 벡터화합니다.',icon:'◈'},{title:'전체 데이터 재적재',desc:'전체 427개 청크를 테스트 인덱스에 적재합니다.',icon:'⇧'}];
  return <><PageHeading eyebrow="PIPELINE CONTROL" title="파이프라인 작업 관리" description="재수집·재청킹·재적재를 Job으로 실행하고 단계별 상태를 확인합니다." /><div className="risk-banner"><b>!</b><span><strong>안전한 테스트 모드</strong>현재 프로토타입의 실행 버튼은 운영 데이터를 변경하지 않고 Job만 생성합니다.</span></div><section className="action-grid">{actions.map((action) => <article className="panel action-card" key={action.title}><span>{action.icon}</span><h3>{action.title}</h3><p>{action.desc}</p><button onClick={() => onTrigger(action.title)}>작업 생성</button></article>)}</section><section className="panel jobs-panel"><div className="panel-head"><div><span className="section-kicker">JOB QUEUE</span><h2>작업 목록</h2></div><select><option>전체 상태</option><option>처리 중</option><option>실패</option></select></div><JobTable jobs={jobs} detailed /></section></>;
}

function Evaluation({ progress, done, onRun }: { progress: number; done: boolean; onRun: () => void }) {
  return <><PageHeading eyebrow="EVALUATION" title="테스트셋 기반 품질 비교" description="Evaluation DataSet v5의 Gold 청크로 운영 설정과 후보 설정을 정량 비교합니다." action={<button className="primary-btn" disabled={progress > 0 && progress < 100} onClick={onRun}>{progress > 0 && progress < 100 ? '평가 실행 중' : '평가 실행'}</button>} /><section className="panel eval-setup"><div className="form-grid"><label><span>평가 데이터셋</span><select><option>Evaluation DataSet v5 · 270문항</option></select></label><label><span>운영 설정</span><select><option>kdic-hybrid-c-v3</option></select></label><label><span>후보 설정</span><select><option>최근 파라미터 테스트 설정</option></select></label><label><span>평가 업무</span><select><option>6개 업무 전체</option></select></label></div></section>{progress > 0 && progress < 100 && <section className="panel evaluation-progress"><div><span>평가 실행 중</span><strong>{progress}%</strong></div><i><em style={{width:`${progress}%`}} /></i><p>{Math.round(270 * progress / 100)} / 270문항 완료 · 오류 0건</p></section>}{done && <><section className="eval-metrics"><Metric label="Hit@3" value="91.2%" detail="+3.0%p 개선" /><Metric label="Recall@5" value="81.5%" detail="+3.1%p 개선" tone="purple" /><Metric label="Complete@5" value="62.8%" detail="+4.6%p 개선" tone="cyan" /><Metric label="평균 검색시간" value="498ms" detail="+114ms 증가" tone="red" /></section><section className="panel verdict"><div><span className="verdict-icon">✓</span><div><span className="section-kicker">EVALUATION COMPLETE</span><h2>품질은 개선됐지만 지연시간 검토가 필요합니다.</h2><p>개선 38문항 · 악화 11문항 · 변화 없음 221문항</p></div></div><button className="secondary-btn">문항별 결과 보기</button></section></>}</>;
}
