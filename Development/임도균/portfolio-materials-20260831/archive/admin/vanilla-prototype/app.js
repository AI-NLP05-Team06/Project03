(function () {
  'use strict';

  const documents = Array.isArray(window.KDIC_DOCUMENTS) ? window.KDIC_DOCUMENTS : [];
  const chunks = Array.isArray(window.KDIC_CHUNKS) ? window.KDIC_CHUNKS : [];
  const root = document.getElementById('app');

  const navItems = [
    ['dashboard', '▦', '대시보드', '운영 현황'],
    ['data', '▤', '데이터 관리', '페이지·청크'],
    ['new', '＋', '신규 데이터', '파싱·청킹 검수'],
    ['parameters', '⇄', '파라미터 테스트', '설정 A/B 비교'],
    ['pipeline', '↻', '파이프라인', '재수집·재적재'],
    ['evaluation', '◎', '평가 관리', '테스트셋 평가']
  ];
  const domains = ['전체', '예금자보호제도', '예금보험금 안내', '고객 미수령금 신청', '착오송금 반환 신청', '채무조정 안내', '은닉재산 신고'];
  const state = {
    view: 'dashboard',
    dark: localStorage.getItem('kdic-admin-theme') === 'dark',
    selectedDoc: documents[0]?.doc_id || '',
    expandedChunk: '',
    query: '',
    domain: '전체',
    inactive: new Set(),
    previewed: false,
    approved: false,
    url: 'https://www.kdic.or.kr/example',
    chunkSize: 500,
    overlap: 50,
    topK: 5,
    denseWeight: 70,
    candidateDepth: 20,
    rrfK: 10,
    compared: false,
    evalProgress: 0,
    evalDone: false,
    jobs: [
      { id: 'JOB-20260824-003', type: '선택 페이지 재수집', target: 'DP-001 외 2건', status: 'SUCCEEDED', progress: 100, step: '완료', started: '오늘 09:42' },
      { id: 'JOB-20260824-002', type: '테스트 인덱스 생성', target: '427개 청크', status: 'RUNNING', progress: 68, step: '임베딩', started: '오늘 09:18' },
      { id: 'JOB-20260823-008', type: '평가 실행', target: 'v5 · 270문항', status: 'FAILED', progress: 74, step: '평가', started: '어제 17:35' }
    ]
  };

  const mockResults = [
    ['MT-013_chunk_001', '반환지원 비용', 1, 2, '.0458', true],
    ['MT-013_chunk_003', '회수금액별 비용률', 4, 1, '.0412', true],
    ['MT-004_chunk_002', '반환지원 신청 대상', 2, 6, '.0365', false],
    ['MT-005_chunk_001', '신청 가능 기간', 7, 3, '.0317', false],
    ['BI-002_chunk_004', '예금보험금 지급절차', 5, 8, '.0248', false]
  ];

  function esc(value) {
    return String(value ?? '').replace(/[&<>'"]/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' })[ch]);
  }
  function attr(value) { return esc(value).replace(/`/g, '&#96;'); }
  function statusBadge(status) {
    const labels = { ACTIVE: '운영 중', REVIEW_REQUIRED: '검수 필요', RUNNING: '처리 중', SUCCEEDED: '완료', FAILED: '실패', INACTIVE: '비활성', QUEUED: '대기', APPROVED: '승인', DRAFT: '초안' };
    return `<span class="status status-${status.toLowerCase()}"><i></i>${labels[status] || esc(status)}</span>`;
  }
  function heading(eyebrow, title, description, action = '') {
    return `<div class="page-heading"><div><span>${eyebrow}</span><h1>${title}</h1><p>${description}</p></div>${action}</div>`;
  }
  function metric(label, value, detail, tone = 'blue', demo = false) {
    return `<article class="metric metric-${tone}${demo ? ' metric-demo' : ''}"><div class="metric-top"><span>${label}</span><b>↗</b></div><strong>${value}</strong><small>${detail}</small></article>`;
  }
  function domainCounts() {
    const counts = {};
    documents.forEach(doc => { counts[doc.business_function] = (counts[doc.business_function] || 0) + 1; });
    return counts;
  }
  function chunkCountMap() {
    const counts = {};
    chunks.forEach(chunk => { counts[chunk.parent_doc_id] = (counts[chunk.parent_doc_id] || 0) + 1; });
    return counts;
  }
  function showToast(message, error = false) {
    document.querySelector('.toast')?.remove();
    const toast = document.createElement('div');
    toast.className = `toast${error ? ' error' : ''}`;
    toast.innerHTML = `<b>${error ? '!' : '✓'}</b>${esc(message)}`;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);
  }

  function shell(content) {
    const nav = navItems.map(([id, icon, label, caption]) => `
      <button class="nav-item ${state.view === id ? 'active' : ''}" data-view="${id}">
        <span class="nav-icon">${icon}</span><span><b>${label}</b><small>${caption}</small></span>
      </button>`).join('');
    const currentLabel = navItems.find(item => item[0] === state.view)?.[2] || '';
    return `<div class="admin-app ${state.dark ? 'dark' : ''}">
      <aside class="sidebar">
        <div class="brand"><div class="brand-mark">K</div><div><strong>KDIC RAG</strong><span>ADMIN CONSOLE</span></div></div>
        <nav aria-label="관리자 메뉴"><p class="nav-label">WORKSPACE</p>${nav}</nav>
        <div class="sidebar-config"><p>기준 검색 설정</p><strong>Structured Hybrid</strong><span>Dense 0.7 · BM25 0.3</span><span>RRF K=10 · Top-K 5</span><span class="simulation">화면 프로토타입</span></div>
        <div class="profile"><div class="avatar">관</div><div><strong>관리자</strong><span>RAG 운영팀</span></div><button aria-label="프로필 메뉴">⋮</button></div>
      </aside>
      <main class="main">
        <header class="topbar"><div><span class="crumb">KDIC RAG Admin /</span><strong>${currentLabel}</strong></div><div class="system-pills"><span><i class="green-dot"></i> 실제 JSONL 조회</span><span><i class="green-dot"></i> 임베딩 데이터 · 1024차원</span><span class="top-note">챗봇 미연동</span><button data-action="theme" aria-label="테마 변경">${state.dark ? '☀' : '☾'}</button></div></header>
        <div class="content">${content}</div>
        <footer><span>표시 설정 <b>kdic-hybrid-c-v3</b></span><span>페이지 ${documents.length} · 청크 ${chunks.length}</span><span>HTML/CSS/Vanilla JS</span></footer>
      </main>
    </div>`;
  }

  function jobTable(jobs, detailed = false) {
    return `<div class="job-table ${detailed ? 'detailed' : ''}">
      <div class="job-row job-header"><span>Job ID</span><span>작업</span><span>대상</span><span>상태</span>${detailed ? '<span>현재 단계</span>' : ''}<span>진행률</span><span>시작</span></div>
      ${jobs.map(job => `<div class="job-row"><b>${job.id}</b><span>${job.type}</span><span>${job.target}</span>${statusBadge(job.status)}${detailed ? `<span>${job.step}</span>` : ''}<div class="progress-cell"><i><em style="width:${job.progress}%"></em></i><small>${job.progress}%</small></div><span>${job.started}</span></div>`).join('')}
    </div>`;
  }

  function dashboard() {
    const counts = domainCounts();
    const countCards = Object.entries(counts).map(([domain, count]) => `<div><span>${esc(domain)}</span><strong>${count}</strong><small>페이지</small></div>`).join('');
    return `${heading('OVERVIEW', '관리자 대시보드', '실제 데이터 현황과 검색·파이프라인 관리 화면을 확인합니다.', '<button class="primary-btn" data-view="new">＋ 신규 URL 추가</button>')}
      <section class="metric-grid">
        ${metric('적재 페이지', documents.length, '6개 업무 도메인')}
        ${metric('검색 청크', chunks.length, '실제 청크 데이터', 'purple')}
        ${metric('평가 데이터셋', '270', '표시값 · 아직 미연동', 'cyan', true)}
        ${metric('실패 Job', '1', '시뮬레이션 작업', 'red', true)}
      </section>
      <section class="dashboard-grid">
        <article class="panel quality-panel"><div class="panel-head"><div><span class="section-kicker">DATA DISTRIBUTION</span><h2>업무별 페이지 분포</h2></div><button data-view="data">데이터 관리 →</button></div><div class="quality-row">${countCards}</div><div class="spark-bars">${[42,58,49,66,61,74,68,83,77,92,86,96].map(h => `<i style="height:${h}%"></i>`).join('')}</div></article>
        <article class="panel config-panel"><div class="panel-head"><div><span class="section-kicker">REFERENCE CONFIG</span><h2>현재 선택한 검색안</h2></div><span class="simulation">표시 설정</span></div><dl><div><dt>검색 방식</dt><dd>Structured Hybrid</dd></div><div><dt>Dense / BM25</dt><dd>0.7 / 0.3</dd></div><div><dt>RRF K</dt><dd>10</dd></div><div><dt>Candidate Depth</dt><dd>각 20</dd></div><div><dt>최종 Top-K</dt><dd>5</dd></div><div><dt>Reranker</dt><dd>미사용</dd></div></dl></article>
      </section>
      <section class="panel jobs-panel"><div class="panel-head"><div><span class="section-kicker">RECENT JOBS · SIMULATION</span><h2>최근 파이프라인</h2></div><button data-view="pipeline">전체 작업 보기 →</button></div>${jobTable(state.jobs.slice(0, 3))}</section>`;
  }

  function dataManagement() {
    const counts = chunkCountMap();
    const q = state.query.trim().toLowerCase();
    const filtered = documents.filter(doc => {
      const text = `${doc.doc_id} ${doc.title} ${doc.source_url}`.toLowerCase();
      return (!q || text.includes(q)) && (state.domain === '전체' || doc.business_function === state.domain);
    });
    const currentDoc = filtered.find(doc => doc.doc_id === state.selectedDoc) || filtered[0] || null;
    const currentChunks = chunks.filter(chunk => chunk.parent_doc_id === currentDoc?.doc_id);
    const rows = filtered.slice(0, 60).map(doc => `<button class="doc-row ${currentDoc?.doc_id === doc.doc_id ? 'selected' : ''}" data-action="select-doc" data-id="${attr(doc.doc_id)}"><span class="doc-id">${esc(doc.doc_id)}</span><span class="doc-main"><b>${esc(doc.title)}</b><small>${esc(doc.business_function)}</small></span><span class="chunk-count">${counts[doc.doc_id] || 0}</span>${statusBadge(state.inactive.has(doc.doc_id) ? 'INACTIVE' : 'ACTIVE')}</button>`).join('') || '<div class="empty-list">조건에 맞는 페이지가 없습니다.</div>';
    const chunkCards = currentChunks.map(chunk => {
      const open = state.expandedChunk === chunk.chunk_id;
      return `<div class="chunk-card ${open ? 'open' : ''}"><button data-action="toggle-chunk" data-id="${attr(chunk.chunk_id)}"><span><b>${esc(chunk.chunk_id)}</b><small>${esc(chunk.section_title || chunk.title)}</small></span><span>${String(chunk.content || '').length}자</span><i>${open ? '−' : '+'}</i></button>${open ? `<div class="chunk-body"><p>${esc(chunk.content)}</p><dl><div><dt>Parent</dt><dd>${esc(chunk.parent_doc_id)}</dd></div><div><dt>업무</dt><dd>${esc(chunk.business_function)}</dd></div></dl></div>` : ''}</div>`;
    }).join('') || '<div class="empty-list">연결된 청크가 없습니다.</div>';
    const detail = currentDoc ? `<div class="notice"><b>안내:</b> 비활성화 버튼은 화면에서만 상태를 바꾸며 원본·청크·챗봇 검색에는 영향을 주지 않습니다.</div><div class="detail-head"><div><span class="doc-id">${esc(currentDoc.doc_id)}</span><h2>${esc(currentDoc.title)}</h2><p>${esc(currentDoc.business_function)} · ${esc(currentDoc.page_type || 'page')}</p></div><div class="detail-actions"><a href="${attr(currentDoc.source_url)}" target="_blank" rel="noreferrer">원문 열기 ↗</a><button class="danger-btn ${state.inactive.has(currentDoc.doc_id) ? 'active' : ''}" data-action="toggle-inactive" data-id="${attr(currentDoc.doc_id)}">${state.inactive.has(currentDoc.doc_id) ? '다시 활성화' : '비활성화(시연)'}</button></div></div><div class="meta-strip"><span><small>연결 청크</small><b>${currentChunks.length}개</b></span><span><small>품질 상태</small><b>${currentDoc.quality?.status === 'pass' ? 'PASS' : '검토'}</b></span><span><small>최종 수집</small><b>${esc(String(currentDoc.parsed_at || '2026-07-21').slice(0, 10))}</b></span></div><h3 class="sub-title">연결된 청크</h3><div class="chunk-stack">${chunkCards}</div>` : '<div class="empty-state">페이지를 선택해 주세요.</div>';
    return `${heading('DATA CATALOG', '적재 데이터 관리', '실제 페이지·청크·메타데이터 복사본을 조회합니다.', '<button class="secondary-btn" data-action="job" data-type="선택 페이지 재수집">↻ 선택 페이지 재수집(시연)</button>')}
      <section class="data-layout"><article class="panel document-list"><div class="filter-row"><input id="search-docs" value="${attr(state.query)}" placeholder="제목·ID·URL 검색"><select id="domain-filter">${domains.map(item => `<option ${state.domain === item ? 'selected' : ''}>${item}</option>`).join('')}</select></div><div class="list-summary"><b>${filtered.length}개 페이지</b><span>최대 60개 표시</span></div><div class="doc-scroll">${rows}</div></article><article class="panel doc-detail">${detail}</article></section>`;
  }

  function newData() {
    const previewChunks = [
      ['TEMP_chunk_000', '지원 대상', 486, '정상', '착오송금 반환지원 신청 대상과 기본 요건에 관한 파싱 결과입니다.'],
      ['TEMP_chunk_001', '신청 절차', 522, '경고', '온라인 신청과 방문 신청 절차, 신청에 필요한 확인사항을 포함합니다.'],
      ['TEMP_chunk_002', '구비 서류', 394, '정상', '본인 및 대리인 신청 시 필요한 서류와 양식 안내입니다.']
    ];
    const preview = state.previewed ? `<section class="preview-grid"><article class="panel parsed-preview"><div class="panel-head"><div><span class="section-kicker">PARSED CONTENT · SIMULATION</span><h2>파싱 원문</h2></div>${statusBadge('APPROVED')}</div><div class="parse-stats"><span>HTTP <b>200</b></span><span>본문 <b>1,402자</b></span><span>제거 태그 <b>34개</b></span></div><div class="paper"><h3>착오송금 반환지원 신청 안내</h3><p>착오송금인이 잘못 송금한 금전을 돌려받지 못한 경우 예금보험공사의 반환지원 제도를 신청할 수 있습니다.</p><h4>지원 대상</h4><p>금융회사를 통한 반환 요청을 먼저 진행한 경우 신청할 수 있습니다.</p><h4>신청 절차</h4><p>온라인 또는 방문을 통해 신청하며 사실관계 확인에 필요한 서류를 제출합니다.</p></div></article><article class="panel chunk-preview"><div class="panel-head"><div><span class="section-kicker">CHUNK PREVIEW · SIMULATION</span><h2>청킹 결과</h2></div><span class="warning-count">경고 1건</span></div>${previewChunks.map(([id, section, chars, status, text]) => `<div class="preview-chunk"><div><b>${id}</b><span class="${status === '경고' ? 'mini-warning' : 'mini-ok'}">${status}</span></div><h3>${section}</h3><p>${text}</p><small>${chars}자 · overlap ${state.overlap}</small>${status === '경고' ? '<em>권장 크기보다 22자 깁니다. 문장 경계를 확인하세요.</em>' : ''}</div>`).join('')}<div class="approve-bar"><button class="secondary-btn" data-action="preview">다시 생성</button><button class="primary-btn" data-action="approve" ${state.approved ? 'disabled' : ''}>${state.approved ? '승인 대기 등록됨' : '검수 승인·테스트 적재'}</button></div></article></section>` : '';
    return `${heading('INGESTION REVIEW', '신규 URL 추가', '적재 전에 파싱 원문과 청킹 결과를 확인하는 화면입니다. 현재는 시뮬레이션입니다.')}
      <div class="stepper"><span class="done"><i>1</i>URL 입력</span><b></b><span class="${state.previewed ? 'done' : ''}"><i>2</i>파싱 결과</span><b></b><span class="${state.previewed ? 'active' : ''}"><i>3</i>청크 검수</span><b></b><span class="${state.approved ? 'done' : ''}"><i>4</i>승인·적재</span></div>
      <section class="panel ingest-form"><div class="form-grid"><label class="wide"><span>신규 URL</span><input id="new-url" value="${attr(state.url)}"></label><label><span>업무 카테고리</span><select><option>착오송금 반환 신청</option><option>예금자보호제도</option><option>예금보험금 안내</option></select></label><label><span>문서 유형</span><select><option>일반 안내 페이지</option><option>FAQ</option><option>신청·처리 페이지</option></select></label><label><span>Chunk Size</span><input id="chunk-size" type="number" value="${state.chunkSize}"></label><label><span>Overlap</span><input id="overlap" type="number" value="${state.overlap}"></label></div><div class="form-actions"><p><b>시뮬레이션:</b> 입력한 URL에 실제 요청을 보내지 않습니다.</p><button class="primary-btn" data-action="preview">미리보기 생성</button></div></section>${preview}`;
  }

  function parameterTest() {
    const rows = mockResults.map((r, index) => `<div class="result-row"><b>${index + 1}</b><span><strong>${r[0]}</strong><small>${r[1]}</small></span><span>${r[2]}</span><span>${r[3]}</span><span>${r[4]}</span><span class="rank-${index < 2 ? 'up' : 'same'}">${index < 2 ? '↑' : '—'}</span><span>${r[5] ? '<small class="gold-mark">GOLD</small>' : '—'}</span></div>`).join('');
    const result = state.compared ? `<section class="panel result-compare"><div class="panel-head"><div><span class="section-kicker">COMPARISON · SIMULATION</span><h2>후보 B 검색 결과</h2></div><div class="latency"><span>기준 A 384ms</span><span>후보 B 416ms</span></div></div><div class="result-table"><div class="result-row result-header"><span>순위</span><span>청크</span><span>Dense</span><span>BM25</span><span>RRF</span><span>변동</span><span>Gold</span></div>${rows}</div><div class="comparison-summary"><span><small>Gold Hit@3</small><b>2 / 3</b></span><span><small>예상 지연 증가</small><b class="cost-up">+32ms</b></span><button class="primary-btn" data-action="job" data-type="후보 설정 테스트 반영">후보 설정 저장(시연)</button></div></section>` : '';
    return `${heading('SEARCH LAB', '파라미터 변경 전후 비교', '동일 질문에서 기준 설정과 후보 설정을 나란히 비교합니다. 결과는 시연용입니다.')}
      <section class="panel query-box"><label><span>대표 질문</span><input value="착오송금 반환지원 비용은 얼마인가요?"></label><span class="env-tag">TEST INDEX</span><button class="primary-btn" data-action="compare">두 설정 비교</button></section>
      <section class="config-compare"><article class="panel config-card"><div class="config-title"><span>A</span><div><h2>현재 기준 설정</h2><p>고정된 기준선</p></div>${statusBadge('ACTIVE')}</div><dl><div><dt>Dense / BM25</dt><dd>0.7 / 0.3</dd></div><div><dt>Candidate Depth</dt><dd>각 20</dd></div><div><dt>RRF K</dt><dd>10</dd></div><div><dt>Top-K</dt><dd>5</dd></div></dl></article>
      <article class="panel config-card candidate"><div class="config-title"><span>B</span><div><h2>후보 설정</h2><p>브라우저에서 조정</p></div><span class="test-badge">TEST ONLY</span></div>
      ${slider('denseWeight', 'Dense 가중치', state.denseWeight, 0, 100, `${state.denseWeight}% / BM25 ${100 - state.denseWeight}%`)}${slider('candidateDepth', 'Candidate Depth', state.candidateDepth, 5, 50, `${state.candidateDepth}`)}${slider('rrfK', 'RRF K', state.rrfK, 1, 60, `${state.rrfK}`)}${slider('topK', '최종 Top-K', state.topK, 1, 10, `${state.topK}`)}</article></section>${result}`;
  }
  function slider(key, label, value, min, max, output) {
    return `<label><span>${label}<b data-output="${key}">${output}</b></span><input type="range" data-slider="${key}" value="${value}" min="${min}" max="${max}"></label>`;
  }

  function pipeline() {
    const actions = [
      ['↻', '선택 페이지 재수집', '선택한 페이지의 원문을 다시 수집합니다.'],
      ['✂', '전체 데이터 재청킹', '적재 시점 파라미터를 적용해 다시 나눕니다.'],
      ['◇', '전체 임베딩 재생성', '청크 임베딩을 다시 생성합니다.'],
      ['⚙', '전체 재적재', '수집부터 인덱싱까지 순서대로 실행합니다.']
    ];
    return `${heading('PIPELINE CONTROL', '재수집·재적재 관리', '데이터 파이프라인 작업을 생성하고 상태를 확인합니다. 현재 버튼은 시뮬레이션입니다.')}
      <div class="risk-banner"><b>!</b><span><strong>실제 원본이나 인덱스를 변경하지 않습니다.</strong>향후 관리자 API와 작업 큐가 연결되면 승인 단계를 추가해야 합니다.</span></div>
      <section class="action-grid">${actions.map(([icon, title, desc]) => `<article class="panel action-card"><span>${icon}</span><h3>${title}</h3><p>${desc}</p><button data-action="job" data-type="${title}">작업 생성(시연)</button></article>`).join('')}</section>
      <section class="panel jobs-panel"><div class="panel-head"><div><span class="section-kicker">JOB HISTORY · SIMULATION</span><h2>작업 상태</h2></div><span class="simulation">API 미연동</span></div>${jobTable(state.jobs, true)}</section>`;
  }

  function evaluation() {
    const progress = state.evalProgress;
    const progressPanel = progress > 0 ? `<section class="panel evaluation-progress"><div><span>${progress < 100 ? '평가 실행 중' : '평가 완료'}</span><strong>${progress}%</strong></div><i><em style="width:${progress}%"></em></i><p>${progress < 100 ? `${Math.round(270 * progress / 100)} / 270문항 · 검색 결과 계산 중` : '270 / 270문항 · 시연 결과 생성 완료'}</p></section>` : '';
    const results = state.evalDone ? `<section class="eval-metrics eval-result">${metric('Hit@3', '91.2%', '기준 대비 +2.8%p', 'blue', true)}${metric('Recall@5', '78.4%', '기준 대비 +3.1%p', 'purple', true)}${metric('nDCG@5', '82.1%', '기준 대비 +1.6%p', 'cyan', true)}${metric('평균 지연', '384ms', '기준 대비 +24ms', 'red', true)}</section><section class="panel verdict"><div><span class="verdict-icon">✓</span><span><span class="section-kicker">SIMULATION VERDICT</span><h2>후보 설정은 검토 대상으로 표시되었습니다.</h2><p>실제 평가 결과가 아니라 UI 동작 확인용 값입니다.</p></span></div><button class="secondary-btn">결과 내보내기(시연)</button></section>` : '';
    return `${heading('EVALUATION', '테스트셋 품질 비교', '평가데이터셋으로 기준 설정과 후보 설정을 비교하는 화면입니다.')}
      <section class="panel eval-setup"><div class="form-grid"><label><span>평가데이터셋</span><select><option>Evaluation DataSet v5</option></select></label><label><span>평가 문항</span><input value="270문항" disabled></label><label><span>기준 설정</span><select><option>Structured Hybrid v3</option></select></label><label><span>후보 설정</span><select><option>Candidate B</option></select></label></div><div class="form-actions"><p><b>시뮬레이션:</b> 실제 평가기는 아직 연결되지 않았습니다.</p><button class="primary-btn" data-action="run-eval" ${progress > 0 && progress < 100 ? 'disabled' : ''}>${progress > 0 && progress < 100 ? '실행 중…' : '변경 전후 평가 실행'}</button></div></section>${progressPanel}${results}`;
  }

  function render() {
    const views = { dashboard, data: dataManagement, new: newData, parameters: parameterTest, pipeline, evaluation };
    root.innerHTML = shell(views[state.view]());
  }

  function triggerJob(type) {
    const id = `JOB-20260824-${String(state.jobs.length + 4).padStart(3, '0')}`;
    state.jobs.unshift({ id, type, target: type.includes('전체') ? `${documents.length}개 페이지` : '선택 대상', status: 'QUEUED', progress: 0, step: '대기', started: '방금 전' });
    render();
    showToast(`${type} 작업을 화면에 추가했습니다. 실제 API는 연결되지 않았습니다.`);
  }

  root.addEventListener('click', event => {
    const target = event.target.closest('button, a');
    if (!target) return;
    if (target.dataset.view) { state.view = target.dataset.view; render(); return; }
    const action = target.dataset.action;
    if (!action) return;
    if (action === 'theme') { state.dark = !state.dark; localStorage.setItem('kdic-admin-theme', state.dark ? 'dark' : 'light'); render(); }
    if (action === 'select-doc') { state.selectedDoc = target.dataset.id; state.expandedChunk = ''; render(); }
    if (action === 'toggle-chunk') { state.expandedChunk = state.expandedChunk === target.dataset.id ? '' : target.dataset.id; render(); }
    if (action === 'toggle-inactive') {
      const id = target.dataset.id;
      state.inactive.has(id) ? state.inactive.delete(id) : state.inactive.add(id);
      render();
      showToast('화면 상태만 변경했습니다. 청크와 챗봇 검색 결과는 변경되지 않습니다.');
    }
    if (action === 'preview') { state.previewed = true; state.approved = false; render(); showToast('시연용 파싱·청킹 미리보기를 생성했습니다.'); }
    if (action === 'approve') { state.approved = true; triggerJob('신규 URL 테스트 적재'); }
    if (action === 'compare') { state.compared = true; render(); showToast('시연용 비교 결과를 표시했습니다.'); }
    if (action === 'job') triggerJob(target.dataset.type || '관리자 작업');
    if (action === 'run-eval') runEvaluation();
  });

  root.addEventListener('input', event => {
    const el = event.target;
    if (el.id === 'search-docs') {
      const caret = el.selectionStart;
      state.query = el.value;
      render();
      const next = document.getElementById('search-docs');
      next?.focus();
      next?.setSelectionRange(caret, caret);
    }
    if (el.id === 'new-url') state.url = el.value;
    if (el.id === 'chunk-size') state.chunkSize = Number(el.value);
    if (el.id === 'overlap') state.overlap = Number(el.value);
    if (el.dataset.slider) {
      const key = el.dataset.slider;
      state[key] = Number(el.value);
      const output = document.querySelector(`[data-output="${key}"]`);
      if (output) output.textContent = key === 'denseWeight' ? `${state.denseWeight}% / BM25 ${100 - state.denseWeight}%` : String(state[key]);
    }
  });
  root.addEventListener('change', event => {
    if (event.target.id === 'domain-filter') { state.domain = event.target.value; render(); }
  });

  function runEvaluation() {
    state.evalProgress = 1;
    state.evalDone = false;
    render();
    const timer = setInterval(() => {
      state.evalProgress = Math.min(100, state.evalProgress + 8);
      if (state.evalProgress === 100) { clearInterval(timer); state.evalDone = true; }
      render();
    }, 180);
  }

  render();
})();
