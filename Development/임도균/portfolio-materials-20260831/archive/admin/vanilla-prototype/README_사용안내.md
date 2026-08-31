# KDIC RAG 관리자 UI — HTML/CSS/Vanilla JS

## 실행

`실행하기.bat` 또는 `index.html`을 더블클릭합니다.

별도의 Python, Node.js, Streamlit, React 설치가 필요하지 않습니다.

## 파일 구성

- `index.html`: 화면 진입점
- `styles.css`: 관리자 화면 디자인
- `app.js`: 메뉴, 검색, 토글, 시뮬레이션 동작
- `data.js`: 실제 87개 페이지와 427개 청크의 브라우저용 복사본

## 실제 동작

- 페이지와 청크 조회
- 제목·ID·URL 검색
- 6개 업무 필터
- 청크 본문 펼쳐보기
- 원문 URL 열기
- 업무별 실제 데이터 수 집계

## 시뮬레이션

- 비활성화
- 신규 URL 파싱·청킹 미리보기
- 파라미터 A/B 검색
- 재수집·재청킹·재임베딩
- 테스트셋 평가

시뮬레이션은 `data.js`나 원본 JSONL을 수정하지 않으며 챗봇 검색에도 영향을 주지 않습니다.

## 향후 실제 연동

백엔드 API가 연결되면 `app.js`의 시뮬레이션 함수들을 `fetch()` API 호출로 교체합니다. API 키는 HTML이나 JavaScript에 넣지 않고 Python 서버 환경변수에 보관해야 합니다.
