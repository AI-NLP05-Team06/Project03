# KDIC 최종 챗봇 + 검색 A/B·운영 반영 관리자 UI

## 가장 먼저 실행할 파일

`2026-08-24-KDIC-최종챗봇-관리자UI통합-v3-API키관리-Cloudflare-Colab.ipynb`

원본 노트북은 수정하지 않았으며, 관리자 자산을 통합한 별도 사본입니다.

## 실행 방법

1. 통합본 노트북을 Google Colab에서 엽니다.
2. 원본과 동일하게 셀을 위에서부터 실행합니다.
3. 안내에 따라 다음 자료를 업로드합니다.
   - KDIC 문서 ZIP
   - Dense Structured V2 임베딩 캐시
   - Fact Index JSON
4. HCX API 키는 Colab 보안 비밀에 `HCX`라는 이름으로 등록하고 노트북 액세스를 허용합니다.
5. 마지막 FastAPI·Cloudflare 셀을 실행합니다.
6. 출력되는 버튼을 사용합니다.
   - `KDIC HTML 챗봇 열기`
   - `KDIC 관리자 UI 열기`

관리자 링크는 일회용입니다. 처음 접속하면 8시간짜리 HttpOnly 관리자 세션 쿠키로 교환됩니다. 같은 브라우저에서는 Cloudflare 주소 뒤에 `/admin`을 붙여 다시 열 수 있습니다.

## 실제로 연결된 관리자 기능

- 최종 파이프라인 연결 상태
- 실제 런타임 검색·답변 상수
- Elasticsearch 클러스터 상태
- 실제 인덱스 목록
- 실제 인덱스 문서·청크 검색
- 벡터 필드를 제외한 문서 상세 조회
- 실제 챗봇 Job 이력
- 실제 최종 챗봇 전체 흐름 테스트
  - 질의분석
  - Hybrid 7:3 Min-Max
  - 다중질의 결합
  - BGE Reranker
  - Parent-Child 8192
  - 단일·동일업무 C안
  - 교차업무 D-C 2Call
  - Fact Index
  - 검증된 Action Link
  - 사용자 친화 답변 근거
- 평가데이터셋 XLSX·CSV 업로드 및 Gold 연결 검증
- 현재 운영값(A)과 후보 파라미터(B)의 일괄 검색 A/B 평가
  - Hit@3, Recall@5, MRR@10, MAP@10, Complete@5
  - nDCG@5, Precision@5, F1@5, 평균 검색 지연시간
- 검색 시점 파라미터 초안 저장 및 승인 반영
- 청크 추가·제거 초안, 신규 청크 자동 임베딩 및 운영 인덱스 반영
- 반영 전 런타임 스냅샷을 이용한 롤백
- HCX API 키 상태·지문 확인, 임베딩/답변 연결 테스트 및 런타임 교체

## 현재 비활성화된 기능

이번 통합본은 `STAGED_WRITE` 모드입니다. 아래처럼 별도 파이프라인이 필요한 기능은 아직 실행하지 않습니다.

- 신규 URL 실제 파싱·청킹
- 재수집·재적재
- 인덱스 삭제와 Alias 전환
- Chunk Size·청킹 방식·임베딩 모델의 즉시 변경

청크 추가·제거와 검색 시점 파라미터는 `초안 → 검증/A-B → 승인 → 반영 → 롤백`으로 처리합니다.

## API 키 보안

- HCX API 키는 Colab 서버에서만 사용합니다.
- 관리자 HTML과 JavaScript에는 API 키가 포함되지 않습니다.
- 관리자 화면은 키 설정 여부만 간접적으로 확인합니다.
- Elasticsearch 벡터와 인증정보도 관리자 응답에서 제외합니다.
- 관리자 인증은 HCX 키와 별도의 일회성 토큰과 HttpOnly 세션을 사용합니다.
- 관리자 UI는 기존 키를 조회할 수 없고, 새 키도 응답·브라우저 저장소에 저장하지 않습니다.
- 초기 키는 Colab 보안 비밀의 `HCX`에서만 읽습니다. 관리자에서 교체한 키는 현재 Colab 런타임에만 적용되며, 다음 런타임에도 유지하려면 Colab 보안 비밀 `HCX`를 갱신합니다.
- `KDIC_output ZIP`, Dense 캐시, Fact Index는 최초 통합 업로드 셀에서 한번에 모두 선택합니다. 뒤의 셀은 업로드 창을 다시 열지 않습니다.
- Gradio는 사용하지 않으며, HTML/CSS/Vanilla JS + FastAPI + Cloudflare Tunnel로 실행됩니다.

## 소스 구성

- `index.html`, `styles.css`, `admin.js`: 관리자 UI 원본
- `kdic_admin_bundled.html`: Colab에 내장되는 단일 HTML
- `kdic_admin_extension.py`: 관리자 인증·평가·초안·반영·롤백 API 확장
- `build_integrated_notebook.js`: 원본 노트북을 보존하면서 통합본을 다시 만드는 스크립트

## 운영상 주의점

1. 반영 버튼은 실행 중인 챗봇 Job이 없을 때만 동작합니다.
2. 신규 청크는 HCX BGE-M3로 임베딩한 뒤 같은 Elasticsearch 인덱스와 메모리 Parent-Child 맵에 함께 반영됩니다.
3. 현재 변경 이력은 Colab 런타임 메모리에 저장되므로 런타임 종료 후에도 유지하려면 외부 DB 또는 Drive 저장소를 추가해야 합니다.
4. A/B 평가는 검색 계층만 비교하며 질의분해와 최종 답변 생성 품질 평가는 별도 단계입니다.
