# KDIC BGE-M3 Dense 검색 평가기

이 평가기는 챗봇 답변을 생성하지 않습니다. 평가 질문을 BGE-M3로 임베딩하고,
`KDIC_output.zip`에 저장된 427개 Dense 청크 벡터와 코사인 유사도를 계산한 뒤
Top-10 청크를 Gold 청크와 비교합니다.

## 현재 고정 조건

- 임베딩 모델: `bge-m3`
- 검색 방식: Dense cosine similarity
- 최종 검색 수: Top-10
- 업무 필터: Gold 업무를 사용한 사전 필터
- 최소 유사도: 사용하지 않음
- 평가 질문: `검색평가대상=Y`

업무 자동분류 성능이 검색 성능에 섞이지 않도록 첫 검색 실험에서는 Gold 업무를
사용합니다. 추후 `--no-domain-filter`로 전체 검색 결과도 별도로 확인할 수 있습니다.

## 1. 사전 점검

API를 사용하지 않고 데이터 연결만 검사합니다.

```powershell
python evaluate_bge_m3_dense.py `
  --dataset "C:\Users\임도균\Desktop\평가데이터셋_검색평가지표용.xlsx" `
  --kdic-zip "C:\Users\임도균\Downloads\KDIC_RAG_V4_7_INTERACTIVE_CHAT (2)\KDIC_output.zip" `
  --output-dir ".\results_dry" `
  --dry-run
```

또는 `run_dry_test.bat`을 실행합니다.

## 2. 실제 평가

먼저 필요한 라이브러리를 설치합니다.

```powershell
python -m pip install -r requirements.txt
```

`run_eval.bat`을 실행하면 API 키를 화면에 표시하지 않고 입력받습니다. 환경 변수로
미리 등록해도 됩니다.

```powershell
$env:HCX_API_KEY="발급받은 API 키"
.\run_eval.bat
```

## 산출물

- `question_results.csv`: 질문별 Top-10, 점수 및 개별 지표
- `summary.json`: 전체 평균과 업무별 평균
- `summary_by_domain.csv`: 업무별 비교표
- `run_config.json`: 실행 조건
- `query_embedding_cache.jsonl`: 질문 임베딩 캐시
- `dry_run_report.json`: 데이터 연결 점검 결과
- `missing_gold.csv`: 존재하지 않는 Gold 청크가 있을 때 생성

질문 임베딩은 캐시되므로 같은 질문을 다시 실행해도 API를 재호출하지 않습니다.

## 지표 정의

- Hit@3: Top-3에 전체 Gold가 하나라도 있으면 1
- Recall@5: Top-5에서 찾은 Gold 수 / 전체 Gold 수
- MRR@10: 첫 Gold 순위의 역수
- MAP@10: 질문별 AP@10의 평균. AP 분모는 `min(Gold 수, 10)`
- Complete@5: `multi_chunk_required=Y`, 주요 Gold 2~5개인 질문에만 적용
- nDCG@5: Primary 관련도 2, Supporting 관련도 1
- Precision@5: Top-5에서 찾은 Gold 수 / 5
- F1@5: Precision@5와 Recall@5의 조화평균

## 검수 상태

기본 실행은 `pending_review`를 포함한 모든 검색평가대상 질문을 평가합니다.
최종 보고용으로 승인된 질문만 평가하려면 `--approved-only`를 추가합니다.
