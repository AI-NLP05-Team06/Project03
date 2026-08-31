# KDIC 검색평가 v4.1 Colab 비교 패키지

## 비교 대상

1. BM25-Nori Discard
2. BGE-M3 Dense
3. BGE-M3 Sparse
4. Hybrid A — Dense 0.85 + Nori 0.15, Weighted RRF
5. Hybrid A + BGE Reranker v2-m3

모든 방식은 같은 `Evaluation_DataSet_v4_1_SearchReady.xlsx`와
`KDIC_output.zip`을 사용합니다. 평가기는 `검색평가용` 시트에서
`검색평가대상=Y`인 121개 질문만 불러오며, 업무 필터를 적용합니다.

## 실행 순서

각 검색 방식 폴더의 Colab 노트북을 하나씩 엽니다.

1. 노트북의 파일 업로드 단계에서 다음 3개 파일을 업로드합니다.
   - 해당 방식 폴더의 `*_평가기.zip`
   - `06_공통입력/Evaluation_DataSet_v4_1_SearchReady.xlsx`
   - `06_공통입력/KDIC_output.zip`
2. 위에서 아래로 셀을 실행합니다.
3. 마지막 셀에서 생성되는 결과 ZIP을 내려받습니다.
4. 다섯 결과 ZIP을 `07_결과비교_대시보드/KDIC_검색평가_비교대시보드.html`에
   모두 업로드합니다.

API 키가 필요한 노트북에서는 Colab Secret의 `HCX_API_KEY`를 사용합니다.
코드에 키 값을 직접 적지 마세요.

## 방식별 실행 환경

| 방식 | HCX API | GPU | 비고 |
|---|---:|---:|---|
| BM25-Nori Discard | 불필요 | 불필요 | Java/Lucene Nori 사용 |
| BGE-M3 Dense | 필요 | 불필요 | 질의 임베딩 API 호출 |
| BGE-M3 Sparse | 불필요 | 권장 | BAAI/bge-m3 모델 실행 |
| Hybrid A | 필요 | 불필요 | Dense + Nori + Weighted RRF |
| Hybrid Reranker | 필요 | 필수 권장 | Hybrid 후보를 BGE Reranker로 재정렬 |

## 공통 평가 지표

| 지표 | 설정 | 의미 |
|---|---:|---|
| Hit@K | K=3 | 정답 Gold가 상위 3개 안에 하나라도 포함된 비율 |
| Recall@K | K=5 | 전체 Gold 중 상위 5개에서 찾은 비율 |
| MRR@K | K=10 | 첫 번째 정답의 순위를 반영한 점수 |
| MAP@K | K=10 | 여러 정답의 순위 품질을 반영한 평균 정밀도 |
| Complete@K | K=5 | 다중 청크 필수 질문에서 Primary Gold를 모두 찾은 비율 |
| nDCG@K | K=5 | Primary·Supporting 중요도와 순위를 함께 반영 |
| Precision@K | K=5 | 상위 5개 중 Gold가 차지하는 비율 |
| F1@K | K=5 | Precision@5와 Recall@5의 조화평균 |

`Complete@5`는 `multi_chunk_required=Y`이고 Primary Gold가 2~5개인
질문에만 계산합니다. Primary Gold가 5개를 초과하면 Top-5에서 완전 회수가
불가능하므로 검색평가에서 제외하거나 별도 분석해야 합니다.

## 데이터셋 정리 내용

- 주 질문 134개 중 검색평가 대상 121개, 제외 13개
- 중복 질문, 교차 업무 Gold, Complete@5 측정 불가능 문항 제외
- 깨진 JSON 배열과 일부 Primary/Supporting 관계 수정
- 추가 질문 10개는 `추가질문_검수대기` 시트로 분리
- 원본 내용은 `추가질문_원본_v4` 시트에 보존

