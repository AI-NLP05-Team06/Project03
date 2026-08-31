# KDIC RAG 검색 방식 평가 산출물

## 1. 산출물 개요

예금보험공사 RAG 챗봇의 검색 방식을 선정하기 위해 동일한 평가데이터와 업무 필터를 사용하여 다음 7개 방식을 비교했다.

1. BM25 기본
2. BM25 + Nori Mixed
3. BM25 + Nori Discard
4. BGE-M3 Dense
5. BGE-M3 Sparse
6. Hybrid A: BGE-M3 Dense 0.85 + BM25-Nori Discard 0.15, Weighted RRF
7. Hybrid A + BGE Reranker

초기 BM25-Nori 공통 버전은 Mixed·Discard 실험으로 세분화되었으므로 제출본에서는 중복을 피하기 위해 제외했다.

## 2. 폴더 구성

- `01_Colab_노트북`: Google Colab에서 실행하는 검색 방식별 평가 노트북
- `02_평가기_코드`: 노트북에 업로드하여 사용하는 평가기 ZIP
- `03_평가결과`: 당시 실행으로 생성된 질문별·도메인별 평가 결과 ZIP
- `04_공통입력`: 당시 실험에 사용한 113문항 평가데이터셋과 `KDIC_output.zip`
- `05_결과비교_도구`: 여러 평가 결과를 그래프로 비교하는 HTML 대시보드
- `평가결과_요약.csv`: 7개 검색 방식의 전체 지표 요약

각 노트북과 평가기 ZIP은 파일명 앞 번호가 서로 대응한다.

## 3. 실험 흐름

`평가데이터셋 + KDIC_output.zip + 검색 방식별 평가기`를 Colab에 업로드한 후 노트북을 위에서부터 순서대로 실행한다.

검색 결과 Top-10을 Gold 청크와 비교하여 다음 지표를 계산한다.

- Hit@3
- Recall@5
- MRR@10
- MAP@10
- Complete@5
- nDCG@5
- Precision@5
- F1@5
- 평균 검색 지연시간

## 4. 검색 방식별 핵심 설정

| 검색 방식 | 핵심 설정 | API/GPU |
|---|---|---|
| BM25 기본 | 정규식 기반 토큰화, `k1=1.5`, `b=0.75` | API·GPU 불필요 |
| BM25-Nori Mixed | Lucene KoreanAnalyzer, 복합어 mixed | API·GPU 불필요 |
| BM25-Nori Discard | Lucene KoreanAnalyzer, 복합어 discard | API·GPU 불필요 |
| BGE-M3 Dense | 질문과 청크의 Dense 벡터 유사도 검색 | HCX 임베딩 API 사용 |
| BGE-M3 Sparse | BGE-M3 lexical weight 기반 검색 | Colab GPU에서 로컬 실행 |
| Hybrid A | Dense 0.85 + Nori 0.15, RRF 상수 10 | HCX API 사용 |
| Hybrid A + Reranker | Hybrid 후보 최대 20개를 `BAAI/bge-reranker-v2-m3`로 재정렬 | HCX API + Colab GPU |

API 키는 노트북에 저장하지 않았으며, Colab Secrets의 `HCX_API_KEY` 또는 비공개 입력창을 사용한다.

## 5. 실행 결과 요약

모든 결과 파일은 113행·고유 평가ID 113개로 확인했으며, Reranker 결과에도 중복 문항이 없다.

| 검색 방식 | Hit@3 | Recall@5 | MRR@10 | nDCG@5 |
|---|---:|---:|---:|---:|
| BM25 기본 | 28.32% | 35.93% | 27.02% | 26.15% |
| BM25-Nori Mixed | 46.02% | 51.50% | 42.48% | 40.83% |
| BM25-Nori Discard | 50.44% | 52.92% | 42.88% | 41.29% |
| BGE-M3 Dense | 61.95% | 61.92% | 55.88% | 54.00% |
| BGE-M3 Sparse | 52.21% | 48.97% | 46.41% | 42.88% |
| Hybrid A | 59.29% | 62.21% | 55.32% | 53.35% |
| Hybrid A + BGE Reranker | 71.68% | 69.91% | 61.08% | 60.80% |

지연시간은 실행 환경, API 상태, 캐시 여부에 영향을 받으므로 검색 품질 지표와 분리하여 참고한다.

## 6. 제출 시 주의사항

- 포함된 결과는 당시 사용한 113문항 평가데이터셋 기준이다.
- 이후 Gold 검수 및 검색평가 대상 제외 정책이 반영된 최신 데이터셋과는 문항 수가 다를 수 있다.
- 최신 데이터셋으로 공식 수치를 확정하려면 7개 방식을 동일 조건으로 다시 실행해야 한다.
- 평가 결과 ZIP에는 API 키가 포함되어 있지 않다.
