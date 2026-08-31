# 검색평가 패키지

| 폴더 | 목적 |
|---|---|
| `dataset-4.1-no-filter` | test_dataset_4.1 호환, 업무필터 없이 7개 검색 구성 평가 |
| `structured-hybrid-parent-child-reranker` | Structured Hybrid + Parent–Child + Reranker 추가 실험 |
| `baselines` | 초기 BM25/Nori/Dense/Sparse/Hybrid/Reranker 개별 소스 |
| `../archive/evaluators` | 데이터셋·업무필터·호환성 변경 전후 기록 |

각 평가기 ZIP 옆의 `source/`에서 압축을 풀지 않고 코드를 읽을 수 있습니다. 코드 ZIP에는 운영 데이터·설치 라이브러리를 넣지 않았습니다. 초기 개별 소스는 다른 평가기의 공통 모듈을 참조할 수 있으므로 실제 실행은 해당 Colab 패키지를 우선합니다.

Hit@3, Recall@5, MRR@10, MAP@10, nDCG@5, Precision@5, F1@5, Complete 관련 정의는 버전별 코드를 확인하세요. 과거 계산식이나 판단 기준을 이번 업로드에서 임의로 수정하지 않았습니다.
