# KDIC Structured Hybrid + Parent-Child + Reranker 평가기

업무 필터 없이 전체 KDIC 청크를 대상으로 다음 순서로 검색 품질을 평가합니다.

1. `title + section_title + heading_path + content`를 BGE-M3로 임베딩한 Structured Dense 검색
2. BM25-Nori Discard 검색
3. Weighted RRF 결합 (`Dense 0.85`, `Nori 0.15`, `c=10`)
4. `BAAI/bge-reranker-v2-m3`로 후보 Child 재정렬
5. Reranker 상위 Child 5개를 보존하고 같은 Parent의 인접 청크 `±1` 확장

## 기본 파라미터

- 각 1차 검색 후보: 20개
- Reranker 입력 후보: 20개
- 검색 평가 Top-K: 10개
- Parent-Child seed: Reranker 상위 5개
- 인접 범위: 앞뒤 1개
- 인접 확장 Parent: 최대 3개
- Parent별 추가 청크: 최대 3개
- 최종 답변 컨텍스트: 최대 10개 청크
- 업무 필터: 사용하지 않음

`max_parents`는 인접 청크 확장에만 적용됩니다. Reranker가 선택한 상위 5개 seed Child는 서로 다른 Parent에 속해도 모두 먼저 보존합니다.

## 지표 해석

- 일반 `Hit@3`, `Recall@5`, `MRR@10`, `MAP@10`, `nDCG@5`, `Precision@5`, `F1@5`: Reranker 이후 Child 순위
- `before_*`: Reranker 이전 Hybrid Child 순위
- `child_seed_recall`: Parent 확장 전 상위 5개 Child의 Gold Recall
- `expanded_gold_recall`: Parent-Child 확장 컨텍스트의 Gold Recall
- `expansion_recall_gain`: Parent-Child 확장으로 증가한 Recall
- `expanded_non_gold_ratio`: Gold로 표기되지 않은 확장 청크의 비율이며 무관 청크 비율과 동일하지 않음

이 평가기는 답변 생성이나 LLM 답변 평가는 수행하지 않습니다.
