# KDIC Structured Hybrid + Parent-Child 평가기

업무 분류와 업무 필터 없이 전체 427개 Child 청크를 검색합니다.

## 검색 순서

1. `title + section_title + heading_path + content`를 BGE-M3로 임베딩
2. Structured Dense Top-20 검색
3. BM25-Nori Discard Top-20 검색
4. Weighted RRF로 결합 (`Dense 0.85`, `Nori 0.15`, `c=10`)
5. 최종 Child Top-10으로 기존 검색 지표 계산
6. 상위 Child 5개를 seed로 선택
7. 최대 Parent 3개 안에서 인접 Child ±1 확장
8. Parent당 최대 3개, 전체 최대 10개 컨텍스트 구성

## 지표 해석

- `Hit@3`, `Recall@5`, `MRR@10`, `MAP@10`, `Complete@5`, `nDCG@5`,
  `Precision@5`, `F1@5`: **확장 전 Child 검색 순위** 평가
- `expanded_gold_recall`: Parent-Child 확장 후 컨텍스트의 Gold 포함률
- `expansion_recall_gain`: 확장 전 seed Child 대비 Gold Recall 증가량
- `parent_hit_at_3`: 상위 3개 Parent 중 Gold Parent 포함 여부
- `expanded_non_gold_ratio`: Gold로 지정되지 않은 컨텍스트 비율. Gold가 모든
  유효 문맥을 완전하게 열거하지 않을 수 있으므로 곧바로 오류율로 해석하지 않습니다.

## 주의

- 원본 XLSX는 읽기만 하며 수정하거나 다시 저장하지 않습니다.
- Structured 문서 임베딩 427개를 처음 실행할 때 HCX API 호출이 발생합니다.
- 이후 같은 결과 폴더의 `structured_embedding_cache.jsonl`을 재사용합니다.
- Parent-Child는 순위 조작이 아니라 답변에 전달할 문맥 확장입니다. 따라서 기존
  검색 지표와 확장 지표를 분리했습니다.
