# KDIC Hybrid A + BGE Reranker 평가기

Hybrid A가 생성한 후보를 `BAAI/bge-reranker-v2-m3` Cross-Encoder로
재정렬합니다.

## 검색 흐름

1. BGE-M3 Dense Top-10
2. BM25-Nori Discard Top-10
3. Dense 0.85 + Nori 0.15, Weighted RRF(c=10)
4. 합집합 후보 최대 20개를 BGE Reranker로 재정렬
5. 최종 Top-10 평가

HCX API 키는 질문 Dense 임베딩에만 사용합니다. Reranker는 Colab GPU에서
직접 실행하며 별도 API 키가 필요하지 않습니다. 공개 모델을 처음 실행할 때
Hugging Face에서 모델 파일을 다운로드합니다.

Colab 런타임은 T4 GPU를 권장합니다. CPU는 매우 느리므로 기본적으로 실행을
차단합니다.

## 주요 결과 파일

- `question_results.csv`: Reranker 적용 후 최종 결과
- `comparison_before_after.csv`: 지표별 전후 차이
- `summary_before_rerank.json`: Hybrid A 적용 전 요약
- `summary.json`
- `summary_by_domain.csv`
- `query_embedding_cache.jsonl`
- `reranker_score_cache.jsonl`

`question_results.csv` 하나에 질문별 Hybrid 원래 순위, 적용 전 지표,
Reranker 점수와 최종 지표를 함께 기록합니다. 질문 결과 CSV를 하나만
생성하므로 대시보드에서 113개 질문이 중복 집계되지 않습니다.
