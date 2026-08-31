# KDIC Hybrid A 검색 평가기

`BGE-M3 Dense 85% + BM25-Nori Discard 15%`를 가중 RRF로 결합합니다.

## 고정 조건

- Dense 모델: `bge-m3`
- Dense 질문 임베딩: HCX OpenAI 호환 Embeddings API
- Dense 청크 벡터: `KDIC_output.zip`의 `chunk_embeddings_hcx.jsonl`
- 키워드 검색: Apache Lucene Nori `discard`
- 후보 수: 각 검색기 Top-10
- 결합 방식: Weighted RRF
- RRF 상수: 10
- 가중치: Dense 0.85, Nori 0.15
- 최종 순위: Top-10
- 업무 필터: Gold 업무 사전 필터

## 필요한 파일

1. `KDIC_Hybrid_A_평가기.zip`
2. `KDIC_output.zip`
3. `평가데이터셋_검색평가지표용.xlsx`

실제 평가에는 HCX API 키가 필요합니다. Colab Secrets에 이름을
`HCX_API_KEY`로 등록하면 노트북이 자동으로 불러옵니다. 해당 Secret의
노트북 접근 권한을 켜야 합니다. Secret을 읽지 못한 경우에만 `getpass`
보안 입력창이 표시됩니다. Dry-run은 API를 호출하지 않습니다.

## 결과

- `question_results.csv`
- `summary.json`
- `summary_by_domain.csv`
- `run_config.json`
- `query_embedding_cache.jsonl`
- `dry_run_report.json`

질문별 결과에는 Dense 후보, Nori 후보, 최종 RRF 순위와 각 청크의 원래
순위가 함께 기록되어 결과 변화를 추적할 수 있습니다.
