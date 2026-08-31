# KDIC BGE-M3 Sparse 검색 평가기

`BAAI/bge-m3`의 Sparse 표현인 `lexical_weights`로 청크를 검색하고, Dense
평가와 동일한 Gold 및 8개 지표로 평가합니다. 답변 생성 LLM과 외부 API 키는
사용하지 않습니다.

## 비교 조건

- 모델: `BAAI/bge-m3`
- 검색 방식: Sparse lexical matching
- 문서 입력: `chunks.jsonl`의 `content`
- 문서 최대 길이: 2,048 tokens
- 질문 최대 길이: 512 tokens
- Top-K: 10
- 업무 필터: Gold 업무 사전 필터
- 평가 지표: Hit@3, Recall@5, MRR@10, MAP@10, Complete@5,
  nDCG@5, Precision@5, F1@5

Colab에서는 GPU 런타임을 권장합니다. 최초 실행 시 약 2.3GB의 BGE-M3 모델을
내려받고 427개 청크의 Sparse 벡터를 생성합니다. 생성 결과는
`chunk_sparse_vectors.jsonl`에 캐시됩니다.

Colab 기본 패키지와의 충돌을 피하기 위해 pandas는 `2.2.2`로 고정합니다.

## 실행 예

```bash
python evaluate_bge_m3_sparse.py \
  --dataset 평가데이터셋_검색평가지표용.xlsx \
  --kdic-zip KDIC_output.zip \
  --output-dir results
```

메모리가 부족하면 `--batch-size 2`를 사용하세요. Dense·Sparse·Hybrid를 공정하게
비교하려면 데이터셋, 업무 필터, Top-K, 최대 입력 길이를 같은 조건으로 유지해야
합니다.
