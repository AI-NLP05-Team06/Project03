# KDIC BM25 + Lucene Nori 검색 평가기

Apache Lucene `KoreanAnalyzer`를 이용하여 mecab-ko-dic 기반 한국어 형태소
분석을 수행한 뒤, 기본 BM25 평가기와 동일한 BM25 점수식으로 검색합니다.

## 고정 조건

- 검색 방식: BM25
- 토큰화: Apache Lucene KoreanAnalyzer(Nori)
- 복합어 처리: `--decompound-mode discard` 또는 `mixed`
- Lucene: 9.12.2
- BM25 파라미터: `k1=1.5`, `b=0.75`
- Top-K: 10
- 업무 필터: Gold 업무 사전 필터
- 문서 입력: `chunks.jsonl`의 `content`
- 평가 데이터와 지표: Dense·Sparse·기본 BM25와 동일

최초 실행 시 Maven Central에서 Lucene core, analysis-common, analysis-nori
JAR를 내려받습니다. 외부 API 키와 GPU는 필요하지 않습니다.

기본 BM25와의 공정한 비교를 위해 점수식과 파라미터는 같고 토큰화만 Nori로
교체했습니다. Elasticsearch Nori 플러그인도 Apache Lucene Nori 분석 모듈을
통합하여 사용합니다.

- `discard`: 복합어 원형을 버리고 분해된 형태소만 사용
- `mixed`: 복합어 원형과 분해된 형태소를 모두 사용
