# RAG_project3

`KDIC_RAG_V4_7_INTERACTIVE_CHAT.ipynb`(Colab 노트북)을 셀 단위로 쪼개서 파일로 나눈 버전이며, 로컬에서도 그대로 돌아가도록 손봤습니다. 그러나 '크롤링 → 청킹 → 임베딩 모델로 벡터화 → 질문 벡터화 → 검색 → 답변 생성

## 1. 디렉터리 구조

```
RAG_project3/
├── production/                # 실제 서비스 파이프라인
│   ├── core/                  # 공통 인프라 — config.py(모델명·Top-K·경로), upload_extract.py, load_data.py, integrity_check.py, hcx_client.py, hcx_api.py
│   ├── retrieval/             # 검색 — hybrid_search.py(Dense+BM25, Weighted-sum·Min-Max, 확정값 7:3), reranker.py(BGE-Reranker-v2-m3, pool=50→상위25 재정렬, 하위질문 병렬처리용 스레드 락 포함)
│   ├── classification/        # 질의분석 — pipeline.py(업무+의도 분류 rule→embedding→LLM), decomposition.py(복합질의 분해), context_rewrite.py(맥락재작성), normalization.py(구어체 정규화), rules.py/embedding_tier.py/llm_tier.py(분류 3단 캐스케이드)
│   ├── generation/             # 답변생성 — compound_answer.py(질의분석 포함 최종 진입점 `answer_query()`), answer_generation.py(근거기반 답변+수치검증 2차패스), complaint_template.py(민원처리 3단계 템플릿)
│   └── scripts/                 # 실행 진입점 — run_rag.py(질문 1건, JSON 저장), interactive_chat.py(터미널, 검색+답변만), interactive_chat_qa.py(터미널, 질의분석 전체 파이프라인), streamlit_app.py(브라우저 채팅 UI), build_*.py/adopt_*.py(임베딩 빌드)
├── experiments/                # 시행착오·평가용 — production은 이 폴더를 참조하지 않음
│   ├── evaluation/              # 지표 계산·튜닝 하니스 — eval_search.py(hit@3·recall@5·mrr@10 등), tune_hybrid.py, tune_reranker.py, judge_answer_quality.py(LLM-judge 답변품질) 등
│   ├── eval01/                   # 검색기법 확정 + 질의분석 raw/qa/hedged 비교 벤치마크(run_bench.py, run_bench_reranked.py)
│   └── scripts/                   # 단독 평가·측정 스크립트 — run_eval_*.py(Dense/BM25/Hybrid/Rerank baseline), measure_latency.py, compare_parallel_latency.py
├── data/                        # documents.jsonl, chunks.jsonl, chunk_embeddings_hcx.jsonl, 평가 데이터셋(.xlsx) — 로컬 전용, .gitignore 처리
├── rag_baseline/                 # uploaded_output(작업용 복사본), results(평가 로그·상세 CSV) — 실행 시 자동 생성
├── artifacts/                     # 분석용 스크래치 파일
├── requirements.txt               # openai·numpy·pandas·FlagEmbedding 등
└── README.md
```

`production/`과 `experiments/`는 코드 트리 전체(하위 패키지 포함)를 통째로 옮긴 구조라, 각 폴더 내부의 상대 import는 그대로입니다. 다만 `experiments/`의 일부 파일(예: `evaluation/tune_dense_structured_hybrid.py`)은 `production/`의 검색·분류 모듈을 실험에 재사용하고 있어서, 진입점 스크립트 상단에서 `production/`과 `RAG_project3/`(=`experiments`의 상위) 두 경로를 모두 `sys.path`에 추가해둡니다 — 실행하는 입장에서는 신경 쓸 필요 없이 `RAG_project3`를 작업 디렉토리로 두고 그대로 실행하면 됩니다(예: `python production/scripts/interactive_chat.py`).

참고로 `production/retrieval`·`production/classification`·`production/generation` 안에는 실제 파이프라인엔 안 쓰이지만(`bge_m3_structured_search.py`, `llm_reranker.py`, `query_expansion.py`, `context_expansion.py`, `classification/evaluate_*.py` 등) 평가 스크립트가 같이 참조해서 완전히 분리하지 못하고 남아있는 파일이 일부 있습니다 — 실행 결과에는 영향 없습니다.

## 2. 실행 흐름 (import 체인)

`interactive_chat.py`를 실행하면 아래 화살표를 타고 나머지 파일이 전부 순서대로 실행됩니다. 초록 = 로컬 파일 I/O, 파랑 = 외부 API 호출.

![RAG_project3 실행 흐름](docs/rag-architecture.png)

## 3. 실행 타임라인

**공통 초기화** — 어느 진입점을 실행하든 항상 먼저 이 순서로 준비됩니다.

1. **`production/core/config.py` 로드** — 모델명(`HCX-005`, `bge-m3`), Top-K, `WORK_ROOT` 설정
2. **`production/core/upload_extract.py`** — `data/` 또는 `KDIC_ZIP_PATH` 경로를 `rag_baseline/uploaded_output/`에 복사
3. **`production/core/load_data.py`** — documents / chunks / embeddings 로드
4. **`production/core/integrity_check.py`** — chunk_id 중복·누락, 임베딩 차원(1024) 검사
5. **`production/core/hcx_client.py`** — `HCX_API_KEY` 읽기, OpenAI 호환 클라이언트 생성

**이후 실행 목적에 따라 진입점이 갈라집니다.**

- **검색+답변만 (질의분석 없이)**: `production/scripts/interactive_chat.py` → `production/scripts/run_rag.py`(`run_kdic_rag()`) → `production/retrieval/reranker.py`(`hybrid_rerank_search`: Hybrid Dense+BM25 pool=50 → Cross-Encoder 상위25 재정렬 → top5) → `production/generation/answer_generation.py`(근거기반 답변 + 수치 검증 2차 패스) → 질문마다 결과 JSON 저장
- **질의분석 포함 전체 파이프라인 (실제 프로덕션 답변)**: `production/scripts/interactive_chat_qa.py`(터미널) 또는 `production/scripts/streamlit_app.py`(브라우저 UI, `streamlit run`으로 실행) → `production/generation/compound_answer.py`(`answer_query()`) → 구어체 정규화(`classification/normalization.py`) → 맥락재작성(`classification/context_rewrite.py`, 이전 턴 있을 때만) → 복합질의 분해(`classification/decomposition.py`) → 하위질문별 업무+의도 분류(`classification/pipeline.py`, rule→embedding→LLM) → 검색(단일질의는 `hybrid_rerank_search`, 복합질의 하위질문은 `hybrid_rerank_search_hedged`) → 답변 생성(정보질문은 `answer_generation.py`, 민원처리질문은 `complaint_template.py`) → 복합질의면 하위답변 병합
- **검색기법 확정 + 질의분석 효과 비교**: `experiments/eval01/run_bench.py` — 확정된 `hybrid_minmax_7_3` 기준으로 질의분석 미적용(raw) / 적용(qa) / 원문 안전망 병합(hedged) 3갈래를 pre-rerank 지표로 비교. `run_bench_reranked.py`는 같은 비교를 실제 프로덕션 재정렬까지 포함해서 수행
- **Dense / BM25 / Hybrid / Reranker 단독 평가**: `experiments/scripts/run_eval_dense.py` · `run_eval_bm25.py` · `run_eval_hybrid.py` · `run_eval_rerank.py` → `production/retrieval/`의 해당 검색 함수 호출 → `experiments/evaluation/eval_search.py`의 `evaluate_search()`로 지표(hit@3·recall@5·mrr@10·map@10·precision@5·f1@5·ndcg@5) 계산 → `rag_baseline/results/search_eval_log.csv` 및 문항별 `detail_*.csv`에 저장
- **하이브리드 하이퍼파라미터 튜닝**: `experiments/evaluation/tune_hybrid.py` — 질문당 Dense/BM25 후보를 한 번만 캐싱한 뒤 `candidate_pool_size → fusion 방식(RRF vs Weighted-sum) → weights(Dense:BM25)` 순으로 단계적 스윕 (확정값: pool=50, Weighted-sum·Min-Max, 7:3)
- **Reranker 튜닝**: `experiments/evaluation/tune_reranker.py` — `production/retrieval/hybrid_search.py`로 넉넉한 후보 pool을 확보한 뒤 `production/retrieval/reranker.py`(BGE-Reranker-v2-m3, 로컬 cross-encoder)로 재정렬 → 재정렬에 포함할 후보 개수(N) 스윕
- **답변 품질 LLM-judge**: `experiments/evaluation/judge_answer_quality.py` — 확정 프로덕션 파이프라인으로 실제 답변을 생성한 뒤, 답변의 모든 구체적 사실이 근거 원문에 실제로 있는지 별도 HCX 호출로 채점(fully_grounded 여부)

## 환경 설정 및 실행

```powershell
cd RAG_project3
pip install -r requirements.txt
```

### HCX_API_KEY 설정 방법

`.env` 파일이나 코드에 직접 넣는 게 아니라, 터미널(PowerShell)에서 환경변수로 등록해서 씁니다.

**방법 1. 매번 입력 (제일 간단)**

PowerShell 켜고 이렇게 입력:
```powershell
$env:HCX_API_KEY = "여기에_실제_키"
```

같은 터미널 창에서 바로 이어서:
```powershell
cd RAG_project3
python production/scripts/interactive_chat.py
```

이 터미널 창을 닫으면 값이 사라지기 때문에, 새 창을 열 때마다 다시 입력해야 합니다.

**방법 2. 매번 안 치고 싶다면 (영구 등록)**

```powershell
setx HCX_API_KEY "여기에_실제_키"
```

한 번만 실행하면 이후 새로 여는 모든 터미널에 자동으로 반영됩니다. 단, 지금 열려 있는 창에는 바로 적용되지 않으니 새 터미널을 열어서 확인해야 합니다.

**주의: 이미 열려 있던 PowerShell 창은 `setx`로 키를 바꿔도 자동으로 반영되지 않습니다.** Windows는 이미 실행 중인 프로세스의 환경변수를 새로고침하지 않기 때문입니다. 새 창을 여는 대신 같은 창에서 계속 쓰고 싶다면, 실행 직전에 아래 명령으로 레지스트리 최신값을 강제로 다시 읽어오면 됩니다:

```powershell
$env:HCX_API_KEY = [System.Environment]::GetEnvironmentVariable("HCX_API_KEY","User")
```

### 데이터 준비 및 실행

`documents.jsonl`, `chunks.jsonl`, `chunk_embeddings_hcx.jsonl`을 `RAG_project3/data/` 폴더에 넣어주세요. (다른 위치를 쓰고 싶으면 `KDIC_ZIP_PATH` 환경변수로 zip 경로 또는 폴더 경로를 지정하면 됩니다)

`RAG_project3`를 작업 디렉토리로 두고, 원하는 진입점을 실행하면 import 체인을 타고 순서대로 다 실행됩니다:

```powershell
python production/scripts/interactive_chat.py       # 검색+답변만 (질의분석 없이, 가장 단순)
python production/scripts/interactive_chat_qa.py     # 질의분석 포함 전체 파이프라인 (후속질문·복합질의 지원, 실제 프로덕션 답변)
streamlit run production/scripts/streamlit_app.py     # 위와 동일한 파이프라인을 브라우저 채팅창으로
```

질문을 입력하면 검색 결과와 HCX 답변이 나옵니다. 터미널 스크립트는 종료하려면 `종료`/`끝`/`exit`/`quit`/`q` 중 아무거나 입력하고, `interactive_chat_qa.py`에서는 `초기화`/`reset`으로 대화 맥락만 리셋할 수 있습니다.

결과 JSON은 `rag_baseline/results/` 아래에 저장됩니다 (`.gitignore`로 제외돼 있어서 커밋은 안 됩니다).

평가·튜닝 스크립트(`experiments/evaluation/`, `experiments/eval01/`, `experiments/scripts/`)도 같은 방식으로 `RAG_project3`에서 `python experiments/eval01/run_bench.py`처럼 실행하면 됩니다.
