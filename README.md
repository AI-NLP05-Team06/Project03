# RAG_project3

`KDIC_RAG_V4_7_INTERACTIVE_CHAT.ipynb`(Colab 노트북)을 셀 단위로 쪼개서 파일로 나눈 버전이며, 로컬에서도 그대로 돌아가도록 손봤습니다.

## 1. 디렉터리 구조

```
RAG_project3/
├── config.py              # 공통 설정 (모델명, Top-K, 작업 폴더 경로)
├── upload_extract.py      # data 자동 인식하도록 분기 추가
├── load_data.py           # jsonl 3종 로드
├── integrity_check.py     # ID 중복·임베딩 차원 검사
├── hcx_client.py          # HCX_API_KEY 읽기, 클라이언트 생성
├── hcx_api.py             # chat·embed 호출 함수
├── search_answer.py       # 코사인 유사도 검색 + 근거 기반 답변
├── run_rag.py             # 질문 1건 실행, JSON 저장
├── interactive_chat.py    # 터미널에서 실행하는 메인 파일
├── download_result.py     # 로컬에선 결과 경로만 출력
├── requirements.txt       # numpy·pandas·ipython 추가
├── README.md              # 변경 이력 정리
├── data/                  # documents.jsonl, chunks.jsonl, chunk_embeddings_hcx.jsonl (로컬 전용, .gitignore 처리)
└── rag_baseline/          # uploaded_output, results (실행 시 자동 생성)
```

## 2. 실행 흐름 (import 체인)

`interactive_chat.py`를 실행하면 아래 화살표를 타고 나머지 파일이 전부 순서대로 실행됩니다. 초록 = 로컬 파일 I/O, 파랑 = 외부 API 호출.

![RAG_project3 실행 흐름](docs/rag-architecture.png)

## 3. 실행 타임라인

1. **`config.py` 로드** — 모델명(`HCX-005`, `bge-m3`), Top-K, `WORK_ROOT` 설정
2. **`upload_extract.py`** — 로컬이면 `data` 또는 `KDIC_ZIP_PATH` 경로를 `rag_baseline/uploaded_output/`에 복사
3. **`load_data.py`** — documents / chunks / embeddings 로드
4. **`integrity_check.py`** — chunk_id 중복·누락, 임베딩 차원(1024) 검사
5. **`hcx_client.py`** — `HCX_API_KEY` 읽기, OpenAI 호환 클라이언트 생성
6. **`hcx_api.py` / `search_answer.py`** — 임베딩 호출, 코사인 유사도 검색, 근거 기반 답변 준비
7. **`run_rag.py`** — `run_kdic_rag()` 준비
8. **`interactive_chat.py`** — 질문 입력 → 임베딩 요청 → 상위 5개 청크 검색 → 답변 생성 → JSON 저장
9. *(optional)* **`download_result.py`** — 마지막 결과 경로 확인

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
python interactive_chat.py
```

이 터미널 창을 닫으면 값이 사라지기 때문에, 새 창을 열 때마다 다시 입력해야 합니다.

**방법 2. 매번 안 치고 싶다면 (영구 등록)**

```powershell
setx HCX_API_KEY "여기에_실제_키"
```

한 번만 실행하면 이후 새로 여는 모든 터미널에 자동으로 반영됩니다. 단, 지금 열려 있는 창에는 바로 적용되지 않으니 새 터미널을 열어서 확인해야 합니다.

### 데이터 준비 및 실행

`documents.jsonl`, `chunks.jsonl`, `chunk_embeddings_hcx.jsonl`을 `RAG_project3/data/` 폴더에 넣어주세요. (다른 위치를 쓰고 싶으면 `KDIC_ZIP_PATH` 환경변수로 zip 경로 또는 폴더 경로를 지정하면 됩니다)

실행하면 import 체인을 타고 순서대로 다 실행됩니다:
```powershell
python interactive_chat.py
```

질문을 입력하면 검색 결과와 HCX 답변이 나옵니다. 종료하려면 `종료`/`끝`/`exit`/`quit`/`q` 중 아무거나 입력.

결과 JSON은 `rag_baseline/results/` 아래에 저장됩니다 (`.gitignore`로 제외돼 있어서 커밋은 안 됩니다).
