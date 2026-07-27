# RAG_project3

`KDIC_RAG_V4_7_INTERACTIVE_CHAT.ipynb`(Colab 노트북)을 셀 단위로 쪼개서 파일로 나눈 버전입니다.
원래 Colab 전용으로 짜여있던 걸 로컬에서도 그대로 돌아가게 손봤고, 이 문서에는 원본 노트북 대비 뭐가 바뀌었는지 정리해뒀습니다.

## 파일 구성 (실행 순서대로)

| 파일 | 역할 |
|---|---|
| `config.py` | 공통 설정 (모델명, Top-K, 작업 폴더 경로 등) |
| `upload_extract.py` | `KDIC_output.zip`(또는 데이터 폴더) 업로드/해제 |
| `load_data.py` | documents/chunks/embeddings jsonl 로드 |
| `integrity_check.py` | 데이터 무결성 검사 (ID 중복, 임베딩 차원 등) |
| `hcx_client.py` | HCX API 키·클라이언트 설정 |
| `hcx_api.py` | HCX 채팅/임베딩 호출 함수 |
| `search_answer.py` | Dense 검색 + 근거 기반 답변 생성 |
| `run_rag.py` | 질문 1건 실행 함수 (`run_kdic_rag`) |
| `interactive_chat.py` | 반복 질문 입력 루프 |
| `download_result.py` | 결과 다운로드(Colab) / 경로 출력(로컬) |

> 각 파일 맨 위에 `from config import *` 같은 import가 붙어있는 이유: 원래 노트북은 한 커널 안에서 셀들이 전역 변수를 공유하는 구조였습니다 (`load_data.py`가 만든 `documents`를 `integrity_check.py`가 그대로 받아쓰는 식). 그 구조를 유지하려고 이전 단계 파일을 불러오는 import만 추가했고, 로직 자체는 원본 셀 그대로입니다.

## Colab 노트북에서 뭘 바꿨나

로컬에서 막히는 부분은 결국 "Colab 전용 API를 쓰는 곳"뿐이었고, 검색·답변 생성 같은 핵심 로직은 손대지 않았습니다.

**1. `upload_extract.py` — 로컬 데이터 소스 지원 추가**
원본은 `google.colab.files.upload()`로 zip을 직접 업로드받는 구조라 로컬에서 돌리면 `RuntimeError`가 났습니다. `google.colab`을 import할 수 없으면(=로컬 환경) `resolve_local_kdic_output()`을 타도록 분기를 추가했습니다.
- `KDIC_ZIP_PATH` 환경변수(기본값 `data`)로 지정된 경로를 사용
- 경로가 **zip 파일**이면 기존처럼 압축 해제
- 경로가 **폴더**(jsonl 3종이 든)면 그대로 복사해서 사용 — 저희는 zip 없이 jsonl 3개를 바로 갖고 있어서 이 방식을 씁니다

**2. `config.py` — 로컬 작업 폴더 경로**
Colab에서는 원래대로 `/content/kdic_rag_baseline`을 쓰고, 로컬 기본 작업 폴더명은 `./rag_baseline`으로 지정했습니다 (`KDIC_WORK_ROOT` 환경변수로 덮어쓰기 가능).

**3. `requirements.txt` 보강**
원래 `openai`만 명시돼 있었는데, `config.py`가 실제로 쓰는 `numpy`, `pandas`, `ipython`(IPython.display)을 추가했습니다. Colab 기본 이미지에는 이미 깔려있어서 원본엔 빠져있었던 것으로 보입니다.

**4. 로컬 데이터 폴더 `RAG_project3/data/` 신설**
`documents.jsonl`, `chunks.jsonl`, `chunk_embeddings_hcx.jsonl`을 로컬에 넣어두는 위치입니다. 실제 데이터 파일은 `.gitignore`로 제외했고, 폴더 구조만 보이도록 `.gitkeep`만 커밋했습니다.

**5. 손대지 않은 것**
이미 로컬/Colab 겸용으로 짜여있던 부분이라 그대로 뒀습니다.
- `hcx_client.py`: Colab Secret이 없으면 `HCX_API_KEY` 환경변수를 읽는 fallback이 원래 있었음
- `download_result.py`: Colab이 아니면 다운로드 대신 결과 JSON 경로를 콘솔에 출력하는 fallback이 원래 있었음

## 로컬에서 돌리는 법

```powershell
cd RAG_project3
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

환경변수 설정 (터미널 세션마다 다시 해줘야 함):

## HCX_API_KEY 설정 방법

`.env` 파일이나 코드에 직접 넣는 게 아니라, 터미널(PowerShell)에서 환경변수로 등록해서 씁니다.

### 방법 1. 매번 입력 (제일 간단)

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

### 방법 2. 매번 안 치고 싶다면 (영구 등록)

```powershell
setx HCX_API_KEY "여기에_실제_키"
```

한 번만 실행하면 이후 새로 여는 모든 터미널에 자동으로 반영됩니다. 단, 지금 열려 있는 창에는 바로 적용되지 않으니 새 터미널을 열어서 확인해야 합니다.

## 데이터 준비 및 실행

`documents.jsonl`, `chunks.jsonl`, `chunk_embeddings_hcx.jsonl`을 `RAG_project3/data/` 폴더에 넣어주세요. (다른 위치를 쓰고 싶으면 `KDIC_ZIP_PATH` 환경변수로 zip 경로 또는 폴더 경로를 지정하면 됩니다)

실행하면 import 체인을 타고 순서대로 다 실행됩니다:
```powershell
python interactive_chat.py
```

질문을 입력하면 검색 결과와 HCX 답변이 나옵니다. 종료하려면 `종료`/`끝`/`exit`/`quit`/`q` 중 아무거나 입력.

결과 JSON은 `rag_baseline/results/` 아래에 저장됩니다 (`.gitignore`로 제외돼 있어서 커밋은 안 됩니다).