# RAG_project

`KDIC_RAG_V4_7_INTERACTIVE_CHAT.ipynb` 노트북의 각 코드 셀을 **내용은 그대로 유지한 채** 파일 단위로 나눈 버전입니다.

## 파일 구성 (실행 순서)

| 파일 | 원본 셀 | 역할 |
|---|---|---|
| `config.py` | 셀 4 | 공통 설정 (모델, 경로 등) |
| `upload_extract.py` | 셀 6 | `KDIC_output.zip` 업로드/해제 |
| `load_data.py` | 셀 9 | documents/chunks/embeddings 로드 |
| `integrity_check.py` | 셀 11 | 데이터 무결성 검사 |
| `hcx_client.py` | 셀 13 | HCX API 키·클라이언트 설정 |
| `hcx_api.py` | 셀 15 | HCX 채팅/임베딩 호출 함수 |
| `search_answer.py` | 셀 17 | Dense 검색 + 근거 기반 답변 생성 |
| `run_rag.py` | 셀 19 | 질문 1건 실행 함수 (`run_kdic_rag`) |
| `interactive_chat.py` | 셀 21 | 반복 질문 입력 루프 |
| `download_result.py` | 셀 23 | 마지막 결과 다운로드 |

## 왜 파일마다 `from x import *`가 있나요

노트북은 하나의 커널 안에서 셀들이 전역 변수를 공유합니다 (예: `load_data.py`가 만든 `documents`를 `integrity_check.py`가 그대로 사용). 이 구조를 유지하면서 파일로 나누기 위해, 각 파일 맨 위에 **이전 단계 파일을 불러오는 import만 추가**했습니다. 그 외 로직·문자열·주석은 원본 노트북 셀과 동일합니다.

## 로컬에서 실행하기

```bash
pip install -r requirements.txt
```

- `HCX_API_KEY` 환경변수를 설정해야 합니다 (Colab Secrets 대신 사용).
- `upload_extract.py`의 `upload_kdic_output_zip()`은 Colab 업로드 위젯 전용이라 로컬에서는 동작하지 않습니다. 로컬에서 돌리려면 이 부분을 압축 파일 경로를 직접 지정하는 코드로 바꿔야 합니다 (노트북 셀 7의 안내와 동일).
- 결과/작업 폴더 기본 경로(`WORK_ROOT`)는 Colab에서는 `/content/kdic_rag_baseline`을 그대로 쓰고, 로컬에서는 현재 폴더 아래 `./kdic_rag_baseline`을 사용합니다. `KDIC_WORK_ROOT` 환경변수로 다른 경로를 지정할 수 있습니다.

실행 순서대로 마지막 파일까지 실행하면 (`python interactive_chat.py`) 이전 단계가 import를 통해 전부 함께 실행됩니다.
