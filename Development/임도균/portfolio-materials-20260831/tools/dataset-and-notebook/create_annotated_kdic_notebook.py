from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path


SOURCE = Path(os.environ["KDIC_SOURCE_NOTEBOOK"])
OUTPUT = Path(os.environ["KDIC_ANNOTATED_NOTEBOOK"])


def markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.rstrip() + "\n",
    }


EXPLANATIONS = {
    2: r"""
### 코드 해설 — 라이브러리 설치

이 셀은 네이버 클로바 스튜디오 HCX API를 **OpenAI 호환 방식**으로 호출할 수 있도록
`openai` 파이썬 패키지를 설치합니다.

- `%pip`: 현재 노트북 커널에 패키지를 설치하는 Jupyter 전용 명령입니다.
- `>=1.68,<2`: 1.68 이상, 2.0 미만 버전만 허용합니다.
- `only-if-needed`: 이미 호환되는 의존성이 있으면 불필요하게 전부 바꾸지 않습니다.
- 이 단계는 모델을 PC에 설치하는 것이 아닙니다. API 요청용 클라이언트만 설치합니다.

실행 결과가 조용한 이유는 `-q(quiet)` 옵션 때문입니다. 설치 후 import 오류가 나면
런타임을 다시 시작하고 다음 셀부터 실행하면 됩니다.
""",
    4: r"""
### 코드 해설 — import, 모델 설정, 작업 폴더

이 셀에는 앞으로 모든 함수가 공통으로 사용하는 **환경 설정값**이 모여 있습니다.

1. `import`는 JSONL, ZIP, 경로, 벡터 계산, 표 출력, API 호출 도구를 불러옵니다.
2. `HCX_BASE_URL`은 요청을 보낼 클로바 스튜디오 서버 주소입니다.
3. `HCX_CHAT_MODEL`은 최종 문장 생성 모델, `HCX_EMBEDDING_MODEL`은 질문을 숫자 벡터로 바꾸는 모델입니다.
4. `TOP_K=5`는 유사도가 높은 청크를 최대 5개 가져온다는 뜻입니다.
5. `MIN_SCORE=0.30`은 0.30 미만인 검색 결과를 버리는 1차 기준입니다.
6. `BUSINESS_FUNCTION=None`이면 6개 업무 전체가 검색 대상입니다.
7. `/content/...` 경로는 Google Colab의 임시 저장 공간입니다. 런타임이 종료되면 사라질 수 있습니다.

핵심 구분:

- 임베딩 모델: 질문과 청크의 **의미상 거리 계산**
- 채팅 모델: 검색 근거를 읽고 **사람이 읽을 답변 생성**
- `Top-K`, 최소 점수: 모델이 아니라 **검색기가 사용하는 설정**
""",
    6: r"""
### 코드 해설 — ZIP 업로드와 압축 해제

두 함수의 역할이 분리되어 있습니다.

- `upload_kdic_output_zip()`: 사용자가 올린 ZIP 바이트를 Colab 파일로 저장
- `extract_kdic_output()`: ZIP 손상 여부를 확인한 후 작업 폴더에 압축 해제

실행 흐름:

```text
files.upload()
→ ZIP 파일인지 확인
→ 정확히 1개인지 확인
→ KDIC_output_uploaded.zip으로 저장
→ 기존 압축 해제 폴더 정리
→ ZIP 검사(testzip)
→ 압축 해제
```

마지막 두 줄은 함수를 정의만 하는 것이 아니라 실제로 호출합니다. 따라서 이 셀을 실행하면
즉시 업로드 창이 나타납니다. 로컬 Jupyter에서는 바로 아래 안내처럼 경로를 직접 지정해야 합니다.
""",
    9: r"""
### 코드 해설 — 산출물 위치 탐색과 JSONL 로드

`JSONL`은 한 줄에 JSON 객체 하나가 저장된 형식입니다. 파일 전체를 하나의 JSON 배열로
읽지 않고, 줄마다 `json.loads()`를 수행합니다.

- `find_unique_file`: 압축을 푼 폴더 아래에서 원하는 파일명을 재귀 검색합니다.
- `processed` 폴더 안의 파일을 우선하여 엉뚱한 복사본 선택을 줄입니다.
- 후보가 0개이거나 여러 개면 조용히 추측하지 않고 오류로 중단합니다.
- `load_jsonl`: 빈 줄은 건너뛰고, 잘못된 JSON이 있으면 정확한 줄 번호를 알려줍니다.
- `RESULT`: 기존 검색 함수가 기대하는 `documents`, `chunks` 구조를 유지하기 위한 묶음입니다.

여기서는 아직 검색하지 않습니다. 디스크에 있는 산출물을 파이썬 리스트로 올리는 단계입니다.
""",
    11: r"""
### 코드 해설 — 데이터 무결성 검사

검색을 시작하기 전에 데이터가 서로 정확히 연결되는지 검사합니다. 이 단계가 없으면
검색 점수는 나왔는데 대응하는 본문 청크가 없거나, 잘못된 청크가 답변 근거로 들어갈 수 있습니다.

검사 항목:

1. 문서·청크·임베딩 레코드의 ID가 비어 있지 않은지
2. 각각의 ID가 중복되지 않는지
3. 모든 청크에 임베딩이 있는지(`missing_embeddings`)
4. 원본 청크가 없는 임베딩이 있는지(`orphan_embeddings`)
5. 모든 임베딩 벡터의 차원이 같은지
6. 벡터에 `NaN`, 무한대 같은 계산 불가능 값이 없는지
7. 실제 데이터에 들어 있는 업무 목록이 무엇인지

`set A - set B`는 A에는 있지만 B에는 없는 항목을 찾는 집합 연산입니다. 이 노트북에서
가장 중요한 검사는 `chunk_id`를 기준으로 청크와 임베딩이 1:1 대응하는지 확인하는 부분입니다.
""",
    13: r"""
### 코드 해설 — API 키 읽기와 HCX 클라이언트 생성

API 키는 코드에 직접 적지 않고 다음 순서로 찾습니다.

```text
Colab Secrets의 HCX_API_KEY
→ 없으면 운영체제 환경 변수 HCX_API_KEY
→ 둘 다 없으면 오류
```

그 후 줄바꿈, `Bearer ` 접두어, 공백이 섞이지 않았는지 검사합니다. 정상 키를
`OpenAI(...)`에 전달하되 `base_url`을 HCX 주소로 설정하여 클로바 API를 호출합니다.

보안상 중요한 점:

- API 키 값을 출력하지 않고 “등록됨”만 출력합니다.
- 노트북을 팀원에게 공유할 때 키 자체가 셀이나 출력에 들어 있지 않아야 합니다.
- 팀원은 각자 자신의 Secret 또는 환경 변수에 키를 등록해야 합니다.
""",
    15: r"""
### 코드 해설 — 채팅 API와 임베딩 API 포장 함수

이 셀은 외부 API 호출을 두 함수로 분리합니다.

- `hcx_chat_text`: 시스템 지침과 사용자 프롬프트를 보내 최종 텍스트 답변을 받습니다.
- `hcx_embed_text`: 질문 하나를 `bge-m3` 숫자 벡터 하나로 바꿉니다.

`normalize_hcx_embedding_text`는 NULL 문자와 줄바꿈 형식을 정리하고 빈 질문을 차단합니다.
임베딩 결과는 다음 조건을 확인합니다.

- 요청 1건에 벡터도 정확히 1개인지
- 벡터가 비어 있지 않은지
- `NaN` 또는 무한대가 없는지
- 저장된 청크 임베딩과 차원이 같은지

질문 벡터와 청크 벡터의 차원이 다르면 코사인 유사도를 계산할 수 없으므로 즉시 중단합니다.
""",
    17: r"""
### 코드 해설 — 검색과 근거 기반 답변의 핵심

이 셀이 현재 RAG의 핵심입니다.

#### 1. `cosine_similarity`

두 벡터의 방향이 얼마나 비슷한지 계산합니다. 값이 클수록 질문과 청크의 의미가 가깝다고
간주합니다. 두 벡터 중 하나가 영벡터면 0을 반환하여 0으로 나누는 오류를 막습니다.

#### 2. `semantic_search_hcx`

```text
질문 → 질문 임베딩 → 모든 저장 청크 벡터와 비교
→ 선택한 업무가 아니면 제외 → 최소 점수 미만 제외
→ 점수 내림차순 정렬 → 앞에서 Top-K개 반환
```

현재는 **Dense 임베딩 검색만** 사용합니다. BM25 키워드 검색, Hybrid, Reranking,
업무 자동 분류는 아직 포함되지 않았습니다.

#### 3. 허용 정보 수집·사후 필터

`_allowed_evidence_values`가 검색된 청크 안의 URL과 전화번호만 허용 목록으로 만듭니다.
`_remove_unsupported_urls_and_phones`는 LLM이 허용 목록 밖의 값을 만들면 삭제합니다.
이는 환각을 줄이는 안전장치지만, 답변 내용 전체의 사실성을 완벽히 보장하는 평가는 아닙니다.

#### 4. `generate_grounded_hcx_answer`

검색 결과가 없거나 최고 점수가 기준 미만이면 LLM을 호출하지 않고 “근거 부족” 답변을 냅니다.
근거가 있으면 청크 내용을 번호별로 묶어 프롬프트에 넣고, 낮은 temperature로 답변을 생성합니다.
마지막에는 실제 검색에 사용된 청크 ID와 공식 출처 URL을 코드가 직접 덧붙입니다.
""",
    19: r"""
### 코드 해설 — 질문 1건을 끝까지 처리하는 파이프라인

`run_kdic_rag()`가 질문 한 건에 대한 전체 작업을 지휘합니다.

```text
입력값 검증
→ semantic_search_hcx()
→ 화면용 검색 결과 표 생성
→ generate_grounded_hcx_answer()
→ 질문·설정·검색결과·답변을 payload로 묶음
→ 선택적으로 JSON 저장
→ payload 반환
```

- `build_search_result_rows`: 내부 청크 객체에서 검토에 필요한 필드만 골라 표 형태로 만듭니다.
- `utc_now_iso`: 결과 생성 시각을 시간대에 덜 의존하는 UTC ISO 형식으로 남깁니다.
- `save_result=True`: 질문마다 결과를 JSON으로 보존합니다.
- 반환되는 `payload`는 이후 평가 코드나 UI가 재사용하기 쉬운 구조입니다.

주의: 파일명은 초 단위 시각이므로 같은 초에 여러 질문을 저장하면 덮어쓸 가능성이 있습니다.
학습용 초기 버전에서는 괜찮지만 서비스화할 때 UUID나 밀리초를 추가하는 편이 안전합니다.
""",
    21: r"""
### 코드 해설 — 반복 입력형 챗봇

이 셀은 웹 UI가 아니라 터미널 입력창을 반복 사용하는 간단한 대화 루프입니다.

- `BUSINESS_FUNCTION_FILTER=None`: 업무를 자동 분류하지 않고 6개 업무 전체 검색
- `chat_history`: 현재 실행 세션에서 성공한 결과를 순서대로 보관
- `while True`: 사용자가 종료 명령을 입력할 때까지 반복
- 빈 질문은 다시 입력받고, API 오류가 나도 전체 프로그램을 끝내지 않고 다음 질문으로 진행
- 성공한 결과는 `chat_history`와 전역 `rag_output`에 저장

중요한 한계: 이름은 `chat_history`이지만 이전 대화 내용을 다음 질문의 검색·답변 프롬프트에
넣지는 않습니다. 즉, 현재 구현은 “여러 질문을 연속 실행”할 뿐 **문맥을 기억하는 멀티턴 챗봇은 아닙니다.**
""",
    23: r"""
### 코드 해설 — 마지막 결과 다운로드

대화형 셀에서 마지막으로 성공한 결과가 `rag_output`에 남아 있습니다. 이 셀은 그 객체의
`result_path`를 찾아 Colab 다운로드를 시작합니다.

- Colab이면 `files.download(...)`를 호출합니다.
- 로컬 Jupyter이면 자동 다운로드 대신 저장 경로를 출력합니다.
- 질문이 한 번도 성공하지 않았다면 다운로드할 파일이 없다는 안내를 표시합니다.

모든 질문의 결과는 이미 `results` 폴더에 각각 저장되지만, 이 셀은 그중 마지막 결과 한 건만
편하게 내려받는 기능입니다.
""",
}


INLINE_COMMENTS = {
    2: [
        "# 현재 노트북 커널에 HCX API 호출용 openai 패키지를 설치합니다.",
    ],
    4: [
        "# 타입 힌트에서 현재 클래스명을 바로 참조할 수 있게 평가 시점을 늦춥니다.",
    ],
    6: [
        "# 1단계: 사용자가 올린 KDIC_output ZIP을 Colab 작업 공간에 저장합니다.",
    ],
    9: [
        "# 압축 내부 경로가 달라도 파일명으로 핵심 산출물을 찾아냅니다.",
    ],
    11: [
        "# 검색 전에 문서-청크-임베딩 연결이 안전한지 검증합니다.",
    ],
    13: [
        "# API 키는 코드에 직접 쓰지 않고 Secret 또는 환경 변수에서만 읽습니다.",
    ],
    15: [
        "# 최종 답변 생성과 질문 임베딩 생성을 각각 함수로 감쌉니다.",
    ],
    17: [
        "# [검색 단계] 질문과 청크 벡터의 의미 유사도를 계산합니다.",
    ],
    19: [
        "# 질문 한 건을 검색부터 답변 저장까지 연결하는 실행 파이프라인입니다.",
    ],
    21: [
        "# 사용자가 종료 명령을 입력할 때까지 질문을 반복해서 처리합니다.",
    ],
    23: [
        "# 마지막으로 성공한 질문 결과 JSON을 내려받습니다.",
    ],
}


INTRO = r"""
# KDIC RAG V4.7 — 전체 로직 주석·학습용 버전

이 파일은 원본 `KDIC_RAG_V4_7_INTERACTIVE_CHAT.ipynb`의 **실행 로직을 유지하면서**
각 코드 셀 앞에 상세 설명과 코드 내부 주석을 추가한 학습용 사본입니다.

## 전체 흐름을 먼저 한 문장으로 보기

사용자 질문을 숫자 벡터로 바꾸고, 미리 임베딩된 청크 중 의미가 가까운 Top-K를 찾은 다음,
그 청크만 HCX-005에 근거로 제공하여 답변하고 결과를 JSON으로 저장합니다.

```text
KDIC_output.zip
  ├─ documents.jsonl
  ├─ chunks.jsonl
  └─ chunk_embeddings_hcx.jsonl
             ↓ 로드·연결 검사
사용자 질문 → bge-m3 질문 임베딩
             ↓
저장된 청크 벡터와 코사인 유사도 계산
             ↓
업무 필터(선택) → 최소 점수 → Top-K
             ↓
검색 청크를 HCX-005 프롬프트에 삽입
             ↓
답변 생성 → 미지원 URL·전화번호 제거
             ↓
답변 + 근거 청크 + 공식 출처 + 결과 JSON
```

## 꼭 구분해서 이해할 용어

| 용어 | 이 노트북에서 하는 일 |
|---|---|
| 문서(document) | 원문 페이지 단위의 자료 |
| 청크(chunk) | 검색하기 쉽도록 문서를 작게 나눈 텍스트 |
| 임베딩(embedding) | 텍스트 의미를 비교하기 위한 숫자 벡터 |
| Dense 검색 | 질문 벡터와 청크 벡터의 유사도로 검색 |
| 코사인 유사도 | 두 벡터 방향의 유사성을 나타내는 점수 |
| Top-K | 점수가 높은 결과를 최대 몇 개 사용할지 지정 |
| 최소 점수 | 관련성이 너무 낮은 청크를 버리는 기준 |
| RAG | 검색한 근거를 LLM에 함께 주어 답변하게 하는 구조 |
| Grounded answer | 검색 근거 범위 안에서 생성한 답변 |

## 현재 초기형에서 아직 하지 않는 것

이 버전에는 BM25 키워드 검색, Hybrid 검색, Reranking, 업무 자동 분류,
Parent-Child 문맥 확장, 평가셋 일괄 실행, 멀티턴 기억, 웹 UI가 없습니다.
따라서 이 노트북은 완성 서비스가 아니라 **Dense RAG의 기본 동작을 이해하고 검증하는 기준선**입니다.
"""


def annotate_code(cell: dict, index: int) -> dict:
    copied = deepcopy(cell)
    source = "".join(copied.get("source", []))
    prefix = "\n".join(INLINE_COMMENTS.get(index, []))
    if prefix:
        source = prefix + "\n" + source
    copied["source"] = source
    return copied


def main() -> None:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    output_notebook = deepcopy(notebook)
    new_cells = [markdown_cell(INTRO)]

    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") == "code" and index in EXPLANATIONS:
            new_cells.append(markdown_cell(EXPLANATIONS[index]))
            new_cells.append(annotate_code(cell, index))
        else:
            new_cells.append(deepcopy(cell))

    output_notebook["cells"] = new_cells
    output_notebook.setdefault("metadata", {})["kdic_learning_version"] = {
        "source_notebook": SOURCE.name,
        "purpose": "원본 실행 로직을 유지한 상세 주석·학습용 버전",
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(output_notebook, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )

    print(OUTPUT)
    print("original_cells=", len(notebook.get("cells", [])))
    print("annotated_cells=", len(new_cells))


if __name__ == "__main__":
    main()
