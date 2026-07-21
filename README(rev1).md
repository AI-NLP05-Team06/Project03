# Research Plan

# RAG 기반 AI 챗봇 서비스 구축

## 프로젝트 개요

### 프로젝트 연계

**실전 프로젝트 1**에서는 예금보험공사 6개 업무 도메인을 대상으로 AI 검색 서비스를 위한 데이터 파이프라인을 구축하였다.

주요 수행 내용은 다음과 같다.

- 6개 업무 도메인 웹페이지 크롤링
- HTML을 결정론적으로 Markdown으로 변환
- Parent-Child Chunk 생성
- 업무별 메타데이터(`business_function`, `breadcrumb`, `parent_doc_id`) 구축
- 첨부파일 및 출처(URL) 메타데이터 생성
- Vector Store 적재를 위한 데이터셋 구축
- RAG 평가용 테스트셋 구축

본 프로젝트에서는 위 산출물을 기반으로 **검색 기반 생성(RAG) 챗봇**을 구현하고 검색 정확도와 답변 신뢰성을 향상시키는 것을 목표로 한다.

---

# 1. Problem Definition

## Task

본 프로젝트에서는 **Retrieval-Augmented Generation(RAG)** 기반 Question Answering Task를 수행한다.

사용자의 자연어 질문에 대해 관련 문서를 검색하고, 검색된 문서를 근거로 LLM이 답변을 생성하는 AI 챗봇을 구축한다.

---

## 해결하고자 하는 문제

기존 예금보험공사의 '예솜24'는 키워드 기반 챗봇으로 운영되어 자연어 질의에 대한 대응이 어렵다.

또한 예금보험 업무는 신청 대상, 신청 기한, 제출 서류 등 조건이 복잡하여 잘못된 답변이 민원으로 이어질 가능성이 있다.

본 프로젝트에서는

- 자연어 질의응답
- 근거 기반 답변 생성
- 민원 처리 안내
- 출처 제공

이 가능한 AI 챗봇을 구축하여 이러한 문제를 해결하고자 한다.

---

## 연구의 필요성

공공기관 AI 챗봇은 정확성과 신뢰성이 매우 중요하다.

따라서 검색된 근거를 기반으로 답변을 생성하고, 출처를 함께 제공하며, 근거가 없는 경우에는 답변을 생성하지 않는 Hallucination 방지 구조가 필요하다.

---

# 2. Background & Baseline

## 기존 접근 방법

최근 RAG 시스템에서는 검색 정확도를 높이기 위해 다음과 같은 방법들이 널리 사용된다.

- Dense Retrieval
- BM25
- Hybrid Search
- Reranking

또한 Parent-Child Retrieval을 활용하여 작은 단위로 검색하고, 원문 문맥을 함께 제공하는 방식이 많이 활용되고 있다.

---

## Baseline

Baseline은 **실전 프로젝트 1에서 구축한 RAG 데이터셋**을 그대로 활용한 기본 RAG 시스템이다.

활용 데이터

- Parent-Child Chunk
- Vector Store
- 업무별 메타데이터
- 출처 URL
- 첨부파일 메타데이터

검색된 Chunk를 LLM에 전달하여 답변을 생성하는 기본 구조를 Baseline으로 설정한다.

---

# 3. Proposed Method

Baseline RAG 시스템을 기반으로 다음과 같은 개선 사항을 적용한다.

## 1. Hybrid Retrieval

Vector Search와 Keyword Search를 함께 활용하여 검색 정확도를 향상시킨다.

---

## 2. 업무 도메인 필터링

실전 프로젝트 1에서 구축한 `business_function` 메타데이터를 활용하여 질의를 6개 업무 중 하나로 분류하고 해당 업무 내에서만 검색을 수행한다.

이를 통해 서로 다른 업무의 유사 문서가 검색되는 문제를 최소화한다.

---

## 3. Parent-Child Retrieval

검색은 Child Chunk 단위로 수행하고, 답변 생성 시 Parent Document를 함께 제공하여 조건이나 예외 사항이 누락되지 않도록 한다.

---

## 4. 민원 처리 안내

질의를

- 정보 조회
- 민원 처리

로 구분한다.

민원 처리 질문인 경우에는

1. 신청 절차 안내
2. 첨부파일 다운로드 제공
3. 신청 페이지 URL 안내

를 순차적으로 제공한다.

이 과정에서는 실전 프로젝트 1에서 구축한 첨부파일 및 URL 메타데이터를 활용한다.

---

## 기대 효과

기존 Baseline 대비

- 검색 정확도 향상
- Hallucination 감소
- 답변 신뢰성 향상
- 사용자 편의성 향상

을 기대한다.

---

# 4. Dataset & Preprocessing

## Dataset

실전 프로젝트 1에서 구축한 예금보험공사 데이터를 사용한다.

데이터는

- Markdown 문서
- Parent-Child Chunk
- 업무 메타데이터
- Breadcrumb
- 출처 URL
- 첨부파일 메타데이터

로 구성되어 있다.

---

## 전처리

실전 프로젝트 1에서

- HTML → Markdown 변환
- Parent-Child Chunk 생성
- 메타데이터 구축
- Vector Store 적재

를 완료하였다.

본 프로젝트에서는 구축된 데이터를 기반으로 Retrieval 및 Question Answering 시스템을 구현한다.

---

# 5. Experiment Design

## 실험 내용

다음 Retrieval 방식을 비교한다.

- Dense Retrieval
- BM25
- Hybrid Search
- Hybrid Search + Reranking

또한

- 업무 도메인 필터링 적용 여부
- Parent-Child Retrieval 적용 여부

에 따른 성능 차이를 비교한다.

---

## Evaluation Metrics

### Retrieval

- Recall@K
- Hit@K

### Answer Quality

- Answer Correctness
- Faithfulness
- Groundedness

### 민원 처리 기능

- 절차 안내 성공 여부
- 첨부파일 안내 여부
- 신청 페이지 연결 여부

실전 프로젝트 1에서 구축한 평가 테스트셋을 이용하여 성능을 비교한다.

---

# 6. Plan

| 일정 | 수행 내용 |
|------|-----------|
| Day 1 | 실전 프로젝트 1 산출물 검토 및 Baseline 구축 |
| Day 2 | 기본 RAG 파이프라인 구현 |
| Day 3 | Hybrid Search 및 업무 도메인 필터링 구현 |
| Day 4 | Parent-Child Retrieval 및 민원 처리 기능 구현 |
| Day 5 | 검색 성능 및 답변 품질 평가 |
| Day 6 | 검색 파라미터 및 프롬프트 최적화 |
| Day 7 | 데모 UI 구현 및 최종 발표 준비 |

---

# 기대 산출물

- RAG 기반 AI 챗봇
- 검색 및 답변 파이프라인
- 성능 평가 리포트
- 대화형 웹 데모
