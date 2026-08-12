# KDIC Lightweight RAG V3.1 구현 보고서

구현일: 2026-08-12  
파이프라인 버전: `KDIC_LIGHTWEIGHT_RAG_QUERY_ANALYZER_V3_1_2026_08_12`

## 1. 구현 목적

V3는 라우팅 정확도 100%를 달성했지만 227개의 Hard Filter 중 4개가 정답 업무를 검색 대상에서 제거했다. V3.1은 라우팅 정책을 다시 보수적인 확인 질문으로 되돌리지 않으면서, 업무 불확실성을 `SOFT` 또는 `RETRIEVE_RELAXED`로 처리한다.

핵심 원칙은 다음과 같다.

> `HARD`는 기본 필터가 아니라 need 단위 안전조건을 모두 통과한 경우에만 허용한다.

## 2. 기존 V3 보존

기존 V3 빌더, 노트북, 도식, 배포 ZIP은 수정하지 않았다. V3.1은 별도 폴더에서 V3 노트북을 읽어 새 노트북을 생성한다.

## 3. 구현한 변경사항

### 3.1 복합 질문 업무 전파 제한

V3는 need의 부분 질문에서 업무를 찾지 못하면 전체 원문에서 발견한 단일 업무를 모든 need에 적용했다. 이 구조는 Q026의 첫 번째 need처럼 올바른 모델 업무를 전체 원문의 다른 업무 키워드로 덮어쓸 수 있다.

V3.1은 전체 원문의 업무 fallback을 need가 하나인 경우에만 허용한다. 복합 질문은 각 need의 부분 질문과 모델 분석 결과를 유지한다.

### 3.2 강한 업무 근거와 약한 근거 분리

다음과 같이 업무를 직접 지칭하는 표현은 강한 근거로 사용한다.

- `예금자보호`, `보호한도`
- `예금보험금`, `보험금 지급`
- `미수령금`, `파산배당금`, `지급대행점`
- `착오송금`, `반환지원`
- `채무조정`, `개인회생`, `워크아웃`
- `은닉재산`, `차명재산`, `신고 포상금`

`보험사고`, `가지급금`, `개산지급금`처럼 여러 업무에서 사용될 수 있는 표현은 약하거나 교차 가능한 근거로 처리하며, 이 표현만으로 Hard Filter를 허용하지 않는다.

### 3.3 need 단위 Filter Safety 정보

각 need에 다음 정보를 기록한다.

```json
{
  "business_candidates": [],
  "decomposition_status": "COMPLETE",
  "model_rule_conflict": false,
  "cross_business_ambiguity": false,
  "business_confidence": 0.99,
  "business_candidate_margin": 1.0,
  "hard_filter_eligible": true,
  "hard_filter_denial_reasons": []
}
```

Hard Filter를 거부할 때는 다음과 같은 사유를 남긴다.

- `INCOMPLETE_DECOMPOSITION`
- `NO_BUSINESS_CANDIDATE`
- `MULTIPLE_BUSINESS_CANDIDATES`
- `MODEL_RULE_CONFLICT`
- `CROSS_BUSINESS_AMBIGUITY`
- `NO_STRONG_EXPLICIT_EVIDENCE`
- `LOW_BUSINESS_CONFIDENCE`
- `LOW_CANDIDATE_MARGIN`

### 3.4 HARD allow-only 정책

다음 조건을 모두 만족해야 한다.

1. 분해 상태가 `COMPLETE`이다.
2. 업무 후보가 정확히 하나다.
3. 모델과 규칙이 충돌하지 않는다.
4. 업무 간 모호성이 없다.
5. 강한 업무 근거 또는 사용자가 확인한 문맥이 있다.
6. 업무 confidence가 0.98 이상이다.
7. 후보 점수 차이가 0.20 이상이다.

하나라도 실패하면 업무가 있을 때 `SOFT`, 업무가 없을 때 `NONE`과 `RETRIEVE_RELAXED`를 사용한다.

### 3.5 Fail-open 검색 계약

실제 검색기는 이 작업 폴더에 포함되어 있지 않다. 따라서 V3.1 query plan에 검색기가 구현해야 하는 fallback 계약을 포함했다.

```json
{
  "fallback_policy": {
    "enabled": true,
    "on": ["NO_RESULTS", "LOW_TOP_SCORE", "LOW_COVERAGE"],
    "next_filter_modes": ["SOFT", "NONE"],
    "fail_open": true
  }
}
```

`relax_query_plan_v31()` helper도 제공하여 검색기가 Hard Filter를 Soft Filter로, Soft Filter를 None으로 완화할 수 있게 했다.

### 3.6 N2 Intent override 제한

V3 평가에서 `N2_RULE_OVERRIDE`는 8회 적용되어 개선 0건, 회귀 3건이었다. V3.1은 N2 이상의 알려진 모델 intent를 규칙이 강제로 덮어쓰지 않는다. 규칙과 모델이 충돌하면 모델 intent를 유지하고 `RULE_OVERRIDE_BLOCKED_V31`을 기록한다.

### 3.7 하드 게이트 평가 내장

V3.1 평가 셀은 다음을 직접 계산한다.

- 실행 성공률
- RETRIEVE 검색계획 유효율
- False OOS 수
- False DIRECT 수
- Hard Filter 수
- Wrong Hard Filter 수
- CLARIFY Precision과 Recall
- `hard_gate_pass`

하드 게이트 실패 문항은 별도 CSV로 저장한다.

## 4. 오프라인 정책 재생 결과

기존 V3의 270개 raw 결과에서 최종 need를 가져와 V3.1 필터 정책을 재생했다. 외부 API는 호출하지 않았다.

| 필터 모드 | V3 | V3.1 재생 | 변화 |
|---|---:|---:|---:|
| `HARD` | 227 | 193 | -34 |
| `SOFT` | 34 | 68 | +34 |
| `NONE` | 4 | 4 | 0 |

| 안전성 결과 | V3 | V3.1 재생 |
|---|---:|---:|
| Wrong Hard Filter | 4 | **0** |
| Q026 Hard Filter | 2 | **0** |
| Q074 Hard Filter | 1 | **0** |
| Q083 Hard Filter | 1 | **0** |

V3.1은 모든 Hard Filter를 끈 것이 아니다. 전체 265개 검색계획 중 193개는 계속 Hard Filter를 사용하고, 안전조건을 통과하지 못한 34개만 추가로 Soft Filter로 내렸다.

## 5. 검증 결과

- V3.1 노트북 16개 셀 및 11개 코드 셀 확인
- Colab magic을 제외한 모든 코드 셀 Python 문법검사 통과
- 파이프라인 메타데이터 `v3.1` 확인
- `hard_filter_eligible`, denial reason, fallback policy 필드 확인
- Wrong Hard Filter 하드 게이트 계산 코드 확인
- N2 override 차단 코드 확인
- 배포 ZIP 구성 확인
- 기존 V3 파일의 크기와 수정시각 유지 확인

## 6. 아직 완료되지 않은 검증

HCX-007 API를 새로 호출하는 FULL 평가는 실행하지 않았다. 따라서 다음 값은 Colab에서 V3.1 노트북을 실행한 후 확인해야 한다.

- 새로운 모델 출력에서 Wrong Hard Filter가 실제로 0건인지
- 라우팅 정확도 100% 유지 여부
- Core Exact와 요청 쌍 F1 변화
- N2 override 제한에 따른 개선·회귀
- 토큰과 지연시간
- 실제 검색기의 `Recall@K`

오프라인 재생은 필터 안전정책의 효과를 검증하지만 새로운 모델 응답과 실제 문서 검색 결과를 대신하지 않는다.

## 7. 권장 실행 순서

1. V3.1 Colab 노트북을 연다.
2. `HCX_API_KEY`를 Colab Secret으로 등록한다.
3. 빠른 기능 확인에서 Q026·Q074·Q083의 filter mode가 `SOFT`인지 확인한다.
4. `SMOKE` 평가를 먼저 실행한다.
5. 하드 게이트 결과가 정상일 때만 `FULL` 270개 평가를 실행한다.
6. `hard_gate_pass == true`와 Wrong Hard Filter 0건을 확인한다.
7. V3 대비 Core Exact·요청 쌍 F1·MULTI 성능의 회귀가 없는지 확인한다.
8. 실제 검색기를 연결한 후 `Recall@5`와 fail-open 동작을 검증한다.

