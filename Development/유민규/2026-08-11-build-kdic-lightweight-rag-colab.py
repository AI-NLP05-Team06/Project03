from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "2026-08-11-KDIC-경량-RAG-질의분석-Colab.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(source).strip() + "\n"}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip() + "\n",
    }


cells = [
    md(
        """
        # KDIC 경량 RAG 질의 분석 파이프라인

        이 노트북은 RAG 검색 직전의 최소 Query Plan을 생성합니다.

        - 고정밀 Fast Path: 명확한 인사·감사·사용법·범위 외만 0회 호출로 처리
        - 일반 질의: HCX-007 네이티브 Structured Outputs 1회
        - 복합질의: 한 번의 호출에서 Need별 검색어·업무·Intent 생성
        - 일부 필드 오류: 코드 정규화 후 Soft/No Filter
        - API·분석 실패: `FALLBACK`으로 명시하고 원문 Broad Retrieval
        - 사용자에게 질문해야만 해결되는 필수 정보 누락일 때만 `CLARIFY`

        이 노트북은 검색 실행·Reranking·답변 생성을 포함하지 않습니다.
        """
    ),
    md(
        """
        ## 경량 흐름

        ```mermaid
        flowchart TD
            A["사용자 원문"] --> B["최소 정규화·확정 문맥"]
            B --> C["고정밀 Fast Path"]
            C --> D{"Fast Path 확정?"}
            D -- "예" --> E["0회 호출 계획"]
            D -- "아니오" --> F["HCX-007 통합 분석 1회"]
            F --> G["간단한 Schema·Enum 검증"]
            G --> H{"Valid?"}
            H -- "예" --> I["Need별 Query Plan"]
            H -- "일부 오류" --> J["코드 정규화·필터 완화"]
            H -- "전체 실패" --> K["원문 Broad Retrieval"]
            J --> I
            K --> I
            E --> L["Hybrid Retriever"]
            I --> L
        ```
        """
    ),
    code("""
        %pip -q install -U requests pandas openpyxl
    """),
    code(
        '''
        from __future__ import annotations

        import json
        import math
        import os
        import random
        import re
        import time
        import unicodedata
        import uuid
        import zipfile
        from collections import Counter
        from dataclasses import asdict, dataclass
        from pathlib import Path
        from typing import Any, Sequence

        import pandas as pd
        import requests

        PIPELINE_VERSION = "KDIC_LIGHTWEIGHT_RAG_QUERY_ANALYZER_V1_2026_08_11"

        BUSINESS_FUNCTIONS = [
            "예금자보호제도",
            "예금보험금 안내",
            "고객 미수령금 신청",
            "착오송금 반환 신청",
            "채무조정 안내",
            "은닉재산 신고",
        ]
        INTENTS = ["AMOUNT", "ELIGIBILITY", "TIME", "APPLICATION", "OVERVIEW", "STATUS", "DOCUMENTS", "CONTACT"]
        ROUTES = ["RETRIEVE", "CLARIFY", "DIRECT", "OUT_OF_SCOPE"]
        APPLICANT_TYPES = ["SELF", "PROXY", "HEIR", "LEGAL_REPRESENTATIVE", "CORPORATION"]
        USER_ROLES = ["DEPOSITOR", "SENDER", "RECIPIENT", "DEBTOR", "REPORTER", "CLAIMANT", "GENERAL_USER"]
        MISSING_FIELDS = ["business_function", "applicant_type", "target_type", "case_details"]

        INTENT_ALIASES = {
            "신청 방법": "APPLICATION", "신청 절차": "APPLICATION", "접수 방법": "APPLICATION",
            "필요 서류": "DOCUMENTS", "구비서류": "DOCUMENTS", "준비 서류": "DOCUMENTS",
            "자격": "ELIGIBILITY", "대상": "ELIGIBILITY", "조건": "ELIGIBILITY",
            "금액": "AMOUNT", "한도": "AMOUNT", "보호한도": "AMOUNT",
            "기간": "TIME", "기한": "TIME", "처리 기간": "TIME",
            "조회": "STATUS", "진행 상태": "STATUS", "처리 상태": "STATUS",
            "문의": "CONTACT", "연락처": "CONTACT",
            "안내": "OVERVIEW", "개요": "OVERVIEW", "무엇": "OVERVIEW",
        }

        BUSINESS_KEYWORDS = {
            "예금자보호제도": ["예금자보호", "보호한도", "보호대상", "예금 보호"],
            "예금보험금 안내": ["예금보험금", "보험금 지급", "보험사고", "가지급금"],
            "고객 미수령금 신청": ["미수령금", "파산배당금", "개산지급금 정산금", "상속인 금융거래 조회"],
            "착오송금 반환 신청": ["착오송금", "잘못 보낸 돈", "잘못 송금", "착오 송금"],
            "채무조정 안내": ["채무조정", "신용회복지원", "파산선고", "면책", "채무감면"],
            "은닉재산 신고": ["은닉재산", "은닉 재산"],
        }

        EXPLICIT_TYPO_MAP = (
            ("예금보헝금", "예금보험금"),
            ("예금보혐금", "예금보험금"),
            ("착오송금반한", "착오송금 반환"),
            ("미수령금신정", "미수령금 신청"),
        )

        @dataclass(frozen=True)
        class PipelineConfig:
            model: str = "HCX-007"
            base_url: str = "https://clovastudio.stream.ntruss.com"
            timeout_seconds: float = 90.0
            max_api_attempts: int = 2
            max_completion_tokens: int = 1400
            temperature: float = 0.1
            top_p: float = 0.8
            top_k: int = 0
            request_interval_seconds: float = 0.3
            max_context_turns: int = 2
            intent_soft_boost: float = 0.15

        class HCXAPIError(RuntimeError):
            def __init__(self, message: str, *, error_type: str, telemetry: dict[str, Any]):
                super().__init__(message)
                self.error_type = error_type
                self.telemetry = telemetry

        def get_hcx_api_key(secret_name: str = "HCX_API_KEY") -> str:
            try:
                from google.colab import userdata
                value = str(userdata.get(secret_name) or "").strip()
            except ImportError:
                value = os.getenv(secret_name, "").strip()
            if not value:
                raise ValueError(f"Colab Secrets 또는 환경변수에 {secret_name}가 없습니다.")
            if value.lower().startswith("bearer ") or any(ch.isspace() for ch in value):
                raise ValueError(f"{secret_name}에는 Bearer 접두사 없이 API 키 값만 저장하세요.")
            return value
        '''
    ),
    code(
        '''
        FOLLOW_UP_PATTERN = re.compile(r"(?:그럼|그러면|그거|그건|그 경우|이거|이건|이 경우|앞서|방금|그때|그것|그 서류|그 신청)")

        def normalize_query(text: str) -> dict[str, Any]:
            original = str(text or "")
            value = unicodedata.normalize("NFKC", original)
            changes = []
            cleaned = re.sub(r"[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f\\x7f]", " ", value)
            if cleaned != value:
                changes.append("CONTROL_CHARACTER")
                value = cleaned
            cleaned = re.sub(r"([!?ㅋㅎㅠㅜ])\\1{2,}", r"\\1\\1", value)
            if cleaned != value:
                changes.append("REPEATED_CHARACTER")
                value = cleaned
            for wrong, correct in EXPLICIT_TYPO_MAP:
                if wrong in value:
                    value = value.replace(wrong, correct)
                    changes.append("EXPLICIT_TYPO")
            cleaned = re.sub(r"\\s+", " ", value).strip()
            if cleaned != value:
                changes.append("WHITESPACE")
            if not cleaned:
                raise ValueError("사용자 질의가 비어 있습니다.")
            return {"original_query": original, "normalized_query": cleaned, "changes": changes}

        def build_context(query: str, conversation_state: dict[str, Any] | None, max_turns: int) -> dict[str, Any]:
            state = dict(conversation_state or {})
            if not FOLLOW_UP_PATTERN.search(query):
                return {"used": False, "confirmed": {}, "recent_turns": []}
            confirmed = state.get("confirmed") if isinstance(state.get("confirmed"), dict) else {}
            turns = state.get("recent_turns") if isinstance(state.get("recent_turns"), list) else []
            return {"used": True, "confirmed": confirmed, "recent_turns": turns[-max_turns:]}

        def exact_fullmatch(pattern: str, query: str) -> bool:
            return re.fullmatch(pattern, query.strip(), flags=re.I) is not None

        def detect_fast_path(query: str, conversation_state: dict[str, Any] | None = None) -> dict[str, Any] | None:
            # 문장 전체가 규칙에 일치할 때만 처리해 실제 질문을 잘라내지 않는다.
            detectors = []
            if exact_fullmatch(r"(?:안녕|안녕하세요|반갑습니다|반가워요)[.!?]*", query):
                detectors.append({"route": "DIRECT", "action": "GREETING"})
            if exact_fullmatch(r"(?:고마워요|고맙습니다|감사합니다|도움이 됐어요|알겠습니다)[.!?]*", query):
                detectors.append({"route": "DIRECT", "action": "ACKNOWLEDGEMENT"})
            if exact_fullmatch(r"(?:무슨 질문을 할 수 있나요|지원하는 업무를 (?:알려주세요|목록으로 보여주세요)|이 챗봇은 어떻게 사용하면 되나요)[.!?]*", query):
                detectors.append({"route": "DIRECT", "action": "CAPABILITY_GUIDE"})
            has_previous = bool((conversation_state or {}).get("has_previous_answer"))
            if has_previous and exact_fullmatch(r"(?:쉽게 설명해 주세요|핵심만 알려주세요|표로 정리해 주세요|더 자세히 설명해 주세요)[.!?]*", query):
                detectors.append({"route": "DIRECT", "action": "REFORMAT_PREVIOUS_ANSWER"})
            if exact_fullmatch(r"(?:오늘|내일|이번 주말)?\\s*(?:서울 )?(?:날씨|기온|미세먼지)(?:를|가|은|는)?.*", query):
                detectors.append({"route": "OUT_OF_SCOPE", "action": "EXPLICIT_WEATHER"})
            if exact_fullmatch(r"(?:주식 종목을 추천해 주세요|로또 번호를 알려주세요)[.!?]*", query):
                detectors.append({"route": "OUT_OF_SCOPE", "action": "EXPLICIT_NON_KDIC"})
            return detectors[0] if len(detectors) == 1 else None

        def find_businesses(text: str) -> list[str]:
            compact = re.sub(r"\\s+", "", text)
            found = []
            for business, keywords in BUSINESS_KEYWORDS.items():
                if any(re.sub(r"\\s+", "", keyword) in compact for keyword in keywords):
                    found.append(business)
            return found

        def normalize_intent(value: Any) -> str | None:
            if value is None:
                return None
            text = re.sub(r"\\s+", " ", str(value)).strip()
            if text in INTENTS:
                return text
            if text.upper() in INTENTS:
                return text.upper()
            return INTENT_ALIASES.get(text)
        '''
    ),
    code(
        '''
        def query_analysis_schema() -> dict[str, Any]:
            # HCX-007 공식 지원 타입에 null이 없으므로 UNKNOWN/빈 문자열을 sentinel로 쓴다.
            business_value = {"type": "string", "enum": [*BUSINESS_FUNCTIONS, "UNKNOWN"]}
            intent_value = {"type": "string", "enum": [*INTENTS, "UNKNOWN"]}
            applicant_value = {"type": "string", "enum": [*APPLICANT_TYPES, "UNKNOWN"]}
            user_role_value = {"type": "string", "enum": [*USER_ROLES, "UNKNOWN"]}
            return {
                "type": "object",
                "properties": {
                    "route": {"type": "string", "enum": ROUTES},
                    "needs": {
                        "type": "array",
                        "maxItems": 6,
                        "items": {
                            "type": "object",
                            "properties": {
                                "need_id": {"type": "string"},
                                "query": {"type": "string"},
                                "business_function": business_value,
                                "business_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "intent": intent_value,
                                "intent_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "user_role": user_role_value,
                                "applicant_type": applicant_value,
                                "target_type": {"type": "string"},
                                "case_details": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": [
                                "need_id", "query", "business_function", "business_confidence",
                                "intent", "intent_confidence", "user_role",
                                "applicant_type", "target_type", "case_details",
                            ],
                        },
                    },
                    "missing_information": {"type": "array", "items": {"type": "string", "enum": MISSING_FIELDS}},
                },
                "required": ["route", "needs", "missing_information"],
            }

        SYSTEM_PROMPT = f"""
        역할: 예금보험공사 RAG 시스템의 경량 질의 분석기.
        목적: 사용자 요구를 검색 가능한 최소 Need로 나누고 각 Need의 검색어·업무·의도·핵심 조건을 반환한다.

        규칙:
        1. 서로 다른 정보 요구는 N1, N2, ...로 분리한다.
        2. query는 해당 Need를 독립적으로 검색할 수 있는 한국어 문장으로 쓴다.
        3. 원문이 이미 독립적이면 불필요하게 바꾸지 않는다.
        4. 확정 context는 후속 질문일 때만 사용한다.
        5. 근거가 없는 범주형 값은 UNKNOWN, target_type은 빈 문자열, 목록은 []로 두고 추측하지 않는다.
           business_confidence와 intent_confidence는 각각 해당 분류가 맞을 확률을 0~1로 쓴다.
        6. 업무가 불확실해도 원문으로 검색 가능하면 RETRIEVE다.
        7. 사용자만 제공할 수 있는 필수 정보가 없어 정답 대상이 바뀐 때만 CLARIFY다.
        8. 인사·감사·사용법·이전 답변 재구성은 DIRECT다.
        9. 예금보험공사 업무와 명확히 무관한 요청은 OUT_OF_SCOPE다.

        허용 business_function: {BUSINESS_FUNCTIONS}
        허용 intent: {INTENTS}
        허용 user_role: {USER_ROLES}
        허용 applicant_type: {APPLICANT_TYPES}
        """

        class HCX007StructuredClient:
            def __init__(self, api_key: str, config: PipelineConfig | None = None):
                self.config = config or PipelineConfig()
                self.api_key = api_key
                self.session = requests.Session()

            @property
            def url(self) -> str:
                return f"{self.config.base_url}/v3/chat-completions/{self.config.model}"

            def analyze(self, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
                started = time.perf_counter()
                attempts = []
                total_tokens = 0
                last_error = None
                body = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                    "topP": self.config.top_p,
                    "topK": self.config.top_k,
                    "maxCompletionTokens": self.config.max_completion_tokens,
                    "temperature": self.config.temperature,
                    "repetitionPenalty": 1.05,
                    "thinking": {"effort": "none"},
                    "stop": [],
                    "responseFormat": {"type": "json", "schema": query_analysis_schema()},
                }
                for attempt in range(1, self.config.max_api_attempts + 1):
                    attempt_started = time.perf_counter()
                    request_id = str(uuid.uuid4())
                    try:
                        response = self.session.post(
                            self.url,
                            headers={
                                "Authorization": f"Bearer {self.api_key}",
                                "X-NCP-CLOVASTUDIO-REQUEST-ID": request_id,
                                "Content-Type": "application/json",
                                "Accept": "application/json",
                            },
                            json=body,
                            timeout=self.config.timeout_seconds,
                        )
                        if response.status_code >= 400:
                            try:
                                error_body = response.json()
                            except Exception:
                                error_body = {"text": response.text[:1000]}
                            retryable = response.status_code in {408, 429, 500, 502, 503, 504}
                            attempts.append({
                                "attempt": attempt, "request_id": request_id, "http_status": response.status_code,
                                "retryable": retryable, "error": error_body,
                                "latency_ms": round((time.perf_counter() - attempt_started) * 1000, 3),
                            })
                            last_error = f"HTTP {response.status_code}: {error_body}"
                            if retryable and attempt < self.config.max_api_attempts:
                                retry_after = response.headers.get("Retry-After")
                                delay = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else 1.5 + random.random()
                                time.sleep(min(delay, 10.0))
                                continue
                            raise HCXAPIError(last_error, error_type="API_RETRYABLE" if retryable else "API_FATAL", telemetry={
                                "api_request_count": len(attempts), "attempts": attempts, "total_tokens": total_tokens,
                                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                            })

                        envelope = response.json()
                        result = envelope.get("result", envelope)
                        usage = result.get("usage") or {}
                        used = int(usage.get("totalTokens") or 0)
                        total_tokens += used
                        content = str((result.get("message") or {}).get("content") or "")
                        parsed = json.loads(content)
                        if not isinstance(parsed, dict):
                            raise TypeError("모델 결과의 최상위가 object가 아닙니다.")
                        attempts.append({
                            "attempt": attempt, "request_id": request_id, "http_status": response.status_code,
                            "tokens": used, "finish_reason": result.get("finishReason"),
                            "latency_ms": round((time.perf_counter() - attempt_started) * 1000, 3),
                        })
                        return parsed, {
                            "api_request_count": len(attempts), "attempts": attempts, "total_tokens": total_tokens,
                            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                        }
                    except HCXAPIError:
                        raise
                    except (requests.Timeout, requests.ConnectionError) as exc:
                        last_error = f"{type(exc).__name__}: {exc}"
                        attempts.append({
                            "attempt": attempt, "request_id": request_id, "retryable": True, "error": last_error,
                            "latency_ms": round((time.perf_counter() - attempt_started) * 1000, 3),
                        })
                        if attempt < self.config.max_api_attempts:
                            time.sleep(1.5 + random.random())
                            continue
                    except Exception as exc:
                        last_error = f"{type(exc).__name__}: {exc}"
                        attempts.append({
                            "attempt": attempt, "request_id": request_id, "retryable": False, "error": last_error,
                            "latency_ms": round((time.perf_counter() - attempt_started) * 1000, 3),
                        })
                        break
                raise HCXAPIError(last_error or "HCX-007 호출 실패", error_type="API_OR_PARSE_ERROR", telemetry={
                    "api_request_count": len(attempts), "attempts": attempts, "total_tokens": total_tokens,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                })
        '''
    ),
    code(
        '''
        def validate_analysis(raw: dict[str, Any], normalized_query: str) -> tuple[dict[str, Any], list[str]]:
            warnings = []
            route = raw.get("route")
            if route not in ROUTES:
                route = "RETRIEVE"
                warnings.append("INVALID_ROUTE_TO_RETRIEVE")
            rows = raw.get("needs")
            if not isinstance(rows, list):
                rows = []
                warnings.append("NEEDS_NOT_LIST")
            needs = []
            for index, row in enumerate(rows, 1):
                if not isinstance(row, dict):
                    warnings.append(f"N{index}_NOT_OBJECT")
                    continue
                query = str(row.get("query") or "").strip()
                if not query:
                    query = normalized_query
                    warnings.append(f"N{index}_EMPTY_QUERY_USED_ORIGINAL")
                business = row.get("business_function")
                if business not in BUSINESS_FUNCTIONS:
                    if business not in (None, "", "UNKNOWN"):
                        warnings.append(f"N{index}_INVALID_BUSINESS_TO_NULL")
                    business = None
                intent = normalize_intent(row.get("intent"))
                if row.get("intent") not in (None, "", "UNKNOWN") and intent is None:
                    warnings.append(f"N{index}_INVALID_INTENT_TO_NULL")
                applicant = row.get("applicant_type")
                if applicant not in APPLICANT_TYPES:
                    applicant = None
                user_role = row.get("user_role")
                if user_role not in USER_ROLES:
                    user_role = None
                try:
                    business_confidence = min(1.0, max(0.0, float(row.get("business_confidence") or 0.0)))
                except (TypeError, ValueError):
                    business_confidence = 0.0
                    warnings.append(f"N{index}_INVALID_BUSINESS_CONFIDENCE_TO_ZERO")
                try:
                    intent_confidence = min(1.0, max(0.0, float(row.get("intent_confidence") or 0.0)))
                except (TypeError, ValueError):
                    intent_confidence = 0.0
                    warnings.append(f"N{index}_INVALID_INTENT_CONFIDENCE_TO_ZERO")
                target = row.get("target_type")
                target = str(target).strip() if target not in (None, "") else None
                cases = row.get("case_details")
                cases = [str(x).strip() for x in cases if str(x).strip()] if isinstance(cases, list) else []
                needs.append({
                    "need_id": f"N{index}", "query": query, "business_function": business,
                    "business_confidence": business_confidence,
                    "intent": intent, "intent_confidence": intent_confidence,
                    "user_role": user_role, "applicant_type": applicant, "target_type": target,
                    "case_details": cases,
                })
            if route == "RETRIEVE" and not needs:
                needs = [{
                    "need_id": "N1", "query": normalized_query, "business_function": None,
                    "business_confidence": 0.0, "intent": None, "intent_confidence": 0.0,
                    "user_role": None, "applicant_type": None, "target_type": None, "case_details": [],
                }]
                warnings.append("EMPTY_RETRIEVE_NEEDS_USED_ORIGINAL")
            if route in {"DIRECT", "OUT_OF_SCOPE"}:
                needs = []
            missing = raw.get("missing_information")
            missing = [x for x in missing if x in MISSING_FIELDS] if isinstance(missing, list) else []
            return {"route": route, "needs": needs, "missing_information": missing}, warnings

        def build_keyword_query(need: dict[str, Any]) -> str:
            values = [
                need.get("business_function"), need.get("intent"), need.get("user_role"),
                need.get("applicant_type"), need.get("target_type"),
            ]
            values.extend(need.get("case_details") or [])
            return " ".join(str(x) for x in values if x)

        def determine_filter_policy(need: dict[str, Any], original: str, context: dict[str, Any], manual: dict[str, Any]) -> dict[str, Any]:
            business = need.get("business_function")
            if business not in BUSINESS_FUNCTIONS:
                return {"mode": "NONE", "value": None, "soft_hint": None, "evidence": "UNKNOWN"}
            if manual.get("business_function") == business:
                return {"mode": "HARD", "value": business, "soft_hint": None, "evidence": "MANUAL"}
            explicit = business in find_businesses(original)
            if explicit:
                return {"mode": "HARD", "value": business, "soft_hint": None, "evidence": "ORIGINAL"}
            if context.get("confirmed", {}).get("business_function") == business:
                return {"mode": "HARD", "value": business, "soft_hint": None, "evidence": "CONTEXT"}
            if float(need.get("business_confidence") or 0.0) >= 0.65:
                return {"mode": "SOFT", "value": None, "soft_hint": business, "evidence": "MODEL"}
            return {"mode": "NONE", "value": None, "soft_hint": None, "evidence": "LOW_CONFIDENCE_MODEL"}

        def build_query_plans(analysis: dict[str, Any], original: str, context: dict[str, Any], manual: dict[str, Any], config: PipelineConfig) -> list[dict[str, Any]]:
            if analysis["route"] != "RETRIEVE":
                return []
            plans = []
            for need in analysis["needs"]:
                policy = determine_filter_policy(need, original, context, manual)
                plans.append({
                    "need_id": need["need_id"],
                    "semantic_query": need["query"],
                    "keyword_query": build_keyword_query(need) or need["query"],
                    "business_filter": policy,
                    "intent_boost": {
                        "mode": "SOFT" if need.get("intent") in INTENTS else "NONE",
                        "value": need.get("intent"),
                        "weight": config.intent_soft_boost if need.get("intent") in INTENTS else 0.0,
                    },
                    "entities": {
                        "user_role": need.get("user_role"),
                        "applicant_type": need.get("applicant_type"),
                        "target_type": need.get("target_type"),
                        "case_details": need.get("case_details") or [],
                    },
                })
            return plans

        GENERIC_CLARIFY_PATTERN = re.compile(r"^(?:신청 방법|필요한 서류|제출해야 하는 서류|신청 기한|처리 기간|문의처)(?:을|가|은|는|이|가)?(?: 어떻게 되나요| 언제까지인가요| 알려주세요| 무엇인가요)?[.!?]*$")

        def fallback_analysis(normalized_query: str, context: dict[str, Any], reason: str) -> dict[str, Any]:
            businesses = find_businesses(normalized_query)
            confirmed_business = context.get("confirmed", {}).get("business_function")
            if not businesses and not confirmed_business and GENERIC_CLARIFY_PATTERN.fullmatch(normalized_query):
                return {
                    "route": "CLARIFY", "needs": [], "missing_information": ["business_function"],
                    "fallback_reason": reason,
                }
            return {
                "route": "RETRIEVE",
                "needs": [{
                    "need_id": "N1", "query": normalized_query,
                    "business_function": businesses[0] if len(businesses) == 1 else confirmed_business,
                    "business_confidence": 1.0 if len(businesses) == 1 else 0.0,
                    "intent": None, "intent_confidence": 0.0,
                    "user_role": None, "applicant_type": None, "target_type": None, "case_details": [],
                }],
                "missing_information": [], "fallback_reason": reason,
            }

        class KDICLightweightRAGAnalyzer:
            def __init__(self, client: HCX007StructuredClient, config: PipelineConfig | None = None):
                self.client = client
                self.config = config or client.config

            def run(self, query: str, *, conversation_state=None, manual_selection=None) -> dict[str, Any]:
                started = time.perf_counter()
                normalized = normalize_query(query)
                original = normalized["original_query"]
                text = normalized["normalized_query"]
                manual = dict(manual_selection or {})
                context = build_context(text, conversation_state, self.config.max_context_turns)

                fast = detect_fast_path(text, conversation_state)
                if fast:
                    analysis = {"route": fast["route"], "needs": [], "missing_information": []}
                    return {
                        "pipeline_version": PIPELINE_VERSION, "analysis_status": "FAST_PATH",
                        "original_query": original, "normalized_query": text, "context": context,
                        "analysis": analysis, "fast_path": fast, "query_plans": [],
                        "runtime": {
                            "api_request_count": 0, "total_tokens": 0,
                            "latency_ms": round((time.perf_counter() - started) * 1000, 3), "attempts": [],
                        },
                    }

                payload = {
                    "query": text,
                    "confirmed_context": context["confirmed"],
                    "recent_turns": context["recent_turns"],
                    "manual_selection": manual,
                }
                try:
                    raw, telemetry = self.client.analyze(payload)
                    analysis, warnings = validate_analysis(raw, text)
                    status = "REPAIRED" if warnings else "OK"
                except HCXAPIError as exc:
                    telemetry = exc.telemetry
                    analysis = fallback_analysis(text, context, f"{exc.error_type}: {exc}")
                    warnings = ["MODEL_ANALYSIS_FAILED_USED_FALLBACK"]
                    status = "FALLBACK"

                plans = build_query_plans(analysis, original, context, manual, self.config)
                return {
                    "pipeline_version": PIPELINE_VERSION, "analysis_status": status,
                    "original_query": original, "normalized_query": text, "context": context,
                    "analysis": analysis, "validation_warnings": warnings, "query_plans": plans,
                    "runtime": {
                        **telemetry,
                        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                    },
                }

        CFG = PipelineConfig()
        HCX_CLIENT = HCX007StructuredClient(get_hcx_api_key("HCX_API_KEY"), CFG)
        PIPELINE = KDICLightweightRAGAnalyzer(HCX_CLIENT, CFG)
        print("파이프라인 준비 완료:", PIPELINE_VERSION)
        '''
    ),
    md(
        """
        ## 단일·복합·후속 질의 테스트

        전체 평가 전에 이 세 유형의 `analysis_status`, `needs`, `query_plans`, `api_request_count`를 확인합니다.
        """
    ),
    code(
        '''
        test_queries = [
            "예금자보호 한도는 얼마인가요?",
            "예금 보호한도와 예금보험금 신청 서류를 알려주세요.",
            "안녕하세요.",
        ]
        for q in test_queries:
            result = PIPELINE.run(q)
            print("\\nQUERY:", q)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        '''
    ),
    code(
        '''
        state = {
            "confirmed": {"business_function": "예금보험금 안내", "applicant_type": "HEIR"},
            "recent_turns": [{"user": "상속인이 예금보험금을 신청하려고 해요."}],
            "has_previous_answer": True,
        }
        follow_up = PIPELINE.run("그럼 필요한 서류는요?", conversation_state=state)
        print(json.dumps(follow_up, ensure_ascii=False, indent=2))
        '''
    ),
    md(
        """
        ## Evaluation_DataSet_v5 평가

        처음에는 `SMOKE` 20건으로 Schema·Route·MULTI 분해·실제 API 호출 수를 확인하세요.
        기준을 통과한 뒤 `FULL`로 변경합니다.
        """
    ),
    code(
        '''
        EVAL_MODE = "SMOKE"  # 최종 검증 시 "FULL"
        SMOKE_LIMIT = 20
        PROGRESS_EVERY = 10

        try:
            from google.colab import files
            uploaded = files.upload()
            candidates = [Path(name) for name in uploaded if name.lower().endswith(".xlsx")]
        except ImportError:
            candidates = list(Path(".").glob("*.xlsx"))
        if not candidates:
            raise FileNotFoundError("Evaluation_DataSet_v5 XLSX 파일을 업로드하세요.")
        dataset_path = next((p for p in candidates if "evaluation_dataset_v5" in p.name.lower()), candidates[0])
        df = pd.read_excel(dataset_path, dtype=object)
        required = ["evaluation_id", "question", "route_type", "business_functions", "user_roles", "requests", "missing_slots"]
        missing_columns = [x for x in required if x not in df.columns]
        if missing_columns:
            raise ValueError(f"필수 칼럼 누락: {missing_columns}")

        def parse_list(value, eid, column):
            if value is None or (isinstance(value, float) and math.isnan(value)):
                return []
            if isinstance(value, list):
                return value
            parsed = json.loads(str(value).strip()) if str(value).strip() else []
            if not isinstance(parsed, list):
                raise TypeError(f"{eid}/{column}: list가 아님")
            return parsed

        gold_rows = []
        for _, row in df[required].dropna(subset=["question"]).iterrows():
            eid = str(row["evaluation_id"]).strip()
            route = str(row["route_type"]).strip()
            requests_gold = parse_list(row["requests"], eid, "requests")
            roles = parse_list(row["user_roles"], eid, "user_roles")
            pairs = [(x.get("business_function"), x.get("information_need")) for x in requests_gold]
            gold_rows.append({
                "evaluation_id": eid, "question": str(row["question"]).strip(), "gold_route": route,
                "gold_query_type": "MULTI" if route == "RETRIEVE" and len(requests_gold) >= 2 else ("SINGLE" if route == "RETRIEVE" else "NONE"),
                "gold_request_pairs": pairs,
                "gold_missing_slots": parse_list(row["missing_slots"], eid, "missing_slots"),
                "gold_user_roles": [x for x in roles if x in USER_ROLES],
                "gold_applicant_types": [x for x in roles if x in APPLICANT_TYPES],
            })
        if EVAL_MODE == "FULL":
            eval_rows = gold_rows
        else:
            # 정렬된 데이터의 앞부분만 뽑지 않고 Route/단일·복합 버킷을 번갈아 선택한다.
            smoke_buckets = {}
            for row in gold_rows:
                key = (row["gold_route"], row["gold_query_type"])
                smoke_buckets.setdefault(key, []).append(row)
            smoke_cursor = Counter()
            eval_rows = []
            while len(eval_rows) < min(SMOKE_LIMIT, len(gold_rows)):
                progressed = False
                for key, bucket in smoke_buckets.items():
                    cursor = smoke_cursor[key]
                    if cursor < len(bucket):
                        eval_rows.append(bucket[cursor])
                        smoke_cursor[key] += 1
                        progressed = True
                        if len(eval_rows) >= SMOKE_LIMIT:
                            break
                if not progressed:
                    break
        print("평가 건수:", len(eval_rows), "/ mode:", EVAL_MODE)
        print("표본 분포:", Counter((x["gold_route"], x["gold_query_type"]) for x in eval_rows))
        '''
    ),
    code(
        '''
        def counter_overlap(gold, pred):
            g, p = Counter(gold), Counter(pred)
            return sum((g & p).values()), sum((p - g).values()), sum((g - p).values())

        records = []
        raw_path = Path("kdic_lightweight_rag_eval_raw.jsonl")
        with raw_path.open("w", encoding="utf-8") as raw_file:
            for index, gold in enumerate(eval_rows, 1):
                try:
                    result = PIPELINE.run(gold["question"])
                except Exception as exc:
                    # 한 샘플의 예외 때문에 전체 평가가 중단되지 않게 하되,
                    # 실패를 정상 fallback과 구분해 결과에 명시한다.
                    fallback = PIPELINE.fallback_analysis(gold["question"], f"EVALUATION_ERROR: {type(exc).__name__}: {exc}")
                    result = {
                        "pipeline_version": PIPELINE_VERSION, "analysis_status": "ERROR",
                        "analysis": fallback, "validation_warnings": [str(exc)], "query_plans": [],
                        "runtime": {"api_request_count": 0, "total_tokens": 0, "latency_ms": 0.0},
                    }
                analysis = result["analysis"]
                pred_route_internal = analysis["route"]
                # 파이프라인 내부의 짧은 이름을 기존 평가셋 라벨에 맞춘다.
                pred_route = {"DIRECT": "DIRECT_RESPONSE"}.get(pred_route_internal, pred_route_internal)
                needs = analysis.get("needs") or []
                pred_query_type = "MULTI" if pred_route_internal == "RETRIEVE" and len(needs) >= 2 else ("SINGLE" if pred_route_internal == "RETRIEVE" else "NONE")
                pred_pairs = [(n.get("business_function"), n.get("intent")) for n in needs if n.get("business_function") and n.get("intent")]
                pred_user_roles = sorted({n.get("user_role") for n in needs if n.get("user_role")})
                pred_applicant_types = sorted({n.get("applicant_type") for n in needs if n.get("applicant_type")})
                retrieve_case = gold["gold_route"] == "RETRIEVE"
                clarify_case = gold["gold_route"] == "CLARIFY"
                pair_exact = Counter(gold["gold_request_pairs"]) == Counter(pred_pairs) if retrieve_case else None
                missing_exact = set(gold["gold_missing_slots"]) == set(analysis.get("missing_information") or []) if clarify_case else None
                route_correct = pred_route == gold["gold_route"]
                query_type_correct = pred_query_type == gold["gold_query_type"]
                core_exact = route_correct and (query_type_correct and pair_exact if retrieve_case else (missing_exact if clarify_case else True))
                runtime = result["runtime"]
                record = {
                    **gold, "analysis_status": result["analysis_status"], "pred_route": pred_route,
                    "pred_route_internal": pred_route_internal,
                    "pred_query_type": pred_query_type, "pred_request_pairs": pred_pairs,
                    "pred_user_roles": pred_user_roles, "pred_applicant_types": pred_applicant_types,
                    "pred_missing_slots": analysis.get("missing_information") or [],
                    "route_correct": route_correct, "query_type_correct": query_type_correct,
                    "request_pair_exact": pair_exact, "missing_slot_exact": missing_exact,
                    "user_role_exact": set(gold["gold_user_roles"]) == set(pred_user_roles),
                    "applicant_type_exact": set(gold["gold_applicant_types"]) == set(pred_applicant_types),
                    "core_exact": bool(core_exact), "query_plan_count": len(result["query_plans"]),
                    "api_request_count": int(runtime.get("api_request_count") or 0),
                    "total_tokens": int(runtime.get("total_tokens") or 0),
                    "wall_latency_ms": float(runtime.get("latency_ms") or 0),
                    "warning_count": len(result.get("validation_warnings") or []),
                }
                records.append(record)
                raw_file.write(json.dumps({"gold": gold, "result": result}, ensure_ascii=False, default=str) + "\\n")
                if index == 1 or index % PROGRESS_EVERY == 0 or index == len(eval_rows):
                    print(f"[{index}/{len(eval_rows)}] status={Counter(r['analysis_status'] for r in records)}")
                time.sleep(CFG.request_interval_seconds)

        result_df = pd.DataFrame(records)
        pair_tp = pair_fp = pair_fn = 0
        for _, row in result_df[result_df.gold_route == "RETRIEVE"].iterrows():
            tp, fp, fn = counter_overlap(row.gold_request_pairs, row.pred_request_pairs)
            pair_tp += tp; pair_fp += fp; pair_fn += fn
        pair_precision = pair_tp / (pair_tp + pair_fp) if pair_tp + pair_fp else 0.0
        pair_recall = pair_tp / (pair_tp + pair_fn) if pair_tp + pair_fn else 0.0
        pair_f1 = 2 * pair_precision * pair_recall / (pair_precision + pair_recall) if pair_precision + pair_recall else 0.0
        clarify_rows = result_df[result_df.gold_route == "CLARIFY"]
        pred_clarify = result_df[result_df.pred_route == "CLARIFY"]
        clarify_tp = int(((result_df.gold_route == "CLARIFY") & (result_df.pred_route == "CLARIFY")).sum())
        llm_rows = result_df[result_df.analysis_status != "FAST_PATH"]

        summary = {
            "pipeline_version": PIPELINE_VERSION, "evaluation_mode": EVAL_MODE, "evaluation_count": len(result_df),
            "analysis_success_rate": float(result_df.analysis_status.isin(["FAST_PATH", "OK", "REPAIRED"]).mean()),
            "llm_structured_success_rate": float(llm_rows.analysis_status.isin(["OK", "REPAIRED"]).mean()) if len(llm_rows) else 1.0,
            "fast_path_rate": float((result_df.analysis_status == "FAST_PATH").mean()),
            "fallback_rate": float((result_df.analysis_status == "FALLBACK").mean()),
            "repaired_rate": float((result_df.analysis_status == "REPAIRED").mean()),
            "route_accuracy": float(result_df.route_correct.mean()),
            "query_type_accuracy": float(result_df.query_type_correct.mean()),
            "core_exact": float(result_df.core_exact.mean()),
            "request_pair_micro_precision": pair_precision,
            "request_pair_micro_recall": pair_recall,
            "request_pair_micro_f1": pair_f1,
            "user_role_exact": float(result_df.user_role_exact.mean()),
            "applicant_type_exact": float(result_df.applicant_type_exact.mean()),
            "clarify_precision": clarify_tp / len(pred_clarify) if len(pred_clarify) else 0.0,
            "clarify_recall": clarify_tp / len(clarify_rows) if len(clarify_rows) else 0.0,
            "average_api_requests": float(result_df.api_request_count.mean()),
            "zero_call_rate": float((result_df.api_request_count == 0).mean()),
            "average_tokens": float(result_df.total_tokens.mean()),
            "average_latency_ms": float(result_df.wall_latency_ms.mean()),
            "p95_latency_ms": float(result_df.wall_latency_ms.quantile(0.95)),
        }
        for query_type in ["SINGLE", "MULTI", "NONE"]:
            subset = result_df[result_df.gold_query_type == query_type]
            if len(subset):
                key = query_type.lower()
                summary[f"{key}_route_accuracy"] = float(subset.route_correct.mean())
                summary[f"{key}_query_type_accuracy"] = float(subset.query_type_correct.mean())
                summary[f"{key}_average_api_requests"] = float(subset.api_request_count.mean())
                summary[f"{key}_average_latency_ms"] = float(subset.wall_latency_ms.mean())
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        '''
    ),
    code(
        '''
        summary_path = Path("kdic_lightweight_rag_eval_summary.json")
        detail_path = Path("kdic_lightweight_rag_eval_per_question.csv")
        confusion_path = Path("kdic_lightweight_rag_eval_route_confusion.csv")
        status_path = Path("kdic_lightweight_rag_eval_status.csv")
        zip_path = Path("kdic_lightweight_rag_evaluation_bundle.zip")
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        confusion = pd.crosstab(result_df.gold_route, result_df.pred_route, margins=True).reset_index()
        status_table = result_df.analysis_status.value_counts(dropna=False).rename_axis("analysis_status").reset_index(name="count")
        result_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
        confusion.to_csv(confusion_path, index=False, encoding="utf-8-sig")
        status_table.to_csv(status_path, index=False, encoding="utf-8-sig")

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in [summary_path, detail_path, confusion_path, status_path, raw_path]:
                archive.write(path, arcname=path.name)
        print("생성:", zip_path.resolve())
        try:
            from google.colab import files
            files.download(str(zip_path))
        except ImportError:
            pass
        '''
    ),
    md(
        """
        ## FULL 실행 전 통과 기준

        - Schema Valid Rate ≥ 98%
        - Fallback Rate ≤ 5%
        - 평균 API 요청 수 ≤ 1.1회
        - SINGLE Query Type Accuracy ≥ 85%
        - MULTI Query Type Accuracy ≥ 75%
        - CLARIFY Recall ≥ 80%
        - Request Pair Micro F1 ≥ 50%

        SMOKE에서 기준을 만족하지 못하면 FULL 270건을 실행하지 말고 Raw JSONL의 오답·Fallback을 먼저 수정합니다.
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(OUTPUT)
