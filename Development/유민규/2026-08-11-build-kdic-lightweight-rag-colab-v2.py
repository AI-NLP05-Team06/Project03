from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parent
BASE_NOTEBOOK = ROOT / "2026-08-11-KDIC-경량-RAG-질의분석-Colab.ipynb"
OUTPUT = ROOT / "2026-08-11-KDIC-경량-RAG-질의분석-Colab-v2.ipynb"


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


if not BASE_NOTEBOOK.exists():
    raise FileNotFoundError(f"기준 노트북이 없습니다: {BASE_NOTEBOOK}")

notebook = json.loads(BASE_NOTEBOOK.read_text(encoding="utf-8"))
cells = notebook["cells"]

# v1의 클래스·보조 함수는 재사용하지만 v1 PIPELINE 인스턴스는 만들지 않는다.
base_pipeline_source = cells[6]["source"]
cells[6]["source"] = base_pipeline_source.split("\nCFG = PipelineConfig()", 1)[0].rstrip() + "\n"

cells[0] = md(
    """
    # KDIC 경량 RAG 질의 분석 파이프라인 v2

    v1 FULL 270건 평가에서 확인한 네 가지 병목을 수정한 버전입니다.

    - Structured Output 뒤에 코드 기반 Single Gate 적용
    - Intent 규칙 우선 + HCX 보조 방식
    - 명시된 역할·신청자 유형만 검색 조건으로 사용
    - 업무 용어 사전 확장, 중복 Need 축소, 평가 지표 분해

    기존 v1과 동일하게 일반 업무 질의는 HCX-007을 원칙상 한 번 호출합니다.
    """
)

cells[1] = md(
    """
    ## v2 실행 흐름

    ```mermaid
    flowchart TD
        A["사용자 질의"] --> B["최소 정규화"]
        B --> C{"고정밀 Fast Path"}
        C -->|"일치"| D["DIRECT / OUT_OF_SCOPE<br/>API 0회"]
        C -->|"불일치"| E["HCX-007 Structured Output<br/>최대 1회 + 재시도 1회"]
        E --> F["스키마 검증"]
        F --> G["Single Gate<br/>Route 일관성·필수정보 판정"]
        G --> H["Intent 규칙·복합질의 보정"]
        H --> I["명시 역할·신청자 유형 추출"]
        I --> J["코드 기반 Query Plan"]
        D --> K["최종 결과"]
        J --> K
    ```

    모델이 반환한 `model_route`와 Gate 이후의 최종 `analysis.route`를 모두 저장합니다.
    """
)

v2_core = code(
    r'''
    PIPELINE_VERSION = "KDIC_LIGHTWEIGHT_RAG_QUERY_ANALYZER_V2_2026_08_11"

    # v1 평가에서 누락된 KDIC 연관어를 확장한다.
    BUSINESS_KEYWORDS = {
        "예금자보호제도": [
            "예금자보호", "보호한도", "보호대상", "예금 보호", "보호 한도",
        ],
        "예금보험금 안내": [
            "예금보험금", "보험금 지급", "보험사고", "가지급금", "개산지급금",
            "1종 보험사고", "2종 보험사고",
        ],
        "고객 미수령금 신청": [
            "미수령금", "파산배당금", "개산지급금 정산금", "상속인 금융거래 조회",
            "지급대행점", "상속인 금융거래 조회서비스",
        ],
        "착오송금 반환 신청": [
            "착오송금", "잘못 보낸 돈", "잘못 송금", "착오 송금", "반환지원",
            "매입계약", "지급명령", "강제집행", "송금인", "수취인",
        ],
        "채무조정 안내": [
            "채무조정", "신용회복지원", "파산선고", "면책", "채무감면",
            "개인회생", "개인파산", "워크아웃", "변제기간",
        ],
        "은닉재산 신고": [
            "은닉재산", "은닉 재산", "금융부실관련자", "부실관련자", "차명 재산",
            "차명재산", "신고 포상금",
        ],
    }

    DIRECT_META_PATTERN = re.compile(
        r"^(?:안녕|안녕하세요|반갑습니다|반가워요|고마워요|고맙습니다|감사합니다|도움이 됐어요|"
        r"알겠습니다|무슨 질문을 할 수 있나요|지원하는 업무를 (?:알려주세요|목록으로 보여주세요)|"
        r"이 챗봇은 어떻게 사용하면 되나요|답변을 쉽게 설명해 줄 수 있나요|"
        r"전문가 수준으로 자세히 설명해 주세요|긴 설명보다 핵심 내용만 먼저 알려주세요|"
        r"질문을 잘못 입력했어요[.]? 다시 물어볼게요)[.!?]*$",
        re.I,
    )
    OOS_EXACT_PATTERN = re.compile(
        r".*(?:코스피|비트코인|주택담보대출 금리|신용점수|실손보험|국민연금|보이스피싱|"
        r"은행 계좌를 새로|해외송금 수수료|카드 결제|전세대출|세금 환급|퇴직금|"
        r"개인정보 유출|상속세|환율|주식 투자|신용카드 연회비|사업자등록|서울 날씨|"
        r"이번 주말.*날씨).*",
        re.I,
    )

    def detect_fast_path(query: str, conversation_state: dict[str, Any] | None = None) -> dict[str, Any] | None:
        text = query.strip()
        if DIRECT_META_PATTERN.fullmatch(text):
            return {"route": "DIRECT", "action": "META_OR_SOCIAL"}
        has_previous = bool((conversation_state or {}).get("has_previous_answer"))
        if has_previous and exact_fullmatch(
            r"(?:쉽게 설명해 주세요|핵심만 알려주세요|표로 정리해 주세요|더 자세히 설명해 주세요)[.!?]*",
            text,
        ):
            return {"route": "DIRECT", "action": "REFORMAT_PREVIOUS_ANSWER"}
        # KDIC 업무 용어가 함께 있으면 OOS 키워드보다 업무 근거를 우선한다.
        if OOS_EXACT_PATTERN.fullmatch(text) and not find_businesses(text):
            return {"route": "OUT_OF_SCOPE", "action": "EXPLICIT_NON_KDIC"}
        return None

    # 규칙이 명확한 Intent만 덮어쓴다. OVERVIEW는 마지막 순위다.
    INTENT_RULE_PATTERNS = [
        ("DOCUMENTS", [r"서류", r"증빙", r"신분증", r"위임장", r"구비", r"준비물", r"준비할 사항", r"필요한 것"]),
        ("AMOUNT", [r"얼마(?!나)", r"얼마나\s*(?:감면|지급|보상|돌려|받)", r"한도", r"금액", r"계산", r"수수료", r"비용", r"포상금", r"채무\s*감면"]),
        ("TIME", [r"기간", r"기한", r"언제", r"시점", r"얼마나 걸", r"소요"]),
        ("ELIGIBILITY", [r"대상", r"자격", r"요건", r"조건", r"해당\s*(?:여부|하는지)", r"받을 수 있", r"신청할 수 있"]),
        ("STATUS", [r"조회", r"진행 상황", r"처리 결과", r"결과.*확인", r"지급 여부", r"어디서 확인"]),
        ("CONTACT", [r"연락처", r"전화", r"문의처", r"어디로 연락", r"어느 기관", r"도움을 주는 기관"]),
        ("APPLICATION", [r"신청\s*(?:방법|절차)", r"접수\s*(?:방법|절차)", r"제출\s*방법", r"어떻게\s*신청", r"어디(?:서|로).*신청", r"절차", r"취소", r"철회"]),
        ("OVERVIEW", [r"의미", r"차이", r"종류", r"설명", r"지원 내용"]),
    ]
    MULTI_INTENT_PATTERNS = [
        ("DOCUMENTS", [r"서류", r"증빙", r"신분증", r"위임장", r"준비할 사항", r"필요한 것"]),
        ("AMOUNT", [r"얼마(?!나)", r"얼마나\s*(?:감면|지급|보상|돌려|받)", r"한도", r"금액", r"계산", r"채무\s*감면"]),
        ("TIME", [r"기간", r"기한", r"언제", r"시점", r"얼마나 걸"]),
        ("ELIGIBILITY", [r"대상", r"자격", r"요건", r"조건", r"해당\s*(?:여부|하는지)"]),
        ("STATUS", [r"조회", r"진행 상황", r"처리 결과", r"결과.*확인"]),
        ("CONTACT", [r"연락처", r"전화", r"문의처", r"어디로 연락", r"도움을 주는 기관"]),
        ("APPLICATION", [r"신청\s*(?:방법|절차)", r"접수\s*(?:방법|절차)", r"제출\s*방법", r"어떻게\s*신청", r"절차", r"취소", r"철회", r"신고\s*채널"]),
        ("OVERVIEW", [r"의미", r"차이", r"종류", r"설명", r"지원 내용"]),
    ]
    INTENT_FOCUS = {
        "DOCUMENTS": "필요 서류", "AMOUNT": "금액·한도", "TIME": "기간·기한",
        "ELIGIBILITY": "대상·자격", "STATUS": "조회·처리 상태", "CONTACT": "문의처",
        "APPLICATION": "신청·접수 방법", "OVERVIEW": "의미·개요",
    }

    def rule_intent(text: str) -> str | None:
        # 의미가 겹치는 표현은 일반 우선순위보다 먼저 판정한다.
        if re.search(r"조회서비스.*(?:무엇|의미)", text, flags=re.I):
            return "OVERVIEW"
        if re.search(r"서류|증빙|신분증|위임장|구비|준비물", text, flags=re.I):
            return "DOCUMENTS"
        if re.search(r"(?:기한|기간|언제|언제까지|얼마나\s*걸|소요)", text, flags=re.I):
            return "TIME"
        if re.search(r"얼마나\s*(?:감면|지급|보상|돌려|받)", text, flags=re.I):
            return "AMOUNT"
        if re.search(r"(?:신청|접수|제출)\s*(?:방법|절차)|어떻게\s*신청", text, flags=re.I):
            return "APPLICATION"
        if re.search(r"대상|자격|요건|신청.*필요(?:한지|한가|합니까|해요)|신청 대상|신청 자격", text, flags=re.I):
            return "ELIGIBILITY"
        if re.search(r"(?:온라인|방문|직접).*신청.*(?:가능|할 수)|어디.*접수|접수.*어디", text, flags=re.I):
            return "APPLICATION"
        if re.search(r"어디.*(?:확인|검색|조회)|결과.*확인|정보.*보는 방법", text, flags=re.I):
            return "STATUS"
        if re.search(r"(?:예금|계좌|금융상품|상품|원금|이자).{0,20}(?:보호 대상|보호되|포함되)|대상인지|자격.*해당", text, flags=re.I):
            return "ELIGIBILITY"
        for intent, patterns in INTENT_RULE_PATTERNS:
            if any(re.search(pattern, text, flags=re.I) for pattern in patterns):
                return intent
        return None

    def multi_intent_candidates(text: str) -> list[str]:
        found = []
        for intent, patterns in MULTI_INTENT_PATTERNS:
            positions = [m.start() for pattern in patterns for m in re.finditer(pattern, text, flags=re.I)]
            if positions:
                found.append((min(positions), intent))
        found.sort()
        return list(dict.fromkeys(intent for _, intent in found))

    USER_ROLE_PATTERNS = [
        ("SENDER", r"송금인|돈을 보낸 사람"),
        ("RECIPIENT", r"수취인|돈을 받은 사람"),
        ("DEBTOR", r"채무자"),
        ("REPORTER", r"신고자|신고인"),
        ("CLAIMANT", r"청구인"),
        ("DEPOSITOR", r"예금자(?!보호)"),
    ]
    APPLICANT_PATTERNS = [
        ("LEGAL_REPRESENTATIVE", r"법정대리인|후견인"),
        ("PROXY", r"대리인|대신해 신청|대리 신청|위임받"),
        ("HEIR", r"상속인"),
        ("CORPORATION", r"법인(?: 명의|이|으로| 신청)"),
        ("SELF", r"본인이 직접|본인 신청|제가 직접|직접 신청"),
    ]

    def extract_explicit_value(text: str, patterns: list[tuple[str, str]]) -> str | None:
        for value, pattern in patterns:
            if re.search(pattern, text, flags=re.I):
                return value
        return None

    def query_analysis_schema() -> dict[str, Any]:
        business_value = {"type": "string", "enum": [*BUSINESS_FUNCTIONS, "UNKNOWN"]}
        intent_value = {"type": "string", "enum": [*INTENTS, "UNKNOWN"]}
        return {
            "type": "object",
            "properties": {
                "route": {"type": "string", "enum": ROUTES},
                "needs": {
                    "type": "array", "maxItems": 6,
                    "items": {
                        "type": "object",
                        "properties": {
                            "need_id": {"type": "string"},
                            "query": {"type": "string"},
                            "business_function": business_value,
                            "intent": intent_value,
                            "target_type": {"type": "string"},
                            "case_details": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["need_id", "query", "business_function", "intent", "target_type", "case_details"],
                    },
                },
                "missing_information": {"type": "array", "items": {"type": "string", "enum": MISSING_FIELDS}},
            },
            "required": ["route", "needs", "missing_information"],
        }

    SYSTEM_PROMPT = f"""
    역할: 예금보험공사 RAG의 경량 질의 분석기.
    출력: JSON Schema만 준수한다.

    업무 사전:
    - 예금자보호제도: 보호대상, 보호한도, 금융상품 보호 여부
    - 예금보험금 안내: 보험사고, 예금보험금, 가지급금, 개산지급금
    - 고객 미수령금 신청: 미수령금, 파산배당금, 정산금, 지급대행점, 상속인 금융거래 조회
    - 착오송금 반환 신청: 반환지원, 매입계약, 지급명령, 강제집행
    - 채무조정 안내: 개인회생, 개인파산, 워크아웃, 신용회복지원, 면책
    - 은닉재산 신고: 금융부실관련자, 차명재산, 신고, 포상금

    규칙:
    1. 서로 다른 정보 요구만 N1, N2로 분리한다. 서류와 제출방법처럼 Intent가 다르면 분리한다.
    2. query는 해당 Need를 독립 검색할 수 있게 쓰되 원문 의미를 보존한다.
    3. OVERVIEW는 의미·개요·차이 질문에만 사용한다.
    4. 개인회생·개인파산·워크아웃과 위 업무 사전 용어는 범위 내다.
    5. missing_information은 단순 미추출 필드가 아니다. 답을 바꾸는 필수정보만 넣는다.
    6. route가 CLARIFY가 아니면 missing_information은 []다.
    7. 업무가 불확실해도 원문으로 검색 가능하면 RETRIEVE다.
    8. 범주형 근거가 없으면 UNKNOWN, 자유 텍스트는 빈 문자열, 목록은 []다.

    예시 판단:
    - '신청 방법을 알려주세요' → CLARIFY, business_function 부족
    - '이 상품도 보호되나요' → CLARIFY, target_type 부족
    - '예금자보호 한도' → RETRIEVE / 예금자보호제도 / AMOUNT
    - '필요 서류와 제출 방법' → DOCUMENTS와 APPLICATION 두 Need

    허용 business_function: {BUSINESS_FUNCTIONS}
    허용 intent: {INTENTS}
    """

    class HCX007StructuredClientV2(HCX007StructuredClient):
        def analyze(self, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            started = time.perf_counter()
            attempts = []
            totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            last_error = None
            body = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                "topP": self.config.top_p, "topK": self.config.top_k,
                "maxCompletionTokens": self.config.max_completion_tokens,
                "temperature": self.config.temperature, "repetitionPenalty": 1.05,
                "thinking": {"effort": "none"}, "stop": [],
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
                            "Content-Type": "application/json", "Accept": "application/json",
                        },
                        json=body, timeout=self.config.timeout_seconds,
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
                            "api_request_count": len(attempts), "attempts": attempts, **totals,
                            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                        })
                    envelope = response.json()
                    result = envelope.get("result", envelope)
                    usage = result.get("usage") or {}
                    used = {
                        "prompt_tokens": int(usage.get("promptTokens") or 0),
                        "completion_tokens": int(usage.get("completionTokens") or 0),
                        "total_tokens": int(usage.get("totalTokens") or 0),
                    }
                    for key in totals:
                        totals[key] += used[key]
                    content = str((result.get("message") or {}).get("content") or "")
                    parsed = json.loads(content)
                    if not isinstance(parsed, dict):
                        raise TypeError("모델 결과의 최상위가 object가 아닙니다.")
                    attempts.append({
                        "attempt": attempt, "request_id": request_id, "http_status": response.status_code,
                        **used, "finish_reason": result.get("finishReason"),
                        "latency_ms": round((time.perf_counter() - attempt_started) * 1000, 3),
                    })
                    return parsed, {
                        "api_request_count": len(attempts), "attempts": attempts, **totals,
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
                "api_request_count": len(attempts), "attempts": attempts, **totals,
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            })

    def validate_analysis_v2(raw: dict[str, Any], normalized_query: str) -> tuple[dict[str, Any], list[str]]:
        warnings = []
        route = raw.get("route") if raw.get("route") in ROUTES else "RETRIEVE"
        if raw.get("route") not in ROUTES:
            warnings.append("INVALID_ROUTE_TO_RETRIEVE")
        rows = raw.get("needs") if isinstance(raw.get("needs"), list) else []
        needs = []
        for index, row in enumerate(rows, 1):
            if not isinstance(row, dict):
                warnings.append(f"N{index}_NOT_OBJECT")
                continue
            query = str(row.get("query") or normalized_query).strip() or normalized_query
            business = row.get("business_function")
            business = business if business in BUSINESS_FUNCTIONS else None
            intent = normalize_intent(row.get("intent"))
            target = str(row.get("target_type") or "").strip() or None
            cases = row.get("case_details")
            cases = [str(x).strip() for x in cases if str(x).strip()] if isinstance(cases, list) else []
            needs.append({
                "need_id": f"N{index}", "query": query,
                "business_function": business, "business_source": "MODEL" if business else None,
                "intent": intent, "intent_source": "MODEL" if intent else None,
                "user_role": None, "user_role_source": None,
                "applicant_type": None, "applicant_type_source": None,
                "target_type": target, "case_details": cases,
            })
        missing = raw.get("missing_information")
        missing = list(dict.fromkeys(x for x in missing if x in MISSING_FIELDS)) if isinstance(missing, list) else []
        return {"route": route, "needs": needs, "missing_information": missing}, warnings

    TARGET_REFERENCE_PHRASES = [
        "제가 가입한 상품", "제가 가진 이 채무", "제가 본 이 재산", "이 재산이나 행위",
        "어떤 송금 건", "어떤 예금에 대해", "어떤 돈을 신청", "어떤 예금이나 금융상품",
    ]
    TARGET_DEMONSTRATIVE_PATTERN = re.compile(
        r"(?:^|[\s,.(])이\s*(?:금융상품|계좌|상품|돈|송금|거래|금액|채무|재산)(?:을|를|이|가|도|은|는|의|이나|과|와|\s|[,.!?]|$)"
    )
    APPLICANT_REFERENCE_PHRASES = [
        "제 신청 유형", "제 신청 자격", "제 경우", "누구를 신청인", "누가 방문",
        "신고 주체 유형", "신청인란", "본인 대신 대리인",
    ]
    CASE_REFERENCE_PHRASES = [
        "제 상황", "현재 상황", "제 채무 상태", "제 신고 상황", "반려", "거절",
        "보완 요청", "진행되지 않", "여러 금융회사에 예금", "여러 계좌에 나뉘",
    ]
    GENERIC_TRANSACTION_PATTERN = re.compile(
        r"(?:신청|접수|조회|처리|서류|기한|대리인|상속인|수수료|비용|문의)", re.I,
    )

    def create_retrieve_need(query: str, business: str | None = None) -> dict[str, Any]:
        intent = rule_intent(query)
        return {
            "need_id": "N1", "query": query,
            "business_function": business, "business_source": "ORIGINAL" if business else None,
            "intent": intent, "intent_source": "RULE" if intent else None,
            "user_role": None, "user_role_source": None,
            "applicant_type": None, "applicant_type_source": None,
            "target_type": None, "case_details": [],
        }

    def detect_blocking_slot(query: str, analysis: dict[str, Any]) -> tuple[str | None, str | None]:
        if any(phrase in query for phrase in APPLICANT_REFERENCE_PHRASES):
            return "applicant_type", "UNRESOLVED_APPLICANT_REFERENCE"
        if any(phrase in query for phrase in TARGET_REFERENCE_PHRASES) or TARGET_DEMONSTRATIVE_PATTERN.search(query):
            return "target_type", "UNRESOLVED_TARGET_REFERENCE"
        if any(phrase in query for phrase in CASE_REFERENCE_PHRASES):
            return "case_details", "PERSONAL_CASE_REQUIRES_DETAILS"
        domain_hits = find_businesses(query)
        has_business_need = any(n.get("business_function") in BUSINESS_FUNCTIONS for n in analysis.get("needs") or [])
        if not domain_hits and not has_business_need and GENERIC_TRANSACTION_PATTERN.search(query):
            return "business_function", "MISSING_BUSINESS_FOR_GENERIC_TRANSACTION"
        return None, None

    def apply_single_gate(query: str, analysis: dict[str, Any]) -> tuple[dict[str, Any], list[str], str | None]:
        fixed = {**analysis, "needs": [dict(x) for x in analysis.get("needs") or []]}
        reasons = []
        model_route = fixed["route"]
        domain_hits = find_businesses(query)

        if model_route in {"DIRECT", "OUT_OF_SCOPE"} and domain_hits:
            fixed["route"] = "RETRIEVE"
            reasons.append(f"DOMAIN_EVIDENCE_OVERRIDES_{model_route}")
            if not fixed["needs"]:
                fixed["needs"] = [create_retrieve_need(query, domain_hits[0] if len(domain_hits) == 1 else None)]

        if fixed["route"] != "OUT_OF_SCOPE":
            blocking_slot, blocking_reason = detect_blocking_slot(query, fixed)
        else:
            blocking_slot, blocking_reason = None, None
        if blocking_slot:
            fixed["route"] = "CLARIFY"
            fixed["needs"] = []
            fixed["missing_information"] = [blocking_slot]
            reasons.append(blocking_reason)
            return fixed, reasons, blocking_slot

        if fixed["route"] == "CLARIFY":
            minimal = next((x for x in fixed.get("missing_information") or [] if x in MISSING_FIELDS), "case_details")
            fixed["missing_information"] = [minimal]
            fixed["needs"] = []
            reasons.append("MODEL_CLARIFY_MINIMAL_SLOT")
            return fixed, reasons, minimal

        fixed["missing_information"] = []
        if fixed["route"] == "RETRIEVE" and not fixed["needs"]:
            fixed["needs"] = [create_retrieve_need(query, domain_hits[0] if len(domain_hits) == 1 else None)]
            reasons.append("EMPTY_RETRIEVE_NEED_RECOVERED")
        if fixed["route"] in {"DIRECT", "OUT_OF_SCOPE"}:
            fixed["needs"] = []
        return fixed, reasons, None

    def apply_intent_and_multi_rules(analysis: dict[str, Any], original: str) -> tuple[dict[str, Any], list[str]]:
        if analysis["route"] != "RETRIEVE":
            return analysis, []
        warnings = []
        needs = [dict(x) for x in analysis["needs"]]
        for need in needs:
            explicit_businesses = find_businesses(need.get("query") or original)
            if len(explicit_businesses) == 1 and explicit_businesses[0] != need.get("business_function"):
                need["business_function"] = explicit_businesses[0]
                need["business_source"] = "RULE"
                warnings.append(f"{need['need_id']}_BUSINESS_RULE_OVERRIDE")
            ruled = rule_intent(need.get("query") or original)
            if ruled and ruled != need.get("intent"):
                need["intent"] = ruled
                need["intent_source"] = "RULE"
                warnings.append(f"{need['need_id']}_INTENT_RULE_OVERRIDE")
            elif ruled:
                need["intent_source"] = "RULE"

        strong_intents = multi_intent_candidates(original)
        if len(needs) == 1 and len(strong_intents) >= 2:
            base = needs[0]
            needs = []
            for intent in strong_intents[:4]:
                item = dict(base)
                item["intent"] = intent
                item["intent_source"] = "RULE"
                item["query"] = f"{original} (검색 초점: {INTENT_FOCUS[intent]})"
                needs.append(item)
            warnings.append("MULTI_INTENT_RULE_SPLIT")

        deduped = []
        seen = {}
        for need in needs:
            business, intent = need.get("business_function"), need.get("intent")
            normalized_need_query = normalize_query(need.get("query") or "")["normalized_query"]
            case_key = tuple(sorted(str(x) for x in (need.get("case_details") or [])))
            key = (business, intent, normalized_need_query, case_key)
            if key in seen:
                warnings.append("EXACT_DUPLICATE_NEED_COLLAPSED")
                continue
            seen[key] = len(deduped)
            deduped.append(need)
        for index, need in enumerate(deduped, 1):
            need["need_id"] = f"N{index}"
        return {**analysis, "needs": deduped}, list(dict.fromkeys(warnings))

    def enrich_evidence(
        analysis: dict[str, Any], original: str, context: dict[str, Any], manual: dict[str, Any]
    ) -> dict[str, Any]:
        explicit_businesses = find_businesses(original)
        explicit_role = extract_explicit_value(original, USER_ROLE_PATTERNS)
        explicit_applicant = extract_explicit_value(original, APPLICANT_PATTERNS)
        confirmed = context.get("confirmed", {}) if context.get("used") else {}
        for need in analysis.get("needs") or []:
            business = need.get("business_function")
            if manual.get("business_function") == business:
                need["business_source"] = "MANUAL"
            elif business in explicit_businesses:
                need["business_source"] = "ORIGINAL"
            elif confirmed.get("business_function") == business:
                need["business_source"] = "CONTEXT"
            elif business:
                need["business_source"] = "MODEL"

            if manual.get("user_role") in USER_ROLES:
                need["user_role"], need["user_role_source"] = manual["user_role"], "MANUAL"
            elif explicit_role:
                need["user_role"], need["user_role_source"] = explicit_role, "ORIGINAL"
            elif confirmed.get("user_role") in USER_ROLES:
                need["user_role"], need["user_role_source"] = confirmed["user_role"], "CONTEXT"

            if manual.get("applicant_type") in APPLICANT_TYPES:
                need["applicant_type"], need["applicant_type_source"] = manual["applicant_type"], "MANUAL"
            elif explicit_applicant:
                need["applicant_type"], need["applicant_type_source"] = explicit_applicant, "ORIGINAL"
            elif confirmed.get("applicant_type") in APPLICANT_TYPES:
                need["applicant_type"], need["applicant_type_source"] = confirmed["applicant_type"], "CONTEXT"
        return analysis

    def build_keyword_query_v2(need: dict[str, Any]) -> str:
        values = [need.get("business_function"), need.get("intent"), need.get("target_type")]
        values.extend(need.get("case_details") or [])
        if need.get("user_role_source") in {"ORIGINAL", "CONTEXT", "MANUAL"}:
            values.append(need.get("user_role"))
        if need.get("applicant_type_source") in {"ORIGINAL", "CONTEXT", "MANUAL"}:
            values.append(need.get("applicant_type"))
        return " ".join(str(x) for x in values if x)

    def build_query_plans_v2(
        analysis: dict[str, Any], original: str, context: dict[str, Any], manual: dict[str, Any], config: PipelineConfig
    ) -> list[dict[str, Any]]:
        if analysis["route"] != "RETRIEVE":
            return []
        plans = []
        for need in analysis["needs"]:
            business = need.get("business_function")
            source = need.get("business_source")
            if not business:
                business_filter = {"mode": "NONE", "value": None, "soft_hint": None, "evidence": "UNKNOWN"}
            elif source in {"ORIGINAL", "CONTEXT", "MANUAL"}:
                business_filter = {"mode": "HARD", "value": business, "soft_hint": None, "evidence": source}
            else:
                business_filter = {"mode": "SOFT", "value": None, "soft_hint": business, "evidence": "MODEL"}
            intent = need.get("intent")
            intent_weight = 0.20 if need.get("intent_source") == "RULE" else config.intent_soft_boost
            plans.append({
                "need_id": need["need_id"], "semantic_query": need["query"],
                "keyword_query": build_keyword_query_v2(need) or need["query"],
                "business_filter": business_filter,
                "intent_boost": {
                    "mode": "SOFT" if intent in INTENTS else "NONE", "value": intent,
                    "weight": intent_weight if intent in INTENTS else 0.0,
                    "evidence": need.get("intent_source"),
                },
                "entities": {
                    "user_role": need.get("user_role"), "user_role_source": need.get("user_role_source"),
                    "applicant_type": need.get("applicant_type"), "applicant_type_source": need.get("applicant_type_source"),
                    "target_type": need.get("target_type"), "case_details": need.get("case_details") or [],
                },
            })
        return plans

    class KDICLightweightRAGAnalyzerV2:
        def __init__(self, client: HCX007StructuredClientV2, config: PipelineConfig | None = None):
            self.client = client
            self.config = config or client.config

        def run(self, query: str, *, conversation_state=None, manual_selection=None) -> dict[str, Any]:
            started = time.perf_counter()
            normalized = normalize_query(query)
            original, text = normalized["original_query"], normalized["normalized_query"]
            manual = dict(manual_selection or {})
            context = build_context(text, conversation_state, self.config.max_context_turns)
            fast = detect_fast_path(text, conversation_state)
            if fast:
                analysis = {"route": fast["route"], "needs": [], "missing_information": []}
                return {
                    "pipeline_version": PIPELINE_VERSION, "analysis_status": "FAST_PATH",
                    "original_query": original, "normalized_query": text, "context": context,
                    "model_route": fast["route"], "model_missing_information": [],
                    "gate_reasons": ["FAST_PATH"], "rule_actions": [], "blocking_slot": None,
                    "analysis": analysis, "fast_path": fast, "validation_warnings": [], "query_plans": [],
                    "runtime": {
                        "api_request_count": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                        "latency_ms": round((time.perf_counter() - started) * 1000, 3), "attempts": [],
                    },
                }
            payload = {
                "query": text, "confirmed_context": context["confirmed"],
                "recent_turns": context["recent_turns"], "manual_selection": manual,
            }
            try:
                raw, telemetry = self.client.analyze(payload)
                analysis, warnings = validate_analysis_v2(raw, text)
                model_route = analysis["route"]
                model_missing = list(analysis.get("missing_information") or [])
                analysis, gate_reasons, blocking_slot = apply_single_gate(text, analysis)
                analysis, rule_actions = apply_intent_and_multi_rules(analysis, text)
                status = "REPAIRED" if warnings else "OK"
            except HCXAPIError as exc:
                telemetry = exc.telemetry
                raw_fallback = {
                    "route": "RETRIEVE",
                    "needs": [{
                        "need_id": "N1", "query": text, "business_function": "UNKNOWN",
                        "intent": "UNKNOWN", "target_type": "", "case_details": [],
                    }],
                    "missing_information": [],
                }
                analysis, warnings = validate_analysis_v2(raw_fallback, text)
                model_route, model_missing = "FALLBACK", []
                analysis, gate_reasons, blocking_slot = apply_single_gate(text, analysis)
                analysis, rule_actions = apply_intent_and_multi_rules(analysis, text)
                warnings.append("MODEL_ANALYSIS_FAILED_USED_FALLBACK")
                status = "FALLBACK"
            analysis = enrich_evidence(analysis, original, context, manual)
            plans = build_query_plans_v2(analysis, original, context, manual, self.config)
            return {
                "pipeline_version": PIPELINE_VERSION, "analysis_status": status,
                "original_query": original, "normalized_query": text, "context": context,
                "model_route": model_route, "model_missing_information": model_missing,
                "gate_reasons": gate_reasons, "rule_actions": rule_actions, "blocking_slot": blocking_slot,
                "analysis": analysis, "validation_warnings": list(dict.fromkeys(warnings)),
                "query_plans": plans,
                "runtime": {**telemetry, "latency_ms": round((time.perf_counter() - started) * 1000, 3)},
            }

    CFG_V2 = PipelineConfig(max_completion_tokens=700, temperature=0.1)
    HCX_CLIENT_V2 = HCX007StructuredClientV2(get_hcx_api_key("HCX_API_KEY"), CFG_V2)
    PIPELINE = KDICLightweightRAGAnalyzerV2(HCX_CLIENT_V2, CFG_V2)
    CFG = CFG_V2
    print("v2 파이프라인 준비 완료:", PIPELINE_VERSION)
    '''
)

# v1 파이프라인 초기화 뒤 v2 정의를 삽입한다. 이후 셀은 새 PIPELINE을 사용한다.
cells.insert(7, v2_core)

cells[8] = md(
    """
    ## v2 단위 확인

    다음 질의로 Fast Path, Generic Clarify, 미해결 대상, Intent 분리, 업무 범위 복구를 확인합니다.
    실제 API 호출 전에는 `HCX_API_KEY`가 필요합니다.
    """
)

cells[9] = code(
    r'''
    test_queries = [
        "안녕하세요.",
        "이번 주말 서울 날씨를 알려주세요.",
        "신청 방법을 알려주세요.",
        "제가 가입한 이 상품도 예금자보호 대상인가요?",
        "예금자 본인이 신청할 때 필요한 서류와 제출 방법은 무엇인가요?",
        "개인회생과 개인파산은 대상과 절차가 어떻게 다른가요?",
        "은닉재산 신고에 대해 전화로 문의하려면 어디로 연락하나요?",
    ]
    for q in test_queries:
        result = PIPELINE.run(q)
        print("\nQUERY:", q)
        print(json.dumps({
            "status": result["analysis_status"], "model_route": result["model_route"],
            "final_route": result["analysis"]["route"], "gate_reasons": result["gate_reasons"],
            "rule_actions": result.get("rule_actions"),
            "missing": result["analysis"].get("missing_information"),
            "needs": result["analysis"].get("needs"), "query_plans": result["query_plans"],
            "runtime": result["runtime"],
        }, ensure_ascii=False, indent=2))
    '''
)

cells[11] = md(
    """
    ## Evaluation_DataSet_v5 v2 평가

    기본 `SMOKE`는 Route·단일/복합 버킷별 4건씩 총 20건입니다. 통과 후 `FULL`로 변경합니다.

    v2는 모델 Route와 Gate 이후 Route를 함께 저장하여 Gate의 개선·회귀를 분리 측정합니다.
    """
)

cells[13] = code(
    r'''
    def counter_overlap(gold, pred):
        g, p = Counter(gold), Counter(pred)
        return sum((g & p).values()), sum((p - g).values()), sum((g - p).values())

    def micro_from_columns(frame, gold_column, pred_column, *, use_set=False):
        tp = fp = fn = 0
        for _, row in frame.iterrows():
            gold = set(row[gold_column]) if use_set else Counter(row[gold_column])
            pred = set(row[pred_column]) if use_set else Counter(row[pred_column])
            if use_set:
                tp += len(gold & pred); fp += len(pred - gold); fn += len(gold - pred)
            else:
                tp += sum((gold & pred).values()); fp += sum((pred - gold).values()); fn += sum((gold - pred).values())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

    records = []
    raw_path = Path("kdic_lightweight_rag_v2_eval_raw.jsonl")
    with raw_path.open("w", encoding="utf-8") as raw_file:
        for index, gold in enumerate(eval_rows, 1):
            try:
                result = PIPELINE.run(gold["question"])
            except Exception as exc:
                result = {
                    "pipeline_version": PIPELINE_VERSION, "analysis_status": "ERROR",
                    "model_route": "ERROR", "model_missing_information": [],
                    "gate_reasons": [f"EVALUATION_ERROR: {type(exc).__name__}"], "rule_actions": [], "blocking_slot": None,
                    "analysis": {
                        "route": "RETRIEVE", "needs": [create_retrieve_need(gold["question"])],
                        "missing_information": [],
                    },
                    "validation_warnings": [str(exc)], "query_plans": [],
                    "runtime": {
                        "api_request_count": 0, "prompt_tokens": 0, "completion_tokens": 0,
                        "total_tokens": 0, "latency_ms": 0.0, "attempts": [],
                    },
                }
            analysis = result["analysis"]
            model_route_internal = result.get("model_route")
            model_route = {"DIRECT": "DIRECT_RESPONSE"}.get(model_route_internal, model_route_internal)
            pred_route_internal = analysis["route"]
            pred_route = {"DIRECT": "DIRECT_RESPONSE"}.get(pred_route_internal, pred_route_internal)
            needs = analysis.get("needs") or []
            pred_query_type = "MULTI" if pred_route_internal == "RETRIEVE" and len(needs) >= 2 else ("SINGLE" if pred_route_internal == "RETRIEVE" else "NONE")
            pred_pairs = [
                (n.get("business_function"), n.get("intent"))
                for n in needs
                if n.get("business_function") not in {None, "UNKNOWN"}
                and n.get("intent") not in {None, "UNKNOWN"}
            ]
            pred_businesses = [x[0] for x in pred_pairs]
            pred_intents = [x[1] for x in pred_pairs]
            gold_businesses = [x[0] for x in gold["gold_request_pairs"]]
            gold_intents = [x[1] for x in gold["gold_request_pairs"]]
            pred_user_roles = sorted({n.get("user_role") for n in needs if n.get("user_role")})
            pred_applicant_types = sorted({n.get("applicant_type") for n in needs if n.get("applicant_type")})
            retrieve_case = gold["gold_route"] == "RETRIEVE"
            clarify_case = gold["gold_route"] == "CLARIFY"
            pair_exact = Counter(gold["gold_request_pairs"]) == Counter(pred_pairs) if retrieve_case else None
            pair_set_exact = set(gold["gold_request_pairs"]) == set(pred_pairs) if retrieve_case else None
            missing_exact = set(gold["gold_missing_slots"]) == set(analysis.get("missing_information") or []) if clarify_case else None
            route_correct = pred_route == gold["gold_route"]
            model_route_correct = model_route == gold["gold_route"]
            query_type_correct = pred_query_type == gold["gold_query_type"]
            core_exact = route_correct and (query_type_correct and pair_exact if retrieve_case else (missing_exact if clarify_case else True))
            core_set_exact = route_correct and (pair_set_exact if retrieve_case else (missing_exact if clarify_case else True))
            runtime = result["runtime"]
            record = {
                **gold, "analysis_status": result["analysis_status"],
                "model_route": model_route, "pred_route": pred_route, "pred_route_internal": pred_route_internal,
                "gate_reasons": result.get("gate_reasons") or [], "rule_actions": result.get("rule_actions") or [],
                "blocking_slot": result.get("blocking_slot"),
                "model_route_correct": model_route_correct, "route_correct": route_correct,
                "gate_improved": (not model_route_correct) and route_correct,
                "gate_regressed": model_route_correct and (not route_correct),
                "pred_query_type": pred_query_type, "query_type_correct": query_type_correct,
                "pred_request_pairs": pred_pairs, "pred_businesses": pred_businesses, "pred_intents": pred_intents,
                "gold_businesses": gold_businesses, "gold_intents": gold_intents,
                "pred_user_roles": pred_user_roles, "pred_applicant_types": pred_applicant_types,
                "pred_missing_slots": analysis.get("missing_information") or [],
                "request_pair_exact": pair_exact, "request_pair_set_exact": pair_set_exact,
                "missing_slot_exact": missing_exact,
                "user_role_exact": set(gold["gold_user_roles"]) == set(pred_user_roles),
                "applicant_type_exact": set(gold["gold_applicant_types"]) == set(pred_applicant_types),
                "core_exact": bool(core_exact), "core_set_exact": bool(core_set_exact),
                "query_plan_count": len(result["query_plans"]),
                "api_request_count": int(runtime.get("api_request_count") or 0),
                "prompt_tokens": int(runtime.get("prompt_tokens") or 0),
                "completion_tokens": int(runtime.get("completion_tokens") or 0),
                "total_tokens": int(runtime.get("total_tokens") or 0),
                "wall_latency_ms": float(runtime.get("latency_ms") or 0),
                "warning_count": len(result.get("validation_warnings") or []),
            }
            records.append(record)
            raw_file.write(json.dumps({"gold": gold, "result": result}, ensure_ascii=False, default=str) + "\n")
            if index == 1 or index % PROGRESS_EVERY == 0 or index == len(eval_rows):
                print(f"[{index}/{len(eval_rows)}] status={Counter(r['analysis_status'] for r in records)}")
            time.sleep(CFG.request_interval_seconds)

    result_df = pd.DataFrame(records)
    retrieve_df = result_df[result_df.gold_route == "RETRIEVE"].copy()
    pair_multi = micro_from_columns(retrieve_df, "gold_request_pairs", "pred_request_pairs")
    pair_set = micro_from_columns(retrieve_df, "gold_request_pairs", "pred_request_pairs", use_set=True)
    business_micro = micro_from_columns(retrieve_df, "gold_businesses", "pred_businesses")
    business_set_micro = micro_from_columns(retrieve_df, "gold_businesses", "pred_businesses", use_set=True)
    intent_micro = micro_from_columns(retrieve_df, "gold_intents", "pred_intents")
    intent_set_micro = micro_from_columns(retrieve_df, "gold_intents", "pred_intents", use_set=True)
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
        "model_route_accuracy": float(result_df.model_route_correct.mean()),
        "final_route_accuracy": float(result_df.route_correct.mean()),
        "gate_improved_count": int(result_df.gate_improved.sum()),
        "gate_regressed_count": int(result_df.gate_regressed.sum()),
        "query_type_accuracy": float(result_df.query_type_correct.mean()),
        "core_exact": float(result_df.core_exact.mean()), "core_set_exact": float(result_df.core_set_exact.mean()),
        "business_micro_f1": business_micro["f1"], "business_set_micro_f1": business_set_micro["f1"],
        "intent_micro_f1": intent_micro["f1"], "intent_set_micro_f1": intent_set_micro["f1"],
        "request_pair_micro_precision": pair_multi["precision"], "request_pair_micro_recall": pair_multi["recall"],
        "request_pair_micro_f1": pair_multi["f1"], "request_pair_set_micro_f1": pair_set["f1"],
        "clarify_precision": clarify_tp / len(pred_clarify) if len(pred_clarify) else 0.0,
        "clarify_recall": clarify_tp / len(clarify_rows) if len(clarify_rows) else 0.0,
        "clarify_missing_exact": float(clarify_rows.missing_slot_exact.fillna(False).mean()) if len(clarify_rows) else 1.0,
        "user_role_exact": float(result_df.user_role_exact.mean()),
        "applicant_type_exact": float(result_df.applicant_type_exact.mean()),
        "average_api_requests": float(result_df.api_request_count.mean()),
        "zero_call_rate": float((result_df.api_request_count == 0).mean()),
        "average_prompt_tokens": float(result_df.prompt_tokens.mean()),
        "average_completion_tokens": float(result_df.completion_tokens.mean()),
        "average_total_tokens": float(result_df.total_tokens.mean()),
        "average_latency_ms": float(result_df.wall_latency_ms.mean()),
        "p95_latency_ms": float(result_df.wall_latency_ms.quantile(0.95)),
    }
    for query_type in ["SINGLE", "MULTI", "NONE"]:
        subset = result_df[result_df.gold_query_type == query_type]
        if len(subset):
            key = query_type.lower()
            summary[f"{key}_route_accuracy"] = float(subset.route_correct.mean())
            summary[f"{key}_query_type_accuracy"] = float(subset.query_type_correct.mean())
            summary[f"{key}_core_exact"] = float(subset.core_exact.mean())
            summary[f"{key}_average_latency_ms"] = float(subset.wall_latency_ms.mean())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    '''
)

cells[14] = code(
    r'''
    summary_path = Path("kdic_lightweight_rag_v2_eval_summary.json")
    detail_path = Path("kdic_lightweight_rag_v2_eval_per_question.csv")
    route_confusion_path = Path("kdic_lightweight_rag_v2_final_route_confusion.csv")
    model_route_confusion_path = Path("kdic_lightweight_rag_v2_model_route_confusion.csv")
    query_type_confusion_path = Path("kdic_lightweight_rag_v2_query_type_confusion.csv")
    gate_reason_path = Path("kdic_lightweight_rag_v2_gate_reasons.csv")
    rule_action_path = Path("kdic_lightweight_rag_v2_rule_actions.csv")
    status_path = Path("kdic_lightweight_rag_v2_eval_status.csv")
    zip_path = Path("kdic_lightweight_rag_v2_evaluation_bundle.zip")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    final_route_confusion = pd.crosstab(result_df.gold_route, result_df.pred_route, margins=True).reset_index()
    model_route_confusion = pd.crosstab(result_df.gold_route, result_df.model_route, margins=True).reset_index()
    query_type_confusion = pd.crosstab(result_df.gold_query_type, result_df.pred_query_type, margins=True).reset_index()
    gate_reason_counts = Counter(reason for reasons in result_df.gate_reasons for reason in reasons)
    gate_reason_table = pd.DataFrame([{"gate_reason": k, "count": v} for k, v in gate_reason_counts.most_common()])
    rule_action_counts = Counter(action for actions in result_df.rule_actions for action in actions)
    rule_action_table = pd.DataFrame([{"rule_action": k, "count": v} for k, v in rule_action_counts.most_common()])
    status_table = result_df.analysis_status.value_counts(dropna=False).rename_axis("analysis_status").reset_index(name="count")

    result_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    final_route_confusion.to_csv(route_confusion_path, index=False, encoding="utf-8-sig")
    model_route_confusion.to_csv(model_route_confusion_path, index=False, encoding="utf-8-sig")
    query_type_confusion.to_csv(query_type_confusion_path, index=False, encoding="utf-8-sig")
    gate_reason_table.to_csv(gate_reason_path, index=False, encoding="utf-8-sig")
    rule_action_table.to_csv(rule_action_path, index=False, encoding="utf-8-sig")
    status_table.to_csv(status_path, index=False, encoding="utf-8-sig")

    export_paths = [
        summary_path, detail_path, route_confusion_path, model_route_confusion_path,
        query_type_confusion_path, gate_reason_path, rule_action_path, status_path, raw_path,
    ]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in export_paths:
            archive.write(path, arcname=path.name)
    print("생성:", zip_path.resolve())
    '''
)

cells[15] = md(
    """
    ## v2 FULL 실행 전 기준

    - LLM Structured Success ≥ 98%
    - Fallback ≤ 5%
    - Final Route Accuracy ≥ 90%
    - CLARIFY Precision ≥ 90%, Recall ≥ 80%
    - Business micro F1 ≥ 90%
    - Intent micro F1 ≥ 75%
    - MULTI Query Type Accuracy ≥ 80%
    - Gate Regression = 0에 가깝게 유지
    - P95 질의분석 지연시간 ≤ 4초를 1차 목표로 사용

    SMOKE에서 Gate 회귀 문항과 Intent override를 확인한 뒤 FULL을 실행합니다.
    """
)

OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(OUTPUT)
