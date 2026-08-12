from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parent
V2_NOTEBOOK = ROOT / "2026-08-11-KDIC-경량-RAG-질의분석-Colab-v2.ipynb"
OUTPUT = ROOT / "2026-08-12-KDIC-경량-RAG-질의분석-Colab-v3.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(source).strip() + "\n"}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": textwrap.dedent(source).strip() + "\n"}


if not V2_NOTEBOOK.exists():
    raise FileNotFoundError(f"v2 노트북을 찾을 수 없습니다: {V2_NOTEBOOK}")

notebook = json.loads(V2_NOTEBOOK.read_text(encoding="utf-8"))
cells = notebook["cells"]

cells[0] = md(
    """
    # KDIC 경량 RAG 질의분석 v3

    v2 FULL 평가의 핵심 문제였던 모델 주도 CLARIFY를 제거한 버전입니다.

    - HCX-007: Atomic Need, 업무, Intent, 대상·사례조건만 분석
    - 코드 Single Gate: 최종 Route를 단독 결정
    - 강한 필수정보 누락만 CLARIFY
    - 검색 가능한 불확실성은 RETRIEVE_RELAXED
    - 모델 Intent를 무조건 덮어쓰지 않고 고정밀 구문만 교정
    - 모든 질의를 재작성하지 않고 필요한 Atomic Need만 독립 검색문으로 사용
    """
)

cells[1] = md(
    """
    ## v3 파이프라인

    ```mermaid
    flowchart TD
        A["사용자 원문"] --> B["최소 정규화"]
        B --> C{"Fast Path"}
        C -->|"인사·메타"| D["DIRECT / API 0회"]
        C -->|"명확한 범위 외"| E["OUT_OF_SCOPE / API 0회"]
        C -->|"일반 질의"| F["HCX-007 Atomic Need 분석 / 기본 1회"]
        F --> G["스키마 검증·빈 Need 복구"]
        G --> H["고정밀 Business·Intent·Need 보정"]
        H --> I{"코드 Single Gate"}
        I -->|"필수정보 부족"| J["CLARIFY"]
        I -->|"검색 가능·업무 불확실"| K["RETRIEVE_RELAXED"]
        I -->|"검색 조건 충분"| L["RETRIEVE"]
        J --> M["질문 생성"]
        K --> N["완화 Query Plan"]
        L --> O["일반 Query Plan"]
    ```

    `RETRIEVE_RELAXED`는 사용자에게 다시 묻지 않고 원문 Semantic Search를 넓게 수행하는 내부 검색 Route입니다.
    """
)

# v2 override 셀을 v3 override로 완전히 교체한다. v1/v2의 공통 클래스와 정규화 함수는 앞 셀에서 재사용한다.
cells[7] = code(
    r'''
    PIPELINE_VERSION = "KDIC_LIGHTWEIGHT_RAG_QUERY_ANALYZER_V3_2026_08_12"
    V3_ROUTES = ["RETRIEVE", "RETRIEVE_RELAXED", "CLARIFY", "DIRECT", "OUT_OF_SCOPE"]

    BUSINESS_KEYWORDS = {
        "예금자보호제도": ["예금자보호", "보호한도", "보호대상", "예금 보호", "보호 한도"],
        "예금보험금 안내": ["예금보험금", "보험금 지급", "보험사고", "가지급금", "개산지급금", "1종 보험사고", "2종 보험사고"],
        "고객 미수령금 신청": ["미수령금", "파산배당금", "개산지급금 정산금", "지급대행점", "상속인 금융거래 조회", "상속인 금융거래 조회서비스"],
        "착오송금 반환 신청": ["착오송금", "잘못 보낸 돈", "잘못 송금", "착오 송금", "반환지원", "매입계약", "지급명령", "강제집행", "송금인", "수취인"],
        "채무조정 안내": ["채무조정", "신용회복지원", "파산선고", "면책", "채무감면", "개인회생", "개인파산", "워크아웃", "변제기간", "부채증명원", "채무정보"],
        "은닉재산 신고": ["은닉재산", "은닉 재산", "금융부실관련자", "부실관련자", "차명 재산", "차명재산", "신고 포상금"],
    }

    DIRECT_META_PATTERN = re.compile(
        r"^(?:안녕|안녕하세요|반갑습니다|반가워요|고마워요|고맙습니다|감사합니다|도움이 됐어요|"
        r"알겠습니다|무슨 질문을 할 수 있나요|지원하는 업무를 (?:알려주세요|목록으로 보여주세요)|"
        r"이 챗봇은 어떻게 사용하면 되나요|답변을 쉽게 설명해 줄 수 있나요|"
        r"전문가 수준으로 자세히 설명해 주세요|긴 설명보다 핵심 내용만 먼저 알려주세요|"
        r"질문을 잘못 입력했어요[.]? 다시 물어볼게요)[.!?]*$", re.I,
    )
    OOS_EXACT_PATTERN = re.compile(
        r".*(?:코스피|비트코인|주택담보대출 금리|신용점수|실손보험|국민연금|보이스피싱|"
        r"은행 계좌를 새로|해외송금 수수료|카드 결제|전세대출|세금 환급|퇴직금|"
        r"개인정보 유출|상속세|환율|주식 투자|신용카드 연회비|사업자등록|서울 날씨|"
        r"이번 주말.*날씨).*", re.I,
    )
    USER_ROLE_PATTERNS = [
        ("SENDER", r"송금인|돈을 보낸 사람"), ("RECIPIENT", r"수취인|돈을 받은 사람"),
        ("DEBTOR", r"채무자"), ("REPORTER", r"신고자|신고인"),
        ("CLAIMANT", r"청구인"), ("DEPOSITOR", r"예금자(?!보호)"),
    ]
    APPLICANT_PATTERNS = [
        ("LEGAL_REPRESENTATIVE", r"법정대리인|후견인"),
        ("PROXY", r"대리인|대신해 신청|대리 신청|위임받"),
        ("HEIR", r"상속인"), ("CORPORATION", r"법인(?: 명의|이|으로| 신청)"),
        ("SELF", r"본인이 직접|본인 신청|제가 직접|직접 신청"),
    ]

    def extract_explicit_value(text: str, patterns: list[tuple[str, str]]) -> str | None:
        for value, pattern in patterns:
            if re.search(pattern, text, flags=re.I):
                return value
        return None

    def detect_fast_path_v3(query: str, conversation_state: dict[str, Any] | None = None) -> dict[str, Any] | None:
        text = query.strip()
        if DIRECT_META_PATTERN.fullmatch(text):
            return {"route": "DIRECT", "action": "META_OR_SOCIAL"}
        has_previous = bool((conversation_state or {}).get("has_previous_answer"))
        if has_previous and exact_fullmatch(
            r"(?:쉽게 설명해 주세요|핵심만 알려주세요|표로 정리해 주세요|더 자세히 설명해 주세요)[.!?]*", text
        ):
            return {"route": "DIRECT", "action": "REFORMAT_PREVIOUS_ANSWER"}
        if OOS_EXACT_PATTERN.fullmatch(text) and not find_businesses(text):
            return {"route": "OUT_OF_SCOPE", "action": "EXPLICIT_NON_KDIC"}
        return None

    # Intent 규칙은 단어가 아니라 완성 구문을 우선한다.
    HIGH_PRECISION_INTENT_RULES = [
        ("DOCUMENTS", [r"필요한?\s*(?:서류|증빙)", r"제출(?:해야 하는|할)?\s*서류", r"(?:서류|증빙)(?:가|는|은)?\s*무엇", r"구비\s*서류", r"신분증", r"위임장", r"준비\s*서류"]),
        ("STATUS", [r"어디(?:서|에서)\s*(?:확인|조회)", r"조회\s*(?:방법|결과)", r"처리\s*결과", r"진행\s*상황", r"지급\s*정보.*보는\s*방법"]),
        ("APPLICATION", [r"신청\s*(?:방법|절차)", r"접수\s*(?:방법|절차)", r"제출\s*방법", r"신고\s*채널", r"어떻게\s*(?:신청|접수)", r"(?:온라인|방문|직접).*신청.*(?:가능|할\s*수)", r"취소.*방법", r"철회.*방법"]),
        ("TIME", [r"언제(?:부터|까지)", r"신청.*(?:기한|기간)", r"처리\s*기간", r"소요\s*(?:기간|시간)", r"얼마나\s*걸"]),
        ("AMOUNT", [r"보호\s*한도", r"지급\s*금액", r"금액\s*계산", r"금액.*얼마(?:여야|이어야)", r"계산\s*(?:기준|방법)", r"수수료\s*(?:금액|비용)", r"얼마나\s*(?:감면|지급|보상|돌려|받)"]),
        ("CONTACT", [r"연락처", r"전화번호", r"문의처", r"어디로\s*연락", r"어느\s*기관.*문의"]),
        ("ELIGIBILITY", [r"신청\s*(?:대상|자격|요건)", r"가능한\s*대상", r"제외되는?\s*경우", r"받을\s*수\s*있", r"신청할\s*수\s*있", r"포함되", r"어떤\s*경우.*(?:지급|지원|보호)", r"(?:대상|자격)에\s*해당", r"(?:예금|계좌|금융상품|상품|원금|이자).{0,20}보호(?:가)?\s*되", r"지원\s*대상"]),
        ("OVERVIEW", [r"무엇(?:인가요|인지)", r"의미", r"정의", r"차이", r"종류", r"개요", r"설명"]),
    ]
    WEAK_INTENT_RULES = [
        ("DOCUMENTS", [r"서류", r"증빙"]), ("STATUS", [r"조회", r"확인"]),
        ("APPLICATION", [r"신청", r"접수", r"절차", r"제출"]),
        ("TIME", [r"기간", r"기한", r"시점"]), ("AMOUNT", [r"한도", r"금액", r"계산", r"비용", r"포상금"]),
        ("CONTACT", [r"연락", r"문의", r"전화"]), ("ELIGIBILITY", [r"대상", r"자격", r"요건", r"조건", r"가능"]),
        ("OVERVIEW", [r"설명", r"관계", r"방식"]),
    ]

    def match_intent(text: str, rules: list[tuple[str, list[str]]]) -> tuple[str | None, str | None]:
        for intent, patterns in rules:
            for pattern in patterns:
                if re.search(pattern, text, flags=re.I):
                    return intent, pattern
        return None, None

    def match_all_intents(text: str, rules: list[tuple[str, list[str]]]) -> list[tuple[str, str]]:
        matches = []
        for intent, patterns in rules:
            for pattern in patterns:
                if re.search(pattern, text, flags=re.I):
                    matches.append((intent, pattern))
                    break
        return matches

    def resolve_intent_v3(text: str, model_intent: str | None) -> tuple[str | None, str, str | None]:
        high_matches = match_all_intents(text, HIGH_PRECISION_INTENT_RULES)
        high_intents = list(dict.fromkeys(intent for intent, _ in high_matches))
        if len(high_intents) > 1:
            if model_intent in INTENTS:
                pattern = next((p for i, p in high_matches if i == model_intent), high_matches[0][1])
                return model_intent, "RULE_AMBIGUOUS_MODEL_KEPT", pattern
            return None, "RULE_AMBIGUOUS_UNKNOWN", high_matches[0][1]
        if len(high_intents) == 1:
            high, pattern = high_matches[0]
            if model_intent == high:
                return high, "RULE_CONFIRMED", pattern
            return high, "RULE_OVERRIDE", pattern
        if model_intent in INTENTS:
            weak, weak_pattern = match_intent(text, WEAK_INTENT_RULES)
            if weak and weak != model_intent:
                return model_intent, "RULE_CONFLICT_MODEL_KEPT", weak_pattern
            return model_intent, "MODEL", None
        weak, weak_pattern = match_intent(text, WEAK_INTENT_RULES)
        return weak, "RULE_FILLED_UNKNOWN" if weak else "UNKNOWN", weak_pattern

    def query_analysis_schema_v3() -> dict[str, Any]:
        business_value = {"type": "string", "enum": [*BUSINESS_FUNCTIONS, "UNKNOWN"]}
        intent_value = {"type": "string", "enum": [*INTENTS, "UNKNOWN"]}
        return {
            "type": "object",
            "properties": {
                "needs": {
                    "type": "array", "maxItems": 6,
                    "items": {
                        "type": "object",
                        "properties": {
                            "need_id": {"type": "string"}, "query": {"type": "string"},
                            "business_function": business_value, "intent": intent_value,
                            "target_type": {"type": "string"},
                            "case_details": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["need_id", "query", "business_function", "intent", "target_type", "case_details"],
                    },
                },
            },
            "required": ["needs"],
        }

    SYSTEM_PROMPT_V3 = f"""
    역할: 예금보험공사 RAG의 Atomic Need 분석기.
    출력: JSON Schema만 준수한다. Route와 확인 질문은 결정하지 않는다.

    업무 사전:
    - 예금자보호제도: 보호대상, 보호한도, 금융상품 보호 여부
    - 예금보험금 안내: 보험사고, 예금보험금, 가지급금, 개산지급금
    - 고객 미수령금 신청: 미수령금, 파산배당금, 정산금, 지급대행점, 상속인 금융거래 조회
    - 착오송금 반환 신청: 반환지원, 매입계약, 지급명령, 강제집행
    - 채무조정 안내: 개인회생, 개인파산, 워크아웃, 신용회복지원, 면책, 부채증명원
    - 은닉재산 신고: 금융부실관련자, 차명재산, 신고, 포상금

    규칙:
    1. 사용자에게 다시 질문하지 말고 현재 입력에서 검색할 Need를 최대한 생성한다.
    2. 서로 다른 정보 요구는 N1, N2로 분리한다. 서류와 제출방법은 서로 다른 Need다.
    3. 같은 Intent여도 검색 대상·상황이 다르면 분리한다.
    4. 비교 질문은 대상별 Cartesian 분리보다 정보 차원별로 분리한다.
       예: 개인회생과 워크아웃의 조건과 변제방식 → 조건 비교, 변제방식 비교의 두 Need.
    5. query는 해당 Need만 독립 검색 가능한 문장으로 쓰고 원문 의미를 보존한다.
    6. 원문이 이미 독립적이면 불필요하게 재작성하지 않는다.
    7. 업무나 Intent가 불확실하면 UNKNOWN을 사용하되 Need를 삭제하지 않는다.
    8. target_type과 case_details는 원문에 명시된 검색 조건만 기록한다.

    허용 business_function: {BUSINESS_FUNCTIONS}
    허용 intent: {INTENTS}
    """

    class HCX007AtomicNeedClientV3(HCX007StructuredClient):
        def analyze(self, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            started = time.perf_counter()
            attempts = []
            totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            last_error = None
            body = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT_V3},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                "topP": self.config.top_p, "topK": self.config.top_k,
                "maxCompletionTokens": self.config.max_completion_tokens,
                "temperature": self.config.temperature, "repetitionPenalty": 1.05,
                "thinking": {"effort": "none"}, "stop": [],
                "responseFormat": {"type": "json", "schema": query_analysis_schema_v3()},
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

    def validate_atomic_needs_v3(raw: dict[str, Any], normalized_query: str) -> tuple[list[dict[str, Any]], list[str]]:
        warnings = []
        rows = raw.get("needs") if isinstance(raw.get("needs"), list) else []
        needs = []
        for index, row in enumerate(rows[:6], 1):
            if not isinstance(row, dict):
                warnings.append(f"N{index}_NOT_OBJECT")
                continue
            query = str(row.get("query") or normalized_query).strip() or normalized_query
            business = row.get("business_function")
            business = business if business in BUSINESS_FUNCTIONS else None
            model_intent = normalize_intent(row.get("intent"))
            target = str(row.get("target_type") or "").strip() or None
            cases = row.get("case_details")
            cases = [str(x).strip() for x in cases if str(x).strip()] if isinstance(cases, list) else []
            needs.append({
                "need_id": f"N{index}", "query": query,
                "business_function": business, "business_source": "MODEL" if business else None,
                "model_intent": model_intent, "intent": model_intent, "intent_source": "MODEL" if model_intent else "UNKNOWN",
                "intent_rule_pattern": None, "target_type": target, "case_details": cases,
                "user_role": None, "user_role_source": None,
                "applicant_type": None, "applicant_type_source": None,
            })
        if not needs:
            warnings.append("EMPTY_MODEL_NEEDS_RECOVERED")
            needs = [create_retrieve_need_v3(normalized_query)]
        return needs, warnings

    def create_retrieve_need_v3(query: str, business: str | None = None) -> dict[str, Any]:
        intent, source, pattern = resolve_intent_v3(query, None)
        return {
            "need_id": "N1", "query": query,
            "business_function": business, "business_source": "ORIGINAL" if business else None,
            "model_intent": None, "intent": intent, "intent_source": source, "intent_rule_pattern": pattern,
            "target_type": None, "case_details": [],
            "user_role": None, "user_role_source": None,
            "applicant_type": None, "applicant_type_source": None,
        }

    GENERIC_BUSINESS_CLARIFY_PATTERN = re.compile(
        r"^(?:신청\s*(?:방법|자격|기한|과정)|접수\s*(?:방법|절차)|제출해야 하는 서류|필요한 서류|"
        r"조회는 어디에서|온라인으로 신청|방문해서 접수|접수 후 처리 기간|신청 대상|신청 자격과 제외 조건|"
        r"신청 과정에서 수수료나 비용|처리 결과는 어디에서 확인|이미 접수한 신청을 취소|"
        r"문의하거나 접수하려면 어느 기관|제가 신청 대상에 해당|본인 대신 대리인이 신청|"
        r"상속인이 신청하거나 받을 수).*$", re.I,
    )
    TARGET_DEMONSTRATIVE_PATTERN_V3 = re.compile(
        r"(?:^|[\s,.(])(?:제가\s*(?:말한|가입한|가진|본)\s*)?이\s*(?:금융상품|계좌|상품|돈|송금|거래|금액|채무|재산)(?:을|를|이|가|도|은|는|의|이나|과|와|\s|[,.!?]|$)"
    )
    TARGET_REFERENCE_PHRASES_V3 = [
        "제가 가입한 상품", "어떤 송금 건", "어떤 예금에 대해", "어떤 돈을 신청", "어떤 예금이나 금융상품",
    ]
    APPLICANT_REFERENCE_PHRASES_V3 = [
        "제 신청 유형", "제 신청 자격", "제 경우", "누구를 신청인", "누가 방문", "신고 주체 유형", "신청인란",
    ]
    CASE_REFERENCE_PHRASES_V3 = [
        "제 상황", "현재 상황", "제 채무 상태", "제 신고 상황", "반려", "거절", "보완 요청", "진행되지 않",
        "여러 금융회사에 예금", "여러 계좌에 나뉘",
    ]

    def has_resolved_target(query: str, needs: list[dict[str, Any]]) -> bool:
        if any(n.get("target_type") for n in needs):
            return True
        demonstrative = TARGET_DEMONSTRATIVE_PATTERN_V3.search(query)
        if not demonstrative:
            return False
        phrase = demonstrative.group(0)
        # 지시 표현 자체만 있을 뿐 구체 상품명·거래정보는 없으므로 unresolved로 본다.
        return False if phrase else True

    def detect_blocking_slot_v3(
        query: str, needs: list[dict[str, Any]], context: dict[str, Any]
    ) -> tuple[str | None, str | None]:
        confirmed = context.get("confirmed", {}) if context.get("used") else {}
        if GENERIC_BUSINESS_CLARIFY_PATTERN.fullmatch(query.strip()) and not find_businesses(query) and not confirmed.get("business_function"):
            return "business_function", "GENERIC_BUSINESS_EXACT_MATCH"
        if any(x in query for x in APPLICANT_REFERENCE_PHRASES_V3):
            if not confirmed.get("applicant_type") and not extract_explicit_value(query, APPLICANT_PATTERNS):
                return "applicant_type", "UNRESOLVED_APPLICANT_REFERENCE"
        if any(x in query for x in TARGET_REFERENCE_PHRASES_V3) or TARGET_DEMONSTRATIVE_PATTERN_V3.search(query):
            if not confirmed.get("target_type") and not has_resolved_target(query, needs):
                return "target_type", "UNRESOLVED_TARGET_REFERENCE"
        if any(x in query for x in CASE_REFERENCE_PHRASES_V3):
            return "case_details", "PERSONAL_CASE_REQUIRES_DETAILS"
        return None, None

    def apply_business_and_intent_rules_v3(needs: list[dict[str, Any]], original: str) -> tuple[list[dict[str, Any]], list[str]]:
        actions = []
        for need in needs:
            query = need.get("query") or original
            explicit_businesses = find_businesses(query)
            if not explicit_businesses:
                explicit_businesses = find_businesses(original)
            if len(explicit_businesses) == 1 and explicit_businesses[0] != need.get("business_function"):
                need["model_business_function"] = need.get("business_function")
                need["business_function"] = explicit_businesses[0]
                need["business_source"] = "ORIGINAL"
                actions.append(f"{need['need_id']}_BUSINESS_ORIGINAL_OVERRIDE")
            final_intent, source, pattern = resolve_intent_v3(query, need.get("model_intent"))
            need["intent"] = final_intent
            need["intent_source"] = source
            need["intent_rule_pattern"] = pattern
            if source in {"RULE_OVERRIDE", "RULE_FILLED_UNKNOWN", "RULE_CONFLICT_MODEL_KEPT", "RULE_AMBIGUOUS_MODEL_KEPT", "RULE_AMBIGUOUS_UNKNOWN"}:
                actions.append(f"{need['need_id']}_{source}")

        # 완전 중복만 제거한다. 같은 업무·Intent라도 독립 질의나 조건이 다르면 유지한다.
        deduped, seen = [], set()
        for need in needs:
            key = (
                need.get("business_function"), need.get("intent"),
                normalize_query(need.get("query") or "")["normalized_query"],
                need.get("target_type"), tuple(sorted(need.get("case_details") or [])),
            )
            if key in seen:
                actions.append("EXACT_DUPLICATE_NEED_COLLAPSED")
                continue
            seen.add(key)
            deduped.append(need)
        for index, need in enumerate(deduped, 1):
            need["need_id"] = f"N{index}"
        return deduped, list(dict.fromkeys(actions))

    def decide_route_v3(
        query: str, needs: list[dict[str, Any]], context: dict[str, Any]
    ) -> tuple[str, list[str], str | None]:
        blocking_slot, reason = detect_blocking_slot_v3(query, needs, context)
        if blocking_slot:
            return "CLARIFY", [reason], blocking_slot
        businesses = [n.get("business_function") for n in needs if n.get("business_function") in BUSINESS_FUNCTIONS]
        if not businesses or any(n.get("business_function") not in BUSINESS_FUNCTIONS for n in needs):
            return "RETRIEVE_RELAXED", ["BUSINESS_UNCERTAIN_BROAD_RETRIEVAL"], None
        return "RETRIEVE", ["SEARCH_CONDITIONS_AVAILABLE"], None

    def enrich_evidence_v3(
        needs: list[dict[str, Any]], original: str, context: dict[str, Any], manual: dict[str, Any]
    ) -> list[dict[str, Any]]:
        explicit_businesses = find_businesses(original)
        explicit_role = extract_explicit_value(original, USER_ROLE_PATTERNS)
        explicit_applicant = extract_explicit_value(original, APPLICANT_PATTERNS)
        confirmed = context.get("confirmed", {}) if context.get("used") else {}
        for need in needs:
            business = need.get("business_function")
            if manual.get("business_function") == business:
                need["business_source"] = "MANUAL"
            elif business in explicit_businesses:
                need["business_source"] = "ORIGINAL"
            elif confirmed.get("business_function") == business:
                need["business_source"] = "CONTEXT"
            elif business:
                need["business_source"] = need.get("business_source") or "MODEL"
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
        return needs

    def build_keyword_query_v3(need: dict[str, Any]) -> str:
        values = [need.get("business_function"), need.get("intent"), need.get("target_type")]
        values.extend(need.get("case_details") or [])
        if need.get("user_role_source") in {"ORIGINAL", "CONTEXT", "MANUAL"}:
            values.append(need.get("user_role"))
        if need.get("applicant_type_source") in {"ORIGINAL", "CONTEXT", "MANUAL"}:
            values.append(need.get("applicant_type"))
        return " ".join(str(x) for x in values if x and x != "UNKNOWN")

    def build_query_plans_v3(analysis: dict[str, Any], config: PipelineConfig) -> list[dict[str, Any]]:
        if analysis["route"] not in {"RETRIEVE", "RETRIEVE_RELAXED"}:
            return []
        relaxed = analysis["route"] == "RETRIEVE_RELAXED"
        plans = []
        for need in analysis["needs"]:
            business, source = need.get("business_function"), need.get("business_source")
            if relaxed or not business:
                business_filter = {"mode": "NONE", "value": None, "soft_hint": business, "evidence": source or "UNKNOWN"}
            elif source in {"ORIGINAL", "CONTEXT", "MANUAL"}:
                business_filter = {"mode": "HARD", "value": business, "soft_hint": None, "evidence": source}
            else:
                business_filter = {"mode": "SOFT", "value": None, "soft_hint": business, "evidence": "MODEL"}
            intent = need.get("intent")
            intent_weight = 0.20 if need.get("intent_source") in {"RULE_OVERRIDE", "RULE_CONFIRMED", "RULE_FILLED_UNKNOWN"} else config.intent_soft_boost
            plans.append({
                "need_id": need["need_id"], "retrieval_mode": "RELAXED" if relaxed else "STANDARD",
                "semantic_query": need.get("query"), "keyword_query": build_keyword_query_v3(need) or need.get("query"),
                "business_filter": business_filter,
                "intent_boost": {
                    "mode": "SOFT" if intent in INTENTS else "NONE", "value": intent,
                    "weight": intent_weight if intent in INTENTS else 0.0, "evidence": need.get("intent_source"),
                },
                "entities": {
                    "user_role": need.get("user_role"), "user_role_source": need.get("user_role_source"),
                    "applicant_type": need.get("applicant_type"), "applicant_type_source": need.get("applicant_type_source"),
                    "target_type": need.get("target_type"), "case_details": need.get("case_details") or [],
                },
            })
        return plans

    class KDICLightweightRAGAnalyzerV3:
        def __init__(self, client: HCX007AtomicNeedClientV3, config: PipelineConfig | None = None):
            self.client = client
            self.config = config or client.config

        def run(self, query: str, *, conversation_state=None, manual_selection=None) -> dict[str, Any]:
            started = time.perf_counter()
            normalized = normalize_query(query)
            original, text = normalized["original_query"], normalized["normalized_query"]
            manual = dict(manual_selection or {})
            context = build_context(text, conversation_state, self.config.max_context_turns)
            fast = detect_fast_path_v3(text, conversation_state)
            if fast:
                analysis = {"route": fast["route"], "needs": [], "missing_information": []}
                return {
                    "pipeline_version": PIPELINE_VERSION, "analysis_status": "FAST_PATH",
                    "original_query": original, "normalized_query": text, "context": context,
                    "gate_reasons": ["FAST_PATH"], "rule_actions": [], "blocking_slot": None,
                    "analysis": analysis, "fast_path": fast, "validation_warnings": [], "query_plans": [],
                    "runtime": {"api_request_count": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                                "latency_ms": round((time.perf_counter() - started) * 1000, 3), "attempts": []},
                }
            payload = {"query": text, "confirmed_context": context["confirmed"], "recent_turns": context["recent_turns"], "manual_selection": manual}
            try:
                raw, telemetry = self.client.analyze(payload)
                needs, warnings = validate_atomic_needs_v3(raw, text)
                status = "REPAIRED" if warnings else "OK"
            except HCXAPIError as exc:
                telemetry = exc.telemetry
                needs = [create_retrieve_need_v3(text, find_businesses(text)[0] if len(find_businesses(text)) == 1 else None)]
                warnings = ["MODEL_ANALYSIS_FAILED_USED_FALLBACK"]
                status = "FALLBACK"
            model_needs = json.loads(json.dumps(needs, ensure_ascii=False))
            needs, rule_actions = apply_business_and_intent_rules_v3(needs, original)
            needs = enrich_evidence_v3(needs, original, context, manual)
            route, gate_reasons, blocking_slot = decide_route_v3(text, needs, context)
            analysis_needs = [] if route == "CLARIFY" else needs
            analysis = {"route": route, "needs": analysis_needs, "missing_information": [blocking_slot] if blocking_slot else []}
            plans = build_query_plans_v3(analysis, self.config)
            return {
                "pipeline_version": PIPELINE_VERSION, "analysis_status": status,
                "original_query": original, "normalized_query": text, "context": context,
                "model_needs": model_needs, "gate_reasons": gate_reasons,
                "rule_actions": rule_actions, "blocking_slot": blocking_slot,
                "analysis": analysis, "validation_warnings": warnings, "query_plans": plans,
                "runtime": {**telemetry, "latency_ms": round((time.perf_counter() - started) * 1000, 3)},
            }

    CFG_V3 = PipelineConfig(max_completion_tokens=700, temperature=0.1)
    HCX_CLIENT_V3 = HCX007AtomicNeedClientV3(get_hcx_api_key("HCX_API_KEY"), CFG_V3)
    PIPELINE = KDICLightweightRAGAnalyzerV3(HCX_CLIENT_V3, CFG_V3)
    CFG = CFG_V3
    print("v3 파이프라인 준비 완료:", PIPELINE_VERSION)
    '''
)

cells[8] = md(
    """
    ## 빠른 기능 확인

    일반 문항은 HCX를 한 번 호출합니다. Fast Path 문항은 API 호출 없이 처리됩니다.
    `model_needs`와 최종 `analysis.needs`를 비교하면 규칙의 교정 내용을 확인할 수 있습니다.
    """
)

cells[9] = code(
    r'''
    test_queries = [
        "안녕하세요.",
        "이번 주말 서울 날씨를 알려주세요.",
        "신청 방법을 알려주세요.",
        "예금보험금 신청 절차를 알려주세요.",
        "제가 가입한 이 상품도 예금자보호 대상인가요?",
        "예금자 본인이 신청할 때 필요한 서류와 제출 방법은 무엇인가요?",
        "개인회생과 워크아웃의 신청 조건과 변제 방식 차이를 알려주세요.",
    ]
    for q in test_queries:
        result = PIPELINE.run(q)
        print("\nQUERY:", q)
        print(json.dumps({
            "status": result["analysis_status"], "route": result["analysis"]["route"],
            "gate_reasons": result["gate_reasons"], "rule_actions": result["rule_actions"],
            "missing": result["analysis"].get("missing_information"),
            "model_needs": result.get("model_needs"), "final_needs": result["analysis"].get("needs"),
            "query_plans": result["query_plans"], "runtime": result["runtime"],
        }, ensure_ascii=False, indent=2))
    '''
)

cells[10] = code(
    r'''
    state = {
        "confirmed": {"business_function": "예금보험금 안내", "applicant_type": "HEIR", "target_type": "정기예금"},
        "recent_turns": [{"user": "상속인이 정기예금의 예금보험금을 신청하려고 해요."}],
        "has_previous_answer": True,
    }
    follow_up = PIPELINE.run("그 경우 필요한 서류만 알려주세요.", conversation_state=state)
    print(json.dumps(follow_up, ensure_ascii=False, indent=2))
    '''
)

cells[11] = md(
    """
    ## Evaluation_DataSet_v5 v3 평가

    기본 `SMOKE` 20건으로 먼저 실행합니다. 정상 동작 후 `EVAL_MODE = "FULL"`로 변경합니다.

    평가에서는 `RETRIEVE_RELAXED`를 검색 행동 기준으로 `RETRIEVE`와 동일하게 정규화하되, 내부 Route도 별도로 저장합니다.
    모델 Need와 규칙 적용 후 Need를 함께 저장하여 Business·Intent 규칙의 개선과 회귀를 측정합니다.
    """
)

# v2의 데이터셋 업로드/파싱 셀은 그대로 사용한다.
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

    def public_route(route):
        return {"DIRECT": "DIRECT_RESPONSE", "RETRIEVE_RELAXED": "RETRIEVE"}.get(route, route)

    records = []
    raw_path = Path("kdic_lightweight_rag_v3_eval_raw.jsonl")
    with raw_path.open("w", encoding="utf-8") as raw_file:
        for index, gold in enumerate(eval_rows, 1):
            try:
                result = PIPELINE.run(gold["question"])
            except Exception as exc:
                result = {
                    "pipeline_version": PIPELINE_VERSION, "analysis_status": "ERROR",
                    "gate_reasons": [f"EVALUATION_ERROR: {type(exc).__name__}"], "rule_actions": [], "blocking_slot": None,
                    "model_needs": [], "analysis": {"route": "RETRIEVE_RELAXED", "needs": [create_retrieve_need_v3(gold["question"])], "missing_information": []},
                    "validation_warnings": [str(exc)], "query_plans": [],
                    "runtime": {"api_request_count": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "latency_ms": 0.0, "attempts": []},
                }
            analysis = result["analysis"]
            internal_route = analysis["route"]
            pred_route = public_route(internal_route)
            needs = analysis.get("needs") or []
            model_needs = result.get("model_needs") or []
            pred_query_type = "MULTI" if pred_route == "RETRIEVE" and len(needs) >= 2 else ("SINGLE" if pred_route == "RETRIEVE" else "NONE")
            pred_pairs = [
                (n.get("business_function"), n.get("intent")) for n in needs
                if n.get("business_function") not in {None, "UNKNOWN"} and n.get("intent") not in {None, "UNKNOWN"}
            ]
            model_pairs = [
                (n.get("business_function"), n.get("model_intent") or n.get("intent")) for n in model_needs
                if n.get("business_function") not in {None, "UNKNOWN"}
                and (n.get("model_intent") or n.get("intent")) not in {None, "UNKNOWN"}
            ]
            gold_pairs = gold["gold_request_pairs"]
            pred_businesses, pred_intents = [x[0] for x in pred_pairs], [x[1] for x in pred_pairs]
            gold_businesses, gold_intents = [x[0] for x in gold_pairs], [x[1] for x in gold_pairs]
            retrieve_case, clarify_case = gold["gold_route"] == "RETRIEVE", gold["gold_route"] == "CLARIFY"
            pair_exact = Counter(gold_pairs) == Counter(pred_pairs) if retrieve_case else None
            model_pair_exact = Counter(gold_pairs) == Counter(model_pairs) if retrieve_case else None
            missing_exact = set(gold["gold_missing_slots"]) == set(analysis.get("missing_information") or []) if clarify_case else None
            route_correct = pred_route == gold["gold_route"]
            query_type_correct = pred_query_type == gold["gold_query_type"]
            core_exact = route_correct and (query_type_correct and pair_exact if retrieve_case else (missing_exact if clarify_case else True))
            runtime = result["runtime"]
            record = {
                **gold, "analysis_status": result["analysis_status"],
                "pred_route": pred_route, "pred_route_internal": internal_route,
                "gate_reasons": result.get("gate_reasons") or [], "rule_actions": result.get("rule_actions") or [],
                "blocking_slot": result.get("blocking_slot"), "route_correct": route_correct,
                "pred_query_type": pred_query_type, "query_type_correct": query_type_correct,
                "model_request_pairs": model_pairs, "pred_request_pairs": pred_pairs,
                "model_pair_exact": model_pair_exact, "request_pair_exact": pair_exact,
                "rule_improved_pair": bool(pair_exact and model_pair_exact is False),
                "rule_regressed_pair": bool(model_pair_exact and pair_exact is False),
                "gold_businesses": gold_businesses, "pred_businesses": pred_businesses,
                "gold_intents": gold_intents, "pred_intents": pred_intents,
                "pred_missing_slots": analysis.get("missing_information") or [], "missing_slot_exact": missing_exact,
                "core_exact": bool(core_exact), "query_plan_count": len(result["query_plans"]),
                "api_request_count": int(runtime.get("api_request_count") or 0),
                "prompt_tokens": int(runtime.get("prompt_tokens") or 0), "completion_tokens": int(runtime.get("completion_tokens") or 0),
                "total_tokens": int(runtime.get("total_tokens") or 0), "wall_latency_ms": float(runtime.get("latency_ms") or 0),
                "warning_count": len(result.get("validation_warnings") or []),
            }
            records.append(record)
            raw_file.write(json.dumps({"gold": gold, "result": result}, ensure_ascii=False, default=str) + "\n")
            if index == 1 or index % PROGRESS_EVERY == 0 or index == len(eval_rows):
                print(f"[{index}/{len(eval_rows)}] status={Counter(r['analysis_status'] for r in records)}")
            time.sleep(CFG.request_interval_seconds)

    result_df = pd.DataFrame(records)
    retrieve_df = result_df[result_df.gold_route == "RETRIEVE"].copy()
    pair_micro = micro_from_columns(retrieve_df, "gold_request_pairs", "pred_request_pairs")
    pair_set = micro_from_columns(retrieve_df, "gold_request_pairs", "pred_request_pairs", use_set=True)
    business_micro = micro_from_columns(retrieve_df, "gold_businesses", "pred_businesses")
    business_set = micro_from_columns(retrieve_df, "gold_businesses", "pred_businesses", use_set=True)
    intent_micro = micro_from_columns(retrieve_df, "gold_intents", "pred_intents")
    intent_set = micro_from_columns(retrieve_df, "gold_intents", "pred_intents", use_set=True)
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
        "final_route_accuracy": float(result_df.route_correct.mean()),
        "standard_retrieve_rate": float((result_df.pred_route_internal == "RETRIEVE").mean()),
        "relaxed_retrieve_rate": float((result_df.pred_route_internal == "RETRIEVE_RELAXED").mean()),
        "query_type_accuracy": float(result_df.query_type_correct.mean()), "core_exact": float(result_df.core_exact.mean()),
        "business_micro_f1": business_micro["f1"], "business_set_micro_f1": business_set["f1"],
        "intent_micro_f1": intent_micro["f1"], "intent_set_micro_f1": intent_set["f1"],
        "request_pair_micro_precision": pair_micro["precision"], "request_pair_micro_recall": pair_micro["recall"],
        "request_pair_micro_f1": pair_micro["f1"], "request_pair_set_micro_f1": pair_set["f1"],
        "clarify_precision": clarify_tp / len(pred_clarify) if len(pred_clarify) else 0.0,
        "clarify_recall": clarify_tp / len(clarify_rows) if len(clarify_rows) else 0.0,
        "clarify_missing_exact": float(sum(bool(x) for x in clarify_rows.missing_slot_exact if pd.notna(x)) / len(clarify_rows)) if len(clarify_rows) else 1.0,
        "rule_improved_pair_count": int(result_df.rule_improved_pair.sum()),
        "rule_regressed_pair_count": int(result_df.rule_regressed_pair.sum()),
        "average_api_requests": float(result_df.api_request_count.mean()), "zero_call_rate": float((result_df.api_request_count == 0).mean()),
        "average_prompt_tokens": float(result_df.prompt_tokens.mean()), "average_completion_tokens": float(result_df.completion_tokens.mean()),
        "average_total_tokens": float(result_df.total_tokens.mean()), "average_latency_ms": float(result_df.wall_latency_ms.mean()),
        "p95_latency_ms": float(result_df.wall_latency_ms.quantile(0.95)),
    }
    for query_type in ["SINGLE", "MULTI", "NONE"]:
        subset = result_df[result_df.gold_query_type == query_type]
        if len(subset):
            key = query_type.lower()
            summary[f"{key}_route_accuracy"] = float(subset.route_correct.mean())
            summary[f"{key}_query_type_accuracy"] = float(subset.query_type_correct.mean())
            summary[f"{key}_core_exact"] = float(subset.core_exact.mean())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    '''
)

cells[14] = code(
    r'''
    summary_path = Path("kdic_lightweight_rag_v3_eval_summary.json")
    detail_path = Path("kdic_lightweight_rag_v3_eval_per_question.csv")
    route_confusion_path = Path("kdic_lightweight_rag_v3_route_confusion.csv")
    query_type_confusion_path = Path("kdic_lightweight_rag_v3_query_type_confusion.csv")
    gate_reason_path = Path("kdic_lightweight_rag_v3_gate_reasons.csv")
    rule_action_path = Path("kdic_lightweight_rag_v3_rule_actions.csv")
    improvement_path = Path("kdic_lightweight_rag_v3_rule_improvements.csv")
    regression_path = Path("kdic_lightweight_rag_v3_rule_regressions.csv")
    error_type_path = Path("kdic_lightweight_rag_v3_error_types.csv")
    status_path = Path("kdic_lightweight_rag_v3_eval_status.csv")
    zip_path = Path("kdic_lightweight_rag_v3_evaluation_bundle.zip")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    route_confusion = pd.crosstab(result_df.gold_route, result_df.pred_route, margins=True).reset_index()
    query_type_confusion = pd.crosstab(result_df.gold_query_type, result_df.pred_query_type, margins=True).reset_index()
    gate_reason_counts = Counter(reason for reasons in result_df.gate_reasons for reason in reasons)
    gate_reason_table = pd.DataFrame([{"gate_reason": k, "count": v} for k, v in gate_reason_counts.most_common()])
    rule_action_counts = Counter(action for actions in result_df.rule_actions for action in actions)
    rule_action_table = pd.DataFrame([{"rule_action": k, "count": v} for k, v in rule_action_counts.most_common()])
    status_table = result_df.analysis_status.value_counts(dropna=False).rename_axis("analysis_status").reset_index(name="count")

    def classify_error(row):
        errors = []
        if not row.route_correct: errors.append("ROUTE")
        if row.gold_route == "RETRIEVE":
            if Counter(row.gold_businesses) != Counter(row.pred_businesses): errors.append("BUSINESS_OR_COUNT")
            if Counter(row.gold_intents) != Counter(row.pred_intents): errors.append("INTENT_OR_COUNT")
            if not row.query_type_correct: errors.append("QUERY_TYPE")
        if row.gold_route == "CLARIFY" and row.missing_slot_exact == False: errors.append("MISSING_SLOT")
        return errors or ["CORRECT"]

    result_df["error_types"] = result_df.apply(classify_error, axis=1)
    error_counts = Counter(error for errors in result_df.error_types for error in errors)
    error_table = pd.DataFrame([{"error_type": k, "count": v} for k, v in error_counts.most_common()])

    result_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    route_confusion.to_csv(route_confusion_path, index=False, encoding="utf-8-sig")
    query_type_confusion.to_csv(query_type_confusion_path, index=False, encoding="utf-8-sig")
    gate_reason_table.to_csv(gate_reason_path, index=False, encoding="utf-8-sig")
    rule_action_table.to_csv(rule_action_path, index=False, encoding="utf-8-sig")
    result_df[result_df.rule_improved_pair].to_csv(improvement_path, index=False, encoding="utf-8-sig")
    result_df[result_df.rule_regressed_pair].to_csv(regression_path, index=False, encoding="utf-8-sig")
    error_table.to_csv(error_type_path, index=False, encoding="utf-8-sig")
    status_table.to_csv(status_path, index=False, encoding="utf-8-sig")

    export_paths = [
        summary_path, detail_path, route_confusion_path, query_type_confusion_path,
        gate_reason_path, rule_action_path, improvement_path, regression_path,
        error_type_path, status_path, raw_path,
    ]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in export_paths:
            archive.write(path, arcname=path.name)
    print("생성:", zip_path.resolve())
    '''
)

cells[15] = md(
    """
    ## v3 FULL 실행 전 기준

    - LLM Structured Success ≥ 98%
    - Fallback ≤ 5%
    - Final Route Accuracy ≥ 90%
    - CLARIFY Precision ≥ 90%, Recall ≥ 80%
    - Business set micro F1 ≥ 90%
    - Intent micro F1 ≥ 75%
    - MULTI Query Type Accuracy ≥ 80%
    - 규칙 Pair Regression은 개선 건수보다 충분히 작아야 함
    - P95 질의분석 지연시간 ≤ 4초

    v3의 첫 FULL 결과는 v2 점수와 직접 비교하되, `RETRIEVE_RELAXED`는 공개 평가 Route에서 `RETRIEVE`로 정규화합니다.
    """
)

notebook["metadata"]["kdic_pipeline_version"] = "v3"
OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(OUTPUT)
