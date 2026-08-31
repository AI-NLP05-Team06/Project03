from __future__ import annotations

"""KDIC 간편 라우터 V1.

설계 목표
---------
1. 명백한 DIRECT/OUT_OF_SCOPE/CLARIFY만 규칙으로 차단하고 나머지는 검색으로 보낸다.
2. 단일질의의 검색 문자열은 사용자 원문을 그대로 보존한다.
3. 실제로 독립 검색이 필요한 복합질의만 보수적으로 분해한다.
4. 업무 필터는 SOFT/NONE만 사용해 잘못된 HARD 필터를 구조적으로 막는다.
5. 외부 API나 모델을 호출하지 않아 라우팅 지연을 최소화한다.
"""

import json
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


PIPELINE_VERSION = "KDIC_LIGHTWEIGHT_ROUTER_V1_2026_08_28_NATURAL_TRANSFER"

BUSINESS_FUNCTIONS = (
    "예금자보호제도",
    "예금보험금 안내",
    "고객 미수령금 신청",
    "착오송금 반환 신청",
    "채무조정 안내",
    "은닉재산 신고",
)

INTENTS = (
    "AMOUNT",
    "ELIGIBILITY",
    "TIME",
    "APPLICATION",
    "OVERVIEW",
    "STATUS",
    "DOCUMENTS",
    "CONTACT",
)

BUSINESS_KEYWORDS: dict[str, tuple[str, ...]] = {
    "예금자보호제도": (
        "예금자보호", "보호한도", "보호 한도", "보호대상", "예금 보호",
        "예금은 얼마까지 보호", "금융상품이 보호", "금융회사가 보호 대상",
    ),
    "예금보험금 안내": (
        "예금보험금", "보험금 지급", "보험사고", "가지급금", "개산지급금",
        "1종 보험사고", "2종 보험사고",
    ),
    "고객 미수령금 신청": (
        "고객 미수령금", "미수령금", "파산배당금", "개산지급금 정산금",
        "지급대행점", "상속인 금융거래 조회", "상속인 금융거래 조회서비스",
    ),
    "착오송금 반환 신청": (
        "착오송금", "착오 송금", "잘못 보낸 돈", "잘못 송금", "반환지원",
        "매입계약", "지급명령", "강제집행", "송금인", "수취인",
        "계좌번호를 잘못", "엉뚱한 사람에게 보낸 돈",
    ),
    "채무조정 안내": (
        "채무조정", "신용회복지원", "파산선고", "채무감면", "개인회생",
        "개인파산", "워크아웃", "변제기간", "부채증명원", "채무정보", "면책",
    ),
    "은닉재산 신고": (
        "은닉재산", "은닉 재산", "금융부실관련자", "부실관련자", "차명재산",
        "차명 재산", "신고 포상금",
    ),
}

STRONG_BUSINESS_KEYWORDS: dict[str, tuple[str, ...]] = {
    "예금자보호제도": ("예금자보호", "보호한도", "보호 한도", "예금 보호"),
    "예금보험금 안내": ("예금보험금", "보험금 지급", "1종 보험사고", "2종 보험사고"),
    "고객 미수령금 신청": (
        "고객 미수령금", "미수령금", "파산배당금", "지급대행점", "상속인 금융거래 조회",
    ),
    "착오송금 반환 신청": (
        "착오송금", "착오 송금", "반환지원", "잘못 보낸 돈", "지급명령", "강제집행",
    ),
    "채무조정 안내": (
        "채무조정", "신용회복지원", "개인회생", "개인파산", "워크아웃", "부채증명원",
    ),
    "은닉재산 신고": (
        "은닉재산", "은닉 재산", "금융부실관련자", "차명재산", "차명 재산", "신고 포상금",
    ),
}

_DEPOSIT_INSURANCE_SYSTEM_PATTERN = re.compile(
    r"예금\s*보험(?:\s*제도)?(?!\s*금)",
    re.I,
)

TYPO_MAP = (
    ("예금보헝금", "예금보험금"),
    ("예금보혐금", "예금보험금"),
    ("착오송금반한", "착오송금 반환"),
    ("반한지원", "반환지원"),
    ("반환지웜", "반환지원"),
    ("미수령금신정", "미수령금 신청"),
    ("통합신정", "통합신청"),
    ("검새", "검색"),
    ("발샐", "발생"),
    ("관게", "관계"),
    ("요정", "요청"),
    ("언재", "언제"),
    ("제외돼는", "제외되는"),
)

DIRECT_META_PATTERN = re.compile(
    r"^(?:안녕|안녕하세요|반갑습니다|반가워요|고마워요|고맙습니다|감사합니다|도움이 됐어요|"
    r"알겠습니다|무슨 질문을 할 수 있나요|지원하는 업무를 (?:알려주세요|목록으로 보여주세요)|"
    r"이 챗봇은 어떻게 사용하면 되나요|답변을 쉽게 설명해 줄 수 있나요|"
    r"전문가 수준으로 자세히 설명해 주세요|긴 설명보다 핵심 내용만 먼저 알려주세요|"
    r"질문을 잘못 입력했어요[.]? 다시 물어볼게요)[.!?]*$",
    re.I,
)

DIRECT_ACTION_RESPONSES: dict[str, str] = {
    "GREETING": "안녕하세요! 예금보험공사 관련 제도나 신청 절차를 편하게 물어보세요.",
    "THANKS": "도움이 되었다니 다행입니다. 다른 궁금한 내용도 편하게 물어보세요.",
    "ACKNOWLEDGEMENT": "네, 알겠습니다. 이어서 궁금한 내용이 있으면 말씀해 주세요.",
    "CLOSING": "이용해 주셔서 감사합니다. 필요할 때 다시 찾아주세요.",
    "REACTION": "네! 예금자보호·예금보험금·미수령금·착오송금 반환지원 등 궁금한 내용을 물어보세요.",
    "CAPABILITY": (
        "예금자보호, 예금보험금, 고객 미수령금, 착오송금 반환지원, "
        "채무조정, 은닉재산 신고 관련 내용을 질문할 수 있습니다."
    ),
    "CANCEL": "알겠습니다. 현재 요청을 중단했습니다.",
    "META_OR_SOCIAL": "예금보험공사 관련 제도나 신청 절차를 편하게 물어보세요.",
    "LOW_INFORMATION": (
        "질문의 뜻을 정확히 파악하기 어렵습니다. "
        "예금자보호, 예금보험금, 미수령금처럼 궁금한 주제를 조금 더 구체적으로 적어 주세요."
    ),
}

# 착오송금은 사용자가 제도명을 모른 채 일상어로 설명하는 경우가 많다.
# 단순 부분 문자열 목록만으로는 ``잘못 보낸 돈``과 어순이 다른
# ``돈을 잘못보냈다``를 놓치므로, 송금 행위와 실수 표현이 함께 있는
# 문장 패턴을 별도의 강한 업무 근거로 사용한다.
MISTAKEN_TRANSFER_NATURAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "MISTAKEN_TRANSFER_SENDER_OBJECT_FIRST",
        re.compile(
            r"(?:돈|금액|송금|이체)(?:을|를)?\s*"
            r"(?:잘못|실수(?:로)?|착각(?:해서|으로)?)\s*"
            r"(?:보내|보냈|송금|이체)",
            re.I,
        ),
    ),
    (
        "MISTAKEN_TRANSFER_SENDER_ERROR_FIRST",
        re.compile(
            r"(?:잘못|실수(?:로)?|착각(?:해서|으로)?).{0,10}"
            r"(?:돈|금액)(?:을|를)?\s*(?:보내|보냈|송금|이체)",
            re.I,
        ),
    ),
    (
        "MISTAKEN_TRANSFER_ACTION_ERROR",
        re.compile(
            r"(?:송금|이체)(?:을|를)?\s*"
            r"(?:잘못(?:했|해서|한|했다)?|실수(?:했|해서|로)?|착각(?:했|해서|으로)?)",
            re.I,
        ),
    ),
    (
        "MISTAKEN_TRANSFER_WRONG_DESTINATION",
        re.compile(
            r"(?:계좌(?:번호)?|받는\s*사람|수취인).{0,12}"
            r"(?:잘못|틀리|실수).{0,12}(?:보내|보냈|송금|이체)",
            re.I,
        ),
    ),
    (
        "MISTAKEN_TRANSFER_OTHER_DESTINATION",
        re.compile(
            r"(?:엉뚱한|다른|모르는)\s*(?:계좌|사람|수취인)"
            r"(?:에게|으로|로)?.{0,12}(?:보내|보냈|송금|이체)",
            re.I,
        ),
    ),
    (
        "MISTAKEN_TRANSFER_RECIPIENT",
        re.compile(
            r"(?:(?:모르는|잘못\s*온|뜻하지\s*않은)\s*(?:돈|금액|송금|입금)"
            r"|(?:돈|금액|송금|입금)(?:이|을)?\s*(?:잘못|모르게))"
            r".{0,10}(?:받|들어오|입금되|입금됐)",
            re.I,
        ),
    ),
)


def _mistaken_transfer_natural_evidence(text: str) -> list[str]:
    return [
        rule_id
        for rule_id, pattern in MISTAKEN_TRANSFER_NATURAL_PATTERNS
        if pattern.search(str(text or ""))
    ]

# NFKC 정규화는 단독 호환 자모(예: ㅋ, ㅎ, ㅇ)를 초성 자모로 바꾼다.
# 소셜 표현 비교 때만 다시 익숙한 호환 자모로 접어 두 형태를 동일하게 처리한다.
_SOCIAL_JAMO_TRANSLATION = str.maketrans({
    "ᄀ": "ㄱ", "ᄇ": "ㅂ", "ᄉ": "ㅅ", "ᄋ": "ㅇ", "ᄏ": "ㅋ", "ᄒ": "ㅎ",
    "ᅮ": "ㅜ", "ᅲ": "ㅠ",
})
_SOCIAL_JAMO_PLACEHOLDERS = str.maketrans({
    "ㄱ": "\ue000", "ㅂ": "\ue001", "ㅅ": "\ue002", "ㅇ": "\ue003",
    "ㅋ": "\ue004", "ㅎ": "\ue005", "ㅠ": "\ue006", "ㅜ": "\ue007",
})
_SOCIAL_JAMO_RESTORE = str.maketrans({
    "\ue000": "ㄱ", "\ue001": "ㅂ", "\ue002": "ㅅ", "\ue003": "ㅇ",
    "\ue004": "ㅋ", "\ue005": "ㅎ", "\ue006": "ㅠ", "\ue007": "ㅜ",
})


def _nfkc_preserving_social_jamo(text: Any) -> str:
    protected = str(text or "").translate(_SOCIAL_JAMO_PLACEHOLDERS)
    return unicodedata.normalize("NFKC", protected).translate(_SOCIAL_JAMO_RESTORE)


def _social_compact(text: str) -> str:
    folded = _nfkc_preserving_social_jamo(text).lower().translate(_SOCIAL_JAMO_TRANSLATION)
    return re.sub(r"[\W_]+", "", folded, flags=re.UNICODE)


_SOCIAL_ACTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("CAPABILITY", re.compile(
        r"(?:(?:너|넌|너는|이챗봇|챗봇|ai챗봇|봇)(?:은|는|이|가)?"
        r"(?:누구(?:야|예요|인가요)?|뭐야|무엇(?:인가요)?|이름(?:이)?뭐야|"
        r"뭐해|뭘할수있(?:어|어요|나요)|무슨질문(?:을)?할수있(?:어|어요|나요)))|"
        r"(?:무슨|어떤)질문(?:을)?(?:할수있(?:어|어요|나요)|하면돼(?:요)?)|"
        r"(?:지원|안내)(?:하|하는)(?:업무|내용)(?:를)?(?:알려(?:줘|주세요)|보여(?:줘|주세요))|"
        r"(?:이)?챗봇(?:은)?(?:어떻게)?사용(?:하|하면)(?:면되나요|나요|돼요)?|"
        r"사용방법|도움말|help"
    )),
    ("GREETING", re.compile(
        r"(?:안녕(?:하세요|하세용|하세여|하세욤|하십니까)?|안뇽(?:하세요|하세용)?|"
        r"방가(?:방가)*|반가워(?:요)?|반갑(?:습니다|네요)|하이(?:하이)*|헬로|"
        r"ㅎㅇ|ㅂㄱ|hi+|hello+|hey+)"
    )),
    ("THANKS", re.compile(
        r"(?:(?:(?:그거|이거|답변|안내|설명|도움)(?:은|는|이|가)?)?"
        r"(?:정말|너무|진짜)?(?:감사(?:합니다|해요|해|용|링|드립니다|드려요)?|고마워(?:요|용)?|"
        r"고맙(?:습니다|네요|다)?|땡큐|thank(?:s|you)|thx|ty|ㄱㅅ)|"
        r"도움이됐(?:어|어요|습니다))"
    )),
    ("CANCEL", re.compile(
        r"(?:그만(?:할게요)?|취소(?:할게요|해줘|해주세요)?|됐(?:어|어요|습니다)|"
        r"괜찮(?:아|아요|습니다)|필요없(?:어|어요|습니다)|필요없음)"
    )),
    ("ACKNOWLEDGEMENT", re.compile(
        r"(?:네+|넵|넹+|예+|응+|ㅇ+|오케이|오키(?:도키)?|okay|ok|ㅇㅋ|"
        r"알겠(?:어|어요|습니다)|잘알겠(?:어|어요|습니다)|"
        r"이해했(?:어|어요|습니다)|좋아(?:요)?|좋습니다|"
        r"확인했(?:어|어요|습니다))"
    )),
    ("CLOSING", re.compile(
        r"(?:안녕히(?:계세요|가세요)|잘가(?:요)?|다음에(?:봐|뵐게요?)?|"
        r"수고(?:했어|했어요|하셨어요|하세요|해요)?|ㅅㄱ|bye|goodbye|"
        r"끝|종료(?:할게요)?|이만)"
    )),
    ("REACTION", re.compile(
        r"(?:[ㅋㅎㅠㅜ]+|아하+|오+|와+|우와+|오호+|굿|좋네(?:요)?|"
        r"대박|그렇구나|그렇군요|멋져(?:요)?)"
    )),
)

_SOCIAL_EMOJI_ONLY_PATTERN = re.compile(r"^[👍🙏👏👋🙂😊😁😄😃😂🤣🥹😭🙌👌❤♥♡✨🎉🏻🏼🏽🏾🏿]+$")
_RETRIEVAL_PROTECT_PATTERN = re.compile(
    r"(?:예금|적금|원금|이자|보험|보험금|보험사고|예금자|보호|한도|부보|"
    r"금융|은행|저축은행|금융회사|금융상품|예금보험공사|예보|kdic|금융안심포털|파산|"
    r"배당|가지급|개산지급|미수령|수령|지급|대행점|송금|반환|수취인|"
    r"채무|부채|채권|재산|신고|포상금|계좌|상속|대리인|청구|압류|상계|"
    r"회생|면책|소멸시효|퇴직연금|cma|isa|irp|dc형)",
    re.I,
)
_INFORMATION_REQUEST_PATTERN = re.compile(
    r"(?:뭐|무엇|어디|언제|왜|어떻게|얼마|얼마나|누구|알려|설명|뜻|의미|정의|"
    r"차이|비교|방법|절차|신청|조회|확인|자격|대상|조건|한도|금액|기간|기한|"
    r"서류|비용|수수료|가능|되나요|인가요|예요|인가|문의)",
    re.I,
)
_LOW_INFORMATION_PATTERN = re.compile(
    r"(?:뭐|뭔데|음+|어+|아+|저기+|글쎄|모르겠어|아무거나|질문|테스트|test|"
    r"[0-9]+|[a-z]{1,12})",
    re.I,
)
_NOISE_ONLY_PATTERN = re.compile(r"^[\s!?.,~…·'\"`^_*+\-=()\[\]{}<>:/\\|]+$")


def direct_response_for_action(action: Any) -> str:
    """DIRECT/저정보 action에 대응하는 외부 호출 없는 고정 안내를 반환한다."""
    key = str(action or "META_OR_SOCIAL").strip().upper()
    return DIRECT_ACTION_RESPONSES.get(key, DIRECT_ACTION_RESPONSES["META_OR_SOCIAL"])


def classify_non_retrieval_utterance(query: str) -> dict[str, str] | None:
    """검색 전에 처리할 소셜 발화와 저정보 입력을 문장 전체 형태로 판정한다.

    업무·금융 용어 또는 실질적인 정보 요청이 섞인 문장은 이 함수가 가로채지
    않는다. 따라서 ``하이하이, 예금자보호 한도 얼마야?`` 같은 혼합 질문은
    기존 RETRIEVE 흐름을 그대로 탄다.
    """
    text = str(query or "").strip()
    if not text:
        return None
    compact = _social_compact(text)

    compact_candidates = [compact]
    without_trailing_reaction = re.sub(r"[ㅋㅎㅠㅜ]+$", "", compact)
    if without_trailing_reaction and without_trailing_reaction != compact:
        compact_candidates.append(without_trailing_reaction)
    for action, pattern in _SOCIAL_ACTION_PATTERNS:
        if any(candidate and pattern.fullmatch(candidate) for candidate in compact_candidates):
            return {
                "route": "DIRECT",
                "action": action,
                "reason": f"SOCIAL_{action}",
                "response": direct_response_for_action(action),
            }

    emoji_text = re.sub(r"[\s.!?~…\ufe0f]", "", text)
    if emoji_text and _SOCIAL_EMOJI_ONLY_PATTERN.fullmatch(emoji_text):
        return {
            "route": "DIRECT",
            "action": "REACTION",
            "reason": "SOCIAL_REACTION_EMOJI",
            "response": direct_response_for_action("REACTION"),
        }

    # 실제 KDIC 단문과 정보 요청은 짧아도 검색/기존 명확화 규칙에 맡긴다.
    if find_businesses(text) or _RETRIEVAL_PROTECT_PATTERN.search(text):
        return None
    if _INFORMATION_REQUEST_PATTERN.search(text):
        return {
            "route": "CLARIFY",
            "action": "LOW_INFORMATION",
            "reason": "NON_KDIC_INFORMATION_REQUEST_NO_RETRIEVAL",
            "response": direct_response_for_action("LOW_INFORMATION"),
        }

    low_information = bool(
        _NOISE_ONLY_PATTERN.fullmatch(text)
        or _LOW_INFORMATION_PATTERN.fullmatch(compact)
        or not compact
        or (compact and len(compact) <= 12)
    )
    if low_information:
        return {
            "route": "CLARIFY",
            "action": "LOW_INFORMATION",
            "reason": "LOW_INFORMATION_NO_RETRIEVAL",
            "response": direct_response_for_action("LOW_INFORMATION"),
        }
    return None

REFORMAT_PATTERN = re.compile(
    r"^(?:쉽게 설명해 주세요|핵심만 알려주세요|표로 정리해 주세요|더 자세히 설명해 주세요)[.!?]*$",
    re.I,
)

OOS_PATTERN = re.compile(
    r"(?:코스피|비트코인|주택담보대출\s*금리|대출금리|신용점수|실손보험|국민연금|"
    r"보이스피싱.*(?:경찰|신고)|은행\s*계좌를\s*새로|계좌\s*개설|해외송금\s*수수료|"
    r"카드\s*결제.*환불|전세대출|세금\s*환급|퇴직금|개인정보\s*유출|상속세|"
    r"환율|환전소|주식\s*(?:투자|포트폴리오)|신용카드\s*연회비|사업자등록|"
    r"(?:서울|오늘|내일|이번\s*주말)?.{0,8}(?:날씨|기온|미세먼지))",
    re.I,
)

GENERIC_BUSINESS_CLARIFY_PATTERN = re.compile(
    r"^(?:신청\s*(?:방법|자격|기한|과정|대상)|접수\s*(?:방법|절차)|제출해야\s*하는\s*서류|"
    r"필요한\s*서류|조회는\s*어디에서|온라인으로\s*신청|방문해서\s*접수|접수\s*후\s*처리\s*기간|"
    r"신청\s*자격과\s*제외\s*조건|신청\s*과정에서\s*수수료나\s*비용|"
    r"처리\s*결과는\s*어디에서\s*확인|이미\s*접수한\s*신청을\s*취소|"
    r"문의하거나\s*접수하려면\s*어느\s*기관|제가\s*신청\s*대상에\s*해당|"
    r"본인\s*대신\s*대리인이\s*신청|상속인이\s*신청하거나\s*받을\s*수|"
    r"처리\s*기간|문의처|신청\s*비용).*$",
    re.I,
)

TARGET_DEMONSTRATIVE_PATTERN = re.compile(
    r"(?:^|[\s,.(])(?:제가\s*(?:말한|가입한|가진|본)\s*)?"
    r"이\s*(?:금융상품|계좌|상품|돈|송금|거래|금액|채무|재산)"
    r"(?:을|를|이|가|도|은|는|의|이나|과|와|\s|[,.!?]|$)",
    re.I,
)

TARGET_REFERENCE_PHRASES = (
    "제가 가입한 상품", "어떤 송금 건", "어떤 예금에 대해", "어떤 돈을 신청",
    "어떤 예금이나 금융상품",
)

APPLICANT_REFERENCE_PHRASES = (
    "제 신청 유형", "제 신청 자격", "제 경우", "누구를 신청인", "누가 방문",
    "신고 주체 유형", "신청인란",
)

CASE_REFERENCE_PHRASES = (
    "제 상황", "현재 상황", "제 채무 상태", "제 신고 상황", "반려", "거절",
    "보완 요청", "진행되지 않", "여러 금융회사에 예금", "여러 계좌에 나뉘",
)

HIGH_PRECISION_INTENT_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("DOCUMENTS", (
        r"필요한?\s*(?:서류|증빙)", r"제출(?:해야\s*하는|할)?\s*서류", r"구비\s*서류",
        r"신분증", r"위임장", r"준비(?:해야\s*할|할)?\s*서류", r"무엇을\s*(?:더\s*)?준비",
    )),
    ("STATUS", (
        r"어디(?:서|에서)\s*(?:확인|조회|검색)", r"조회\s*(?:방법|결과)", r"처리\s*결과",
        r"진행\s*(?:상황|상태)", r"지급\s*정보.*보는\s*방법", r"있는지.*조회",
    )),
    ("APPLICATION", (
        r"신청\s*(?:방법|절차)", r"접수\s*(?:방법|절차)", r"제출\s*방법", r"신고\s*채널",
        r"어떻게\s*(?:신청|접수|청구)", r"(?:온라인|방문|직접).*신청.*(?:가능|할\s*수)",
        r"취소.*방법", r"철회.*방법", r"무엇을\s*해야", r"어디에\s*접수",
    )),
    ("TIME", (
        r"언제(?:부터|까지)", r"신청.*(?:기한|기간|시점)", r"처리\s*기간", r"소요\s*(?:기간|시간)",
        r"얼마나\s*걸", r"언제\s*(?:지급|찾)",
    )),
    ("AMOUNT", (
        r"보호\s*한도", r"지급\s*금액", r"금액\s*계산", r"금액.*얼마(?:여야|이어야)",
        r"계산\s*(?:기준|방법)", r"수수료\s*(?:금액|비용)", r"얼마나\s*(?:감면|지급|보상|돌려|받|보호)",
        r"비용\s*차감", r"최종\s*보호금액",
    )),
    ("CONTACT", (
        r"연락처", r"전화번호", r"문의처", r"어디로\s*연락", r"어느\s*기관.*문의",
    )),
    ("ELIGIBILITY", (
        r"신청\s*(?:대상|자격|요건)", r"가능한\s*대상", r"제외되는?\s*경우", r"받을\s*수\s*있",
        r"신청할\s*수\s*있", r"포함되", r"어떤\s*경우.*(?:지급|지원|보호)",
        r"(?:대상|자격)에\s*해당", r"(?:예금|계좌|금융상품|상품|원금|이자|채권).{0,30}보호(?:가)?\s*되",
        r"지원\s*대상", r"누가\s*(?:신청|수령)", r"보호\s*대상",
    )),
    ("OVERVIEW", (
        r"무엇(?:인가요|인지|이며)", r"뭐예요", r"의미", r"정의", r"차이", r"종류", r"개요",
        r"설명", r"관계", r"왜\s*(?:발생|제외)", r"어떤\s*성격",
    )),
)

WEAK_INTENT_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("DOCUMENTS", (r"서류", r"증빙", r"준비")),
    ("STATUS", (r"조회", r"확인", r"검색")),
    ("APPLICATION", (r"신청", r"접수", r"절차", r"제출", r"청구", r"신고")),
    ("TIME", (r"기간", r"기한", r"시점", r"언제")),
    ("AMOUNT", (r"한도", r"금액", r"계산", r"비용", r"포상금")),
    ("CONTACT", (r"연락", r"문의", r"전화")),
    ("ELIGIBILITY", (r"대상", r"자격", r"요건", r"조건", r"가능", r"보호")),
    ("OVERVIEW", (r"설명", r"관계", r"방식", r"종류", r"의미")),
)

# 두 정보가 서로 밀접한 하나의 검색 문서에서 함께 해결될 가능성이 높은 결합입니다.
# 이런 결합은 요구가 두 개여도 원문을 유지합니다.
COHESIVE_NO_SPLIT_PATTERNS = (
    re.compile(r"(?:대상|자격|조건).{0,28}(?:서류|준비)|(?:서류|준비).{0,28}(?:제출|신청|접수)\s*방법", re.I),
    re.compile(r"(?:누가|대리인|상속인|본인|법인).{0,35}(?:서류|준비)", re.I),
    re.compile(r"(?:한도|금액).{0,30}(?:포함|계산|합산|상계)|(?:포함|계산|합산|상계).{0,30}(?:한도|금액)", re.I),
    re.compile(r"(?:사유|이유).{0,25}(?:조건|해제)|(?:의미|무엇).{0,25}(?:차이|관계|종류)", re.I),
    re.compile(r"(?:언제까지|기한|기간).{0,25}(?:어디에서|어디에)\s*(?:신청|확인)", re.I),
    re.compile(r"(?:조회|확인)한?\s*(?:뒤|다음).{0,35}(?:신청|지급)", re.I),
    re.compile(r"(?:퇴직연금).{0,45}(?:연금저축).{0,45}(?:보호|한도)", re.I),
)

# 서로 다른 사전 용어가 잡혀도 목적이 관계·차이 또는 결합 가능성 확인이면 원문을 유지합니다.
CROSS_TERM_RELATION_KEEP_PATTERN = re.compile(
    r"(?:와|과|및).{0,38}(?:관계|차이)|(?:관계|차이).{0,38}(?:와|과|및)|"
    r"(?:와|과|및).{0,38}(?:함께|동시에)\s*(?:조회|확인|신청|보호).*(?:가능|할\s*수)",
    re.I,
)

# 독립된 대상·상황·처리 단계가 명시된 경우에만 같은 업무 안에서도 분해합니다.
STRONG_SAME_BUSINESS_SPLIT_PATTERNS = (
    re.compile(r"(?:때|경우)와.{0,55}(?:때|경우)", re.I),
    re.compile(r"(?:외화예금|간편송금|온라인\s*신청).{0,45}(?:후순위채권|해외\s*계좌|방문\s*신청)", re.I),
    re.compile(r"(?:1종\s*보험사고).{0,45}(?:2종\s*보험사고)", re.I),
    re.compile(r"(?:영업정지).{0,35}(?:기존\s*대출|대출\s*거래)", re.I),
    re.compile(r"(?:미리\s*신청|신청\s*전).{0,45}(?:실제\s*보험사고|접수\s*후)", re.I),
    re.compile(r"(?:지급되는\s*조건|지급\s*조건).{0,35}(?:실제\s*)?신청\s*절차", re.I),
    re.compile(r"(?:온라인\s*신청).{0,45}(?:지급대행점\s*)?방문\s*신청", re.I),
    re.compile(r"(?:신청(?:하기)?\s*전|신청\s*전에).{0,45}(?:접수|신청)\s*후", re.I),
    re.compile(r"(?:접수|신청).{0,25}(?:결과|진행\s*절차).{0,25}(?:확인|진행)", re.I),
    re.compile(r"(?:파산\s*금융회사).{0,40}(?:남은\s*)?미수령금.*신청", re.I),
    re.compile(r"(?:기간|얼마나\s*걸).{0,40}(?:비용\s*차감|차감\s*방식)", re.I),
    re.compile(r"(?:금융회사).{0,30}보호\s*대상.{0,30}(?:금융상품).{0,20}보호", re.I),
    re.compile(r"(?:미성년자).{0,35}보호되.{0,35}(?:누가|수령)", re.I),
    re.compile(r"(?:상속인).{0,30}(?:조회).{0,20}(?:뒤|다음).{0,30}(?:지급을\s*)?신청", re.I),
    re.compile(r"(?:제외되는\s*경우).{0,35}(?:제외되는\s*이유|왜\s*제외)", re.I),
)

CLAUSE_BOUNDARY_PATTERN = re.compile(
    r"\s*(?:[.!?;]+|,?\s*(?:그리고|또|혹시|별도로|그와\s*별개로|반면에|반면|뿐만\s*아니라)\s+)\s*",
    re.I,
)

CONJUNCTION_BOUNDARY_PATTERN = re.compile(
    r"\s*(?:,\s*|\s+)(?:그리고|또|혹시|별도로|반면에|반면)\s*",
    re.I,
)

NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d+)*(?:\s*(?:원|만원|억원|개월|년|일|%))?")
NEGATION_TERMS = ("아니", "못", "제외", "불가", "없", "않", "전혀")


@dataclass(frozen=True)
class RouterConfig:
    max_subqueries: int = 4
    include_original_anchor_for_multi: bool = True
    allow_hard_filter: bool = False
    min_subquery_chars: int = 5


def normalize_query(text: Any) -> dict[str, Any]:
    original = str(text or "")
    value = _nfkc_preserving_social_jamo(original)
    changes: list[str] = []
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", value)
    if cleaned != value:
        changes.append("CONTROL_CHARACTER")
        value = cleaned
    cleaned = re.sub(r"([!?ㅋㅎㅠㅜ])\1{2,}", r"\1\1", value)
    if cleaned != value:
        changes.append("REPEATED_CHARACTER")
        value = cleaned
    for wrong, correct in TYPO_MAP:
        if wrong in value:
            value = value.replace(wrong, correct)
            changes.append("EXPLICIT_TYPO")
    cleaned = re.sub(r"\s+", " ", value).strip()
    if cleaned != value:
        changes.append("WHITESPACE")
    if not cleaned:
        raise ValueError("사용자 질의가 비어 있습니다.")
    return {"original_query": original, "normalized_query": cleaned, "changes": list(dict.fromkeys(changes))}


def _ordered_unique(values: Iterable[Any]) -> list[Any]:
    output: list[Any] = []
    seen: set[Any] = set()
    for value in values:
        if value is None or value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def find_business_matches(text: str) -> list[dict[str, Any]]:
    compact = _compact(text)
    found: list[dict[str, Any]] = []
    for business, keywords in BUSINESS_KEYWORDS.items():
        evidence = [keyword for keyword in keywords if _compact(keyword) in compact]
        deposit_insurance_system_match = (
            _DEPOSIT_INSURANCE_SYSTEM_PATTERN.search(text)
            if business == "예금자보호제도"
            else None
        )
        if deposit_insurance_system_match:
            evidence.append(deposit_insurance_system_match.group(0))
        natural_evidence = (
            _mistaken_transfer_natural_evidence(text)
            if business == "착오송금 반환 신청"
            else []
        )
        evidence.extend(natural_evidence)
        if not evidence:
            continue
        strong = [term for term in STRONG_BUSINESS_KEYWORDS[business] if _compact(term) in compact]
        if deposit_insurance_system_match:
            strong.append(deposit_insurance_system_match.group(0))
        strong.extend(natural_evidence)
        found.append({
            "business_function": business,
            "evidence": _ordered_unique(evidence),
            "strong_evidence": _ordered_unique(strong),
            "confidence": 0.99 if strong else 0.80,
        })
    return found


def find_businesses(text: str) -> list[str]:
    return [row["business_function"] for row in find_business_matches(text)]


def find_intent_matches(text: str) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for source, rules in (("HIGH_PRECISION_RULE", HIGH_PRECISION_INTENT_RULES), ("WEAK_RULE", WEAK_INTENT_RULES)):
        for intent, patterns in rules:
            hit = next((m for pattern in patterns if (m := re.search(pattern, text, flags=re.I))), None)
            if hit and intent not in {row["intent"] for row in matches}:
                matches.append({
                    "intent": intent,
                    "source": source,
                    "evidence": hit.group(0),
                    "start": hit.start(),
                    "end": hit.end(),
                })
    natural_transfer = _mistaken_transfer_natural_evidence(text)
    if (
        natural_transfer
        and "APPLICATION" not in {row["intent"] for row in matches}
        and re.search(r"(?:어떻게|어쩌|해야\s*(?:해|돼|하)|돌려\s*받)", text, flags=re.I)
    ):
        matches.append({
            "intent": "APPLICATION",
            "source": "NATURAL_BUSINESS_PATTERN",
            "evidence": natural_transfer[0],
            "start": 0,
            "end": len(text),
        })
    return matches


def _parse_previous_turns(value: Any) -> list[dict[str, str]]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == "nan":
            return []
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return [{"user": text, "assistant": ""}]
    if not isinstance(value, list):
        return []
    output = []
    for row in value[-3:]:
        if isinstance(row, Mapping):
            user = str(row.get("user") or row.get("question") or "").strip()
            assistant = str(row.get("assistant") or row.get("answer") or "").strip()
            if user or assistant:
                output.append({"user": user, "assistant": assistant})
        elif str(row).strip():
            output.append({"user": str(row).strip(), "assistant": ""})
    return output


def build_context(previous_turns: Any = None, conversation_state: Mapping[str, Any] | None = None) -> dict[str, Any]:
    state = dict(conversation_state or {})
    turns = _parse_previous_turns(previous_turns if previous_turns is not None else state.get("recent_turns"))
    confirmed = dict(state.get("confirmed") or {}) if isinstance(state.get("confirmed"), Mapping) else {}
    context_text = " ".join(row["user"] for row in turns if row.get("user"))
    context_businesses = find_businesses(context_text)
    if len(context_businesses) == 1 and not confirmed.get("business_function"):
        confirmed["business_function"] = context_businesses[0]
    return {
        "used": bool(turns or confirmed),
        "recent_turns": turns,
        "confirmed": confirmed,
        "context_businesses": context_businesses,
    }


def detect_route(
    query: str,
    *,
    context: Mapping[str, Any],
) -> tuple[str, list[str], list[str], str | None]:
    """보수적 라우팅. 반환값은 route, reasons, missing, direct_action 순서입니다."""
    businesses = find_businesses(query)
    context_business = str((context.get("confirmed") or {}).get("business_function") or "")
    has_context = bool(context.get("used"))

    non_retrieval = classify_non_retrieval_utterance(query)
    if non_retrieval and non_retrieval["route"] == "DIRECT":
        return (
            "DIRECT",
            [non_retrieval["reason"]],
            [],
            non_retrieval["action"],
        )

    if DIRECT_META_PATTERN.fullmatch(query):
        return "DIRECT", ["EXPLICIT_META_OR_SOCIAL"], [], "META_OR_SOCIAL"
    if REFORMAT_PATTERN.fullmatch(query) and has_context:
        return "DIRECT", ["REFORMAT_PREVIOUS_ANSWER"], [], "REFORMAT_PREVIOUS_ANSWER"
    if OOS_PATTERN.search(query) and not businesses:
        return "OUT_OF_SCOPE", ["EXPLICIT_NON_KDIC_TOPIC"], [], None

    if GENERIC_BUSINESS_CLARIFY_PATTERN.fullmatch(query) and not businesses and not context_business:
        return "CLARIFY", ["BUSINESS_NOT_SPECIFIED"], ["business_function"], None

    if non_retrieval and non_retrieval["route"] == "CLARIFY":
        return (
            "CLARIFY",
            [non_retrieval["reason"]],
            ["question_topic"],
            non_retrieval["action"],
        )

    has_resolved_context = has_context or bool(context_business)
    if any(phrase in query for phrase in APPLICANT_REFERENCE_PHRASES) and not has_resolved_context:
        return "CLARIFY", ["UNRESOLVED_APPLICANT_REFERENCE"], ["applicant_type"], None
    if (any(phrase in query for phrase in TARGET_REFERENCE_PHRASES) or TARGET_DEMONSTRATIVE_PATTERN.search(query)) and not has_resolved_context:
        return "CLARIFY", ["UNRESOLVED_TARGET_REFERENCE"], ["target_type"], None
    if any(phrase in query for phrase in CASE_REFERENCE_PHRASES) and not has_resolved_context:
        return "CLARIFY", ["PERSONAL_CASE_REQUIRES_DETAILS"], ["case_details"], None

    return "RETRIEVE", ["DEFAULT_FAIL_OPEN_RETRIEVAL"], [], None


def _is_cohesive_no_split(query: str) -> bool:
    return any(pattern.search(query) for pattern in COHESIVE_NO_SPLIT_PATTERNS)


def _has_strong_same_business_split_signal(query: str) -> tuple[bool, str | None]:
    for index, pattern in enumerate(STRONG_SAME_BUSINESS_SPLIT_PATTERNS, 1):
        if pattern.search(query):
            return True, f"SAME_BUSINESS_STRONG_PATTERN_{index:02d}"
    return False, None


def detect_complexity(query: str) -> dict[str, Any]:
    businesses = find_businesses(query)
    intents = find_intent_matches(query)
    strong_same, strong_rule = _has_strong_same_business_split_signal(query)
    sentence_clauses = [part.strip(" ,") for part in CLAUSE_BOUNDARY_PATTERN.split(query) if part.strip(" ,")]

    reasons: list[str] = []
    cross_relation_keep = bool(CROSS_TERM_RELATION_KEEP_PATTERN.search(query))
    if len(businesses) >= 2 and not cross_relation_keep:
        reasons.append("MULTIPLE_BUSINESS_FUNCTIONS")
    elif len(businesses) >= 2 and cross_relation_keep:
        reasons.append("CROSS_TERM_RELATION_KEEP_ORIGINAL")
    if strong_same:
        reasons.append(strong_rule or "STRONG_SAME_BUSINESS_SIGNAL")
    if len(sentence_clauses) >= 2:
        clause_businesses = [find_businesses(part) for part in sentence_clauses]
        if sum(bool(values) for values in clause_businesses) >= 2:
            reasons.append("INDEPENDENT_BUSINESS_CLAUSES")
        elif len(intents) >= 2 and not _is_cohesive_no_split(query):
            reasons.append("INDEPENDENT_INTENT_CLAUSES")

    is_multi = (len(businesses) >= 2 and not cross_relation_keep) or strong_same or "INDEPENDENT_INTENT_CLAUSES" in reasons
    if _is_cohesive_no_split(query) and len(businesses) <= 1 and not strong_same:
        is_multi = False
        reasons.append("COHESIVE_SAME_BUSINESS_KEEP_ORIGINAL")
    if not is_multi:
        reasons.append("NO_SAFE_SPLIT_EVIDENCE")

    return {
        "question_type": "MULTI" if is_multi else "SINGLE",
        "businesses": businesses,
        "intents": [row["intent"] for row in intents],
        "clause_count": len(sentence_clauses),
        "reasons": _ordered_unique(reasons),
    }


def _clean_clause(text: str) -> str:
    text = re.sub(r"^(?:그리고|또|혹시|별도로|그럼|그러면)\s*", "", text.strip(), flags=re.I)
    text = re.sub(r"\s+", " ", text).strip(" ,.;")
    if text and not re.search(r"[?요다까]$", text):
        text += " 관련 정보"
    return text


def _sentence_clauses(query: str) -> list[str]:
    return [_clean_clause(part) for part in CLAUSE_BOUNDARY_PATTERN.split(query) if _clean_clause(part)]


def _business_anchor_positions(query: str) -> list[tuple[int, int, str, str]]:
    positions: list[tuple[int, int, str, str]] = []
    compact_query = query.lower()
    for business, keywords in BUSINESS_KEYWORDS.items():
        for keyword in sorted(keywords, key=len, reverse=True):
            start = compact_query.find(keyword.lower())
            if start >= 0:
                positions.append((start, start + len(keyword), business, keyword))
                break
    positions.sort(key=lambda row: row[0])
    return positions


def _split_cross_business(query: str, businesses: Sequence[str]) -> list[str]:
    clauses = _sentence_clauses(query)
    if len(clauses) >= 2:
        enriched: list[str] = []
        for clause in clauses:
            local_businesses = find_businesses(clause)
            if local_businesses:
                enriched.append(clause)
        if len(enriched) >= 2:
            return enriched

    anchors = _business_anchor_positions(query)
    if len(anchors) < 2:
        return []
    output: list[str] = []
    for index, (start, _end, business, keyword) in enumerate(anchors):
        next_start = anchors[index + 1][0] if index + 1 < len(anchors) else len(query)
        previous_end = anchors[index - 1][1] if index > 0 else 0
        raw = query[previous_end:next_start]
        raw = re.sub(r"^(?:와|과|및|하고|,|\s)+", "", raw)
        raw = re.sub(r"(?:와|과|및|하고|,|\s)+$", "", raw)
        clause = _clean_clause(raw)
        if business not in find_businesses(clause):
            clause = f"{business} {clause}".strip()
        # 너무 짧거나 명사구뿐이면 전체 문장의 해당 업무 관련 의도 단서를 붙입니다.
        local_intents = find_intent_matches(clause)
        if not local_intents:
            global_intents = find_intent_matches(query)
            if global_intents:
                clause = f"{clause} {global_intents[min(index, len(global_intents)-1)]['evidence']}"
        output.append(_clean_clause(clause))
    return output


def _split_same_business(query: str, business: str | None) -> list[str]:
    # 처리 전후나 서로 다른 처리 대상을 한 문장에 묶은 대표 구조는 의미 단위로 직접 분리합니다.
    match = re.search(
        r"^(?P<context>.*?영업정지되면)\s*(?P<first>예금은.*?)(?:고|며)\s*(?P<second>기존\s*대출\s*거래.*?)(?:[?]?)$",
        query,
        flags=re.I,
    )
    if match:
        context = match.group("context").strip()
        return [
            _clean_clause(f"{context} {match.group('first')}"),
            _clean_clause(f"{context} {match.group('second')}"),
        ]

    match = re.search(
        r"^(?P<actor>상속인이\s*고인의)\s*(?P<target>미수령금)을?\s*조회한?\s*(?:뒤|다음)\s*"
        r"(?P<action>지급을\s*신청하는\s*방법).*?$",
        query,
        flags=re.I,
    )
    if match:
        prefix = f"{match.group('actor')} {match.group('target')}"
        return [
            _clean_clause(f"{prefix} 조회 방법"),
            _clean_clause(f"{prefix} {match.group('action')}"),
        ]

    clauses = _sentence_clauses(query)
    if len(clauses) >= 2:
        output = []
        for clause in clauses:
            if business and business not in find_businesses(clause):
                clause = f"{business} {clause}"
            output.append(_clean_clause(clause))
        return output

    # 쉼표·연결 표현을 먼저 이용합니다.
    candidates = [part.strip(" ,") for part in re.split(r"\s*(?:,|이고|이며|인지,?|는지와|과|와)\s*", query) if part.strip(" ,")]
    if len(candidates) >= 2:
        candidates = candidates[:3]
        output = []
        for clause in candidates:
            if len(clause) < 4:
                continue
            if business and business not in find_businesses(clause):
                clause = f"{business} {clause}"
            output.append(_clean_clause(clause))
        if len(output) >= 2:
            return output

    # 안전한 절단점을 못 찾으면 분해 실패로 두고 원문 fallback을 사용합니다.
    return []


def validate_decomposition(original: str, subqueries: Sequence[str], expected_businesses: Sequence[str]) -> dict[str, Any]:
    queries = [_clean_clause(str(value)) for value in subqueries if _clean_clause(str(value))]
    issues: list[str] = []
    if len(queries) < 2:
        issues.append("TOO_FEW_SUBQUERIES")
    if len(set(_compact(value) for value in queries)) != len(queries):
        issues.append("DUPLICATE_SUBQUERIES")
    if any(len(value) < 5 for value in queries):
        issues.append("SUBQUERY_TOO_SHORT")

    reconstructed = " ".join(queries)
    missing_businesses = [business for business in expected_businesses if business not in find_businesses(reconstructed)]
    if missing_businesses:
        issues.append("MISSING_BUSINESS_COVERAGE")

    original_numbers = NUMBER_PATTERN.findall(original)
    missing_numbers = [value for value in original_numbers if value not in reconstructed]
    if missing_numbers:
        issues.append("MISSING_NUMERIC_CONSTRAINT")

    original_negations = [term for term in NEGATION_TERMS if term in original]
    missing_negations = [term for term in original_negations if term not in reconstructed]
    if missing_negations:
        issues.append("MISSING_NEGATION")

    status = "COMPLETE" if not issues else ("PARTIAL" if len(queries) >= 2 else "FAILED")
    return {
        "status": status,
        "issues": issues,
        "subqueries": queries,
        "missing_businesses": missing_businesses,
        "missing_numbers": missing_numbers,
        "missing_negations": missing_negations,
    }


def decompose_query(query: str, complexity: Mapping[str, Any], config: RouterConfig) -> dict[str, Any]:
    businesses = list(complexity.get("businesses") or [])
    if complexity.get("question_type") != "MULTI":
        return {"status": "NOT_REQUIRED", "subqueries": [query], "issues": [], "fallback_to_original": False}

    if len(businesses) >= 2:
        candidates = _split_cross_business(query, businesses)
    else:
        candidates = _split_same_business(query, businesses[0] if businesses else None)
    candidates = _ordered_unique(candidates)[: config.max_subqueries]
    validation = validate_decomposition(query, candidates, businesses)
    validation["fallback_to_original"] = validation["status"] != "COMPLETE"
    return validation


def _business_filter_for_query(query: str) -> dict[str, Any]:
    matches = find_business_matches(query)
    if len(matches) == 1:
        match = matches[0]
        return {
            "mode": "SOFT",
            "value": None,
            "soft_hint": match["business_function"],
            "confidence": match["confidence"],
            "evidence": match["evidence"],
            "hard_filter_eligible": False,
            "hard_filter_denial_reasons": ["HARD_DISABLED_BY_ROUTER_POLICY"],
        }
    return {
        "mode": "NONE",
        "value": None,
        "soft_hint": None,
        "confidence": 0.0,
        "evidence": [],
        "hard_filter_eligible": False,
        "hard_filter_denial_reasons": [
            "HARD_DISABLED_BY_ROUTER_POLICY",
            "MULTIPLE_OR_UNKNOWN_BUSINESS_CANDIDATES",
        ],
    }


def _make_need(need_id: str, query: str, *, source: str) -> dict[str, Any]:
    business_matches = find_business_matches(query)
    intent_matches = find_intent_matches(query)
    return {
        "need_id": need_id,
        "query": query,
        "query_source": source,
        "business_function": business_matches[0]["business_function"] if len(business_matches) == 1 else None,
        "business_candidates": business_matches,
        "intents": [row["intent"] for row in intent_matches],
        "intent_evidence": intent_matches,
    }


def build_query_plans(
    original_query: str,
    route: str,
    complexity: Mapping[str, Any],
    decomposition: Mapping[str, Any],
    config: RouterConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if route != "RETRIEVE":
        return [], []

    is_multi = complexity.get("question_type") == "MULTI"
    decomposition_complete = decomposition.get("status") == "COMPLETE"
    if is_multi and decomposition_complete:
        retrieval_queries = list(decomposition.get("subqueries") or [])
        source = "CONSERVATIVE_DECOMPOSITION"
    else:
        retrieval_queries = [original_query]
        source = "ORIGINAL_PASSTHROUGH" if not is_multi else "ORIGINAL_FALLBACK"

    needs = [_make_need(f"N{index}", query, source=source) for index, query in enumerate(retrieval_queries, 1)]
    plans: list[dict[str, Any]] = []
    for need in needs:
        query = need["query"]
        plans.append({
            "need_id": need["need_id"],
            "retrieval_mode": "STANDARD",
            "semantic_query": query,
            "keyword_query": query,
            "query_source": need["query_source"],
            "business_filter": _business_filter_for_query(query),
            "fallback_policy": {
                "enabled": True,
                "on": ["NO_RESULTS", "LOW_TOP_SCORE", "LOW_COVERAGE"],
                "next_filter_modes": ["NONE"],
                "fail_open": True,
                "original_anchor_query": original_query if is_multi and config.include_original_anchor_for_multi else None,
            },
            "intent_boost": {
                "mode": "SOFT" if need["intents"] else "NONE",
                "values": need["intents"],
                "weight": 0.10 if need["intents"] else 0.0,
            },
        })
    return needs, plans


def query_plan_is_valid(result: Mapping[str, Any]) -> bool:
    route = str((result.get("analysis") or {}).get("route") or "")
    plans = result.get("query_plans") or []
    if route == "RETRIEVE":
        if not plans:
            return False
        for plan in plans:
            if not str(plan.get("semantic_query") or "").strip():
                return False
            if not str(plan.get("keyword_query") or "").strip():
                return False
            if (plan.get("business_filter") or {}).get("mode") == "HARD":
                return False
    elif plans:
        return False
    return True


class KDICLightweightRouterV1:
    def __init__(self, config: RouterConfig | None = None):
        self.config = config or RouterConfig()

    def run(
        self,
        query: str,
        *,
        previous_turns: Any = None,
        conversation_state: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        normalized = normalize_query(query)
        original = normalized["original_query"]
        normalized_text = normalized["normalized_query"]
        context = build_context(previous_turns, conversation_state)
        route, route_reasons, missing, direct_action = detect_route(normalized_text, context=context)

        if route == "RETRIEVE":
            complexity = detect_complexity(normalized_text)
            decomposition = decompose_query(normalized_text, complexity, self.config)
        else:
            complexity = {
                "question_type": "NONE",
                "businesses": [],
                "intents": [],
                "clause_count": 0,
                "reasons": ["NO_RETRIEVAL_ROUTE"],
            }
            decomposition = {
                "status": "NOT_APPLICABLE",
                "subqueries": [],
                "issues": [],
                "fallback_to_original": False,
            }

        needs, plans = build_query_plans(original, route, complexity, decomposition, self.config)
        analysis = {
            "route": route,
            "question_type": complexity["question_type"],
            "business_functions": complexity["businesses"],
            "intents": complexity["intents"],
            "needs": needs,
            "missing_information": missing,
            "decomposition_status": decomposition["status"],
        }
        result = {
            "pipeline_version": PIPELINE_VERSION,
            "analysis_status": "OK",
            "original_query": original,
            "normalized_query": normalized_text,
            "normalization_changes": normalized["changes"],
            "context": context,
            "route_reasons": route_reasons,
            "direct_action": direct_action,
            "complexity": complexity,
            "decomposition": decomposition,
            "analysis": analysis,
            "query_plans": plans,
            "validation_warnings": [],
            "runtime": {
                "api_request_count": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            },
        }
        if not query_plan_is_valid(result):
            result["analysis_status"] = "INVALID_PLAN"
            result["validation_warnings"].append("QUERY_PLAN_VALIDATION_FAILED")
        return result


def route_query(
    query: str,
    *,
    previous_turns: Any = None,
    conversation_state: Mapping[str, Any] | None = None,
    config: RouterConfig | None = None,
) -> dict[str, Any]:
    return KDICLightweightRouterV1(config).run(
        query,
        previous_turns=previous_turns,
        conversation_state=conversation_state,
    )


if __name__ == "__main__":
    examples = (
        "예금자보호 한도는 얼마인가요?",
        "예금자보호 한도는 얼마인가요? 그리고 착오송금 반환지원은 누가 신청할 수 있나요?",
        "신청 방법을 알려주세요.",
        "안녕하세요.",
    )
    router = KDICLightweightRouterV1()
    for example in examples:
        print(json.dumps(router.run(example), ensure_ascii=False, indent=2))
