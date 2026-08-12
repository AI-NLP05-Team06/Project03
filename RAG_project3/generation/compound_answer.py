# [파이프라인 완성] 답변생성 이전 전체 흐름의 단일 진입점:
# 구어체 정규화(2번) -> 맥락반영 재작성(4번, 이전 턴 있을 때만) ->
# 복합질의 판별+분리(3번) -> 하위질문마다 업무+의도 분류(5번) ->
# 검색(6번)+답변생성(정보/민원처리 템플릿 분기) -> 하위답변 병합.
# 단일 질의면 그냥 하나만 답변하고 병합 단계는 건너뛴다.
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from classification.context_rewrite import rewrite_with_context
from classification.decomposition import decompose_query
from classification.normalization import normalize_colloquial
from classification.pipeline import classify
from generation.answer_generation import generate_grounded_hcx_answer
from generation.complaint_template import generate_complaint_template_answer
from retrieval.reranker import hybrid_rerank_search

_DIRECT_RESPONSE_TEXT = "안녕하세요! 예금보험공사 챗봇입니다. 예금자보호, 예금보험금, 미수령금, 착오송금, 채무조정, 은닉재산 신고 관련 문의를 도와드릴 수 있어요."
_CLARIFY_TEXT = "질문하신 내용이 어떤 업무에 대한 것인지 조금 더 구체적으로 말씀해 주시겠어요?"
_OUT_OF_SCOPE_TEXT = "죄송합니다, 해당 문의는 예금보험공사 업무 범위 밖이라 답변드리기 어렵습니다."


def answer_query(question: str, *, previous_question: str | None = None) -> str:
    normalized = normalize_colloquial(question)

    if previous_question:
        normalized = rewrite_with_context(normalized, previous_question)

    sub_questions = decompose_query(normalized)
    if len(sub_questions) == 1:
        return _answer_single(sub_questions[0])

    # Type A/B 하위질문은 서로 독립적이라 병렬로 처리한다 — 순차 처리 시
    # 하위질문 수만큼 지연이 배로 늘어나는데(실측: 2개=114초), 병렬화하면
    # 가장 느린 하나 수준으로 유지된다(설계 초기부터 계획했으나 누락돼있었음).
    with ThreadPoolExecutor(max_workers=len(sub_questions)) as executor:
        sub_answers = list(executor.map(_answer_single, sub_questions))
    return _merge_answers(sub_questions, sub_answers)


def _answer_single(question: str) -> str:
    result = classify(question)
    route = result["route"]

    if route == "DIRECT_RESPONSE":
        return _DIRECT_RESPONSE_TEXT
    if route == "OUT_OF_SCOPE":
        return _OUT_OF_SCOPE_TEXT
    if route == "CLARIFY":
        return _CLARIFY_TEXT

    search_results = hybrid_rerank_search(question, top_k=5, business_function=None)
    if result["intent"] == "민원처리":
        return generate_complaint_template_answer(question, search_results)
    return generate_grounded_hcx_answer(question, search_results)


def _merge_answers(sub_questions: list[str], sub_answers: list[str]) -> str:
    """각 하위답변이 이미 자체적으로 근거·출처를 포함하고 있으므로, 별도
    LLM 병합 호출 없이 하위질문 제목 아래 그대로 이어붙인다(불필요한
    재작성으로 할루시네이션 위험을 만들지 않기 위함)."""
    blocks = [
        f"[{i}] {question}\n{answer}"
        for i, (question, answer) in enumerate(zip(sub_questions, sub_answers), start=1)
    ]
    return "\n\n".join(blocks)
