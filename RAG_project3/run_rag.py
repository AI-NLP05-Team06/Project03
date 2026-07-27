# [8/10] 질문 1건을 받아 검색→답변 생성까지 실행하고 결과를 JSON으로 저장하는 run_kdic_rag()를 정의합니다.
from config import *
from integrity_check import *
from search_answer import *


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_search_result_rows(
    search_results: list[dict],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for rank, result in enumerate(search_results, start=1):
        chunk = result["chunk"]
        rows.append({
            "rank": rank,
            "score": round(float(result["score"]), 6),
            "document_id": chunk.get("document_id"),
            "chunk_id": chunk.get("chunk_id"),
            "business_function": chunk.get("business_function"),
            "chunk_type": chunk.get("chunk_type"),
            "title": chunk.get("title"),
            "section_title": chunk.get("section_title"),
            "source_url": chunk.get("source_url"),
            "preview": chunk.get("content", "")[:250],
        })

    return rows


def run_kdic_rag(
    question: str,
    *,
    top_k: int = HCX_RAG_TOP_K,
    business_function: str | None = HCX_RAG_BUSINESS_FUNCTION,
    min_score: float = HCX_RAG_MIN_SCORE,
    save_result: bool = True,
) -> dict[str, Any]:
    question = str(question).strip()
    if not question:
        raise ValueError("질문이 비어 있습니다.")
    if top_k < 1:
        raise ValueError("top_k는 1 이상이어야 합니다.")
    if business_function and business_function not in business_functions:
        raise ValueError(
            "존재하지 않는 업무 필터입니다. "
            f"허용값={business_functions}"
        )

    search_results = semantic_search_hcx(
        question,
        top_k=top_k,
        business_function=business_function,
        min_score=min_score,
    )
    result_rows = build_search_result_rows(search_results)
    result_df = pd.DataFrame(result_rows)

    print("질문:", question)
    print("업무 필터:", business_function)
    print("Top-K:", top_k)
    print("최소 유사도:", min_score)
    print("\n검색 결과")
    display(result_df)

    answer = generate_grounded_hcx_answer(
        question,
        search_results,
        min_score=min_score,
    )

    print("\nHCX 답변\n")
    print(answer)

    payload = {
        "question": question,
        "business_function_filter": business_function,
        "minimum_score": min_score,
        "chat_model": HCX_CHAT_MODEL,
        "embedding_model": HCX_EMBEDDING_MODEL,
        "top_k": top_k,
        "retrieved": result_rows,
        "answer": answer,
        "generated_at": utc_now_iso(),
    }

    if save_result:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = RESULT_ROOT / f"hcx_rag_{timestamp}.json"
        result_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        payload["result_path"] = str(result_path)
        print("\n저장:", result_path)

    return payload


print("질문 실행 함수 준비 완료")
