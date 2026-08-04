# [7/10] 코사인 유사도 기반 Dense 검색 함수를 정의합니다.
# (답변 생성은 generation/answer_generation.py로 분리되어 있습니다)
from core.config import *
from core.load_data import *
from core.hcx_api import *


def cosine_similarity(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
) -> float:
    denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if denominator == 0:
        return 0.0
    return float(np.dot(vector_a, vector_b) / denominator)


def semantic_search_hcx(
    question: str,
    *,
    top_k: int = 5,
    business_function: str | None = None,
    min_score: float | None = None,
) -> list[dict]:
    if not embedding_records:
        raise RuntimeError(
            "임베딩 결과가 없습니다. KDIC_output을 먼저 로드하세요."
        )

    query_vector = np.asarray(
        hcx_embed_text(question),
        dtype=np.float32,
    )
    chunks_by_id = {
        chunk["chunk_id"]: chunk
        for chunk in RESULT["chunks"]
    }
    scored = []

    for record in embedding_records:
        if (
            business_function
            and record.get("business_function") != business_function
        ):
            continue

        vector = np.asarray(
            record["embedding"],
            dtype=np.float32,
        )
        score = cosine_similarity(query_vector, vector)

        if min_score is not None and score < min_score:
            continue

        chunk = chunks_by_id.get(record["chunk_id"])
        if chunk:
            scored.append({"score": score, "chunk": chunk})

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


print("Dense 검색 함수 준비 완료")
