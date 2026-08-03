# [7/10] 코사인 유사도 기반 Dense 검색과, 검색된 근거 청크만 사용하는 HCX 답변 생성 함수를 정의합니다.
from config import *
from load_data import *
from hcx_api import *


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


def _allowed_evidence_values(
    search_results: list[dict],
) -> tuple[set[str], set[str]]:
    urls: set[str] = set()
    phones: set[str] = set()
    phone_pattern = re.compile(
        r"(?:0\d{1,2}[-)]\s*)?\d{3,4}-\d{4}"
    )

    for result in search_results:
        chunk = result["chunk"]

        for value in [
            chunk.get("source_url"),
            chunk.get("official_download_url"),
        ]:
            if value:
                urls.add(value.rstrip("/"))

        for link in chunk.get("related_links", []):
            if link.get("url"):
                urls.add(link["url"].rstrip("/"))

        phones.update(
            phone_pattern.findall(chunk.get("content", ""))
        )

    return urls, phones


def _remove_unsupported_urls_and_phones(
    answer: str,
    allowed_urls: set[str],
    allowed_phones: set[str],
) -> str:
    url_pattern = re.compile(r"https?://[^\s)\]}>]+")
    phone_pattern = re.compile(
        r"(?:0\d{1,2}[-)]\s*)?\d{3,4}-\d{4}"
    )

    def url_replacer(match: re.Match) -> str:
        candidate = match.group(0).rstrip(".")
        return (
            candidate
            if candidate.rstrip("/") in allowed_urls
            else ""
        )

    def phone_replacer(match: re.Match) -> str:
        candidate = match.group(0)
        return candidate if candidate in allowed_phones else ""

    answer = url_pattern.sub(url_replacer, answer)
    answer = phone_pattern.sub(phone_replacer, answer)
    answer = re.sub(r"[ \t]+\n", "\n", answer)
    answer = re.sub(r"\n{3,}", "\n\n", answer)
    return answer.strip()


def generate_grounded_hcx_answer(
    question: str,
    search_results: list[dict],
    *,
    min_score: float = HCX_RAG_MIN_SCORE,
) -> str:
    if (
        not search_results
        or search_results[0]["score"] < min_score
    ):
        return (
            "검색된 공식 근거의 관련도가 충분하지 않아 "
            "현재 수집 데이터만으로는 정확하게 답할 수 없습니다."
        )

    evidence_blocks = []

    for rank, result in enumerate(search_results, start=1):
        chunk = result["chunk"]
        evidence_blocks.append("\n".join([
            f"[근거 {rank}]",
            f"문서 ID: {chunk.get('document_id')}",
            f"청크 ID: {chunk.get('chunk_id')}",
            f"업무: {chunk.get('business_function')}",
            f"출처: {chunk.get('source_url')}",
            "내용:",
            chunk.get("content", ""),
        ]))

    allowed_urls, allowed_phones = _allowed_evidence_values(
        search_results
    )
    evidence_text = "\n\n".join(evidence_blocks)

    raw_answer = hcx_chat_text(
        system_prompt=(
            "당신은 예금보험공사 공식 문서 기반 RAG 답변기입니다. "
            "제공된 근거에 명시된 정보만 사용하세요. "
            "근거에 없는 금액, 기한, 조건, 기관, URL, 전화번호를 "
            "만들지 마세요. 근거가 부족하면 확인할 수 없다고 답하세요. "
            "출처 목록은 작성하지 마세요."
        ),
        user_prompt=f"""
[사용자 질문]
{question}

[검색된 공식 근거]
{evidence_text}

[답변에 사용 가능한 URL]
{sorted(allowed_urls)}

[답변에 사용 가능한 전화번호]
{sorted(allowed_phones)}

한국어로 답변하세요. URL과 전화번호는 위 허용 목록에 있는 값만 그대로 사용할 수 있습니다.
""".strip(),
        max_tokens=900,
        temperature=0.0,
    )

    answer = _remove_unsupported_urls_and_phones(
        raw_answer,
        allowed_urls,
        allowed_phones,
    )

    used_chunks = [
        result["chunk"].get("chunk_id")
        for result in search_results
    ]
    source_urls = list(dict.fromkeys(
        result["chunk"].get("source_url")
        for result in search_results
        if result["chunk"].get("source_url")
    ))

    appendix = "\n\n근거 청크: " + ", ".join(used_chunks)
    if source_urls:
        appendix += "\n공식 출처:\n" + "\n".join(
            f"- {url}" for url in source_urls
        )

    return answer + appendix


print("Baseline 검색·답변 함수 준비 완료")
