# [6/10] HCX 채팅 완성(hcx_chat_text)과 질문 임베딩(hcx_embed_text) 호출 함수를 정의합니다.
from config import *
from hcx_client import *
from integrity_check import *


def hcx_chat_text(
    *,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 500,
    temperature: float = 0.1,
) -> str:
    response = hcx_client.chat.completions.create(
        model=HCX_CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    text = response.choices[0].message.content
    if not text or not text.strip():
        raise RuntimeError("HCX 응답 본문이 비어 있습니다.")
    return text.strip()


def normalize_hcx_embedding_text(text: str) -> str:
    cleaned = (
        str(text)
        .replace("\x00", "")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .strip()
    )

    if not cleaned:
        raise ValueError("임베딩 입력 텍스트가 비어 있습니다.")
    return cleaned


def hcx_embed_text(text: str) -> list[float]:
    """질문 문자열 한 건을 bge-m3 벡터 하나로 변환합니다."""
    cleaned_text = normalize_hcx_embedding_text(text)

    response = hcx_client.embeddings.create(
        model=HCX_EMBEDDING_MODEL,
        input=cleaned_text,
        encoding_format=HCX_EMBEDDING_ENCODING_FORMAT,
    )

    if len(response.data) != 1:
        raise RuntimeError(
            "단일 임베딩 요청의 응답 개수가 1개가 아닙니다. "
            f"응답 개수={len(response.data)}"
        )

    vector = [float(value) for value in response.data[0].embedding]

    if not vector:
        raise RuntimeError("HCX 임베딩 벡터가 비어 있습니다.")
    if not all(math.isfinite(value) for value in vector):
        raise RuntimeError(
            "HCX 임베딩 벡터에 NaN 또는 무한대가 포함되어 있습니다."
        )

    stored_dimensions = next(iter(embedding_dimensions))
    if len(vector) != stored_dimensions:
        raise RuntimeError(
            "질문 임베딩과 저장 청크 임베딩의 차원이 다릅니다: "
            f"query={len(vector)}, stored={stored_dimensions}"
        )

    return vector


print("HCX 호출 함수 준비 완료")
