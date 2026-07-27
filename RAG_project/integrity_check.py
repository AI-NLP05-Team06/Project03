from config import *
from upload_extract import *
from load_data import *


def assert_unique(values: list[str], label: str) -> None:
    duplicates = pd.Series(values).value_counts()
    duplicates = duplicates[duplicates > 1]
    if not duplicates.empty:
        raise RuntimeError(
            f"{label} 중복 발견: {duplicates.index.tolist()[:10]}"
        )


document_ids = [
    str(document.get("doc_id", "")).strip()
    for document in documents
]
chunk_ids = [
    str(chunk.get("chunk_id", "")).strip()
    for chunk in chunks
]
embedding_chunk_ids = [
    str(record.get("chunk_id", "")).strip()
    for record in embedding_records
]

if any(not value for value in document_ids):
    raise RuntimeError("빈 doc_id가 존재합니다.")
if any(not value for value in chunk_ids):
    raise RuntimeError("빈 chunk_id가 존재합니다.")
if any(not value for value in embedding_chunk_ids):
    raise RuntimeError("임베딩에 빈 chunk_id가 존재합니다.")

assert_unique(document_ids, "doc_id")
assert_unique(chunk_ids, "chunk_id")
assert_unique(embedding_chunk_ids, "embedding chunk_id")

chunk_id_set = set(chunk_ids)
embedding_chunk_id_set = set(embedding_chunk_ids)

missing_embeddings = sorted(chunk_id_set - embedding_chunk_id_set)
orphan_embeddings = sorted(embedding_chunk_id_set - chunk_id_set)

if missing_embeddings:
    raise RuntimeError(
        f"임베딩이 없는 청크가 있습니다: {missing_embeddings[:20]}"
    )
if orphan_embeddings:
    raise RuntimeError(
        f"원본 청크가 없는 임베딩이 있습니다: {orphan_embeddings[:20]}"
    )

embedding_dimensions = {
    int(record.get("dimensions", 0))
    for record in embedding_records
}
embedding_models = {
    str(record.get("model", ""))
    for record in embedding_records
}

if len(embedding_dimensions) != 1 or 0 in embedding_dimensions:
    raise RuntimeError(
        f"임베딩 차원이 일치하지 않습니다: {embedding_dimensions}"
    )

for record in embedding_records:
    vector = record.get("embedding")
    if not isinstance(vector, list):
        raise TypeError(
            f"embedding이 list가 아닙니다: {record.get('chunk_id')}"
        )
    if len(vector) != record.get("dimensions"):
        raise RuntimeError(
            f"임베딩 길이 불일치: {record.get('chunk_id')}"
        )
    if not all(math.isfinite(float(value)) for value in vector):
        raise RuntimeError(
            f"NaN 또는 무한대 임베딩 발견: {record.get('chunk_id')}"
        )

business_functions = sorted({
    str(chunk.get("business_function", "")).strip()
    for chunk in chunks
    if str(chunk.get("business_function", "")).strip()
})

print("무결성 검사 통과")
print("- 청크-임베딩 연결:", len(chunk_ids), "건")
print("- 임베딩 차원:", embedding_dimensions)
print("- 저장 임베딩 모델:", embedding_models)
print("- 업무 목록:")
for value in business_functions:
    print("  -", value)
