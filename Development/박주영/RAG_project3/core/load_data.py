# [3/10] 압축 해제된 폴더에서 documents/chunks/chunk_embeddings_hcx jsonl 파일을 찾아 로드합니다.
from core.config import *
from core.upload_extract import *


def find_unique_file(root: Path, filename: str) -> Path:
    matches = list(root.rglob(filename))

    if not matches:
        raise FileNotFoundError(
            f"압축 해제 결과에서 {filename}을 찾지 못했습니다."
        )

    # processed 폴더 내부 파일을 우선합니다.
    processed_matches = [
        path for path in matches
        if path.parent.name == "processed"
    ]
    candidates = processed_matches or matches

    if len(candidates) != 1:
        raise RuntimeError(
            f"{filename} 후보가 여러 개입니다: "
            + ", ".join(str(path) for path in candidates)
        )

    return candidates[0]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSONL 파싱 실패: {path}, line={line_number}"
                ) from error

            if not isinstance(record, dict):
                raise TypeError(
                    f"JSONL 레코드가 객체가 아닙니다: {path}, line={line_number}"
                )
            records.append(record)

    return records


DOCUMENTS_PATH = find_unique_file(EXTRACT_ROOT, "documents.jsonl")
CHUNKS_PATH = find_unique_file(EXTRACT_ROOT, "chunks.jsonl")
EMBEDDINGS_PATH = find_unique_file(
    EXTRACT_ROOT,
    "chunk_embeddings_hcx.jsonl",
)


documents = load_jsonl(DOCUMENTS_PATH)
chunks = load_jsonl(CHUNKS_PATH)
embedding_records = load_jsonl(EMBEDDINGS_PATH)

# 기존 V4.6 검색 함수가 사용하던 구조를 유지합니다.
RESULT = {
    "documents": documents,
    "chunks": chunks,
}

print("로드 완료")
print("- documents:", len(documents), DOCUMENTS_PATH)
print("- chunks:", len(chunks), CHUNKS_PATH)
print("- embeddings:", len(embedding_records), EMBEDDINGS_PATH)
