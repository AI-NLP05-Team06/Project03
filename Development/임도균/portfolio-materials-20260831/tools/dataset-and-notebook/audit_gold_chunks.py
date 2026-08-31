import csv
import json
import sys
from collections import Counter
from pathlib import Path


def parse_list(value):
    text = (value or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None
    if any(not isinstance(item, str) for item in parsed):
        return None
    return parsed


dataset_path = Path(sys.argv[1])
chunks_path = Path(sys.argv[2])

with dataset_path.open(encoding="utf-8-sig", newline="") as file:
    rows = list(csv.DictReader(file, delimiter="\t"))

chunks = {}
with chunks_path.open(encoding="utf-8-sig") as file:
    for line in file:
        if line.strip():
            chunk = json.loads(line)
            chunks[chunk["chunk_id"]] = chunk

results = []
parse_errors = []

for row_number, row in enumerate(rows, start=2):
    question_id = row.get("질문ID", "")
    document_ids = parse_list(row.get("gold_document_ids"))
    section_titles = parse_list(row.get("gold_section_titles"))
    chunk_ids = parse_list(row.get("gold_chunk_ids"))
    primary_ids = parse_list(row.get("gold_primary_chunk_ids"))
    supporting_ids = parse_list(row.get("gold_supporting_chunk_ids"))

    for field, parsed in (
        ("gold_document_ids", document_ids),
        ("gold_section_titles", section_titles),
        ("gold_chunk_ids", chunk_ids),
        ("gold_primary_chunk_ids", primary_ids),
        ("gold_supporting_chunk_ids", supporting_ids),
    ):
        if parsed is None:
            parse_errors.append(
                {
                    "row": row_number,
                    "question_id": question_id,
                    "field": field,
                    "value": row.get(field, ""),
                }
            )

    if any(
        value is None
        for value in (document_ids, section_titles, chunk_ids, primary_ids, supporting_ids)
    ):
        continue

    missing_chunks = [chunk_id for chunk_id in chunk_ids if chunk_id not in chunks]
    existing = [chunks[chunk_id] for chunk_id in chunk_ids if chunk_id in chunks]
    actual_documents = sorted(
        {
            str(chunk.get("parent_doc_id") or chunk.get("document_id") or "")
            for chunk in existing
        }
        - {""}
    )
    wrong_documents = [
        doc_id for doc_id in actual_documents if doc_id not in set(document_ids)
    ]

    actual_labels = []
    for chunk in existing:
        labels = []
        for key in ("title", "section_title"):
            value = str(chunk.get(key) or "").strip()
            if value and value not in labels:
                labels.append(value)
        for value in chunk.get("heading_path") or []:
            value = str(value or "").strip()
            if value and value not in labels:
                labels.append(value)
        canonical_label = str(chunk.get("section_title") or "").strip()
        if not canonical_label:
            canonical_label = str(chunk.get("title") or "").strip()
        if not canonical_label:
            heading_path = chunk.get("heading_path") or []
            canonical_label = str(heading_path[-1] if heading_path else "").strip()
        actual_labels.append(
            {
                "chunk_id": chunk["chunk_id"],
                "labels": labels,
                "canonical_label": canonical_label,
                "title": chunk.get("title") or "",
                "section_title": chunk.get("section_title") or "",
                "heading_path": chunk.get("heading_path") or [],
                "content": chunk.get("content") or "",
            }
        )

    normalized_actual = {
        item["canonical_label"] for item in actual_labels if item["canonical_label"]
    }
    unmatched_sections = [
        title for title in section_titles if title.strip() not in normalized_actual
    ]
    chunk_labels_not_declared = [
        {
            "chunk_id": item["chunk_id"],
            "labels": [item["canonical_label"]],
        }
        for item in actual_labels
        if item["canonical_label"]
        and item["canonical_label"] not in set(section_titles)
    ]

    primary_not_gold = [chunk_id for chunk_id in primary_ids if chunk_id not in chunk_ids]
    supporting_not_gold = [
        chunk_id for chunk_id in supporting_ids if chunk_id not in chunk_ids
    ]
    gold_not_classified = [
        chunk_id
        for chunk_id in chunk_ids
        if chunk_id not in set(primary_ids + supporting_ids)
    ]

    issues = []
    if missing_chunks:
        issues.append("missing_chunk")
    if wrong_documents:
        issues.append("document_mismatch")
    if unmatched_sections:
        issues.append("section_not_found_in_chunk_metadata")
    if chunk_labels_not_declared:
        issues.append("chunk_label_not_in_gold_section_titles")
    if primary_not_gold or supporting_not_gold:
        issues.append("primary_supporting_not_in_gold")
    if gold_not_classified and (primary_ids or supporting_ids):
        issues.append("gold_chunk_not_classified")

    results.append(
        {
            "row": row_number,
            "question_id": question_id,
            "question": row.get("예상질문", ""),
            "gold_document_ids": document_ids,
            "gold_section_titles": section_titles,
            "gold_chunk_ids": chunk_ids,
            "actual_documents": actual_documents,
            "actual_labels": actual_labels,
            "missing_chunks": missing_chunks,
            "wrong_documents": wrong_documents,
            "unmatched_sections": unmatched_sections,
            "chunk_labels_not_declared": chunk_labels_not_declared,
            "primary_not_gold": primary_not_gold,
            "supporting_not_gold": supporting_not_gold,
            "gold_not_classified": gold_not_classified,
            "issues": issues,
        }
    )

summary = {
    "row_count": len(rows),
    "column_count": len(rows[0]) if rows else 0,
    "headers": list(rows[0].keys()) if rows else [],
    "chunk_count": len(chunks),
    "parse_errors": parse_errors,
    "issue_counts": Counter(
        issue for result in results for issue in result["issues"]
    ),
    "rows_with_issues": sum(bool(result["issues"]) for result in results),
}

payload = {"summary": summary, "results": results}
if len(sys.argv) >= 4:
    output_path = Path(sys.argv[3])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=list))
    print(f"OUTPUT={output_path}")
else:
    print(json.dumps(payload, ensure_ascii=False, indent=2))
