from evaluate_structured_hybrid_parent_child_reranker import (
    ParentChildExpander,
    parent_child_metrics,
)


def sample_chunks():
    return [
        {
            "chunk_id": f"A_chunk_{index:03d}",
            "parent_doc_id": "A",
            "document_id": "A",
            "chunk_index": index,
            "content": f"A {index}",
        }
        for index in range(5)
    ] + [
        {
            "chunk_id": f"B_chunk_{index:03d}",
            "parent_doc_id": "B",
            "document_id": "B",
            "chunk_index": index,
            "content": f"B {index}",
        }
        for index in range(3)
    ]


def test_seed_preservation_before_neighbor_expansion():
    expander = ParentChildExpander(sample_chunks())
    expanded, _ = expander.expand(
        ["A_chunk_001", "A_chunk_003", "B_chunk_001"],
        seed_child_k=3,
        neighbor_window=1,
        max_parents=2,
        max_chunks_per_parent=3,
        max_context_chunks=6,
    )
    assert expanded[:3] == ["A_chunk_001", "A_chunk_003", "B_chunk_001"]
    assert len(expanded) <= 6
    assert len(set(expanded)) == len(expanded)


def test_expansion_recall_gain():
    expander = ParentChildExpander(sample_chunks())
    ranked = ["A_chunk_001", "B_chunk_001"]
    expanded, _ = expander.expand(
        ranked,
        seed_child_k=2,
        neighbor_window=1,
        max_parents=2,
        max_chunks_per_parent=3,
        max_context_chunks=6,
    )
    metrics = parent_child_metrics(
        expander=expander,
        ranked_child_ids=ranked,
        expanded_ids=expanded,
        gold_ids=["A_chunk_002"],
        primary_gold_ids=["A_chunk_002"],
        supporting_gold_ids=[],
        seed_child_k=2,
    )
    assert metrics["child_seed_recall"] == 0.0
    assert metrics["expanded_gold_recall"] == 1.0
    assert metrics["expansion_recall_gain"] == 1.0


def test_all_seed_children_survive_parent_limit():
    chunks = [
        {
            "chunk_id": f"P{index}_chunk_000",
            "parent_doc_id": f"P{index}",
            "document_id": f"P{index}",
            "chunk_index": 0,
            "content": f"P{index}",
        }
        for index in range(5)
    ]
    ranked = [chunk["chunk_id"] for chunk in chunks]
    expander = ParentChildExpander(chunks)
    expanded, _ = expander.expand(
        ranked,
        seed_child_k=5,
        neighbor_window=1,
        max_parents=3,
        max_chunks_per_parent=3,
        max_context_chunks=10,
    )
    assert expanded[:5] == ranked
