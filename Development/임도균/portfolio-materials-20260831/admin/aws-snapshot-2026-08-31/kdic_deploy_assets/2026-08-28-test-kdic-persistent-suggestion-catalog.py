from __future__ import annotations

import ast
import importlib.util
import json
import sys
import time
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
CORE_FILE = BASE_DIR / "2026-08-23-kdic-service-core.py"
POSTGRES_FILE = BASE_DIR / "kdic_postgres_store.py"
PREWARM_FILE = BASE_DIR / "2026-08-26-prewarm-kdic-suggestion-answer-cache.py"
MIGRATION_FILE = (
    BASE_DIR.parent / "2026-08-28-kdic-persistent-suggestion-catalog-migration.sql"
)
GRAMMAR_MIGRATION_FILE = (
    BASE_DIR.parent
    / "2026-08-28-kdic-fixed-suggestion-query-grammar-migration.sql"
)


def _load_core():
    spec = importlib.util.spec_from_file_location("kdic_persistent_catalog_core", CORE_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(CORE_FILE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _bundle(core, row, cache_key: str, answer: str):
    return core.CachedAnswerBundle(
        cache_key=cache_key,
        suggestion_id=row["suggestion_id"],
        business=row["business"],
        keyword=row["label"],
        question=row["query"],
        public_result={"answer": answer},
        raw_result={"answer": answer},
        basis_result={"schema_version": core.BASIS_EXPLANATION_SCHEMA_VERSION},
        pipeline_name="TEST",
        runtime_revision="test-revision",
        created_at=time.time() - 86_400,
        updated_at=time.time() - 86_400,
    )


def test_static_contracts() -> dict[str, str]:
    for path in (CORE_FILE, POSTGRES_FILE, PREWARM_FILE):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    postgres = POSTGRES_FILE.read_text(encoding="utf-8")
    prewarm = PREWARM_FILE.read_text(encoding="utf-8")
    migration = MIGRATION_FILE.read_text(encoding="utf-8")
    grammar_migration = GRAMMAR_MIGRATION_FILE.read_text(encoding="utf-8")
    assert "CREATE TABLE IF NOT EXISTS suggestion_catalog" in migration
    assert "ALTER COLUMN expires_at DROP NOT NULL" in migration
    assert "SET expires_at = NULL" in migration
    assert "UPDATE suggestion_catalog AS catalog" in grammar_migration
    assert "UPDATE suggestion_answer_cache AS answer" in grammar_migration
    assert grammar_migration.count("SQ-") == 22
    assert not any(
        bad in grammar_migration
        for bad in ("서류을", "절차을", "시기을", "한도을", "자료을")
    )
    assert "DELETE FROM suggestion_answer_cache" not in postgres
    assert "'VALIDATED', NULL" in postgres
    assert "def get_active(" in postgres
    assert "def activate(" in postgres
    assert '"retention_policy": "MANUAL_REPLACEMENT_NO_TTL"' in postgres
    assert "cache.sync_catalog(catalog)" in prewarm
    assert "cache.peek_active" in prewarm
    return {
        "migration": "passed",
        "fixed_query_grammar_migration": "passed",
        "no_ttl_delete": "passed",
        "postgres_active_pointer": "passed",
        "prewarm_uses_persistent_catalog": "passed",
    }


def test_fixed_catalog(core) -> dict[str, object]:
    catalog = core.suggestion_catalog()
    assert len(catalog) == 26
    assert len({row["suggestion_id"] for row in catalog}) == 26
    assert {row["suggestion_id"] for row in catalog} == set(
        core.FOLLOWUP_SUGGESTION_IDS.values()
    )
    assert not any(
        bad in row["query"]
        for row in catalog
        for bad in ("서류을", "절차을", "시기을", "한도을", "자료을")
    )

    cache = core.InMemorySuggestionAnswerCache(ttl_seconds=60, max_entries=100)
    cache.sync_catalog(catalog)
    first = catalog[0]
    first_key = "history:first:v1"
    replacement_key = "history:first:v2"
    cache.put(_bundle(core, first, first_key, "첫 번째 승인 답변"))
    cache.put(_bundle(core, first, replacement_key, "교체 대기 답변"))

    # Time does not expire an approved answer, and merely generating a new row
    # does not replace what users currently receive.
    assert cache.peek_active(first["suggestion_id"]).cache_key == first_key
    cache.activate(first["suggestion_id"], replacement_key)
    assert cache.peek_active(first["suggestion_id"]).cache_key == replacement_key

    for index, row in enumerate(catalog[1:], start=2):
        cache.put(_bundle(core, row, f"history:{index}:v1", f"승인 답변 {index}"))
    stats = cache.stats()
    assert stats["catalog_count"] == 26
    assert stats["active_count"] == 26
    assert stats["missing_active_count"] == 0
    assert stats["ready"] is True
    assert stats["ttl_seconds"] is None
    assert stats["retention_policy"] == "MANUAL_REPLACEMENT_NO_TTL"
    return {
        "catalog_count": stats["catalog_count"],
        "active_count": stats["active_count"],
        "fixed_26_ready": stats["ready"],
        "manual_replacement": "passed",
        "time_expiry_removed": "passed",
    }


def main() -> None:
    core = _load_core()
    print(
        json.dumps(
            {
                "status": "passed",
                "static": test_static_contracts(),
                "catalog": test_fixed_catalog(core),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
