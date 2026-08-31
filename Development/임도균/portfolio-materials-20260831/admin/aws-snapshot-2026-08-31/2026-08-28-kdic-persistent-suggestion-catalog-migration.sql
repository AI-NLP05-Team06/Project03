BEGIN;

-- The 26 administrator-managed recommendation answers are durable content,
-- not disposable runtime cache entries.  NULL means "keep until replaced".
ALTER TABLE suggestion_answer_cache
    ALTER COLUMN expires_at DROP NOT NULL;

UPDATE suggestion_answer_cache
SET expires_at = NULL
WHERE validation_status = 'VALIDATED';

CREATE TABLE IF NOT EXISTS suggestion_catalog (
    suggestion_id       VARCHAR(100) PRIMARY KEY,
    business            TEXT NOT NULL,
    keyword             TEXT NOT NULL,
    canonical_question  TEXT NOT NULL,
    display_order       SMALLINT NOT NULL,
    is_enabled          BOOLEAN NOT NULL DEFAULT true,
    is_locked           BOOLEAN NOT NULL DEFAULT true,
    active_cache_key    TEXT REFERENCES suggestion_answer_cache(cache_key)
                            ON DELETE SET NULL,
    row_version         BIGINT NOT NULL DEFAULT 1,
    created_by          TEXT NOT NULL DEFAULT 'SYSTEM',
    updated_by          TEXT NOT NULL DEFAULT 'SYSTEM',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT suggestion_catalog_display_order_check
        CHECK (display_order BETWEEN 1 AND 26),
    CONSTRAINT suggestion_catalog_row_version_check CHECK (row_version >= 1)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_suggestion_catalog_display_order
    ON suggestion_catalog (display_order);

CREATE UNIQUE INDEX IF NOT EXISTS idx_suggestion_catalog_business_keyword
    ON suggestion_catalog (business, keyword);

CREATE INDEX IF NOT EXISTS idx_suggestion_catalog_active_cache
    ON suggestion_catalog (active_cache_key);

COMMIT;
