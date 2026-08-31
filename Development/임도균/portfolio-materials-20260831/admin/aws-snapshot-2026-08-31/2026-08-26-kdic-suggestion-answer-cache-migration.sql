BEGIN;

CREATE TABLE IF NOT EXISTS suggestion_answer_cache (
    cache_key TEXT PRIMARY KEY,
    suggestion_id VARCHAR(100) NOT NULL,
    business TEXT NOT NULL,
    keyword TEXT NOT NULL,
    question TEXT NOT NULL,
    public_result JSONB NOT NULL,
    raw_result JSONB NOT NULL,
    basis_result JSONB NOT NULL DEFAULT '{}'::jsonb,
    pipeline_name TEXT NOT NULL,
    runtime_revision TEXT NOT NULL,
    validation_status VARCHAR(32) NOT NULL DEFAULT 'VALIDATED',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ NOT NULL,
    last_hit_at TIMESTAMPTZ,
    hit_count BIGINT NOT NULL DEFAULT 0,
    CONSTRAINT suggestion_answer_cache_validation_status_check
        CHECK (validation_status IN ('VALIDATED', 'INVALIDATED')),
    CONSTRAINT suggestion_answer_cache_hit_count_check CHECK (hit_count >= 0)
);

CREATE INDEX IF NOT EXISTS idx_suggestion_answer_cache_lookup
    ON suggestion_answer_cache (suggestion_id, pipeline_name, runtime_revision);

CREATE INDEX IF NOT EXISTS idx_suggestion_answer_cache_expiry
    ON suggestion_answer_cache (expires_at);

COMMIT;
