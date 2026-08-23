-- KDIC 관리자 화면 + 운영 파이프라인용 RDS Postgres 스키마 (v2 — 팀 피드백 반영)
CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- gen_random_uuid()

-- ============================================================
-- 1. 인증 — /admin/* 보호용 (최소 토큰 인증)
-- ============================================================
CREATE TABLE admin_api_tokens (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    label           TEXT NOT NULL,
    token_hash      TEXT NOT NULL UNIQUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at      TIMESTAMPTZ,
    revoked_at      TIMESTAMPTZ,
    CONSTRAINT chk_admin_token_expiry CHECK (expires_at IS NULL OR expires_at > created_at)
);

-- ============================================================
-- 2. jobs — 비동기 작업 큐 (채팅 답변 / 색인 / 재수집 / 평가 배치 공용)
-- ============================================================
CREATE TABLE jobs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_type        TEXT NOT NULL CHECK (job_type IN ('chat', 'ingest', 'reingest', 'delete', 'eval_run')),
    status          TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'running', 'done', 'failed')),
    session_id      TEXT,
    progress        SMALLINT NOT NULL DEFAULT 0 CHECK (progress BETWEEN 0 AND 100),
    stage           TEXT,
    payload         JSONB NOT NULL DEFAULT '{}' CHECK (jsonb_typeof(payload) = 'object'),
    result          JSONB,
    error_message   TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at      TIMESTAMPTZ,
    finished_at     TIMESTAMPTZ,
    CONSTRAINT chk_jobs_time CHECK (finished_at IS NULL OR started_at IS NULL OR finished_at >= started_at)
);
CREATE INDEX idx_jobs_status ON jobs (status);
CREATE INDEX idx_jobs_type_status ON jobs (job_type, status);

-- ============================================================
-- 3. pages / chunks_index — 데이터 관리(요구사항 1)
-- ============================================================
CREATE TABLE pages (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_url          TEXT NOT NULL UNIQUE,
    title               TEXT,
    business_category   TEXT,
    status              TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'stale', 'deleted')),
    last_ingest_job_id  UUID REFERENCES jobs(id),
    last_ingested_at    TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE chunks_index (
    chunk_id        TEXT PRIMARY KEY,
    page_id         UUID NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
    parent_doc_id   TEXT,
    status          TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'deleted')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_chunks_index_page ON chunks_index (page_id);

-- ============================================================
-- 4. staging_preview — 처리결과 미리보기(요구사항 2)
-- ============================================================
CREATE TABLE staging_preview (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id          UUID REFERENCES jobs(id),
    source_url      TEXT NOT NULL,
    parsed_text     TEXT,
    chunk_preview   JSONB NOT NULL DEFAULT '[]' CHECK (jsonb_typeof(chunk_preview) = 'array'),
    review_status   TEXT NOT NULL DEFAULT 'pending' CHECK (review_status IN ('pending', 'approved', 'rejected')),
    reviewed_by     TEXT,
    reviewed_at     TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_staging_preview_status ON staging_preview (review_status);

-- ============================================================
-- 5. search_params — 파라미터 테스트(요구사항 3)
--    dense/bm25 weight는 소수점 3자리까지 실험 가능하게 NUMERIC(4,3)
-- ============================================================
CREATE TABLE search_params (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    label                   TEXT NOT NULL,
    dense_weight            NUMERIC(4,3) NOT NULL,
    bm25_weight             NUMERIC(4,3) NOT NULL,
    candidate_depth         SMALLINT NOT NULL,
    final_top_k             SMALLINT NOT NULL,
    rrf_k                   SMALLINT,
    reranker_model          TEXT NOT NULL,
    reranker_candidate_depth SMALLINT,
    is_active               BOOLEAN NOT NULL DEFAULT false,
    created_by              TEXT,
    note                    TEXT,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT chk_weight_sum CHECK (dense_weight + bm25_weight = 1.0)
);
CREATE UNIQUE INDEX idx_search_params_single_active
    ON search_params (is_active) WHERE is_active = true;

-- ============================================================
-- 6. eval_queries / eval_runs / eval_run_results — 평가 연동(요구사항 5)
-- ============================================================
CREATE TABLE eval_queries (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    question            TEXT NOT NULL,
    business_category   TEXT,
    expected_chunk_ids  JSONB CHECK (expected_chunk_ids IS NULL OR jsonb_typeof(expected_chunk_ids) = 'array'),
    is_active           BOOLEAN NOT NULL DEFAULT true,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE eval_runs (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    search_params_id  UUID NOT NULL REFERENCES search_params(id),
    job_id            UUID REFERENCES jobs(id),
    status            TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'done', 'failed')),
    triggered_by      TEXT,
    note              TEXT,
    started_at        TIMESTAMPTZ,
    finished_at       TIMESTAMPTZ,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE eval_run_results (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    eval_run_id         UUID NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
    eval_query_id       UUID NOT NULL REFERENCES eval_queries(id),
    retrieved_chunk_ids JSONB NOT NULL DEFAULT '[]' CHECK (jsonb_typeof(retrieved_chunk_ids) = 'array'),
    rank_of_expected    SMALLINT,
    hit                 BOOLEAN NOT NULL DEFAULT false,
    latency_ms          NUMERIC(10,2),
    answer_text         TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (eval_run_id, eval_query_id)   -- 같은 run에서 같은 질문 중복 저장 방지 (Hit@K/MRR 집계 오염 방지)
);
CREATE INDEX idx_eval_run_results_run ON eval_run_results (eval_run_id);

-- ============================================================
-- 7. hcx_call_log — 429/HCX 호출 추적 (job_id로 어느 작업에서 났는지 추적 가능)
-- ============================================================
CREATE TABLE hcx_call_log (
    id              BIGSERIAL PRIMARY KEY,
    job_id          UUID REFERENCES jobs(id),
    stage           TEXT NOT NULL,   -- query_embedding / answer_generation / decomposition
    success         BOOLEAN NOT NULL,
    rate_limit_hit  BOOLEAN NOT NULL DEFAULT false,
    pacing_wait_ms  NUMERIC(10,2),
    retry_wait_ms   NUMERIC(10,2),
    wall_latency_ms NUMERIC(10,2),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_hcx_call_log_job ON hcx_call_log (job_id);
CREATE INDEX idx_hcx_call_log_created ON hcx_call_log (created_at);

-- ============================================================
-- (보류) chat_sessions / chat_messages
-- 지금 5개 관리자 요구사항엔 대화이력 분석이 없어서 일단 안 만듦.
-- "대화 이력/사용 chunk/적용 파라미터 분석"이 실제 요구사항으로 나오면 그때 추가:
--   chat_sessions(id, started_at, ...)
--   chat_messages(id, session_id FK, role, content, used_chunk_ids JSONB,
--                 search_params_id FK, job_id FK, created_at)
-- ============================================================
