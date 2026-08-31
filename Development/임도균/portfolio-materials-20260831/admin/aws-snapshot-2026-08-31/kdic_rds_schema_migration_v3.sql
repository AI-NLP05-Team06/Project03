-- v2 스키마 위에 얹는 추가분 (DB 연동 작업 중 실제로 필요해져서 추가)
-- jobs 테이블: basis() 조회용 raw_result 컬럼 필요
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS raw_result JSONB;

-- 세션(멀티턴 대화) 상태 저장 — InMemorySessionStore 대체용
-- (보류했던 chat_sessions를 이제 실제로 필요해져서 최소 형태로 추가)
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id  TEXT PRIMARY KEY,
    state       JSONB NOT NULL DEFAULT '{}' CHECK (jsonb_typeof(state) = 'object'),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_updated ON chat_sessions (updated_at);
