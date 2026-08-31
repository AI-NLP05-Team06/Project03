-- v3 위에 얹는 추가분: 승인된 미리보기가 실제 ES에 반영됐는지 추적
ALTER TABLE staging_preview ADD COLUMN IF NOT EXISTS published_at TIMESTAMPTZ;
