BEGIN;

-- The original migration fixed the catalog at exactly 26 rows.  The manager UI
-- keeps those defaults, but administrators may add or retire keyword buttons.
ALTER TABLE suggestion_catalog
    ADD COLUMN IF NOT EXISTS business_key TEXT;

UPDATE suggestion_catalog
SET business_key = CASE
    WHEN business LIKE '%착오송금%' THEN '착오송금'
    WHEN business LIKE '%예금보험금%' THEN '예금보험금'
    WHEN business LIKE '%예금자보호%' THEN '예금자보호'
    WHEN business LIKE '%미수령금%' THEN '미수령금'
    WHEN business LIKE '%채무조정%' THEN '채무조정'
    WHEN business LIKE '%은닉재산%' THEN '은닉재산'
    ELSE business
END
WHERE business_key IS NULL OR btrim(business_key) = '';

ALTER TABLE suggestion_catalog
    ALTER COLUMN business_key SET NOT NULL;

ALTER TABLE suggestion_catalog
    DROP CONSTRAINT IF EXISTS suggestion_catalog_display_order_check;

ALTER TABLE suggestion_catalog
    ADD CONSTRAINT suggestion_catalog_display_order_check
        CHECK (display_order BETWEEN 1 AND 32767);

CREATE INDEX IF NOT EXISTS idx_suggestion_catalog_business_key
    ON suggestion_catalog (business_key, display_order);

COMMIT;
