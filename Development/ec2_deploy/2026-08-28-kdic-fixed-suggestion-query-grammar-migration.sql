BEGIN;

WITH fixed_questions (suggestion_id, canonical_question) AS (
    VALUES
        ('SQ-2060AAB635105137', '착오송금 반환 신청의 필요 서류를 알려주세요.'),
        ('SQ-B791CB88BFC45CF3', '착오송금 반환 신청의 처리 절차를 알려주세요.'),
        ('SQ-459DECFF580A5384', '예금보험금 안내의 필요 서류를 알려주세요.'),
        ('SQ-320F0AB620FF5A12', '예금보험금 안내의 지급 절차를 알려주세요.'),
        ('SQ-FF6F9636C6005A67', '예금보험금 안내의 지급 시기를 알려주세요.'),
        ('SQ-AC6E709EA06D5BD8', '예금자보호제도의 보호 한도를 알려주세요.'),
        ('SQ-A8BD396C8E29557D', '고객 미수령금 신청의 지급 절차를 알려주세요.'),
        ('SQ-BC43EF4C7E93539C', '채무조정 안내의 필요 서류를 알려주세요.'),
        ('SQ-66BFA311A16459DB', '채무조정 안내의 조정 절차를 알려주세요.'),
        ('SQ-2052C10901E15DEF', '은닉재산 신고의 필요 자료를 알려주세요.'),
        ('SQ-A049AF6FABE55BA4', '은닉재산 신고의 처리 절차를 알려주세요.')
)
UPDATE suggestion_catalog AS catalog
SET canonical_question = fixed.canonical_question,
    updated_at = now()
FROM fixed_questions AS fixed
WHERE catalog.suggestion_id = fixed.suggestion_id
  AND catalog.canonical_question IS DISTINCT FROM fixed.canonical_question;

WITH fixed_questions (suggestion_id, canonical_question) AS (
    VALUES
        ('SQ-2060AAB635105137', '착오송금 반환 신청의 필요 서류를 알려주세요.'),
        ('SQ-B791CB88BFC45CF3', '착오송금 반환 신청의 처리 절차를 알려주세요.'),
        ('SQ-459DECFF580A5384', '예금보험금 안내의 필요 서류를 알려주세요.'),
        ('SQ-320F0AB620FF5A12', '예금보험금 안내의 지급 절차를 알려주세요.'),
        ('SQ-FF6F9636C6005A67', '예금보험금 안내의 지급 시기를 알려주세요.'),
        ('SQ-AC6E709EA06D5BD8', '예금자보호제도의 보호 한도를 알려주세요.'),
        ('SQ-A8BD396C8E29557D', '고객 미수령금 신청의 지급 절차를 알려주세요.'),
        ('SQ-BC43EF4C7E93539C', '채무조정 안내의 필요 서류를 알려주세요.'),
        ('SQ-66BFA311A16459DB', '채무조정 안내의 조정 절차를 알려주세요.'),
        ('SQ-2052C10901E15DEF', '은닉재산 신고의 필요 자료를 알려주세요.'),
        ('SQ-A049AF6FABE55BA4', '은닉재산 신고의 처리 절차를 알려주세요.')
)
UPDATE suggestion_answer_cache AS answer
SET question = fixed.canonical_question,
    updated_at = now()
FROM fixed_questions AS fixed
WHERE answer.suggestion_id = fixed.suggestion_id
  AND answer.question IS DISTINCT FROM fixed.canonical_question;

COMMIT;
