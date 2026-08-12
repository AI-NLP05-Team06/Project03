from __future__ import annotations

import json
import textwrap
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
SOURCE_NOTEBOOK = WORKSPACE / "2026-08-12-KDIC-경량-RAG-질의분석-Colab-v3.ipynb"
OUTPUT_NOTEBOOK = ROOT / "2026-08-12-KDIC-경량-RAG-질의분석-Colab-v3-1.ipynb"
DIAGRAM = ROOT / "2026-08-12-KDIC-경량-RAG-질의분석-도식-v3-1.md"
VALIDATOR = ROOT / "2026-08-12-validate-kdic-v3-1-hard-filter-policy.py"
RELEASE_ZIP = ROOT / "2026-08-12-KDIC-경량-RAG-질의분석-v3-1-배포파일.zip"


def source_text(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def replace_block(source: str, start: str, end: str, replacement: str) -> str:
    start_index = source.index(start)
    end_index = source.index(end, start_index)
    return source[:start_index] + textwrap.dedent(replacement).strip() + "\n\n" + source[end_index:]


if not SOURCE_NOTEBOOK.exists():
    raise FileNotFoundError(f"V3 노트북을 찾을 수 없습니다: {SOURCE_NOTEBOOK}")

notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
cells = notebook["cells"]

cells[0]["source"] = textwrap.dedent(
    """
    # KDIC 경량 RAG 질의분석 v3.1

    v3의 정확한 라우팅은 유지하면서 잘못된 업무 Hard Filter가 정답 문서를 제거하지 않도록
    need 단위 Filter Safety Gate를 추가한 버전입니다.

    - `HARD`는 기본값이 아니라 모든 안전조건을 통과한 경우에만 허용
    - 복합 질문에서 전체 원문의 업무를 모든 need에 전파하지 않음
    - 업무 후보·근거·confidence·후보 점수 차이 기록
    - 분해 불완전, 다중 후보, 모델·규칙 충돌, 업무 간 모호성에서는 `SOFT`
    - 업무 후보가 없으면 `RETRIEVE_RELAXED`
    - 검색 결과가 없거나 품질이 낮을 때 `HARD → SOFT → NONE` fail-open 정책 제공
    - N2 이상에서 정확한 모델 intent를 규칙이 덮어쓰는 회귀 방지
    """
).strip() + "\n"

cells[1]["source"] = textwrap.dedent(
    """
    ## v3.1 파이프라인

    ```mermaid
    flowchart TD
        A["사용자 원문"] --> B["최소 정규화"]
        B --> C{"Fast Path"}
        C -->|"인사·메타"| D["DIRECT / API 0회"]
        C -->|"명확한 범위 외"| E["OUT_OF_SCOPE / API 0회"]
        C -->|"일반 질의"| F["HCX-007 Atomic Need 분석"]
        F --> G["Business·Intent 보정"]
        G --> H["Need 분해 완전성·업무 후보·충돌 분석"]
        H --> I{"필수정보 부족"}
        I -->|"예"| J["CLARIFY"]
        I -->|"아니오"| K{"업무 후보 존재"}
        K -->|"없음"| L["RETRIEVE_RELAXED / Filter NONE"]
        K -->|"있음"| M{"HARD 안전조건 모두 통과"}
        M -->|"예"| N["RETRIEVE / HARD"]
        M -->|"아니오"| O["RETRIEVE / SOFT"]
        N --> P["결과 없음·낮은 점수 시 SOFT → NONE"]
    ```

    V3.1은 불확실성을 확인 질문으로 전환하지 않습니다. 검색 자체가 가능한 경우에는
    `SOFT` 또는 `RETRIEVE_RELAXED`로 정답 업무 문서가 후보에서 제거되지 않게 합니다.
    """
).strip() + "\n"

pipeline_source = source_text(cells[7])
pipeline_source = pipeline_source.replace(
    'PIPELINE_VERSION = "KDIC_LIGHTWEIGHT_RAG_QUERY_ANALYZER_V3_2026_08_12"',
    'PIPELINE_VERSION = "KDIC_LIGHTWEIGHT_RAG_QUERY_ANALYZER_V3_1_2026_08_12"\n'
    'HARD_BUSINESS_CONFIDENCE_THRESHOLD = 0.98\n'
    'HARD_BUSINESS_MARGIN_THRESHOLD = 0.20',
)

pipeline_source = replace_block(
    pipeline_source,
    "def apply_business_and_intent_rules_v3",
    "def decide_route_v3",
    r'''
    STRONG_BUSINESS_KEYWORDS_V31 = {
        "예금자보호제도": ["예금자보호", "보호한도", "예금 보호", "보호 한도"],
        "예금보험금 안내": ["예금보험금", "보험금 지급", "1종 보험사고", "2종 보험사고"],
        "고객 미수령금 신청": ["미수령금", "파산배당금", "지급대행점", "상속인 금융거래 조회"],
        "착오송금 반환 신청": ["착오송금", "착오 송금", "반환지원", "매입계약", "지급명령", "강제집행"],
        "채무조정 안내": ["채무조정", "신용회복지원", "개인회생", "개인파산", "워크아웃", "부채증명원"],
        "은닉재산 신고": ["은닉재산", "은닉 재산", "금융부실관련자", "차명재산", "차명 재산", "신고 포상금"],
    }
    WEAK_OR_CROSS_BUSINESS_TERMS_V31 = {
        "예금보험금 안내": ["보험사고", "가지급금", "개산지급금"],
        "고객 미수령금 신청": ["개산지급금 정산금"],
    }
    MULTI_DECOMPOSITION_SIGNAL_V31 = re.compile(
        r"각각|뿐만\s*아니라|함께\s*알려|동시에\s*알려|"
        r"(?:와|과).{0,30}(?:관계|차이|비교)|"
        r"(?:대상|기준|방법|절차|서류|금액|시점).{0,25}(?:와|과|그리고).{0,25}(?:대상|기준|방법|절차|서류|금액|시점)",
        re.I,
    )

    def matched_terms_v31(text: str, mapping: dict[str, list[str]]) -> dict[str, list[str]]:
        lowered = (text or "").lower()
        return {
            business: [term for term in terms if term.lower() in lowered]
            for business, terms in mapping.items()
            if any(term.lower() in lowered for term in terms)
        }

    def decomposition_status_v31(original: str, needs: list[dict[str, Any]]) -> str:
        if not needs or any(not (need.get("query") or "").strip() for need in needs):
            return "PARTIAL"
        if len(needs) == 1 and MULTI_DECOMPOSITION_SIGNAL_V31.search(original):
            return "PARTIAL"
        return "COMPLETE"

    def apply_business_and_intent_rules_v31(
        needs: list[dict[str, Any]], original: str
    ) -> tuple[list[dict[str, Any]], list[str]]:
        actions = []
        need_count = len(needs)
        for need in needs:
            query = need.get("query") or original
            explicit_businesses = find_businesses(query)
            # V3의 오류 원인이었던 복합 질문 원문 업무의 전체 need 전파를 금지한다.
            if not explicit_businesses and need_count == 1:
                explicit_businesses = find_businesses(original)
            if len(explicit_businesses) == 1 and explicit_businesses[0] != need.get("business_function"):
                need["model_business_function"] = need.get("business_function")
                need["business_function"] = explicit_businesses[0]
                need["business_source"] = "ORIGINAL"
                actions.append(f"{need['need_id']}_BUSINESS_ORIGINAL_OVERRIDE")

            model_intent = need.get("model_intent")
            final_intent, source, pattern = resolve_intent_v3(query, model_intent)
            # V3 FULL에서 N2 override는 개선 0, 회귀 3이었다. 알려진 모델 intent는 보존한다.
            if need.get("need_id") != "N1" and source == "RULE_OVERRIDE" and model_intent in INTENTS:
                final_intent = model_intent
                source = "RULE_CONFLICT_MODEL_KEPT_V31"
                actions.append(f"{need['need_id']}_RULE_OVERRIDE_BLOCKED_V31")
            need["intent"] = final_intent
            need["intent_source"] = source
            need["intent_rule_pattern"] = pattern
            if source in {
                "RULE_OVERRIDE", "RULE_FILLED_UNKNOWN", "RULE_CONFLICT_MODEL_KEPT",
                "RULE_CONFLICT_MODEL_KEPT_V31", "RULE_AMBIGUOUS_MODEL_KEPT", "RULE_AMBIGUOUS_UNKNOWN",
            }:
                actions.append(f"{need['need_id']}_{source}")

        deduped, seen = [], set()
        for need in needs:
            key = (
                need.get("business_function"), need.get("intent"),
                normalize_query(need.get("query") or "")["normalized_query"],
                need.get("target_type"), tuple(sorted(need.get("case_details") or [])),
            )
            if key in seen:
                actions.append("EXACT_DUPLICATE_NEED_COLLAPSED")
                continue
            seen.add(key)
            deduped.append(need)
        for index, need in enumerate(deduped, 1):
            need["need_id"] = f"N{index}"
        return deduped, list(dict.fromkeys(actions))

    def annotate_business_safety_v31(
        needs: list[dict[str, Any]], original: str
    ) -> list[dict[str, Any]]:
        decomposition = decomposition_status_v31(original, needs)
        need_count = len(needs)
        original_candidates = find_businesses(original)
        for need in needs:
            query = need.get("query") or original
            business = need.get("business_function")
            source = need.get("business_source") or "UNKNOWN"
            query_candidates = find_businesses(query)
            candidates = query_candidates or (original_candidates if need_count == 1 else [])
            candidates = list(dict.fromkeys(candidates))
            strong_matches = matched_terms_v31(query, STRONG_BUSINESS_KEYWORDS_V31)
            weak_matches = matched_terms_v31(query, WEAK_OR_CROSS_BUSINESS_TERMS_V31)
            if not strong_matches and need_count == 1:
                strong_matches = matched_terms_v31(original, STRONG_BUSINESS_KEYWORDS_V31)
            if not weak_matches and need_count == 1:
                weak_matches = matched_terms_v31(original, WEAK_OR_CROSS_BUSINESS_TERMS_V31)

            model_business = need.get("model_business_function")
            model_rule_conflict = bool(model_business and business and model_business != business)
            cross_business_ambiguity = len(candidates) > 1
            strong_for_business = business in strong_matches

            if source == "MANUAL":
                confidence = 1.0
            elif source == "CONTEXT":
                confidence = 0.995
            elif strong_for_business and len(candidates) == 1:
                confidence = 0.99
            elif business in candidates and len(candidates) == 1:
                confidence = 0.90
            elif business:
                confidence = 0.70
            else:
                confidence = 0.0
            candidate_margin = 1.0 if len(candidates) <= 1 else 0.0

            denial_reasons = []
            if decomposition != "COMPLETE": denial_reasons.append("INCOMPLETE_DECOMPOSITION")
            if not business: denial_reasons.append("NO_BUSINESS_CANDIDATE")
            if len(candidates) > 1: denial_reasons.append("MULTIPLE_BUSINESS_CANDIDATES")
            if model_rule_conflict: denial_reasons.append("MODEL_RULE_CONFLICT")
            if cross_business_ambiguity: denial_reasons.append("CROSS_BUSINESS_AMBIGUITY")
            if source not in {"MANUAL", "CONTEXT"} and not strong_for_business:
                denial_reasons.append("NO_STRONG_EXPLICIT_EVIDENCE")
            if confidence < HARD_BUSINESS_CONFIDENCE_THRESHOLD:
                denial_reasons.append("LOW_BUSINESS_CONFIDENCE")
            if candidate_margin < HARD_BUSINESS_MARGIN_THRESHOLD:
                denial_reasons.append("LOW_CANDIDATE_MARGIN")

            need["business_candidates"] = [
                {
                    "value": candidate,
                    "confidence": 0.99 if candidate in strong_matches else 0.80,
                    "strong_evidence": strong_matches.get(candidate, []),
                    "weak_evidence": weak_matches.get(candidate, []),
                }
                for candidate in candidates
            ]
            need["decomposition_status"] = decomposition
            need["model_rule_conflict"] = model_rule_conflict
            need["cross_business_ambiguity"] = cross_business_ambiguity
            need["business_confidence"] = confidence
            need["business_candidate_margin"] = candidate_margin
            need["hard_filter_eligible"] = not denial_reasons
            need["hard_filter_denial_reasons"] = list(dict.fromkeys(denial_reasons))
        return needs
    ''',
)

pipeline_source = replace_block(
    pipeline_source,
    "def build_query_plans_v3",
    "class KDICLightweightRAGAnalyzerV3",
    r'''
    def build_query_plans_v31(analysis: dict[str, Any], config: PipelineConfig) -> list[dict[str, Any]]:
        if analysis["route"] not in {"RETRIEVE", "RETRIEVE_RELAXED"}:
            return []
        relaxed = analysis["route"] == "RETRIEVE_RELAXED"
        plans = []
        for need in analysis["needs"]:
            business, source = need.get("business_function"), need.get("business_source")
            eligible = bool(need.get("hard_filter_eligible"))
            denial_reasons = need.get("hard_filter_denial_reasons") or []
            if relaxed or not business:
                business_filter = {
                    "mode": "NONE", "value": None, "soft_hint": business,
                    "candidates": need.get("business_candidates") or [],
                    "evidence": source or "UNKNOWN", "denial_reasons": denial_reasons,
                }
                fallback_chain = []
            elif eligible:
                business_filter = {
                    "mode": "HARD", "value": business, "soft_hint": None,
                    "candidates": need.get("business_candidates") or [],
                    "evidence": source, "denial_reasons": [],
                }
                fallback_chain = ["SOFT", "NONE"]
            else:
                business_filter = {
                    "mode": "SOFT", "value": None, "soft_hint": business,
                    "candidates": need.get("business_candidates") or [],
                    "evidence": source or "UNKNOWN", "denial_reasons": denial_reasons,
                }
                fallback_chain = ["NONE"]
            intent = need.get("intent")
            intent_weight = 0.20 if need.get("intent_source") in {
                "RULE_OVERRIDE", "RULE_CONFIRMED", "RULE_FILLED_UNKNOWN"
            } else config.intent_soft_boost
            plans.append({
                "need_id": need["need_id"],
                "retrieval_mode": "RELAXED" if relaxed else "STANDARD",
                "semantic_query": need.get("query"),
                "keyword_query": build_keyword_query_v3(need) or need.get("query"),
                "business_filter": business_filter,
                "filter_safety": {
                    "hard_filter_eligible": eligible,
                    "decomposition_status": need.get("decomposition_status"),
                    "model_rule_conflict": need.get("model_rule_conflict"),
                    "cross_business_ambiguity": need.get("cross_business_ambiguity"),
                    "business_confidence": need.get("business_confidence"),
                    "candidate_margin": need.get("business_candidate_margin"),
                    "denial_reasons": denial_reasons,
                },
                "fallback_policy": {
                    "enabled": bool(fallback_chain),
                    "on": ["NO_RESULTS", "LOW_TOP_SCORE", "LOW_COVERAGE"],
                    "next_filter_modes": fallback_chain,
                    "fail_open": True,
                },
                "intent_boost": {
                    "mode": "SOFT" if intent in INTENTS else "NONE", "value": intent,
                    "weight": intent_weight if intent in INTENTS else 0.0,
                    "evidence": need.get("intent_source"),
                },
                "entities": {
                    "user_role": need.get("user_role"), "user_role_source": need.get("user_role_source"),
                    "applicant_type": need.get("applicant_type"), "applicant_type_source": need.get("applicant_type_source"),
                    "target_type": need.get("target_type"), "case_details": need.get("case_details") or [],
                },
            })
        return plans

    def relax_query_plan_v31(plan: dict[str, Any], reason: str) -> dict[str, Any]:
        """검색기가 결과 부족 시 호출할 수 있는 fail-open helper."""
        relaxed_plan = json.loads(json.dumps(plan, ensure_ascii=False))
        current_mode = relaxed_plan.get("business_filter", {}).get("mode")
        if current_mode == "HARD":
            relaxed_plan["business_filter"]["mode"] = "SOFT"
            relaxed_plan["business_filter"]["soft_hint"] = relaxed_plan["business_filter"].get("value")
            relaxed_plan["business_filter"]["value"] = None
        elif current_mode == "SOFT":
            relaxed_plan["business_filter"]["mode"] = "NONE"
            relaxed_plan["retrieval_mode"] = "RELAXED"
        relaxed_plan.setdefault("fallback_history", []).append({"from": current_mode, "reason": reason})
        return relaxed_plan
    ''',
)

pipeline_source = pipeline_source.replace("class KDICLightweightRAGAnalyzerV3", "class KDICLightweightRAGAnalyzerV31")
pipeline_source = pipeline_source.replace("apply_business_and_intent_rules_v3(needs, original)", "apply_business_and_intent_rules_v31(needs, original)")
pipeline_source = pipeline_source.replace(
    "needs = enrich_evidence_v3(needs, original, context, manual)\n        route, gate_reasons, blocking_slot",
    "needs = enrich_evidence_v3(needs, original, context, manual)\n"
    "        needs = annotate_business_safety_v31(needs, original)\n"
    "        route, gate_reasons, blocking_slot",
)
pipeline_source = pipeline_source.replace("build_query_plans_v3(analysis, self.config)", "build_query_plans_v31(analysis, self.config)")
pipeline_source = pipeline_source.replace("KDICLightweightRAGAnalyzerV3(HCX_CLIENT_V3", "KDICLightweightRAGAnalyzerV31(HCX_CLIENT_V3")
pipeline_source = pipeline_source.replace('print("v3 파이프라인 준비 완료:"', 'print("v3.1 파이프라인 준비 완료:"')
cells[7]["source"] = pipeline_source

cells[8]["source"] = textwrap.dedent(
    """
    ## v3.1 빠른 기능 확인

    `filter_safety`, `business_candidates`, `hard_filter_denial_reasons`를 확인합니다.
    Q026·Q074·Q083에서는 잘못된 `HARD`가 생성되지 않아야 합니다.
    """
).strip() + "\n"

quick_test_source = source_text(cells[9])
quick_test_source = quick_test_source.replace(
    '"개인회생과 워크아웃의 신청 조건과 변제 방식 차이를 알려주세요.",',
    '"개인회생과 워크아웃의 신청 조건과 변제 방식 차이를 알려주세요.",\n'
    '    "보호받기 위해 사전 신청이 필요한지와 실제 보험사고가 나면 무엇을 해야 하는지 알려주세요.",\n'
    '    "가지급금과 미수령금은 어떤 관계인가요?",\n'
    '    "가지급금과 예금보험금이 미수령금으로 남는 경우를 각각 설명해 주세요.",',
)
cells[9]["source"] = quick_test_source

evaluation_source = source_text(cells[13])
evaluation_source = evaluation_source.replace("kdic_lightweight_rag_v3_eval_raw.jsonl", "kdic_lightweight_rag_v3_1_eval_raw.jsonl")
evaluation_source = evaluation_source.replace(
    "records = []",
    textwrap.dedent(
        r'''
        def query_plan_valid_v31(plan):
            return bool((plan.get("semantic_query") or "").strip() and (plan.get("keyword_query") or "").strip())

        def hard_filter_failures_v31(gold, result):
            gold_pairs = gold.get("gold_request_pairs") or []
            gold_businesses = {pair[0] for pair in gold_pairs}
            failures = []
            for index, plan in enumerate(result.get("query_plans") or []):
                business_filter = plan.get("business_filter") or {}
                if business_filter.get("mode") != "HARD":
                    continue
                value = business_filter.get("value")
                if not value or value not in gold_businesses:
                    failures.append({
                        "need_id": plan.get("need_id"), "filter_value": value,
                        "gold_businesses": sorted(gold_businesses), "reason": "GOLD_BUSINESS_EXCLUDED",
                    })
            return failures

        records = []
        '''
    ).strip(),
)
evaluation_source = evaluation_source.replace(
    'runtime = result["runtime"]\n        record = {',
    'runtime = result["runtime"]\n'
    '        plan_valid = bool(result["query_plans"]) and all(query_plan_valid_v31(p) for p in result["query_plans"]) if pred_route == "RETRIEVE" else True\n'
    '        hard_filter_failures = hard_filter_failures_v31(gold, result) if pred_route == "RETRIEVE" else []\n'
    '        false_oos = gold["gold_route"] != "OUT_OF_SCOPE" and pred_route == "OUT_OF_SCOPE"\n'
    '        false_direct = gold["gold_route"] != "DIRECT_RESPONSE" and pred_route == "DIRECT_RESPONSE"\n'
    '        record = {',
)
evaluation_source = evaluation_source.replace(
    '"warning_count": len(result.get("validation_warnings") or []),',
    '"warning_count": len(result.get("validation_warnings") or []),\n'
    '            "query_plan_valid": plan_valid, "hard_filter_count": sum(\n'
    '                1 for p in result.get("query_plans") or [] if (p.get("business_filter") or {}).get("mode") == "HARD"\n'
    '            ),\n'
    '            "wrong_hard_filter_count": len(hard_filter_failures),\n'
    '            "hard_filter_failures": hard_filter_failures,\n'
    '            "false_oos": false_oos, "false_direct": false_direct,',
)
evaluation_source = evaluation_source.replace(
    '"rule_regressed_pair_count": int(result_df.rule_regressed_pair.sum()),',
    '"rule_regressed_pair_count": int(result_df.rule_regressed_pair.sum()),\n'
    '        "retrieve_query_plan_valid_rate": float(result_df[result_df.pred_route == "RETRIEVE"].query_plan_valid.mean()),\n'
    '        "false_oos_count": int(result_df.false_oos.sum()),\n'
    '        "false_direct_count": int(result_df.false_direct.sum()),\n'
    '        "hard_filter_count": int(result_df.hard_filter_count.sum()),\n'
    '        "wrong_hard_filter_count": int(result_df.wrong_hard_filter_count.sum()),',
)
evaluation_source = evaluation_source.replace(
    'print(json.dumps(summary, ensure_ascii=False, indent=2))',
    'summary["hard_gate_pass"] = bool(\n'
    '        summary["analysis_success_rate"] >= 0.995\n'
    '        and summary["retrieve_query_plan_valid_rate"] >= 0.99\n'
    '        and summary["false_oos_count"] == 0\n'
    '        and summary["false_direct_count"] == 0\n'
    '        and summary["wrong_hard_filter_count"] == 0\n'
    '        and summary["clarify_precision"] >= 0.95\n'
    '        and summary["clarify_recall"] >= 0.95\n'
    '    )\n'
    'print(json.dumps(summary, ensure_ascii=False, indent=2))',
)
cells[13]["source"] = evaluation_source

export_source = source_text(cells[14])
export_source = export_source.replace("kdic_lightweight_rag_v3_", "kdic_lightweight_rag_v3_1_")
export_source = export_source.replace(
    'status_path = Path("kdic_lightweight_rag_v3_1_eval_status.csv")',
    'status_path = Path("kdic_lightweight_rag_v3_1_eval_status.csv")\n'
    'hard_gate_failure_path = Path("kdic_lightweight_rag_v3_1_hard_gate_failures.csv")',
)
export_source = export_source.replace(
    'status_table.to_csv(status_path, index=False, encoding="utf-8-sig")',
    'status_table.to_csv(status_path, index=False, encoding="utf-8-sig")\n'
    'result_df[(~result_df.query_plan_valid) | result_df.false_oos | result_df.false_direct | '
    '(result_df.wrong_hard_filter_count > 0)].to_csv(hard_gate_failure_path, index=False, encoding="utf-8-sig")',
)
export_source = export_source.replace(
    'error_type_path, status_path, raw_path,',
    'error_type_path, status_path, hard_gate_failure_path, raw_path,',
)
cells[14]["source"] = export_source

cells[15]["source"] = textwrap.dedent(
    """
    ## v3.1 FULL 실행 하드 게이트

    - 실행 성공률 ≥ 99.5%
    - RETRIEVE 검색계획 유효율 ≥ 99%
    - 정상 질문의 False OOS = 0
    - 정상 질문의 False DIRECT = 0
    - 정답 업무를 제외하는 Wrong Hard Filter = 0
    - CLARIFY Precision ≥ 95%
    - CLARIFY Recall ≥ 95%

    하드 게이트 통과 후 Core Exact, Request Pair F1, MULTI Core Exact, 토큰과 지연시간을
    V3와 비교합니다. `HARD` 적용률은 안전성 게이트가 아니라 검색 효율 최적화 지표로 별도 보고합니다.
    """
).strip() + "\n"

notebook.setdefault("metadata", {})["kdic_pipeline_version"] = "v3.1"
notebook["metadata"]["hard_filter_policy"] = "ALLOW_ONLY_IF_ALL_SAFETY_CHECKS_PASS"
OUTPUT_NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")

missing_release_files = [path for path in (DIAGRAM, VALIDATOR) if not path.exists()]
if missing_release_files:
    print("노트북 생성 완료. 배포 ZIP은 다음 파일 생성 후 빌더를 다시 실행하면 생성됩니다:")
    for path in missing_release_files:
        print(" -", path)
else:
    with zipfile.ZipFile(RELEASE_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in (OUTPUT_NOTEBOOK, DIAGRAM, Path(__file__).resolve(), VALIDATOR):
            archive.write(path, arcname=path.name)
    print("생성:", RELEASE_ZIP)

print("생성:", OUTPUT_NOTEBOOK)
