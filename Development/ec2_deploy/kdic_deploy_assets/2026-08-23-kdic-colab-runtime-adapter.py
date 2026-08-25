from __future__ import annotations

from typing import Any, Callable, Mapping, MutableMapping


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


class LatestKDICNotebookAdapter:
    """Bridge the final notebook globals to the shared FastAPI service contract.

    The notebook owns query analysis, evidence gates and answer generation. The
    adapter calls one production entrypoint whose routing policy is deliberately
    narrow:

    * every normal question -> C 1Call
    * valid multi-business COMPARE -> D-C 2Call
    * D-C 1Call -> disabled
    * CLARIFY / OOS / DIRECT -> route response without answer LLM calls
    """

    name = "V1.5_C_DEFAULT_DC2_COMPARE_ONLY"
    build_info = {
        "source_notebook": (
            "2026-08-21-KDIC-D-교차업무전용-1Call-vs-2Call-v1_3-Colab.ipynb"
        ),
        "routing_policy": "C_DEFAULT_DC2_COMPARE_ONLY_V1",
        "default_variant": "C_1CALL",
        "comparison_variant": "DC_2CALL",
        "dc1_enabled": False,
        "build_sha256": "F9A908D62A43EA3A3566A5D8DF0E982F214373FFF96470A749DC1EFE79E25083",
        "overlay_file": "2026-08-25-kdic-production-overlay.py",
        "overlay_revision": "2026-08-26-declared-citation-canonicalization-v12",
        "cache_compatible_overlay_revisions": [
            "2026-08-26-explicit-allowed-citation-ids-v11",
            "2026-08-26-declared-citation-canonicalization-v12",
        ],
        "adapter_version": "2026-08-26-ec2-production-v13",
        "suggestion_cache_schema": "kdic-suggestion-answer-bundle-v5.0",
    }

    def __init__(self, runtime_globals: Mapping[str, Any]):
        self.runtime = runtime_globals
        self._validate_runtime()

    def _required_callable(self, name: str) -> Callable[..., Any]:
        value = self.runtime.get(name)
        if not callable(value):
            raise RuntimeError(
                f"현재 Colab 런타임에서 {name}()을 찾지 못했습니다. "
                "최신 KDIC 질의분석·검색·답변 셀을 먼저 실행하세요."
            )
        return value

    def _validate_runtime(self) -> None:
        has_production_policy = callable(self.runtime.get("execute_production_variant_v1"))
        has_c_compatibility = callable(self.runtime.get("execute_bcd_variant_v3"))
        has_b_fallback = callable(self.runtime.get("run_fixed_pipeline"))
        if not has_production_policy and not has_c_compatibility and not has_b_fallback:
            raise RuntimeError(
                "연결 가능한 KDIC 실행 함수를 찾지 못했습니다. "
                "execute_production_variant_v1, execute_bcd_variant_v3 또는 "
                "run_fixed_pipeline이 필요합니다."
            )

    def _with_build_info(self, result: Mapping[str, Any]) -> dict[str, Any]:
        output = dict(result)
        notebook_build = output.get("runtime_build")
        if not isinstance(notebook_build, Mapping):
            notebook_build = {}
        output["runtime_build"] = {**dict(notebook_build), **dict(self.build_info)}
        output["routing_policy"] = self.build_info["routing_policy"]
        return output

    def _new_holder(self) -> dict[str, Any]:
        for factory_name in (
            "new_dc_controller_state_v1",
            "new_bcd_controller_state_c1",
            "new_comparison_state",
        ):
            factory = self.runtime.get(factory_name)
            if callable(factory):
                value = factory()
                if isinstance(value, Mapping):
                    return dict(value)
        return {
            "conversation": {"turns": []},
            "current_question": "",
            "common": None,
            "answer_cache": {},
            "committed": False,
            "committed_variant": None,
            "events": [],
        }

    def _holder(self, state: MutableMapping[str, Any]) -> dict[str, Any]:
        holder = state.get("_kdic_controller")
        runtime_revision = str(self.build_info["overlay_revision"])
        if (
            not isinstance(holder, dict)
            or holder.get("_runtime_revision") != runtime_revision
        ):
            holder = self._new_holder()
            holder["_runtime_revision"] = runtime_revision
            state["_kdic_controller"] = holder
        return holder

    def _apply_output_guardrails(self, result: Mapping[str, Any]) -> dict[str, Any]:
        output = dict(result)
        manager = self.runtime.get("KDIC_GUARDRAIL_MANAGER")
        if manager is None:
            return output
        audits = []
        for key in ("answer", "display_answer", "basic_answer", "route_message"):
            if not isinstance(output.get(key), str):
                continue
            audit = manager.evaluate(output[key], "output")
            output[key] = audit["text"]
            audits.extend(audit["hits"])
        payload = output.get("payload")
        if isinstance(payload, Mapping) and isinstance(payload.get("answer"), str):
            payload_copy = dict(payload)
            audit = manager.evaluate(payload_copy["answer"], "output")
            payload_copy["answer"] = audit["text"]
            output["payload"] = payload_copy
            audits.extend(audit["hits"])
        output["guardrail_audit"] = {
            "version": manager.public()["active_version"],
            "output_hit_rule_ids": list(dict.fromkeys(row["rule_id"] for row in audits)),
        }
        return output

    def configure(self, api_key: str) -> None:
        configure = self.runtime.get("_configure_hcx_runtime")
        if callable(configure):
            configure(api_key)
            return
        current = _clean(self.runtime.get("HCX_API_KEY"))
        if current:
            # The notebook already constructed the embedding/decomposition/answer
            # clients with this key, so there is nothing to rebuild.
            return
        raise RuntimeError(
            "현재 노트북은 실행 중 API 키 재설정을 지원하지 않습니다. "
            "HCX 키 설정 셀을 먼저 실행한 뒤 API를 연결하세요."
        )

    def record_cached_turn(
        self,
        question: str,
        answer: str,
        state: MutableMapping[str, Any],
    ) -> None:
        """Commit a cached standalone answer to the notebook conversation state."""

        holder = self._holder(state)
        conversation = holder.get("conversation")
        if not isinstance(conversation, dict):
            conversation = {"turns": []}
            holder["conversation"] = conversation
        turns = conversation.get("turns")
        if not isinstance(turns, list):
            turns = []
            conversation["turns"] = turns
        turns.extend(
            [
                {"role": "user", "content": _clean(question)},
                {"role": "assistant", "content": str(answer or "").strip()},
            ]
        )
        holder["current_question"] = _clean(question)
        holder["common"] = None
        holder["answer_cache"] = {}
        holder["committed"] = True
        holder["committed_variant"] = "SUGGESTION_ANSWER_CACHE"

    def __call__(
        self,
        question: str,
        state: MutableMapping[str, Any],
        progress: Callable[[int, str], None] | None = None,
    ) -> Mapping[str, Any]:
        lock = self.runtime.get("KDIC_RUNTIME_EXECUTION_LOCK")
        if lock is None:
            return self._execute(question, state, progress)
        with lock:
            return self._execute(question, state, progress)

    def _execute(
        self,
        question: str,
        state: MutableMapping[str, Any],
        progress: Callable[[int, str], None] | None = None,
    ) -> Mapping[str, Any]:
        progress = progress or (lambda *_: None)
        holder = self._holder(state)
        progress(10, "질문의 문맥과 업무를 확인하고 있습니다.")

        execute_production = self.runtime.get("execute_production_variant_v1")
        if callable(execute_production):
            progress(25, "C 기본·비교형 D-C 2Call 정책으로 근거를 확인하고 있습니다.")
            result = execute_production(question, holder)
            if not isinstance(result, Mapping):
                raise TypeError("KDIC 운영 정책 실행 결과가 dict가 아닙니다.")
            routing = result.get("production_routing")
            selected_variant = (
                _clean(routing.get("selected_variant")).upper()
                if isinstance(routing, Mapping)
                else _clean(result.get("variant")).upper()
            )
            if selected_variant == "DC_1CALL":
                raise RuntimeError("운영 정책에서 DC_1CALL은 비활성화되어 있습니다.")
            if selected_variant == "DC_2CALL":
                progress(63, "다중업무 비교 근거를 D-C 2Call로 구성하고 있습니다.")
            else:
                progress(63, "기본 C안으로 공식 근거 답변을 구성하고 있습니다.")
            progress(94, "공식 출처와 후속 행동 링크를 확인하고 있습니다.")
            return self._with_build_info(self._apply_output_guardrails(result))

        # Safe compatibility mode for a stale engine. It never guesses that a
        # question is cross-business and therefore never invokes DC1/DC2.
        execute_bcd = self.runtime.get("execute_bcd_variant_v3")
        if callable(execute_bcd):
            progress(35, "호환 모드에서 C안 근거 답변을 구성하고 있습니다.")
            result = execute_bcd("C", question, holder)
            if not isinstance(result, Mapping):
                raise TypeError("KDIC C안 호환 실행 결과가 dict가 아닙니다.")
            output = self._with_build_info(self._apply_output_guardrails(result))
            output["runtime_compatibility_mode"] = "C_ONLY_STALE_ENGINE"
            progress(94, "공식 출처와 후속 행동 링크를 확인하고 있습니다.")
            return output

        progress(35, "호환용 B 파이프라인을 실행하고 있습니다.")
        run_fixed = self._required_callable("run_fixed_pipeline")
        comparison_state = holder.get("conversation") or holder
        result = run_fixed("v15", question, state=comparison_state)
        if not isinstance(result, Mapping):
            raise TypeError("KDIC 호환 파이프라인 결과가 dict가 아닙니다.")
        progress(94, "공식 출처를 확인하고 있습니다.")
        output = self._with_build_info(self._apply_output_guardrails(result))
        output["runtime_compatibility_mode"] = "B_FALLBACK_STALE_ENGINE"
        return output


def build_latest_kdic_pipeline(
    runtime_globals: Mapping[str, Any],
) -> LatestKDICNotebookAdapter:
    return LatestKDICNotebookAdapter(runtime_globals)

