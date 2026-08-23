from __future__ import annotations

from typing import Any, Callable, Mapping, MutableMapping


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


class LatestKDICNotebookAdapter:
    """Bridge the final notebook globals to the shared FastAPI service contract.

    The adapter intentionally contains no query-analysis or answer-generation logic.
    It selects the already-defined notebook entry points:

    * non-cross-business RETRIEVE -> C
    * cross-business RETRIEVE -> D-C 2Call
    * CLARIFY / OOS / DIRECT -> route response without answer LLM calls
    """

    name = "V1.5_RELATIONAL_MULTITURN_C_OR_CROSS_DC_2CALL"

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
        has_final_policy = callable(self.runtime.get("execute_dc_variant_v1")) and callable(
            self.runtime.get("execute_bcd_variant_v3")
        )
        has_b_fallback = callable(self.runtime.get("run_fixed_pipeline"))
        if not has_final_policy and not has_b_fallback:
            raise RuntimeError(
                "연결 가능한 KDIC 실행 함수를 찾지 못했습니다. "
                "execute_dc_variant_v1 + execute_bcd_variant_v3 또는 run_fixed_pipeline이 필요합니다."
            )

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
        if not isinstance(holder, dict):
            holder = self._new_holder()
            state["_kdic_controller"] = holder
        return holder

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

    def __call__(
        self,
        question: str,
        state: MutableMapping[str, Any],
        progress: Callable[[int, str], None] | None = None,
    ) -> Mapping[str, Any]:
        progress = progress or (lambda *_: None)
        holder = self._holder(state)
        progress(10, "질문의 문맥과 업무를 확인하고 있습니다.")

        execute_dc = self.runtime.get("execute_dc_variant_v1")
        execute_bcd = self.runtime.get("execute_bcd_variant_v3")
        if callable(execute_dc) and callable(execute_bcd):
            progress(25, "교차업무 여부와 검색 계획을 확인하고 있습니다.")
            result = execute_dc("DC_2CALL", question, holder)
            route = _clean(result.get("route")).upper() if isinstance(result, Mapping) else ""
            if route == "C_POLICY_TARGET":
                progress(63, "단일·동일업무 질문을 C안 근거로 답변하고 있습니다.")
                result = execute_bcd("C", question, holder)
            elif route == "RETRIEVE":
                progress(63, "교차업무 근거를 D-C 2Call로 구성하고 있습니다.")
            else:
                progress(85, "추가 확인 또는 안내 범위 응답을 정리하고 있습니다.")
            if not isinstance(result, Mapping):
                raise TypeError("KDIC 최종 정책 실행 결과가 dict가 아닙니다.")
            progress(94, "공식 출처와 후속 행동 링크를 확인하고 있습니다.")
            return dict(result)

        progress(35, "호환용 B 파이프라인을 실행하고 있습니다.")
        run_fixed = self._required_callable("run_fixed_pipeline")
        comparison_state = holder.get("conversation") or holder
        result = run_fixed("v15", question, state=comparison_state)
        if not isinstance(result, Mapping):
            raise TypeError("KDIC 호환 파이프라인 결과가 dict가 아닙니다.")
        progress(94, "공식 출처를 확인하고 있습니다.")
        return dict(result)


def build_latest_kdic_pipeline(
    runtime_globals: Mapping[str, Any],
) -> LatestKDICNotebookAdapter:
    return LatestKDICNotebookAdapter(runtime_globals)

