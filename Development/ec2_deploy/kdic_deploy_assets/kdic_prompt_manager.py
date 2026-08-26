from __future__ import annotations

"""Versioned answer-prompt management for the KDIC administrator UI."""

import copy
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Mapping


PROMPT_SCHEMA_VERSION = "c-cross-direct-dc-twocall-v1"
LEGACY_C_PROMPT_SLOT = "C_STRUCTURED_SYSTEM_PROMPT_V3"


PROMPT_SLOTS: dict[str, dict[str, str]] = {
    "C_CROSS_DIRECT_SYSTEM_PROMPT_V1": {
        "label": "C안 기본·직접 답변",
        "route": "C 1Call",
        "description": "최신 C안 Evidence Pack을 바탕으로 기본 답변과 비교용 직접 답변을 생성합니다.",
    },
    "DC_SKELETON_SYSTEM_PROMPT_V1": {
        "label": "교차업무 답변 골격",
        "route": "D-C 2Call · 1차",
        "description": "여러 업무가 섞인 질문에서 답변 항목과 사용할 근거를 먼저 구조화합니다.",
    },
    "DC_FINAL_SYSTEM_PROMPT_V1": {
        "label": "교차업무 최종 답변",
        "route": "D-C 2Call · 2차",
        "description": "검증된 골격과 선택 근거로 사용자에게 보여줄 최종 답변을 작성합니다.",
    },
}

SECRET_PATTERN = re.compile(r"(?i)(?:nv-|sk-)[A-Za-z0-9_-]{16,}")


def _validate_values(raw: Mapping[str, Any]) -> dict[str, str]:
    unknown = set(raw) - set(PROMPT_SLOTS)
    if unknown:
        raise ValueError("관리 대상이 아닌 프롬프트가 포함되어 있습니다: " + ", ".join(sorted(unknown)))
    values: dict[str, str] = {}
    for slot in PROMPT_SLOTS:
        value = str(raw.get(slot) or "").strip()
        if len(value) < 100:
            raise ValueError(f"{PROMPT_SLOTS[slot]['label']}: 프롬프트는 100자 이상이어야 합니다.")
        if len(value) > 30_000:
            raise ValueError(f"{PROMPT_SLOTS[slot]['label']}: 프롬프트는 30,000자 이하여야 합니다.")
        if SECRET_PATTERN.search(value):
            raise ValueError(f"{PROMPT_SLOTS[slot]['label']}: API 키로 보이는 값이 포함되어 있습니다.")
        values[slot] = value
    return values


def _migrate_values(raw: Mapping[str, Any], defaults: Mapping[str, Any]) -> dict[str, Any]:
    source = dict(raw or {})
    unknown = set(source) - set(PROMPT_SLOTS) - {LEGACY_C_PROMPT_SLOT}
    if unknown:
        raise ValueError("관리 대상이 아닌 프롬프트가 포함되어 있습니다: " + ", ".join(sorted(unknown)))
    return {
        slot: source.get(slot) or defaults.get(slot)
        for slot in PROMPT_SLOTS
    }


class PromptManager:
    def __init__(self, path: str | Path, defaults: Mapping[str, Any]):
        self.path = Path(path).resolve()
        self.lock = threading.RLock()
        self.defaults = _validate_values(defaults)
        self.state = self._load()

    def _default_payload(self) -> dict[str, Any]:
        return {
            "schema_version": PROMPT_SCHEMA_VERSION,
            "active_version": "prompt-runtime-default-v1",
            "active_values": copy.deepcopy(self.defaults),
            "draft_values": copy.deepcopy(self.defaults),
            "history": [],
            "updated_at": time.time(),
        }

    def _load(self) -> dict[str, Any]:
        if not self.path.is_file():
            payload = self._default_payload()
            self._write(payload)
            return payload
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            raw_active = dict(payload.get("active_values") or self.defaults)
            raw_draft = dict(payload.get("draft_values") or raw_active)
            active_values = _validate_values(_migrate_values(raw_active, self.defaults))
            draft_values = _validate_values(_migrate_values(raw_draft, active_values))
            migrated = (
                payload.get("schema_version") != PROMPT_SCHEMA_VERSION
                or raw_active != active_values
                or raw_draft != draft_values
            )
            if migrated and (LEGACY_C_PROMPT_SLOT in raw_active or LEGACY_C_PROMPT_SLOT in raw_draft):
                legacy_values = dict(payload.get("legacy_values") or {})
                legacy_values.setdefault(
                    LEGACY_C_PROMPT_SLOT,
                    {
                        "active": raw_active.get(LEGACY_C_PROMPT_SLOT),
                        "draft": raw_draft.get(LEGACY_C_PROMPT_SLOT),
                        "migrated_at": time.time(),
                    },
                )
                payload["legacy_values"] = legacy_values
            payload["schema_version"] = PROMPT_SCHEMA_VERSION
            payload["active_values"] = active_values
            payload["draft_values"] = draft_values
            payload["history"] = list(payload.get("history") or [])[-20:]
            if migrated:
                payload["schema_migrated_at"] = time.time()
                self._write(payload)
            return payload
        except Exception:
            backup = self.path.with_suffix(self.path.suffix + f".invalid-{int(time.time())}")
            os.replace(self.path, backup)
            payload = self._default_payload()
            self._write(payload)
            return payload

    def _write(self, payload: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, self.path)

    def active_values(self) -> dict[str, str]:
        with self.lock:
            return copy.deepcopy(self.state["active_values"])

    def draft_values(self) -> dict[str, str]:
        with self.lock:
            return copy.deepcopy(self.state["draft_values"])

    def public(self) -> dict[str, Any]:
        with self.lock:
            active = self.state["active_values"]
            draft = self.state["draft_values"]
            return {
                "schema_version": self.state.get("schema_version", PROMPT_SCHEMA_VERSION),
                "active_version": self.state["active_version"],
                "updated_at": self.state["updated_at"],
                "has_changes": active != draft,
                "slots": [
                    {
                        "slot": slot,
                        **meta,
                        "active": active[slot],
                        "draft": draft[slot],
                        "changed": active[slot] != draft[slot],
                    }
                    for slot, meta in PROMPT_SLOTS.items()
                ],
                "history": copy.deepcopy(self.state.get("history") or []),
            }

    def save_draft(self, values: Mapping[str, Any]) -> dict[str, Any]:
        validated = _validate_values(values)
        with self.lock:
            self.state["draft_values"] = validated
            self.state["updated_at"] = time.time()
            self._write(self.state)
            return self.public()

    def clear_draft(self) -> dict[str, Any]:
        with self.lock:
            self.state["draft_values"] = copy.deepcopy(self.state["active_values"])
            self.state["updated_at"] = time.time()
            self._write(self.state)
            return self.public()

    def apply_draft(self) -> dict[str, Any]:
        with self.lock:
            previous = {
                "version": self.state["active_version"],
                "values": copy.deepcopy(self.state["active_values"]),
                "archived_at": time.time(),
            }
            self.state["history"] = (list(self.state.get("history") or []) + [previous])[-20:]
            self.state["active_version"] = f"prompt-{time.strftime('%Y%m%d-%H%M%S')}"
            self.state["active_values"] = copy.deepcopy(self.state["draft_values"])
            self.state["updated_at"] = time.time()
            self._write(self.state)
            return self.public()

    def activate(
        self,
        values: Mapping[str, Any],
        *,
        version: str | None = None,
        archive_current: bool = True,
    ) -> dict[str, Any]:
        """Activate validated values for a unified administrator release."""
        validated = _validate_values(_migrate_values(values, self.defaults))
        with self.lock:
            if archive_current:
                previous = {
                    "version": self.state["active_version"],
                    "values": copy.deepcopy(self.state["active_values"]),
                    "archived_at": time.time(),
                }
                self.state["history"] = (list(self.state.get("history") or []) + [previous])[-20:]
            self.state["active_version"] = version or f"prompt-{time.strftime('%Y%m%d-%H%M%S')}"
            self.state["active_values"] = copy.deepcopy(validated)
            self.state["draft_values"] = copy.deepcopy(validated)
            self.state["updated_at"] = time.time()
            self._write(self.state)
            return self.public()

    def rollback(self, version: str) -> dict[str, Any]:
        with self.lock:
            row = next((item for item in self.state.get("history") or [] if item.get("version") == version), None)
            if row is None:
                raise KeyError(version)
            restored = _validate_values(_migrate_values(row.get("values") or {}, self.defaults))
            current = {
                "version": self.state["active_version"],
                "values": copy.deepcopy(self.state["active_values"]),
                "archived_at": time.time(),
            }
            self.state["history"] = [item for item in self.state["history"] if item.get("version") != version]
            self.state["history"] = (self.state["history"] + [current])[-20:]
            self.state["active_version"] = str(row["version"])
            self.state["active_values"] = restored
            self.state["draft_values"] = copy.deepcopy(restored)
            self.state["updated_at"] = time.time()
            self._write(self.state)
            return self.public()


def build_prompt_manager(runtime_globals: Mapping[str, Any]) -> PromptManager:
    defaults = {slot: runtime_globals.get(slot) for slot in PROMPT_SLOTS}
    path = os.getenv("KDIC_PROMPT_PATH", "/opt/kdic/runtime/admin_prompts.json")
    return PromptManager(path, defaults)
