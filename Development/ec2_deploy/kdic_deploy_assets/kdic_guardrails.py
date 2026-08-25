from __future__ import annotations

"""Versioned guardrails shared by the public chatbot and administrator UI."""

import copy
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_RULES: list[dict[str, Any]] = [
    {
        "rule_id": "pii-resident-number",
        "name": "주민등록번호 마스킹",
        "scope": "both",
        "action": "mask",
        "match_type": "regex",
        "pattern": r"(?<!\d)\d{6}\s*[- ]?\s*[1-4]\d{6}(?!\d)",
        "replacement": "[주민등록번호]",
        "enabled": True,
        "description": "질문과 답변에 포함된 주민등록번호 형식을 가립니다.",
    },
    {
        "rule_id": "pii-phone",
        "name": "휴대전화 번호 마스킹",
        "scope": "both",
        "action": "mask",
        "match_type": "regex",
        "pattern": r"(?<!\d)01[016789]\s*[- ]?\s*\d{3,4}\s*[- ]?\s*\d{4}(?!\d)",
        "replacement": "[전화번호]",
        "enabled": True,
        "description": "휴대전화 번호를 모델 입력 전과 최종 답변 출력 전에 가립니다.",
    },
    {
        "rule_id": "pii-email",
        "name": "이메일 주소 마스킹",
        "scope": "both",
        "action": "mask",
        "match_type": "regex",
        "pattern": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "replacement": "[이메일]",
        "enabled": True,
        "description": "이메일 주소를 모델 입력과 사용자 출력에서 가립니다.",
    },
    {
        "rule_id": "secret-api-key",
        "name": "API 키 형태 차단",
        "scope": "input",
        "action": "block",
        "match_type": "regex",
        "pattern": r"(?i)(?:nv-[A-Za-z0-9_-]{16,}|sk-[A-Za-z0-9_-]{16,})",
        "replacement": "",
        "enabled": True,
        "description": "실수로 API 키를 질문창에 입력하면 모델 호출 전에 차단합니다.",
    },
]

VALID_SCOPES = {"input", "output", "both"}
VALID_ACTIONS = {"mask", "block"}
VALID_MATCH_TYPES = {"literal", "regex"}


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _default_payload() -> dict[str, Any]:
    return {
        "active_version": "guardrail-default-v1",
        "active_rules": copy.deepcopy(DEFAULT_RULES),
        "draft_rules": copy.deepcopy(DEFAULT_RULES),
        "history": [],
        "updated_at": time.time(),
    }


def _validate_rule(raw: Mapping[str, Any]) -> dict[str, Any]:
    rule_id = _clean(raw.get("rule_id")) or f"rule-{uuid.uuid4().hex[:10]}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{2,79}", rule_id):
        raise ValueError("규칙 ID는 영문·숫자·하이픈·밑줄 3~80자로 입력해 주세요.")
    name = _clean(raw.get("name"))
    scope = _clean(raw.get("scope")).lower()
    action = _clean(raw.get("action")).lower()
    match_type = _clean(raw.get("match_type")).lower()
    pattern = str(raw.get("pattern") or "").strip()
    replacement = str(raw.get("replacement") or "").strip()
    if not name or len(name) > 100:
        raise ValueError(f"{rule_id}: 규칙 이름은 1~100자여야 합니다.")
    if scope not in VALID_SCOPES or action not in VALID_ACTIONS or match_type not in VALID_MATCH_TYPES:
        raise ValueError(f"{rule_id}: scope/action/match_type 값이 올바르지 않습니다.")
    if not pattern or len(pattern) > 300:
        raise ValueError(f"{rule_id}: 패턴은 1~300자여야 합니다.")
    if action == "mask" and (not replacement or len(replacement) > 100):
        raise ValueError(f"{rule_id}: 마스킹 규칙에는 1~100자의 대체 문구가 필요합니다.")
    if match_type == "regex":
        if re.search(r"(?:\([^)]*[+*][^)]*\))[+*{]", pattern):
            raise ValueError(f"{rule_id}: 과도한 반복을 유발할 수 있는 정규식은 사용할 수 없습니다.")
        try:
            re.compile(pattern)
        except re.error as error:
            raise ValueError(f"{rule_id}: 정규식 오류: {error}") from error
    return {
        "rule_id": rule_id,
        "name": name,
        "scope": scope,
        "action": action,
        "match_type": match_type,
        "pattern": pattern,
        "replacement": replacement if action == "mask" else "",
        "enabled": bool(raw.get("enabled", True)),
        "description": _clean(raw.get("description"))[:300],
    }


def _validate_rules(raw_rules: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(raw_rules) > 100:
        raise ValueError("가드레일 규칙은 최대 100개까지 등록할 수 있습니다.")
    rules = [_validate_rule(raw) for raw in raw_rules]
    ids = [rule["rule_id"] for rule in rules]
    if len(ids) != len(set(ids)):
        raise ValueError("중복된 규칙 ID가 있습니다.")
    return rules


class GuardrailManager:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.lock = threading.RLock()
        self.state = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.is_file():
            payload = _default_payload()
            self._write(payload)
            return payload
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            payload["active_rules"] = _validate_rules(payload.get("active_rules") or [])
            payload["draft_rules"] = _validate_rules(payload.get("draft_rules") or payload["active_rules"])
            payload["history"] = list(payload.get("history") or [])[-20:]
            return payload
        except Exception:
            backup = self.path.with_suffix(self.path.suffix + f".invalid-{int(time.time())}")
            os.replace(self.path, backup)
            payload = _default_payload()
            self._write(payload)
            return payload

    def _write(self, payload: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, self.path)

    def public(self) -> dict[str, Any]:
        with self.lock:
            return copy.deepcopy(self.state)

    def save_draft(self, rules: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        validated = _validate_rules(rules)
        with self.lock:
            self.state["draft_rules"] = validated
            self.state["updated_at"] = time.time()
            self._write(self.state)
            return self.public()

    def clear_draft(self) -> dict[str, Any]:
        with self.lock:
            self.state["draft_rules"] = copy.deepcopy(self.state["active_rules"])
            self.state["updated_at"] = time.time()
            self._write(self.state)
            return self.public()

    def apply_draft(self) -> dict[str, Any]:
        with self.lock:
            previous = {
                "version": self.state["active_version"],
                "rules": copy.deepcopy(self.state["active_rules"]),
                "archived_at": time.time(),
            }
            self.state["history"] = (list(self.state.get("history") or []) + [previous])[-20:]
            self.state["active_version"] = f"guardrail-{time.strftime('%Y%m%d-%H%M%S')}"
            self.state["active_rules"] = copy.deepcopy(self.state["draft_rules"])
            self.state["updated_at"] = time.time()
            self._write(self.state)
            return self.public()

    def rollback(self, version: str) -> dict[str, Any]:
        with self.lock:
            row = next((item for item in self.state.get("history") or [] if item.get("version") == version), None)
            if row is None:
                raise KeyError(version)
            current = {
                "version": self.state["active_version"],
                "rules": copy.deepcopy(self.state["active_rules"]),
                "archived_at": time.time(),
            }
            self.state["history"] = [item for item in self.state["history"] if item.get("version") != version]
            self.state["history"] = (self.state["history"] + [current])[-20:]
            self.state["active_version"] = str(row["version"])
            self.state["active_rules"] = copy.deepcopy(row["rules"])
            self.state["draft_rules"] = copy.deepcopy(row["rules"])
            self.state["updated_at"] = time.time()
            self._write(self.state)
            return self.public()

    @staticmethod
    def _matches(rule: Mapping[str, Any], text: str) -> bool:
        if rule["match_type"] == "literal":
            return str(rule["pattern"]).lower() in text.lower()
        return re.search(str(rule["pattern"]), text) is not None

    @staticmethod
    def _mask(rule: Mapping[str, Any], text: str) -> str:
        if rule["match_type"] == "literal":
            return re.sub(re.escape(str(rule["pattern"])), str(rule["replacement"]), text, flags=re.I)
        return re.sub(str(rule["pattern"]), str(rule["replacement"]), text)

    def evaluate(self, text: Any, scope: str, *, use_draft: bool = False) -> dict[str, Any]:
        value = str(text or "")
        scope = _clean(scope).lower()
        if scope not in {"input", "output"}:
            raise ValueError("scope은 input 또는 output이어야 합니다.")
        with self.lock:
            rules = copy.deepcopy(self.state["draft_rules" if use_draft else "active_rules"])
            version = "DRAFT" if use_draft else self.state["active_version"]
        hits, sanitized, blocked = [], value, False
        for rule in rules:
            if not rule["enabled"] or rule["scope"] not in {scope, "both"}:
                continue
            if not self._matches(rule, sanitized):
                continue
            hits.append({"rule_id": rule["rule_id"], "name": rule["name"], "action": rule["action"]})
            if rule["action"] == "block":
                blocked = True
            else:
                sanitized = self._mask(rule, sanitized)
        return {"text": sanitized, "blocked": blocked, "hits": hits, "version": version, "scope": scope}


def build_guardrail_manager() -> GuardrailManager:
    path = os.getenv("KDIC_GUARDRAIL_PATH", "/opt/kdic/runtime/admin_guardrails.json")
    return GuardrailManager(path)
