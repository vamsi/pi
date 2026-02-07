"""Context sync (log.jsonl -> SessionManager) and MomSettingsManager."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


# ============================================================================
# Sync log.jsonl to SessionManager
# ============================================================================


def sync_log_to_session_manager(
    session_manager: Any,
    channel_dir: str,
    exclude_slack_ts: str | None = None,
) -> int:
    """Sync user messages from log.jsonl to SessionManager.

    Ensures that messages logged while mom wasn't running (channel chatter,
    backfilled messages, messages while busy) are added to the LLM context.

    Returns the number of messages synced.
    """
    log_file = os.path.join(channel_dir, "log.jsonl")
    if not os.path.exists(log_file):
        return 0

    # Build set of existing message content from session
    _TS_PREFIX_RE = re.compile(
        r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}\] "
    )

    existing_messages: set[str] = set()
    for entry in session_manager.get_entries():
        if entry.get("type") != "message":
            continue
        msg = entry.get("message", {})
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if content is None:
            continue

        if isinstance(content, str):
            normalized = _TS_PREFIX_RE.sub("", content)
            att_idx = normalized.find("\n\n<slack_attachments>\n")
            if att_idx != -1:
                normalized = normalized[:att_idx]
            existing_messages.add(normalized)
        elif isinstance(content, list):
            for part in content:
                if (
                    isinstance(part, dict)
                    and part.get("type") == "text"
                    and "text" in part
                ):
                    normalized = _TS_PREFIX_RE.sub("", part["text"])
                    att_idx = normalized.find("\n\n<slack_attachments>\n")
                    if att_idx != -1:
                        normalized = normalized[:att_idx]
                    existing_messages.add(normalized)

    # Read log.jsonl and find user messages not in context
    with open(log_file, "r", encoding="utf-8") as fh:
        log_content = fh.read()

    log_lines = [l for l in log_content.strip().split("\n") if l]

    new_messages: list[tuple[float, dict[str, Any]]] = []

    for line in log_lines:
        try:
            log_msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        slack_ts = log_msg.get("ts")
        date_str = log_msg.get("date")
        if not slack_ts or not date_str:
            continue

        if exclude_slack_ts and slack_ts == exclude_slack_ts:
            continue

        if log_msg.get("isBot"):
            continue

        user_name = (
            log_msg.get("userName") or log_msg.get("user") or "unknown"
        )
        message_text = f"[{user_name}]: {log_msg.get('text', '')}"

        if message_text in existing_messages:
            continue

        try:
            msg_time = datetime.fromisoformat(date_str).timestamp() * 1000
        except (ValueError, TypeError):
            msg_time = _now_ms()

        user_message = {
            "role": "user",
            "content": [{"type": "text", "text": message_text}],
            "timestamp": msg_time,
        }

        new_messages.append((msg_time, user_message))
        existing_messages.add(message_text)

    if not new_messages:
        return 0

    new_messages.sort(key=lambda x: x[0])

    for _, message in new_messages:
        session_manager.append_message(message)

    return len(new_messages)


def _now_ms() -> float:
    return datetime.now(timezone.utc).timestamp() * 1000


# ============================================================================
# MomSettingsManager
# ============================================================================

_DEFAULT_COMPACTION = {
    "enabled": True,
    "reserveTokens": 16384,
    "keepRecentTokens": 20000,
}

_DEFAULT_RETRY = {
    "enabled": True,
    "maxRetries": 3,
    "baseDelayMs": 2000,
}


class MomSettingsManager:
    """Simple settings manager for mom.

    Stores settings in ``settings.json`` inside *workspace_dir*.
    """

    def __init__(self, workspace_dir: str) -> None:
        self._settings_path = os.path.join(workspace_dir, "settings.json")
        self._settings: dict[str, Any] = self._load()

    # ── Private persistence ──────────────────────────────────────────

    def _load(self) -> dict[str, Any]:
        if not os.path.exists(self._settings_path):
            return {}
        try:
            with open(self._settings_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {}

    def _save(self) -> None:
        try:
            d = os.path.dirname(self._settings_path)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(self._settings_path, "w", encoding="utf-8") as fh:
                json.dump(self._settings, fh, indent=2)
        except Exception as exc:
            print(f"Warning: Could not save settings file: {exc}")

    # ── Compaction ───────────────────────────────────────────────────

    def get_compaction_settings(self) -> dict[str, Any]:
        return {**_DEFAULT_COMPACTION, **(self._settings.get("compaction") or {})}

    def get_compaction_enabled(self) -> bool:
        comp = self._settings.get("compaction") or {}
        return comp.get("enabled", _DEFAULT_COMPACTION["enabled"])

    def set_compaction_enabled(self, enabled: bool) -> None:
        comp = self._settings.setdefault("compaction", {})
        comp["enabled"] = enabled
        self._save()

    # ── Retry ────────────────────────────────────────────────────────

    def get_retry_settings(self) -> dict[str, Any]:
        return {**_DEFAULT_RETRY, **(self._settings.get("retry") or {})}

    def get_retry_enabled(self) -> bool:
        retry = self._settings.get("retry") or {}
        return retry.get("enabled", _DEFAULT_RETRY["enabled"])

    def set_retry_enabled(self, enabled: bool) -> None:
        retry = self._settings.setdefault("retry", {})
        retry["enabled"] = enabled
        self._save()

    # ── Model / Provider ─────────────────────────────────────────────

    def get_default_model(self) -> str | None:
        return self._settings.get("defaultModel")

    def get_default_provider(self) -> str | None:
        return self._settings.get("defaultProvider")

    def set_default_model_and_provider(
        self, provider: str, model_id: str
    ) -> None:
        self._settings["defaultProvider"] = provider
        self._settings["defaultModel"] = model_id
        self._save()

    # ── Thinking level ───────────────────────────────────────────────

    def get_default_thinking_level(self) -> str:
        return self._settings.get("defaultThinkingLevel", "off")

    def set_default_thinking_level(self, level: str) -> None:
        self._settings["defaultThinkingLevel"] = level
        self._save()

    # ── Compatibility methods for AgentSession ───────────────────────

    def get_steering_mode(self) -> str:
        return "one-at-a-time"

    def set_steering_mode(self, _mode: str) -> None:
        pass

    def get_follow_up_mode(self) -> str:
        return "one-at-a-time"

    def set_follow_up_mode(self, _mode: str) -> None:
        pass

    def get_hook_paths(self) -> list[str]:
        return []

    def get_hook_timeout(self) -> int:
        return 30000
