"""Tests for pi.mom.context."""

import json
import os
import tempfile

import pytest

from pi.mom.context import MomSettingsManager, sync_log_to_session_manager


# Minimal mock SessionManager for testing sync
class MockSessionManager:
    def __init__(self) -> None:
        self.entries: list[dict] = []
        self.appended: list[dict] = []

    def get_entries(self) -> list[dict]:
        return self.entries

    def append_message(self, msg: dict) -> None:
        self.appended.append(msg)


@pytest.fixture
def tmpdir() -> str:
    with tempfile.TemporaryDirectory() as d:
        yield d


class TestSyncLogToSessionManager:
    def test_no_log_file(self, tmpdir: str) -> None:
        sm = MockSessionManager()
        count = sync_log_to_session_manager(sm, tmpdir)
        assert count == 0
        assert len(sm.appended) == 0

    def test_syncs_user_messages(self, tmpdir: str) -> None:
        log_path = os.path.join(tmpdir, "log.jsonl")
        with open(log_path, "w") as f:
            f.write(
                json.dumps(
                    {
                        "date": "2025-01-01T10:00:00Z",
                        "ts": "1000.0",
                        "user": "U1",
                        "userName": "mario",
                        "text": "hello",
                        "isBot": False,
                    }
                )
                + "\n"
            )
            f.write(
                json.dumps(
                    {
                        "date": "2025-01-01T10:01:00Z",
                        "ts": "1001.0",
                        "user": "U2",
                        "userName": "luigi",
                        "text": "world",
                        "isBot": False,
                    }
                )
                + "\n"
            )

        sm = MockSessionManager()
        count = sync_log_to_session_manager(sm, tmpdir)
        assert count == 2
        assert len(sm.appended) == 2
        assert sm.appended[0]["content"][0]["text"] == "[mario]: hello"

    def test_skips_bot_messages(self, tmpdir: str) -> None:
        log_path = os.path.join(tmpdir, "log.jsonl")
        with open(log_path, "w") as f:
            f.write(
                json.dumps(
                    {
                        "date": "2025-01-01T10:00:00Z",
                        "ts": "1000.0",
                        "user": "bot",
                        "text": "bot response",
                        "isBot": True,
                    }
                )
                + "\n"
            )

        sm = MockSessionManager()
        count = sync_log_to_session_manager(sm, tmpdir)
        assert count == 0

    def test_skips_excluded_ts(self, tmpdir: str) -> None:
        log_path = os.path.join(tmpdir, "log.jsonl")
        with open(log_path, "w") as f:
            f.write(
                json.dumps(
                    {
                        "date": "2025-01-01T10:00:00Z",
                        "ts": "1000.0",
                        "user": "U1",
                        "userName": "mario",
                        "text": "hi",
                        "isBot": False,
                    }
                )
                + "\n"
            )

        sm = MockSessionManager()
        count = sync_log_to_session_manager(sm, tmpdir, exclude_slack_ts="1000.0")
        assert count == 0

    def test_skips_existing_messages(self, tmpdir: str) -> None:
        log_path = os.path.join(tmpdir, "log.jsonl")
        with open(log_path, "w") as f:
            f.write(
                json.dumps(
                    {
                        "date": "2025-01-01T10:00:00Z",
                        "ts": "1000.0",
                        "user": "U1",
                        "userName": "mario",
                        "text": "hi",
                        "isBot": False,
                    }
                )
                + "\n"
            )

        sm = MockSessionManager()
        sm.entries = [
            {
                "type": "message",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "[mario]: hi"}],
                },
            }
        ]
        count = sync_log_to_session_manager(sm, tmpdir)
        assert count == 0


class TestMomSettingsManager:
    def test_defaults(self, tmpdir: str) -> None:
        mgr = MomSettingsManager(tmpdir)
        assert mgr.get_compaction_enabled() is True
        assert mgr.get_retry_enabled() is True
        assert mgr.get_default_model() is None
        assert mgr.get_default_thinking_level() == "off"

    def test_set_compaction(self, tmpdir: str) -> None:
        mgr = MomSettingsManager(tmpdir)
        mgr.set_compaction_enabled(False)
        assert mgr.get_compaction_enabled() is False

        # Reload from file
        mgr2 = MomSettingsManager(tmpdir)
        assert mgr2.get_compaction_enabled() is False

    def test_set_retry(self, tmpdir: str) -> None:
        mgr = MomSettingsManager(tmpdir)
        mgr.set_retry_enabled(False)
        assert mgr.get_retry_enabled() is False

    def test_set_model_and_provider(self, tmpdir: str) -> None:
        mgr = MomSettingsManager(tmpdir)
        mgr.set_default_model_and_provider("anthropic", "claude-sonnet-4-5")
        assert mgr.get_default_provider() == "anthropic"
        assert mgr.get_default_model() == "claude-sonnet-4-5"

    def test_set_thinking_level(self, tmpdir: str) -> None:
        mgr = MomSettingsManager(tmpdir)
        mgr.set_default_thinking_level("medium")
        assert mgr.get_default_thinking_level() == "medium"

    def test_compatibility_methods(self, tmpdir: str) -> None:
        mgr = MomSettingsManager(tmpdir)
        assert mgr.get_steering_mode() == "one-at-a-time"
        assert mgr.get_follow_up_mode() == "one-at-a-time"
        assert mgr.get_hook_paths() == []
        assert mgr.get_hook_timeout() == 30000

    def test_compaction_settings(self, tmpdir: str) -> None:
        mgr = MomSettingsManager(tmpdir)
        settings = mgr.get_compaction_settings()
        assert settings["enabled"] is True
        assert settings["reserveTokens"] == 16384
        assert settings["keepRecentTokens"] == 20000

    def test_retry_settings(self, tmpdir: str) -> None:
        mgr = MomSettingsManager(tmpdir)
        settings = mgr.get_retry_settings()
        assert settings["enabled"] is True
        assert settings["maxRetries"] == 3
        assert settings["baseDelayMs"] == 2000
