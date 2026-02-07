"""Comprehensive tests for pi.web.ws.protocol."""

from __future__ import annotations

import pytest

from pi.web.ws.protocol import (
    AbortMessage,
    DeleteSessionMessage,
    LoadSessionMessage,
    NewSessionMessage,
    PromptMessage,
    SetApiKeyMessage,
    SetModelMessage,
    SetThinkingLevelMessage,
    api_key_required_message,
    error_message,
    models_message,
    parse_client_message,
    sessions_message,
    state_message,
)


# ---------------------------------------------------------------------------
# parse_client_message  --  PromptMessage
# ---------------------------------------------------------------------------


class TestParsePromptMessage:
    def test_prompt_with_text_and_attachments(self):
        data = {"type": "prompt", "text": "hello world", "attachments": ["file1.txt", "file2.png"]}
        msg = parse_client_message(data)
        assert isinstance(msg, PromptMessage)
        assert msg.type == "prompt"
        assert msg.text == "hello world"
        assert msg.attachments == ["file1.txt", "file2.png"]

    def test_prompt_with_text_only_no_attachments_key(self):
        data = {"type": "prompt", "text": "just text"}
        msg = parse_client_message(data)
        assert isinstance(msg, PromptMessage)
        assert msg.text == "just text"
        assert msg.attachments == []

    def test_prompt_with_empty_text(self):
        data = {"type": "prompt"}
        msg = parse_client_message(data)
        assert isinstance(msg, PromptMessage)
        assert msg.text == ""
        assert msg.attachments == []

    def test_prompt_with_empty_attachments_list(self):
        data = {"type": "prompt", "text": "hi", "attachments": []}
        msg = parse_client_message(data)
        assert isinstance(msg, PromptMessage)
        assert msg.text == "hi"
        assert msg.attachments == []

    def test_prompt_attachments_default_not_shared(self):
        """Each PromptMessage should get its own attachments list."""
        msg1 = parse_client_message({"type": "prompt", "text": "a"})
        msg2 = parse_client_message({"type": "prompt", "text": "b"})
        msg1.attachments.append("x")
        assert msg2.attachments == []


# ---------------------------------------------------------------------------
# parse_client_message  --  AbortMessage
# ---------------------------------------------------------------------------


class TestParseAbortMessage:
    def test_abort(self):
        msg = parse_client_message({"type": "abort"})
        assert isinstance(msg, AbortMessage)
        assert msg.type == "abort"

    def test_abort_ignores_extra_keys(self):
        msg = parse_client_message({"type": "abort", "reason": "user cancelled"})
        assert isinstance(msg, AbortMessage)


# ---------------------------------------------------------------------------
# parse_client_message  --  SetModelMessage
# ---------------------------------------------------------------------------


class TestParseSetModelMessage:
    def test_set_model_with_camel_case_model_id(self):
        data = {"type": "set_model", "provider": "openai", "modelId": "gpt-4"}
        msg = parse_client_message(data)
        assert isinstance(msg, SetModelMessage)
        assert msg.type == "set_model"
        assert msg.provider == "openai"
        assert msg.model_id == "gpt-4"

    def test_set_model_with_snake_case_model_id(self):
        data = {"type": "set_model", "provider": "anthropic", "model_id": "claude-3"}
        msg = parse_client_message(data)
        assert isinstance(msg, SetModelMessage)
        assert msg.provider == "anthropic"
        assert msg.model_id == "claude-3"

    def test_set_model_camel_case_takes_precedence(self):
        """When both modelId and model_id are present, modelId wins."""
        data = {"type": "set_model", "provider": "p", "modelId": "camel", "model_id": "snake"}
        msg = parse_client_message(data)
        assert msg.model_id == "camel"

    def test_set_model_missing_fields(self):
        data = {"type": "set_model"}
        msg = parse_client_message(data)
        assert isinstance(msg, SetModelMessage)
        assert msg.provider == ""
        assert msg.model_id == ""


# ---------------------------------------------------------------------------
# parse_client_message  --  SetThinkingLevelMessage
# ---------------------------------------------------------------------------


class TestParseSetThinkingLevelMessage:
    def test_set_thinking_level(self):
        data = {"type": "set_thinking_level", "level": "high"}
        msg = parse_client_message(data)
        assert isinstance(msg, SetThinkingLevelMessage)
        assert msg.type == "set_thinking_level"
        assert msg.level == "high"

    def test_set_thinking_level_default(self):
        data = {"type": "set_thinking_level"}
        msg = parse_client_message(data)
        assert isinstance(msg, SetThinkingLevelMessage)
        assert msg.level == "off"


# ---------------------------------------------------------------------------
# parse_client_message  --  LoadSessionMessage
# ---------------------------------------------------------------------------


class TestParseLoadSessionMessage:
    def test_load_session_with_camel_case(self):
        data = {"type": "load_session", "sessionId": "abc-123"}
        msg = parse_client_message(data)
        assert isinstance(msg, LoadSessionMessage)
        assert msg.type == "load_session"
        assert msg.session_id == "abc-123"

    def test_load_session_with_snake_case(self):
        data = {"type": "load_session", "session_id": "xyz-789"}
        msg = parse_client_message(data)
        assert isinstance(msg, LoadSessionMessage)
        assert msg.session_id == "xyz-789"

    def test_load_session_camel_case_takes_precedence(self):
        data = {"type": "load_session", "sessionId": "camel", "session_id": "snake"}
        msg = parse_client_message(data)
        assert msg.session_id == "camel"

    def test_load_session_missing_id(self):
        data = {"type": "load_session"}
        msg = parse_client_message(data)
        assert isinstance(msg, LoadSessionMessage)
        assert msg.session_id == ""


# ---------------------------------------------------------------------------
# parse_client_message  --  NewSessionMessage
# ---------------------------------------------------------------------------


class TestParseNewSessionMessage:
    def test_new_session(self):
        msg = parse_client_message({"type": "new_session"})
        assert isinstance(msg, NewSessionMessage)
        assert msg.type == "new_session"

    def test_new_session_ignores_extra_keys(self):
        msg = parse_client_message({"type": "new_session", "foo": "bar"})
        assert isinstance(msg, NewSessionMessage)


# ---------------------------------------------------------------------------
# parse_client_message  --  SetApiKeyMessage
# ---------------------------------------------------------------------------


class TestParseSetApiKeyMessage:
    def test_set_api_key(self):
        data = {"type": "set_api_key", "provider": "openai", "key": "sk-abc123"}
        msg = parse_client_message(data)
        assert isinstance(msg, SetApiKeyMessage)
        assert msg.type == "set_api_key"
        assert msg.provider == "openai"
        assert msg.key == "sk-abc123"

    def test_set_api_key_missing_fields(self):
        data = {"type": "set_api_key"}
        msg = parse_client_message(data)
        assert isinstance(msg, SetApiKeyMessage)
        assert msg.provider == ""
        assert msg.key == ""


# ---------------------------------------------------------------------------
# parse_client_message  --  DeleteSessionMessage
# ---------------------------------------------------------------------------


class TestParseDeleteSessionMessage:
    def test_delete_session_with_camel_case(self):
        data = {"type": "delete_session", "sessionId": "sess-1"}
        msg = parse_client_message(data)
        assert isinstance(msg, DeleteSessionMessage)
        assert msg.type == "delete_session"
        assert msg.session_id == "sess-1"

    def test_delete_session_with_snake_case(self):
        data = {"type": "delete_session", "session_id": "sess-2"}
        msg = parse_client_message(data)
        assert isinstance(msg, DeleteSessionMessage)
        assert msg.session_id == "sess-2"

    def test_delete_session_camel_case_takes_precedence(self):
        data = {"type": "delete_session", "sessionId": "camel", "session_id": "snake"}
        msg = parse_client_message(data)
        assert msg.session_id == "camel"

    def test_delete_session_missing_id(self):
        data = {"type": "delete_session"}
        msg = parse_client_message(data)
        assert isinstance(msg, DeleteSessionMessage)
        assert msg.session_id == ""


# ---------------------------------------------------------------------------
# parse_client_message  --  Edge cases / unknown types
# ---------------------------------------------------------------------------


class TestParseClientMessageEdgeCases:
    def test_unknown_type_returns_none(self):
        assert parse_client_message({"type": "unknown_thing"}) is None

    def test_empty_dict_returns_none(self):
        assert parse_client_message({}) is None

    def test_missing_type_key_returns_none(self):
        assert parse_client_message({"text": "hello"}) is None

    def test_type_value_is_none_returns_none(self):
        """data.get('type', '') returns None when the key exists but value is None."""
        # None won't match any case, so falls through to default
        assert parse_client_message({"type": None}) is None

    def test_empty_string_type_returns_none(self):
        assert parse_client_message({"type": ""}) is None

    def test_numeric_type_returns_none(self):
        assert parse_client_message({"type": 42}) is None


# ---------------------------------------------------------------------------
# Dataclass defaults
# ---------------------------------------------------------------------------


class TestDataclassDefaults:
    def test_prompt_message_defaults(self):
        msg = PromptMessage()
        assert msg.type == "prompt"
        assert msg.text == ""
        assert msg.attachments == []

    def test_abort_message_defaults(self):
        msg = AbortMessage()
        assert msg.type == "abort"

    def test_set_model_message_defaults(self):
        msg = SetModelMessage()
        assert msg.type == "set_model"
        assert msg.provider == ""
        assert msg.model_id == ""

    def test_set_thinking_level_message_defaults(self):
        msg = SetThinkingLevelMessage()
        assert msg.type == "set_thinking_level"
        assert msg.level == "off"

    def test_load_session_message_defaults(self):
        msg = LoadSessionMessage()
        assert msg.type == "load_session"
        assert msg.session_id == ""

    def test_new_session_message_defaults(self):
        msg = NewSessionMessage()
        assert msg.type == "new_session"

    def test_set_api_key_message_defaults(self):
        msg = SetApiKeyMessage()
        assert msg.type == "set_api_key"
        assert msg.provider == ""
        assert msg.key == ""

    def test_delete_session_message_defaults(self):
        msg = DeleteSessionMessage()
        assert msg.type == "delete_session"
        assert msg.session_id == ""


# ---------------------------------------------------------------------------
# Server -> Client message builders
# ---------------------------------------------------------------------------


class TestStateMessage:
    def test_basic_state_message(self):
        result = state_message(
            session_id="sess-1",
            model={"provider": "anthropic", "id": "claude-3"},
            thinking_level="high",
            messages=[{"role": "user", "content": "hi"}],
            is_streaming=False,
        )
        assert result == {
            "type": "state",
            "sessionId": "sess-1",
            "model": {"provider": "anthropic", "id": "claude-3"},
            "thinkingLevel": "high",
            "messages": [{"role": "user", "content": "hi"}],
            "isStreaming": False,
        }

    def test_state_message_with_none_model(self):
        result = state_message(
            session_id="s",
            model=None,
            thinking_level="off",
            messages=[],
            is_streaming=True,
        )
        assert result["type"] == "state"
        assert result["model"] is None
        assert result["isStreaming"] is True
        assert result["messages"] == []

    def test_state_message_with_empty_messages(self):
        result = state_message("id", {}, "off", [], False)
        assert result["messages"] == []

    def test_state_message_with_multiple_messages(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
            {"role": "user", "content": "how are you?"},
        ]
        result = state_message("sess", None, "medium", msgs, True)
        assert result["messages"] == msgs
        assert len(result["messages"]) == 3

    def test_state_message_keys(self):
        result = state_message("s", None, "off", [], False)
        assert set(result.keys()) == {"type", "sessionId", "model", "thinkingLevel", "messages", "isStreaming"}


class TestErrorMessage:
    def test_error_message(self):
        result = error_message("something went wrong")
        assert result == {"type": "error", "message": "something went wrong"}

    def test_error_message_empty_string(self):
        result = error_message("")
        assert result == {"type": "error", "message": ""}

    def test_error_message_keys(self):
        result = error_message("x")
        assert set(result.keys()) == {"type", "message"}


class TestApiKeyRequiredMessage:
    def test_api_key_required_message(self):
        result = api_key_required_message("openai")
        assert result == {"type": "api_key_required", "provider": "openai"}

    def test_api_key_required_message_empty_provider(self):
        result = api_key_required_message("")
        assert result == {"type": "api_key_required", "provider": ""}

    def test_api_key_required_message_keys(self):
        result = api_key_required_message("anthropic")
        assert set(result.keys()) == {"type", "provider"}


class TestModelsMessage:
    def test_models_message_with_providers(self):
        providers = [
            {"name": "openai", "models": [{"id": "gpt-4"}, {"id": "gpt-3.5"}]},
            {"name": "anthropic", "models": [{"id": "claude-3"}]},
        ]
        result = models_message(providers)
        assert result == {"type": "models", "providers": providers}

    def test_models_message_empty_list(self):
        result = models_message([])
        assert result == {"type": "models", "providers": []}

    def test_models_message_keys(self):
        result = models_message([])
        assert set(result.keys()) == {"type", "providers"}


class TestSessionsMessage:
    def test_sessions_message_with_sessions(self):
        sessions = [
            {"id": "sess-1", "title": "Chat 1"},
            {"id": "sess-2", "title": "Chat 2"},
        ]
        result = sessions_message(sessions)
        assert result == {"type": "sessions", "sessions": sessions}

    def test_sessions_message_empty_list(self):
        result = sessions_message([])
        assert result == {"type": "sessions", "sessions": []}

    def test_sessions_message_keys(self):
        result = sessions_message([])
        assert set(result.keys()) == {"type", "sessions"}
