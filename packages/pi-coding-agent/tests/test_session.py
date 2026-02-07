"""Tests for AgentSession and its helpers."""

from __future__ import annotations

from typing import Any

from pi.agent.agent import Agent
from pi.agent.types import AgentState
from pi.ai.types import (
    AssistantMessage,
    Model,
    ModelCost,
    TextContent,
    ToolCall,
    Usage,
    UserMessage,
)
from pi.coding.core.session import AgentSession, AgentSessionConfig
from pi.coding.core.session.events import (
    AutoCompactionEndEvent,
    AutoCompactionStartEvent,
    AutoRetryEndEvent,
    AutoRetryStartEvent,
    SessionForkedEvent,
    SessionSwitchedEvent,
)
from pi.coding.core.session.models import (
    THINKING_LEVELS_WITH_XHIGH,
    _clamp_thinking_level,
    _model_supports_xhigh,
)
from pi.coding.core.session.navigation import (
    _extract_user_message_text,
)
from pi.coding.core.sessions import SessionManager
from pi.coding.core.settings import SettingsManager


def _make_model(
    model_id: str = "test-model",
    provider: str = "test-provider",
    reasoning: bool = False,
    context_window: int = 100000,
) -> Model:
    return Model(
        id=model_id,
        name=model_id,
        api="anthropic-messages",
        provider=provider,
        base_url="https://test.api",
        reasoning=reasoning,
        input=["text"],
        cost=ModelCost(),
        context_window=context_window,
        max_tokens=4096,
    )


def _make_assistant_message(
    text: str = "Hello",
    error_message: str | None = None,
    stop_reason: str = "stop",
    usage: Usage | None = None,
    tool_calls: list[ToolCall] | None = None,
) -> AssistantMessage:
    content: list[Any] = [TextContent(text=text)]
    if tool_calls:
        content.extend(tool_calls)
    return AssistantMessage(
        role="assistant",
        content=content,
        api="anthropic-messages",
        provider="test",
        model="test-model",
        usage=usage or Usage(),
        stop_reason=stop_reason,
        error_message=error_message,
        timestamp=1000,
    )


def _make_user_message(text: str = "Hello") -> UserMessage:
    return UserMessage(
        role="user",
        content=[TextContent(text=text)],
        timestamp=1000,
    )


def _make_session_config(
    model: Model | None = None,
    cwd: str = "/test",
) -> AgentSessionConfig:
    model = model or _make_model()
    agent = Agent(initial_state=AgentState(model=model))
    sm = SessionManager.in_memory(cwd)
    settings = SettingsManager.in_memory()
    return AgentSessionConfig(
        agent=agent,
        session_manager=sm,
        settings_manager=settings,
        cwd=cwd,
    )


# --- AgentSession Events ---


class TestAgentSessionEvents:
    """Test custom event types."""

    def test_auto_compaction_start_event(self) -> None:
        event = AutoCompactionStartEvent(reason="threshold")
        assert event.type == "auto_compaction_start"
        assert event.reason == "threshold"

    def test_auto_compaction_end_event(self) -> None:
        event = AutoCompactionEndEvent(aborted=True, error_message="cancelled")
        assert event.type == "auto_compaction_end"
        assert event.aborted is True
        assert event.error_message == "cancelled"

    def test_auto_retry_start_event(self) -> None:
        event = AutoRetryStartEvent(attempt=2, max_attempts=5, delay_ms=4000, error_message="429")
        assert event.type == "auto_retry_start"
        assert event.attempt == 2
        assert event.delay_ms == 4000

    def test_auto_retry_end_event(self) -> None:
        event = AutoRetryEndEvent(success=True, attempt=3)
        assert event.type == "auto_retry_end"
        assert event.success is True

    def test_session_switched_event(self) -> None:
        event = SessionSwitchedEvent(session_path="/sessions/test.jsonl", reason="resume")
        assert event.type == "session_switched"
        assert event.session_path == "/sessions/test.jsonl"

    def test_session_forked_event(self) -> None:
        event = SessionForkedEvent(entry_id="abc123")
        assert event.type == "session_forked"


# --- Thinking Level Clamping ---


class TestClampThinkingLevel:
    """Test the _clamp_thinking_level function."""

    def test_exact_match(self) -> None:
        assert _clamp_thinking_level("medium", ["off", "low", "medium", "high"]) == "medium"

    def test_clamp_forward(self) -> None:
        # "low" not in available, nearest forward is "medium"
        assert _clamp_thinking_level("low", ["off", "medium", "high"]) == "medium"

    def test_clamp_backward(self) -> None:
        # "high" not available but "medium" is
        assert _clamp_thinking_level("high", ["off", "medium"]) == "medium"

    def test_clamp_to_off(self) -> None:
        assert _clamp_thinking_level("high", ["off"]) == "off"

    def test_clamp_empty_available(self) -> None:
        assert _clamp_thinking_level("medium", []) == "off"

    def test_clamp_unknown_level(self) -> None:
        assert _clamp_thinking_level("super", ["off", "medium"]) == "off"

    def test_xhigh_available(self) -> None:
        levels = THINKING_LEVELS_WITH_XHIGH
        assert _clamp_thinking_level("xhigh", levels) == "xhigh"


# --- Model Supports xhigh ---


class TestModelSupportsXhigh:
    def test_anthropic_opus_supports_xhigh(self) -> None:
        model = _make_model("claude-opus-4-6", "anthropic", reasoning=True)
        assert _model_supports_xhigh(model) is True

    def test_non_reasoning_no_xhigh(self) -> None:
        model = _make_model("claude-opus-4-6", "anthropic", reasoning=False)
        assert _model_supports_xhigh(model) is False

    def test_non_anthropic_no_xhigh(self) -> None:
        model = _make_model("gpt-5", "openai", reasoning=True)
        assert _model_supports_xhigh(model) is False

    def test_anthropic_sonnet_no_xhigh(self) -> None:
        model = _make_model("claude-sonnet-4-5", "anthropic", reasoning=True)
        assert _model_supports_xhigh(model) is False


# --- Extract User Message Text ---


class TestExtractUserMessageText:
    def test_string_content(self) -> None:
        assert _extract_user_message_text("hello") == "hello"

    def test_content_blocks(self) -> None:
        content = [{"type": "text", "text": "hello"}, {"type": "image", "data": "..."}]
        assert _extract_user_message_text(content) == "hello"

    def test_multiple_text_blocks(self) -> None:
        content = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
        assert _extract_user_message_text(content) == "a b"

    def test_empty_content(self) -> None:
        assert _extract_user_message_text([]) == ""
        assert _extract_user_message_text("") == ""
        assert _extract_user_message_text(None) == ""


# --- AgentSession Construction ---


class TestAgentSessionConstruction:
    def test_basic_construction(self) -> None:
        config = _make_session_config()
        session = AgentSession(config)
        assert session.agent is config.agent
        assert session.session_manager is config.session_manager
        assert session.settings_manager is config.settings_manager
        assert session.cwd == "/test"

    def test_model_property(self) -> None:
        model = _make_model("my-model")
        config = _make_session_config(model=model)
        session = AgentSession(config)
        assert session.model is not None
        assert session.model.id == "my-model"

    def test_thinking_level_default(self) -> None:
        config = _make_session_config()
        session = AgentSession(config)
        assert session.thinking_level == "off"

    def test_is_streaming_default(self) -> None:
        config = _make_session_config()
        session = AgentSession(config)
        assert session.is_streaming is False

    def test_messages_empty(self) -> None:
        config = _make_session_config()
        session = AgentSession(config)
        assert session.messages == []

    def test_session_id(self) -> None:
        config = _make_session_config()
        session = AgentSession(config)
        assert session.session_id is not None
        assert len(session.session_id) > 0

    def test_tools_registered(self) -> None:
        config = _make_session_config()
        session = AgentSession(config)
        tools = session.get_all_tools()
        # Should have all 7 base tools
        assert "read" in tools
        assert "bash" in tools
        assert "edit" in tools
        assert "write" in tools
        assert "grep" in tools
        assert "find" in tools
        assert "ls" in tools

    def test_system_prompt_set(self) -> None:
        config = _make_session_config()
        session = AgentSession(config)
        assert "expert coding assistant" in session.system_prompt

    def test_custom_prompt(self) -> None:
        config = _make_session_config()
        config.custom_prompt = "You are a test bot."
        session = AgentSession(config)
        assert "You are a test bot." in session.system_prompt

    def test_append_system_prompt(self) -> None:
        config = _make_session_config()
        config.append_system_prompt = "Extra instructions."
        session = AgentSession(config)
        assert "Extra instructions." in session.system_prompt


# --- Tool Management ---


class TestToolManagement:
    def test_get_active_tool_names(self) -> None:
        config = _make_session_config()
        session = AgentSession(config)
        names = session.get_active_tool_names()
        assert "read" in names
        assert "bash" in names

    def test_set_active_tools_by_name(self) -> None:
        config = _make_session_config()
        session = AgentSession(config)
        session.set_active_tools_by_name(["read", "bash"])
        names = session.get_active_tool_names()
        assert set(names) == {"read", "bash"}

    def test_set_active_tools_updates_agent(self) -> None:
        config = _make_session_config()
        session = AgentSession(config)
        session.set_active_tools_by_name(["read"])
        assert len(session.agent.state.tools) == 1
        assert session.agent.state.tools[0].name == "read"


# --- Event Handling ---


class TestEventHandling:
    def test_subscribe_returns_unsubscribe(self) -> None:
        config = _make_session_config()
        session = AgentSession(config)
        events: list[Any] = []
        unsub = session.subscribe(lambda e: events.append(e))
        assert callable(unsub)

    def test_unsubscribe_removes_listener(self) -> None:
        config = _make_session_config()
        session = AgentSession(config)
        events: list[Any] = []
        unsub = session.subscribe(lambda e: events.append(e))
        unsub()
        session._emit_session_event(AutoCompactionStartEvent())
        assert len(events) == 0

    def test_session_event_emitted_to_listeners(self) -> None:
        config = _make_session_config()
        session = AgentSession(config)
        events: list[Any] = []
        session.subscribe(lambda e: events.append(e))
        session._emit_session_event(AutoCompactionStartEvent(reason="threshold"))
        assert len(events) == 1
        assert events[0].type == "auto_compaction_start"

    def test_dispose_clears_listeners(self) -> None:
        config = _make_session_config()
        session = AgentSession(config)
        events: list[Any] = []
        session.subscribe(lambda e: events.append(e))
        session.dispose()
        session._emit_session_event(AutoCompactionStartEvent())
        assert len(events) == 0


# --- Model Management ---


class TestModelManagement:
    def test_available_thinking_levels_no_reasoning(self) -> None:
        config = _make_session_config(model=_make_model(reasoning=False))
        session = AgentSession(config)
        levels = session.get_available_thinking_levels()
        assert levels == ["off"]

    def test_available_thinking_levels_with_reasoning(self) -> None:
        config = _make_session_config(model=_make_model(reasoning=True))
        session = AgentSession(config)
        levels = session.get_available_thinking_levels()
        assert "off" in levels
        assert "medium" in levels
        assert "high" in levels

    def test_set_thinking_level(self) -> None:
        config = _make_session_config(model=_make_model(reasoning=True))
        session = AgentSession(config)
        result = session.set_thinking_level("high")
        assert result == "high"
        assert session.thinking_level == "high"

    def test_set_thinking_level_clamps(self) -> None:
        config = _make_session_config(model=_make_model(reasoning=False))
        session = AgentSession(config)
        result = session.set_thinking_level("high")
        assert result == "off"  # Clamped because model doesn't support reasoning

    def test_cycle_thinking_level(self) -> None:
        model = _make_model(reasoning=True)
        config = _make_session_config(model=model)
        session = AgentSession(config)

        # Set to "off" first
        session.set_thinking_level("off")
        # Cycle should go to "minimal"
        result = session.cycle_thinking_level()
        assert result == "minimal"

    def test_cycle_thinking_level_no_reasoning(self) -> None:
        config = _make_session_config(model=_make_model(reasoning=False))
        session = AgentSession(config)
        result = session.cycle_thinking_level()
        assert result is None


# --- Compaction ---


class TestCompactionHelper:
    def test_is_retryable_error_rate_limit(self) -> None:
        config = _make_session_config()
        session = AgentSession(config)
        msg = _make_assistant_message(error_message="rate limit exceeded")
        assert session._compaction.is_retryable_error(msg) is True

    def test_is_retryable_error_429(self) -> None:
        config = _make_session_config()
        session = AgentSession(config)
        msg = _make_assistant_message(error_message="HTTP 429 Too Many Requests")
        assert session._compaction.is_retryable_error(msg) is True

    def test_is_retryable_error_overloaded(self) -> None:
        config = _make_session_config()
        session = AgentSession(config)
        msg = _make_assistant_message(error_message="Service overloaded")
        assert session._compaction.is_retryable_error(msg) is True

    def test_is_retryable_error_500(self) -> None:
        config = _make_session_config()
        session = AgentSession(config)
        msg = _make_assistant_message(error_message="HTTP 500 Internal Server Error")
        assert session._compaction.is_retryable_error(msg) is True

    def test_overflow_not_retryable(self) -> None:
        config = _make_session_config()
        session = AgentSession(config)
        msg = _make_assistant_message(error_message="context window exceeded, too many tokens")
        assert session._compaction.is_retryable_error(msg) is False

    def test_no_error_not_retryable(self) -> None:
        config = _make_session_config()
        session = AgentSession(config)
        msg = _make_assistant_message()
        assert session._compaction.is_retryable_error(msg) is False

    def test_connection_error_retryable(self) -> None:
        config = _make_session_config()
        session = AgentSession(config)
        msg = _make_assistant_message(error_message="connection reset by peer")
        assert session._compaction.is_retryable_error(msg) is True


# --- Navigation ---


class TestNavigationHelper:
    def test_session_stats_empty(self) -> None:
        config = _make_session_config()
        session = AgentSession(config)
        stats = session.get_session_stats()
        assert stats.user_messages == 0
        assert stats.assistant_messages == 0
        assert stats.total_cost == 0.0

    def test_session_stats_with_messages(self) -> None:
        config = _make_session_config()
        session = AgentSession(config)

        # Add messages directly to agent state
        user_msg = _make_user_message("Hi")
        assistant_msg = _make_assistant_message(
            "Hello!",
            usage=Usage(input=100, output=50),
        )
        session.agent.append_message(user_msg)
        session.agent.append_message(assistant_msg)

        stats = session.get_session_stats()
        assert stats.user_messages == 1
        assert stats.assistant_messages == 1
        assert stats.input_tokens == 100
        assert stats.output_tokens == 50

    def test_session_stats_counts_tool_calls(self) -> None:
        config = _make_session_config()
        session = AgentSession(config)

        tool_call = ToolCall(
            type="tool_call",
            id="tc1",
            name="read",
            arguments={"path": "/test.py"},
        )
        msg = _make_assistant_message(tool_calls=[tool_call])
        session.agent.append_message(msg)

        stats = session.get_session_stats()
        assert stats.tool_calls == 1

    def test_context_usage_empty(self) -> None:
        config = _make_session_config()
        session = AgentSession(config)
        usage = session.get_context_usage()
        assert usage.tokens == 0
        assert usage.percentage == 0.0

    def test_context_usage_with_model(self) -> None:
        model = _make_model(context_window=100000)
        config = _make_session_config(model=model)
        session = AgentSession(config)
        usage = session.get_context_usage()
        assert usage.context_window == 100000

    def test_get_last_assistant_text_empty(self) -> None:
        config = _make_session_config()
        session = AgentSession(config)
        assert session.get_last_assistant_text() == ""

    def test_get_last_assistant_text(self) -> None:
        config = _make_session_config()
        session = AgentSession(config)
        msg = _make_assistant_message("Final response.")
        session.agent.append_message(msg)
        assert session.get_last_assistant_text() == "Final response."

    def test_get_last_assistant_text_skips_aborted(self) -> None:
        config = _make_session_config()
        session = AgentSession(config)
        good_msg = _make_assistant_message("Good response.")
        aborted_msg = _make_assistant_message("", stop_reason="aborted")
        session.agent.append_message(good_msg)
        session.agent.append_message(aborted_msg)
        assert session.get_last_assistant_text() == "Good response."

    def test_get_user_messages_for_forking(self) -> None:
        config = _make_session_config()
        session = AgentSession(config)

        # Add entries to session manager
        session.session_manager.append_message({"role": "user", "content": "First question"})
        session.session_manager.append_message({"role": "assistant", "content": "Answer"})
        session.session_manager.append_message({"role": "user", "content": "Second question"})

        result = session.get_user_messages_for_forking()
        assert len(result) == 2
        assert result[0].text == "First question"
        assert result[1].text == "Second question"


# --- Queue Management ---


class TestQueueManagement:
    def test_clear_queue(self) -> None:
        config = _make_session_config()
        session = AgentSession(config)
        session._steering_messages.append("steer1")
        session._follow_up_messages.append("follow1")

        steering, follow_up = session.clear_queue()
        assert steering == ["steer1"]
        assert follow_up == ["follow1"]
        assert session._steering_messages == []
        assert session._follow_up_messages == []
