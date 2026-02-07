"""Tests for pi.web.ws.serializer module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, Field

from pi.agent.types import (
    AgentEndEvent,
    AgentStartEvent,
    AgentToolResult,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from pi.ai.types import (
    AssistantMessage,
    TextContent,
    UserMessage,
)
from pi.web.ws.serializer import serialize_event, serialize_message


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


class _MockAssistantMessageEvent:
    """Minimal stand-in for AssistantMessageEvent (a union of many Pydantic models).

    We only need *something* to satisfy the ``assistant_message_event`` field
    on ``MessageUpdateEvent``; the serializer never inspects it.
    """

    type: str = "text_delta"


class _PlainObject:
    """A plain Python object with public and private attributes."""

    def __init__(self, x: int, y: str) -> None:
        self.x = x
        self.y = y
        self._private = "hidden"


class _PydanticResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    tool_call_id: str = Field(alias="toolCallId")
    output: str


# ---------------------------------------------------------------------------
# serialize_message
# ---------------------------------------------------------------------------


class TestSerializeMessage:
    """Tests for ``serialize_message``."""

    def test_pydantic_model_uses_model_dump_by_alias(self) -> None:
        msg = AssistantMessage(
            role="assistant",
            content=[TextContent(type="text", text="hello")],
            timestamp=1000,
        )
        result = serialize_message(msg)

        assert isinstance(result, dict)
        # by_alias=True means camelCase keys are used
        assert result["role"] == "assistant"
        assert result["stopReason"] == "stop"  # alias for stop_reason
        assert isinstance(result["content"], list)
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "hello"

    def test_user_message_pydantic(self) -> None:
        msg = UserMessage(role="user", content="hi there", timestamp=12345)
        result = serialize_message(msg)

        assert result["role"] == "user"
        assert result["content"] == "hi there"
        assert result["timestamp"] == 12345

    def test_plain_object_uses_dict(self) -> None:
        obj = _PlainObject(x=42, y="abc")
        result = serialize_message(obj)

        assert result == {"x": 42, "y": "abc"}
        assert "_private" not in result

    def test_empty_input_returns_empty_dict(self) -> None:
        # An int has no model_dump and no __dict__ with public attrs we care about,
        # but actually int does have __dict__... it doesn't. Let's use a type that
        # truly has neither.
        result = serialize_message(42)
        assert result == {}

    def test_none_input_returns_empty_dict(self) -> None:
        result = serialize_message(None)
        assert result == {}

    def test_string_input_returns_empty_dict(self) -> None:
        result = serialize_message("just a string")
        assert result == {}

    def test_plain_object_excludes_private_attrs(self) -> None:
        obj = _PlainObject(x=1, y="two")
        result = serialize_message(obj)
        assert "_private" not in result
        assert "x" in result
        assert "y" in result

    def test_dataclass_uses_dict_fallback(self) -> None:
        @dataclass
        class SimpleMsg:
            role: str = "custom"
            text: str = "stuff"

        msg = SimpleMsg()
        result = serialize_message(msg)
        assert result == {"role": "custom", "text": "stuff"}


# ---------------------------------------------------------------------------
# serialize_event - lifecycle events
# ---------------------------------------------------------------------------


class TestSerializeEventLifecycle:
    """Tests for agent/turn lifecycle events."""

    def test_agent_start(self) -> None:
        event = AgentStartEvent()
        result = serialize_event(event)
        assert result == {"type": "agent_start"}

    def test_agent_end(self) -> None:
        event = AgentEndEvent(messages=[])
        result = serialize_event(event)
        assert result == {"type": "agent_end"}

    def test_agent_end_with_messages_still_returns_type_only(self) -> None:
        """AgentEndEvent carries messages but serialize_event ignores them."""
        msg = UserMessage(role="user", content="bye", timestamp=0)
        event = AgentEndEvent(messages=[msg])
        result = serialize_event(event)
        assert result == {"type": "agent_end"}

    def test_turn_start(self) -> None:
        event = TurnStartEvent()
        result = serialize_event(event)
        assert result == {"type": "turn_start"}

    def test_turn_end(self) -> None:
        msg = AssistantMessage(role="assistant", content=[], timestamp=0)
        event = TurnEndEvent(message=msg)
        result = serialize_event(event)
        assert result == {"type": "turn_end"}

    def test_turn_end_with_tool_results_still_returns_type_only(self) -> None:
        msg = AssistantMessage(role="assistant", content=[], timestamp=0)
        event = TurnEndEvent(message=msg, tool_results=[])
        result = serialize_event(event)
        assert result == {"type": "turn_end"}


# ---------------------------------------------------------------------------
# serialize_event - message events
# ---------------------------------------------------------------------------


class TestSerializeEventMessages:
    """Tests for message start/update/end events."""

    def _make_assistant_message(self, text: str = "hello") -> AssistantMessage:
        return AssistantMessage(
            role="assistant",
            content=[TextContent(type="text", text=text)],
            timestamp=5000,
        )

    def test_message_start(self) -> None:
        msg = self._make_assistant_message("hi")
        event = MessageStartEvent(message=msg)
        result = serialize_event(event)

        assert result is not None
        assert result["type"] == "message_start"
        assert result["message"]["role"] == "assistant"
        assert result["message"]["content"][0]["text"] == "hi"

    def test_message_update(self) -> None:
        msg = self._make_assistant_message("partial")
        mock_event = _MockAssistantMessageEvent()
        event = MessageUpdateEvent(message=msg, assistant_message_event=mock_event)
        result = serialize_event(event)

        assert result is not None
        assert result["type"] == "message_update"
        assert result["message"]["role"] == "assistant"
        assert result["message"]["content"][0]["text"] == "partial"

    def test_message_end(self) -> None:
        msg = self._make_assistant_message("done")
        event = MessageEndEvent(message=msg)
        result = serialize_event(event)

        assert result is not None
        assert result["type"] == "message_end"
        assert result["message"]["role"] == "assistant"
        assert result["message"]["content"][0]["text"] == "done"

    def test_message_start_with_user_message(self) -> None:
        msg = UserMessage(role="user", content="question?", timestamp=100)
        event = MessageStartEvent(message=msg)
        result = serialize_event(event)

        assert result is not None
        assert result["type"] == "message_start"
        assert result["message"]["role"] == "user"
        assert result["message"]["content"] == "question?"

    def test_message_events_use_camel_case_aliases(self) -> None:
        msg = self._make_assistant_message("check aliases")
        event = MessageStartEvent(message=msg)
        result = serialize_event(event)

        # AssistantMessage has stop_reason -> stopReason alias
        assert "stopReason" in result["message"]
        # errorMessage alias
        assert "errorMessage" in result["message"]


# ---------------------------------------------------------------------------
# serialize_event - tool execution events
# ---------------------------------------------------------------------------


class TestSerializeEventToolExecution:
    """Tests for tool execution start/update/end events."""

    def test_tool_execution_start(self) -> None:
        event = ToolExecutionStartEvent(
            tool_call_id="tc_001",
            tool_name="read_file",
            args={"path": "/tmp/foo.txt"},
        )
        result = serialize_event(event)

        assert result == {
            "type": "tool_start",
            "toolCallId": "tc_001",
            "toolName": "read_file",
            "args": {"path": "/tmp/foo.txt"},
        }

    def test_tool_execution_start_with_string_args(self) -> None:
        event = ToolExecutionStartEvent(
            tool_call_id="tc_002",
            tool_name="bash",
            args="ls -la",
        )
        result = serialize_event(event)

        assert result["args"] == "ls -la"

    def test_tool_execution_update(self) -> None:
        event = ToolExecutionUpdateEvent(
            tool_call_id="tc_001",
            tool_name="read_file",
            args={"path": "/tmp/foo.txt"},
            partial_result="partial content...",
        )
        result = serialize_event(event)

        assert result == {
            "type": "tool_update",
            "toolCallId": "tc_001",
            "toolName": "read_file",
            "args": {"path": "/tmp/foo.txt"},
            "partialResult": "partial content...",
        }

    def test_tool_execution_update_with_none_partial(self) -> None:
        event = ToolExecutionUpdateEvent(
            tool_call_id="tc_003",
            tool_name="search",
            args={"query": "test"},
            partial_result=None,
        )
        result = serialize_event(event)

        assert result["partialResult"] is None

    def test_tool_execution_end_basic(self) -> None:
        event = ToolExecutionEndEvent(
            tool_call_id="tc_001",
            tool_name="read_file",
            result="file contents here",
            is_error=False,
        )
        result = serialize_event(event)

        assert result == {
            "type": "tool_end",
            "toolCallId": "tc_001",
            "toolName": "read_file",
            "result": "file contents here",
            "isError": False,
        }

    def test_tool_execution_end_with_error(self) -> None:
        event = ToolExecutionEndEvent(
            tool_call_id="tc_004",
            tool_name="write_file",
            result="Permission denied",
            is_error=True,
        )
        result = serialize_event(event)

        assert result["isError"] is True
        assert result["result"] == "Permission denied"

    def test_tool_execution_end_is_error_defaults_false(self) -> None:
        event = ToolExecutionEndEvent(
            tool_call_id="tc_005",
            tool_name="bash",
            result="ok",
        )
        result = serialize_event(event)
        assert result["isError"] is False


# ---------------------------------------------------------------------------
# serialize_event - unknown event type
# ---------------------------------------------------------------------------


class TestSerializeEventUnknown:
    """Tests for unknown/unhandled event types."""

    def test_returns_none_for_unknown_event(self) -> None:
        @dataclass
        class CustomEvent:
            type: str = "custom_unknown"

        result = serialize_event(CustomEvent())  # type: ignore[arg-type]
        assert result is None

    def test_returns_none_for_plain_dict(self) -> None:
        result = serialize_event({"type": "something"})  # type: ignore[arg-type]
        assert result is None

    def test_returns_none_for_string(self) -> None:
        result = serialize_event("not_an_event")  # type: ignore[arg-type]
        assert result is None


# ---------------------------------------------------------------------------
# _serialize_tool_result (tested via ToolExecutionEndEvent)
# ---------------------------------------------------------------------------


class TestSerializeToolResult:
    """Tests for ``_serialize_tool_result`` via ``serialize_event`` with ToolExecutionEndEvent."""

    def _make_tool_end(self, result: Any, is_error: bool = False) -> ToolExecutionEndEvent:
        return ToolExecutionEndEvent(
            tool_call_id="tc_100",
            tool_name="test_tool",
            result=result,
            is_error=is_error,
        )

    def test_none_result(self) -> None:
        event = self._make_tool_end(None)
        result = serialize_event(event)

        assert result is not None
        assert result["result"] is None

    def test_pydantic_model_result(self) -> None:
        pydantic_result = _PydanticResult(toolCallId="tc_99", output="success")
        event = self._make_tool_end(pydantic_result)
        result = serialize_event(event)

        assert result is not None
        serialized = result["result"]
        assert serialized["toolCallId"] == "tc_99"
        assert serialized["output"] == "success"

    def test_agent_tool_result_with_text_content(self) -> None:
        tool_result = AgentToolResult(
            content=[TextContent(type="text", text="found it")],
            details={"line": 42},
        )
        event = self._make_tool_end(tool_result)
        result = serialize_event(event)

        assert result is not None
        serialized = result["result"]
        assert "content" in serialized
        assert len(serialized["content"]) == 1
        assert serialized["content"][0]["type"] == "text"
        assert serialized["content"][0]["text"] == "found it"
        assert serialized["details"] == {"line": 42}

    def test_agent_tool_result_with_multiple_content_items(self) -> None:
        tool_result = AgentToolResult(
            content=[
                TextContent(type="text", text="line 1"),
                TextContent(type="text", text="line 2"),
            ],
            details=None,
        )
        event = self._make_tool_end(tool_result)
        result = serialize_event(event)

        assert result is not None
        serialized = result["result"]
        assert len(serialized["content"]) == 2
        assert serialized["content"][0]["text"] == "line 1"
        assert serialized["content"][1]["text"] == "line 2"
        assert serialized["details"] is None

    def test_agent_tool_result_empty_content(self) -> None:
        tool_result = AgentToolResult(content=[], details=None)
        event = self._make_tool_end(tool_result)
        result = serialize_event(event)

        assert result is not None
        serialized = result["result"]
        assert serialized["content"] == []
        assert serialized["details"] is None

    def test_agent_tool_result_with_non_pydantic_content_items(self) -> None:
        """Content items that lack model_dump should be stringified."""

        @dataclass
        class PlainContent:
            text: str = "fallback"

        tool_result = AgentToolResult(
            content=[PlainContent(text="raw")],  # type: ignore[list-item]
            details="extra",
        )
        event = self._make_tool_end(tool_result)
        result = serialize_event(event)

        assert result is not None
        serialized = result["result"]
        # PlainContent has no model_dump, so it should be str(item)
        assert len(serialized["content"]) == 1
        assert "raw" in serialized["content"][0]  # str representation includes "raw"
        assert serialized["details"] == "extra"

    def test_plain_string_result(self) -> None:
        event = self._make_tool_end("just a string")
        result = serialize_event(event)

        assert result is not None
        assert result["result"] == "just a string"

    def test_plain_dict_result(self) -> None:
        event = self._make_tool_end({"key": "value", "num": 123})
        result = serialize_event(event)

        assert result is not None
        assert result["result"] == {"key": "value", "num": 123}

    def test_plain_list_result(self) -> None:
        event = self._make_tool_end([1, 2, 3])
        result = serialize_event(event)

        assert result is not None
        assert result["result"] == [1, 2, 3]

    def test_integer_result(self) -> None:
        event = self._make_tool_end(42)
        result = serialize_event(event)

        assert result is not None
        assert result["result"] == 42

    def test_boolean_result(self) -> None:
        event = self._make_tool_end(True)
        result = serialize_event(event)

        assert result is not None
        assert result["result"] is True

    def test_agent_tool_result_without_details_attr(self) -> None:
        """An object with .content but no .details should get details=None via getattr default."""

        class ContentOnlyResult:
            def __init__(self) -> None:
                self.content = [TextContent(type="text", text="ok")]

        event = self._make_tool_end(ContentOnlyResult())
        result = serialize_event(event)

        assert result is not None
        serialized = result["result"]
        assert serialized["content"][0]["text"] == "ok"
        assert serialized["details"] is None

    def test_pydantic_result_uses_aliases(self) -> None:
        """Pydantic results should be serialized with camelCase aliases."""
        pydantic_result = _PydanticResult(toolCallId="tc_alias", output="done")
        event = self._make_tool_end(pydantic_result)
        result = serialize_event(event)

        assert result is not None
        serialized = result["result"]
        # The alias "toolCallId" should be used, not "tool_call_id"
        assert "toolCallId" in serialized
        assert "tool_call_id" not in serialized


# ---------------------------------------------------------------------------
# Round-trip / integration-style tests
# ---------------------------------------------------------------------------


class TestSerializeEventIntegration:
    """Integration tests ensuring all event types produce JSON-serializable output."""

    def test_all_event_types_are_json_serializable(self) -> None:
        """Every event type should produce a dict that can be passed to json.dumps."""
        import json

        msg = AssistantMessage(role="assistant", content=[], timestamp=0)
        mock_ame = _MockAssistantMessageEvent()

        events: list[Any] = [
            AgentStartEvent(),
            AgentEndEvent(messages=[]),
            TurnStartEvent(),
            TurnEndEvent(message=msg),
            MessageStartEvent(message=msg),
            MessageUpdateEvent(message=msg, assistant_message_event=mock_ame),
            MessageEndEvent(message=msg),
            ToolExecutionStartEvent(tool_call_id="t1", tool_name="tool", args={}),
            ToolExecutionUpdateEvent(tool_call_id="t1", tool_name="tool", args={}, partial_result=None),
            ToolExecutionEndEvent(tool_call_id="t1", tool_name="tool", result="ok"),
        ]

        for event in events:
            serialized = serialize_event(event)
            assert serialized is not None, f"serialize_event returned None for {type(event).__name__}"
            # Must not raise
            json_str = json.dumps(serialized)
            assert isinstance(json_str, str)

    def test_every_event_has_type_field(self) -> None:
        msg = AssistantMessage(role="assistant", content=[], timestamp=0)
        mock_ame = _MockAssistantMessageEvent()

        events: list[Any] = [
            AgentStartEvent(),
            AgentEndEvent(messages=[]),
            TurnStartEvent(),
            TurnEndEvent(message=msg),
            MessageStartEvent(message=msg),
            MessageUpdateEvent(message=msg, assistant_message_event=mock_ame),
            MessageEndEvent(message=msg),
            ToolExecutionStartEvent(tool_call_id="t1", tool_name="tool", args={}),
            ToolExecutionUpdateEvent(tool_call_id="t1", tool_name="tool", args={}, partial_result=None),
            ToolExecutionEndEvent(tool_call_id="t1", tool_name="tool", result="ok"),
        ]

        for event in events:
            serialized = serialize_event(event)
            assert serialized is not None
            assert "type" in serialized, f"Missing 'type' key for {type(event).__name__}"

    def test_tool_end_with_nested_pydantic_agent_tool_result(self) -> None:
        """Full pipeline: ToolExecutionEndEvent -> AgentToolResult -> TextContent -> serialized dict."""
        tool_result = AgentToolResult(
            content=[
                TextContent(type="text", text="line one"),
                TextContent(type="text", text="line two"),
            ],
            details={"status": "complete", "elapsed_ms": 150},
        )
        event = ToolExecutionEndEvent(
            tool_call_id="tc_integration",
            tool_name="read_file",
            result=tool_result,
            is_error=False,
        )
        result = serialize_event(event)

        assert result is not None
        assert result["type"] == "tool_end"
        assert result["toolCallId"] == "tc_integration"
        assert result["toolName"] == "read_file"
        assert result["isError"] is False

        inner = result["result"]
        assert len(inner["content"]) == 2
        assert inner["content"][0] == {"type": "text", "text": "line one", "textSignature": None}
        assert inner["content"][1] == {"type": "text", "text": "line two", "textSignature": None}
        assert inner["details"] == {"status": "complete", "elapsed_ms": 150}
