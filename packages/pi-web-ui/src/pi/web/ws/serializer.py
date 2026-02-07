"""Serialize AgentEvents to JSON-safe dicts for WebSocket transmission."""

from __future__ import annotations

from typing import Any

from pi.agent.types import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
)


def serialize_message(msg: Any) -> dict[str, Any]:
    """Serialize a Pydantic message (or any AgentMessage) to a camelCase dict."""
    if hasattr(msg, "model_dump"):
        return msg.model_dump(by_alias=True)
    if hasattr(msg, "__dict__"):
        return {k: v for k, v in msg.__dict__.items() if not k.startswith("_")}
    return {}


def serialize_event(event: AgentEvent) -> dict[str, Any] | None:
    """Convert an AgentEvent to a JSON-serializable dict for the WebSocket."""
    match event:
        case AgentStartEvent():
            return {"type": "agent_start"}

        case AgentEndEvent():
            return {"type": "agent_end"}

        case TurnStartEvent():
            return {"type": "turn_start"}

        case TurnEndEvent():
            return {"type": "turn_end"}

        case MessageStartEvent():
            return {
                "type": "message_start",
                "message": serialize_message(event.message),
            }

        case MessageUpdateEvent():
            return {
                "type": "message_update",
                "message": serialize_message(event.message),
            }

        case MessageEndEvent():
            return {
                "type": "message_end",
                "message": serialize_message(event.message),
            }

        case ToolExecutionStartEvent():
            return {
                "type": "tool_start",
                "toolCallId": event.tool_call_id,
                "toolName": event.tool_name,
                "args": event.args,
            }

        case ToolExecutionUpdateEvent():
            return {
                "type": "tool_update",
                "toolCallId": event.tool_call_id,
                "toolName": event.tool_name,
                "args": event.args,
                "partialResult": event.partial_result,
            }

        case ToolExecutionEndEvent():
            return {
                "type": "tool_end",
                "toolCallId": event.tool_call_id,
                "toolName": event.tool_name,
                "result": _serialize_tool_result(event.result),
                "isError": event.is_error,
            }

        case _:
            return None


def _serialize_tool_result(result: Any) -> Any:
    """Serialize a tool result, handling AgentToolResult and Pydantic models."""
    if result is None:
        return None
    if hasattr(result, "model_dump"):
        return result.model_dump(by_alias=True)
    if hasattr(result, "content"):
        content = result.content
        serialized_content = []
        for item in content:
            if hasattr(item, "model_dump"):
                serialized_content.append(item.model_dump(by_alias=True))
            else:
                serialized_content.append(str(item))
        return {"content": serialized_content, "details": getattr(result, "details", None)}
    return result
