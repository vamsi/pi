"""Core types for the agent runtime."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pi.ai.events import AssistantMessageEventStream
from pi.ai.types import (
    AssistantMessageEvent,
    ImageContent,
    Message,
    Model,
    TextContent,
    ThinkingBudgets,
    ToolResultMessage,
)

if TYPE_CHECKING:
    import asyncio

ThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh"]

# Stream function type - can return sync or async
StreamFn = Callable[..., AssistantMessageEventStream | Awaitable[AssistantMessageEventStream]]


@dataclass
class AgentToolResult:
    """Result from executing a tool."""

    content: list[TextContent | ImageContent] = field(default_factory=list)
    details: Any = None


AgentToolUpdateCallback = Callable[[AgentToolResult], None]


@dataclass
class AgentTool:
    """Tool with execution capability."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    label: str = ""
    execute: Callable[..., Awaitable[AgentToolResult]] | None = None


# AgentMessage is a union of LLM messages + any custom app messages
AgentMessage = Message  # Apps can use their own union type


@dataclass
class AgentState:
    """Current state of the agent."""

    system_prompt: str = ""
    model: Model | None = None
    thinking_level: ThinkingLevel = "off"
    tools: list[AgentTool] = field(default_factory=list)
    messages: list[AgentMessage] = field(default_factory=list)
    is_streaming: bool = False
    stream_message: AgentMessage | None = None
    pending_tool_calls: set[str] = field(default_factory=set)
    error: str | None = None


@dataclass
class AgentContext:
    """Context for agent execution."""

    system_prompt: str = ""
    messages: list[AgentMessage] = field(default_factory=list)
    tools: list[AgentTool] | None = None


@dataclass
class AgentLoopConfig:
    """Configuration for the agent loop."""

    model: Model
    convert_to_llm: Callable[[list[AgentMessage]], list[Message] | Awaitable[list[Message]]]
    reasoning: str | None = None
    session_id: str | None = None
    thinking_budgets: ThinkingBudgets | None = None
    max_retry_delay_ms: int | None = None
    api_key: str | None = None
    transform_context: Callable[[list[AgentMessage], asyncio.Event | None], Awaitable[list[AgentMessage]]] | None = None
    get_api_key: Callable[[str], Awaitable[str | None] | str | None] | None = None
    get_steering_messages: Callable[[], Awaitable[list[AgentMessage]]] | None = None
    get_follow_up_messages: Callable[[], Awaitable[list[AgentMessage]]] | None = None


# --- Agent Events ---


@dataclass
class AgentStartEvent:
    type: Literal["agent_start"] = "agent_start"


@dataclass
class AgentEndEvent:
    messages: list[AgentMessage]
    type: Literal["agent_end"] = "agent_end"


@dataclass
class TurnStartEvent:
    type: Literal["turn_start"] = "turn_start"


@dataclass
class TurnEndEvent:
    message: AgentMessage
    tool_results: list[ToolResultMessage] = field(default_factory=list)
    type: Literal["turn_end"] = "turn_end"


@dataclass
class MessageStartEvent:
    message: AgentMessage
    type: Literal["message_start"] = "message_start"


@dataclass
class MessageUpdateEvent:
    message: AgentMessage
    assistant_message_event: AssistantMessageEvent
    type: Literal["message_update"] = "message_update"


@dataclass
class MessageEndEvent:
    message: AgentMessage
    type: Literal["message_end"] = "message_end"


@dataclass
class ToolExecutionStartEvent:
    tool_call_id: str
    tool_name: str
    args: Any
    type: Literal["tool_execution_start"] = "tool_execution_start"


@dataclass
class ToolExecutionUpdateEvent:
    tool_call_id: str
    tool_name: str
    args: Any
    partial_result: Any
    type: Literal["tool_execution_update"] = "tool_execution_update"


@dataclass
class ToolExecutionEndEvent:
    tool_call_id: str
    tool_name: str
    result: Any
    is_error: bool = False
    type: Literal["tool_execution_end"] = "tool_execution_end"


AgentEvent = (
    AgentStartEvent
    | AgentEndEvent
    | TurnStartEvent
    | TurnEndEvent
    | MessageStartEvent
    | MessageUpdateEvent
    | MessageEndEvent
    | ToolExecutionStartEvent
    | ToolExecutionUpdateEvent
    | ToolExecutionEndEvent
)
