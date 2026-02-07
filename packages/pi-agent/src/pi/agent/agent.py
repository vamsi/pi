"""Agent class - main interface for using the agent runtime.

Manages state, message queues, and orchestrates the agent loop.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from pi.agent.loop import agent_loop, agent_loop_continue
from pi.agent.types import (
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentLoopConfig,
    AgentMessage,
    AgentState,
    AgentTool,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    StreamFn,
    ThinkingLevel,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    TurnEndEvent,
)
from pi.ai.stream import stream_simple
from pi.ai.types import (
    AssistantMessage,
    ImageContent,
    Message,
    Model,
    TextContent,
    ThinkingBudgets,
    Usage,
)


def _default_convert_to_llm(messages: list[AgentMessage]) -> list[Message]:
    """Keep only LLM-compatible messages."""
    return [m for m in messages if m.role in ("user", "assistant", "tool_result")]


class Agent:
    """Stateful agent that orchestrates LLM calls and tool execution."""

    def __init__(
        self,
        *,
        initial_state: AgentState | None = None,
        convert_to_llm: Any = None,
        transform_context: Any = None,
        steering_mode: str = "one-at-a-time",
        follow_up_mode: str = "one-at-a-time",
        stream_fn: StreamFn | None = None,
        session_id: str | None = None,
        get_api_key: Any = None,
        thinking_budgets: ThinkingBudgets | None = None,
        max_retry_delay_ms: int | None = None,
    ) -> None:
        self._state = initial_state or AgentState()
        self._convert_to_llm = convert_to_llm or _default_convert_to_llm
        self._transform_context = transform_context
        self._steering_mode = steering_mode
        self._follow_up_mode = follow_up_mode
        self.stream_fn: StreamFn = stream_fn or stream_simple
        self._session_id = session_id
        self.get_api_key = get_api_key
        self._thinking_budgets = thinking_budgets
        self._max_retry_delay_ms = max_retry_delay_ms

        self._listeners: set[Any] = set()
        self._cancel_event: asyncio.Event | None = None
        self._steering_queue: list[AgentMessage] = []
        self._follow_up_queue: list[AgentMessage] = []
        self._running_task: asyncio.Task | None = None
        self._idle_future: asyncio.Future | None = None

    # --- Properties ---

    @property
    def state(self) -> AgentState:
        return self._state

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @session_id.setter
    def session_id(self, value: str | None) -> None:
        self._session_id = value

    @property
    def thinking_budgets(self) -> ThinkingBudgets | None:
        return self._thinking_budgets

    @thinking_budgets.setter
    def thinking_budgets(self, value: ThinkingBudgets | None) -> None:
        self._thinking_budgets = value

    @property
    def max_retry_delay_ms(self) -> int | None:
        return self._max_retry_delay_ms

    @max_retry_delay_ms.setter
    def max_retry_delay_ms(self, value: int | None) -> None:
        self._max_retry_delay_ms = value

    # --- Subscriptions ---

    def subscribe(self, fn: Any) -> Any:
        """Subscribe to agent events. Returns an unsubscribe function."""
        self._listeners.add(fn)

        def unsubscribe() -> None:
            self._listeners.discard(fn)

        return unsubscribe

    def _emit(self, event: AgentEvent) -> None:
        for listener in self._listeners:
            listener(event)

    # --- State mutators ---

    def set_system_prompt(self, prompt: str) -> None:
        self._state.system_prompt = prompt

    def set_model(self, model: Model) -> None:
        self._state.model = model

    def set_thinking_level(self, level: ThinkingLevel) -> None:
        self._state.thinking_level = level

    def set_tools(self, tools: list[AgentTool]) -> None:
        self._state.tools = tools

    def replace_messages(self, messages: list[AgentMessage]) -> None:
        self._state.messages = list(messages)

    def append_message(self, message: AgentMessage) -> None:
        self._state.messages = [*self._state.messages, message]

    def clear_messages(self) -> None:
        self._state.messages = []

    # --- Queue management ---

    def steer(self, message: AgentMessage) -> None:
        """Queue a steering message to interrupt the agent mid-run."""
        self._steering_queue.append(message)

    def follow_up(self, message: AgentMessage) -> None:
        """Queue a follow-up message for after the agent finishes."""
        self._follow_up_queue.append(message)

    def clear_steering_queue(self) -> None:
        self._steering_queue = []

    def clear_follow_up_queue(self) -> None:
        self._follow_up_queue = []

    def clear_all_queues(self) -> None:
        self._steering_queue = []
        self._follow_up_queue = []

    def has_queued_messages(self) -> bool:
        return len(self._steering_queue) > 0 or len(self._follow_up_queue) > 0

    def _dequeue_steering(self) -> list[AgentMessage]:
        if self._steering_mode == "one-at-a-time":
            if self._steering_queue:
                first = self._steering_queue[0]
                self._steering_queue = self._steering_queue[1:]
                return [first]
            return []
        result = list(self._steering_queue)
        self._steering_queue = []
        return result

    def _dequeue_follow_up(self) -> list[AgentMessage]:
        if self._follow_up_mode == "one-at-a-time":
            if self._follow_up_queue:
                first = self._follow_up_queue[0]
                self._follow_up_queue = self._follow_up_queue[1:]
                return [first]
            return []
        result = list(self._follow_up_queue)
        self._follow_up_queue = []
        return result

    # --- Main methods ---

    async def prompt(
        self, input: str | AgentMessage | list[AgentMessage], images: list[ImageContent] | None = None
    ) -> None:
        """Send a prompt to the agent."""
        if self._state.is_streaming:
            raise RuntimeError("Agent is already processing. Use steer() or follow_up() to queue messages.")

        if self._state.model is None:
            raise RuntimeError("No model configured")

        if isinstance(input, list):
            msgs = input
        elif isinstance(input, str):
            content: list[TextContent | ImageContent] = [TextContent(text=input)]
            if images:
                content.extend(images)
            msgs = [
                type(
                    "UserMsg",
                    (),
                    {
                        "role": "user",
                        "content": content,
                        "timestamp": int(time.time() * 1000),
                    },
                )()
            ]
            # Actually use the proper type
            from pi.ai.types import UserMessage

            msgs = [UserMessage(role="user", content=content, timestamp=int(time.time() * 1000))]
        else:
            msgs = [input]

        await self._run_loop(msgs)

    async def continue_(self) -> None:
        """Continue from current context (for retries and queued messages)."""
        if self._state.is_streaming:
            raise RuntimeError("Agent is already processing.")

        messages = self._state.messages
        if not messages:
            raise RuntimeError("No messages to continue from")

        if messages[-1].role == "assistant":
            steering = self._dequeue_steering()
            if steering:
                await self._run_loop(steering, skip_initial_steering=True)
                return

            follow_up = self._dequeue_follow_up()
            if follow_up:
                await self._run_loop(follow_up)
                return

            raise RuntimeError("Cannot continue from message role: assistant")

        await self._run_loop(None)

    def abort(self) -> None:
        """Abort the current execution."""
        if self._cancel_event:
            self._cancel_event.set()

    async def wait_for_idle(self) -> None:
        """Wait for the agent to finish processing."""
        if self._running_task:
            await self._running_task

    def reset(self) -> None:
        """Reset agent state and queues."""
        self._state.messages = []
        self._state.is_streaming = False
        self._state.stream_message = None
        self._state.pending_tool_calls = set()
        self._state.error = None
        self._steering_queue = []
        self._follow_up_queue = []

    # --- Internal ---

    async def _run_loop(
        self,
        messages: list[AgentMessage] | None = None,
        skip_initial_steering: bool = False,
    ) -> None:
        model = self._state.model
        if model is None:
            raise RuntimeError("No model configured")

        self._cancel_event = asyncio.Event()
        self._state.is_streaming = True
        self._state.stream_message = None
        self._state.error = None

        reasoning = None if self._state.thinking_level == "off" else self._state.thinking_level

        context = AgentContext(
            system_prompt=self._state.system_prompt,
            messages=list(self._state.messages),
            tools=self._state.tools,
        )

        _skip = skip_initial_steering

        async def get_steering() -> list[AgentMessage]:
            nonlocal _skip
            if _skip:
                _skip = False
                return []
            return self._dequeue_steering()

        async def get_follow_up() -> list[AgentMessage]:
            return self._dequeue_follow_up()

        config = AgentLoopConfig(
            model=model,
            reasoning=reasoning,
            session_id=self._session_id,
            thinking_budgets=self._thinking_budgets,
            max_retry_delay_ms=self._max_retry_delay_ms,
            convert_to_llm=self._convert_to_llm,
            transform_context=self._transform_context,
            get_api_key=self.get_api_key,
            get_steering_messages=get_steering,
            get_follow_up_messages=get_follow_up,
        )

        try:
            if messages:
                event_stream = agent_loop(messages, context, config, self._cancel_event, self.stream_fn)
            else:
                event_stream = agent_loop_continue(context, config, self._cancel_event, self.stream_fn)

            async for event in event_stream:
                if isinstance(event, (MessageStartEvent, MessageUpdateEvent)):
                    self._state.stream_message = event.message
                elif isinstance(event, MessageEndEvent):
                    self._state.stream_message = None
                    self.append_message(event.message)
                elif isinstance(event, ToolExecutionStartEvent):
                    self._state.pending_tool_calls = self._state.pending_tool_calls | {event.tool_call_id}
                elif isinstance(event, ToolExecutionEndEvent):
                    self._state.pending_tool_calls = self._state.pending_tool_calls - {event.tool_call_id}
                elif isinstance(event, TurnEndEvent):
                    if hasattr(event.message, "error_message") and event.message.error_message:
                        self._state.error = event.message.error_message
                elif isinstance(event, AgentEndEvent):
                    self._state.is_streaming = False
                    self._state.stream_message = None

                self._emit(event)

        except Exception as err:
            error_msg = AssistantMessage(
                role="assistant",
                content=[TextContent(text="")],
                api=model.api,
                provider=model.provider,
                model=model.id,
                usage=Usage(),
                stop_reason="aborted" if (self._cancel_event and self._cancel_event.is_set()) else "error",
                error_message=str(err),
                timestamp=int(time.time() * 1000),
            )
            self.append_message(error_msg)
            self._state.error = str(err)
            self._emit(AgentEndEvent(messages=[error_msg]))

        finally:
            self._state.is_streaming = False
            self._state.stream_message = None
            self._state.pending_tool_calls = set()
            self._cancel_event = None
