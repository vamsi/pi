"""Agent loop implementation.

Transforms AgentMessage[] to Message[] only at the LLM call boundary.
Handles tool execution, steering, and follow-up message processing.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from copy import deepcopy
from typing import Any

from pi.agent.types import (
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentLoopConfig,
    AgentMessage,
    AgentStartEvent,
    AgentTool,
    AgentToolResult,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    StreamFn,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from pi.ai.events import EventStream
from pi.ai.stream import stream_simple
from pi.ai.types import (
    AssistantMessage,
    Context,
    SimpleStreamOptions,
    TextContent,
    Tool,
    ToolCall,
    ToolResultMessage,
)
from pi.ai.utils.validation import validate_tool_arguments


def _create_agent_stream() -> EventStream[AgentEvent, list[AgentMessage]]:
    return EventStream[AgentEvent, list[AgentMessage]](
        is_complete=lambda event: isinstance(event, AgentEndEvent),
        extract_result=lambda event: event.messages if isinstance(event, AgentEndEvent) else [],
    )


def agent_loop(
    prompts: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    cancel_event: asyncio.Event | None = None,
    stream_fn: StreamFn | None = None,
) -> EventStream[AgentEvent, list[AgentMessage]]:
    """Start an agent loop with new prompt messages."""
    stream = _create_agent_stream()

    async def _run() -> None:
        new_messages: list[AgentMessage] = list(prompts)
        current_context = AgentContext(
            system_prompt=context.system_prompt,
            messages=list(context.messages) + list(prompts),
            tools=context.tools,
        )

        stream.push(AgentStartEvent())
        stream.push(TurnStartEvent())
        for prompt in prompts:
            stream.push(MessageStartEvent(message=prompt))
            stream.push(MessageEndEvent(message=prompt))

        await _run_loop(current_context, new_messages, config, cancel_event, stream, stream_fn)

    stream._background_task = asyncio.ensure_future(_run())
    return stream


def agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    cancel_event: asyncio.Event | None = None,
    stream_fn: StreamFn | None = None,
) -> EventStream[AgentEvent, list[AgentMessage]]:
    """Continue an agent loop from existing context without adding new messages."""
    if not context.messages:
        raise ValueError("Cannot continue: no messages in context")
    if context.messages[-1].role == "assistant":
        raise ValueError("Cannot continue from message role: assistant")

    stream = _create_agent_stream()

    async def _run() -> None:
        new_messages: list[AgentMessage] = []
        current_context = AgentContext(
            system_prompt=context.system_prompt,
            messages=list(context.messages),
            tools=context.tools,
        )

        stream.push(AgentStartEvent())
        stream.push(TurnStartEvent())

        await _run_loop(current_context, new_messages, config, cancel_event, stream, stream_fn)

    stream._background_task = asyncio.ensure_future(_run())
    return stream


async def _run_loop(
    current_context: AgentContext,
    new_messages: list[AgentMessage],
    config: AgentLoopConfig,
    cancel_event: asyncio.Event | None,
    stream: EventStream[AgentEvent, list[AgentMessage]],
    stream_fn: StreamFn | None,
) -> None:
    """Main loop logic shared by agent_loop and agent_loop_continue."""
    first_turn = True
    pending_messages: list[AgentMessage] = []

    if config.get_steering_messages:
        pending_messages = await config.get_steering_messages()

    while True:
        has_more_tool_calls = True
        steering_after_tools: list[AgentMessage] | None = None

        while has_more_tool_calls or pending_messages:
            if not first_turn:
                stream.push(TurnStartEvent())
            else:
                first_turn = False

            if pending_messages:
                for message in pending_messages:
                    stream.push(MessageStartEvent(message=message))
                    stream.push(MessageEndEvent(message=message))
                    current_context.messages.append(message)
                    new_messages.append(message)
                pending_messages = []

            # Stream assistant response
            message = await _stream_assistant_response(current_context, config, cancel_event, stream, stream_fn)
            new_messages.append(message)

            if message.stop_reason in ("error", "aborted"):
                stream.push(TurnEndEvent(message=message, tool_results=[]))
                stream.push(AgentEndEvent(messages=new_messages))
                stream.end(new_messages)
                return

            # Check for tool calls
            tool_calls = [c for c in message.content if isinstance(c, ToolCall)]
            has_more_tool_calls = len(tool_calls) > 0

            tool_results: list[ToolResultMessage] = []
            if has_more_tool_calls:
                execution = await _execute_tool_calls(
                    current_context.tools,
                    message,
                    cancel_event,
                    stream,
                    config.get_steering_messages,
                )
                tool_results.extend(execution["tool_results"])
                steering_after_tools = execution.get("steering_messages")

                for result in tool_results:
                    current_context.messages.append(result)
                    new_messages.append(result)

            stream.push(TurnEndEvent(message=message, tool_results=tool_results))

            if steering_after_tools:
                pending_messages = steering_after_tools
                steering_after_tools = None
            elif config.get_steering_messages:
                pending_messages = await config.get_steering_messages()

        # Check for follow-up messages
        if config.get_follow_up_messages:
            follow_up = await config.get_follow_up_messages()
            if follow_up:
                pending_messages = follow_up
                continue

        break

    stream.push(AgentEndEvent(messages=new_messages))
    stream.end(new_messages)


async def _stream_assistant_response(
    context: AgentContext,
    config: AgentLoopConfig,
    cancel_event: asyncio.Event | None,
    stream: EventStream[AgentEvent, list[AgentMessage]],
    stream_fn: StreamFn | None,
) -> AssistantMessage:
    """Stream an assistant response from the LLM."""
    messages = context.messages
    if config.transform_context:
        messages = await config.transform_context(messages, cancel_event)

    # Convert to LLM messages
    result = config.convert_to_llm(messages)
    if inspect.isawaitable(result):
        llm_messages = await result
    else:
        llm_messages = result

    llm_context = Context(
        system_prompt=context.system_prompt,
        messages=llm_messages,
        tools=[Tool(name=t.name, description=t.description, parameters=t.parameters) for t in (context.tools or [])],
    )

    fn = stream_fn or stream_simple

    # Resolve API key
    resolved_key = config.api_key
    if config.get_api_key:
        key_result = config.get_api_key(config.model.provider)
        if inspect.isawaitable(key_result):
            resolved_key = await key_result or resolved_key
        else:
            resolved_key = key_result or resolved_key

    options = SimpleStreamOptions(
        reasoning=config.reasoning,
        session_id=config.session_id,
        thinking_budgets=config.thinking_budgets,
        max_retry_delay_ms=config.max_retry_delay_ms,
        api_key=resolved_key,
    )

    response_result = fn(config.model, llm_context, options)
    if inspect.isawaitable(response_result):
        response = await response_result
    else:
        response = response_result

    partial_message: AssistantMessage | None = None
    added_partial = False

    async for event in response:
        if event.type == "start":
            partial_message = event.partial
            context.messages.append(partial_message)
            added_partial = True
            stream.push(MessageStartEvent(message=deepcopy(partial_message)))

        elif event.type in (
            "text_start",
            "text_delta",
            "text_end",
            "thinking_start",
            "thinking_delta",
            "thinking_end",
            "toolcall_start",
            "toolcall_delta",
            "toolcall_end",
        ):
            if partial_message:
                partial_message = event.partial
                context.messages[-1] = partial_message
                stream.push(
                    MessageUpdateEvent(
                        assistant_message_event=event,
                        message=deepcopy(partial_message),
                    )
                )

        elif event.type in ("done", "error"):
            final_message = await response.result()
            if added_partial:
                context.messages[-1] = final_message
            else:
                context.messages.append(final_message)
            if not added_partial:
                stream.push(MessageStartEvent(message=deepcopy(final_message)))
            stream.push(MessageEndEvent(message=final_message))
            return final_message

    return await response.result()


async def _execute_tool_calls(
    tools: list[AgentTool] | None,
    assistant_message: AssistantMessage,
    cancel_event: asyncio.Event | None,
    stream: EventStream[AgentEvent, list[AgentMessage]],
    get_steering_messages: Any | None,
) -> dict[str, Any]:
    """Execute tool calls from an assistant message."""
    tool_calls = [c for c in assistant_message.content if isinstance(c, ToolCall)]
    results: list[ToolResultMessage] = []
    steering_messages: list[AgentMessage] | None = None

    for index, tc in enumerate(tool_calls):
        tool = next((t for t in (tools or []) if t.name == tc.name), None)

        stream.push(ToolExecutionStartEvent(tool_call_id=tc.id, tool_name=tc.name, args=tc.arguments))

        result: AgentToolResult
        is_error = False

        try:
            if tool is None:
                raise ValueError(f"Tool {tc.name} not found")
            if tool.execute is None:
                raise ValueError(f"Tool {tc.name} has no execute function")

            # Validate arguments
            errors = validate_tool_arguments(tool.parameters, tc.arguments)
            if errors:
                raise ValueError(f"Invalid arguments: {'; '.join(errors)}")

            def on_update(partial: AgentToolResult, _tc: Any = tc) -> None:
                stream.push(
                    ToolExecutionUpdateEvent(
                        tool_call_id=_tc.id,
                        tool_name=_tc.name,
                        args=_tc.arguments,
                        partial_result=partial,
                    )
                )

            result = await tool.execute(tc.id, tc.arguments, cancel_event, on_update)

        except Exception as e:
            result = AgentToolResult(content=[TextContent(text=str(e))], details={})
            is_error = True

        stream.push(
            ToolExecutionEndEvent(
                tool_call_id=tc.id,
                tool_name=tc.name,
                result=result,
                is_error=is_error,
            )
        )

        tool_result_msg = ToolResultMessage(
            tool_call_id=tc.id,
            tool_name=tc.name,
            content=result.content,
            details=result.details,
            is_error=is_error,
            timestamp=int(time.time() * 1000),
        )

        results.append(tool_result_msg)
        stream.push(MessageStartEvent(message=tool_result_msg))
        stream.push(MessageEndEvent(message=tool_result_msg))

        # Check for steering
        if get_steering_messages:
            steering = await get_steering_messages()
            if steering:
                steering_messages = steering
                remaining = tool_calls[index + 1 :]
                for skipped in remaining:
                    results.append(_skip_tool_call(skipped, stream))
                break

    return {"tool_results": results, "steering_messages": steering_messages}


def _skip_tool_call(
    tool_call: ToolCall,
    stream: EventStream[AgentEvent, list[AgentMessage]],
) -> ToolResultMessage:
    """Create a skipped tool result."""
    result = AgentToolResult(
        content=[TextContent(text="Skipped due to queued user message.")],
        details={},
    )

    stream.push(ToolExecutionStartEvent(tool_call_id=tool_call.id, tool_name=tool_call.name, args=tool_call.arguments))
    stream.push(
        ToolExecutionEndEvent(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            result=result,
            is_error=True,
        )
    )

    msg = ToolResultMessage(
        tool_call_id=tool_call.id,
        tool_name=tool_call.name,
        content=result.content,
        details={},
        is_error=True,
        timestamp=int(time.time() * 1000),
    )

    stream.push(MessageStartEvent(message=msg))
    stream.push(MessageEndEvent(message=msg))
    return msg
