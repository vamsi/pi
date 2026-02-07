"""Anthropic Messages API provider implementation."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any, Literal

import anthropic

from pi.ai.env import get_env_api_key
from pi.ai.events import AssistantMessageEventStream
from pi.ai.models import calculate_cost
from pi.ai.providers.options import adjust_max_tokens_for_thinking, build_base_options
from pi.ai.providers.transform import transform_messages
from pi.ai.types import (
    AssistantMessage,
    CacheRetention,
    Context,
    DoneEvent,
    ErrorEvent,
    ImageContent,
    Message,
    Model,
    SimpleStreamOptions,
    StartEvent,
    StopReason,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingContent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingLevel,
    ThinkingStartEvent,
    Tool,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    Usage,
)
from pi.ai.utils.json import parse_streaming_json

AnthropicEffort = Literal["low", "medium", "high", "max"]


@dataclass
class AnthropicOptions:
    """Extended options for the Anthropic provider."""

    temperature: float | None = None
    max_tokens: int | None = None
    api_key: str | None = None
    cache_retention: CacheRetention = "short"
    session_id: str | None = None
    headers: dict[str, str] | None = None
    max_retry_delay_ms: int | None = None
    thinking_enabled: bool = False
    thinking_budget_tokens: int | None = None
    effort: AnthropicEffort | None = None
    interleaved_thinking: bool = True
    tool_choice: str | dict[str, str] | None = None
    on_payload: Any = None


def _resolve_cache_retention(cache_retention: CacheRetention | None = None) -> CacheRetention:
    if cache_retention:
        return cache_retention
    if os.environ.get("PI_CACHE_RETENTION") == "long":
        return "long"
    return "short"


def _get_cache_control(
    base_url: str,
    cache_retention: CacheRetention | None = None,
) -> tuple[CacheRetention, dict[str, Any] | None]:
    retention = _resolve_cache_retention(cache_retention)
    if retention == "none":
        return retention, None
    ttl = "1h" if retention == "long" and "api.anthropic.com" in base_url else None
    cache_control: dict[str, Any] = {"type": "ephemeral"}
    if ttl:
        cache_control["ttl"] = ttl
    return retention, cache_control


def _normalize_tool_call_id(id: str) -> str:
    """Normalize tool call IDs to Anthropic's required pattern."""
    import re

    return re.sub(r"[^a-zA-Z0-9_-]", "_", id)[:64]


def _is_oauth_token(api_key: str) -> bool:
    return "sk-ant-oat" in api_key


def _map_stop_reason(reason: str) -> StopReason:
    mapping = {
        "end_turn": "stop",
        "max_tokens": "length",
        "tool_use": "tool_use",
        "refusal": "error",
        "pause_turn": "stop",
        "stop_sequence": "stop",
        "sensitive": "error",
    }
    result = mapping.get(reason)
    if result is None:
        raise ValueError(f"Unhandled stop reason: {reason}")
    return result


def _supports_adaptive_thinking(model_id: str) -> bool:
    return "opus-4-6" in model_id or "opus-4.6" in model_id


def _map_thinking_level_to_effort(level: ThinkingLevel | None) -> AnthropicEffort:
    mapping = {
        "minimal": "low",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": "max",
    }
    return mapping.get(level or "high", "high")


def _convert_content_blocks(content: list[TextContent | ImageContent]) -> str | list[dict[str, Any]]:
    """Convert content blocks to Anthropic API format."""
    has_images = any(c.type == "image" for c in content)
    if not has_images:
        return "\n".join(c.text for c in content if isinstance(c, TextContent))

    blocks: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, TextContent):
            blocks.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageContent):
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": block.mime_type,
                        "data": block.data,
                    },
                }
            )

    if not any(b["type"] == "text" for b in blocks):
        blocks.insert(0, {"type": "text", "text": "(see attached image)"})

    return blocks


def _convert_messages(
    messages: list[Message],
    model: Model,
    cache_control: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Convert messages to Anthropic API format."""
    params: list[dict[str, Any]] = []
    transformed = transform_messages(messages, current_model=model.id, normalize_tool_id=_normalize_tool_call_id)

    i = 0
    while i < len(transformed):
        msg = transformed[i]

        if isinstance(msg, type) and hasattr(msg, "role"):
            # This shouldn't happen but handle gracefully
            i += 1
            continue

        if msg.role == "user":
            if isinstance(msg.content, str):
                if msg.content.strip():
                    params.append({"role": "user", "content": msg.content})
            else:
                blocks = []
                for item in msg.content:
                    if isinstance(item, TextContent):
                        if item.text.strip():
                            blocks.append({"type": "text", "text": item.text})
                    elif isinstance(item, ImageContent) and "image" in model.input:
                        blocks.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": item.mime_type,
                                    "data": item.data,
                                },
                            }
                        )
                if blocks:
                    params.append({"role": "user", "content": blocks})

        elif msg.role == "assistant":
            blocks = []
            for block in msg.content:
                if isinstance(block, TextContent):
                    if block.text.strip():
                        blocks.append({"type": "text", "text": block.text})
                elif isinstance(block, ThinkingContent):
                    if block.thinking.strip():
                        if block.thinking_signature and block.thinking_signature.strip():
                            blocks.append(
                                {
                                    "type": "thinking",
                                    "thinking": block.thinking,
                                    "signature": block.thinking_signature,
                                }
                            )
                        else:
                            blocks.append({"type": "text", "text": block.thinking})
                elif isinstance(block, ToolCall):
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.arguments or {},
                        }
                    )
            if blocks:
                params.append({"role": "assistant", "content": blocks})

        elif msg.role == "tool_result":
            tool_results = []
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id,
                    "content": _convert_content_blocks(msg.content),
                    "is_error": msg.is_error,
                }
            )

            # Collect consecutive tool results
            j = i + 1
            while j < len(transformed) and transformed[j].role == "tool_result":
                next_msg = transformed[j]
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": next_msg.tool_call_id,
                        "content": _convert_content_blocks(next_msg.content),
                        "is_error": next_msg.is_error,
                    }
                )
                j += 1
            i = j - 1

            params.append({"role": "user", "content": tool_results})

        i += 1

    # Add cache control to last user message
    if cache_control and params:
        last = params[-1]
        if last["role"] == "user":
            if isinstance(last["content"], list) and last["content"]:
                last_block = last["content"][-1]
                if last_block.get("type") in ("text", "image", "tool_result"):
                    last_block["cache_control"] = cache_control
            elif isinstance(last["content"], str):
                last["content"] = [{"type": "text", "text": last["content"], "cache_control": cache_control}]

    return params


def _convert_tools(tools: list[Tool]) -> list[dict[str, Any]]:
    """Convert tools to Anthropic API format."""
    result = []
    for tool in tools:
        schema = tool.parameters
        result.append(
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": {
                    "type": "object",
                    "properties": schema.get("properties", {}),
                    "required": schema.get("required", []),
                },
            }
        )
    return result


def stream_anthropic(
    model: Model,
    context: Context,
    options: AnthropicOptions | None = None,
) -> AssistantMessageEventStream:
    """Stream a response from the Anthropic Messages API."""
    event_stream = AssistantMessageEventStream()

    async def _run() -> None:
        output = AssistantMessage(
            role="assistant",
            content=[],
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage=Usage(),
            stop_reason="stop",
            timestamp=int(time.time() * 1000),
        )

        try:
            api_key = (options and options.api_key) or get_env_api_key(model.provider) or ""
            _, cache_control = _get_cache_control(model.base_url, options.cache_retention if options else None)

            beta_features = ["fine-grained-tool-streaming-2025-05-14"]
            if options is None or options.interleaved_thinking:
                beta_features.append("interleaved-thinking-2025-05-14")

            headers = {"anthropic-beta": ",".join(beta_features)}
            if model.headers:
                headers.update(model.headers)
            if options and options.headers:
                headers.update(options.headers)

            client = anthropic.AsyncAnthropic(
                api_key=api_key,
                base_url=model.base_url,
                default_headers=headers,
            )

            # Build params
            params: dict[str, Any] = {
                "model": model.id,
                "messages": _convert_messages(context.messages, model, cache_control),
                "max_tokens": (options and options.max_tokens) or (model.max_tokens // 3) or 4096,
                "stream": True,
            }

            if context.system_prompt:
                system_blocks = [{"type": "text", "text": context.system_prompt}]
                if cache_control:
                    system_blocks[0]["cache_control"] = cache_control
                params["system"] = system_blocks

            if options and options.temperature is not None:
                params["temperature"] = options.temperature

            if context.tools:
                params["tools"] = _convert_tools(context.tools)

            if options and options.thinking_enabled and model.reasoning:
                if _supports_adaptive_thinking(model.id):
                    params["thinking"] = {"type": "adaptive"}
                    if options.effort:
                        params["output_config"] = {"effort": options.effort}
                else:
                    params["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": options.thinking_budget_tokens or 1024,
                    }

            if options and options.tool_choice:
                if isinstance(options.tool_choice, str):
                    params["tool_choice"] = {"type": options.tool_choice}
                else:
                    params["tool_choice"] = options.tool_choice

            if options and options.on_payload:
                options.on_payload(params)

            event_stream.push(StartEvent(partial=output))

            # Track blocks with their API index for matching deltas
            block_api_indices: list[int] = []  # maps our content index -> API block index

            async with client.messages.stream(**params) as stream:
                async for event in stream:
                    if event.type == "message_start":
                        u = event.message.usage
                        output.usage.input = u.input_tokens or 0
                        output.usage.output = u.output_tokens or 0
                        output.usage.cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
                        output.usage.cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0
                        output.usage.total_tokens = (
                            output.usage.input
                            + output.usage.output
                            + output.usage.cache_read
                            + output.usage.cache_write
                        )
                        calculate_cost(model, output.usage)

                    elif event.type == "content_block_start":
                        cb = event.content_block
                        if cb.type == "text":
                            output.content.append(TextContent(text=""))
                            block_api_indices.append(event.index)
                            event_stream.push(TextStartEvent(content_index=len(output.content) - 1, partial=output))
                        elif cb.type == "thinking":
                            output.content.append(ThinkingContent(thinking="", thinking_signature=""))
                            block_api_indices.append(event.index)
                            event_stream.push(ThinkingStartEvent(content_index=len(output.content) - 1, partial=output))
                        elif cb.type == "tool_use":
                            output.content.append(
                                ToolCall(
                                    id=cb.id,
                                    name=cb.name,
                                    arguments=cb.input if isinstance(cb.input, dict) else {},
                                )
                            )
                            block_api_indices.append(event.index)
                            event_stream.push(ToolCallStartEvent(content_index=len(output.content) - 1, partial=output))

                    elif event.type == "content_block_delta":
                        # Find our content index from API block index
                        try:
                            idx = block_api_indices.index(event.index)
                        except ValueError:
                            continue

                        block = output.content[idx]
                        delta = event.delta

                        if delta.type == "text_delta":
                            if isinstance(block, TextContent):
                                block.text += delta.text
                                event_stream.push(TextDeltaEvent(content_index=idx, delta=delta.text, partial=output))
                        elif delta.type == "thinking_delta":
                            if isinstance(block, ThinkingContent):
                                block.thinking += delta.thinking
                                event_stream.push(
                                    ThinkingDeltaEvent(content_index=idx, delta=delta.thinking, partial=output)
                                )
                        elif delta.type == "input_json_delta":
                            if isinstance(block, ToolCall):
                                # Accumulate partial JSON
                                if not hasattr(block, "_partial_json"):
                                    block._partial_json = ""
                                block._partial_json += delta.partial_json
                                block.arguments = parse_streaming_json(block._partial_json)
                                event_stream.push(
                                    ToolCallDeltaEvent(content_index=idx, delta=delta.partial_json, partial=output)
                                )
                        elif delta.type == "signature_delta" and isinstance(block, ThinkingContent):
                            block.thinking_signature = (block.thinking_signature or "") + delta.signature

                    elif event.type == "content_block_stop":
                        try:
                            idx = block_api_indices.index(event.index)
                        except ValueError:
                            continue

                        block = output.content[idx]
                        if isinstance(block, TextContent):
                            event_stream.push(TextEndEvent(content_index=idx, content=block.text, partial=output))
                        elif isinstance(block, ThinkingContent):
                            event_stream.push(
                                ThinkingEndEvent(content_index=idx, content=block.thinking, partial=output)
                            )
                        elif isinstance(block, ToolCall):
                            if hasattr(block, "_partial_json"):
                                block.arguments = parse_streaming_json(block._partial_json)
                                del block._partial_json
                            event_stream.push(ToolCallEndEvent(content_index=idx, tool_call=block, partial=output))

                    elif event.type == "message_delta":
                        if hasattr(event.delta, "stop_reason") and event.delta.stop_reason:
                            output.stop_reason = _map_stop_reason(event.delta.stop_reason)
                        u = event.usage
                        if hasattr(u, "input_tokens") and u.input_tokens is not None:
                            output.usage.input = u.input_tokens
                        if hasattr(u, "output_tokens") and u.output_tokens is not None:
                            output.usage.output = u.output_tokens
                        if hasattr(u, "cache_read_input_tokens") and u.cache_read_input_tokens is not None:
                            output.usage.cache_read = u.cache_read_input_tokens
                        if hasattr(u, "cache_creation_input_tokens") and u.cache_creation_input_tokens is not None:
                            output.usage.cache_write = u.cache_creation_input_tokens
                        output.usage.total_tokens = (
                            output.usage.input
                            + output.usage.output
                            + output.usage.cache_read
                            + output.usage.cache_write
                        )
                        calculate_cost(model, output.usage)

            event_stream.push(DoneEvent(reason=output.stop_reason, message=output))
            event_stream.end()

        except Exception as e:
            output.stop_reason = "aborted" if "aborted" in str(e).lower() or "cancelled" in str(e).lower() else "error"
            output.error_message = str(e)
            event_stream.push(ErrorEvent(reason=output.stop_reason, error=output))
            event_stream.end()

    event_stream._background_task = asyncio.ensure_future(_run())
    return event_stream


def stream_simple_anthropic(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    """Stream using the simple API with thinking level support."""
    api_key = (options and options.api_key) or get_env_api_key(model.provider)
    if not api_key:
        raise ValueError(f"No API key for provider: {model.provider}")

    base = build_base_options(model, options)

    if not options or not options.reasoning:
        return stream_anthropic(
            model,
            context,
            AnthropicOptions(
                temperature=base.temperature,
                max_tokens=base.max_tokens,
                api_key=api_key,
                cache_retention=base.cache_retention,
                session_id=base.session_id,
                headers=base.headers,
                thinking_enabled=False,
            ),
        )

    if _supports_adaptive_thinking(model.id):
        effort = _map_thinking_level_to_effort(options.reasoning)
        return stream_anthropic(
            model,
            context,
            AnthropicOptions(
                temperature=base.temperature,
                max_tokens=base.max_tokens,
                api_key=api_key,
                cache_retention=base.cache_retention,
                session_id=base.session_id,
                headers=base.headers,
                thinking_enabled=True,
                effort=effort,
            ),
        )

    custom_budgets = None
    if options.thinking_budgets:
        custom_budgets = {
            k: v
            for k, v in {
                "minimal": options.thinking_budgets.minimal,
                "low": options.thinking_budgets.low,
                "medium": options.thinking_budgets.medium,
                "high": options.thinking_budgets.high,
            }.items()
            if v is not None
        }

    adjusted_max, thinking_budget = adjust_max_tokens_for_thinking(
        base.max_tokens or 0,
        options.reasoning,
        custom_budgets,
    )

    return stream_anthropic(
        model,
        context,
        AnthropicOptions(
            temperature=base.temperature,
            max_tokens=adjusted_max,
            api_key=api_key,
            cache_retention=base.cache_retention,
            session_id=base.session_id,
            headers=base.headers,
            thinking_enabled=True,
            thinking_budget_tokens=thinking_budget,
        ),
    )
