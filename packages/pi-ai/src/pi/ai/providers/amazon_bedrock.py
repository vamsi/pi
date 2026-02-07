"""Amazon Bedrock Converse Stream provider implementation."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from pi.ai.events import AssistantMessageEventStream
from pi.ai.models import calculate_cost
from pi.ai.providers.options import adjust_max_tokens_for_thinking, build_base_options, clamp_reasoning
from pi.ai.providers.transform import transform_messages
from pi.ai.types import (
    AssistantMessage,
    CacheRetention,
    Context,
    DoneEvent,
    ErrorEvent,
    ImageContent,
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
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    ToolResultMessage,
    Usage,
)
from pi.ai.utils.json import parse_streaming_json

_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def _sanitize(text: str) -> str:
    return _SURROGATE_RE.sub("\ufffd", text)


@dataclass
class BedrockOptions:
    temperature: float | None = None
    max_tokens: int | None = None
    api_key: str | None = None
    cache_retention: CacheRetention = "short"
    session_id: str | None = None
    headers: dict[str, str] | None = None
    max_retry_delay_ms: int | None = None
    on_payload: Any = None
    region: str | None = None
    profile: str | None = None
    tool_choice: str | dict[str, Any] | None = None
    reasoning: ThinkingLevel | None = None
    thinking_budgets: dict[str, int] | None = None
    interleaved_thinking: bool = False


def stream_bedrock(
    model: Model,
    context: Context,
    options: BedrockOptions | None = None,
) -> AssistantMessageEventStream:
    """Stream a response from Amazon Bedrock Converse Stream API."""
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

        # Block tracking with API indices
        block_indices: dict[int, int] = {}  # api_index -> content_index
        partial_json: dict[int, str] = {}  # content_index -> accumulated json

        try:
            import boto3

            opts = options or BedrockOptions()
            region = opts.region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"

            client_kwargs: dict[str, Any] = {"service_name": "bedrock-runtime", "region_name": region}
            if opts.profile:
                import botocore.session

                session = botocore.session.Session(profile=opts.profile)
                client_kwargs["session"] = boto3.Session(botocore_session=session)

            if os.environ.get("AWS_BEDROCK_SKIP_AUTH") == "1":
                client_kwargs["aws_access_key_id"] = "dummy-access-key"
                client_kwargs["aws_secret_access_key"] = "dummy-secret-key"

            client = boto3.client(**client_kwargs)

            cache_retention = _resolve_cache_retention(opts.cache_retention)
            command_input = _build_command_input(model, context, opts, cache_retention)
            if opts.on_payload:
                opts.on_payload(command_input)

            response = client.converse_stream(**command_input)

            for item_wrapper in response.get("stream", []):
                if "messageStart" in item_wrapper:
                    event_stream.push(StartEvent(partial=output))

                elif "contentBlockStart" in item_wrapper:
                    event_data = item_wrapper["contentBlockStart"]
                    api_index = event_data.get("contentBlockIndex", 0)
                    start = event_data.get("start", {})

                    if "toolUse" in start:
                        tc = ToolCall(
                            id=start["toolUse"].get("toolUseId", ""),
                            name=start["toolUse"].get("name", ""),
                            arguments={},
                        )
                        output.content.append(tc)
                        ci = len(output.content) - 1
                        block_indices[api_index] = ci
                        partial_json[ci] = ""
                        event_stream.push(ToolCallStartEvent(content_index=ci, partial=output))

                elif "contentBlockDelta" in item_wrapper:
                    event_data = item_wrapper["contentBlockDelta"]
                    api_index = event_data.get("contentBlockIndex", 0)
                    delta = event_data.get("delta", {})

                    if "text" in delta:
                        if api_index not in block_indices:
                            output.content.append(TextContent(text=""))
                            ci = len(output.content) - 1
                            block_indices[api_index] = ci
                            event_stream.push(TextStartEvent(content_index=ci, partial=output))
                        ci = block_indices[api_index]
                        block = output.content[ci]
                        if isinstance(block, TextContent):
                            block.text += delta["text"]
                            event_stream.push(TextDeltaEvent(content_index=ci, delta=delta["text"], partial=output))

                    elif "toolUse" in delta:
                        if api_index in block_indices:
                            ci = block_indices[api_index]
                            block = output.content[ci]
                            if isinstance(block, ToolCall):
                                input_text = delta["toolUse"].get("input", "")
                                partial_json[ci] = partial_json.get(ci, "") + input_text
                                block.arguments = parse_streaming_json(partial_json[ci])
                                event_stream.push(
                                    ToolCallDeltaEvent(content_index=ci, delta=input_text, partial=output)
                                )

                    elif "reasoningContent" in delta:
                        rc = delta["reasoningContent"]
                        if api_index not in block_indices:
                            output.content.append(ThinkingContent(thinking="", thinking_signature=""))
                            ci = len(output.content) - 1
                            block_indices[api_index] = ci
                            event_stream.push(ThinkingStartEvent(content_index=ci, partial=output))
                        ci = block_indices[api_index]
                        block = output.content[ci]
                        if isinstance(block, ThinkingContent):
                            text = rc.get("text", "")
                            if text:
                                block.thinking += text
                                event_stream.push(ThinkingDeltaEvent(content_index=ci, delta=text, partial=output))
                            sig = rc.get("signature", "")
                            if sig:
                                block.thinking_signature = (block.thinking_signature or "") + sig

                elif "contentBlockStop" in item_wrapper:
                    event_data = item_wrapper["contentBlockStop"]
                    api_index = event_data.get("contentBlockIndex", 0)
                    if api_index in block_indices:
                        ci = block_indices[api_index]
                        block = output.content[ci]
                        if isinstance(block, TextContent):
                            event_stream.push(TextEndEvent(content_index=ci, content=block.text, partial=output))
                        elif isinstance(block, ThinkingContent):
                            event_stream.push(
                                ThinkingEndEvent(content_index=ci, content=block.thinking, partial=output)
                            )
                        elif isinstance(block, ToolCall):
                            if ci in partial_json:
                                block.arguments = parse_streaming_json(partial_json[ci])
                            event_stream.push(ToolCallEndEvent(content_index=ci, tool_call=block, partial=output))

                elif "messageStop" in item_wrapper:
                    stop = item_wrapper["messageStop"].get("stopReason", "")
                    output.stop_reason = _map_stop_reason(stop)

                elif "metadata" in item_wrapper:
                    meta = item_wrapper["metadata"]
                    usage_data = meta.get("usage", {})
                    if usage_data:
                        output.usage.input = usage_data.get("inputTokens", 0)
                        output.usage.output = usage_data.get("outputTokens", 0)
                        output.usage.cache_read = usage_data.get("cacheReadInputTokens", 0)
                        output.usage.cache_write = usage_data.get("cacheWriteInputTokens", 0)
                        output.usage.total_tokens = usage_data.get(
                            "totalTokens", output.usage.input + output.usage.output
                        )
                        calculate_cost(model, output.usage)

                elif "internalServerException" in item_wrapper:
                    raise RuntimeError(f"Internal server error: {item_wrapper['internalServerException'].get('message', '')}")
                elif "modelStreamErrorException" in item_wrapper:
                    raise RuntimeError(f"Model stream error: {item_wrapper['modelStreamErrorException'].get('message', '')}")
                elif "validationException" in item_wrapper:
                    raise RuntimeError(f"Validation error: {item_wrapper['validationException'].get('message', '')}")
                elif "throttlingException" in item_wrapper:
                    raise RuntimeError(f"Throttling error: {item_wrapper['throttlingException'].get('message', '')}")
                elif "serviceUnavailableException" in item_wrapper:
                    raise RuntimeError(f"Service unavailable: {item_wrapper['serviceUnavailableException'].get('message', '')}")

            if output.stop_reason in ("error", "aborted"):
                raise RuntimeError("An unknown error occurred")

            event_stream.push(DoneEvent(reason=output.stop_reason, message=output))
            event_stream.end()

        except Exception as e:
            output.stop_reason = "aborted" if "aborted" in str(e).lower() or "cancelled" in str(e).lower() else "error"
            output.error_message = str(e)
            event_stream.push(ErrorEvent(reason=output.stop_reason, error=output))
            event_stream.end()

    event_stream._background_task = asyncio.ensure_future(_run())
    return event_stream


def stream_simple_bedrock(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    """Stream using the simple API with thinking level support."""
    base = build_base_options(model, options)

    if not options or not options.reasoning:
        return stream_bedrock(
            model,
            context,
            BedrockOptions(
                temperature=base.temperature,
                max_tokens=base.max_tokens,
                cache_retention=base.cache_retention,
                session_id=base.session_id,
                headers=base.headers,
            ),
        )

    is_claude = "anthropic.claude" in model.id or "anthropic/claude" in model.id
    if is_claude:
        if _supports_adaptive_thinking(model.id):
            return stream_bedrock(
                model,
                context,
                BedrockOptions(
                    temperature=base.temperature,
                    max_tokens=base.max_tokens,
                    cache_retention=base.cache_retention,
                    session_id=base.session_id,
                    headers=base.headers,
                    reasoning=options.reasoning,
                    thinking_budgets=_budgets_to_dict(options.thinking_budgets),
                ),
            )

        custom_budgets = _budgets_to_dict(options.thinking_budgets)
        adjusted_max, thinking_budget = adjust_max_tokens_for_thinking(
            base.max_tokens or 0,
            options.reasoning,
            custom_budgets,
        )

        level = clamp_reasoning(options.reasoning)
        return stream_bedrock(
            model,
            context,
            BedrockOptions(
                temperature=base.temperature,
                max_tokens=adjusted_max,
                cache_retention=base.cache_retention,
                session_id=base.session_id,
                headers=base.headers,
                reasoning=options.reasoning,
                thinking_budgets={**(custom_budgets or {}), level: thinking_budget},
            ),
        )

    return stream_bedrock(
        model,
        context,
        BedrockOptions(
            temperature=base.temperature,
            max_tokens=base.max_tokens,
            cache_retention=base.cache_retention,
            session_id=base.session_id,
            headers=base.headers,
            reasoning=options.reasoning,
            thinking_budgets=_budgets_to_dict(options.thinking_budgets),
        ),
    )


def _budgets_to_dict(budgets: Any) -> dict[str, int] | None:
    if not budgets:
        return None
    result = {}
    for k in ("minimal", "low", "medium", "high"):
        v = getattr(budgets, k, None)
        if v is not None:
            result[k] = v
    return result or None


def _supports_adaptive_thinking(model_id: str) -> bool:
    return "opus-4-6" in model_id or "opus-4.6" in model_id


def _supports_prompt_caching(model: Model) -> bool:
    if model.cost.cache_read or model.cost.cache_write:
        return True
    mid = model.id.lower()
    if "claude" in mid and ("-4-" in mid or "-4." in mid):
        return True
    if "claude-3-7-sonnet" in mid:
        return True
    if "claude-3-5-haiku" in mid:
        return True
    return False


def _supports_thinking_signature(model: Model) -> bool:
    mid = model.id.lower()
    return "anthropic.claude" in mid or "anthropic/claude" in mid


def _resolve_cache_retention(retention: CacheRetention | None = None) -> CacheRetention:
    if retention:
        return retention
    if os.environ.get("PI_CACHE_RETENTION") == "long":
        return "long"
    return "short"


def _normalize_tool_call_id(id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", id)[:64]


def _map_stop_reason(reason: str) -> StopReason:
    mapping: dict[str, StopReason] = {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
        "model_context_window_exceeded": "length",
        "tool_use": "tool_use",
    }
    return mapping.get(reason, "error")


def _create_image_block(mime_type: str, data_b64: str) -> dict[str, Any]:
    format_map = {
        "image/jpeg": "jpeg",
        "image/jpg": "jpeg",
        "image/png": "png",
        "image/gif": "gif",
        "image/webp": "webp",
    }
    fmt = format_map.get(mime_type)
    if not fmt:
        raise ValueError(f"Unknown image type: {mime_type}")
    return {"source": {"bytes": base64.b64decode(data_b64)}, "format": fmt}


def _build_command_input(
    model: Model,
    context: Context,
    options: BedrockOptions,
    cache_retention: CacheRetention,
) -> dict[str, Any]:
    transformed = transform_messages(context.messages, current_model=model.id, normalize_tool_id=_normalize_tool_call_id)
    messages = _convert_messages(transformed, model, cache_retention)

    command: dict[str, Any] = {
        "modelId": model.id,
        "messages": messages,
        "inferenceConfig": {},
    }

    if options.max_tokens:
        command["inferenceConfig"]["maxTokens"] = options.max_tokens
    if options.temperature is not None:
        command["inferenceConfig"]["temperature"] = options.temperature

    # System prompt
    if context.system_prompt:
        system_blocks: list[dict[str, Any]] = [{"text": _sanitize(context.system_prompt)}]
        if cache_retention != "none" and _supports_prompt_caching(model):
            cache_point: dict[str, Any] = {"type": "default"}
            if cache_retention == "long":
                cache_point["ttl"] = "ONE_HOUR"
            system_blocks.append({"cachePoint": cache_point})
        command["system"] = system_blocks

    # Tools
    if context.tools:
        tool_config = _convert_tool_config(context.tools, options.tool_choice)
        if tool_config:
            command["toolConfig"] = tool_config

    # Thinking / reasoning
    additional = _build_additional_model_fields(model, options)
    if additional:
        command["additionalModelRequestFields"] = additional

    return command


def _convert_messages(
    transformed: list[Any],
    model: Model,
    cache_retention: CacheRetention,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    i = 0
    while i < len(transformed):
        msg = transformed[i]

        if msg.role == "user":
            if isinstance(msg.content, str):
                result.append({"role": "user", "content": [{"text": _sanitize(msg.content)}]})
            else:
                content_blocks = []
                for c in msg.content:
                    if isinstance(c, TextContent):
                        content_blocks.append({"text": _sanitize(c.text)})
                    elif isinstance(c, ImageContent):
                        content_blocks.append({"image": _create_image_block(c.mime_type, c.data)})
                result.append({"role": "user", "content": content_blocks})

        elif msg.role == "assistant":
            if not msg.content:
                i += 1
                continue
            content_blocks = []
            for c in msg.content:
                if isinstance(c, TextContent):
                    if not c.text.strip():
                        continue
                    content_blocks.append({"text": _sanitize(c.text)})
                elif isinstance(c, ToolCall):
                    content_blocks.append({"toolUse": {"toolUseId": c.id, "name": c.name, "input": c.arguments}})
                elif isinstance(c, ThinkingContent):
                    if not c.thinking.strip():
                        continue
                    reasoning_text: dict[str, Any] = {"text": _sanitize(c.thinking)}
                    if _supports_thinking_signature(model) and c.thinking_signature:
                        reasoning_text["signature"] = c.thinking_signature
                    content_blocks.append({"reasoningContent": {"reasoningText": reasoning_text}})
            if not content_blocks:
                i += 1
                continue
            result.append({"role": "assistant", "content": content_blocks})

        elif msg.role == "tool_result":
            tool_results: list[dict[str, Any]] = []
            j = i
            while j < len(transformed) and transformed[j].role == "tool_result":
                tr = transformed[j]
                content_items = []
                for c in tr.content:
                    if isinstance(c, ImageContent):
                        content_items.append({"image": _create_image_block(c.mime_type, c.data)})
                    else:
                        content_items.append({"text": _sanitize(c.text)})
                tool_results.append(
                    {
                        "toolResult": {
                            "toolUseId": tr.tool_call_id,
                            "content": content_items,
                            "status": "error" if tr.is_error else "success",
                        }
                    }
                )
                j += 1
            i = j - 1
            result.append({"role": "user", "content": tool_results})

        i += 1

    # Add cache point to last user message
    if cache_retention != "none" and _supports_prompt_caching(model) and result:
        last = result[-1]
        if last.get("role") == "user" and last.get("content"):
            cache_point: dict[str, Any] = {"type": "default"}
            if cache_retention == "long":
                cache_point["ttl"] = "ONE_HOUR"
            last["content"].append({"cachePoint": cache_point})

    return result


def _convert_tool_config(tools: list[Any], tool_choice: str | dict[str, Any] | None) -> dict[str, Any] | None:
    if not tools or tool_choice == "none":
        return None

    bedrock_tools = [
        {"toolSpec": {"name": t.name, "description": t.description, "inputSchema": {"json": t.parameters}}}
        for t in tools
    ]

    tc: dict[str, Any] | None = None
    if tool_choice == "auto":
        tc = {"auto": {}}
    elif tool_choice == "any":
        tc = {"any": {}}
    elif isinstance(tool_choice, dict) and tool_choice.get("type") == "tool":
        tc = {"tool": {"name": tool_choice["name"]}}

    result: dict[str, Any] = {"tools": bedrock_tools}
    if tc:
        result["toolChoice"] = tc
    return result


def _build_additional_model_fields(model: Model, options: BedrockOptions) -> dict[str, Any] | None:
    if not options.reasoning or not model.reasoning:
        return None

    if "anthropic.claude" in model.id:
        if _supports_adaptive_thinking(model.id):
            result: dict[str, Any] = {
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": _map_thinking_level_to_effort(options.reasoning)},
            }
        else:
            default_budgets: dict[str, int] = {
                "minimal": 1024,
                "low": 2048,
                "medium": 8192,
                "high": 16384,
                "xhigh": 16384,
            }
            level = "high" if options.reasoning == "xhigh" else options.reasoning
            budget = (options.thinking_budgets or {}).get(level, default_budgets.get(options.reasoning, 8192))
            result = {"thinking": {"type": "enabled", "budget_tokens": budget}}

        if options.interleaved_thinking and not _supports_adaptive_thinking(model.id):
            result["anthropic_beta"] = ["interleaved-thinking-2025-05-14"]

        return result

    return None


def _map_thinking_level_to_effort(level: ThinkingLevel | None) -> str:
    mapping = {"minimal": "low", "low": "low", "medium": "medium", "high": "high", "xhigh": "max"}
    return mapping.get(level or "high", "high")
