"""OpenAI Chat Completions API provider implementation.

Supports OpenAI and 15+ compatible providers (OpenRouter, Groq, Mistral, Cerebras,
xAI, GitHub Copilot, etc.) via compatibility detection.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from pi.ai.env import get_env_api_key
from pi.ai.events import AssistantMessageEventStream
from pi.ai.models import calculate_cost, supports_xhigh
from pi.ai.providers.options import build_base_options, clamp_reasoning
from pi.ai.providers.transform import transform_messages
from pi.ai.types import (
    AssistantMessage,
    Context,
    DoneEvent,
    ErrorEvent,
    ImageContent,
    Message,
    Model,
    OpenAICompletionsCompat,
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
    ToolResultMessage,
    Usage,
)
from pi.ai.utils.json import parse_streaming_json

_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def _sanitize(text: str) -> str:
    return _SURROGATE_RE.sub("\ufffd", text)


def _normalize_mistral_tool_id(id: str) -> str:
    """Normalize tool call ID for Mistral (exactly 9 alphanumeric chars)."""
    normalized = re.sub(r"[^a-zA-Z0-9]", "", id)
    if len(normalized) < 9:
        padding = "ABCDEFGHI"
        normalized = normalized + padding[: 9 - len(normalized)]
    elif len(normalized) > 9:
        normalized = normalized[:9]
    return normalized


def _has_tool_history(messages: list[Message]) -> bool:
    for msg in messages:
        if msg.role == "tool_result":
            return True
        if msg.role == "assistant" and any(isinstance(b, ToolCall) for b in msg.content):
            return True
    return False


@dataclass
class OpenAICompletionsOptions:
    temperature: float | None = None
    max_tokens: int | None = None
    api_key: str | None = None
    cache_retention: str = "short"
    session_id: str | None = None
    headers: dict[str, str] | None = None
    max_retry_delay_ms: int | None = None
    on_payload: Any = None
    tool_choice: str | dict[str, Any] | None = None
    reasoning_effort: str | None = None


def stream_openai_completions(
    model: Model,
    context: Context,
    options: OpenAICompletionsOptions | None = None,
) -> AssistantMessageEventStream:
    """Stream a response from the OpenAI Chat Completions API."""
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
            import openai

            api_key = (options and options.api_key) or get_env_api_key(model.provider) or ""
            if not api_key:
                raise ValueError(
                    "OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass it as an argument."
                )

            client = _create_client(model, context, api_key, options.headers if options else None)
            params = _build_params(model, context, options)
            if options and options.on_payload:
                options.on_payload(params)

            openai_stream = await client.chat.completions.create(**params)
            event_stream.push(StartEvent(partial=output))

            current_block: dict[str, Any] | None = None
            blocks = output.content

            def block_index() -> int:
                return len(blocks) - 1

            def finish_current(block: dict[str, Any] | None) -> None:
                if not block:
                    return
                idx = block_index()
                if block["type"] == "text":
                    event_stream.push(TextEndEvent(content_index=idx, content=block["text"], partial=output))
                elif block["type"] == "thinking":
                    event_stream.push(ThinkingEndEvent(content_index=idx, content=block["thinking"], partial=output))
                elif block["type"] == "toolCall":
                    tc_block = output.content[idx]
                    if isinstance(tc_block, ToolCall):
                        try:
                            tc_block.arguments = json.loads(block.get("partial_args", "{}"))
                        except json.JSONDecodeError:
                            tc_block.arguments = {}
                    event_stream.push(ToolCallEndEvent(content_index=idx, tool_call=tc_block, partial=output))

            async for chunk in openai_stream:
                usage_data = getattr(chunk, "usage", None)
                if usage_data:
                    cached = getattr(getattr(usage_data, "prompt_tokens_details", None), "cached_tokens", 0) or 0
                    reasoning_tokens = (
                        getattr(getattr(usage_data, "completion_tokens_details", None), "reasoning_tokens", 0) or 0
                    )
                    input_tokens = (getattr(usage_data, "prompt_tokens", 0) or 0) - cached
                    output_tokens = (getattr(usage_data, "completion_tokens", 0) or 0) + reasoning_tokens
                    output.usage = Usage(
                        input=input_tokens,
                        output=output_tokens,
                        cache_read=cached,
                        cache_write=0,
                        total_tokens=input_tokens + output_tokens + cached,
                    )
                    calculate_cost(model, output.usage)

                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                choice = choices[0]

                finish_reason = getattr(choice, "finish_reason", None)
                if finish_reason:
                    output.stop_reason = _map_stop_reason(finish_reason)

                delta = getattr(choice, "delta", None)
                if not delta:
                    continue

                # Text content
                content_text = getattr(delta, "content", None)
                if content_text is not None and len(content_text) > 0:
                    if not current_block or current_block["type"] != "text":
                        finish_current(current_block)
                        current_block = {"type": "text", "text": ""}
                        output.content.append(TextContent(text=""))
                        event_stream.push(TextStartEvent(content_index=block_index(), partial=output))

                    idx = block_index()
                    block = output.content[idx]
                    if isinstance(block, TextContent):
                        block.text += content_text
                        current_block["text"] += content_text
                        event_stream.push(TextDeltaEvent(content_index=idx, delta=content_text, partial=output))

                # Reasoning content (multiple field names)
                reasoning_fields = ["reasoning_content", "reasoning", "reasoning_text"]
                found_field = None
                for rf in reasoning_fields:
                    val = getattr(delta, rf, None)
                    if val is not None and len(val) > 0:
                        found_field = rf
                        break

                if found_field:
                    reasoning_text = getattr(delta, found_field, "")
                    if not current_block or current_block["type"] != "thinking":
                        finish_current(current_block)
                        current_block = {"type": "thinking", "thinking": "", "field": found_field}
                        output.content.append(ThinkingContent(thinking="", thinking_signature=found_field))
                        event_stream.push(ThinkingStartEvent(content_index=block_index(), partial=output))

                    idx = block_index()
                    block = output.content[idx]
                    if isinstance(block, ThinkingContent):
                        block.thinking += reasoning_text
                        current_block["thinking"] += reasoning_text
                        event_stream.push(ThinkingDeltaEvent(content_index=idx, delta=reasoning_text, partial=output))

                # Tool calls
                tool_calls = getattr(delta, "tool_calls", None)
                if tool_calls:
                    for tc in tool_calls:
                        tc_id = getattr(tc, "id", None)
                        tc_fn = getattr(tc, "function", None)
                        tc_name = getattr(tc_fn, "name", "") if tc_fn else ""
                        tc_args = getattr(tc_fn, "arguments", "") if tc_fn else ""

                        if not current_block or current_block["type"] != "toolCall" or (tc_id and current_block.get("id") != tc_id):
                            finish_current(current_block)
                            current_block = {
                                "type": "toolCall",
                                "id": tc_id or "",
                                "name": tc_name,
                                "partial_args": "",
                            }
                            output.content.append(ToolCall(id=tc_id or "", name=tc_name, arguments={}))
                            event_stream.push(ToolCallStartEvent(content_index=block_index(), partial=output))

                        if tc_id:
                            current_block["id"] = tc_id
                            block = output.content[block_index()]
                            if isinstance(block, ToolCall):
                                block.id = tc_id
                        if tc_name:
                            current_block["name"] = tc_name
                            block = output.content[block_index()]
                            if isinstance(block, ToolCall):
                                block.name = tc_name

                        if tc_args:
                            current_block["partial_args"] = current_block.get("partial_args", "") + tc_args
                            parsed = parse_streaming_json(current_block["partial_args"])
                            block = output.content[block_index()]
                            if isinstance(block, ToolCall):
                                block.arguments = parsed
                            event_stream.push(
                                ToolCallDeltaEvent(content_index=block_index(), delta=tc_args, partial=output)
                            )

                # Reasoning details (encrypted signatures for tool calls)
                reasoning_details = getattr(delta, "reasoning_details", None)
                if reasoning_details and isinstance(reasoning_details, list):
                    for detail in reasoning_details:
                        if (
                            getattr(detail, "type", None) == "reasoning.encrypted"
                            and getattr(detail, "id", None)
                            and getattr(detail, "data", None)
                        ):
                            for b in output.content:
                                if isinstance(b, ToolCall) and b.id == detail.id:
                                    b.thought_signature = json.dumps(
                                        {"type": detail.type, "id": detail.id, "data": detail.data}
                                    )

            finish_current(current_block)

            if output.stop_reason in ("aborted", "error"):
                raise RuntimeError("An unknown error occurred")

            event_stream.push(DoneEvent(reason=output.stop_reason, message=output))
            event_stream.end()

        except Exception as e:
            output.stop_reason = "aborted" if "aborted" in str(e).lower() or "cancelled" in str(e).lower() else "error"
            output.error_message = str(e)
            # Check for raw metadata
            raw_metadata = getattr(getattr(e, "error", None), "metadata", None)
            if raw_metadata:
                raw = getattr(raw_metadata, "raw", None)
                if raw:
                    output.error_message += f"\n{raw}"
            event_stream.push(ErrorEvent(reason=output.stop_reason, error=output))
            event_stream.end()

    event_stream._background_task = asyncio.ensure_future(_run())
    return event_stream


def stream_simple_openai_completions(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    """Stream using the simple API with reasoning effort support."""
    api_key = (options and options.api_key) or get_env_api_key(model.provider)
    if not api_key:
        raise ValueError(f"No API key for provider: {model.provider}")

    base = build_base_options(model, options)
    reasoning_effort = options.reasoning if options and supports_xhigh(model) else (clamp_reasoning(options.reasoning) if options and options.reasoning else None)

    return stream_openai_completions(
        model,
        context,
        OpenAICompletionsOptions(
            temperature=base.temperature,
            max_tokens=base.max_tokens,
            api_key=api_key,
            cache_retention=base.cache_retention,
            session_id=base.session_id,
            headers=base.headers,
            reasoning_effort=reasoning_effort,
        ),
    )


def _create_client(
    model: Model,
    context: Context,
    api_key: str,
    options_headers: dict[str, str] | None = None,
) -> Any:
    import openai

    headers = dict(model.headers or {})

    if model.provider == "github-copilot":
        messages = context.messages or []
        last_message = messages[-1] if messages else None
        is_agent_call = last_message.role != "user" if last_message else False
        headers["X-Initiator"] = "agent" if is_agent_call else "user"
        headers["Openai-Intent"] = "conversation-edits"

        has_images = any(
            (msg.role == "user" and isinstance(msg.content, list) and any(isinstance(c, ImageContent) for c in msg.content))
            or (msg.role == "tool_result" and any(isinstance(c, ImageContent) for c in msg.content))
            for msg in messages
        )
        if has_images:
            headers["Copilot-Vision-Request"] = "true"

    if options_headers:
        headers.update(options_headers)

    return openai.AsyncOpenAI(
        api_key=api_key,
        base_url=model.base_url,
        default_headers=headers if headers else None,
    )


def _build_params(
    model: Model,
    context: Context,
    options: OpenAICompletionsOptions | None = None,
) -> dict[str, Any]:
    compat = _get_compat(model)
    messages = _convert_messages(model, context, compat)
    _maybe_add_openrouter_cache_control(model, messages)

    params: dict[str, Any] = {
        "model": model.id,
        "messages": messages,
        "stream": True,
    }

    if compat.get("supports_usage_in_streaming", True):
        params["stream_options"] = {"include_usage": True}

    if compat.get("supports_store"):
        params["store"] = False

    if options and options.max_tokens:
        if compat.get("max_tokens_field") == "max_tokens":
            params["max_tokens"] = options.max_tokens
        else:
            params["max_completion_tokens"] = options.max_tokens

    if options and options.temperature is not None:
        params["temperature"] = options.temperature

    if context.tools:
        params["tools"] = _convert_tools(context.tools, compat)
    elif _has_tool_history(context.messages):
        params["tools"] = []

    if options and options.tool_choice:
        params["tool_choice"] = options.tool_choice

    thinking_format = compat.get("thinking_format", "openai")
    if thinking_format == "zai" and model.reasoning:
        params["thinking"] = {"type": "enabled" if (options and options.reasoning_effort) else "disabled"}
    elif thinking_format == "qwen" and model.reasoning:
        params["enable_thinking"] = bool(options and options.reasoning_effort)
    elif options and options.reasoning_effort and model.reasoning and compat.get("supports_reasoning_effort"):
        params["reasoning_effort"] = options.reasoning_effort

    # OpenRouter provider routing
    if model.base_url and "openrouter.ai" in model.base_url and model.compat:
        routing = getattr(model.compat, "open_router_routing", None)
        if routing:
            params["provider"] = routing

    return params


def _maybe_add_openrouter_cache_control(model: Model, messages: list[dict[str, Any]]) -> None:
    if model.provider != "openrouter" or not model.id.startswith("anthropic/"):
        return

    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") not in ("user", "assistant"):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
            return
        if isinstance(content, list):
            for j in range(len(content) - 1, -1, -1):
                part = content[j]
                if isinstance(part, dict) and part.get("type") == "text":
                    part["cache_control"] = {"type": "ephemeral"}
                    return


def _convert_messages(
    model: Model,
    context: Context,
    compat: dict[str, Any],
) -> list[dict[str, Any]]:
    """Convert messages to OpenAI Chat Completions format."""
    params: list[dict[str, Any]] = []

    def normalize_tool_call_id(id: str) -> str:
        if compat.get("requires_mistral_tool_ids"):
            return _normalize_mistral_tool_id(id)
        if "|" in id:
            call_id = id.split("|", 1)[0]
            return re.sub(r"[^a-zA-Z0-9_-]", "_", call_id)[:40]
        if model.provider == "openai":
            return id[:40] if len(id) > 40 else id
        if model.provider == "github-copilot" and "claude" in model.id.lower():
            return re.sub(r"[^a-zA-Z0-9_-]", "_", id)[:64]
        return id

    transformed = transform_messages(
        context.messages, current_model=model.id, normalize_tool_id=normalize_tool_call_id
    )

    if context.system_prompt:
        use_developer = model.reasoning and compat.get("supports_developer_role")
        role = "developer" if use_developer else "system"
        params.append({"role": role, "content": _sanitize(context.system_prompt)})

    last_role: str | None = None
    i = 0
    while i < len(transformed):
        msg = transformed[i]

        if compat.get("requires_assistant_after_tool_result") and last_role == "tool_result" and msg.role == "user":
            params.append({"role": "assistant", "content": "I have processed the tool results."})

        if msg.role == "user":
            if isinstance(msg.content, str):
                params.append({"role": "user", "content": _sanitize(msg.content)})
            else:
                content: list[dict[str, Any]] = []
                for item in msg.content:
                    if isinstance(item, TextContent):
                        content.append({"type": "text", "text": _sanitize(item.text)})
                    elif isinstance(item, ImageContent):
                        content.append(
                            {"type": "image_url", "image_url": {"url": f"data:{item.mime_type};base64,{item.data}"}}
                        )
                filtered = [c for c in content if c["type"] != "image_url"] if "image" not in model.input else content
                if not filtered:
                    i += 1
                    continue
                params.append({"role": "user", "content": filtered})

        elif msg.role == "assistant":
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": "" if compat.get("requires_assistant_after_tool_result") else None,
            }

            text_blocks = [b for b in msg.content if isinstance(b, TextContent) and b.text and b.text.strip()]
            if text_blocks:
                if model.provider == "github-copilot":
                    assistant_msg["content"] = "".join(_sanitize(b.text) for b in text_blocks)
                else:
                    assistant_msg["content"] = [{"type": "text", "text": _sanitize(b.text)} for b in text_blocks]

            thinking_blocks = [
                b for b in msg.content if isinstance(b, ThinkingContent) and b.thinking and b.thinking.strip()
            ]
            if thinking_blocks:
                if compat.get("requires_thinking_as_text"):
                    thinking_text = "\n\n".join(b.thinking for b in thinking_blocks)
                    existing = assistant_msg.get("content")
                    if isinstance(existing, list):
                        existing.insert(0, {"type": "text", "text": thinking_text})
                    else:
                        assistant_msg["content"] = [{"type": "text", "text": thinking_text}]
                else:
                    sig = thinking_blocks[0].thinking_signature
                    if sig and len(sig) > 0:
                        assistant_msg[sig] = "\n".join(b.thinking for b in thinking_blocks)

            tool_calls = [b for b in msg.content if isinstance(b, ToolCall)]
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in tool_calls
                ]
                reasoning_details = []
                for tc in tool_calls:
                    if tc.thought_signature:
                        try:
                            reasoning_details.append(json.loads(tc.thought_signature))
                        except json.JSONDecodeError:
                            pass
                if reasoning_details:
                    assistant_msg["reasoning_details"] = reasoning_details

            content_val = assistant_msg.get("content")
            has_content = content_val is not None and (
                (isinstance(content_val, str) and len(content_val) > 0)
                or (isinstance(content_val, list) and len(content_val) > 0)
            )
            if not has_content and "tool_calls" not in assistant_msg:
                i += 1
                continue
            params.append(assistant_msg)

        elif msg.role == "tool_result":
            image_blocks: list[dict[str, Any]] = []
            j = i
            while j < len(transformed) and transformed[j].role == "tool_result":
                tool_msg = transformed[j]
                text_parts = [c.text for c in tool_msg.content if isinstance(c, TextContent)]
                text_result = "\n".join(text_parts)
                has_images = any(isinstance(c, ImageContent) for c in tool_msg.content)
                has_text = len(text_result) > 0

                tool_result_msg: dict[str, Any] = {
                    "role": "tool",
                    "content": _sanitize(text_result if has_text else "(see attached image)"),
                    "tool_call_id": tool_msg.tool_call_id,
                }
                if compat.get("requires_tool_result_name") and tool_msg.tool_name:
                    tool_result_msg["name"] = tool_msg.tool_name
                params.append(tool_result_msg)

                if has_images and "image" in model.input:
                    for block in tool_msg.content:
                        if isinstance(block, ImageContent):
                            image_blocks.append(
                                {"type": "image_url", "image_url": {"url": f"data:{block.mime_type};base64,{block.data}"}}
                            )
                j += 1
            i = j - 1

            if image_blocks:
                if compat.get("requires_assistant_after_tool_result"):
                    params.append({"role": "assistant", "content": "I have processed the tool results."})
                params.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Attached image(s) from tool result:"}, *image_blocks],
                    }
                )
                last_role = "user"
            else:
                last_role = "tool_result"
            i += 1
            continue

        last_role = msg.role
        i += 1

    return params


def _convert_tools(tools: list[Tool], compat: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for tool in tools:
        fn: dict[str, Any] = {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
        if compat.get("supports_strict_mode", True):
            fn["strict"] = False
        result.append({"type": "function", "function": fn})
    return result


def _map_stop_reason(reason: str | None) -> StopReason:
    if reason is None:
        return "stop"
    mapping: dict[str, StopReason] = {
        "stop": "stop",
        "length": "length",
        "function_call": "tool_use",
        "tool_calls": "tool_use",
        "content_filter": "error",
    }
    return mapping.get(reason, "stop")


def _detect_compat(model: Model) -> dict[str, Any]:
    """Detect compatibility settings from provider and baseUrl."""
    provider = model.provider
    base_url = model.base_url or ""

    is_zai = provider == "zai" or "api.z.ai" in base_url
    is_non_standard = (
        provider in ("cerebras", "xai", "mistral", "opencode")
        or any(s in base_url for s in ("cerebras.ai", "api.x.ai", "mistral.ai", "chutes.ai", "deepseek.com", "opencode.ai"))
        or is_zai
    )
    use_max_tokens = provider == "mistral" or "mistral.ai" in base_url or "chutes.ai" in base_url
    is_grok = provider == "xai" or "api.x.ai" in base_url
    is_mistral = provider == "mistral" or "mistral.ai" in base_url

    return {
        "supports_store": not is_non_standard,
        "supports_developer_role": not is_non_standard,
        "supports_reasoning_effort": not is_grok and not is_zai,
        "supports_usage_in_streaming": True,
        "max_tokens_field": "max_tokens" if use_max_tokens else "max_completion_tokens",
        "requires_tool_result_name": is_mistral,
        "requires_assistant_after_tool_result": False,
        "requires_thinking_as_text": is_mistral,
        "requires_mistral_tool_ids": is_mistral,
        "thinking_format": "zai" if is_zai else "openai",
        "supports_strict_mode": True,
    }


def _get_compat(model: Model) -> dict[str, Any]:
    """Get resolved compatibility settings for a model."""
    detected = _detect_compat(model)
    if not model.compat or not isinstance(model.compat, OpenAICompletionsCompat):
        return detected

    c = model.compat
    return {
        "supports_store": c.supports_store if c.supports_store is not None else detected["supports_store"],
        "supports_developer_role": c.supports_developer_role if c.supports_developer_role is not None else detected["supports_developer_role"],
        "supports_reasoning_effort": c.supports_reasoning_effort if c.supports_reasoning_effort is not None else detected["supports_reasoning_effort"],
        "supports_usage_in_streaming": c.supports_usage_in_streaming if c.supports_usage_in_streaming is not None else detected["supports_usage_in_streaming"],
        "max_tokens_field": c.max_tokens_field or detected["max_tokens_field"],
        "requires_tool_result_name": c.requires_tool_result_name if c.requires_tool_result_name is not None else detected["requires_tool_result_name"],
        "requires_assistant_after_tool_result": c.requires_assistant_after_tool_result if c.requires_assistant_after_tool_result is not None else detected["requires_assistant_after_tool_result"],
        "requires_thinking_as_text": c.requires_thinking_as_text if c.requires_thinking_as_text is not None else detected["requires_thinking_as_text"],
        "requires_mistral_tool_ids": c.requires_mistral_tool_ids if c.requires_mistral_tool_ids is not None else detected["requires_mistral_tool_ids"],
        "thinking_format": c.thinking_format or detected["thinking_format"],
        "supports_strict_mode": c.supports_strict_mode if c.supports_strict_mode is not None else detected["supports_strict_mode"],
    }
