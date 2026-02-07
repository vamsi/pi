"""Shared utilities for OpenAI Responses API providers.

Used by openai_responses.py, azure_openai_responses.py, and openai_codex_responses.py.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pi.ai.models import calculate_cost
from pi.ai.providers.transform import transform_messages
from pi.ai.types import (
    AssistantMessage,
    Context,
    ImageContent,
    Model,
    StopReason,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingContent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    Usage,
)
from pi.ai.utils.json import parse_streaming_json

# Regex to strip surrogate pairs
_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def _sanitize(text: str) -> str:
    return _SURROGATE_RE.sub("\ufffd", text)


def _short_hash(s: str) -> str:
    """Fast deterministic hash to shorten long strings."""
    h1 = 0xDEADBEEF
    h2 = 0x41C6CE57
    for ch in s:
        c = ord(ch)
        h1 = ((h1 ^ c) * 2654435761) & 0xFFFFFFFF
        h2 = ((h2 ^ c) * 1597334677) & 0xFFFFFFFF
    h1 = (((h1 ^ (h1 >> 16)) * 2246822507) ^ ((h2 ^ (h2 >> 13)) * 3266489909)) & 0xFFFFFFFF
    h2 = (((h2 ^ (h2 >> 16)) * 2246822507) ^ ((h1 ^ (h1 >> 13)) * 3266489909)) & 0xFFFFFFFF

    import string

    digits = string.digits + string.ascii_lowercase

    def to_base36(n: int) -> str:
        if n == 0:
            return "0"
        result = []
        while n:
            result.append(digits[n % 36])
            n //= 36
        return "".join(reversed(result))

    return to_base36(h2) + to_base36(h1)


# =============================================================================
# Message conversion
# =============================================================================


def convert_responses_messages(
    model: Model,
    context: Context,
    allowed_tool_call_providers: set[str],
    *,
    include_system_prompt: bool = True,
) -> list[dict[str, Any]]:
    """Convert internal messages to OpenAI Responses API input format."""
    messages: list[dict[str, Any]] = []

    def normalize_tool_call_id(id: str) -> str:
        if model.provider not in allowed_tool_call_providers:
            return id
        if "|" not in id:
            return id
        call_id, item_id = id.split("|", 1)
        sanitized_call_id = re.sub(r"[^a-zA-Z0-9_-]", "_", call_id)
        sanitized_item_id = re.sub(r"[^a-zA-Z0-9_-]", "_", item_id)
        if not sanitized_item_id.startswith("fc"):
            sanitized_item_id = f"fc_{sanitized_item_id}"
        normalized_call_id = sanitized_call_id[:64].rstrip("_")
        normalized_item_id = sanitized_item_id[:64].rstrip("_")
        return f"{normalized_call_id}|{normalized_item_id}"

    transformed = transform_messages(context.messages, current_model=model.id, normalize_tool_id=normalize_tool_call_id)

    if include_system_prompt and context.system_prompt:
        role = "developer" if model.reasoning else "system"
        messages.append({"role": role, "content": _sanitize(context.system_prompt)})

    msg_index = 0
    for msg in transformed:
        if msg.role == "user":
            if isinstance(msg.content, str):
                messages.append(
                    {"role": "user", "content": [{"type": "input_text", "text": _sanitize(msg.content)}]}
                )
            else:
                content: list[dict[str, Any]] = []
                for item in msg.content:
                    if isinstance(item, TextContent):
                        content.append({"type": "input_text", "text": _sanitize(item.text)})
                    elif isinstance(item, ImageContent):
                        content.append(
                            {
                                "type": "input_image",
                                "detail": "auto",
                                "image_url": f"data:{item.mime_type};base64,{item.data}",
                            }
                        )
                filtered = [c for c in content if c["type"] != "input_image"] if "image" not in model.input else content
                if not filtered:
                    continue
                messages.append({"role": "user", "content": filtered})

        elif msg.role == "assistant":
            output_items: list[dict[str, Any]] = []
            is_different_model = msg.model != model.id and msg.provider == model.provider and msg.api == model.api

            for block in msg.content:
                if isinstance(block, ThinkingContent):
                    if block.thinking_signature:
                        try:
                            reasoning_item = json.loads(block.thinking_signature)
                            output_items.append(reasoning_item)
                        except json.JSONDecodeError:
                            pass
                elif isinstance(block, TextContent):
                    msg_id = block.text_signature
                    if not msg_id:
                        msg_id = f"msg_{msg_index}"
                    elif len(msg_id) > 64:
                        msg_id = f"msg_{_short_hash(msg_id)}"
                    output_items.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": _sanitize(block.text), "annotations": []}],
                            "status": "completed",
                            "id": msg_id,
                        }
                    )
                elif isinstance(block, ToolCall):
                    parts = block.id.split("|", 1)
                    call_id = parts[0]
                    item_id = parts[1] if len(parts) > 1 else None

                    if is_different_model and item_id and item_id.startswith("fc_"):
                        item_id = None

                    output_items.append(
                        {
                            "type": "function_call",
                            "id": item_id,
                            "call_id": call_id,
                            "name": block.name,
                            "arguments": json.dumps(block.arguments),
                        }
                    )

            if not output_items:
                continue
            messages.extend(output_items)

        elif msg.role == "tool_result":
            text_parts = [c.text for c in msg.content if isinstance(c, TextContent)]
            text_result = "\n".join(text_parts)
            has_images = any(isinstance(c, ImageContent) for c in msg.content)

            has_text = len(text_result) > 0
            call_id = msg.tool_call_id.split("|", 1)[0]
            messages.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": _sanitize(text_result if has_text else "(see attached image)"),
                }
            )

            if has_images and "image" in model.input:
                content_parts: list[dict[str, Any]] = [
                    {"type": "input_text", "text": "Attached image(s) from tool result:"}
                ]
                for block in msg.content:
                    if isinstance(block, ImageContent):
                        content_parts.append(
                            {
                                "type": "input_image",
                                "detail": "auto",
                                "image_url": f"data:{block.mime_type};base64,{block.data}",
                            }
                        )
                messages.append({"role": "user", "content": content_parts})

        msg_index += 1

    return messages


# =============================================================================
# Tool conversion
# =============================================================================


def convert_responses_tools(tools: list[Any], *, strict: bool | None = False) -> list[dict[str, Any]]:
    """Convert tools to OpenAI Responses function format."""
    return [
        {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
            "strict": strict,
        }
        for tool in tools
    ]


# =============================================================================
# Stream processing
# =============================================================================


async def process_responses_stream(
    openai_stream: Any,
    output: AssistantMessage,
    stream: Any,
    model: Model,
    *,
    service_tier: str | None = None,
    apply_service_tier_pricing: Any | None = None,
) -> None:
    """Process an OpenAI Responses API stream, pushing events to the event stream."""
    current_item: dict[str, Any] | None = None
    current_block: dict[str, Any] | None = None
    blocks = output.content

    def bi() -> int:
        return len(blocks) - 1

    async for event in openai_stream:
        event_type = getattr(event, "type", None)
        if not event_type:
            continue

        if event_type == "response.output_item.added":
            item = event.item
            item_type = getattr(item, "type", None)
            if item_type == "reasoning":
                current_item = {"type": "reasoning", "summary": []}
                current_block = {"type": "thinking", "thinking": ""}
                output.content.append(ThinkingContent(thinking=""))
                stream.push(ThinkingStartEvent(content_index=bi(), partial=output))
            elif item_type == "message":
                current_item = {"type": "message", "content": [], "id": getattr(item, "id", None)}
                current_block = {"type": "text", "text": ""}
                output.content.append(TextContent(text=""))
                stream.push(TextStartEvent(content_index=bi(), partial=output))
            elif item_type == "function_call":
                call_id = getattr(item, "call_id", "")
                item_id = getattr(item, "id", "")
                name = getattr(item, "name", "")
                args = getattr(item, "arguments", "") or ""
                current_item = {"type": "function_call", "call_id": call_id, "id": item_id, "name": name}
                current_block = {
                    "type": "toolCall",
                    "id": f"{call_id}|{item_id}",
                    "name": name,
                    "arguments": {},
                    "partial_json": args,
                }
                output.content.append(ToolCall(id=f"{call_id}|{item_id}", name=name, arguments={}))
                stream.push(ToolCallStartEvent(content_index=bi(), partial=output))

        elif event_type == "response.reasoning_summary_part.added":
            if current_item and current_item.get("type") == "reasoning":
                part = getattr(event, "part", None)
                if part:
                    current_item["summary"].append({"text": getattr(part, "text", "")})

        elif event_type == "response.reasoning_summary_text.delta":
            if (
                current_item
                and current_item.get("type") == "reasoning"
                and current_block
                and current_block.get("type") == "thinking"
            ):
                summary = current_item.get("summary", [])
                delta = getattr(event, "delta", "")
                if summary:
                    last_part = summary[-1]
                    idx = bi()
                    block = output.content[idx]
                    if isinstance(block, ThinkingContent):
                        block.thinking += delta
                        last_part["text"] += delta
                        current_block["thinking"] += delta
                        stream.push(ThinkingDeltaEvent(content_index=idx, delta=delta, partial=output))

        elif event_type == "response.reasoning_summary_part.done":
            if (
                current_item
                and current_item.get("type") == "reasoning"
                and current_block
                and current_block.get("type") == "thinking"
            ):
                summary = current_item.get("summary", [])
                if summary:
                    idx = bi()
                    block = output.content[idx]
                    if isinstance(block, ThinkingContent):
                        block.thinking += "\n\n"
                        summary[-1]["text"] += "\n\n"
                        current_block["thinking"] += "\n\n"
                        stream.push(ThinkingDeltaEvent(content_index=idx, delta="\n\n", partial=output))

        elif event_type == "response.content_part.added":
            if current_item and current_item.get("type") == "message":
                part = getattr(event, "part", None)
                if part:
                    part_type = getattr(part, "type", None)
                    if part_type in ("output_text", "refusal"):
                        current_item["content"].append(
                            {"type": part_type, "text": getattr(part, "text", ""), "refusal": getattr(part, "refusal", "")}
                        )

        elif event_type == "response.output_text.delta":
            if (
                current_item
                and current_item.get("type") == "message"
                and current_block
                and current_block.get("type") == "text"
            ):
                content_list = current_item.get("content", [])
                if not content_list:
                    continue
                last_part = content_list[-1]
                if last_part.get("type") == "output_text":
                    delta = getattr(event, "delta", "")
                    idx = bi()
                    block = output.content[idx]
                    if isinstance(block, TextContent):
                        block.text += delta
                        current_block["text"] += delta
                        last_part["text"] += delta
                        stream.push(TextDeltaEvent(content_index=idx, delta=delta, partial=output))

        elif event_type == "response.refusal.delta":
            if (
                current_item
                and current_item.get("type") == "message"
                and current_block
                and current_block.get("type") == "text"
            ):
                content_list = current_item.get("content", [])
                if not content_list:
                    continue
                last_part = content_list[-1]
                if last_part.get("type") == "refusal":
                    delta = getattr(event, "delta", "")
                    idx = bi()
                    block = output.content[idx]
                    if isinstance(block, TextContent):
                        block.text += delta
                        current_block["text"] += delta
                        last_part["refusal"] += delta
                        stream.push(TextDeltaEvent(content_index=idx, delta=delta, partial=output))

        elif event_type == "response.function_call_arguments.delta":
            if (
                current_item
                and current_item.get("type") == "function_call"
                and current_block
                and current_block.get("type") == "toolCall"
            ):
                delta = getattr(event, "delta", "")
                current_block["partial_json"] += delta
                parsed = parse_streaming_json(current_block["partial_json"])
                idx = bi()
                block = output.content[idx]
                if isinstance(block, ToolCall):
                    block.arguments = parsed
                    current_block["arguments"] = parsed
                    stream.push(ToolCallDeltaEvent(content_index=idx, delta=delta, partial=output))

        elif event_type == "response.function_call_arguments.done":
            if (
                current_item
                and current_item.get("type") == "function_call"
                and current_block
                and current_block.get("type") == "toolCall"
            ):
                args_str = getattr(event, "arguments", "")
                current_block["partial_json"] = args_str
                parsed = parse_streaming_json(args_str)
                idx = bi()
                block = output.content[idx]
                if isinstance(block, ToolCall):
                    block.arguments = parsed
                    current_block["arguments"] = parsed

        elif event_type == "response.output_item.done":
            item = event.item
            item_type = getattr(item, "type", None)

            if item_type == "reasoning" and current_block and current_block.get("type") == "thinking":
                summary_parts = getattr(item, "summary", None) or []
                thinking_text = "\n\n".join(getattr(s, "text", "") for s in summary_parts)
                idx = bi()
                block = output.content[idx]
                if isinstance(block, ThinkingContent):
                    block.thinking = thinking_text
                    try:
                        block.thinking_signature = json.dumps(
                            {
                                "type": "reasoning",
                                "id": getattr(item, "id", ""),
                                "summary": [
                                    {"type": getattr(s, "type", "summary_text"), "text": getattr(s, "text", "")}
                                    for s in summary_parts
                                ],
                            }
                        )
                    except Exception:
                        pass
                    stream.push(ThinkingEndEvent(content_index=idx, content=thinking_text, partial=output))
                current_block = None

            elif item_type == "message" and current_block and current_block.get("type") == "text":
                content_list = getattr(item, "content", [])
                text = "".join(
                    getattr(c, "text", "") if getattr(c, "type", "") == "output_text" else getattr(c, "refusal", "")
                    for c in content_list
                )
                idx = bi()
                block = output.content[idx]
                if isinstance(block, TextContent):
                    block.text = text
                    block.text_signature = getattr(item, "id", None)
                    stream.push(TextEndEvent(content_index=idx, content=text, partial=output))
                current_block = None

            elif item_type == "function_call":
                if current_block and current_block.get("type") == "toolCall" and current_block.get("partial_json"):
                    try:
                        args = json.loads(current_block["partial_json"])
                    except json.JSONDecodeError:
                        try:
                            args = json.loads(getattr(item, "arguments", "{}"))
                        except json.JSONDecodeError:
                            args = {}
                else:
                    try:
                        args = json.loads(getattr(item, "arguments", "{}"))
                    except json.JSONDecodeError:
                        args = {}

                call_id = getattr(item, "call_id", "")
                item_id = getattr(item, "id", "")
                name = getattr(item, "name", "")
                tc = ToolCall(id=f"{call_id}|{item_id}", name=name, arguments=args)

                idx = bi()
                block = output.content[idx]
                if isinstance(block, ToolCall):
                    block.arguments = args
                    block.id = tc.id
                    block.name = tc.name

                current_block = None
                stream.push(ToolCallEndEvent(content_index=idx, tool_call=tc, partial=output))

        elif event_type == "response.completed":
            response = getattr(event, "response", None)
            if response:
                usage_data = getattr(response, "usage", None)
                if usage_data:
                    input_tokens = getattr(usage_data, "input_tokens", 0) or 0
                    output_tokens = getattr(usage_data, "output_tokens", 0) or 0
                    total_tokens = getattr(usage_data, "total_tokens", 0) or 0
                    details = getattr(usage_data, "input_tokens_details", None)
                    cached_tokens = getattr(details, "cached_tokens", 0) if details else 0
                    output.usage = Usage(
                        input=input_tokens - cached_tokens,
                        output=output_tokens,
                        cache_read=cached_tokens,
                        cache_write=0,
                        total_tokens=total_tokens,
                    )
                calculate_cost(model, output.usage)

                if apply_service_tier_pricing:
                    tier = getattr(response, "service_tier", None) or service_tier
                    apply_service_tier_pricing(output.usage, tier)

                status = getattr(response, "status", None)
                output.stop_reason = map_stop_reason(status)
                if any(isinstance(b, ToolCall) for b in output.content) and output.stop_reason == "stop":
                    output.stop_reason = "tool_use"

        elif event_type == "error":
            code = getattr(event, "code", "")
            message = getattr(event, "message", "")
            raise RuntimeError(f"Error Code {code}: {message}" if message else "Unknown error")

        elif event_type == "response.failed":
            raise RuntimeError("Unknown error")


def map_stop_reason(status: str | None) -> StopReason:
    """Map OpenAI Responses API status to StopReason."""
    if not status:
        return "stop"
    mapping: dict[str, StopReason] = {
        "completed": "stop",
        "incomplete": "length",
        "failed": "error",
        "cancelled": "error",
        "in_progress": "stop",
        "queued": "stop",
    }
    return mapping.get(status, "stop")
