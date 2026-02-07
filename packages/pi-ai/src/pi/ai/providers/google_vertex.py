"""Google Vertex AI (Gemini via ADC) provider implementation."""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Any

from pi.ai.events import AssistantMessageEventStream
from pi.ai.models import calculate_cost
from pi.ai.providers.google_shared import (
    _sanitize,
    convert_messages,
    convert_tools,
    is_thinking_part,
    map_stop_reason,
    map_tool_choice,
    retain_thought_signature,
)
from pi.ai.providers.options import build_base_options, clamp_reasoning
from pi.ai.types import (
    AssistantMessage,
    Context,
    DoneEvent,
    ErrorEvent,
    Model,
    SimpleStreamOptions,
    StartEvent,
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
    Usage,
)

# Thinking level type for Gemini 3 models
GoogleThinkingLevel = str  # "THINKING_LEVEL_UNSPECIFIED" | "MINIMAL" | "LOW" | "MEDIUM" | "HIGH"

API_VERSION = "v1"

# Counter for generating unique tool call IDs
_tool_call_counter = 0


@dataclass
class GoogleVertexOptions:
    temperature: float | None = None
    max_tokens: int | None = None
    project: str | None = None
    location: str | None = None
    cache_retention: str = "short"
    session_id: str | None = None
    headers: dict[str, str] | None = None
    max_retry_delay_ms: int | None = None
    on_payload: Any = None
    tool_choice: str | None = None
    thinking: dict[str, Any] | None = None  # {enabled, budgetTokens?, level?}


def stream_google_vertex(
    model: Model,
    context: Context,
    options: GoogleVertexOptions | None = None,
) -> AssistantMessageEventStream:
    """Stream a response from Google Vertex AI."""
    event_stream = AssistantMessageEventStream()

    async def _run() -> None:
        global _tool_call_counter

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
            project = _resolve_project(options)
            location = _resolve_location(options)
            client = _create_client(model, project, location, options.headers if options else None)
            params = _build_params(model, context, options)
            if options and options.on_payload:
                options.on_payload(params)

            google_stream = await client.models.generate_content_stream(**params)

            event_stream.push(StartEvent(partial=output))
            current_block: dict[str, Any] | None = None
            blocks = output.content

            def bi() -> int:
                return len(blocks) - 1

            async for chunk in google_stream:
                candidates = getattr(chunk, "candidates", None)
                candidate = candidates[0] if candidates else None

                if candidate:
                    content = getattr(candidate, "content", None)
                    parts = getattr(content, "parts", None) if content else None
                    if parts:
                        for part in parts:
                            part_dict = part if isinstance(part, dict) else _part_to_dict(part)
                            text = part_dict.get("text")

                            if text is not None:
                                is_thinking = is_thinking_part(part_dict)

                                if (
                                    not current_block
                                    or (is_thinking and current_block["type"] != "thinking")
                                    or (not is_thinking and current_block["type"] != "text")
                                ):
                                    if current_block:
                                        _finish_block(current_block, blocks, bi, output, event_stream)
                                    if is_thinking:
                                        current_block = {"type": "thinking", "thinking": ""}
                                        output.content.append(ThinkingContent(thinking=""))
                                        event_stream.push(ThinkingStartEvent(content_index=bi(), partial=output))
                                    else:
                                        current_block = {"type": "text", "text": ""}
                                        output.content.append(TextContent(text=""))
                                        event_stream.push(TextStartEvent(content_index=bi(), partial=output))

                                idx = bi()
                                block = output.content[idx]
                                if isinstance(block, ThinkingContent):
                                    block.thinking += text
                                    block.thinking_signature = retain_thought_signature(
                                        block.thinking_signature, part_dict.get("thoughtSignature")
                                    )
                                    current_block["thinking"] += text
                                    event_stream.push(ThinkingDeltaEvent(content_index=idx, delta=text, partial=output))
                                elif isinstance(block, TextContent):
                                    block.text += text
                                    block.text_signature = retain_thought_signature(
                                        block.text_signature, part_dict.get("thoughtSignature")
                                    )
                                    current_block["text"] += text
                                    event_stream.push(TextDeltaEvent(content_index=idx, delta=text, partial=output))

                            fc = part_dict.get("functionCall")
                            if fc:
                                if current_block:
                                    _finish_block(current_block, blocks, bi, output, event_stream)
                                    current_block = None

                                provided_id = fc.get("id")
                                needs_new = not provided_id or any(
                                    isinstance(b, ToolCall) and b.id == provided_id for b in output.content
                                )
                                _tool_call_counter += 1
                                tc_id = (
                                    f"{fc.get('name', '')}_{int(time.time() * 1000)}_{_tool_call_counter}"
                                    if needs_new
                                    else provided_id
                                )

                                tc = ToolCall(
                                    id=tc_id,
                                    name=fc.get("name", ""),
                                    arguments=fc.get("args", {}),
                                    thought_signature=part_dict.get("thoughtSignature"),
                                )
                                output.content.append(tc)
                                event_stream.push(ToolCallStartEvent(content_index=bi(), partial=output))
                                event_stream.push(
                                    ToolCallDeltaEvent(
                                        content_index=bi(), delta=json.dumps(tc.arguments), partial=output
                                    )
                                )
                                event_stream.push(ToolCallEndEvent(content_index=bi(), tool_call=tc, partial=output))

                    finish_reason = getattr(candidate, "finish_reason", None)
                    if finish_reason:
                        reason_str = finish_reason if isinstance(finish_reason, str) else str(finish_reason)
                        # Handle enum-style values like "FinishReason.STOP"
                        if "." in reason_str:
                            reason_str = reason_str.split(".")[-1]
                        output.stop_reason = map_stop_reason(reason_str)
                        if any(isinstance(b, ToolCall) for b in output.content):
                            output.stop_reason = "tool_use"

                usage_meta = getattr(chunk, "usage_metadata", None)
                if usage_meta:
                    output.usage = Usage(
                        input=getattr(usage_meta, "prompt_token_count", 0) or 0,
                        output=(getattr(usage_meta, "candidates_token_count", 0) or 0)
                        + (getattr(usage_meta, "thoughts_token_count", 0) or 0),
                        cache_read=getattr(usage_meta, "cached_content_token_count", 0) or 0,
                        cache_write=0,
                        total_tokens=getattr(usage_meta, "total_token_count", 0) or 0,
                    )
                    calculate_cost(model, output.usage)

            if current_block:
                _finish_block(current_block, blocks, bi, output, event_stream)

            if output.stop_reason in ("aborted", "error"):
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


def stream_simple_google_vertex(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    """Stream using the simple API with reasoning support."""
    base = build_base_options(model, options)

    if not options or not options.reasoning:
        return stream_google_vertex(
            model,
            context,
            GoogleVertexOptions(
                temperature=base.temperature,
                max_tokens=base.max_tokens,
                cache_retention=base.cache_retention,
                session_id=base.session_id,
                headers=base.headers,
                thinking={"enabled": False},
            ),
        )

    effort = clamp_reasoning(options.reasoning)

    if _is_gemini3_pro(model) or _is_gemini3_flash(model):
        return stream_google_vertex(
            model,
            context,
            GoogleVertexOptions(
                temperature=base.temperature,
                max_tokens=base.max_tokens,
                cache_retention=base.cache_retention,
                session_id=base.session_id,
                headers=base.headers,
                thinking={"enabled": True, "level": _get_gemini3_thinking_level(effort, model)},
            ),
        )

    return stream_google_vertex(
        model,
        context,
        GoogleVertexOptions(
            temperature=base.temperature,
            max_tokens=base.max_tokens,
            cache_retention=base.cache_retention,
            session_id=base.session_id,
            headers=base.headers,
            thinking={"enabled": True, "budgetTokens": _get_google_budget(model, effort, options.thinking_budgets)},
        ),
    )


def _part_to_dict(part: Any) -> dict[str, Any]:
    """Convert a google-genai Part object to dict."""
    result: dict[str, Any] = {}
    if hasattr(part, "text") and part.text is not None:
        result["text"] = part.text
    if hasattr(part, "thought"):
        result["thought"] = part.thought
    if hasattr(part, "thought_signature") and part.thought_signature:
        result["thoughtSignature"] = part.thought_signature
    if hasattr(part, "function_call") and part.function_call:
        fc = part.function_call
        result["functionCall"] = {
            "name": getattr(fc, "name", ""),
            "args": getattr(fc, "args", {}) or {},
            "id": getattr(fc, "id", None),
        }
    return result


def _finish_block(
    block: dict[str, Any],
    blocks: list[Any],
    bi: Any,
    output: AssistantMessage,
    stream: AssistantMessageEventStream,
) -> None:
    idx = bi()
    if block["type"] == "text":
        stream.push(TextEndEvent(content_index=idx, content=block["text"], partial=output))
    elif block["type"] == "thinking":
        stream.push(ThinkingEndEvent(content_index=idx, content=block["thinking"], partial=output))


def _create_client(
    model: Model,
    project: str,
    location: str,
    options_headers: dict[str, str] | None = None,
) -> Any:
    from google import genai

    http_options: dict[str, Any] = {}
    if model.headers or options_headers:
        http_options["headers"] = {**(model.headers or {}), **(options_headers or {})}

    return genai.Client(
        vertexai=True,
        project=project,
        location=location,
        api_version=API_VERSION,
        http_options=http_options if http_options else None,
    )


def _build_params(model: Model, context: Context, options: GoogleVertexOptions | None = None) -> dict[str, Any]:
    contents = convert_messages(model, context)

    config: dict[str, Any] = {}
    if options and options.temperature is not None:
        config["temperature"] = options.temperature
    if options and options.max_tokens is not None:
        config["max_output_tokens"] = options.max_tokens
    if context.system_prompt:
        config["system_instruction"] = _sanitize(context.system_prompt)
    if context.tools:
        config["tools"] = convert_tools(context.tools)
    if context.tools and options and options.tool_choice:
        config["tool_config"] = {"function_calling_config": {"mode": map_tool_choice(options.tool_choice)}}

    if options and options.thinking and options.thinking.get("enabled") and model.reasoning:
        thinking_config: dict[str, Any] = {"include_thoughts": True}
        if options.thinking.get("level") is not None:
            thinking_config["thinking_level"] = options.thinking["level"]
        elif options.thinking.get("budgetTokens") is not None:
            thinking_config["thinking_budget"] = options.thinking["budgetTokens"]
        config["thinking_config"] = thinking_config

    return {"model": model.id, "contents": contents, "config": config}


def _resolve_project(options: GoogleVertexOptions | None = None) -> str:
    """Resolve the GCP project ID from options or environment variables."""
    project = (
        (options and options.project)
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
        or os.environ.get("GCLOUD_PROJECT")
    )
    if not project:
        raise ValueError(
            "Vertex AI requires a project ID. Set GOOGLE_CLOUD_PROJECT/GCLOUD_PROJECT or pass project in options."
        )
    return project


def _resolve_location(options: GoogleVertexOptions | None = None) -> str:
    """Resolve the GCP location from options or environment variables."""
    location = (options and options.location) or os.environ.get("GOOGLE_CLOUD_LOCATION")
    if not location:
        raise ValueError("Vertex AI requires a location. Set GOOGLE_CLOUD_LOCATION or pass location in options.")
    return location


def _is_gemini3_pro(model: Model) -> bool:
    return "3-pro" in model.id


def _is_gemini3_flash(model: Model) -> bool:
    return "3-flash" in model.id


def _get_gemini3_thinking_level(effort: ThinkingLevel, model: Model) -> GoogleThinkingLevel:
    if _is_gemini3_pro(model):
        if effort in ("minimal", "low"):
            return "LOW"
        return "HIGH"
    mapping: dict[str, str] = {"minimal": "MINIMAL", "low": "LOW", "medium": "MEDIUM", "high": "HIGH"}
    return mapping.get(effort, "MEDIUM")


def _get_google_budget(model: Model, effort: ThinkingLevel, custom_budgets: Any = None) -> int:
    if custom_budgets:
        val = getattr(custom_budgets, effort, None)
        if val is not None:
            return val

    if "2.5-pro" in model.id:
        budgets = {"minimal": 128, "low": 2048, "medium": 8192, "high": 32768}
        return budgets.get(effort, 8192)

    if "2.5-flash" in model.id:
        budgets = {"minimal": 128, "low": 2048, "medium": 8192, "high": 24576}
        return budgets.get(effort, 8192)

    return -1  # Dynamic
