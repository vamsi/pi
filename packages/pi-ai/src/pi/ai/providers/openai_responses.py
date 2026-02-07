"""OpenAI Responses API provider implementation."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any

from pi.ai.env import get_env_api_key
from pi.ai.events import AssistantMessageEventStream
from pi.ai.models import supports_xhigh
from pi.ai.providers.openai_shared import (
    convert_responses_messages,
    convert_responses_tools,
    process_responses_stream,
)
from pi.ai.providers.options import build_base_options, clamp_reasoning
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
    Usage,
)

OPENAI_TOOL_CALL_PROVIDERS = {"openai", "openai-codex", "opencode"}


def _resolve_cache_retention(cache_retention: CacheRetention | None = None) -> CacheRetention:
    if cache_retention:
        return cache_retention
    if os.environ.get("PI_CACHE_RETENTION") == "long":
        return "long"
    return "short"


def _get_prompt_cache_retention(base_url: str, cache_retention: CacheRetention) -> str | None:
    if cache_retention != "long":
        return None
    if "api.openai.com" in base_url:
        return "24h"
    return None


@dataclass
class OpenAIResponsesOptions:
    temperature: float | None = None
    max_tokens: int | None = None
    api_key: str | None = None
    cache_retention: CacheRetention = "short"
    session_id: str | None = None
    headers: dict[str, str] | None = None
    max_retry_delay_ms: int | None = None
    on_payload: Any = None
    reasoning_effort: str | None = None
    reasoning_summary: str | None = None
    service_tier: str | None = None


def stream_openai_responses(
    model: Model,
    context: Context,
    options: OpenAIResponsesOptions | None = None,
) -> AssistantMessageEventStream:
    """Stream a response from the OpenAI Responses API."""
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
                raise ValueError("OpenAI API key is required.")

            client = _create_client(model, context, api_key, options.headers if options else None)
            params = _build_params(model, context, options)
            if options and options.on_payload:
                options.on_payload(params)

            openai_stream = await client.responses.create(**params)
            event_stream.push(StartEvent(partial=output))

            await process_responses_stream(
                openai_stream,
                output,
                event_stream,
                model,
                service_tier=options.service_tier if options else None,
                apply_service_tier_pricing=_apply_service_tier_pricing,
            )

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


def stream_simple_openai_responses(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    """Stream using the simple API with reasoning effort support."""
    api_key = (options and options.api_key) or get_env_api_key(model.provider)
    if not api_key:
        raise ValueError(f"No API key for provider: {model.provider}")

    base = build_base_options(model, options)
    reasoning_effort = (
        options.reasoning
        if options and supports_xhigh(model)
        else (clamp_reasoning(options.reasoning) if options and options.reasoning else None)
    )

    return stream_openai_responses(
        model,
        context,
        OpenAIResponsesOptions(
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


def _build_params(model: Model, context: Context, options: OpenAIResponsesOptions | None = None) -> dict[str, Any]:
    messages = convert_responses_messages(model, context, OPENAI_TOOL_CALL_PROVIDERS)

    cache_retention = _resolve_cache_retention(options.cache_retention if options else None)
    params: dict[str, Any] = {
        "model": model.id,
        "input": messages,
        "stream": True,
        "store": False,
    }

    if cache_retention != "none" and options and options.session_id:
        params["prompt_cache_key"] = options.session_id

    retention = _get_prompt_cache_retention(model.base_url, cache_retention)
    if retention:
        params["prompt_cache_retention"] = retention

    if options and options.max_tokens:
        params["max_output_tokens"] = options.max_tokens
    if options and options.temperature is not None:
        params["temperature"] = options.temperature
    if options and options.service_tier is not None:
        params["service_tier"] = options.service_tier
    if context.tools:
        params["tools"] = convert_responses_tools(context.tools)

    if model.reasoning:
        if (options and options.reasoning_effort) or (options and options.reasoning_summary):
            params["reasoning"] = {
                "effort": (options.reasoning_effort if options else None) or "medium",
                "summary": (options.reasoning_summary if options else None) or "auto",
            }
            params["include"] = ["reasoning.encrypted_content"]
        elif model.name.startswith("gpt-5"):
            messages.append(
                {"role": "developer", "content": [{"type": "input_text", "text": "# Juice: 0 !important"}]}
            )

    return params


def _get_service_tier_multiplier(service_tier: str | None) -> float:
    if service_tier == "flex":
        return 0.5
    if service_tier == "priority":
        return 2.0
    return 1.0


def _apply_service_tier_pricing(usage: Usage, service_tier: str | None) -> None:
    multiplier = _get_service_tier_multiplier(service_tier)
    if multiplier == 1.0:
        return
    usage.cost.input *= multiplier
    usage.cost.output *= multiplier
    usage.cost.cache_read *= multiplier
    usage.cost.cache_write *= multiplier
    usage.cost.total = usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write
