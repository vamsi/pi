"""Azure OpenAI Responses API provider implementation."""

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
    Context,
    DoneEvent,
    ErrorEvent,
    Model,
    SimpleStreamOptions,
    StartEvent,
    Usage,
)

DEFAULT_AZURE_API_VERSION = "v1"
AZURE_TOOL_CALL_PROVIDERS = {"openai", "openai-codex", "opencode", "azure-openai-responses"}


def _parse_deployment_name_map(value: str | None) -> dict[str, str]:
    result: dict[str, str] = {}
    if not value:
        return result
    for entry in value.split(","):
        trimmed = entry.strip()
        if not trimmed:
            continue
        parts = trimmed.split("=", 1)
        if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
            continue
        result[parts[0].strip()] = parts[1].strip()
    return result


@dataclass
class AzureOpenAIResponsesOptions:
    temperature: float | None = None
    max_tokens: int | None = None
    api_key: str | None = None
    cache_retention: str = "short"
    session_id: str | None = None
    headers: dict[str, str] | None = None
    max_retry_delay_ms: int | None = None
    on_payload: Any = None
    reasoning_effort: str | None = None
    reasoning_summary: str | None = None
    azure_api_version: str | None = None
    azure_resource_name: str | None = None
    azure_base_url: str | None = None
    azure_deployment_name: str | None = None


def stream_azure_openai_responses(
    model: Model,
    context: Context,
    options: AzureOpenAIResponsesOptions | None = None,
) -> AssistantMessageEventStream:
    """Stream a response from the Azure OpenAI Responses API."""
    event_stream = AssistantMessageEventStream()

    async def _run() -> None:
        opts = options or AzureOpenAIResponsesOptions()
        deployment_name = _resolve_deployment_name(model, opts)

        output = AssistantMessage(
            role="assistant",
            content=[],
            api="azure-openai-responses",
            provider=model.provider,
            model=model.id,
            usage=Usage(),
            stop_reason="stop",
            timestamp=int(time.time() * 1000),
        )

        try:
            api_key = opts.api_key or get_env_api_key(model.provider) or ""
            if not api_key:
                raise ValueError("Azure OpenAI API key is required.")

            client = _create_client(model, api_key, opts)
            params = _build_params(model, context, opts, deployment_name)
            if opts.on_payload:
                opts.on_payload(params)

            openai_stream = await client.responses.create(**params)
            event_stream.push(StartEvent(partial=output))

            await process_responses_stream(openai_stream, output, event_stream, model)

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


def stream_simple_azure_openai_responses(
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

    return stream_azure_openai_responses(
        model,
        context,
        AzureOpenAIResponsesOptions(
            temperature=base.temperature,
            max_tokens=base.max_tokens,
            api_key=api_key,
            cache_retention=base.cache_retention,
            session_id=base.session_id,
            headers=base.headers,
            reasoning_effort=reasoning_effort,
        ),
    )


def _resolve_deployment_name(model: Model, options: AzureOpenAIResponsesOptions) -> str:
    if options.azure_deployment_name:
        return options.azure_deployment_name
    mapping = _parse_deployment_name_map(os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME_MAP"))
    return mapping.get(model.id, model.id)


def _resolve_azure_config(model: Model, options: AzureOpenAIResponsesOptions) -> tuple[str, str]:
    api_version = (
        options.azure_api_version
        or os.environ.get("AZURE_OPENAI_API_VERSION")
        or DEFAULT_AZURE_API_VERSION
    )
    base_url = (options.azure_base_url or "").strip() or (os.environ.get("AZURE_OPENAI_BASE_URL") or "").strip() or None
    resource_name = options.azure_resource_name or os.environ.get("AZURE_OPENAI_RESOURCE_NAME")

    if not base_url and resource_name:
        base_url = f"https://{resource_name}.openai.azure.com/openai/v1"
    if not base_url:
        base_url = model.base_url
    if not base_url:
        raise ValueError("Azure OpenAI base URL is required.")

    return base_url.rstrip("/"), api_version


def _create_client(model: Model, api_key: str, options: AzureOpenAIResponsesOptions) -> Any:
    from openai import AsyncAzureOpenAI

    headers = dict(model.headers or {})
    if options.headers:
        headers.update(options.headers)

    base_url, api_version = _resolve_azure_config(model, options)

    return AsyncAzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=base_url,
        default_headers=headers if headers else None,
    )


def _build_params(
    model: Model,
    context: Context,
    options: AzureOpenAIResponsesOptions,
    deployment_name: str,
) -> dict[str, Any]:
    messages = convert_responses_messages(model, context, AZURE_TOOL_CALL_PROVIDERS)

    params: dict[str, Any] = {
        "model": deployment_name,
        "input": messages,
        "stream": True,
    }

    if options.session_id:
        params["prompt_cache_key"] = options.session_id
    if options.max_tokens:
        params["max_output_tokens"] = options.max_tokens
    if options.temperature is not None:
        params["temperature"] = options.temperature
    if context.tools:
        params["tools"] = convert_responses_tools(context.tools)

    if model.reasoning:
        if options.reasoning_effort or options.reasoning_summary:
            params["reasoning"] = {
                "effort": options.reasoning_effort or "medium",
                "summary": options.reasoning_summary or "auto",
            }
            params["include"] = ["reasoning.encrypted_content"]
        elif model.name.lower().startswith("gpt-5"):
            messages.append(
                {"role": "developer", "content": [{"type": "input_text", "text": "# Juice: 0 !important"}]}
            )

    return params
