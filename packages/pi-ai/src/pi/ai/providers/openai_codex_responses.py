"""OpenAI Codex Responses API provider implementation.

Uses raw HTTP via httpx (no SDK) with JWT token parsing,
SSE stream parsing, and retry logic.
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import platform
import re
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx

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
    StopReason,
    Usage,
)

DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api"
JWT_CLAIM_PATH = "https://api.openai.com/auth"
MAX_RETRIES = 3
BASE_DELAY_MS = 1000
CODEX_TOOL_CALL_PROVIDERS = {"openai", "openai-codex", "opencode"}
CODEX_RESPONSE_STATUSES = {"completed", "incomplete", "failed", "cancelled", "queued", "in_progress"}


@dataclass
class OpenAICodexResponsesOptions:
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
    text_verbosity: str | None = None


def stream_openai_codex_responses(
    model: Model,
    context: Context,
    options: OpenAICodexResponsesOptions | None = None,
) -> AssistantMessageEventStream:
    """Stream a response from the OpenAI Codex Responses API."""
    event_stream = AssistantMessageEventStream()

    async def _run() -> None:
        output = AssistantMessage(
            role="assistant",
            content=[],
            api="openai-codex-responses",
            provider=model.provider,
            model=model.id,
            usage=Usage(),
            stop_reason="stop",
            timestamp=int(time.time() * 1000),
        )

        try:
            opts = options or OpenAICodexResponsesOptions()
            api_key = opts.api_key or get_env_api_key(model.provider) or ""
            if not api_key:
                raise ValueError(f"No API key for provider: {model.provider}")

            account_id = _extract_account_id(api_key)
            body = _build_request_body(model, context, opts)
            if opts.on_payload:
                opts.on_payload(body)

            headers = _build_headers(model.headers, opts.headers, account_id, api_key, opts.session_id)
            body_json = json.dumps(body)
            url = _resolve_codex_url(model.base_url)

            # Retry logic
            response: httpx.Response | None = None
            last_error: Exception | None = None

            async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
                for attempt in range(MAX_RETRIES + 1):
                    try:
                        response = await client.post(url, headers=headers, content=body_json)

                        if response.status_code < 400:
                            break

                        error_text = response.text
                        if attempt < MAX_RETRIES and _is_retryable_error(response.status_code, error_text):
                            delay_ms = BASE_DELAY_MS * (2**attempt)
                            await asyncio.sleep(delay_ms / 1000)
                            continue

                        info = _parse_error_response(error_text, response.status_code)
                        raise RuntimeError(info.get("friendly_message") or info["message"])

                    except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout) as e:
                        last_error = e
                        if attempt < MAX_RETRIES:
                            delay_ms = BASE_DELAY_MS * (2**attempt)
                            await asyncio.sleep(delay_ms / 1000)
                            continue
                        raise RuntimeError(str(e)) from e

                if response is None or response.status_code >= 400:
                    raise last_error or RuntimeError("Failed after retries")

                event_stream.push(StartEvent(partial=output))

                # Process SSE stream
                await _process_stream(response, output, event_stream, model)

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


def stream_simple_openai_codex_responses(
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

    return stream_openai_codex_responses(
        model,
        context,
        OpenAICodexResponsesOptions(
            temperature=base.temperature,
            max_tokens=base.max_tokens,
            api_key=api_key,
            cache_retention=base.cache_retention,
            session_id=base.session_id,
            headers=base.headers,
            reasoning_effort=reasoning_effort,
        ),
    )


def _build_request_body(model: Model, context: Context, options: OpenAICodexResponsesOptions) -> dict[str, Any]:
    messages = convert_responses_messages(model, context, CODEX_TOOL_CALL_PROVIDERS, include_system_prompt=False)

    body: dict[str, Any] = {
        "model": model.id,
        "store": False,
        "stream": True,
        "instructions": context.system_prompt,
        "input": messages,
        "text": {"verbosity": options.text_verbosity or "medium"},
        "include": ["reasoning.encrypted_content"],
        "prompt_cache_key": options.session_id,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
    }

    if options.temperature is not None:
        body["temperature"] = options.temperature
    if context.tools:
        body["tools"] = convert_responses_tools(context.tools, strict=None)

    if options.reasoning_effort is not None:
        body["reasoning"] = {
            "effort": _clamp_reasoning_effort(model.id, options.reasoning_effort),
            "summary": options.reasoning_summary or "auto",
        }

    return body


def _clamp_reasoning_effort(model_id: str, effort: str) -> str:
    mid = model_id.split("/")[-1] if "/" in model_id else model_id
    if (mid.startswith("gpt-5.2") or mid.startswith("gpt-5.3")) and effort == "minimal":
        return "low"
    if mid == "gpt-5.1" and effort == "xhigh":
        return "high"
    if mid == "gpt-5.1-codex-mini":
        return "high" if effort in ("high", "xhigh") else "medium"
    return effort


def _resolve_codex_url(base_url: str | None) -> str:
    raw = base_url.strip() if base_url and base_url.strip() else DEFAULT_CODEX_BASE_URL
    normalized = raw.rstrip("/")
    if normalized.endswith("/codex/responses"):
        return normalized
    if normalized.endswith("/codex"):
        return f"{normalized}/responses"
    return f"{normalized}/codex/responses"


async def _process_stream(
    response: httpx.Response,
    output: AssistantMessage,
    stream: AssistantMessageEventStream,
    model: Model,
) -> None:
    await process_responses_stream(_map_codex_events(_parse_sse(response)), output, stream, model)


async def _map_codex_events(events: AsyncIterator[dict[str, Any]]) -> AsyncIterator[Any]:
    """Map Codex events to OpenAI Responses stream events."""

    class _Event:
        def __init__(self, data: dict[str, Any]):
            self._data = data
            self.type = data.get("type", "")

        def __getattr__(self, name: str) -> Any:
            if name == "_data":
                return super().__getattribute__("_data")
            val = self._data.get(name)
            if isinstance(val, dict):
                return _Event(val)
            if isinstance(val, list):
                return [_Event(v) if isinstance(v, dict) else v for v in val]
            return val

    async for event_data in events:
        event_type = event_data.get("type", "")
        if not event_type:
            continue

        if event_type == "error":
            code = event_data.get("code", "")
            message = event_data.get("message", "")
            raise RuntimeError(f"Codex error: {message or code or json.dumps(event_data)}")

        if event_type == "response.failed":
            msg = (event_data.get("response") or {}).get("error", {}).get("message", "")
            raise RuntimeError(msg or "Codex response failed")

        if event_type in ("response.done", "response.completed"):
            resp = event_data.get("response", {})
            status = resp.get("status") if resp else None
            normalized_status = status if isinstance(status, str) and status in CODEX_RESPONSE_STATUSES else None
            if resp:
                resp = {**resp, "status": normalized_status}
            yield _Event({**event_data, "type": "response.completed", "response": resp})
            continue

        yield _Event(event_data)


async def _parse_sse(response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
    """Parse SSE events from an httpx response."""
    buffer = ""
    async for chunk in response.aiter_text():
        buffer += chunk
        while "\n\n" in buffer:
            idx = buffer.index("\n\n")
            raw = buffer[:idx]
            buffer = buffer[idx + 2 :]

            data_lines = [line[5:].strip() for line in raw.split("\n") if line.startswith("data:")]
            if data_lines:
                data = "\n".join(data_lines).strip()
                if data and data != "[DONE]":
                    try:
                        yield json.loads(data)
                    except json.JSONDecodeError:
                        pass


def _is_retryable_error(status: int, error_text: str) -> bool:
    if status in (429, 500, 502, 503, 504):
        return True
    return bool(re.search(r"rate.?limit|overloaded|service.?unavailable|upstream.?connect|connection.?refused", error_text, re.IGNORECASE))


def _parse_error_response(raw: str, status_code: int) -> dict[str, Any]:
    message = raw or f"Request failed with status {status_code}"
    friendly_message: str | None = None

    try:
        parsed = json.loads(raw)
        err = parsed.get("error", {})
        if err:
            code = err.get("code", "") or err.get("type", "")
            if re.search(r"usage_limit_reached|usage_not_included|rate_limit_exceeded", code, re.IGNORECASE) or status_code == 429:
                plan = f" ({err.get('plan_type', '').lower()} plan)" if err.get("plan_type") else ""
                resets_at = err.get("resets_at")
                when = ""
                if resets_at:
                    mins = max(0, round((resets_at * 1000 - time.time() * 1000) / 60000))
                    when = f" Try again in ~{mins} min."
                friendly_message = f"You have hit your ChatGPT usage limit{plan}.{when}".strip()
            message = err.get("message") or friendly_message or message
    except json.JSONDecodeError:
        pass

    return {"message": message, "friendly_message": friendly_message}


def _extract_account_id(token: str) -> str:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid token")
        # Add padding if needed
        payload_b64 = parts[1]
        payload_b64 += "=" * (4 - len(payload_b64) % 4)
        payload = json.loads(base64.b64decode(payload_b64))
        account_id = payload.get(JWT_CLAIM_PATH, {}).get("chatgpt_account_id")
        if not account_id:
            raise ValueError("No account ID in token")
        return account_id
    except Exception as e:
        raise ValueError(f"Failed to extract accountId from token: {e}") from e


def _build_headers(
    init_headers: dict[str, str] | None,
    additional_headers: dict[str, str] | None,
    account_id: str,
    token: str,
    session_id: str | None = None,
) -> dict[str, str]:
    headers: dict[str, str] = dict(init_headers or {})
    headers["Authorization"] = f"Bearer {token}"
    headers["chatgpt-account-id"] = account_id
    headers["OpenAI-Beta"] = "responses=experimental"
    headers["originator"] = "pi"
    headers["User-Agent"] = f"pi ({platform.system().lower()} {platform.release()}; {platform.machine()})"
    headers["accept"] = "text/event-stream"
    headers["content-type"] = "application/json"

    if additional_headers:
        headers.update(additional_headers)
    if session_id:
        headers["session_id"] = session_id

    return headers
