"""Top-level stream dispatch functions.

These are the primary entry points for making LLM calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pi.ai.registry import get_api_provider

if TYPE_CHECKING:
    from pi.ai.events import AssistantMessageEventStream
    from pi.ai.types import AssistantMessage, Context, Model, SimpleStreamOptions, StreamOptions


def _resolve_api_provider(api: str):
    provider = get_api_provider(api)
    if provider is None:
        raise ValueError(f"No API provider registered for api: {api}")
    return provider


def stream(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
) -> AssistantMessageEventStream:
    """Stream an LLM response using the registered provider for the model's API."""
    provider = _resolve_api_provider(model.api)
    return provider.stream(model, context, options)


async def complete(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
) -> AssistantMessage:
    """Complete an LLM call and return the final message."""
    s = stream(model, context, options)
    return await s.result()


def stream_simple(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    """Stream using the simple API with reasoning level support."""
    provider = _resolve_api_provider(model.api)
    return provider.stream_simple(model, context, options)


async def complete_simple(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessage:
    """Complete using the simple API and return the final message."""
    s = stream_simple(model, context, options)
    return await s.result()
