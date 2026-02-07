"""API provider registry for managing LLM API implementations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pi.ai.events import AssistantMessageEventStream
from pi.ai.types import Context, Model, SimpleStreamOptions, StreamOptions

StreamFunction = Callable[[Model, Context, StreamOptions | None], AssistantMessageEventStream]
SimpleStreamFunction = Callable[[Model, Context, SimpleStreamOptions | None], AssistantMessageEventStream]


@dataclass
class ApiProvider:
    """An API provider implementation."""

    api: str
    stream: StreamFunction
    stream_simple: SimpleStreamFunction


@dataclass
class _RegisteredProvider:
    provider: ApiProvider
    source_id: str | None = None


_registry: dict[str, _RegisteredProvider] = {}


def register_api_provider(provider: ApiProvider, source_id: str | None = None) -> None:
    """Register an API provider implementation."""
    _registry[provider.api] = _RegisteredProvider(provider=provider, source_id=source_id)


def get_api_provider(api: str) -> ApiProvider | None:
    """Get a registered API provider by API name."""
    entry = _registry.get(api)
    return entry.provider if entry else None


def get_api_providers() -> list[ApiProvider]:
    """Get all registered API providers."""
    return [entry.provider for entry in _registry.values()]


def unregister_api_providers(source_id: str) -> None:
    """Remove all providers registered with a given source ID."""
    to_remove = [api for api, entry in _registry.items() if entry.source_id == source_id]
    for api in to_remove:
        del _registry[api]


def clear_api_providers() -> None:
    """Remove all registered providers."""
    _registry.clear()
