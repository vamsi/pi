"""Model registry for managing LLM model definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pi.ai.types import Model, Usage

_model_registry: dict[str, dict[str, Model]] = {}


def register_models(provider: str, models: dict[str, Model]) -> None:
    """Register models for a provider."""
    _model_registry[provider] = models


def get_model(provider: str, model_id: str) -> Model | None:
    """Get a model by provider and model ID."""
    provider_models = _model_registry.get(provider)
    if provider_models is None:
        return None
    return provider_models.get(model_id)


def get_providers() -> list[str]:
    """Get all registered provider names."""
    return list(_model_registry.keys())


def get_models(provider: str) -> list[Model]:
    """Get all models for a provider."""
    provider_models = _model_registry.get(provider)
    return list(provider_models.values()) if provider_models else []


def calculate_cost(model: Model, usage: Usage) -> Usage:
    """Calculate cost based on model pricing and usage, updating usage in-place."""
    usage.cost.input = (model.cost.input / 1_000_000) * usage.input
    usage.cost.output = (model.cost.output / 1_000_000) * usage.output
    usage.cost.cache_read = (model.cost.cache_read / 1_000_000) * usage.cache_read
    usage.cost.cache_write = (model.cost.cache_write / 1_000_000) * usage.cache_write
    usage.cost.total = usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write
    return usage


def supports_xhigh(model: Model) -> bool:
    """Check if a model supports xhigh thinking level."""
    if "gpt-5.2" in model.id or "gpt-5.3" in model.id:
        return True
    if model.api == "anthropic-messages":
        return "opus-4-6" in model.id or "opus-4.6" in model.id
    return False


def models_are_equal(a: Model | None, b: Model | None) -> bool:
    """Check if two models are equal by ID and provider."""
    if a is None or b is None:
        return False
    return a.id == b.id and a.provider == b.provider
