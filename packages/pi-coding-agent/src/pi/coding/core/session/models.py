"""Agent session model management helper.

Handles model selection, cycling, thinking level management,
and persistence of model/thinking state to session and settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pi.ai.env import get_env_api_key

if TYPE_CHECKING:
    from pi.ai.types import Model
    from pi.coding.core.resolver import ScopedModel

# Ordered thinking levels (without xhigh)
THINKING_LEVELS: list[str] = ["off", "minimal", "low", "medium", "high"]

# Ordered thinking levels (with xhigh)
THINKING_LEVELS_WITH_XHIGH: list[str] = ["off", "minimal", "low", "medium", "high", "xhigh"]


@dataclass
class ModelCycleResult:
    """Result of cycling to the next/previous model."""

    model: Model
    thinking_level: str | None = None
    is_scoped: bool = False


class AgentSessionModels:
    """Model and thinking level management for AgentSession.

    Uses composition — takes a reference to the parent session.
    """

    def __init__(self, session: Any) -> None:
        self._session = session

    async def set_model(self, model: Model) -> None:
        """Set the active model, validate API key, persist, and clamp thinking level."""
        session = self._session
        prev_model = session.agent.state.model

        # Validate API key exists
        api_key = get_env_api_key(model.provider)
        if not api_key and session.agent.get_api_key:
            import inspect

            result = session.agent.get_api_key(model.provider)
            if inspect.isawaitable(result):
                api_key = await result
            else:
                api_key = result

        # Set model on agent
        session.agent.set_model(model)

        # Persist to session
        session.session_manager.append_model_change(model.id, model.provider)

        # Persist to settings
        session.settings_manager.set_default_model_and_provider(model.id, model.provider)

        # Clamp thinking level to what this model supports
        available = self.get_available_thinking_levels()
        current_level = session.agent.state.thinking_level
        clamped = _clamp_thinking_level(current_level, available)
        if clamped != current_level:
            session.agent.set_thinking_level(clamped)
            session.session_manager.append_thinking_level_change(clamped)
            session.settings_manager.set_default_thinking_level(clamped)

        # Emit model_select to extensions
        self._emit_model_select(model, prev_model)

    def cycle_model(self, direction: int = 1) -> ModelCycleResult | None:
        """Cycle to the next/previous model.

        Args:
            direction: 1 for next, -1 for previous.

        Returns:
            ModelCycleResult or None if no models to cycle through.
        """
        session = self._session
        scoped = session.scoped_models

        if scoped:
            return self._cycle_scoped_model(scoped, direction)
        return self._cycle_available_model(direction)

    def _cycle_scoped_model(self, scoped: list[ScopedModel], direction: int) -> ModelCycleResult | None:
        """Cycle through scoped models."""
        if not scoped:
            return None

        session = self._session
        current_model = session.agent.state.model

        # Find current index
        current_idx = -1
        if current_model:
            for i, s in enumerate(scoped):
                if s.model.provider == current_model.provider and s.model.id == current_model.id:
                    current_idx = i
                    break

        # Wrap around
        next_idx = (current_idx + direction) % len(scoped)
        next_scoped = scoped[next_idx]

        return ModelCycleResult(
            model=next_scoped.model,
            thinking_level=next_scoped.thinking_level,
            is_scoped=True,
        )

    def _cycle_available_model(self, direction: int) -> ModelCycleResult | None:
        """Cycle through all available models (those with API keys)."""
        session = self._session
        registry = session.model_registry
        if not registry:
            return None

        available = registry.get_available()
        if not available:
            return None

        current_model = session.agent.state.model

        # Find current index
        current_idx = -1
        if current_model:
            for i, m in enumerate(available):
                if m.provider == current_model.provider and m.id == current_model.id:
                    current_idx = i
                    break

        # Wrap around
        next_idx = (current_idx + direction) % len(available)
        return ModelCycleResult(model=available[next_idx])

    def set_thinking_level(self, level: str) -> str:
        """Set the thinking level, clamping to available levels.

        Returns the actual level set (may differ from requested if clamped).
        """
        session = self._session
        available = self.get_available_thinking_levels()
        clamped = _clamp_thinking_level(level, available)

        if clamped != session.agent.state.thinking_level:
            session.agent.set_thinking_level(clamped)
            session.session_manager.append_thinking_level_change(clamped)
            session.settings_manager.set_default_thinking_level(clamped)

        return clamped

    def cycle_thinking_level(self) -> str | None:
        """Cycle to the next thinking level.

        Returns the new level, or None if model doesn't support thinking.
        """
        session = self._session
        available = self.get_available_thinking_levels()

        if len(available) <= 1:
            return None

        current = session.agent.state.thinking_level
        try:
            current_idx = available.index(current)
        except ValueError:
            current_idx = 0

        next_idx = (current_idx + 1) % len(available)
        next_level = available[next_idx]

        session.agent.set_thinking_level(next_level)
        session.session_manager.append_thinking_level_change(next_level)
        session.settings_manager.set_default_thinking_level(next_level)

        return next_level

    def get_available_thinking_levels(self) -> list[str]:
        """Get thinking levels available for the current model."""
        session = self._session
        model = session.agent.state.model

        if not model or not model.reasoning:
            return ["off"]

        # Check if model supports xhigh
        if _model_supports_xhigh(model):
            return list(THINKING_LEVELS_WITH_XHIGH)

        return list(THINKING_LEVELS)

    def _emit_model_select(self, next_model: Model, prev_model: Model | None) -> None:
        """Emit model_select event to extensions if model actually changed."""
        if prev_model and prev_model.provider == next_model.provider and prev_model.id == next_model.id:
            return

        runner = self._session.extension_runner
        if runner:
            import asyncio

            from pi.coding.core.extensions.types import ModelSelectEvent

            event = ModelSelectEvent(
                model_id=next_model.id,
                provider=next_model.provider,
            )
            # Fire and forget — extension events are informational
            try:
                loop = asyncio.get_running_loop()
                _task = loop.create_task(runner.emit(event))  # noqa: RUF006
            except RuntimeError:
                pass


def _clamp_thinking_level(level: str, available: list[str]) -> str:
    """Find the nearest available thinking level.

    Searches forward first in the ordered levels array, then backward.
    """
    if level in available:
        return level

    if not available:
        return "off"

    # Find position of requested level in the full ordered list
    full = THINKING_LEVELS_WITH_XHIGH
    try:
        requested_idx = full.index(level)
    except ValueError:
        return available[0]

    # Search forward then backward
    for offset in range(1, len(full)):
        # Forward
        forward_idx = requested_idx + offset
        if forward_idx < len(full) and full[forward_idx] in available:
            return full[forward_idx]
        # Backward
        backward_idx = requested_idx - offset
        if backward_idx >= 0 and full[backward_idx] in available:
            return full[backward_idx]

    return available[0]


def _model_supports_xhigh(model: Model) -> bool:
    """Check if a model supports xhigh thinking level.

    Currently xhigh is only for specific Anthropic models.
    """
    if not model.reasoning:
        return False
    # Heuristic: Anthropic reasoning models with large context windows
    return model.provider == "anthropic" and "opus" in model.id.lower()
