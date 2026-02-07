"""Context overflow detection across multiple LLM providers."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pi.ai.types import AssistantMessage, Model

_OVERFLOW_PATTERNS: list[re.Pattern[str]] = [
    # Anthropic
    re.compile(r"prompt is too long", re.IGNORECASE),
    re.compile(r"exceeds the model's maximum context", re.IGNORECASE),
    # OpenAI
    re.compile(r"maximum context length", re.IGNORECASE),
    re.compile(r"context_length_exceeded", re.IGNORECASE),
    re.compile(r"max_tokens.*exceeds.*model maximum", re.IGNORECASE),
    # Google
    re.compile(r"exceeds the maximum number of tokens", re.IGNORECASE),
    re.compile(r"Request payload size exceeds the limit", re.IGNORECASE),
    # xAI / Groq / general
    re.compile(r"token limit", re.IGNORECASE),
    re.compile(r"too many tokens", re.IGNORECASE),
    re.compile(r"rate_limit_exceeded.*tokens", re.IGNORECASE),
    # Cerebras / Mistral
    re.compile(r"context window", re.IGNORECASE),
    re.compile(r"input.*too long", re.IGNORECASE),
]


def get_overflow_patterns() -> list[re.Pattern[str]]:
    """Return the overflow detection patterns (for testing)."""
    return list(_OVERFLOW_PATTERNS)


def is_context_overflow(message: AssistantMessage, model: Model | None = None) -> bool:
    """Detect if an assistant message indicates context overflow.

    Checks both error messages against known provider patterns and
    silent overflow (usage.input exceeding context window).
    """
    # Check error message patterns
    if message.stop_reason in ("error", "aborted") and message.error_message:
        for pattern in _OVERFLOW_PATTERNS:
            if pattern.search(message.error_message):
                return True

    # Silent overflow detection (e.g., z.ai)
    return bool(model and model.context_window > 0 and message.usage.input > model.context_window)
