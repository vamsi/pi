"""Helper functions for building stream options with thinking/reasoning support."""

from __future__ import annotations

from pi.ai.types import Model, SimpleStreamOptions, StreamOptions, ThinkingLevel

DEFAULT_BUDGETS: dict[str, int] = {
    "minimal": 1024,
    "low": 2048,
    "medium": 8192,
    "high": 16384,
}


def build_base_options(model: Model, options: SimpleStreamOptions | None = None) -> StreamOptions:
    """Build base StreamOptions from SimpleStreamOptions.

    Defaults maxTokens to min(model.max_tokens, 32000).
    """
    max_tokens = min(model.max_tokens, 32000) if model.max_tokens > 0 else 32000
    if options is None:
        return StreamOptions(max_tokens=max_tokens)

    return StreamOptions(
        temperature=options.temperature,
        max_tokens=options.max_tokens or max_tokens,
        api_key=options.api_key,
        cache_retention=options.cache_retention,
        session_id=options.session_id,
        headers=options.headers,
        max_retry_delay_ms=options.max_retry_delay_ms,
        on_payload=options.on_payload,
    )


def clamp_reasoning(level: ThinkingLevel) -> ThinkingLevel:
    """Clamp xhigh thinking level to high."""
    return "high" if level == "xhigh" else level


def adjust_max_tokens_for_thinking(
    max_tokens: int,
    thinking_level: ThinkingLevel,
    custom_budgets: dict[str, int] | None = None,
) -> tuple[int, int]:
    """Calculate adjusted max_tokens and thinking budget for reasoning.

    Returns (adjusted_max_tokens, thinking_budget).
    Ensures minimum 1024 tokens for output.
    """
    budgets = {**DEFAULT_BUDGETS, **(custom_budgets or {})}
    thinking_budget = budgets.get(thinking_level, DEFAULT_BUDGETS["medium"])

    total = max_tokens + thinking_budget
    min_output = 1024

    if max_tokens < min_output:
        # Reduce thinking budget to ensure minimum output tokens
        thinking_budget = max(0, total - min_output)
        max_tokens = min_output
    else:
        max_tokens = total

    return max_tokens, thinking_budget
