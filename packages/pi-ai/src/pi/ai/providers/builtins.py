"""Register all built-in API providers."""

from __future__ import annotations

from pi.ai.providers.anthropic import stream_anthropic, stream_simple_anthropic
from pi.ai.registry import ApiProvider, register_api_provider


def register_builtin_providers() -> None:
    """Register all built-in LLM API providers."""
    register_api_provider(
        ApiProvider(
            api="anthropic-messages",
            stream=stream_anthropic,
            stream_simple=stream_simple_anthropic,
        )
    )
    # Additional providers will be registered here as they are ported:
    # - openai-completions
    # - openai-responses
    # - google-generative-ai
    # - bedrock-converse-stream
    # etc.
