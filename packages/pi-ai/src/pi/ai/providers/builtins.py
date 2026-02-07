"""Register all built-in API providers."""

from __future__ import annotations

from pi.ai.providers.anthropic import stream_anthropic, stream_simple_anthropic
from pi.ai.providers.openai_completions import stream_openai_completions, stream_simple_openai_completions
from pi.ai.providers.openai_responses import stream_openai_responses, stream_simple_openai_responses
from pi.ai.providers.openai_codex_responses import stream_openai_codex_responses, stream_simple_openai_codex_responses
from pi.ai.providers.azure_openai_responses import stream_azure_openai_responses, stream_simple_azure_openai_responses
from pi.ai.registry import ApiProvider, register_api_provider


def register_builtin_providers() -> None:
    """Register all built-in LLM API providers."""
    # Core providers (always available)
    register_api_provider(
        ApiProvider(
            api="anthropic-messages",
            stream=stream_anthropic,
            stream_simple=stream_simple_anthropic,
        )
    )
    register_api_provider(
        ApiProvider(
            api="openai-completions",
            stream=stream_openai_completions,
            stream_simple=stream_simple_openai_completions,
        )
    )
    register_api_provider(
        ApiProvider(
            api="openai-responses",
            stream=stream_openai_responses,
            stream_simple=stream_simple_openai_responses,
        )
    )
    register_api_provider(
        ApiProvider(
            api="openai-codex-responses",
            stream=stream_openai_codex_responses,
            stream_simple=stream_simple_openai_codex_responses,
        )
    )
    register_api_provider(
        ApiProvider(
            api="azure-openai-responses",
            stream=stream_azure_openai_responses,
            stream_simple=stream_simple_azure_openai_responses,
        )
    )

    # Optional providers (require extra SDK dependencies)
    try:
        from pi.ai.providers.google import stream_google, stream_simple_google

        register_api_provider(
            ApiProvider(
                api="google-generative-ai",
                stream=stream_google,
                stream_simple=stream_simple_google,
            )
        )
    except ImportError:
        pass

    try:
        from pi.ai.providers.google_vertex import stream_google_vertex, stream_simple_google_vertex

        register_api_provider(
            ApiProvider(
                api="google-vertex",
                stream=stream_google_vertex,
                stream_simple=stream_simple_google_vertex,
            )
        )
    except ImportError:
        pass

    # google-gemini-cli uses httpx (core dep), so always available
    from pi.ai.providers.google_gemini_cli import stream_google_gemini_cli, stream_simple_google_gemini_cli

    register_api_provider(
        ApiProvider(
            api="google-gemini-cli",
            stream=stream_google_gemini_cli,
            stream_simple=stream_simple_google_gemini_cli,
        )
    )

    try:
        from pi.ai.providers.amazon_bedrock import stream_bedrock, stream_simple_bedrock

        register_api_provider(
            ApiProvider(
                api="bedrock-converse-stream",
                stream=stream_bedrock,
                stream_simple=stream_simple_bedrock,
            )
        )
    except ImportError:
        pass
