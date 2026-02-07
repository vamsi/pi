"""Built-in model definitions for all providers.

Ported from models.generated.ts — includes the ~60 most commonly used models.
"""

from __future__ import annotations

from pi.ai.models import register_models
from pi.ai.types import Model, ModelCost, OpenAICompletionsCompat

# Shared constants
_ANTHROPIC_BASE = "https://api.anthropic.com"
_OPENAI_BASE = "https://api.openai.com/v1"
_GOOGLE_BASE = "https://generativelanguage.googleapis.com/v1beta"
_BEDROCK_BASE = "https://bedrock-runtime.us-east-1.amazonaws.com"
_CODEX_BASE = "https://chatgpt.com/backend-api"
_COPILOT_BASE = "https://api.individual.githubcopilot.com"

_COPILOT_HEADERS = {
    "User-Agent": "GitHubCopilotChat/0.35.0",
    "Editor-Version": "vscode/1.107.0",
    "Editor-Plugin-Version": "copilot-chat/0.35.0",
    "Copilot-Integration-Id": "vscode-chat",
}
_COPILOT_COMPAT = OpenAICompletionsCompat(
    supports_store=False,
    supports_developer_role=False,
    supports_reasoning_effort=False,
)


def _m(
    id: str,
    name: str,
    api: str,
    provider: str,
    base_url: str,
    *,
    reasoning: bool = False,
    input: list[str] | None = None,
    cost_in: float = 0,
    cost_out: float = 0,
    cache_read: float = 0,
    cache_write: float = 0,
    context_window: int = 0,
    max_tokens: int = 0,
    headers: dict[str, str] | None = None,
    compat: OpenAICompletionsCompat | None = None,
) -> Model:
    return Model(
        id=id,
        name=name,
        api=api,
        provider=provider,
        baseUrl=base_url,
        reasoning=reasoning,
        input=input or ["text"],
        cost=ModelCost(input=cost_in, output=cost_out, cacheRead=cache_read, cacheWrite=cache_write),
        contextWindow=context_window,
        maxTokens=max_tokens,
        headers=headers,
        compat=compat,
    )


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------
_ANTHROPIC_MODELS: dict[str, Model] = {}

for _id, _name, _reasoning, _cost_in, _cost_out, _cr, _cw, _mt in [
    ("claude-opus-4-6", "Claude Opus 4.6", True, 5, 25, 0.5, 6.25, 128000),
    ("claude-opus-4-5", "Claude Opus 4.5 (latest)", True, 5, 25, 0.5, 6.25, 64000),
    ("claude-opus-4-5-20251101", "Claude Opus 4.5", True, 5, 25, 0.5, 6.25, 64000),
    ("claude-opus-4-1", "Claude Opus 4.1 (latest)", True, 15, 75, 1.5, 18.75, 32000),
    ("claude-opus-4-1-20250805", "Claude Opus 4.1", True, 15, 75, 1.5, 18.75, 32000),
    ("claude-opus-4-0", "Claude Opus 4 (latest)", True, 15, 75, 1.5, 18.75, 32000),
    ("claude-opus-4-20250514", "Claude Opus 4", True, 15, 75, 1.5, 18.75, 32000),
    ("claude-sonnet-4-5", "Claude Sonnet 4.5 (latest)", True, 3, 15, 0.3, 3.75, 64000),
    ("claude-sonnet-4-5-20250929", "Claude Sonnet 4.5", True, 3, 15, 0.3, 3.75, 64000),
    ("claude-sonnet-4-0", "Claude Sonnet 4 (latest)", True, 3, 15, 0.3, 3.75, 64000),
    ("claude-sonnet-4-20250514", "Claude Sonnet 4", True, 3, 15, 0.3, 3.75, 64000),
    ("claude-3-7-sonnet-20250219", "Claude Sonnet 3.7", True, 3, 15, 0.3, 3.75, 64000),
    ("claude-3-5-sonnet-20241022", "Claude Sonnet 3.5 v2", False, 3, 15, 0.3, 3.75, 8192),
    ("claude-haiku-4-5", "Claude Haiku 4.5 (latest)", True, 1, 5, 0.1, 1.25, 64000),
    ("claude-haiku-4-5-20251001", "Claude Haiku 4.5", True, 1, 5, 0.1, 1.25, 64000),
    ("claude-3-5-haiku-20241022", "Claude Haiku 3.5", False, 0.8, 4, 0.08, 1, 8192),
]:
    _ANTHROPIC_MODELS[_id] = _m(
        _id, _name, "anthropic-messages", "anthropic", _ANTHROPIC_BASE,
        reasoning=_reasoning, input=["text", "image"],
        cost_in=_cost_in, cost_out=_cost_out, cache_read=_cr, cache_write=_cw,
        context_window=200000, max_tokens=_mt,
    )


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------
_OPENAI_MODELS: dict[str, Model] = {}

for _id, _name, _reasoning, _inp, _cost_in, _cost_out, _cr, _cw, _ctx, _mt in [
    ("gpt-5.2", "GPT-5.2", True, ["text", "image"], 1.75, 14, 0.175, 0, 400000, 128000),
    ("gpt-5.1", "GPT-5.1", True, ["text", "image"], 1.25, 10, 0.13, 0, 400000, 128000),
    ("gpt-4.1", "GPT-4.1", False, ["text", "image"], 2, 8, 0.5, 0, 1047576, 32768),
    ("gpt-4.1-mini", "GPT-4.1 mini", False, ["text", "image"], 0.4, 1.6, 0.1, 0, 1047576, 32768),
    ("gpt-4.1-nano", "GPT-4.1 nano", False, ["text", "image"], 0.1, 0.4, 0.03, 0, 1047576, 32768),
    ("o4-mini", "o4-mini", True, ["text", "image"], 1.1, 4.4, 0.28, 0, 200000, 100000),
    ("o3", "o3", True, ["text", "image"], 2, 8, 0.5, 0, 200000, 100000),
    ("o3-mini", "o3-mini", True, ["text"], 1.1, 4.4, 0.55, 0, 200000, 100000),
    ("gpt-4o", "GPT-4o", False, ["text", "image"], 2.5, 10, 1.25, 0, 128000, 16384),
    ("gpt-4o-mini", "GPT-4o mini", False, ["text", "image"], 0.15, 0.6, 0.08, 0, 128000, 16384),
]:
    _OPENAI_MODELS[_id] = _m(
        _id, _name, "openai-responses", "openai", _OPENAI_BASE,
        reasoning=_reasoning, input=_inp,
        cost_in=_cost_in, cost_out=_cost_out, cache_read=_cr, cache_write=_cw,
        context_window=_ctx, max_tokens=_mt,
    )


# ---------------------------------------------------------------------------
# Google
# ---------------------------------------------------------------------------
_GOOGLE_MODELS: dict[str, Model] = {}

for _id, _name, _reasoning, _cost_in, _cost_out, _cr, _ctx, _mt in [
    ("gemini-3-pro-preview", "Gemini 3 Pro Preview", True, 2, 12, 0.2, 1000000, 64000),
    ("gemini-3-flash-preview", "Gemini 3 Flash Preview", True, 0.5, 3, 0.05, 1048576, 65536),
    ("gemini-2.5-pro", "Gemini 2.5 Pro", True, 1.25, 10, 0.31, 1048576, 65536),
    ("gemini-2.5-flash", "Gemini 2.5 Flash", True, 0.3, 2.5, 0.075, 1048576, 65536),
    ("gemini-2.0-flash", "Gemini 2.0 Flash", False, 0.1, 0.4, 0.025, 1048576, 8192),
]:
    _GOOGLE_MODELS[_id] = _m(
        _id, _name, "google-generative-ai", "google", _GOOGLE_BASE,
        reasoning=_reasoning, input=["text", "image"],
        cost_in=_cost_in, cost_out=_cost_out, cache_read=_cr,
        context_window=_ctx, max_tokens=_mt,
    )


# ---------------------------------------------------------------------------
# Amazon Bedrock
# ---------------------------------------------------------------------------
_BEDROCK_MODELS: dict[str, Model] = {}

# Bedrock Claude models (bare and US-prefixed variants)
for _id, _name, _reasoning, _cost_in, _cost_out, _cr, _cw, _mt in [
    ("anthropic.claude-opus-4-6-v1", "Claude Opus 4.6", True, 5, 25, 0.5, 6.25, 128000),
    ("us.anthropic.claude-opus-4-6-v1", "Claude Opus 4.6 (US)", True, 5, 25, 0.5, 6.25, 128000),
    ("anthropic.claude-opus-4-5-20251101-v1:0", "Claude Opus 4.5", True, 5, 25, 0.5, 6.25, 64000),
    ("us.anthropic.claude-opus-4-5-20251101-v1:0", "Claude Opus 4.5 (US)", True, 5, 25, 0.5, 6.25, 64000),
    ("anthropic.claude-opus-4-1-20250805-v1:0", "Claude Opus 4.1", True, 15, 75, 1.5, 18.75, 32000),
    ("us.anthropic.claude-opus-4-1-20250805-v1:0", "Claude Opus 4.1 (US)", True, 15, 75, 1.5, 18.75, 32000),
    ("anthropic.claude-opus-4-20250514-v1:0", "Claude Opus 4", True, 15, 75, 1.5, 18.75, 32000),
    ("us.anthropic.claude-opus-4-20250514-v1:0", "Claude Opus 4 (US)", True, 15, 75, 1.5, 18.75, 32000),
    ("anthropic.claude-sonnet-4-5-20250929-v1:0", "Claude Sonnet 4.5", True, 3, 15, 0.3, 3.75, 64000),
    ("us.anthropic.claude-sonnet-4-5-20250929-v1:0", "Claude Sonnet 4.5 (US)", True, 3, 15, 0.3, 3.75, 64000),
    ("anthropic.claude-sonnet-4-20250514-v1:0", "Claude Sonnet 4", True, 3, 15, 0.3, 3.75, 64000),
    ("us.anthropic.claude-sonnet-4-20250514-v1:0", "Claude Sonnet 4 (US)", True, 3, 15, 0.3, 3.75, 64000),
    ("anthropic.claude-haiku-4-5-20251001-v1:0", "Claude Haiku 4.5", True, 1, 5, 0.1, 1.25, 64000),
    ("us.anthropic.claude-haiku-4-5-20251001-v1:0", "Claude Haiku 4.5 (US)", True, 1, 5, 0.1, 1.25, 64000),
    ("anthropic.claude-3-5-sonnet-20241022-v2:0", "Claude Sonnet 3.5 v2", False, 3, 15, 0.3, 3.75, 8192),
    ("anthropic.claude-3-5-haiku-20241022-v1:0", "Claude Haiku 3.5", False, 0.8, 4, 0.08, 1, 8192),
]:
    _BEDROCK_MODELS[_id] = _m(
        _id, _name, "bedrock-converse-stream", "amazon-bedrock", _BEDROCK_BASE,
        reasoning=_reasoning, input=["text", "image"],
        cost_in=_cost_in, cost_out=_cost_out, cache_read=_cr, cache_write=_cw,
        context_window=200000, max_tokens=_mt,
    )


# ---------------------------------------------------------------------------
# OpenAI Codex
# ---------------------------------------------------------------------------
_CODEX_MODELS: dict[str, Model] = {}

for _id, _name, _cost_in, _cost_out, _cr in [
    ("gpt-5.1", "GPT-5.1", 1.25, 10, 0.125),
    ("gpt-5.1-codex-max", "GPT-5.1 Codex Max", 1.25, 10, 0.125),
    ("gpt-5.1-codex-mini", "GPT-5.1 Codex Mini", 0.25, 2, 0.025),
    ("gpt-5.2", "GPT-5.2", 1.75, 14, 0.175),
    ("gpt-5.2-codex", "GPT-5.2 Codex", 1.75, 14, 0.175),
    ("gpt-5.3-codex", "GPT-5.3 Codex", 1.75, 14, 0.175),
]:
    _CODEX_MODELS[_id] = _m(
        _id, _name, "openai-codex-responses", "openai-codex", _CODEX_BASE,
        reasoning=True, input=["text", "image"],
        cost_in=_cost_in, cost_out=_cost_out, cache_read=_cr,
        context_window=272000, max_tokens=128000,
    )


# ---------------------------------------------------------------------------
# GitHub Copilot (openai-completions models with compat)
# ---------------------------------------------------------------------------
_COPILOT_MODELS: dict[str, Model] = {}

for _id, _name, _reasoning, _inp, _ctx, _mt in [
    ("claude-sonnet-4", "Claude Sonnet 4", True, ["text", "image"], 128000, 16000),
    ("claude-sonnet-4.5", "Claude Sonnet 4.5", True, ["text", "image"], 128000, 16000),
    ("claude-opus-4.5", "Claude Opus 4.5", True, ["text", "image"], 128000, 16000),
    ("claude-opus-4.6", "Claude Opus 4.6", True, ["text", "image"], 128000, 64000),
    ("claude-haiku-4.5", "Claude Haiku 4.5", True, ["text", "image"], 128000, 16000),
    ("gpt-4.1", "GPT-4.1", False, ["text", "image"], 128000, 16384),
    ("gpt-4o", "GPT-4o", False, ["text", "image"], 64000, 16384),
    ("gemini-2.5-pro", "Gemini 2.5 Pro", False, ["text", "image"], 128000, 64000),
    ("gemini-3-flash-preview", "Gemini 3 Flash", True, ["text", "image"], 128000, 64000),
    ("gemini-3-pro-preview", "Gemini 3 Pro Preview", True, ["text", "image"], 128000, 64000),
]:
    _COPILOT_MODELS[_id] = _m(
        _id, _name, "openai-completions", "github-copilot", _COPILOT_BASE,
        reasoning=_reasoning, input=_inp,
        context_window=_ctx, max_tokens=_mt,
        headers=_COPILOT_HEADERS, compat=_COPILOT_COMPAT,
    )

# Copilot GPT-5 models use openai-responses (no compat)
for _id, _name, _ctx, _mt in [
    ("gpt-5.2", "GPT-5.2", 128000, 64000),
    ("gpt-5.1", "GPT-5.1", 128000, 128000),
]:
    _COPILOT_MODELS[_id] = _m(
        _id, _name, "openai-responses", "github-copilot", _COPILOT_BASE,
        reasoning=True, input=["text", "image"],
        context_window=_ctx, max_tokens=_mt,
        headers=_COPILOT_HEADERS,
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_builtin_models() -> None:
    """Register all built-in models for all providers."""
    register_models("anthropic", _ANTHROPIC_MODELS)
    register_models("openai", _OPENAI_MODELS)
    register_models("google", _GOOGLE_MODELS)
    register_models("amazon-bedrock", _BEDROCK_MODELS)
    register_models("openai-codex", _CODEX_MODELS)
    register_models("github-copilot", _COPILOT_MODELS)
