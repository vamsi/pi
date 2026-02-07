"""Model resolution with fuzzy matching and provider selection.

Supports exact, partial, and glob-based model matching with optional
thinking level extraction from pattern suffixes.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pi.ai.env import get_env_api_key
from pi.ai.types import Model, ModelCost

if TYPE_CHECKING:
    from pi.coding.core.settings import SettingsManager

# --- Constants ---

VALID_THINKING_LEVELS = {"off", "minimal", "low", "medium", "high", "xhigh"}

DEFAULT_THINKING_LEVEL = "medium"

# Default model IDs per known provider
DEFAULT_MODEL_PER_PROVIDER: dict[str, str] = {
    "amazon-bedrock": "us.anthropic.claude-opus-4-6-v1",
    "anthropic": "claude-opus-4-6",
    "google": "gemini-2.5-pro",
    "google-gemini-cli": "gemini-2.5-pro",
    "google-antigravity": "gemini-3-pro-high",
    "google-vertex": "gemini-3-pro-preview",
    "openai": "gpt-5.1-codex",
    "azure-openai-responses": "gpt-5.2",
    "openai-codex": "gpt-5.3-codex",
    "github-copilot": "gpt-4o",
    "xai": "grok-4-fast-non-reasoning",
    "groq": "openai/gpt-oss-120b",
    "cerebras": "zai-glm-4.6",
    "openrouter": "openai/gpt-5.1-codex",
    "vercel-ai-gateway": "anthropic/claude-opus-4-6",
    "zai": "glm-4.6",
    "mistral": "devstral-medium-latest",
    "minimax": "MiniMax-M2.1",
    "minimax-cn": "MiniMax-M2.1",
    "huggingface": "moonshotai/Kimi-K2.5",
    "opencode": "claude-opus-4-6",
    "kimi-coding": "kimi-k2-thinking",
}

# Date suffix pattern for distinguishing aliases from dated versions
_DATE_SUFFIX_RE = re.compile(r"-\d{8}$")


# --- Types ---


@dataclass
class ScopedModel:
    """A resolved model with optional thinking level override."""

    model: Model
    thinking_level: str | None = None


@dataclass
class ParsedModelResult:
    """Result of parsing a model pattern."""

    model: Model | None = None
    thinking_level: str | None = None
    warning: str | None = None


@dataclass
class InitialModelResult:
    """Result of finding the initial model with fallback logic."""

    model: Model | None = None
    thinking_level: str = DEFAULT_THINKING_LEVEL
    fallback_message: str | None = None


# --- Model registry ---


class ModelRegistry:
    """Registry of available models, combining built-in and custom models.

    Loads custom models from a models.json configuration file.
    """

    def __init__(self, agent_dir: str | None = None) -> None:
        self._models: dict[str, Model] = {}
        self._providers: dict[str, dict[str, Model]] = {}
        self._custom_errors: list[str] = []
        self._agent_dir = agent_dir or _default_agent_dir()
        self._load_custom_models()

    def register(self, model: Model) -> None:
        """Register a model."""
        key = f"{model.provider}/{model.id}"
        self._models[key] = model
        if model.provider not in self._providers:
            self._providers[model.provider] = {}
        self._providers[model.provider][model.id] = model

    def register_provider(
        self,
        provider: str,
        *,
        base_url: str = "",
        api: str = "",
        models: list[dict[str, Any]] | None = None,
    ) -> None:
        """Register a provider with its models."""
        for model_def in models or []:
            model = Model(
                id=model_def["id"],
                name=model_def.get("name", model_def["id"]),
                api=model_def.get("api", api),
                provider=provider,
                baseUrl=model_def.get("baseUrl", base_url),
                reasoning=model_def.get("reasoning", False),
                input=model_def.get("input", ["text"]),
                cost=ModelCost(**model_def["cost"]) if "cost" in model_def else ModelCost(),
                contextWindow=model_def.get("contextWindow", 0),
                maxTokens=model_def.get("maxTokens", 0),
                headers=model_def.get("headers"),
            )
            self.register(model)

    def get_all(self) -> list[Model]:
        """Return all registered models."""
        return list(self._models.values())

    def get_available(self) -> list[Model]:
        """Return models with configured authentication (fast check, no OAuth refresh)."""
        available: list[Model] = []
        for model in self._models.values():
            if get_env_api_key(model.provider):
                available.append(model)
        return available

    def find(self, provider: str, model_id: str) -> Model | None:
        """Direct lookup by provider and model ID."""
        return self._models.get(f"{provider}/{model_id}")

    def get_providers(self) -> list[str]:
        """Return all provider names."""
        return list(self._providers.keys())

    def get_models_for_provider(self, provider: str) -> list[Model]:
        """Return all models for a provider."""
        return list((self._providers.get(provider) or {}).values())

    def get_custom_errors(self) -> list[str]:
        """Return any errors from loading custom models."""
        return list(self._custom_errors)

    def _load_custom_models(self) -> None:
        """Load custom model definitions from models.json."""
        models_path = os.path.join(self._agent_dir, "models.json")
        if not os.path.exists(models_path):
            return

        try:
            content = Path(models_path).read_text(encoding="utf-8")
            config = json.loads(content)
        except (OSError, json.JSONDecodeError) as e:
            self._custom_errors.append(f"Failed to load models.json: {e}")
            return

        providers = config.get("providers", {})
        if not isinstance(providers, dict):
            self._custom_errors.append("models.json: 'providers' must be an object")
            return

        for provider_name, provider_config in providers.items():
            if not isinstance(provider_config, dict):
                continue

            base_url = _resolve_config_value(provider_config.get("baseUrl", ""))
            _resolve_config_value(provider_config.get("apiKey", ""))  # validate but not stored yet
            api = provider_config.get("api", "")
            headers = provider_config.get("headers")

            # Register custom models
            for model_def in provider_config.get("models", []):
                if not isinstance(model_def, dict) or "id" not in model_def:
                    continue

                model_api = model_def.get("api", api)
                if not model_api:
                    self._custom_errors.append(f"Model {model_def['id']} in {provider_name}: missing 'api'")
                    continue

                cost_data = model_def.get("cost", {})
                model = Model(
                    id=model_def["id"],
                    name=model_def.get("name", model_def["id"]),
                    api=model_api,
                    provider=provider_name,
                    baseUrl=model_def.get("baseUrl", base_url),
                    reasoning=model_def.get("reasoning", False),
                    input=model_def.get("input", ["text"]),
                    cost=ModelCost(
                        input=cost_data.get("input", 0.0),
                        output=cost_data.get("output", 0.0),
                        cacheRead=cost_data.get("cacheRead", 0.0),
                        cacheWrite=cost_data.get("cacheWrite", 0.0),
                    ),
                    contextWindow=model_def.get("contextWindow", 0),
                    maxTokens=model_def.get("maxTokens", 0),
                    headers={**(headers or {}), **(model_def.get("headers") or {})},
                )
                self.register(model)

            # Apply model overrides to existing models
            for model_id, overrides in provider_config.get("modelOverrides", {}).items():
                existing = self.find(provider_name, model_id)
                if existing and isinstance(overrides, dict):
                    # Apply overrides by creating a new model with merged values
                    data = existing.model_dump(by_alias=True)
                    for key, value in overrides.items():
                        if key in data:
                            data[key] = value
                    self.register(Model.model_validate(data))


# --- Matching functions ---


def _is_alias(model_id: str) -> bool:
    """Check if a model ID is an alias (no date suffix like -YYYYMMDD)."""
    return not bool(_DATE_SUFFIX_RE.search(model_id))


def try_match_model(pattern: str, models: list[Model]) -> Model | None:
    """Try to match a pattern to a model using a 3-step hierarchy.

    1. Provider/ModelId format (e.g., "anthropic/claude-opus-4-6")
    2. Exact ID match (case-insensitive)
    3. Partial/fuzzy match (substring, prefers aliases)
    """
    pattern_lower = pattern.lower()

    # Step 1: Provider/ModelId format
    if "/" in pattern:
        parts = pattern.split("/", 1)
        provider_part = parts[0].lower()
        model_part = parts[1].lower()
        for model in models:
            if model.provider.lower() == provider_part and model.id.lower() == model_part:
                return model

    # Step 2: Exact ID match
    for model in models:
        if model.id.lower() == pattern_lower:
            return model

    # Step 3: Partial/fuzzy match
    matches: list[Model] = []
    for model in models:
        if pattern_lower in model.id.lower() or pattern_lower in model.name.lower():
            matches.append(model)

    if not matches:
        return None

    if len(matches) == 1:
        return matches[0]

    # Prefer aliases over dated versions
    aliases = [m for m in matches if _is_alias(m.id)]
    if aliases:
        aliases.sort(key=lambda m: m.id, reverse=True)
        return aliases[0]

    # Fall back to latest dated version
    matches.sort(key=lambda m: m.id, reverse=True)
    return matches[0]


def parse_model_pattern(pattern: str, models: list[Model]) -> ParsedModelResult:
    """Parse a model pattern with optional thinking level suffix.

    Supports patterns like "claude-opus:high" where ":high" specifies
    the thinking level.
    """
    # Try exact match on full pattern first
    model = try_match_model(pattern, models)
    if model:
        return ParsedModelResult(model=model)

    # Try splitting on last colon for thinking level suffix
    if ":" in pattern:
        last_colon = pattern.rfind(":")
        prefix = pattern[:last_colon]
        suffix = pattern[last_colon + 1 :]

        if suffix.lower() in VALID_THINKING_LEVELS:
            result = parse_model_pattern(prefix, models)
            result.thinking_level = suffix.lower()
            return result

        # Invalid suffix - warn and try prefix anyway
        result = parse_model_pattern(prefix, models)
        if result.model:
            result.warning = f"Unknown thinking level '{suffix}', using default"
        return result

    return ParsedModelResult(warning=f"No model found matching '{pattern}'")


def resolve_model_scope(
    patterns: list[str],
    models: list[Model],
) -> tuple[list[ScopedModel], list[str]]:
    """Convert one or more patterns to a list of ScopedModels.

    Supports glob patterns (*, ?, [) and plain patterns with thinking level suffixes.
    Returns (scoped_models, warnings).
    """
    results: list[ScopedModel] = []
    warnings: list[str] = []
    seen: set[str] = set()

    for pattern in patterns:
        # Extract thinking level from glob patterns
        thinking_level: str | None = None
        match_pattern = pattern

        if ":" in pattern:
            last_colon = pattern.rfind(":")
            suffix = pattern[last_colon + 1 :]
            if suffix.lower() in VALID_THINKING_LEVELS:
                thinking_level = suffix.lower()
                match_pattern = pattern[:last_colon]

        # Check if it's a glob pattern
        if any(c in match_pattern for c in "*?["):
            matched = False
            for model in models:
                full_id = f"{model.provider}/{model.id}"
                if fnmatch.fnmatch(full_id.lower(), match_pattern.lower()) or fnmatch.fnmatch(
                    model.id.lower(), match_pattern.lower()
                ):
                    key = f"{model.provider}/{model.id}"
                    if key not in seen:
                        seen.add(key)
                        results.append(ScopedModel(model=model, thinking_level=thinking_level))
                        matched = True
            if not matched:
                warnings.append(f"No models match pattern '{pattern}'")
        else:
            parsed = parse_model_pattern(pattern, models)
            if parsed.model:
                key = f"{parsed.model.provider}/{parsed.model.id}"
                if key not in seen:
                    seen.add(key)
                    level = thinking_level or parsed.thinking_level
                    results.append(ScopedModel(model=parsed.model, thinking_level=level))
            if parsed.warning:
                warnings.append(parsed.warning)

    return results, warnings


# --- Initial model finding ---


def find_initial_model(
    *,
    models: list[Model],
    cli_provider: str | None = None,
    cli_model: str | None = None,
    settings: SettingsManager | None = None,
    scoped_models: list[ScopedModel] | None = None,
    is_continue: bool = False,
) -> InitialModelResult:
    """Find the initial model using priority-based selection.

    Priority:
    1. CLI args (--provider + --model)
    2. First scoped model (only if not continuing)
    3. Saved default from settings
    4. First available model with auth
    5. Fallback to first model in list
    """
    # 1. CLI args
    if cli_provider and cli_model:
        for model in models:
            if model.provider == cli_provider and model.id == cli_model:
                return InitialModelResult(model=model)
        return InitialModelResult(fallback_message=f"Model {cli_provider}/{cli_model} not found")

    if cli_model:
        parsed = parse_model_pattern(cli_model, models)
        if parsed.model:
            return InitialModelResult(
                model=parsed.model,
                thinking_level=parsed.thinking_level or DEFAULT_THINKING_LEVEL,
            )
        return InitialModelResult(fallback_message=f"Model '{cli_model}' not found")

    # 2. Scoped models (only if not continuing)
    if scoped_models and not is_continue:
        first = scoped_models[0]
        return InitialModelResult(
            model=first.model,
            thinking_level=first.thinking_level or DEFAULT_THINKING_LEVEL,
        )

    # 3. Saved default from settings
    if settings:
        default_provider = settings.get_default_provider()
        default_model = settings.get_default_model()
        if default_provider and default_model:
            for model in models:
                if model.provider == default_provider and model.id == default_model:
                    return InitialModelResult(model=model)

    # 4. First available model with auth
    for model in models:
        if get_env_api_key(model.provider):
            return InitialModelResult(model=model)

    # 5. Fallback to first model
    if models:
        return InitialModelResult(
            model=models[0],
            fallback_message="No API key found, using first available model",
        )

    return InitialModelResult(fallback_message="No models available")


def restore_model_from_session(
    *,
    provider: str,
    model_id: str,
    models: list[Model],
) -> InitialModelResult:
    """Restore a model from a saved session, with fallback cascade.

    1. Try to restore exact model
    2. Try default model for that provider
    3. Use first available model with auth
    4. Return None
    """
    # 1. Exact restore
    for model in models:
        if model.provider == provider and model.id == model_id:
            if get_env_api_key(model.provider):
                return InitialModelResult(model=model)
            return InitialModelResult(
                fallback_message=f"No API key for {provider}, trying fallback",
            )

    # 2. Default model for provider
    default_id = DEFAULT_MODEL_PER_PROVIDER.get(provider)
    if default_id:
        for model in models:
            if model.provider == provider and model.id == default_id and get_env_api_key(model.provider):
                return InitialModelResult(
                    model=model,
                    fallback_message=f"Model {model_id} not found, using {default_id}",
                )

    # 3. First available with auth
    for model in models:
        if get_env_api_key(model.provider):
            return InitialModelResult(
                model=model,
                fallback_message=f"Could not restore {provider}/{model_id}, using {model.provider}/{model.id}",
            )

    return InitialModelResult(fallback_message="No models available")


# --- Config value resolution ---


def _resolve_config_value(value: str) -> str:
    """Resolve config values that reference environment variables or commands.

    Supports:
        env:VARIABLE_NAME -> os.environ["VARIABLE_NAME"]
        cmd:command -> subprocess output (not implemented for safety)
        literal string -> returned as-is
    """
    if not value:
        return ""

    if value.startswith("env:"):
        env_name = value[4:]
        return os.environ.get(env_name, "")

    return value


def _default_agent_dir() -> str:
    """Default agent data directory (~/.pi)."""
    return os.path.join(os.path.expanduser("~"), ".pi")
