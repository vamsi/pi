"""Tests for environment-based API key resolution."""

import os
from unittest.mock import patch

from pi.ai.env import get_env_api_key


def test_anthropic_key():
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test-123"}):
        assert get_env_api_key("anthropic") == "sk-test-123"


def test_anthropic_oauth_takes_precedence():
    with patch.dict(
        os.environ,
        {
            "ANTHROPIC_API_KEY": "sk-api",
            "ANTHROPIC_OAUTH_TOKEN": "sk-oauth",
        },
    ):
        assert get_env_api_key("anthropic") == "sk-oauth"


def test_openai_key():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-openai"}, clear=False):
        assert get_env_api_key("openai") == "sk-openai"


def test_unknown_provider():
    assert get_env_api_key("unknown-provider") is None


def test_github_copilot_key():
    with patch.dict(os.environ, {"GITHUB_TOKEN": "ghp-test"}, clear=False):
        assert get_env_api_key("github-copilot") == "ghp-test"
