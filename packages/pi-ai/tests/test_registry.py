"""Tests for the API provider registry."""

from pi.ai.events import AssistantMessageEventStream
from pi.ai.registry import (
    ApiProvider,
    clear_api_providers,
    get_api_provider,
    get_api_providers,
    register_api_provider,
    unregister_api_providers,
)


def _mock_stream(*args, **kwargs):
    return AssistantMessageEventStream()


def _mock_stream_simple(*args, **kwargs):
    return AssistantMessageEventStream()


def setup_function():
    clear_api_providers()


def test_register_and_get():
    provider = ApiProvider(api="test-api", stream=_mock_stream, stream_simple=_mock_stream_simple)
    register_api_provider(provider)
    result = get_api_provider("test-api")
    assert result is not None
    assert result.api == "test-api"


def test_get_nonexistent():
    assert get_api_provider("nonexistent") is None


def test_get_all_providers():
    p1 = ApiProvider(api="api-1", stream=_mock_stream, stream_simple=_mock_stream_simple)
    p2 = ApiProvider(api="api-2", stream=_mock_stream, stream_simple=_mock_stream_simple)
    register_api_provider(p1)
    register_api_provider(p2)
    all_providers = get_api_providers()
    apis = [p.api for p in all_providers]
    assert "api-1" in apis
    assert "api-2" in apis


def test_unregister_by_source():
    p1 = ApiProvider(api="api-a", stream=_mock_stream, stream_simple=_mock_stream_simple)
    p2 = ApiProvider(api="api-b", stream=_mock_stream, stream_simple=_mock_stream_simple)
    register_api_provider(p1, source_id="ext-1")
    register_api_provider(p2, source_id="ext-2")
    unregister_api_providers("ext-1")
    assert get_api_provider("api-a") is None
    assert get_api_provider("api-b") is not None


def test_clear():
    register_api_provider(ApiProvider(api="x", stream=_mock_stream, stream_simple=_mock_stream_simple))
    clear_api_providers()
    assert get_api_providers() == []
