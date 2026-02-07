"""Tests for model registry and utility functions."""

from pi.ai.models import calculate_cost, get_model, models_are_equal, register_models, supports_xhigh
from pi.ai.types import Model, ModelCost, Usage


def test_register_and_get_model():
    model = Model(
        id="test-model-1",
        name="Test Model",
        api="anthropic-messages",
        provider="test-provider",
        base_url="https://test.api.com",
        cost=ModelCost(input=3.0, output=15.0),
        context_window=100000,
        max_tokens=4096,
    )
    register_models("test-provider", {"test-model-1": model})
    result = get_model("test-provider", "test-model-1")
    assert result is not None
    assert result.id == "test-model-1"


def test_calculate_cost():
    model = Model(
        id="test",
        name="Test",
        api="test",
        provider="test",
        base_url="https://test.com",
        cost=ModelCost(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75),
        context_window=100000,
        max_tokens=4096,
    )
    usage = Usage(input=1000, output=500, cache_read=200, cache_write=100, total_tokens=1800)
    calculate_cost(model, usage)
    assert usage.cost.input == 3.0 / 1_000_000 * 1000
    assert usage.cost.output == 15.0 / 1_000_000 * 500
    assert usage.cost.total > 0


def test_models_are_equal():
    m1 = Model(
        id="a",
        name="A",
        api="test",
        provider="p1",
        base_url="https://test.com",
        cost=ModelCost(),
        context_window=0,
        max_tokens=0,
    )
    m2 = Model(
        id="a",
        name="A",
        api="test",
        provider="p1",
        base_url="https://test.com",
        cost=ModelCost(),
        context_window=0,
        max_tokens=0,
    )
    m3 = Model(
        id="b",
        name="B",
        api="test",
        provider="p1",
        base_url="https://test.com",
        cost=ModelCost(),
        context_window=0,
        max_tokens=0,
    )
    assert models_are_equal(m1, m2) is True
    assert models_are_equal(m1, m3) is False
    assert models_are_equal(None, m1) is False


def test_supports_xhigh():
    opus = Model(
        id="claude-opus-4-6",
        name="Opus",
        api="anthropic-messages",
        provider="anthropic",
        base_url="https://api.anthropic.com",
        cost=ModelCost(),
        context_window=0,
        max_tokens=0,
    )
    assert supports_xhigh(opus) is True

    sonnet = Model(
        id="claude-sonnet-4-5",
        name="Sonnet",
        api="anthropic-messages",
        provider="anthropic",
        base_url="https://api.anthropic.com",
        cost=ModelCost(),
        context_window=0,
        max_tokens=0,
    )
    assert supports_xhigh(sonnet) is False
