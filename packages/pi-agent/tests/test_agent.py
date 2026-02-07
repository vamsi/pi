"""Tests for the Agent class."""

from pi.agent.agent import Agent


def test_agent_creation():
    agent = Agent()
    assert agent.state.is_streaming is False
    assert agent.state.messages == []


def test_agent_set_system_prompt():
    agent = Agent()
    agent.set_system_prompt("Be helpful")
    assert agent.state.system_prompt == "Be helpful"


def test_agent_set_thinking_level():
    agent = Agent()
    agent.set_thinking_level("high")
    assert agent.state.thinking_level == "high"


def test_agent_queue_messages():
    from pi.ai.types import UserMessage

    agent = Agent()
    msg = UserMessage(content="test", timestamp=123)
    agent.steer(msg)
    assert agent.has_queued_messages()

    agent.clear_all_queues()
    assert not agent.has_queued_messages()


def test_agent_reset():
    from pi.ai.types import UserMessage

    agent = Agent()
    agent.append_message(UserMessage(content="test", timestamp=123))
    agent.steer(UserMessage(content="steer", timestamp=456))
    assert len(agent.state.messages) == 1
    assert agent.has_queued_messages()

    agent.reset()
    assert agent.state.messages == []
    assert not agent.has_queued_messages()


def test_agent_subscribe():
    agent = Agent()
    events = []
    unsub = agent.subscribe(lambda e: events.append(e))
    assert callable(unsub)

    # Unsubscribe
    unsub()
    # No error calling unsub twice
    unsub()
