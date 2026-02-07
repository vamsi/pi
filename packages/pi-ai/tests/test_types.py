"""Tests for pi-ai core types."""

from pi.ai.types import (
    AssistantMessage,
    Context,
    DoneEvent,
    ErrorEvent,
    ImageContent,
    Model,
    ModelCost,
    SimpleStreamOptions,
    StartEvent,
    StreamOptions,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)


def test_text_content():
    tc = TextContent(text="hello")
    assert tc.type == "text"
    assert tc.text == "hello"
    assert tc.text_signature is None


def test_thinking_content():
    tc = ThinkingContent(thinking="let me think")
    assert tc.type == "thinking"
    assert tc.thinking == "let me think"


def test_image_content():
    ic = ImageContent(data="base64data", mime_type="image/png")
    assert ic.type == "image"
    assert ic.data == "base64data"
    assert ic.mime_type == "image/png"


def test_tool_call():
    tc = ToolCall(id="tc1", name="bash", arguments={"command": "ls"})
    assert tc.type == "tool_call"
    assert tc.id == "tc1"
    assert tc.name == "bash"


def test_usage():
    u = Usage(input=100, output=50, cache_read=10, cache_write=5, total_tokens=165)
    assert u.input == 100
    assert u.total_tokens == 165
    assert u.cost.total == 0.0


def test_user_message():
    msg = UserMessage(content="hello", timestamp=1234567890)
    assert msg.role == "user"
    assert msg.content == "hello"


def test_assistant_message():
    msg = AssistantMessage(
        content=[TextContent(text="hi")],
        api="anthropic-messages",
        provider="anthropic",
        model="claude-3-5-sonnet",
        timestamp=1234567890,
    )
    assert msg.role == "assistant"
    assert len(msg.content) == 1


def test_tool_result_message():
    msg = ToolResultMessage(
        tool_call_id="tc1",
        tool_name="bash",
        content=[TextContent(text="output")],
        is_error=False,
        timestamp=1234567890,
    )
    assert msg.role == "tool_result"
    assert msg.tool_call_id == "tc1"


def test_tool():
    t = Tool(
        name="test_tool",
        description="A test tool",
        parameters={"type": "object", "properties": {"x": {"type": "string"}}},
    )
    assert t.name == "test_tool"


def test_model():
    m = Model(
        id="test-model",
        name="Test Model",
        api="anthropic-messages",
        provider="anthropic",
        base_url="https://api.anthropic.com",
        reasoning=False,
        input=["text"],
        cost=ModelCost(input=3.0, output=15.0),
        context_window=200000,
        max_tokens=8192,
    )
    assert m.id == "test-model"
    assert m.cost.input == 3.0
    assert m.context_window == 200000


def test_context():
    ctx = Context(
        system_prompt="You are helpful",
        messages=[UserMessage(content="hello", timestamp=1234567890)],
    )
    assert ctx.system_prompt == "You are helpful"
    assert len(ctx.messages) == 1


def test_stream_options():
    opts = StreamOptions(temperature=0.7, max_tokens=4096)
    assert opts.temperature == 0.7
    assert opts.max_tokens == 4096


def test_simple_stream_options():
    opts = SimpleStreamOptions(reasoning="high", temperature=0.5)
    assert opts.reasoning == "high"
    assert opts.temperature == 0.5


def test_start_event():
    msg = AssistantMessage(timestamp=0)
    event = StartEvent(partial=msg)
    assert event.type == "start"


def test_done_event():
    msg = AssistantMessage(timestamp=0)
    event = DoneEvent(reason="stop", message=msg)
    assert event.type == "done"
    assert event.reason == "stop"


def test_error_event():
    msg = AssistantMessage(timestamp=0, stop_reason="error", error_message="oops")
    event = ErrorEvent(reason="error", error=msg)
    assert event.type == "error"


def test_serialization_roundtrip():
    """Test that models can be serialized and deserialized."""
    msg = AssistantMessage(
        content=[TextContent(text="hello"), ToolCall(id="tc1", name="bash", arguments={"cmd": "ls"})],
        api="anthropic-messages",
        provider="anthropic",
        model="claude-3-5-sonnet",
        usage=Usage(input=100, output=50, total_tokens=150),
        timestamp=1234567890,
    )
    data = msg.model_dump()
    restored = AssistantMessage.model_validate(data)
    assert restored.model == msg.model
    assert len(restored.content) == 2
