"""Tests for the extension system."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from pi.agent.types import AgentTool, AgentToolResult
from pi.ai.types import TextContent
from pi.coding.core.extensions.loader import (
    discover_extensions_in_dir,
    load_extension_from_factory,
    load_extensions,
)
from pi.coding.core.extensions.runner import ExtensionRunner
from pi.coding.core.extensions.types import (
    AgentStartEvent,
    Extension,
    ExtensionAPI,
    ExtensionError,
    RegisteredCommand,
    RegisteredTool,
    ToolCallEvent,
    ToolDefinition,
)
from pi.coding.core.extensions.wrapper import (
    wrap_registered_tool,
    wrap_tool_with_extensions,
    wrap_tools_with_extensions,
)

# --- Extension API ---


def test_extension_api_register_handler():
    ext = Extension(path="test", resolved_path="test")
    api = ExtensionAPI(ext, "/tmp")

    handler_called = []

    def my_handler(event, ctx):
        handler_called.append(event.type)

    api.on("agent_start", my_handler)
    assert "agent_start" in ext.handlers
    assert len(ext.handlers["agent_start"]) == 1


def test_extension_api_register_multiple_handlers():
    ext = Extension(path="test", resolved_path="test")
    api = ExtensionAPI(ext, "/tmp")

    api.on("agent_start", lambda e, c: None)
    api.on("agent_start", lambda e, c: None)
    api.on("agent_end", lambda e, c: None)

    assert len(ext.handlers["agent_start"]) == 2
    assert len(ext.handlers["agent_end"]) == 1


def test_extension_api_register_tool():
    ext = Extension(path="test", resolved_path="test")
    api = ExtensionAPI(ext, "/tmp")

    async def execute(tool_call_id, params, **kwargs):
        return AgentToolResult(content=[TextContent(text="result")])

    defn = ToolDefinition(
        name="my_tool",
        label="My Tool",
        description="A test tool",
        parameters={"type": "object"},
        execute=execute,
    )
    api.register_tool(defn)

    assert "my_tool" in ext.tools
    assert ext.tools["my_tool"].definition.name == "my_tool"


def test_extension_api_register_command():
    ext = Extension(path="test", resolved_path="test")
    api = ExtensionAPI(ext, "/tmp")

    async def handler(ctx):
        pass

    api.register_command("test", description="Test command", handler=handler)
    assert "test" in ext.commands
    assert ext.commands["test"].description == "Test command"


def test_extension_api_register_command_requires_handler():
    ext = Extension(path="test", resolved_path="test")
    api = ExtensionAPI(ext, "/tmp")

    with pytest.raises(ValueError, match="requires a handler"):
        api.register_command("test", description="Test")


def test_extension_api_register_flag():
    ext = Extension(path="test", resolved_path="test")
    api = ExtensionAPI(ext, "/tmp")

    api.register_flag("verbose", description="Verbose mode", flag_type="boolean", default=False)
    assert "verbose" in ext.flags
    assert ext.flags["verbose"].flag_type == "boolean"


def test_extension_api_register_shortcut():
    ext = Extension(path="test", resolved_path="test")
    api = ExtensionAPI(ext, "/tmp")

    async def handler(ctx):
        pass

    api.register_shortcut("ctrl+t", description="Test shortcut", handler=handler)
    assert "ctrl+t" in ext.shortcuts


def test_extension_api_cwd():
    ext = Extension(path="test", resolved_path="test")
    api = ExtensionAPI(ext, "/my/cwd")
    assert api.cwd == "/my/cwd"


# --- Extension Runner ---


@pytest.mark.asyncio
async def test_runner_emit():
    ext = Extension(path="test", resolved_path="test")
    events_received: list[str] = []

    def handler(event, ctx):
        events_received.append(event.type)

    ext.handlers["agent_start"] = [handler]

    runner = ExtensionRunner([ext], "/tmp")
    await runner.emit(AgentStartEvent())
    assert events_received == ["agent_start"]


@pytest.mark.asyncio
async def test_runner_emit_async_handler():
    ext = Extension(path="test", resolved_path="test")
    events_received: list[str] = []

    async def handler(event, ctx):
        events_received.append(event.type)

    ext.handlers["agent_start"] = [handler]

    runner = ExtensionRunner([ext], "/tmp")
    await runner.emit(AgentStartEvent())
    assert events_received == ["agent_start"]


@pytest.mark.asyncio
async def test_runner_error_isolation():
    ext = Extension(path="test", resolved_path="test")
    second_called = []

    def bad_handler(event, ctx):
        raise RuntimeError("Extension error")

    def good_handler(event, ctx):
        second_called.append(True)

    ext.handlers["agent_start"] = [bad_handler, good_handler]

    errors: list[ExtensionError] = []
    runner = ExtensionRunner([ext], "/tmp")
    runner.on_error(errors.append)

    await runner.emit(AgentStartEvent())
    assert len(errors) == 1
    assert "Extension error" in errors[0].error
    assert second_called == [True]


@pytest.mark.asyncio
async def test_runner_tool_call_block():
    ext = Extension(path="test", resolved_path="test")

    def block_handler(event, ctx):
        event.blocked = True
        event.block_reason = "Not allowed"

    ext.handlers["tool_call"] = [block_handler]

    runner = ExtensionRunner([ext], "/tmp")
    event = ToolCallEvent(tool_name="bash", arguments={"command": "rm -rf /"})
    result = await runner.emit_tool_call(event)
    assert result.blocked is True
    assert result.block_reason == "Not allowed"


@pytest.mark.asyncio
async def test_runner_context_modification():
    ext = Extension(path="test", resolved_path="test")

    def context_handler(event, ctx):
        event.messages.append({"role": "user", "content": "injected"})

    ext.handlers["context"] = [context_handler]

    runner = ExtensionRunner([ext], "/tmp")
    messages = [{"role": "user", "content": "original"}]
    result = await runner.emit_context(messages)
    assert len(result) == 2
    assert result[1]["content"] == "injected"
    # Original should not be modified (deep copy)
    assert len(messages) == 1


@pytest.mark.asyncio
async def test_runner_input_transformation():
    ext = Extension(path="test", resolved_path="test")

    def input_handler(event, ctx):
        event.transformed_text = event.text.upper()

    ext.handlers["input"] = [input_handler]

    runner = ExtensionRunner([ext], "/tmp")
    result = await runner.emit_input("hello world")
    assert result == "HELLO WORLD"


@pytest.mark.asyncio
async def test_runner_input_no_transform():
    runner = ExtensionRunner([], "/tmp")
    result = await runner.emit_input("hello")
    assert result == "hello"


def test_runner_has_handlers():
    ext = Extension(path="test", resolved_path="test")
    ext.handlers["agent_start"] = [lambda e, c: None]

    runner = ExtensionRunner([ext], "/tmp")
    assert runner.has_handlers("agent_start")
    assert not runner.has_handlers("agent_end")


def test_runner_get_tools():
    ext = Extension(path="test", resolved_path="test")

    async def execute(tool_call_id, params, **kwargs):
        return AgentToolResult()

    defn = ToolDefinition(name="my_tool", label="My Tool", description="Test", parameters={}, execute=execute)
    ext.tools["my_tool"] = RegisteredTool(definition=defn, extension_path="test")

    runner = ExtensionRunner([ext], "/tmp")
    tools = runner.get_all_registered_tools()
    assert "my_tool" in tools

    assert runner.get_tool_definition("my_tool") is not None
    assert runner.get_tool_definition("nonexistent") is None


def test_runner_get_commands():
    ext = Extension(path="test", resolved_path="test")

    async def handler(ctx):
        pass

    ext.commands["test"] = RegisteredCommand(
        name="test", description="Test", handler=handler, extension_path="test", aliases=["t"]
    )

    runner = ExtensionRunner([ext], "/tmp")
    commands = runner.get_registered_commands()
    assert len(commands) == 1

    assert runner.get_command("test") is not None
    assert runner.get_command("t") is not None  # alias
    assert runner.get_command("unknown") is None


def test_runner_set_context():
    runner = ExtensionRunner([], "/tmp")
    runner.set_context(model_id="claude-opus-4-6", has_ui=True)
    assert runner.context.model_id == "claude-opus-4-6"
    assert runner.context.has_ui is True


def test_runner_error_unsubscribe():
    runner = ExtensionRunner([], "/tmp")
    errors: list[ExtensionError] = []
    unsub = runner.on_error(errors.append)

    runner._emit_error(ExtensionError(extension_path="test", event="test", error="err"))
    assert len(errors) == 1

    unsub()
    runner._emit_error(ExtensionError(extension_path="test", event="test", error="err2"))
    assert len(errors) == 1  # Unsubscribed, shouldn't receive


# --- Extension loader ---


def test_discover_extensions_in_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a .py file
        Path(os.path.join(tmpdir, "my_ext.py")).write_text("def extension(pi): pass\n", encoding="utf-8")
        # Create a directory with __init__.py
        ext_dir = os.path.join(tmpdir, "pkg_ext")
        os.makedirs(ext_dir)
        Path(os.path.join(ext_dir, "__init__.py")).write_text("def extension(pi): pass\n", encoding="utf-8")

        paths = discover_extensions_in_dir(tmpdir)
        assert len(paths) == 2


def test_discover_extensions_nonexistent_dir():
    paths = discover_extensions_in_dir("/nonexistent/dir")
    assert paths == []


@pytest.mark.asyncio
async def test_load_extension_from_factory():
    events: list[str] = []

    def my_factory(pi: ExtensionAPI):
        pi.on("agent_start", lambda e, c: events.append("started"))

    ext = await load_extension_from_factory(my_factory, "/tmp")
    assert "agent_start" in ext.handlers
    assert len(ext.handlers["agent_start"]) == 1


@pytest.mark.asyncio
async def test_load_extension_from_async_factory():
    async def my_factory(pi: ExtensionAPI):
        pi.on("agent_end", lambda e, c: None)

    ext = await load_extension_from_factory(my_factory, "/tmp")
    assert "agent_end" in ext.handlers


@pytest.mark.asyncio
async def test_load_extensions_from_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        ext_file = os.path.join(tmpdir, "test_ext.py")
        Path(ext_file).write_text(
            "def extension(pi):\n    pi.on('agent_start', lambda e, c: None)\n",
            encoding="utf-8",
        )

        extensions, errors = await load_extensions([ext_file], "/tmp")
        assert len(extensions) == 1
        assert len(errors) == 0
        assert "agent_start" in extensions[0].handlers


@pytest.mark.asyncio
async def test_load_extensions_bad_file():
    extensions, errors = await load_extensions(["/nonexistent.py"], "/tmp")
    assert len(extensions) == 0
    assert len(errors) == 1


@pytest.mark.asyncio
async def test_load_extensions_dedup():
    with tempfile.TemporaryDirectory() as tmpdir:
        ext_file = os.path.join(tmpdir, "test_ext.py")
        Path(ext_file).write_text("def extension(pi): pass\n", encoding="utf-8")

        # Pass same file twice
        extensions, _errors = await load_extensions([ext_file, ext_file], "/tmp")
        assert len(extensions) == 1  # Deduped


# --- Tool wrapping ---


@pytest.mark.asyncio
async def test_wrap_registered_tool():
    async def execute(tool_call_id, params, **kwargs):
        return AgentToolResult(content=[TextContent(text="done")])

    defn = ToolDefinition(name="my_tool", label="My Tool", description="Test", parameters={}, execute=execute)
    registered = RegisteredTool(definition=defn, extension_path="test")

    runner = ExtensionRunner([], "/tmp")
    agent_tool = wrap_registered_tool(registered, runner)

    assert agent_tool.name == "my_tool"
    assert agent_tool.execute is not None
    result = await agent_tool.execute("tc1", {})
    assert result.content[0].text == "done"


@pytest.mark.asyncio
async def test_wrap_tool_with_extensions_block():
    ext = Extension(path="test", resolved_path="test")

    def block_handler(event, ctx):
        if event.tool_name == "bash":
            event.blocked = True
            event.block_reason = "Blocked by extension"

    ext.handlers["tool_call"] = [block_handler]
    runner = ExtensionRunner([ext], "/tmp")

    async def execute(tool_call_id, arguments, **kwargs):
        return AgentToolResult(content=[TextContent(text="executed")])

    tool = AgentTool(name="bash", description="Run bash", parameters={}, execute=execute)
    wrapped = wrap_tool_with_extensions(tool, runner)

    result = await wrapped.execute("tc1", {})
    assert "blocked" in result.content[0].text.lower()


@pytest.mark.asyncio
async def test_wrap_tool_with_extensions_passthrough():
    runner = ExtensionRunner([], "/tmp")

    async def execute(tool_call_id, arguments, **kwargs):
        return AgentToolResult(content=[TextContent(text="ok")])

    tool = AgentTool(name="read", description="Read file", parameters={}, execute=execute)
    wrapped = wrap_tool_with_extensions(tool, runner)

    result = await wrapped.execute("tc1", {"path": "/test"})
    assert result.content[0].text == "ok"


def test_wrap_tools_with_extensions():
    runner = ExtensionRunner([], "/tmp")
    tools = [
        AgentTool(name="read", description="Read", parameters={}, execute=None),
        AgentTool(name="write", description="Write", parameters={}, execute=None),
    ]
    wrapped = wrap_tools_with_extensions(tools, runner)
    assert len(wrapped) == 2
    # Tools without execute should pass through unchanged
    assert wrapped[0].execute is None
