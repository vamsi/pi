"""Tool wrapping: adds extension hooks around tool execution.

Wraps AgentTools so that tool_call events are emitted before execution
(allowing extensions to block) and tool_result events after (allowing
extensions to modify results).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pi.agent.types import AgentTool, AgentToolResult
from pi.ai.types import TextContent
from pi.coding.core.extensions.types import (
    RegisteredTool,
    ToolCallEvent,
    ToolResultEvent,
)

if TYPE_CHECKING:
    from pi.coding.core.extensions.runner import ExtensionRunner


def wrap_registered_tool(registered: RegisteredTool, runner: ExtensionRunner) -> AgentTool:
    """Convert a RegisteredTool to an AgentTool with extension hooks."""
    defn = registered.definition

    async def execute(
        tool_call_id: str,
        arguments: dict[str, Any],
        **kwargs: Any,
    ) -> AgentToolResult:
        # Emit tool_call event (can block)
        call_event = ToolCallEvent(
            tool_name=defn.name,
            tool_call_id=tool_call_id,
            arguments=arguments,
        )
        call_event = await runner.emit_tool_call(call_event)

        if call_event.blocked:
            return AgentToolResult(
                content=[TextContent(text=f"Tool blocked: {call_event.block_reason}")],
            )

        # Execute the tool
        result = await defn.execute(tool_call_id, arguments, **kwargs)

        # Emit tool_result event (can modify result)
        result_event = ToolResultEvent(
            tool_name=defn.name,
            tool_call_id=tool_call_id,
            result=result,
        )
        result_event = await runner.emit_tool_result(result_event)

        return result_event.result or result

    return AgentTool(
        name=defn.name,
        description=defn.description,
        parameters=defn.parameters,
        label=defn.label,
        execute=execute,
    )


def wrap_registered_tools(
    registered_tools: dict[str, RegisteredTool],
    runner: ExtensionRunner,
) -> list[AgentTool]:
    """Convert all RegisteredTools to AgentTools."""
    return [wrap_registered_tool(rt, runner) for rt in registered_tools.values()]


def wrap_tool_with_extensions(
    tool: AgentTool,
    runner: ExtensionRunner,
) -> AgentTool:
    """Wrap an existing AgentTool with extension event hooks."""
    original_execute = tool.execute
    if original_execute is None:
        return tool

    async def execute(
        tool_call_id: str,
        arguments: dict[str, Any],
        **kwargs: Any,
    ) -> AgentToolResult:
        # Emit tool_call event
        call_event = ToolCallEvent(
            tool_name=tool.name,
            tool_call_id=tool_call_id,
            arguments=arguments,
        )
        call_event = await runner.emit_tool_call(call_event)

        if call_event.blocked:
            return AgentToolResult(
                content=[TextContent(text=f"Tool blocked: {call_event.block_reason}")],
            )

        # Execute original
        assert original_execute is not None
        result = await original_execute(tool_call_id, arguments, **kwargs)

        # Emit tool_result event
        result_event = ToolResultEvent(
            tool_name=tool.name,
            tool_call_id=tool_call_id,
            result=result,
        )
        result_event = await runner.emit_tool_result(result_event)

        return result_event.result or result

    return AgentTool(
        name=tool.name,
        description=tool.description,
        parameters=tool.parameters,
        label=tool.label,
        execute=execute,
    )


def wrap_tools_with_extensions(
    tools: list[AgentTool],
    runner: ExtensionRunner,
) -> list[AgentTool]:
    """Wrap all AgentTools with extension event hooks."""
    return [wrap_tool_with_extensions(t, runner) for t in tools]
