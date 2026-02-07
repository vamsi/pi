"""File writing tool."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pi.agent.types import AgentTool, AgentToolResult

if TYPE_CHECKING:
    from pi.mom.sandbox import Executor


def _shell_escape(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def create_write_tool(executor: Executor, cwd: str) -> AgentTool:
    async def execute(
        tool_call_id: str,
        args: dict[str, Any],
        **kwargs: Any,
    ) -> AgentToolResult:
        path = args["path"]
        content = args["content"]
        abort_event = kwargs.get("abort_event")

        d = path[: path.rfind("/")] if "/" in path else "."
        cmd = (
            f"mkdir -p {_shell_escape(d)} && "
            f"printf '%s' {_shell_escape(content)} > {_shell_escape(path)}"
        )

        result = await executor.exec(cmd, abort_event=abort_event)
        if result.code != 0:
            raise RuntimeError(result.stderr or f"Failed to write file: {path}")

        return AgentToolResult(
            content=[
                {
                    "type": "text",
                    "text": f"Successfully wrote {len(content)} bytes to {path}",
                }
            ],
        )

    return AgentTool(
        name="write",
        label="write",
        description=(
            "Write content to a file. Creates the file if it doesn't exist, "
            "overwrites if it does. Automatically creates parent directories."
        ),
        parameters={
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "description": "Brief description of what you're writing (shown to user)",
                },
                "path": {
                    "type": "string",
                    "description": "Path to the file to write (relative or absolute)",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file",
                },
            },
            "required": ["label", "path", "content"],
        },
        execute=execute,
    )
