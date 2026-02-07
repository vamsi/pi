"""Write tool: write content to files with automatic directory creation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pi.agent.types import AgentTool, AgentToolResult, AgentToolUpdateCallback
from pi.ai.types import TextContent

WRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "The file path to write to (relative or absolute)."},
        "content": {"type": "string", "description": "The content to write to the file."},
    },
    "required": ["path", "content"],
}


def _resolve_path(path: str, cwd: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = Path(cwd) / p
    return p.resolve()


async def execute_write(
    tool_call_id: str,
    params: dict[str, Any],
    cancel_event: Any = None,
    on_update: AgentToolUpdateCallback | None = None,
    *,
    cwd: str = ".",
) -> AgentToolResult:
    """Write content to a file."""
    file_path = _resolve_path(params["path"], cwd)
    content = params["content"]

    # Create parent directories
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Write file
    file_path.write_text(content, encoding="utf-8")
    bytes_written = len(content.encode("utf-8"))

    return AgentToolResult(
        content=[TextContent(text=f"Successfully wrote {bytes_written} bytes to {file_path}")],
        details={"bytes_written": bytes_written},
    )


def create_write_tool(cwd: str) -> AgentTool:
    """Create a file writing tool."""

    async def execute(
        tool_call_id: str,
        params: dict[str, Any],
        cancel_event: Any = None,
        on_update: AgentToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        return await execute_write(tool_call_id, params, cancel_event, on_update, cwd=cwd)

    return AgentTool(
        name="write",
        description="Write content to a file. Creates parent directories automatically. Overwrites existing files.",
        parameters=WRITE_SCHEMA,
        label="Write",
        execute=execute,
    )
