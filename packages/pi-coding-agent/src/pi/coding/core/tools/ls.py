"""Ls tool: list directory contents with type indicators."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pi.agent.types import AgentTool, AgentToolResult, AgentToolUpdateCallback
from pi.ai.types import TextContent
from pi.coding.core.truncate import DEFAULT_MAX_BYTES, truncate_head

LS_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Directory to list (defaults to current directory)."},
        "limit": {"type": "integer", "description": "Maximum number of entries (default 500)."},
    },
}

MAX_ENTRIES = 500


@dataclass
class LsToolDetails:
    truncation: Any = None
    limit_reached: bool = False


def _resolve_path(path: str, cwd: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = Path(cwd) / p
    return p.resolve()


async def execute_ls(
    tool_call_id: str,
    params: dict[str, Any],
    cancel_event: Any = None,
    on_update: AgentToolUpdateCallback | None = None,
    *,
    cwd: str = ".",
) -> AgentToolResult:
    """List directory contents."""
    dir_path = _resolve_path(params.get("path", "."), cwd)
    limit = min(params.get("limit", MAX_ENTRIES), MAX_ENTRIES)

    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    if not dir_path.is_dir():
        raise ValueError(f"Not a directory: {dir_path}")

    entries: list[str] = []
    limit_reached = False

    try:
        items = sorted(dir_path.iterdir(), key=lambda p: p.name.lower())
    except PermissionError:
        raise PermissionError(f"Permission denied: {dir_path}") from None

    for item in items:
        if len(entries) >= limit:
            limit_reached = True
            break
        try:
            name = item.name
            if item.is_dir():
                name += "/"
            entries.append(name)
        except OSError:
            continue

    output = "\n".join(entries)
    truncation = truncate_head(output, max_lines=limit, max_bytes=DEFAULT_MAX_BYTES)

    return AgentToolResult(
        content=[TextContent(text=truncation.content)],
        details=LsToolDetails(
            truncation=truncation if truncation.truncated else None,
            limit_reached=limit_reached,
        ),
    )


def create_ls_tool(cwd: str) -> AgentTool:
    """Create a directory listing tool."""

    async def execute(
        tool_call_id: str,
        params: dict[str, Any],
        cancel_event: Any = None,
        on_update: AgentToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        return await execute_ls(tool_call_id, params, cancel_event, on_update, cwd=cwd)

    return AgentTool(
        name="ls",
        description="List directory contents with type indicators (/ for directories). Sorted alphabetically.",
        parameters=LS_SCHEMA,
        label="List",
        execute=execute,
    )
