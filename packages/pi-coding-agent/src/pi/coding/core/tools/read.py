"""Read tool: read files with smart type detection and truncation."""

from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pi.agent.types import AgentTool, AgentToolResult, AgentToolUpdateCallback
from pi.ai.types import ImageContent, TextContent
from pi.coding.core.truncate import TruncationResult, truncate_head

READ_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "The file path to read (relative or absolute)."},
        "offset": {"type": "integer", "description": "Line number to start reading from (1-indexed)."},
        "limit": {"type": "integer", "description": "Maximum number of lines to read."},
    },
    "required": ["path"],
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg"}


@dataclass
class ReadToolDetails:
    truncation: TruncationResult | None = None


def _resolve_path(path: str, cwd: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = Path(cwd) / p
    return p.resolve()


async def execute_read(
    tool_call_id: str,
    params: dict[str, Any],
    cancel_event: Any = None,
    on_update: AgentToolUpdateCallback | None = None,
    *,
    cwd: str = ".",
) -> AgentToolResult:
    """Read a file and return its contents."""
    file_path = _resolve_path(params["path"], cwd)
    offset = params.get("offset")
    limit = params.get("limit")

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"Not a file: {file_path}")

    # Check for image files
    suffix = file_path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        data = file_path.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        mime_type = mimetypes.guess_type(str(file_path))[0] or "image/png"
        return AgentToolResult(
            content=[ImageContent(data=b64, mime_type=mime_type)],
            details=ReadToolDetails(),
        )

    # Read text file
    text = file_path.read_text(encoding="utf-8", errors="replace")
    lines = text.split("\n")

    # Apply offset
    if offset is not None and offset > 1:
        start = offset - 1  # Convert to 0-indexed
        lines = lines[start:]
        text = "\n".join(lines)

    # Apply limit
    if limit is not None:
        lines = lines[:limit]
        text = "\n".join(lines)

    # Add line numbers (cat -n format)
    numbered_lines = []
    start_line = offset or 1
    for i, line in enumerate(lines):
        line_num = start_line + i
        numbered_lines.append(f"{line_num:>6}\t{line}")
    text = "\n".join(numbered_lines)

    # Truncate
    truncation = truncate_head(text)
    details = ReadToolDetails()
    if truncation.truncated:
        details.truncation = truncation

    return AgentToolResult(
        content=[TextContent(text=truncation.content)],
        details=details,
    )


def create_read_tool(cwd: str) -> AgentTool:
    """Create a file reading tool."""

    async def execute(
        tool_call_id: str,
        params: dict[str, Any],
        cancel_event: Any = None,
        on_update: AgentToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        return await execute_read(tool_call_id, params, cancel_event, on_update, cwd=cwd)

    return AgentTool(
        name="read",
        description="Read the contents of a file. Returns text with line numbers, or base64 image data for image files.",
        parameters=READ_SCHEMA,
        label="Read",
        execute=execute,
    )
