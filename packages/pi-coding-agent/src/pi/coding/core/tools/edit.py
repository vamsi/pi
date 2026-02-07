"""Edit tool: surgically edit files by finding and replacing exact text."""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pi.agent.types import AgentTool, AgentToolResult, AgentToolUpdateCallback
from pi.ai.types import TextContent

EDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "The file path to edit (relative or absolute)."},
        "old_text": {"type": "string", "description": "The exact text to find and replace."},
        "new_text": {"type": "string", "description": "The replacement text."},
    },
    "required": ["path", "old_text", "new_text"],
}


@dataclass
class EditToolDetails:
    diff: str = ""
    first_changed_line: int = 0


def _resolve_path(path: str, cwd: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = Path(cwd) / p
    return p.resolve()


def _strip_bom(text: str) -> str:
    """Strip UTF-8 BOM if present."""
    if text.startswith("\ufeff"):
        return text[1:]
    return text


async def execute_edit(
    tool_call_id: str,
    params: dict[str, Any],
    cancel_event: Any = None,
    on_update: AgentToolUpdateCallback | None = None,
    *,
    cwd: str = ".",
) -> AgentToolResult:
    """Edit a file by replacing exact text."""
    file_path = _resolve_path(params["path"], cwd)
    old_text = params["old_text"]
    new_text = params["new_text"]

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if old_text == new_text:
        raise ValueError("old_text and new_text are identical")

    content = file_path.read_text(encoding="utf-8", errors="replace")
    content = _strip_bom(content)

    # Normalize line endings
    content = content.replace("\r\n", "\n")
    old_text = old_text.replace("\r\n", "\n")
    new_text = new_text.replace("\r\n", "\n")

    # Find occurrences
    count = content.count(old_text)

    if count == 0:
        raise ValueError(f"Could not find the specified text in {file_path}")

    if count > 1:
        raise ValueError(f"Found {count} occurrences of the text. Provide more context to make the match unique.")

    # Replace
    new_content = content.replace(old_text, new_text, 1)

    # Generate diff
    old_lines = content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    diff = "".join(difflib.unified_diff(old_lines, new_lines, fromfile=str(file_path), tofile=str(file_path)))

    # Find first changed line
    first_changed = 0
    for i, (old_line, new_line) in enumerate(zip(old_lines, new_lines, strict=False)):
        if old_line != new_line:
            first_changed = i + 1
            break

    # Write
    file_path.write_text(new_content, encoding="utf-8")

    return AgentToolResult(
        content=[TextContent(text=diff or "Edit applied successfully.")],
        details=EditToolDetails(diff=diff, first_changed_line=first_changed),
    )


def create_edit_tool(cwd: str) -> AgentTool:
    """Create a file editing tool."""

    async def execute(
        tool_call_id: str,
        params: dict[str, Any],
        cancel_event: Any = None,
        on_update: AgentToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        return await execute_edit(tool_call_id, params, cancel_event, on_update, cwd=cwd)

    return AgentTool(
        name="edit",
        description="Edit a file by finding and replacing exact text. The old_text must match exactly (including whitespace).",
        parameters=EDIT_SCHEMA,
        label="Edit",
        execute=execute,
    )
