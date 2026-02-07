"""Grep tool: search file contents for patterns using ripgrep."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pi.agent.types import AgentTool, AgentToolResult, AgentToolUpdateCallback
from pi.ai.types import TextContent
from pi.coding.core.truncate import truncate_head, truncate_line

GREP_SCHEMA = {
    "type": "object",
    "properties": {
        "pattern": {"type": "string", "description": "Search pattern (regex or literal)."},
        "path": {"type": "string", "description": "File or directory to search (defaults to current directory)."},
        "glob": {"type": "string", "description": "Filter files by glob pattern (e.g., '*.py')."},
        "ignore_case": {"type": "boolean", "description": "Case-insensitive search."},
        "literal": {"type": "boolean", "description": "Treat pattern as literal string, not regex."},
        "context": {"type": "integer", "description": "Lines of context before and after matches."},
        "limit": {"type": "integer", "description": "Maximum number of matches (default 100)."},
    },
    "required": ["pattern"],
}

MAX_MATCHES = 100


@dataclass
class GrepToolDetails:
    truncation: Any = None
    limit_reached: bool = False
    lines_truncated: bool = False


async def execute_grep(
    tool_call_id: str,
    params: dict[str, Any],
    cancel_event: Any = None,
    on_update: AgentToolUpdateCallback | None = None,
    *,
    cwd: str = ".",
) -> AgentToolResult:
    """Search file contents for a pattern."""
    pattern = params["pattern"]
    search_path = params.get("path", ".")
    file_glob = params.get("glob")
    ignore_case = params.get("ignore_case", False)
    literal = params.get("literal", False)
    context = params.get("context", 0)
    limit = min(params.get("limit", MAX_MATCHES), MAX_MATCHES)

    if not Path(search_path).is_absolute():
        search_path = str(Path(cwd) / search_path)

    # Build ripgrep command
    cmd = ["rg", "--json"]
    if ignore_case:
        cmd.append("-i")
    if literal:
        cmd.append("-F")
    if context > 0:
        cmd.extend(["-C", str(context)])
    if file_glob:
        cmd.extend(["-g", file_glob])
    cmd.extend(["--max-count", str(limit)])
    cmd.append(pattern)
    cmd.append(search_path)

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        # ripgrep not installed, fall back to Python grep
        return _python_grep(pattern, search_path, file_glob, ignore_case, literal, context, limit)
    except subprocess.TimeoutExpired:
        raise TimeoutError("Search timed out after 30 seconds") from None

    # Parse ripgrep JSON output
    matches: list[str] = []
    lines_truncated = False
    limit_reached = False

    for line in proc.stdout.strip().split("\n"):
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        if data.get("type") == "match":
            match_data = data["data"]
            path = match_data.get("path", {}).get("text", "")
            line_num = match_data.get("line_number", 0)
            text = match_data.get("lines", {}).get("text", "").rstrip("\n")

            truncated_text, was_truncated = truncate_line(text)
            if was_truncated:
                lines_truncated = True

            try:
                rel_path = str(Path(path).relative_to(cwd))
            except ValueError:
                rel_path = path

            matches.append(f"{rel_path}:{line_num}: {truncated_text}")

            if len(matches) >= limit:
                limit_reached = True
                break

        elif data.get("type") == "context":
            ctx_data = data["data"]
            path = ctx_data.get("path", {}).get("text", "")
            line_num = ctx_data.get("line_number", 0)
            text = ctx_data.get("lines", {}).get("text", "").rstrip("\n")

            truncated_text, was_truncated = truncate_line(text)
            if was_truncated:
                lines_truncated = True

            try:
                rel_path = str(Path(path).relative_to(cwd))
            except ValueError:
                rel_path = path

            matches.append(f"{rel_path}:{line_num}  {truncated_text}")

    if not matches:
        return AgentToolResult(
            content=[TextContent(text="No matches found.")],
            details=GrepToolDetails(),
        )

    output = "\n".join(matches)
    truncation = truncate_head(output)

    suffix = f"\n[{len(matches)} matches" + (" - limit reached]" if limit_reached else "]")

    return AgentToolResult(
        content=[TextContent(text=truncation.content + suffix)],
        details=GrepToolDetails(
            truncation=truncation if truncation.truncated else None,
            limit_reached=limit_reached,
            lines_truncated=lines_truncated,
        ),
    )


def _python_grep(
    pattern: str,
    search_path: str,
    file_glob: str | None,
    ignore_case: bool,
    literal: bool,
    context: int,
    limit: int,
) -> AgentToolResult:
    """Fallback grep using Python's re module."""
    import re

    flags = re.IGNORECASE if ignore_case else 0
    compiled = re.compile(re.escape(pattern), flags) if literal else re.compile(pattern, flags)

    search = Path(search_path)
    matches: list[str] = []

    if search.is_file():
        files = [search]
    else:
        glob_pattern = file_glob or "**/*"
        files = [f for f in search.rglob(glob_pattern) if f.is_file()]

    for file in files:
        if len(matches) >= limit:
            break
        try:
            lines = file.read_text(encoding="utf-8", errors="replace").split("\n")
        except PermissionError, OSError:
            continue

        for i, line in enumerate(lines):
            if compiled.search(line):
                truncated, _ = truncate_line(line)
                try:
                    rel = str(file.relative_to(search_path))
                except ValueError:
                    rel = str(file)
                matches.append(f"{rel}:{i + 1}: {truncated}")
                if len(matches) >= limit:
                    break

    if not matches:
        return AgentToolResult(
            content=[TextContent(text="No matches found.")],
            details=GrepToolDetails(),
        )

    return AgentToolResult(
        content=[TextContent(text="\n".join(matches))],
        details=GrepToolDetails(limit_reached=len(matches) >= limit),
    )


def create_grep_tool(cwd: str) -> AgentTool:
    """Create a content search tool."""

    async def execute(
        tool_call_id: str,
        params: dict[str, Any],
        cancel_event: Any = None,
        on_update: AgentToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        return await execute_grep(tool_call_id, params, cancel_event, on_update, cwd=cwd)

    return AgentTool(
        name="grep",
        description="Search file contents for a pattern. Uses ripgrep if available, falls back to Python regex.",
        parameters=GREP_SCHEMA,
        label="Grep",
        execute=execute,
    )
