"""Find tool: search for files by glob pattern."""

from __future__ import annotations

import glob as globmod
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pi.agent.types import AgentTool, AgentToolResult, AgentToolUpdateCallback
from pi.ai.types import TextContent
from pi.coding.core.truncate import DEFAULT_MAX_BYTES, TruncationResult, truncate_head

FIND_SCHEMA = {
    "type": "object",
    "properties": {
        "pattern": {"type": "string", "description": "Glob pattern to match (e.g., '*.py', '**/*.json')."},
        "path": {"type": "string", "description": "Directory to search in (defaults to current directory)."},
        "limit": {"type": "integer", "description": "Maximum number of results (default 1000)."},
    },
    "required": ["pattern"],
}

MAX_RESULTS = 1000


@dataclass
class FindToolDetails:
    truncation: TruncationResult | None = None
    limit_reached: bool = False


def _resolve_path(path: str, cwd: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = Path(cwd) / p
    return p.resolve()


async def execute_find(
    tool_call_id: str,
    params: dict[str, Any],
    cancel_event: Any = None,
    on_update: AgentToolUpdateCallback | None = None,
    *,
    cwd: str = ".",
) -> AgentToolResult:
    """Find files matching a glob pattern."""
    pattern = params["pattern"]
    search_path = _resolve_path(params.get("path", "."), cwd)
    limit = min(params.get("limit", MAX_RESULTS), MAX_RESULTS)

    if not search_path.exists():
        raise FileNotFoundError(f"Directory not found: {search_path}")

    if not search_path.is_dir():
        raise ValueError(f"Not a directory: {search_path}")

    # Try using fd for speed, fall back to glob
    results: list[str] = []
    limit_reached = False

    try:
        proc = subprocess.run(
            ["fd", "--glob", pattern, "--max-results", str(limit + 1), str(search_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode == 0:
            for line in proc.stdout.strip().split("\n"):
                if line:
                    try:
                        rel = str(Path(line).relative_to(search_path))
                    except ValueError:
                        rel = line
                    results.append(rel)
    except FileNotFoundError, subprocess.TimeoutExpired:
        # fd not available or timed out, use glob
        full_pattern = str(search_path / pattern)
        for match in globmod.iglob(full_pattern, recursive=True):
            p = Path(match)
            if p.name.startswith(".git") or "node_modules" in p.parts:
                continue
            try:
                rel = str(p.relative_to(search_path))
            except ValueError:
                rel = match
            results.append(rel)
            if len(results) >= limit + 1:
                break

    if len(results) > limit:
        results = results[:limit]
        limit_reached = True

    results.sort()
    output = "\n".join(results)

    truncation = truncate_head(output, max_lines=limit, max_bytes=DEFAULT_MAX_BYTES)
    details = FindToolDetails(limit_reached=limit_reached)
    if truncation.truncated:
        details.truncation = truncation

    suffix = f"\n[{len(results)} results" + (" - limit reached]" if limit_reached else "]")

    return AgentToolResult(
        content=[TextContent(text=truncation.content + suffix)],
        details=details,
    )


def create_find_tool(cwd: str) -> AgentTool:
    """Create a file finding tool."""

    async def execute(
        tool_call_id: str,
        params: dict[str, Any],
        cancel_event: Any = None,
        on_update: AgentToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        return await execute_find(tool_call_id, params, cancel_event, on_update, cwd=cwd)

    return AgentTool(
        name="find",
        description="Find files matching a glob pattern. Respects .gitignore. Returns relative paths.",
        parameters=FIND_SCHEMA,
        label="Find",
        execute=execute,
    )
