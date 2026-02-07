"""Bash tool: execute shell commands with streaming output and truncation."""

from __future__ import annotations

import asyncio
import contextlib
import os
import tempfile
from dataclasses import dataclass
from typing import Any

from pi.agent.types import AgentTool, AgentToolResult, AgentToolUpdateCallback
from pi.ai.types import TextContent
from pi.coding.core.truncate import TruncationResult, truncate_tail

BASH_SCHEMA = {
    "type": "object",
    "properties": {
        "command": {"type": "string", "description": "The bash command to execute."},
        "timeout": {"type": "integer", "description": "Optional timeout in seconds."},
    },
    "required": ["command"],
}


@dataclass
class BashToolDetails:
    exit_code: int = 0
    truncation: TruncationResult | None = None
    full_output_path: str | None = None


async def execute_bash(
    tool_call_id: str,
    params: dict[str, Any],
    cancel_event: asyncio.Event | None = None,
    on_update: AgentToolUpdateCallback | None = None,
    *,
    cwd: str = ".",
) -> AgentToolResult:
    """Execute a bash command and return its output."""
    command = params["command"]
    timeout = params.get("timeout")

    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=cwd,
        env={**os.environ, "TERM": "dumb"},
    )

    output_lines: list[str] = []
    full_output = []

    async def read_output() -> None:
        assert process.stdout is not None
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            decoded = line.decode("utf-8", errors="replace")
            output_lines.append(decoded)
            full_output.append(decoded)

            if on_update and len(output_lines) % 50 == 0:
                on_update(
                    AgentToolResult(
                        content=[TextContent(text="".join(output_lines[-50:]))],
                        details=BashToolDetails(),
                    )
                )

    try:
        if timeout:
            await asyncio.wait_for(read_output(), timeout=timeout)
        else:
            await read_output()

        await process.wait()
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        await process.wait()
        output_lines.append(f"\n[Command timed out after {timeout}s]\n")
    except Exception:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        raise

    raw_output = "".join(output_lines)
    exit_code = process.returncode or 0

    # Truncate output
    truncation = truncate_tail(raw_output)
    details = BashToolDetails(exit_code=exit_code)

    if truncation.truncated:
        details.truncation = truncation
        # Write full output to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, prefix="pi-bash-") as f:
            f.write(raw_output)
            details.full_output_path = f.name

    result_text = truncation.content
    if exit_code != 0:
        result_text += f"\n[Exit code: {exit_code}]"

    return AgentToolResult(
        content=[TextContent(text=result_text)],
        details=details,
    )


def create_bash_tool(cwd: str) -> AgentTool:
    """Create a bash execution tool."""

    async def execute(
        tool_call_id: str,
        params: dict[str, Any],
        cancel_event: asyncio.Event | None = None,
        on_update: AgentToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        return await execute_bash(tool_call_id, params, cancel_event, on_update, cwd=cwd)

    return AgentTool(
        name="bash",
        description="Execute a bash command. Use for running scripts, installing packages, git operations, and other shell tasks.",
        parameters=BASH_SCHEMA,
        label="Bash",
        execute=execute,
    )
