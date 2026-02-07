"""Bash command execution tool."""

from __future__ import annotations

import os
import secrets
import tempfile
from typing import TYPE_CHECKING, Any

from pi.agent.types import AgentTool, AgentToolResult
from pi.mom.tools.truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    format_size,
    truncate_tail,
)

if TYPE_CHECKING:
    from pi.mom.sandbox import Executor


def _get_temp_file_path() -> str:
    return os.path.join(tempfile.gettempdir(), f"mom-bash-{secrets.token_hex(8)}.log")


def _shell_escape(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def create_bash_tool(executor: Executor, cwd: str) -> AgentTool:
    async def execute(
        tool_call_id: str,
        args: dict[str, Any],
        **kwargs: Any,
    ) -> AgentToolResult:
        command = args["command"]
        timeout = args.get("timeout")

        abort_event = kwargs.get("abort_event")
        result = await executor.exec(
            command, timeout=timeout, abort_event=abort_event
        )

        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            if output:
                output += "\n"
            output += result.stderr

        total_bytes = len(output.encode("utf-8"))

        temp_file_path: str | None = None
        if total_bytes > DEFAULT_MAX_BYTES:
            temp_file_path = _get_temp_file_path()
            with open(temp_file_path, "w", encoding="utf-8") as fh:
                fh.write(output)

        truncation = truncate_tail(output)
        output_text = truncation.content or "(no output)"

        if truncation.truncated:
            start_line = truncation.total_lines - truncation.output_lines + 1
            end_line = truncation.total_lines

            if truncation.last_line_partial:
                last_line = output.split("\n")[-1] if output else ""
                last_line_size = format_size(len(last_line.encode("utf-8")))
                output_text += (
                    f"\n\n[Showing last {format_size(truncation.output_bytes)} "
                    f"of line {end_line} (line is {last_line_size}). "
                    f"Full output: {temp_file_path}]"
                )
            elif truncation.truncated_by == "lines":
                output_text += (
                    f"\n\n[Showing lines {start_line}-{end_line} "
                    f"of {truncation.total_lines}. "
                    f"Full output: {temp_file_path}]"
                )
            else:
                output_text += (
                    f"\n\n[Showing lines {start_line}-{end_line} "
                    f"of {truncation.total_lines} "
                    f"({format_size(DEFAULT_MAX_BYTES)} limit). "
                    f"Full output: {temp_file_path}]"
                )

        if result.code != 0:
            raise RuntimeError(
                f"{output_text}\n\nCommand exited with code {result.code}".strip()
            )

        return AgentToolResult(
            content=[{"type": "text", "text": output_text}],
            details={"truncation": truncation, "fullOutputPath": temp_file_path},
        )

    return AgentTool(
        name="bash",
        label="bash",
        description=(
            f"Execute a bash command in the current working directory. "
            f"Returns stdout and stderr. Output is truncated to last "
            f"{DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB "
            f"(whichever is hit first). If truncated, full output is saved "
            f"to a temp file. Optionally provide a timeout in seconds."
        ),
        parameters={
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "description": "Brief description of what this command does (shown to user)",
                },
                "command": {
                    "type": "string",
                    "description": "Bash command to execute",
                },
                "timeout": {
                    "type": "number",
                    "description": "Timeout in seconds (optional, no default timeout)",
                },
            },
            "required": ["label", "command"],
        },
        execute=execute,
    )
