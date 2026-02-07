"""File reading tool with image support."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from pi.agent.types import AgentTool, AgentToolResult
from pi.mom.tools.truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    format_size,
    truncate_head,
)

if TYPE_CHECKING:
    from pi.mom.sandbox import Executor

IMAGE_MIME_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def _is_image_file(file_path: str) -> str | None:
    ext = os.path.splitext(file_path)[1].lower()
    return IMAGE_MIME_TYPES.get(ext)


def _shell_escape(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def create_read_tool(executor: Executor, cwd: str) -> AgentTool:
    async def execute(
        tool_call_id: str,
        args: dict[str, Any],
        **kwargs: Any,
    ) -> AgentToolResult:
        path = args["path"]
        offset = args.get("offset")
        limit = args.get("limit")
        abort_event = kwargs.get("abort_event")

        mime_type = _is_image_file(path)

        if mime_type:
            result = await executor.exec(
                f"base64 < {_shell_escape(path)}", abort_event=abort_event
            )
            if result.code != 0:
                raise RuntimeError(result.stderr or f"Failed to read file: {path}")
            base64_data = result.stdout.replace(" ", "").replace("\n", "").replace("\r", "")
            return AgentToolResult(
                content=[
                    {"type": "text", "text": f"Read image file [{mime_type}]"},
                    {"type": "image", "data": base64_data, "mimeType": mime_type},
                ],
            )

        # Get total line count
        count_result = await executor.exec(
            f"wc -l < {_shell_escape(path)}", abort_event=abort_event
        )
        if count_result.code != 0:
            raise RuntimeError(count_result.stderr or f"Failed to read file: {path}")
        total_file_lines = int(count_result.stdout.strip()) + 1

        start_line = max(1, offset) if offset else 1

        if start_line > total_file_lines:
            raise RuntimeError(
                f"Offset {offset} is beyond end of file ({total_file_lines} lines total)"
            )

        if start_line == 1:
            cmd = f"cat {_shell_escape(path)}"
        else:
            cmd = f"tail -n +{start_line} {_shell_escape(path)}"

        result = await executor.exec(cmd, abort_event=abort_event)
        if result.code != 0:
            raise RuntimeError(result.stderr or f"Failed to read file: {path}")

        selected_content = result.stdout
        user_limited_lines: int | None = None

        if limit is not None:
            lines = selected_content.split("\n")
            end_line = min(limit, len(lines))
            selected_content = "\n".join(lines[:end_line])
            user_limited_lines = end_line

        truncation = truncate_head(selected_content)

        if truncation.first_line_exceeds_limit:
            first_line = selected_content.split("\n")[0]
            first_line_size = format_size(len(first_line.encode("utf-8")))
            output_text = (
                f"[Line {start_line} is {first_line_size}, exceeds "
                f"{format_size(DEFAULT_MAX_BYTES)} limit. Use bash: "
                f"sed -n '{start_line}p' {path} | head -c {DEFAULT_MAX_BYTES}]"
            )
        elif truncation.truncated:
            end_line_display = start_line + truncation.output_lines - 1
            next_offset = end_line_display + 1
            output_text = truncation.content

            if truncation.truncated_by == "lines":
                output_text += (
                    f"\n\n[Showing lines {start_line}-{end_line_display} "
                    f"of {total_file_lines}. Use offset={next_offset} to continue]"
                )
            else:
                output_text += (
                    f"\n\n[Showing lines {start_line}-{end_line_display} "
                    f"of {total_file_lines} "
                    f"({format_size(DEFAULT_MAX_BYTES)} limit). "
                    f"Use offset={next_offset} to continue]"
                )
        elif user_limited_lines is not None:
            lines_from_start = start_line - 1 + user_limited_lines
            if lines_from_start < total_file_lines:
                remaining = total_file_lines - lines_from_start
                next_offset = start_line + user_limited_lines
                output_text = truncation.content
                output_text += (
                    f"\n\n[{remaining} more lines in file. "
                    f"Use offset={next_offset} to continue]"
                )
            else:
                output_text = truncation.content
        else:
            output_text = truncation.content

        return AgentToolResult(
            content=[{"type": "text", "text": output_text}],
            details={"truncation": truncation} if truncation.truncated else None,
        )

    return AgentTool(
        name="read",
        label="read",
        description=(
            f"Read the contents of a file. Supports text files and images "
            f"(jpg, png, gif, webp). Images are sent as attachments. "
            f"For text files, output is truncated to {DEFAULT_MAX_LINES} lines "
            f"or {DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first). "
            f"Use offset/limit for large files."
        ),
        parameters={
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "description": "Brief description of what you're reading and why (shown to user)",
                },
                "path": {
                    "type": "string",
                    "description": "Path to the file to read (relative or absolute)",
                },
                "offset": {
                    "type": "number",
                    "description": "Line number to start reading from (1-indexed)",
                },
                "limit": {
                    "type": "number",
                    "description": "Maximum number of lines to read",
                },
            },
            "required": ["label", "path"],
        },
        execute=execute,
    )
