"""File editing tool with unified diff."""

from __future__ import annotations

import difflib
from typing import TYPE_CHECKING, Any

from pi.agent.types import AgentTool, AgentToolResult

if TYPE_CHECKING:
    from pi.mom.sandbox import Executor


def _shell_escape(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def _generate_diff_string(
    old_content: str, new_content: str, context_lines: int = 4
) -> str:
    """Generate a unified diff string with line numbers and context."""
    old_lines = old_content.split("\n")
    new_lines = new_content.split("\n")

    sm = difflib.SequenceMatcher(None, old_lines, new_lines)
    max_line_num = max(len(old_lines), len(new_lines))
    line_num_width = len(str(max_line_num))

    output: list[str] = []
    old_line_num = 1
    new_line_num = 1

    opcodes = sm.get_opcodes()
    for idx, (tag, i1, i2, j1, j2) in enumerate(opcodes):
        if tag == "equal":
            chunk = old_lines[i1:i2]
            # Show context around changes
            prev_is_change = idx > 0 and opcodes[idx - 1][0] != "equal"
            next_is_change = (
                idx < len(opcodes) - 1 and opcodes[idx + 1][0] != "equal"
            )

            if prev_is_change or next_is_change:
                lines_to_show = chunk
                skip_start = 0
                skip_end = 0

                if not prev_is_change:
                    skip_start = max(0, len(chunk) - context_lines)
                    lines_to_show = chunk[skip_start:]

                if not next_is_change and len(lines_to_show) > context_lines:
                    skip_end = len(lines_to_show) - context_lines
                    lines_to_show = lines_to_show[:context_lines]

                if skip_start > 0:
                    output.append(f" {''.ljust(line_num_width)} ...")

                for line in lines_to_show:
                    ln = str(old_line_num + skip_start).rjust(line_num_width)
                    output.append(f" {ln} {line}")
                    old_line_num += 1
                    new_line_num += 1

                if skip_end > 0:
                    output.append(f" {''.ljust(line_num_width)} ...")

                old_line_num += skip_start + skip_end - len(lines_to_show) + len(chunk) - skip_start
                new_line_num += skip_start + skip_end - len(lines_to_show) + len(chunk) - skip_start
            else:
                old_line_num += len(chunk)
                new_line_num += len(chunk)
        elif tag == "replace":
            for line in old_lines[i1:i2]:
                ln = str(old_line_num).rjust(line_num_width)
                output.append(f"-{ln} {line}")
                old_line_num += 1
            for line in new_lines[j1:j2]:
                ln = str(new_line_num).rjust(line_num_width)
                output.append(f"+{ln} {line}")
                new_line_num += 1
        elif tag == "delete":
            for line in old_lines[i1:i2]:
                ln = str(old_line_num).rjust(line_num_width)
                output.append(f"-{ln} {line}")
                old_line_num += 1
        elif tag == "insert":
            for line in new_lines[j1:j2]:
                ln = str(new_line_num).rjust(line_num_width)
                output.append(f"+{ln} {line}")
                new_line_num += 1

    return "\n".join(output)


def create_edit_tool(executor: Executor, cwd: str) -> AgentTool:
    async def execute(
        tool_call_id: str,
        args: dict[str, Any],
        **kwargs: Any,
    ) -> AgentToolResult:
        path = args["path"]
        old_text = args["oldText"]
        new_text = args["newText"]
        abort_event = kwargs.get("abort_event")

        # Read the file
        read_result = await executor.exec(
            f"cat {_shell_escape(path)}", abort_event=abort_event
        )
        if read_result.code != 0:
            raise RuntimeError(read_result.stderr or f"File not found: {path}")

        content = read_result.stdout

        if old_text not in content:
            raise RuntimeError(
                f"Could not find the exact text in {path}. "
                "The old text must match exactly including all whitespace and newlines."
            )

        occurrences = content.count(old_text)
        if occurrences > 1:
            raise RuntimeError(
                f"Found {occurrences} occurrences of the text in {path}. "
                "The text must be unique. Please provide more context to make it unique."
            )

        idx = content.index(old_text)
        new_content = content[:idx] + new_text + content[idx + len(old_text) :]

        if content == new_content:
            raise RuntimeError(
                f"No changes made to {path}. The replacement produced "
                "identical content."
            )

        write_result = await executor.exec(
            f"printf '%s' {_shell_escape(new_content)} > {_shell_escape(path)}",
            abort_event=abort_event,
        )
        if write_result.code != 0:
            raise RuntimeError(write_result.stderr or f"Failed to write file: {path}")

        return AgentToolResult(
            content=[
                {
                    "type": "text",
                    "text": (
                        f"Successfully replaced text in {path}. "
                        f"Changed {len(old_text)} characters to {len(new_text)} characters."
                    ),
                }
            ],
            details={"diff": _generate_diff_string(content, new_content)},
        )

    return AgentTool(
        name="edit",
        label="edit",
        description=(
            "Edit a file by replacing exact text. The oldText must match "
            "exactly (including whitespace). Use this for precise, surgical edits."
        ),
        parameters={
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "description": "Brief description of the edit you're making (shown to user)",
                },
                "path": {
                    "type": "string",
                    "description": "Path to the file to edit (relative or absolute)",
                },
                "oldText": {
                    "type": "string",
                    "description": "Exact text to find and replace (must match exactly)",
                },
                "newText": {
                    "type": "string",
                    "description": "New text to replace the old text with",
                },
            },
            "required": ["label", "path", "oldText", "newText"],
        },
        execute=execute,
    )
