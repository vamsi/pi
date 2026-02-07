"""System prompt construction and project context loading.

Builds the system prompt from available tools, dynamic guidelines,
optional custom prompts, context files, and environment info.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime

# Tool descriptions for system prompt
TOOL_DESCRIPTIONS: dict[str, str] = {
    "read": "Read file contents",
    "bash": "Execute bash commands (ls, grep, find, etc.)",
    "edit": "Make surgical edits to files (find exact text and replace)",
    "write": "Create or overwrite files",
    "grep": "Search file contents for patterns (respects .gitignore)",
    "find": "Find files by glob pattern (respects .gitignore)",
    "ls": "List directory contents",
}

DEFAULT_TOOLS = ["read", "bash", "edit", "write"]


@dataclass
class ContextFile:
    """A project context file (e.g., AGENTS.md) to include in the prompt."""

    path: str
    content: str


def build_system_prompt(
    *,
    selected_tools: list[str] | None = None,
    custom_prompt: str | None = None,
    append_system_prompt: str | None = None,
    cwd: str | None = None,
    context_files: list[ContextFile] | None = None,
) -> str:
    """Build the system prompt with tools, guidelines, and context.

    Args:
        selected_tools: Which tools to include in prompt. Default: [read, bash, edit, write].
        custom_prompt: Custom system prompt (replaces default).
        append_system_prompt: Text to append to system prompt.
        cwd: Working directory. Default: os.getcwd().
        context_files: Pre-loaded project context files.

    Returns:
        The complete system prompt string.
    """
    resolved_cwd = cwd or os.getcwd()

    now = datetime.now(UTC).astimezone()
    date_time = now.strftime("%A, %B %d, %Y %I:%M:%S %p %Z")

    append_section = f"\n\n{append_system_prompt}" if append_system_prompt else ""

    files = context_files or []

    if custom_prompt:
        return _build_custom_prompt(
            custom_prompt=custom_prompt,
            append_section=append_section,
            context_files=files,
            selected_tools=selected_tools,
            date_time=date_time,
            cwd=resolved_cwd,
        )

    return _build_default_prompt(
        selected_tools=selected_tools,
        append_section=append_section,
        context_files=files,
        date_time=date_time,
        cwd=resolved_cwd,
    )


def _build_custom_prompt(
    *,
    custom_prompt: str,
    append_section: str,
    context_files: list[ContextFile],
    selected_tools: list[str] | None,
    date_time: str,
    cwd: str,
) -> str:
    """Build a prompt when a custom prompt is provided."""
    prompt = custom_prompt

    if append_section:
        prompt += append_section

    if context_files:
        prompt += "\n\n# Project Context\n\n"
        prompt += "Project-specific instructions and guidelines:\n\n"
        for cf in context_files:
            prompt += f"## {cf.path}\n\n{cf.content}\n\n"

    prompt += f"\nCurrent date and time: {date_time}"
    prompt += f"\nCurrent working directory: {cwd}"

    return prompt


def _build_guidelines(tools: list[str]) -> str:
    """Build dynamic guidelines based on available tools."""
    guidelines: list[str] = []

    has_bash = "bash" in tools
    has_edit = "edit" in tools
    has_write = "write" in tools
    has_grep = "grep" in tools
    has_find = "find" in tools
    has_ls = "ls" in tools
    has_read = "read" in tools

    # File exploration guidelines
    if has_bash and not has_grep and not has_find and not has_ls:
        guidelines.append("Use bash for file operations like ls, rg, find")
    elif has_bash and (has_grep or has_find or has_ls):
        guidelines.append("Prefer grep/find/ls tools over bash for file exploration (faster, respects .gitignore)")

    # Read before edit
    if has_read and has_edit:
        guidelines.append("Use read to examine files before editing. You must use this tool instead of cat or sed.")

    # Edit guideline
    if has_edit:
        guidelines.append("Use edit for precise changes (old text must match exactly)")

    # Write guideline
    if has_write:
        guidelines.append("Use write only for new files or complete rewrites")

    # Output guideline
    if has_edit or has_write:
        guidelines.append(
            "When summarizing your actions, output plain text directly - do NOT use cat or bash to display what you did"
        )

    # Always include
    guidelines.append("Be concise in your responses")
    guidelines.append("Show file paths clearly when working with files")

    return "\n".join(f"- {g}" for g in guidelines)


def _build_default_prompt(
    *,
    selected_tools: list[str] | None,
    append_section: str,
    context_files: list[ContextFile],
    date_time: str,
    cwd: str,
) -> str:
    """Build the default system prompt."""
    # Filter to known built-in tools
    tools = [t for t in (DEFAULT_TOOLS if selected_tools is None else selected_tools) if t in TOOL_DESCRIPTIONS]
    tools_list = "\n".join(f"- {t}: {TOOL_DESCRIPTIONS[t]}" for t in tools) if tools else "(none)"

    guidelines = _build_guidelines(tools)

    prompt = f"""You are an expert coding assistant operating inside pi, a coding agent harness. You help users by reading files, executing commands, editing code, and writing new files.

Available tools:
{tools_list}

In addition to the tools above, you may have access to other custom tools depending on the project.

Guidelines:
{guidelines}"""

    if append_section:
        prompt += append_section

    # Append project context files
    if context_files:
        prompt += "\n\n# Project Context\n\n"
        prompt += "Project-specific instructions and guidelines:\n\n"
        for cf in context_files:
            prompt += f"## {cf.path}\n\n{cf.content}\n\n"

    # Add date/time and working directory last
    prompt += f"\nCurrent date and time: {date_time}"
    prompt += f"\nCurrent working directory: {cwd}"

    return prompt
