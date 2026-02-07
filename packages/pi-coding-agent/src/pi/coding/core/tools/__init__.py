"""Coding agent tools: bash, read, write, edit, find, grep, ls."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pi.agent.types import AgentTool

from pi.coding.core.tools.bash import create_bash_tool
from pi.coding.core.tools.edit import create_edit_tool
from pi.coding.core.tools.find import create_find_tool
from pi.coding.core.tools.grep import create_grep_tool
from pi.coding.core.tools.ls import create_ls_tool
from pi.coding.core.tools.read import create_read_tool
from pi.coding.core.tools.write import create_write_tool


def create_coding_tools(cwd: str) -> list[AgentTool]:
    """Create the standard set of coding tools (read, bash, edit, write)."""
    return [
        create_read_tool(cwd),
        create_bash_tool(cwd),
        create_edit_tool(cwd),
        create_write_tool(cwd),
    ]


def create_read_only_tools(cwd: str) -> list[AgentTool]:
    """Create read-only exploration tools (read, grep, find, ls)."""
    return [
        create_read_tool(cwd),
        create_grep_tool(cwd),
        create_find_tool(cwd),
        create_ls_tool(cwd),
    ]


def create_all_tools(cwd: str) -> dict[str, AgentTool]:
    """Create all tools as a dictionary keyed by name."""
    tools = [
        create_read_tool(cwd),
        create_bash_tool(cwd),
        create_edit_tool(cwd),
        create_write_tool(cwd),
        create_grep_tool(cwd),
        create_find_tool(cwd),
        create_ls_tool(cwd),
    ]
    return {t.name: t for t in tools}


__all__ = [
    "create_all_tools",
    "create_bash_tool",
    "create_coding_tools",
    "create_edit_tool",
    "create_find_tool",
    "create_grep_tool",
    "create_ls_tool",
    "create_read_only_tools",
    "create_read_tool",
    "create_write_tool",
]
