"""Mom tool factory – aggregates all mom-specific tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Coroutine, Any

if TYPE_CHECKING:
    from pi.agent.types import AgentTool
    from pi.mom.sandbox import Executor

# Module-level upload function set by agent runner at runtime.
_upload_fn: Callable[[str, str | None], Coroutine[Any, Any, None]] | None = None


def set_upload_function(
    fn: Callable[[str, str | None], Coroutine[Any, Any, None]],
) -> None:
    global _upload_fn
    _upload_fn = fn


def get_upload_function() -> Callable[[str, str | None], Coroutine[Any, Any, None]] | None:
    return _upload_fn


def create_mom_tools(executor: Executor, cwd: str) -> list[AgentTool]:
    from pi.mom.tools.bash import create_bash_tool
    from pi.mom.tools.read import create_read_tool
    from pi.mom.tools.write import create_write_tool
    from pi.mom.tools.edit import create_edit_tool
    from pi.mom.tools.attach import create_attach_tool

    return [
        create_bash_tool(executor, cwd),
        create_read_tool(executor, cwd),
        create_write_tool(executor, cwd),
        create_edit_tool(executor, cwd),
        create_attach_tool(),
    ]
