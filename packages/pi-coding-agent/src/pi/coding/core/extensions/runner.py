"""Extension runner: dispatches events to registered handlers.

Manages the lifecycle of loaded extensions and provides methods
for emitting events, querying registered tools/commands, and
error handling.
"""

from __future__ import annotations

import contextlib
import inspect
import traceback
from copy import deepcopy
from typing import Any

from pi.coding.core.extensions.types import (
    BeforeAgentStartEvent,
    ContextEvent,
    Extension,
    ExtensionContext,
    ExtensionError,
    ExtensionEvent,
    InputEvent,
    RegisteredCommand,
    RegisteredTool,
    ToolCallEvent,
    ToolResultEvent,
)


class ExtensionRunner:
    """Dispatches events to extension handlers and manages extension state.

    Thread-safe event emission with error isolation - errors in one
    extension handler do not affect others.
    """

    def __init__(
        self,
        extensions: list[Extension],
        cwd: str,
    ) -> None:
        self._extensions = list(extensions)
        self._cwd = cwd
        self._error_listeners: list[Any] = []
        self._context = ExtensionContext(cwd=cwd)

    @property
    def extensions(self) -> list[Extension]:
        return list(self._extensions)

    @property
    def context(self) -> ExtensionContext:
        return self._context

    def set_context(self, **kwargs: Any) -> None:
        """Update context fields."""
        for key, value in kwargs.items():
            if hasattr(self._context, key):
                setattr(self._context, key, value)

    # --- Event emission ---

    async def emit(self, event: ExtensionEvent) -> None:
        """Emit an event to all registered handlers."""
        for ext in self._extensions:
            handlers = ext.handlers.get(event.type, [])
            for handler in handlers:
                try:
                    result = handler(event, self._context)
                    if inspect.isawaitable(result):
                        await result
                except Exception as e:
                    self._emit_error(
                        ExtensionError(
                            extension_path=ext.path,
                            event=event.type,
                            error=str(e),
                            stack=traceback.format_exc(),
                        )
                    )

    async def emit_tool_call(self, event: ToolCallEvent) -> ToolCallEvent:
        """Emit a tool_call event. Handlers can block execution."""
        for ext in self._extensions:
            handlers = ext.handlers.get("tool_call", [])
            for handler in handlers:
                try:
                    result = handler(event, self._context)
                    if inspect.isawaitable(result):
                        await result
                    if event.blocked:
                        return event
                except Exception as e:
                    self._emit_error(
                        ExtensionError(
                            extension_path=ext.path,
                            event="tool_call",
                            error=str(e),
                            stack=traceback.format_exc(),
                        )
                    )
        return event

    async def emit_tool_result(self, event: ToolResultEvent) -> ToolResultEvent:
        """Emit a tool_result event. Handlers can modify the result."""
        for ext in self._extensions:
            handlers = ext.handlers.get("tool_result", [])
            for handler in handlers:
                try:
                    result = handler(event, self._context)
                    if inspect.isawaitable(result):
                        await result
                except Exception as e:
                    self._emit_error(
                        ExtensionError(
                            extension_path=ext.path,
                            event="tool_result",
                            error=str(e),
                            stack=traceback.format_exc(),
                        )
                    )
        return event

    async def emit_context(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Emit a context event before LLM call. Handlers can modify messages."""
        event = ContextEvent(messages=deepcopy(messages))
        for ext in self._extensions:
            handlers = ext.handlers.get("context", [])
            for handler in handlers:
                try:
                    result = handler(event, self._context)
                    if inspect.isawaitable(result):
                        await result
                except Exception as e:
                    self._emit_error(
                        ExtensionError(
                            extension_path=ext.path,
                            event="context",
                            error=str(e),
                            stack=traceback.format_exc(),
                        )
                    )
        return event.messages

    async def emit_before_agent_start(
        self,
        prompt: str,
        system_prompt: str,
    ) -> tuple[str, str]:
        """Emit before_agent_start. Handlers can modify system prompt."""
        event = BeforeAgentStartEvent(prompt=prompt, system_prompt=system_prompt)
        for ext in self._extensions:
            handlers = ext.handlers.get("before_agent_start", [])
            for handler in handlers:
                try:
                    result = handler(event, self._context)
                    if inspect.isawaitable(result):
                        await result
                except Exception as e:
                    self._emit_error(
                        ExtensionError(
                            extension_path=ext.path,
                            event="before_agent_start",
                            error=str(e),
                            stack=traceback.format_exc(),
                        )
                    )
        return event.prompt, event.system_prompt

    async def emit_input(self, text: str) -> str:
        """Emit an input event. Handlers can transform text."""
        event = InputEvent(text=text)
        for ext in self._extensions:
            handlers = ext.handlers.get("input", [])
            for handler in handlers:
                try:
                    result = handler(event, self._context)
                    if inspect.isawaitable(result):
                        await result
                except Exception as e:
                    self._emit_error(
                        ExtensionError(
                            extension_path=ext.path,
                            event="input",
                            error=str(e),
                            stack=traceback.format_exc(),
                        )
                    )
        return event.transformed_text if event.transformed_text is not None else event.text

    # --- Query methods ---

    def has_handlers(self, event_type: str) -> bool:
        """Check if any extensions have handlers for an event type."""
        return any(event_type in ext.handlers and ext.handlers[event_type] for ext in self._extensions)

    def get_all_registered_tools(self) -> dict[str, RegisteredTool]:
        """Get all custom tools registered by extensions."""
        tools: dict[str, RegisteredTool] = {}
        for ext in self._extensions:
            tools.update(ext.tools)
        return tools

    def get_tool_definition(self, name: str) -> RegisteredTool | None:
        """Get a specific tool by name."""
        for ext in self._extensions:
            if name in ext.tools:
                return ext.tools[name]
        return None

    def get_registered_commands(self, reserved: set[str] | None = None) -> list[RegisteredCommand]:
        """Get all registered commands, optionally excluding reserved names."""
        reserved = reserved or set()
        commands: list[RegisteredCommand] = []
        for ext in self._extensions:
            for name, cmd in ext.commands.items():
                if name not in reserved:
                    commands.append(cmd)
        return commands

    def get_command(self, name: str) -> RegisteredCommand | None:
        """Get a command by name or alias."""
        for ext in self._extensions:
            if name in ext.commands:
                return ext.commands[name]
            for cmd in ext.commands.values():
                if name in cmd.aliases:
                    return cmd
        return None

    def get_flags(self) -> dict[str, Any]:
        """Get all extension flags."""
        flags: dict[str, Any] = {}
        for ext in self._extensions:
            flags.update(ext.flags)
        return flags

    def get_extension_paths(self) -> list[str]:
        """Get paths of all loaded extensions."""
        return [ext.path for ext in self._extensions]

    # --- Error handling ---

    def on_error(self, listener: Any) -> Any:
        """Register an error listener. Returns unsubscribe function."""
        self._error_listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._error_listeners:
                self._error_listeners.remove(listener)

        return unsubscribe

    def _emit_error(self, error: ExtensionError) -> None:
        """Emit an error to all registered listeners."""
        for listener in self._error_listeners:
            with contextlib.suppress(Exception):
                listener(error)
