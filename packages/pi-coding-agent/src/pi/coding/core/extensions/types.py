"""Extension system types: events, API, tool definitions, and contexts."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from pi.agent.types import AgentToolResult

# --- Extension factory ---

ExtensionFactory = Callable[["ExtensionAPI"], None | Awaitable[None]]

# --- Event types ---

EventType = Literal[
    "session_start",
    "session_switch",
    "session_before_switch",
    "session_fork",
    "session_before_fork",
    "session_before_compact",
    "session_compact",
    "session_before_tree",
    "session_tree",
    "session_shutdown",
    "context",
    "before_agent_start",
    "agent_start",
    "agent_end",
    "turn_start",
    "turn_end",
    "model_select",
    "tool_call",
    "tool_result",
    "user_bash",
    "input",
    "resources_discover",
]


@dataclass
class ExtensionEvent:
    """Base event passed to extension handlers."""

    type: str
    data: dict[str, Any] = field(default_factory=dict)


# --- Session events ---


@dataclass
class SessionStartEvent(ExtensionEvent):
    type: str = "session_start"


@dataclass
class SessionSwitchEvent(ExtensionEvent):
    type: str = "session_switch"
    session_path: str = ""


@dataclass
class SessionBeforeCompactEvent(ExtensionEvent):
    type: str = "session_before_compact"


@dataclass
class SessionCompactEvent(ExtensionEvent):
    type: str = "session_compact"
    from_extension: bool = False


@dataclass
class SessionShutdownEvent(ExtensionEvent):
    type: str = "session_shutdown"


# --- Agent events ---


@dataclass
class ContextEvent(ExtensionEvent):
    """Before LLM call - can modify messages."""

    type: str = "context"
    messages: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class BeforeAgentStartEvent(ExtensionEvent):
    """After user input, before agent - can modify system prompt."""

    type: str = "before_agent_start"
    prompt: str = ""
    system_prompt: str = ""


@dataclass
class AgentStartEvent(ExtensionEvent):
    type: str = "agent_start"


@dataclass
class AgentEndEvent(ExtensionEvent):
    type: str = "agent_end"


@dataclass
class TurnStartEvent(ExtensionEvent):
    type: str = "turn_start"


@dataclass
class TurnEndEvent(ExtensionEvent):
    type: str = "turn_end"


# --- Tool events ---


@dataclass
class ToolCallEvent(ExtensionEvent):
    """Before tool execution - can block."""

    type: str = "tool_call"
    tool_name: str = ""
    tool_call_id: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    blocked: bool = False
    block_reason: str = ""


@dataclass
class ToolResultEvent(ExtensionEvent):
    """After tool execution - can modify result."""

    type: str = "tool_result"
    tool_name: str = ""
    tool_call_id: str = ""
    result: AgentToolResult | None = None
    is_error: bool = False


# --- Other events ---


@dataclass
class ModelSelectEvent(ExtensionEvent):
    type: str = "model_select"
    model_id: str = ""
    provider: str = ""


@dataclass
class InputEvent(ExtensionEvent):
    """User input received - can transform."""

    type: str = "input"
    text: str = ""
    transformed_text: str | None = None


@dataclass
class ResourcesDiscoverEvent(ExtensionEvent):
    type: str = "resources_discover"
    cwd: str = ""
    reason: str = ""


# --- Tool definition ---


@dataclass
class ToolDefinition:
    """Definition for a custom tool registered by an extension."""

    name: str
    label: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    execute: Callable[..., Awaitable[AgentToolResult]]
    render_call: Callable[..., Any] | None = None
    render_result: Callable[..., Any] | None = None


@dataclass
class RegisteredTool:
    """A tool registered by an extension, with its source."""

    definition: ToolDefinition
    extension_path: str


# --- Command definition ---


@dataclass
class RegisteredCommand:
    """A slash command registered by an extension."""

    name: str
    description: str
    handler: Callable[..., Awaitable[None]]
    extension_path: str
    aliases: list[str] = field(default_factory=list)


# --- Extension flag ---


@dataclass
class ExtensionFlag:
    """CLI flag registered by an extension."""

    name: str
    description: str
    flag_type: Literal["boolean", "string"] = "boolean"
    default: bool | str = False
    extension_path: str = ""


# --- Extension shortcut ---


@dataclass
class ExtensionShortcut:
    """Keyboard shortcut registered by an extension."""

    key_id: str
    description: str
    handler: Callable[..., Awaitable[None]]
    extension_path: str


# --- Extension context ---


@dataclass
class ExtensionContext:
    """Context available in all event handlers."""

    cwd: str = ""
    has_ui: bool = False
    model_id: str | None = None
    model_provider: str | None = None


# --- Extension error ---


@dataclass
class ExtensionError:
    """Error from an extension handler."""

    extension_path: str
    event: str
    error: str
    stack: str | None = None


# --- Extension structure ---


@dataclass
class Extension:
    """A loaded extension with its handlers and registrations."""

    path: str
    resolved_path: str
    handlers: dict[str, list[Callable[..., Any]]] = field(default_factory=dict)
    tools: dict[str, RegisteredTool] = field(default_factory=dict)
    commands: dict[str, RegisteredCommand] = field(default_factory=dict)
    flags: dict[str, ExtensionFlag] = field(default_factory=dict)
    shortcuts: dict[str, ExtensionShortcut] = field(default_factory=dict)


# --- Handler function type ---

HandlerFn = Callable[..., Any | Awaitable[Any]]


# --- Extension API ---


class ExtensionAPI:
    """API provided to extensions during initialization.

    Extensions receive this as their sole argument and use it to
    register handlers, tools, commands, shortcuts, and flags.
    """

    def __init__(self, extension: Extension, cwd: str) -> None:
        self._extension = extension
        self._cwd = cwd

    @property
    def cwd(self) -> str:
        return self._cwd

    def on(self, event_type: str, handler: HandlerFn) -> None:
        """Register a handler for an event type."""
        if event_type not in self._extension.handlers:
            self._extension.handlers[event_type] = []
        self._extension.handlers[event_type].append(handler)

    def register_tool(self, definition: ToolDefinition) -> None:
        """Register a custom tool callable by the LLM."""
        self._extension.tools[definition.name] = RegisteredTool(
            definition=definition,
            extension_path=self._extension.path,
        )

    def register_command(
        self,
        name: str,
        *,
        description: str = "",
        handler: Callable[..., Awaitable[None]] | None = None,
        aliases: list[str] | None = None,
    ) -> None:
        """Register a custom slash command."""
        if handler is None:
            msg = f"Command '{name}' requires a handler"
            raise ValueError(msg)
        self._extension.commands[name] = RegisteredCommand(
            name=name,
            description=description,
            handler=handler,
            extension_path=self._extension.path,
            aliases=aliases or [],
        )

    def register_flag(
        self,
        name: str,
        *,
        description: str = "",
        flag_type: Literal["boolean", "string"] = "boolean",
        default: bool | str = False,
    ) -> None:
        """Register a CLI flag."""
        self._extension.flags[name] = ExtensionFlag(
            name=name,
            description=description,
            flag_type=flag_type,
            default=default,
            extension_path=self._extension.path,
        )

    def register_shortcut(
        self,
        key_id: str,
        *,
        description: str = "",
        handler: Callable[..., Awaitable[None]] | None = None,
    ) -> None:
        """Register a keyboard shortcut."""
        if handler is None:
            msg = f"Shortcut '{key_id}' requires a handler"
            raise ValueError(msg)
        self._extension.shortcuts[key_id] = ExtensionShortcut(
            key_id=key_id,
            description=description,
            handler=handler,
            extension_path=self._extension.path,
        )
