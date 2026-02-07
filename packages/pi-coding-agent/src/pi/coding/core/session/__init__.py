"""Main AgentSession orchestrator.

Central class that wires together the Agent, SessionManager, SettingsManager,
ExtensionRunner, ModelRegistry, and tools into a cohesive session.

Handles:
- Runtime setup (tool registration, system prompt building)
- Prompt handling (prompt, steer, follow_up)
- Event routing (agent events → extensions → session persistence → subscribers)
- Lifecycle (subscribe, dispose, abort, new_session, reload)
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pi.agent.types import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    MessageEndEvent,
    MessageStartEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from pi.ai.types import (
    ImageContent,
    TextContent,
    UserMessage,
)
from pi.coding.core.prompt import ContextFile, build_system_prompt
from pi.coding.core.session.compaction import AgentSessionCompaction
from pi.coding.core.session.events import AgentSessionEvent, AgentSessionOrAgentEvent
from pi.coding.core.session.models import AgentSessionModels
from pi.coding.core.session.navigation import AgentSessionNavigation

if TYPE_CHECKING:
    from pi.agent.agent import Agent
    from pi.agent.types import AgentMessage, AgentTool
    from pi.ai.types import Model
    from pi.coding.core.extensions.runner import ExtensionRunner
    from pi.coding.core.extensions.types import Extension
    from pi.coding.core.resolver import ModelRegistry, ScopedModel
    from pi.coding.core.sessions import SessionManager
    from pi.coding.core.settings import SettingsManager


EventListener = Callable[[AgentSessionOrAgentEvent], None]


@dataclass
class AgentSessionConfig:
    """Configuration for constructing an AgentSession."""

    agent: Agent
    session_manager: SessionManager
    settings_manager: SettingsManager
    cwd: str
    scoped_models: list[ScopedModel] | None = None
    model_registry: ModelRegistry | None = None
    extension_runner: ExtensionRunner | None = None
    extensions: list[Extension] | None = None
    custom_tools: list[AgentTool] | None = None
    initial_active_tool_names: list[str] | None = None
    base_tools_override: dict[str, AgentTool] | None = None
    context_files: list[ContextFile] | None = None
    custom_prompt: str | None = None
    append_system_prompt: str | None = None


class AgentSession:
    """Main session orchestrator for the coding agent.

    Wires together the Agent runtime, SessionManager persistence,
    SettingsManager configuration, ExtensionRunner hooks, and tools.
    """

    def __init__(self, config: AgentSessionConfig) -> None:
        # Core components
        self._agent = config.agent
        self._session_manager = config.session_manager
        self._settings_manager = config.settings_manager
        self._cwd = config.cwd

        # Model/extension state
        self._scoped_models = list(config.scoped_models or [])
        self._model_registry = config.model_registry
        self._extension_runner = config.extension_runner
        self._extensions = list(config.extensions or [])

        # Tool registries
        self._base_tool_registry: dict[str, AgentTool] = {}
        self._tool_registry: dict[str, AgentTool] = {}
        self._active_tool_names: set[str] = set()

        # System prompt state
        self._base_system_prompt = ""
        self._context_files = list(config.context_files or [])
        self._custom_prompt = config.custom_prompt
        self._append_system_prompt = config.append_system_prompt

        # Event listeners
        self._event_listeners: list[EventListener] = []
        self._unsubscribe_agent: Any = None

        # Message tracking
        self._steering_messages: list[str] = []
        self._follow_up_messages: list[str] = []
        self._turn_index = 0

        # Composition helpers
        self._models = AgentSessionModels(self)
        self._compaction = AgentSessionCompaction(self)
        self._navigation = AgentSessionNavigation(self)

        # Subscribe to agent events
        self._unsubscribe_agent = self._agent.subscribe(self._handle_agent_event)

        # Build runtime (tools + system prompt)
        self._build_runtime(config)

    # --- Properties ---

    @property
    def agent(self) -> Agent:
        return self._agent

    @property
    def session_manager(self) -> SessionManager:
        return self._session_manager

    @session_manager.setter
    def session_manager(self, value: SessionManager) -> None:
        self._session_manager = value

    @property
    def settings_manager(self) -> SettingsManager:
        return self._settings_manager

    @property
    def cwd(self) -> str:
        return self._cwd

    @property
    def model(self) -> Model | None:
        return self._agent.state.model

    @property
    def thinking_level(self) -> str:
        return self._agent.state.thinking_level

    @property
    def is_streaming(self) -> bool:
        return self._agent.state.is_streaming

    @property
    def is_compacting(self) -> bool:
        return self._compaction.is_compacting

    @property
    def retry_attempt(self) -> int:
        return self._compaction.retry_attempt

    @property
    def messages(self) -> list[AgentMessage]:
        return self._agent.state.messages

    @property
    def system_prompt(self) -> str:
        return self._agent.state.system_prompt

    @property
    def session_file(self) -> str | None:
        return self._session_manager.session_file

    @property
    def session_id(self) -> str:
        return self._session_manager.session_id

    @property
    def session_name(self) -> str | None:
        return self._session_manager.get_session_name()

    @property
    def scoped_models(self) -> list[ScopedModel]:
        return list(self._scoped_models)

    @property
    def model_registry(self) -> ModelRegistry | None:
        return self._model_registry

    @property
    def extension_runner(self) -> ExtensionRunner | None:
        return self._extension_runner

    @property
    def steering_mode(self) -> str:
        return self._settings_manager.get_steering_mode() or "one-at-a-time"

    @property
    def follow_up_mode(self) -> str:
        return self._settings_manager.get_follow_up_mode() or "one-at-a-time"

    # --- Runtime setup ---

    def _build_runtime(self, config: AgentSessionConfig) -> None:
        """Set up tools, extensions, and system prompt."""
        from pi.coding.core.tools import create_all_tools

        # 1. Build base tools
        if config.base_tools_override:
            self._base_tool_registry = dict(config.base_tools_override)
        else:
            self._base_tool_registry = create_all_tools(self._cwd)

        # 2. Start with base tools
        self._tool_registry = dict(self._base_tool_registry)

        # 3. Add custom tools
        if config.custom_tools:
            for tool in config.custom_tools:
                self._tool_registry[tool.name] = tool

        # 4. Wrap tools with extension hooks if extension runner exists
        if self._extension_runner:
            from pi.coding.core.extensions.wrapper import (
                wrap_registered_tools,
                wrap_tools_with_extensions,
            )

            # Register extension tools
            registered = self._extension_runner.get_all_registered_tools()
            if registered:
                wrapped = wrap_registered_tools(list(registered.values()), self._extension_runner)
                for tool in wrapped:
                    self._tool_registry[tool.name] = tool

            # Wrap all tools with extension middleware
            base_tools_list = list(self._base_tool_registry.values())
            wrapped_base = wrap_tools_with_extensions(base_tools_list, self._extension_runner)
            for tool in wrapped_base:
                self._tool_registry[tool.name] = tool

        # 5. Set active tools
        if config.initial_active_tool_names:
            self._active_tool_names = set(config.initial_active_tool_names)
        else:
            self._active_tool_names = set(self._tool_registry.keys())

        active_tools = [self._tool_registry[name] for name in self._active_tool_names if name in self._tool_registry]
        self._agent.set_tools(active_tools)

        # 6. Build and set system prompt
        self._rebuild_system_prompt()

    def _rebuild_system_prompt(self, tool_names: list[str] | None = None) -> None:
        """Rebuild and set the system prompt."""
        selected = tool_names or [name for name in self._active_tool_names if name in self._base_tool_registry]

        self._base_system_prompt = build_system_prompt(
            selected_tools=selected,
            custom_prompt=self._custom_prompt,
            append_system_prompt=self._append_system_prompt,
            cwd=self._cwd,
            context_files=self._context_files,
        )
        self._agent.set_system_prompt(self._base_system_prompt)

    # --- Tool management ---

    def get_active_tool_names(self) -> list[str]:
        """Get names of currently active tools."""
        return list(self._active_tool_names)

    def get_all_tools(self) -> dict[str, AgentTool]:
        """Get all registered tools (active and inactive)."""
        return dict(self._tool_registry)

    def set_active_tools_by_name(self, tool_names: list[str]) -> None:
        """Set which tools are active by name."""
        self._active_tool_names = set(tool_names)
        active_tools = [self._tool_registry[name] for name in tool_names if name in self._tool_registry]
        self._agent.set_tools(active_tools)
        self._rebuild_system_prompt(tool_names)

    # --- Prompting ---

    async def prompt(
        self,
        text: str,
        *,
        images: list[ImageContent] | None = None,
    ) -> None:
        """Send a prompt to the agent.

        Processes input through extensions, validates model and API key,
        checks compaction, then sends to agent.
        """
        # Emit input event to extensions for interception/transformation
        if self._extension_runner:
            text = await self._extension_runner.emit_input(text)

        # Validate model
        model = self._agent.state.model
        if not model:
            raise RuntimeError("No model configured")

        # Emit before_agent_start to extensions
        if self._extension_runner:
            _, system_prompt = await self._extension_runner.emit_before_agent_start(text, self._base_system_prompt)
            if system_prompt != self._base_system_prompt:
                self._agent.set_system_prompt(system_prompt)

        # Build user message
        content: list[TextContent | ImageContent] = [TextContent(text=text)]
        if images:
            content.extend(images)

        user_msg = UserMessage(
            role="user",
            content=content,
            timestamp=int(time.time() * 1000),
        )

        # Send to agent
        await self._agent.prompt(user_msg)

        # Wait for any retry to complete
        await self._compaction.wait_for_retry()

    def steer(self, text: str, images: list[ImageContent] | None = None) -> None:
        """Queue a steering message (interrupts current turn)."""
        self._steering_messages.append(text)

        content: list[TextContent | ImageContent] = [TextContent(text=text)]
        if images:
            content.extend(images)

        msg = UserMessage(
            role="user",
            content=content,
            timestamp=int(time.time() * 1000),
        )
        self._agent.steer(msg)

    def follow_up(self, text: str, images: list[ImageContent] | None = None) -> None:
        """Queue a follow-up message (sent after agent finishes)."""
        self._follow_up_messages.append(text)

        content: list[TextContent | ImageContent] = [TextContent(text=text)]
        if images:
            content.extend(images)

        msg = UserMessage(
            role="user",
            content=content,
            timestamp=int(time.time() * 1000),
        )
        self._agent.follow_up(msg)

    def clear_queue(self) -> tuple[list[str], list[str]]:
        """Clear and return pending steering and follow-up queues."""
        steering = list(self._steering_messages)
        follow_up = list(self._follow_up_messages)
        self._steering_messages.clear()
        self._follow_up_messages.clear()
        self._agent.clear_all_queues()
        return steering, follow_up

    # --- Event handling ---

    def _handle_agent_event(self, event: AgentEvent) -> None:
        """Handle events from the Agent runtime.

        Routes to extensions, persists to session, and emits to listeners.
        """
        # Track steering/follow-up messages
        if isinstance(event, MessageStartEvent):
            msg = event.message
            if hasattr(msg, "role") and msg.role == "user":
                text = _extract_message_text(msg)
                if text in self._steering_messages:
                    self._steering_messages.remove(text)
                elif text in self._follow_up_messages:
                    self._follow_up_messages.remove(text)

        # Emit to extensions
        self._emit_extension_event(event)

        # Emit to all registered listeners
        for listener in self._event_listeners:
            listener(event)

        # Persist messages to session on message_end
        if isinstance(event, MessageEndEvent):
            self._persist_message(event.message)

            # Reset retry on successful assistant message
            if (
                hasattr(event.message, "role")
                and event.message.role == "assistant"
                and not (hasattr(event.message, "error_message") and event.message.error_message)
            ):
                self._compaction.reset_retry_on_success()

        # Check for retryable errors and compaction on agent_end
        if isinstance(event, AgentEndEvent):
            self._on_agent_end(event)

    def _emit_extension_event(self, event: AgentEvent) -> None:
        """Map agent events to extension events and emit."""
        runner = self._extension_runner
        if not runner:
            return

        if isinstance(event, AgentStartEvent):
            self._turn_index = 0
            from pi.coding.core.extensions.types import AgentStartEvent as ExtAgentStart

            asyncio.get_event_loop().call_soon(lambda: asyncio.ensure_future(runner.emit(ExtAgentStart())))

        elif isinstance(event, AgentEndEvent):
            from pi.coding.core.extensions.types import AgentEndEvent as ExtAgentEnd

            asyncio.get_event_loop().call_soon(lambda: asyncio.ensure_future(runner.emit(ExtAgentEnd())))

        elif isinstance(event, TurnStartEvent):
            from pi.coding.core.extensions.types import TurnStartEvent as ExtTurnStart

            asyncio.get_event_loop().call_soon(lambda: asyncio.ensure_future(runner.emit(ExtTurnStart())))

        elif isinstance(event, TurnEndEvent):
            self._turn_index += 1
            from pi.coding.core.extensions.types import TurnEndEvent as ExtTurnEnd

            asyncio.get_event_loop().call_soon(lambda: asyncio.ensure_future(runner.emit(ExtTurnEnd())))

    def _persist_message(self, message: AgentMessage) -> None:
        """Persist a message to the session manager."""
        if not hasattr(message, "role"):
            return

        # Serialize the message for session storage
        msg_dict = _message_to_dict(message)
        if msg_dict:
            self._session_manager.append_message(msg_dict)

    def _on_agent_end(self, event: AgentEndEvent) -> None:
        """Handle agent_end: check for retryable errors, then compaction."""
        if not event.messages:
            return

        last_msg = event.messages[-1]

        # Check for retryable error
        if (
            hasattr(last_msg, "error_message")
            and last_msg.error_message
            and self._compaction.is_retryable_error(last_msg)
        ):
            asyncio.get_event_loop().call_soon(
                lambda: asyncio.ensure_future(self._compaction.handle_retryable_error(last_msg))
            )
            return

        # Check if compaction is needed
        assistant_msg = last_msg if hasattr(last_msg, "stop_reason") else None
        self._compaction.check_compaction(assistant_msg)

    # --- Session event emission ---

    def _emit_session_event(self, event: AgentSessionEvent) -> None:
        """Emit an AgentSession-specific event to listeners."""
        for listener in self._event_listeners:
            listener(event)

    # --- Subscriptions ---

    def subscribe(self, fn: EventListener) -> Callable[[], None]:
        """Subscribe to agent session events. Returns unsubscribe function."""
        self._event_listeners.append(fn)

        def unsubscribe() -> None:
            if fn in self._event_listeners:
                self._event_listeners.remove(fn)

        return unsubscribe

    # --- Disconnect/reconnect ---

    def _disconnect_from_agent(self) -> None:
        """Temporarily stop receiving agent events."""
        if self._unsubscribe_agent:
            self._unsubscribe_agent()
            self._unsubscribe_agent = None

    def _reconnect_to_agent(self) -> None:
        """Resume receiving agent events."""
        if not self._unsubscribe_agent:
            self._unsubscribe_agent = self._agent.subscribe(self._handle_agent_event)

    # --- Lifecycle ---

    def dispose(self) -> None:
        """Clean up and disconnect."""
        self._disconnect_from_agent()
        self._event_listeners.clear()

    def abort(self) -> None:
        """Abort current operation."""
        self._compaction.abort_retry()
        self._agent.abort()

    async def new_session(self) -> None:
        """Create a new session, resetting agent state."""
        # Emit session_before_switch
        runner = self._extension_runner
        if runner:
            from pi.coding.core.extensions.types import SessionSwitchEvent

            await runner.emit(SessionSwitchEvent())

        # Disconnect and reset
        self._disconnect_from_agent()
        self._agent.abort()
        self._agent.reset()
        self._agent.clear_all_queues()

        # Create new session
        from pi.coding.core.sessions import SessionManager

        new_sm = SessionManager.create(self._cwd, self._session_manager.session_dir)
        self._session_manager = new_sm

        # Clear queues
        self._steering_messages.clear()
        self._follow_up_messages.clear()

        # Reconnect
        self._reconnect_to_agent()

        # Emit session_switch
        if runner:
            from pi.coding.core.extensions.types import SessionStartEvent

            await runner.emit(SessionStartEvent())

    def reload(self) -> None:
        """Reload settings and rebuild runtime."""
        self._settings_manager.reload()
        self._rebuild_system_prompt()

    # --- API key resolution ---

    async def _get_api_key(self, provider: str) -> str | None:
        """Get API key for a provider."""
        from pi.ai.env import get_env_api_key

        api_key = get_env_api_key(provider)
        if api_key:
            return api_key

        if self._agent.get_api_key:
            result = self._agent.get_api_key(provider)
            if inspect.isawaitable(result):
                return await result
            return result

        return None

    # --- Model management (delegated) ---

    async def set_model(self, model: Model) -> None:
        await self._models.set_model(model)

    def cycle_model(self, direction: int = 1) -> Any:
        return self._models.cycle_model(direction)

    def set_thinking_level(self, level: str) -> str:
        return self._models.set_thinking_level(level)

    def cycle_thinking_level(self) -> str | None:
        return self._models.cycle_thinking_level()

    def get_available_thinking_levels(self) -> list[str]:
        return self._models.get_available_thinking_levels()

    # --- Compaction (delegated) ---

    async def compact(self, custom_instructions: str | None = None) -> Any:
        return await self._compaction.compact_manual(custom_instructions)

    def abort_compaction(self) -> None:
        self._compaction.abort_compaction()

    # --- Navigation (delegated) ---

    async def switch_session(self, session_path: str) -> None:
        await self._navigation.switch_session(session_path)

    async def fork(self, entry_id: str) -> dict[str, Any]:
        return await self._navigation.fork(entry_id)

    def get_user_messages_for_forking(self) -> list[Any]:
        return self._navigation.get_user_messages_for_forking()

    def get_session_stats(self) -> Any:
        return self._navigation.get_session_stats()

    def get_context_usage(self) -> Any:
        return self._navigation.get_context_usage()

    def get_last_assistant_text(self) -> str:
        return self._navigation.get_last_assistant_text()


# --- Helpers ---


def _extract_message_text(message: Any) -> str:
    """Extract text from a message's content."""
    if not hasattr(message, "content"):
        return ""

    content = message.content
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        for item in content:
            if hasattr(item, "type") and item.type == "text" and hasattr(item, "text"):
                return item.text

    return ""


def _message_to_dict(message: Any) -> dict[str, Any] | None:
    """Serialize a message to a dict for session storage."""
    if hasattr(message, "model_dump"):
        return message.model_dump(by_alias=True, exclude_none=True)

    if not hasattr(message, "role"):
        return None

    result: dict[str, Any] = {"role": message.role}

    if hasattr(message, "content"):
        content = message.content
        if isinstance(content, str):
            result["content"] = content
        elif isinstance(content, list):
            serialized = []
            for item in content:
                if hasattr(item, "model_dump"):
                    serialized.append(item.model_dump(by_alias=True, exclude_none=True))
                elif isinstance(item, dict):
                    serialized.append(item)
            result["content"] = serialized

    if hasattr(message, "timestamp"):
        result["timestamp"] = message.timestamp

    # Assistant-specific fields
    if hasattr(message, "api"):
        result["api"] = message.api
    if hasattr(message, "provider"):
        result["provider"] = message.provider
    if hasattr(message, "model"):
        result["model"] = message.model
    if hasattr(message, "usage") and message.usage and hasattr(message.usage, "model_dump"):
        result["usage"] = message.usage.model_dump(by_alias=True, exclude_none=True)
    if hasattr(message, "stop_reason"):
        result["stopReason"] = message.stop_reason
    if hasattr(message, "error_message") and message.error_message:
        result["errorMessage"] = message.error_message

    # ToolResult-specific fields
    if hasattr(message, "tool_call_id"):
        result["toolCallId"] = message.tool_call_id
    if hasattr(message, "tool_name"):
        result["toolName"] = message.tool_name
    if hasattr(message, "is_error"):
        result["isError"] = message.is_error
    if hasattr(message, "details"):
        result["details"] = message.details

    return result
