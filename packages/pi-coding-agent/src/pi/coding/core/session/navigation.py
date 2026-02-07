"""Agent session navigation helper.

Handles session switching, forking, tree navigation, and
session statistics/context usage calculations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pi.coding.core.compaction.compact import (
    estimate_context_tokens,
)
from pi.coding.core.resolver import restore_model_from_session


@dataclass
class SessionStats:
    """Aggregate statistics for the current session."""

    user_messages: int = 0
    assistant_messages: int = 0
    tool_result_messages: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_cost: float = 0.0


@dataclass
class ContextUsage:
    """Current context window usage."""

    tokens: int = 0
    context_window: int = 0
    percentage: float = 0.0


@dataclass
class ForkableMessage:
    """A user message that can be forked from."""

    entry_id: str
    text: str


class AgentSessionNavigation:
    """Session lifecycle and tree navigation for AgentSession.

    Uses composition — takes a reference to the parent session.
    """

    def __init__(self, session: Any) -> None:
        self._session = session

    async def switch_session(self, session_path: str) -> None:
        """Load and switch to a different session file.

        Disconnects from agent, loads new session, restores model/thinking,
        and reconnects.
        """
        session = self._session

        # Emit session_before_switch to extensions
        runner = session.extension_runner
        if runner:
            from pi.coding.core.extensions.types import SessionSwitchEvent

            await runner.emit(SessionSwitchEvent(session_path=session_path))

        # Disconnect and abort
        session._disconnect_from_agent()
        session.agent.abort()
        session.agent.reset()
        session.agent.clear_all_queues()

        try:
            # Open the new session
            from pi.coding.core.sessions import SessionManager

            new_sm = SessionManager.open(session_path)
            session._session_manager = new_sm

            # Rebuild context from the session
            context = new_sm.build_session_context()
            session.agent.replace_messages(context.messages)

            # Restore model from session
            if context.model_id and context.provider and session.model_registry:
                result = restore_model_from_session(
                    provider=context.provider,
                    model_id=context.model_id,
                    models=session.model_registry.get_all(),
                )
                if result.model:
                    session.agent.set_model(result.model)

            # Restore thinking level
            if context.thinking_level:
                session.agent.set_thinking_level(context.thinking_level)

            # Emit session_switch to extensions
            if runner:
                from pi.coding.core.extensions.types import SessionSwitchEvent

                await runner.emit(SessionSwitchEvent(session_path=session_path))

        finally:
            session._reconnect_to_agent()

    async def fork(self, entry_id: str) -> dict[str, Any]:
        """Fork the session from a specific entry.

        Creates a branched session from the given entry point.

        Args:
            entry_id: The entry ID to fork from (must be a user message).

        Returns:
            Dict with 'selected_text' (str or None) and 'cancelled' (bool).
        """
        session = self._session
        sm = session.session_manager

        # Validate the entry exists and is a user message
        entry = sm.get_entry(entry_id)
        if not entry:
            return {"selected_text": None, "cancelled": True}

        # Extract user message text for editor pre-fill
        selected_text = None
        if entry.get("type") == "message":
            msg = entry.get("message", {})
            if msg.get("role") == "user":
                selected_text = _extract_user_message_text(msg.get("content", ""))

        # Create branched session
        branched_path = sm.create_branched_session()
        if not branched_path:
            # Fallback: just branch in-place
            parent_id = entry.get("parentId")
            if parent_id:
                sm.branch(parent_id)
            else:
                sm.reset_leaf()

        # Rebuild context
        context = sm.build_session_context()
        session.agent.replace_messages(context.messages)

        # Emit session_fork to extensions
        runner = session.extension_runner
        if runner:
            from pi.coding.core.extensions.types import ExtensionEvent

            await runner.emit(ExtensionEvent(type="session_fork"))

        return {"selected_text": selected_text, "cancelled": False}

    def get_user_messages_for_forking(self) -> list[ForkableMessage]:
        """Get all user messages that can be forked from.

        Returns entry IDs and text for fork UI.
        """
        session = self._session
        entries = session.session_manager.entries
        result: list[ForkableMessage] = []

        for entry in entries:
            if entry.get("type") != "message":
                continue
            msg = entry.get("message", {})
            if msg.get("role") != "user":
                continue

            text = _extract_user_message_text(msg.get("content", ""))
            if text:
                result.append(
                    ForkableMessage(
                        entry_id=entry.get("id", ""),
                        text=text,
                    )
                )

        return result

    def get_session_stats(self) -> SessionStats:
        """Calculate aggregate statistics for the current session."""
        session = self._session
        messages = session.agent.state.messages
        stats = SessionStats()

        for msg in messages:
            role = msg.role if hasattr(msg, "role") else ""

            if role == "user":
                stats.user_messages += 1
            elif role == "assistant":
                stats.assistant_messages += 1
                # Count tool calls
                if hasattr(msg, "content") and isinstance(msg.content, list):
                    for block in msg.content:
                        if hasattr(block, "type") and block.type == "tool_call":
                            stats.tool_calls += 1
                # Accumulate usage
                if hasattr(msg, "usage") and msg.usage:
                    usage = msg.usage
                    stats.input_tokens += usage.input
                    stats.output_tokens += usage.output
                    stats.cache_read_tokens += usage.cache_read
                    stats.cache_write_tokens += usage.cache_write
                    if hasattr(usage, "cost") and usage.cost:
                        stats.total_cost += usage.cost.total
            elif role == "tool_result":
                stats.tool_result_messages += 1

        return stats

    def get_context_usage(self) -> ContextUsage:
        """Get current token usage as a percentage of context window."""
        session = self._session
        model = session.agent.state.model

        if not model or not model.context_window:
            return ContextUsage()

        entries = session.session_manager.entries
        estimate = estimate_context_tokens(entries)

        percentage = (estimate.tokens / model.context_window * 100) if model.context_window > 0 else 0.0

        return ContextUsage(
            tokens=estimate.tokens,
            context_window=model.context_window,
            percentage=percentage,
        )

    def get_last_assistant_text(self) -> str:
        """Get text from the last non-empty assistant message."""
        session = self._session
        messages = session.agent.state.messages

        for msg in reversed(messages):
            if not hasattr(msg, "role") or msg.role != "assistant":
                continue
            # Skip aborted messages with no content
            if (
                hasattr(msg, "stop_reason")
                and msg.stop_reason == "aborted"
                and (not hasattr(msg, "content") or not msg.content)
            ):
                continue
            # Collect text blocks
            if hasattr(msg, "content") and isinstance(msg.content, list):
                texts = []
                for block in msg.content:
                    if hasattr(block, "type") and block.type == "text" and hasattr(block, "text"):
                        texts.append(block.text)
                text = "".join(texts)
                if text.strip():
                    return text

        return ""


def _extract_user_message_text(content: Any) -> str:
    """Extract text from user message content (string or content blocks)."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(item.get("text", ""))
            elif hasattr(item, "type") and item.type == "text" and hasattr(item, "text"):
                texts.append(item.text)
        return " ".join(texts)

    return ""
