"""Agent session compaction and auto-retry helper.

Handles automatic context compaction when thresholds are exceeded,
overflow detection, and exponential backoff retry for transient errors.
"""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, Any

from pi.coding.core.compaction.compact import (
    CompactionResult,
    CompactionSettings,
    compact,
    estimate_context_tokens,
    prepare_compaction,
    should_compact,
)
from pi.coding.core.session.events import (
    AutoCompactionEndEvent,
    AutoCompactionStartEvent,
    AutoRetryEndEvent,
    AutoRetryStartEvent,
)

if TYPE_CHECKING:
    from pi.ai.types import AssistantMessage

# Regex for detecting retryable errors
_RETRYABLE_ERROR_RE = re.compile(
    r"overloaded|rate.?limit|429|5\d{2}|service.?unavailable"
    r"|connection.?(reset|refused|timeout|error)"
    r"|fetch.?failed|terminated|ECONNRESET|ETIMEDOUT"
    r"|retry.?delay|too.?many.?requests",
    re.IGNORECASE,
)

# Context overflow is NOT retryable (handled by compaction instead)
_OVERFLOW_RE = re.compile(
    r"context.?(window|length|limit)|too.?many.?tokens|max.?context",
    re.IGNORECASE,
)


class AgentSessionCompaction:
    """Compaction and retry management for AgentSession.

    Uses composition — takes a reference to the parent session.
    """

    def __init__(self, session: Any) -> None:
        self._session = session

        # Abort signals
        self._compaction_abort: asyncio.Event | None = None
        self._auto_compaction_abort: asyncio.Event | None = None
        self._retry_abort: asyncio.Event | None = None

        # Retry state
        self._retry_attempt = 0
        self._retry_future: asyncio.Future[None] | None = None

        # Compaction state
        self._is_compacting = False

    @property
    def is_compacting(self) -> bool:
        return self._is_compacting

    @property
    def retry_attempt(self) -> int:
        return self._retry_attempt

    async def compact_manual(self, custom_instructions: str | None = None) -> CompactionResult | None:
        """Run manual compaction (disconnect, compact, reconnect).

        Returns the CompactionResult or None if nothing to compact.
        """
        session = self._session

        # Disconnect from agent events during compaction
        session._disconnect_from_agent()
        session.agent.abort()

        try:
            self._is_compacting = True
            self._compaction_abort = asyncio.Event()

            model = session.agent.state.model
            if not model:
                return None

            # Get API key
            api_key = await session._get_api_key(model.provider)

            # Build entries from session manager
            entries = session.session_manager.entries

            # Emit session_before_compact to extensions
            runner = session.extension_runner
            if runner:
                from pi.coding.core.extensions.types import SessionBeforeCompactEvent

                event = SessionBeforeCompactEvent()
                await runner.emit(event)

            # Prepare and execute compaction
            preparation = prepare_compaction(entries)
            if not preparation:
                return None

            result = await compact(
                preparation,
                model,
                api_key=api_key,
                custom_instructions=custom_instructions,
                abort_event=self._compaction_abort,
            )

            # Persist compaction to session
            details_dict = None
            if result.details:
                details_dict = {
                    "readFiles": result.details.read_files,
                    "modifiedFiles": result.details.modified_files,
                }

            session.session_manager.append_compaction(
                result.summary,
                first_kept_entry_id=result.first_kept_entry_id,
                tokens_before=result.tokens_before,
                details=details_dict,
            )

            # Rebuild messages from session context
            context = session.session_manager.build_session_context()
            session.agent.replace_messages(context.messages)

            # Emit session_compact to extensions
            if runner:
                from pi.coding.core.extensions.types import SessionCompactEvent

                await runner.emit(SessionCompactEvent())

            return result

        finally:
            self._is_compacting = False
            self._compaction_abort = None
            session._reconnect_to_agent()

    def abort_compaction(self) -> None:
        """Cancel an in-progress compaction."""
        if self._compaction_abort:
            self._compaction_abort.set()
        if self._auto_compaction_abort:
            self._auto_compaction_abort.set()

    def check_compaction(self, assistant_message: AssistantMessage | None = None) -> None:
        """Check if compaction is needed after agent_end.

        Two cases:
        1. Overflow: LLM returned context overflow error → auto-compact with retry
        2. Threshold: Context tokens exceed threshold → auto-compact without retry
        """
        session = self._session
        model = session.agent.state.model
        if not model or not model.context_window:
            return

        settings = self._get_compaction_settings()
        if not settings.enabled:
            return

        # Case 1: Overflow detection
        if assistant_message and self._is_overflow_error(assistant_message):
            # Schedule auto-compaction with retry
            asyncio.get_event_loop().call_soon(
                lambda: asyncio.ensure_future(self._run_auto_compaction("overflow", will_retry=True))
            )
            return

        # Case 2: Threshold check
        entries = session.session_manager.entries
        estimate = estimate_context_tokens(entries)
        if should_compact(estimate.tokens, model.context_window, settings):
            asyncio.get_event_loop().call_soon(
                lambda: asyncio.ensure_future(self._run_auto_compaction("threshold", will_retry=False))
            )

    async def _run_auto_compaction(
        self,
        reason: str,
        will_retry: bool,
    ) -> None:
        """Execute auto-compaction with events."""
        session = self._session
        self._auto_compaction_abort = asyncio.Event()

        # Emit start event
        session._emit_session_event(AutoCompactionStartEvent(reason=reason))

        try:
            self._is_compacting = True
            session._disconnect_from_agent()

            model = session.agent.state.model
            if not model:
                session._emit_session_event(AutoCompactionEndEvent(error_message="No model configured"))
                return

            api_key = await session._get_api_key(model.provider)
            entries = session.session_manager.entries

            # Emit session_before_compact to extensions
            runner = session.extension_runner
            if runner:
                from pi.coding.core.extensions.types import SessionBeforeCompactEvent

                await runner.emit(SessionBeforeCompactEvent())

            preparation = prepare_compaction(entries)
            if not preparation:
                session._emit_session_event(AutoCompactionEndEvent(error_message="Nothing to compact"))
                return

            result = await compact(
                preparation,
                model,
                api_key=api_key,
                abort_event=self._auto_compaction_abort,
            )

            # Persist
            details_dict = None
            if result.details:
                details_dict = {
                    "readFiles": result.details.read_files,
                    "modifiedFiles": result.details.modified_files,
                }

            session.session_manager.append_compaction(
                result.summary,
                first_kept_entry_id=result.first_kept_entry_id,
                tokens_before=result.tokens_before,
                details=details_dict,
            )

            # Rebuild messages
            context = session.session_manager.build_session_context()
            session.agent.replace_messages(context.messages)

            if runner:
                from pi.coding.core.extensions.types import SessionCompactEvent

                await runner.emit(SessionCompactEvent())

            session._emit_session_event(AutoCompactionEndEvent(result=result, will_retry=will_retry))

            # If will_retry, remove the error message and continue
            if will_retry:
                messages = session.agent.state.messages
                if messages and hasattr(messages[-1], "error_message") and messages[-1].error_message:
                    session.agent.replace_messages(messages[:-1])
                # Continue the agent loop
                asyncio.get_event_loop().call_soon(lambda: asyncio.ensure_future(session.agent.continue_()))

        except Exception as e:
            aborted = self._auto_compaction_abort is not None and self._auto_compaction_abort.is_set()
            session._emit_session_event(
                AutoCompactionEndEvent(
                    aborted=aborted,
                    error_message=str(e),
                )
            )
        finally:
            self._is_compacting = False
            self._auto_compaction_abort = None
            session._reconnect_to_agent()

    def is_retryable_error(self, message: AssistantMessage) -> bool:
        """Check if an error message indicates a retryable condition."""
        if not hasattr(message, "error_message") or not message.error_message:
            return False

        error = message.error_message

        # Overflow is handled by compaction, not retry
        if _OVERFLOW_RE.search(error):
            return False

        return bool(_RETRYABLE_ERROR_RE.search(error))

    async def handle_retryable_error(self, message: AssistantMessage) -> bool:
        """Handle a retryable error with exponential backoff.

        Returns True if retry was initiated, False if max retries exceeded.
        """
        session = self._session
        retry_settings = session.settings_manager.get_retry_settings()

        if not retry_settings.enabled:
            return False

        max_retries = retry_settings.max_retries or 3
        base_delay_ms = retry_settings.base_delay_ms or 2000
        max_delay_ms = retry_settings.max_delay_ms or 60000

        self._retry_attempt += 1

        # Create retry future on first attempt
        if self._retry_attempt == 1:
            loop = asyncio.get_running_loop()
            self._retry_future = loop.create_future()

        # Check max retries
        if self._retry_attempt > max_retries:
            session._emit_session_event(
                AutoRetryEndEvent(
                    success=False,
                    attempt=self._retry_attempt - 1,
                    final_error=message.error_message if hasattr(message, "error_message") else None,
                )
            )
            self._reset_retry()
            return False

        # Calculate delay with exponential backoff
        delay_ms = min(
            base_delay_ms * (2 ** (self._retry_attempt - 1)),
            max_delay_ms,
        )

        # Emit retry start event
        session._emit_session_event(
            AutoRetryStartEvent(
                attempt=self._retry_attempt,
                max_attempts=max_retries,
                delay_ms=int(delay_ms),
                error_message=message.error_message if hasattr(message, "error_message") else "",
            )
        )

        # Remove error message from agent state
        messages = session.agent.state.messages
        if messages and hasattr(messages[-1], "error_message") and messages[-1].error_message:
            session.agent.replace_messages(messages[:-1])

        # Sleep with abort support
        self._retry_abort = asyncio.Event()
        try:
            await asyncio.wait_for(
                self._retry_abort.wait(),
                timeout=delay_ms / 1000.0,
            )
            # Abort was triggered
            self._reset_retry()
            return False
        except TimeoutError:
            pass  # Normal timeout = sleep complete
        finally:
            self._retry_abort = None

        # Continue agent loop
        asyncio.get_event_loop().call_soon(lambda: asyncio.ensure_future(session.agent.continue_()))

        return True

    def abort_retry(self) -> None:
        """Cancel an in-progress retry."""
        if self._retry_abort:
            self._retry_abort.set()
        self._resolve_retry()

    async def wait_for_retry(self) -> None:
        """Wait for any active retry sequence to complete."""
        if self._retry_future:
            await self._retry_future

    def reset_retry_on_success(self) -> None:
        """Reset retry counter on successful response."""
        if self._retry_attempt > 0:
            session = self._session
            session._emit_session_event(
                AutoRetryEndEvent(
                    success=True,
                    attempt=self._retry_attempt,
                )
            )
            self._reset_retry()

    def _reset_retry(self) -> None:
        """Reset retry state."""
        self._retry_attempt = 0
        self._resolve_retry()

    def _resolve_retry(self) -> None:
        """Resolve the retry future if pending."""
        if self._retry_future and not self._retry_future.done():
            self._retry_future.set_result(None)
        self._retry_future = None

    def _is_overflow_error(self, message: AssistantMessage) -> bool:
        """Check if a message indicates context overflow."""
        if not hasattr(message, "error_message") or not message.error_message:
            return False
        return bool(_OVERFLOW_RE.search(message.error_message))

    def _get_compaction_settings(self) -> CompactionSettings:
        """Get compaction settings from settings manager."""
        session = self._session
        sm_settings = session.settings_manager.get_compaction_settings()
        return CompactionSettings(
            enabled=sm_settings.enabled if sm_settings.enabled is not None else True,
            reserve_tokens=sm_settings.reserve_tokens or 16384,
            keep_recent_tokens=sm_settings.keep_recent_tokens or 20000,
        )
