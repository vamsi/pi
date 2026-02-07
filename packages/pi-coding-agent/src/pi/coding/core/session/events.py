"""Agent session event types.

Custom events emitted by AgentSession to subscribers, distinct from
the agent-level events in pi.agent.types and extension events.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from pi.coding.core.compaction.compact import CompactionResult


@dataclass
class AgentSessionEvent:
    """Base class for agent session events."""

    type: str


@dataclass
class AutoCompactionStartEvent(AgentSessionEvent):
    """Emitted when auto-compaction begins."""

    reason: Literal["threshold", "overflow"] = "threshold"
    type: str = "auto_compaction_start"


@dataclass
class AutoCompactionEndEvent(AgentSessionEvent):
    """Emitted when auto-compaction finishes."""

    result: CompactionResult | None = None
    aborted: bool = False
    will_retry: bool = False
    error_message: str | None = None
    type: str = "auto_compaction_end"


@dataclass
class AutoRetryStartEvent(AgentSessionEvent):
    """Emitted when an auto-retry attempt begins."""

    attempt: int = 0
    max_attempts: int = 0
    delay_ms: int = 0
    error_message: str = ""
    type: str = "auto_retry_start"


@dataclass
class AutoRetryEndEvent(AgentSessionEvent):
    """Emitted when the auto-retry sequence ends."""

    success: bool = False
    attempt: int = 0
    final_error: str | None = None
    type: str = "auto_retry_end"


@dataclass
class SessionSwitchedEvent(AgentSessionEvent):
    """Emitted when the session is switched."""

    session_path: str = ""
    reason: str = ""
    type: str = "session_switched"


@dataclass
class SessionForkedEvent(AgentSessionEvent):
    """Emitted when a session fork occurs."""

    entry_id: str = ""
    new_session_path: str = ""
    type: str = "session_forked"


# Union of session events and agent events for subscriber callbacks
AgentSessionOrAgentEvent = AgentSessionEvent | Any  # AgentEvent at runtime
