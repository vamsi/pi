"""Main compaction logic: token estimation, cut point detection, and summary generation.

Handles both automatic (threshold-based) and manual compaction of
conversation history, generating LLM summaries of discarded content.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pi.coding.core.compaction.utils import (
    SUMMARIZATION_PROMPT,
    SUMMARIZATION_SYSTEM_PROMPT,
    TURN_PREFIX_SUMMARIZATION_PROMPT,
    UPDATE_SUMMARIZATION_PROMPT,
    FileOperations,
    compute_file_lists,
    create_file_ops,
    extract_file_ops_from_message,
    format_file_operations,
    serialize_conversation,
)

if TYPE_CHECKING:
    import asyncio

    from pi.ai.types import Model, Usage

# --- Constants ---

IMAGE_ESTIMATED_CHARS = 4800  # ~1,200 tokens at 4 chars/token

# Valid entry types for cut points
VALID_CUT_TYPES = {"user", "assistant", "custom", "custom_message", "branch_summary"}


# --- Settings ---


@dataclass
class CompactionSettings:
    """Configuration for context compaction."""

    enabled: bool = True
    reserve_tokens: int = 16384
    keep_recent_tokens: int = 20000


DEFAULT_COMPACTION_SETTINGS = CompactionSettings()


# --- Result types ---


@dataclass
class CompactionDetails:
    """File tracking metadata stored in CompactionEntry.details."""

    read_files: list[str] = field(default_factory=list)
    modified_files: list[str] = field(default_factory=list)


@dataclass
class CompactionResult:
    """Result from a compaction operation."""

    summary: str
    first_kept_entry_id: str
    tokens_before: int
    details: CompactionDetails | None = None


@dataclass
class ContextUsageEstimate:
    """Token usage estimate for the current context."""

    tokens: int = 0
    usage_tokens: int = 0
    trailing_tokens: int = 0
    last_usage_index: int | None = None


@dataclass
class CutPointResult:
    """Result of finding a cut point for compaction."""

    first_kept_entry_index: int
    turn_start_index: int = -1
    is_split_turn: bool = False


# --- Token estimation ---


def calculate_context_tokens(usage: Usage) -> int:
    """Calculate actual context tokens from LLM usage data."""
    total = usage.total_tokens
    if total:
        return total
    return usage.input + usage.output + usage.cache_read + usage.cache_write


def estimate_tokens_from_text(text: str) -> int:
    """Estimate tokens from text using character-based heuristic (chars / 4)."""
    return len(text) // 4


def estimate_tokens(message: dict[str, Any]) -> int:
    """Estimate token count for a single message.

    Uses character-based heuristic: chars / 4 (conservative estimate).
    """
    content = message.get("content", "")
    total_chars = 0

    if isinstance(content, str):
        total_chars = len(content)
    elif isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type", "")

            if item_type == "text":
                total_chars += len(item.get("text", ""))
            elif item_type == "thinking":
                total_chars += len(item.get("thinking", ""))
            elif item_type == "tool_call":
                total_chars += len(item.get("name", ""))
                total_chars += len(str(item.get("arguments", {})))
            elif item_type == "image":
                total_chars += IMAGE_ESTIMATED_CHARS

    return total_chars // 4


def estimate_entry_tokens(entry: dict[str, Any]) -> int:
    """Estimate tokens for a session entry."""
    entry_type = entry.get("type", "")

    if entry_type == "message":
        return estimate_tokens(entry.get("message", {}))
    elif entry_type == "compaction" or entry_type == "branch_summary":
        return estimate_tokens_from_text(entry.get("summary", ""))
    elif entry_type == "custom_message":
        content = entry.get("content", [])
        total = 0
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                total += len(item.get("text", ""))
        return total // 4

    return 0


# --- Context usage estimation ---


def estimate_context_tokens(entries: list[dict[str, Any]]) -> ContextUsageEstimate:
    """Estimate total context tokens, using actual usage when available.

    Prefers actual usage from the last assistant message, then estimates
    trailing messages that came after.
    """
    if not entries:
        return ContextUsageEstimate()

    # Find last assistant message with usage data
    last_usage_index: int | None = None
    usage_tokens = 0

    for i in range(len(entries) - 1, -1, -1):
        entry = entries[i]
        if entry.get("type") != "message":
            continue
        msg = entry.get("message", {})
        if msg.get("role") != "assistant":
            continue
        usage = msg.get("usage")
        if usage and (usage.get("totalTokens", 0) or usage.get("input", 0)):
            last_usage_index = i
            usage_tokens = calculate_context_tokens_from_dict(usage)
            break

    if last_usage_index is not None:
        # Estimate trailing messages after last usage
        trailing = 0
        for entry in entries[last_usage_index + 1 :]:
            trailing += estimate_entry_tokens(entry)
        return ContextUsageEstimate(
            tokens=usage_tokens + trailing,
            usage_tokens=usage_tokens,
            trailing_tokens=trailing,
            last_usage_index=last_usage_index,
        )

    # No usage data - pure estimation
    total = sum(estimate_entry_tokens(e) for e in entries)
    return ContextUsageEstimate(tokens=total)


def calculate_context_tokens_from_dict(usage: dict[str, Any]) -> int:
    """Calculate context tokens from a usage dict."""
    total = usage.get("totalTokens", 0) or usage.get("total_tokens", 0)
    if total:
        return total
    return usage.get("input", 0) + usage.get("output", 0) + usage.get("cacheRead", 0) + usage.get("cacheWrite", 0)


# --- Compaction threshold ---


def should_compact(context_tokens: int, context_window: int, settings: CompactionSettings) -> bool:
    """Check if context tokens exceed the compaction threshold."""
    if not settings.enabled or context_window <= 0:
        return False
    return context_tokens > context_window - settings.reserve_tokens


# --- Cut point detection ---


def find_valid_cut_points(
    entries: list[dict[str, Any]],
    start: int,
    end: int,
) -> list[int]:
    """Find valid indices where compaction can cut.

    Valid cut points are entries of type: user, assistant, custom,
    custom_message, branch_summary. Never cuts at tool results
    (must follow their tool call).
    """
    valid: list[int] = []
    for i in range(start, min(end, len(entries))):
        entry = entries[i]
        entry_type = entry.get("type", "")

        if entry_type == "message":
            role = entry.get("message", {}).get("role", "")
            if role in VALID_CUT_TYPES:
                valid.append(i)
        elif entry_type in ("custom", "custom_message", "branch_summary"):
            valid.append(i)

    return valid


def find_turn_start_index(
    entries: list[dict[str, Any]],
    index: int,
    start: int,
) -> int:
    """Find the user message that started the turn containing the given index."""
    for i in range(index, start - 1, -1):
        entry = entries[i]
        if entry.get("type") == "message":
            msg = entry.get("message", {})
            if msg.get("role") == "user":
                return i
    return -1


def find_cut_point(
    entries: list[dict[str, Any]],
    start: int,
    end: int,
    keep_tokens: int,
) -> CutPointResult:
    """Find the optimal cut point for compaction.

    Walks backwards from the end, accumulating token estimates.
    When accumulated tokens >= keep_tokens, finds the nearest
    valid cut point.
    """
    if not entries or start >= end:
        return CutPointResult(first_kept_entry_index=start)

    valid_points = find_valid_cut_points(entries, start, end)
    if not valid_points:
        return CutPointResult(first_kept_entry_index=start)

    # Walk backwards accumulating tokens
    accumulated = 0
    cut_index = start

    for i in range(end - 1, start - 1, -1):
        accumulated += estimate_entry_tokens(entries[i])
        if accumulated >= keep_tokens:
            # Find closest valid cut point at or after this index
            for vp in valid_points:
                if vp >= i:
                    cut_index = vp
                    break
            else:
                cut_index = valid_points[-1]
            break
    else:
        # Didn't reach threshold - keep everything from start
        cut_index = start

    # Check if this splits a turn
    turn_start = find_turn_start_index(entries, cut_index, start)
    is_split = turn_start != -1 and turn_start < cut_index

    return CutPointResult(
        first_kept_entry_index=cut_index,
        turn_start_index=turn_start if is_split else -1,
        is_split_turn=is_split,
    )


# --- File operations extraction ---


def _extract_file_ops(entries: list[dict[str, Any]], start: int, end: int) -> FileOperations:
    """Extract file operations from a range of entries."""
    ops = create_file_ops()
    for entry in entries[start:end]:
        if entry.get("type") == "message":
            msg = entry.get("message", {})
            if msg.get("role") == "assistant":
                extract_file_ops_from_message(msg, ops)
        elif entry.get("type") == "compaction":
            # Accumulate from previous compaction details
            details = entry.get("details")
            if isinstance(details, dict):
                for f in details.get("readFiles", details.get("read_files", [])):
                    ops.read.add(f)
                for f in details.get("modifiedFiles", details.get("modified_files", [])):
                    ops.written.add(f)
    return ops


# --- Preparation ---


@dataclass
class CompactionPreparation:
    """Preparation result for compaction."""

    entries: list[dict[str, Any]]
    cut_point: CutPointResult
    context_tokens: int
    keep_entries: list[dict[str, Any]]
    discard_entries: list[dict[str, Any]]
    previous_summary: str | None = None
    file_ops: FileOperations = field(default_factory=create_file_ops)


def prepare_compaction(
    entries: list[dict[str, Any]],
    settings: CompactionSettings | None = None,
) -> CompactionPreparation | None:
    """Analyze entries and prepare for compaction.

    Returns None if there's nothing to compact.
    """
    if settings is None:
        settings = DEFAULT_COMPACTION_SETTINGS

    if len(entries) < 2:
        return None

    # Find cut point
    cut = find_cut_point(entries, 0, len(entries), settings.keep_recent_tokens)

    if cut.first_kept_entry_index <= 0:
        return None

    # Split entries
    discard = entries[: cut.first_kept_entry_index]
    keep = entries[cut.first_kept_entry_index :]

    # Extract file operations from discarded entries
    file_ops = _extract_file_ops(entries, 0, cut.first_kept_entry_index)

    # Find previous compaction summary
    previous_summary: str | None = None
    for entry in reversed(discard):
        if entry.get("type") == "compaction":
            previous_summary = entry.get("summary")
            break

    # Estimate tokens
    estimate = estimate_context_tokens(entries)

    return CompactionPreparation(
        entries=entries,
        cut_point=cut,
        context_tokens=estimate.tokens,
        keep_entries=keep,
        discard_entries=discard,
        previous_summary=previous_summary,
        file_ops=file_ops,
    )


# --- Summary generation ---


async def generate_summary(
    messages: list[dict[str, Any]],
    model: Model,
    reserve_tokens: int,
    *,
    api_key: str | None = None,
    abort_event: asyncio.Event | None = None,
    custom_instructions: str | None = None,
    previous_summary: str | None = None,
) -> str:
    """Generate a compaction summary using an LLM.

    Uses the conversation history and optionally a previous summary
    to produce a structured summary.
    """
    from pi.ai.stream import complete_simple

    # Build the prompt
    conversation_text = serialize_conversation(messages)

    if previous_summary:
        prompt = UPDATE_SUMMARIZATION_PROMPT + "\n\n"
        prompt += f"<previous-summary>\n{previous_summary}\n</previous-summary>\n\n"
        prompt += f"<conversation>\n{conversation_text}\n</conversation>"
    else:
        prompt = SUMMARIZATION_PROMPT + "\n\n"
        prompt += f"<conversation>\n{conversation_text}\n</conversation>"

    if custom_instructions:
        prompt += f"\n\nAdditional instructions: {custom_instructions}"

    # Calculate max tokens for summary (80% of reserve)
    max_summary_tokens = int(reserve_tokens * 0.8)

    result = await complete_simple(
        model=model,
        system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
        prompt=prompt,
        reasoning="high",
        max_tokens=max_summary_tokens,
        api_key=api_key,
    )

    # Extract text from result
    for item in result.content:
        if hasattr(item, "text"):
            return item.text

    return ""


async def generate_turn_prefix_summary(
    messages: list[dict[str, Any]],
    model: Model,
    reserve_tokens: int,
    *,
    api_key: str | None = None,
) -> str:
    """Generate a summary for the prefix of a split turn."""
    from pi.ai.stream import complete_simple

    conversation_text = serialize_conversation(messages)
    prompt = TURN_PREFIX_SUMMARIZATION_PROMPT + "\n\n"
    prompt += f"<conversation>\n{conversation_text}\n</conversation>"

    max_tokens = int(reserve_tokens * 0.3)

    result = await complete_simple(
        model=model,
        system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
        prompt=prompt,
        reasoning="medium",
        max_tokens=max_tokens,
        api_key=api_key,
    )

    for item in result.content:
        if hasattr(item, "text"):
            return item.text

    return ""


# --- Main compaction ---


async def compact(
    preparation: CompactionPreparation,
    model: Model,
    *,
    api_key: str | None = None,
    custom_instructions: str | None = None,
    abort_event: asyncio.Event | None = None,
) -> CompactionResult:
    """Execute compaction: generate summary and return result.

    The caller is responsible for appending the CompactionEntry
    to the session manager.
    """
    # Build messages from discarded entries
    discard_messages: list[dict[str, Any]] = []
    for entry in preparation.discard_entries:
        if entry.get("type") == "message":
            discard_messages.append(entry["message"])
        elif entry.get("type") == "compaction":
            # Include previous summary as context
            discard_messages.append(
                {
                    "role": "user",
                    "content": f"[Previous summary]\n{entry.get('summary', '')}",
                }
            )

    # Handle split turn - summarize prefix separately
    if preparation.cut_point.is_split_turn and preparation.cut_point.turn_start_index >= 0:
        turn_start = preparation.cut_point.turn_start_index
        prefix_messages: list[dict[str, Any]] = []
        for entry in preparation.entries[turn_start : preparation.cut_point.first_kept_entry_index]:
            if entry.get("type") == "message":
                prefix_messages.append(entry["message"])

        if prefix_messages:
            prefix_summary = await generate_turn_prefix_summary(
                prefix_messages,
                model,
                DEFAULT_COMPACTION_SETTINGS.reserve_tokens,
                api_key=api_key,
            )
            if prefix_summary:
                discard_messages.append(
                    {
                        "role": "user",
                        "content": f"[Turn prefix summary]\n{prefix_summary}",
                    }
                )

    # Generate summary
    summary = await generate_summary(
        discard_messages,
        model,
        DEFAULT_COMPACTION_SETTINGS.reserve_tokens,
        api_key=api_key,
        abort_event=abort_event,
        custom_instructions=custom_instructions,
        previous_summary=preparation.previous_summary,
    )

    # Compute file operation details
    read_files, modified_files = compute_file_lists(preparation.file_ops)
    details = CompactionDetails(read_files=read_files, modified_files=modified_files)

    # Add file operations to summary
    file_ops_text = format_file_operations(read_files, modified_files)
    if file_ops_text:
        summary += "\n\n" + file_ops_text

    # Get first kept entry ID
    first_kept = preparation.keep_entries[0] if preparation.keep_entries else None
    first_kept_id = first_kept.get("id", "") if first_kept else ""

    return CompactionResult(
        summary=summary,
        first_kept_entry_id=first_kept_id,
        tokens_before=preparation.context_tokens,
        details=details,
    )
