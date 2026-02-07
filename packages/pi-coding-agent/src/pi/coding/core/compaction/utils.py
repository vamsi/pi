"""Compaction utilities: file tracking, message serialization, prompts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# --- System prompt for summarization ---

SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a context summarization assistant. Your task is to read a conversation "
    "between a user and an AI coding assistant, then produce a structured summary following "
    "the exact format specified. Do NOT continue the conversation."
)

# --- Summarization prompts ---

SUMMARIZATION_PROMPT = """\
Summarize the following conversation between a user and an AI coding assistant.
Produce a structured summary with EXACTLY these sections:

## Goal
What is the user trying to accomplish?

## Constraints & Preferences
Any requirements, constraints, or preferences mentioned.

## Progress
### Done
- Completed items

### In Progress
- Items being worked on

### Blocked
- Items that are stuck

## Key Decisions
Important decisions made during the conversation.

## Next Steps
What should happen next.

## Critical Context
Any other information essential to continuing the work.
"""

UPDATE_SUMMARIZATION_PROMPT = """\
You are given a previous summary and new conversation turns since that summary was created.
Update the summary by:
1. PRESERVING all existing information from the previous summary
2. ADDING new progress, decisions, and context from the new turns
3. UPDATING the Progress section (move items to "Done" if completed)
4. UPDATING "Next Steps" based on new accomplishments
5. Keeping the EXACT SAME format as the previous summary

Output ONLY the updated summary, nothing else.
"""

TURN_PREFIX_SUMMARIZATION_PROMPT = """\
Summarize the beginning of this conversation turn that is being split.
Produce a brief summary with these sections:

## Original Request
What the user asked for in this turn.

## Early Progress
What was accomplished before the split point.

## Context for Suffix
Key information needed to understand the remaining messages.
"""


# --- File operations tracking ---


@dataclass
class FileOperations:
    """Tracks file read/write/edit operations during a session."""

    read: set[str] = field(default_factory=set)
    written: set[str] = field(default_factory=set)
    edited: set[str] = field(default_factory=set)


def create_file_ops() -> FileOperations:
    """Create empty file operations tracker."""
    return FileOperations()


def extract_file_ops_from_message(message: dict[str, Any], file_ops: FileOperations) -> None:
    """Extract file operations from an assistant message's tool calls."""
    content = message.get("content", [])
    if not isinstance(content, list):
        return

    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "tool_call":
            continue

        name = item.get("name", "")
        args = item.get("arguments", {})
        if not isinstance(args, dict):
            continue

        path = args.get("file_path", "") or args.get("path", "")
        if not path:
            continue

        if name == "read":
            file_ops.read.add(path)
        elif name == "write":
            file_ops.written.add(path)
        elif name == "edit":
            file_ops.edited.add(path)


def compute_file_lists(file_ops: FileOperations) -> tuple[list[str], list[str]]:
    """Compute read-only and modified file lists from operations.

    Returns (read_only_files, modified_files) both sorted alphabetically.
    """
    modified = file_ops.written | file_ops.edited
    read_only = file_ops.read - modified
    return sorted(read_only), sorted(modified)


def format_file_operations(read_files: list[str], modified_files: list[str]) -> str:
    """Format file operations as XML tags for inclusion in summaries."""
    parts: list[str] = []

    if read_files:
        parts.append("<read-files>")
        parts.extend(read_files)
        parts.append("</read-files>")

    if modified_files:
        parts.append("<modified-files>")
        parts.extend(modified_files)
        parts.append("</modified-files>")

    return "\n".join(parts)


# --- Message serialization ---


def serialize_conversation(messages: list[dict[str, Any]]) -> str:
    """Serialize messages into a text format for summarization.

    Produces a readable text representation of the conversation
    suitable for passing to a summarization LLM.
    """
    parts: list[str] = []

    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        if isinstance(content, str):
            parts.append(f"[{role}]\n{content}")
        elif isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type", "")

                if item_type == "text":
                    text_parts.append(item.get("text", ""))
                elif item_type == "thinking":
                    text_parts.append(f"<thinking>{item.get('thinking', '')}</thinking>")
                elif item_type == "tool_call":
                    name = item.get("name", "")
                    args = item.get("arguments", {})
                    text_parts.append(f"<tool_call name='{name}'>{_truncate_args(args)}</tool_call>")
                elif item_type == "image":
                    text_parts.append("[image]")

            if text_parts:
                parts.append(f"[{role}]\n" + "\n".join(text_parts))
        else:
            parts.append(f"[{role}]\n{content}")

    return "\n\n".join(parts)


def _truncate_args(args: Any, max_len: int = 500) -> str:
    """Truncate tool arguments for serialization."""
    text = str(args)
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text
