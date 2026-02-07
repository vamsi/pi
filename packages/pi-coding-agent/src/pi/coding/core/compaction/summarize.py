"""Branch summarization for preserving context when navigating tree branches.

When abandoning a conversation branch, generates a summary that captures
the key context of the abandoned path for use in the new branch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pi.coding.core.compaction.compact import (
    DEFAULT_COMPACTION_SETTINGS,
    estimate_entry_tokens,
)
from pi.coding.core.compaction.utils import (
    SUMMARIZATION_SYSTEM_PROMPT,
    serialize_conversation,
)

if TYPE_CHECKING:
    import asyncio

    from pi.ai.types import Model


# --- Types ---


@dataclass
class BranchSummaryDetails:
    """Details stored in a BranchSummaryEntry."""

    read_files: list[str] = field(default_factory=list)
    modified_files: list[str] = field(default_factory=list)


@dataclass
class BranchSummaryResult:
    """Result of generating a branch summary."""

    summary: str
    branch_entry_ids: list[str] = field(default_factory=list)
    details: BranchSummaryDetails | None = None


@dataclass
class BranchPreparation:
    """Preparation result for branch summarization."""

    entries: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    branch_entry_ids: list[str]
    total_tokens: int = 0


BRANCH_SUMMARIZATION_PROMPT = """\
Summarize the following conversation branch that is being abandoned.
The user is switching to a different conversation path, and this summary
will provide context about what was explored in this branch.

Focus on:
1. What was attempted in this branch
2. Key findings or results
3. Any important decisions or changes made
4. Why this path might have been abandoned (if apparent)

Keep the summary concise but informative.
"""


# --- Collection ---


def collect_entries_for_branch_summary(
    entries: list[dict[str, Any]],
    old_leaf_id: str,
    target_id: str,
    by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collect entries that are unique to the abandoned branch.

    Finds entries on the path from old_leaf_id back to the branch point
    (where the path diverges from the target_id path).
    """
    # Walk from old_leaf to root
    old_path: list[str] = []
    current: str | None = old_leaf_id
    while current is not None:
        old_path.append(current)
        entry = by_id.get(current)
        current = entry.get("parentId") if entry else None

    # Walk from target to root
    target_path: list[str] = []
    current = target_id
    while current is not None:
        target_path.append(current)
        entry = by_id.get(current)
        current = entry.get("parentId") if entry else None
    target_path_set = set(target_path)

    # Branch-only entries = entries in old path but not in target path
    branch_only_ids = [eid for eid in old_path if eid not in target_path_set]

    # Collect actual entries in order (reverse since we walked leaf→root)
    branch_entries: list[dict[str, Any]] = []
    for eid in reversed(branch_only_ids):
        entry = by_id.get(eid)
        if entry:
            branch_entries.append(entry)

    return branch_entries


def prepare_branch_entries(
    entries: list[dict[str, Any]],
    token_budget: int,
) -> BranchPreparation:
    """Prepare branch entries for summarization, respecting token budget.

    If entries exceed the budget, truncates from the middle,
    keeping the beginning and end for context.
    """
    messages: list[dict[str, Any]] = []
    branch_ids: list[str] = []
    total_tokens = 0

    for entry in entries:
        tokens = estimate_entry_tokens(entry)
        total_tokens += tokens
        branch_ids.append(entry.get("id", ""))

        if entry.get("type") == "message":
            messages.append(entry["message"])
        elif entry.get("type") == "compaction":
            messages.append(
                {
                    "role": "user",
                    "content": f"[Summary]\n{entry.get('summary', '')}",
                }
            )
        elif entry.get("type") == "branch_summary":
            messages.append(
                {
                    "role": "user",
                    "content": f"[Branch summary]\n{entry.get('summary', '')}",
                }
            )

    # Truncate if over budget
    if total_tokens > token_budget and len(messages) > 4:
        # Keep first 2 and last 2 messages
        kept = [*messages[:2], {"role": "user", "content": "[... middle of branch omitted ...]"}, *messages[-2:]]
        messages = kept

    return BranchPreparation(
        entries=entries,
        messages=messages,
        branch_entry_ids=branch_ids,
        total_tokens=total_tokens,
    )


# --- Summary generation ---


async def generate_branch_summary(
    preparation: BranchPreparation,
    model: Model,
    *,
    api_key: str | None = None,
    abort_event: asyncio.Event | None = None,
    reserve_tokens: int | None = None,
) -> BranchSummaryResult:
    """Generate a summary for an abandoned branch."""
    from pi.ai.stream import complete_simple

    if reserve_tokens is None:
        reserve_tokens = DEFAULT_COMPACTION_SETTINGS.reserve_tokens

    conversation_text = serialize_conversation(preparation.messages)
    prompt = BRANCH_SUMMARIZATION_PROMPT + "\n\n"
    prompt += f"<branch-conversation>\n{conversation_text}\n</branch-conversation>"

    max_tokens = int(reserve_tokens * 0.5)

    result = await complete_simple(
        model=model,
        system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
        prompt=prompt,
        reasoning="medium",
        max_tokens=max_tokens,
        api_key=api_key,
    )

    summary = ""
    for item in result.content:
        if hasattr(item, "text"):
            summary = item.text
            break

    return BranchSummaryResult(
        summary=summary,
        branch_entry_ids=preparation.branch_entry_ids,
    )
