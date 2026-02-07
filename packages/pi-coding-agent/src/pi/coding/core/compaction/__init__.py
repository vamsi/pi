"""Context compaction system for managing conversation history size.

Provides automatic and manual compaction of conversation history using
LLM-generated summaries, with configurable token thresholds and
file operation tracking.
"""

from pi.coding.core.compaction.compact import (
    DEFAULT_COMPACTION_SETTINGS,
    CompactionDetails,
    CompactionResult,
    CompactionSettings,
    ContextUsageEstimate,
    CutPointResult,
    calculate_context_tokens,
    compact,
    estimate_context_tokens,
    estimate_tokens,
    find_cut_point,
    prepare_compaction,
    should_compact,
)
from pi.coding.core.compaction.utils import (
    SUMMARIZATION_SYSTEM_PROMPT,
    FileOperations,
    compute_file_lists,
    create_file_ops,
    extract_file_ops_from_message,
    format_file_operations,
    serialize_conversation,
)

__all__ = [
    "DEFAULT_COMPACTION_SETTINGS",
    "SUMMARIZATION_SYSTEM_PROMPT",
    "CompactionDetails",
    "CompactionResult",
    "CompactionSettings",
    "ContextUsageEstimate",
    "CutPointResult",
    "FileOperations",
    "calculate_context_tokens",
    "compact",
    "compute_file_lists",
    "create_file_ops",
    "estimate_context_tokens",
    "estimate_tokens",
    "extract_file_ops_from_message",
    "find_cut_point",
    "format_file_operations",
    "prepare_compaction",
    "serialize_conversation",
    "should_compact",
]
