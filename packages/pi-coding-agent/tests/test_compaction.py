"""Tests for the compaction system."""

from __future__ import annotations

from pi.coding.core.compaction.compact import (
    DEFAULT_COMPACTION_SETTINGS,
    CompactionSettings,
    CutPointResult,
    calculate_context_tokens_from_dict,
    estimate_context_tokens,
    estimate_entry_tokens,
    estimate_tokens,
    find_cut_point,
    find_valid_cut_points,
    prepare_compaction,
    should_compact,
)
from pi.coding.core.compaction.utils import (
    FileOperations,
    compute_file_lists,
    create_file_ops,
    extract_file_ops_from_message,
    format_file_operations,
    serialize_conversation,
)

# --- Token estimation ---


def test_estimate_tokens_text():
    msg = {"role": "user", "content": "Hello, world!"}
    tokens = estimate_tokens(msg)
    assert tokens == len("Hello, world!") // 4


def test_estimate_tokens_content_list():
    msg = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "This is a response."},
            {"type": "thinking", "thinking": "Let me think about this."},
        ],
    }
    tokens = estimate_tokens(msg)
    expected = (len("This is a response.") + len("Let me think about this.")) // 4
    assert tokens == expected


def test_estimate_tokens_tool_call():
    msg = {
        "role": "assistant",
        "content": [
            {
                "type": "tool_call",
                "name": "read",
                "arguments": {"file_path": "/test.py"},
            },
        ],
    }
    tokens = estimate_tokens(msg)
    assert tokens > 0


def test_estimate_tokens_image():
    msg = {
        "role": "assistant",
        "content": [{"type": "image", "data": "base64..."}],
    }
    tokens = estimate_tokens(msg)
    assert tokens == 4800 // 4  # IMAGE_ESTIMATED_CHARS / 4


def test_estimate_tokens_empty():
    assert estimate_tokens({"role": "user", "content": ""}) == 0
    assert estimate_tokens({}) == 0


def test_estimate_entry_tokens_message():
    entry = {"type": "message", "message": {"role": "user", "content": "test message"}}
    tokens = estimate_entry_tokens(entry)
    assert tokens == len("test message") // 4


def test_estimate_entry_tokens_compaction():
    entry = {"type": "compaction", "summary": "This is a summary."}
    tokens = estimate_entry_tokens(entry)
    assert tokens == len("This is a summary.") // 4


def test_estimate_entry_tokens_unknown():
    entry = {"type": "thinking_level_change", "thinkingLevel": "high"}
    assert estimate_entry_tokens(entry) == 0


# --- Context token estimation ---


def test_calculate_context_tokens_from_dict():
    usage = {"totalTokens": 5000}
    assert calculate_context_tokens_from_dict(usage) == 5000

    usage2 = {"input": 1000, "output": 500, "cacheRead": 200, "cacheWrite": 100}
    assert calculate_context_tokens_from_dict(usage2) == 1800


def test_estimate_context_tokens_with_usage():
    entries = [
        {"type": "message", "message": {"role": "user", "content": "hello"}},
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "content": "hi",
                "usage": {"totalTokens": 1000, "input": 800, "output": 200},
            },
        },
        {"type": "message", "message": {"role": "user", "content": "follow up question"}},
    ]
    estimate = estimate_context_tokens(entries)
    assert estimate.usage_tokens == 1000
    assert estimate.last_usage_index == 1
    assert estimate.trailing_tokens > 0
    assert estimate.tokens > 1000


def test_estimate_context_tokens_no_usage():
    entries = [
        {"type": "message", "message": {"role": "user", "content": "hello"}},
        {"type": "message", "message": {"role": "assistant", "content": "hi"}},
    ]
    estimate = estimate_context_tokens(entries)
    assert estimate.tokens > 0
    assert estimate.last_usage_index is None


def test_estimate_context_tokens_empty():
    estimate = estimate_context_tokens([])
    assert estimate.tokens == 0


# --- Should compact ---


def test_should_compact_true():
    assert should_compact(190000, 200000, CompactionSettings(reserve_tokens=16384))


def test_should_compact_false():
    assert not should_compact(100000, 200000, CompactionSettings(reserve_tokens=16384))


def test_should_compact_disabled():
    assert not should_compact(190000, 200000, CompactionSettings(enabled=False))


def test_should_compact_no_window():
    assert not should_compact(190000, 0, CompactionSettings())


# --- Cut point detection ---


def _make_entries(roles: list[str]) -> list[dict]:
    return [
        {"type": "message", "id": f"e{i}", "message": {"role": role, "content": f"msg {i}" * 100}}
        for i, role in enumerate(roles)
    ]


def test_find_valid_cut_points():
    entries = _make_entries(["user", "assistant", "tool_result", "user", "assistant"])
    valid = find_valid_cut_points(entries, 0, len(entries))
    # tool_result at index 2 should not be valid
    assert 2 not in valid
    assert 0 in valid  # user
    assert 1 in valid  # assistant
    assert 3 in valid  # user
    assert 4 in valid  # assistant


def test_find_cut_point_basic():
    entries = _make_entries(["user", "assistant", "user", "assistant", "user", "assistant"])
    cut = find_cut_point(entries, 0, len(entries), keep_tokens=100)
    assert isinstance(cut, CutPointResult)
    assert cut.first_kept_entry_index >= 0


def test_find_cut_point_empty():
    cut = find_cut_point([], 0, 0, keep_tokens=100)
    assert cut.first_kept_entry_index == 0


def test_find_cut_point_small_keep():
    # Very small keep_tokens means we want to keep very little
    entries = _make_entries(["user", "assistant"] * 10)
    cut = find_cut_point(entries, 0, len(entries), keep_tokens=1)
    # Should keep almost nothing (cut near the end)
    assert cut.first_kept_entry_index > 0


# --- Preparation ---


def test_prepare_compaction():
    entries = _make_entries(["user", "assistant"] * 5)
    # Give each an ID
    for i, e in enumerate(entries):
        e["id"] = f"entry_{i}"

    prep = prepare_compaction(entries, CompactionSettings(keep_recent_tokens=200))
    assert prep is not None
    assert len(prep.keep_entries) > 0
    assert len(prep.discard_entries) > 0
    assert prep.context_tokens > 0


def test_prepare_compaction_too_few_entries():
    entries = [{"type": "message", "id": "e0", "message": {"role": "user", "content": "hi"}}]
    prep = prepare_compaction(entries)
    assert prep is None


def test_prepare_compaction_no_cut_needed():
    # Very short entries with large keep budget
    entries = [
        {"type": "message", "id": "e0", "message": {"role": "user", "content": "hi"}},
        {"type": "message", "id": "e1", "message": {"role": "assistant", "content": "hello"}},
    ]
    prep = prepare_compaction(entries, CompactionSettings(keep_recent_tokens=100000))
    # Should return None since cut_point would be at 0
    assert prep is None


# --- File operations ---


def test_create_file_ops():
    ops = create_file_ops()
    assert isinstance(ops, FileOperations)
    assert len(ops.read) == 0
    assert len(ops.written) == 0
    assert len(ops.edited) == 0


def test_extract_file_ops():
    msg = {
        "content": [
            {"type": "tool_call", "name": "read", "arguments": {"file_path": "/a.py"}},
            {"type": "tool_call", "name": "write", "arguments": {"file_path": "/b.py"}},
            {"type": "tool_call", "name": "edit", "arguments": {"file_path": "/c.py"}},
            {"type": "tool_call", "name": "read", "arguments": {"file_path": "/b.py"}},
        ]
    }
    ops = create_file_ops()
    extract_file_ops_from_message(msg, ops)
    assert "/a.py" in ops.read
    assert "/b.py" in ops.written
    assert "/b.py" in ops.read
    assert "/c.py" in ops.edited


def test_compute_file_lists():
    ops = FileOperations(read={"/a.py", "/b.py"}, written={"/b.py"}, edited={"/c.py"})
    read_only, modified = compute_file_lists(ops)
    assert read_only == ["/a.py"]  # b.py was modified
    assert modified == ["/b.py", "/c.py"]


def test_format_file_operations():
    text = format_file_operations(["/a.py"], ["/b.py"])
    assert "<read-files>" in text
    assert "/a.py" in text
    assert "<modified-files>" in text
    assert "/b.py" in text


def test_format_file_operations_empty():
    assert format_file_operations([], []) == ""


# --- Serialization ---


def test_serialize_conversation_text():
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    text = serialize_conversation(messages)
    assert "[user]" in text
    assert "Hello" in text
    assert "[assistant]" in text
    assert "Hi there" in text


def test_serialize_conversation_content_list():
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Response"},
                {"type": "tool_call", "name": "read", "arguments": {"path": "/f.py"}},
            ],
        },
    ]
    text = serialize_conversation(messages)
    assert "Response" in text
    assert "tool_call" in text
    assert "read" in text


def test_serialize_conversation_empty():
    assert serialize_conversation([]) == ""


# --- Default settings ---


def test_default_settings():
    assert DEFAULT_COMPACTION_SETTINGS.enabled is True
    assert DEFAULT_COMPACTION_SETTINGS.reserve_tokens == 16384
    assert DEFAULT_COMPACTION_SETTINGS.keep_recent_tokens == 20000
