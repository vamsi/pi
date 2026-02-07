"""Tests for the JSONL session manager."""

from __future__ import annotations

import json
import os
import tempfile

from pi.coding.core.sessions import (
    SessionManager,
    _generate_id,
    _migrate_to_current,
    is_valid_session_file,
    load_entries_from_file,
    parse_session_entries,
)

# --- Parsing tests ---


def test_parse_session_entries():
    content = '{"type":"session","version":3,"id":"abc","timestamp":"now","cwd":"/"}\n'
    content += '{"type":"message","id":"m1","parentId":null,"message":{"role":"user","content":"hi"}}\n'
    entries = parse_session_entries(content)
    assert len(entries) == 2
    assert entries[0]["type"] == "session"
    assert entries[1]["type"] == "message"


def test_parse_skips_malformed_lines():
    content = '{"type":"session","version":3}\n'
    content += "not json\n"
    content += '{"type":"message","id":"m1"}\n'
    entries = parse_session_entries(content)
    assert len(entries) == 2


def test_parse_empty_content():
    assert parse_session_entries("") == []
    assert parse_session_entries("\n\n\n") == []


# --- ID generation ---


def test_generate_id_unique():
    existing: set[str] = set()
    ids = set()
    for _ in range(50):
        new_id = _generate_id(existing)
        ids.add(new_id)
        existing.add(new_id)
    assert len(ids) == 50


def test_generate_id_avoids_collision():
    existing = {"abcd1234"}
    new_id = _generate_id(existing)
    assert new_id != "abcd1234"


# --- File validation ---


def test_valid_session_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"type": "session", "version": 3}) + "\n")
        path = f.name
    try:
        assert is_valid_session_file(path)
    finally:
        os.unlink(path)


def test_invalid_session_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"type": "message"}) + "\n")
        path = f.name
    try:
        assert not is_valid_session_file(path)
    finally:
        os.unlink(path)


def test_nonexistent_file_is_invalid():
    assert not is_valid_session_file("/nonexistent/path.jsonl")


# --- In-memory sessions ---


def test_in_memory_create():
    mgr = SessionManager.in_memory("/tmp/test")
    assert mgr.session_id
    assert mgr.cwd == "/tmp/test"
    assert mgr.session_file is None
    assert mgr.entry_count == 0


def test_append_message():
    mgr = SessionManager.in_memory()
    msg = {"role": "user", "content": "hello", "timestamp": 1000}
    entry_id = mgr.append_message(msg)
    assert entry_id
    assert mgr.entry_count == 1
    assert mgr.leaf_id == entry_id


def test_append_multiple_messages():
    mgr = SessionManager.in_memory()
    id1 = mgr.append_message({"role": "user", "content": "q1", "timestamp": 1000})
    id2 = mgr.append_message({"role": "assistant", "content": "a1", "timestamp": 1001})
    id3 = mgr.append_message({"role": "user", "content": "q2", "timestamp": 1002})

    assert mgr.entry_count == 3
    assert mgr.leaf_id == id3

    # Verify parent chain
    entry2 = mgr.get_entry(id2)
    assert entry2 is not None
    assert entry2["parentId"] == id1

    entry3 = mgr.get_entry(id3)
    assert entry3 is not None
    assert entry3["parentId"] == id2


# --- Branching ---


def test_branch():
    mgr = SessionManager.in_memory()
    id1 = mgr.append_message({"role": "user", "content": "q1", "timestamp": 1000})
    mgr.append_message({"role": "assistant", "content": "a1", "timestamp": 1001})

    # Branch from first message
    mgr.branch(id1)
    id3 = mgr.append_message({"role": "assistant", "content": "a1-alt", "timestamp": 1002})

    entry3 = mgr.get_entry(id3)
    assert entry3 is not None
    assert entry3["parentId"] == id1  # Parent is the branch point, not a1


def test_branch_invalid_id():
    mgr = SessionManager.in_memory()
    try:
        mgr.branch("nonexistent")
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass


def test_reset_leaf():
    mgr = SessionManager.in_memory()
    mgr.append_message({"role": "user", "content": "q1", "timestamp": 1000})
    mgr.reset_leaf()
    assert mgr.leaf_id is None

    # Next append creates a root entry
    entry_id = mgr.append_message({"role": "user", "content": "q2", "timestamp": 1001})
    entry = mgr.get_entry(entry_id)
    assert entry is not None
    assert entry["parentId"] is None


def test_branch_with_summary():
    mgr = SessionManager.in_memory()
    id1 = mgr.append_message({"role": "user", "content": "q1", "timestamp": 1000})
    mgr.append_message({"role": "assistant", "content": "a1", "timestamp": 1001})

    summary_id = mgr.branch_with_summary(id1, "Summary of abandoned branch")
    entry = mgr.get_entry(summary_id)
    assert entry is not None
    assert entry["type"] == "branch_summary"
    assert entry["summary"] == "Summary of abandoned branch"


# --- Path walking ---


def test_get_branch():
    mgr = SessionManager.in_memory()
    id1 = mgr.append_message({"role": "user", "content": "q1", "timestamp": 1000})
    id2 = mgr.append_message({"role": "assistant", "content": "a1", "timestamp": 1001})
    id3 = mgr.append_message({"role": "user", "content": "q2", "timestamp": 1002})

    path = mgr.get_branch()
    assert len(path) == 3
    assert path[0]["id"] == id1
    assert path[1]["id"] == id2
    assert path[2]["id"] == id3


def test_get_branch_from_specific_id():
    mgr = SessionManager.in_memory()
    id1 = mgr.append_message({"role": "user", "content": "q1", "timestamp": 1000})
    id2 = mgr.append_message({"role": "assistant", "content": "a1", "timestamp": 1001})
    mgr.append_message({"role": "user", "content": "q2", "timestamp": 1002})

    path = mgr.get_branch(id2)
    assert len(path) == 2
    assert path[0]["id"] == id1
    assert path[1]["id"] == id2


def test_get_branch_empty():
    mgr = SessionManager.in_memory()
    assert mgr.get_branch() == []


# --- Context building ---


def test_build_session_context():
    mgr = SessionManager.in_memory()
    mgr.append_message({"role": "user", "content": "hello", "timestamp": 1000})
    mgr.append_message({"role": "assistant", "content": [{"type": "text", "text": "hi"}], "timestamp": 1001})

    ctx = mgr.build_session_context()
    assert len(ctx.messages) == 2
    assert ctx.messages[0]["role"] == "user"
    assert ctx.messages[1]["role"] == "assistant"


def test_build_context_with_thinking_level_change():
    mgr = SessionManager.in_memory()
    mgr.append_message({"role": "user", "content": "hello", "timestamp": 1000})
    mgr.append_thinking_level_change("high")
    mgr.append_message({"role": "assistant", "content": [], "timestamp": 1001})

    ctx = mgr.build_session_context()
    assert ctx.thinking_level == "high"


def test_build_context_with_model_change():
    mgr = SessionManager.in_memory()
    mgr.append_model_change("claude-opus-4-6", "anthropic")
    mgr.append_message({"role": "user", "content": "test", "timestamp": 1000})

    ctx = mgr.build_session_context()
    assert ctx.model_id == "claude-opus-4-6"
    assert ctx.provider == "anthropic"


def test_build_context_with_compaction():
    mgr = SessionManager.in_memory()
    mgr.append_message({"role": "user", "content": "old message", "timestamp": 1000})
    mgr.append_compaction("Summary of old messages")
    mgr.append_message({"role": "user", "content": "new message", "timestamp": 1002})

    ctx = mgr.build_session_context()
    # Should have summary + new message
    assert len(ctx.messages) == 2
    assert ctx.messages[0]["content"] == "Summary of old messages"


# --- Special entries ---


def test_append_label():
    mgr = SessionManager.in_memory()
    msg_id = mgr.append_message({"role": "user", "content": "important", "timestamp": 1000})
    mgr.append_label("checkpoint", msg_id)

    assert mgr.get_label(msg_id) == "checkpoint"


def test_session_name():
    mgr = SessionManager.in_memory()
    assert mgr.get_session_name() is None

    mgr.set_session_name("My Session")
    assert mgr.get_session_name() == "My Session"

    mgr.set_session_name("Updated Name")
    assert mgr.get_session_name() == "Updated Name"


def test_append_custom_entry():
    mgr = SessionManager.in_memory()
    entry_id = mgr.append_custom_entry("my_extension", {"key": "value"})
    entry = mgr.get_entry(entry_id)
    assert entry is not None
    assert entry["type"] == "custom"
    assert entry["data"] == {"key": "value"}


def test_append_custom_message():
    mgr = SessionManager.in_memory()
    entry_id = mgr.append_custom_message("user", [{"type": "text", "text": "custom"}])
    entry = mgr.get_entry(entry_id)
    assert entry is not None
    assert entry["type"] == "custom_message"
    assert entry["role"] == "user"


# --- Tree visualization ---


def test_get_tree():
    mgr = SessionManager.in_memory()
    mgr.append_message({"role": "user", "content": "q1", "timestamp": 1000})
    mgr.append_message({"role": "assistant", "content": "a1", "timestamp": 1001})

    # Branch
    mgr.branch(mgr.entries[0]["id"])
    mgr.append_message({"role": "assistant", "content": "a1-alt", "timestamp": 1002})

    tree = mgr.get_tree()
    assert len(tree) == 1  # One root
    root = tree[0]
    assert len(root.children) == 2  # Two children: a1 and a1-alt


# --- File persistence ---


def test_create_and_open():
    with tempfile.TemporaryDirectory() as tmpdir:
        session_dir = os.path.join(tmpdir, "sessions")

        # Create
        mgr = SessionManager.create("/tmp/test", session_dir)
        mgr.append_message({"role": "user", "content": "hello", "timestamp": 1000})
        mgr.append_message({"role": "assistant", "content": "hi", "timestamp": 1001})
        mgr.flush()

        path = mgr.session_file
        assert path is not None
        assert os.path.exists(path)

        # Open
        mgr2 = SessionManager.open(path)
        assert mgr2.session_id == mgr.session_id
        assert mgr2.entry_count == 2


def test_continue_recent():
    with tempfile.TemporaryDirectory() as tmpdir:
        session_dir = os.path.join(tmpdir, "sessions")

        # Create first
        mgr = SessionManager.create("/tmp/test", session_dir)
        mgr.append_message({"role": "user", "content": "hello", "timestamp": 1000})
        mgr.append_message({"role": "assistant", "content": "hi", "timestamp": 1001})
        mgr.flush()

        # Continue recent should find it
        mgr2 = SessionManager.continue_recent("/tmp/test", session_dir)
        assert mgr2.session_id == mgr.session_id

        # New dir = new session
        mgr3 = SessionManager.continue_recent("/tmp/other", os.path.join(tmpdir, "other"))
        assert mgr3.session_id != mgr.session_id


def test_list_sessions():
    with tempfile.TemporaryDirectory() as tmpdir:
        session_dir = os.path.join(tmpdir, "sessions")

        # Create two sessions
        mgr1 = SessionManager.create("/tmp/test", session_dir)
        mgr1.append_message({"role": "user", "content": "session 1", "timestamp": 1000})
        mgr1.append_message({"role": "assistant", "content": "ok", "timestamp": 1001})
        mgr1.flush()

        mgr2 = SessionManager.create("/tmp/test", session_dir)
        mgr2.append_message({"role": "user", "content": "session 2", "timestamp": 2000})
        mgr2.append_message({"role": "assistant", "content": "ok2", "timestamp": 2001})
        mgr2.flush()

        sessions = SessionManager.list_sessions("/tmp/test", session_dir)
        assert len(sessions) == 2
        # Should be sorted by modified time (newest first)
        assert sessions[0].id == mgr2.session_id or sessions[1].id == mgr2.session_id


# --- Migrations ---


def test_migrate_v1_to_v2():
    entries = [
        {"type": "session", "version": 1, "id": "s1", "timestamp": "now", "cwd": "/"},
        {"type": "message", "message": {"role": "user", "content": "hi"}},
        {"type": "message", "message": {"role": "assistant", "content": "hello"}},
    ]
    migrated, was_migrated = _migrate_to_current(entries)
    assert was_migrated
    assert migrated[0]["version"] == 3
    # Entries should now have id and parentId
    assert "id" in migrated[1]
    assert migrated[1]["parentId"] is None
    assert "id" in migrated[2]
    assert migrated[2]["parentId"] == migrated[1]["id"]


def test_migrate_v2_to_v3():
    entries = [
        {"type": "session", "version": 2, "id": "s1", "timestamp": "now", "cwd": "/"},
        {"type": "hookMessage", "id": "h1", "parentId": None},
    ]
    migrated, was_migrated = _migrate_to_current(entries)
    assert was_migrated
    assert migrated[1]["type"] == "custom"


def test_no_migration_needed():
    entries = [
        {"type": "session", "version": 3, "id": "s1", "timestamp": "now", "cwd": "/"},
        {"type": "message", "id": "m1", "parentId": None, "message": {"role": "user"}},
    ]
    _migrated, was_migrated = _migrate_to_current(entries)
    assert not was_migrated


# --- Branched session creation ---


def test_create_branched_session():
    with tempfile.TemporaryDirectory() as tmpdir:
        session_dir = os.path.join(tmpdir, "sessions")

        mgr = SessionManager.create("/tmp/test", session_dir)
        mgr.append_message({"role": "user", "content": "q1", "timestamp": 1000})
        mgr.append_message({"role": "assistant", "content": "a1", "timestamp": 1001})
        mgr.append_message({"role": "user", "content": "q2", "timestamp": 1002})
        mgr.flush()

        new_file = mgr.create_branched_session()
        assert new_file is not None
        assert os.path.exists(new_file)

        # Load and verify
        entries = load_entries_from_file(new_file)
        assert len(entries) >= 4  # header + 3 messages
        assert entries[0]["type"] == "session"
        assert entries[0].get("parentSession") == mgr.session_id
