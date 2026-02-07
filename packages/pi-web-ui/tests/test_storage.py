"""Comprehensive tests for the storage layer: Database, SessionStore, ProviderKeyStore, SettingsStore."""

from __future__ import annotations

import asyncio
import json

import pytest

from pi.web.storage.database import Database
from pi.web.storage.sessions import SessionStore
from pi.web.storage.provider_keys import ProviderKeyStore
from pi.web.storage.settings import SettingsStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def db():
    """Create an in-memory Database, connect it, yield, then close."""
    database = Database(":memory:")
    await database.connect()
    yield database
    await database.close()


@pytest.fixture
def session_store(db: Database) -> SessionStore:
    return SessionStore(db)


@pytest.fixture
def provider_key_store(db: Database) -> ProviderKeyStore:
    return ProviderKeyStore(db)


@pytest.fixture
def settings_store(db: Database) -> SettingsStore:
    return SettingsStore(db)


# ===========================================================================
# Database tests
# ===========================================================================


class TestDatabase:
    async def test_connect_creates_tables(self, db: Database):
        """After connect(), all schema tables should exist."""
        cursor = await db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        rows = await cursor.fetchall()
        table_names = sorted(row["name"] for row in rows)
        assert "attachments" in table_names
        assert "provider_keys" in table_names
        assert "session_metadata" in table_names
        assert "sessions" in table_names
        assert "settings" in table_names

    async def test_conn_raises_when_not_connected(self):
        """Accessing .conn before connect() should raise RuntimeError."""
        database = Database(":memory:")
        with pytest.raises(RuntimeError, match="Database not connected"):
            _ = database.conn

    async def test_conn_raises_after_close(self):
        """Accessing .conn after close() should raise RuntimeError."""
        database = Database(":memory:")
        await database.connect()
        await database.close()
        with pytest.raises(RuntimeError, match="Database not connected"):
            _ = database.conn

    async def test_close_when_not_connected_is_noop(self):
        """Calling close() before connect() should not raise."""
        database = Database(":memory:")
        await database.close()  # should not raise

    async def test_double_connect(self):
        """Calling connect() twice should work (opens a new connection)."""
        database = Database(":memory:")
        await database.connect()
        conn1 = database.conn
        await database.connect()
        conn2 = database.conn
        assert conn2 is not conn1
        await database.close()

    async def test_row_factory_returns_dict_like_rows(self, db: Database):
        """Row factory should allow dict-style access on rows."""
        await db.conn.execute(
            "INSERT INTO settings (key, value_json) VALUES (?, ?)",
            ("test_key", '"hello"'),
        )
        cursor = await db.conn.execute("SELECT * FROM settings WHERE key = ?", ("test_key",))
        row = await cursor.fetchone()
        assert row["key"] == "test_key"
        assert row["value_json"] == '"hello"'


# ===========================================================================
# SessionStore tests
# ===========================================================================


class TestSessionStore:
    # --- save / load ---

    async def test_save_and_load(self, session_store: SessionStore):
        """A saved session should be loadable by its ID."""
        await session_store.save(
            "sess-1",
            model_json='{"name": "gpt-4"}',
            thinking_level="high",
            messages_json='[{"role": "user", "content": "hi"}]',
            title="Test Session",
            message_count=1,
            model_id="gpt-4",
            preview="hi",
        )
        result = await session_store.load("sess-1")
        assert result is not None
        assert result["id"] == "sess-1"
        assert result["model_json"] == '{"name": "gpt-4"}'
        assert result["thinking_level"] == "high"
        assert result["messages_json"] == '[{"role": "user", "content": "hi"}]'
        assert result["created_at"] is not None
        assert result["last_modified"] is not None

    async def test_save_with_defaults(self, session_store: SessionStore):
        """Saving with only session_id uses sensible defaults."""
        await session_store.save("sess-default")
        result = await session_store.load("sess-default")
        assert result is not None
        assert result["model_json"] == "{}"
        assert result["thinking_level"] == "off"
        assert result["messages_json"] == "[]"

    async def test_load_nonexistent_returns_none(self, session_store: SessionStore):
        """Loading a session that does not exist returns None."""
        result = await session_store.load("nonexistent")
        assert result is None

    # --- upsert (save same id) ---

    async def test_save_updates_existing_session(self, session_store: SessionStore):
        """Saving with the same ID should update rather than duplicate."""
        await session_store.save("sess-1", model_json='{"v": 1}', message_count=1)
        await session_store.save("sess-1", model_json='{"v": 2}', message_count=5)

        result = await session_store.load("sess-1")
        assert result is not None
        assert result["model_json"] == '{"v": 2}'

    async def test_upsert_updates_metadata(self, session_store: SessionStore):
        """Upserting should update the session_metadata row too."""
        await session_store.save("sess-1", title="First", message_count=1)
        await session_store.save("sess-1", title="Updated", message_count=10)

        metadata = await session_store.get_all_metadata()
        assert len(metadata) == 1
        assert metadata[0]["title"] == "Updated"
        assert metadata[0]["message_count"] == 10

    # --- delete ---

    async def test_delete_removes_session_and_metadata(self, session_store: SessionStore):
        """Deleting should remove both session and metadata rows."""
        await session_store.save("sess-del", title="To Delete")
        await session_store.delete("sess-del")

        assert await session_store.load("sess-del") is None
        metadata = await session_store.get_all_metadata()
        assert all(m["id"] != "sess-del" for m in metadata)

    async def test_delete_nonexistent_is_noop(self, session_store: SessionStore):
        """Deleting a nonexistent session should not raise."""
        await session_store.delete("nonexistent")  # should not raise

    # --- get_all_metadata ordering ---

    async def test_get_all_metadata_ordered_by_last_modified_desc(
        self, session_store: SessionStore
    ):
        """Metadata should be returned most-recently-modified first."""
        await session_store.save("sess-a", title="A")
        await asyncio.sleep(0.01)  # ensure distinct timestamps
        await session_store.save("sess-b", title="B")
        await asyncio.sleep(0.01)
        await session_store.save("sess-c", title="C")

        metadata = await session_store.get_all_metadata()
        assert len(metadata) == 3
        # Most recent first
        assert metadata[0]["id"] == "sess-c"
        assert metadata[1]["id"] == "sess-b"
        assert metadata[2]["id"] == "sess-a"

    async def test_get_all_metadata_empty(self, session_store: SessionStore):
        """Should return empty list when no sessions exist."""
        metadata = await session_store.get_all_metadata()
        assert metadata == []

    async def test_get_all_metadata_reorder_on_update(self, session_store: SessionStore):
        """Updating an older session should move it to the front."""
        await session_store.save("sess-a", title="A")
        await asyncio.sleep(0.01)
        await session_store.save("sess-b", title="B")
        await asyncio.sleep(0.01)
        # Update sess-a, making it the most recently modified
        await session_store.save("sess-a", title="A updated")

        metadata = await session_store.get_all_metadata()
        assert metadata[0]["id"] == "sess-a"
        assert metadata[0]["title"] == "A updated"

    # --- extract_preview ---

    async def test_extract_preview_string_content(self):
        """Preview should extract from the last user message (string content)."""
        messages = [
            {"role": "user", "content": "Hello there"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "How are you?"},
        ]
        preview = SessionStore.extract_preview(json.dumps(messages))
        assert preview == "How are you?"

    async def test_extract_preview_list_content(self):
        """Preview should extract text parts from structured content."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Part A"},
                    {"type": "image", "url": "http://x"},
                    {"type": "text", "text": "Part B"},
                ],
            }
        ]
        preview = SessionStore.extract_preview(json.dumps(messages))
        assert preview == "Part A Part B"

    async def test_extract_preview_truncation(self):
        """Preview should be truncated to max_len."""
        messages = [{"role": "user", "content": "x" * 200}]
        preview = SessionStore.extract_preview(json.dumps(messages), max_len=50)
        assert len(preview) == 50

    async def test_extract_preview_no_user_message(self):
        """Preview should be empty if there are no user messages."""
        messages = [{"role": "assistant", "content": "Hi"}]
        assert SessionStore.extract_preview(json.dumps(messages)) == ""

    async def test_extract_preview_empty_messages(self):
        """Preview should be empty for an empty messages list."""
        assert SessionStore.extract_preview("[]") == ""

    async def test_extract_preview_invalid_json(self):
        """Preview should be empty for invalid JSON."""
        assert SessionStore.extract_preview("not json") == ""

    async def test_extract_preview_non_string_non_list_content(self):
        """Messages with content that is neither string nor list should be skipped."""
        messages = [
            {"role": "user", "content": 42},
            {"role": "user", "content": "fallback"},
        ]
        preview = SessionStore.extract_preview(json.dumps(messages))
        assert preview == "fallback"

    # --- extract_title ---

    async def test_extract_title_string_content(self):
        """Title should come from the first user message."""
        messages = [
            {"role": "user", "content": "First question"},
            {"role": "user", "content": "Second question"},
        ]
        title = SessionStore.extract_title(json.dumps(messages))
        assert title == "First question"

    async def test_extract_title_list_content(self):
        """Title should extract text parts from structured content."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "text", "text": "World"},
                ],
            }
        ]
        title = SessionStore.extract_title(json.dumps(messages))
        assert title == "Hello World"

    async def test_extract_title_takes_first_line(self):
        """Title should use only the first line of multiline content."""
        messages = [{"role": "user", "content": "First line\nSecond line\nThird"}]
        title = SessionStore.extract_title(json.dumps(messages))
        assert title == "First line"

    async def test_extract_title_strips_outer_whitespace(self):
        """Title should strip leading/trailing whitespace from the whole text before splitting lines."""
        messages = [{"role": "user", "content": "  \n  Actual first line  \n  Other  "}]
        title = SessionStore.extract_title(json.dumps(messages))
        # .strip() removes leading newline, so "Actual first line  " is the first line
        assert title == "Actual first line  "

    async def test_extract_title_strips_leading_whitespace(self):
        """Leading whitespace on the content is stripped so the first non-empty line is used."""
        messages = [{"role": "user", "content": "   Hello"}]
        title = SessionStore.extract_title(json.dumps(messages))
        assert title == "Hello"

    async def test_extract_title_truncation(self):
        """Title should be truncated to max_len."""
        messages = [{"role": "user", "content": "a" * 100}]
        title = SessionStore.extract_title(json.dumps(messages), max_len=20)
        assert len(title) == 20

    async def test_extract_title_no_user_message(self):
        """Title should be 'New Chat' if there are no user messages."""
        messages = [{"role": "assistant", "content": "Hi"}]
        assert SessionStore.extract_title(json.dumps(messages)) == "New Chat"

    async def test_extract_title_empty_messages(self):
        """Title should be 'New Chat' for empty messages list."""
        assert SessionStore.extract_title("[]") == "New Chat"

    async def test_extract_title_invalid_json(self):
        """Title should be 'New Chat' for invalid JSON."""
        assert SessionStore.extract_title("{bad json}") == "New Chat"

    async def test_extract_title_skips_non_string_non_list_content(self):
        """First user message with non-string/non-list content is skipped."""
        messages = [
            {"role": "user", "content": {"not": "a string"}},
            {"role": "user", "content": "Actual title"},
        ]
        title = SessionStore.extract_title(json.dumps(messages))
        assert title == "Actual title"

    async def test_extract_title_skips_assistant_before_user(self):
        """Assistant messages before the first user message are ignored."""
        messages = [
            {"role": "assistant", "content": "Welcome!"},
            {"role": "user", "content": "My question"},
        ]
        title = SessionStore.extract_title(json.dumps(messages))
        assert title == "My question"


# ===========================================================================
# ProviderKeyStore tests
# ===========================================================================


class TestProviderKeyStore:
    async def test_get_nonexistent_returns_none(self, provider_key_store: ProviderKeyStore):
        """Getting a key for a provider that has none returns None."""
        result = await provider_key_store.get("openai")
        assert result is None

    async def test_set_and_get(self, provider_key_store: ProviderKeyStore):
        """Setting a key should make it retrievable."""
        await provider_key_store.set("openai", "sk-abc123")
        result = await provider_key_store.get("openai")
        assert result == "sk-abc123"

    async def test_upsert_overwrites(self, provider_key_store: ProviderKeyStore):
        """Setting a key for an existing provider should overwrite."""
        await provider_key_store.set("openai", "sk-old")
        await provider_key_store.set("openai", "sk-new")
        result = await provider_key_store.get("openai")
        assert result == "sk-new"

    async def test_delete(self, provider_key_store: ProviderKeyStore):
        """Deleting a provider key should make it no longer retrievable."""
        await provider_key_store.set("anthropic", "sk-ant-xyz")
        await provider_key_store.delete("anthropic")
        result = await provider_key_store.get("anthropic")
        assert result is None

    async def test_delete_nonexistent_is_noop(self, provider_key_store: ProviderKeyStore):
        """Deleting a nonexistent provider should not raise."""
        await provider_key_store.delete("nonexistent")  # should not raise

    async def test_get_all_empty(self, provider_key_store: ProviderKeyStore):
        """get_all on empty store returns empty dict."""
        result = await provider_key_store.get_all()
        assert result == {}

    async def test_get_all_multiple(self, provider_key_store: ProviderKeyStore):
        """get_all should return all stored provider keys."""
        await provider_key_store.set("openai", "sk-openai")
        await provider_key_store.set("anthropic", "sk-anthropic")
        await provider_key_store.set("google", "sk-google")

        result = await provider_key_store.get_all()
        assert result == {
            "openai": "sk-openai",
            "anthropic": "sk-anthropic",
            "google": "sk-google",
        }

    async def test_get_all_reflects_updates(self, provider_key_store: ProviderKeyStore):
        """get_all should reflect upserts and deletes."""
        await provider_key_store.set("openai", "sk-1")
        await provider_key_store.set("anthropic", "sk-2")

        await provider_key_store.set("openai", "sk-1-updated")
        await provider_key_store.delete("anthropic")

        result = await provider_key_store.get_all()
        assert result == {"openai": "sk-1-updated"}

    async def test_different_providers_are_independent(
        self, provider_key_store: ProviderKeyStore
    ):
        """Setting/deleting one provider should not affect others."""
        await provider_key_store.set("openai", "sk-openai")
        await provider_key_store.set("anthropic", "sk-anthropic")
        await provider_key_store.delete("openai")

        assert await provider_key_store.get("openai") is None
        assert await provider_key_store.get("anthropic") == "sk-anthropic"


# ===========================================================================
# SettingsStore tests
# ===========================================================================


class TestSettingsStore:
    async def test_get_nonexistent_returns_none(self, settings_store: SettingsStore):
        """Getting a key that does not exist returns None."""
        result = await settings_store.get("missing_key")
        assert result is None

    async def test_set_and_get_string(self, settings_store: SettingsStore):
        """String values round-trip correctly."""
        await settings_store.set("theme", "dark")
        result = await settings_store.get("theme")
        assert result == "dark"

    async def test_set_and_get_int(self, settings_store: SettingsStore):
        """Integer values round-trip correctly."""
        await settings_store.set("max_tokens", 4096)
        result = await settings_store.get("max_tokens")
        assert result == 4096

    async def test_set_and_get_float(self, settings_store: SettingsStore):
        """Float values round-trip correctly."""
        await settings_store.set("temperature", 0.7)
        result = await settings_store.get("temperature")
        assert result == pytest.approx(0.7)

    async def test_set_and_get_bool(self, settings_store: SettingsStore):
        """Boolean values round-trip correctly."""
        await settings_store.set("stream", True)
        assert await settings_store.get("stream") is True

        await settings_store.set("debug", False)
        assert await settings_store.get("debug") is False

    async def test_set_and_get_none(self, settings_store: SettingsStore):
        """JSON null round-trips as Python None (distinguishable from missing key by checking db)."""
        await settings_store.set("nullable", None)
        # Note: get() returns None for both missing keys and stored None.
        # We verify it was stored by checking the raw database.
        cursor = await settings_store._db.conn.execute(
            "SELECT value_json FROM settings WHERE key = ?", ("nullable",)
        )
        row = await cursor.fetchone()
        assert row is not None
        assert row["value_json"] == "null"

    async def test_set_and_get_list(self, settings_store: SettingsStore):
        """List values round-trip correctly."""
        data = [1, "two", 3.0, None, True]
        await settings_store.set("my_list", data)
        result = await settings_store.get("my_list")
        assert result == data

    async def test_set_and_get_dict(self, settings_store: SettingsStore):
        """Dict values round-trip correctly."""
        data = {"model": "gpt-4", "temperature": 0.5, "tags": ["a", "b"]}
        await settings_store.set("config", data)
        result = await settings_store.get("config")
        assert result == data

    async def test_set_and_get_nested_structure(self, settings_store: SettingsStore):
        """Deeply nested structures round-trip correctly."""
        data = {
            "level1": {
                "level2": {
                    "level3": [1, 2, {"level4": True}]
                }
            }
        }
        await settings_store.set("nested", data)
        result = await settings_store.get("nested")
        assert result == data

    async def test_upsert_overwrites(self, settings_store: SettingsStore):
        """Setting the same key again should overwrite the value."""
        await settings_store.set("key", "old_value")
        await settings_store.set("key", "new_value")
        result = await settings_store.get("key")
        assert result == "new_value"

    async def test_upsert_changes_type(self, settings_store: SettingsStore):
        """Overwriting a key can change the value type."""
        await settings_store.set("key", "string_value")
        await settings_store.set("key", 42)
        result = await settings_store.get("key")
        assert result == 42

    async def test_delete(self, settings_store: SettingsStore):
        """Deleting a key makes it no longer retrievable."""
        await settings_store.set("to_delete", "bye")
        await settings_store.delete("to_delete")
        result = await settings_store.get("to_delete")
        assert result is None

    async def test_delete_nonexistent_is_noop(self, settings_store: SettingsStore):
        """Deleting a nonexistent key should not raise."""
        await settings_store.delete("nonexistent")  # should not raise

    async def test_multiple_keys_independent(self, settings_store: SettingsStore):
        """Different keys should not interfere with each other."""
        await settings_store.set("a", 1)
        await settings_store.set("b", 2)
        await settings_store.set("c", 3)

        assert await settings_store.get("a") == 1
        assert await settings_store.get("b") == 2
        assert await settings_store.get("c") == 3

        await settings_store.delete("b")
        assert await settings_store.get("a") == 1
        assert await settings_store.get("b") is None
        assert await settings_store.get("c") == 3

    async def test_empty_string_value(self, settings_store: SettingsStore):
        """Empty string is a valid value, distinct from None/missing."""
        await settings_store.set("empty", "")
        result = await settings_store.get("empty")
        assert result == ""

    async def test_empty_list_value(self, settings_store: SettingsStore):
        """Empty list is a valid value."""
        await settings_store.set("empty_list", [])
        result = await settings_store.get("empty_list")
        assert result == []

    async def test_empty_dict_value(self, settings_store: SettingsStore):
        """Empty dict is a valid value."""
        await settings_store.set("empty_dict", {})
        result = await settings_store.get("empty_dict")
        assert result == {}
