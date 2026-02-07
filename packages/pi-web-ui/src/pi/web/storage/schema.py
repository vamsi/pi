"""SQLite schema definitions."""

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS session_metadata (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    last_modified TEXT NOT NULL,
    message_count INTEGER DEFAULT 0,
    model_id TEXT DEFAULT '',
    thinking_level TEXT DEFAULT 'off',
    preview TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    model_json TEXT NOT NULL DEFAULT '{}',
    thinking_level TEXT DEFAULT 'off',
    messages_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    last_modified TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS provider_keys (
    provider TEXT PRIMARY KEY,
    api_key TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS attachments (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    file_name TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    size INTEGER NOT NULL,
    content BLOB NOT NULL,
    created_at TEXT NOT NULL
);
"""
