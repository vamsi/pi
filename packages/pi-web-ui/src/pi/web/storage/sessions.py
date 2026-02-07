"""Session CRUD operations."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from pi.web.storage.database import Database


class SessionStore:
    """Manages session persistence in SQLite."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def save(
        self,
        session_id: str,
        *,
        model_json: str = "{}",
        thinking_level: str = "off",
        messages_json: str = "[]",
        title: str = "",
        message_count: int = 0,
        model_id: str = "",
        preview: str = "",
    ) -> None:
        """Save or update a session and its metadata."""
        now = datetime.now(timezone.utc).isoformat()
        conn = self._db.conn

        await conn.execute(
            """INSERT INTO sessions (id, model_json, thinking_level, messages_json, created_at, last_modified)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   model_json=excluded.model_json,
                   thinking_level=excluded.thinking_level,
                   messages_json=excluded.messages_json,
                   last_modified=excluded.last_modified""",
            (session_id, model_json, thinking_level, messages_json, now, now),
        )
        await conn.execute(
            """INSERT INTO session_metadata (id, title, created_at, last_modified, message_count, model_id, thinking_level, preview)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   title=excluded.title,
                   last_modified=excluded.last_modified,
                   message_count=excluded.message_count,
                   model_id=excluded.model_id,
                   thinking_level=excluded.thinking_level,
                   preview=excluded.preview""",
            (session_id, title, now, now, message_count, model_id, thinking_level, preview),
        )
        await conn.commit()

    async def load(self, session_id: str) -> dict[str, Any] | None:
        """Load a session by ID. Returns dict with model_json, thinking_level, messages_json."""
        cursor = await self._db.conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        return dict(row)

    async def get_all_metadata(self) -> list[dict[str, Any]]:
        """Get metadata for all sessions, sorted by last_modified descending."""
        cursor = await self._db.conn.execute(
            "SELECT * FROM session_metadata ORDER BY last_modified DESC"
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def delete(self, session_id: str) -> None:
        """Delete a session and its metadata."""
        conn = self._db.conn
        await conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        await conn.execute("DELETE FROM session_metadata WHERE id = ?", (session_id,))
        await conn.execute("DELETE FROM attachments WHERE session_id = ?", (session_id,))
        await conn.commit()

    @staticmethod
    def extract_preview(messages_json: str, max_len: int = 100) -> str:
        """Extract a preview string from the last user message."""
        try:
            messages = json.loads(messages_json)
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        text = content
                    elif isinstance(content, list):
                        text = " ".join(
                            c.get("text", "") for c in content if c.get("type") == "text"
                        )
                    else:
                        continue
                    return text[:max_len]
        except (json.JSONDecodeError, TypeError):
            pass
        return ""

    @staticmethod
    def extract_title(messages_json: str, max_len: int = 60) -> str:
        """Extract a title from the first user message."""
        try:
            messages = json.loads(messages_json)
            for msg in messages:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        text = content
                    elif isinstance(content, list):
                        text = " ".join(
                            c.get("text", "") for c in content if c.get("type") == "text"
                        )
                    else:
                        continue
                    text = text.strip().split("\n")[0]
                    return text[:max_len]
        except (json.JSONDecodeError, TypeError):
            pass
        return "New Chat"
