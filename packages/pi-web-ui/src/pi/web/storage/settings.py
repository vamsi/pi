"""Key-value settings storage."""

from __future__ import annotations

import json
from typing import Any

from pi.web.storage.database import Database


class SettingsStore:
    """Manages key-value settings in SQLite."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def get(self, key: str) -> Any:
        """Get a setting value by key. Returns None if not found."""
        cursor = await self._db.conn.execute(
            "SELECT value_json FROM settings WHERE key = ?", (key,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return json.loads(row["value_json"])

    async def set(self, key: str, value: Any) -> None:
        """Set a setting value."""
        await self._db.conn.execute(
            """INSERT INTO settings (key, value_json) VALUES (?, ?)
               ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json""",
            (key, json.dumps(value)),
        )
        await self._db.conn.commit()

    async def delete(self, key: str) -> None:
        """Delete a setting."""
        await self._db.conn.execute("DELETE FROM settings WHERE key = ?", (key,))
        await self._db.conn.commit()
