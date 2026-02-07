"""API key storage for LLM providers."""

from __future__ import annotations

from pi.web.storage.database import Database


class ProviderKeyStore:
    """Manages provider API keys in SQLite."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def get(self, provider: str) -> str | None:
        """Get API key for a provider."""
        cursor = await self._db.conn.execute(
            "SELECT api_key FROM provider_keys WHERE provider = ?", (provider,)
        )
        row = await cursor.fetchone()
        return row["api_key"] if row else None

    async def set(self, provider: str, api_key: str) -> None:
        """Set API key for a provider."""
        await self._db.conn.execute(
            """INSERT INTO provider_keys (provider, api_key) VALUES (?, ?)
               ON CONFLICT(provider) DO UPDATE SET api_key=excluded.api_key""",
            (provider, api_key),
        )
        await self._db.conn.commit()

    async def delete(self, provider: str) -> None:
        """Delete API key for a provider."""
        await self._db.conn.execute("DELETE FROM provider_keys WHERE provider = ?", (provider,))
        await self._db.conn.commit()

    async def get_all(self) -> dict[str, str]:
        """Get all stored provider keys (provider -> key)."""
        cursor = await self._db.conn.execute("SELECT provider, api_key FROM provider_keys")
        rows = await cursor.fetchall()
        return {row["provider"]: row["api_key"] for row in rows}
