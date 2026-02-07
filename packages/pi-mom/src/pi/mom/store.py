"""ChannelStore: message logging and attachment downloads."""

from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

from pi.mom import log


@dataclass
class Attachment:
    original: str  # original filename from uploader
    local: str  # path relative to working dir


@dataclass
class LoggedMessage:
    date: str  # ISO 8601
    ts: str  # slack timestamp or epoch ms
    user: str  # user ID (or "bot" for bot responses)
    text: str
    attachments: list[Attachment]
    is_bot: bool
    user_name: str | None = None
    display_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "date": self.date,
            "ts": self.ts,
            "user": self.user,
            "text": self.text,
            "attachments": [
                {"original": a.original, "local": a.local}
                for a in self.attachments
            ],
            "isBot": self.is_bot,
        }
        if self.user_name is not None:
            d["userName"] = self.user_name
        if self.display_name is not None:
            d["displayName"] = self.display_name
        return d


@dataclass
class _PendingDownload:
    channel_id: str
    local_path: str
    url: str


class ChannelStore:
    def __init__(self, working_dir: str, bot_token: str) -> None:
        self._working_dir = working_dir
        self._bot_token = bot_token
        self._pending_downloads: list[_PendingDownload] = []
        self._is_downloading = False
        # Track recently logged message timestamps to prevent duplicates
        self._recently_logged: dict[str, float] = {}

        os.makedirs(self._working_dir, exist_ok=True)

    # ── Public API ───────────────────────────────────────────────────

    def get_channel_dir(self, channel_id: str) -> str:
        d = os.path.join(self._working_dir, channel_id)
        os.makedirs(d, exist_ok=True)
        return d

    @staticmethod
    def generate_local_filename(original_name: str, timestamp: str) -> str:
        ts = math.floor(float(timestamp) * 1000)
        sanitized = re.sub(r"[^a-zA-Z0-9._-]", "_", original_name)
        return f"{ts}_{sanitized}"

    def process_attachments(
        self,
        channel_id: str,
        files: list[dict[str, Any]],
        timestamp: str,
    ) -> list[Attachment]:
        attachments: list[Attachment] = []

        for f in files:
            url = f.get("url_private_download") or f.get("url_private")
            if not url:
                continue
            name = f.get("name")
            if not name:
                log.log_warning("Attachment missing name, skipping", url)
                continue

            filename = self.generate_local_filename(name, timestamp)
            local_path = f"{channel_id}/attachments/{filename}"

            attachments.append(Attachment(original=name, local=local_path))
            self._pending_downloads.append(
                _PendingDownload(
                    channel_id=channel_id, local_path=local_path, url=url
                )
            )

        # Trigger background download
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._process_download_queue())
        except RuntimeError:
            pass  # No event loop – downloads will be processed later

        return attachments

    async def log_message(
        self, channel_id: str, message: LoggedMessage
    ) -> bool:
        dedupe_key = f"{channel_id}:{message.ts}"
        if dedupe_key in self._recently_logged:
            return False

        self._recently_logged[dedupe_key] = time.time()
        # Schedule cleanup after 60 seconds
        self._schedule_cleanup(dedupe_key)

        log_path = os.path.join(self.get_channel_dir(channel_id), "log.jsonl")

        if not message.date:
            if "." in message.ts:
                dt = datetime.fromtimestamp(float(message.ts), tz=timezone.utc)
            else:
                dt = datetime.fromtimestamp(
                    int(message.ts) / 1000, tz=timezone.utc
                )
            message.date = dt.isoformat()

        line = json.dumps(message.to_dict()) + "\n"
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line)
        return True

    async def log_bot_response(
        self, channel_id: str, text: str, ts: str
    ) -> None:
        await self.log_message(
            channel_id,
            LoggedMessage(
                date=datetime.now(timezone.utc).isoformat(),
                ts=ts,
                user="bot",
                text=text,
                attachments=[],
                is_bot=True,
            ),
        )

    def get_last_timestamp(self, channel_id: str) -> str | None:
        log_path = os.path.join(self._working_dir, channel_id, "log.jsonl")
        if not os.path.exists(log_path):
            return None

        try:
            with open(log_path, "r", encoding="utf-8") as fh:
                content = fh.read()
            lines = content.strip().split("\n")
            if not lines or lines[0] == "":
                return None
            last = json.loads(lines[-1])
            return last.get("ts")
        except Exception:
            return None

    # ── Private ──────────────────────────────────────────────────────

    def _schedule_cleanup(self, key: str) -> None:
        import asyncio

        async def _cleanup() -> None:
            await asyncio.sleep(60)
            self._recently_logged.pop(key, None)

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_cleanup())
        except RuntimeError:
            pass

    async def _process_download_queue(self) -> None:
        if self._is_downloading or not self._pending_downloads:
            return

        self._is_downloading = True
        try:
            while self._pending_downloads:
                item = self._pending_downloads.pop(0)
                try:
                    await self._download_attachment(item.local_path, item.url)
                except Exception as exc:
                    log.log_warning(
                        "Failed to download attachment",
                        f"{item.local_path}: {exc}",
                    )
        finally:
            self._is_downloading = False

    async def _download_attachment(
        self, local_path: str, url: str
    ) -> None:
        file_path = os.path.join(self._working_dir, local_path)
        d = os.path.dirname(file_path)
        os.makedirs(d, exist_ok=True)

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                url, headers={"Authorization": f"Bearer {self._bot_token}"}
            )
            resp.raise_for_status()
            with open(file_path, "wb") as fh:
                fh.write(resp.content)
