"""SlackBot: Socket Mode, Web API, message queuing."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Protocol

from slack_sdk.socket_mode.aiohttp import SocketModeClient as AsyncSocketModeClient
from slack_sdk.web.async_client import AsyncWebClient

from pi.mom import log
from pi.mom.store import Attachment, ChannelStore

# ============================================================================
# Types
# ============================================================================


@dataclass
class SlackEvent:
    type: str  # "mention" or "dm"
    channel: str
    ts: str
    user: str
    text: str
    files: list[dict[str, Any]] | None = None
    attachments: list[Attachment] | None = None


@dataclass
class SlackUser:
    id: str
    user_name: str
    display_name: str


@dataclass
class SlackChannel:
    id: str
    name: str


@dataclass
class ChannelInfo:
    id: str
    name: str


@dataclass
class UserInfo:
    id: str
    user_name: str
    display_name: str


@dataclass
class SlackMessage:
    text: str
    raw_text: str
    user: str
    user_name: str | None
    channel: str
    ts: str
    attachments: list[dict[str, str]]


@dataclass
class SlackContext:
    message: SlackMessage
    channel_name: str | None
    channels: list[ChannelInfo]
    users: list[UserInfo]
    respond: Callable[[str, bool], Coroutine[Any, Any, None]]
    replace_message: Callable[[str], Coroutine[Any, Any, None]]
    respond_in_thread: Callable[[str], Coroutine[Any, Any, None]]
    set_typing: Callable[[bool], Coroutine[Any, Any, None]]
    upload_file: Callable[[str, str | None], Coroutine[Any, Any, None]]
    set_working: Callable[[bool], Coroutine[Any, Any, None]]
    delete_message: Callable[[], Coroutine[Any, Any, None]]


class MomHandler(Protocol):
    def is_running(self, channel_id: str) -> bool: ...
    async def handle_event(
        self, event: SlackEvent, slack: SlackBot, is_event: bool = False
    ) -> None: ...
    async def handle_stop(self, channel_id: str, slack: SlackBot) -> None: ...


# ============================================================================
# Per-channel queue for sequential processing
# ============================================================================


class _ChannelQueue:
    def __init__(self) -> None:
        self._queue: list[Callable[[], Coroutine[Any, Any, None]]] = []
        self._processing = False

    def enqueue(self, work: Callable[[], Coroutine[Any, Any, None]]) -> None:
        self._queue.append(work)
        asyncio.ensure_future(self._process_next())

    def size(self) -> int:
        return len(self._queue)

    async def _process_next(self) -> None:
        if self._processing or not self._queue:
            return
        self._processing = True
        work = self._queue.pop(0)
        try:
            await work()
        except Exception as exc:
            log.log_warning("Queue error", str(exc))
        self._processing = False
        await self._process_next()


# ============================================================================
# SlackBot
# ============================================================================

_MENTION_RE = re.compile(r"<@[A-Z0-9]+>", re.IGNORECASE)


class SlackBot:
    def __init__(
        self,
        handler: MomHandler,
        *,
        app_token: str,
        bot_token: str,
        working_dir: str,
        store: ChannelStore,
    ) -> None:
        self._handler = handler
        self._working_dir = working_dir
        self._store = store
        self._socket_client = AsyncSocketModeClient(
            app_token=app_token, web_client=AsyncWebClient(token=bot_token)
        )
        self._web_client = AsyncWebClient(token=bot_token)
        self._bot_user_id: str | None = None
        self._startup_ts: str | None = None

        self._users: dict[str, SlackUser] = {}
        self._channels: dict[str, SlackChannel] = {}
        self._queues: dict[str, _ChannelQueue] = {}

    # ── Public API ───────────────────────────────────────────────────

    async def start(self) -> None:
        auth = await self._web_client.auth_test()
        self._bot_user_id = auth["user_id"]

        await asyncio.gather(self._fetch_users(), self._fetch_channels())
        log.log_info(
            f"Loaded {len(self._channels)} channels, {len(self._users)} users"
        )

        await self._backfill_all_channels()

        self._setup_event_handlers()
        await self._socket_client.connect()

        self._startup_ts = f"{time.time():.6f}"
        log.log_connected()

    def get_user(self, user_id: str) -> SlackUser | None:
        return self._users.get(user_id)

    def get_channel(self, channel_id: str) -> SlackChannel | None:
        return self._channels.get(channel_id)

    def get_all_users(self) -> list[SlackUser]:
        return list(self._users.values())

    def get_all_channels(self) -> list[SlackChannel]:
        return list(self._channels.values())

    async def post_message(self, channel: str, text: str) -> str:
        result = await self._web_client.chat_postMessage(channel=channel, text=text)
        return result["ts"]

    async def update_message(self, channel: str, ts: str, text: str) -> None:
        await self._web_client.chat_update(channel=channel, ts=ts, text=text)

    async def delete_message(self, channel: str, ts: str) -> None:
        await self._web_client.chat_delete(channel=channel, ts=ts)

    async def post_in_thread(
        self, channel: str, thread_ts: str, text: str
    ) -> str:
        result = await self._web_client.chat_postMessage(
            channel=channel, thread_ts=thread_ts, text=text
        )
        return result["ts"]

    async def upload_file(
        self, channel: str, file_path: str, title: str | None = None
    ) -> None:
        file_name = title or os.path.basename(file_path)
        with open(file_path, "rb") as fh:
            file_content = fh.read()
        await self._web_client.files_upload_v2(
            channel=channel,
            file=file_content,
            filename=file_name,
            title=file_name,
        )

    def log_to_file(self, channel: str, entry: dict[str, Any]) -> None:
        d = os.path.join(self._working_dir, channel)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "log.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    def log_bot_response(self, channel: str, text: str, ts: str) -> None:
        self.log_to_file(
            channel,
            {
                "date": datetime.now(timezone.utc).isoformat(),
                "ts": ts,
                "user": "bot",
                "text": text,
                "attachments": [],
                "isBot": True,
            },
        )

    # ── Events integration ───────────────────────────────────────────

    def enqueue_event(self, event: SlackEvent) -> bool:
        queue = self._get_queue(event.channel)
        if queue.size() >= 5:
            log.log_warning(
                f"Event queue full for {event.channel}, "
                f"discarding: {event.text[:50]}"
            )
            return False
        log.log_info(
            f"Enqueueing event for {event.channel}: {event.text[:50]}"
        )
        queue.enqueue(lambda: self._handler.handle_event(event, self, True))
        return True

    # ── Private – event handlers ─────────────────────────────────────

    def _get_queue(self, channel_id: str) -> _ChannelQueue:
        if channel_id not in self._queues:
            self._queues[channel_id] = _ChannelQueue()
        return self._queues[channel_id]

    def _setup_event_handlers(self) -> None:
        from slack_sdk.socket_mode.request import SocketModeRequest
        from slack_sdk.socket_mode.response import SocketModeResponse

        async def _on_events_api(
            client: AsyncSocketModeClient, req: SocketModeRequest
        ) -> None:
            # Acknowledge immediately
            await client.send_socket_mode_response(
                SocketModeResponse(envelope_id=req.envelope_id)
            )

            payload = req.payload
            event = payload.get("event", {})
            event_type = event.get("type")

            if event_type == "app_mention":
                self._handle_app_mention(event)
            elif event_type == "message":
                self._handle_message(event)

        self._socket_client.socket_mode_request_listeners.append(_on_events_api)

    def _handle_app_mention(self, event: dict[str, Any]) -> None:
        channel = event.get("channel", "")
        user = event.get("user", "")
        ts = event.get("ts", "")
        text = event.get("text", "")
        files = event.get("files")

        if channel.startswith("D"):
            return

        slack_event = SlackEvent(
            type="mention",
            channel=channel,
            ts=ts,
            user=user,
            text=_MENTION_RE.sub("", text).strip(),
            files=files,
        )

        slack_event.attachments = self._log_user_message(slack_event)

        if self._startup_ts and ts < self._startup_ts:
            log.log_info(
                f"[{channel}] Logged old message (pre-startup), "
                f"not triggering: {slack_event.text[:30]}"
            )
            return

        if slack_event.text.lower().strip() == "stop":
            if self._handler.is_running(channel):
                asyncio.ensure_future(self._handler.handle_stop(channel, self))
            else:
                asyncio.ensure_future(self.post_message(channel, "_Nothing running_"))
            return

        if self._handler.is_running(channel):
            asyncio.ensure_future(
                self.post_message(
                    channel, "_Already working. Say `@mom stop` to cancel._"
                )
            )
        else:
            self._get_queue(channel).enqueue(
                lambda ev=slack_event: self._handler.handle_event(ev, self)
            )

    def _handle_message(self, event: dict[str, Any]) -> None:
        channel = event.get("channel", "")
        user = event.get("user")
        ts = event.get("ts", "")
        text = event.get("text", "")
        channel_type = event.get("channel_type")
        subtype = event.get("subtype")
        bot_id = event.get("bot_id")
        files = event.get("files")

        if bot_id or not user or user == self._bot_user_id:
            return
        if subtype is not None and subtype != "file_share":
            return
        if not text and (not files or len(files) == 0):
            return

        is_dm = channel_type == "im"
        is_bot_mention = self._bot_user_id and f"<@{self._bot_user_id}>" in (text or "")

        if not is_dm and is_bot_mention:
            return

        slack_event = SlackEvent(
            type="dm" if is_dm else "mention",
            channel=channel,
            ts=ts,
            user=user,
            text=_MENTION_RE.sub("", text or "").strip(),
            files=files,
        )

        slack_event.attachments = self._log_user_message(slack_event)

        if self._startup_ts and ts < self._startup_ts:
            log.log_info(
                f"[{channel}] Skipping old message (pre-startup): "
                f"{slack_event.text[:30]}"
            )
            return

        if is_dm:
            if slack_event.text.lower().strip() == "stop":
                if self._handler.is_running(channel):
                    asyncio.ensure_future(
                        self._handler.handle_stop(channel, self)
                    )
                else:
                    asyncio.ensure_future(
                        self.post_message(channel, "_Nothing running_")
                    )
                return

            if self._handler.is_running(channel):
                asyncio.ensure_future(
                    self.post_message(
                        channel,
                        "_Already working. Say `stop` to cancel._",
                    )
                )
            else:
                self._get_queue(channel).enqueue(
                    lambda ev=slack_event: self._handler.handle_event(ev, self)
                )

    def _log_user_message(self, event: SlackEvent) -> list[Attachment]:
        user = self._users.get(event.user)
        attachments = (
            self._store.process_attachments(event.channel, event.files, event.ts)
            if event.files
            else []
        )
        self.log_to_file(
            event.channel,
            {
                "date": datetime.fromtimestamp(
                    float(event.ts), tz=timezone.utc
                ).isoformat(),
                "ts": event.ts,
                "user": event.user,
                "userName": user.user_name if user else None,
                "displayName": user.display_name if user else None,
                "text": event.text,
                "attachments": [
                    {"original": a.original, "local": a.local} for a in attachments
                ],
                "isBot": False,
            },
        )
        return attachments

    # ── Private – backfill ───────────────────────────────────────────

    def _get_existing_timestamps(self, channel_id: str) -> set[str]:
        log_path = os.path.join(self._working_dir, channel_id, "log.jsonl")
        timestamps: set[str] = set()
        if not os.path.exists(log_path):
            return timestamps
        with open(log_path, "r", encoding="utf-8") as fh:
            content = fh.read()
        for line in content.strip().split("\n"):
            if not line:
                continue
            try:
                entry = json.loads(line)
                ts = entry.get("ts")
                if ts:
                    timestamps.add(ts)
            except json.JSONDecodeError:
                pass
        return timestamps

    async def _backfill_channel(self, channel_id: str) -> int:
        existing_ts = self._get_existing_timestamps(channel_id)

        latest_ts: str | None = None
        for ts in existing_ts:
            if latest_ts is None or float(ts) > float(latest_ts):
                latest_ts = ts

        all_messages: list[dict[str, Any]] = []
        cursor: str | None = None
        page_count = 0
        max_pages = 3

        while True:
            kwargs: dict[str, Any] = {
                "channel": channel_id,
                "inclusive": False,
                "limit": 1000,
            }
            if latest_ts:
                kwargs["oldest"] = latest_ts
            if cursor:
                kwargs["cursor"] = cursor

            result = await self._web_client.conversations_history(**kwargs)
            messages = result.get("messages", [])
            if messages:
                all_messages.extend(messages)
            cursor = (result.get("response_metadata") or {}).get("next_cursor")
            page_count += 1
            if not cursor or page_count >= max_pages:
                break

        relevant = [
            msg
            for msg in all_messages
            if msg.get("ts")
            and msg["ts"] not in existing_ts
            and (
                msg.get("user") == self._bot_user_id
                or (
                    not msg.get("bot_id")
                    and (msg.get("subtype") is None or msg.get("subtype") == "file_share")
                    and msg.get("user")
                    and (msg.get("text") or msg.get("files"))
                )
            )
        ]

        relevant.reverse()

        for msg in relevant:
            is_mom = msg.get("user") == self._bot_user_id
            user = self._users.get(msg["user"]) if msg.get("user") else None
            text = _MENTION_RE.sub("", msg.get("text", "")).strip()
            attachments = (
                self._store.process_attachments(
                    channel_id, msg["files"], msg["ts"]
                )
                if msg.get("files")
                else []
            )
            self.log_to_file(
                channel_id,
                {
                    "date": datetime.fromtimestamp(
                        float(msg["ts"]), tz=timezone.utc
                    ).isoformat(),
                    "ts": msg["ts"],
                    "user": "bot" if is_mom else msg["user"],
                    "userName": None if is_mom else (user.user_name if user else None),
                    "displayName": None if is_mom else (user.display_name if user else None),
                    "text": text,
                    "attachments": [
                        {"original": a.original, "local": a.local}
                        for a in attachments
                    ],
                    "isBot": is_mom,
                },
            )

        return len(relevant)

    async def _backfill_all_channels(self) -> None:
        start_time = time.time()
        channels_to_backfill: list[tuple[str, SlackChannel]] = []
        for ch_id, ch in self._channels.items():
            log_path = os.path.join(self._working_dir, ch_id, "log.jsonl")
            if os.path.exists(log_path):
                channels_to_backfill.append((ch_id, ch))

        log.log_backfill_start(len(channels_to_backfill))

        total_messages = 0
        for ch_id, ch in channels_to_backfill:
            try:
                count = await self._backfill_channel(ch_id)
                if count > 0:
                    log.log_backfill_channel(ch.name, count)
                total_messages += count
            except Exception as exc:
                log.log_warning(f"Failed to backfill #{ch.name}", str(exc))

        duration_ms = (time.time() - start_time) * 1000
        log.log_backfill_complete(total_messages, duration_ms)

    # ── Private – fetch users/channels ───────────────────────────────

    async def _fetch_users(self) -> None:
        cursor: str | None = None
        while True:
            kwargs: dict[str, Any] = {"limit": 200}
            if cursor:
                kwargs["cursor"] = cursor
            result = await self._web_client.users_list(**kwargs)
            members = result.get("members", [])
            for u in members:
                uid = u.get("id")
                name = u.get("name")
                if uid and name and not u.get("deleted"):
                    self._users[uid] = SlackUser(
                        id=uid,
                        user_name=name,
                        display_name=u.get("real_name") or name,
                    )
            cursor = (result.get("response_metadata") or {}).get("next_cursor")
            if not cursor:
                break

    async def _fetch_channels(self) -> None:
        # Public + private
        cursor: str | None = None
        while True:
            kwargs: dict[str, Any] = {
                "types": "public_channel,private_channel",
                "exclude_archived": True,
                "limit": 200,
            }
            if cursor:
                kwargs["cursor"] = cursor
            result = await self._web_client.conversations_list(**kwargs)
            channels = result.get("channels", [])
            for c in channels:
                cid = c.get("id")
                cname = c.get("name")
                if cid and cname and c.get("is_member"):
                    self._channels[cid] = SlackChannel(id=cid, name=cname)
            cursor = (result.get("response_metadata") or {}).get("next_cursor")
            if not cursor:
                break

        # DMs
        cursor = None
        while True:
            kwargs = {"types": "im", "limit": 200}
            if cursor:
                kwargs["cursor"] = cursor
            result = await self._web_client.conversations_list(**kwargs)
            ims = result.get("channels", [])
            for im in ims:
                im_id = im.get("id")
                if im_id:
                    im_user = im.get("user")
                    user = self._users.get(im_user) if im_user else None
                    name = f"DM:{user.user_name}" if user else f"DM:{im_id}"
                    self._channels[im_id] = SlackChannel(id=im_id, name=name)
            cursor = (result.get("response_metadata") or {}).get("next_cursor")
            if not cursor:
                break
