"""EventsWatcher: cron scheduling, file watching for event-driven triggers."""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from croniter import croniter

from pi.mom import log

if TYPE_CHECKING:
    from pi.mom.slack import SlackBot, SlackEvent


# ============================================================================
# Event Types
# ============================================================================


@dataclass
class ImmediateEvent:
    type: str  # "immediate"
    channel_id: str
    text: str


@dataclass
class OneShotEvent:
    type: str  # "one-shot"
    channel_id: str
    text: str
    at: str  # ISO 8601 with timezone offset


@dataclass
class PeriodicEvent:
    type: str  # "periodic"
    channel_id: str
    text: str
    schedule: str  # cron syntax
    tz: str  # IANA timezone


MomEvent = ImmediateEvent | OneShotEvent | PeriodicEvent


# ============================================================================
# EventsWatcher
# ============================================================================

_DEBOUNCE_MS = 0.1  # 100 ms
_MAX_RETRIES = 3
_RETRY_BASE_S = 0.1


class EventsWatcher:
    def __init__(self, events_dir: str, slack: SlackBot) -> None:
        self._events_dir = events_dir
        self._slack = slack
        self._start_time = time.time()
        self._timers: dict[str, asyncio.TimerHandle] = {}
        self._cron_tasks: dict[str, asyncio.Task[None]] = {}
        self._debounce_handles: dict[str, asyncio.TimerHandle] = {}
        self._known_files: set[str] = set()
        self._observer: Any = None  # watchdog Observer
        self._running = False

    def start(self) -> None:
        os.makedirs(self._events_dir, exist_ok=True)
        log.log_info(f"Events watcher starting, dir: {self._events_dir}")

        # Scan existing files
        self._scan_existing()

        # Start filesystem watcher using watchdog
        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer

            watcher = self

            class _Handler(FileSystemEventHandler):
                def on_any_event(self, event: Any) -> None:
                    if event.is_directory:
                        return
                    src = getattr(event, "src_path", "")
                    if not src.endswith(".json"):
                        return
                    filename = os.path.basename(src)
                    watcher._debounce(filename, lambda fn=filename: watcher._handle_file_change(fn))

            self._observer = Observer()
            self._observer.schedule(_Handler(), self._events_dir, recursive=False)
            self._observer.start()
        except ImportError:
            log.log_warning(
                "watchdog not installed – events directory will not be watched for changes"
            )

        self._running = True
        log.log_info(f"Events watcher started, tracking {len(self._known_files)} files")

    def stop(self) -> None:
        if self._observer is not None:
            self._observer.stop()
            self._observer.join()
            self._observer = None

        for handle in self._debounce_handles.values():
            handle.cancel()
        self._debounce_handles.clear()

        for handle in self._timers.values():
            handle.cancel()
        self._timers.clear()

        for task in self._cron_tasks.values():
            task.cancel()
        self._cron_tasks.clear()

        self._known_files.clear()
        self._running = False
        log.log_info("Events watcher stopped")

    # ── Internal ─────────────────────────────────────────────────────

    def _debounce(self, filename: str, fn: Any) -> None:
        existing = self._debounce_handles.pop(filename, None)
        if existing is not None:
            existing.cancel()
        try:
            loop = asyncio.get_running_loop()
            handle = loop.call_later(_DEBOUNCE_MS, fn)
            self._debounce_handles[filename] = handle
        except RuntimeError:
            fn()

    def _scan_existing(self) -> None:
        try:
            files = [f for f in os.listdir(self._events_dir) if f.endswith(".json")]
        except OSError as exc:
            log.log_warning("Failed to read events directory", str(exc))
            return
        for filename in files:
            asyncio.ensure_future(self._handle_file(filename))

    def _handle_file_change(self, filename: str) -> None:
        file_path = os.path.join(self._events_dir, filename)
        if not os.path.exists(file_path):
            self._handle_delete(filename)
        elif filename in self._known_files:
            self._cancel_scheduled(filename)
            asyncio.ensure_future(self._handle_file(filename))
        else:
            asyncio.ensure_future(self._handle_file(filename))

    def _handle_delete(self, filename: str) -> None:
        if filename not in self._known_files:
            return
        log.log_info(f"Event file deleted: {filename}")
        self._cancel_scheduled(filename)
        self._known_files.discard(filename)

    def _cancel_scheduled(self, filename: str) -> None:
        handle = self._timers.pop(filename, None)
        if handle is not None:
            handle.cancel()
        task = self._cron_tasks.pop(filename, None)
        if task is not None:
            task.cancel()

    async def _handle_file(self, filename: str) -> None:
        file_path = os.path.join(self._events_dir, filename)

        event: MomEvent | None = None
        last_error: Exception | None = None

        for i in range(_MAX_RETRIES):
            try:
                with open(file_path, "r", encoding="utf-8") as fh:
                    content = fh.read()
                event = self._parse_event(content, filename)
                break
            except Exception as exc:
                last_error = exc
                if i < _MAX_RETRIES - 1:
                    await asyncio.sleep(_RETRY_BASE_S * (2**i))

        if event is None:
            log.log_warning(
                f"Failed to parse event file after {_MAX_RETRIES} retries: {filename}",
                str(last_error) if last_error else None,
            )
            self._delete_file(filename)
            return

        self._known_files.add(filename)

        if event.type == "immediate":
            self._handle_immediate(filename, event)  # type: ignore[arg-type]
        elif event.type == "one-shot":
            self._handle_one_shot(filename, event)  # type: ignore[arg-type]
        elif event.type == "periodic":
            self._handle_periodic(filename, event)  # type: ignore[arg-type]

    @staticmethod
    def _parse_event(content: str, filename: str) -> MomEvent:
        data = json.loads(content)
        if not data.get("type") or not data.get("channelId") or not data.get("text"):
            raise ValueError(
                f"Missing required fields (type, channelId, text) in {filename}"
            )

        etype = data["type"]
        if etype == "immediate":
            return ImmediateEvent(
                type="immediate",
                channel_id=data["channelId"],
                text=data["text"],
            )
        elif etype == "one-shot":
            if not data.get("at"):
                raise ValueError(f"Missing 'at' field for one-shot event in {filename}")
            return OneShotEvent(
                type="one-shot",
                channel_id=data["channelId"],
                text=data["text"],
                at=data["at"],
            )
        elif etype == "periodic":
            if not data.get("schedule"):
                raise ValueError(
                    f"Missing 'schedule' field for periodic event in {filename}"
                )
            if not data.get("timezone"):
                raise ValueError(
                    f"Missing 'timezone' field for periodic event in {filename}"
                )
            return PeriodicEvent(
                type="periodic",
                channel_id=data["channelId"],
                text=data["text"],
                schedule=data["schedule"],
                tz=data["timezone"],
            )
        else:
            raise ValueError(f"Unknown event type '{etype}' in {filename}")

    def _handle_immediate(self, filename: str, event: ImmediateEvent) -> None:
        file_path = os.path.join(self._events_dir, filename)
        try:
            mtime = os.path.getmtime(file_path)
            if mtime < self._start_time:
                log.log_info(f"Stale immediate event, deleting: {filename}")
                self._delete_file(filename)
                return
        except OSError:
            return

        log.log_info(f"Executing immediate event: {filename}")
        self._execute(filename, event)

    def _handle_one_shot(self, filename: str, event: OneShotEvent) -> None:
        at_time = datetime.fromisoformat(event.at).timestamp()
        now = time.time()

        if at_time <= now:
            log.log_info(f"One-shot event in the past, deleting: {filename}")
            self._delete_file(filename)
            return

        delay = at_time - now
        log.log_info(
            f"Scheduling one-shot event: {filename} in {round(delay)}s"
        )

        try:
            loop = asyncio.get_running_loop()
            handle = loop.call_later(
                delay,
                lambda: self._fire_one_shot(filename, event),
            )
            self._timers[filename] = handle
        except RuntimeError:
            pass

    def _fire_one_shot(self, filename: str, event: OneShotEvent) -> None:
        self._timers.pop(filename, None)
        log.log_info(f"Executing one-shot event: {filename}")
        self._execute(filename, event)

    def _handle_periodic(self, filename: str, event: PeriodicEvent) -> None:
        try:
            import zoneinfo

            tz = zoneinfo.ZoneInfo(event.tz)
            cron = croniter(event.schedule, datetime.now(tz))
            next_run = cron.get_next(datetime)
            log.log_info(
                f"Scheduled periodic event: {filename}, "
                f"next run: {next_run.isoformat()}"
            )

            async def _cron_loop() -> None:
                nonlocal cron
                while True:
                    now = datetime.now(tz)
                    next_dt = cron.get_next(datetime)
                    delay = (next_dt - now).total_seconds()
                    if delay > 0:
                        await asyncio.sleep(delay)
                    log.log_info(f"Executing periodic event: {filename}")
                    self._execute(filename, event, delete_after=False)

            task = asyncio.ensure_future(_cron_loop())
            self._cron_tasks[filename] = task
        except Exception as exc:
            log.log_warning(
                f"Invalid cron schedule for {filename}: {event.schedule}",
                str(exc),
            )
            self._delete_file(filename)

    def _execute(
        self,
        filename: str,
        event: MomEvent,
        delete_after: bool = True,
    ) -> None:
        if isinstance(event, ImmediateEvent):
            schedule_info = "immediate"
        elif isinstance(event, OneShotEvent):
            schedule_info = event.at
        elif isinstance(event, PeriodicEvent):
            schedule_info = event.schedule
        else:
            schedule_info = "unknown"

        message = f"[EVENT:{filename}:{event.type}:{schedule_info}] {event.text}"

        from pi.mom.slack import SlackEvent as SE

        synthetic_event = SE(
            type="mention",
            channel=event.channel_id,
            user="EVENT",
            text=message,
            ts=str(time.time()),
        )

        enqueued = self._slack.enqueue_event(synthetic_event)
        if enqueued and delete_after:
            self._delete_file(filename)
        elif not enqueued:
            log.log_warning(f"Event queue full, discarded: {filename}")
            if delete_after:
                self._delete_file(filename)

    def _delete_file(self, filename: str) -> None:
        file_path = os.path.join(self._events_dir, filename)
        try:
            os.unlink(file_path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            log.log_warning(f"Failed to delete event file: {filename}", str(exc))
        self._known_files.discard(filename)


def create_events_watcher(workspace_dir: str, slack: SlackBot) -> EventsWatcher:
    events_dir = os.path.join(workspace_dir, "events")
    return EventsWatcher(events_dir, slack)
