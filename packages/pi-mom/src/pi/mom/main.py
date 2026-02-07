"""Entry point: CLI args, channel state, signal handling."""

from __future__ import annotations

import asyncio
import os
import re
import signal
import sys
from dataclasses import dataclass, field
from typing import Any

from pi.mom import log
from pi.mom.agent import AgentRunner, get_or_create_runner
from pi.mom.download import download_channel
from pi.mom.events import create_events_watcher
from pi.mom.sandbox import SandboxConfig, parse_sandbox_arg, validate_sandbox
from pi.mom.slack import (
    ChannelInfo,
    MomHandler,
    SlackBot,
    SlackContext,
    SlackEvent,
    SlackMessage,
    UserInfo,
)
from pi.mom.store import ChannelStore


# ============================================================================
# Arg parsing
# ============================================================================


def _parse_args() -> dict[str, Any]:
    args = sys.argv[1:]
    sandbox: SandboxConfig = SandboxConfig(type="host")
    working_dir: str | None = None
    download_channel_id: str | None = None

    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--sandbox="):
            sandbox = parse_sandbox_arg(arg[len("--sandbox="):])
        elif arg == "--sandbox":
            i += 1
            sandbox = parse_sandbox_arg(args[i] if i < len(args) else "")
        elif arg.startswith("--download="):
            download_channel_id = arg[len("--download="):]
        elif arg == "--download":
            i += 1
            download_channel_id = args[i] if i < len(args) else None
        elif not arg.startswith("-"):
            working_dir = os.path.abspath(arg)
        i += 1

    return {
        "working_dir": working_dir,
        "sandbox": sandbox,
        "download_channel": download_channel_id,
    }


# ============================================================================
# SlackContext adapter
# ============================================================================


def _create_slack_context(
    event: SlackEvent,
    slack: SlackBot,
    state: _ChannelState,
    is_event: bool = False,
) -> SlackContext:
    message_ts: str | None = None
    thread_message_ts: list[str] = []
    accumulated_text = ""
    is_working = True
    working_indicator = " ..."
    update_lock = asyncio.Lock()

    user = slack.get_user(event.user)

    event_filename: str | None = None
    if is_event:
        m = re.match(r"^\[EVENT:([^:]+):", event.text)
        event_filename = m.group(1) if m else None

    async def respond(text: str, should_log: bool = True) -> None:
        nonlocal message_ts, accumulated_text
        async with update_lock:
            accumulated_text = f"{accumulated_text}\n{text}" if accumulated_text else text
            display = accumulated_text + working_indicator if is_working else accumulated_text
            if message_ts:
                await slack.update_message(event.channel, message_ts, display)
            else:
                message_ts = await slack.post_message(event.channel, display)
            if should_log and message_ts:
                slack.log_bot_response(event.channel, text, message_ts)

    async def replace_message(text: str) -> None:
        nonlocal message_ts, accumulated_text
        async with update_lock:
            accumulated_text = text
            display = accumulated_text + working_indicator if is_working else accumulated_text
            if message_ts:
                await slack.update_message(event.channel, message_ts, display)
            else:
                message_ts = await slack.post_message(event.channel, display)

    async def respond_in_thread(text: str) -> None:
        async with update_lock:
            if message_ts:
                ts = await slack.post_in_thread(event.channel, message_ts, text)
                thread_message_ts.append(ts)

    async def set_typing(typing: bool) -> None:
        nonlocal message_ts, accumulated_text
        if typing and not message_ts:
            async with update_lock:
                if not message_ts:
                    accumulated_text = (
                        f"_Starting event: {event_filename}_"
                        if event_filename
                        else "_Thinking_"
                    )
                    message_ts = await slack.post_message(
                        event.channel, accumulated_text + working_indicator
                    )

    async def upload_file(file_path: str, title: str | None = None) -> None:
        await slack.upload_file(event.channel, file_path, title)

    async def set_working(working: bool) -> None:
        nonlocal is_working
        async with update_lock:
            is_working = working
            if message_ts:
                display = (
                    accumulated_text + working_indicator
                    if is_working
                    else accumulated_text
                )
                await slack.update_message(event.channel, message_ts, display)

    async def delete_message_fn() -> None:
        nonlocal message_ts
        async with update_lock:
            for ts in reversed(thread_message_ts):
                try:
                    await slack.delete_message(event.channel, ts)
                except Exception:
                    pass
            thread_message_ts.clear()
            if message_ts:
                await slack.delete_message(event.channel, message_ts)
                message_ts = None

    return SlackContext(
        message=SlackMessage(
            text=event.text,
            raw_text=event.text,
            user=event.user,
            user_name=user.user_name if user else None,
            channel=event.channel,
            ts=event.ts,
            attachments=[
                {"local": a.local} for a in (event.attachments or [])
            ],
        ),
        channel_name=slack.get_channel(event.channel).name
        if slack.get_channel(event.channel)
        else None,
        channels=[
            ChannelInfo(id=c.id, name=c.name) for c in slack.get_all_channels()
        ],
        users=[
            UserInfo(id=u.id, user_name=u.user_name, display_name=u.display_name)
            for u in slack.get_all_users()
        ],
        respond=respond,
        replace_message=replace_message,
        respond_in_thread=respond_in_thread,
        set_typing=set_typing,
        upload_file=upload_file,
        set_working=set_working,
        delete_message=delete_message_fn,
    )


# ============================================================================
# Channel state
# ============================================================================


@dataclass
class _ChannelState:
    running: bool = False
    runner: AgentRunner | None = None
    store: ChannelStore | None = None
    stop_requested: bool = False
    stop_message_ts: str | None = None


# ============================================================================
# Handler
# ============================================================================


class _Handler:
    def __init__(
        self,
        sandbox: SandboxConfig,
        working_dir: str,
        bot_token: str,
    ) -> None:
        self._sandbox = sandbox
        self._working_dir = working_dir
        self._bot_token = bot_token
        self._states: dict[str, _ChannelState] = {}

    def _get_state(self, channel_id: str) -> _ChannelState:
        if channel_id not in self._states:
            channel_dir = os.path.join(self._working_dir, channel_id)
            self._states[channel_id] = _ChannelState(
                runner=get_or_create_runner(
                    self._sandbox, channel_id, channel_dir
                ),
                store=ChannelStore(self._working_dir, self._bot_token),
            )
        return self._states[channel_id]

    def is_running(self, channel_id: str) -> bool:
        st = self._states.get(channel_id)
        return st.running if st else False

    async def handle_stop(self, channel_id: str, slack: SlackBot) -> None:
        st = self._states.get(channel_id)
        if st and st.running:
            st.stop_requested = True
            st.runner.abort()  # type: ignore[union-attr]
            ts = await slack.post_message(channel_id, "_Stopping..._")
            st.stop_message_ts = ts
        else:
            await slack.post_message(channel_id, "_Nothing running_")

    async def handle_event(
        self,
        event: SlackEvent,
        slack: SlackBot,
        is_event: bool = False,
    ) -> None:
        st = self._get_state(event.channel)
        st.running = True
        st.stop_requested = False

        log.log_info(f"[{event.channel}] Starting run: {event.text[:50]}")

        try:
            ctx = _create_slack_context(event, slack, st, is_event)
            await ctx.set_typing(True)
            await ctx.set_working(True)
            result = await st.runner.run(ctx, st.store)  # type: ignore[union-attr, arg-type]
            await ctx.set_working(False)

            if result.get("stopReason") == "aborted" and st.stop_requested:
                if st.stop_message_ts:
                    await slack.update_message(
                        event.channel, st.stop_message_ts, "_Stopped_"
                    )
                    st.stop_message_ts = None
                else:
                    await slack.post_message(event.channel, "_Stopped_")
        except Exception as exc:
            log.log_warning(f"[{event.channel}] Run error", str(exc))
        finally:
            st.running = False


# ============================================================================
# Entry point
# ============================================================================


async def _async_main() -> None:
    parsed = _parse_args()

    bot_token = os.environ.get("MOM_SLACK_BOT_TOKEN")
    app_token = os.environ.get("MOM_SLACK_APP_TOKEN")

    # Handle --download mode
    if parsed["download_channel"]:
        if not bot_token:
            print("Missing env: MOM_SLACK_BOT_TOKEN", file=sys.stderr)
            sys.exit(1)
        await download_channel(parsed["download_channel"], bot_token)
        return

    working_dir: str | None = parsed["working_dir"]
    sandbox: SandboxConfig = parsed["sandbox"]

    if not working_dir:
        print(
            "Usage: pi-mom [--sandbox=host|docker:<name>] <working-directory>",
            file=sys.stderr,
        )
        print("       pi-mom --download <channel-id>", file=sys.stderr)
        sys.exit(1)

    if not app_token or not bot_token:
        print(
            "Missing env: MOM_SLACK_APP_TOKEN, MOM_SLACK_BOT_TOKEN",
            file=sys.stderr,
        )
        sys.exit(1)

    await validate_sandbox(sandbox)

    sandbox_desc = (
        "host"
        if sandbox.type == "host"
        else f"docker:{sandbox.container}"
    )
    log.log_startup(working_dir, sandbox_desc)

    shared_store = ChannelStore(working_dir, bot_token)

    handler = _Handler(sandbox, working_dir, bot_token)

    bot = SlackBot(
        handler,  # type: ignore[arg-type]
        app_token=app_token,
        bot_token=bot_token,
        working_dir=working_dir,
        store=shared_store,
    )

    events_watcher = create_events_watcher(working_dir, bot)
    events_watcher.start()

    loop = asyncio.get_event_loop()

    def _shutdown() -> None:
        log.log_info("Shutting down...")
        events_watcher.stop()
        loop.stop()

    loop.add_signal_handler(signal.SIGINT, _shutdown)
    loop.add_signal_handler(signal.SIGTERM, _shutdown)

    await bot.start()

    # Keep running
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        pass


def main() -> None:
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
