"""Channel history export utility."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from typing import Any

from slack_sdk.web.async_client import AsyncWebClient


def _format_ts(ts: str) -> str:
    dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _format_message(
    ts: str, user: str, text: str, indent: str = ""
) -> str:
    prefix = f"[{_format_ts(ts)}] {user}: "
    lines = text.split("\n")
    first_line = f"{indent}{prefix}{lines[0]}"
    if len(lines) == 1:
        return first_line
    content_indent = indent + " " * len(prefix)
    return "\n".join(
        [first_line] + [content_indent + l for l in lines[1:]]
    )


async def download_channel(channel_id: str, bot_token: str) -> None:
    client = AsyncWebClient(token=bot_token)

    print(f"Fetching channel info for {channel_id}...", file=sys.stderr)

    channel_name = channel_id
    try:
        info = await client.conversations_info(channel=channel_id)
        channel_name = info.get("channel", {}).get("name", channel_id)
    except Exception:
        pass

    print(
        f"Downloading history for #{channel_name} ({channel_id})...",
        file=sys.stderr,
    )

    messages: list[dict[str, Any]] = []
    cursor: str | None = None

    while True:
        kwargs: dict[str, Any] = {"channel": channel_id, "limit": 200}
        if cursor:
            kwargs["cursor"] = cursor
        response = await client.conversations_history(**kwargs)
        msgs = response.get("messages", [])
        if msgs:
            messages.extend(msgs)
        cursor = (response.get("response_metadata") or {}).get("next_cursor")
        print(f"  Fetched {len(messages)} messages...", file=sys.stderr)
        if not cursor:
            break

    messages.reverse()

    # Build thread replies
    thread_replies: dict[str, list[dict[str, Any]]] = {}
    threads_to_fetch = [
        m for m in messages if m.get("reply_count") and m["reply_count"] > 0
    ]

    print(
        f"Fetching {len(threads_to_fetch)} threads...", file=sys.stderr
    )

    for idx, parent in enumerate(threads_to_fetch):
        print(
            f"  Thread {idx + 1}/{len(threads_to_fetch)} "
            f"({parent.get('reply_count', 0)} replies)...",
            file=sys.stderr,
        )

        replies: list[dict[str, Any]] = []
        t_cursor: str | None = None

        while True:
            kwargs = {
                "channel": channel_id,
                "ts": parent["ts"],
                "limit": 200,
            }
            if t_cursor:
                kwargs["cursor"] = t_cursor
            response = await client.conversations_replies(**kwargs)
            r_msgs = response.get("messages", [])
            if r_msgs:
                replies.extend(r_msgs[1:])  # skip parent
            t_cursor = (response.get("response_metadata") or {}).get(
                "next_cursor"
            )
            if not t_cursor:
                break

        thread_replies[parent["ts"]] = replies

    total_replies = 0
    for msg in messages:
        print(
            _format_message(
                msg.get("ts", "0"),
                msg.get("user", "unknown"),
                msg.get("text", ""),
            )
        )
        replies = thread_replies.get(msg.get("ts", ""))
        if replies:
            for reply in replies:
                print(
                    _format_message(
                        reply.get("ts", "0"),
                        reply.get("user", "unknown"),
                        reply.get("text", ""),
                        indent="  ",
                    )
                )
                total_replies += 1

    print(
        f"Done! {len(messages)} messages, {total_replies} thread replies",
        file=sys.stderr,
    )
