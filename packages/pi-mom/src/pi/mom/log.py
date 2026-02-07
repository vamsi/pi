"""Colored console logging with timestamps and context."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from typing import Any


class LogContext:
    __slots__ = ("channel_id", "user_name", "channel_name")

    def __init__(
        self,
        channel_id: str,
        user_name: str | None = None,
        channel_name: str | None = None,
    ) -> None:
        self.channel_id = channel_id
        self.user_name = user_name
        self.channel_name = channel_name


# ── ANSI helpers ─────────────────────────────────────────────────────

_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_BLUE = "\033[34m"
_DIM = "\033[2m"
_RESET = "\033[0m"


def _timestamp() -> str:
    now = datetime.now()
    return f"[{now.strftime('%H:%M:%S')}]"


def _format_context(ctx: LogContext) -> str:
    if ctx.channel_id.startswith("D"):
        return f"[DM:{ctx.user_name or ctx.channel_id}]"
    channel = ctx.channel_name or ctx.channel_id
    user = ctx.user_name or "unknown"
    ch = channel if channel.startswith("#") else f"#{channel}"
    return f"[{ch}:{user}]"


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return f"{text[:max_len]}\n(truncated at {max_len} chars)"


def _format_tool_args(args: dict[str, Any]) -> str:
    lines: list[str] = []
    for key, value in args.items():
        if key == "label":
            continue
        if key == "path" and isinstance(value, str):
            offset = args.get("offset")
            limit = args.get("limit")
            if offset is not None and limit is not None:
                lines.append(f"{value}:{offset}-{offset + limit}")
            else:
                lines.append(value)
            continue
        if key in ("offset", "limit"):
            continue
        if isinstance(value, str):
            lines.append(value)
        else:
            lines.append(json.dumps(value))
    return "\n".join(lines)


def _indent(text: str) -> str:
    return "\n".join(f"           {line}" for line in text.split("\n"))


# ── Public logging functions ─────────────────────────────────────────


def log_user_message(ctx: LogContext, text: str) -> None:
    print(f"{_GREEN}{_timestamp()} {_format_context(ctx)} {text}{_RESET}")


def log_tool_start(
    ctx: LogContext,
    tool_name: str,
    label: str,
    args: dict[str, Any],
) -> None:
    formatted_args = _format_tool_args(args)
    print(f"{_YELLOW}{_timestamp()} {_format_context(ctx)} ↳ {tool_name}: {label}{_RESET}")
    if formatted_args:
        print(f"{_DIM}{_indent(formatted_args)}{_RESET}")


def log_tool_success(
    ctx: LogContext, tool_name: str, duration_ms: float, result: str
) -> None:
    duration = f"{duration_ms / 1000:.1f}"
    print(f"{_YELLOW}{_timestamp()} {_format_context(ctx)} ✓ {tool_name} ({duration}s){_RESET}")
    truncated = _truncate(result, 1000)
    if truncated:
        print(f"{_DIM}{_indent(truncated)}{_RESET}")


def log_tool_error(
    ctx: LogContext, tool_name: str, duration_ms: float, error: str
) -> None:
    duration = f"{duration_ms / 1000:.1f}"
    print(f"{_YELLOW}{_timestamp()} {_format_context(ctx)} ✗ {tool_name} ({duration}s){_RESET}")
    truncated = _truncate(error, 1000)
    print(f"{_DIM}{_indent(truncated)}{_RESET}")


def log_response_start(ctx: LogContext) -> None:
    print(f"{_YELLOW}{_timestamp()} {_format_context(ctx)} → Streaming response...{_RESET}")


def log_thinking(ctx: LogContext, thinking: str) -> None:
    print(f"{_YELLOW}{_timestamp()} {_format_context(ctx)} 💭 Thinking{_RESET}")
    truncated = _truncate(thinking, 1000)
    print(f"{_DIM}{_indent(truncated)}{_RESET}")


def log_response(ctx: LogContext, text: str) -> None:
    print(f"{_YELLOW}{_timestamp()} {_format_context(ctx)} 💬 Response{_RESET}")
    truncated = _truncate(text, 1000)
    print(f"{_DIM}{_indent(truncated)}{_RESET}")


def log_download_start(ctx: LogContext, filename: str, local_path: str) -> None:
    print(f"{_YELLOW}{_timestamp()} {_format_context(ctx)} ↓ Downloading attachment{_RESET}")
    print(f"{_DIM}           {filename} → {local_path}{_RESET}")


def log_download_success(ctx: LogContext, size_kb: float) -> None:
    print(
        f"{_YELLOW}{_timestamp()} {_format_context(ctx)} "
        f"✓ Downloaded ({size_kb:,.0f} KB){_RESET}"
    )


def log_download_error(ctx: LogContext, filename: str, error: str) -> None:
    print(f"{_YELLOW}{_timestamp()} {_format_context(ctx)} ✗ Download failed{_RESET}")
    print(f"{_DIM}           {filename}: {error}{_RESET}")


def log_stop_request(ctx: LogContext) -> None:
    print(f"{_GREEN}{_timestamp()} {_format_context(ctx)} stop{_RESET}")
    print(f"{_YELLOW}{_timestamp()} {_format_context(ctx)} ⊗ Stop requested - aborting{_RESET}")


def log_info(message: str) -> None:
    print(f"{_BLUE}{_timestamp()} [system] {message}{_RESET}")


def log_warning(message: str, details: str | None = None) -> None:
    print(f"{_YELLOW}{_timestamp()} [system] ⚠ {message}{_RESET}")
    if details:
        print(f"{_DIM}{_indent(details)}{_RESET}")


def log_agent_error(ctx: LogContext | str, error: str) -> None:
    if isinstance(ctx, str):
        context = "[system]"
    else:
        context = _format_context(ctx)
    print(f"{_YELLOW}{_timestamp()} {context} ✗ Agent error{_RESET}")
    print(f"{_DIM}{_indent(error)}{_RESET}")


def log_usage_summary(
    ctx: LogContext,
    usage: dict[str, Any],
    context_tokens: int | None = None,
    context_window: int | None = None,
) -> str:
    def format_tokens(count: int) -> str:
        if count < 1000:
            return str(count)
        if count < 10000:
            return f"{count / 1000:.1f}k"
        if count < 1_000_000:
            return f"{round(count / 1000)}k"
        return f"{count / 1_000_000:.1f}M"

    cost = usage.get("cost", {})
    lines: list[str] = []
    lines.append("*Usage Summary*")
    lines.append(f"Tokens: {usage['input']:,} in, {usage['output']:,} out")
    if usage.get("cacheRead", 0) > 0 or usage.get("cacheWrite", 0) > 0:
        lines.append(
            f"Cache: {usage['cacheRead']:,} read, {usage['cacheWrite']:,} write"
        )
    if context_tokens and context_window:
        pct = (context_tokens / context_window) * 100
        lines.append(
            f"Context: {format_tokens(context_tokens)} / "
            f"{format_tokens(context_window)} ({pct:.1f}%)"
        )
    cost_line = f"Cost: ${cost.get('input', 0):.4f} in, ${cost.get('output', 0):.4f} out"
    if usage.get("cacheRead", 0) > 0 or usage.get("cacheWrite", 0) > 0:
        cost_line += (
            f", ${cost.get('cacheRead', 0):.4f} cache read, "
            f"${cost.get('cacheWrite', 0):.4f} cache write"
        )
    lines.append(cost_line)
    lines.append(f"*Total: ${cost.get('total', 0):.4f}*")

    summary = "\n".join(lines)

    # Console log
    console_detail = (
        f"{usage['input']:,} in + {usage['output']:,} out"
    )
    if usage.get("cacheRead", 0) > 0 or usage.get("cacheWrite", 0) > 0:
        console_detail += (
            f" ({usage['cacheRead']:,} cache read, "
            f"{usage['cacheWrite']:,} cache write)"
        )
    console_detail += f" = ${cost.get('total', 0):.4f}"
    print(f"{_YELLOW}{_timestamp()} {_format_context(ctx)} 💰 Usage{_RESET}")
    print(f"{_DIM}           {console_detail}{_RESET}")

    return summary


def log_startup(working_dir: str, sandbox: str) -> None:
    print("Starting mom bot...")
    print(f"  Working directory: {working_dir}")
    print(f"  Sandbox: {sandbox}")


def log_connected() -> None:
    print("⚡️ Mom bot connected and listening!")
    print()


def log_disconnected() -> None:
    print("Mom bot disconnected.")


def log_backfill_start(channel_count: int) -> None:
    print(f"{_BLUE}{_timestamp()} [system] Backfilling {channel_count} channels...{_RESET}")


def log_backfill_channel(channel_name: str, message_count: int) -> None:
    print(
        f"{_BLUE}{_timestamp()} [system]   #{channel_name}: {message_count} messages{_RESET}"
    )


def log_backfill_complete(total_messages: int, duration_ms: float) -> None:
    duration = f"{duration_ms / 1000:.1f}"
    print(
        f"{_BLUE}{_timestamp()} [system] Backfill complete: "
        f"{total_messages} messages in {duration}s{_RESET}"
    )
