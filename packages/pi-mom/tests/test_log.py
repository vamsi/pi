"""Tests for pi.mom.log."""

import io
import sys

from pi.mom.log import (
    LogContext,
    log_agent_error,
    log_backfill_channel,
    log_backfill_complete,
    log_backfill_start,
    log_connected,
    log_disconnected,
    log_download_error,
    log_download_start,
    log_download_success,
    log_info,
    log_response,
    log_response_start,
    log_startup,
    log_stop_request,
    log_thinking,
    log_tool_error,
    log_tool_start,
    log_tool_success,
    log_usage_summary,
    log_user_message,
    log_warning,
)


def _capture_output(fn, *args, **kwargs) -> str:
    """Capture stdout from a function call."""
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        result = fn(*args, **kwargs)
    finally:
        sys.stdout = old
    return buf.getvalue()


class TestLogContext:
    def test_dm_context(self) -> None:
        ctx = LogContext(channel_id="D123", user_name="mario")
        output = _capture_output(log_user_message, ctx, "hi")
        assert "DM:mario" in output

    def test_channel_context(self) -> None:
        ctx = LogContext(channel_id="C123", user_name="mario", channel_name="dev")
        output = _capture_output(log_user_message, ctx, "hi")
        assert "#dev:mario" in output

    def test_channel_without_name(self) -> None:
        ctx = LogContext(channel_id="C123", user_name="mario")
        output = _capture_output(log_user_message, ctx, "hi")
        assert "C123:mario" in output


class TestLogFunctions:
    def test_log_user_message(self) -> None:
        ctx = LogContext("C1", "user1")
        output = _capture_output(log_user_message, ctx, "hello world")
        assert "hello world" in output

    def test_log_tool_start(self) -> None:
        ctx = LogContext("C1", "user1")
        output = _capture_output(
            log_tool_start, ctx, "bash", "list files", {"command": "ls"}
        )
        assert "bash" in output
        assert "list files" in output

    def test_log_tool_start_with_path(self) -> None:
        ctx = LogContext("C1", "user1")
        output = _capture_output(
            log_tool_start,
            ctx,
            "read",
            "read file",
            {"path": "/tmp/test.py", "offset": 1, "limit": 10},
        )
        assert "/tmp/test.py:1-11" in output

    def test_log_tool_success(self) -> None:
        ctx = LogContext("C1", "user1")
        output = _capture_output(
            log_tool_success, ctx, "bash", 1500, "output text"
        )
        assert "bash" in output
        assert "1.5s" in output

    def test_log_tool_error(self) -> None:
        ctx = LogContext("C1", "user1")
        output = _capture_output(
            log_tool_error, ctx, "bash", 2000, "error message"
        )
        assert "✗" in output
        assert "error message" in output

    def test_log_info(self) -> None:
        output = _capture_output(log_info, "test message")
        assert "test message" in output
        assert "[system]" in output

    def test_log_warning(self) -> None:
        output = _capture_output(log_warning, "warning!", "details here")
        assert "⚠" in output
        assert "warning!" in output
        assert "details here" in output

    def test_log_warning_no_details(self) -> None:
        output = _capture_output(log_warning, "simple warning")
        assert "simple warning" in output

    def test_log_startup(self) -> None:
        output = _capture_output(log_startup, "/work", "host")
        assert "/work" in output
        assert "host" in output

    def test_log_connected(self) -> None:
        output = _capture_output(log_connected)
        assert "connected" in output.lower()

    def test_log_disconnected(self) -> None:
        output = _capture_output(log_disconnected)
        assert "disconnected" in output.lower()

    def test_log_agent_error_with_context(self) -> None:
        ctx = LogContext("C1", "user1")
        output = _capture_output(log_agent_error, ctx, "boom")
        assert "Agent error" in output

    def test_log_agent_error_system(self) -> None:
        output = _capture_output(log_agent_error, "system", "boom")
        assert "[system]" in output

    def test_log_usage_summary(self) -> None:
        ctx = LogContext("C1", "user1")
        usage = {
            "input": 1000,
            "output": 500,
            "cacheRead": 200,
            "cacheWrite": 100,
            "cost": {
                "input": 0.01,
                "output": 0.02,
                "cacheRead": 0.001,
                "cacheWrite": 0.002,
                "total": 0.033,
            },
        }
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            result = log_usage_summary(ctx, usage, 50000, 200000)
        finally:
            sys.stdout = old

        assert "*Usage Summary*" in result
        assert "1,000 in" in result
        assert "500 out" in result
        assert "$0.0330" in result

    def test_log_usage_summary_no_cache(self) -> None:
        ctx = LogContext("C1", "user1")
        usage = {
            "input": 100,
            "output": 50,
            "cacheRead": 0,
            "cacheWrite": 0,
            "cost": {
                "input": 0.001,
                "output": 0.002,
                "cacheRead": 0,
                "cacheWrite": 0,
                "total": 0.003,
            },
        }
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            result = log_usage_summary(ctx, usage)
        finally:
            sys.stdout = old

        assert "Cache" not in result


class TestLogDownload:
    def test_log_download_start(self) -> None:
        ctx = LogContext("C1", "user1")
        output = _capture_output(log_download_start, ctx, "file.png", "/tmp/file.png")
        assert "file.png" in output

    def test_log_download_success(self) -> None:
        ctx = LogContext("C1", "user1")
        output = _capture_output(log_download_success, ctx, 1024)
        assert "1,024" in output

    def test_log_download_error(self) -> None:
        ctx = LogContext("C1", "user1")
        output = _capture_output(log_download_error, ctx, "file.png", "404")
        assert "file.png" in output
        assert "404" in output


class TestLogBackfill:
    def test_log_backfill_start(self) -> None:
        output = _capture_output(log_backfill_start, 5)
        assert "5 channels" in output

    def test_log_backfill_channel(self) -> None:
        output = _capture_output(log_backfill_channel, "general", 42)
        assert "#general" in output
        assert "42" in output

    def test_log_backfill_complete(self) -> None:
        output = _capture_output(log_backfill_complete, 100, 2500)
        assert "100 messages" in output
        assert "2.5s" in output
