"""Shared truncation utilities for all tools."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 50 * 1024  # 50KB
GREP_MAX_LINE_LENGTH = 500


@dataclass
class TruncationResult:
    content: str
    truncated: bool
    truncated_by: str | None = None  # "lines" | "bytes" | None
    total_lines: int = 0
    total_bytes: int = 0
    output_lines: int = 0
    output_bytes: int = 0
    last_line_partial: bool = False
    first_line_exceeds_limit: bool = False
    max_lines: int = DEFAULT_MAX_LINES
    max_bytes: int = DEFAULT_MAX_BYTES


def truncate_head(
    text: str,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> TruncationResult:
    """Keep the first N lines, truncating to max_bytes.

    Used for file reads where the beginning is most relevant.
    """
    lines = text.split("\n")
    total_lines = len(lines)
    total_bytes = len(text.encode("utf-8"))

    if total_lines <= max_lines and total_bytes <= max_bytes:
        return TruncationResult(
            content=text,
            truncated=False,
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=total_lines,
            output_bytes=total_bytes,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    # Truncate by lines first
    kept = lines[:max_lines]
    result = "\n".join(kept)
    result_bytes = len(result.encode("utf-8"))
    truncated_by = "lines" if total_lines > max_lines else None

    # Then check bytes
    if result_bytes > max_bytes:
        result = truncate_string_to_bytes(result, max_bytes)
        result_bytes = len(result.encode("utf-8"))
        truncated_by = "bytes"

    output_lines = result.count("\n") + 1 if result else 0

    # Check if first line alone exceeds limit
    first_line_exceeds = False
    if lines and len(lines[0].encode("utf-8")) > max_bytes:
        first_line_exceeds = True

    return TruncationResult(
        content=result,
        truncated=True,
        truncated_by=truncated_by,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=output_lines,
        output_bytes=result_bytes,
        first_line_exceeds_limit=first_line_exceeds,
        max_lines=max_lines,
        max_bytes=max_bytes,
    )


def truncate_tail(
    text: str,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> TruncationResult:
    """Keep the last N lines, truncating to max_bytes.

    Used for bash output where the end is most relevant.
    """
    lines = text.split("\n")
    total_lines = len(lines)
    total_bytes = len(text.encode("utf-8"))

    if total_lines <= max_lines and total_bytes <= max_bytes:
        return TruncationResult(
            content=text,
            truncated=False,
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=total_lines,
            output_bytes=total_bytes,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    kept = lines[-max_lines:]
    result = "\n".join(kept)
    result_bytes = len(result.encode("utf-8"))
    truncated_by = "lines" if total_lines > max_lines else None

    if result_bytes > max_bytes:
        result = truncate_string_to_bytes_from_end(result, max_bytes)
        result_bytes = len(result.encode("utf-8"))
        truncated_by = "bytes"

    output_lines = result.count("\n") + 1 if result else 0

    return TruncationResult(
        content=result,
        truncated=True,
        truncated_by=truncated_by,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=output_lines,
        output_bytes=result_bytes,
        max_lines=max_lines,
        max_bytes=max_bytes,
    )


def truncate_line(line: str, max_length: int = GREP_MAX_LINE_LENGTH) -> tuple[str, bool]:
    """Truncate a single line to max characters."""
    if len(line) <= max_length:
        return line, False
    return line[:max_length] + "...", True


def format_size(bytes_count: int) -> str:
    """Format bytes as human-readable string."""
    if bytes_count < 1024:
        return f"{bytes_count}B"
    if bytes_count < 1024 * 1024:
        return f"{bytes_count / 1024:.1f}KB"
    return f"{bytes_count / (1024 * 1024):.1f}MB"


def truncate_string_to_bytes(text: str, max_bytes: int) -> str:
    """Truncate string to fit within max_bytes (UTF-8 safe, from start)."""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def truncate_string_to_bytes_from_end(text: str, max_bytes: int) -> str:
    """Truncate string to fit within max_bytes (UTF-8 safe, keep end)."""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[-max_bytes:].decode("utf-8", errors="ignore")
