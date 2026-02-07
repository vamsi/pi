"""Tests for pi.mom.tools.truncate."""

from pi.mom.tools.truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    TruncationResult,
    format_size,
    truncate_head,
    truncate_tail,
)


class TestFormatSize:
    def test_bytes(self) -> None:
        assert format_size(0) == "0B"
        assert format_size(512) == "512B"
        assert format_size(1023) == "1023B"

    def test_kilobytes(self) -> None:
        assert format_size(1024) == "1.0KB"
        assert format_size(1536) == "1.5KB"
        assert format_size(100 * 1024) == "100.0KB"

    def test_megabytes(self) -> None:
        assert format_size(1024 * 1024) == "1.0MB"
        assert format_size(int(2.5 * 1024 * 1024)) == "2.5MB"


class TestTruncateHead:
    def test_no_truncation_needed(self) -> None:
        content = "line1\nline2\nline3"
        result = truncate_head(content)
        assert not result.truncated
        assert result.truncated_by is None
        assert result.content == content
        assert result.total_lines == 3
        assert result.output_lines == 3

    def test_empty_string(self) -> None:
        result = truncate_head("")
        assert not result.truncated
        assert result.content == ""
        assert result.total_lines == 1  # "".split("\n") => [""]

    def test_truncate_by_lines(self) -> None:
        content = "\n".join(f"line{i}" for i in range(100))
        result = truncate_head(content, max_lines=10)
        assert result.truncated
        assert result.truncated_by == "lines"
        assert result.output_lines == 10
        assert result.total_lines == 100

    def test_truncate_by_bytes(self) -> None:
        content = "\n".join("x" * 100 for _ in range(20))
        result = truncate_head(content, max_bytes=500)
        assert result.truncated
        assert result.truncated_by == "bytes"
        assert result.output_bytes <= 500

    def test_first_line_exceeds_limit(self) -> None:
        content = "x" * 200 + "\nline2"
        result = truncate_head(content, max_bytes=100)
        assert result.truncated
        assert result.first_line_exceeds_limit
        assert result.content == ""
        assert result.output_lines == 0

    def test_never_returns_partial_lines(self) -> None:
        content = "short\n" + "x" * 100 + "\nthird"
        result = truncate_head(content, max_bytes=50)
        assert result.truncated
        # Should only include "short" since adding the second line would exceed
        assert result.output_lines == 1
        assert result.content == "short"

    def test_multibyte_utf8(self) -> None:
        content = "héllo\nwörld"
        result = truncate_head(content, max_bytes=10)
        assert result.truncated
        # "héllo" in UTF-8 is 6 bytes
        assert result.output_lines == 1


class TestTruncateTail:
    def test_no_truncation_needed(self) -> None:
        content = "line1\nline2\nline3"
        result = truncate_tail(content)
        assert not result.truncated
        assert result.content == content

    def test_truncate_by_lines(self) -> None:
        content = "\n".join(f"line{i}" for i in range(100))
        result = truncate_tail(content, max_lines=10)
        assert result.truncated
        assert result.truncated_by == "lines"
        assert result.output_lines == 10
        # Should keep the LAST 10 lines
        assert "line99" in result.content
        assert "line90" in result.content

    def test_truncate_by_bytes(self) -> None:
        content = "\n".join("x" * 100 for _ in range(20))
        result = truncate_tail(content, max_bytes=500)
        assert result.truncated
        assert result.truncated_by == "bytes"
        assert result.output_bytes <= 500

    def test_last_line_partial_truncation(self) -> None:
        # Single very long line
        content = "x" * 200
        result = truncate_tail(content, max_bytes=100)
        assert result.truncated
        assert result.last_line_partial
        # Should keep the END of the line
        assert len(result.content.encode("utf-8")) <= 100

    def test_keeps_end_of_content(self) -> None:
        lines = [f"line{i}" for i in range(50)]
        content = "\n".join(lines)
        result = truncate_tail(content, max_lines=5)
        assert result.truncated
        assert "line49" in result.content
        assert "line45" in result.content
        assert "line44" not in result.content

    def test_multibyte_utf8_from_end(self) -> None:
        content = "àáâãäå" * 20  # lots of 2-byte characters
        result = truncate_tail(content, max_bytes=20)
        assert result.truncated
        assert len(result.content.encode("utf-8")) <= 20
