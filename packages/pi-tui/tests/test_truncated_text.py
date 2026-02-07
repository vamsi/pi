"""Tests for the TruncatedText component."""

from __future__ import annotations

from pi.tui.components.truncated_text import TruncatedText
from pi.tui.utils import visible_width


class TestTruncatedTextRendersSingleLine:
    """TruncatedText renders exactly one content line (plus optional padding)."""

    def test_renders_one_content_line_no_padding(self) -> None:
        tt = TruncatedText("hello world", padding_x=0, padding_y=0)
        lines = tt.render(80)
        assert len(lines) == 1

    def test_renders_one_content_line_with_vertical_padding(self) -> None:
        tt = TruncatedText("hello world", padding_x=0, padding_y=2)
        lines = tt.render(80)
        # 2 top padding + 1 content + 2 bottom padding = 5
        assert len(lines) == 5

    def test_multiline_text_uses_only_first_line(self) -> None:
        tt = TruncatedText("first line\nsecond line", padding_x=0, padding_y=0)
        lines = tt.render(80)
        assert len(lines) == 1
        assert "first line" in lines[0]
        assert "second line" not in lines[0]


class TestTruncatedTextShortTextGetsPadded:
    """Short text is right-padded with spaces to fill the full width."""

    def test_short_text_padded_to_width(self) -> None:
        tt = TruncatedText("hi", padding_x=0, padding_y=0)
        lines = tt.render(20)
        assert visible_width(lines[0]) == 20

    def test_empty_text_padded_to_width(self) -> None:
        tt = TruncatedText("", padding_x=0, padding_y=0)
        lines = tt.render(30)
        assert visible_width(lines[0]) == 30
        assert lines[0].strip() == ""

    def test_short_text_with_horizontal_padding(self) -> None:
        tt = TruncatedText("hi", padding_x=2, padding_y=0)
        lines = tt.render(20)
        # Line starts with 2 spaces of left padding
        assert lines[0][:2] == "  "
        assert visible_width(lines[0]) == 20

    def test_exact_width_text_padded_correctly(self) -> None:
        text = "x" * 40
        tt = TruncatedText(text, padding_x=0, padding_y=0)
        lines = tt.render(40)
        assert visible_width(lines[0]) == 40


class TestTruncatedTextLongTextGetsTruncated:
    """Long text is truncated to fit within the available width."""

    def test_long_text_fits_within_width(self) -> None:
        long_text = "a" * 100
        tt = TruncatedText(long_text, padding_x=0, padding_y=0)
        lines = tt.render(20)
        assert visible_width(lines[0]) == 20

    def test_long_text_with_padding_fits_within_width(self) -> None:
        long_text = "b" * 100
        tt = TruncatedText(long_text, padding_x=3, padding_y=0)
        lines = tt.render(30)
        assert visible_width(lines[0]) == 30

    def test_truncated_text_contains_ellipsis(self) -> None:
        long_text = "abcdefghijklmnopqrstuvwxyz"
        tt = TruncatedText(long_text, padding_x=0, padding_y=0)
        lines = tt.render(10)
        # truncate_to_width uses "..." by default
        assert "..." in lines[0]

    def test_very_narrow_width_does_not_crash(self) -> None:
        tt = TruncatedText("hello world", padding_x=0, padding_y=0)
        lines = tt.render(1)
        assert len(lines) == 1
        assert visible_width(lines[0]) <= 1


class TestTruncatedTextInvalidate:
    """The invalidate method exists and is callable (no-op since no cache)."""

    def test_invalidate_is_callable(self) -> None:
        tt = TruncatedText("hello")
        # invalidate is a no-op for TruncatedText but should not raise
        tt.invalidate()

    def test_render_after_invalidate_returns_same_result(self) -> None:
        tt = TruncatedText("test content", padding_x=0, padding_y=0)
        lines_before = tt.render(40)
        tt.invalidate()
        lines_after = tt.render(40)
        assert lines_before == lines_after


class TestTruncatedTextVerticalPadding:
    """Vertical padding lines are empty lines padded to full width."""

    def test_vertical_padding_lines_are_spaces(self) -> None:
        tt = TruncatedText("content", padding_x=0, padding_y=1)
        lines = tt.render(20)
        # First line is top padding, last line is bottom padding
        assert lines[0] == " " * 20
        assert lines[-1] == " " * 20

    def test_vertical_padding_count_is_correct(self) -> None:
        tt = TruncatedText("content", padding_x=0, padding_y=3)
        lines = tt.render(20)
        # 3 top + 1 content + 3 bottom = 7
        assert len(lines) == 7
