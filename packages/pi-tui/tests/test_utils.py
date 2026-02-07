"""Tests for pi.tui.utils -- terminal text utilities."""

from __future__ import annotations

from pi.tui.utils import (
    get_segmenter,
    is_punctuation_char,
    is_whitespace_char,
    truncate_to_width,
    visible_width,
    wrap_text_with_ansi,
)


# ---------------------------------------------------------------------------
# visible_width
# ---------------------------------------------------------------------------


class TestVisibleWidth:
    """Measure the visible terminal width of text."""

    def test_plain_ascii(self) -> None:
        assert visible_width("hello") == 5

    def test_empty_string(self) -> None:
        assert visible_width("") == 0

    def test_ansi_codes_do_not_count(self) -> None:
        # Bold "hi" then reset -- only "hi" contributes width.
        text = "\x1b[1mhi\x1b[0m"
        assert visible_width(text) == 2

    def test_multiple_ansi_codes(self) -> None:
        # Red bold "abc" then reset.
        text = "\x1b[1m\x1b[31mabc\x1b[0m"
        assert visible_width(text) == 3

    def test_wide_cjk_characters_count_as_two(self) -> None:
        # Chinese character U+4E16 ("world") is a wide character.
        assert visible_width("\u4e16") == 2

    def test_mixed_ascii_and_wide(self) -> None:
        # "A" (1) + U+4E16 (2) + "B" (1) = 4
        assert visible_width("A\u4e16B") == 4

    def test_tab_counts_as_three_spaces(self) -> None:
        assert visible_width("\t") == 3

    def test_osc8_hyperlink_does_not_count(self) -> None:
        # OSC 8 hyperlink wrapping "link".
        text = "\x1b]8;;https://example.com\x07link\x1b]8;;\x07"
        assert visible_width(text) == 4

    def test_apc_sequence_does_not_count(self) -> None:
        text = "\x1b_payload\x07visible"
        assert visible_width(text) == 7


# ---------------------------------------------------------------------------
# truncate_to_width
# ---------------------------------------------------------------------------


class TestTruncateToWidth:
    """Truncate text to a maximum visible width."""

    def test_short_text_unchanged(self) -> None:
        assert truncate_to_width("hi", 10) == "hi"

    def test_exact_width_unchanged(self) -> None:
        assert truncate_to_width("hello", 5) == "hello"

    def test_truncates_with_ellipsis(self) -> None:
        result = truncate_to_width("hello world", 5)
        # 5 columns total: 2 chars + "..." (3 cols) = 5
        assert visible_width(result) <= 5
        assert result.endswith("...")

    def test_truncate_with_custom_ellipsis(self) -> None:
        result = truncate_to_width("hello world", 6, ellipsis="..")
        assert visible_width(result) <= 6
        assert result.endswith("..")

    def test_truncate_zero_width_returns_empty(self) -> None:
        assert truncate_to_width("hello", 0) == ""

    def test_pad_fills_to_max_width(self) -> None:
        result = truncate_to_width("hi", 10, pad=True)
        assert visible_width(result) == 10
        assert result.startswith("hi")

    def test_truncate_with_ansi_preserves_codes(self) -> None:
        text = "\x1b[31mhello world\x1b[0m"
        result = truncate_to_width(text, 8)
        assert visible_width(result) <= 8


# ---------------------------------------------------------------------------
# wrap_text_with_ansi
# ---------------------------------------------------------------------------


class TestWrapTextWithAnsi:
    """Word-wrapping text while preserving ANSI codes."""

    def test_short_text_no_wrap(self) -> None:
        lines = wrap_text_with_ansi("hello", 80)
        assert lines == ["hello"]

    def test_wraps_at_word_boundary(self) -> None:
        lines = wrap_text_with_ansi("hello world", 6)
        assert len(lines) >= 2
        # First line should contain "hello" and second should contain "world".
        joined = " ".join(lines)
        assert "hello" in joined
        assert "world" in joined

    def test_preserves_embedded_newlines(self) -> None:
        lines = wrap_text_with_ansi("line1\nline2", 80)
        assert len(lines) == 2
        assert "line1" in lines[0]
        assert "line2" in lines[1]

    def test_long_word_forced_break(self) -> None:
        # A single long word that exceeds the width must be broken.
        lines = wrap_text_with_ansi("abcdefghijklmnop", 5)
        assert len(lines) >= 2
        for line in lines:
            assert visible_width(line) <= 5

    def test_zero_width_returns_text_as_is(self) -> None:
        lines = wrap_text_with_ansi("hello", 0)
        assert lines == ["hello"]

    def test_ansi_codes_preserved_across_wrap(self) -> None:
        # Bold text that wraps -- ANSI state should persist on continuation line.
        text = "\x1b[1m" + "a " * 20 + "\x1b[0m"
        lines = wrap_text_with_ansi(text, 10)
        assert len(lines) >= 2
        # The continuation lines should re-apply the bold code.
        for line in lines[1:]:
            assert "\x1b[1m" in line

    def test_empty_string(self) -> None:
        lines = wrap_text_with_ansi("", 10)
        assert lines == [""]


# ---------------------------------------------------------------------------
# is_whitespace_char / is_punctuation_char
# ---------------------------------------------------------------------------


class TestCharacterClassification:
    """Character classification helpers."""

    def test_space_is_whitespace(self) -> None:
        assert is_whitespace_char(" ") is True

    def test_tab_is_whitespace(self) -> None:
        assert is_whitespace_char("\t") is True

    def test_newline_is_whitespace(self) -> None:
        assert is_whitespace_char("\n") is True

    def test_carriage_return_is_whitespace(self) -> None:
        assert is_whitespace_char("\r") is True

    def test_letter_is_not_whitespace(self) -> None:
        assert is_whitespace_char("a") is False

    def test_digit_is_not_whitespace(self) -> None:
        assert is_whitespace_char("5") is False

    def test_period_is_punctuation(self) -> None:
        assert is_punctuation_char(".") is True

    def test_comma_is_punctuation(self) -> None:
        assert is_punctuation_char(",") is True

    def test_exclamation_is_punctuation(self) -> None:
        assert is_punctuation_char("!") is True

    def test_parentheses_are_punctuation(self) -> None:
        assert is_punctuation_char("(") is True
        assert is_punctuation_char(")") is True

    def test_brackets_are_punctuation(self) -> None:
        assert is_punctuation_char("[") is True
        assert is_punctuation_char("]") is True

    def test_operators_are_punctuation(self) -> None:
        assert is_punctuation_char("+") is True
        assert is_punctuation_char("-") is True
        assert is_punctuation_char("*") is True
        assert is_punctuation_char("/") is True

    def test_letter_is_not_punctuation(self) -> None:
        assert is_punctuation_char("a") is False

    def test_digit_is_not_punctuation(self) -> None:
        assert is_punctuation_char("7") is False

    def test_space_is_not_punctuation(self) -> None:
        assert is_punctuation_char(" ") is False


# ---------------------------------------------------------------------------
# get_segmenter
# ---------------------------------------------------------------------------


class TestGetSegmenter:
    """Grapheme segmenter wrapper."""

    def test_returns_object_with_segment_method(self) -> None:
        seg = get_segmenter()
        assert hasattr(seg, "segment")
        assert callable(seg.segment)

    def test_segment_ascii(self) -> None:
        seg = get_segmenter()
        result = seg.segment("hello")
        assert result == ["h", "e", "l", "l", "o"]

    def test_segment_empty_string(self) -> None:
        seg = get_segmenter()
        result = seg.segment("")
        assert result == []

    def test_segment_grapheme_clusters(self) -> None:
        """Combined characters should stay together as a single cluster."""
        seg = get_segmenter()
        # e + combining acute accent (U+0301) is one grapheme cluster
        text = "e\u0301"
        result = seg.segment(text)
        assert len(result) == 1
        assert result[0] == "e\u0301"

    def test_segment_mixed_text(self) -> None:
        seg = get_segmenter()
        # "A" + wide char + "B" should produce 3 clusters
        result = seg.segment("A\u4e16B")
        assert len(result) == 3
        assert result == ["A", "\u4e16", "B"]
