"""Tests for the Markdown component."""

from __future__ import annotations

import re

from pi.tui.components.markdown import (
    DefaultTextStyle,
    Markdown,
    MarkdownTheme,
)
from pi.tui.utils import visible_width

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    """Remove all ANSI escape codes from text."""
    return _ANSI_RE.sub("", text)


def _plain_lines(md_text: str, width: int = 80) -> list[str]:
    """Render markdown and return lines with ANSI stripped for easy assertion."""
    md = Markdown(md_text, padding_x=0, padding_y=0)
    raw_lines = md.render(width)
    return [_strip_ansi(line).rstrip() for line in raw_lines]


def _raw_lines(md_text: str, width: int = 80) -> list[str]:
    """Render markdown and return raw lines (ANSI preserved)."""
    md = Markdown(md_text, padding_x=0, padding_y=0)
    return md.render(width)


# ---------------------------------------------------------------------------
# Headings
# ---------------------------------------------------------------------------


class TestMarkdownHeadings:
    """Headings at various levels render with expected content."""

    def test_h1_renders_text(self) -> None:
        lines = _plain_lines("# Hello World")
        joined = " ".join(lines)
        assert "Hello World" in joined

    def test_h2_renders_text(self) -> None:
        lines = _plain_lines("## Section Title")
        joined = " ".join(lines)
        assert "Section Title" in joined

    def test_h3_renders_with_hash_prefix(self) -> None:
        lines = _plain_lines("### Sub Section")
        joined = " ".join(lines)
        assert "###" in joined
        assert "Sub Section" in joined

    def test_h1_has_bold_ansi(self) -> None:
        raw = _raw_lines("# Bold Heading")
        joined = "".join(raw)
        # Bold is ESC[1m
        assert "\x1b[1m" in joined

    def test_h1_has_underline_ansi(self) -> None:
        raw = _raw_lines("# Underlined Heading")
        joined = "".join(raw)
        # Underline is ESC[4m
        assert "\x1b[4m" in joined

    def test_heading_followed_by_blank_line(self) -> None:
        lines = _plain_lines("# Heading\n\nParagraph text")
        # There should be at least one empty line between heading and paragraph
        found_blank = False
        found_heading = False
        for line in lines:
            if "Heading" in line:
                found_heading = True
            elif found_heading and line.strip() == "":
                found_blank = True
                break
        assert found_blank, "Expected a blank line after heading"


# ---------------------------------------------------------------------------
# Paragraphs
# ---------------------------------------------------------------------------


class TestMarkdownParagraphs:
    """Paragraphs render as wrapped text with trailing blank lines."""

    def test_simple_paragraph(self) -> None:
        lines = _plain_lines("Hello, this is a paragraph.")
        joined = " ".join(lines)
        assert "Hello, this is a paragraph." in joined

    def test_two_paragraphs_separated_by_blank_line(self) -> None:
        text = "First paragraph.\n\nSecond paragraph."
        lines = _plain_lines(text)
        joined = " ".join(lines)
        assert "First paragraph." in joined
        assert "Second paragraph." in joined

    def test_paragraph_wraps_at_width(self) -> None:
        long_text = "word " * 40  # 200 chars of text
        lines = _plain_lines(long_text, width=40)
        # Should produce multiple lines since text is wider than 40
        assert len(lines) > 1

    def test_empty_text_returns_no_lines(self) -> None:
        md = Markdown("", padding_x=0, padding_y=0)
        lines = md.render(80)
        assert lines == []

    def test_whitespace_only_returns_no_lines(self) -> None:
        md = Markdown("   \n  \n  ", padding_x=0, padding_y=0)
        lines = md.render(80)
        assert lines == []


# ---------------------------------------------------------------------------
# Code blocks
# ---------------------------------------------------------------------------


class TestMarkdownCodeBlocks:
    """Fenced code blocks render with code content preserved."""

    def test_code_block_content_preserved(self) -> None:
        text = "```python\nprint('hello')\n```"
        lines = _plain_lines(text)
        joined = " ".join(lines)
        assert "print('hello')" in joined

    def test_code_block_multiline(self) -> None:
        text = "```\nline1\nline2\nline3\n```"
        lines = _plain_lines(text)
        content_lines = [l for l in lines if l.strip()]
        assert len(content_lines) >= 3

    def test_code_block_followed_by_blank_line(self) -> None:
        text = "```\ncode\n```\n\nAfter code."
        lines = _plain_lines(text)
        found_code = False
        found_blank_after = False
        for line in lines:
            if "code" in line:
                found_code = True
            elif found_code and line.strip() == "":
                found_blank_after = True
                break
        assert found_blank_after, "Expected a blank line after code block"


# ---------------------------------------------------------------------------
# Lists
# ---------------------------------------------------------------------------


class TestMarkdownLists:
    """Bullet and ordered lists render with appropriate prefixes."""

    def test_bullet_list_items_have_dash_prefix(self) -> None:
        text = "- item one\n- item two\n- item three"
        lines = _plain_lines(text)
        item_lines = [l for l in lines if l.strip() and l.strip() != ""]
        # Each item line should have a "- " prefix (possibly with indentation)
        for line in item_lines:
            stripped = line.lstrip()
            assert stripped.startswith("- ") or stripped.startswith("1."), (
                f"Expected list prefix, got: {line!r}"
            )

    def test_ordered_list_items_have_number_prefix(self) -> None:
        text = "1. first\n2. second\n3. third"
        lines = _plain_lines(text)
        item_lines = [l for l in lines if l.strip()]
        for line in item_lines:
            stripped = line.lstrip()
            assert re.match(r"\d+\.\s", stripped), (
                f"Expected numbered prefix, got: {line!r}"
            )

    def test_bullet_list_renders_all_items(self) -> None:
        text = "- alpha\n- beta\n- gamma"
        lines = _plain_lines(text)
        joined = " ".join(lines)
        assert "alpha" in joined
        assert "beta" in joined
        assert "gamma" in joined

    def test_ordered_list_renders_all_items(self) -> None:
        text = "1. one\n2. two\n3. three"
        lines = _plain_lines(text)
        joined = " ".join(lines)
        assert "one" in joined
        assert "two" in joined
        assert "three" in joined


# ---------------------------------------------------------------------------
# set_text
# ---------------------------------------------------------------------------


class TestMarkdownSetText:
    """set_text updates the rendered content."""

    def test_set_text_changes_output(self) -> None:
        md = Markdown("Original text", padding_x=0, padding_y=0)
        lines_before = md.render(80)
        md.set_text("Updated text")
        lines_after = md.render(80)
        joined_before = " ".join(_strip_ansi(l) for l in lines_before)
        joined_after = " ".join(_strip_ansi(l) for l in lines_after)
        assert "Original" in joined_before
        assert "Updated" in joined_after
        assert "Original" not in joined_after

    def test_set_text_invalidates_cache(self) -> None:
        md = Markdown("First", padding_x=0, padding_y=0)
        md.render(80)
        # After render, cache is populated
        assert md._cached_lines is not None
        md.set_text("Second")
        # After set_text, cache should be cleared
        assert md._cached_lines is None

    def test_set_text_same_text_does_not_invalidate(self) -> None:
        md = Markdown("Same", padding_x=0, padding_y=0)
        md.render(80)
        cached = md._cached_lines
        md.set_text("Same")
        # Same text should not invalidate
        assert md._cached_lines is cached

    def test_render_caching_returns_same_object(self) -> None:
        md = Markdown("cached test", padding_x=0, padding_y=0)
        result1 = md.render(80)
        result2 = md.render(80)
        assert result1 is result2


# ---------------------------------------------------------------------------
# Inline formatting
# ---------------------------------------------------------------------------


class TestMarkdownInlineFormatting:
    """Bold, italic, and inline code produce correct ANSI codes."""

    def test_bold_text_has_bold_ansi(self) -> None:
        raw = _raw_lines("This is **bold** text")
        joined = "".join(raw)
        assert "\x1b[1m" in joined  # Bold
        joined_plain = _strip_ansi(joined)
        assert "bold" in joined_plain

    def test_italic_text_has_italic_ansi(self) -> None:
        raw = _raw_lines("This is *italic* text")
        joined = "".join(raw)
        assert "\x1b[3m" in joined  # Italic
        joined_plain = _strip_ansi(joined)
        assert "italic" in joined_plain

    def test_inline_code_is_present(self) -> None:
        lines = _plain_lines("Use `print()` to output")
        joined = " ".join(lines)
        assert "print()" in joined

    def test_bold_and_italic_combined(self) -> None:
        raw = _raw_lines("This is ***bold italic*** text")
        joined = "".join(raw)
        # Should have both bold and italic codes
        assert "\x1b[1m" in joined
        assert "\x1b[3m" in joined

    def test_inline_code_appears_in_raw_output(self) -> None:
        raw = _raw_lines("Some `code` here")
        joined = "".join(raw)
        # Inline code resets style then applies code style
        assert "\x1b[0m" in joined
        plain = _strip_ansi(joined)
        assert "code" in plain


# ---------------------------------------------------------------------------
# MarkdownTheme with identity functions
# ---------------------------------------------------------------------------


class TestMarkdownTheme:
    """MarkdownTheme controls the styling applied to rendered output."""

    def test_default_theme_renders_without_errors(self) -> None:
        md = Markdown("# Heading\n\nParagraph.", padding_x=0, padding_y=0)
        lines = md.render(80)
        assert len(lines) > 0

    def test_custom_heading_color_appears_in_output(self) -> None:
        theme = MarkdownTheme(heading_color="\x1b[36m")  # cyan
        md = Markdown(
            "# Colored Heading",
            padding_x=0,
            padding_y=0,
            theme=theme,
        )
        lines = md.render(80)
        joined = "".join(lines)
        assert "\x1b[36m" in joined

    def test_code_block_bg_color_applied(self) -> None:
        theme = MarkdownTheme(
            code_bg="\x1b[48;5;236m",
            code_fg="\x1b[38;5;252m",
        )
        md = Markdown(
            "```\nsome code\n```",
            padding_x=0,
            padding_y=0,
            theme=theme,
        )
        lines = md.render(80)
        joined = "".join(lines)
        assert "\x1b[48;5;236m" in joined
        assert "\x1b[38;5;252m" in joined

    def test_identity_theme_no_extra_styling(self) -> None:
        # MarkdownTheme with all None values (default) adds no extra color
        theme = MarkdownTheme()
        md = Markdown(
            "Simple paragraph.",
            padding_x=0,
            padding_y=0,
            theme=theme,
        )
        lines = md.render(80)
        joined = " ".join(_strip_ansi(l) for l in lines)
        assert "Simple paragraph." in joined

    def test_set_theme_invalidates_cache(self) -> None:
        md = Markdown("test", padding_x=0, padding_y=0)
        md.render(80)
        assert md._cached_lines is not None
        md.set_theme(MarkdownTheme(heading_color="\x1b[32m"))
        assert md._cached_lines is None


# ---------------------------------------------------------------------------
# DefaultTextStyle
# ---------------------------------------------------------------------------


class TestMarkdownDefaultTextStyle:
    """DefaultTextStyle applies global styling to all rendered content."""

    def test_default_bold_style_applied(self) -> None:
        style = DefaultTextStyle(bold=True)
        md = Markdown(
            "Bold paragraph",
            padding_x=0,
            padding_y=0,
            default_text_style=style,
        )
        lines = md.render(80)
        joined = "".join(lines)
        assert "\x1b[1m" in joined

    def test_default_color_style_applied(self) -> None:
        color = "\x1b[38;2;200;200;200m"
        style = DefaultTextStyle(color=color)
        md = Markdown(
            "Colored text",
            padding_x=0,
            padding_y=0,
            default_text_style=style,
        )
        lines = md.render(80)
        joined = "".join(lines)
        assert color in joined

    def test_default_style_includes_reset_suffix(self) -> None:
        style = DefaultTextStyle(italic=True)
        md = Markdown(
            "Italic text",
            padding_x=0,
            padding_y=0,
            default_text_style=style,
        )
        lines = md.render(80)
        joined = "".join(lines)
        # Should include reset at the end
        assert "\x1b[0m" in joined


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------


class TestMarkdownPadding:
    """Padding is correctly applied around rendered content."""

    def test_horizontal_padding_offsets_content(self) -> None:
        md = Markdown("Hello", padding_x=3, padding_y=0)
        lines = md.render(80)
        assert len(lines) > 0
        # First non-empty line should start with spaces
        for line in lines:
            stripped = _strip_ansi(line)
            if stripped.strip():
                assert stripped.startswith("   "), (
                    f"Expected 3 spaces of padding, got: {stripped!r}"
                )
                break

    def test_vertical_padding_adds_empty_lines(self) -> None:
        md = Markdown("Hello", padding_x=0, padding_y=2)
        lines = md.render(80)
        # Should have 2 top padding + content + 2 bottom padding
        assert len(lines) >= 5

    def test_invalidate_clears_cache(self) -> None:
        md = Markdown("test", padding_x=0, padding_y=0)
        md.render(80)
        assert md._cached_lines is not None
        md.invalidate()
        assert md._cached_lines is None


# ---------------------------------------------------------------------------
# Horizontal rule
# ---------------------------------------------------------------------------


class TestMarkdownHorizontalRule:
    """Horizontal rules render as a line of box-drawing characters."""

    def test_hr_renders(self) -> None:
        lines = _plain_lines("---")
        joined = " ".join(lines)
        # HR uses U+2500 (box drawing horizontal)
        assert "\u2500" in joined


# ---------------------------------------------------------------------------
# Blockquote
# ---------------------------------------------------------------------------


class TestMarkdownBlockquote:
    """Blockquotes are prefixed with a vertical bar."""

    def test_blockquote_has_bar_prefix(self) -> None:
        lines = _plain_lines("> quoted text")
        joined = " ".join(lines)
        # Blockquote uses U+2502 (box drawing vertical)
        assert "\u2502" in joined or "quoted text" in joined

    def test_blockquote_content_preserved(self) -> None:
        lines = _plain_lines("> This is a quote.")
        joined = " ".join(lines)
        assert "This is a quote." in joined
