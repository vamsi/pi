"""Markdown component -- renders markdown text to styled terminal output.

Faithfully ported from the TypeScript Markdown component (~770 lines),
adapted to use ``markdown-it-py`` instead of the ``marked`` library.

Key differences from the TypeScript original:
- markdown-it-py uses an open/close tag model (``heading_open`` / ``heading_close``)
  rather than marked's nested token structure.
- Inline content lives in ``token.children`` of ``inline`` tokens.
- Code fences use ``fence`` token type with ``token.info`` for the language.
- Lists: ``bullet_list_open/close``, ``ordered_list_open/close``,
  ``list_item_open/close``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from markdown_it import MarkdownIt
from markdown_it.token import Token

from pi.tui.terminal_image import is_image_line
from pi.tui.utils import apply_background_to_line, visible_width, wrap_text_with_ansi

# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

_RESET = "\x1b[0m"
_BOLD = "\x1b[1m"
_DIM = "\x1b[2m"
_ITALIC = "\x1b[3m"
_UNDERLINE = "\x1b[4m"
_STRIKETHROUGH = "\x1b[9m"

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


# ---------------------------------------------------------------------------
# DefaultTextStyle
# ---------------------------------------------------------------------------


@dataclass
class DefaultTextStyle:
    """Default text styling applied to all rendered markdown content."""

    color: str | None = None  # e.g. "\x1b[38;2;R;G;Bm"
    bold: bool = False
    italic: bool = False
    dim: bool = False
    underline: bool = False


# ---------------------------------------------------------------------------
# MarkdownTheme
# ---------------------------------------------------------------------------


@dataclass
class MarkdownTheme:
    """Colour / style theme for the markdown renderer."""

    heading_color: str | None = None
    code_bg: str | None = None
    code_fg: str | None = None
    inline_code_bg: str | None = None
    inline_code_fg: str | None = None
    link_color: str | None = None
    blockquote_color: str | None = None
    hr_color: str | None = None
    table_border_color: str | None = None
    table_header_color: str | None = None


# ---------------------------------------------------------------------------
# InlineStyleContext (internal)
# ---------------------------------------------------------------------------


@dataclass
class _InlineStyleContext:
    """Tracks ANSI state while walking inline tokens."""

    bold: bool = False
    italic: bool = False
    strikethrough: bool = False
    code: bool = False
    link_href: str | None = None


# ---------------------------------------------------------------------------
# SyntaxHighlightCallback protocol
# ---------------------------------------------------------------------------

SyntaxHighlightFn = Callable[[str, str], str]  # (code, language) -> highlighted


# ---------------------------------------------------------------------------
# markdown-it singleton (with GFM table + strikethrough)
# ---------------------------------------------------------------------------

# We use "gfm-like" to get table + strikethrough enabled by default.
# linkify is also enabled but we don't rely on it.
_md_parser = MarkdownIt("gfm-like")


# ---------------------------------------------------------------------------
# Markdown component
# ---------------------------------------------------------------------------


class Markdown:
    """Renders a markdown string to a list of terminal lines."""

    def __init__(
        self,
        text: str = "",
        *,
        padding_x: int = 1,
        padding_y: int = 0,
        theme: MarkdownTheme | None = None,
        default_text_style: DefaultTextStyle | None = None,
        syntax_highlight_fn: SyntaxHighlightFn | None = None,
        custom_bg_fn: Callable[[str], str] | None = None,
    ) -> None:
        self._text = text
        self._padding_x = padding_x
        self._padding_y = padding_y
        self._theme = theme or MarkdownTheme()
        self._default_text_style = default_text_style or DefaultTextStyle()
        self._syntax_highlight_fn = syntax_highlight_fn
        self._custom_bg_fn = custom_bg_fn

        # Rendering cache
        self._cached_text: str | None = None
        self._cached_width: int | None = None
        self._cached_lines: list[str] | None = None

    # -- public API ---------------------------------------------------------

    def set_text(self, text: str) -> None:
        if text != self._text:
            self._text = text
            self._invalidate_cache()

    def set_theme(self, theme: MarkdownTheme) -> None:
        self._theme = theme
        self._invalidate_cache()

    def set_default_text_style(self, style: DefaultTextStyle) -> None:
        self._default_text_style = style
        self._invalidate_cache()

    def set_syntax_highlight_fn(self, fn: SyntaxHighlightFn | None) -> None:
        self._syntax_highlight_fn = fn
        self._invalidate_cache()

    def set_custom_bg_fn(self, fn: Callable[[str], str] | None) -> None:
        self._custom_bg_fn = fn
        self._invalidate_cache()

    def invalidate(self) -> None:
        self._invalidate_cache()

    def render(self, width: int) -> list[str]:
        if (
            self._cached_lines is not None
            and self._cached_text == self._text
            and self._cached_width == width
        ):
            return self._cached_lines

        lines = self._render_markdown(width)

        self._cached_text = self._text
        self._cached_width = width
        self._cached_lines = lines
        return lines

    # -- cache --------------------------------------------------------------

    def _invalidate_cache(self) -> None:
        self._cached_text = None
        self._cached_width = None
        self._cached_lines = None

    # -- default text style prefix / suffix ---------------------------------

    def _default_style_prefix(self) -> str:
        parts: list[str] = []
        s = self._default_text_style
        if s.color:
            parts.append(s.color)
        if s.bold:
            parts.append(_BOLD)
        if s.italic:
            parts.append(_ITALIC)
        if s.dim:
            parts.append(_DIM)
        if s.underline:
            parts.append(_UNDERLINE)
        return "".join(parts)

    def _default_style_suffix(self) -> str:
        s = self._default_text_style
        if s.color or s.bold or s.italic or s.dim or s.underline:
            return _RESET
        return ""

    # -- main rendering entry -----------------------------------------------

    def _render_markdown(self, width: int) -> list[str]:
        if not self._text or not self._text.strip():
            return []

        content_width = max(1, width - self._padding_x * 2)

        # Parse markdown into tokens
        tokens = _md_parser.parse(self._text)

        # Render block tokens into lines
        raw_lines = self._render_tokens(tokens, content_width)

        # Apply padding and background
        result = self._apply_padding_and_bg(raw_lines, width, content_width)

        return result

    # -- block-level token dispatch -----------------------------------------

    def _render_tokens(self, tokens: list[Token], width: int) -> list[str]:
        """Walk the top-level token list and dispatch to renderers."""
        lines: list[str] = []
        i = 0
        n = len(tokens)

        while i < n:
            tok = tokens[i]
            t = tok.type

            # Heading: heading_open ... inline ... heading_close
            if t == "heading_open":
                level = int(tok.tag[1]) if tok.tag and tok.tag[0] == "h" else 1
                # The next token should be inline
                inline_tok = tokens[i + 1] if i + 1 < n else None
                text = self._render_inline(inline_tok) if inline_tok and inline_tok.type == "inline" else ""
                lines.extend(self._render_heading(text, level, width))
                # skip to heading_close
                i = self._skip_to_close(tokens, i, "heading_close")
                i += 1
                continue

            # Paragraph: paragraph_open ... inline ... paragraph_close
            if t == "paragraph_open":
                inline_tok = tokens[i + 1] if i + 1 < n else None
                text = self._render_inline(inline_tok) if inline_tok and inline_tok.type == "inline" else ""
                lines.extend(self._render_paragraph(text, width))
                i = self._skip_to_close(tokens, i, "paragraph_close")
                i += 1
                continue

            # Fence (code block)
            if t == "fence":
                lang = tok.info.strip() if tok.info else ""
                code = tok.content
                # Remove trailing newline if present
                if code.endswith("\n"):
                    code = code[:-1]
                lines.extend(self._render_code_block(code, lang, width))
                i += 1
                continue

            # Indented code block
            if t == "code_block":
                code = tok.content
                if code.endswith("\n"):
                    code = code[:-1]
                lines.extend(self._render_code_block(code, "", width))
                i += 1
                continue

            # Bullet list
            if t == "bullet_list_open":
                close_idx = self._find_matching_close(tokens, i, "bullet_list_open", "bullet_list_close")
                sub_tokens = tokens[i + 1 : close_idx]
                lines.extend(self._render_list(sub_tokens, ordered=False, width=width, depth=0))
                i = close_idx + 1
                continue

            # Ordered list
            if t == "ordered_list_open":
                start = 1
                start_attr = tok.attrs.get("start")
                if start_attr is not None:
                    try:
                        start = int(start_attr)
                    except (ValueError, TypeError):
                        start = 1
                close_idx = self._find_matching_close(tokens, i, "ordered_list_open", "ordered_list_close")
                sub_tokens = tokens[i + 1 : close_idx]
                lines.extend(self._render_list(sub_tokens, ordered=True, width=width, depth=0, start=start))
                i = close_idx + 1
                continue

            # Blockquote
            if t == "blockquote_open":
                close_idx = self._find_matching_close(tokens, i, "blockquote_open", "blockquote_close")
                sub_tokens = tokens[i + 1 : close_idx]
                lines.extend(self._render_blockquote(sub_tokens, width))
                i = close_idx + 1
                continue

            # Horizontal rule
            if t == "hr":
                lines.extend(self._render_hr(width))
                i += 1
                continue

            # Table
            if t == "table_open":
                close_idx = self._find_matching_close(tokens, i, "table_open", "table_close")
                sub_tokens = tokens[i + 1 : close_idx]
                lines.extend(self._render_table(sub_tokens, width))
                i = close_idx + 1
                continue

            # HTML block -- render as-is
            if t == "html_block":
                content = tok.content.rstrip("\n")
                if content:
                    prefix = self._default_style_prefix()
                    suffix = self._default_style_suffix()
                    for line in content.split("\n"):
                        lines.extend(wrap_text_with_ansi(prefix + line + suffix, width))
                    lines.append("")
                i += 1
                continue

            # Standalone inline token (shouldn't normally happen at top level)
            if t == "inline":
                text = self._render_inline(tok)
                if text:
                    prefix = self._default_style_prefix()
                    suffix = self._default_style_suffix()
                    wrapped = wrap_text_with_ansi(prefix + text + suffix, width)
                    lines.extend(wrapped)
                    lines.append("")
                i += 1
                continue

            # Skip unknown / closing tokens
            i += 1

        # Remove trailing empty lines
        while lines and lines[-1] == "":
            lines.pop()

        return lines

    # -- heading ------------------------------------------------------------

    def _render_heading(self, text: str, level: int, width: int) -> list[str]:
        lines: list[str] = []
        theme = self._theme

        heading_color = theme.heading_color or ""
        prefix = self._default_style_prefix()
        suffix = self._default_style_suffix()

        if level == 1:
            # Bold + underline
            styled = f"{prefix}{heading_color}{_BOLD}{_UNDERLINE}{text}{_RESET}{suffix}"
            wrapped = wrap_text_with_ansi(styled, width)
            lines.extend(wrapped)
        elif level == 2:
            # Bold
            styled = f"{prefix}{heading_color}{_BOLD}{text}{_RESET}{suffix}"
            wrapped = wrap_text_with_ansi(styled, width)
            lines.extend(wrapped)
        else:
            # h3+ : dim prefix "### " + bold text
            hashes = "#" * level
            styled = f"{prefix}{_DIM}{hashes}{_RESET} {prefix}{heading_color}{_BOLD}{text}{_RESET}{suffix}"
            wrapped = wrap_text_with_ansi(styled, width)
            lines.extend(wrapped)

        # blank line after heading
        lines.append("")
        return lines

    # -- paragraph ----------------------------------------------------------

    def _render_paragraph(self, text: str, width: int) -> list[str]:
        lines: list[str] = []
        prefix = self._default_style_prefix()
        suffix = self._default_style_suffix()

        styled = prefix + text + suffix
        wrapped = wrap_text_with_ansi(styled, width)
        lines.extend(wrapped)
        lines.append("")  # blank line after paragraph
        return lines

    # -- code block ---------------------------------------------------------

    def _render_code_block(self, code: str, lang: str, width: int) -> list[str]:
        lines: list[str] = []
        theme = self._theme

        code_bg = theme.code_bg or ""
        code_fg = theme.code_fg or ""

        highlighted = code
        if lang and self._syntax_highlight_fn:
            try:
                highlighted = self._syntax_highlight_fn(code, lang)
            except Exception:
                highlighted = code

        code_lines = highlighted.split("\n")

        for code_line in code_lines:
            # Skip image lines (they contain terminal image escape sequences)
            if is_image_line(code_line):
                lines.append(code_line)
                continue

            # Replace tabs with spaces
            code_line = code_line.replace("\t", "   ")

            styled = f"{code_bg}{code_fg}{code_line}{_RESET}"
            # Pad code line to full width for background
            line_width = visible_width(styled)
            if line_width < width:
                padding = " " * (width - line_width)
                styled = f"{code_bg}{code_fg}{code_line}{padding}{_RESET}"

            lines.append(styled)

        # blank line after code block
        lines.append("")
        return lines

    # -- inline rendering ---------------------------------------------------

    def _render_inline(self, tok: Token | None) -> str:
        """Render an ``inline`` token's children into a flat styled string."""
        if tok is None or tok.children is None:
            return tok.content if tok else ""

        parts: list[str] = []
        ctx = _InlineStyleContext()
        theme = self._theme
        prefix = self._default_style_prefix()
        suffix = self._default_style_suffix()

        for child in tok.children:
            ct = child.type

            # Plain text
            if ct == "text":
                parts.append(self._styled_text(child.content, ctx))
                continue

            # Soft/hard break
            if ct == "softbreak":
                parts.append(" ")
                continue
            if ct == "hardbreak":
                parts.append("\n")
                continue

            # Bold open/close
            if ct == "strong_open":
                ctx.bold = True
                continue
            if ct == "strong_close":
                ctx.bold = False
                continue

            # Italic open/close
            if ct == "em_open":
                ctx.italic = True
                continue
            if ct == "em_close":
                ctx.italic = False
                continue

            # Strikethrough open/close
            if ct == "s_open":
                ctx.strikethrough = True
                continue
            if ct == "s_close":
                ctx.strikethrough = False
                continue

            # Inline code
            if ct == "code_inline":
                code_text = child.content
                inline_code_bg = theme.inline_code_bg or ""
                inline_code_fg = theme.inline_code_fg or ""
                parts.append(f"{_RESET}{inline_code_bg}{inline_code_fg} {code_text} {_RESET}{prefix}")
                continue

            # Link open/close
            if ct == "link_open":
                href = child.attrs.get("href", "")
                ctx.link_href = str(href) if href else None
                continue
            if ct == "link_close":
                href = ctx.link_href
                if href:
                    link_color = theme.link_color or ""
                    parts.append(f"{_RESET}{_DIM} ({href}){_RESET}{prefix}")
                ctx.link_href = None
                continue

            # Image
            if ct == "image":
                alt = child.content or "image"
                src = child.attrs.get("src", "")
                parts.append(f"[{alt}]")
                if src:
                    link_color = theme.link_color or ""
                    parts.append(f"{_RESET}{_DIM} ({src}){_RESET}{prefix}")
                continue

            # HTML inline -- just include as-is
            if ct == "html_inline":
                # Strip HTML tags for terminal display
                stripped = re.sub(r"<[^>]+>", "", child.content)
                if stripped:
                    parts.append(self._styled_text(stripped, ctx))
                continue

            # Fallback: include content if any
            if child.content:
                parts.append(self._styled_text(child.content, ctx))

        return "".join(parts)

    def _styled_text(self, text: str, ctx: _InlineStyleContext) -> str:
        """Apply inline style context to plain text."""
        if not text:
            return ""

        parts: list[str] = []
        theme = self._theme

        if ctx.link_href:
            link_color = theme.link_color or ""
            if link_color:
                parts.append(link_color)
            parts.append(_UNDERLINE)

        if ctx.bold:
            parts.append(_BOLD)
        if ctx.italic:
            parts.append(_ITALIC)
        if ctx.strikethrough:
            parts.append(_STRIKETHROUGH)

        style_prefix = "".join(parts)

        if style_prefix:
            return f"{style_prefix}{text}{_RESET}{self._default_style_prefix()}"
        return text

    # -- list ---------------------------------------------------------------

    def _render_list(
        self,
        tokens: list[Token],
        *,
        ordered: bool,
        width: int,
        depth: int,
        start: int = 1,
    ) -> list[str]:
        lines: list[str] = []
        item_index = start
        indent = "  " * depth
        i = 0
        n = len(tokens)

        while i < n:
            tok = tokens[i]

            if tok.type == "list_item_open":
                # Find the matching list_item_close
                close_idx = self._find_matching_close(tokens, i, "list_item_open", "list_item_close")
                item_tokens = tokens[i + 1 : close_idx]

                # Build the bullet/number prefix
                if ordered:
                    bullet = f"{indent}{item_index}. "
                    item_index += 1
                else:
                    bullet = f"{indent}- "

                bullet_width = visible_width(bullet)
                continuation_indent = " " * bullet_width

                # Render the item body
                item_lines = self._render_list_item(item_tokens, width - bullet_width, depth)

                # Prepend bullet to first line, indent continuation lines
                for j, item_line in enumerate(item_lines):
                    if j == 0:
                        lines.append(bullet + item_line)
                    else:
                        if item_line == "":
                            lines.append("")
                        else:
                            lines.append(continuation_indent + item_line)

                i = close_idx + 1
                continue

            i += 1

        # Blank line after list (only at top-level depth 0)
        if depth == 0:
            lines.append("")

        return lines

    def _render_list_item(self, tokens: list[Token], width: int, depth: int) -> list[str]:
        """Render the content inside a list item."""
        lines: list[str] = []
        i = 0
        n = len(tokens)

        while i < n:
            tok = tokens[i]
            t = tok.type

            # Paragraph inside list item (may be hidden for tight lists)
            if t == "paragraph_open":
                inline_tok = tokens[i + 1] if i + 1 < n else None
                text = self._render_inline(inline_tok) if inline_tok and inline_tok.type == "inline" else ""
                prefix = self._default_style_prefix()
                suffix = self._default_style_suffix()
                styled = prefix + text + suffix
                wrapped = wrap_text_with_ansi(styled, width)
                lines.extend(wrapped)
                i = self._skip_to_close(tokens, i, "paragraph_close")
                i += 1
                continue

            # Inline token directly (tight list)
            if t == "inline":
                text = self._render_inline(tok)
                prefix = self._default_style_prefix()
                suffix = self._default_style_suffix()
                styled = prefix + text + suffix
                wrapped = wrap_text_with_ansi(styled, width)
                lines.extend(wrapped)
                i += 1
                continue

            # Nested bullet list
            if t == "bullet_list_open":
                close_idx = self._find_matching_close(tokens, i, "bullet_list_open", "bullet_list_close")
                sub_tokens = tokens[i + 1 : close_idx]
                nested_lines = self._render_list(sub_tokens, ordered=False, width=width, depth=depth + 1)
                lines.extend(nested_lines)
                i = close_idx + 1
                continue

            # Nested ordered list
            if t == "ordered_list_open":
                s = 1
                start_attr = tok.attrs.get("start")
                if start_attr is not None:
                    try:
                        s = int(start_attr)
                    except (ValueError, TypeError):
                        s = 1
                close_idx = self._find_matching_close(tokens, i, "ordered_list_open", "ordered_list_close")
                sub_tokens = tokens[i + 1 : close_idx]
                nested_lines = self._render_list(sub_tokens, ordered=True, width=width, depth=depth + 1, start=s)
                lines.extend(nested_lines)
                i = close_idx + 1
                continue

            # Blockquote inside list item
            if t == "blockquote_open":
                close_idx = self._find_matching_close(tokens, i, "blockquote_open", "blockquote_close")
                sub_tokens = tokens[i + 1 : close_idx]
                lines.extend(self._render_blockquote(sub_tokens, width))
                i = close_idx + 1
                continue

            # Fence inside list item
            if t == "fence":
                lang = tok.info.strip() if tok.info else ""
                code = tok.content
                if code.endswith("\n"):
                    code = code[:-1]
                lines.extend(self._render_code_block(code, lang, width))
                i += 1
                continue

            i += 1

        return lines

    # -- blockquote ---------------------------------------------------------

    def _render_blockquote(self, tokens: list[Token], width: int) -> list[str]:
        theme = self._theme
        bq_color = theme.blockquote_color or ""

        border = f"{bq_color}\u2502 {_RESET}"
        border_width = 2  # "| " is 2 visible chars

        # Render inner content with reduced width
        inner_width = max(1, width - border_width)
        inner_lines = self._render_tokens(tokens, inner_width)

        lines: list[str] = []
        for line in inner_lines:
            lines.append(f"{border}{line}")

        lines.append("")  # blank line after blockquote
        return lines

    # -- horizontal rule ----------------------------------------------------

    def _render_hr(self, width: int) -> list[str]:
        theme = self._theme
        hr_color = theme.hr_color or ""
        rule = "\u2500" * width
        styled = f"{hr_color}{_DIM}{rule}{_RESET}"
        return [styled, ""]

    # -- table --------------------------------------------------------------

    def _render_table(self, tokens: list[Token], width: int) -> list[str]:
        """Render a GFM table with box-drawing borders."""
        theme = self._theme
        border_color = theme.table_border_color or ""
        header_color = theme.table_header_color or ""

        # Parse the table tokens into header cells and body rows
        header_cells: list[str] = []
        body_rows: list[list[str]] = []

        self._parse_table_tokens(tokens, header_cells, body_rows)

        if not header_cells:
            return []

        num_cols = len(header_cells)

        # Calculate column widths
        col_widths = self._calculate_column_widths(header_cells, body_rows, num_cols, width)

        lines: list[str] = []

        # Top border: ┌───┬───┐
        lines.append(self._table_top_border(col_widths, border_color))

        # Header row
        header_lines = self._render_table_row(
            header_cells, col_widths, border_color, cell_style=f"{header_color}{_BOLD}"
        )
        lines.extend(header_lines)

        # Header separator: ├───┼───┤
        lines.append(self._table_mid_border(col_widths, border_color))

        # Body rows
        for row_idx, row in enumerate(body_rows):
            row_lines = self._render_table_row(row, col_widths, border_color)
            lines.extend(row_lines)
            # Add row separator between body rows (but not after the last)
            if row_idx < len(body_rows) - 1:
                lines.append(self._table_mid_border(col_widths, border_color))

        # Bottom border: └───┴───┘
        lines.append(self._table_bottom_border(col_widths, border_color))

        lines.append("")  # blank line after table
        return lines

    def _parse_table_tokens(
        self,
        tokens: list[Token],
        header_cells: list[str],
        body_rows: list[list[str]],
    ) -> None:
        """Walk table sub-tokens and extract header cells and body rows."""
        i = 0
        n = len(tokens)
        in_thead = False
        in_tbody = False
        current_row: list[str] | None = None

        while i < n:
            tok = tokens[i]
            t = tok.type

            if t == "thead_open":
                in_thead = True
                i += 1
                continue
            if t == "thead_close":
                in_thead = False
                i += 1
                continue
            if t == "tbody_open":
                in_tbody = True
                i += 1
                continue
            if t == "tbody_close":
                in_tbody = False
                i += 1
                continue

            if t == "tr_open":
                current_row = []
                i += 1
                continue
            if t == "tr_close":
                if current_row is not None:
                    if in_thead:
                        header_cells.extend(current_row)
                    elif in_tbody:
                        body_rows.append(current_row)
                current_row = None
                i += 1
                continue

            if t in ("th_open", "td_open"):
                # Next token should be inline
                inline_tok = tokens[i + 1] if i + 1 < n else None
                cell_text = ""
                if inline_tok and inline_tok.type == "inline":
                    cell_text = self._render_inline(inline_tok)
                    i += 1  # skip the inline token
                if current_row is not None:
                    current_row.append(cell_text)
                i += 1
                continue

            if t in ("th_close", "td_close"):
                i += 1
                continue

            i += 1

    def _calculate_column_widths(
        self,
        header_cells: list[str],
        body_rows: list[list[str]],
        num_cols: int,
        available_width: int,
    ) -> list[int]:
        """Calculate column widths for a table."""
        # Minimum width per column
        min_col_width = 3

        # Calculate natural widths (max content width per column)
        natural_widths: list[int] = []
        for col in range(num_cols):
            max_w = visible_width(header_cells[col]) if col < len(header_cells) else 0
            for row in body_rows:
                if col < len(row):
                    w = visible_width(row[col])
                    if w > max_w:
                        max_w = w
            natural_widths.append(max(max_w, min_col_width))

        # Account for borders: |_content_| = 3 chars per column + 1 for the final border
        # Total border overhead: (num_cols + 1) border chars + num_cols * 2 padding chars
        border_overhead = (num_cols + 1) + (num_cols * 2)
        content_budget = max(num_cols * min_col_width, available_width - border_overhead)

        total_natural = sum(natural_widths)
        if total_natural <= content_budget:
            return natural_widths

        # Proportionally shrink columns
        col_widths: list[int] = []
        for nw in natural_widths:
            w = max(min_col_width, int(nw * content_budget / total_natural))
            col_widths.append(w)

        # Distribute any remaining budget
        remaining = content_budget - sum(col_widths)
        for ci in range(min(remaining, num_cols)):
            col_widths[ci] += 1

        return col_widths

    def _table_top_border(self, col_widths: list[int], color: str) -> str:
        parts = [f"{color}\u250c"]
        for i, w in enumerate(col_widths):
            parts.append("\u2500" * (w + 2))
            if i < len(col_widths) - 1:
                parts.append("\u252c")
        parts.append(f"\u2510{_RESET}")
        return "".join(parts)

    def _table_mid_border(self, col_widths: list[int], color: str) -> str:
        parts = [f"{color}\u251c"]
        for i, w in enumerate(col_widths):
            parts.append("\u2500" * (w + 2))
            if i < len(col_widths) - 1:
                parts.append("\u253c")
        parts.append(f"\u2524{_RESET}")
        return "".join(parts)

    def _table_bottom_border(self, col_widths: list[int], color: str) -> str:
        parts = [f"{color}\u2514"]
        for i, w in enumerate(col_widths):
            parts.append("\u2500" * (w + 2))
            if i < len(col_widths) - 1:
                parts.append("\u2534")
        parts.append(f"\u2518{_RESET}")
        return "".join(parts)

    def _render_table_row(
        self,
        cells: list[str],
        col_widths: list[int],
        border_color: str,
        cell_style: str = "",
    ) -> list[str]:
        """Render a single table row, possibly spanning multiple display lines.

        Each cell is wrapped to its column width; the row may be multiple lines
        tall if any cell wraps.
        """
        num_cols = len(col_widths)

        # Wrap each cell to its column width
        wrapped_cells: list[list[str]] = []
        for col in range(num_cols):
            cell_text = cells[col] if col < len(cells) else ""
            cell_w = col_widths[col]
            cell_lines = wrap_text_with_ansi(_strip_ansi(cell_text) if not cell_text else cell_text, cell_w)
            if not cell_lines:
                cell_lines = [""]
            wrapped_cells.append(cell_lines)

        # Max number of display lines for this row
        max_lines = max(len(cl) for cl in wrapped_cells) if wrapped_cells else 1

        row_lines: list[str] = []
        for line_idx in range(max_lines):
            parts: list[str] = [f"{border_color}\u2502{_RESET}"]
            for col in range(num_cols):
                cell_w = col_widths[col]
                cell_lines = wrapped_cells[col] if col < len(wrapped_cells) else [""]
                cell_line = cell_lines[line_idx] if line_idx < len(cell_lines) else ""

                # Pad cell content to column width
                content_width = visible_width(cell_line)
                padding = max(0, cell_w - content_width)

                if cell_style:
                    parts.append(f" {cell_style}{cell_line}{_RESET}{' ' * padding} ")
                else:
                    prefix = self._default_style_prefix()
                    suffix = self._default_style_suffix()
                    parts.append(f" {prefix}{cell_line}{suffix}{' ' * padding} ")

                parts.append(f"{border_color}\u2502{_RESET}")

            row_lines.append("".join(parts))

        return row_lines

    # -- padding and background ---------------------------------------------

    def _apply_padding_and_bg(
        self, raw_lines: list[str], width: int, content_width: int
    ) -> list[str]:
        """Add horizontal padding, vertical padding, and optional background."""
        left_pad = " " * self._padding_x
        result: list[str] = []

        # Top padding
        for _ in range(self._padding_y):
            line = " " * width
            if self._custom_bg_fn:
                line = apply_background_to_line(line, width, self._custom_bg_fn)
            result.append(line)

        # Content lines
        for raw_line in raw_lines:
            # Skip image lines (pass through untouched)
            if is_image_line(raw_line):
                result.append(raw_line)
                continue

            padded = left_pad + raw_line
            line_width = visible_width(padded)
            right_padding = max(0, width - line_width)
            padded = padded + " " * right_padding

            if self._custom_bg_fn:
                padded = apply_background_to_line(padded, width, self._custom_bg_fn)

            result.append(padded)

        # Bottom padding
        for _ in range(self._padding_y):
            line = " " * width
            if self._custom_bg_fn:
                line = apply_background_to_line(line, width, self._custom_bg_fn)
            result.append(line)

        return result

    # -- token navigation helpers -------------------------------------------

    @staticmethod
    def _skip_to_close(tokens: list[Token], start: int, close_type: str) -> int:
        """Advance index past the next token of *close_type*."""
        i = start + 1
        while i < len(tokens):
            if tokens[i].type == close_type:
                return i
            i += 1
        return len(tokens) - 1

    @staticmethod
    def _find_matching_close(
        tokens: list[Token], start: int, open_type: str, close_type: str
    ) -> int:
        """Find the matching close token for a given open token, respecting nesting."""
        depth = 0
        i = start
        while i < len(tokens):
            if tokens[i].type == open_type:
                depth += 1
            elif tokens[i].type == close_type:
                depth -= 1
                if depth == 0:
                    return i
            i += 1
        return len(tokens) - 1
