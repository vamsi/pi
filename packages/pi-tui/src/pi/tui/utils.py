"""Terminal text utilities: ANSI handling, width measurement, word wrapping.

Provides functions for measuring visible terminal widths, tracking ANSI SGR
state, word-wrapping text with ANSI codes preserved, and column-based slicing.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Callable

import grapheme
import wcwidth as _wcwidth


# ---------------------------------------------------------------------------
# Grapheme segmenter wrapper (mirrors Intl.Segmenter API)
# ---------------------------------------------------------------------------


class _GraphemeSegmenter:
    """Thin wrapper around ``grapheme.graphemes`` matching the Intl.Segmenter API."""

    @staticmethod
    def segment(text: str) -> list[str]:
        return list(grapheme.graphemes(text))


def get_segmenter() -> _GraphemeSegmenter:
    """Return a grapheme segmenter instance."""
    return _GraphemeSegmenter()


# ---------------------------------------------------------------------------
# Regex patterns for ANSI / OSC / APC sequences
# ---------------------------------------------------------------------------

# CSI sequences: ESC[ <params> <final byte>
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mGKHJ]")
# OSC 8 hyperlinks: ESC]8;;<uri> BEL
_OSC8_RE = re.compile(r"\x1b\]8;;[^\x07]*\x07")
# APC sequences: ESC_ <payload> (BEL | ST)
_APC_RE = re.compile(r"\x1b_[^\x07\x1b]*(?:\x07|\x1b\\)")

_STRIP_RE = re.compile(
    r"\x1b\[[0-9;]*[mGKHJ]"        # CSI
    r"|\x1b\]8;;[^\x07]*\x07"       # OSC 8
    r"|\x1b_[^\x07\x1b]*(?:\x07|\x1b\\)"  # APC
)

# Punctuation characters for word-break classification
_PUNCTUATION_REGEX = re.compile(r"[(){}\[\]<>.,;:'\"!?\+\-=*/\\|&%\^$#@~`]")

# ---------------------------------------------------------------------------
# Width cache (capped at 512 entries)
# ---------------------------------------------------------------------------

_width_cache: dict[str, int] = {}
_WIDTH_CACHE_MAX = 512


def _cache_width(key: str, value: int) -> int:
    if len(_width_cache) >= _WIDTH_CACHE_MAX:
        _width_cache.clear()
    _width_cache[key] = value
    return value


# ---------------------------------------------------------------------------
# Grapheme width
# ---------------------------------------------------------------------------

def _grapheme_width(g: str) -> int:
    """Return the terminal display width of a single grapheme cluster.

    Rules:
    1. Zero-width characters (control, combining marks, etc.) -> 0
    2. Emoji (multi-codepoint, contains VS16 U+FE0F, ZWJ sequences, etc.) -> 2
    3. Otherwise delegate to wcwidth for the first meaningful codepoint.
    """
    if not g:
        return 0

    # Single codepoint fast path
    if len(g) == 1:
        cp = ord(g)
        # Control characters
        if cp < 0x20 or (0x7F <= cp <= 0x9F):
            return 0
        w = _wcwidth.wcwidth(g)
        return max(w, 0)

    # Multi-codepoint grapheme cluster
    codepoints = list(g)

    # Check for emoji indicators:
    # - Contains VS16 (U+FE0F) -> emoji presentation
    # - Contains ZWJ (U+200D) -> ZWJ sequence
    # - Contains regional indicators (U+1F1E6..U+1F1FF)
    # - Contains emoji modifiers (U+1F3FB..U+1F3FF)
    for ch in codepoints:
        cp = ord(ch)
        if cp == 0xFE0F:  # VS16
            return 2
        if cp == 0x200D:  # ZWJ
            return 2
        if 0x1F3FB <= cp <= 0x1F3FF:  # Skin tone modifiers
            return 2
        if 0x1F1E6 <= cp <= 0x1F1FF:  # Regional indicators
            return 2

    # Check Unicode categories for emoji detection
    first_cp = ord(codepoints[0])

    # Common emoji ranges
    if first_cp >= 0x1F000:
        return 2

    # Miscellaneous symbols, dingbats, etc.
    if 0x2600 <= first_cp <= 0x27BF:
        return 2

    # Enclosed alphanumeric supplement
    if 0x1F100 <= first_cp <= 0x1F1FF:
        return 2

    # Multi-codepoint but not clearly emoji: check category
    cat = unicodedata.category(codepoints[0])
    if cat.startswith("M"):  # Mark
        return 0
    if cat == "Cf":  # Format
        return 0

    # Fall back to wcwidth on the first codepoint
    w = _wcwidth.wcwidth(codepoints[0])
    return max(w, 0)


# ---------------------------------------------------------------------------
# visible_width
# ---------------------------------------------------------------------------

def visible_width(text: str) -> int:
    """Calculate the visible terminal width of *text*.

    * Strips ANSI escape sequences.
    * Treats tabs as 3 spaces.
    * Uses a fast ASCII path when possible.
    * Caches results for non-ASCII strings.
    """
    if not text:
        return 0

    # Strip ANSI codes first
    stripped = _STRIP_RE.sub("", text)
    if not stripped:
        return 0

    # Replace tabs with 3 spaces for width calculation
    stripped = stripped.replace("\t", "   ")

    # Fast ASCII path: all codepoints in 0x20..0x7E
    is_ascii = True
    for ch in stripped:
        cp = ord(ch)
        if cp < 0x20 or cp > 0x7E:
            is_ascii = False
            break

    if is_ascii:
        return len(stripped)

    # Check cache
    cached = _width_cache.get(stripped)
    if cached is not None:
        return cached

    # Full grapheme-cluster measurement
    total = 0
    for g in grapheme.graphemes(stripped):
        total += _grapheme_width(g)

    return _cache_width(stripped, total)


# ---------------------------------------------------------------------------
# extract_ansi_code
# ---------------------------------------------------------------------------

def extract_ansi_code(text: str, pos: int) -> tuple[str, int] | None:
    """Extract an ANSI escape sequence starting at *pos* in *text*.

    Returns ``(code, length)`` where *code* is the full escape sequence string
    and *length* is the number of characters consumed, or ``None`` if there is
    no escape sequence at *pos*.

    Handles:
    * CSI sequences: ``ESC[`` ... ``m`` / ``G`` / ``K`` / ``H`` / ``J``
    * OSC sequences: ``ESC]`` ... ``BEL`` / ``ST``
    * APC sequences: ``ESC_`` ... ``BEL`` / ``ST``
    """
    if pos >= len(text) or text[pos] != "\x1b":
        return None

    if pos + 1 >= len(text):
        return None

    next_ch = text[pos + 1]

    # CSI: ESC[ <params> <final>
    if next_ch == "[":
        i = pos + 2
        while i < len(text):
            ch = text[i]
            if ch in "mGKHJ":
                code = text[pos : i + 1]
                return (code, len(code))
            if ch.isdigit() or ch == ";":
                i += 1
                continue
            # Unexpected character -- not a recognized CSI sequence
            break
        return None

    # OSC: ESC] ... (BEL | ESC\)
    if next_ch == "]":
        i = pos + 2
        while i < len(text):
            ch = text[i]
            if ch == "\x07":  # BEL
                code = text[pos : i + 1]
                return (code, len(code))
            if ch == "\x1b" and i + 1 < len(text) and text[i + 1] == "\\":
                code = text[pos : i + 2]
                return (code, len(code))
            i += 1
        return None

    # APC: ESC_ ... (BEL | ESC\)
    if next_ch == "_":
        i = pos + 2
        while i < len(text):
            ch = text[i]
            if ch == "\x07":  # BEL
                code = text[pos : i + 1]
                return (code, len(code))
            if ch == "\x1b" and i + 1 < len(text) and text[i + 1] == "\\":
                code = text[pos : i + 2]
                return (code, len(code))
            i += 1
        return None

    return None


# ---------------------------------------------------------------------------
# AnsiCodeTracker
# ---------------------------------------------------------------------------

class AnsiCodeTracker:
    """Track active ANSI SGR (Select Graphic Rendition) state.

    Processes CSI SGR sequences (``ESC[...m``) and maintains which attributes
    are currently active so that they can be re-applied after line breaks.
    """

    def __init__(self) -> None:
        self.bold: str | None = None
        self.dim: str | None = None
        self.italic: str | None = None
        self.underline: str | None = None
        self.blink: str | None = None
        self.inverse: str | None = None
        self.hidden: str | None = None
        self.strikethrough: str | None = None
        self.fg_color: str | None = None
        self.bg_color: str | None = None

    def process(self, code: str) -> None:
        """Update tracked state from an SGR sequence like ``\\x1b[1;31m``."""
        if not code.startswith("\x1b[") or not code.endswith("m"):
            return

        params_str = code[2:-1]  # strip ESC[ and m
        if not params_str:
            # ESC[m is equivalent to reset
            self.clear()
            return

        params = params_str.split(";")
        i = 0
        while i < len(params):
            p = params[i]

            # Handle empty params as 0 (reset)
            val = int(p) if p else 0

            if val == 0:
                self.clear()
            elif val == 1:
                self.bold = "\x1b[1m"
            elif val == 2:
                self.dim = "\x1b[2m"
            elif val == 3:
                self.italic = "\x1b[3m"
            elif val == 4:
                self.underline = "\x1b[4m"
            elif val == 5:
                self.blink = "\x1b[5m"
            elif val == 7:
                self.inverse = "\x1b[7m"
            elif val == 8:
                self.hidden = "\x1b[8m"
            elif val == 9:
                self.strikethrough = "\x1b[9m"
            elif val == 22:
                self.bold = None
                self.dim = None
            elif val == 23:
                self.italic = None
            elif val == 24:
                self.underline = None
            elif val == 25:
                self.blink = None
            elif val == 27:
                self.inverse = None
            elif val == 28:
                self.hidden = None
            elif val == 29:
                self.strikethrough = None
            # Foreground colors
            elif 30 <= val <= 37:
                self.fg_color = f"\x1b[{val}m"
            elif val == 38:
                # 256-color or RGB
                if i + 1 < len(params):
                    mode = int(params[i + 1]) if params[i + 1] else 0
                    if mode == 5 and i + 2 < len(params):
                        # 256-color: 38;5;N
                        n = params[i + 2]
                        self.fg_color = f"\x1b[38;5;{n}m"
                        i += 2
                    elif mode == 2 and i + 4 < len(params):
                        # RGB: 38;2;R;G;B
                        r = params[i + 2]
                        g = params[i + 3]
                        b = params[i + 4]
                        self.fg_color = f"\x1b[38;2;{r};{g};{b}m"
                        i += 4
                    else:
                        i += 1
            elif val == 39:
                self.fg_color = None
            # Background colors
            elif 40 <= val <= 47:
                self.bg_color = f"\x1b[{val}m"
            elif val == 48:
                # 256-color or RGB
                if i + 1 < len(params):
                    mode = int(params[i + 1]) if params[i + 1] else 0
                    if mode == 5 and i + 2 < len(params):
                        n = params[i + 2]
                        self.bg_color = f"\x1b[48;5;{n}m"
                        i += 2
                    elif mode == 2 and i + 4 < len(params):
                        r = params[i + 2]
                        g = params[i + 3]
                        b = params[i + 4]
                        self.bg_color = f"\x1b[48;2;{r};{g};{b}m"
                        i += 4
                    else:
                        i += 1
            elif val == 49:
                self.bg_color = None
            # Bright foreground colors
            elif 90 <= val <= 97:
                self.fg_color = f"\x1b[{val}m"
            # Bright background colors
            elif 100 <= val <= 107:
                self.bg_color = f"\x1b[{val}m"

            i += 1

    def clear(self) -> None:
        """Reset all tracked attributes to off."""
        self.bold = None
        self.dim = None
        self.italic = None
        self.underline = None
        self.blink = None
        self.inverse = None
        self.hidden = None
        self.strikethrough = None
        self.fg_color = None
        self.bg_color = None

    def get_active_codes(self) -> str:
        """Return a string of ANSI codes that reactivate the current state."""
        parts: list[str] = []
        if self.bold is not None:
            parts.append(self.bold)
        if self.dim is not None:
            parts.append(self.dim)
        if self.italic is not None:
            parts.append(self.italic)
        if self.underline is not None:
            parts.append(self.underline)
        if self.blink is not None:
            parts.append(self.blink)
        if self.inverse is not None:
            parts.append(self.inverse)
        if self.hidden is not None:
            parts.append(self.hidden)
        if self.strikethrough is not None:
            parts.append(self.strikethrough)
        if self.fg_color is not None:
            parts.append(self.fg_color)
        if self.bg_color is not None:
            parts.append(self.bg_color)
        return "".join(parts)

    def has_active_codes(self) -> bool:
        """Return ``True`` if any SGR attribute is currently active."""
        return (
            self.bold is not None
            or self.dim is not None
            or self.italic is not None
            or self.underline is not None
            or self.blink is not None
            or self.inverse is not None
            or self.hidden is not None
            or self.strikethrough is not None
            or self.fg_color is not None
            or self.bg_color is not None
        )

    def get_line_end_reset(self) -> str:
        """Return a reset sequence if any attribute is active, else empty."""
        if self.has_active_codes():
            return "\x1b[0m"
        return ""


# ---------------------------------------------------------------------------
# wrap_text_with_ansi
# ---------------------------------------------------------------------------

def wrap_text_with_ansi(text: str, width: int) -> list[str]:
    """Word-wrap *text* to *width* columns, preserving ANSI escape codes.

    Handles embedded newlines by processing each physical line separately.
    ANSI state is tracked across lines so that colours/attributes persist
    correctly after wrapping.

    Returns a list of wrapped lines (without trailing newlines).
    """
    if width <= 0:
        return [text]

    physical_lines = text.split("\n")
    result: list[str] = []
    tracker = AnsiCodeTracker()

    for physical_line in physical_lines:
        wrapped = _wrap_single_line(physical_line, width, tracker)
        result.extend(wrapped)

    return result


def _wrap_single_line(
    line: str,
    width: int,
    tracker: AnsiCodeTracker,
) -> list[str]:
    """Wrap a single line (no embedded newlines) to *width* columns."""
    if not line:
        # Still need to propagate any ANSI codes in an empty line
        return [""]

    result_lines: list[str] = []
    current_line: list[str] = []
    current_width = 0

    # Re-apply active codes at the start of a continuation line
    prefix = tracker.get_active_codes()
    if prefix:
        current_line.append(prefix)

    i = 0
    while i < len(line):
        # Check for ANSI escape sequence
        extracted = extract_ansi_code(line, i)
        if extracted is not None:
            code, length = extracted
            tracker.process(code)
            current_line.append(code)
            i += length
            continue

        ch = line[i]

        # Handle tab -> 3 spaces
        if ch == "\t":
            ch_text = "   "
            ch_width = 3
        else:
            ch_text = ch
            ch_width = _grapheme_width(ch)

        # Check if this character fits on the current line
        if current_width + ch_width > width and current_width > 0:
            # Try to find a word break point
            break_result = _find_word_break(current_line, current_width, width)

            if break_result is not None:
                before, after, before_width = break_result
                # Emit the 'before' portion as a finished line
                if tracker.has_active_codes():
                    before += "\x1b[0m"
                result_lines.append(before)

                # Start new line with carried-over content
                current_line = []
                active = tracker.get_active_codes()
                if active:
                    current_line.append(active)
                if after:
                    current_line.append(after)
                current_width = visible_width(after)
            else:
                # No good break point; break at current position
                finished = "".join(current_line)
                if tracker.has_active_codes():
                    finished += "\x1b[0m"
                result_lines.append(finished)
                current_line = []
                active = tracker.get_active_codes()
                if active:
                    current_line.append(active)
                current_width = 0

        current_line.append(ch_text)
        current_width += ch_width
        i += 1

    # Flush remaining content
    finished = "".join(current_line)
    if tracker.has_active_codes():
        finished += "\x1b[0m"
    result_lines.append(finished)

    return result_lines


def _find_word_break(
    parts: list[str],
    current_width: int,
    max_width: int,
) -> tuple[str, str, int] | None:
    """Find the last word-break position in the accumulated *parts*.

    Returns ``(before, after, before_width)`` or ``None`` if no suitable
    break point is found.
    """
    joined = "".join(parts)
    # Walk backwards through visible characters to find whitespace
    # We need to find a space in the visible text
    stripped = _STRIP_RE.sub("", joined)
    last_space = stripped.rfind(" ")

    if last_space <= 0:
        return None

    # Now reconstruct based on visible character positions
    visible_idx = 0
    split_pos = 0
    i = 0
    while i < len(joined):
        extracted = extract_ansi_code(joined, i)
        if extracted is not None:
            _code, length = extracted
            i += length
            continue

        if visible_idx == last_space:
            split_pos = i
            break
        visible_idx += 1
        i += 1

    if split_pos <= 0:
        return None

    # Split: include the space in 'before', strip leading spaces from 'after'
    before = joined[:split_pos]
    after = joined[split_pos:]

    # Strip ANSI codes from after to get just visible content, then re-add
    # Actually, keep ANSI codes but strip leading whitespace from visible text
    after_stripped = ""
    j = 0
    leading = True
    while j < len(after):
        extracted = extract_ansi_code(after, j)
        if extracted is not None:
            code, length = extracted
            after_stripped += code
            j += length
            continue
        if leading and after[j] == " ":
            j += 1
            continue
        leading = False
        after_stripped += after[j]
        j += 1

    before_w = visible_width(before)
    return (before, after_stripped, before_w)


# ---------------------------------------------------------------------------
# apply_background_to_line
# ---------------------------------------------------------------------------

def apply_background_to_line(
    line: str,
    width: int,
    bg_fn: Callable[[str], str],
) -> str:
    """Apply a background colour function to *line*, padding to *width*.

    *bg_fn* is called with the content (potentially padded with spaces) and
    should return the string wrapped in background ANSI codes.
    """
    line_width = visible_width(line)
    if line_width >= width:
        return bg_fn(line)

    padding = " " * (width - line_width)
    return bg_fn(line + padding)


# ---------------------------------------------------------------------------
# truncate_to_width
# ---------------------------------------------------------------------------

def truncate_to_width(
    text: str,
    max_width: int,
    ellipsis: str = "...",
    pad: bool = False,
) -> str:
    """Truncate *text* to fit within *max_width* visible columns.

    If the text is wider than *max_width*, it is truncated and *ellipsis* is
    appended (the ellipsis counts towards the width).  If *pad* is ``True``,
    the result is right-padded with spaces to exactly *max_width*.
    """
    if max_width <= 0:
        return ""

    text_width = visible_width(text)
    if text_width <= max_width:
        if pad:
            return text + " " * (max_width - text_width)
        return text

    ellipsis_width = visible_width(ellipsis)
    target_width = max_width - ellipsis_width
    if target_width <= 0:
        # Ellipsis alone exceeds max_width -- just truncate ellipsis
        return _take_columns(ellipsis, max_width)

    result = _take_columns(text, target_width)
    result += ellipsis

    if pad:
        result_width = visible_width(result)
        if result_width < max_width:
            result += " " * (max_width - result_width)

    return result


def _take_columns(text: str, max_cols: int) -> str:
    """Return a prefix of *text* that fits within *max_cols* visible columns.

    ANSI codes are preserved; the text is cut at grapheme boundaries.
    """
    result: list[str] = []
    cols = 0
    i = 0

    while i < len(text):
        extracted = extract_ansi_code(text, i)
        if extracted is not None:
            code, length = extracted
            result.append(code)
            i += length
            continue

        ch = text[i]
        w = _grapheme_width(ch)
        if cols + w > max_cols:
            break
        result.append(ch)
        cols += w
        i += 1

    return "".join(result)


# ---------------------------------------------------------------------------
# slice_by_column / slice_with_width
# ---------------------------------------------------------------------------

def slice_by_column(
    line: str,
    start_col: int,
    length: int,
    strict: bool = False,
) -> str:
    """Extract *length* visible columns starting at *start_col* from *line*.

    If *strict* is ``True``, wide characters that partially overlap the
    boundaries are replaced with spaces rather than being included.
    """
    result, _width = _slice_impl(line, start_col, length, strict)
    return result


def slice_with_width(
    line: str,
    start_col: int,
    length: int,
    strict: bool = False,
) -> tuple[str, int]:
    """Like :func:`slice_by_column` but also returns the actual visible width."""
    return _slice_impl(line, start_col, length, strict)


def _slice_impl(
    line: str,
    start_col: int,
    length: int,
    strict: bool,
) -> tuple[str, int]:
    """Core implementation for column-based slicing."""
    if length <= 0:
        return ("", 0)

    end_col = start_col + length
    result: list[str] = []
    col = 0
    result_width = 0
    i = 0

    while i < len(line) and col < end_col:
        # Collect ANSI codes
        extracted = extract_ansi_code(line, i)
        if extracted is not None:
            code, code_len = extracted
            # Include ANSI codes if we are within or past start_col
            if col >= start_col:
                result.append(code)
            i += code_len
            continue

        ch = line[i]
        w = _grapheme_width(ch)

        # Character spans [col, col + w)
        char_end = col + w

        if char_end <= start_col:
            # Entirely before the slice region
            col = char_end
            i += 1
            continue

        if col >= end_col:
            # Past the slice region
            break

        # Character overlaps with the slice region
        if col < start_col:
            # Character starts before start_col (partial overlap at start)
            if strict and w > 1:
                # Replace with spaces for the portion inside the region
                overlap = char_end - start_col
                spaces = " " * overlap
                result.append(spaces)
                result_width += overlap
            else:
                result.append(ch)
                result_width += w
        elif char_end > end_col:
            # Character extends past end_col (partial overlap at end)
            if strict and w > 1:
                overlap = end_col - col
                spaces = " " * overlap
                result.append(spaces)
                result_width += overlap
            else:
                result.append(ch)
                result_width += w
        else:
            # Fully within the region
            result.append(ch)
            result_width += w

        col = char_end
        i += 1

    # Collect trailing ANSI codes
    while i < len(line):
        extracted = extract_ansi_code(line, i)
        if extracted is not None:
            code, code_len = extracted
            result.append(code)
            i += code_len
        else:
            break

    return ("".join(result), result_width)


# ---------------------------------------------------------------------------
# extract_segments
# ---------------------------------------------------------------------------

def extract_segments(
    line: str,
    before_end: int,
    after_start: int,
    after_len: int,
    strict_after: bool = False,
) -> tuple[str, str]:
    """Extract "before" and "after" segments from *line* in a single pass.

    Used for overlay compositing where a region of the line is replaced:

    * ``before``: columns ``[0, before_end)``
    * ``after``:  columns ``[after_start, after_start + after_len)``

    *strict_after* controls whether wide characters that partially overlap
    the *after* boundaries are replaced with spaces.
    """
    before_parts: list[str] = []
    after_parts: list[str] = []
    col = 0
    after_end = after_start + after_len
    i = 0

    while i < len(line):
        extracted = extract_ansi_code(line, i)
        if extracted is not None:
            code, code_len = extracted
            if col < before_end:
                before_parts.append(code)
            if col >= after_start and col < after_end:
                after_parts.append(code)
            i += code_len
            continue

        ch = line[i]
        w = _grapheme_width(ch)
        char_end = col + w

        # Before segment: columns [0, before_end)
        if col < before_end:
            if char_end <= before_end:
                before_parts.append(ch)
            else:
                # Partial overlap at end of before region
                before_parts.append(ch)

        # After segment: columns [after_start, after_end)
        if char_end > after_start and col < after_end:
            if col < after_start:
                # Starts before after_start (partial overlap at start)
                if strict_after and w > 1:
                    overlap = min(char_end, after_end) - after_start
                    after_parts.append(" " * overlap)
                else:
                    after_parts.append(ch)
            elif char_end > after_end:
                # Extends past after_end (partial overlap at end)
                if strict_after and w > 1:
                    overlap = after_end - col
                    after_parts.append(" " * overlap)
                else:
                    after_parts.append(ch)
            else:
                after_parts.append(ch)

        col = char_end
        i += 1

    return ("".join(before_parts), "".join(after_parts))


# ---------------------------------------------------------------------------
# Character classification
# ---------------------------------------------------------------------------

def is_whitespace_char(char: str) -> bool:
    """Return ``True`` if *char* is a whitespace character."""
    return char in (" ", "\t", "\n", "\r", "\f", "\v")


def is_punctuation_char(char: str) -> bool:
    """Return ``True`` if *char* is a punctuation character."""
    return bool(_PUNCTUATION_REGEX.match(char))
