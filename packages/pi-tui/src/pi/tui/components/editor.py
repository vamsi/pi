"""Multi-line editor with autocomplete, undo, word wrap, history, and kill ring.

Faithful Python port of the TypeScript Editor component from
``pi-mono/packages/tui/src/components/editor.ts``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Literal, Protocol

import grapheme as _grapheme

from pi.tui.keybindings import get_editor_keybindings
from pi.tui.keys import matches_key
from pi.tui.kill_ring import KillRing
from pi.tui.undo_stack import UndoStack
from pi.tui.utils import is_punctuation_char, is_whitespace_char, visible_width

if TYPE_CHECKING:
    from pi.tui.components.select_list import SelectList, SelectListTheme


# ---------------------------------------------------------------------------
# Protocols for TUI / Autocomplete (avoids circular imports)
# ---------------------------------------------------------------------------


class _Terminal(Protocol):
    @property
    def rows(self) -> int: ...

    @property
    def columns(self) -> int: ...


class TUI(Protocol):
    @property
    def terminal(self) -> _Terminal: ...

    def request_render(self) -> None: ...


class AutocompleteItem(Protocol):
    @property
    def value(self) -> str: ...

    @property
    def label(self) -> str: ...

    @property
    def description(self) -> str | None: ...


class AutocompleteProvider(Protocol):
    def get_suggestions(
        self,
        lines: list[str],
        cursor_line: int,
        cursor_col: int,
    ) -> dict[str, object] | None: ...

    def apply_completion(
        self,
        lines: list[str],
        cursor_line: int,
        cursor_col: int,
        item: object,
        prefix: str,
    ) -> dict[str, object]: ...


class CombinedAutocompleteProvider(AutocompleteProvider, Protocol):
    def should_trigger_file_completion(
        self, lines: list[str], cursor_line: int, cursor_col: int
    ) -> bool: ...

    def get_force_file_suggestions(
        self,
        lines: list[str],
        cursor_line: int,
        cursor_col: int,
    ) -> dict[str, object] | None: ...


# ---------------------------------------------------------------------------
# Component / Focusable interfaces (duck-typed)
# ---------------------------------------------------------------------------


class Component(Protocol):
    def invalidate(self) -> None: ...

    def render(self, width: int) -> list[str]: ...

    def handle_input(self, data: str) -> None: ...


class Focusable(Protocol):
    focused: bool


# ---------------------------------------------------------------------------
# CURSOR_MARKER
# ---------------------------------------------------------------------------

CURSOR_MARKER = "\x1b_pi:c\x07"

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TextChunk:
    """Represents a chunk of text for word-wrap layout."""

    text: str
    start_index: int
    end_index: int


@dataclass
class LayoutLine:
    """A single visual line produced by layout_text."""

    text: str
    has_cursor: bool
    cursor_pos: int | None = None


@dataclass
class EditorState:
    """Internal mutable state of the editor."""

    lines: list[str] = field(default_factory=lambda: [""])
    cursor_line: int = 0
    cursor_col: int = 0


class EditorTheme(Protocol):
    @property
    def border_color(self) -> Callable[[str], str]: ...

    @property
    def select_list(self) -> SelectListTheme: ...


@dataclass
class EditorOptions:
    padding_x: int = 0
    autocomplete_max_visible: int = 5


# ---------------------------------------------------------------------------
# word_wrap_line
# ---------------------------------------------------------------------------


def word_wrap_line(line: str, max_width: int) -> list[TextChunk]:
    """Split a line into word-wrapped chunks.

    Wraps at word boundaries when possible, falling back to character-level
    wrapping for words longer than the available width.
    """
    if not line or max_width <= 0:
        return [TextChunk(text="", start_index=0, end_index=0)]

    line_width = visible_width(line)
    if line_width <= max_width:
        return [TextChunk(text=line, start_index=0, end_index=len(line))]

    chunks: list[TextChunk] = []

    # Build segments list: (grapheme_string, index_in_line)
    segments: list[tuple[str, int]] = []
    idx = 0
    for g in _grapheme.graphemes(line):
        segments.append((g, idx))
        idx += len(g)

    current_width = 0
    chunk_start = 0

    # Wrap opportunity: position after the last whitespace before a non-ws grapheme
    wrap_opp_index = -1
    wrap_opp_width = 0

    for i, (grapheme_str, char_index) in enumerate(segments):
        g_width = visible_width(grapheme_str)
        is_ws = is_whitespace_char(grapheme_str)

        # Overflow check before advancing.
        if current_width + g_width > max_width:
            if wrap_opp_index >= 0:
                # Backtrack to last wrap opportunity.
                chunks.append(
                    TextChunk(
                        text=line[chunk_start:wrap_opp_index],
                        start_index=chunk_start,
                        end_index=wrap_opp_index,
                    )
                )
                chunk_start = wrap_opp_index
                current_width -= wrap_opp_width
            elif chunk_start < char_index:
                # No wrap opportunity: force-break at current position.
                chunks.append(
                    TextChunk(
                        text=line[chunk_start:char_index],
                        start_index=chunk_start,
                        end_index=char_index,
                    )
                )
                chunk_start = char_index
                current_width = 0
            wrap_opp_index = -1

        # Advance.
        current_width += g_width

        # Record wrap opportunity: whitespace followed by non-whitespace.
        if is_ws and i + 1 < len(segments):
            next_g, next_idx = segments[i + 1]
            if not is_whitespace_char(next_g):
                wrap_opp_index = next_idx
                wrap_opp_width = current_width

    # Push final chunk.
    chunks.append(
        TextChunk(
            text=line[chunk_start:],
            start_index=chunk_start,
            end_index=len(line),
        )
    )

    return chunks


# ---------------------------------------------------------------------------
# decode_kitty_printable
# ---------------------------------------------------------------------------

_KITTY_CSI_U_REGEX = re.compile(
    r"^\x1b\[(\d+)(?::(\d*))?(?::(\d+))?(?:;(\d+))?(?::(\d+))?u$"
)
_KITTY_MOD_SHIFT = 1
_KITTY_MOD_ALT = 2
_KITTY_MOD_CTRL = 4


def decode_kitty_printable(data: str) -> str | None:
    """Decode a printable CSI-u sequence, preferring the shifted key when present."""
    m = _KITTY_CSI_U_REGEX.match(data)
    if not m:
        return None

    try:
        codepoint = int(m.group(1))
    except (ValueError, TypeError):
        return None
    if not math.isfinite(codepoint):
        return None

    shifted_key: int | None = None
    if m.group(2) is not None and len(m.group(2)) > 0:
        try:
            shifted_key = int(m.group(2))
        except (ValueError, TypeError):
            shifted_key = None

    try:
        mod_value = int(m.group(4)) if m.group(4) else 1
    except (ValueError, TypeError):
        mod_value = 1
    modifier = mod_value - 1 if math.isfinite(mod_value) else 0

    # Ignore CSI-u sequences used for Alt/Ctrl shortcuts.
    if modifier & (_KITTY_MOD_ALT | _KITTY_MOD_CTRL):
        return None

    # Prefer the shifted keycode when Shift is held.
    effective_codepoint = codepoint
    if (modifier & _KITTY_MOD_SHIFT) and isinstance(shifted_key, int):
        effective_codepoint = shifted_key

    # Drop control characters or invalid codepoints.
    if not math.isfinite(effective_codepoint) or effective_codepoint < 32:
        return None

    try:
        return chr(effective_codepoint)
    except (ValueError, OverflowError):
        return None


# ---------------------------------------------------------------------------
# Editor class
# ---------------------------------------------------------------------------


class Editor:
    """Multi-line terminal editor with autocomplete, undo, history and kill ring.

    Implements the Component and Focusable interfaces for TUI integration.
    """

    def __init__(
        self,
        tui: TUI,
        theme: EditorTheme,
        options: EditorOptions | None = None,
    ) -> None:
        if options is None:
            options = EditorOptions()

        self._state = EditorState()
        self.focused: bool = False
        self._tui = tui
        self._theme = theme
        self.border_color: Callable[[str], str] = theme.border_color

        padding_x = options.padding_x
        if not math.isfinite(padding_x):
            padding_x = 0
        self._padding_x: int = max(0, int(padding_x))

        max_vis = options.autocomplete_max_visible
        if not math.isfinite(max_vis):
            max_vis = 5
        self._autocomplete_max_visible: int = max(3, min(20, int(max_vis)))

        self._last_width: int = 80
        self._scroll_offset: int = 0

        # Autocomplete support
        self._autocomplete_provider: AutocompleteProvider | None = None
        self._autocomplete_list: SelectList | None = None
        self._autocomplete_state: Literal["regular", "force"] | None = None
        self._autocomplete_prefix: str = ""

        # Paste tracking for large pastes
        self._pastes: dict[int, str] = {}
        self._paste_counter: int = 0

        # Bracketed paste mode buffering
        self._paste_buffer: str = ""
        self._is_in_paste: bool = False

        # Prompt history
        self._history: list[str] = []
        self._history_index: int = -1

        # Kill ring
        self._kill_ring = KillRing()
        self._last_action: Literal["kill", "yank", "type-word"] | None = None

        # Character jump mode
        self._jump_mode: Literal["forward", "backward"] | None = None

        # Preferred visual column for vertical cursor movement (sticky column)
        self._preferred_visual_col: int | None = None

        # Undo support
        self._undo_stack: UndoStack[EditorState] = UndoStack()

        # Public callbacks
        self.on_submit: Callable[[str], None] | None = None
        self.on_change: Callable[[str], None] | None = None
        self.disable_submit: bool = False

    # -- Padding accessors ---------------------------------------------------

    def get_padding_x(self) -> int:
        return self._padding_x

    def set_padding_x(self, padding: int) -> None:
        if not math.isfinite(padding):
            padding = 0
        new_padding = max(0, int(padding))
        if self._padding_x != new_padding:
            self._padding_x = new_padding
            self._tui.request_render()

    # -- Autocomplete max visible accessors ----------------------------------

    def get_autocomplete_max_visible(self) -> int:
        return self._autocomplete_max_visible

    def set_autocomplete_max_visible(self, max_visible: int) -> None:
        if not math.isfinite(max_visible):
            max_visible = 5
        new_max_visible = max(3, min(20, int(max_visible)))
        if self._autocomplete_max_visible != new_max_visible:
            self._autocomplete_max_visible = new_max_visible
            self._tui.request_render()

    def set_autocomplete_provider(self, provider: AutocompleteProvider) -> None:
        self._autocomplete_provider = provider

    # -- History -------------------------------------------------------------

    def add_to_history(self, text: str) -> None:
        """Add a prompt to history for up/down arrow navigation."""
        trimmed = text.strip()
        if not trimmed:
            return
        # Don't add consecutive duplicates
        if self._history and self._history[0] == trimmed:
            return
        self._history.insert(0, trimmed)
        # Limit history size
        if len(self._history) > 100:
            self._history.pop()

    # -- Internal helpers ----------------------------------------------------

    def _is_editor_empty(self) -> bool:
        return len(self._state.lines) == 1 and self._state.lines[0] == ""

    def _is_on_first_visual_line(self) -> bool:
        visual_lines = self._build_visual_line_map(self._last_width)
        current_visual_line = self._find_current_visual_line(visual_lines)
        return current_visual_line == 0

    def _is_on_last_visual_line(self) -> bool:
        visual_lines = self._build_visual_line_map(self._last_width)
        current_visual_line = self._find_current_visual_line(visual_lines)
        return current_visual_line == len(visual_lines) - 1

    def _navigate_history(self, direction: int) -> None:
        """direction: 1 (down) or -1 (up)."""
        self._last_action = None
        if not self._history:
            return

        new_index = self._history_index - direction  # Up(-1) increases index, Down(1) decreases
        if new_index < -1 or new_index >= len(self._history):
            return

        # Capture state when first entering history browsing mode
        if self._history_index == -1 and new_index >= 0:
            self._push_undo_snapshot()

        self._history_index = new_index

        if self._history_index == -1:
            # Returned to "current" state - clear editor
            self._set_text_internal("")
        else:
            self._set_text_internal(self._history[self._history_index] if self._history_index < len(self._history) else "")

    def _set_text_internal(self, text: str) -> None:
        """Internal setText that doesn't reset history state -- used by _navigate_history."""
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        self._state.lines = lines if lines else [""]
        self._state.cursor_line = len(self._state.lines) - 1
        self._set_cursor_col(len(self._state.lines[self._state.cursor_line]) if self._state.lines[self._state.cursor_line] else 0)
        # Reset scroll - render() will adjust to show cursor
        self._scroll_offset = 0

        if self.on_change:
            self.on_change(self.get_text())

    # -- Component interface -------------------------------------------------

    def invalidate(self) -> None:
        """No cached state to invalidate currently."""

    def render(self, width: int) -> list[str]:
        max_padding = max(0, (width - 1) // 2)
        padding_x = min(self._padding_x, max_padding)
        content_width = max(1, width - padding_x * 2)

        # Layout width: with padding the cursor can overflow into it,
        # without padding we reserve 1 column for the cursor.
        layout_width = max(1, content_width - (0 if padding_x else 1))

        # Store for cursor navigation (must match wrapping width)
        self._last_width = layout_width

        horizontal = self.border_color("\u2500")

        # Layout the text
        layout_lines = self._layout_text(layout_width)

        # Calculate max visible lines: 30% of terminal height, minimum 5 lines
        terminal_rows = self._tui.terminal.rows
        max_visible_lines = max(5, terminal_rows * 3 // 10)

        # Find the cursor line index in layout_lines
        cursor_line_index = 0
        for i, ll in enumerate(layout_lines):
            if ll.has_cursor:
                cursor_line_index = i
                break

        # Adjust scroll offset to keep cursor visible
        if cursor_line_index < self._scroll_offset:
            self._scroll_offset = cursor_line_index
        elif cursor_line_index >= self._scroll_offset + max_visible_lines:
            self._scroll_offset = cursor_line_index - max_visible_lines + 1

        # Clamp scroll offset to valid range
        max_scroll_offset = max(0, len(layout_lines) - max_visible_lines)
        self._scroll_offset = max(0, min(self._scroll_offset, max_scroll_offset))

        # Get visible lines slice
        visible_lines = layout_lines[self._scroll_offset : self._scroll_offset + max_visible_lines]

        result: list[str] = []
        left_padding = " " * padding_x
        right_padding = left_padding

        # Render top border (with scroll indicator if scrolled down)
        if self._scroll_offset > 0:
            indicator = f"\u2500\u2500\u2500 \u2191 {self._scroll_offset} more "
            remaining = width - visible_width(indicator)
            result.append(self.border_color(indicator + "\u2500" * max(0, remaining)))
        else:
            result.append(horizontal * width)

        # Render each visible layout line
        # Emit hardware cursor marker only when focused and not showing autocomplete
        emit_cursor_marker = self.focused and not self._autocomplete_state

        for layout_line in visible_lines:
            display_text = layout_line.text
            line_visible_width = visible_width(layout_line.text)
            cursor_in_padding = False

            # Add cursor if this line has it
            if layout_line.has_cursor and layout_line.cursor_pos is not None:
                before = display_text[: layout_line.cursor_pos]
                after = display_text[layout_line.cursor_pos :]

                # Hardware cursor marker (zero-width, emitted before fake cursor for IME positioning)
                marker = CURSOR_MARKER if emit_cursor_marker else ""

                if after:
                    # Cursor is on a character (grapheme) - replace it with highlighted version
                    after_graphemes = list(_grapheme.graphemes(after))
                    first_grapheme = after_graphemes[0] if after_graphemes else ""
                    rest_after = after[len(first_grapheme) :]
                    cursor = f"\x1b[7m{first_grapheme}\x1b[0m"
                    display_text = before + marker + cursor + rest_after
                    # line_visible_width stays the same - we're replacing, not adding
                else:
                    # Cursor is at the end - add highlighted space
                    cursor = "\x1b[7m \x1b[0m"
                    display_text = before + marker + cursor
                    line_visible_width = line_visible_width + 1
                    # If cursor overflows content width into the padding, flag it
                    if line_visible_width > content_width and padding_x > 0:
                        cursor_in_padding = True

            # Calculate padding based on actual visible width
            padding = " " * max(0, content_width - line_visible_width)
            line_right_padding = right_padding[1:] if cursor_in_padding else right_padding

            # Render the line (no side borders, just horizontal lines above and below)
            result.append(f"{left_padding}{display_text}{padding}{line_right_padding}")

        # Render bottom border (with scroll indicator if more content below)
        lines_below = len(layout_lines) - (self._scroll_offset + len(visible_lines))
        if lines_below > 0:
            indicator = f"\u2500\u2500\u2500 \u2193 {lines_below} more "
            remaining = width - visible_width(indicator)
            result.append(self.border_color(indicator + "\u2500" * max(0, remaining)))
        else:
            result.append(horizontal * width)

        # Add autocomplete list if active
        if self._autocomplete_state and self._autocomplete_list:
            autocomplete_result = self._autocomplete_list.render(content_width)
            for ac_line in autocomplete_result:
                ac_line_width = visible_width(ac_line)
                line_padding = " " * max(0, content_width - ac_line_width)
                result.append(f"{left_padding}{ac_line}{line_padding}{right_padding}")

        return result

    # -- Input handling ------------------------------------------------------

    def handle_input(self, data: str) -> None:  # noqa: C901
        kb = get_editor_keybindings()

        # Handle character jump mode (awaiting next character to jump to)
        if self._jump_mode is not None:
            # Cancel if the hotkey is pressed again
            if kb.matches(data, "jumpForward") or kb.matches(data, "jumpBackward"):
                self._jump_mode = None
                return

            if ord(data[0]) >= 32 if data else False:
                # Printable character - perform the jump
                direction = self._jump_mode
                self._jump_mode = None
                self._jump_to_char(data, direction)
                return

            # Control character - cancel and fall through to normal handling
            self._jump_mode = None

        # Handle bracketed paste mode
        if "\x1b[200~" in data:
            self._is_in_paste = True
            self._paste_buffer = ""
            data = data.replace("\x1b[200~", "")

        if self._is_in_paste:
            self._paste_buffer += data
            end_index = self._paste_buffer.find("\x1b[201~")
            if end_index != -1:
                paste_content = self._paste_buffer[:end_index]
                if paste_content:
                    self._handle_paste(paste_content)
                self._is_in_paste = False
                remaining = self._paste_buffer[end_index + 6 :]
                self._paste_buffer = ""
                if remaining:
                    self.handle_input(remaining)
                return
            return

        # Ctrl+C - let parent handle (exit/clear)
        if kb.matches(data, "copy"):
            return

        # Undo
        if kb.matches(data, "undo"):
            self._undo()
            return

        # Handle autocomplete mode
        if self._autocomplete_state and self._autocomplete_list:
            if kb.matches(data, "selectCancel"):
                self._cancel_autocomplete()
                return

            if kb.matches(data, "selectUp") or kb.matches(data, "selectDown"):
                self._autocomplete_list.handle_input(data)
                return

            if kb.matches(data, "tab"):
                selected = self._autocomplete_list.get_selected_item()
                if selected and self._autocomplete_provider:
                    self._push_undo_snapshot()
                    self._last_action = None
                    result = self._autocomplete_provider.apply_completion(
                        self._state.lines,
                        self._state.cursor_line,
                        self._state.cursor_col,
                        selected,
                        self._autocomplete_prefix,
                    )
                    self._state.lines = result["lines"]  # type: ignore[assignment]
                    self._state.cursor_line = result["cursor_line"]  # type: ignore[assignment]
                    self._set_cursor_col(result["cursor_col"])  # type: ignore[arg-type]
                    self._cancel_autocomplete()
                    if self.on_change:
                        self.on_change(self.get_text())
                return

            if kb.matches(data, "selectConfirm"):
                selected = self._autocomplete_list.get_selected_item()
                if selected and self._autocomplete_provider:
                    self._push_undo_snapshot()
                    self._last_action = None
                    result = self._autocomplete_provider.apply_completion(
                        self._state.lines,
                        self._state.cursor_line,
                        self._state.cursor_col,
                        selected,
                        self._autocomplete_prefix,
                    )
                    self._state.lines = result["lines"]  # type: ignore[assignment]
                    self._state.cursor_line = result["cursor_line"]  # type: ignore[assignment]
                    self._set_cursor_col(result["cursor_col"])  # type: ignore[arg-type]

                    if self._autocomplete_prefix.startswith("/"):
                        self._cancel_autocomplete()
                        # Fall through to submit
                    else:
                        self._cancel_autocomplete()
                        if self.on_change:
                            self.on_change(self.get_text())
                        return

        # Tab - trigger completion
        if kb.matches(data, "tab") and not self._autocomplete_state:
            self._handle_tab_completion()
            return

        # Deletion actions
        if kb.matches(data, "deleteToLineEnd"):
            self._delete_to_end_of_line()
            return
        if kb.matches(data, "deleteToLineStart"):
            self._delete_to_start_of_line()
            return
        if kb.matches(data, "deleteWordBackward"):
            self._delete_word_backwards()
            return
        if kb.matches(data, "deleteWordForward"):
            self._delete_word_forward()
            return
        if kb.matches(data, "deleteCharBackward") or matches_key(data, "shift+backspace"):
            self._handle_backspace()
            return
        if kb.matches(data, "deleteCharForward") or matches_key(data, "shift+delete"):
            self._handle_forward_delete()
            return

        # Kill ring actions
        if kb.matches(data, "yank"):
            self._yank()
            return
        if kb.matches(data, "yankPop"):
            self._yank_pop()
            return

        # Cursor movement actions
        if kb.matches(data, "cursorLineStart"):
            self._move_to_line_start()
            return
        if kb.matches(data, "cursorLineEnd"):
            self._move_to_line_end()
            return
        if kb.matches(data, "cursorWordLeft"):
            self._move_word_backwards()
            return
        if kb.matches(data, "cursorWordRight"):
            self._move_word_forwards()
            return

        # New line
        if (
            kb.matches(data, "newLine")
            or (len(data) > 1 and ord(data[0]) == 10)
            or data == "\x1b\r"
            or data == "\x1b[13;2~"
            or (len(data) > 1 and "\x1b" in data and "\r" in data)
            or (data == "\n" and len(data) == 1)
        ):
            if self._should_submit_on_backslash_enter(data, kb):
                self._handle_backspace()
                self._submit_value()
                return
            self._add_new_line()
            return

        # Submit (Enter)
        if kb.matches(data, "submit"):
            if self.disable_submit:
                return

            # Workaround for terminals without Shift+Enter support:
            # If char before cursor is \, delete it and insert newline instead of submitting.
            current_line = self._state.lines[self._state.cursor_line] if self._state.cursor_line < len(self._state.lines) else ""
            if self._state.cursor_col > 0 and self._state.cursor_col <= len(current_line) and current_line[self._state.cursor_col - 1] == "\\":
                self._handle_backspace()
                self._add_new_line()
                return

            self._submit_value()
            return

        # Arrow key navigation (with history support)
        if kb.matches(data, "cursorUp"):
            if self._is_editor_empty():
                self._navigate_history(-1)
            elif self._history_index > -1 and self._is_on_first_visual_line():
                self._navigate_history(-1)
            elif self._is_on_first_visual_line():
                # Already at top - jump to start of line
                self._move_to_line_start()
            else:
                self._move_cursor(-1, 0)
            return
        if kb.matches(data, "cursorDown"):
            if self._history_index > -1 and self._is_on_last_visual_line():
                self._navigate_history(1)
            elif self._is_on_last_visual_line():
                # Already at bottom - jump to end of line
                self._move_to_line_end()
            else:
                self._move_cursor(1, 0)
            return
        if kb.matches(data, "cursorRight"):
            self._move_cursor(0, 1)
            return
        if kb.matches(data, "cursorLeft"):
            self._move_cursor(0, -1)
            return

        # Page up/down - scroll by page and move cursor
        if kb.matches(data, "pageUp"):
            self._page_scroll(-1)
            return
        if kb.matches(data, "pageDown"):
            self._page_scroll(1)
            return

        # Character jump mode triggers
        if kb.matches(data, "jumpForward"):
            self._jump_mode = "forward"
            return
        if kb.matches(data, "jumpBackward"):
            self._jump_mode = "backward"
            return

        # Shift+Space - insert regular space
        if matches_key(data, "shift+space"):
            self._insert_character(" ")
            return

        kitty_printable = decode_kitty_printable(data)
        if kitty_printable is not None:
            self._insert_character(kitty_printable)
            return

        # Regular characters
        if data and ord(data[0]) >= 32:
            self._insert_character(data)

    # -- Layout --------------------------------------------------------------

    def _layout_text(self, content_width: int) -> list[LayoutLine]:
        layout_lines: list[LayoutLine] = []

        if not self._state.lines or (len(self._state.lines) == 1 and self._state.lines[0] == ""):
            # Empty editor
            layout_lines.append(LayoutLine(text="", has_cursor=True, cursor_pos=0))
            return layout_lines

        # Process each logical line
        for i, line in enumerate(self._state.lines):
            is_current_line = i == self._state.cursor_line
            line_vis_width = visible_width(line)

            if line_vis_width <= content_width:
                # Line fits in one layout line
                if is_current_line:
                    layout_lines.append(
                        LayoutLine(text=line, has_cursor=True, cursor_pos=self._state.cursor_col)
                    )
                else:
                    layout_lines.append(LayoutLine(text=line, has_cursor=False))
            else:
                # Line needs wrapping - use word-aware wrapping
                chunks = word_wrap_line(line, content_width)

                for chunk_index, chunk in enumerate(chunks):
                    cursor_pos = self._state.cursor_col
                    is_last_chunk = chunk_index == len(chunks) - 1

                    # Determine if cursor is in this chunk
                    has_cursor_in_chunk = False
                    adjusted_cursor_pos = 0

                    if is_current_line:
                        if is_last_chunk:
                            # Last chunk: cursor belongs here if >= start_index
                            has_cursor_in_chunk = cursor_pos >= chunk.start_index
                            adjusted_cursor_pos = cursor_pos - chunk.start_index
                        else:
                            # Non-last chunk: cursor belongs here if in range [start_index, end_index)
                            has_cursor_in_chunk = chunk.start_index <= cursor_pos < chunk.end_index
                            if has_cursor_in_chunk:
                                adjusted_cursor_pos = cursor_pos - chunk.start_index
                                # Clamp to text length (in case cursor was in trimmed whitespace)
                                if adjusted_cursor_pos > len(chunk.text):
                                    adjusted_cursor_pos = len(chunk.text)

                    if has_cursor_in_chunk:
                        layout_lines.append(
                            LayoutLine(
                                text=chunk.text,
                                has_cursor=True,
                                cursor_pos=adjusted_cursor_pos,
                            )
                        )
                    else:
                        layout_lines.append(LayoutLine(text=chunk.text, has_cursor=False))

        return layout_lines

    # -- Text accessors ------------------------------------------------------

    def get_text(self) -> str:
        return "\n".join(self._state.lines)

    def get_expanded_text(self) -> str:
        """Get text with paste markers expanded to their actual content."""
        result = "\n".join(self._state.lines)
        for paste_id, paste_content in self._pastes.items():
            marker_regex = re.compile(
                rf"\[paste #{paste_id}( (\+\d+ lines|\d+ chars))?\]"
            )
            result = marker_regex.sub(paste_content, result)
        return result

    def get_lines(self) -> list[str]:
        return list(self._state.lines)

    def get_cursor(self) -> dict[str, int]:
        return {"line": self._state.cursor_line, "col": self._state.cursor_col}

    def set_text(self, text: str) -> None:
        self._last_action = None
        self._history_index = -1  # Exit history browsing mode
        # Push undo snapshot if content differs (makes programmatic changes undoable)
        if self.get_text() != text:
            self._push_undo_snapshot()
        self._set_text_internal(text)

    def insert_text_at_cursor(self, text: str) -> None:
        """Insert text at the current cursor position.

        Used for programmatic insertion (e.g., clipboard image markers).
        This is atomic for undo - single undo restores entire pre-insert state.
        """
        if not text:
            return
        self._push_undo_snapshot()
        self._last_action = None
        self._history_index = -1
        self._insert_text_at_cursor_internal(text)

    def _insert_text_at_cursor_internal(self, text: str) -> None:
        """Internal text insertion at cursor. Handles single and multi-line text.

        Does not push undo snapshots or trigger autocomplete - caller is responsible.
        Normalizes line endings and calls on_change once at the end.
        """
        if not text:
            return

        # Normalize line endings
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        inserted_lines = normalized.split("\n")

        current_line = self._state.lines[self._state.cursor_line] if self._state.cursor_line < len(self._state.lines) else ""
        before_cursor = current_line[: self._state.cursor_col]
        after_cursor = current_line[self._state.cursor_col :]

        if len(inserted_lines) == 1:
            # Single line - insert at cursor position
            self._state.lines[self._state.cursor_line] = before_cursor + normalized + after_cursor
            self._set_cursor_col(self._state.cursor_col + len(normalized))
        else:
            # Multi-line insertion
            new_lines: list[str] = []
            # All lines before current line
            new_lines.extend(self._state.lines[: self._state.cursor_line])
            # The first inserted line merged with text before cursor
            new_lines.append(before_cursor + inserted_lines[0])
            # All middle inserted lines
            new_lines.extend(inserted_lines[1:-1])
            # The last inserted line with text after cursor
            new_lines.append(inserted_lines[-1] + after_cursor)
            # All lines after current line
            new_lines.extend(self._state.lines[self._state.cursor_line + 1 :])

            self._state.lines = new_lines
            self._state.cursor_line += len(inserted_lines) - 1
            self._set_cursor_col(len(inserted_lines[-1]))

        if self.on_change:
            self.on_change(self.get_text())

    # -- Character insertion -------------------------------------------------

    def _insert_character(self, char: str, skip_undo_coalescing: bool = False) -> None:
        self._history_index = -1  # Exit history browsing mode

        # Undo coalescing (fish-style):
        # - Consecutive word chars coalesce into one undo unit
        # - Space captures state before itself (so undo removes space+following word together)
        # - Each space is separately undoable
        if not skip_undo_coalescing:
            if is_whitespace_char(char) or self._last_action != "type-word":
                self._push_undo_snapshot()
            self._last_action = "type-word"

        line = self._state.lines[self._state.cursor_line] if self._state.cursor_line < len(self._state.lines) else ""

        before = line[: self._state.cursor_col]
        after = line[self._state.cursor_col :]

        self._state.lines[self._state.cursor_line] = before + char + after
        self._set_cursor_col(self._state.cursor_col + len(char))

        if self.on_change:
            self.on_change(self.get_text())

        # Check if we should trigger or update autocomplete
        if not self._autocomplete_state:
            # Auto-trigger for "/" at the start of a line (slash commands)
            if char == "/" and self._is_at_start_of_message():
                self._try_trigger_autocomplete()
            # Auto-trigger for "@" file reference (fuzzy search)
            elif char == "@":
                current_line = self._state.lines[self._state.cursor_line] if self._state.cursor_line < len(self._state.lines) else ""
                text_before_cursor = current_line[: self._state.cursor_col]
                # Only trigger if @ is after whitespace or at start of line
                char_before_at = text_before_cursor[-2] if len(text_before_cursor) >= 2 else None
                if len(text_before_cursor) == 1 or char_before_at == " " or char_before_at == "\t":
                    self._try_trigger_autocomplete()
            # Also auto-trigger when typing letters in a slash command context
            elif re.match(r"[a-zA-Z0-9.\-_]", char):
                current_line = self._state.lines[self._state.cursor_line] if self._state.cursor_line < len(self._state.lines) else ""
                text_before_cursor = current_line[: self._state.cursor_col]
                # Check if we're in a slash command (with or without space for arguments)
                if self._is_in_slash_command_context(text_before_cursor):
                    self._try_trigger_autocomplete()
                # Check if we're in an @ file reference context
                elif re.search(r"(?:^|[\s])@[^\s]*$", text_before_cursor):
                    self._try_trigger_autocomplete()
        else:
            self._update_autocomplete()

    # -- Paste handling ------------------------------------------------------

    def _handle_paste(self, pasted_text: str) -> None:
        self._history_index = -1  # Exit history browsing mode
        self._last_action = None

        self._push_undo_snapshot()

        # Clean the pasted text
        clean_text = pasted_text.replace("\r\n", "\n").replace("\r", "\n")

        # Convert tabs to spaces (4 spaces per tab)
        tab_expanded_text = clean_text.replace("\t", "    ")

        # Filter out non-printable characters except newlines
        filtered_text = "".join(
            ch for ch in tab_expanded_text if ch == "\n" or ord(ch) >= 32
        )

        # If pasting a file path (starts with /, ~, or .) and the character before
        # the cursor is a word character, prepend a space for better readability
        if re.match(r"^[/~.]", filtered_text):
            current_line = self._state.lines[self._state.cursor_line] if self._state.cursor_line < len(self._state.lines) else ""
            char_before_cursor = current_line[self._state.cursor_col - 1] if self._state.cursor_col > 0 and self._state.cursor_col <= len(current_line) else ""
            if char_before_cursor and re.match(r"\w", char_before_cursor):
                filtered_text = f" {filtered_text}"

        # Split into lines to check for large paste
        pasted_lines = filtered_text.split("\n")

        # Check if this is a large paste (> 10 lines or > 1000 characters)
        total_chars = len(filtered_text)
        if len(pasted_lines) > 10 or total_chars > 1000:
            # Store the paste and insert a marker
            self._paste_counter += 1
            paste_id = self._paste_counter
            self._pastes[paste_id] = filtered_text

            # Insert marker like "[paste #1 +123 lines]" or "[paste #1 1234 chars]"
            if len(pasted_lines) > 10:
                marker = f"[paste #{paste_id} +{len(pasted_lines)} lines]"
            else:
                marker = f"[paste #{paste_id} {total_chars} chars]"
            self._insert_text_at_cursor_internal(marker)
            return

        if len(pasted_lines) == 1:
            # Single line - insert character by character to trigger autocomplete
            for ch in filtered_text:
                self._insert_character(ch, True)
            return

        # Multi-line paste - use direct state manipulation
        self._insert_text_at_cursor_internal(filtered_text)

    # -- New line / submit ---------------------------------------------------

    def _add_new_line(self) -> None:
        self._history_index = -1  # Exit history browsing mode
        self._last_action = None

        self._push_undo_snapshot()

        current_line = self._state.lines[self._state.cursor_line] if self._state.cursor_line < len(self._state.lines) else ""

        before = current_line[: self._state.cursor_col]
        after = current_line[self._state.cursor_col :]

        # Split current line
        self._state.lines[self._state.cursor_line] = before
        self._state.lines.insert(self._state.cursor_line + 1, after)

        # Move cursor to start of new line
        self._state.cursor_line += 1
        self._set_cursor_col(0)

        if self.on_change:
            self.on_change(self.get_text())

    def _should_submit_on_backslash_enter(self, data: str, kb: object) -> bool:
        if self.disable_submit:
            return False
        if not matches_key(data, "enter"):
            return False
        submit_keys = kb.get_keys("submit")  # type: ignore[attr-defined]
        has_shift_enter = "shift+enter" in submit_keys or "shift+return" in submit_keys
        if not has_shift_enter:
            return False

        current_line = self._state.lines[self._state.cursor_line] if self._state.cursor_line < len(self._state.lines) else ""
        return self._state.cursor_col > 0 and self._state.cursor_col <= len(current_line) and current_line[self._state.cursor_col - 1] == "\\"

    def _submit_value(self) -> None:
        result = "\n".join(self._state.lines).strip()
        for paste_id, paste_content in self._pastes.items():
            marker_regex = re.compile(
                rf"\[paste #{paste_id}( (\+\d+ lines|\d+ chars))?\]"
            )
            result = marker_regex.sub(paste_content, result)

        self._state = EditorState()
        self._pastes.clear()
        self._paste_counter = 0
        self._history_index = -1
        self._scroll_offset = 0
        self._undo_stack.clear()
        self._last_action = None

        if self.on_change:
            self.on_change("")
        if self.on_submit:
            self.on_submit(result)

    # -- Backspace / Forward delete ------------------------------------------

    def _handle_backspace(self) -> None:
        self._history_index = -1  # Exit history browsing mode
        self._last_action = None

        if self._state.cursor_col > 0:
            self._push_undo_snapshot()

            # Delete grapheme before cursor (handles emojis, combining characters, etc.)
            line = self._state.lines[self._state.cursor_line] if self._state.cursor_line < len(self._state.lines) else ""
            before_cursor = line[: self._state.cursor_col]

            # Find the last grapheme in the text before cursor
            grapheme_list = list(_grapheme.graphemes(before_cursor))
            last_grapheme = grapheme_list[-1] if grapheme_list else ""
            grapheme_length = len(last_grapheme) if last_grapheme else 1

            before = line[: self._state.cursor_col - grapheme_length]
            after = line[self._state.cursor_col :]

            self._state.lines[self._state.cursor_line] = before + after
            self._set_cursor_col(self._state.cursor_col - grapheme_length)
        elif self._state.cursor_line > 0:
            self._push_undo_snapshot()

            # Merge with previous line
            current_line = self._state.lines[self._state.cursor_line] if self._state.cursor_line < len(self._state.lines) else ""
            previous_line = self._state.lines[self._state.cursor_line - 1] if self._state.cursor_line - 1 < len(self._state.lines) else ""

            self._state.lines[self._state.cursor_line - 1] = previous_line + current_line
            del self._state.lines[self._state.cursor_line]

            self._state.cursor_line -= 1
            self._set_cursor_col(len(previous_line))

        if self.on_change:
            self.on_change(self.get_text())

        # Update or re-trigger autocomplete after backspace
        if self._autocomplete_state:
            self._update_autocomplete()
        else:
            # If autocomplete was cancelled (no matches), re-trigger if we're in a completable context
            current_line = self._state.lines[self._state.cursor_line] if self._state.cursor_line < len(self._state.lines) else ""
            text_before_cursor = current_line[: self._state.cursor_col]
            # Slash command context
            if self._is_in_slash_command_context(text_before_cursor):
                self._try_trigger_autocomplete()
            # @ file reference context
            elif re.search(r"(?:^|[\s])@[^\s]*$", text_before_cursor):
                self._try_trigger_autocomplete()

    def _handle_forward_delete(self) -> None:
        self._history_index = -1  # Exit history browsing mode
        self._last_action = None

        current_line = self._state.lines[self._state.cursor_line] if self._state.cursor_line < len(self._state.lines) else ""

        if self._state.cursor_col < len(current_line):
            self._push_undo_snapshot()

            # Delete grapheme at cursor position (handles emojis, combining characters, etc.)
            after_cursor = current_line[self._state.cursor_col :]

            # Find the first grapheme at cursor
            grapheme_list = list(_grapheme.graphemes(after_cursor))
            first_grapheme = grapheme_list[0] if grapheme_list else ""
            grapheme_length = len(first_grapheme) if first_grapheme else 1

            before = current_line[: self._state.cursor_col]
            after = current_line[self._state.cursor_col + grapheme_length :]
            self._state.lines[self._state.cursor_line] = before + after
        elif self._state.cursor_line < len(self._state.lines) - 1:
            self._push_undo_snapshot()

            # At end of line - merge with next line
            next_line = self._state.lines[self._state.cursor_line + 1] if self._state.cursor_line + 1 < len(self._state.lines) else ""
            self._state.lines[self._state.cursor_line] = current_line + next_line
            del self._state.lines[self._state.cursor_line + 1]

        if self.on_change:
            self.on_change(self.get_text())

        # Update or re-trigger autocomplete after forward delete
        if self._autocomplete_state:
            self._update_autocomplete()
        else:
            current_line = self._state.lines[self._state.cursor_line] if self._state.cursor_line < len(self._state.lines) else ""
            text_before_cursor = current_line[: self._state.cursor_col]
            # Slash command context
            if self._is_in_slash_command_context(text_before_cursor):
                self._try_trigger_autocomplete()
            # @ file reference context
            elif re.search(r"(?:^|[\s])@[^\s]*$", text_before_cursor):
                self._try_trigger_autocomplete()

    # -- Cursor column setter ------------------------------------------------

    def _set_cursor_col(self, col: int) -> None:
        """Set cursor column and clear preferred_visual_col.

        Use this for all non-vertical cursor movements to reset sticky column behavior.
        """
        self._state.cursor_col = col
        self._preferred_visual_col = None

    # -- Delete to line start/end -------------------------------------------

    def _delete_to_start_of_line(self) -> None:
        self._history_index = -1  # Exit history browsing mode

        current_line = self._state.lines[self._state.cursor_line] if self._state.cursor_line < len(self._state.lines) else ""

        if self._state.cursor_col > 0:
            self._push_undo_snapshot()

            # Calculate text to be deleted and save to kill ring (backward deletion = prepend)
            deleted_text = current_line[: self._state.cursor_col]
            self._kill_ring.push(deleted_text, prepend=True, accumulate=self._last_action == "kill")
            self._last_action = "kill"

            # Delete from start of line up to cursor
            self._state.lines[self._state.cursor_line] = current_line[self._state.cursor_col :]
            self._set_cursor_col(0)
        elif self._state.cursor_line > 0:
            self._push_undo_snapshot()

            # At start of line - merge with previous line, treating newline as deleted text
            self._kill_ring.push("\n", prepend=True, accumulate=self._last_action == "kill")
            self._last_action = "kill"

            previous_line = self._state.lines[self._state.cursor_line - 1] if self._state.cursor_line - 1 < len(self._state.lines) else ""
            self._state.lines[self._state.cursor_line - 1] = previous_line + current_line
            del self._state.lines[self._state.cursor_line]
            self._state.cursor_line -= 1
            self._set_cursor_col(len(previous_line))

        if self.on_change:
            self.on_change(self.get_text())

    def _delete_to_end_of_line(self) -> None:
        self._history_index = -1  # Exit history browsing mode

        current_line = self._state.lines[self._state.cursor_line] if self._state.cursor_line < len(self._state.lines) else ""

        if self._state.cursor_col < len(current_line):
            self._push_undo_snapshot()

            # Calculate text to be deleted and save to kill ring (forward deletion = append)
            deleted_text = current_line[self._state.cursor_col :]
            self._kill_ring.push(deleted_text, prepend=False, accumulate=self._last_action == "kill")
            self._last_action = "kill"

            # Delete from cursor to end of line
            self._state.lines[self._state.cursor_line] = current_line[: self._state.cursor_col]
        elif self._state.cursor_line < len(self._state.lines) - 1:
            self._push_undo_snapshot()

            # At end of line - merge with next line, treating newline as deleted text
            self._kill_ring.push("\n", prepend=False, accumulate=self._last_action == "kill")
            self._last_action = "kill"

            next_line = self._state.lines[self._state.cursor_line + 1] if self._state.cursor_line + 1 < len(self._state.lines) else ""
            self._state.lines[self._state.cursor_line] = current_line + next_line
            del self._state.lines[self._state.cursor_line + 1]

        if self.on_change:
            self.on_change(self.get_text())

    # -- Delete word backward/forward ----------------------------------------

    def _delete_word_backwards(self) -> None:
        self._history_index = -1  # Exit history browsing mode

        current_line = self._state.lines[self._state.cursor_line] if self._state.cursor_line < len(self._state.lines) else ""

        # If at start of line, behave like backspace at column 0 (merge with previous line)
        if self._state.cursor_col == 0:
            if self._state.cursor_line > 0:
                self._push_undo_snapshot()

                # Treat newline as deleted text (backward deletion = prepend)
                self._kill_ring.push("\n", prepend=True, accumulate=self._last_action == "kill")
                self._last_action = "kill"

                previous_line = self._state.lines[self._state.cursor_line - 1] if self._state.cursor_line - 1 < len(self._state.lines) else ""
                self._state.lines[self._state.cursor_line - 1] = previous_line + current_line
                del self._state.lines[self._state.cursor_line]
                self._state.cursor_line -= 1
                self._set_cursor_col(len(previous_line))
        else:
            self._push_undo_snapshot()

            # Save last_action before cursor movement (move_word_backwards resets it)
            was_kill = self._last_action == "kill"

            old_cursor_col = self._state.cursor_col
            self._move_word_backwards()
            delete_from = self._state.cursor_col
            self._set_cursor_col(old_cursor_col)

            deleted_text = current_line[delete_from : self._state.cursor_col]
            self._kill_ring.push(deleted_text, prepend=True, accumulate=was_kill)
            self._last_action = "kill"

            self._state.lines[self._state.cursor_line] = (
                current_line[:delete_from] + current_line[self._state.cursor_col :]
            )
            self._set_cursor_col(delete_from)

        if self.on_change:
            self.on_change(self.get_text())

    def _delete_word_forward(self) -> None:
        self._history_index = -1  # Exit history browsing mode

        current_line = self._state.lines[self._state.cursor_line] if self._state.cursor_line < len(self._state.lines) else ""

        # If at end of line, merge with next line (delete the newline)
        if self._state.cursor_col >= len(current_line):
            if self._state.cursor_line < len(self._state.lines) - 1:
                self._push_undo_snapshot()

                # Treat newline as deleted text (forward deletion = append)
                self._kill_ring.push("\n", prepend=False, accumulate=self._last_action == "kill")
                self._last_action = "kill"

                next_line = self._state.lines[self._state.cursor_line + 1] if self._state.cursor_line + 1 < len(self._state.lines) else ""
                self._state.lines[self._state.cursor_line] = current_line + next_line
                del self._state.lines[self._state.cursor_line + 1]
        else:
            self._push_undo_snapshot()

            # Save last_action before cursor movement (move_word_forwards resets it)
            was_kill = self._last_action == "kill"

            old_cursor_col = self._state.cursor_col
            self._move_word_forwards()
            delete_to = self._state.cursor_col
            self._set_cursor_col(old_cursor_col)

            deleted_text = current_line[self._state.cursor_col : delete_to]
            self._kill_ring.push(deleted_text, prepend=False, accumulate=was_kill)
            self._last_action = "kill"

            self._state.lines[self._state.cursor_line] = (
                current_line[: self._state.cursor_col] + current_line[delete_to:]
            )

        if self.on_change:
            self.on_change(self.get_text())

    # -- Visual line map -----------------------------------------------------

    def _build_visual_line_map(
        self, width: int
    ) -> list[dict[str, int]]:
        """Build a mapping from visual lines to logical positions.

        Returns a list where each element represents a visual line with:
        - logical_line: index into self._state.lines
        - start_col: starting column in the logical line
        - length: length of this visual line segment
        """
        visual_lines: list[dict[str, int]] = []

        for i, line in enumerate(self._state.lines):
            line_vis_width = visible_width(line)
            if not line:
                # Empty line still takes one visual line
                visual_lines.append({"logical_line": i, "start_col": 0, "length": 0})
            elif line_vis_width <= width:
                visual_lines.append({"logical_line": i, "start_col": 0, "length": len(line)})
            else:
                # Line needs wrapping - use word-aware wrapping
                chunks = word_wrap_line(line, width)
                for chunk in chunks:
                    visual_lines.append(
                        {
                            "logical_line": i,
                            "start_col": chunk.start_index,
                            "length": chunk.end_index - chunk.start_index,
                        }
                    )

        return visual_lines

    def _find_current_visual_line(
        self, visual_lines: list[dict[str, int]]
    ) -> int:
        """Find the visual line index for the current cursor position."""
        for i, vl in enumerate(visual_lines):
            if vl["logical_line"] == self._state.cursor_line:
                col_in_segment = self._state.cursor_col - vl["start_col"]
                # Cursor is in this segment if it's within range
                # For the last segment of a logical line, cursor can be at length (end position)
                is_last_segment_of_line = (
                    i == len(visual_lines) - 1
                    or visual_lines[i + 1]["logical_line"] != vl["logical_line"]
                )
                if col_in_segment >= 0 and (
                    col_in_segment < vl["length"]
                    or (is_last_segment_of_line and col_in_segment <= vl["length"])
                ):
                    return i
        # Fallback: return last visual line
        return len(visual_lines) - 1

    # -- Cursor movement -----------------------------------------------------

    def _move_to_visual_line(
        self,
        visual_lines: list[dict[str, int]],
        current_visual_line: int,
        target_visual_line: int,
    ) -> None:
        """Move cursor to a target visual line, applying sticky column logic."""
        current_vl = visual_lines[current_visual_line] if current_visual_line < len(visual_lines) else None
        target_vl = visual_lines[target_visual_line] if target_visual_line < len(visual_lines) else None

        if current_vl and target_vl:
            current_visual_col = self._state.cursor_col - current_vl["start_col"]

            # For non-last segments, clamp to length-1 to stay within the segment
            is_last_source_segment = (
                current_visual_line == len(visual_lines) - 1
                or visual_lines[current_visual_line + 1]["logical_line"] != current_vl["logical_line"]
            )
            source_max_visual_col = (
                current_vl["length"]
                if is_last_source_segment
                else max(0, current_vl["length"] - 1)
            )

            is_last_target_segment = (
                target_visual_line == len(visual_lines) - 1
                or visual_lines[target_visual_line + 1]["logical_line"] != target_vl["logical_line"]
            )
            target_max_visual_col = (
                target_vl["length"]
                if is_last_target_segment
                else max(0, target_vl["length"] - 1)
            )

            move_to_visual_col = self._compute_vertical_move_column(
                current_visual_col,
                source_max_visual_col,
                target_max_visual_col,
            )

            # Set cursor position
            self._state.cursor_line = target_vl["logical_line"]
            target_col = target_vl["start_col"] + move_to_visual_col
            logical_line = self._state.lines[target_vl["logical_line"]] if target_vl["logical_line"] < len(self._state.lines) else ""
            self._state.cursor_col = min(target_col, len(logical_line))

    def _compute_vertical_move_column(
        self,
        current_visual_col: int,
        source_max_visual_col: int,
        target_max_visual_col: int,
    ) -> int:
        """Compute the target visual column for vertical cursor movement.

        Implements the sticky column decision table.
        """
        has_preferred = self._preferred_visual_col is not None  # P
        cursor_in_middle = current_visual_col < source_max_visual_col  # S
        target_too_short = target_max_visual_col < current_visual_col  # T

        if not has_preferred or cursor_in_middle:
            if target_too_short:
                # Cases 2 and 7
                self._preferred_visual_col = current_visual_col
                return target_max_visual_col

            # Cases 1 and 6
            self._preferred_visual_col = None
            return current_visual_col

        target_cant_fit_preferred = target_max_visual_col < self._preferred_visual_col  # type: ignore[operator]  # U
        if target_too_short or target_cant_fit_preferred:
            # Cases 4 and 5
            return target_max_visual_col

        # Case 3
        result = self._preferred_visual_col  # type: ignore[assignment]
        self._preferred_visual_col = None
        return result  # type: ignore[return-value]

    def _move_to_line_start(self) -> None:
        self._last_action = None
        self._set_cursor_col(0)

    def _move_to_line_end(self) -> None:
        self._last_action = None
        current_line = self._state.lines[self._state.cursor_line] if self._state.cursor_line < len(self._state.lines) else ""
        self._set_cursor_col(len(current_line))

    def _move_cursor(self, delta_line: int, delta_col: int) -> None:
        self._last_action = None
        visual_lines = self._build_visual_line_map(self._last_width)
        current_visual_line = self._find_current_visual_line(visual_lines)

        if delta_line != 0:
            target_visual_line = current_visual_line + delta_line

            if 0 <= target_visual_line < len(visual_lines):
                self._move_to_visual_line(visual_lines, current_visual_line, target_visual_line)

        if delta_col != 0:
            current_line = self._state.lines[self._state.cursor_line] if self._state.cursor_line < len(self._state.lines) else ""

            if delta_col > 0:
                # Moving right - move by one grapheme
                if self._state.cursor_col < len(current_line):
                    after_cursor = current_line[self._state.cursor_col :]
                    grapheme_list = list(_grapheme.graphemes(after_cursor))
                    first_grapheme = grapheme_list[0] if grapheme_list else ""
                    self._set_cursor_col(self._state.cursor_col + (len(first_grapheme) if first_grapheme else 1))
                elif self._state.cursor_line < len(self._state.lines) - 1:
                    # Wrap to start of next logical line
                    self._state.cursor_line += 1
                    self._set_cursor_col(0)
                else:
                    # At end of last line - can't move, but set preferred_visual_col
                    current_vl = visual_lines[current_visual_line] if current_visual_line < len(visual_lines) else None
                    if current_vl:
                        self._preferred_visual_col = self._state.cursor_col - current_vl["start_col"]
            else:
                # Moving left - move by one grapheme
                if self._state.cursor_col > 0:
                    before_cursor = current_line[: self._state.cursor_col]
                    grapheme_list = list(_grapheme.graphemes(before_cursor))
                    last_grapheme = grapheme_list[-1] if grapheme_list else ""
                    self._set_cursor_col(self._state.cursor_col - (len(last_grapheme) if last_grapheme else 1))
                elif self._state.cursor_line > 0:
                    # Wrap to end of previous logical line
                    self._state.cursor_line -= 1
                    prev_line = self._state.lines[self._state.cursor_line] if self._state.cursor_line < len(self._state.lines) else ""
                    self._set_cursor_col(len(prev_line))

    def _page_scroll(self, direction: int) -> None:
        """Scroll by a page (direction: -1 for up, 1 for down)."""
        self._last_action = None
        terminal_rows = self._tui.terminal.rows
        page_size = max(5, terminal_rows * 3 // 10)

        visual_lines = self._build_visual_line_map(self._last_width)
        current_visual_line = self._find_current_visual_line(visual_lines)
        target_visual_line = max(
            0, min(len(visual_lines) - 1, current_visual_line + direction * page_size)
        )

        self._move_to_visual_line(visual_lines, current_visual_line, target_visual_line)

    # -- Word movement -------------------------------------------------------

    def _move_word_backwards(self) -> None:
        self._last_action = None
        current_line = self._state.lines[self._state.cursor_line] if self._state.cursor_line < len(self._state.lines) else ""

        # If at start of line, move to end of previous line
        if self._state.cursor_col == 0:
            if self._state.cursor_line > 0:
                self._state.cursor_line -= 1
                prev_line = self._state.lines[self._state.cursor_line] if self._state.cursor_line < len(self._state.lines) else ""
                self._set_cursor_col(len(prev_line))
            return

        text_before_cursor = current_line[: self._state.cursor_col]
        grapheme_list = list(_grapheme.graphemes(text_before_cursor))
        new_col = self._state.cursor_col

        # Skip trailing whitespace
        while grapheme_list and is_whitespace_char(grapheme_list[-1]):
            new_col -= len(grapheme_list.pop())

        if grapheme_list:
            last_g = grapheme_list[-1]
            if is_punctuation_char(last_g):
                # Skip punctuation run
                while grapheme_list and is_punctuation_char(grapheme_list[-1]):
                    new_col -= len(grapheme_list.pop())
            else:
                # Skip word run
                while (
                    grapheme_list
                    and not is_whitespace_char(grapheme_list[-1])
                    and not is_punctuation_char(grapheme_list[-1])
                ):
                    new_col -= len(grapheme_list.pop())

        self._set_cursor_col(new_col)

    def _move_word_forwards(self) -> None:
        self._last_action = None
        current_line = self._state.lines[self._state.cursor_line] if self._state.cursor_line < len(self._state.lines) else ""

        # If at end of line, move to start of next line
        if self._state.cursor_col >= len(current_line):
            if self._state.cursor_line < len(self._state.lines) - 1:
                self._state.cursor_line += 1
                self._set_cursor_col(0)
            return

        text_after_cursor = current_line[self._state.cursor_col :]
        grapheme_iter = iter(_grapheme.graphemes(text_after_cursor))
        new_col = self._state.cursor_col

        # Get first grapheme
        current_g: str | None = next(grapheme_iter, None)

        # Skip leading whitespace
        while current_g is not None and is_whitespace_char(current_g):
            new_col += len(current_g)
            current_g = next(grapheme_iter, None)

        if current_g is not None:
            if is_punctuation_char(current_g):
                # Skip punctuation run
                while current_g is not None and is_punctuation_char(current_g):
                    new_col += len(current_g)
                    current_g = next(grapheme_iter, None)
            else:
                # Skip word run
                while (
                    current_g is not None
                    and not is_whitespace_char(current_g)
                    and not is_punctuation_char(current_g)
                ):
                    new_col += len(current_g)
                    current_g = next(grapheme_iter, None)

        self._set_cursor_col(new_col)

    # -- Kill ring (yank / yank-pop) -----------------------------------------

    def _yank(self) -> None:
        """Yank (paste) the most recent kill ring entry at cursor position."""
        if self._kill_ring.length == 0:
            return

        self._push_undo_snapshot()

        text = self._kill_ring.peek()
        if text is not None:
            self._insert_yanked_text(text)

        self._last_action = "yank"

    def _yank_pop(self) -> None:
        """Cycle through kill ring (only works immediately after yank or yank-pop)."""
        # Only works if we just yanked and have more than one entry
        if self._last_action != "yank" or self._kill_ring.length <= 1:
            return

        self._push_undo_snapshot()

        # Delete the previously yanked text (still at end of ring before rotation)
        self._delete_yanked_text()

        # Rotate the ring: move end to front
        self._kill_ring.rotate()

        # Insert the new most recent entry (now at end after rotation)
        text = self._kill_ring.peek()
        if text is not None:
            self._insert_yanked_text(text)

        self._last_action = "yank"

    def _insert_yanked_text(self, text: str) -> None:
        """Insert text at cursor position (used by yank operations)."""
        self._history_index = -1  # Exit history browsing mode
        lines = text.split("\n")

        if len(lines) == 1:
            # Single line - insert at cursor
            current_line = self._state.lines[self._state.cursor_line] if self._state.cursor_line < len(self._state.lines) else ""
            before = current_line[: self._state.cursor_col]
            after = current_line[self._state.cursor_col :]
            self._state.lines[self._state.cursor_line] = before + text + after
            self._set_cursor_col(self._state.cursor_col + len(text))
        else:
            # Multi-line insert
            current_line = self._state.lines[self._state.cursor_line] if self._state.cursor_line < len(self._state.lines) else ""
            before = current_line[: self._state.cursor_col]
            after = current_line[self._state.cursor_col :]

            # First line merges with text before cursor
            self._state.lines[self._state.cursor_line] = before + (lines[0] or "")

            # Insert middle lines
            for j in range(1, len(lines) - 1):
                self._state.lines.insert(self._state.cursor_line + j, lines[j] or "")

            # Last line merges with text after cursor
            last_line_index = self._state.cursor_line + len(lines) - 1
            self._state.lines.insert(last_line_index, (lines[-1] or "") + after)

            # Update cursor position
            self._state.cursor_line = last_line_index
            self._set_cursor_col(len(lines[-1] or ""))

        if self.on_change:
            self.on_change(self.get_text())

    def _delete_yanked_text(self) -> None:
        """Delete the previously yanked text (used by yank-pop)."""
        yanked_text = self._kill_ring.peek()
        if not yanked_text:
            return

        yank_lines = yanked_text.split("\n")

        if len(yank_lines) == 1:
            # Single line - delete backward from cursor
            current_line = self._state.lines[self._state.cursor_line] if self._state.cursor_line < len(self._state.lines) else ""
            delete_len = len(yanked_text)
            before = current_line[: self._state.cursor_col - delete_len]
            after = current_line[self._state.cursor_col :]
            self._state.lines[self._state.cursor_line] = before + after
            self._set_cursor_col(self._state.cursor_col - delete_len)
        else:
            # Multi-line delete - cursor is at end of last yanked line
            start_line = self._state.cursor_line - (len(yank_lines) - 1)
            start_line_text = self._state.lines[start_line] if start_line < len(self._state.lines) else ""
            start_col = len(start_line_text) - len(yank_lines[0] or "")

            # Get text after cursor on current line
            current_line_text = self._state.lines[self._state.cursor_line] if self._state.cursor_line < len(self._state.lines) else ""
            after_cursor = current_line_text[self._state.cursor_col :]

            # Get text before yank start position
            before_yank = start_line_text[:start_col]

            # Remove all lines from start_line to cursor_line and replace with merged line
            self._state.lines[start_line : start_line + len(yank_lines)] = [before_yank + after_cursor]

            # Update cursor
            self._state.cursor_line = start_line
            self._set_cursor_col(start_col)

        if self.on_change:
            self.on_change(self.get_text())

    # -- Undo ----------------------------------------------------------------

    def _push_undo_snapshot(self) -> None:
        self._undo_stack.push(self._state)

    def _undo(self) -> None:
        self._history_index = -1  # Exit history browsing mode
        snapshot = self._undo_stack.pop()
        if snapshot is None:
            return
        self._state.lines = snapshot.lines
        self._state.cursor_line = snapshot.cursor_line
        self._state.cursor_col = snapshot.cursor_col
        self._last_action = None
        self._preferred_visual_col = None
        if self.on_change:
            self.on_change(self.get_text())

    # -- Character jump ------------------------------------------------------

    def _jump_to_char(self, char: str, direction: str) -> None:
        """Jump to the first occurrence of a character in the specified direction.

        Multi-line search. Case-sensitive. Skips the current cursor position.
        """
        self._last_action = None
        is_forward = direction == "forward"
        lines = self._state.lines

        if is_forward:
            line_range = range(self._state.cursor_line, len(lines))
        else:
            line_range = range(self._state.cursor_line, -1, -1)

        for line_idx in line_range:
            line = lines[line_idx] if line_idx < len(lines) else ""
            is_current_line = line_idx == self._state.cursor_line

            if is_current_line:
                if is_forward:
                    search_from = self._state.cursor_col + 1
                else:
                    search_from = self._state.cursor_col - 1
            else:
                search_from = None

            if is_forward:
                idx = line.find(char, search_from if search_from is not None else 0)
            else:
                if search_from is not None:
                    idx = line.rfind(char, 0, search_from + 1) if search_from >= 0 else -1
                else:
                    idx = line.rfind(char)

            if idx != -1:
                self._state.cursor_line = line_idx
                self._set_cursor_col(idx)
                return
        # No match found - cursor stays in place

    # -- Slash command / autocomplete helpers ---------------------------------

    def _is_slash_menu_allowed(self) -> bool:
        """Slash menu only allowed on the first line of the editor."""
        return self._state.cursor_line == 0

    def _is_at_start_of_message(self) -> bool:
        """Check if cursor is at start of message (for slash command detection)."""
        if not self._is_slash_menu_allowed():
            return False
        current_line = self._state.lines[self._state.cursor_line] if self._state.cursor_line < len(self._state.lines) else ""
        before_cursor = current_line[: self._state.cursor_col]
        return before_cursor.strip() == "" or before_cursor.strip() == "/"

    def _is_in_slash_command_context(self, text_before_cursor: str) -> bool:
        return self._is_slash_menu_allowed() and text_before_cursor.lstrip().startswith("/")

    # -- Autocomplete --------------------------------------------------------

    def _try_trigger_autocomplete(self, explicit_tab: bool = False) -> None:
        if not self._autocomplete_provider:
            return

        # Check if we should trigger file completion on Tab
        if explicit_tab:
            provider = self._autocomplete_provider
            should_trigger_fn = getattr(provider, "should_trigger_file_completion", None)
            if should_trigger_fn is not None:
                should_trigger = should_trigger_fn(
                    self._state.lines, self._state.cursor_line, self._state.cursor_col
                )
                if not should_trigger:
                    return

        suggestions = self._autocomplete_provider.get_suggestions(
            self._state.lines,
            self._state.cursor_line,
            self._state.cursor_col,
        )

        if suggestions and suggestions.get("items"):
            items = suggestions["items"]
            prefix = suggestions.get("prefix", "")
            self._autocomplete_prefix = prefix  # type: ignore[assignment]
            # Import SelectList here to avoid circular import at module level
            from pi.tui.components.select_list import SelectList

            self._autocomplete_list = SelectList(items, self._autocomplete_max_visible, self._theme.select_list)  # type: ignore[arg-type]
            self._autocomplete_state = "regular"
        else:
            self._cancel_autocomplete()

    def _handle_tab_completion(self) -> None:
        if not self._autocomplete_provider:
            return

        current_line = self._state.lines[self._state.cursor_line] if self._state.cursor_line < len(self._state.lines) else ""
        before_cursor = current_line[: self._state.cursor_col]

        # Check if we're in a slash command context
        if self._is_in_slash_command_context(before_cursor) and " " not in before_cursor.lstrip():
            self._handle_slash_command_completion()
        else:
            self._force_file_autocomplete(True)

    def _handle_slash_command_completion(self) -> None:
        self._try_trigger_autocomplete(True)

    def _force_file_autocomplete(self, explicit_tab: bool = False) -> None:
        if not self._autocomplete_provider:
            return

        # Check if provider supports force file suggestions via runtime check
        get_force_fn = getattr(self._autocomplete_provider, "get_force_file_suggestions", None)
        if not callable(get_force_fn):
            self._try_trigger_autocomplete(True)
            return

        suggestions = get_force_fn(
            self._state.lines,
            self._state.cursor_line,
            self._state.cursor_col,
        )

        if suggestions and suggestions.get("items"):
            items = suggestions["items"]
            prefix = suggestions.get("prefix", "")

            # If there's exactly one suggestion, apply it immediately
            if explicit_tab and len(items) == 1:
                item = items[0]
                self._push_undo_snapshot()
                self._last_action = None
                result = self._autocomplete_provider.apply_completion(
                    self._state.lines,
                    self._state.cursor_line,
                    self._state.cursor_col,
                    item,
                    prefix,
                )
                self._state.lines = result["lines"]  # type: ignore[assignment]
                self._state.cursor_line = result["cursor_line"]  # type: ignore[assignment]
                self._set_cursor_col(result["cursor_col"])  # type: ignore[arg-type]
                if self.on_change:
                    self.on_change(self.get_text())
                return

            self._autocomplete_prefix = prefix  # type: ignore[assignment]
            from pi.tui.components.select_list import SelectList

            self._autocomplete_list = SelectList(items, self._autocomplete_max_visible, self._theme.select_list)  # type: ignore[arg-type]
            self._autocomplete_state = "force"
        else:
            self._cancel_autocomplete()

    def _cancel_autocomplete(self) -> None:
        self._autocomplete_state = None
        self._autocomplete_list = None
        self._autocomplete_prefix = ""

    def is_showing_autocomplete(self) -> bool:
        return self._autocomplete_state is not None

    def _update_autocomplete(self) -> None:
        if not self._autocomplete_state or not self._autocomplete_provider:
            return

        if self._autocomplete_state == "force":
            self._force_file_autocomplete()
            return

        suggestions = self._autocomplete_provider.get_suggestions(
            self._state.lines,
            self._state.cursor_line,
            self._state.cursor_col,
        )
        if suggestions and suggestions.get("items"):
            items = suggestions["items"]
            prefix = suggestions.get("prefix", "")
            self._autocomplete_prefix = prefix  # type: ignore[assignment]
            # Always create new SelectList to ensure update
            from pi.tui.components.select_list import SelectList

            self._autocomplete_list = SelectList(items, self._autocomplete_max_visible, self._theme.select_list)  # type: ignore[arg-type]
        else:
            self._cancel_autocomplete()
