"""Input component - single-line text input with horizontal scrolling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from pi.tui.keybindings import get_editor_keybindings
from pi.tui.kill_ring import KillRing
from pi.tui.tui import CURSOR_MARKER
from pi.tui.undo_stack import UndoStack
from pi.tui.utils import get_segmenter, is_punctuation_char, is_whitespace_char, visible_width

_segmenter = get_segmenter()


@dataclass
class _InputState:
    value: str = ""
    cursor: int = 0


class Input:
    """Input component - single-line text input with horizontal scrolling."""

    def __init__(self) -> None:
        self._value: str = ""
        self._cursor: int = 0

        self.on_submit: Callable[[str], None] | None = None
        self.on_escape: Callable[[], None] | None = None

        # Focusable interface
        self.focused: bool = False

        # Bracketed paste mode
        self._paste_buffer: str = ""
        self._is_in_paste: bool = False

        # Kill ring
        self._kill_ring = KillRing()
        self._last_action: str | None = None  # "kill", "yank", "type-word"

        # Undo
        self._undo_stack: UndoStack[_InputState] = UndoStack()

    def get_value(self) -> str:
        return self._value

    def set_value(self, value: str) -> None:
        self._value = value
        self._cursor = min(self._cursor, len(value))

    def handle_input(self, data: str) -> None:
        # Handle bracketed paste
        if "\x1b[200~" in data:
            self._is_in_paste = True
            self._paste_buffer = ""
            data = data.replace("\x1b[200~", "")

        if self._is_in_paste:
            self._paste_buffer += data
            end_index = self._paste_buffer.find("\x1b[201~")
            if end_index != -1:
                paste_content = self._paste_buffer[:end_index]
                self._handle_paste(paste_content)
                self._is_in_paste = False
                remaining = self._paste_buffer[end_index + 6:]
                self._paste_buffer = ""
                if remaining:
                    self.handle_input(remaining)
            return

        kb = get_editor_keybindings()

        if kb.matches(data, "selectCancel"):
            if self.on_escape:
                self.on_escape()
            return

        if kb.matches(data, "undo"):
            self._undo()
            return

        if kb.matches(data, "submit") or data == "\n":
            if self.on_submit:
                self.on_submit(self._value)
            return

        if kb.matches(data, "deleteCharBackward"):
            self._handle_backspace()
            return

        if kb.matches(data, "deleteCharForward"):
            self._handle_forward_delete()
            return

        if kb.matches(data, "deleteWordBackward"):
            self._delete_word_backwards()
            return

        if kb.matches(data, "deleteWordForward"):
            self._delete_word_forward()
            return

        if kb.matches(data, "deleteToLineStart"):
            self._delete_to_line_start()
            return

        if kb.matches(data, "deleteToLineEnd"):
            self._delete_to_line_end()
            return

        if kb.matches(data, "yank"):
            self._yank()
            return

        if kb.matches(data, "yankPop"):
            self._yank_pop()
            return

        if kb.matches(data, "cursorLeft"):
            self._last_action = None
            if self._cursor > 0:
                before = self._value[: self._cursor]
                graphemes = list(_segmenter.segment(before))
                last = graphemes[-1] if graphemes else None
                self._cursor -= len(last) if last else 1
            return

        if kb.matches(data, "cursorRight"):
            self._last_action = None
            if self._cursor < len(self._value):
                after = self._value[self._cursor :]
                graphemes = list(_segmenter.segment(after))
                first = graphemes[0] if graphemes else None
                self._cursor += len(first) if first else 1
            return

        if kb.matches(data, "cursorLineStart"):
            self._last_action = None
            self._cursor = 0
            return

        if kb.matches(data, "cursorLineEnd"):
            self._last_action = None
            self._cursor = len(self._value)
            return

        if kb.matches(data, "cursorWordLeft"):
            self._move_word_backwards()
            return

        if kb.matches(data, "cursorWordRight"):
            self._move_word_forwards()
            return

        # Regular character input
        has_control = any(
            ord(ch) < 32 or ord(ch) == 0x7F or (0x80 <= ord(ch) <= 0x9F)
            for ch in data
        )
        if not has_control:
            self._insert_character(data)

    def _insert_character(self, char: str) -> None:
        if is_whitespace_char(char) or self._last_action != "type-word":
            self._push_undo()
        self._last_action = "type-word"

        self._value = self._value[: self._cursor] + char + self._value[self._cursor :]
        self._cursor += len(char)

    def _handle_backspace(self) -> None:
        self._last_action = None
        if self._cursor > 0:
            self._push_undo()
            before = self._value[: self._cursor]
            graphemes = list(_segmenter.segment(before))
            last = graphemes[-1] if graphemes else None
            gl = len(last) if last else 1
            self._value = self._value[: self._cursor - gl] + self._value[self._cursor :]
            self._cursor -= gl

    def _handle_forward_delete(self) -> None:
        self._last_action = None
        if self._cursor < len(self._value):
            self._push_undo()
            after = self._value[self._cursor :]
            graphemes = list(_segmenter.segment(after))
            first = graphemes[0] if graphemes else None
            gl = len(first) if first else 1
            self._value = self._value[: self._cursor] + self._value[self._cursor + gl :]

    def _delete_to_line_start(self) -> None:
        if self._cursor == 0:
            return
        self._push_undo()
        deleted = self._value[: self._cursor]
        self._kill_ring.push(deleted, prepend=True, accumulate=self._last_action == "kill")
        self._last_action = "kill"
        self._value = self._value[self._cursor :]
        self._cursor = 0

    def _delete_to_line_end(self) -> None:
        if self._cursor >= len(self._value):
            return
        self._push_undo()
        deleted = self._value[self._cursor :]
        self._kill_ring.push(deleted, prepend=False, accumulate=self._last_action == "kill")
        self._last_action = "kill"
        self._value = self._value[: self._cursor]

    def _delete_word_backwards(self) -> None:
        if self._cursor == 0:
            return
        was_kill = self._last_action == "kill"
        self._push_undo()
        old_cursor = self._cursor
        self._move_word_backwards()
        delete_from = self._cursor
        self._cursor = old_cursor
        deleted = self._value[delete_from : self._cursor]
        self._kill_ring.push(deleted, prepend=True, accumulate=was_kill)
        self._last_action = "kill"
        self._value = self._value[:delete_from] + self._value[self._cursor :]
        self._cursor = delete_from

    def _delete_word_forward(self) -> None:
        if self._cursor >= len(self._value):
            return
        was_kill = self._last_action == "kill"
        self._push_undo()
        old_cursor = self._cursor
        self._move_word_forwards()
        delete_to = self._cursor
        self._cursor = old_cursor
        deleted = self._value[self._cursor : delete_to]
        self._kill_ring.push(deleted, prepend=False, accumulate=was_kill)
        self._last_action = "kill"
        self._value = self._value[: self._cursor] + self._value[delete_to:]

    def _yank(self) -> None:
        text = self._kill_ring.peek()
        if not text:
            return
        self._push_undo()
        self._value = self._value[: self._cursor] + text + self._value[self._cursor :]
        self._cursor += len(text)
        self._last_action = "yank"

    def _yank_pop(self) -> None:
        if self._last_action != "yank" or self._kill_ring.length <= 1:
            return
        self._push_undo()
        prev_text = self._kill_ring.peek() or ""
        self._value = (
            self._value[: self._cursor - len(prev_text)] + self._value[self._cursor :]
        )
        self._cursor -= len(prev_text)
        self._kill_ring.rotate()
        text = self._kill_ring.peek() or ""
        self._value = self._value[: self._cursor] + text + self._value[self._cursor :]
        self._cursor += len(text)
        self._last_action = "yank"

    def _push_undo(self) -> None:
        self._undo_stack.push(_InputState(value=self._value, cursor=self._cursor))

    def _undo(self) -> None:
        snapshot = self._undo_stack.pop()
        if not snapshot:
            return
        self._value = snapshot.value
        self._cursor = snapshot.cursor
        self._last_action = None

    def _move_word_backwards(self) -> None:
        if self._cursor == 0:
            return
        self._last_action = None
        before = self._value[: self._cursor]
        graphemes = list(_segmenter.segment(before))

        # Skip trailing whitespace
        while graphemes and is_whitespace_char(graphemes[-1]):
            self._cursor -= len(graphemes.pop())

        if graphemes:
            last = graphemes[-1]
            if is_punctuation_char(last):
                while graphemes and is_punctuation_char(graphemes[-1]):
                    self._cursor -= len(graphemes.pop())
            else:
                while (
                    graphemes
                    and not is_whitespace_char(graphemes[-1])
                    and not is_punctuation_char(graphemes[-1])
                ):
                    self._cursor -= len(graphemes.pop())

    def _move_word_forwards(self) -> None:
        if self._cursor >= len(self._value):
            return
        self._last_action = None
        after = self._value[self._cursor :]
        graphemes = list(_segmenter.segment(after))
        idx = 0

        # Skip leading whitespace
        while idx < len(graphemes) and is_whitespace_char(graphemes[idx]):
            self._cursor += len(graphemes[idx])
            idx += 1

        if idx < len(graphemes):
            first = graphemes[idx]
            if is_punctuation_char(first):
                while idx < len(graphemes) and is_punctuation_char(graphemes[idx]):
                    self._cursor += len(graphemes[idx])
                    idx += 1
            else:
                while (
                    idx < len(graphemes)
                    and not is_whitespace_char(graphemes[idx])
                    and not is_punctuation_char(graphemes[idx])
                ):
                    self._cursor += len(graphemes[idx])
                    idx += 1

    def _handle_paste(self, pasted_text: str) -> None:
        self._last_action = None
        self._push_undo()
        clean_text = pasted_text.replace("\r\n", "").replace("\r", "").replace("\n", "")
        self._value = self._value[: self._cursor] + clean_text + self._value[self._cursor :]
        self._cursor += len(clean_text)

    def invalidate(self) -> None:
        pass

    def render(self, width: int) -> list[str]:
        prompt = "> "
        available_width = width - len(prompt)

        if available_width <= 0:
            return [prompt]

        visible_text = ""
        cursor_display = self._cursor

        if len(self._value) < available_width:
            visible_text = self._value
        else:
            scroll_width = (
                available_width - 1
                if self._cursor == len(self._value)
                else available_width
            )
            half_width = scroll_width // 2

            def find_valid_start(start: int) -> int:
                while start < len(self._value):
                    code = ord(self._value[start])
                    if 0xDC00 <= code < 0xE000:
                        start += 1
                        continue
                    break
                return start

            def find_valid_end(end: int) -> int:
                while end > 0:
                    code = ord(self._value[end - 1])
                    if 0xD800 <= code < 0xDC00:
                        end -= 1
                        continue
                    break
                return end

            if self._cursor < half_width:
                visible_text = self._value[: find_valid_end(scroll_width)]
                cursor_display = self._cursor
            elif self._cursor > len(self._value) - half_width:
                start = find_valid_start(len(self._value) - scroll_width)
                visible_text = self._value[start:]
                cursor_display = self._cursor - start
            else:
                start = find_valid_start(self._cursor - half_width)
                visible_text = self._value[start : find_valid_end(start + scroll_width)]
                cursor_display = half_width

        # Build line with cursor
        after_cursor_text = visible_text[cursor_display:]
        graphemes = list(_segmenter.segment(after_cursor_text)) if after_cursor_text else []
        cursor_grapheme = graphemes[0] if graphemes else None

        before_cursor = visible_text[:cursor_display]
        at_cursor = cursor_grapheme if cursor_grapheme else " "
        after_cursor = visible_text[cursor_display + len(at_cursor) :]

        marker = CURSOR_MARKER if self.focused else ""
        cursor_char = f"\x1b[7m{at_cursor}\x1b[27m"
        text_with_cursor = before_cursor + marker + cursor_char + after_cursor

        visual_length = visible_width(text_with_cursor)
        padding = " " * max(0, available_width - visual_length)
        line = prompt + text_with_cursor + padding

        return [line]
