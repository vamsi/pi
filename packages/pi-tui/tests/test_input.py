"""Tests for the Input component."""

from __future__ import annotations

from pi.tui.components.input import Input
from pi.tui.utils import visible_width

# Raw escape codes for key sequences
KEY_UP = "\x1b[A"
KEY_DOWN = "\x1b[B"
KEY_LEFT = "\x1b[D"
KEY_RIGHT = "\x1b[C"
KEY_HOME = "\x1b[H"
KEY_END = "\x1b[F"
KEY_ENTER = "\r"
KEY_ESCAPE = "\x1b"
KEY_BACKSPACE = "\x7f"
KEY_DELETE = "\x1b[3~"


class TestInputInitialState:
    """Input starts empty and renders a prompt with cursor."""

    def test_initial_value_is_empty(self) -> None:
        inp = Input()
        assert inp.get_value() == ""

    def test_initial_render_shows_prompt(self) -> None:
        inp = Input()
        lines = inp.render(40)
        assert len(lines) == 1
        assert lines[0].startswith("> ")

    def test_initial_render_shows_cursor_block(self) -> None:
        inp = Input()
        lines = inp.render(40)
        # The cursor is rendered as reverse video: ESC[7m ... ESC[27m
        assert "\x1b[7m" in lines[0]
        assert "\x1b[27m" in lines[0]

    def test_focused_flag_defaults_to_false(self) -> None:
        inp = Input()
        assert inp.focused is False


class TestInputCharacterInsertion:
    """handle_input with regular characters inserts text at cursor."""

    def test_single_char_insertion(self) -> None:
        inp = Input()
        inp.handle_input("a")
        assert inp.get_value() == "a"

    def test_multiple_char_insertion(self) -> None:
        inp = Input()
        inp.handle_input("h")
        inp.handle_input("i")
        assert inp.get_value() == "hi"

    def test_multi_char_string_insertion(self) -> None:
        inp = Input()
        inp.handle_input("hello")
        assert inp.get_value() == "hello"

    def test_insertion_at_cursor_position(self) -> None:
        inp = Input()
        inp.handle_input("ac")
        # Move cursor left once to be between 'a' and 'c'
        inp.handle_input(KEY_LEFT)
        inp.handle_input("b")
        assert inp.get_value() == "abc"

    def test_control_chars_are_not_inserted(self) -> None:
        inp = Input()
        inp.handle_input("ab")
        # Control character (e.g. \x01 which is ctrl+a / home binding)
        # should not insert as text
        value_before = inp.get_value()
        inp.handle_input("\x03")  # ctrl+c triggers escape callback
        # Value shouldn't have \x03 inserted
        assert "\x03" not in inp.get_value()


class TestInputBackspaceAndDelete:
    """Backspace deletes character before cursor, delete removes after."""

    def test_backspace_deletes_last_char(self) -> None:
        inp = Input()
        inp.handle_input("abc")
        inp.handle_input(KEY_BACKSPACE)
        assert inp.get_value() == "ab"

    def test_backspace_on_empty_does_nothing(self) -> None:
        inp = Input()
        inp.handle_input(KEY_BACKSPACE)
        assert inp.get_value() == ""

    def test_backspace_in_middle(self) -> None:
        inp = Input()
        inp.handle_input("abc")
        inp.handle_input(KEY_LEFT)  # cursor after 'b'
        inp.handle_input(KEY_BACKSPACE)  # deletes 'b'
        assert inp.get_value() == "ac"

    def test_multiple_backspaces(self) -> None:
        inp = Input()
        inp.handle_input("hello")
        inp.handle_input(KEY_BACKSPACE)
        inp.handle_input(KEY_BACKSPACE)
        assert inp.get_value() == "hel"

    def test_forward_delete_removes_char_after_cursor(self) -> None:
        inp = Input()
        inp.handle_input("abc")
        inp.handle_input(KEY_HOME)  # cursor at start
        inp.handle_input(KEY_DELETE)  # deletes 'a'
        assert inp.get_value() == "bc"

    def test_forward_delete_at_end_does_nothing(self) -> None:
        inp = Input()
        inp.handle_input("abc")
        inp.handle_input(KEY_DELETE)
        assert inp.get_value() == "abc"

    def test_forward_delete_in_middle(self) -> None:
        inp = Input()
        inp.handle_input("abc")
        inp.handle_input(KEY_HOME)
        inp.handle_input(KEY_RIGHT)  # cursor after 'a'
        inp.handle_input(KEY_DELETE)  # deletes 'b'
        assert inp.get_value() == "ac"


class TestInputCursorMovement:
    """Cursor moves correctly with arrow keys, home, and end."""

    def test_left_arrow_moves_cursor_left(self) -> None:
        inp = Input()
        inp.handle_input("abc")
        inp.handle_input(KEY_LEFT)
        inp.handle_input("X")
        assert inp.get_value() == "abXc"

    def test_right_arrow_moves_cursor_right(self) -> None:
        inp = Input()
        inp.handle_input("abc")
        inp.handle_input(KEY_HOME)
        inp.handle_input(KEY_RIGHT)
        inp.handle_input("X")
        assert inp.get_value() == "aXbc"

    def test_left_at_beginning_stays_at_beginning(self) -> None:
        inp = Input()
        inp.handle_input("abc")
        inp.handle_input(KEY_HOME)
        inp.handle_input(KEY_LEFT)  # should be no-op
        inp.handle_input("X")
        assert inp.get_value() == "Xabc"

    def test_right_at_end_stays_at_end(self) -> None:
        inp = Input()
        inp.handle_input("abc")
        inp.handle_input(KEY_RIGHT)  # should be no-op (already at end)
        inp.handle_input("X")
        assert inp.get_value() == "abcX"

    def test_home_moves_to_beginning(self) -> None:
        inp = Input()
        inp.handle_input("hello")
        inp.handle_input(KEY_HOME)
        inp.handle_input("X")
        assert inp.get_value() == "Xhello"

    def test_end_moves_to_end(self) -> None:
        inp = Input()
        inp.handle_input("hello")
        inp.handle_input(KEY_HOME)
        inp.handle_input(KEY_END)
        inp.handle_input("X")
        assert inp.get_value() == "helloX"

    def test_cursor_movement_sequence(self) -> None:
        inp = Input()
        inp.handle_input("abcd")
        inp.handle_input(KEY_HOME)
        inp.handle_input(KEY_RIGHT)
        inp.handle_input(KEY_RIGHT)
        inp.handle_input("X")
        assert inp.get_value() == "abXcd"


class TestInputGetValue:
    """get_value returns the current text in the input."""

    def test_get_value_empty(self) -> None:
        inp = Input()
        assert inp.get_value() == ""

    def test_get_value_after_typing(self) -> None:
        inp = Input()
        inp.handle_input("test")
        assert inp.get_value() == "test"

    def test_get_value_after_deletion(self) -> None:
        inp = Input()
        inp.handle_input("test")
        inp.handle_input(KEY_BACKSPACE)
        assert inp.get_value() == "tes"

    def test_get_value_reflects_all_edits(self) -> None:
        inp = Input()
        inp.handle_input("hello")
        inp.handle_input(KEY_HOME)
        inp.handle_input(KEY_DELETE)
        inp.handle_input("H")
        assert inp.get_value() == "Hello"


class TestInputSetValue:
    """set_value updates the text and adjusts cursor position."""

    def test_set_value_updates_text(self) -> None:
        inp = Input()
        inp.set_value("new text")
        assert inp.get_value() == "new text"

    def test_set_value_replaces_existing_text(self) -> None:
        inp = Input()
        inp.handle_input("old text")
        inp.set_value("new text")
        assert inp.get_value() == "new text"

    def test_set_value_clamps_cursor(self) -> None:
        inp = Input()
        inp.handle_input("long text here")
        # Cursor is at end (position 14)
        inp.set_value("hi")
        # Cursor should be clamped to len("hi") = 2
        inp.handle_input("X")
        assert inp.get_value() == "hiX"

    def test_set_value_to_empty(self) -> None:
        inp = Input()
        inp.handle_input("something")
        inp.set_value("")
        assert inp.get_value() == ""

    def test_typing_after_set_value(self) -> None:
        # set_value clamps cursor within bounds but doesn't move it to the end.
        # Cursor starts at 0, so typing inserts at position 0.
        inp = Input()
        inp.set_value("prefix")
        inp.handle_input("_suffix")
        assert inp.get_value() == "_suffixprefix"

    def test_typing_after_set_value_with_end(self) -> None:
        # Move cursor to end first, then set_value keeps it clamped at end
        inp = Input()
        inp.handle_input("x")  # cursor moves to 1
        inp.set_value("prefix")  # cursor clamped to min(1, 6) = 1
        inp.handle_input(KEY_END)  # move cursor to end
        inp.handle_input("_suffix")
        assert inp.get_value() == "prefix_suffix"


class TestInputSubmitAndEscape:
    """Enter triggers on_submit, Escape triggers on_escape."""

    def test_enter_triggers_on_submit(self) -> None:
        inp = Input()
        inp.handle_input("hello")
        submitted_values: list[str] = []
        inp.on_submit = lambda v: submitted_values.append(v)
        inp.handle_input(KEY_ENTER)
        assert submitted_values == ["hello"]

    def test_escape_triggers_on_escape(self) -> None:
        inp = Input()
        cancelled = []
        inp.on_escape = lambda: cancelled.append(True)
        inp.handle_input(KEY_ESCAPE)
        assert cancelled == [True]

    def test_enter_without_callback_does_not_raise(self) -> None:
        inp = Input()
        inp.handle_input("text")
        inp.handle_input(KEY_ENTER)  # Should not raise

    def test_escape_without_callback_does_not_raise(self) -> None:
        inp = Input()
        inp.handle_input(KEY_ESCAPE)  # Should not raise


class TestInputRender:
    """Render produces correctly sized output."""

    def test_render_returns_single_line(self) -> None:
        inp = Input()
        inp.handle_input("test")
        lines = inp.render(40)
        assert len(lines) == 1

    def test_render_includes_prompt(self) -> None:
        inp = Input()
        inp.handle_input("hello")
        lines = inp.render(40)
        assert "> " in lines[0]

    def test_render_narrow_width(self) -> None:
        inp = Input()
        lines = inp.render(3)
        assert len(lines) == 1
