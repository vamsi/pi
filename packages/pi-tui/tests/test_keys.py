"""Tests for pi.tui.keys — keyboard input parsing and matching."""

from __future__ import annotations

import pytest

from pi.tui.keys import (
    Key,
    LEGACY_KEY_SEQUENCES,
    LEGACY_SHIFT_SEQUENCES,
    LEGACY_CTRL_SEQUENCES,
    LEGACY_ALT_SEQUENCES,
    LEGACY_CTRL_SHIFT_SEQUENCES,
    LEGACY_CTRL_ALT_SEQUENCES,
    LEGACY_SHIFT_ALT_SEQUENCES,
    LEGACY_CTRL_SHIFT_ALT_SEQUENCES,
    SHIFTED_KEY_MAP,
    UNSHIFTED_KEY_MAP,
    SYMBOL_KEYS,
    MODIFIERS,
    CODEPOINTS,
    ParsedKittySequence,
    is_key_release,
    is_key_repeat,
    is_kitty_protocol_active,
    matches_key,
    matches_kitty_sequence,
    matches_modify_other_keys,
    parse_key,
    parse_key_id,
    parse_kitty_sequence,
    raw_ctrl_char,
    set_kitty_protocol_active,
)


# ---------------------------------------------------------------------------
# Key helper class
# ---------------------------------------------------------------------------


class TestKeyConstants:
    """Key class exposes named constants for common keys."""

    def test_arrow_keys(self):
        assert Key.up == "up"
        assert Key.down == "down"
        assert Key.left == "left"
        assert Key.right == "right"

    def test_special_keys(self):
        assert Key.escape == "escape"
        assert Key.esc == "esc"
        assert Key.enter == "enter"
        assert Key.tab == "tab"
        assert Key.space == "space"
        assert Key.backspace == "backspace"
        assert Key.delete == "delete"
        assert Key.insert == "insert"
        assert Key.home == "home"
        assert Key.end == "end"
        assert Key.page_up == "pageUp"
        assert Key.page_down == "pageDown"

    def test_function_keys(self):
        assert Key.f1 == "f1"
        assert Key.f5 == "f5"
        assert Key.f12 == "f12"

    def test_symbol_keys(self):
        assert Key.backtick == "`"
        assert Key.hyphen == "-"
        assert Key.equals == "="
        assert Key.open_bracket == "["
        assert Key.close_bracket == "]"
        assert Key.backslash == "\\"
        assert Key.semicolon == ";"
        assert Key.quote == "'"
        assert Key.comma == ","
        assert Key.period == "."
        assert Key.slash == "/"


class TestKeyModifierCombinators:
    """Key.ctrl / Key.alt / Key.shift / etc. produce correct identifiers."""

    def test_ctrl(self):
        assert Key.ctrl("c") == "ctrl+c"
        assert Key.ctrl("a") == "ctrl+a"

    def test_alt(self):
        assert Key.alt("x") == "alt+x"
        assert Key.alt("f") == "alt+f"

    def test_shift(self):
        assert Key.shift("A") == "shift+A"
        assert Key.shift("enter") == "shift+enter"

    def test_ctrl_shift(self):
        assert Key.ctrl_shift("a") == "ctrl+shift+a"

    def test_ctrl_alt(self):
        assert Key.ctrl_alt("x") == "ctrl+alt+x"

    def test_shift_alt(self):
        assert Key.shift_alt("x") == "shift+alt+x"

    def test_ctrl_shift_alt(self):
        assert Key.ctrl_shift_alt("x") == "ctrl+shift+alt+x"


# ---------------------------------------------------------------------------
# Kitty protocol global toggle
# ---------------------------------------------------------------------------


class TestKittyProtocolToggle:
    """set_kitty_protocol_active / is_kitty_protocol_active toggle global state."""

    def teardown_method(self):
        # Reset to default after each test
        set_kitty_protocol_active(False)

    def test_default_inactive(self):
        set_kitty_protocol_active(False)
        assert is_kitty_protocol_active() is False

    def test_activate(self):
        set_kitty_protocol_active(True)
        assert is_kitty_protocol_active() is True

    def test_deactivate(self):
        set_kitty_protocol_active(True)
        set_kitty_protocol_active(False)
        assert is_kitty_protocol_active() is False


# ---------------------------------------------------------------------------
# parse_key_id
# ---------------------------------------------------------------------------


class TestParseKeyId:
    """parse_key_id splits a key identifier into modifiers bitmask + key."""

    def test_simple_letter(self):
        result = parse_key_id("a")
        assert result is not None
        assert result["modifiers"] == 0
        assert result["key"] == "a"

    def test_ctrl_modifier(self):
        result = parse_key_id("ctrl+c")
        assert result is not None
        assert result["modifiers"] == MODIFIERS["ctrl"]
        assert result["key"] == "c"

    def test_shift_modifier(self):
        result = parse_key_id("shift+enter")
        assert result is not None
        assert result["modifiers"] == MODIFIERS["shift"]
        assert result["key"] == "enter"

    def test_alt_modifier(self):
        result = parse_key_id("alt+x")
        assert result is not None
        assert result["modifiers"] == MODIFIERS["alt"]
        assert result["key"] == "x"

    def test_combined_modifiers(self):
        result = parse_key_id("ctrl+shift+a")
        assert result is not None
        assert result["modifiers"] == (MODIFIERS["ctrl"] | MODIFIERS["shift"])
        assert result["key"] == "a"

    def test_all_three_modifiers(self):
        result = parse_key_id("ctrl+shift+alt+f5")
        assert result is not None
        expected_mod = MODIFIERS["ctrl"] | MODIFIERS["shift"] | MODIFIERS["alt"]
        assert result["modifiers"] == expected_mod
        assert result["key"] == "f5"

    def test_empty_returns_none(self):
        assert parse_key_id("") is None

    def test_modifier_only_returns_none(self):
        # "ctrl+" splits into ["ctrl", ""] -- key part is empty
        assert parse_key_id("ctrl+") is None


# ---------------------------------------------------------------------------
# raw_ctrl_char
# ---------------------------------------------------------------------------


class TestRawCtrlChar:
    """raw_ctrl_char converts a base key to the corresponding control byte."""

    def test_ctrl_a(self):
        assert raw_ctrl_char("a") == "\x01"

    def test_ctrl_c(self):
        assert raw_ctrl_char("c") == "\x03"

    def test_ctrl_z(self):
        assert raw_ctrl_char("z") == "\x1a"

    def test_ctrl_uppercase(self):
        # Should handle uppercase by lowering
        assert raw_ctrl_char("A") == "\x01"

    def test_ctrl_bracket(self):
        assert raw_ctrl_char("[") == chr(27)  # ESC

    def test_ctrl_question_mark(self):
        assert raw_ctrl_char("?") == chr(127)  # DEL

    def test_ctrl_at(self):
        assert raw_ctrl_char("@") == chr(0)  # NUL

    def test_multi_char_returns_none(self):
        assert raw_ctrl_char("ab") is None

    def test_nonmappable_returns_none(self):
        assert raw_ctrl_char("1") is None


# ---------------------------------------------------------------------------
# matches_key — simple characters
# ---------------------------------------------------------------------------


class TestMatchesKeySimpleChars:
    """matches_key correctly matches plain single-character input."""

    def test_letter_a(self):
        assert matches_key("a", "a") is True

    def test_letter_mismatch(self):
        assert matches_key("a", "b") is False

    def test_digit(self):
        assert matches_key("5", "5") is True

    def test_symbol(self):
        assert matches_key("/", "/") is True

    def test_empty_key_id(self):
        assert matches_key("a", "") is False


# ---------------------------------------------------------------------------
# matches_key — ctrl sequences
# ---------------------------------------------------------------------------


class TestMatchesKeyCtrl:
    """matches_key recognises ctrl+letter via raw control characters."""

    def test_ctrl_c(self):
        assert matches_key("\x03", "ctrl+c") is True

    def test_ctrl_a(self):
        assert matches_key("\x01", "ctrl+a") is True

    def test_ctrl_z(self):
        assert matches_key("\x1a", "ctrl+z") is True

    def test_ctrl_c_mismatch(self):
        assert matches_key("\x03", "ctrl+a") is False

    def test_ctrl_d(self):
        assert matches_key("\x04", "ctrl+d") is True


# ---------------------------------------------------------------------------
# matches_key — alt sequences
# ---------------------------------------------------------------------------


class TestMatchesKeyAlt:
    """matches_key recognises alt+key via ESC prefix."""

    def test_alt_x(self):
        assert matches_key("\x1bx", "alt+x") is True

    def test_alt_f(self):
        assert matches_key("\x1bf", "alt+f") is True

    def test_alt_b(self):
        assert matches_key("\x1bb", "alt+b") is True

    def test_alt_mismatch(self):
        assert matches_key("\x1bx", "alt+y") is False


# ---------------------------------------------------------------------------
# matches_key — shift
# ---------------------------------------------------------------------------


class TestMatchesKeyShift:
    """matches_key recognises shift+letter as uppercase characters."""

    def test_shift_a(self):
        assert matches_key("A", "shift+a") is True

    def test_shift_z(self):
        assert matches_key("Z", "shift+z") is True

    def test_shift_symbol_produces_shifted_char(self):
        # shift+1 should match "!"
        assert matches_key("!", "shift+1") is True

    def test_shift_minus_underscore(self):
        assert matches_key("_", "shift+-") is True

    def test_shift_equals_plus(self):
        assert matches_key("+", "shift+=") is True


# ---------------------------------------------------------------------------
# matches_key — special keys
# ---------------------------------------------------------------------------


class TestMatchesKeySpecial:
    """matches_key handles escape, enter, tab, space, backspace."""

    def test_escape(self):
        assert matches_key("\x1b", "escape") is True

    def test_esc_alias(self):
        assert matches_key("\x1b", "esc") is True

    def test_enter_cr(self):
        assert matches_key("\r", "enter") is True

    def test_enter_lf(self):
        assert matches_key("\n", "enter") is True

    def test_tab(self):
        assert matches_key("\t", "tab") is True

    def test_space(self):
        assert matches_key(" ", "space") is True

    def test_backspace_del(self):
        assert matches_key("\x7f", "backspace") is True

    def test_backspace_bs(self):
        assert matches_key("\x08", "backspace") is True

    def test_shift_tab(self):
        assert matches_key("\x1b[Z", "shift+tab") is True

    def test_alt_enter(self):
        assert matches_key("\x1b\r", "alt+enter") is True

    def test_alt_backspace(self):
        assert matches_key("\x1b\x7f", "alt+backspace") is True

    def test_ctrl_space(self):
        assert matches_key("\x00", "ctrl+space") is True

    def test_alt_escape(self):
        assert matches_key("\x1b\x1b", "alt+escape") is True

    def test_alt_space(self):
        assert matches_key("\x1b ", "alt+space") is True

    def test_alt_tab(self):
        assert matches_key("\x1b\t", "alt+tab") is True


# ---------------------------------------------------------------------------
# matches_key — arrow keys (legacy sequences)
# ---------------------------------------------------------------------------


class TestMatchesKeyArrows:
    """matches_key handles legacy arrow key escape sequences."""

    def test_up(self):
        assert matches_key("\x1b[A", "up") is True

    def test_down(self):
        assert matches_key("\x1b[B", "down") is True

    def test_right(self):
        assert matches_key("\x1b[C", "right") is True

    def test_left(self):
        assert matches_key("\x1b[D", "left") is True

    def test_up_alternate(self):
        assert matches_key("\x1bOA", "up") is True

    def test_arrow_mismatch(self):
        assert matches_key("\x1b[A", "down") is False

    def test_ctrl_up(self):
        assert matches_key("\x1b[1;5A", "ctrl+up") is True

    def test_ctrl_down(self):
        assert matches_key("\x1b[1;5B", "ctrl+down") is True

    def test_shift_right(self):
        assert matches_key("\x1b[1;2C", "shift+right") is True

    def test_alt_left(self):
        assert matches_key("\x1b[1;3D", "alt+left") is True

    def test_ctrl_shift_up(self):
        assert matches_key("\x1b[1;6A", "ctrl+shift+up") is True


# ---------------------------------------------------------------------------
# matches_key — function keys
# ---------------------------------------------------------------------------


class TestMatchesKeyFunctionKeys:
    """matches_key handles function key sequences."""

    def test_f1(self):
        assert matches_key("\x1bOP", "f1") is True

    def test_f2(self):
        assert matches_key("\x1bOQ", "f2") is True

    def test_f5(self):
        assert matches_key("\x1b[15~", "f5") is True

    def test_f12(self):
        assert matches_key("\x1b[24~", "f12") is True

    def test_shift_f1(self):
        assert matches_key("\x1b[1;2P", "shift+f1") is True

    def test_ctrl_f5(self):
        assert matches_key("\x1b[15;5~", "ctrl+f5") is True

    def test_alt_f3(self):
        assert matches_key("\x1b[1;3R", "alt+f3") is True

    def test_f_key_mismatch(self):
        assert matches_key("\x1bOP", "f2") is False


# ---------------------------------------------------------------------------
# matches_key — navigation keys
# ---------------------------------------------------------------------------


class TestMatchesKeyNavigation:
    """matches_key handles home, end, insert, delete, page up/down."""

    def test_home(self):
        assert matches_key("\x1b[H", "home") is True

    def test_home_alternate(self):
        assert matches_key("\x1bOH", "home") is True

    def test_home_alternate_2(self):
        assert matches_key("\x1b[1~", "home") is True

    def test_end(self):
        assert matches_key("\x1b[F", "end") is True

    def test_end_alternate(self):
        assert matches_key("\x1b[4~", "end") is True

    def test_insert(self):
        assert matches_key("\x1b[2~", "insert") is True

    def test_delete(self):
        assert matches_key("\x1b[3~", "delete") is True

    def test_page_up(self):
        assert matches_key("\x1b[5~", "pageUp") is True

    def test_page_down(self):
        assert matches_key("\x1b[6~", "pageDown") is True

    def test_clear(self):
        assert matches_key("\x1b[E", "clear") is True

    def test_shift_delete(self):
        assert matches_key("\x1b[3;2~", "shift+delete") is True

    def test_ctrl_home(self):
        assert matches_key("\x1b[1;5H", "ctrl+home") is True

    def test_ctrl_end(self):
        assert matches_key("\x1b[1;5F", "ctrl+end") is True


# ---------------------------------------------------------------------------
# matches_key — combined modifiers for characters
# ---------------------------------------------------------------------------


class TestMatchesKeyCombinedModifiers:
    """matches_key handles ctrl+alt, shift+alt, etc. for character keys."""

    def test_ctrl_alt_letter(self):
        # ctrl+alt+a = ESC + ctrl-a
        assert matches_key("\x1b\x01", "ctrl+alt+a") is True

    def test_shift_alt_letter(self):
        # shift+alt+a = ESC + A
        assert matches_key("\x1bA", "shift+alt+a") is True

    def test_shift_alt_symbol(self):
        # shift+alt+1 = ESC + !
        assert matches_key("\x1b!", "shift+alt+1") is True


# ---------------------------------------------------------------------------
# matches_key — kitty protocol sequences
# ---------------------------------------------------------------------------


class TestMatchesKeyKitty:
    """matches_key handles kitty keyboard protocol CSI u sequences."""

    def test_kitty_plain_a(self):
        # CSI 97 u  (codepoint for 'a', no modifiers)
        assert matches_key("\x1b[97u", "a") is True

    def test_kitty_ctrl_a(self):
        # CSI 97;5u  (modifier 5 means ctrl; 5-1=4, ctrl bit=4)
        assert matches_key("\x1b[97;5u", "ctrl+a") is True

    def test_kitty_alt_a(self):
        # CSI 97;3u  (modifier 3 means alt; 3-1=2, alt bit=2)
        assert matches_key("\x1b[97;3u", "alt+a") is True

    def test_kitty_shift_a(self):
        # CSI 97;2u  (modifier 2 means shift; 2-1=1, shift bit=1)
        assert matches_key("\x1b[97;2u", "shift+a") is True

    def test_kitty_ctrl_shift_a(self):
        # modifier 6 = ctrl+shift; 6-1=5, ctrl=4, shift=1
        assert matches_key("\x1b[97;6u", "ctrl+shift+a") is True

    def test_kitty_enter(self):
        # codepoint 13 = enter
        assert matches_key("\x1b[13u", "enter") is True

    def test_kitty_escape(self):
        # codepoint 27 = escape
        assert matches_key("\x1b[27u", "escape") is True

    def test_kitty_space(self):
        # codepoint 32 = space
        assert matches_key("\x1b[32u", "space") is True

    def test_kitty_tab(self):
        # codepoint 9 = tab
        assert matches_key("\x1b[9u", "tab") is True

    def test_kitty_backspace(self):
        # codepoint 127 = backspace
        assert matches_key("\x1b[127u", "backspace") is True

    def test_kitty_arrow_up(self):
        # Arrow up with modifier: CSI 1;1A (modifier 1 = no modifier, 1-1=0)
        assert matches_key("\x1b[1;1A", "up") is True

    def test_kitty_ctrl_arrow_right(self):
        # CSI 1;5C (modifier 5 = ctrl)
        assert matches_key("\x1b[1;5C", "ctrl+right") is True

    def test_kitty_f1_via_csi_u(self):
        # F1 codepoint in kitty: 57364
        assert matches_key("\x1b[57364u", "f1") is True

    def test_kitty_f5_via_functional(self):
        # F5 as functional: CSI 15;1~
        assert matches_key("\x1b[15;1~", "f5") is True

    def test_kitty_ctrl_f5(self):
        # CSI 15;5~
        assert matches_key("\x1b[15;5~", "ctrl+f5") is True


# ---------------------------------------------------------------------------
# matches_key — modifyOtherKeys format
# ---------------------------------------------------------------------------


class TestMatchesKeyModifyOtherKeys:
    """matches_key handles CSI 27;modifier;keycode~ format."""

    def test_modify_other_keys_ctrl_a(self):
        # CSI 27;5;97~ means ctrl+a
        assert matches_modify_other_keys("\x1b[27;5;97~", ord("a"), MODIFIERS["ctrl"]) is True

    def test_modify_other_keys_no_match(self):
        assert matches_modify_other_keys("plain text", ord("a"), 0) is False

    def test_modify_other_keys_wrong_keycode(self):
        assert matches_modify_other_keys("\x1b[27;5;98~", ord("a"), MODIFIERS["ctrl"]) is False


# ---------------------------------------------------------------------------
# parse_kitty_sequence
# ---------------------------------------------------------------------------


class TestParseKittySequence:
    """parse_kitty_sequence extracts components from kitty protocol data."""

    def test_csi_u_simple(self):
        result = parse_kitty_sequence("\x1b[97u")
        assert result is not None
        assert result.codepoint == 97
        assert result.shifted_key is None
        assert result.base_layout_key is None
        assert result.modifier == 1
        assert result.event_type == 1

    def test_csi_u_with_modifier(self):
        result = parse_kitty_sequence("\x1b[97;5u")
        assert result is not None
        assert result.codepoint == 97
        assert result.modifier == 5
        assert result.event_type == 1

    def test_csi_u_with_event_type(self):
        result = parse_kitty_sequence("\x1b[97;5:3u")
        assert result is not None
        assert result.codepoint == 97
        assert result.modifier == 5
        assert result.event_type == 3  # release

    def test_csi_u_with_shifted_key(self):
        result = parse_kitty_sequence("\x1b[97:65u")
        assert result is not None
        assert result.codepoint == 97
        assert result.shifted_key == 65
        assert result.base_layout_key is None

    def test_csi_u_with_all_fields(self):
        result = parse_kitty_sequence("\x1b[97:65:98;5:2u")
        assert result is not None
        assert result.codepoint == 97
        assert result.shifted_key == 65
        assert result.base_layout_key == 98
        assert result.modifier == 5
        assert result.event_type == 2

    def test_arrow_key(self):
        result = parse_kitty_sequence("\x1b[1;1A")
        assert result is not None
        assert result.codepoint == -1  # up
        assert result.modifier == 1

    def test_arrow_with_event_type(self):
        result = parse_kitty_sequence("\x1b[1;5:3A")
        assert result is not None
        assert result.codepoint == -1  # up
        assert result.modifier == 5
        assert result.event_type == 3

    def test_functional_key(self):
        result = parse_kitty_sequence("\x1b[3;5~")
        assert result is not None
        assert result.codepoint == -10  # delete
        assert result.modifier == 5

    def test_home_end(self):
        result = parse_kitty_sequence("\x1b[1;5H")
        assert result is not None
        assert result.codepoint == -14  # home
        assert result.modifier == 5

    def test_f1_f4(self):
        result = parse_kitty_sequence("\x1b[1;5P")
        assert result is not None
        assert result.codepoint == -16  # f1
        assert result.modifier == 5

    def test_no_match(self):
        assert parse_kitty_sequence("plain") is None

    def test_no_match_empty(self):
        assert parse_kitty_sequence("") is None


# ---------------------------------------------------------------------------
# is_key_release / is_key_repeat
# ---------------------------------------------------------------------------


class TestIsKeyRelease:
    """is_key_release detects release event patterns in kitty protocol."""

    def test_csi_u_release(self):
        # event_type 3 in CSI u format: \x1b[97;5:3u
        assert is_key_release("\x1b[97;5:3u") is True

    def test_functional_release(self):
        # event_type 3 in functional format: \x1b[3;5:3~
        assert is_key_release("\x1b[3;5:3~") is True

    def test_arrow_release(self):
        # event_type 3 in arrow format: \x1b[1;5:3A
        assert is_key_release("\x1b[1;5:3A") is True

    def test_home_release(self):
        assert is_key_release("\x1b[1;5:3H") is True

    def test_f_key_release(self):
        assert is_key_release("\x1b[1;5:3P") is True

    def test_press_is_not_release(self):
        assert is_key_release("\x1b[97;5:1u") is False

    def test_repeat_is_not_release(self):
        assert is_key_release("\x1b[97;5:2u") is False

    def test_plain_char_is_not_release(self):
        assert is_key_release("a") is False

    def test_bracketed_paste_not_treated_as_release(self):
        # Even if data looks like release, bracketed paste overrides
        assert is_key_release("\x1b[200~\x1b[97;5:3u") is False


class TestIsKeyRepeat:
    """is_key_repeat detects repeat event patterns in kitty protocol."""

    def test_csi_u_repeat(self):
        assert is_key_repeat("\x1b[97;5:2u") is True

    def test_functional_repeat(self):
        assert is_key_repeat("\x1b[3;5:2~") is True

    def test_arrow_repeat(self):
        assert is_key_repeat("\x1b[1;5:2A") is True

    def test_home_repeat(self):
        assert is_key_repeat("\x1b[1;5:2H") is True

    def test_f_key_repeat(self):
        assert is_key_repeat("\x1b[1;5:2P") is True

    def test_press_is_not_repeat(self):
        assert is_key_repeat("\x1b[97;5:1u") is False

    def test_release_is_not_repeat(self):
        assert is_key_repeat("\x1b[97;5:3u") is False

    def test_plain_char_is_not_repeat(self):
        assert is_key_repeat("a") is False

    def test_bracketed_paste_not_treated_as_repeat(self):
        assert is_key_repeat("\x1b[200~\x1b[97;5:2u") is False


# ---------------------------------------------------------------------------
# parse_key — returns human-readable key name from raw input
# ---------------------------------------------------------------------------


class TestParseKeySimple:
    """parse_key converts raw single-byte input to key names."""

    def test_printable_char(self):
        assert parse_key("a") == "a"

    def test_digit(self):
        assert parse_key("5") == "5"

    def test_uppercase_preserved(self):
        # Uppercase 'A' is a printable character
        assert parse_key("A") == "A"

    def test_escape(self):
        assert parse_key("\x1b") == "escape"

    def test_enter_cr(self):
        assert parse_key("\r") == "enter"

    def test_enter_lf(self):
        assert parse_key("\n") == "enter"

    def test_tab(self):
        assert parse_key("\t") == "tab"

    def test_space(self):
        assert parse_key(" ") == "space"

    def test_backspace_del(self):
        assert parse_key("\x7f") == "backspace"

    def test_backspace_bs(self):
        assert parse_key("\x08") == "backspace"

    def test_ctrl_space(self):
        assert parse_key("\x00") == "ctrl+space"

    def test_empty_returns_none(self):
        assert parse_key("") is None


class TestParseKeyCtrl:
    """parse_key correctly identifies ctrl+letter sequences."""

    def test_ctrl_a(self):
        assert parse_key("\x01") == "ctrl+a"

    def test_ctrl_c(self):
        assert parse_key("\x03") == "ctrl+c"

    def test_ctrl_z(self):
        assert parse_key("\x1a") == "ctrl+z"

    def test_ctrl_d(self):
        assert parse_key("\x04") == "ctrl+d"


class TestParseKeyAlt:
    """parse_key correctly identifies alt+key sequences."""

    def test_alt_letter(self):
        assert parse_key("\x1bx") == "alt+x"

    def test_alt_digit(self):
        assert parse_key("\x1b5") == "alt+5"

    def test_alt_escape(self):
        assert parse_key("\x1b\x1b") == "alt+escape"

    def test_alt_enter(self):
        assert parse_key("\x1b\r") == "alt+enter"

    def test_alt_tab(self):
        assert parse_key("\x1b\t") == "alt+tab"

    def test_alt_space(self):
        assert parse_key("\x1b ") == "alt+space"

    def test_alt_backspace(self):
        assert parse_key("\x1b\x7f") == "alt+backspace"

    def test_ctrl_alt_letter(self):
        # ESC + ctrl-a byte
        assert parse_key("\x1b\x01") == "ctrl+alt+a"

    def test_shift_alt_letter(self):
        # ESC + uppercase letter
        assert parse_key("\x1bA") == "shift+alt+a"


class TestParseKeyLegacySequences:
    """parse_key handles legacy escape sequences for special keys."""

    def test_arrow_up(self):
        assert parse_key("\x1b[A") == "up"

    def test_arrow_down(self):
        assert parse_key("\x1b[B") == "down"

    def test_arrow_right(self):
        assert parse_key("\x1b[C") == "right"

    def test_arrow_left(self):
        assert parse_key("\x1b[D") == "left"

    def test_home(self):
        assert parse_key("\x1b[H") == "home"

    def test_end(self):
        assert parse_key("\x1b[F") == "end"

    def test_insert(self):
        assert parse_key("\x1b[2~") == "insert"

    def test_delete(self):
        assert parse_key("\x1b[3~") == "delete"

    def test_page_up(self):
        assert parse_key("\x1b[5~") == "pageUp"

    def test_page_down(self):
        assert parse_key("\x1b[6~") == "pageDown"

    def test_f1(self):
        assert parse_key("\x1bOP") == "f1"

    def test_f5(self):
        assert parse_key("\x1b[15~") == "f5"

    def test_f12(self):
        assert parse_key("\x1b[24~") == "f12"

    def test_shift_tab(self):
        assert parse_key("\x1b[Z") == "shift+tab"

    def test_ctrl_up(self):
        assert parse_key("\x1b[1;5A") == "ctrl+up"

    def test_shift_right(self):
        assert parse_key("\x1b[1;2C") == "shift+right"

    def test_alt_left(self):
        assert parse_key("\x1b[1;3D") == "alt+left"

    def test_ctrl_shift_down(self):
        assert parse_key("\x1b[1;6B") == "ctrl+shift+down"

    def test_ctrl_alt_up(self):
        assert parse_key("\x1b[1;7A") == "ctrl+alt+up"

    def test_shift_alt_right(self):
        assert parse_key("\x1b[1;4C") == "shift+alt+right"

    def test_ctrl_shift_alt_left(self):
        assert parse_key("\x1b[1;8D") == "ctrl+shift+alt+left"

    def test_shift_f1(self):
        assert parse_key("\x1b[1;2P") == "shift+f1"

    def test_ctrl_f5(self):
        assert parse_key("\x1b[15;5~") == "ctrl+f5"


class TestParseKeyKitty:
    """parse_key handles kitty keyboard protocol sequences."""

    def test_kitty_plain_a(self):
        assert parse_key("\x1b[97u") == "a"

    def test_kitty_ctrl_a(self):
        assert parse_key("\x1b[97;5u") == "ctrl+a"

    def test_kitty_alt_x(self):
        assert parse_key("\x1b[120;3u") == "alt+x"

    def test_kitty_shift_a(self):
        assert parse_key("\x1b[97;2u") == "shift+a"

    def test_kitty_enter(self):
        assert parse_key("\x1b[13u") == "enter"

    def test_kitty_escape(self):
        assert parse_key("\x1b[27u") == "escape"

    def test_kitty_tab(self):
        assert parse_key("\x1b[9u") == "tab"

    def test_kitty_space(self):
        assert parse_key("\x1b[32u") == "space"

    def test_kitty_backspace(self):
        assert parse_key("\x1b[127u") == "backspace"

    def test_kitty_f1(self):
        assert parse_key("\x1b[57364u") == "f1"

    def test_kitty_f12(self):
        assert parse_key("\x1b[57375u") == "f12"

    def test_kitty_kp_enter(self):
        # kp_enter codepoint 57414 should be reported as "enter"
        assert parse_key("\x1b[57414u") == "enter"

    def test_kitty_ctrl_shift_alt(self):
        # modifier 8 = ctrl+shift+alt; 8-1=7, ctrl=4,shift=1,alt=2
        assert parse_key("\x1b[97;8u") == "ctrl+shift+alt+a"


class TestParseKeyModifyOtherKeys:
    """parse_key handles modifyOtherKeys CSI 27;modifier;keycode~ format."""

    def test_modify_other_keys_ctrl_a(self):
        assert parse_key("\x1b[27;5;97~") == "ctrl+a"

    def test_modify_other_keys_alt_x(self):
        assert parse_key("\x1b[27;3;120~") == "alt+x"


# ---------------------------------------------------------------------------
# Constant lookups — sanity checks
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify key constant dictionaries are internally consistent."""

    def test_symbol_keys_set(self):
        assert "`" in SYMBOL_KEYS
        assert "-" in SYMBOL_KEYS
        assert "/" in SYMBOL_KEYS

    def test_modifiers_dict(self):
        assert MODIFIERS["shift"] == 1
        assert MODIFIERS["alt"] == 2
        assert MODIFIERS["ctrl"] == 4

    def test_codepoints_dict(self):
        assert CODEPOINTS["escape"] == 27
        assert CODEPOINTS["enter"] == 13
        assert CODEPOINTS["space"] == 32
        assert CODEPOINTS["tab"] == 9
        assert CODEPOINTS["backspace"] == 127

    def test_shifted_key_map_round_trip(self):
        for base, shifted in SHIFTED_KEY_MAP.items():
            assert UNSHIFTED_KEY_MAP[shifted] == base

    def test_legacy_key_sequences_has_arrows(self):
        assert "\x1b[A" in LEGACY_KEY_SEQUENCES
        assert LEGACY_KEY_SEQUENCES["\x1b[A"] == "up"

    def test_legacy_shift_sequences_has_entries(self):
        assert len(LEGACY_SHIFT_SEQUENCES) > 0

    def test_legacy_ctrl_sequences_has_entries(self):
        assert len(LEGACY_CTRL_SEQUENCES) > 0

    def test_legacy_alt_sequences_has_entries(self):
        assert len(LEGACY_ALT_SEQUENCES) > 0


# ---------------------------------------------------------------------------
# matches_kitty_sequence — low-level helper
# ---------------------------------------------------------------------------


class TestMatchesKittySequence:
    """matches_kitty_sequence checks a parsed kitty sequence against expectations."""

    def test_match_plain(self):
        assert matches_kitty_sequence("\x1b[97u", 97, 0) is True

    def test_match_with_ctrl(self):
        # modifier 5, actual_mod = (5-1) & ~LOCK_MASK = 4 = ctrl
        assert matches_kitty_sequence("\x1b[97;5u", 97, 4) is True

    def test_no_match_wrong_codepoint(self):
        assert matches_kitty_sequence("\x1b[97u", 98, 0) is False

    def test_no_match_wrong_modifier(self):
        assert matches_kitty_sequence("\x1b[97;5u", 97, 2) is False

    def test_non_kitty_data(self):
        assert matches_kitty_sequence("a", 97, 0) is False
