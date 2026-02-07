"""Tests for pi.tui.keybindings — editor keybindings manager."""

from __future__ import annotations

import pytest

from pi.tui.keybindings import (
    DEFAULT_EDITOR_KEYBINDINGS,
    EditorKeybindingsManager,
    get_editor_keybindings,
    set_editor_keybindings,
)
from pi.tui.keys import matches_key


# ---------------------------------------------------------------------------
# DEFAULT_EDITOR_KEYBINDINGS constant
# ---------------------------------------------------------------------------


class TestDefaultEditorKeybindings:
    """DEFAULT_EDITOR_KEYBINDINGS has the expected shape and actions."""

    def test_is_dict(self):
        assert isinstance(DEFAULT_EDITOR_KEYBINDINGS, dict)

    def test_has_cursor_movement_actions(self):
        for action in [
            "cursorUp", "cursorDown", "cursorLeft", "cursorRight",
            "cursorWordLeft", "cursorWordRight",
            "cursorLineStart", "cursorLineEnd",
        ]:
            assert action in DEFAULT_EDITOR_KEYBINDINGS, f"Missing action: {action}"

    def test_has_deletion_actions(self):
        for action in [
            "deleteCharBackward", "deleteCharForward",
            "deleteWordBackward", "deleteWordForward",
            "deleteToLineStart", "deleteToLineEnd",
        ]:
            assert action in DEFAULT_EDITOR_KEYBINDINGS, f"Missing action: {action}"

    def test_has_text_input_actions(self):
        assert "newLine" in DEFAULT_EDITOR_KEYBINDINGS
        assert "submit" in DEFAULT_EDITOR_KEYBINDINGS
        assert "tab" in DEFAULT_EDITOR_KEYBINDINGS

    def test_has_selection_actions(self):
        for action in [
            "selectUp", "selectDown", "selectPageUp", "selectPageDown",
            "selectConfirm", "selectCancel",
        ]:
            assert action in DEFAULT_EDITOR_KEYBINDINGS, f"Missing action: {action}"

    def test_has_clipboard_and_kill_ring(self):
        assert "copy" in DEFAULT_EDITOR_KEYBINDINGS
        assert "yank" in DEFAULT_EDITOR_KEYBINDINGS
        assert "yankPop" in DEFAULT_EDITOR_KEYBINDINGS

    def test_has_undo(self):
        assert "undo" in DEFAULT_EDITOR_KEYBINDINGS

    def test_has_session_actions(self):
        for action in [
            "toggleSessionPath", "toggleSessionSort",
            "renameSession", "deleteSession", "deleteSessionNoninvasive",
        ]:
            assert action in DEFAULT_EDITOR_KEYBINDINGS, f"Missing action: {action}"

    def test_has_expand_tools(self):
        assert "expandTools" in DEFAULT_EDITOR_KEYBINDINGS

    def test_cursor_up_is_up(self):
        assert DEFAULT_EDITOR_KEYBINDINGS["cursorUp"] == "up"

    def test_cursor_left_is_list(self):
        keys = DEFAULT_EDITOR_KEYBINDINGS["cursorLeft"]
        assert isinstance(keys, list)
        assert "left" in keys
        assert "ctrl+b" in keys

    def test_submit_is_enter(self):
        assert DEFAULT_EDITOR_KEYBINDINGS["submit"] == "enter"

    def test_new_line_is_shift_enter(self):
        assert DEFAULT_EDITOR_KEYBINDINGS["newLine"] == "shift+enter"

    def test_select_cancel_includes_escape_and_ctrl_c(self):
        keys = DEFAULT_EDITOR_KEYBINDINGS["selectCancel"]
        assert isinstance(keys, list)
        assert "escape" in keys
        assert "ctrl+c" in keys

    def test_copy_is_ctrl_c(self):
        assert DEFAULT_EDITOR_KEYBINDINGS["copy"] == "ctrl+c"

    def test_yank_is_ctrl_y(self):
        assert DEFAULT_EDITOR_KEYBINDINGS["yank"] == "ctrl+y"

    def test_yank_pop_is_alt_y(self):
        assert DEFAULT_EDITOR_KEYBINDINGS["yankPop"] == "alt+y"


# ---------------------------------------------------------------------------
# EditorKeybindingsManager — construction
# ---------------------------------------------------------------------------


class TestEditorKeybindingsManagerConstruction:
    """EditorKeybindingsManager loads defaults and applies overrides."""

    def test_default_construction(self):
        mgr = EditorKeybindingsManager()
        # Should have all default actions
        for action in DEFAULT_EDITOR_KEYBINDINGS:
            keys = mgr.get_keys(action)
            assert len(keys) > 0, f"Action {action} should have at least one key"

    def test_construction_with_none_config(self):
        mgr = EditorKeybindingsManager(config=None)
        assert mgr.get_keys("submit") == ["enter"]

    def test_construction_with_empty_config(self):
        mgr = EditorKeybindingsManager(config={})
        assert mgr.get_keys("submit") == ["enter"]

    def test_construction_with_override(self):
        mgr = EditorKeybindingsManager(config={"submit": "ctrl+enter"})
        assert mgr.get_keys("submit") == ["ctrl+enter"]

    def test_override_preserves_other_defaults(self):
        mgr = EditorKeybindingsManager(config={"submit": "ctrl+enter"})
        # cursorUp should still have its default
        assert mgr.get_keys("cursorUp") == ["up"]

    def test_override_with_list(self):
        mgr = EditorKeybindingsManager(config={"submit": ["enter", "ctrl+enter"]})
        assert mgr.get_keys("submit") == ["enter", "ctrl+enter"]


# ---------------------------------------------------------------------------
# EditorKeybindingsManager.get_keys
# ---------------------------------------------------------------------------


class TestEditorKeybindingsManagerGetKeys:
    """get_keys returns the list of KeyIds bound to an action."""

    def test_single_key_action(self):
        mgr = EditorKeybindingsManager()
        keys = mgr.get_keys("cursorUp")
        assert keys == ["up"]

    def test_multi_key_action(self):
        mgr = EditorKeybindingsManager()
        keys = mgr.get_keys("cursorLeft")
        assert "left" in keys
        assert "ctrl+b" in keys

    def test_unknown_action_returns_empty(self):
        mgr = EditorKeybindingsManager()
        # Using a string that is not a valid action -- get_keys returns []
        keys = mgr.get_keys("nonExistentAction")  # type: ignore[arg-type]
        assert keys == []


# ---------------------------------------------------------------------------
# EditorKeybindingsManager.matches
# ---------------------------------------------------------------------------


class TestEditorKeybindingsManagerMatches:
    """matches checks raw terminal input against an action's keybindings."""

    def test_submit_matches_enter(self):
        mgr = EditorKeybindingsManager()
        assert mgr.matches("\r", "submit") is True

    def test_submit_matches_newline(self):
        mgr = EditorKeybindingsManager()
        assert mgr.matches("\n", "submit") is True

    def test_submit_does_not_match_space(self):
        mgr = EditorKeybindingsManager()
        assert mgr.matches(" ", "submit") is False

    def test_cursor_up_matches_arrow(self):
        mgr = EditorKeybindingsManager()
        assert mgr.matches("\x1b[A", "cursorUp") is True

    def test_cursor_down_matches_arrow(self):
        mgr = EditorKeybindingsManager()
        assert mgr.matches("\x1b[B", "cursorDown") is True

    def test_cursor_left_matches_arrow(self):
        mgr = EditorKeybindingsManager()
        assert mgr.matches("\x1b[D", "cursorLeft") is True

    def test_cursor_left_matches_ctrl_b(self):
        mgr = EditorKeybindingsManager()
        assert mgr.matches("\x02", "cursorLeft") is True

    def test_cursor_right_matches_arrow(self):
        mgr = EditorKeybindingsManager()
        assert mgr.matches("\x1b[C", "cursorRight") is True

    def test_cursor_right_matches_ctrl_f(self):
        mgr = EditorKeybindingsManager()
        assert mgr.matches("\x06", "cursorRight") is True

    def test_cursor_word_left_matches_alt_left(self):
        mgr = EditorKeybindingsManager()
        assert mgr.matches("\x1b[1;3D", "cursorWordLeft") is True

    def test_cursor_word_left_matches_alt_b(self):
        mgr = EditorKeybindingsManager()
        assert mgr.matches("\x1bb", "cursorWordLeft") is True

    def test_cursor_word_right_matches_alt_right(self):
        mgr = EditorKeybindingsManager()
        assert mgr.matches("\x1b[1;3C", "cursorWordRight") is True

    def test_cursor_word_right_matches_alt_f(self):
        mgr = EditorKeybindingsManager()
        assert mgr.matches("\x1bf", "cursorWordRight") is True

    def test_cursor_line_start_matches_ctrl_a(self):
        mgr = EditorKeybindingsManager()
        assert mgr.matches("\x01", "cursorLineStart") is True

    def test_cursor_line_start_matches_home(self):
        mgr = EditorKeybindingsManager()
        assert mgr.matches("\x1b[H", "cursorLineStart") is True

    def test_cursor_line_end_matches_ctrl_e(self):
        mgr = EditorKeybindingsManager()
        assert mgr.matches("\x05", "cursorLineEnd") is True

    def test_cursor_line_end_matches_end(self):
        mgr = EditorKeybindingsManager()
        assert mgr.matches("\x1b[F", "cursorLineEnd") is True

    def test_delete_char_backward_matches_backspace(self):
        mgr = EditorKeybindingsManager()
        assert mgr.matches("\x7f", "deleteCharBackward") is True

    def test_delete_char_forward_matches_ctrl_d(self):
        mgr = EditorKeybindingsManager()
        assert mgr.matches("\x04", "deleteCharForward") is True

    def test_delete_char_forward_matches_delete(self):
        mgr = EditorKeybindingsManager()
        assert mgr.matches("\x1b[3~", "deleteCharForward") is True

    def test_delete_word_backward_matches_ctrl_w(self):
        mgr = EditorKeybindingsManager()
        assert mgr.matches("\x17", "deleteWordBackward") is True

    def test_delete_word_backward_matches_alt_backspace(self):
        mgr = EditorKeybindingsManager()
        assert mgr.matches("\x1b\x7f", "deleteWordBackward") is True

    def test_delete_word_forward_matches_alt_d(self):
        mgr = EditorKeybindingsManager()
        assert mgr.matches("\x1bd", "deleteWordForward") is True

    def test_delete_to_line_start_matches_ctrl_u(self):
        mgr = EditorKeybindingsManager()
        assert mgr.matches("\x15", "deleteToLineStart") is True

    def test_delete_to_line_end_matches_ctrl_k(self):
        mgr = EditorKeybindingsManager()
        assert mgr.matches("\x0b", "deleteToLineEnd") is True

    def test_tab_matches_tab(self):
        mgr = EditorKeybindingsManager()
        assert mgr.matches("\t", "tab") is True

    def test_select_cancel_matches_escape(self):
        mgr = EditorKeybindingsManager()
        assert mgr.matches("\x1b", "selectCancel") is True

    def test_select_cancel_matches_ctrl_c(self):
        mgr = EditorKeybindingsManager()
        assert mgr.matches("\x03", "selectCancel") is True

    def test_copy_matches_ctrl_c(self):
        mgr = EditorKeybindingsManager()
        assert mgr.matches("\x03", "copy") is True

    def test_yank_matches_ctrl_y(self):
        mgr = EditorKeybindingsManager()
        assert mgr.matches("\x19", "yank") is True

    def test_yank_pop_matches_alt_y(self):
        mgr = EditorKeybindingsManager()
        assert mgr.matches("\x1by", "yankPop") is True

    def test_expand_tools_matches_ctrl_o(self):
        mgr = EditorKeybindingsManager()
        assert mgr.matches("\x0f", "expandTools") is True

    def test_page_up_matches_page_up(self):
        mgr = EditorKeybindingsManager()
        assert mgr.matches("\x1b[5~", "pageUp") is True

    def test_page_down_matches_page_down(self):
        mgr = EditorKeybindingsManager()
        assert mgr.matches("\x1b[6~", "pageDown") is True

    def test_unknown_action_does_not_match(self):
        mgr = EditorKeybindingsManager()
        assert mgr.matches("\r", "nonExistentAction") is False  # type: ignore[arg-type]

    def test_matches_after_override(self):
        mgr = EditorKeybindingsManager(config={"submit": "ctrl+enter"})
        # Default "enter" should no longer match
        assert mgr.matches("\r", "submit") is False


# ---------------------------------------------------------------------------
# EditorKeybindingsManager.set_config
# ---------------------------------------------------------------------------


class TestEditorKeybindingsManagerSetConfig:
    """set_config replaces the active keybinding configuration."""

    def test_set_config_overrides_action(self):
        mgr = EditorKeybindingsManager()
        assert mgr.matches("\r", "submit") is True

        mgr.set_config({"submit": "space"})
        assert mgr.matches("\r", "submit") is False
        assert mgr.matches(" ", "submit") is True

    def test_set_config_preserves_defaults_for_unspecified_actions(self):
        mgr = EditorKeybindingsManager()
        mgr.set_config({"submit": "space"})
        # cursorUp should still work with default
        assert mgr.matches("\x1b[A", "cursorUp") is True

    def test_set_config_with_list(self):
        mgr = EditorKeybindingsManager()
        mgr.set_config({"submit": ["enter", "space"]})
        assert mgr.matches("\r", "submit") is True
        assert mgr.matches(" ", "submit") is True


# ---------------------------------------------------------------------------
# Global singleton — get_editor_keybindings / set_editor_keybindings
# ---------------------------------------------------------------------------


class TestGlobalKeybindings:
    """get_editor_keybindings / set_editor_keybindings manage a global instance."""

    def teardown_method(self):
        # Reset global to ensure test isolation
        import pi.tui.keybindings as kb_module
        kb_module._global_editor_keybindings = None

    def test_get_returns_manager(self):
        mgr = get_editor_keybindings()
        assert isinstance(mgr, EditorKeybindingsManager)

    def test_get_returns_same_instance(self):
        mgr1 = get_editor_keybindings()
        mgr2 = get_editor_keybindings()
        assert mgr1 is mgr2

    def test_set_replaces_global(self):
        original = get_editor_keybindings()
        custom = EditorKeybindingsManager(config={"submit": "space"})
        set_editor_keybindings(custom)
        current = get_editor_keybindings()
        assert current is custom
        assert current is not original

    def test_set_then_get_uses_new_bindings(self):
        custom = EditorKeybindingsManager(config={"submit": "space"})
        set_editor_keybindings(custom)
        mgr = get_editor_keybindings()
        assert mgr.matches(" ", "submit") is True
        assert mgr.matches("\r", "submit") is False

    def test_reset_to_none_creates_new_default(self):
        import pi.tui.keybindings as kb_module

        mgr1 = get_editor_keybindings()
        kb_module._global_editor_keybindings = None
        mgr2 = get_editor_keybindings()
        assert mgr2 is not mgr1
        # New instance should still have defaults
        assert mgr2.matches("\r", "submit") is True


# ---------------------------------------------------------------------------
# Integration: matches_key is correctly delegated by manager.matches
# ---------------------------------------------------------------------------


class TestKeybindingsIntegration:
    """End-to-end: raw terminal bytes routed through the manager to actions."""

    def test_kitty_enter_matches_submit(self):
        mgr = EditorKeybindingsManager()
        # Kitty protocol enter: CSI 13 u
        assert mgr.matches("\x1b[13u", "submit") is True

    def test_kitty_up_matches_cursor_up(self):
        mgr = EditorKeybindingsManager()
        assert mgr.matches("\x1b[1;1A", "cursorUp") is True

    def test_kitty_ctrl_a_matches_cursor_line_start(self):
        mgr = EditorKeybindingsManager()
        assert mgr.matches("\x1b[97;5u", "cursorLineStart") is True

    def test_alt_b_matches_cursor_word_left(self):
        mgr = EditorKeybindingsManager()
        assert mgr.matches("\x1bb", "cursorWordLeft") is True

    def test_custom_binding_kitty(self):
        mgr = EditorKeybindingsManager(config={"submit": "ctrl+m"})
        # ctrl+m = \r in legacy, but also codepoint 109 with ctrl in kitty
        assert mgr.matches("\r", "submit") is True
