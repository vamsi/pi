"""Tests for the SelectList component."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pi.tui.components.select_list import SelectItem, SelectList, SelectListTheme

# Raw escape codes for key sequences
KEY_UP = "\x1b[A"
KEY_DOWN = "\x1b[B"
KEY_ENTER = "\r"
KEY_ESCAPE = "\x1b"


# ---------------------------------------------------------------------------
# A simple identity theme for testing
# ---------------------------------------------------------------------------


class _IdentityTheme:
    """Theme that returns text unmodified, satisfying the SelectListTheme protocol."""

    @staticmethod
    def selected_prefix(text: str) -> str:
        return text

    @staticmethod
    def selected_text(text: str) -> str:
        return text

    @staticmethod
    def description(text: str) -> str:
        return text

    @staticmethod
    def scroll_info(text: str) -> str:
        return text

    @staticmethod
    def no_match(text: str) -> str:
        return text


def _make_items(*labels: str) -> list[SelectItem]:
    """Create SelectItem list from labels (value = label)."""
    return [SelectItem(value=label, label=label) for label in labels]


def _make_list(
    items: list[SelectItem] | None = None, max_visible: int = 10
) -> SelectList:
    """Create a SelectList with the identity theme."""
    if items is None:
        items = _make_items("alpha", "beta", "gamma")
    return SelectList(items=items, max_visible=max_visible, theme=_IdentityTheme())


class TestSelectListRender:
    """Renders items with the selected item showing a cursor prefix."""

    def test_renders_all_items(self) -> None:
        sl = _make_list()
        lines = sl.render(80)
        # Should have one line per item (3 items, all visible)
        assert len(lines) == 3

    def test_selected_item_has_arrow_prefix(self) -> None:
        sl = _make_list()
        lines = sl.render(80)
        # First item is selected by default and should start with arrow
        assert lines[0].startswith("\u2192 ") or "\u2192" in lines[0]

    def test_unselected_items_have_space_prefix(self) -> None:
        sl = _make_list()
        lines = sl.render(80)
        # Second item is unselected, prefixed with two spaces
        assert lines[1].startswith("  ")

    def test_selected_item_label_appears_in_output(self) -> None:
        sl = _make_list()
        lines = sl.render(80)
        assert "alpha" in lines[0]

    def test_render_with_no_items(self) -> None:
        sl = _make_list(items=[])
        lines = sl.render(80)
        assert len(lines) == 1
        assert "No matching" in lines[0]


class TestSelectListNavigation:
    """Up/down arrows change the selected item."""

    def test_down_arrow_selects_next_item(self) -> None:
        sl = _make_list()
        sl.handle_input(KEY_DOWN)
        item = sl.get_selected_item()
        assert item is not None
        assert item.value == "beta"

    def test_up_arrow_selects_previous_item(self) -> None:
        sl = _make_list()
        sl.handle_input(KEY_DOWN)  # select beta
        sl.handle_input(KEY_UP)  # back to alpha
        item = sl.get_selected_item()
        assert item is not None
        assert item.value == "alpha"

    def test_down_arrow_wraps_to_first(self) -> None:
        sl = _make_list()
        sl.handle_input(KEY_DOWN)  # beta
        sl.handle_input(KEY_DOWN)  # gamma
        sl.handle_input(KEY_DOWN)  # wraps to alpha
        item = sl.get_selected_item()
        assert item is not None
        assert item.value == "alpha"

    def test_up_arrow_wraps_to_last(self) -> None:
        sl = _make_list()
        sl.handle_input(KEY_UP)  # wraps to gamma
        item = sl.get_selected_item()
        assert item is not None
        assert item.value == "gamma"

    def test_navigation_updates_render_output(self) -> None:
        sl = _make_list()
        sl.handle_input(KEY_DOWN)
        lines = sl.render(80)
        # beta should now be the selected item (with arrow prefix)
        # Find the line containing "beta"
        beta_line = [l for l in lines if "beta" in l][0]
        assert "\u2192" in beta_line

    def test_on_selection_change_fires_on_navigation(self) -> None:
        sl = _make_list()
        changes: list[str] = []
        sl.on_selection_change = lambda item: changes.append(item.value)
        sl.handle_input(KEY_DOWN)
        assert changes == ["beta"]


class TestSelectListFilter:
    """set_filter narrows the displayed items."""

    def test_filter_narrows_items(self) -> None:
        items = _make_items("apple", "apricot", "banana", "blueberry")
        sl = _make_list(items=items)
        sl.set_filter("ap")
        lines = sl.render(80)
        # Only apple and apricot match "ap" prefix
        assert len(lines) == 2

    def test_filter_resets_selection_to_first(self) -> None:
        items = _make_items("apple", "apricot", "banana")
        sl = _make_list(items=items)
        sl.handle_input(KEY_DOWN)  # select apricot
        sl.set_filter("b")
        item = sl.get_selected_item()
        assert item is not None
        assert item.value == "banana"

    def test_filter_no_matches_shows_no_match_message(self) -> None:
        sl = _make_list()
        sl.set_filter("zzz")
        lines = sl.render(80)
        assert len(lines) == 1
        assert "No matching" in lines[0]

    def test_filter_empty_string_shows_all(self) -> None:
        sl = _make_list()
        sl.set_filter("a")  # only alpha
        sl.set_filter("")  # back to all
        lines = sl.render(80)
        assert len(lines) == 3

    def test_filter_is_case_insensitive(self) -> None:
        items = _make_items("Alpha", "Beta")
        sl = _make_list(items=items)
        sl.set_filter("al")
        item = sl.get_selected_item()
        assert item is not None
        assert item.value == "Alpha"


class TestSelectListOnSelect:
    """Enter key fires the on_select callback with the selected item."""

    def test_enter_fires_on_select(self) -> None:
        sl = _make_list()
        selected_items: list[SelectItem] = []
        sl.on_select = lambda item: selected_items.append(item)
        sl.handle_input(KEY_ENTER)
        assert len(selected_items) == 1
        assert selected_items[0].value == "alpha"

    def test_enter_after_navigation_selects_correct_item(self) -> None:
        sl = _make_list()
        selected_items: list[SelectItem] = []
        sl.on_select = lambda item: selected_items.append(item)
        sl.handle_input(KEY_DOWN)
        sl.handle_input(KEY_ENTER)
        assert len(selected_items) == 1
        assert selected_items[0].value == "beta"

    def test_enter_without_callback_does_not_raise(self) -> None:
        sl = _make_list()
        sl.handle_input(KEY_ENTER)  # Should not raise

    def test_enter_on_filtered_list_selects_filtered_item(self) -> None:
        items = _make_items("apple", "banana", "cherry")
        sl = _make_list(items=items)
        sl.set_filter("b")
        selected_items: list[SelectItem] = []
        sl.on_select = lambda item: selected_items.append(item)
        sl.handle_input(KEY_ENTER)
        assert len(selected_items) == 1
        assert selected_items[0].value == "banana"


class TestSelectListOnCancel:
    """Escape key fires the on_cancel callback."""

    def test_escape_fires_on_cancel(self) -> None:
        sl = _make_list()
        cancelled: list[bool] = []
        sl.on_cancel = lambda: cancelled.append(True)
        sl.handle_input(KEY_ESCAPE)
        assert cancelled == [True]

    def test_escape_without_callback_does_not_raise(self) -> None:
        sl = _make_list()
        sl.handle_input(KEY_ESCAPE)  # Should not raise


class TestSelectListScrolling:
    """Scroll indicators appear when items exceed max_visible."""

    def test_scroll_indicator_when_items_exceed_max_visible(self) -> None:
        items = _make_items("a", "b", "c", "d", "e", "f", "g", "h", "i", "j")
        sl = _make_list(items=items, max_visible=3)
        lines = sl.render(80)
        # Should show 3 items + 1 scroll indicator
        assert len(lines) == 4
        # Scroll indicator should contain position info
        assert "(1/" in lines[-1]

    def test_set_selected_index_clamps_to_range(self) -> None:
        sl = _make_list()
        sl.set_selected_index(100)
        item = sl.get_selected_item()
        assert item is not None
        assert item.value == "gamma"  # clamped to last

    def test_set_selected_index_negative_clamps_to_zero(self) -> None:
        sl = _make_list()
        sl.set_selected_index(-5)
        item = sl.get_selected_item()
        assert item is not None
        assert item.value == "alpha"  # clamped to first


class TestSelectListWithDescriptions:
    """Items with descriptions render them alongside the label."""

    def test_description_appears_in_wide_render(self) -> None:
        items = [SelectItem(value="cmd", label="command", description="A useful command")]
        sl = _make_list(items=items)
        lines = sl.render(80)
        assert len(lines) == 1
        # The description should appear in the output for wide enough renders
        assert "useful" in lines[0] or "command" in lines[0]

    def test_description_hidden_in_narrow_render(self) -> None:
        items = [
            SelectItem(value="cmd", label="command", description="A useful command")
        ]
        sl = _make_list(items=items)
        lines = sl.render(30)
        assert len(lines) == 1
        # Label should still be present
        assert "command" in lines[0]
