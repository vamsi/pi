"""Test theme helpers -- identity-styling themes for use in component tests.

Provides ``PlainTheme`` and ``plain_select_list_theme`` that apply no ANSI
formatting, making rendered output easy to assert against in tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


# ---------------------------------------------------------------------------
# Plain (identity) theme helpers
# ---------------------------------------------------------------------------


def _identity(text: str) -> str:
    """Return *text* unchanged -- an identity styling function."""
    return text


@dataclass
class PlainSelectListTheme:
    """A ``SelectListTheme``-compatible object with identity styling."""

    selected_prefix: Callable[[str], str] = _identity
    selected_text: Callable[[str], str] = _identity
    description: Callable[[str], str] = _identity
    scroll_info: Callable[[str], str] = _identity
    no_match: Callable[[str], str] = _identity


@dataclass
class PlainEditorTheme:
    """An ``EditorTheme``-compatible object with identity styling."""

    _border_color: Callable[[str], str] = _identity
    _select_list: PlainSelectListTheme | None = None

    def __post_init__(self) -> None:
        if self._select_list is None:
            self._select_list = PlainSelectListTheme()

    @property
    def border_color(self) -> Callable[[str], str]:
        return self._border_color

    @property
    def select_list(self) -> PlainSelectListTheme:
        assert self._select_list is not None
        return self._select_list


# ---------------------------------------------------------------------------
# Tests -- verify the theme objects are well-formed
# ---------------------------------------------------------------------------


class TestPlainSelectListTheme:
    """Verify that PlainSelectListTheme passes through text unchanged."""

    def test_selected_prefix_identity(self) -> None:
        theme = PlainSelectListTheme()
        assert theme.selected_prefix(">>") == ">>"

    def test_selected_text_identity(self) -> None:
        theme = PlainSelectListTheme()
        assert theme.selected_text("item") == "item"

    def test_description_identity(self) -> None:
        theme = PlainSelectListTheme()
        assert theme.description("desc") == "desc"

    def test_scroll_info_identity(self) -> None:
        theme = PlainSelectListTheme()
        assert theme.scroll_info("1/10") == "1/10"

    def test_no_match_identity(self) -> None:
        theme = PlainSelectListTheme()
        assert theme.no_match("nothing found") == "nothing found"


class TestPlainEditorTheme:
    """Verify that PlainEditorTheme passes through text unchanged."""

    def test_border_color_identity(self) -> None:
        theme = PlainEditorTheme()
        assert theme.border_color("---") == "---"

    def test_select_list_is_plain(self) -> None:
        theme = PlainEditorTheme()
        assert isinstance(theme.select_list, PlainSelectListTheme)

    def test_select_list_identity(self) -> None:
        theme = PlainEditorTheme()
        assert theme.select_list.selected_text("foo") == "foo"


class TestIdentityFunction:
    """Verify the identity helper itself."""

    def test_returns_input_unchanged(self) -> None:
        assert _identity("hello") == "hello"

    def test_empty_string(self) -> None:
        assert _identity("") == ""

    def test_ansi_codes_preserved(self) -> None:
        ansi = "\x1b[31mred\x1b[0m"
        assert _identity(ansi) == ansi
