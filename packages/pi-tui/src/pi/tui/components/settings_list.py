"""SettingsList component for settings management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pi.tui.fuzzy import fuzzy_filter
from pi.tui.keybindings import get_editor_keybindings
from pi.tui.utils import truncate_to_width, visible_width, wrap_text_with_ansi


@dataclass
class SettingItem:
    """A setting entry."""

    id: str
    label: str
    description: str | None = None
    current_value: str = ""
    values: list[str] | None = None
    submenu: Callable[[str, Callable[[str | None], None]], Any] | None = None


@dataclass
class SettingsListTheme:
    label: Callable[[str, bool], str]
    value: Callable[[str, bool], str]
    description: Callable[[str], str]
    cursor: str
    hint: Callable[[str], str]


@dataclass
class SettingsListOptions:
    enable_search: bool = False


class SettingsList:
    """Settings list component."""

    def __init__(
        self,
        items: list[SettingItem],
        max_visible: int,
        theme: SettingsListTheme,
        on_change: Callable[[str, str], None],
        on_cancel: Callable[[], None],
        options: SettingsListOptions | None = None,
    ) -> None:
        self._items = items
        self._filtered_items = list(items)
        self._theme = theme
        self._selected_index = 0
        self._max_visible = max_visible
        self._on_change = on_change
        self._on_cancel = on_cancel
        self._search_enabled = (options or SettingsListOptions()).enable_search

        self._search_input: Any | None = None
        if self._search_enabled:
            from pi.tui.components.input import Input

            self._search_input = Input()

        # Submenu state
        self._submenu_component: Any | None = None
        self._submenu_item_index: int | None = None

    def update_value(self, id: str, new_value: str) -> None:
        """Update an item's current_value."""
        for item in self._items:
            if item.id == id:
                item.current_value = new_value
                break

    def invalidate(self) -> None:
        if self._submenu_component and hasattr(self._submenu_component, "invalidate"):
            self._submenu_component.invalidate()

    def render(self, width: int) -> list[str]:
        if self._submenu_component:
            return self._submenu_component.render(width)
        return self._render_main_list(width)

    def _render_main_list(self, width: int) -> list[str]:
        lines: list[str] = []

        if self._search_enabled and self._search_input:
            lines.extend(self._search_input.render(width))
            lines.append("")

        if not self._items:
            lines.append(self._theme.hint("  No settings available"))
            if self._search_enabled:
                self._add_hint_line(lines, width)
            return lines

        display_items = self._filtered_items if self._search_enabled else self._items
        if not display_items:
            lines.append(
                truncate_to_width(self._theme.hint("  No matching settings"), width)
            )
            self._add_hint_line(lines, width)
            return lines

        # Calculate visible range
        start_index = max(
            0,
            min(
                self._selected_index - self._max_visible // 2,
                len(display_items) - self._max_visible,
            ),
        )
        end_index = min(start_index + self._max_visible, len(display_items))

        # Max label width for alignment
        max_label_width = min(
            30,
            max(visible_width(item.label) for item in self._items) if self._items else 0,
        )

        for i in range(start_index, end_index):
            item = display_items[i]
            is_selected = i == self._selected_index
            prefix = self._theme.cursor if is_selected else "  "
            prefix_width = visible_width(prefix)

            label_padded = item.label + " " * max(
                0, max_label_width - visible_width(item.label)
            )
            label_text = self._theme.label(label_padded, is_selected)

            separator = "  "
            used_width = prefix_width + max_label_width + visible_width(separator)
            value_max_width = width - used_width - 2

            value_text = self._theme.value(
                truncate_to_width(item.current_value, value_max_width, ""),
                is_selected,
            )

            lines.append(
                truncate_to_width(prefix + label_text + separator + value_text, width)
            )

        # Scroll indicator
        if start_index > 0 or end_index < len(display_items):
            scroll_text = f"  ({self._selected_index + 1}/{len(display_items)})"
            lines.append(
                self._theme.hint(truncate_to_width(scroll_text, width - 2, ""))
            )

        # Description
        selected_item = (
            display_items[self._selected_index]
            if self._selected_index < len(display_items)
            else None
        )
        if selected_item and selected_item.description:
            lines.append("")
            wrapped_desc = wrap_text_with_ansi(selected_item.description, width - 4)
            for line in wrapped_desc:
                lines.append(self._theme.description(f"  {line}"))

        self._add_hint_line(lines, width)
        return lines

    def handle_input(self, data: str) -> None:
        if self._submenu_component:
            if hasattr(self._submenu_component, "handle_input"):
                self._submenu_component.handle_input(data)
            return

        kb = get_editor_keybindings()
        display_items = self._filtered_items if self._search_enabled else self._items

        if kb.matches(data, "selectUp"):
            if not display_items:
                return
            self._selected_index = (
                len(display_items) - 1
                if self._selected_index == 0
                else self._selected_index - 1
            )
        elif kb.matches(data, "selectDown"):
            if not display_items:
                return
            self._selected_index = (
                0
                if self._selected_index == len(display_items) - 1
                else self._selected_index + 1
            )
        elif kb.matches(data, "selectConfirm") or data == " ":
            self._activate_item()
        elif kb.matches(data, "selectCancel"):
            self._on_cancel()
        elif self._search_enabled and self._search_input:
            sanitized = data.replace(" ", "")
            if not sanitized:
                return
            self._search_input.handle_input(sanitized)
            self._apply_filter(self._search_input.get_value())

    def _activate_item(self) -> None:
        items = self._filtered_items if self._search_enabled else self._items
        if self._selected_index >= len(items):
            return
        item = items[self._selected_index]

        if item.submenu:
            self._submenu_item_index = self._selected_index

            def done(selected_value: str | None = None) -> None:
                if selected_value is not None:
                    item.current_value = selected_value
                    self._on_change(item.id, selected_value)
                self._close_submenu()

            self._submenu_component = item.submenu(item.current_value, done)
        elif item.values and len(item.values) > 0:
            try:
                current_index = item.values.index(item.current_value)
            except ValueError:
                current_index = -1
            next_index = (current_index + 1) % len(item.values)
            new_value = item.values[next_index]
            item.current_value = new_value
            self._on_change(item.id, new_value)

    def _close_submenu(self) -> None:
        self._submenu_component = None
        if self._submenu_item_index is not None:
            self._selected_index = self._submenu_item_index
            self._submenu_item_index = None

    def _apply_filter(self, query: str) -> None:
        self._filtered_items = fuzzy_filter(
            self._items, query, lambda item: item.label
        )
        self._selected_index = 0

    def _add_hint_line(self, lines: list[str], width: int) -> None:
        lines.append("")
        hint_text = (
            "  Type to search · Enter/Space to change · Esc to cancel"
            if self._search_enabled
            else "  Enter/Space to change · Esc to cancel"
        )
        lines.append(truncate_to_width(self._theme.hint(hint_text), width))
