"""SelectList component with filtering and keyboard navigation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Protocol

from pi.tui.keybindings import get_editor_keybindings
from pi.tui.utils import truncate_to_width


def _normalize_to_single_line(text: str) -> str:
    return re.sub(r"[\r\n]+", " ", text).strip()


@dataclass
class SelectItem:
    value: str
    label: str
    description: str | None = None


class SelectListTheme(Protocol):
    selected_prefix: Callable[[str], str]
    selected_text: Callable[[str], str]
    description: Callable[[str], str]
    scroll_info: Callable[[str], str]
    no_match: Callable[[str], str]


class SelectList:
    """SelectList with filtering and keyboard navigation."""

    def __init__(
        self,
        items: list[SelectItem],
        max_visible: int,
        theme: SelectListTheme,
    ) -> None:
        self._items = items
        self._filtered_items = list(items)
        self._selected_index = 0
        self._max_visible = max_visible
        self._theme = theme

        self.on_select: Callable[[SelectItem], None] | None = None
        self.on_cancel: Callable[[], None] | None = None
        self.on_selection_change: Callable[[SelectItem], None] | None = None

    def set_filter(self, filter_text: str) -> None:
        self._filtered_items = [
            item
            for item in self._items
            if item.value.lower().startswith(filter_text.lower())
        ]
        self._selected_index = 0

    def set_selected_index(self, index: int) -> None:
        self._selected_index = max(
            0, min(index, len(self._filtered_items) - 1)
        )

    def invalidate(self) -> None:
        pass

    def render(self, width: int) -> list[str]:
        lines: list[str] = []

        if not self._filtered_items:
            lines.append(self._theme.no_match("  No matching commands"))
            return lines

        # Calculate visible range with scrolling
        start_index = max(
            0,
            min(
                self._selected_index - self._max_visible // 2,
                len(self._filtered_items) - self._max_visible,
            ),
        )
        end_index = min(
            start_index + self._max_visible, len(self._filtered_items)
        )

        # Render visible items
        for i in range(start_index, end_index):
            item = self._filtered_items[i]
            is_selected = i == self._selected_index
            desc_single = (
                _normalize_to_single_line(item.description)
                if item.description
                else None
            )

            line = ""
            if is_selected:
                prefix_width = 2  # "→ "
                display_value = item.label or item.value

                if desc_single and width > 40:
                    max_value_width = min(30, width - prefix_width - 4)
                    truncated_value = truncate_to_width(
                        display_value, max_value_width, ""
                    )
                    spacing = " " * max(1, 32 - len(truncated_value))

                    desc_start = prefix_width + len(truncated_value) + len(spacing)
                    remaining_width = width - desc_start - 2

                    if remaining_width > 10:
                        truncated_desc = truncate_to_width(
                            desc_single, remaining_width, ""
                        )
                        line = self._theme.selected_text(
                            f"→ {truncated_value}{spacing}{truncated_desc}"
                        )
                    else:
                        max_w = width - prefix_width - 2
                        line = self._theme.selected_text(
                            f"→ {truncate_to_width(display_value, max_w, '')}"
                        )
                else:
                    max_w = width - prefix_width - 2
                    line = self._theme.selected_text(
                        f"→ {truncate_to_width(display_value, max_w, '')}"
                    )
            else:
                display_value = item.label or item.value
                prefix = "  "

                if desc_single and width > 40:
                    max_value_width = min(30, width - len(prefix) - 4)
                    truncated_value = truncate_to_width(
                        display_value, max_value_width, ""
                    )
                    spacing = " " * max(1, 32 - len(truncated_value))

                    desc_start = len(prefix) + len(truncated_value) + len(spacing)
                    remaining_width = width - desc_start - 2

                    if remaining_width > 10:
                        truncated_desc = truncate_to_width(
                            desc_single, remaining_width, ""
                        )
                        desc_text = self._theme.description(
                            spacing + truncated_desc
                        )
                        line = prefix + truncated_value + desc_text
                    else:
                        max_w = width - len(prefix) - 2
                        line = prefix + truncate_to_width(
                            display_value, max_w, ""
                        )
                else:
                    max_w = width - len(prefix) - 2
                    line = prefix + truncate_to_width(
                        display_value, max_w, ""
                    )

            lines.append(line)

        # Add scroll indicators if needed
        if start_index > 0 or end_index < len(self._filtered_items):
            scroll_text = f"  ({self._selected_index + 1}/{len(self._filtered_items)})"
            lines.append(
                self._theme.scroll_info(
                    truncate_to_width(scroll_text, width - 2, "")
                )
            )

        return lines

    def handle_input(self, key_data: str) -> None:
        kb = get_editor_keybindings()

        if kb.matches(key_data, "selectUp"):
            self._selected_index = (
                len(self._filtered_items) - 1
                if self._selected_index == 0
                else self._selected_index - 1
            )
            self._notify_selection_change()
        elif kb.matches(key_data, "selectDown"):
            self._selected_index = (
                0
                if self._selected_index == len(self._filtered_items) - 1
                else self._selected_index + 1
            )
            self._notify_selection_change()
        elif kb.matches(key_data, "selectConfirm"):
            if self._selected_index < len(self._filtered_items) and self.on_select:
                self.on_select(self._filtered_items[self._selected_index])
        elif kb.matches(key_data, "selectCancel"):
            if self.on_cancel:
                self.on_cancel()

    def _notify_selection_change(self) -> None:
        if (
            self._selected_index < len(self._filtered_items)
            and self.on_selection_change
        ):
            self.on_selection_change(
                self._filtered_items[self._selected_index]
            )

    def get_selected_item(self) -> SelectItem | None:
        if self._selected_index < len(self._filtered_items):
            return self._filtered_items[self._selected_index]
        return None
