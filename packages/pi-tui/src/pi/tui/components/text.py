"""Text component - displays multi-line text with word wrapping."""

from __future__ import annotations

from typing import Callable

from pi.tui.utils import apply_background_to_line, visible_width, wrap_text_with_ansi


class Text:
    """Text component - displays multi-line text with word wrapping."""

    def __init__(
        self,
        text: str = "",
        padding_x: int = 1,
        padding_y: int = 1,
        custom_bg_fn: Callable[[str], str] | None = None,
    ) -> None:
        self._text = text
        self._padding_x = padding_x
        self._padding_y = padding_y
        self._custom_bg_fn = custom_bg_fn

        # Cache
        self._cached_text: str | None = None
        self._cached_width: int | None = None
        self._cached_lines: list[str] | None = None

    def set_text(self, text: str) -> None:
        self._text = text
        self._cached_text = None
        self._cached_width = None
        self._cached_lines = None

    def set_custom_bg_fn(self, custom_bg_fn: Callable[[str], str] | None = None) -> None:
        self._custom_bg_fn = custom_bg_fn
        self._cached_text = None
        self._cached_width = None
        self._cached_lines = None

    def invalidate(self) -> None:
        self._cached_text = None
        self._cached_width = None
        self._cached_lines = None

    def render(self, width: int) -> list[str]:
        # Check cache
        if (
            self._cached_lines is not None
            and self._cached_text == self._text
            and self._cached_width == width
        ):
            return self._cached_lines

        # Don't render anything if there's no actual text
        if not self._text or self._text.strip() == "":
            result: list[str] = []
            self._cached_text = self._text
            self._cached_width = width
            self._cached_lines = result
            return result

        # Replace tabs with 3 spaces
        normalized_text = self._text.replace("\t", "   ")

        # Calculate content width (subtract left/right margins)
        content_width = max(1, width - self._padding_x * 2)

        # Wrap text (preserves ANSI codes but does NOT pad)
        wrapped_lines = wrap_text_with_ansi(normalized_text, content_width)

        # Add margins and background to each line
        left_margin = " " * self._padding_x
        right_margin = " " * self._padding_x
        content_lines: list[str] = []

        for line in wrapped_lines:
            line_with_margins = left_margin + line + right_margin

            if self._custom_bg_fn:
                content_lines.append(
                    apply_background_to_line(line_with_margins, width, self._custom_bg_fn)
                )
            else:
                visible_len = visible_width(line_with_margins)
                padding_needed = max(0, width - visible_len)
                content_lines.append(line_with_margins + " " * padding_needed)

        # Add top/bottom padding (empty lines)
        empty_line = " " * width
        empty_lines: list[str] = []
        for _ in range(self._padding_y):
            if self._custom_bg_fn:
                line = apply_background_to_line(empty_line, width, self._custom_bg_fn)
            else:
                line = empty_line
            empty_lines.append(line)

        result = [*empty_lines, *content_lines, *empty_lines]

        # Update cache
        self._cached_text = self._text
        self._cached_width = width
        self._cached_lines = result

        return result if len(result) > 0 else [""]
