"""Box component - a container that applies padding and background to all children."""

from __future__ import annotations

from typing import Any, Callable

from pi.tui.utils import apply_background_to_line, visible_width


class Box:
    """Box component - a container that applies padding and background to all children."""

    def __init__(
        self,
        padding_x: int = 1,
        padding_y: int = 1,
        bg_fn: Callable[[str], str] | None = None,
    ) -> None:
        self.children: list[Any] = []  # list[Component]
        self._padding_x = padding_x
        self._padding_y = padding_y
        self._bg_fn = bg_fn

        # Cache
        self._cache: dict[str, Any] | None = None

    def add_child(self, component: Any) -> None:
        self.children.append(component)
        self._invalidate_cache()

    def remove_child(self, component: Any) -> None:
        try:
            index = self.children.index(component)
            self.children.pop(index)
            self._invalidate_cache()
        except ValueError:
            pass

    def clear(self) -> None:
        self.children = []
        self._invalidate_cache()

    def set_bg_fn(self, bg_fn: Callable[[str], str] | None = None) -> None:
        self._bg_fn = bg_fn

    def _invalidate_cache(self) -> None:
        self._cache = None

    def _match_cache(
        self, width: int, child_lines: list[str], bg_sample: str | None
    ) -> bool:
        cache = self._cache
        if cache is None:
            return False
        return (
            cache["width"] == width
            and cache["bg_sample"] == bg_sample
            and len(cache["child_lines"]) == len(child_lines)
            and all(a == b for a, b in zip(cache["child_lines"], child_lines))
        )

    def invalidate(self) -> None:
        self._invalidate_cache()
        for child in self.children:
            if hasattr(child, "invalidate"):
                child.invalidate()

    def render(self, width: int) -> list[str]:
        if not self.children:
            return []

        content_width = max(1, width - self._padding_x * 2)
        left_pad = " " * self._padding_x

        # Render all children
        child_lines: list[str] = []
        for child in self.children:
            lines = child.render(content_width)
            for line in lines:
                child_lines.append(left_pad + line)

        if not child_lines:
            return []

        # Check if bg_fn output changed by sampling
        bg_sample = self._bg_fn("test") if self._bg_fn else None

        # Check cache validity
        if self._match_cache(width, child_lines, bg_sample):
            return self._cache["lines"]  # type: ignore

        # Apply background and padding
        result: list[str] = []

        # Top padding
        for _ in range(self._padding_y):
            result.append(self._apply_bg("", width))

        # Content
        for line in child_lines:
            result.append(self._apply_bg(line, width))

        # Bottom padding
        for _ in range(self._padding_y):
            result.append(self._apply_bg("", width))

        # Update cache
        self._cache = {
            "child_lines": child_lines,
            "width": width,
            "bg_sample": bg_sample,
            "lines": result,
        }

        return result

    def _apply_bg(self, line: str, width: int) -> str:
        vis_len = visible_width(line)
        pad_needed = max(0, width - vis_len)
        padded = line + " " * pad_needed

        if self._bg_fn:
            return apply_background_to_line(padded, width, self._bg_fn)
        return padded
