"""Spacer component that renders empty lines."""

from __future__ import annotations


class Spacer:
    """Spacer component that renders empty lines."""

    def __init__(self, lines: int = 1) -> None:
        self._lines = lines

    def set_lines(self, lines: int) -> None:
        self._lines = lines

    def invalidate(self) -> None:
        pass

    def render(self, width: int) -> list[str]:
        return [""] * self._lines
