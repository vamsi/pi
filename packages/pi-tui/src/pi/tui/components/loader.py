"""Loader component that updates every 80ms with spinning animation."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Callable

from pi.tui.components.text import Text

if TYPE_CHECKING:
    from pi.tui.tui import TUI


class Loader(Text):
    """Loader component that updates every 80ms with spinning animation."""

    _frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(
        self,
        ui: TUI,
        spinner_color_fn: Callable[[str], str],
        message_color_fn: Callable[[str], str],
        message: str = "Loading...",
    ) -> None:
        super().__init__("", 1, 0)
        self._ui = ui
        self._spinner_color_fn = spinner_color_fn
        self._message_color_fn = message_color_fn
        self._message = message
        self._current_frame = 0
        self._timer_handle: asyncio.TimerHandle | None = None
        self.start()

    def render(self, width: int) -> list[str]:
        return ["", *super().render(width)]

    def start(self) -> None:
        self._update_display()
        self._schedule_next()

    def _schedule_next(self) -> None:
        try:
            loop = asyncio.get_event_loop()
            self._timer_handle = loop.call_later(0.08, self._tick)
        except RuntimeError:
            pass

    def _tick(self) -> None:
        self._current_frame = (self._current_frame + 1) % len(self._frames)
        self._update_display()
        self._schedule_next()

    def stop(self) -> None:
        if self._timer_handle is not None:
            self._timer_handle.cancel()
            self._timer_handle = None

    def set_message(self, message: str) -> None:
        self._message = message
        self._update_display()

    def _update_display(self) -> None:
        frame = self._frames[self._current_frame]
        self.set_text(
            f"{self._spinner_color_fn(frame)} {self._message_color_fn(self._message)}"
        )
        if self._ui is not None:
            self._ui.request_render()
