"""Loader that can be cancelled with Escape."""

from __future__ import annotations

from typing import Callable

from pi.tui.components.loader import Loader
from pi.tui.keybindings import get_editor_keybindings


class CancellableLoader(Loader):
    """Loader that can be cancelled with Escape.

    Extends Loader with cancellation support for async operations.

    Example::

        loader = CancellableLoader(tui, cyan, dim, "Working...")
        loader.on_abort = lambda: done(None)
    """

    def __init__(self, *args, **kwargs) -> None:  # type: ignore
        super().__init__(*args, **kwargs)
        self._aborted = False
        self.on_abort: Callable[[], None] | None = None

    @property
    def aborted(self) -> bool:
        return self._aborted

    def handle_input(self, data: str) -> None:
        kb = get_editor_keybindings()
        if kb.matches(data, "selectCancel"):
            self._aborted = True
            if self.on_abort:
                self.on_abort()

    def dispose(self) -> None:
        self.stop()
