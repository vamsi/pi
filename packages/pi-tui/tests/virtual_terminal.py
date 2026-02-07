"""Virtual terminal for testing -- implements the Terminal protocol in-memory.

This module provides a ``VirtualTerminal`` class that satisfies the
``pi.tui.terminal.Terminal`` protocol without performing any real I/O.
All output is captured in a buffer for assertions.
"""

from __future__ import annotations

from typing import Callable


class VirtualTerminal:
    """In-memory terminal that records all writes for test inspection.

    Implements the ``Terminal`` protocol from ``pi.tui.terminal``.

    Parameters
    ----------
    rows:
        Number of terminal rows (height).
    columns:
        Number of terminal columns (width).
    """

    def __init__(self, rows: int = 24, columns: int = 80) -> None:
        self._rows = rows
        self._columns = columns
        self._buffer: list[str] = []
        self._started = False
        self._input_handler: Callable[[str], None] | None = None
        self._resize_handler: Callable[[], None] | None = None
        self._kitty_protocol_active = False
        self._cursor_visible = True
        self._title: str = ""

    # -- Terminal protocol: properties --------------------------------------

    @property
    def rows(self) -> int:
        return self._rows

    @rows.setter
    def rows(self, value: int) -> None:
        self._rows = value

    @property
    def columns(self) -> int:
        return self._columns

    @columns.setter
    def columns(self, value: int) -> None:
        self._columns = value

    @property
    def kitty_protocol_active(self) -> bool:
        return self._kitty_protocol_active

    # -- Terminal protocol: lifecycle ---------------------------------------

    def start(
        self,
        on_input: Callable[[str], None],
        on_resize: Callable[[], None],
    ) -> None:
        self._input_handler = on_input
        self._resize_handler = on_resize
        self._started = True

    def stop(self) -> None:
        self._started = False
        self._input_handler = None
        self._resize_handler = None

    async def drain_input(
        self,
        max_ms: int = 1000,
        idle_ms: int = 50,
    ) -> None:
        """No-op for virtual terminal -- there is no real input to drain."""
        pass

    # -- Terminal protocol: output ------------------------------------------

    def write(self, data: str) -> None:
        """Append *data* to the internal buffer."""
        self._buffer.append(data)

    def flush(self) -> None:
        """No-op -- the virtual terminal has no underlying stream to flush."""
        pass

    # -- Terminal protocol: cursor/screen manipulation ----------------------

    def move_by(self, lines: int) -> None:
        if lines < 0:
            self.write(f"\x1b[{-lines}A")
        elif lines > 0:
            self.write(f"\x1b[{lines}B")

    def hide_cursor(self) -> None:
        self._cursor_visible = False
        self.write("\x1b[?25l")

    def show_cursor(self) -> None:
        self._cursor_visible = True
        self.write("\x1b[?25h")

    def clear_line(self) -> None:
        self.write("\x1b[2K\r")

    def clear_from_cursor(self) -> None:
        self.write("\x1b[0J")

    def clear_screen(self) -> None:
        self.write("\x1b[2J\x1b[H")

    def set_title(self, title: str) -> None:
        self._title = title
        self.write(f"\x1b]0;{title}\x07")

    # -- Test helpers -------------------------------------------------------

    @property
    def output(self) -> str:
        """Return everything written to the terminal as a single string."""
        return "".join(self._buffer)

    @property
    def write_count(self) -> int:
        """Return the number of individual ``write`` calls made."""
        return len(self._buffer)

    def clear_buffer(self) -> None:
        """Discard all recorded output."""
        self._buffer.clear()

    def simulate_input(self, data: str) -> None:
        """Feed *data* into the registered input handler.

        Raises ``RuntimeError`` if no input handler has been registered
        (i.e. ``start`` was not called).
        """
        if self._input_handler is None:
            raise RuntimeError(
                "No input handler registered -- call start() first"
            )
        self._input_handler(data)

    def simulate_resize(self, rows: int | None = None, columns: int | None = None) -> None:
        """Change terminal dimensions and fire the resize callback.

        If *rows* or *columns* is ``None`` the corresponding dimension
        is left unchanged.
        """
        if rows is not None:
            self._rows = rows
        if columns is not None:
            self._columns = columns
        if self._resize_handler is not None:
            self._resize_handler()

    def on_resize(self) -> None:
        """Fire the resize handler (if registered) without changing dimensions."""
        if self._resize_handler is not None:
            self._resize_handler()
