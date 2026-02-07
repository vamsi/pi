"""Terminal abstraction for raw-mode stdin/stdout interaction.

Provides a ``Terminal`` protocol and a concrete ``ProcessTerminal``
implementation that manages raw mode, bracketed paste, the Kitty keyboard
protocol, cursor visibility, and screen clearing via ANSI escape sequences.

This is a faithful Python port of the TypeScript ``terminal.ts`` module.
"""

from __future__ import annotations

import asyncio
import os
import re
import signal
import sys
import termios
import tty
from typing import Callable, Protocol

from pi.tui.keys import set_kitty_protocol_active
from pi.tui.stdin_buffer import StdinBuffer

# ---------------------------------------------------------------------------
# ANSI escape constants
# ---------------------------------------------------------------------------

_BRACKETED_PASTE_ENABLE = "\x1b[?2004h"
_BRACKETED_PASTE_DISABLE = "\x1b[?2004l"
_BRACKETED_PASTE_START = "\x1b[200~"
_BRACKETED_PASTE_END = "\x1b[201~"

_KITTY_QUERY = "\x1b[?u"
_KITTY_ENABLE = "\x1b[>1u"
_KITTY_DISABLE = "\x1b[<u"

_KITTY_RESPONSE_RE = re.compile(r"^\x1b\[\?(\d+)u$")

_HIDE_CURSOR = "\x1b[?25l"
_SHOW_CURSOR = "\x1b[?25h"
_CLEAR_LINE = "\x1b[2K\r"
_CLEAR_FROM_CURSOR = "\x1b[0J"
_CLEAR_SCREEN = "\x1b[2J\x1b[H"
_CURSOR_UP_FMT = "\x1b[{}A"
_CURSOR_DOWN_FMT = "\x1b[{}B"

_SET_TITLE_FMT = "\x1b]0;{}\x07"


# ---------------------------------------------------------------------------
# Terminal protocol
# ---------------------------------------------------------------------------


class Terminal(Protocol):
    """Interface for terminal I/O operations."""

    def start(
        self,
        on_input: Callable[[str], None],
        on_resize: Callable[[], None],
    ) -> None: ...

    def stop(self) -> None: ...

    async def drain_input(
        self,
        max_ms: int = 1000,
        idle_ms: int = 50,
    ) -> None: ...

    def write(self, data: str) -> None: ...

    @property
    def columns(self) -> int: ...

    @property
    def rows(self) -> int: ...

    @property
    def kitty_protocol_active(self) -> bool: ...

    def move_by(self, lines: int) -> None: ...

    def hide_cursor(self) -> None: ...

    def show_cursor(self) -> None: ...

    def clear_line(self) -> None: ...

    def clear_from_cursor(self) -> None: ...

    def clear_screen(self) -> None: ...

    def set_title(self, title: str) -> None: ...


# ---------------------------------------------------------------------------
# ProcessTerminal implementation
# ---------------------------------------------------------------------------


class ProcessTerminal:
    """Concrete terminal implementation backed by ``sys.stdin``/``sys.stdout``.

    Manages raw mode via :mod:`tty` and :mod:`termios`, the Kitty keyboard
    protocol, bracketed paste mode, and SIGWINCH-based resize detection.
    """

    def __init__(self) -> None:
        self._was_raw: bool = False
        self._input_handler: Callable[[str], None] | None = None
        self._resize_handler: Callable[[], None] | None = None
        self._kitty_protocol_active: bool = False
        self._stdin_buffer: StdinBuffer | None = None
        self._stdin_reader_active: bool = False
        self._original_termios: list | None = None
        self._prev_sigwinch_handler: signal.Handlers | None = None
        self._write_log_path: str = os.environ.get("PI_TUI_WRITE_LOG", "")

    # -- properties ---------------------------------------------------------

    @property
    def kitty_protocol_active(self) -> bool:
        return self._kitty_protocol_active

    @property
    def columns(self) -> int:
        try:
            return os.get_terminal_size(sys.stdout.fileno()).columns
        except (ValueError, OSError):
            return 80

    @property
    def rows(self) -> int:
        try:
            return os.get_terminal_size(sys.stdout.fileno()).lines
        except (ValueError, OSError):
            return 24

    # -- start / stop -------------------------------------------------------

    def start(
        self,
        on_input: Callable[[str], None],
        on_resize: Callable[[], None],
    ) -> None:
        """Enable raw mode, bracketed paste, and begin reading stdin."""
        self._input_handler = on_input
        self._resize_handler = on_resize

        fd = sys.stdin.fileno()

        # Save previous terminal state
        self._original_termios = termios.tcgetattr(fd)
        self._was_raw = _is_raw_mode(fd)

        # Enable raw mode
        tty.setraw(fd)

        # Enable bracketed paste
        self._raw_write(_BRACKETED_PASTE_ENABLE)

        # Set up SIGWINCH handler for resize events
        self._prev_sigwinch_handler = signal.getsignal(signal.SIGWINCH)
        signal.signal(signal.SIGWINCH, self._on_sigwinch)

        # Query and enable the Kitty keyboard protocol
        self._query_and_enable_kitty_protocol()

    def stop(self) -> None:
        """Restore terminal state and clean up all handlers."""
        # Disable bracketed paste
        self._raw_write(_BRACKETED_PASTE_DISABLE)

        # Disable Kitty protocol if it was activated
        if self._kitty_protocol_active:
            self._raw_write(_KITTY_DISABLE)
            self._kitty_protocol_active = False
            set_kitty_protocol_active(False)

        # Clean up StdinBuffer
        if self._stdin_buffer is not None:
            self._stdin_buffer.destroy()
            self._stdin_buffer = None

        # Remove stdin reader
        self._remove_stdin_reader()

        # Restore SIGWINCH handler
        if self._prev_sigwinch_handler is not None:
            signal.signal(signal.SIGWINCH, self._prev_sigwinch_handler)
            self._prev_sigwinch_handler = None

        # Restore terminal attributes
        fd = sys.stdin.fileno()
        if self._original_termios is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, self._original_termios)
            self._original_termios = None

        self._input_handler = None
        self._resize_handler = None

    # -- drainInput ---------------------------------------------------------

    async def drain_input(
        self,
        max_ms: int = 1000,
        idle_ms: int = 50,
    ) -> None:
        """Drain pending input, disabling Kitty protocol first.

        Waits until no more input arrives for *idle_ms* milliseconds, with an
        overall timeout of *max_ms* milliseconds.
        """
        # Disable Kitty protocol while draining so responses don't confuse us
        if self._kitty_protocol_active:
            self._raw_write(_KITTY_DISABLE)

        max_seconds = max_ms / 1000.0
        idle_seconds = idle_ms / 1000.0

        loop = asyncio.get_event_loop()
        done_event = asyncio.Event()
        idle_handle: asyncio.TimerHandle | None = None
        timeout_handle: asyncio.TimerHandle | None = None

        def _on_drain_data() -> None:
            nonlocal idle_handle
            # Data arrived -- reset idle timer
            try:
                os.read(sys.stdin.fileno(), 4096)
            except OSError:
                pass
            if idle_handle is not None:
                idle_handle.cancel()
            idle_handle = loop.call_later(idle_seconds, _finish)

        def _finish() -> None:
            nonlocal timeout_handle, idle_handle
            if timeout_handle is not None:
                timeout_handle.cancel()
                timeout_handle = None
            if idle_handle is not None:
                idle_handle.cancel()
                idle_handle = None
            try:
                loop.remove_reader(sys.stdin.fileno())
            except Exception:
                pass
            done_event.set()

        # Install a temporary reader that consumes and discards data
        try:
            loop.add_reader(sys.stdin.fileno(), _on_drain_data)
        except Exception:
            done_event.set()

        # Set overall timeout
        timeout_handle = loop.call_later(max_seconds, _finish)

        # Kick off the first idle timer
        idle_handle = loop.call_later(idle_seconds, _finish)

        await done_event.wait()

    # -- write --------------------------------------------------------------

    def write(self, data: str) -> None:
        """Write data to stdout and optionally to the write log."""
        self._raw_write(data)

        if self._write_log_path:
            try:
                with open(self._write_log_path, "a") as f:
                    f.write(data)
            except OSError:
                pass

    # -- cursor / screen manipulation --------------------------------------

    def move_by(self, lines: int) -> None:
        """Move the cursor up (negative) or down (positive) by *lines*."""
        if lines < 0:
            self._raw_write(_CURSOR_UP_FMT.format(-lines))
        elif lines > 0:
            self._raw_write(_CURSOR_DOWN_FMT.format(lines))

    def hide_cursor(self) -> None:
        self._raw_write(_HIDE_CURSOR)

    def show_cursor(self) -> None:
        self._raw_write(_SHOW_CURSOR)

    def clear_line(self) -> None:
        self._raw_write(_CLEAR_LINE)

    def clear_from_cursor(self) -> None:
        self._raw_write(_CLEAR_FROM_CURSOR)

    def clear_screen(self) -> None:
        self._raw_write(_CLEAR_SCREEN)

    def set_title(self, title: str) -> None:
        self._raw_write(_SET_TITLE_FMT.format(title))

    # -- private: Kitty protocol setup -------------------------------------

    def _setup_stdin_buffer(self) -> None:
        """Create the :class:`StdinBuffer` and wire up handlers."""
        self._stdin_buffer = StdinBuffer(timeout=0.01)

        def _on_buffer_data(data: str) -> None:
            # Check whether the data is a Kitty keyboard protocol response
            match = _KITTY_RESPONSE_RE.match(data)
            if match:
                self._kitty_protocol_active = True
                set_kitty_protocol_active(True)
                # Now send the enable sequence for Kitty flags mode 1
                self._raw_write(_KITTY_ENABLE)
                return

            # Forward to the user-provided input handler
            if self._input_handler is not None:
                self._input_handler(data)

        def _on_buffer_paste(data: str) -> None:
            # Re-wrap with bracketed paste markers and forward
            if self._input_handler is not None:
                self._input_handler(
                    _BRACKETED_PASTE_START + data + _BRACKETED_PASTE_END
                )

        self._stdin_buffer.on_data(_on_buffer_data)
        self._stdin_buffer.on_paste(_on_buffer_paste)

    def _query_and_enable_kitty_protocol(self) -> None:
        """Set up the stdin buffer, start reading, and query Kitty support."""
        self._setup_stdin_buffer()
        self._start_stdin_reader()
        # Send the Kitty query sequence
        self._raw_write(_KITTY_QUERY)

    # -- private: stdin reading --------------------------------------------

    def _start_stdin_reader(self) -> None:
        """Register an asyncio reader on stdin to feed the stdin buffer."""
        if self._stdin_reader_active:
            return
        try:
            loop = asyncio.get_event_loop()
            loop.add_reader(sys.stdin.fileno(), self._on_stdin_readable)
            self._stdin_reader_active = True
        except RuntimeError:
            # No running event loop -- cannot register reader
            pass

    def _remove_stdin_reader(self) -> None:
        """Remove the asyncio reader from stdin."""
        if not self._stdin_reader_active:
            return
        try:
            loop = asyncio.get_event_loop()
            loop.remove_reader(sys.stdin.fileno())
        except (RuntimeError, ValueError):
            pass
        self._stdin_reader_active = False

    def _on_stdin_readable(self) -> None:
        """Callback invoked by the event loop when stdin has data."""
        try:
            raw = os.read(sys.stdin.fileno(), 4096)
        except OSError:
            return

        if not raw:
            return

        data = raw.decode("utf-8", errors="replace")

        if self._stdin_buffer is not None:
            self._stdin_buffer.process(data)
        elif self._input_handler is not None:
            self._input_handler(data)

    # -- private: SIGWINCH -------------------------------------------------

    def _on_sigwinch(
        self,
        signum: int,
        frame: object,
    ) -> None:
        """Handle terminal resize signals."""
        if self._resize_handler is not None:
            self._resize_handler()

    # -- private: raw write ------------------------------------------------

    def _raw_write(self, data: str) -> None:
        """Write directly to stdout, bypassing buffering."""
        try:
            sys.stdout.write(data)
            sys.stdout.flush()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_raw_mode(fd: int) -> bool:
    """Heuristic check for whether the terminal fd is already in raw mode.

    Raw mode is characterised by the absence of ICANON and ECHO in the
    local-mode flags.
    """
    try:
        attrs = termios.tcgetattr(fd)
        lflag = attrs[3]  # c_lflag
        return not bool(lflag & (termios.ICANON | termios.ECHO))
    except termios.error:
        return False
