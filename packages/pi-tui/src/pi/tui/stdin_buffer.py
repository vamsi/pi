"""StdinBuffer buffers input and emits complete sequences.

This is necessary because stdin data events can arrive in partial chunks,
especially for escape sequences like mouse events. Without buffering,
partial sequences can be misinterpreted as regular keypresses.
"""

from __future__ import annotations

import asyncio
from typing import Callable

ESC = "\x1b"
BRACKETED_PASTE_START = "\x1b[200~"
BRACKETED_PASTE_END = "\x1b[201~"


def _is_complete_sequence(data: str) -> str:
    """Check if a string is a complete escape sequence or needs more data.

    Returns 'complete', 'incomplete', or 'not-escape'.
    """
    if not data.startswith(ESC):
        return "not-escape"

    if len(data) == 1:
        return "incomplete"

    after_esc = data[1:]

    # CSI sequences: ESC [
    if after_esc.startswith("["):
        if after_esc.startswith("[M"):
            return "complete" if len(data) >= 6 else "incomplete"
        return _is_complete_csi_sequence(data)

    # OSC sequences: ESC ]
    if after_esc.startswith("]"):
        return _is_complete_osc_sequence(data)

    # DCS sequences: ESC P
    if after_esc.startswith("P"):
        return _is_complete_dcs_sequence(data)

    # APC sequences: ESC _
    if after_esc.startswith("_"):
        return _is_complete_apc_sequence(data)

    # SS3 sequences: ESC O
    if after_esc.startswith("O"):
        return "complete" if len(after_esc) >= 2 else "incomplete"

    # Meta key sequences: ESC followed by a single character
    if len(after_esc) == 1:
        return "complete"

    return "complete"


def _is_complete_csi_sequence(data: str) -> str:
    if not data.startswith(f"{ESC}["):
        return "complete"

    if len(data) < 3:
        return "incomplete"

    payload = data[2:]
    last_char = payload[-1]
    last_char_code = ord(last_char)

    if 0x40 <= last_char_code <= 0x7E:
        # Special handling for SGR mouse sequences
        if payload.startswith("<"):
            import re

            mouse_match = re.match(r"^<\d+;\d+;\d+[Mm]$", payload)
            if mouse_match:
                return "complete"
            if last_char in ("M", "m"):
                parts = payload[1:-1].split(";")
                if len(parts) == 3 and all(
                    p.isdigit() for p in parts
                ):
                    return "complete"
            return "incomplete"
        return "complete"

    return "incomplete"


def _is_complete_osc_sequence(data: str) -> str:
    if not data.startswith(f"{ESC}]"):
        return "complete"
    if data.endswith(f"{ESC}\\") or data.endswith("\x07"):
        return "complete"
    return "incomplete"


def _is_complete_dcs_sequence(data: str) -> str:
    if not data.startswith(f"{ESC}P"):
        return "complete"
    if data.endswith(f"{ESC}\\"):
        return "complete"
    return "incomplete"


def _is_complete_apc_sequence(data: str) -> str:
    if not data.startswith(f"{ESC}_"):
        return "complete"
    if data.endswith(f"{ESC}\\"):
        return "complete"
    return "incomplete"


def _extract_complete_sequences(buffer: str) -> tuple[list[str], str]:
    """Split accumulated buffer into complete sequences.

    Returns (sequences, remainder).
    """
    sequences: list[str] = []
    pos = 0

    while pos < len(buffer):
        remaining = buffer[pos:]

        if remaining.startswith(ESC):
            seq_end = 1
            while seq_end <= len(remaining):
                candidate = remaining[:seq_end]
                status = _is_complete_sequence(candidate)

                if status == "complete":
                    sequences.append(candidate)
                    pos += seq_end
                    break
                elif status == "incomplete":
                    seq_end += 1
                else:
                    sequences.append(candidate)
                    pos += seq_end
                    break
            else:
                # seq_end > len(remaining)
                if seq_end > len(remaining):
                    return sequences, remaining
        else:
            sequences.append(remaining[0])
            pos += 1

    return sequences, ""


class StdinBuffer:
    """Buffers stdin input and emits complete sequences.

    Handles partial escape sequences that arrive across multiple chunks.
    """

    def __init__(self, *, timeout: float = 0.01) -> None:
        self._buffer: str = ""
        self._timeout_handle: asyncio.TimerHandle | None = None
        self._timeout_ms: float = timeout
        self._paste_mode: bool = False
        self._paste_buffer: str = ""

        self._on_data: Callable[[str], None] | None = None
        self._on_paste: Callable[[str], None] | None = None

    def on_data(self, callback: Callable[[str], None]) -> None:
        """Set callback for complete sequences."""
        self._on_data = callback

    def on_paste(self, callback: Callable[[str], None]) -> None:
        """Set callback for paste content."""
        self._on_paste = callback

    def _emit_data(self, data: str) -> None:
        if self._on_data:
            self._on_data(data)

    def _emit_paste(self, data: str) -> None:
        if self._on_paste:
            self._on_paste(data)

    def process(self, data: str) -> None:
        """Feed input data into the buffer."""
        # Clear any pending timeout
        if self._timeout_handle is not None:
            self._timeout_handle.cancel()
            self._timeout_handle = None

        if len(data) == 0 and len(self._buffer) == 0:
            self._emit_data("")
            return

        self._buffer += data

        if self._paste_mode:
            self._paste_buffer += self._buffer
            self._buffer = ""

            end_index = self._paste_buffer.find(BRACKETED_PASTE_END)
            if end_index != -1:
                pasted_content = self._paste_buffer[:end_index]
                remaining = self._paste_buffer[
                    end_index + len(BRACKETED_PASTE_END) :
                ]

                self._paste_mode = False
                self._paste_buffer = ""

                self._emit_paste(pasted_content)

                if remaining:
                    self.process(remaining)
            return

        start_index = self._buffer.find(BRACKETED_PASTE_START)
        if start_index != -1:
            if start_index > 0:
                before_paste = self._buffer[:start_index]
                sequences, _ = _extract_complete_sequences(before_paste)
                for sequence in sequences:
                    self._emit_data(sequence)

            self._buffer = self._buffer[
                start_index + len(BRACKETED_PASTE_START) :
            ]
            self._paste_mode = True
            self._paste_buffer = self._buffer
            self._buffer = ""

            end_index = self._paste_buffer.find(BRACKETED_PASTE_END)
            if end_index != -1:
                pasted_content = self._paste_buffer[:end_index]
                remaining = self._paste_buffer[
                    end_index + len(BRACKETED_PASTE_END) :
                ]

                self._paste_mode = False
                self._paste_buffer = ""

                self._emit_paste(pasted_content)

                if remaining:
                    self.process(remaining)
            return

        sequences, remainder = _extract_complete_sequences(self._buffer)
        self._buffer = remainder

        for sequence in sequences:
            self._emit_data(sequence)

        if self._buffer:
            try:
                loop = asyncio.get_event_loop()
                self._timeout_handle = loop.call_later(
                    self._timeout_ms, self._flush_timeout
                )
            except RuntimeError:
                # No event loop - flush immediately
                flushed = self.flush()
                for sequence in flushed:
                    self._emit_data(sequence)

    def _flush_timeout(self) -> None:
        self._timeout_handle = None
        flushed = self.flush()
        for sequence in flushed:
            self._emit_data(sequence)

    def flush(self) -> list[str]:
        if self._timeout_handle is not None:
            self._timeout_handle.cancel()
            self._timeout_handle = None

        if not self._buffer:
            return []

        sequences = [self._buffer]
        self._buffer = ""
        return sequences

    def clear(self) -> None:
        if self._timeout_handle is not None:
            self._timeout_handle.cancel()
            self._timeout_handle = None
        self._buffer = ""
        self._paste_mode = False
        self._paste_buffer = ""

    def get_buffer(self) -> str:
        return self._buffer

    def destroy(self) -> None:
        self.clear()
