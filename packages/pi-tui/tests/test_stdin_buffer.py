"""Tests for pi.tui.stdin_buffer.StdinBuffer."""

from __future__ import annotations

import asyncio

import pytest

from pi.tui.stdin_buffer import (
    BRACKETED_PASTE_END,
    BRACKETED_PASTE_START,
    ESC,
    StdinBuffer,
    _extract_complete_sequences,
    _is_complete_sequence,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class Collector:
    """Collects emitted data/paste events for assertions."""

    def __init__(self) -> None:
        self.data: list[str] = []
        self.pastes: list[str] = []

    def on_data(self, d: str) -> None:
        self.data.append(d)

    def on_paste(self, d: str) -> None:
        self.pastes.append(d)


def make_buffer(timeout: float = 0.01) -> tuple[StdinBuffer, Collector]:
    buf = StdinBuffer(timeout=timeout)
    col = Collector()
    buf.on_data(col.on_data)
    buf.on_paste(col.on_paste)
    return buf, col


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_default_timeout(self) -> None:
        buf = StdinBuffer()
        assert buf._timeout_ms == 0.01

    def test_custom_timeout(self) -> None:
        buf = StdinBuffer(timeout=0.05)
        assert buf._timeout_ms == 0.05

    def test_initial_state(self) -> None:
        buf = StdinBuffer()
        assert buf.get_buffer() == ""
        assert buf._paste_mode is False
        assert buf._paste_buffer == ""
        assert buf._on_data is None
        assert buf._on_paste is None

    def test_set_on_data_callback(self) -> None:
        buf = StdinBuffer()
        cb = lambda d: None  # noqa: E731
        buf.on_data(cb)
        assert buf._on_data is cb

    def test_set_on_paste_callback(self) -> None:
        buf = StdinBuffer()
        cb = lambda d: None  # noqa: E731
        buf.on_paste(cb)
        assert buf._on_paste is cb


# ---------------------------------------------------------------------------
# _is_complete_sequence (internal helper, tested for confidence)
# ---------------------------------------------------------------------------


class TestIsCompleteSequence:
    def test_non_escape_returns_not_escape(self) -> None:
        assert _is_complete_sequence("a") == "not-escape"
        assert _is_complete_sequence("hello") == "not-escape"

    def test_lone_esc_is_incomplete(self) -> None:
        assert _is_complete_sequence(ESC) == "incomplete"

    def test_meta_key_sequence_is_complete(self) -> None:
        # ESC followed by a single char = meta key
        assert _is_complete_sequence(f"{ESC}a") == "complete"
        assert _is_complete_sequence(f"{ESC}x") == "complete"

    def test_csi_arrow_key_is_complete(self) -> None:
        assert _is_complete_sequence(f"{ESC}[A") == "complete"
        assert _is_complete_sequence(f"{ESC}[B") == "complete"

    def test_csi_incomplete_without_terminator(self) -> None:
        assert _is_complete_sequence(f"{ESC}[") == "incomplete"
        assert _is_complete_sequence(f"{ESC}[1") == "incomplete"
        assert _is_complete_sequence(f"{ESC}[1;") == "incomplete"

    def test_csi_with_parameters_is_complete(self) -> None:
        # e.g. ESC[1;5A = Ctrl+Up
        assert _is_complete_sequence(f"{ESC}[1;5A") == "complete"

    def test_csi_mouse_normal_mode(self) -> None:
        # Normal mouse: ESC[M followed by 3 bytes (total 6 chars)
        assert _is_complete_sequence(f"{ESC}[M   ") == "complete"
        assert _is_complete_sequence(f"{ESC}[M  ") == "incomplete"

    def test_csi_sgr_mouse_complete(self) -> None:
        assert _is_complete_sequence(f"{ESC}[<0;10;20M") == "complete"
        assert _is_complete_sequence(f"{ESC}[<0;10;20m") == "complete"

    def test_csi_sgr_mouse_incomplete(self) -> None:
        assert _is_complete_sequence(f"{ESC}[<0;10") == "incomplete"
        assert _is_complete_sequence(f"{ESC}[<0;10;") == "incomplete"

    def test_osc_incomplete_without_terminator(self) -> None:
        assert _is_complete_sequence(f"{ESC}]0;title") == "incomplete"

    def test_osc_complete_with_bel(self) -> None:
        assert _is_complete_sequence(f"{ESC}]0;title\x07") == "complete"

    def test_osc_complete_with_st(self) -> None:
        assert _is_complete_sequence(f"{ESC}]0;title{ESC}\\") == "complete"

    def test_dcs_incomplete(self) -> None:
        assert _is_complete_sequence(f"{ESC}Ppayload") == "incomplete"

    def test_dcs_complete(self) -> None:
        assert _is_complete_sequence(f"{ESC}Ppayload{ESC}\\") == "complete"

    def test_apc_incomplete(self) -> None:
        assert _is_complete_sequence(f"{ESC}_payload") == "incomplete"

    def test_apc_complete(self) -> None:
        assert _is_complete_sequence(f"{ESC}_payload{ESC}\\") == "complete"

    def test_ss3_incomplete(self) -> None:
        assert _is_complete_sequence(f"{ESC}O") == "incomplete"

    def test_ss3_complete(self) -> None:
        # ESC O followed by at least one character
        assert _is_complete_sequence(f"{ESC}OP") == "complete"


# ---------------------------------------------------------------------------
# _extract_complete_sequences
# ---------------------------------------------------------------------------


class TestExtractCompleteSequences:
    def test_empty_string(self) -> None:
        seqs, rem = _extract_complete_sequences("")
        assert seqs == []
        assert rem == ""

    def test_single_regular_char(self) -> None:
        seqs, rem = _extract_complete_sequences("a")
        assert seqs == ["a"]
        assert rem == ""

    def test_multiple_regular_chars(self) -> None:
        seqs, rem = _extract_complete_sequences("abc")
        assert seqs == ["a", "b", "c"]
        assert rem == ""

    def test_single_complete_escape_sequence(self) -> None:
        seqs, rem = _extract_complete_sequences(f"{ESC}[A")
        assert seqs == [f"{ESC}[A"]
        assert rem == ""

    def test_incomplete_escape_goes_to_remainder(self) -> None:
        seqs, rem = _extract_complete_sequences(ESC)
        assert seqs == []
        assert rem == ESC

    def test_mixed_chars_and_sequences(self) -> None:
        data = f"a{ESC}[Ab"
        seqs, rem = _extract_complete_sequences(data)
        assert seqs == ["a", f"{ESC}[A", "b"]
        assert rem == ""

    def test_incomplete_csi_at_end(self) -> None:
        data = f"x{ESC}[1"
        seqs, rem = _extract_complete_sequences(data)
        assert seqs == ["x"]
        assert rem == f"{ESC}[1"

    def test_multiple_escape_sequences(self) -> None:
        data = f"{ESC}[A{ESC}[B{ESC}[C"
        seqs, rem = _extract_complete_sequences(data)
        assert seqs == [f"{ESC}[A", f"{ESC}[B", f"{ESC}[C"]
        assert rem == ""


# ---------------------------------------------------------------------------
# StdinBuffer.process — regular characters
# ---------------------------------------------------------------------------


class TestProcessRegularChars:
    def test_single_character(self) -> None:
        buf, col = make_buffer()
        buf.process("a")
        assert col.data == ["a"]

    def test_each_char_emitted_individually(self) -> None:
        buf, col = make_buffer()
        buf.process("abc")
        assert col.data == ["a", "b", "c"]

    def test_empty_data_with_empty_buffer_emits_once(self) -> None:
        buf, col = make_buffer()
        buf.process("")
        assert col.data == [""]

    def test_no_callback_does_not_raise(self) -> None:
        buf = StdinBuffer()
        buf.process("abc")  # No callback set, should not raise


# ---------------------------------------------------------------------------
# StdinBuffer.process — escape sequences
# ---------------------------------------------------------------------------


class TestProcessEscapeSequences:
    def test_complete_csi_emitted(self) -> None:
        buf, col = make_buffer()
        buf.process(f"{ESC}[A")
        assert f"{ESC}[A" in col.data

    def test_complete_meta_key_emitted(self) -> None:
        buf, col = make_buffer()
        buf.process(f"{ESC}x")
        assert f"{ESC}x" in col.data

    def test_complete_osc_with_bel_emitted(self) -> None:
        buf, col = make_buffer()
        buf.process(f"{ESC}]0;mytitle\x07")
        assert f"{ESC}]0;mytitle\x07" in col.data

    def test_complete_osc_with_st_emitted(self) -> None:
        buf, col = make_buffer()
        buf.process(f"{ESC}]0;mytitle{ESC}\\")
        assert f"{ESC}]0;mytitle{ESC}\\" in col.data

    def test_complete_ss3_emitted(self) -> None:
        buf, col = make_buffer()
        buf.process(f"{ESC}OP")
        assert f"{ESC}OP" in col.data

    @pytest.mark.asyncio
    async def test_partial_escape_buffered(self) -> None:
        buf, col = make_buffer()
        buf.process(ESC)
        # The lone ESC is incomplete; should be buffered, not emitted yet
        assert col.data == []
        assert buf.get_buffer() == ESC

    @pytest.mark.asyncio
    async def test_partial_csi_buffered(self) -> None:
        buf, col = make_buffer()
        buf.process(f"{ESC}[")
        assert col.data == []
        assert buf.get_buffer() == f"{ESC}["

    @pytest.mark.asyncio
    async def test_split_csi_across_chunks(self) -> None:
        """Simulates ESC arriving in one chunk and [A in the next."""
        buf, col = make_buffer()
        buf.process(ESC)
        assert col.data == []
        buf.process("[A")
        assert f"{ESC}[A" in col.data

    def test_sgr_mouse_sequence(self) -> None:
        buf, col = make_buffer()
        buf.process(f"{ESC}[<0;50;25M")
        assert f"{ESC}[<0;50;25M" in col.data

    def test_chars_before_escape_emitted_first(self) -> None:
        buf, col = make_buffer()
        buf.process(f"ab{ESC}[A")
        assert col.data == ["a", "b", f"{ESC}[A"]


# ---------------------------------------------------------------------------
# StdinBuffer.process — bracketed paste
# ---------------------------------------------------------------------------


class TestProcessBracketedPaste:
    def test_complete_paste_in_single_chunk(self) -> None:
        buf, col = make_buffer()
        text = "hello world"
        buf.process(f"{BRACKETED_PASTE_START}{text}{BRACKETED_PASTE_END}")
        assert col.pastes == [text]
        assert col.data == []  # Paste content should not appear in data

    def test_paste_content_with_special_chars(self) -> None:
        buf, col = make_buffer()
        text = "line1\nline2\ttab"
        buf.process(f"{BRACKETED_PASTE_START}{text}{BRACKETED_PASTE_END}")
        assert col.pastes == [text]

    def test_data_before_paste_emitted(self) -> None:
        buf, col = make_buffer()
        buf.process(f"xy{BRACKETED_PASTE_START}hello{BRACKETED_PASTE_END}")
        assert "x" in col.data
        assert "y" in col.data
        assert col.pastes == ["hello"]

    def test_data_after_paste_emitted(self) -> None:
        buf, col = make_buffer()
        buf.process(
            f"{BRACKETED_PASTE_START}hello{BRACKETED_PASTE_END}z"
        )
        assert col.pastes == ["hello"]
        assert "z" in col.data

    def test_paste_split_across_chunks(self) -> None:
        buf, col = make_buffer()
        buf.process(BRACKETED_PASTE_START + "hel")
        assert col.pastes == []
        buf.process("lo" + BRACKETED_PASTE_END)
        assert col.pastes == ["hello"]

    def test_empty_paste(self) -> None:
        buf, col = make_buffer()
        buf.process(f"{BRACKETED_PASTE_START}{BRACKETED_PASTE_END}")
        assert col.pastes == [""]

    def test_paste_mode_cleared_after_complete(self) -> None:
        buf, col = make_buffer()
        buf.process(f"{BRACKETED_PASTE_START}x{BRACKETED_PASTE_END}")
        assert buf._paste_mode is False
        assert buf._paste_buffer == ""


# ---------------------------------------------------------------------------
# StdinBuffer.flush
# ---------------------------------------------------------------------------


class TestFlush:
    def test_flush_empty_buffer_returns_empty(self) -> None:
        buf, _ = make_buffer()
        assert buf.flush() == []

    @pytest.mark.asyncio
    async def test_flush_returns_buffered_content(self) -> None:
        buf, col = make_buffer()
        buf.process(ESC)
        assert buf.get_buffer() != ""
        result = buf.flush()
        assert result == [ESC]
        assert buf.get_buffer() == ""

    @pytest.mark.asyncio
    async def test_flush_clears_buffer(self) -> None:
        buf, _ = make_buffer()
        buf.process(ESC)
        buf.flush()
        assert buf.get_buffer() == ""


# ---------------------------------------------------------------------------
# StdinBuffer.clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clear_resets_buffer(self) -> None:
        buf, _ = make_buffer()
        buf.process(ESC)
        buf.clear()
        assert buf.get_buffer() == ""

    def test_clear_resets_paste_mode(self) -> None:
        buf, _ = make_buffer()
        buf._paste_mode = True
        buf._paste_buffer = "something"
        buf.clear()
        assert buf._paste_mode is False
        assert buf._paste_buffer == ""


# ---------------------------------------------------------------------------
# StdinBuffer.destroy
# ---------------------------------------------------------------------------


class TestDestroy:
    def test_destroy_clears_everything(self) -> None:
        buf, _ = make_buffer()
        buf.process(ESC)
        buf.destroy()
        assert buf.get_buffer() == ""
        assert buf._paste_mode is False
        assert buf._paste_buffer == ""


# ---------------------------------------------------------------------------
# StdinBuffer.get_buffer
# ---------------------------------------------------------------------------


class TestGetBuffer:
    def test_empty_initially(self) -> None:
        buf = StdinBuffer()
        assert buf.get_buffer() == ""

    @pytest.mark.asyncio
    async def test_returns_pending_data(self) -> None:
        buf, _ = make_buffer()
        buf.process(ESC)
        assert buf.get_buffer() == ESC


# ---------------------------------------------------------------------------
# Timeout flush behaviour (requires event loop)
# ---------------------------------------------------------------------------


class TestTimeoutFlush:
    @pytest.mark.asyncio
    async def test_incomplete_sequence_flushed_on_timeout(self) -> None:
        buf, col = make_buffer(timeout=0.02)
        buf.process(ESC)
        assert col.data == []
        # Wait for the timeout to fire
        await asyncio.sleep(0.05)
        assert ESC in col.data

    @pytest.mark.asyncio
    async def test_timeout_cancelled_on_new_data(self) -> None:
        buf, col = make_buffer(timeout=0.05)
        buf.process(ESC)
        # Immediately complete the sequence before timeout
        buf.process("[A")
        assert f"{ESC}[A" in col.data
        # After the original timeout window, no duplicate emission
        await asyncio.sleep(0.08)
        count = col.data.count(f"{ESC}[A")
        assert count == 1
