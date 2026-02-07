"""Tests for pi.tui.kill_ring.KillRing -- Emacs-style kill ring."""

from __future__ import annotations

from pi.tui.kill_ring import KillRing


class TestKillRingPushAndPeek:
    """Basic push and peek operations."""

    def test_push_and_peek_single_entry(self) -> None:
        ring = KillRing()
        ring.push("hello", prepend=False)
        assert ring.peek() == "hello"

    def test_push_multiple_peek_returns_most_recent(self) -> None:
        ring = KillRing()
        ring.push("first", prepend=False)
        ring.push("second", prepend=False)
        assert ring.peek() == "second"

    def test_push_empty_string_is_ignored(self) -> None:
        ring = KillRing()
        ring.push("", prepend=False)
        assert ring.peek() is None
        assert ring.length == 0

    def test_length_tracks_entries(self) -> None:
        ring = KillRing()
        assert ring.length == 0
        ring.push("a", prepend=False)
        assert ring.length == 1
        ring.push("b", prepend=False)
        assert ring.length == 2


class TestKillRingEmpty:
    """Behaviour when the ring is empty."""

    def test_peek_on_empty_returns_none(self) -> None:
        ring = KillRing()
        assert ring.peek() is None

    def test_rotate_on_empty_does_nothing(self) -> None:
        ring = KillRing()
        ring.rotate()  # should not raise
        assert ring.peek() is None

    def test_rotate_single_entry_does_nothing(self) -> None:
        ring = KillRing()
        ring.push("only", prepend=False)
        ring.rotate()
        assert ring.peek() == "only"


class TestKillRingRotate:
    """Rotation (yank-pop) cycling."""

    def test_rotate_cycles_through_entries(self) -> None:
        ring = KillRing()
        ring.push("a", prepend=False)
        ring.push("b", prepend=False)
        ring.push("c", prepend=False)
        # Ring is [a, b, c], peek -> c
        assert ring.peek() == "c"

        ring.rotate()
        # rotate moves last to front: [c, a, b], peek -> b
        assert ring.peek() == "b"

        ring.rotate()
        # [b, c, a], peek -> a
        assert ring.peek() == "a"

        ring.rotate()
        # [a, b, c], peek -> c  (full cycle)
        assert ring.peek() == "c"

    def test_rotate_two_entries(self) -> None:
        ring = KillRing()
        ring.push("x", prepend=False)
        ring.push("y", prepend=False)
        assert ring.peek() == "y"

        ring.rotate()
        assert ring.peek() == "x"

        ring.rotate()
        assert ring.peek() == "y"


class TestKillRingAccumulate:
    """Accumulate mode merges text with the most recent entry."""

    def test_accumulate_append(self) -> None:
        ring = KillRing()
        ring.push("hello", prepend=False)
        ring.push(" world", prepend=False, accumulate=True)
        assert ring.peek() == "hello world"
        assert ring.length == 1

    def test_accumulate_prepend(self) -> None:
        ring = KillRing()
        ring.push("world", prepend=True)
        ring.push("hello ", prepend=True, accumulate=True)
        assert ring.peek() == "hello world"
        assert ring.length == 1

    def test_accumulate_on_empty_ring_creates_new_entry(self) -> None:
        ring = KillRing()
        ring.push("text", prepend=False, accumulate=True)
        assert ring.peek() == "text"
        assert ring.length == 1

    def test_accumulate_multiple_times(self) -> None:
        ring = KillRing()
        ring.push("a", prepend=False)
        ring.push("b", prepend=False, accumulate=True)
        ring.push("c", prepend=False, accumulate=True)
        assert ring.peek() == "abc"
        assert ring.length == 1
