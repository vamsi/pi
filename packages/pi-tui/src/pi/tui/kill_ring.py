"""Ring buffer for Emacs-style kill/yank operations."""

from __future__ import annotations


class KillRing:
    """Tracks killed (deleted) text entries.

    Consecutive kills can accumulate into a single entry.
    Supports yank (paste most recent) and yank-pop (cycle through older entries).
    """

    def __init__(self) -> None:
        self._ring: list[str] = []

    def push(self, text: str, *, prepend: bool, accumulate: bool = False) -> None:
        """Add text to the kill ring.

        Args:
            text: The killed text to add.
            prepend: If accumulating, prepend (backward deletion) or append (forward deletion).
            accumulate: Merge with the most recent entry instead of creating a new one.
        """
        if not text:
            return

        if accumulate and self._ring:
            last = self._ring.pop()
            self._ring.append(text + last if prepend else last + text)
        else:
            self._ring.append(text)

    def peek(self) -> str | None:
        """Get most recent entry without modifying the ring."""
        return self._ring[-1] if self._ring else None

    def rotate(self) -> None:
        """Move last entry to front (for yank-pop cycling)."""
        if len(self._ring) > 1:
            last = self._ring.pop()
            self._ring.insert(0, last)

    @property
    def length(self) -> int:
        return len(self._ring)
