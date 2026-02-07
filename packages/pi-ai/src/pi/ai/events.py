"""Async event stream for push/pull streaming pattern.

Uses asyncio.Queue internally for producer/consumer coordination.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from pi.ai.types import AssistantMessage, AssistantMessageEvent

_SENTINEL = object()


class EventStream[T, R]:
    """Generic async event stream supporting push from producers and async iteration by consumers.

    Type parameters:
        T: The event type pushed into the stream.
        R: The final result type extracted from the terminal event.
    """

    def __init__(
        self,
        is_complete: callable,
        extract_result: callable,
    ) -> None:
        self._is_complete = is_complete
        self._extract_result = extract_result
        self._queue: asyncio.Queue[T | object] = asyncio.Queue()
        self._done = False
        self._result_future: asyncio.Future[R] = asyncio.get_event_loop().create_future()

    def push(self, event: T) -> None:
        """Push an event into the stream. No-op if stream is already done."""
        if self._done:
            return

        if self._is_complete(event):
            self._done = True
            if not self._result_future.done():
                self._result_future.set_result(self._extract_result(event))

        self._queue.put_nowait(event)

    def end(self, result: R | None = None) -> None:
        """Signal that the stream is done, optionally providing a final result."""
        self._done = True
        if result is not None and not self._result_future.done():
            self._result_future.set_result(result)
        self._queue.put_nowait(_SENTINEL)

    async def __aiter__(self) -> AsyncIterator[T]:
        while True:
            item = await self._queue.get()
            if item is _SENTINEL:
                return
            if self._done and self._queue.empty():
                yield item  # type: ignore[misc]
                return
            yield item  # type: ignore[misc]

    async def result(self) -> R:
        """Await the final result from the terminal event."""
        return await self._result_future


class AssistantMessageEventStream(EventStream[AssistantMessageEvent, AssistantMessage]):
    """Specialized event stream for assistant message events."""

    def __init__(self) -> None:
        # We need to handle the case where there's no running event loop yet
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()

        super().__init__(
            is_complete=lambda event: event.type in ("done", "error"),
            extract_result=self._extract,
        )
        # Re-create future on the correct loop
        self._result_future = loop.create_future()

    @staticmethod
    def _extract(event: AssistantMessageEvent) -> AssistantMessage:
        if event.type == "done":
            return event.message
        if event.type == "error":
            return event.error
        raise ValueError(f"Unexpected event type for final result: {event.type}")


def create_assistant_message_event_stream() -> AssistantMessageEventStream:
    """Factory function for AssistantMessageEventStream."""
    return AssistantMessageEventStream()
