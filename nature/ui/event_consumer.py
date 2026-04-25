"""EventConsumer — the UI-side dependency-inversion pattern.

UIs that want to render execution events subscribe to
`EventStore.live_tail()` and dispatch to per-type handlers. The
consumer has ZERO back-channel to execution — it's a read-only
projection.

Typical usage:

    consumer = EventConsumer()
    consumer.on(EventType.MESSAGE_APPENDED, render_message)
    consumer.on(EventType.TOOL_STARTED, render_tool_pending)
    consumer.on(EventType.TOOL_COMPLETED, render_tool_done)
    consumer.on_any(debug_print)

    await consumer.consume("s1", store)

Runs indefinitely until the store closes the stream or the caller
cancels the awaiting task. Individual handler exceptions are caught
and logged so a single bad event never kills the consumer.
"""

from __future__ import annotations

import logging
from typing import Awaitable, Callable, Union

from nature.events.store import EventStore
from nature.events.types import Event, EventType

logger = logging.getLogger(__name__)

Handler = Callable[[Event], Union[None, Awaitable[None]]]


class EventConsumer:
    """Dispatches events from a store to registered handlers by type."""

    def __init__(self) -> None:
        self._handlers: dict[EventType, Handler] = {}
        self._fallback: Handler | None = None

    def on(self, event_type: EventType, handler: Handler) -> "EventConsumer":
        """Register a handler for a specific event type. Returns self for chaining."""
        self._handlers[event_type] = handler
        return self

    def on_any(self, handler: Handler) -> "EventConsumer":
        """Register a fallback handler for events with no specific handler."""
        self._fallback = handler
        return self

    async def dispatch(self, event: Event) -> None:
        """Route a single event to the appropriate handler.

        Handler exceptions are caught and logged. A misbehaving
        renderer (e.g., markup parse error on user-supplied text)
        must NOT kill the consumer task — the rest of the events
        keep flowing.
        """
        handler = self._handlers.get(event.type, self._fallback)
        if handler is None:
            return
        try:
            result = handler(event)
            if result is not None:
                await result
        except Exception as exc:
            logger.warning(
                "EventConsumer handler for %s failed: %s",
                event.type, exc,
            )

    async def consume(self, session_id: str, store: EventStore) -> None:
        """Subscribe to the store and dispatch events until the stream ends."""
        async for event in store.live_tail(session_id):
            await self.dispatch(event)

    async def replay(self, session_id: str, store: EventStore) -> None:
        """Dispatch the stored snapshot once, without subscribing for live events.

        Useful for rendering a completed session for inspection.
        """
        for event in store.snapshot(session_id):
            await self.dispatch(event)
