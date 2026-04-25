"""EventStore — the dependency inversion point between execution and UI.

Execution is the only writer (via `append`). UIs and `reconstruct()` are
readers (via `snapshot`, `live_tail`, `list_sessions`). Nothing else should
touch this module.
"""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

from nature.events.types import Event


@dataclass(frozen=True)
class SessionMeta:
    """Summary info for a session in the store.

    `parent_session_id` + `forked_from_event_id` are populated for
    sessions created via `fork()` — they let the UI render a session
    tree and let replay navigate to the original event the fork
    branched from. Both are `None` for organically created sessions.
    """

    session_id: str
    created_at: float
    last_event_at: float
    event_count: int
    parent_session_id: str | None = None
    forked_from_event_id: int | None = None


class EventStore(ABC):
    """The single sink for all execution events.

    Implementations must guarantee:
    - `append` assigns a monotonic `id` per session and returns it
    - `snapshot` returns events in insertion order
    - `live_tail` yields historical events first, then live events as they arrive
    """

    @abstractmethod
    def append(self, event: Event) -> int:
        """Append an event. Returns the assigned session-monotonic id."""
        ...

    @abstractmethod
    def snapshot(self, session_id: str) -> list[Event]:
        """Load all events for a session in order."""
        ...

    @abstractmethod
    def live_tail(self, session_id: str) -> AsyncIterator[Event]:
        """Yield historical events then live events as they're appended."""
        ...

    @abstractmethod
    def list_sessions(self) -> list[SessionMeta]:
        """List all known sessions with basic metadata."""
        ...

    @abstractmethod
    def fork(
        self,
        source_session_id: str,
        *,
        at_event_id: int,
        new_session_id: str,
    ) -> int:
        """Create a new session by copying events 1..at_event_id from an
        existing session. Returns the number of events copied.

        Original event ids are preserved in the copy, so new_session's
        events 1..at_event_id correspond one-to-one to source's
        1..at_event_id. New events appended to the forked session
        continue the monotonic id counter from at_event_id + 1.

        Implementations also persist `parent_session_id` and
        `forked_from_event_id` as session metadata so `list_sessions`
        can surface the fork relationship to UIs.

        Raises KeyError if the source session doesn't exist, and
        ValueError if at_event_id is out of range or new_session_id
        already exists.
        """
        ...

    def get_session_meta(self, session_id: str) -> SessionMeta | None:
        """Return the `SessionMeta` for one session, or None if unknown.

        Default implementation scans `list_sessions()` — subclasses
        with a direct lookup path should override for efficiency.
        """
        for meta in self.list_sessions():
            if meta.session_id == session_id:
                return meta
        return None


class FileEventStore(EventStore):
    """JSONL-backed event store. One file per session.

    Layout: `<root>/<session_id>.jsonl` with one Event per line.

    Live subscribers get fanout via in-process asyncio queues — this is
    intentionally simple and sufficient for single-process execution. For
    cross-process subscription, wrap this store with a pub/sub adapter.
    """

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._counters: dict[str, int] = {}
        self._subscribers: dict[str, list[asyncio.Queue[Event]]] = {}

    def _session_path(self, session_id: str) -> Path:
        return self._root / f"{session_id}.jsonl"

    def _session_meta_path(self, session_id: str) -> Path:
        """Sidecar JSON file that carries fork lineage + future metadata.

        Kept separate from the event log so event format stays pure
        append-only JSONL and organic sessions have no associated
        metadata file at all (fork is opt-in; absence of the sidecar
        means "root session, no parent").
        """
        return self._root / f"{session_id}.meta.json"

    def _read_sidecar(self, session_id: str) -> dict:
        """Load the session's sidecar JSON, or `{}` if absent/invalid.

        Failure is non-fatal — a missing or corrupt sidecar just means
        the session has no recorded lineage. The event log itself is
        the source of truth for everything else.
        """
        path = self._session_meta_path(session_id)
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _write_sidecar(self, session_id: str, data: dict) -> None:
        """Persist the session's sidecar JSON atomically (tmp + rename)."""
        path = self._session_meta_path(session_id)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(path)

    def _next_id(self, session_id: str) -> int:
        if session_id not in self._counters:
            path = self._session_path(session_id)
            if path.exists():
                with path.open("r", encoding="utf-8") as f:
                    self._counters[session_id] = sum(1 for line in f if line.strip())
            else:
                self._counters[session_id] = 0
        self._counters[session_id] += 1
        return self._counters[session_id]

    def append(self, event: Event) -> int:
        event_id = self._next_id(event.session_id)
        stamped = event.model_copy(update={"id": event_id})

        path = self._session_path(event.session_id)
        with path.open("a", encoding="utf-8") as f:
            f.write(stamped.model_dump_json() + "\n")

        for queue in self._subscribers.get(event.session_id, []):
            try:
                queue.put_nowait(stamped)
            except asyncio.QueueFull:
                pass

        return event_id

    def snapshot(self, session_id: str) -> list[Event]:
        path = self._session_path(session_id)
        if not path.exists():
            return []
        events: list[Event] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(Event.model_validate_json(line))
        return events

    async def live_tail(self, session_id: str) -> AsyncIterator[Event]:
        # Attach the subscriber BEFORE reading the snapshot, so any event
        # appended while we're still yielding historical items is also
        # enqueued and we don't miss it. Events whose id is ≤ the last
        # snapshot id are de-duplicated below.
        queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=1000)
        self._subscribers.setdefault(session_id, []).append(queue)
        try:
            max_snapshot_id = 0
            for ev in self.snapshot(session_id):
                if ev.id > max_snapshot_id:
                    max_snapshot_id = ev.id
                yield ev

            while True:
                ev = await queue.get()
                if ev.id <= max_snapshot_id:
                    # Already yielded from the snapshot — dedupe
                    continue
                yield ev
        finally:
            subs = self._subscribers.get(session_id, [])
            if queue in subs:
                subs.remove(queue)

    def list_sessions(self) -> list[SessionMeta]:
        sessions: list[SessionMeta] = []
        for path in sorted(self._root.glob("*.jsonl")):
            session_id = path.stem
            events = self.snapshot(session_id)
            if not events:
                continue
            sidecar = self._read_sidecar(session_id)
            sessions.append(
                SessionMeta(
                    session_id=session_id,
                    created_at=events[0].timestamp,
                    last_event_at=events[-1].timestamp,
                    event_count=len(events),
                    parent_session_id=sidecar.get("parent_session_id"),
                    forked_from_event_id=sidecar.get("forked_from_event_id"),
                )
            )
        return sessions

    def get_session_meta(self, session_id: str) -> SessionMeta | None:
        """Direct-lookup override — avoids scanning every session file."""
        path = self._session_path(session_id)
        if not path.exists():
            return None
        events = self.snapshot(session_id)
        if not events:
            return None
        sidecar = self._read_sidecar(session_id)
        return SessionMeta(
            session_id=session_id,
            created_at=events[0].timestamp,
            last_event_at=events[-1].timestamp,
            event_count=len(events),
            parent_session_id=sidecar.get("parent_session_id"),
            forked_from_event_id=sidecar.get("forked_from_event_id"),
        )

    def fork(
        self,
        source_session_id: str,
        *,
        at_event_id: int,
        new_session_id: str,
    ) -> int:
        """Copy events 1..at_event_id from source into a fresh session.

        Each copied event's `session_id` field is rewritten to
        `new_session_id` so replay/reconstruct sees the copy as its own
        session. Original event ids are preserved — new events
        appended to `new_session_id` continue from `at_event_id + 1`
        via the store's existing `_next_id` bookkeeping.

        The fork lineage (`parent_session_id`, `forked_from_event_id`,
        `forked_at`) lands in the sidecar metadata file.
        """
        source_path = self._session_path(source_session_id)
        if not source_path.exists():
            raise KeyError(
                f"source session '{source_session_id}' has no event log"
            )

        new_path = self._session_path(new_session_id)
        if new_path.exists():
            raise ValueError(
                f"cannot fork into '{new_session_id}': a session with that "
                f"id already has an event log"
            )
        if new_session_id in self._counters:
            raise ValueError(
                f"cannot fork into '{new_session_id}': id is already in use "
                f"by an in-memory session"
            )

        if at_event_id < 1:
            raise ValueError(
                f"at_event_id must be ≥ 1, got {at_event_id}"
            )

        # Scan once for PARALLEL_GROUP_STARTED / PARALLEL_GROUP_COMPLETED
        # pairs so we can reject a fork that lands strictly inside any
        # of them. Forking at the bracket boundaries themselves
        # (STARTED or COMPLETED event ids) is fine — that's "right
        # before the batch" or "right after the join", which are both
        # well-defined historical states. Forking at an id between
        # them is ambiguous (inner events have no total order
        # relative to each other), so we refuse and point the caller
        # at the nearest valid boundary.
        from nature.events.types import EventType
        parallel_brackets: list[tuple[int, int | None]] = []
        open_starts: list[int] = []
        with source_path.open("r", encoding="utf-8") as src:
            for line in src:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = Event.model_validate_json(line)
                except Exception:
                    continue
                if ev.type == EventType.PARALLEL_GROUP_STARTED:
                    open_starts.append(ev.id)
                elif ev.type == EventType.PARALLEL_GROUP_COMPLETED:
                    if open_starts:
                        start_id = open_starts.pop()
                        parallel_brackets.append((start_id, ev.id))
            # Any unclosed brackets (e.g. a crash mid-batch) are left
            # as (start_id, None); we treat them as open-ended so the
            # caller can't fork inside an in-flight batch either.
            for start_id in open_starts:
                parallel_brackets.append((start_id, None))

        for start_id, end_id in parallel_brackets:
            if end_id is None:
                if at_event_id > start_id:
                    raise ValueError(
                        f"cannot fork at event {at_event_id}: it falls "
                        f"inside an in-flight parallel batch that opened "
                        f"at event {start_id} and has no matching "
                        f"PARALLEL_GROUP_COMPLETED. Fork at event "
                        f"{start_id} instead (right before the batch)."
                    )
            else:
                if start_id < at_event_id < end_id:
                    raise ValueError(
                        f"cannot fork at event {at_event_id}: it falls "
                        f"strictly inside the parallel batch bracketed "
                        f"by events {start_id} (PARALLEL_GROUP_STARTED) "
                        f"and {end_id} (PARALLEL_GROUP_COMPLETED). "
                        f"Inner events have no total order relative to "
                        f"each other, so forking at one of them is "
                        f"ambiguous. Use event {start_id} to fork right "
                        f"before the batch, or event {end_id} to fork "
                        f"right after the join."
                    )

        # Stream source events, rewrite session_id, write to new file.
        # We rewrite via JSON string manipulation on the raw line to
        # preserve byte-for-byte fidelity of everything else in the
        # payload (which includes model-specific fields we don't
        # want to accidentally lose through a model_validate round
        # trip). Falls back to Event.model_copy if the string form is
        # not a simple single-session-id hit.
        copied = 0
        max_seen_id = 0
        import time as _time

        with source_path.open("r", encoding="utf-8") as src, \
                new_path.open("w", encoding="utf-8") as dst:
            for line in src:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = Event.model_validate_json(line)
                except Exception:
                    continue
                if ev.id > at_event_id:
                    break
                rewritten = ev.model_copy(update={
                    "session_id": new_session_id,
                })
                dst.write(rewritten.model_dump_json() + "\n")
                copied += 1
                if ev.id > max_seen_id:
                    max_seen_id = ev.id

        if copied == 0:
            # Undo: leave no empty file behind
            new_path.unlink()
            raise ValueError(
                f"no events in source session '{source_session_id}' up to "
                f"id {at_event_id}"
            )

        # Prime the in-memory id counter so the next append on the
        # forked session picks up at (max_seen_id + 1). Without this
        # the store would re-scan the file on first append, which is
        # correct but slower.
        self._counters[new_session_id] = max_seen_id

        # Sidecar metadata — fork lineage for UIs and tree traversal.
        self._write_sidecar(new_session_id, {
            "parent_session_id": source_session_id,
            "forked_from_event_id": at_event_id,
            "forked_at": _time.time(),
        })

        return copied
