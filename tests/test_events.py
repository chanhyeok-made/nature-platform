"""Tests for the event sourcing foundation (Step 1 of the refactor)."""

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path

import pytest

from nature.events import Event, EventStore, EventType, FileEventStore
from nature.events.reconstruct import snapshot_events


def _make_event(
    session_id: str = "s1",
    frame_id: str | None = "f1",
    event_type: EventType = EventType.FRAME_OPENED,
    payload: dict | None = None,
) -> Event:
    return Event(
        id=0,
        session_id=session_id,
        frame_id=frame_id,
        timestamp=time.time(),
        type=event_type,
        payload=payload or {},
    )


def test_todo_written_event_replays_into_frame_body():
    """A TODO_WRITTEN event should land in frame.context.body.todos
    after reconstruct(), with full-list-overwrite semantics."""
    from nature.events.reconstruct import reconstruct
    from nature.events.payloads import (
        FrameOpenedPayload,
        HeaderSnapshotPayload,
        TodoWrittenPayload,
        dump_payload,
    )
    from nature.context.types import AgentRole
    from nature.protocols.todo import TodoItem

    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))

        # Open a root frame
        store.append(_make_event(
            event_type=EventType.FRAME_OPENED,
            payload=dump_payload(FrameOpenedPayload(
                purpose="test", parent_id=None, role_name="r",
                model="m", role_model=None,
            )),
        ))
        store.append(_make_event(
            event_type=EventType.HEADER_SNAPSHOT,
            payload=dump_payload(HeaderSnapshotPayload(
                role=AgentRole(name="r", instructions="be useful"),
                principles=[],
            )),
        ))

        # First TodoWrite — 2 items pending
        store.append(_make_event(
            event_type=EventType.TODO_WRITTEN,
            payload=dump_payload(TodoWrittenPayload(
                todos=[
                    TodoItem(content="A", activeForm="Doing A"),
                    TodoItem(content="B", activeForm="Doing B"),
                ],
            )),
        ))
        result = reconstruct("s1", store)
        frame = result.frames["f1"]
        assert len(frame.context.body.todos) == 2
        assert frame.context.body.todos[0].content == "A"
        assert frame.context.body.todos[0].activeForm == "Doing A"
        assert frame.context.body.todos[0].status == "pending"

        # Second TodoWrite — overwrite with 1 item in_progress + 1 completed
        store.append(_make_event(
            event_type=EventType.TODO_WRITTEN,
            payload=dump_payload(TodoWrittenPayload(
                todos=[
                    TodoItem(content="A", activeForm="Doing A", status="completed"),
                    TodoItem(content="B", activeForm="Doing B", status="in_progress"),
                ],
            )),
        ))
        result = reconstruct("s1", store)
        frame = result.frames["f1"]
        assert len(frame.context.body.todos) == 2
        assert frame.context.body.todos[0].status == "completed"
        assert frame.context.body.todos[1].status == "in_progress"

        # Third TodoWrite — clear (empty list)
        store.append(_make_event(
            event_type=EventType.TODO_WRITTEN,
            payload=dump_payload(TodoWrittenPayload(todos=[])),
        ))
        result = reconstruct("s1", store)
        frame = result.frames["f1"]
        assert frame.context.body.todos == []


def test_append_assigns_monotonic_id():
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))

        id1 = store.append(_make_event(payload={"purpose": "first"}))
        id2 = store.append(_make_event(payload={"purpose": "second"}))
        id3 = store.append(_make_event(payload={"purpose": "third"}))

        assert id1 == 1
        assert id2 == 2
        assert id3 == 3


def test_snapshot_preserves_insertion_order():
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))

        for i in range(5):
            store.append(_make_event(
                event_type=EventType.MESSAGE_APPENDED,
                payload={"message_id": f"m{i}", "idx": i},
            ))

        events = store.snapshot("s1")
        assert len(events) == 5
        assert [e.id for e in events] == [1, 2, 3, 4, 5]
        assert [e.payload["idx"] for e in events] == [0, 1, 2, 3, 4]


def test_session_isolation():
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))

        store.append(_make_event(
            session_id="s1",
            event_type=EventType.USER_INPUT,
            payload={"text": "hello"},
        ))
        store.append(_make_event(
            session_id="s2",
            event_type=EventType.USER_INPUT,
            payload={"text": "world"},
        ))
        store.append(_make_event(
            session_id="s1",
            event_type=EventType.USER_INPUT,
            payload={"text": "again"},
        ))

        s1 = store.snapshot("s1")
        s2 = store.snapshot("s2")

        assert len(s1) == 2
        assert len(s2) == 1
        assert [e.payload["text"] for e in s1] == ["hello", "again"]
        assert s2[0].payload["text"] == "world"
        # Each session has its own monotonic counter
        assert [e.id for e in s1] == [1, 2]
        assert s2[0].id == 1


def test_persistence_across_store_instances():
    """A fresh store instance should pick up existing sessions and continue
    the monotonic counter."""
    with tempfile.TemporaryDirectory() as tmp:
        store1 = FileEventStore(Path(tmp))
        store1.append(_make_event(payload={"n": 1}))
        store1.append(_make_event(payload={"n": 2}))

        store2 = FileEventStore(Path(tmp))
        new_id = store2.append(_make_event(payload={"n": 3}))

        assert new_id == 3
        assert [e.payload["n"] for e in store2.snapshot("s1")] == [1, 2, 3]


def test_list_sessions():
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))

        store.append(_make_event(session_id="s1"))
        store.append(_make_event(session_id="s1"))
        store.append(_make_event(session_id="s2"))

        sessions = store.list_sessions()
        by_id = {s.session_id: s for s in sessions}

        assert set(by_id.keys()) == {"s1", "s2"}
        assert by_id["s1"].event_count == 2
        assert by_id["s2"].event_count == 1


def test_empty_session_snapshot():
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        assert store.snapshot("nonexistent") == []


def test_event_frozen():
    """Events must be immutable once created."""
    ev = _make_event()
    with pytest.raises((ValueError, TypeError)):
        ev.payload = {"mutated": True}  # type: ignore[misc]


async def test_live_tail_yields_historical_then_live():
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))

        # Write two historical events
        store.append(_make_event(payload={"n": 1}))
        store.append(_make_event(payload={"n": 2}))

        received: list[Event] = []

        async def consumer():
            async for ev in store.live_tail("s1"):
                received.append(ev)
                if len(received) >= 4:
                    return

        task = asyncio.create_task(consumer())

        # Give consumer a moment to drain historical
        await asyncio.sleep(0.05)
        assert len(received) == 2
        assert [e.payload["n"] for e in received] == [1, 2]

        # Now emit live events
        store.append(_make_event(payload={"n": 3}))
        store.append(_make_event(payload={"n": 4}))

        await asyncio.wait_for(task, timeout=1.0)

        assert [e.payload["n"] for e in received] == [1, 2, 3, 4]
        assert [e.id for e in received] == [1, 2, 3, 4]


def test_fork_copies_events_and_rewrites_session_id():
    """Forking copies events 1..at_event_id with original ids preserved
    and session_id rewritten to the new session."""
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))

        # Source session: 5 events
        for i in range(1, 6):
            store.append(_make_event(
                session_id="src",
                payload={"step": i},
            ))

        copied = store.fork("src", at_event_id=3, new_session_id="forked")
        assert copied == 3

        src_events = store.snapshot("src")
        fork_events = store.snapshot("forked")

        # Source is untouched
        assert len(src_events) == 5
        assert all(e.session_id == "src" for e in src_events)

        # Forked session has exactly events 1..3, ids preserved
        assert len(fork_events) == 3
        assert [e.id for e in fork_events] == [1, 2, 3]
        assert [e.payload["step"] for e in fork_events] == [1, 2, 3]
        assert all(e.session_id == "forked" for e in fork_events)


def test_fork_writes_lineage_sidecar_read_back_by_list_sessions():
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))

        store.append(_make_event(session_id="src", payload={"n": 1}))
        store.append(_make_event(session_id="src", payload={"n": 2}))

        store.fork("src", at_event_id=2, new_session_id="forked")

        sessions = {s.session_id: s for s in store.list_sessions()}

        # Root session has no lineage
        assert sessions["src"].parent_session_id is None
        assert sessions["src"].forked_from_event_id is None

        # Forked session carries lineage
        assert sessions["forked"].parent_session_id == "src"
        assert sessions["forked"].forked_from_event_id == 2

        # get_session_meta should also return the lineage
        meta = store.get_session_meta("forked")
        assert meta is not None
        assert meta.parent_session_id == "src"
        assert meta.forked_from_event_id == 2


def test_fork_continues_monotonic_ids_after_append():
    """New events appended to a forked session continue from
    at_event_id + 1, preserving the store's monotonic invariant."""
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))

        for i in range(1, 6):
            store.append(_make_event(session_id="src", payload={"n": i}))

        store.fork("src", at_event_id=3, new_session_id="forked")

        new_id = store.append(_make_event(
            session_id="forked",
            payload={"n": "after-fork"},
        ))
        assert new_id == 4

        events = store.snapshot("forked")
        assert len(events) == 4
        assert [e.id for e in events] == [1, 2, 3, 4]
        assert events[-1].payload["n"] == "after-fork"


def test_fork_persists_across_store_instances():
    """A second FileEventStore pointing at the same directory should
    see the forked session, its lineage, and its id counter state."""
    with tempfile.TemporaryDirectory() as tmp:
        store1 = FileEventStore(Path(tmp))
        for i in range(1, 4):
            store1.append(_make_event(session_id="src", payload={"n": i}))
        store1.fork("src", at_event_id=2, new_session_id="forked")

        store2 = FileEventStore(Path(tmp))
        # Lineage survives the process boundary
        meta = store2.get_session_meta("forked")
        assert meta is not None
        assert meta.parent_session_id == "src"
        assert meta.forked_from_event_id == 2

        # Counter recovers — next append must continue from 3
        new_id = store2.append(_make_event(
            session_id="forked",
            payload={"n": "new"},
        ))
        assert new_id == 3


def test_fork_rejects_at_event_id_inside_parallel_bracket():
    """Events strictly between PARALLEL_GROUP_STARTED and
    PARALLEL_GROUP_COMPLETED have no total order relative to each
    other — forking at one of them is ambiguous, so the store must
    reject it and point the caller at the bracket boundaries."""
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))

        # Seed a realistic shape: 2 sequential events, then a parallel
        # batch of 3 Read-style tool events, then 2 more sequential.
        # Event ids end up as 1..10.
        store.append(_make_event(session_id="src", payload={"step": 1}))  # 1
        store.append(_make_event(session_id="src", payload={"step": 2}))  # 2
        store.append(_make_event(
            session_id="src",
            event_type=EventType.PARALLEL_GROUP_STARTED,
            payload={"group_id": "pg_abc", "tool_count": 3},
        ))  # 3
        store.append(_make_event(
            session_id="src",
            event_type=EventType.TOOL_STARTED,
            payload={"tool_use_id": "t1", "tool_name": "Read", "tool_input": {}},
        ))  # 4
        store.append(_make_event(
            session_id="src",
            event_type=EventType.TOOL_STARTED,
            payload={"tool_use_id": "t2", "tool_name": "Read", "tool_input": {}},
        ))  # 5
        store.append(_make_event(
            session_id="src",
            event_type=EventType.TOOL_COMPLETED,
            payload={"tool_use_id": "t1", "tool_name": "Read", "output": "", "is_error": False, "duration_ms": 1},
        ))  # 6
        store.append(_make_event(
            session_id="src",
            event_type=EventType.TOOL_COMPLETED,
            payload={"tool_use_id": "t2", "tool_name": "Read", "output": "", "is_error": False, "duration_ms": 1},
        ))  # 7
        store.append(_make_event(
            session_id="src",
            event_type=EventType.PARALLEL_GROUP_COMPLETED,
            payload={"group_id": "pg_abc", "duration_ms": 5},
        ))  # 8
        store.append(_make_event(session_id="src", payload={"step": 9}))   # 9
        store.append(_make_event(session_id="src", payload={"step": 10}))  # 10

        # Strict-interior ids are rejected with a message that names
        # both boundary ids.
        for bad_id in (4, 5, 6, 7):
            with pytest.raises(ValueError) as exc:
                store.fork(
                    "src", at_event_id=bad_id, new_session_id=f"bad_{bad_id}",
                )
            msg = str(exc.value)
            assert "strictly inside" in msg
            assert "3" in msg  # PARALLEL_GROUP_STARTED id
            assert "8" in msg  # PARALLEL_GROUP_COMPLETED id

        # Boundaries are allowed — forking AT the STARTED or
        # COMPLETED id is fine (both are well-defined points in time).
        store.fork("src", at_event_id=3, new_session_id="ok_started")
        store.fork("src", at_event_id=8, new_session_id="ok_completed")

        # And anything outside the bracket is fine.
        store.fork("src", at_event_id=2, new_session_id="ok_before")
        store.fork("src", at_event_id=9, new_session_id="ok_after")


def test_fork_rejects_inside_an_unclosed_parallel_bracket():
    """If a crash or cancellation leaves a PARALLEL_GROUP_STARTED with
    no matching COMPLETED event, forks at any id greater than that
    start id are rejected — the batch is considered in-flight."""
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))

        store.append(_make_event(session_id="src", payload={"step": 1}))  # 1
        store.append(_make_event(
            session_id="src",
            event_type=EventType.PARALLEL_GROUP_STARTED,
            payload={"group_id": "pg_unfinished", "tool_count": 2},
        ))  # 2
        store.append(_make_event(
            session_id="src",
            event_type=EventType.TOOL_STARTED,
            payload={"tool_use_id": "t1", "tool_name": "Bash", "tool_input": {}},
        ))  # 3
        # No PARALLEL_GROUP_COMPLETED — the batch is "still running"
        # from the store's perspective.

        with pytest.raises(ValueError) as exc:
            store.fork("src", at_event_id=3, new_session_id="bad")
        assert "in-flight parallel batch" in str(exc.value)

        # At the start id is fine — that's "right before the batch".
        store.fork("src", at_event_id=2, new_session_id="ok")
        # Events before the bracket are fine too.
        store.fork("src", at_event_id=1, new_session_id="ok2")


def test_fork_rejects_bad_arguments():
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        store.append(_make_event(session_id="src", payload={"n": 1}))
        store.append(_make_event(session_id="src", payload={"n": 2}))

        # Source doesn't exist
        with pytest.raises(KeyError):
            store.fork("nope", at_event_id=1, new_session_id="forked")

        # at_event_id below 1 is invalid
        with pytest.raises(ValueError):
            store.fork("src", at_event_id=0, new_session_id="forked")

        # at_event_id beyond the source's last event yields 0 copies,
        # which the store treats as an error (the new session file is
        # removed before raising so no zombie empty file is left).
        # First do a legit fork so "forked" is taken:
        store.fork("src", at_event_id=2, new_session_id="forked")

        # Now a second fork into the same id should fail with ValueError
        with pytest.raises(ValueError):
            store.fork("src", at_event_id=2, new_session_id="forked")


def test_snapshot_events_helper():
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        store.append(_make_event(payload={"n": 1}))
        store.append(_make_event(payload={"n": 2}))

        events = snapshot_events("s1", store)
        assert len(events) == 2
        assert [e.payload["n"] for e in events] == [1, 2]


def test_event_type_coverage():
    """Sanity check: all canonical EventType values exist and are stable."""
    expected = {
        "session.started",
        "frame.opened", "frame.resolved", "frame.closed", "frame.errored",
        "frame.reopened",
        "message.appended", "annotation.stored",
        "header.snapshot", "principle.added", "role.changed",
        "body.compacted",
        "llm.request", "llm.response", "llm.error",
        "tool.started", "tool.completed",
        "user.input", "error",
        "hint.injected",
        "todo.written",
        "parallel.started",
        "parallel.completed",
        "budget.consumed", "budget.warning", "budget.blocked",
        "loop.detected", "loop.blocked",
        "edit.miss", "path.invalid", "parse.retry",
        "read_memory.set", "ledger.symbol_confirmed",
        "ledger.approach_rejected", "ledger.test_executed",
        "ledger.rule_set",
        "agent.model_swapped",
    }
    actual = {e.value for e in EventType}
    assert expected == actual


def test_event_category_split_is_exhaustive():
    """Every EventType has a category — state-transition vs trace."""
    from nature.events import EVENT_CATEGORIES, EventCategory
    for et in EventType:
        assert et in EVENT_CATEGORIES, f"missing category for {et}"
    expected_state = {
        EventType.SESSION_STARTED,
        EventType.FRAME_OPENED, EventType.FRAME_RESOLVED,
        EventType.FRAME_CLOSED, EventType.FRAME_ERRORED,
        EventType.FRAME_REOPENED,
        EventType.MESSAGE_APPENDED, EventType.ANNOTATION_STORED,
        EventType.HEADER_SNAPSHOT, EventType.PRINCIPLE_ADDED,
        EventType.ROLE_CHANGED, EventType.BODY_COMPACTED,
        EventType.TODO_WRITTEN,
        EventType.BUDGET_CONSUMED,
        EventType.READ_MEMORY_SET,
        EventType.LEDGER_SYMBOL_CONFIRMED,
        EventType.LEDGER_APPROACH_REJECTED,
        EventType.LEDGER_TEST_EXECUTED,
        EventType.LEDGER_RULE_SET,
        EventType.AGENT_MODEL_SWAPPED,
    }
    actual_state = {
        et for et, cat in EVENT_CATEGORIES.items()
        if cat is EventCategory.STATE_TRANSITION
    }
    assert expected_state == actual_state


def test_eventstore_is_abstract():
    with pytest.raises(TypeError):
        EventStore()  # type: ignore[abstract]


# ──────────────────────────────────────────────────────────────────────
# Phase 1 — new event types round-trip through reconstruct
# ──────────────────────────────────────────────────────────────────────


def _seed_root_frame(store: FileEventStore) -> None:
    """Emit FRAME_OPENED + HEADER_SNAPSHOT for a dummy root frame f1."""
    from nature.context.types import AgentRole
    from nature.events.payloads import (
        FrameOpenedPayload,
        HeaderSnapshotPayload,
        dump_payload,
    )

    store.append(_make_event(
        event_type=EventType.FRAME_OPENED,
        payload=dump_payload(FrameOpenedPayload(
            purpose="test", parent_id=None, role_name="r",
            model="m", role_model=None,
        )),
    ))
    store.append(_make_event(
        event_type=EventType.HEADER_SNAPSHOT,
        payload=dump_payload(HeaderSnapshotPayload(
            role=AgentRole(name="r", instructions="be useful"),
            principles=[],
        )),
    ))


def test_budget_consumed_replays_to_frame_counter():
    from nature.events.payloads import BudgetConsumedPayload, dump_payload
    from nature.events.reconstruct import reconstruct

    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        _seed_root_frame(store)
        for _ in range(3):
            store.append(_make_event(
                event_type=EventType.BUDGET_CONSUMED,
                payload=dump_payload(BudgetConsumedPayload(kind="reads", delta=1)),
            ))
        store.append(_make_event(
            event_type=EventType.BUDGET_CONSUMED,
            payload=dump_payload(BudgetConsumedPayload(kind="turns", delta=2)),
        ))

        result = reconstruct("s1", store)
        frame = result.frames["f1"]
        assert frame.budget_counts["reads"] == 3
        assert frame.budget_counts["turns"] == 2


def test_read_memory_set_replays():
    from nature.events.payloads import ReadMemorySetPayload, dump_payload
    from nature.events.reconstruct import reconstruct

    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        _seed_root_frame(store)
        store.append(_make_event(
            event_type=EventType.READ_MEMORY_SET,
            payload=dump_payload(ReadMemorySetPayload(
                path="src/foo.py", content_hash="abc123",
                mtime_ns=999, lines=42, offset=0, limit=42, depth=0,
            )),
        ))

        result = reconstruct("s1", store)
        frame = result.frames["f1"]
        # ReadMemory entry restored as expired (segments=None)
        read_memory = frame.pack_state.get("read_memory")
        assert read_memory is not None
        assert len(read_memory) == 1
        entry = list(read_memory._entries.values())[0]
        assert entry.expired is True
        assert entry.mtime_ns == 999
        assert entry.total_lines == 42
        assert entry.depth == 0


def test_ledger_symbol_confirmed_replays():
    from nature.events.payloads import LedgerSymbolConfirmedPayload, dump_payload
    from nature.events.reconstruct import reconstruct

    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        _seed_root_frame(store)
        store.append(_make_event(
            event_type=EventType.LEDGER_SYMBOL_CONFIRMED,
            payload=dump_payload(LedgerSymbolConfirmedPayload(
                name="DuplicateEmailError", file="src/err.py",
                line_start=10, line_end=14,
            )),
        ))

        result = reconstruct("s1", store)
        symbols = result.frames["f1"].ledger.symbols_confirmed
        assert len(symbols) == 1
        assert symbols[0]["name"] == "DuplicateEmailError"
        assert symbols[0]["line_end"] == 14


def test_ledger_approach_rejected_replays():
    from nature.events.payloads import LedgerApproachRejectedPayload, dump_payload
    from nature.events.reconstruct import reconstruct

    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        _seed_root_frame(store)
        store.append(_make_event(
            event_type=EventType.LEDGER_APPROACH_REJECTED,
            payload=dump_payload(LedgerApproachRejectedPayload(
                tool="Edit", input_hash="h1", reason="old_string not found",
            )),
        ))

        result = reconstruct("s1", store)
        rejects = result.frames["f1"].ledger.approaches_rejected
        assert len(rejects) == 1
        assert rejects[0]["tool"] == "Edit"
        assert rejects[0]["input_hash"] == "h1"


def test_ledger_test_executed_replays():
    from nature.events.payloads import LedgerTestExecutedPayload, dump_payload
    from nature.events.reconstruct import reconstruct

    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        _seed_root_frame(store)
        store.append(_make_event(
            event_type=EventType.LEDGER_TEST_EXECUTED,
            payload=dump_payload(LedgerTestExecutedPayload(
                command="pytest tests/test_foo.py", status="passed",
                summary="12 passed",
            )),
        ))

        result = reconstruct("s1", store)
        tests = result.frames["f1"].ledger.tests_executed
        assert len(tests) == 1
        assert tests[0]["status"] == "passed"
        assert tests[0]["summary"] == "12 passed"


def test_ledger_rule_set_replays():
    from nature.events.payloads import LedgerRuleSetPayload, dump_payload
    from nature.events.reconstruct import reconstruct

    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        _seed_root_frame(store)
        store.append(_make_event(
            event_type=EventType.LEDGER_RULE_SET,
            payload=dump_payload(LedgerRuleSetPayload(
                rule="do not touch migrations", source="judge",
            )),
        ))

        result = reconstruct("s1", store)
        rules = result.frames["f1"].ledger.rules
        assert len(rules) == 1
        assert rules[0]["rule"] == "do not touch migrations"
        assert rules[0]["source"] == "judge"


def test_agent_model_swapped_replays():
    from nature.events.payloads import AgentModelSwappedPayload, dump_payload
    from nature.events.reconstruct import reconstruct

    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        _seed_root_frame(store)
        # Sanity: frame starts on the opened model
        result = reconstruct("s1", store)
        assert result.frames["f1"].model == "m"

        store.append(_make_event(
            event_type=EventType.AGENT_MODEL_SWAPPED,
            payload=dump_payload(AgentModelSwappedPayload(
                previous_model="m",
                new_model="claude-opus-4-6",
                reason="escalation",
            )),
        ))

        result = reconstruct("s1", store)
        assert result.frames["f1"].model == "claude-opus-4-6"


def test_new_trace_events_round_trip_payload():
    """New trace-only events load back into their declared payload type."""
    from nature.events.payloads import (
        BudgetBlockedPayload,
        BudgetWarningPayload,
        EditMissPayload,
        LoopBlockedPayload,
        LoopDetectedPayload,
        ParseRetryPayload,
        PathInvalidPayload,
        dump_payload,
        load_payload,
    )

    cases = [
        (EventType.BUDGET_WARNING,
         BudgetWarningPayload(kind="reads", used=16, limit=20)),
        (EventType.BUDGET_BLOCKED,
         BudgetBlockedPayload(kind="reads", tool="Read", reason="cap")),
        (EventType.LOOP_DETECTED,
         LoopDetectedPayload(tool="Edit", input_hash="h1", attempts=3)),
        (EventType.LOOP_BLOCKED,
         LoopBlockedPayload(tool="Edit", input_hash="h1")),
        (EventType.EDIT_MISS,
         EditMissPayload(file="src/foo.py", fuzzy_match="    def foo():", lineno=10)),
        (EventType.PATH_INVALID,
         PathInvalidPayload(path="src/nope.py", tool="Read",
                            suggestions=["src/nose.py"])),
        (EventType.PARSE_RETRY,
         ParseRetryPayload(attempt=2, reason="json incomplete")),
    ]

    for et, payload in cases:
        evt = _make_event(event_type=et, payload=dump_payload(payload))
        loaded = load_payload(evt)
        assert type(loaded) is type(payload), f"{et}: got {type(loaded).__name__}"
