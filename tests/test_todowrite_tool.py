"""Tests for the TodoWrite built-in tool + framework wiring.

The tool itself is a thin validator (no side effects); the
framework is responsible for emitting the TODO_WRITTEN event after
a successful run. These tests cover both layers:

- TodoWriteTool.execute() — input validation, dual-form schema,
  status enum, output summary text, concurrency policy.
- AreaManager._emit_todo_written — event lands on the right frame
  with the right payload, reconstruct() picks it up, malformed
  input is rejected silently without breaking the surrounding
  TOOL_COMPLETED emission.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from nature.events import EventType, FileEventStore
from nature.events.payloads import (
    FrameOpenedPayload,
    HeaderSnapshotPayload,
    TodoWrittenPayload,
    dump_payload,
)
from nature.events.reconstruct import reconstruct
from nature.context.types import AgentRole
from nature.protocols.tool import ToolContext
from nature.protocols.todo import TodoItem
from nature.tools.builtin.todowrite import (
    TODO_WRITE_TOOL_NAME,
    TodoWriteInput,
    TodoWriteTool,
)


# ---------------------------------------------------------------------------
# Tool layer (pure)
# ---------------------------------------------------------------------------


@pytest.fixture
def ctx(tmp_path):
    return ToolContext(cwd=str(tmp_path), session_id="s1", agent_id="f1")


@pytest.mark.asyncio
async def test_todowrite_validates_dual_form_required(ctx):
    """activeForm is required — missing it should be a validation error."""
    tool = TodoWriteTool()
    result = await tool.execute({
        "todos": [{"content": "A", "status": "pending"}],
    }, ctx)
    assert result.is_error
    assert "Invalid input" in result.output


@pytest.mark.asyncio
async def test_todowrite_rejects_invalid_status(ctx):
    tool = TodoWriteTool()
    result = await tool.execute({
        "todos": [{"content": "A", "activeForm": "Doing A", "status": "blocked"}],
    }, ctx)
    assert result.is_error


@pytest.mark.asyncio
async def test_todowrite_accepts_full_list(ctx):
    tool = TodoWriteTool()
    result = await tool.execute({
        "todos": [
            {"content": "A", "activeForm": "Doing A", "status": "pending"},
            {"content": "B", "activeForm": "Doing B", "status": "in_progress"},
            {"content": "C", "activeForm": "Doing C", "status": "completed"},
        ],
    }, ctx)
    assert not result.is_error
    assert "3 items" in result.output
    assert "1 pending" in result.output
    assert "1 in_progress" in result.output
    assert "1 completed" in result.output


@pytest.mark.asyncio
async def test_todowrite_empty_list_is_valid_clear(ctx):
    """Passing an empty list = explicit clear. Tool reports it cleanly."""
    tool = TodoWriteTool()
    result = await tool.execute({"todos": []}, ctx)
    assert not result.is_error
    assert "0 items" in result.output
    assert "empty" in result.output


def test_todowrite_is_not_concurrency_safe():
    """Multiple TodoWrites must serialize so events don't race."""
    tool = TodoWriteTool()
    assert tool.is_concurrency_safe({"todos": []}) is False


def test_todowrite_is_not_read_only():
    tool = TodoWriteTool()
    assert tool.is_read_only({"todos": []}) is False


def test_todowrite_input_schema_includes_dual_form():
    """JSON schema sent to the LLM must mention both content + activeForm."""
    tool = TodoWriteTool()
    schema = tool.input_schema
    todos_schema = schema["properties"]["todos"]
    # Pydantic emits the items schema via $ref; just check the def lives
    # somewhere in the dump
    dumped = str(schema)
    assert "content" in dumped
    assert "activeForm" in dumped
    assert "status" in dumped


def test_todowrite_tool_name_constant_matches():
    """The framework's special-case branch in FrameManager keys on the
    string constant; if the tool name changes, the constant must too."""
    tool = TodoWriteTool()
    assert tool.name == TODO_WRITE_TOOL_NAME == "TodoWrite"


# ---------------------------------------------------------------------------
# Verification nudge — embedded in tool output (Rule 2 of B2.4)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verification_nudge_fires_on_3plus_all_completed_no_verifier(ctx):
    """3+ items, all completed, no verification-flavored todo → nudge."""
    tool = TodoWriteTool()
    result = await tool.execute({
        "todos": [
            {"content": "Read app.py", "activeForm": "Reading app.py",
             "status": "completed"},
            {"content": "Add /version handler", "activeForm": "Adding handler",
             "status": "completed"},
            {"content": "Run tests", "activeForm": "Running tests",
             "status": "completed"},
        ],
    }, ctx)
    assert not result.is_error
    assert "NOTE:" in result.output
    assert "verifier agent" in result.output
    assert 'Agent(name="judge"' in result.output


@pytest.mark.asyncio
async def test_verification_nudge_silent_when_under_3_items(ctx):
    """Small lists don't need independent verification."""
    tool = TodoWriteTool()
    result = await tool.execute({
        "todos": [
            {"content": "A", "activeForm": "Doing A", "status": "completed"},
            {"content": "B", "activeForm": "Doing B", "status": "completed"},
        ],
    }, ctx)
    assert "NOTE:" not in result.output


@pytest.mark.asyncio
async def test_verification_nudge_silent_when_pending_remains(ctx):
    """Mid-checklist → not all-done → no nudge yet."""
    tool = TodoWriteTool()
    result = await tool.execute({
        "todos": [
            {"content": "A", "activeForm": "Doing A", "status": "completed"},
            {"content": "B", "activeForm": "Doing B", "status": "completed"},
            {"content": "C", "activeForm": "Doing C", "status": "in_progress"},
        ],
    }, ctx)
    assert "NOTE:" not in result.output


@pytest.mark.asyncio
async def test_verification_nudge_silent_when_verifier_step_present(ctx):
    """A todo whose content mentions a verification keyword (verify,
    review, judge, validate, audit, sanity, double-check) suppresses
    the nudge."""
    tool = TodoWriteTool()
    cases = [
        "Verify the implementation",
        "Ask reviewer to check",
        "Delegate to judge",
        "Validate output schema",
        "Audit the diff",
        "Sanity check the regression",
        "Double-check edge cases",
    ]
    for verifier_text in cases:
        result = await tool.execute({
            "todos": [
                {"content": "A", "activeForm": "Doing A", "status": "completed"},
                {"content": "B", "activeForm": "Doing B", "status": "completed"},
                {"content": verifier_text, "activeForm": "Doing it",
                 "status": "completed"},
            ],
        }, ctx)
        assert "NOTE:" not in result.output, (
            f"Nudge fired despite verification step: {verifier_text!r}"
        )


@pytest.mark.asyncio
async def test_verification_nudge_lands_in_tool_completed_event(tmp_path):
    """End-to-end: when the tool's output contains the nudge, it gets
    stored in TOOL_COMPLETED.output and the message_appended.content's
    tool_result block — both are state-transition events, so
    reconstruct/time-travel preserves the nudge byte-for-byte."""
    from nature.protocols.turn import Action, ActionType

    manager, store = _make_manager_with_todowrite(tmp_path)
    frame = _seed_root_frame(manager, store)

    action = Action(
        type=ActionType.TOOL_CALL,
        tool_name="TodoWrite",
        tool_input={
            "todos": [
                {"content": "A", "activeForm": "Doing A", "status": "completed"},
                {"content": "B", "activeForm": "Doing B", "status": "completed"},
                {"content": "C", "activeForm": "Doing C", "status": "completed"},
            ],
        },
        tool_use_id="tu1",
    )
    block = await manager._execute_single_tool(frame, action)
    assert block is not None

    # The tool_result block carries the nudge text
    assert "NOTE:" in block.content

    # And the TOOL_COMPLETED event's stored output also carries it
    events = store.snapshot("s1")
    completed = next(e for e in events if e.type == EventType.TOOL_COMPLETED)
    assert "NOTE:" in completed.payload["output"]
    assert 'Agent(name="judge"' in completed.payload["output"]


@pytest.mark.asyncio
async def test_time_travel_preserves_verification_nudge(tmp_path):
    """Time-travel guarantee: the verification nudge embedded in
    TodoWrite's output text lives in the TOOL_COMPLETED event's
    payload, so a snapshot taken at *any* event id ≥ that completion
    surfaces the nudge byte-for-byte. The reconstruct() at the
    earlier point doesn't see it (because the event hasn't been
    written yet), and the reconstruct() at the later point does.
    """
    from nature.protocols.turn import Action, ActionType

    manager, store = _make_manager_with_todowrite(tmp_path)
    frame = _seed_root_frame(manager, store)

    # Trigger the all-done nudge
    await manager._execute_single_tool(frame, Action(
        type=ActionType.TOOL_CALL,
        tool_name="TodoWrite",
        tool_input={
            "todos": [
                {"content": "A", "activeForm": "Doing A", "status": "completed"},
                {"content": "B", "activeForm": "Doing B", "status": "completed"},
                {"content": "C", "activeForm": "Doing C", "status": "completed"},
            ],
        },
        tool_use_id="tu1",
    ))

    events = store.snapshot("s1")
    completed = next(e for e in events if e.type == EventType.TOOL_COMPLETED)
    started = next(e for e in events if e.type == EventType.TOOL_STARTED)
    todo_written = next(e for e in events if e.type == EventType.TODO_WRITTEN)

    # The nudge is in the TOOL_COMPLETED event's stored output —
    # a permanent fact in the event log
    assert "NOTE:" in completed.payload["output"]
    assert 'Agent(name="judge"' in completed.payload["output"]
    # And the TOOL_STARTED predates it (no nudge in the started input)
    assert "NOTE:" not in str(started.payload)

    # reconstruct at the TODO_WRITTEN point: todos are applied, but
    # tool isn't completed yet → output not part of any state event
    mid = reconstruct("s1", store, up_to_event_id=todo_written.id)
    assert len(mid.frames["f1"].context.body.todos) == 3
    assert all(t.status == "completed" for t in mid.frames["f1"].context.body.todos)

    # reconstruct past the TOOL_COMPLETED: same todos, and the nudge
    # is reachable by anyone walking the snapshot (which is what UI
    # rendering / view.py uses)
    final = reconstruct("s1", store, up_to_event_id=completed.id)
    assert len(final.frames["f1"].context.body.todos) == 3


# ---------------------------------------------------------------------------
# Framework integration — emit TODO_WRITTEN, replay via reconstruct
# ---------------------------------------------------------------------------


def _make_event(
    event_id: int = 0,
    session_id: str = "s1",
    frame_id: str = "f1",
    event_type: EventType = EventType.FRAME_OPENED,
    payload: dict | None = None,
):
    from nature.events import Event
    import time
    return Event(
        id=event_id,
        session_id=session_id,
        frame_id=frame_id,
        timestamp=time.time(),
        type=event_type,
        payload=payload or {},
    )


def test_todowrite_event_replays_into_frame_via_reconstruct():
    """Opening a frame, writing todos, then reconstructing should
    rebuild the exact list. This is the fundamental B2 promise:
    todo state survives via the event log, no in-memory sidecar."""
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))

        store.append(_make_event(
            event_type=EventType.FRAME_OPENED,
            payload=dump_payload(FrameOpenedPayload(
                purpose="test", parent_id=None, role_name="r",
                model="m",
            )),
        ))
        store.append(_make_event(
            event_type=EventType.HEADER_SNAPSHOT,
            payload=dump_payload(HeaderSnapshotPayload(
                role=AgentRole(name="r", instructions="be useful"),
                principles=[],
            )),
        ))
        store.append(_make_event(
            event_type=EventType.TODO_WRITTEN,
            payload=dump_payload(TodoWrittenPayload(
                todos=[
                    TodoItem(content="A", activeForm="Doing A"),
                    TodoItem(content="B", activeForm="Doing B", status="in_progress"),
                ],
                source="todo_write_tool",
            )),
        ))

        result = reconstruct("s1", store)
        frame = result.frames["f1"]
        assert len(frame.context.body.todos) == 2
        assert frame.context.body.todos[0].content == "A"
        assert frame.context.body.todos[0].activeForm == "Doing A"
        assert frame.context.body.todos[0].status == "pending"
        assert frame.context.body.todos[1].status == "in_progress"


def _make_manager_with_todowrite(tmp_path):
    """Build an AreaManager wired with a fake provider + the TodoWriteTool.

    The provider is never actually called by these tests — we go
    straight into _execute_single_tool / _emit_todo_written — but
    AreaManager's constructor requires it.
    """
    from nature.events import FileEventStore
    from nature.frame.manager import AreaManager
    from tests._fakes import FakeProvider
    from nature.tools.builtin.todowrite import TodoWriteTool

    store = FileEventStore(tmp_path)
    provider = FakeProvider(events=[])
    manager = AreaManager(
        store=store,
        provider=provider,
        tool_registry=[TodoWriteTool()],
        cwd=str(tmp_path),
    )
    return manager, store


def _seed_root_frame(manager, store, session_id="s1", frame_id="f1"):
    """Open a root frame in the store + register on the manager so
    `_execute_single_tool(frame, ...)` finds it. Returns the Frame."""
    from nature.events.payloads import dump_payload
    from nature.frame.frame import Frame
    from nature.context.types import (
        AgentRole, Context, ContextBody, ContextHeader,
    )

    store.append(_make_event(
        session_id=session_id, frame_id=frame_id,
        event_type=EventType.FRAME_OPENED,
        payload=dump_payload(FrameOpenedPayload(
            purpose="test", parent_id=None, role_name="r",
            model="m",
        )),
    ))
    store.append(_make_event(
        session_id=session_id, frame_id=frame_id,
        event_type=EventType.HEADER_SNAPSHOT,
        payload=dump_payload(HeaderSnapshotPayload(
            role=AgentRole(name="r", instructions="be useful"),
            principles=[],
        )),
    ))
    frame = Frame(
        id=frame_id,
        session_id=session_id,
        purpose="test",
        context=Context(
            header=ContextHeader(role=AgentRole(name="r", instructions="hi")),
            body=ContextBody(),
        ),
        model="m",
    )
    return frame


@pytest.mark.asyncio
async def test_areamanager_emits_todo_written_after_tool_call(tmp_path):
    """End-to-end: dispatch a TodoWrite via _execute_single_tool and
    verify TODO_WRITTEN landed on the same frame in the event log,
    in the right order relative to TOOL_STARTED / TOOL_COMPLETED.
    """
    from nature.protocols.turn import Action, ActionType

    manager, store = _make_manager_with_todowrite(tmp_path)
    frame = _seed_root_frame(manager, store)

    action = Action(
        type=ActionType.TOOL_CALL,
        tool_name="TodoWrite",
        tool_input={
            "todos": [
                {"content": "A", "activeForm": "Doing A", "status": "in_progress"},
                {"content": "B", "activeForm": "Doing B", "status": "pending"},
            ],
        },
        tool_use_id="tu1",
    )
    block = await manager._execute_single_tool(frame, action)
    assert block is not None
    assert not block.is_error

    events = store.snapshot("s1")
    types = [e.type for e in events]
    assert EventType.TOOL_STARTED in types
    assert EventType.TODO_WRITTEN in types
    assert EventType.TOOL_COMPLETED in types
    started_idx = types.index(EventType.TOOL_STARTED)
    todo_idx = types.index(EventType.TODO_WRITTEN)
    completed_idx = types.index(EventType.TOOL_COMPLETED)
    assert started_idx < todo_idx < completed_idx

    todo_event = events[todo_idx]
    payload = TodoWrittenPayload.model_validate(todo_event.payload)
    assert len(payload.todos) == 2
    assert payload.todos[0].content == "A"
    assert payload.todos[0].activeForm == "Doing A"
    assert payload.todos[0].status == "in_progress"
    assert payload.todos[1].status == "pending"
    assert payload.source == "todo_write_tool"

    result = reconstruct("s1", store)
    rebuilt = result.frames["f1"]
    assert len(rebuilt.context.body.todos) == 2
    assert rebuilt.context.body.todos[0].status == "in_progress"


@pytest.mark.asyncio
async def test_areamanager_syncs_in_memory_body_todos(tmp_path):
    """Regression: `_emit_todo_written` must update the live
    `frame.context.body.todos` in addition to appending the event.
    Otherwise the next `compose()` call sees stale (empty) todos and
    the footer pipeline misroutes to `synthesis_nudge` instead of
    `todo_continues_after_tool_result`, because the rules key off
    body.todos."""
    from nature.protocols.turn import Action, ActionType

    manager, store = _make_manager_with_todowrite(tmp_path)
    frame = _seed_root_frame(manager, store)
    # Before the tool call, the body has no todos at all.
    assert frame.context.body.todos == []

    action = Action(
        type=ActionType.TOOL_CALL,
        tool_name="TodoWrite",
        tool_input={
            "todos": [
                {"content": "A", "activeForm": "Doing A", "status": "in_progress"},
                {"content": "B", "activeForm": "Doing B", "status": "pending"},
            ],
        },
        tool_use_id="tu1",
    )
    block = await manager._execute_single_tool(frame, action)
    assert block is not None and not block.is_error

    # After the tool call, the live frame must reflect the write — the
    # compose() that builds the next LLM request reads directly from
    # this object, not from a replay of the event log.
    assert len(frame.context.body.todos) == 2
    assert frame.context.body.todos[0].content == "A"
    assert frame.context.body.todos[0].status == "in_progress"
    assert frame.context.body.todos[1].status == "pending"


@pytest.mark.asyncio
async def test_areamanager_skips_todo_written_on_validation_error(tmp_path):
    """If malformed input slips through to the framework's
    re-validation, we must NOT crash — just log and skip the state
    event."""
    manager, store = _make_manager_with_todowrite(tmp_path)
    frame = _seed_root_frame(manager, store)

    manager._emit_todo_written(frame, {
        "todos": [{"content": "A", "status": "pending"}],
    })
    events = store.snapshot("s1")
    assert all(e.type != EventType.TODO_WRITTEN for e in events)
