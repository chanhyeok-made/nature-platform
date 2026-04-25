"""Integration tests for SessionRunner + EventConsumer (Step 6).

These tests wire the full new path end-to-end: execution (SessionRunner
→ AreaManager → llm_agent) writes events to the store, and a pure
consumer (EventConsumer) reads them back. No UI code, no middleware.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from nature.context import AgentRole
from nature.events import Event, EventType, FileEventStore
from nature.frame import AgentTool, FrameState
from nature.protocols.message import (
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolUseContent,
    Usage,
)
from nature.config.constants import StopReason
from nature.session.runner import SessionRunner
from nature.ui.event_consumer import EventConsumer

from tests._fakes import FakeProvider, FakeTool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text_turn(text: str) -> list[StreamEvent]:
    return [
        StreamEvent(type=StreamEventType.MESSAGE_START),
        StreamEvent(
            type=StreamEventType.CONTENT_BLOCK_START,
            index=0,
            content_block=TextContent(text=""),
        ),
        StreamEvent(
            type=StreamEventType.CONTENT_BLOCK_DELTA,
            index=0,
            delta_text=text,
        ),
        StreamEvent(type=StreamEventType.CONTENT_BLOCK_STOP, index=0),
        StreamEvent(
            type=StreamEventType.MESSAGE_STOP,
            usage=Usage(input_tokens=5, output_tokens=3),
            stop_reason=StopReason.END_TURN,
        ),
    ]


def _tool_turn(name: str, inp: dict, tid: str = "t1") -> list[StreamEvent]:
    block = ToolUseContent(id=tid, name=name, input=inp)
    return [
        StreamEvent(type=StreamEventType.MESSAGE_START),
        StreamEvent(
            type=StreamEventType.CONTENT_BLOCK_START,
            index=0,
            content_block=block,
        ),
        StreamEvent(
            type=StreamEventType.CONTENT_BLOCK_STOP,
            index=0,
            content_block=block,
        ),
        StreamEvent(
            type=StreamEventType.MESSAGE_STOP,
            usage=Usage(input_tokens=10, output_tokens=5),
            stop_reason=StopReason.TOOL_USE,
        ),
    ]


def _role(name: str = "receptionist") -> AgentRole:
    return AgentRole(
        name=name,
        instructions=f"you are {name}",
        allowed_tools=None,
    )


# ---------------------------------------------------------------------------
# SessionRunner — basic run
# ---------------------------------------------------------------------------


async def test_session_runner_runs_and_resolves():
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        provider = FakeProvider(_text_turn("hello back"))

        runner = SessionRunner(
            provider=provider,
            tool_registry=[AgentTool()],
            event_store=store,
            cwd="/tmp",
        )

        result = await runner.run(
            session_id="s1",
            role=_role(),
            model="fake",
            user_input="hi",
        )

        assert result.is_resolved is True
        assert result.frame.state == FrameState.RESOLVED
        assert result.session_id == "s1"

        # Events were written to the store
        events = store.snapshot("s1")
        assert len(events) > 0
        assert events[0].type == EventType.FRAME_OPENED
        assert events[-1].type == EventType.FRAME_RESOLVED


async def test_session_runner_tool_execution_writes_events():
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        provider = FakeProvider([
            _tool_turn("FakeRead", {"path": "/x"}, tid="t1"),
            _text_turn("done"),
        ])
        tools = [AgentTool(), FakeTool("FakeRead", output="file body")]

        runner = SessionRunner(
            provider=provider,
            tool_registry=tools,
            event_store=store,
            cwd="/tmp",
        )

        result = await runner.run(
            session_id="s1",
            role=_role(),
            model="fake",
            user_input="read file",
        )

        assert result.is_resolved
        types = [e.type for e in store.snapshot("s1")]
        assert EventType.TOOL_STARTED in types
        assert EventType.TOOL_COMPLETED in types


# ---------------------------------------------------------------------------
# continue_session
# ---------------------------------------------------------------------------


async def test_continue_session_appends_user_input_and_runs_again():
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        provider = FakeProvider([
            _text_turn("first answer"),
            _text_turn("second answer"),
        ])

        runner = SessionRunner(
            provider=provider,
            tool_registry=[AgentTool()],
            event_store=store,
            cwd="/tmp",
        )

        first = await runner.run(
            session_id="s1",
            role=_role(),
            model="fake",
            user_input="first question",
        )
        assert first.is_resolved

        second = await runner.continue_session(
            frame=first.frame,
            user_input="second question",
        )
        assert second.is_resolved

        # Conversation has both turns
        msgs = first.frame.context.body.conversation.messages
        assert len(msgs) == 4
        # user, assistant, user, assistant
        assert msgs[0].from_ == "user"
        assert msgs[1].from_ == "receptionist"
        assert msgs[2].from_ == "user"
        assert msgs[3].from_ == "receptionist"

        # Events show two USER_INPUT
        types = [e.type for e in store.snapshot("s1")]
        assert types.count(EventType.USER_INPUT) == 2


async def test_close_emits_frame_closed():
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        provider = FakeProvider(_text_turn("ok"))
        runner = SessionRunner(
            provider=provider,
            tool_registry=[AgentTool()],
            event_store=store,
            cwd="/tmp",
        )

        result = await runner.run(
            session_id="s1", role=_role(), model="fake", user_input="go",
        )
        runner.close(result.frame)

        types = [e.type for e in store.snapshot("s1")]
        assert types[-1] == EventType.FRAME_CLOSED


# ---------------------------------------------------------------------------
# EventConsumer — pure read-only path
# ---------------------------------------------------------------------------


async def test_event_consumer_replay_dispatches_all_events():
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        provider = FakeProvider(_text_turn("hi"))
        runner = SessionRunner(
            provider=provider,
            tool_registry=[AgentTool()],
            event_store=store,
            cwd="/tmp",
        )
        await runner.run(
            session_id="s1", role=_role(), model="fake", user_input="go",
        )

        # Pure consumer — no writes, no execution references
        seen: list[EventType] = []
        consumer = EventConsumer()
        consumer.on_any(lambda ev: seen.append(ev.type))

        await consumer.replay("s1", store)

        assert EventType.FRAME_OPENED in seen
        assert EventType.MESSAGE_APPENDED in seen
        assert EventType.FRAME_RESOLVED in seen


async def test_event_consumer_per_type_handlers_dispatch_correctly():
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        provider = FakeProvider([
            _tool_turn("FakeRead", {"path": "/x"}, tid="t1"),
            _text_turn("ok"),
        ])
        tools = [AgentTool(), FakeTool("FakeRead", output="body")]
        runner = SessionRunner(
            provider=provider, tool_registry=tools,
            event_store=store, cwd="/tmp",
        )
        await runner.run(
            session_id="s1", role=_role(), model="fake", user_input="read",
        )

        tool_events: list[Event] = []
        message_events: list[Event] = []

        consumer = (
            EventConsumer()
            .on(EventType.TOOL_STARTED, lambda ev: tool_events.append(ev))
            .on(EventType.TOOL_COMPLETED, lambda ev: tool_events.append(ev))
            .on(EventType.MESSAGE_APPENDED, lambda ev: message_events.append(ev))
        )

        await consumer.replay("s1", store)

        assert len(tool_events) == 2  # started + completed
        assert any(e.payload["tool_name"] == "FakeRead" for e in tool_events)
        assert len(message_events) >= 3  # user + assistant + tool result + assistant


async def test_event_consumer_live_tail_sees_events_as_they_arrive():
    """live_tail should yield historical events then stream new ones."""
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        provider = FakeProvider(_text_turn("ok"))

        runner = SessionRunner(
            provider=provider,
            tool_registry=[AgentTool()],
            event_store=store,
            cwd="/tmp",
        )

        seen: list[Event] = []
        done = asyncio.Event()
        target_types = {
            EventType.FRAME_OPENED,
            EventType.USER_INPUT,
            EventType.MESSAGE_APPENDED,
            EventType.LLM_REQUEST,
            EventType.LLM_RESPONSE,
            EventType.ANNOTATION_STORED,
            EventType.FRAME_RESOLVED,
        }

        def record(event: Event) -> None:
            seen.append(event)
            if event.type == EventType.FRAME_RESOLVED:
                done.set()

        consumer = EventConsumer().on_any(record)

        # Start the consumer first — live_tail will attach
        consume_task = asyncio.create_task(consumer.consume("s1", store))
        # Give it a moment to attach its subscriber queue
        await asyncio.sleep(0.02)

        # Now run — events should flow to the consumer in real time
        await runner.run(
            session_id="s1",
            role=_role(),
            model="fake",
            user_input="hi",
        )

        # Wait for the resolved event
        await asyncio.wait_for(done.wait(), timeout=1.0)
        consume_task.cancel()
        try:
            await consume_task
        except asyncio.CancelledError:
            pass

        seen_types = {e.type for e in seen}
        # All expected transitions arrived through live_tail
        assert target_types.issubset(seen_types)


# ---------------------------------------------------------------------------
# UI-side has zero execution imports: architectural boundary smoke check
# ---------------------------------------------------------------------------


async def test_reopen_emits_frame_reopened_and_restores_active():
    """AreaManager.reopen flips terminal state to ACTIVE and records
    the transition as a FRAME_REOPENED event so replay sees it."""
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        provider = FakeProvider(_text_turn("done"))
        runner = SessionRunner(
            provider=provider, tool_registry=[AgentTool()],
            event_store=store, cwd="/tmp",
        )

        # Run a session to resolution
        result = await runner.run(
            session_id="s1", role=_role(), model="fake", user_input="go",
        )
        assert result.frame.state == FrameState.RESOLVED

        # Reopen the already-resolved frame
        runner.manager.reopen(result.frame)
        assert result.frame.state == FrameState.ACTIVE

        events = store.snapshot("s1")
        reopened = [e for e in events if e.type == EventType.FRAME_REOPENED]
        assert len(reopened) == 1
        assert reopened[0].payload["previous_state"] == "resolved"
        assert reopened[0].payload["reason"] == "resume"


async def test_reopen_is_noop_on_already_active_frame():
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        provider = FakeProvider(_text_turn("hello"))
        runner = SessionRunner(
            provider=provider, tool_registry=[AgentTool()],
            event_store=store, cwd="/tmp",
        )
        frame = runner.manager.open_root(
            session_id="s1", role=_role(), model="fake",
            initial_user_input="hi",
        )
        # Frame is ACTIVE right after open_root
        assert frame.state == FrameState.ACTIVE
        n_before = len(store.snapshot("s1"))
        runner.manager.reopen(frame)
        n_after = len(store.snapshot("s1"))
        # No event emitted because there was no terminal-state transition
        assert n_after == n_before


async def test_reconstruct_replays_frame_reopened_as_active():
    """After reopen + new turn, reconstruct should reach ACTIVE through
    REOPENED rather than infer it from a trailing MESSAGE_APPENDED."""
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        provider = FakeProvider([
            _text_turn("first"),
            _text_turn("second"),
        ])
        runner = SessionRunner(
            provider=provider, tool_registry=[AgentTool()],
            event_store=store, cwd="/tmp",
        )
        result = await runner.run(
            session_id="s1", role=_role(), model="fake", user_input="go",
        )
        runner.manager.reopen(result.frame)

        # Reconstruct mid-stream: right after FRAME_REOPENED, state
        # should be ACTIVE (not RESOLVED from the prior turn).
        from nature.events.reconstruct import reconstruct
        events = store.snapshot("s1")
        reopen_id = next(
            e.id for e in events if e.type == EventType.FRAME_REOPENED
        )
        replay = reconstruct("s1", store, up_to_event_id=reopen_id)
        assert replay.frames[result.frame.id].state == FrameState.ACTIVE


async def test_event_consumer_handler_error_does_not_kill_consumer():
    """A handler that raises must not abort the consumer loop."""
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        provider = FakeProvider(_text_turn("hi"))
        runner = SessionRunner(
            provider=provider, tool_registry=[AgentTool()],
            event_store=store, cwd="/tmp",
        )
        await runner.run(
            session_id="s1", role=_role(), model="fake", user_input="go",
        )

        seen: list[EventType] = []

        def good_handler(event):
            seen.append(event.type)

        def bad_handler(event):
            raise RuntimeError("boom — handler is broken")

        consumer = (
            EventConsumer()
            .on(EventType.FRAME_OPENED, bad_handler)  # crashes
            .on_any(good_handler)                      # still runs
        )

        # Replay should NOT raise even though one handler is broken
        await consumer.replay("s1", store)

        # The fallback handler still saw events from after FRAME_OPENED
        assert EventType.MESSAGE_APPENDED in seen


def test_event_consumer_module_does_not_import_frame_or_manager():
    """A read-only consumer must not import execution modules."""
    import nature.ui.event_consumer as consumer_mod

    source = consumer_mod.__file__
    with open(source, "r", encoding="utf-8") as f:
        text = f.read()

    forbidden = [
        "from nature.frame",
        "from nature.agent.llm_agent",
        "from nature.agent.executor",
        "from nature.orchestrator",
    ]
    for pattern in forbidden:
        assert pattern not in text, (
            f"event_consumer must not import execution module: {pattern}"
        )
