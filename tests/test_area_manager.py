"""Tests for AreaManager — the execution loop (Step 4)."""

from __future__ import annotations

import tempfile
from pathlib import Path


from nature.context import AgentRole
from nature.events import EventType, FileEventStore
from nature.events.reconstruct import reconstruct
from nature.frame import AreaManager, Frame, FrameState
from nature.protocols.message import (
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolUseContent,
    Usage,
)
from nature.config.constants import StopReason

from tests._fakes import FakeProvider, FakeTool


# ---------------------------------------------------------------------------
# Stream sequence builders
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


def _tool_turn(
    tool_name: str,
    tool_input: dict,
    tool_id: str = "toolu_1",
) -> list[StreamEvent]:
    tool_block = ToolUseContent(id=tool_id, name=tool_name, input=tool_input)
    return [
        StreamEvent(type=StreamEventType.MESSAGE_START),
        StreamEvent(
            type=StreamEventType.CONTENT_BLOCK_START,
            index=0,
            content_block=tool_block,
        ),
        StreamEvent(
            type=StreamEventType.CONTENT_BLOCK_STOP,
            index=0,
            content_block=tool_block,
        ),
        StreamEvent(
            type=StreamEventType.MESSAGE_STOP,
            usage=Usage(input_tokens=10, output_tokens=5),
            stop_reason=StopReason.TOOL_USE,
        ),
    ]


def _make_manager(
    store: FileEventStore,
    provider: FakeProvider,
    tools: list | None = None,
) -> AreaManager:
    return AreaManager(
        store=store,
        provider=provider,
        tool_registry=tools or [],
        cwd="/tmp",
    )


def _role(name: str = "receptionist", allowed_tools=None) -> AgentRole:
    return AgentRole(
        name=name,
        instructions=f"you are {name}",
        allowed_tools=allowed_tools,
    )


# ---------------------------------------------------------------------------
# open_root
# ---------------------------------------------------------------------------


async def test_open_root_creates_frame_and_emits_opened_and_user_input():
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        provider = FakeProvider(_text_turn("hi"))
        manager = _make_manager(store, provider)

        frame = manager.open_root(
            session_id="s1",
            role=_role(),
            model="fake",
            initial_user_input="hello",
        )

        assert isinstance(frame, Frame)
        assert frame.state == FrameState.ACTIVE
        assert frame.session_id == "s1"
        assert len(frame.context.body.conversation) == 1

        events = store.snapshot("s1")
        types = [e.type for e in events]
        # Canonical open sequence:
        #   FRAME_OPENED → HEADER_SNAPSHOT → USER_INPUT → MESSAGE_APPENDED
        assert types[0] == EventType.FRAME_OPENED
        assert types[1] == EventType.HEADER_SNAPSHOT
        assert types[2] == EventType.USER_INPUT
        assert types[3] == EventType.MESSAGE_APPENDED

        # FRAME_OPENED still carries the thin role header for UI listings
        opened = events[0]
        assert opened.payload["role_name"] == "receptionist"
        assert opened.payload["model"] == "fake"

        # HEADER_SNAPSHOT carries the full role — the authoritative
        # source for replay and cache-boundary calculations
        snap = events[1]
        assert snap.payload["role"]["name"] == "receptionist"
        assert snap.payload["role"]["instructions"] == "you are receptionist"
        assert snap.payload["principles"] == []


# ---------------------------------------------------------------------------
# run — text-only response
# ---------------------------------------------------------------------------


async def test_run_text_response_resolves_frame():
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        provider = FakeProvider(_text_turn("the answer is 42"))
        manager = _make_manager(store, provider)

        frame = manager.open_root(
            session_id="s1",
            role=_role(),
            model="fake",
            initial_user_input="what is the answer",
        )
        await manager.run(frame)

        assert frame.state == FrameState.RESOLVED

        events = store.snapshot("s1")
        types = [e.type for e in events]
        # Expected order: OPENED, USER_INPUT, MESSAGE_APPENDED(user),
        # LLM_REQUEST, LLM_RESPONSE, MESSAGE_APPENDED(assistant),
        # ANNOTATION_STORED, FRAME_RESOLVED
        assert EventType.LLM_REQUEST in types
        assert EventType.LLM_RESPONSE in types
        assert EventType.ANNOTATION_STORED in types
        assert types[-1] == EventType.FRAME_RESOLVED

        # Conversation has user + assistant
        msgs = frame.context.body.conversation.messages
        assert len(msgs) == 2
        assert msgs[0].from_ == "user"
        assert msgs[1].from_ == "receptionist"
        assert msgs[1].content[0].text == "the answer is 42"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# run — tool call, then resolve
# ---------------------------------------------------------------------------


async def test_run_tool_call_then_text_final():
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        provider = FakeProvider([
            _tool_turn("FakeRead", {"path": "x"}, tool_id="toolu_A"),
            _text_turn("done reading"),
        ])
        tools = [FakeTool("FakeRead", output="file body")]
        manager = _make_manager(store, provider, tools=tools)

        frame = manager.open_root(
            session_id="s1",
            role=_role(allowed_tools=["FakeRead"]),
            model="fake",
            initial_user_input="read the file",
        )
        await manager.run(frame)

        assert frame.state == FrameState.RESOLVED

        events = store.snapshot("s1")
        types = [e.type for e in events]

        assert EventType.TOOL_STARTED in types
        assert EventType.TOOL_COMPLETED in types

        # Tool started event carries input
        tool_started = next(e for e in events if e.type == EventType.TOOL_STARTED)
        assert tool_started.payload["tool_name"] == "FakeRead"
        assert tool_started.payload["tool_input"] == {"path": "x"}
        assert tool_started.payload["tool_use_id"] == "toolu_A"

        # Tool completed carries output
        tool_done = next(e for e in events if e.type == EventType.TOOL_COMPLETED)
        assert tool_done.payload["tool_name"] == "FakeRead"
        assert tool_done.payload["output"] == "file body"
        assert tool_done.payload["is_error"] is False

        # Provider was called twice (once for tool, once for final text)
        assert len(provider.requests) == 2


async def test_run_tool_results_appear_in_conversation():
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        provider = FakeProvider([
            _tool_turn("FakeRead", {"path": "x"}, tool_id="toolu_A"),
            _text_turn("done"),
        ])
        tools = [FakeTool("FakeRead", output="result body")]
        manager = _make_manager(store, provider, tools=tools)

        frame = manager.open_root(
            session_id="s1",
            role=_role(allowed_tools=["FakeRead"]),
            model="fake",
            initial_user_input="read the file",
        )
        await manager.run(frame)

        msgs = frame.context.body.conversation.messages
        # user, assistant(tool_use), tool_result, assistant(final text)
        assert len(msgs) == 4
        assert msgs[0].from_ == "user"
        assert msgs[1].from_ == "receptionist"
        assert msgs[2].from_ == "tool"
        assert msgs[2].to == "receptionist"
        assert msgs[3].from_ == "receptionist"


# ---------------------------------------------------------------------------
# Parallel tool batching (_execute_and_apply Phase A)
# ---------------------------------------------------------------------------


class _SafeFakeTool(FakeTool):
    """A FakeTool that declares itself concurrency-safe so the
    manager's `_execute_and_apply` can batch consecutive calls
    through `asyncio.gather`."""

    def is_concurrency_safe(self, input: dict) -> bool:  # type: ignore[override]
        return True

    def is_read_only(self, input: dict) -> bool:  # type: ignore[override]
        return True


def _tool_turn_multi(
    calls: list[tuple[str, dict, str]],
) -> list[StreamEvent]:
    """Build a stream sequence with multiple tool_use blocks in one turn.

    Each `(tool_name, tool_input, tool_id)` tuple becomes a separate
    CONTENT_BLOCK_START/STOP pair at the right block index, so the
    resulting assistant message has N tool_use blocks — the same shape
    a real Claude model produces when it wants parallel reads.
    """
    events = [StreamEvent(type=StreamEventType.MESSAGE_START)]
    for idx, (name, inp, tid) in enumerate(calls):
        block = ToolUseContent(id=tid, name=name, input=inp)
        events.append(StreamEvent(
            type=StreamEventType.CONTENT_BLOCK_START,
            index=idx,
            content_block=block,
        ))
        events.append(StreamEvent(
            type=StreamEventType.CONTENT_BLOCK_STOP,
            index=idx,
            content_block=block,
        ))
    events.append(StreamEvent(
        type=StreamEventType.MESSAGE_STOP,
        usage=Usage(input_tokens=10, output_tokens=5),
        stop_reason=StopReason.TOOL_USE,
    ))
    return events


async def test_run_parallel_batch_emits_bracket_events_and_ordering():
    """When the model emits 3 consecutive concurrency-safe tool_use
    blocks in a single turn, the manager should bracket them with
    PARALLEL_GROUP_STARTED / PARALLEL_GROUP_COMPLETED and tag each
    inner TOOL_STARTED/COMPLETED with the same parallel_group_id."""
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        provider = FakeProvider([
            _tool_turn_multi([
                ("FakeRead", {"path": "a"}, "tu_a"),
                ("FakeRead", {"path": "b"}, "tu_b"),
                ("FakeRead", {"path": "c"}, "tu_c"),
            ]),
            _text_turn("done"),
        ])
        tools = [_SafeFakeTool("FakeRead", output="body")]
        manager = _make_manager(store, provider, tools=tools)

        frame = manager.open_root(
            session_id="s1",
            role=_role(allowed_tools=["FakeRead"]),
            model="fake",
            initial_user_input="read three things at once",
        )
        await manager.run(frame)

        assert frame.state == FrameState.RESOLVED

        events = store.snapshot("s1")

        # Exactly one bracket pair emitted for the batch of 3.
        starts = [e for e in events if e.type == EventType.PARALLEL_GROUP_STARTED]
        ends = [e for e in events if e.type == EventType.PARALLEL_GROUP_COMPLETED]
        assert len(starts) == 1
        assert len(ends) == 1
        start_ev = starts[0]
        end_ev = ends[0]
        group_id = start_ev.payload["group_id"]
        assert start_ev.payload["tool_count"] == 3
        assert end_ev.payload["group_id"] == group_id

        # The bracket and its inner events all share the same
        # parallel_group_id on their envelope.
        inside_ids = [
            e.id for e in events
            if e.parallel_group_id == group_id
        ]
        # 1 start + 3 tool_started + 3 tool_completed + 1 end = 8
        assert len(inside_ids) == 8
        assert min(inside_ids) == start_ev.id
        assert max(inside_ids) == end_ev.id
        # And no gaps — it's a contiguous block in the event log.
        assert inside_ids == list(range(start_ev.id, end_ev.id + 1))

        # Every TOOL_STARTED / TOOL_COMPLETED inside carries group_id.
        inner_tool_events = [
            e for e in events
            if e.type in (EventType.TOOL_STARTED, EventType.TOOL_COMPLETED)
            and e.parallel_group_id == group_id
        ]
        assert len(inner_tool_events) == 6

        # Events outside the batch are NOT tagged.
        outside = [e for e in events if e.parallel_group_id is None]
        assert EventType.USER_INPUT in {e.type for e in outside}
        assert EventType.FRAME_OPENED in {e.type for e in outside}


async def test_run_single_tool_has_no_parallel_bracket():
    """A single tool_use in one turn stays on the non-batch path —
    no PARALLEL_GROUP_* events are emitted."""
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        provider = FakeProvider([
            _tool_turn("FakeRead", {"path": "a"}, tool_id="tu_a"),
            _text_turn("done"),
        ])
        tools = [_SafeFakeTool("FakeRead", output="body")]
        manager = _make_manager(store, provider, tools=tools)

        frame = manager.open_root(
            session_id="s1",
            role=_role(allowed_tools=["FakeRead"]),
            model="fake",
            initial_user_input="read one file",
        )
        await manager.run(frame)

        events = store.snapshot("s1")
        assert all(
            e.type not in (
                EventType.PARALLEL_GROUP_STARTED,
                EventType.PARALLEL_GROUP_COMPLETED,
            )
            for e in events
        )
        assert all(e.parallel_group_id is None for e in events)


async def test_parallel_batch_preserves_tool_result_order():
    """tool_result blocks must come back in the same order as the
    original tool_use blocks, even though gather resolves in
    completion order. The final assistant message's tool_result
    envelope is the shape the model sees."""
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        provider = FakeProvider([
            _tool_turn_multi([
                ("FakeRead", {"path": "first"}, "tu_1"),
                ("FakeRead", {"path": "second"}, "tu_2"),
                ("FakeRead", {"path": "third"}, "tu_3"),
            ]),
            _text_turn("done"),
        ])
        tools = [_SafeFakeTool("FakeRead", output="body")]
        manager = _make_manager(store, provider, tools=tools)

        frame = manager.open_root(
            session_id="s1",
            role=_role(allowed_tools=["FakeRead"]),
            model="fake",
            initial_user_input="read three",
        )
        await manager.run(frame)

        msgs = frame.context.body.conversation.messages
        # Find the tool_result message and check its blocks are in
        # the same order as the original tool_use ids.
        tool_msg = next(m for m in msgs if m.from_ == "tool")
        from nature.protocols.message import ToolResultContent
        tr_ids = [
            b.tool_use_id for b in tool_msg.content
            if isinstance(b, ToolResultContent)
        ]
        assert tr_ids == ["tu_1", "tu_2", "tu_3"]


# ---------------------------------------------------------------------------
# Tool filtering
# ---------------------------------------------------------------------------


async def test_allowed_tools_filter_respected_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        provider = FakeProvider(_text_turn("ok"))
        registry = [
            FakeTool("Read"),
            FakeTool("Bash"),
            FakeTool("Write"),
        ]
        manager = _make_manager(store, provider, tools=registry)

        frame = manager.open_root(
            session_id="s1",
            role=_role(allowed_tools=["Read"]),
            model="fake",
            initial_user_input="go",
        )
        await manager.run(frame)

        # Provider request should see only Read
        assert provider.last_request is not None
        tool_defs = provider.last_request.tools
        assert tool_defs is not None
        assert {t.name for t in tool_defs} == {"Read"}


# ---------------------------------------------------------------------------
# Error path
# ---------------------------------------------------------------------------


class _BrokenProvider(FakeProvider):
    async def stream_request(self, request):  # type: ignore[override]
        self.requests.append(request)
        self.last_request = request
        raise RuntimeError("network boom")
        yield  # pragma: no cover — required for async-generator signature


class _StallingProvider(FakeProvider):
    """Provider that opens a stream and then hangs forever."""

    async def stream_request(self, request):  # type: ignore[override]
        self.requests.append(request)
        self.last_request = request
        import asyncio as _asyncio
        yield StreamEvent(type=StreamEventType.MESSAGE_START)
        await _asyncio.sleep(60)  # pragma: no cover — timeout fires first


async def test_llm_timeout_transitions_frame_to_error_state(monkeypatch):
    """A hung LLM stream must be caught by the timeout wrapper and
    turned into LLM_ERROR + FRAME_ERRORED, matching the self-heal
    contract AreaManager already handles for other exception types."""
    import nature.agent.llm_agent as la

    monkeypatch.setattr(la, "DEFAULT_LLM_TIMEOUT_SECONDS", 0.2)

    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        provider = _StallingProvider(_text_turn("unused"))
        manager = _make_manager(store, provider)

        frame = manager.open_root(
            session_id="s1", role=_role(), model="fake-stall",
            initial_user_input="go",
        )
        await manager.run(frame)

        # Frame transitioned to ERROR — self-heal path engaged
        assert frame.state == FrameState.ERROR

        events = store.snapshot("s1")
        types = [e.type for e in events]
        assert EventType.LLM_ERROR in types
        assert EventType.FRAME_ERRORED in types

        # LLM_ERROR carries the domain exception type so UIs can
        # recognize timeouts vs generic provider errors
        err = next(e for e in events if e.type == EventType.LLM_ERROR)
        assert err.payload["error_type"] == "LLMCallTimeout"
        assert "fake-stall" in err.payload["message"]


async def test_provider_exception_marks_frame_error_and_emits_llm_error():
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        provider = _BrokenProvider(_text_turn("unused"))
        manager = _make_manager(store, provider)

        frame = manager.open_root(
            session_id="s1",
            role=_role(),
            model="fake",
            initial_user_input="go",
        )
        await manager.run(frame)

        assert frame.state == FrameState.ERROR

        types = [e.type for e in store.snapshot("s1")]
        assert EventType.LLM_ERROR in types

        error_event = next(
            e for e in store.snapshot("s1") if e.type == EventType.LLM_ERROR
        )
        assert error_event.payload["error_type"] == "RuntimeError"
        assert "network boom" in error_event.payload["message"]


# ---------------------------------------------------------------------------
# Close
# ---------------------------------------------------------------------------


async def test_close_emits_frame_closed_and_sets_state():
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        provider = FakeProvider(_text_turn("bye"))
        manager = _make_manager(store, provider)

        frame = manager.open_root(
            session_id="s1", role=_role(), model="fake",
            initial_user_input="hi",
        )
        await manager.run(frame)
        manager.close(frame)

        assert frame.state == FrameState.CLOSED
        types = [e.type for e in store.snapshot("s1")]
        assert types[-1] == EventType.FRAME_CLOSED


# ---------------------------------------------------------------------------
# reconstruct — replay end-to-end
# ---------------------------------------------------------------------------


async def test_reconstruct_rebuilds_frame_from_events():
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        provider = FakeProvider([
            _tool_turn("FakeRead", {"path": "x"}, tool_id="toolu_A"),
            _text_turn("analysis complete"),
        ])
        tools = [FakeTool("FakeRead", output="file")]
        manager = _make_manager(store, provider, tools=tools)

        frame = manager.open_root(
            session_id="s1",
            role=_role(allowed_tools=["FakeRead"]),
            model="fake-model",
            initial_user_input="analyze",
        )
        await manager.run(frame)
        manager.close(frame)

        # Reconstruct from events in a fresh store instance
        fresh_store = FileEventStore(Path(tmp))
        replay = reconstruct("s1", fresh_store)

        assert len(replay.frames) == 1
        rebuilt = replay.frames[frame.id]

        assert rebuilt.state == FrameState.CLOSED
        assert rebuilt.self_actor == "receptionist"
        assert rebuilt.model == "fake-model"

        # Same conversation shape
        rebuilt_msgs = rebuilt.context.body.conversation.messages
        live_msgs = frame.context.body.conversation.messages
        assert len(rebuilt_msgs) == len(live_msgs)
        for a, b in zip(rebuilt_msgs, live_msgs):
            assert a.from_ == b.from_
            assert a.to == b.to
            assert len(a.content) == len(b.content)

        # Annotations replayed too
        assistant_msg = live_msgs[1]  # first receptionist output
        assert assistant_msg.id in replay.annotations
        assert replay.annotations[assistant_msg.id][0].stop_reason is not None


async def test_reconstruct_handles_error_state():
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        provider = _BrokenProvider([])
        manager = _make_manager(store, provider)

        frame = manager.open_root(
            session_id="s1", role=_role(), model="fake",
            initial_user_input="go",
        )
        await manager.run(frame)

        # Both the trace event and the state-transition event should exist
        types = [e.type for e in store.snapshot("s1")]
        assert EventType.LLM_ERROR in types
        assert EventType.FRAME_ERRORED in types

        replay = reconstruct("s1", store)
        rebuilt = replay.frames[frame.id]
        assert rebuilt.state == FrameState.ERROR


async def test_message_appended_timestamps_are_single_clock():
    """For every MESSAGE_APPENDED event, the event's timestamp and the
    payload's message timestamp should be identical — they come from a
    single logical clock reading."""
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        provider = FakeProvider([
            _tool_turn("FakeRead", {"path": "x"}, tool_id="toolu_a"),
            _text_turn("done"),
        ])
        tools = [FakeTool("FakeRead", output="result")]
        manager = _make_manager(store, provider, tools=tools)

        frame = manager.open_root(
            session_id="s1",
            role=_role(allowed_tools=["FakeRead"]),
            model="fake",
            initial_user_input="go",
        )
        await manager.run(frame)

        msg_events = [
            e for e in store.snapshot("s1")
            if e.type == EventType.MESSAGE_APPENDED
        ]
        assert len(msg_events) >= 3  # user, assistant(tool_use), tool_result, final
        for e in msg_events:
            assert e.timestamp == e.payload["timestamp"], (
                f"event {e.id}: Event.timestamp={e.timestamp} "
                f"!= payload.timestamp={e.payload['timestamp']}"
            )


async def test_reconstruct_restores_full_header_via_snapshot():
    """HEADER_SNAPSHOT should round-trip role (incl. per-role model)."""
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        provider = FakeProvider(_text_turn("ok"))
        manager = _make_manager(store, provider)

        role = AgentRole(
            name="core",
            description="the primary solver",
            instructions="think hard",
            allowed_tools=["FakeRead", "FakeWrite"],
            model="per-role-model",
        )
        frame = manager.open_root(
            session_id="s1", role=role, model="frame-model",
            initial_user_input="go",
        )
        await manager.run(frame)

        replay = reconstruct("s1", store)
        rebuilt = replay.frames[frame.id]
        rr = rebuilt.context.header.role
        assert rr.name == "core"
        assert rr.description == "the primary solver"
        assert rr.instructions == "think hard"
        assert rr.allowed_tools == ["FakeRead", "FakeWrite"]
        assert rr.model == "per-role-model"
        assert rebuilt.model == "frame-model"


async def test_reconstruct_reports_incomplete_llm_span_after_crash():
    """If a session crashed mid-turn (LLM_REQUEST with no matching
    LLM_RESPONSE), reconstruct should surface it in incomplete_spans."""
    from nature.events.types import Event
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        provider = FakeProvider(_text_turn("done"))
        manager = _make_manager(store, provider)

        frame = manager.open_root(
            session_id="s1", role=_role(), model="fake", initial_user_input="go",
        )
        await manager.run(frame)

        # Full replay: nothing incomplete on a clean session
        clean = reconstruct("s1", store)
        assert clean.incomplete_spans == []

        # Simulate a crash: slice the stream right after LLM_REQUEST
        events = store.snapshot("s1")
        llm_req_id = next(
            e.id for e in events if e.type == EventType.LLM_REQUEST
        )
        sliced = reconstruct("s1", store, up_to_event_id=llm_req_id)
        assert len(sliced.incomplete_spans) == 1
        span = sliced.incomplete_spans[0]
        assert span.kind == "llm_request"
        assert span.frame_id == frame.id
        assert span.identifier  # request_id is non-empty
        assert span.started_event_id == llm_req_id


async def test_reconstruct_reports_incomplete_tool_span_after_crash():
    """TOOL_STARTED with no matching TOOL_COMPLETED is reported the
    same way. Useful when a tool execution was in-flight at crash time."""
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        provider = FakeProvider([
            _tool_turn("FakeRead", {"path": "/x"}, tool_id="toolu_crashy"),
            _text_turn("unused"),
        ])
        tools = [FakeTool("FakeRead", output="body")]
        manager = _make_manager(store, provider, tools=tools)

        frame = manager.open_root(
            session_id="s1",
            role=_role(allowed_tools=["FakeRead"]),
            model="fake",
            initial_user_input="go",
        )
        await manager.run(frame)

        events = store.snapshot("s1")
        tool_start_id = next(
            e.id for e in events if e.type == EventType.TOOL_STARTED
        )
        # Slice to just after TOOL_STARTED: the tool is mid-flight
        sliced = reconstruct("s1", store, up_to_event_id=tool_start_id)
        assert len(sliced.incomplete_spans) == 1
        span = sliced.incomplete_spans[0]
        assert span.kind == "tool_use"
        assert span.identifier == "toolu_crashy"
        assert span.payload.get("tool_name") == "FakeRead"


async def test_reconstruct_up_to_event_id_slices_replay():
    """`up_to_event_id` should replay only events with id ≤ the limit,
    giving callers a deterministic rewind point."""
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        provider = FakeProvider([
            _tool_turn("FakeRead", {"path": "x"}, tool_id="toolu_A"),
            _text_turn("done reading"),
        ])
        tools = [FakeTool("FakeRead", output="file body")]
        manager = _make_manager(store, provider, tools=tools)

        frame = manager.open_root(
            session_id="s1",
            role=_role(allowed_tools=["FakeRead"]),
            model="fake",
            initial_user_input="go",
        )
        await manager.run(frame)

        all_events = store.snapshot("s1")
        total = len(all_events)

        # Full replay (no limit) == explicit limit at last id
        full = reconstruct("s1", store)
        explicit_full = reconstruct("s1", store, up_to_event_id=total)
        assert list(full.frames.keys()) == list(explicit_full.frames.keys())
        assert (
            len(full.frames[frame.id].context.body.conversation.messages)
            == len(explicit_full.frames[frame.id].context.body.conversation.messages)
        )

        # Replay up to the first MESSAGE_APPENDED (user input landing on the frame)
        first_msg_id = next(
            e.id for e in all_events if e.type == EventType.MESSAGE_APPENDED
        )
        partial = reconstruct("s1", store, up_to_event_id=first_msg_id)
        p_frame = partial.frames[frame.id]
        msgs = p_frame.context.body.conversation.messages
        # Exactly one message (the user input), header already restored
        assert len(msgs) == 1
        assert msgs[0].from_ == "user"
        assert p_frame.context.header.role.name == "receptionist"
        # Frame is still ACTIVE at this point (FRAME_RESOLVED came later)
        assert p_frame.state == FrameState.ACTIVE

        # Replay up to id=0 returns empty
        empty = reconstruct("s1", store, up_to_event_id=0)
        assert empty.frames == {}

        # Replay with a limit beyond the end == full replay
        beyond = reconstruct("s1", store, up_to_event_id=total + 1000)
        assert len(beyond.frames) == len(full.frames)


async def test_reconstruct_ignores_trace_events():
    """Invariant: dropping every TRACE event must yield an identical replay."""
    from nature.events.types import EVENT_CATEGORIES, EventCategory

    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        provider = FakeProvider([
            _tool_turn("FakeRead", {"path": "x"}, tool_id="toolu_A"),
            _text_turn("done"),
        ])
        tools = [FakeTool("FakeRead", output="file body")]
        manager = _make_manager(store, provider, tools=tools)

        frame = manager.open_root(
            session_id="s1",
            role=_role(allowed_tools=["FakeRead"]),
            model="fake",
            initial_user_input="analyze",
        )
        await manager.run(frame)
        manager.close(frame)

        replay_full = reconstruct("s1", store)
        rebuilt_full = replay_full.frames[frame.id]

        # Hand-filter to state-transition events only, replay against
        # reconstruct's internal apply path to confirm same state.
        from nature.events.reconstruct import ReplayResult, _apply_event
        trace_types = {
            et for et, cat in EVENT_CATEGORIES.items()
            if cat is EventCategory.TRACE
        }
        filtered = [
            e for e in store.snapshot("s1") if e.type not in trace_types
        ]
        filtered_result = ReplayResult()
        for ev in filtered:
            _apply_event(ev, filtered_result)
        rebuilt_filtered = filtered_result.frames[frame.id]

        # Frame state + conversation must match exactly
        assert rebuilt_filtered.state == rebuilt_full.state
        full_msgs = rebuilt_full.context.body.conversation.messages
        filt_msgs = rebuilt_filtered.context.body.conversation.messages
        assert len(full_msgs) == len(filt_msgs)
        for a, b in zip(full_msgs, filt_msgs):
            assert a.from_ == b.from_ and a.to == b.to
            assert len(a.content) == len(b.content)


# ---------------------------------------------------------------------------
# _emit → OnEvent Listener dispatch (framework wiring)
# ---------------------------------------------------------------------------


async def test_emit_fires_onevent_listeners():
    """A Listener on OnEvent(FRAME_OPENED) should fire when the frame
    is opened — proves _emit now dispatches OnEvent interventions."""
    from nature.packs.registry import PackRegistry
    from nature.packs.types import (
        Intervention, InterventionContext, OnEvent,
    )

    reg = PackRegistry()
    fired: list[EventType] = []
    reg.register_intervention(Intervention(
        id="t.open_listener",
        kind="listener",
        trigger=OnEvent(event_type=EventType.FRAME_OPENED),
        action=lambda ctx: fired.append(EventType.FRAME_OPENED) or [],
    ))

    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        provider = FakeProvider(_text_turn("hi"))
        manager = AreaManager(
            store=store, provider=provider, tool_registry=[],
            cwd="/tmp", pack_registry=reg,
        )
        manager.open_root(
            session_id="s1", role=_role(), model="fake",
            initial_user_input="hi",
        )

    assert fired == [EventType.FRAME_OPENED]


async def test_emit_applies_emit_event_effects_recursively():
    """A Listener returning EmitEvent triggers another _emit cycle.
    Proves the cascade loop works and feeds subsequent Listeners."""
    from nature.packs.registry import PackRegistry
    from nature.packs.types import (
        EmitEvent, Intervention, OnEvent,
    )

    reg = PackRegistry()
    observed: list[EventType] = []

    def on_opened(ctx):
        # React to FRAME_OPENED by emitting EDIT_MISS
        return [EmitEvent(event_type=EventType.EDIT_MISS, payload={})]

    def on_edit_miss(ctx):
        observed.append(EventType.EDIT_MISS)
        return []

    reg.register_intervention(Intervention(
        id="t.cascade_source",
        kind="listener",
        trigger=OnEvent(event_type=EventType.FRAME_OPENED),
        action=on_opened,
    ))
    reg.register_intervention(Intervention(
        id="t.cascade_observer",
        kind="listener",
        trigger=OnEvent(event_type=EventType.EDIT_MISS),
        action=on_edit_miss,
    ))

    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        provider = FakeProvider(_text_turn("hi"))
        manager = AreaManager(
            store=store, provider=provider, tool_registry=[],
            cwd="/tmp", pack_registry=reg,
        )
        manager.open_root(
            session_id="s1", role=_role(), model="fake",
            initial_user_input="hi",
        )

        assert observed == [EventType.EDIT_MISS]
        # The downstream event is persisted alongside FRAME_OPENED.
        types = [e.type for e in store.snapshot("s1")]
        assert EventType.EDIT_MISS in types


async def test_onframe_opened_fires_on_open_root():
    """OnFrame(OPENED) should fire for the root frame's open path."""
    from nature.packs.registry import PackRegistry
    from nature.packs.types import (
        FramePhase, Intervention, OnFrame,
    )

    reg = PackRegistry()
    seen: list[str] = []
    reg.register_intervention(Intervention(
        id="t.frame_opened",
        kind="listener",
        trigger=OnFrame(phase=FramePhase.OPENED),
        action=lambda ctx: seen.append(ctx.frame.id) or [],
    ))

    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        provider = FakeProvider(_text_turn("hi"))
        manager = AreaManager(
            store=store, provider=provider, tool_registry=[],
            cwd="/tmp", pack_registry=reg,
        )
        frame = manager.open_root(
            session_id="s1", role=_role(), model="fake",
            initial_user_input="hi",
        )
        assert seen == [frame.id]


async def test_onframe_resolved_and_closed_fire_end_to_end():
    """Full lifecycle: OPENED (open_root) → RESOLVED (run resolves) →
    CLOSED (close). All three phases fire exactly once."""
    from nature.packs.registry import PackRegistry
    from nature.packs.types import (
        FramePhase, Intervention, OnFrame,
    )

    reg = PackRegistry()
    seen: list[FramePhase] = []

    def record(phase):
        return lambda ctx: seen.append(phase) or []

    for phase in (FramePhase.OPENED, FramePhase.RESOLVED, FramePhase.CLOSED):
        reg.register_intervention(Intervention(
            id=f"t.{phase.value}",
            kind="listener",
            trigger=OnFrame(phase=phase),
            action=record(phase),
        ))

    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        provider = FakeProvider(_text_turn("done"))
        manager = AreaManager(
            store=store, provider=provider, tool_registry=[],
            cwd="/tmp", pack_registry=reg,
        )
        frame = manager.open_root(
            session_id="s1", role=_role(), model="fake",
            initial_user_input="hi",
        )
        await manager.run(frame)
        manager.close(frame)

    assert seen == [
        FramePhase.OPENED,
        FramePhase.RESOLVED,
        FramePhase.CLOSED,
    ]


async def test_onframe_errored_fires_on_provider_exception():
    """A provider that raises mid-run should trigger OnFrame(ERRORED)."""
    from nature.packs.registry import PackRegistry
    from nature.packs.types import (
        FramePhase, Intervention, OnFrame,
    )

    reg = PackRegistry()
    seen: list[FramePhase] = []
    reg.register_intervention(Intervention(
        id="t.frame_errored",
        kind="listener",
        trigger=OnFrame(phase=FramePhase.ERRORED),
        action=lambda ctx: seen.append(FramePhase.ERRORED) or [],
    ))

    class _ExplodingProvider:
        async def stream_request(self, request):  # noqa: D401
            raise RuntimeError("kaboom")
            if False:  # pragma: no cover
                yield None
        async def stream(self, *a, **kw):  # pragma: no cover
            raise RuntimeError("kaboom")
            if False:
                yield None
        async def count_tokens(self, *a, **kw):
            return 0
        @property
        def model_id(self):
            return "fake"

    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        manager = AreaManager(
            store=store, provider=_ExplodingProvider(), tool_registry=[],
            cwd="/tmp", pack_registry=reg,
        )
        frame = manager.open_root(
            session_id="s1", role=_role(), model="fake",
            initial_user_input="trigger",
        )
        await manager.run(frame)

    assert seen == [FramePhase.ERRORED]


async def test_onllm_pre_and_post_fire_around_llm_call():
    """Both OnLLM(PRE) and OnLLM(POST) listeners should fire, and
    POST must see the frame after the LLM request/response pair has
    been emitted."""
    from nature.packs.registry import PackRegistry
    from nature.packs.types import (
        Intervention, LLMPhase, OnLLM,
    )

    reg = PackRegistry()
    sequence: list[str] = []
    reg.register_intervention(Intervention(
        id="t.llm_pre",
        kind="listener",
        trigger=OnLLM(phase=LLMPhase.PRE),
        action=lambda ctx: sequence.append("pre") or [],
    ))
    reg.register_intervention(Intervention(
        id="t.llm_post",
        kind="listener",
        trigger=OnLLM(phase=LLMPhase.POST),
        action=lambda ctx: sequence.append("post") or [],
    ))

    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        provider = FakeProvider(_text_turn("ack"))
        manager = AreaManager(
            store=store, provider=provider, tool_registry=[],
            cwd="/tmp", pack_registry=reg,
        )
        frame = manager.open_root(
            session_id="s1", role=_role(), model="fake",
            initial_user_input="hi",
        )
        await manager.run(frame)

    # Exactly one PRE and one POST for the single-turn exchange.
    assert sequence == ["pre", "post"]


async def test_emit_depth_guard_stops_runaway_cascade():
    """A Listener that unconditionally re-emits the same event must
    not recurse past `_EMIT_MAX_DEPTH`."""
    from nature.packs.registry import PackRegistry
    from nature.packs.types import (
        EmitEvent, Intervention, OnEvent,
    )

    reg = PackRegistry()
    call_count = {"n": 0}

    def runaway(ctx):
        call_count["n"] += 1
        return [EmitEvent(event_type=EventType.LOOP_DETECTED, payload={})]

    reg.register_intervention(Intervention(
        id="t.runaway",
        kind="listener",
        trigger=OnEvent(event_type=EventType.LOOP_DETECTED),
        action=runaway,
    ))

    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        provider = FakeProvider(_text_turn("hi"))
        manager = AreaManager(
            store=store, provider=provider, tool_registry=[],
            cwd="/tmp", pack_registry=reg,
        )
        # Seeding LOOP_DETECTED manually so the cascade kicks off.
        from nature.events.payloads import _PayloadBase
        frame = manager.open_root(
            session_id="s1", role=_role(), model="fake",
            initial_user_input="hi",
        )
        manager._emit(frame, EventType.LOOP_DETECTED, _PayloadBase())

    # Bounded by _EMIT_MAX_DEPTH — listener fires at most that many times.
    assert call_count["n"] <= manager._EMIT_MAX_DEPTH
