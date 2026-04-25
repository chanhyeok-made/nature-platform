"""Tests for multi-agent delegation via AreaManager.open_child (Step 5)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from nature.context import AgentRole
from nature.events import EventType, FileEventStore
from nature.events.reconstruct import reconstruct
from nature.frame import AgentTool, AreaManager, Frame, FrameState
from nature.protocols.message import (
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolResultContent,
    ToolUseContent,
    Usage,
)
from nature.config.constants import StopReason

from tests._fakes import FakeProvider, FakeTool


# ---------------------------------------------------------------------------
# Stream builders
# ---------------------------------------------------------------------------


def _text_turn(text: str, stop: str = StopReason.END_TURN) -> list[StreamEvent]:
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
            stop_reason=stop,
        ),
    ]


def _tool_turn(
    tool_name: str,
    tool_input: dict,
    tool_id: str = "toolu_x",
) -> list[StreamEvent]:
    block = ToolUseContent(id=tool_id, name=tool_name, input=tool_input)
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


def _two_tool_turn(
    calls: list[tuple[str, dict, str]],
) -> list[StreamEvent]:
    """Stream with two tool_use blocks in one assistant message."""
    events: list[StreamEvent] = [StreamEvent(type=StreamEventType.MESSAGE_START)]
    for idx, (name, inp, tid) in enumerate(calls):
        block = ToolUseContent(id=tid, name=name, input=inp)
        events += [
            StreamEvent(
                type=StreamEventType.CONTENT_BLOCK_START,
                index=idx,
                content_block=block,
            ),
            StreamEvent(
                type=StreamEventType.CONTENT_BLOCK_STOP,
                index=idx,
                content_block=block,
            ),
        ]
    events.append(
        StreamEvent(
            type=StreamEventType.MESSAGE_STOP,
            usage=Usage(input_tokens=10, output_tokens=5),
            stop_reason=StopReason.TOOL_USE,
        )
    )
    return events


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manager(
    store: FileEventStore,
    provider: FakeProvider,
    tools: list | None = None,
    role_resolver=None,
) -> AreaManager:
    return AreaManager(
        store=store,
        provider=provider,
        tool_registry=tools or [AgentTool()],
        cwd="/tmp",
        role_resolver=role_resolver,
    )


def _role(name: str = "receptionist") -> AgentRole:
    return AgentRole(
        name=name,
        instructions=f"you are {name}",
        allowed_tools=None,
    )


# ---------------------------------------------------------------------------
# AgentTool schema
# ---------------------------------------------------------------------------


def test_agent_tool_schema_exposes_name_and_inputs():
    tool = AgentTool()
    assert tool.name == "Agent"
    assert "sub-agent" in tool.description.lower() or "delegate" in tool.description.lower()

    schema = tool.input_schema
    assert "properties" in schema
    assert "prompt" in schema["properties"]
    assert "name" in schema["properties"]


async def test_agent_tool_run_directly_returns_error_sentinel():
    """If AreaManager fails to intercept, calling execute should fail loudly."""
    from nature.protocols.tool import ToolContext
    tool = AgentTool()
    result = await tool.execute(
        {"prompt": "hi", "name": "core"},
        ToolContext(cwd="/tmp"),
    )
    assert result.is_error is True
    assert "intercept" in result.output.lower()


# ---------------------------------------------------------------------------
# open_child
# ---------------------------------------------------------------------------


async def test_open_child_creates_frame_with_parent_link_and_fresh_context():
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        provider = FakeProvider(_text_turn("ok"))
        manager = _make_manager(store, provider)

        parent = manager.open_root(
            session_id="s1",
            role=_role("receptionist"),
            model="fake",
            initial_user_input="go",
        )

        child = manager.open_child(
            parent=parent,
            role=_role("core"),
            initial_input="please plan",
        )

        assert child.parent_id == parent.id
        assert child.session_id == parent.session_id
        assert child.id in parent.children_ids
        assert child.state == FrameState.ACTIVE
        assert child.self_actor == "core"

        # Fresh body: only the delegation message, not parent's messages
        assert len(child.context.body.conversation) == 1
        delegation = child.context.body.conversation.messages[0]
        assert delegation.from_ == "receptionist"
        assert delegation.to == "core"
        assert delegation.content[0].text == "please plan"  # type: ignore[attr-defined]

        # FRAME_OPENED event carries parent_id
        events = store.snapshot("s1")
        opened = [e for e in events if e.type == EventType.FRAME_OPENED]
        assert len(opened) == 2  # root + child
        child_opened = opened[1]
        assert child_opened.payload["parent_id"] == parent.id
        assert child_opened.payload["role_name"] == "core"


# ---------------------------------------------------------------------------
# End-to-end delegation
# ---------------------------------------------------------------------------


async def test_agent_tool_call_triggers_delegation_end_to_end():
    """Parent calls Agent → AreaManager opens child → child resolves →
    result bubbles back → parent resolves."""
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        provider = FakeProvider([
            _tool_turn(
                "Agent",
                {"prompt": "find the bug", "name": "researcher"},
                tool_id="toolu_deleg",
            ),
            _text_turn("found the bug in module X"),  # child's response
            _text_turn("summary: bug is in module X"),  # parent's final response
        ])
        manager = _make_manager(store, provider)

        frame = manager.open_root(
            session_id="s1",
            role=_role("receptionist"),
            model="fake",
            initial_user_input="analyze the code",
        )
        await manager.run(frame)

        assert frame.state == FrameState.RESOLVED

        events = store.snapshot("s1")
        types = [e.type for e in events]

        # At least two FRAME_OPENED (root + child) and two FRAME_RESOLVED
        assert types.count(EventType.FRAME_OPENED) == 2
        assert types.count(EventType.FRAME_RESOLVED) == 2

        # Child frame was closed
        assert types.count(EventType.FRAME_CLOSED) == 1

        # Provider was called 3 times (parent turn 1, child, parent turn 2)
        assert len(provider.requests) == 3


async def test_parallel_delegation_batch_opens_children_concurrently():
    """Three consecutive Agent tool_use blocks in one assistant turn
    run via asyncio.gather — we should see PARALLEL_GROUP_STARTED /
    COMPLETED around the whole batch, each Agent's TOOL_STARTED /
    COMPLETED tagged with the same parallel_group_id, three child
    FRAME_OPENED events, and tool_results reassembled in input order.
    """
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        provider = FakeProvider([
            # Parent's first turn emits 3 Agent calls simultaneously.
            _two_tool_turn([
                ("Agent", {"prompt": "find A", "name": "researcher"}, "tu_a"),
                ("Agent", {"prompt": "find B", "name": "researcher"}, "tu_b"),
                ("Agent", {"prompt": "find C", "name": "researcher"}, "tu_c"),
            ]),
            # Each child just responds with its own identifying text.
            _text_turn("child A done"),
            _text_turn("child B done"),
            _text_turn("child C done"),
            # Parent wraps up.
            _text_turn("all three done"),
        ])
        manager = _make_manager(store, provider)

        frame = manager.open_root(
            session_id="s1",
            role=_role("receptionist"),
            model="fake",
            initial_user_input="split into three",
        )
        await manager.run(frame)

        assert frame.state == FrameState.RESOLVED

        events = store.snapshot("s1")

        # Exactly one parallel bracket pair for the delegation batch.
        starts = [e for e in events if e.type == EventType.PARALLEL_GROUP_STARTED]
        ends = [e for e in events if e.type == EventType.PARALLEL_GROUP_COMPLETED]
        assert len(starts) == 1
        assert len(ends) == 1
        group_id = starts[0].payload["group_id"]
        assert starts[0].payload["tool_count"] == 3
        assert ends[0].payload["group_id"] == group_id

        # The whole parallel region is between STARTED.id and
        # COMPLETED.id, and includes every event the children emitted
        # on their own frame_ids (FRAME_OPENED/CLOSED/MESSAGE_APPENDED
        # etc.) simply by virtue of falling in that id range — fork
        # validation catches all of them via the id range check alone.
        bracket_range = range(starts[0].id + 1, ends[0].id)
        assert len(list(bracket_range)) > 0
        child_frames_opened_in_range = [
            e for e in events
            if e.type == EventType.FRAME_OPENED
            and e.id in bracket_range
        ]
        assert len(child_frames_opened_in_range) == 3

        # Each delegation's own TOOL_STARTED and TOOL_COMPLETED on the
        # PARENT frame is tagged with the bracket's group_id.
        parent_agent_events = [
            e for e in events
            if e.type in (EventType.TOOL_STARTED, EventType.TOOL_COMPLETED)
            and e.payload.get("tool_name") == "Agent"
            and e.frame_id == frame.id
        ]
        assert len(parent_agent_events) == 6  # 3 starts + 3 completes
        assert all(
            e.parallel_group_id == group_id for e in parent_agent_events
        )

        # The bundled tool_result message carries 3 blocks in input
        # order (tu_a, tu_b, tu_c) even though asyncio.gather may have
        # resolved them in any order internally.
        msgs = frame.context.body.conversation.messages
        tool_msg = next(m for m in msgs if m.from_ == "tool")
        ids_in_order = [
            b.tool_use_id for b in tool_msg.content
            if isinstance(b, ToolResultContent)
        ]
        assert ids_in_order == ["tu_a", "tu_b", "tu_c"]


async def test_parallel_delegation_fork_rejection_spans_child_events():
    """Forking at ANY event id strictly between PARALLEL_GROUP_STARTED
    and PARALLEL_GROUP_COMPLETED is rejected — including events
    emitted on the child frames (FRAME_OPENED, child MESSAGE_APPENDED,
    child FRAME_RESOLVED, etc.). The store's id-range check is frame-
    agnostic, so this falls out naturally."""
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        provider = FakeProvider([
            _two_tool_turn([
                ("Agent", {"prompt": "A", "name": "researcher"}, "tu_a"),
                ("Agent", {"prompt": "B", "name": "researcher"}, "tu_b"),
            ]),
            _text_turn("child A done"),
            _text_turn("child B done"),
            _text_turn("parent done"),
        ])
        manager = _make_manager(store, provider)

        frame = manager.open_root(
            session_id="s1",
            role=_role("receptionist"),
            model="fake",
            initial_user_input="go",
        )
        await manager.run(frame)

        events = store.snapshot("s1")
        started = next(e for e in events if e.type == EventType.PARALLEL_GROUP_STARTED)
        completed = next(e for e in events if e.type == EventType.PARALLEL_GROUP_COMPLETED)

        # Any id strictly between the bracket is rejected — pick one
        # that's actually a child-frame event to make the test's
        # intent explicit.
        inner_child_events = [
            e for e in events
            if started.id < e.id < completed.id
            and e.frame_id is not None
            and e.frame_id != frame.id
        ]
        assert len(inner_child_events) > 0
        bad_inner_id = inner_child_events[0].id

        with pytest.raises(ValueError) as exc:
            store.fork("s1", at_event_id=bad_inner_id, new_session_id="bad")
        assert "strictly inside" in str(exc.value)
        assert str(started.id) in str(exc.value)
        assert str(completed.id) in str(exc.value)

        # Boundaries remain valid.
        store.fork("s1", at_event_id=started.id, new_session_id="ok_started")
        store.fork("s1", at_event_id=completed.id, new_session_id="ok_completed")


async def test_delegation_result_becomes_tool_result_in_parent_conversation():
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        provider = FakeProvider([
            _tool_turn(
                "Agent",
                {"prompt": "do it", "name": "researcher"},
                tool_id="toolu_d",
            ),
            _text_turn("here is my research"),  # child
            _text_turn("ok"),  # parent final
        ])
        manager = _make_manager(store, provider)

        frame = manager.open_root(
            session_id="s1", role=_role("receptionist"),
            model="fake", initial_user_input="analyze",
        )
        await manager.run(frame)

        msgs = frame.context.body.conversation.messages
        # user, assistant(Agent tool_use), tool_result(from child), assistant(final)
        assert len(msgs) == 4

        tool_result_msg = msgs[2]
        assert tool_result_msg.from_ == "tool"
        assert tool_result_msg.to == "receptionist"

        block = tool_result_msg.content[0]
        assert isinstance(block, ToolResultContent)
        assert block.tool_use_id == "toolu_d"
        assert "here is my research" in (block.content if isinstance(block.content, str) else "")
        assert block.is_error is False


async def test_delegation_uses_fresh_context_without_parent_messages():
    """Child should NOT see parent's user message or system prompt."""
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        provider = FakeProvider([
            _tool_turn(
                "Agent",
                {"prompt": "investigate X", "name": "researcher"},
                tool_id="toolu_d",
            ),
            _text_turn("done investigating"),
            _text_turn("wrapped up"),
        ])
        manager = _make_manager(store, provider)

        frame = manager.open_root(
            session_id="s1", role=_role("receptionist"),
            model="fake", initial_user_input="secret user prompt",
        )
        await manager.run(frame)

        # The second request (to child) should only have the delegation
        # message, not the parent's original user prompt.
        child_request = provider.requests[1]
        assert len(child_request.messages) == 1
        first_msg = child_request.messages[0]
        # The single message in the child's request is the delegation text
        content_text = "".join(
            b.text for b in first_msg.content if hasattr(b, "text")
        )
        assert "investigate X" in content_text
        assert "secret user prompt" not in content_text

        # Child's system prompt comes from the researcher role, NOT the
        # receptionist's — role instructions start with "# Role: Researcher"
        # (from the builtin profile).
        assert any("researcher" in s.lower() for s in child_request.system)


# ---------------------------------------------------------------------------
# Role resolution
# ---------------------------------------------------------------------------


async def test_delegation_resolver_drives_role_allowed_tools():
    """A custom resolver's allowed_tools filter the child's tool set."""
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        provider = FakeProvider([
            _tool_turn(
                "Agent",
                {"prompt": "read files", "name": "researcher"},
                tool_id="toolu_d",
            ),
            _text_turn("read done"),
            _text_turn("finished"),
        ])

        def resolver(name: str) -> AgentRole | None:
            if name == "researcher":
                return AgentRole(
                    name="researcher",
                    instructions="locate files",
                    allowed_tools=["Read", "Glob", "Grep", "Bash"],
                )
            return None

        manager = _make_manager(
            store,
            provider,
            tools=[
                AgentTool(),
                FakeTool("Read"),
                FakeTool("Glob"),
                FakeTool("Grep"),
                FakeTool("Bash"),
                FakeTool("Write"),
            ],
            role_resolver=resolver,
        )

        frame = manager.open_root(
            session_id="s1", role=_role("receptionist"),
            model="fake", initial_user_input="go",
        )
        await manager.run(frame)

        child_request = provider.requests[1]
        child_tool_names = {t.name for t in (child_request.tools or [])}
        # Only the resolver's allowed_tools are exposed to the child.
        assert "Read" in child_tool_names
        assert "Glob" in child_tool_names
        assert "Grep" in child_tool_names
        assert "Bash" in child_tool_names
        assert "Agent" not in child_tool_names
        assert "Write" not in child_tool_names


async def test_delegation_unknown_role_falls_back_to_minimal():
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        provider = FakeProvider([
            _tool_turn(
                "Agent",
                {"prompt": "help", "name": "this_role_does_not_exist"},
                tool_id="toolu_d",
            ),
            _text_turn("helping"),
            _text_turn("done"),
        ])
        manager = _make_manager(store, provider)

        frame = manager.open_root(
            session_id="s1", role=_role("receptionist"),
            model="fake", initial_user_input="go",
        )
        await manager.run(frame)

        # Runs without error; child system prompt contains fallback
        # instructions mentioning the unknown role name
        child_request = provider.requests[1]
        system_text = "\n".join(child_request.system)
        assert "this_role_does_not_exist" in system_text


# ---------------------------------------------------------------------------
# Mixed delegation + regular tool in one turn
# ---------------------------------------------------------------------------


async def test_mixed_delegation_and_regular_tool_in_single_turn():
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        provider = FakeProvider([
            _two_tool_turn([
                ("FakeRead", {"path": "/tmp/x"}, "toolu_read"),
                ("Agent", {"prompt": "analyze", "name": "analyzer"}, "toolu_agent"),
            ]),
            _text_turn("analysis complete"),  # child (analyzer)
            _text_turn("final"),              # parent
        ])
        manager = _make_manager(
            store, provider,
            tools=[AgentTool(), FakeTool("FakeRead", output="file body")],
        )

        frame = manager.open_root(
            session_id="s1", role=_role("receptionist"),
            model="fake", initial_user_input="do both",
        )
        await manager.run(frame)

        # Parent conversation has: user, assistant(2 tool_uses), tool_result(2 blocks), assistant
        msgs = frame.context.body.conversation.messages
        assert len(msgs) == 4
        tool_result = msgs[2]
        assert tool_result.from_ == "tool"
        assert len(tool_result.content) == 2
        # First block is FakeRead result (order preserved)
        assert tool_result.content[0].tool_use_id == "toolu_read"  # type: ignore[attr-defined]
        assert tool_result.content[1].tool_use_id == "toolu_agent"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# reconstruct with parent + child frames
# ---------------------------------------------------------------------------


async def test_counterparty_does_not_drift_after_tool_result():
    """Researcher's outgoing reply must still target the parent (core),
    not 'tool', even after a tool_result message arrives in its conversation.
    Regression for the routing label drift bug.
    """
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        # Sequence:
        #  1. Receptionist root: Agent(name=worker, prompt=...)
        #  2. Worker child, turn 1: tool_use FakeRead
        #  3. Worker child, turn 2: text reply (after tool result)
        #  4. Receptionist root, after Agent returns: text reply
        provider = FakeProvider([
            _tool_turn(
                "Agent",
                {"prompt": "do the job", "name": "researcher"},
                tool_id="toolu_deleg",
            ),
            _tool_turn("FakeRead", {"path": "/x"}, tool_id="toolu_read"),
            _text_turn("findings: file content"),
            _text_turn("summary for user"),
        ])
        manager = _make_manager(
            store, provider,
            tools=[AgentTool(), FakeTool("FakeRead", output="file body")],
        )

        root = manager.open_root(
            session_id="s1",
            role=_role("receptionist"),
            model="fake",
            initial_user_input="go",
        )
        await manager.run(root)

        # Find the child frame's last outgoing message
        events = store.snapshot("s1")
        child_opened = [
            e for e in events if e.type == EventType.FRAME_OPENED
            and e.payload.get("parent_id")
        ]
        assert len(child_opened) == 1
        child_frame_id = child_opened[0].frame_id

        # Find researcher's outgoing messages
        outgoing = [
            e for e in events
            if e.type == EventType.MESSAGE_APPENDED
            and e.frame_id == child_frame_id
            and e.payload.get("from_") == "researcher"
        ]
        # All researcher → ??? messages must target "receptionist"
        # (the parent's actor), never "tool"
        for ev in outgoing:
            assert ev.payload.get("to") == "receptionist", (
                f"counterparty drifted to {ev.payload.get('to')!r} "
                "after tool result — should stay 'receptionist'"
            )


async def test_reconstruct_rebuilds_parent_child_frame_tree():
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        provider = FakeProvider([
            _tool_turn(
                "Agent",
                {"prompt": "explore", "name": "researcher"},
                tool_id="toolu_d",
            ),
            _text_turn("explored"),
            _text_turn("summary"),
        ])
        manager = _make_manager(store, provider)

        parent = manager.open_root(
            session_id="s1", role=_role("receptionist"),
            model="fake", initial_user_input="go",
        )
        await manager.run(parent)
        manager.close(parent)

        # Replay from a fresh store instance
        fresh = FileEventStore(Path(tmp))
        replay = reconstruct("s1", fresh)

        # Both frames exist
        assert len(replay.frames) == 2
        root_frames = replay.root_frames
        assert len(root_frames) == 1
        rebuilt_parent = root_frames[0]
        assert rebuilt_parent.id == parent.id
        assert len(rebuilt_parent.children_ids) == 1

        child_id = rebuilt_parent.children_ids[0]
        rebuilt_child = replay.frames[child_id]
        assert rebuilt_child.parent_id == parent.id
        assert rebuilt_child.self_actor == "researcher"
        assert rebuilt_child.state == FrameState.CLOSED


async def test_reconstruct_restores_child_counterparty_from_parent():
    """Regression: reconstructed child frames must carry the parent's
    self_actor as their counterparty, not the dataclass default 'user'."""
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        provider = FakeProvider([
            _tool_turn(
                "Agent",
                {"prompt": "dig in", "name": "researcher"},
                tool_id="toolu_dc",
            ),
            _text_turn("dug in"),
            _text_turn("summary"),
        ])
        manager = _make_manager(store, provider)

        parent = manager.open_root(
            session_id="s1", role=_role("receptionist"),
            model="fake", initial_user_input="go",
        )
        await manager.run(parent)

        replay = reconstruct("s1", store)
        # Root frame replies to user
        rebuilt_parent = replay.frames[parent.id]
        assert rebuilt_parent.counterparty == "user"

        # Child frame must reply to parent's self_actor (receptionist),
        # not the Frame dataclass default.
        child_id = rebuilt_parent.children_ids[0]
        rebuilt_child = replay.frames[child_id]
        assert rebuilt_child.counterparty == "receptionist"
        assert rebuilt_child.self_actor == "researcher"


async def test_reconstruct_exposes_spawn_by_tool_use_index():
    """ReplayResult should let callers jump parent tool_use → child frame
    without scanning the event stream."""
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        provider = FakeProvider([
            _tool_turn(
                "Agent",
                {"prompt": "please investigate", "name": "researcher"},
                tool_id="toolu_abc",
            ),
            _text_turn("investigation done"),
            _text_turn("wrap-up"),
        ])
        manager = _make_manager(store, provider)

        parent = manager.open_root(
            session_id="s1", role=_role("receptionist"),
            model="fake", initial_user_input="go",
        )
        await manager.run(parent)

        replay = reconstruct("s1", store)
        child_frame = replay.child_of("toolu_abc")
        assert child_frame is not None
        assert child_frame.parent_id == parent.id

        # Reverse link
        parent_id, parent_msg_id, tool_use_id = replay.spawn_origin[child_frame.id]
        assert parent_id == parent.id
        assert tool_use_id == "toolu_abc"
        # parent_msg_id should point to an actual message in the parent
        parent_msg_ids = {
            m.id for m in replay.frames[parent.id].context.body.conversation.messages
        }
        assert parent_msg_id in parent_msg_ids

        # The parent's tool_result MESSAGE_APPENDED should carry the
        # delegation mapping in its payload.
        events = store.snapshot("s1")
        tool_result_events = [
            e for e in events
            if e.type == EventType.MESSAGE_APPENDED
            and e.payload.get("from_") == "tool"
            and e.frame_id == parent.id
        ]
        assert tool_result_events
        assert tool_result_events[0].payload["delegations"] == {
            "toolu_abc": child_frame.id
        }
