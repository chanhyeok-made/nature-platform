"""Tests for body compaction + BODY_COMPACTED event (Phase 5)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from nature.context import AgentRole
from nature.context.body_compaction import (
    BodyCompactionPipeline,
    BodyCompactionResult,
    BodyCompactionStrategy,
    DreamerBodyStrategy,
    MICROCOMPACT_CLEARED,
    MicrocompactBodyStrategy,
)
from nature.context.conversation import Conversation, Message
from nature.context.types import Context, ContextBody, ContextHeader
from nature.events import EventType, FileEventStore
from nature.events.reconstruct import reconstruct
from nature.events.types import EVENT_CATEGORIES, EventCategory
from nature.frame import AreaManager, FrameState
from nature.protocols.context import TokenBudget
from nature.protocols.message import (
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolResultContent,
    Usage,
)
from nature.config.constants import StopReason

from tests._fakes import FakeProvider


class _ByteEstimateFakeProvider(FakeProvider):
    """FakeProvider whose count_tokens fails so the estimator falls back
    to byte-based estimation — needed to actually trigger compaction in
    integration tests (default FakeProvider always returns 0 tokens)."""

    async def count_tokens(self, messages, system, tools=None) -> int:  # type: ignore[override]
        raise RuntimeError("use byte fallback")


# ---------------------------------------------------------------------------
# Unit tests: MicrocompactBodyStrategy
# ---------------------------------------------------------------------------


def _assistant(text: str, from_: str = "receptionist") -> Message:
    return Message(
        from_=from_,
        to="user",
        content=[TextContent(text=text)],
        timestamp=0.0,
    )


def _tool_result(text: str, tool_use_id: str = "t1") -> Message:
    return Message(
        from_="tool",
        to="receptionist",
        content=[ToolResultContent(tool_use_id=tool_use_id, content=text)],
        timestamp=0.0,
    )


def _header(actor: str = "receptionist") -> ContextHeader:
    return ContextHeader(
        role=AgentRole(name=actor, instructions=f"you are {actor}"),
    )


async def test_microcompact_is_noop_when_under_preserve_turns():
    body = ContextBody(conversation=Conversation(messages=[
        _assistant("turn 1"),
        _tool_result("result 1"),
        _assistant("turn 2"),
    ]))
    strat = MicrocompactBodyStrategy(preserve_turns=4)
    result = await strat.compact(
        body, header=_header(), budget=TokenBudget(), current_tokens=100,
    )
    assert result.body is body
    assert result.tokens_after == result.tokens_before


async def test_microcompact_clears_old_tool_results_preserves_recent_turns():
    msgs = [
        _assistant("turn 1"),
        _tool_result("old result 1"),
        _assistant("turn 2"),
        _tool_result("old result 2"),
        _assistant("turn 3"),
        _tool_result("recent 3"),
        _assistant("turn 4"),
        _tool_result("recent 4"),
        _assistant("turn 5"),
    ]
    body = ContextBody(conversation=Conversation(messages=msgs))
    strat = MicrocompactBodyStrategy(preserve_turns=3)

    result = await strat.compact(
        body, header=_header(), budget=TokenBudget(), current_tokens=1000,
    )

    new = result.body.conversation.messages
    assert len(new) == len(msgs)  # message count unchanged; only contents altered

    # Old tool results (before the preserved tail) should be cleared
    old_tr_1 = new[1].content[0]
    assert isinstance(old_tr_1, ToolResultContent)
    assert old_tr_1.content == MICROCOMPACT_CLEARED

    # Last preserved tool_result should still carry original content
    recent = new[-2].content[0]
    assert isinstance(recent, ToolResultContent)
    assert recent.content == "recent 4"

    assert "cleared" in (result.summary or "")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class _AggressiveStrategy(BodyCompactionStrategy):
    """Fake strategy that replaces the entire body with a placeholder msg."""

    def __init__(self) -> None:
        self.called = 0

    @property
    def name(self) -> str:
        return "aggressive"

    async def compact(
        self, body, *, header, budget, current_tokens, provider=None,
    ):
        self.called += 1
        marker = Message(
            from_="user",
            to=header.role.name,
            content=[TextContent(text="[compacted]")],
            timestamp=0.0,
        )
        new_body = ContextBody(conversation=Conversation(messages=[marker]))
        return BodyCompactionResult(
            body=new_body,
            tokens_before=current_tokens,
            tokens_after=1,
            strategy_name=self.name,
            summary="aggressive wipe",
        )


async def test_pipeline_noop_when_under_threshold():
    pipeline = BodyCompactionPipeline(
        strategies=[_AggressiveStrategy()],
        budget=TokenBudget(),
    )
    ctx = Context(
        header=_header(),
        body=ContextBody(conversation=Conversation(messages=[_assistant("hi")])),
    )
    result = await pipeline.run(
        ctx,
        self_actor="receptionist",
        tool_registry=[],
        model="fake",
    )
    assert result.steps == []


async def test_pipeline_runs_strategies_above_threshold():
    # Tiny budget so even a single message crosses the autocompact threshold
    budget = TokenBudget(
        context_window=200,
        output_reservation=10,
        autocompact_buffer=150,
    )
    strat = _AggressiveStrategy()
    pipeline = BodyCompactionPipeline(strategies=[strat], budget=budget)

    big_body = ContextBody(conversation=Conversation(messages=[
        _assistant("x" * 400, from_="user"),
        _assistant("y" * 400),
    ]))
    ctx = Context(header=_header(), body=big_body)

    result = await pipeline.run(
        ctx,
        self_actor="receptionist",
        tool_registry=[],
        model="fake",
    )
    assert strat.called == 1
    assert len(result.steps) == 1
    assert result.final_body.conversation.messages[0].content[0].text == "[compacted]"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# AreaManager integration — emits BODY_COMPACTED + reconstruct round-trip
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


def _role(name: str = "receptionist") -> AgentRole:
    return AgentRole(name=name, instructions=f"you are {name}")


async def test_area_manager_emits_body_compacted_when_pipeline_fires():
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        provider = _ByteEstimateFakeProvider(_text_turn("pong"))

        # Tiny budget that triggers compaction immediately
        budget = TokenBudget(
            context_window=150,
            output_reservation=10,
            autocompact_buffer=130,
        )
        pipeline = BodyCompactionPipeline(
            strategies=[_AggressiveStrategy()],
            budget=budget,
        )

        manager = AreaManager(
            store=store,
            provider=provider,
            tool_registry=[],
            cwd="/tmp",
            compaction_pipeline=pipeline,
        )

        frame = manager.open_root(
            session_id="s1",
            role=_role(),
            model="fake",
            initial_user_input="a" * 400,  # ensures we're over threshold
        )
        await manager.run(frame)
        assert frame.state == FrameState.RESOLVED

        events = store.snapshot("s1")
        types = [e.type for e in events]
        assert EventType.BODY_COMPACTED in types

        # The compacted event must sit BEFORE the llm.request that saw the
        # compacted state — i.e. pipeline ran first, then llm_agent ran.
        req_idx = types.index(EventType.LLM_REQUEST)
        bc_idx = types.index(EventType.BODY_COMPACTED)
        assert bc_idx < req_idx

        # Category check
        assert EVENT_CATEGORIES[EventType.BODY_COMPACTED] is EventCategory.STATE_TRANSITION


async def test_reconstruct_reproduces_compacted_body():
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        provider = _ByteEstimateFakeProvider(_text_turn("ok"))

        budget = TokenBudget(
            context_window=150,
            output_reservation=10,
            autocompact_buffer=130,
        )
        pipeline = BodyCompactionPipeline(
            strategies=[_AggressiveStrategy()],
            budget=budget,
        )

        manager = AreaManager(
            store=store,
            provider=provider,
            tool_registry=[],
            cwd="/tmp",
            compaction_pipeline=pipeline,
        )

        frame = manager.open_root(
            session_id="s1",
            role=_role(),
            model="fake",
            initial_user_input="b" * 400,
        )
        await manager.run(frame)
        manager.close(frame)

        # Replay from a fresh store instance
        fresh = FileEventStore(Path(tmp))
        replay = reconstruct("s1", fresh)
        rebuilt = replay.frames[frame.id]

        # After compaction the body should contain the compacted marker
        # plus any messages appended AFTER compaction by the run loop.
        msgs = rebuilt.context.body.conversation.messages
        texts = []
        for m in msgs:
            for b in m.content:
                if hasattr(b, "text"):
                    texts.append(b.text)  # type: ignore[attr-defined]
        # The aggressive strategy replaced everything with "[compacted]",
        # then llm_agent appended the assistant's response on top.
        assert "[compacted]" in texts
        assert "ok" in texts

        # Header must be untouched by compaction
        assert rebuilt.context.header.role.name == "receptionist"
        assert rebuilt.context.header.role.instructions == "you are receptionist"

        # Trace-strip invariant: dropping trace events yields the same
        # compacted body
        from nature.events.reconstruct import ReplayResult, _apply_event
        from nature.events.types import is_state_transition
        r2 = ReplayResult()
        for ev in store.snapshot("s1"):
            if is_state_transition(ev.type):
                _apply_event(ev, r2)
        assert len(r2.frames[frame.id].context.body.conversation.messages) == len(msgs)


async def test_pipeline_does_not_modify_header():
    """A strategy bug that returns a mutated header should not propagate."""
    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        provider = _ByteEstimateFakeProvider(_text_turn("ok"))

        budget = TokenBudget(
            context_window=150,
            output_reservation=10,
            autocompact_buffer=130,
        )
        pipeline = BodyCompactionPipeline(
            strategies=[_AggressiveStrategy()],
            budget=budget,
        )

        manager = AreaManager(
            store=store,
            provider=provider,
            tool_registry=[],
            cwd="/tmp",
            compaction_pipeline=pipeline,
        )

        original_role = _role()
        frame = manager.open_root(
            session_id="s1",
            role=original_role,
            model="fake",
            initial_user_input="x" * 400,
        )
        await manager.run(frame)

        # Header identity preserved (same role name, same instructions)
        assert frame.context.header.role.name == original_role.name
        assert frame.context.header.role.instructions == original_role.instructions


# ---------------------------------------------------------------------------
# Unit tests: DreamerBodyStrategy
# ---------------------------------------------------------------------------


def _summary_stream_events(text: str) -> list[StreamEvent]:
    """Build a minimal StreamEvent sequence that streams `text` as a
    single content_block_delta — enough for DreamerBodyStrategy to
    collect it."""
    return [
        StreamEvent(type=StreamEventType.MESSAGE_START),
        StreamEvent(type=StreamEventType.CONTENT_BLOCK_START, index=0),
        StreamEvent(
            type=StreamEventType.CONTENT_BLOCK_DELTA,
            index=0, delta_text=text,
        ),
        StreamEvent(type=StreamEventType.CONTENT_BLOCK_STOP, index=0),
        StreamEvent(type=StreamEventType.MESSAGE_STOP),
    ]


def _long_conversation(turns: int, actor: str = "receptionist") -> list[Message]:
    """Build a fake multi-turn conversation. Each turn = user → actor →
    actor tool_use → tool_result, so the self-actor sees `turns` turns."""
    msgs: list[Message] = []
    for i in range(turns):
        msgs.append(Message(
            from_="user", to=actor,
            content=[TextContent(text=f"user msg {i}")],
            timestamp=0.0,
        ))
        msgs.append(Message(
            from_=actor, to="user",
            content=[TextContent(text=f"{actor} turn {i}")],
            timestamp=0.0,
        ))
    return msgs


async def test_dreamer_is_noop_when_few_self_turns():
    body = ContextBody(conversation=Conversation(
        messages=_long_conversation(turns=3),
    ))
    strat = DreamerBodyStrategy(preserve_recent_turns=6)
    provider = FakeProvider(_summary_stream_events("should not be called"))

    result = await strat.compact(
        body, header=_header(), budget=TokenBudget(),
        current_tokens=10_000, provider=provider,
    )
    assert result.body is body
    # Provider should not have been called since no compaction happened.
    assert provider._call_count == 0


async def test_dreamer_is_noop_when_cutoff_is_zero():
    # With preserve_recent_turns == number of self turns, cutoff is 0 —
    # there is nothing older to compact.
    msgs = _long_conversation(turns=4)
    body = ContextBody(conversation=Conversation(messages=msgs))
    strat = DreamerBodyStrategy(preserve_recent_turns=4)
    provider = FakeProvider(_summary_stream_events("unused"))

    result = await strat.compact(
        body, header=_header(), budget=TokenBudget(),
        current_tokens=10_000, provider=provider,
    )
    assert result.body is body


async def test_dreamer_is_noop_without_provider():
    body = ContextBody(conversation=Conversation(
        messages=_long_conversation(turns=10),
    ))
    strat = DreamerBodyStrategy(preserve_recent_turns=3)

    result = await strat.compact(
        body, header=_header(), budget=TokenBudget(),
        current_tokens=10_000, provider=None,
    )
    assert result.body is body


async def test_dreamer_replaces_prefix_with_summary_message():
    msgs = _long_conversation(turns=10)
    body = ContextBody(conversation=Conversation(messages=msgs))
    strat = DreamerBodyStrategy(preserve_recent_turns=3)
    provider = FakeProvider(_summary_stream_events(
        "Goal: testing. Files: none. Next: nothing."
    ))

    result = await strat.compact(
        body, header=_header(), budget=TokenBudget(),
        current_tokens=10_000, provider=provider,
    )
    new_messages = result.body.conversation.messages

    # Cutoff is the index of the N-th-from-last self-actor message,
    # so the tail starts AT that assistant turn — the user msg before
    # it lands in the prefix. For turns=10, preserve=3, that leaves
    # 3 assistant + 2 user = 5 preserved messages behind one summary.
    assert len(new_messages) == 1 + 5
    head = new_messages[0]
    assert head.from_ == "system"
    assert head.to == "receptionist"
    assert isinstance(head.content[0], TextContent)
    assert "Goal: testing" in head.content[0].text

    # Tail preserves the most recent self-actor turns.
    assert new_messages[-1].from_ == "receptionist"
    assert "turn 9" in new_messages[-1].content[0].text


async def test_dreamer_persists_ltm_when_configured(tmp_path):
    msgs = _long_conversation(turns=8)
    body = ContextBody(conversation=Conversation(messages=msgs))
    strat = DreamerBodyStrategy(
        preserve_recent_turns=2,
        session_id="sess-xyz",
        ltm_dir=tmp_path,
    )
    provider = FakeProvider(_summary_stream_events("summary body"))

    result = await strat.compact(
        body, header=_header(), budget=TokenBudget(),
        current_tokens=10_000, provider=provider,
    )

    # LTM file should have been created under <ltm_dir>/<session_id>/
    # tagged by self-actor (the header's role name).
    session_dir = tmp_path / "sess-xyz"
    assert session_dir.exists()
    files = list(session_dir.glob("receptionist-*.md"))
    assert len(files) == 1
    written = files[0].read_text(encoding="utf-8")
    assert "## Summary" in written
    assert "summary body" in written
    assert "## Raw transcript" in written
    # Summary message carries the LTM path so the agent can Read it.
    summary_text = result.body.conversation.messages[0].content[0].text
    assert str(files[0]) in summary_text


async def test_dreamer_empty_summary_is_noop():
    msgs = _long_conversation(turns=10)
    body = ContextBody(conversation=Conversation(messages=msgs))
    strat = DreamerBodyStrategy(preserve_recent_turns=3)
    provider = FakeProvider(_summary_stream_events(""))

    result = await strat.compact(
        body, header=_header(), budget=TokenBudget(),
        current_tokens=10_000, provider=provider,
    )
    assert result.body is body


async def test_dreamer_renders_tool_use_and_result_blocks():
    from nature.protocols.message import ToolUseContent as _TU
    msgs = [
        Message(
            from_="user", to="receptionist",
            content=[TextContent(text="do X")], timestamp=0.0,
        ),
        Message(
            from_="receptionist", to="user",
            content=[
                TextContent(text="running tool"),
                _TU(id="t1", name="Bash", input={"cmd": "ls"}),
            ],
            timestamp=0.0,
        ),
        Message(
            from_="tool", to="receptionist",
            content=[ToolResultContent(
                tool_use_id="t1", content="file1\nfile2",
            )],
            timestamp=0.0,
        ),
    ] + _long_conversation(turns=8)
    body = ContextBody(conversation=Conversation(messages=msgs))
    strat = DreamerBodyStrategy(preserve_recent_turns=2)
    provider = FakeProvider(_summary_stream_events("summary"))

    await strat.compact(
        body, header=_header(), budget=TokenBudget(),
        current_tokens=10_000, provider=provider,
    )
    # The provider saw a user message whose text includes tool_use +
    # tool_result markers. We assert on the content the provider received.
    sent = provider.requests[-1] if provider.requests else None
    # FakeProvider.stream() doesn't populate .requests; check via
    # reading the one user-message body passed to stream() directly:
    # easier to validate the renderer here without reaching into fake.
    from nature.context.body_compaction import _render_message_for_summary
    rendered = _render_message_for_summary(msgs[1])
    assert "tool_use" in rendered
    assert "Bash" in rendered
    rendered_tr = _render_message_for_summary(msgs[2])
    assert "tool_result" in rendered_tr
    assert "file1" in rendered_tr
