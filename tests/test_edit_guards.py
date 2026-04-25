"""Tests for the edit_guards builtin Pack.

Layer coverage:

1. Unit tests for each intervention's action function (no registry)
2. Helper tests (body walker, hash stability)
3. Registry integration (fuzzy_suggest + loop_detector PRIMARY/POST_EFFECT)
4. End-to-end via AreaManager with a synthetic Edit-named tool that
   always errors
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path

import pytest

from nature.context.conversation import Conversation, Message
from nature.context.types import (
    AgentRole,
    Context,
    ContextBody,
    ContextHeader,
)
from nature.events import FileEventStore
from nature.events.types import EventType
from nature.frame import AreaManager, Frame, FrameState
from nature.packs.builtin.edit_guards.fuzzy_suggest import (
    _find_closest_window,
    _fuzzy_suggest_action,
    fuzzy_suggest,
)
from nature.packs.builtin.edit_guards.loop_block import (
    _loop_block_action,
    loop_block,
)
from nature.packs.builtin.edit_guards.loop_detector import (
    THRESHOLD,
    _loop_detector_action,
    count_recent_same_hash_edit_failures,
    hash_edit_input,
    loop_detector,
)
from nature.packs.builtin.edit_guards.pack import (
    edit_guards_capability,
    install,
)
from nature.packs.builtin.edit_guards.reread_hint import (
    _last_edit_error,
    _reread_hint_action,
    reread_hint,
)
from nature.packs.registry import PackRegistry
from nature.packs.types import (
    AppendFooter,
    Block,
    EmitEvent,
    InterventionContext,
    InterventionPhase,
    ModifyToolResult,
    ToolCallInfo,
    ToolPhase,
)
from nature.protocols.message import (
    TextContent,
    ToolResultContent,
    ToolUseContent,
)


# ──────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────


def _make_frame(messages: list[Message]) -> Frame:
    """Build a minimal Frame with a given conversation for body tests."""
    return Frame(
        id="f1",
        session_id="s1",
        purpose="test",
        context=Context(
            header=ContextHeader(role=AgentRole(name="r", instructions="")),
            body=ContextBody(conversation=Conversation(messages=messages)),
        ),
        model="fake",
    )


def _edit_attempt_pair(
    tool_use_id: str,
    file_path: str,
    old_string: str,
    new_string: str = "x",
    is_error: bool = True,
    output: str = "old_string not found",
) -> tuple[Message, Message]:
    """Build (assistant with Edit tool_use, tool with matching result)."""
    assistant = Message(
        id=f"msg_asst_{tool_use_id}",
        from_="self",
        to="tool",
        content=[
            ToolUseContent(
                id=tool_use_id,
                name="Edit",
                input={
                    "file_path": file_path,
                    "old_string": old_string,
                    "new_string": new_string,
                },
            ),
        ],
        timestamp=time.time(),
    )
    tool_result = Message(
        id=f"msg_tool_{tool_use_id}",
        from_="tool",
        to="self",
        content=[
            ToolResultContent(
                tool_use_id=tool_use_id,
                content=output,
                is_error=is_error,
            ),
        ],
        timestamp=time.time(),
    )
    return assistant, tool_result


def _edit_tool_call_info(
    file_path: str,
    old_string: str,
    *,
    phase: ToolPhase,
    is_error: bool | None = None,
) -> ToolCallInfo:
    return ToolCallInfo(
        tool_name="Edit",
        tool_use_id="tu_new",
        tool_input={
            "file_path": file_path,
            "old_string": old_string,
            "new_string": "y",
        },
        phase=phase,
        result_is_error=is_error,
    )


# ──────────────────────────────────────────────────────────────────────
# hash + body walker
# ──────────────────────────────────────────────────────────────────────


def test_hash_edit_input_is_stable_and_length_insensitive_past_80():
    a = hash_edit_input("x.py", "foo" * 100)
    b = hash_edit_input("x.py", "foo" * 100)
    c = hash_edit_input("x.py", "foo" * 100 + "bar")  # differs past 80 chars
    assert a == b
    # After 80 characters of `old_string`, the hash ignores further differences.
    # Because 3*100 = 300 chars and we truncate at 80, the extra "bar" doesn't
    # move the hash.
    assert a == c


def test_hash_edit_input_distinguishes_different_files_and_targets():
    a = hash_edit_input("x.py", "foo")
    b = hash_edit_input("y.py", "foo")
    c = hash_edit_input("x.py", "bar")
    assert a != b
    assert a != c


def test_count_recent_same_hash_zero_for_empty_body():
    frame = _make_frame([])
    assert count_recent_same_hash_edit_failures(frame.context.body.conversation) == 0


def test_count_recent_same_hash_counts_consecutive_matching_pairs():
    a1, t1 = _edit_attempt_pair("tu_1", "a.py", "hallucinated_old")
    a2, t2 = _edit_attempt_pair("tu_2", "a.py", "hallucinated_old")
    frame = _make_frame([a1, t1, a2, t2])
    assert count_recent_same_hash_edit_failures(frame.context.body.conversation) == 2


def test_count_recent_same_hash_breaks_on_different_hash():
    a1, t1 = _edit_attempt_pair("tu_1", "a.py", "hallucinated_old")
    a2, t2 = _edit_attempt_pair("tu_2", "b.py", "different_target")
    a3, t3 = _edit_attempt_pair("tu_3", "b.py", "different_target")
    frame = _make_frame([a1, t1, a2, t2, a3, t3])
    # Walking back from the tail: tu_3 and tu_2 share hash → streak = 2,
    # then tu_1 has a different hash so we stop.
    assert count_recent_same_hash_edit_failures(frame.context.body.conversation) == 2


def test_count_recent_same_hash_ignores_unpaired_trailing_tool_use():
    a1, t1 = _edit_attempt_pair("tu_1", "a.py", "x")
    a2 = Message(
        id="msg_asst_2",
        from_="self",
        to="tool",
        content=[
            ToolUseContent(id="tu_2", name="Edit", input={"file_path": "a.py", "old_string": "x", "new_string": "y"}),
        ],
        timestamp=time.time(),
    )
    frame = _make_frame([a1, t1, a2])
    # a2 has no tool_result yet — shouldn't break or count.
    assert count_recent_same_hash_edit_failures(frame.context.body.conversation) == 1


def test_count_recent_same_hash_respects_target_hash():
    a1, t1 = _edit_attempt_pair("tu_1", "a.py", "foo")
    frame = _make_frame([a1, t1])
    wrong_hash = hash_edit_input("a.py", "bar")
    # Target hash doesn't match the most recent attempt → streak ends at 0.
    assert count_recent_same_hash_edit_failures(
        frame.context.body.conversation, target_hash=wrong_hash,
    ) == 0


# ──────────────────────────────────────────────────────────────────────
# fuzzy_suggest
# ──────────────────────────────────────────────────────────────────────


def test_fuzzy_suggest_finds_close_window_in_file():
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write("def greet():\n    print('hello world')\n\ndef farewell():\n    pass\n")
        path = f.name
    try:
        match = _find_closest_window(path, "print('hello worlds')")  # typo
        assert match is not None
        matched_text, lineno = match
        assert "hello world" in matched_text
        assert lineno in (1, 2)
    finally:
        Path(path).unlink()


def test_fuzzy_suggest_returns_none_below_similarity_floor():
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write("alpha\nbeta\ngamma\n")
        path = f.name
    try:
        match = _find_closest_window(path, "completely_unrelated_content_xyz_123")
        assert match is None
    finally:
        Path(path).unlink()


def test_fuzzy_suggest_action_emits_edit_miss_even_without_match():
    # Nonexistent file → _find_closest_window returns None.
    ctx = InterventionContext(
        session_id="s",
        now=time.time(),
        registry=None,
        tool_call=_edit_tool_call_info(
            "/nowhere/missing_file.py", "anything", phase=ToolPhase.POST, is_error=True,
        ),
    )
    effects = _fuzzy_suggest_action(ctx)
    assert len(effects) == 1
    assert isinstance(effects[0], EmitEvent)
    assert effects[0].event_type == EventType.EDIT_MISS


def test_fuzzy_suggest_action_returns_modify_plus_emit_on_match():
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write("def foo():\n    return 1\n")
        path = f.name
    try:
        ctx = InterventionContext(
            session_id="s",
            now=time.time(),
            registry=None,
            tool_call=_edit_tool_call_info(
                path, "def fooo():\n    return 1", phase=ToolPhase.POST, is_error=True,
            ),
        )
        effects = _fuzzy_suggest_action(ctx)
        assert len(effects) == 2
        kinds = {type(e).__name__ for e in effects}
        assert "ModifyToolResult" in kinds
        assert "EmitEvent" in kinds
        modify = next(e for e in effects if isinstance(e, ModifyToolResult))
        assert modify.append_hint is not None
        assert "Closest match" in modify.append_hint
    finally:
        Path(path).unlink()


def test_fuzzy_suggest_skips_when_tool_call_is_not_errored():
    ctx = InterventionContext(
        session_id="s",
        now=time.time(),
        registry=None,
        tool_call=_edit_tool_call_info(
            "/tmp/any.py", "x", phase=ToolPhase.POST, is_error=False,
        ),
    )
    assert _fuzzy_suggest_action(ctx) == []


# ──────────────────────────────────────────────────────────────────────
# reread_hint (Contributor)
# ──────────────────────────────────────────────────────────────────────


def test_reread_hint_last_edit_error_detects_failed_tool_result():
    _, t1 = _edit_attempt_pair("tu_1", "a.py", "x", output="old_string not found")
    body = ContextBody(conversation=Conversation(messages=[t1]))
    assert _last_edit_error(body) is True


def test_reread_hint_last_edit_error_skips_successful_tool_result():
    _, t1 = _edit_attempt_pair(
        "tu_1", "a.py", "x", is_error=False, output="Edit applied",
    )
    body = ContextBody(conversation=Conversation(messages=[t1]))
    assert _last_edit_error(body) is False


def test_reread_hint_last_edit_error_skips_non_edit_errors():
    t1 = Message(
        id="msg_tool_1",
        from_="tool",
        to="self",
        content=[
            ToolResultContent(
                tool_use_id="tu_bash",
                content="some bash error",
                is_error=True,
            ),
        ],
        timestamp=time.time(),
    )
    body = ContextBody(conversation=Conversation(messages=[t1]))
    # Error but not an Edit-style error.
    assert _last_edit_error(body) is False


def test_reread_hint_action_contributes_footer_when_active():
    _, t1 = _edit_attempt_pair("tu_1", "a.py", "x", output="old_string not found")
    body = ContextBody(conversation=Conversation(messages=[t1]))
    ctx = InterventionContext(
        session_id="s",
        now=time.time(),
        registry=None,
        body=body,
        header=ContextHeader(role=AgentRole(name="r", instructions="")),
        self_actor="r",
    )
    effects = _reread_hint_action(ctx)
    assert len(effects) == 1
    assert isinstance(effects[0], AppendFooter)
    assert "reread_hint" in effects[0].source_id
    assert "Re-read" in effects[0].text


def test_reread_hint_action_no_op_on_fresh_body():
    body = ContextBody(conversation=Conversation(messages=[]))
    ctx = InterventionContext(
        session_id="s",
        now=time.time(),
        registry=None,
        body=body,
        header=ContextHeader(role=AgentRole(name="r", instructions="")),
        self_actor="r",
    )
    assert _reread_hint_action(ctx) == []


# ──────────────────────────────────────────────────────────────────────
# loop_detector (POST_EFFECT)
# ──────────────────────────────────────────────────────────────────────


def _edit_miss_primary_effects() -> list:
    from nature.events.payloads import EditMissPayload
    return [EmitEvent(
        event_type=EventType.EDIT_MISS,
        payload=EditMissPayload(file="a.py"),
    )]


def test_loop_detector_skips_without_edit_miss_in_primary_effects():
    frame = _make_frame([])
    ctx = InterventionContext(
        session_id="s",
        now=time.time(),
        registry=None,
        frame=frame,
        tool_call=_edit_tool_call_info(
            "a.py", "x", phase=ToolPhase.POST, is_error=True,
        ),
        primary_effects=[],  # no EDIT_MISS
    )
    assert _loop_detector_action(ctx) == []


def test_loop_detector_no_emit_below_threshold():
    # One prior same-hash failure + current (in-flight) = 2 < THRESHOLD(3)
    a1, t1 = _edit_attempt_pair("tu_1", "a.py", "ghost")
    frame = _make_frame([a1, t1])
    ctx = InterventionContext(
        session_id="s",
        now=time.time(),
        registry=None,
        frame=frame,
        tool_call=_edit_tool_call_info(
            "a.py", "ghost", phase=ToolPhase.POST, is_error=True,
        ),
        primary_effects=_edit_miss_primary_effects(),
    )
    assert _loop_detector_action(ctx) == []


def test_loop_detector_emits_loop_detected_at_threshold():
    # Two prior same-hash failures + current = 3 == THRESHOLD → emit.
    a1, t1 = _edit_attempt_pair("tu_1", "a.py", "ghost")
    a2, t2 = _edit_attempt_pair("tu_2", "a.py", "ghost")
    frame = _make_frame([a1, t1, a2, t2])
    ctx = InterventionContext(
        session_id="s",
        now=time.time(),
        registry=None,
        frame=frame,
        tool_call=_edit_tool_call_info(
            "a.py", "ghost", phase=ToolPhase.POST, is_error=True,
        ),
        primary_effects=_edit_miss_primary_effects(),
    )
    effects = _loop_detector_action(ctx)
    assert len(effects) == 1
    assert isinstance(effects[0], EmitEvent)
    assert effects[0].event_type == EventType.LOOP_DETECTED
    assert effects[0].payload.attempts == THRESHOLD


# ──────────────────────────────────────────────────────────────────────
# loop_block (Gate)
# ──────────────────────────────────────────────────────────────────────


def test_loop_block_allows_call_below_threshold():
    # One prior same-hash failure + incoming = 2 < THRESHOLD → allow.
    a1, t1 = _edit_attempt_pair("tu_1", "a.py", "ghost")
    frame = _make_frame([a1, t1])
    ctx = InterventionContext(
        session_id="s",
        now=time.time(),
        registry=None,
        frame=frame,
        tool_call=_edit_tool_call_info(
            "a.py", "ghost", phase=ToolPhase.PRE,
        ),
    )
    assert _loop_block_action(ctx) == []


def test_loop_block_refuses_call_at_threshold():
    a1, t1 = _edit_attempt_pair("tu_1", "a.py", "ghost")
    a2, t2 = _edit_attempt_pair("tu_2", "a.py", "ghost")
    frame = _make_frame([a1, t1, a2, t2])
    ctx = InterventionContext(
        session_id="s",
        now=time.time(),
        registry=None,
        frame=frame,
        tool_call=_edit_tool_call_info(
            "a.py", "ghost", phase=ToolPhase.PRE,
        ),
    )
    effects = _loop_block_action(ctx)
    assert len(effects) == 1
    assert isinstance(effects[0], Block)
    assert "ghost" not in effects[0].reason  # reason text is agnostic to the content
    assert effects[0].trace_event == EventType.LOOP_BLOCKED


def test_loop_block_ignores_different_hash():
    # Prior failures exist for one target, but the incoming call has
    # a different old_string → fresh hash → allow.
    a1, t1 = _edit_attempt_pair("tu_1", "a.py", "ghost")
    a2, t2 = _edit_attempt_pair("tu_2", "a.py", "ghost")
    frame = _make_frame([a1, t1, a2, t2])
    ctx = InterventionContext(
        session_id="s",
        now=time.time(),
        registry=None,
        frame=frame,
        tool_call=_edit_tool_call_info(
            "a.py", "a_completely_different_target", phase=ToolPhase.PRE,
        ),
    )
    assert _loop_block_action(ctx) == []


# ──────────────────────────────────────────────────────────────────────
# pack registration
# ──────────────────────────────────────────────────────────────────────


def test_pack_install_registers_all_four_interventions():
    reg = PackRegistry()
    install(reg)
    ids = set(reg.interventions.keys())
    assert "edit_guards.fuzzy_suggest" in ids
    assert "edit_guards.reread_hint" in ids
    assert "edit_guards.loop_detector" in ids
    assert "edit_guards.loop_block" in ids


def test_pack_install_is_idempotent():
    reg = PackRegistry()
    install(reg)
    install(reg)  # second install should not duplicate entries
    post_list = [
        i for i in reg.interventions.values()
        if i.kind == "listener"
        and i.phase == InterventionPhase.POST_EFFECT
        and i.id == "edit_guards.loop_detector"
    ]
    assert len(post_list) == 1


# ──────────────────────────────────────────────────────────────────────
# Registry-level integration (PRIMARY → POST_EFFECT ordering)
# ──────────────────────────────────────────────────────────────────────


def test_registry_runs_fuzzy_before_loop_detector_in_phase_order():
    """fuzzy_suggest (PRIMARY) must run first, then loop_detector
    (POST_EFFECT) must see fuzzy_suggest's EDIT_MISS emit in
    ctx.primary_effects."""
    reg = PackRegistry()
    install(reg)

    # Build a body with 2 prior same-hash failures so the next miss
    # hits the loop_detector threshold.
    a1, t1 = _edit_attempt_pair("tu_1", "/nowhere/z.py", "ghost")
    a2, t2 = _edit_attempt_pair("tu_2", "/nowhere/z.py", "ghost")
    frame = _make_frame([a1, t1, a2, t2])

    ctx = InterventionContext(
        session_id="s",
        now=time.time(),
        registry=reg,
        frame=frame,
        tool_call=_edit_tool_call_info(
            "/nowhere/z.py", "ghost", phase=ToolPhase.POST, is_error=True,
        ),
    )
    effects = asyncio.run(reg.dispatch_tool(ToolPhase.POST, ctx))

    emitted_types = [
        e.event_type for e in effects
        if isinstance(e, EmitEvent)
    ]
    # fuzzy_suggest emits EDIT_MISS in PRIMARY
    assert EventType.EDIT_MISS in emitted_types
    # loop_detector saw it (POST_EFFECT) and emitted LOOP_DETECTED
    assert EventType.LOOP_DETECTED in emitted_types
    # EDIT_MISS must come before LOOP_DETECTED in the final effect list,
    # reflecting phase ordering.
    edit_miss_idx = next(
        i for i, e in enumerate(effects)
        if isinstance(e, EmitEvent) and e.event_type == EventType.EDIT_MISS
    )
    loop_idx = next(
        i for i, e in enumerate(effects)
        if isinstance(e, EmitEvent) and e.event_type == EventType.LOOP_DETECTED
    )
    assert edit_miss_idx < loop_idx
