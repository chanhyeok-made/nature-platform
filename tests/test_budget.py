"""Tests for the budget builtin Pack (Phase 3.1 — reads_budget)."""

from __future__ import annotations

import time

import pytest

from nature.context.conversation import Conversation, Message
from nature.context.types import (
    AgentRole,
    Context,
    ContextBody,
    ContextHeader,
)
from nature.events.types import EventType
from nature.frame.frame import Frame
from nature.packs.builtin.budget.pack import install
from nature.packs.builtin.budget.reads_budget import (
    DEFAULT_LIMIT,
    READ_FAMILY,
    WARN_RATIO,
    count_read_family_calls,
    reads_gate,
    reads_warning,
)
from nature.packs.registry import PackRegistry
from nature.packs.types import (
    AppendFooter,
    Block,
    InterventionContext,
    ToolCallInfo,
    ToolPhase,
)
from nature.protocols.message import (
    ToolResultContent,
    ToolUseContent,
)


# ──────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────


def _make_frame(messages: list[Message]) -> Frame:
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


def _read_pair(tool_use_id: str, tool_name: str = "Read") -> tuple[Message, Message]:
    """Build (assistant with Read tool_use, tool with matching result)."""
    assistant = Message(
        id=f"msg_asst_{tool_use_id}",
        from_="self",
        to="tool",
        content=[
            ToolUseContent(
                id=tool_use_id,
                name=tool_name,
                input={"file_path": "x.py"},
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
                content="file content",
                is_error=False,
            ),
        ],
        timestamp=time.time(),
    )
    return assistant, tool_result


def _build_n_reads(n: int, tool_name: str = "Read") -> list[Message]:
    msgs: list[Message] = []
    for i in range(n):
        a, t = _read_pair(f"tu_{i}", tool_name=tool_name)
        msgs.extend([a, t])
    return msgs


# ──────────────────────────────────────────────────────────────────────
# count_read_family_calls
# ──────────────────────────────────────────────────────────────────────


def test_count_empty_body():
    frame = _make_frame([])
    assert count_read_family_calls(frame.context.body.conversation) == 0


def test_count_reads_only():
    msgs = _build_n_reads(5)
    frame = _make_frame(msgs)
    assert count_read_family_calls(frame.context.body.conversation) == 5


def test_count_grep_and_glob_included():
    msgs = _build_n_reads(3, "Grep") + _build_n_reads(2, "Glob")
    frame = _make_frame(msgs)
    assert count_read_family_calls(frame.context.body.conversation) == 5


def test_count_ignores_edit_and_write():
    msgs = _build_n_reads(2, "Edit") + _build_n_reads(1, "Write") + _build_n_reads(3)
    frame = _make_frame(msgs)
    assert count_read_family_calls(frame.context.body.conversation) == 3


def test_count_ignores_unpaired_trailing_tool_use():
    msgs = _build_n_reads(2)
    # Add an unpaired trailing tool_use
    msgs.append(Message(
        id="trailing",
        from_="self",
        to="tool",
        content=[ToolUseContent(id="tu_orphan", name="Read", input={})],
        timestamp=time.time(),
    ))
    frame = _make_frame(msgs)
    assert count_read_family_calls(frame.context.body.conversation) == 2


# ──────────────────────────────────────────────────────────────────────
# reads_gate
# ──────────────────────────────────────────────────────────────────────


def test_gate_allows_under_limit():
    msgs = _build_n_reads(DEFAULT_LIMIT - 2)
    frame = _make_frame(msgs)
    ctx = InterventionContext(
        session_id="s",
        now=time.time(),
        registry=None,
        frame=frame,
        tool_call=ToolCallInfo(
            tool_name="Read",
            tool_use_id="tu_new",
            tool_input={"file_path": "y.py"},
            phase=ToolPhase.PRE,
        ),
    )
    effects = reads_gate.action(ctx)
    assert effects == []


def test_gate_blocks_at_limit():
    msgs = _build_n_reads(DEFAULT_LIMIT)
    frame = _make_frame(msgs)
    ctx = InterventionContext(
        session_id="s",
        now=time.time(),
        registry=None,
        frame=frame,
        tool_call=ToolCallInfo(
            tool_name="Read",
            tool_use_id="tu_new",
            tool_input={"file_path": "y.py"},
            phase=ToolPhase.PRE,
        ),
    )
    effects = reads_gate.action(ctx)
    assert len(effects) == 1
    assert isinstance(effects[0], Block)
    assert effects[0].trace_event == EventType.BUDGET_BLOCKED
    assert "limit" in effects[0].reason.lower() or "budget" in effects[0].reason.lower()


def test_gate_ignores_non_read_family():
    msgs = _build_n_reads(DEFAULT_LIMIT + 5)
    frame = _make_frame(msgs)
    ctx = InterventionContext(
        session_id="s",
        now=time.time(),
        registry=None,
        frame=frame,
        tool_call=ToolCallInfo(
            tool_name="Edit",  # not in READ_FAMILY
            tool_use_id="tu_edit",
            tool_input={},
            phase=ToolPhase.PRE,
        ),
    )
    assert reads_gate.action(ctx) == []


# ──────────────────────────────────────────────────────────────────────
# reads_warning (Contributor)
# ──────────────────────────────────────────────────────────────────────


def test_warning_silent_below_80_percent():
    n = int(DEFAULT_LIMIT * WARN_RATIO) - 1
    msgs = _build_n_reads(n)
    body = ContextBody(conversation=Conversation(messages=msgs))
    ctx = InterventionContext(
        session_id="s",
        now=time.time(),
        registry=None,
        body=body,
    )
    assert reads_warning.action(ctx) == []


def test_warning_fires_at_80_percent():
    n = int(DEFAULT_LIMIT * WARN_RATIO)
    msgs = _build_n_reads(n)
    body = ContextBody(conversation=Conversation(messages=msgs))
    ctx = InterventionContext(
        session_id="s",
        now=time.time(),
        registry=None,
        body=body,
    )
    effects = reads_warning.action(ctx)
    assert len(effects) == 1
    assert isinstance(effects[0], AppendFooter)
    assert "reads_budget" in effects[0].source_id
    assert str(n) in effects[0].text


def test_warning_silent_at_100_percent():
    msgs = _build_n_reads(DEFAULT_LIMIT)
    body = ContextBody(conversation=Conversation(messages=msgs))
    ctx = InterventionContext(
        session_id="s",
        now=time.time(),
        registry=None,
        body=body,
    )
    # At limit, the gate handles it — no redundant footer hint.
    assert reads_warning.action(ctx) == []


# ──────────────────────────────────────────────────────────────────────
# pack registration
# ──────────────────────────────────────────────────────────────────────


def test_pack_install_registers_gate_and_warning():
    reg = PackRegistry()
    install(reg)
    ids = set(reg.interventions.keys())
    assert "reads_budget.gate" in ids
    assert "reads_budget.warning" in ids


def test_pack_install_idempotent():
    reg = PackRegistry()
    install(reg)
    install(reg)
    assert len([
        i for i in reg.interventions.values()
        if i.id.startswith("reads_budget.")
    ]) == 2
