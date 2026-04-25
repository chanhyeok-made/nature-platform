"""Tests for the Frame dataclass (Step 4)."""

from __future__ import annotations

from nature.context import AgentRole, Context, ContextHeader
from nature.frame import Frame, FrameState


def test_frame_defaults():
    role = AgentRole(name="receptionist", instructions="x")
    ctx = Context(header=ContextHeader(role=role))
    frame = Frame(
        id="f1",
        session_id="s1",
        purpose="root",
        context=ctx,
        model="fake",
    )

    assert frame.state == FrameState.ACTIVE
    assert frame.is_root is True
    assert frame.self_actor == "receptionist"
    assert frame.children_ids == []
    assert frame.parent_id is None


def test_frame_is_root_false_when_parent_set():
    role = AgentRole(name="core", instructions="x")
    ctx = Context(header=ContextHeader(role=role))
    frame = Frame(
        id="f2",
        session_id="s1",
        purpose="delegation",
        context=ctx,
        model="fake",
        parent_id="f1",
    )

    assert frame.is_root is False


def test_frame_state_transitions():
    role = AgentRole(name="r", instructions="x")
    ctx = Context(header=ContextHeader(role=role))
    frame = Frame(id="f", session_id="s", purpose="p", context=ctx, model="m")

    frame.state = FrameState.AWAITING_USER
    assert frame.state == FrameState.AWAITING_USER

    frame.state = FrameState.RESOLVED
    assert frame.state == FrameState.RESOLVED

    frame.state = FrameState.CLOSED
    assert frame.state == FrameState.CLOSED


def test_frame_self_actor_reflects_role_name():
    role = AgentRole(name="researcher", instructions="explore")
    ctx = Context(header=ContextHeader(role=role))
    frame = Frame(id="f", session_id="s", purpose="p", context=ctx, model="m")

    assert frame.self_actor == "researcher"

    # Swapping role changes self_actor
    new_role = AgentRole(name="analyzer", instructions="analyze")
    frame.context = frame.context.with_role(new_role)
    assert frame.self_actor == "analyzer"
