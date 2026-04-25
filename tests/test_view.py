"""Tests for nature.server.view.build_session_view.

Exercises the event-log → structured-turn-tree transform against
synthetic event sequences covering: empty session, single-turn
Q&A, turn with tool calls, sub-agent delegation, error frames,
running-turn state, and multi-turn sessions.
"""

from __future__ import annotations

from typing import Any

import pytest

from nature.events.types import Event, EventType
from nature.server.view import (
    MessageDto,
    SessionViewDto,
    StepDto,
    SubAgentDto,
    ToolDto,
    TurnDto,
    build_session_view,
)


# ---------------------------------------------------------------------------
# Event fixture helpers
# ---------------------------------------------------------------------------


_NEXT_ID = [0]
_NEXT_TS = [1000.0]


def _reset() -> None:
    _NEXT_ID[0] = 0
    _NEXT_TS[0] = 1000.0


def _ev(
    event_type: EventType,
    *,
    frame_id: str = "root",
    payload: dict[str, Any] | None = None,
    dt: float = 0.1,
) -> Event:
    _NEXT_ID[0] += 1
    _NEXT_TS[0] += dt
    return Event(
        id=_NEXT_ID[0],
        session_id="s1",
        frame_id=frame_id,
        timestamp=_NEXT_TS[0],
        type=event_type,
        payload=payload or {},
    )


def _frame_opened(
    frame_id: str,
    role: str = "receptionist",
    *,
    parent: str | None = None,
    tool_use: str | None = None,
    purpose: str = "",
) -> Event:
    return _ev(
        EventType.FRAME_OPENED,
        frame_id=frame_id,
        payload={
            "role_name": role,
            "purpose": purpose,
            "parent_id": parent,
            "model": "qwen2.5",
            "spawned_by_tool_use_id": tool_use,
        },
    )


def _user_msg(text: str, *, frame_id: str = "root", to: str = "receptionist") -> Event:
    mid = f"msg_{_NEXT_ID[0] + 1}u"
    return _ev(
        EventType.MESSAGE_APPENDED,
        frame_id=frame_id,
        payload={
            "message_id": mid,
            "from_": "user",
            "to": to,
            "content": [{"type": "text", "text": text}],
            "timestamp": _NEXT_TS[0] + 0.1,
        },
    )


def _assistant_msg(
    text: str,
    *,
    frame_id: str = "root",
    from_: str = "receptionist",
    to: str = "user",
) -> Event:
    mid = f"msg_{_NEXT_ID[0] + 1}a"
    return _ev(
        EventType.MESSAGE_APPENDED,
        frame_id=frame_id,
        payload={
            "message_id": mid,
            "from_": from_,
            "to": to,
            "content": [{"type": "text", "text": text}],
            "timestamp": _NEXT_TS[0] + 0.1,
        },
    )


def _tool_started(
    name: str,
    tool_use_id: str,
    *,
    frame_id: str = "root",
    tool_input: dict | None = None,
) -> Event:
    return _ev(
        EventType.TOOL_STARTED,
        frame_id=frame_id,
        payload={
            "tool_use_id": tool_use_id,
            "tool_name": name,
            "tool_input": tool_input or {},
        },
    )


def _tool_completed(
    tool_use_id: str,
    *,
    frame_id: str = "root",
    output: str = "",
    is_error: bool = False,
    duration_ms: int = 42,
) -> Event:
    return _ev(
        EventType.TOOL_COMPLETED,
        frame_id=frame_id,
        payload={
            "tool_use_id": tool_use_id,
            "tool_name": "Bash",
            "output": output,
            "is_error": is_error,
            "duration_ms": duration_ms,
        },
    )


def _annotation(
    message_id: str,
    *,
    frame_id: str = "root",
    stop_reason: str = "end_turn",
    input_tokens: int = 100,
    output_tokens: int = 20,
    thinking: list[str] | None = None,
    duration_ms: int = 500,
) -> Event:
    return _ev(
        EventType.ANNOTATION_STORED,
        frame_id=frame_id,
        payload={
            "message_id": message_id,
            "stop_reason": stop_reason,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
            "duration_ms": duration_ms,
            "thinking": thinking,
            "llm_request_id": "req_x",
        },
    )


def _llm_request(frame_id: str = "root") -> Event:
    return _ev(
        EventType.LLM_REQUEST,
        frame_id=frame_id,
        payload={"request_id": "req_x", "model": "qwen2.5", "message_count": 2, "tool_count": 4},
    )


def _llm_response(frame_id: str = "root", usage: dict | None = None) -> Event:
    return _ev(
        EventType.LLM_RESPONSE,
        frame_id=frame_id,
        payload={
            "request_id": "req_x",
            "stop_reason": "end_turn",
            "usage": usage or {"input_tokens": 100, "output_tokens": 20},
        },
    )


def _frame_resolved(frame_id: str = "root") -> Event:
    return _ev(EventType.FRAME_RESOLVED, frame_id=frame_id, payload={})


def _frame_errored(frame_id: str = "root", msg: str = "boom") -> Event:
    return _ev(
        EventType.FRAME_ERRORED,
        frame_id=frame_id,
        payload={"error_type": "RuntimeError", "message": msg},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_empty_session_returns_skeleton():
    _reset()
    v = build_session_view([], session_id="s1", role_name="receptionist", model="qwen2.5")
    assert v.session_id == "s1"
    assert v.turns == []
    assert v.pulse.active is False
    assert v.state == "active"


def test_single_turn_simple_qa():
    """user → assistant → done. The assistant message becomes the final
    reply, steps is empty, summary is zero-everything."""
    _reset()
    events = [
        _frame_opened("root"),
        _user_msg("hi"),
        _llm_request(),
        _assistant_msg("hello there"),
        _annotation("msg_4a"),
        _llm_response(),
    ]
    v = build_session_view(events, session_id="s1")
    assert len(v.turns) == 1
    turn = v.turns[0]
    assert turn.user_message.text == "hi"
    assert turn.final_message is not None
    assert turn.final_message.text == "hello there"
    assert turn.final_message.annotation is not None
    assert turn.final_message.annotation.stop_reason == "end_turn"
    assert turn.steps == []
    assert turn.summary.step_count == 0
    assert turn.summary.tool_count == 0
    assert turn.summary.sub_agent_count == 0


def test_turn_with_tool_calls():
    """Intermediate assistant message + tool call + final assistant.
    The intermediate assistant stays in steps, the tool becomes a step,
    the final assistant is promoted to final_message."""
    _reset()
    events = [
        _frame_opened("root"),
        _user_msg("make todo app"),
        _assistant_msg("I'll create it"),  # intermediate
        _tool_started("Bash", "tu1", tool_input={"cmd": "mkdir todo"}),
        _tool_completed("tu1", output="ok"),
        _assistant_msg("Done — created the folder"),  # final
        _annotation("msg_7a"),
    ]
    v = build_session_view(events, session_id="s1")
    turn = v.turns[0]

    assert turn.user_message.text == "make todo app"
    assert turn.final_message is not None
    assert turn.final_message.text == "Done — created the folder"
    assert len(turn.steps) == 2
    assert turn.steps[0].kind == "message"
    assert turn.steps[0].message.text == "I'll create it"
    assert turn.steps[1].kind == "tool"
    assert turn.steps[1].tool.tool_name == "Bash"
    assert turn.steps[1].tool.tool_input == {"cmd": "mkdir todo"}
    assert turn.steps[1].tool.output == "ok"
    assert turn.steps[1].tool.is_error is False
    assert turn.steps[1].tool.duration_ms == 42
    assert turn.summary.tool_count == 1
    assert turn.summary.step_count == 2


def test_sub_agent_delegation_becomes_sub_agent_step():
    """An Agent tool call that spawned a child frame renders as a
    sub_agent step (not a plain tool step), with the child's turns
    nested inside.

    Real nature emits the child frame's first message with
    from_=<parent_role> (not "user"), so the turn boundary logic has
    to treat any "incoming to self actor" message as a new turn,
    not just ones from "user".
    """
    _reset()
    # Child frame's opening delegation message — from the parent role
    # (receptionist), to the child role (researcher). Use the raw _ev
    # helper since _user_msg hardcodes from_="user".
    def _delegation(text: str) -> Event:
        return _ev(
            EventType.MESSAGE_APPENDED,
            frame_id="child",
            payload={
                "message_id": f"msg_{_NEXT_ID[0] + 1}d",
                "from_": "receptionist",
                "to": "researcher",
                "content": [{"type": "text", "text": text}],
                "timestamp": _NEXT_TS[0] + 0.1,
            },
        )

    events = [
        _frame_opened("root"),
        _user_msg("research X"),
        _tool_started("Agent", "tu_agent", tool_input={"purpose": "research"}),
        _frame_opened("child", role="researcher", parent="root",
                      tool_use="tu_agent", purpose="research X"),
        _delegation("research X in depth"),
        _tool_started("WebSearch", "tu_ws", frame_id="child"),
        _tool_completed("tu_ws", frame_id="child", output="found 10 results"),
        _assistant_msg("Summary of findings…", frame_id="child",
                       from_="researcher", to="receptionist"),
        _frame_resolved("child"),
        _tool_completed("tu_agent", output="delegation complete"),
        _assistant_msg("Here's what I found: …"),
        _annotation("msg_12a"),
    ]
    v = build_session_view(events, session_id="s1")
    turn = v.turns[0]

    # There should be exactly one step on the parent — the sub_agent
    # (the Agent tool call itself is not rendered as a separate tool).
    assert len(turn.steps) == 1
    step = turn.steps[0]
    assert step.kind == "sub_agent"
    assert step.sub_agent is not None
    assert step.sub_agent.role_name == "researcher"
    assert step.sub_agent.state == "resolved"
    assert step.sub_agent.spawned_by_tool_use_id == "tu_agent"

    # The child frame had one turn, seeded by the delegation message
    assert len(step.sub_agent.turns) == 1
    child_turn = step.sub_agent.turns[0]
    assert child_turn.user_message.text == "research X in depth"
    assert child_turn.user_message.from_ == "receptionist"
    assert child_turn.final_message is not None
    assert child_turn.final_message.text == "Summary of findings…"
    assert child_turn.final_message.from_ == "researcher"

    # returned_text mirrors the child's last final_message so the
    # parent card can show "what came back" without digging into
    # nested turns
    assert step.sub_agent.returned_text == "Summary of findings…"
    assert step.sub_agent.returned_message_id == child_turn.final_message.message_id
    # WebSearch tool was regular (not a sub-agent) so it lives in steps
    assert len(child_turn.steps) == 1
    assert child_turn.steps[0].kind == "tool"
    assert child_turn.steps[0].tool.tool_name == "WebSearch"

    # Parent final reply is the outer assistant message
    assert turn.final_message is not None
    assert turn.final_message.text == "Here's what I found: …"
    assert turn.summary.sub_agent_count == 1


def test_multi_turn_session():
    """Two user messages on the same root frame → two turns."""
    _reset()
    events = [
        _frame_opened("root"),
        _user_msg("first"),
        _assistant_msg("reply one"),
        _user_msg("second"),
        _assistant_msg("reply two"),
    ]
    v = build_session_view(events, session_id="s1")
    assert len(v.turns) == 2
    assert v.turns[0].user_message.text == "first"
    assert v.turns[0].final_message.text == "reply one"
    assert v.turns[1].user_message.text == "second"
    assert v.turns[1].final_message.text == "reply two"


def test_running_turn_has_no_final_yet():
    """A turn with only a user message and a pending llm.request should
    have final_message=None and pulse.active=True."""
    _reset()
    events = [
        _frame_opened("root"),
        _user_msg("think hard"),
        _llm_request(),
    ]
    v = build_session_view(events, session_id="s1")
    turn = v.turns[0]
    assert turn.final_message is None
    assert turn.steps == []
    assert v.pulse.active is True
    assert v.pulse.activity == "thinking"


def test_pulse_stops_after_llm_response():
    """After the llm.request/response pair, pulse should be inactive."""
    _reset()
    events = [
        _frame_opened("root"),
        _user_msg("hi"),
        _llm_request(),
        _assistant_msg("hi back"),
        _annotation("msg_4a"),
        _llm_response(),
    ]
    v = build_session_view(events, session_id="s1")
    assert v.pulse.active is False


def test_frame_errored_sets_turn_state_error():
    """A frame.errored event marks the enclosing turn as error state."""
    _reset()
    events = [
        _frame_opened("root"),
        _user_msg("go"),
        _llm_request(),
        _frame_errored("root", msg="llm blew up"),
    ]
    v = build_session_view(events, session_id="s1")
    turn = v.turns[0]
    assert turn.state == "error"
    assert v.state == "error"


def test_tool_result_message_becomes_received_step():
    """Messages with from_='tool' should become a 'received' step so
    the aggregation point (the moment the agent receives tool
    results) is visible in the timeline instead of silently skipped.
    """
    _reset()
    events = [
        _frame_opened("root"),
        _user_msg("go"),
        _tool_started("Bash", "tu1"),
        _tool_completed("tu1", output="ok"),
        # Bundled tool_result message from the tool bus back to the agent
        _ev(
            EventType.MESSAGE_APPENDED,
            payload={
                "message_id": "msg_tool",
                "from_": "tool",
                "to": "receptionist",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tu1", "content": "ok", "is_error": False},
                ],
                "timestamp": _NEXT_TS[0],
            },
        ),
        _assistant_msg("done"),
    ]
    v = build_session_view(events, session_id="s1")
    turn = v.turns[0]
    # steps: [tool, received]
    assert [s.kind for s in turn.steps] == ["tool", "received"]

    rec = turn.steps[1].received
    assert rec is not None
    assert rec.message_id == "msg_tool"
    assert len(rec.results) == 1
    assert rec.results[0].tool_use_id == "tu1"
    assert rec.results[0].content == "ok"
    assert rec.results[0].is_error is False

    assert turn.final_message.text == "done"
    assert turn.summary.received_count == 1
    assert turn.summary.tool_count == 1


def test_received_step_with_multiple_tool_results():
    """A bundled tool_result message with N blocks produces one
    'received' step containing N ToolResultBlockDtos — this is the
    parallel-sub-agent pattern."""
    _reset()
    events = [
        _frame_opened("root", role="core"),
        _ev(EventType.MESSAGE_APPENDED, frame_id="root", payload={
            "message_id": "u1",
            "from_": "receptionist",
            "to": "core",
            "content": [{"type": "text", "text": "do stuff"}],
            "timestamp": _NEXT_TS[0],
        }),
        _tool_started("Agent", "tu_a", tool_input={"name": "a"}),
        _tool_started("Agent", "tu_b", tool_input={"name": "b"}),
        _tool_completed("tu_a", output="a done"),
        _tool_completed("tu_b", output="b done"),
        # Bundled envelope with two tool_result blocks
        _ev(EventType.MESSAGE_APPENDED, frame_id="root", payload={
            "message_id": "msg_bundle",
            "from_": "tool",
            "to": "core",
            "content": [
                {"type": "tool_result", "tool_use_id": "tu_a", "content": "a done"},
                {"type": "tool_result", "tool_use_id": "tu_b", "content": "b done"},
            ],
            "timestamp": _NEXT_TS[0],
        }),
        _assistant_msg("synth", frame_id="root", from_="core", to="receptionist"),
    ]
    v = build_session_view(events, session_id="s1")
    turn = v.turns[0]
    # Two tool steps (both Agent calls without child frames, since
    # this test doesn't open them) + one received step
    rec_steps = [s for s in turn.steps if s.kind == "received"]
    assert len(rec_steps) == 1
    assert len(rec_steps[0].received.results) == 2
    assert rec_steps[0].received.results[0].tool_use_id == "tu_a"
    assert rec_steps[0].received.results[1].tool_use_id == "tu_b"
    assert turn.summary.received_count == 1


def test_session_meta_recovered_from_root_frame_when_not_passed():
    """If the caller doesn't pass role_name/model, they're recovered
    from the root frame.opened payload."""
    _reset()
    events = [
        _frame_opened("root", role="receptionist"),
        _user_msg("hi"),
    ]
    v = build_session_view(events, session_id="s1")
    assert v.role_name == "receptionist"
    assert v.model == "qwen2.5"


def test_message_after_received_is_marked_regenerated_from():
    """An assistant message produced by an LLM call that ingested a
    tool_result envelope gets `regenerated_from` populated with the
    bundled tool_use_ids — so the UI can flag it as a synthesis
    output rather than a plain reply.
    """
    _reset()
    events = [
        _frame_opened("root", role="receptionist"),
        _user_msg("hi"),
        _tool_started("Bash", "tu1"),
        _tool_completed("tu1", output="ok"),
        _ev(EventType.MESSAGE_APPENDED, payload={
            "message_id": "msg_tool",
            "from_": "tool",
            "to": "receptionist",
            "content": [
                {"type": "tool_result", "tool_use_id": "tu1", "content": "ok"},
            ],
            "timestamp": _NEXT_TS[0],
        }),
        _assistant_msg("here is what I did"),
    ]
    v = build_session_view(events, session_id="s1")
    turn = v.turns[0]
    assert turn.final_message is not None
    assert turn.final_message.regenerated_from == ["tu1"]
    # Tool steps and any message that came BEFORE the received envelope
    # don't get the flag
    assert turn.user_message.regenerated_from == []


def test_todo_written_surfaces_on_session_view_root():
    """A TODO_WRITTEN event on the root frame should land in
    SessionViewDto.root_todos as the latest list (full overwrite
    semantics — only the most recent write survives)."""
    _reset()
    from nature.protocols.todo import TodoItem
    events = [
        _frame_opened("root"),
        _user_msg("do things"),
        _ev(EventType.TODO_WRITTEN, payload={
            "todos": [
                {"content": "A", "activeForm": "Doing A", "status": "pending"},
                {"content": "B", "activeForm": "Doing B", "status": "pending"},
            ],
            "source": "todo_write_tool",
        }),
        # Overwrite: A now in_progress, B still pending
        _ev(EventType.TODO_WRITTEN, payload={
            "todos": [
                {"content": "A", "activeForm": "Doing A", "status": "in_progress"},
                {"content": "B", "activeForm": "Doing B", "status": "pending"},
            ],
            "source": "todo_write_tool",
        }),
    ]
    v = build_session_view(events, session_id="s1")
    assert len(v.root_todos) == 2
    assert v.root_todos[0].status == "in_progress"
    assert v.root_todos[0].activeForm == "Doing A"
    assert v.root_todos[1].status == "pending"


def test_todo_written_surfaces_on_sub_agent():
    """A TODO_WRITTEN event on a child frame should land on the
    matching SubAgentDto.todos, not on root_todos."""
    _reset()

    def _delegation(text: str, to: str) -> Event:
        return _ev(
            EventType.MESSAGE_APPENDED,
            frame_id="child",
            payload={
                "message_id": f"msg_{_NEXT_ID[0] + 1}d",
                "from_": "receptionist",
                "to": to,
                "content": [{"type": "text", "text": text}],
                "timestamp": _NEXT_TS[0] + 0.1,
            },
        )

    events = [
        _frame_opened("root"),
        _user_msg("go"),
        _tool_started("Agent", "tu1", tool_input={"name": "researcher"}),
        _frame_opened("child", role="researcher", parent="root",
                      tool_use="tu1"),
        _delegation("research X", "researcher"),
        # Child frame writes its own todo list
        _ev(EventType.TODO_WRITTEN, frame_id="child", payload={
            "todos": [
                {"content": "Grep for X", "activeForm": "Grepping for X",
                 "status": "in_progress"},
                {"content": "Summarize", "activeForm": "Summarizing",
                 "status": "pending"},
            ],
            "source": "todo_write_tool",
        }),
        _assistant_msg("findings", frame_id="child",
                       from_="researcher", to="receptionist"),
        _frame_resolved("child"),
        _tool_completed("tu1"),
        _assistant_msg("done"),
    ]
    v = build_session_view(events, session_id="s1")
    # Root has no todos — the TODO_WRITTEN was scoped to the child
    assert v.root_todos == []
    # Sub-agent carries its own todos
    sub_step = v.turns[0].steps[0]
    assert sub_step.kind == "sub_agent"
    assert len(sub_step.sub_agent.todos) == 2
    assert sub_step.sub_agent.todos[0].status == "in_progress"
    assert sub_step.sub_agent.todos[0].activeForm == "Grepping for X"


def test_hint_injected_attaches_to_next_assistant_message():
    """A HINT_INJECTED event records which footer-rule nudges the
    composer appended to the upcoming LLM_REQUEST. The assistant
    message that the LLM produces in response to that request should
    carry those hints on its `injected_hints` field — and only that
    message, not the user message before it or a later message on
    the same frame."""
    _reset()
    events = [
        _frame_opened("root"),
        _user_msg("go"),
        # An earlier assistant reply, BEFORE any hint fires → clean.
        _assistant_msg("sure, starting now"),
        # Framework whispers into the next LLM_REQUEST.
        _ev(EventType.HINT_INJECTED, payload={
            "request_id": "req_1",
            "hints": [
                {"source": "todo_continues_after_tool_result",
                 "text": "[FRAMEWORK NOTE] keep going"},
                {"source": "todo_needs_in_progress",
                 "text": "[FRAMEWORK NOTE] pick the next item"},
            ],
        }),
        # The message produced by that LLM_REQUEST.
        _assistant_msg("continuing down the list"),
    ]
    v = build_session_view(events, session_id="s1")
    turn = v.turns[0]
    # The first assistant message is now an intermediate step
    first_msg_step = next(
        s for s in turn.steps if s.kind == "message"
    )
    assert first_msg_step.message.injected_hints == []
    # The final assistant message carries both hints
    assert turn.final_message is not None
    sources = [h.source for h in turn.final_message.injected_hints]
    assert sources == [
        "todo_continues_after_tool_result",
        "todo_needs_in_progress",
    ]
    assert (
        "FRAMEWORK NOTE"
        in turn.final_message.injected_hints[0].text
    )


def test_hint_injected_consumed_by_exactly_one_message():
    """Each HINT_INJECTED must attach to exactly ONE downstream message
    — the immediate next one — so a second reply that had no nudges
    stays clean instead of inheriting the previous turn's hints."""
    _reset()
    events = [
        _frame_opened("root"),
        _user_msg("go"),
        _ev(EventType.HINT_INJECTED, payload={
            "request_id": "req_1",
            "hints": [{"source": "synthesis_nudge", "text": "[FRAMEWORK NOTE] X"}],
        }),
        _assistant_msg("nudged reply"),
        # Second reply with no new hint → must be clean.
        _assistant_msg("follow-up"),
    ]
    v = build_session_view(events, session_id="s1")
    turn = v.turns[0]
    # First assistant reply is the intermediate step, second is final.
    msg_step = next(s for s in turn.steps if s.kind == "message")
    assert [h.source for h in msg_step.message.injected_hints] == ["synthesis_nudge"]
    assert turn.final_message is not None
    assert turn.final_message.injected_hints == []


def test_turn_duration_computed_from_frame_lifecycle():
    """When a frame resolves, the last turn gets started_at + ended_at
    and summary.duration_ms."""
    _reset()
    events = [
        _frame_opened("root"),
        _user_msg("hi"),
        _assistant_msg("hello"),
        _frame_resolved("root"),
    ]
    v = build_session_view(events, session_id="s1")
    turn = v.turns[0]
    assert turn.state == "resolved"
    assert turn.ended_at is not None
    assert turn.summary.duration_ms is not None
    assert turn.summary.duration_ms >= 0
