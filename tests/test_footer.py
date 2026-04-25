"""Tests for the context footer rule pipeline.

The footer is a render-time computation in ContextComposer that takes
the current ContextBody (conversation + todos) plus header and emits
ephemeral hints to inject into the upcoming LLM request. This file
covers:

- synthesis_nudge_rule fires on a from_=tool tail message UNLESS
  pending todos are present (the todo-aware rule takes over)
- todo_needs_in_progress_rule fires when there are pending todos but
  none are in_progress
- todo_continues_after_tool_result_rule fires on a from_=tool tail
  message when pending todos remain (mutually exclusive with synth)
- compute_footer_hints aggregates rules and stays exception-safe
- ContextComposer.compose() actually appends a hint as a tail user
  message and surfaces hints on ComposedRequest.hints
"""

from __future__ import annotations

import time

from nature.context.composer import ComposedRequest, ContextComposer
from nature.context.conversation import Conversation, Message
from nature.context.footer import (
    FOOTER_RULES,
    Hint,
    ROLE_REQUIRED_TOOLS,
    compute_footer_hints,
    needs_required_tool_rule,
    synthesis_nudge_rule,
    todo_continues_after_tool_result_rule,
    todo_needs_in_progress_rule,
)
from nature.context.types import (
    AgentRole, Context, ContextBody, ContextHeader,
)
from nature.protocols.message import (
    Role,
    TextContent,
    ToolResultContent,
    ToolUseContent,
)
from nature.protocols.todo import TodoItem


def _ctx(messages: list[Message] | None = None,
         todos: list[TodoItem] | None = None,
         role_name: str = "core") -> Context:
    role = AgentRole(name=role_name, instructions="be useful")
    header = ContextHeader(role=role)
    body = ContextBody(
        conversation=Conversation(messages=list(messages or [])),
        todos=list(todos or []),
    )
    return Context(header=header, body=body)


def _msg(from_: str, to: str, content: list, ts: float | None = None) -> Message:
    return Message(
        from_=from_, to=to, content=content, timestamp=ts or time.time(),
    )


def _todo(content: str, status: str = "pending") -> TodoItem:
    return TodoItem(
        content=content, activeForm=f"Doing {content}", status=status,
    )


# ---------------------------------------------------------------------------
# synthesis_nudge_rule
# ---------------------------------------------------------------------------


def test_synthesis_nudge_fires_after_tool_result_for_reporter_role():
    msgs = [
        _msg("core", "researcher", [TextContent(text="find X")]),
        _msg("researcher", "core", [
            TextContent(text="let me read the file"),
            ToolUseContent(name="Read", input={"file_path": "/p/a.py"}, id="tu1"),
        ]),
        _msg("tool", "researcher", [
            ToolResultContent(tool_use_id="tu1", content="file.py"),
        ]),
    ]
    ctx = _ctx(msgs, role_name="researcher")
    hint = synthesis_nudge_rule(ctx.body, ctx.header, "researcher")
    assert hint is not None
    assert hint.source == "synthesis_nudge"
    assert "1 tool_result" in hint.text
    assert "[FRAMEWORK NOTE" in hint.text


def test_synthesis_nudge_counts_multiple_tool_results():
    msgs = [
        _msg("core", "researcher", [TextContent(text="go")]),
        _msg("researcher", "core", [
            ToolUseContent(name="Read", input={"file_path": "/a.py"}, id="tu_a"),
            ToolUseContent(name="Read", input={"file_path": "/b.py"}, id="tu_b"),
        ]),
        _msg("tool", "researcher", [
            ToolResultContent(tool_use_id="tu_a", content="a out"),
            ToolResultContent(tool_use_id="tu_b", content="b out"),
        ]),
    ]
    ctx = _ctx(msgs, role_name="researcher")
    hint = synthesis_nudge_rule(ctx.body, ctx.header, "researcher")
    assert hint is not None
    assert "2 tool_results" in hint.text


def test_synthesis_nudge_silent_for_delegator_role():
    """Delegator roles (receptionist, core) deliver by calling Agent,
    not by synthesizing text from tool results. Firing synthesis_nudge
    at them after every tool_result turns into hint spam — stage-1
    `1776665006-8ea9ec` logged 18 hints in a 19-turn receptionist
    session. New behaviour: silent unless role is a reporter."""
    msgs = [
        _msg("user", "core", [TextContent(text="analyze")]),
        _msg("core", "user", [
            ToolUseContent(name="Bash", input={"cmd": "ls"}, id="tu1"),
        ]),
        _msg("tool", "core", [
            ToolResultContent(tool_use_id="tu1", content="file.py"),
        ]),
    ]
    ctx = _ctx(msgs, role_name="core")
    assert synthesis_nudge_rule(ctx.body, ctx.header, "core") is None


def test_synthesis_nudge_silent_for_implementer_role():
    """Implementer is a worker role — it finishes when its edit loop
    is done, not when tools return. Synthesis_nudge on every
    tool_result would interrupt a legitimate Read→Edit→Read→Edit
    sequence."""
    msgs = [
        _msg("core", "implementer", [TextContent(text="add field")]),
        _msg("implementer", "core", [
            ToolUseContent(
                name="Edit",
                input={"file_path": "/p/a.py", "old_string": "a", "new_string": "b"},
                id="tu_edit",
            ),
        ]),
        _msg("tool", "implementer", [
            ToolResultContent(tool_use_id="tu_edit", content="edited"),
        ]),
    ]
    ctx = _ctx(msgs, role_name="implementer")
    assert synthesis_nudge_rule(ctx.body, ctx.header, "implementer") is None


def test_synthesis_nudge_silent_for_solo_role():
    """Single-agent preset's `solo` role — worker, not reporter.
    Synthesis_nudge was the dominant hint source for solo cells in
    stage-1 eval, entirely unhelpfully."""
    msgs = [
        _msg("user", "solo", [TextContent(text="fix it")]),
        _msg("solo", "user", [
            ToolUseContent(name="Read", input={"file_path": "/p/a.py"}, id="tu"),
        ]),
        _msg("tool", "solo", [
            ToolResultContent(tool_use_id="tu", content="contents"),
        ]),
    ]
    ctx = _ctx(msgs, role_name="solo")
    assert synthesis_nudge_rule(ctx.body, ctx.header, "solo") is None


def test_synthesis_nudge_silent_when_last_is_assistant():
    msgs = [
        _msg("user", "core", [TextContent(text="hi")]),
        _msg("core", "user", [TextContent(text="hello")]),
    ]
    ctx = _ctx(msgs)
    assert synthesis_nudge_rule(ctx.body, ctx.header, "core") is None


def test_synthesis_nudge_silent_when_tool_msg_has_no_tool_result_block():
    msgs = [
        _msg("tool", "core", [TextContent(text="not a real tool result")]),
    ]
    ctx = _ctx(msgs)
    assert synthesis_nudge_rule(ctx.body, ctx.header, "core") is None


def test_synthesis_nudge_silent_on_empty_body():
    ctx = _ctx()
    assert synthesis_nudge_rule(ctx.body, ctx.header, "core") is None


def test_synthesis_nudge_silent_when_pending_todos_remain():
    """When the LLM is mid-checklist, the todo-continues rule takes
    over — synthesis_nudge must yield."""
    msgs = [
        _msg("user", "core", [TextContent(text="go")]),
        _msg("tool", "core", [
            ToolResultContent(tool_use_id="tu1", content="ok"),
        ]),
    ]
    ctx = _ctx(msgs, todos=[
        _todo("A", "completed"), _todo("B", "pending"),
    ])
    assert synthesis_nudge_rule(ctx.body, ctx.header, "core") is None


def test_synthesis_nudge_silent_when_only_todowrite_in_tail():
    """Regression: session 409b958e — synthesis_nudge fired after every
    TodoWrite tool_result, treating self-bookkeeping as 'real work
    just completed'. The filter must skip TodoWrite results so
    bookkeeping doesn't trigger synthesis pressure."""
    msgs = [
        _msg("user", "core", [TextContent(text="hi")]),
        _msg("core", "user", [
            ToolUseContent(name="TodoWrite", input={"todos": []}, id="tw1"),
        ]),
        _msg("tool", "core", [
            ToolResultContent(tool_use_id="tw1", content="Updated"),
        ]),
    ]
    ctx = _ctx(msgs)  # no todos — would otherwise fire synthesis_nudge
    assert synthesis_nudge_rule(ctx.body, ctx.header, "core") is None


def test_synthesis_nudge_fires_when_all_todos_completed():
    """All todos done + reporter role + required tool used → synthesis_nudge fires."""
    msgs = [
        _msg("core", "researcher", [TextContent(text="go")]),
        _msg("researcher", "core", [
            ToolUseContent(name="Read", input={"file_path": "/a.py"}, id="tu1"),
        ]),
        _msg("tool", "researcher", [
            ToolResultContent(tool_use_id="tu1", content="ok"),
        ]),
    ]
    ctx = _ctx(msgs, todos=[
        _todo("A", "completed"), _todo("B", "completed"),
    ], role_name="researcher")
    hint = synthesis_nudge_rule(ctx.body, ctx.header, "researcher")
    assert hint is not None


# ---------------------------------------------------------------------------
# todo_needs_in_progress_rule
# ---------------------------------------------------------------------------


def test_todo_needs_in_progress_fires_when_pending_with_no_in_progress():
    ctx = _ctx(todos=[
        _todo("A", "pending"),
        _todo("B", "pending"),
    ])
    hint = todo_needs_in_progress_rule(ctx.body, ctx.header, "core")
    assert hint is not None
    assert hint.source == "todo_needs_in_progress"
    assert "in_progress" in hint.text
    # Mentions the next pending item by content
    assert "'A'" in hint.text


def test_todo_needs_in_progress_silent_when_in_progress_exists():
    ctx = _ctx(todos=[
        _todo("A", "in_progress"),
        _todo("B", "pending"),
    ])
    assert todo_needs_in_progress_rule(ctx.body, ctx.header, "core") is None


def test_todo_needs_in_progress_silent_when_all_completed():
    ctx = _ctx(todos=[
        _todo("A", "completed"),
        _todo("B", "completed"),
    ])
    assert todo_needs_in_progress_rule(ctx.body, ctx.header, "core") is None


def test_todo_needs_in_progress_silent_when_no_todos():
    ctx = _ctx()
    assert todo_needs_in_progress_rule(ctx.body, ctx.header, "core") is None


# ---------------------------------------------------------------------------
# todo_continues_after_tool_result_rule
# ---------------------------------------------------------------------------


def test_todo_continues_fires_on_tool_result_with_pending_todos():
    msgs = [
        _msg("user", "core", [TextContent(text="go")]),
        _msg("tool", "core", [
            ToolResultContent(tool_use_id="tu1", content="ok"),
        ]),
    ]
    ctx = _ctx(msgs, todos=[
        _todo("A", "completed"),
        _todo("B", "in_progress"),
        _todo("C", "pending"),
    ])
    hint = todo_continues_after_tool_result_rule(ctx.body, ctx.header, "core")
    assert hint is not None
    assert hint.source == "todo_continues_after_tool_result"
    assert "checkpoint, not a finish line" in hint.text
    # B (in_progress) + C (pending) = 2 unfinished items
    assert "2 unfinished items" in hint.text


def test_todo_continues_fires_when_only_in_progress_remains():
    """Regression: previously synthesis_nudge would fire when all
    remaining items were in_progress (no strict `pending`), forcing the
    LLM to write a completion summary mid-work. Now todo_continues
    stays in charge until nothing is unfinished.
    """
    msgs = [
        _msg("user", "core", [TextContent(text="go")]),
        _msg("tool", "core", [
            ToolResultContent(tool_use_id="tu1", content="ok"),
        ]),
    ]
    ctx = _ctx(msgs, todos=[
        _todo("A", "completed"),
        _todo("B", "in_progress"),
    ])
    # synthesis_nudge must stay silent while any work is in flight
    assert synthesis_nudge_rule(ctx.body, ctx.header, "core") is None
    # todo_continues takes over
    hint = todo_continues_after_tool_result_rule(ctx.body, ctx.header, "core")
    assert hint is not None
    assert "1 unfinished item" in hint.text


def test_todo_continues_silent_when_no_tool_result_tail():
    msgs = [
        _msg("user", "core", [TextContent(text="hi")]),
    ]
    ctx = _ctx(msgs, todos=[_todo("A", "pending")])
    assert todo_continues_after_tool_result_rule(
        ctx.body, ctx.header, "core",
    ) is None


def test_todo_continues_silent_when_only_todowrite_in_tail():
    """Regression: session 409b958e — todo_continues fired after every
    TodoWrite tool_result, the model called TodoWrite again to comply,
    fired the rule again, repeat 79 times. Skip TodoWrite results."""
    msgs = [
        _msg("user", "core", [TextContent(text="add health api")]),
        _msg("core", "user", [
            ToolUseContent(name="TodoWrite", input={"todos": []}, id="tw1"),
        ]),
        _msg("tool", "core", [
            ToolResultContent(tool_use_id="tw1", content="Updated todo list"),
        ]),
    ]
    ctx = _ctx(msgs, todos=[
        _todo("A", "completed"),
        _todo("B", "in_progress"),  # work in flight, normally fires
    ])
    assert todo_continues_after_tool_result_rule(
        ctx.body, ctx.header, "core",
    ) is None


def test_todo_continues_silent_when_no_pending_todos():
    msgs = [
        _msg("tool", "core", [
            ToolResultContent(tool_use_id="tu1", content="ok"),
        ]),
    ]
    ctx = _ctx(msgs, todos=[
        _todo("A", "completed"), _todo("B", "completed"),
    ])
    assert todo_continues_after_tool_result_rule(
        ctx.body, ctx.header, "core",
    ) is None


def test_todo_continues_and_synthesis_are_mutually_exclusive():
    """The whole point of the pending-todos check on synthesis_nudge:
    exactly one of these two rules should fire on a from_=tool tail
    message, never both."""
    msgs = [
        _msg("tool", "core", [
            ToolResultContent(tool_use_id="tu1", content="ok"),
        ]),
    ]
    # With pending todos: only todo_continues (role doesn't matter here —
    # todo_continues applies to any role mid-checklist).
    ctx_pending = _ctx(msgs, todos=[_todo("A", "pending")])
    assert synthesis_nudge_rule(ctx_pending.body, ctx_pending.header, "core") is None
    assert todo_continues_after_tool_result_rule(
        ctx_pending.body, ctx_pending.header, "core",
    ) is not None

    # No pending todos + reporter role: only synthesis_nudge fires.
    # synthesis_nudge is reporter-only now, so we pick researcher;
    # todo_continues stays silent with no pending.
    ctx_done = _ctx(msgs, todos=[_todo("A", "completed")], role_name="researcher")
    msgs_reporter = [
        _msg("core", "researcher", [TextContent(text="go")]),
        _msg("researcher", "core", [
            ToolUseContent(name="Read", input={"file_path": "/a.py"}, id="tu1"),
        ]),
        _msg("tool", "researcher", [
            ToolResultContent(tool_use_id="tu1", content="ok"),
        ]),
    ]
    ctx_done = _ctx(msgs_reporter, todos=[_todo("A", "completed")], role_name="researcher")
    assert synthesis_nudge_rule(ctx_done.body, ctx_done.header, "researcher") is not None
    assert todo_continues_after_tool_result_rule(
        ctx_done.body, ctx_done.header, "researcher",
    ) is None


# ---------------------------------------------------------------------------
# Role-aware synthesis gating + needs_required_tool_rule
# ---------------------------------------------------------------------------


def test_synthesis_nudge_silent_when_implementer_hasnt_edited():
    """Regression (session 5947bf5a429f): implementer ran Glob once,
    synthesis_nudge fired, implementer fabricated code. The role gate
    must silence synthesis_nudge until Edit or Write was actually
    called at least once.
    """
    msgs = [
        _msg("core", "implementer", [TextContent(text="add health check")]),
        _msg("implementer", "core", [
            ToolUseContent(
                name="Glob",
                input={"pattern": "nature/server/**/*.py"},
                id="tu_glob",
            ),
        ]),
        _msg("tool", "implementer", [
            ToolResultContent(
                tool_use_id="tu_glob",
                content="nature/server/app.py\nnature/server/view.py",
            ),
        ]),
    ]
    ctx = _ctx(msgs, role_name="implementer")
    assert synthesis_nudge_rule(ctx.body, ctx.header, "implementer") is None


def test_synthesis_nudge_still_silent_for_implementer_after_edit():
    """Implementer is a worker role — its deliverable is the edit
    itself, not a text summary. Synthesis_nudge after every Edit
    tool_result would tell implementer to `synthesize` prematurely,
    interrupting a legitimate Read→Edit→Read→Edit loop. Stage-1
    eval showed this as the dominant hint source for implementer
    frames in the all-sonnet preset.
    """
    msgs = [
        _msg("core", "implementer", [TextContent(text="add health check")]),
        _msg("implementer", "core", [
            ToolUseContent(
                name="Edit",
                input={"file_path": "/p/app.py", "old_string": "a", "new_string": "b"},
                id="tu_edit",
            ),
        ]),
        _msg("tool", "implementer", [
            ToolResultContent(tool_use_id="tu_edit", content="edited"),
        ]),
    ]
    ctx = _ctx(msgs, role_name="implementer")
    assert synthesis_nudge_rule(ctx.body, ctx.header, "implementer") is None


def test_needs_required_tool_fires_for_implementer_with_only_glob():
    """The complementary rule: when synthesis_nudge is silenced by the
    role gate, needs_required_tool fires instead so the model gets an
    explicit next-step nudge (call Edit/Write, or admit no change)."""
    msgs = [
        _msg("core", "implementer", [TextContent(text="add endpoint")]),
        _msg("implementer", "core", [
            ToolUseContent(
                name="Glob", input={"pattern": "*.py"}, id="tu_glob",
            ),
        ]),
        _msg("tool", "implementer", [
            ToolResultContent(tool_use_id="tu_glob", content="files"),
        ]),
    ]
    ctx = _ctx(msgs, role_name="implementer")
    hint = needs_required_tool_rule(ctx.body, ctx.header, "implementer")
    assert hint is not None
    assert hint.source == "needs_required_tool"
    assert "implementer" in hint.text
    assert "Edit or Write" in hint.text
    # must include the honest-failure escape clause
    assert "no changes are required" in hint.text


def test_needs_required_tool_silent_when_required_tool_used():
    msgs = [
        _msg("core", "implementer", [TextContent(text="edit")]),
        _msg("implementer", "core", [
            ToolUseContent(
                name="Write",
                input={"file_path": "/p/new.py", "content": "x"},
                id="tu_write",
            ),
        ]),
        _msg("tool", "implementer", [
            ToolResultContent(tool_use_id="tu_write", content="written"),
        ]),
    ]
    ctx = _ctx(msgs, role_name="implementer")
    assert needs_required_tool_rule(ctx.body, ctx.header, "implementer") is None


def test_needs_required_tool_silent_for_role_without_requirements():
    """Roles like receptionist / core have no required tools — the rule
    must stay silent for them regardless of trace shape."""
    msgs = [
        _msg("user", "receptionist", [TextContent(text="hi")]),
        _msg("receptionist", "user", [
            ToolUseContent(name="Glob", input={}, id="tu1"),
        ]),
        _msg("tool", "receptionist", [
            ToolResultContent(tool_use_id="tu1", content="x"),
        ]),
    ]
    ctx = _ctx(msgs, role_name="receptionist")
    assert needs_required_tool_rule(
        ctx.body, ctx.header, "receptionist",
    ) is None


def test_role_required_tools_catalog_is_not_empty():
    """Sanity check — the gate only helps if the catalog covers the
    roles we actually ship with."""
    assert "implementer" in ROLE_REQUIRED_TOOLS
    assert "researcher" in ROLE_REQUIRED_TOOLS
    assert "Edit" in ROLE_REQUIRED_TOOLS["implementer"]
    assert "Write" in ROLE_REQUIRED_TOOLS["implementer"]
    assert "Read" in ROLE_REQUIRED_TOOLS["researcher"]


# ---------------------------------------------------------------------------
# compute_footer_hints aggregation + safety
# ---------------------------------------------------------------------------


def test_compute_footer_hints_collects_synthesis_when_no_todos():
    msgs = [
        _msg("core", "researcher", [TextContent(text="go")]),
        _msg("researcher", "core", [
            ToolUseContent(name="Read", input={"file_path": "/a.py"}, id="tu1"),
        ]),
        _msg("tool", "researcher", [
            ToolResultContent(tool_use_id="tu1", content="ok"),
        ]),
    ]
    ctx = _ctx(msgs, role_name="researcher")
    hints = compute_footer_hints(ctx.body, ctx.header, "researcher")
    sources = [h.source for h in hints]
    assert "synthesis_nudge" in sources
    assert "todo_continues_after_tool_result" not in sources
    assert "todo_needs_in_progress" not in sources


def test_compute_footer_hints_collects_todo_continues_with_pending():
    msgs = [
        _msg("user", "core", [TextContent(text="go")]),
        _msg("tool", "core", [
            ToolResultContent(tool_use_id="tu1", content="ok"),
        ]),
    ]
    ctx = _ctx(msgs, todos=[
        _todo("A", "completed"), _todo("B", "pending"),
    ])
    hints = compute_footer_hints(ctx.body, ctx.header, "core")
    sources = [h.source for h in hints]
    assert "synthesis_nudge" not in sources
    assert "todo_continues_after_tool_result" in sources
    # B is pending and nothing is in_progress, so the in_progress rule
    # also fires — both rules can stack
    assert "todo_needs_in_progress" in sources


def test_compute_footer_hints_skips_buggy_rule():
    def boom(body, header, self_actor):
        raise RuntimeError("oops")

    FOOTER_RULES.insert(0, boom)
    try:
        msgs = [
            _msg("core", "researcher", [TextContent(text="hi")]),
            _msg("researcher", "core", [
                ToolUseContent(name="Read", input={"file_path": "/a.py"}, id="tu1"),
            ]),
            _msg("tool", "researcher", [
                ToolResultContent(tool_use_id="tu1", content="ok"),
            ]),
        ]
        ctx = _ctx(msgs, role_name="researcher")
        hints = compute_footer_hints(ctx.body, ctx.header, "researcher")
        assert any(h.source == "synthesis_nudge" for h in hints)
    finally:
        FOOTER_RULES.remove(boom)


# ---------------------------------------------------------------------------
# Composer integration — hints land on ComposedRequest + tail message
# ---------------------------------------------------------------------------


def test_composer_appends_hint_as_tail_user_message():
    msgs = [
        _msg("core", "researcher", [TextContent(text="analyze")]),
        _msg("researcher", "core", [
            ToolUseContent(name="Read", input={"file_path": "/a.py"}, id="tu1"),
        ]),
        _msg("tool", "researcher", [
            ToolResultContent(tool_use_id="tu1", content="file.py"),
        ]),
    ]
    ctx = _ctx(msgs, role_name="researcher")
    composer = ContextComposer()
    composed = composer.compose(
        ctx, self_actor="researcher", tool_registry=[], model="m",
    )
    assert isinstance(composed, ComposedRequest)
    assert len(composed.hints) == 1
    assert composed.hints[0].source == "synthesis_nudge"

    msgs_out = composed.request.messages
    assert len(msgs_out) == 4
    tail = msgs_out[-1]
    assert tail.role == Role.USER
    text = tail.content[0]
    assert isinstance(text, TextContent)
    assert "[FRAMEWORK NOTE" in text.text


def test_composer_no_tail_message_when_no_hint_fires():
    msgs = [
        _msg("user", "core", [TextContent(text="hi")]),
        _msg("core", "user", [TextContent(text="hello")]),
    ]
    ctx = _ctx(msgs)
    composer = ContextComposer()
    composed = composer.compose(
        ctx, self_actor="core", tool_registry=[], model="m",
    )
    assert composed.hints == []
    assert len(composed.request.messages) == 2


def test_composer_request_passthrough_attributes_still_work():
    msgs = [_msg("user", "core", [TextContent(text="hi")])]
    composer = ContextComposer()
    composed = composer.compose(
        _ctx(msgs, role_name="core"),
        self_actor="core", tool_registry=[], model="m",
    )
    assert composed.system == ["be useful"]
    assert composed.request.system == ["be useful"]
    assert composed.messages == composed.request.messages


def test_composer_combines_multiple_hints_into_one_tail_message():
    """When two rules fire (e.g., todo_continues + todo_needs_in_progress),
    they should be merged into a single [FRAMEWORK NOTE] tail message,
    not two separate user messages."""
    msgs = [
        _msg("user", "core", [TextContent(text="go")]),
        _msg("tool", "core", [
            ToolResultContent(tool_use_id="tu1", content="ok"),
        ]),
    ]
    ctx = _ctx(msgs, todos=[_todo("A", "pending"), _todo("B", "pending")])
    composer = ContextComposer()
    composed = composer.compose(
        ctx, self_actor="core", tool_registry=[], model="m",
    )
    assert len(composed.hints) == 2
    sources = [h.source for h in composed.hints]
    assert "todo_continues_after_tool_result" in sources
    assert "todo_needs_in_progress" in sources

    # Tail count: 2 original + 1 combined hint message
    assert len(composed.request.messages) == 3
    tail_text = composed.request.messages[-1].content[0].text
    assert "checkpoint, not a finish line" in tail_text
    assert "checklist is idle" in tail_text
