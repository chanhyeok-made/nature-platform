"""Helper functions shared across footer rules."""

from __future__ import annotations

from nature.context.types import ContextBody
from nature.protocols.message import ToolResultContent, ToolUseContent


def _has_pending_todos(body: ContextBody) -> bool:
    """Frame has ≥1 todo with status exactly 'pending' (not started yet)."""
    return any(t.status == "pending" for t in body.todos)


def _has_in_progress_todo(body: ContextBody) -> bool:
    """Frame has ≥1 in_progress todo."""
    return any(t.status == "in_progress" for t in body.todos)


def _has_incomplete_todos(body: ContextBody) -> bool:
    """Frame has ≥1 todo that isn't completed yet (pending OR in_progress).

    This is the right check for rules that ask "is the checklist still
    mid-flight?". `_has_pending_todos` alone misses the case where the
    only remaining items are `in_progress` — e.g., the LLM marked the
    next step in_progress but hasn't finished it yet. Treating that as
    "no pending work" causes `synthesis_nudge_rule` to fire and coerce
    the LLM into writing a completion summary while work is still
    ongoing, which on weaker models manifests as fabricated answers
    (see session `d952ef8b0757` for the empirical failure case).
    """
    return any(t.status in ("pending", "in_progress") for t in body.todos)


def _frame_used_tools(body: ContextBody) -> set[str]:
    """Return the set of tool names the frame has actually invoked.

    Walks the conversation history and collects every assistant
    `tool_use` block's name. Used by role-aware synthesis gating —
    "did implementer ever call Edit?" is answered by
    `"Edit" in _frame_used_tools(body)`.
    """
    used: set[str] = set()
    for msg in body.conversation.messages:
        for block in msg.content or []:
            if isinstance(block, ToolUseContent):
                used.add(block.name)
    return used


# Role → set of tools the role must use at least once before it is
# allowed to enter synthesis. Rationale: a researcher that only ran
# `Glob` knows filenames but has read no content — synthesizing at
# that point produces fabrication. A "required tool" captures the
# minimum-viable "real work" the role exists to do. Roles not in
# this dict have no requirement (e.g., receptionist, core, judge).
#
# Sessions `0bd80d171a71`, `d952ef8b0757`, `5947bf5a429f` all
# exhibited the same failure shape: specialist ran one metadata tool
# (Glob), synthesis_nudge fired, specialist fabricated content
# without ever reading or writing a real file. This gate prevents
# the nudge from misfiring in that state.
ROLE_REQUIRED_TOOLS: dict[str, frozenset[str]] = {
    "implementer": frozenset({"Edit", "Write"}),
    "researcher":  frozenset({"Read"}),
    "analyzer":    frozenset({"Read"}),
    "reviewer":    frozenset({"Read"}),
}


# Roles whose deliverable is a TEXT REPORT back to a caller. They do
# some research tool calls, then close with a written synthesis.
# synthesis_nudge is designed for exactly this transition; firing it
# on other role classes (delegators like receptionist/core, workers
# like implementer, single-agent like solo) creates hint spam that
# disrupts legitimate multi-turn work. Stage-1 eval run
# `1776665006-8ea9ec` showed 18 synthesis_nudge hints in a 19-turn
# all-haiku receptionist session — almost every tool_result. Gating
# to this set reduces the misfire surface to zero.
REPORTER_ROLES: frozenset[str] = frozenset({
    "researcher", "analyzer", "reviewer", "judge",
})


def _last_message_tool_results(body: ContextBody) -> list[ToolResultContent]:
    """Return the tool_result blocks of the body's most recent message,
    EXCLUDING those whose originating tool_use was a TodoWrite call.

    Why filter TodoWrite specifically? TodoWrite is the agent's own
    bookkeeping — calling it doesn't represent "real work just
    completed", just an updated checklist. Without this filter, the
    todo-oriented footer rules (`todo_continues_after_tool_result`,
    `synthesis_nudge`) fire on the TodoWrite tool_result, which
    nudges the agent to "advance the todo list", which the agent
    does by calling TodoWrite, which produces another tool_result,
    which fires the rule again — a feedback loop with no real work
    in between. Session `409b958e` exhibited exactly this: 79
    consecutive TodoWrite calls in core's frame, all writing the
    same `[completed, completed, completed, in_progress]` state,
    burning ~50 LLM turns and ~$5 of input tokens before the model
    finally broke out by marking the last todo completed.

    Returns [] if the last message isn't a from_="tool" envelope or
    if it contains only TodoWrite results.
    """
    msgs = body.conversation.messages
    if not msgs:
        return []
    last = msgs[-1]
    if last.from_ != "tool":
        return []

    # Build a tool_use_id → tool_name map by walking the entire
    # history. Unknown ids (no matching tool_use in history) are
    # treated as "real work" so synthetic test fixtures and edge
    # cases keep their previous behavior.
    tool_use_names: dict[str, str] = {}
    for m in msgs:
        for block in m.content or []:
            if isinstance(block, ToolUseContent):
                tool_use_names[block.id] = block.name

    return [
        b for b in (last.content or [])
        if isinstance(b, ToolResultContent)
        and tool_use_names.get(b.tool_use_id) != "TodoWrite"
    ]
