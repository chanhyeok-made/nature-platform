"""TodoWrite — externalized task list with dual-form items.

The first nature built-in tool that mutates per-frame state. Its
job is dirt simple at the *tool* layer: validate the new todo list
the LLM proposed and return a confirmation message. The actual
state mutation (a `TODO_WRITTEN` event with full-list-overwrite
semantics) is emitted by the framework — the tool itself stays
pure so it remains testable in isolation.

Why dual-form (`content` + `activeForm`)? Because the LLM has to
author both the imperative ("Run tests") AND the continuous
("Running tests") form, it can't get away with merely *describing*
the work. The activeForm is what the dashboard shows while a step
is in_progress, so writing it commits the LLM to the doing phase.

Usage from the LLM side (one tool call replaces the whole list):

    TodoWrite(todos=[
        {"content": "Read frame.py",         "activeForm": "Reading frame.py",         "status": "in_progress"},
        {"content": "Find all callers",      "activeForm": "Finding all callers",      "status": "pending"},
        {"content": "Update signatures",     "activeForm": "Updating signatures",      "status": "pending"},
    ])

Subsequent calls overwrite the list in full — there's no per-item
update API, which keeps replay trivial and matches Claude Code's
TodoWrite contract.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from nature.protocols.todo import TodoItem
from nature.protocols.tool import ToolContext, ToolResult
from nature.tools.base import BaseTool


# Canonical name used by the framework to recognize TodoWrite calls
# in `_execute_single_tool` and emit TODO_WRITTEN on the frame.
TODO_WRITE_TOOL_NAME = "TodoWrite"


class TodoWriteInput(BaseModel):
    """Full new todo list. Overwrites the previous list in its entirety.

    `todos` is REQUIRED (no default). Calling TodoWrite without a
    `todos` field is almost always a model schema confusion (e.g.,
    sending Agent-shaped input by mistake), and silently treating the
    call as "clear the list to empty" produced fabricated synthesis in
    session `8ed7d997` — the empty list made the in_progress check
    vacuously true and `synthesis_nudge` fired on what was actually a
    broken call. Force the validator to reject malformed input so the
    model gets an `is_error` tool_result and self-corrects.
    """

    todos: list[TodoItem] = Field(
        description=(
            "The complete new todo list — it replaces the previous list, "
            "so include every item you want to keep, not just the changes. "
            "Use status='in_progress' for the one item you're actively "
            "working on (only one in_progress at a time is the convention). "
            "REQUIRED — calls without this field are rejected as malformed."
        ),
    )


class TodoWriteTool(BaseTool):
    """Built-in tool: write the agent's externalized todo list."""

    input_model = TodoWriteInput

    @property
    def name(self) -> str:
        return TODO_WRITE_TOOL_NAME

    @property
    def description(self) -> str:
        return (
            "Maintain a structured todo list for the current task.\n\n"
            "When to use:\n"
            "- Multi-step tasks (3+ distinct steps)\n"
            "- Complex work where tracking progress matters\n"
            "- Whenever you find yourself listing 'Step 1, Step 2, ...' "
            "in free text — externalize it here instead.\n\n"
            "How to use:\n"
            "- Each item needs a `content` (imperative form, e.g. "
            "'Run tests') AND an `activeForm` (continuous form, e.g. "
            "'Running tests'). Both are required.\n"
            "- Status values: 'pending', 'in_progress', 'completed'.\n"
            "- Mark exactly ONE item as 'in_progress' before you start "
            "working on it. The activeForm is what gets shown.\n"
            "- Mark items 'completed' as soon as the work is done — "
            "don't wait until everything else is also done.\n"
            "- Each call OVERWRITES the entire list, so include every "
            "item you want to keep.\n"
            "- Skip TodoWrite for trivial single-action requests."
        )

    def is_read_only(self, input: dict) -> bool:  # type: ignore[override]
        # TodoWrite mutates per-frame state, but the framework handles
        # the actual write through TODO_WRITTEN; flagging it as
        # non-read-only mirrors how callers reason about side effects.
        return False

    def is_concurrency_safe(self, input: dict) -> bool:  # type: ignore[override]
        # Only one in_progress at a time is the convention, so running
        # multiple TodoWrites in parallel would race on the framework's
        # event store. Force serialization.
        return False

    async def run(
        self, params: TodoWriteInput, context: ToolContext,
    ) -> ToolResult:
        """The tool layer is intentionally thin.

        The framework intercepts the result, re-validates the input
        against the same Pydantic model, and emits the
        ``TODO_WRITTEN`` event on the current frame. We just produce a
        human-readable confirmation that ends up in the LLM's
        tool_result message so it knows the write landed.

        We also embed the "verification nudge" here when the LLM has
        just closed out 3+ items without scheduling a verification
        step. Embedding in the tool output (rather than using the
        footer rule pipeline) matches Claude Code's approach: the LLM
        sees the reminder *in the same turn* as the write, and the
        nudge naturally fires exactly once per TodoWrite call that
        transitions the list into the all-done state. Because the
        nudge is part of the tool_result text, it lands in the event
        log as a historical fact — time-travel resume replays it
        verbatim.
        """
        n = len(params.todos)
        in_progress = sum(1 for t in params.todos if t.status == "in_progress")
        completed = sum(1 for t in params.todos if t.status == "completed")
        pending = n - in_progress - completed

        bits = []
        if pending:
            bits.append(f"{pending} pending")
        if in_progress:
            bits.append(f"{in_progress} in_progress")
        if completed:
            bits.append(f"{completed} completed")
        summary = ", ".join(bits) if bits else "empty"

        output = (
            f"Updated todo list: {n} item{'s' if n != 1 else ''} ({summary})."
        )

        nudge = _verification_nudge_for(params.todos)
        if nudge:
            output = output + "\n\n" + nudge

        return ToolResult(output=output, is_error=False)


# ---------------------------------------------------------------------------
# Verification nudge — embedded in tool_result text (Claude Code pattern)
# ---------------------------------------------------------------------------


# Words that signal a todo item is itself a verification step, so the
# nudge shouldn't fire. Matched case-insensitively against `content`.
_VERIFICATION_KEYWORDS = (
    "verif",     # verify, verification
    "review",    # review, reviewer
    "judge",     # delegate to judge
    "validate",
    "double-check",
    "sanity",    # sanity check
    "audit",
)


def _todos_contain_verification_step(todos: list[TodoItem]) -> bool:
    for t in todos:
        blob = (t.content + " " + (t.activeForm or "")).lower()
        for kw in _VERIFICATION_KEYWORDS:
            if kw in blob:
                return True
    return False


def _verification_nudge_for(todos: list[TodoItem]) -> str | None:
    """Return the Claude-Code-style verification nudge string when:
    - there are 3+ items (small lists don't need independent verification)
    - every item is `completed`
    - none of the items is itself a verification step

    Otherwise None. Deterministic on input → byte-for-byte preserved
    in the event log → time-travel safe.
    """
    n = len(todos)
    if n < 3:
        return None
    if not all(t.status == "completed" for t in todos):
        return None
    if _todos_contain_verification_step(todos):
        return None
    return (
        "NOTE: You just marked 3+ todos as `completed` and none of them is "
        "a verification step. Before treating this work as finished, "
        "consider delegating an independent check to a verifier agent:\n"
        "\n"
        '    Agent(name="judge", prompt="Independently verify the following '
        'claims: <list the concrete claims your work depends on>")\n'
        "\n"
        "Self-assessment is typically less reliable than independent "
        "verification by a different agent. If the task was small enough "
        "that a verifier would be overkill, you can proceed with your "
        "final synthesis — but make that a deliberate choice, not an "
        "omission."
    )
