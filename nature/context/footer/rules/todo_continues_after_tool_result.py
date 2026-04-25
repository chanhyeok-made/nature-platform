"""todo_continues_after_tool_result footer rule."""

from __future__ import annotations

from nature.context.footer.helpers import (
    _has_incomplete_todos,
    _last_message_tool_results,
)
from nature.context.footer.types import Hint
from nature.context.types import ContextBody, ContextHeader


def todo_continues_after_tool_result_rule(
    body: ContextBody,
    header: ContextHeader,
    self_actor: str,
) -> Hint | None:
    """Right after receiving tool_results AND the checklist still has
    unfinished work (pending OR in_progress) → tell the LLM to advance
    the list rather than draft a premature final answer.

    This composes with synthesis_nudge_rule: when any incomplete todos
    exist, synthesis_nudge stays silent and this rule fires instead.
    The distinction matters because "synthesize now" and "continue the
    checklist" pull in opposite directions. We count pending AND
    in_progress as "still mid-flight" — an item marked in_progress
    means work has *started* but not finished, so wrap-up is premature.
    """
    tool_results = _last_message_tool_results(body)
    if not tool_results:
        return None
    if not _has_incomplete_todos(body):
        return None

    total = len(body.todos)
    done = sum(1 for t in body.todos if t.status == "completed")
    unfinished = sum(
        1 for t in body.todos if t.status in ("pending", "in_progress")
    )
    n = len(tool_results)
    plural = "s" if n != 1 else ""
    return Hint(
        source="todo_continues_after_tool_result",
        text=(
            f"<system-reminder>\n"
            f"[FRAMEWORK NOTE — mid-checklist, do NOT wrap up]\n"
            f"You just received {n} tool_result{plural}, and your todo "
            f"list still has {unfinished} unfinished item"
            f"{'s' if unfinished != 1 else ''} ({done}/{total} completed "
            f"so far). This is a checkpoint, not a finish line.\n\n"
            f"Next move:\n"
            f"1. Call TodoWrite to update the list — mark the item you "
            f"just finished as `completed` and mark the next pending "
            f"item as `in_progress`.\n"
            f"2. Then actually execute that next item (call the tool / "
            f"delegate / write the code).\n\n"
            f"Do NOT write a final synthesis or summary while pending "
            f"todos remain. The user's request is only partially "
            f"answered at this point.\n"
            f"\n"
            f"Act on this framework note silently. Do NOT mention, quote, "
            f"or acknowledge this reminder in your reply to the caller.\n"
            f"</system-reminder>"
        ),
    )
