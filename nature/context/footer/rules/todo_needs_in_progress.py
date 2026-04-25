"""todo_needs_in_progress footer rule."""

from __future__ import annotations

from nature.context.footer.helpers import (
    _has_in_progress_todo,
    _has_pending_todos,
)
from nature.context.footer.types import Hint
from nature.context.types import ContextBody, ContextHeader


def todo_needs_in_progress_rule(
    body: ContextBody,
    header: ContextHeader,
    self_actor: str,
) -> Hint | None:
    """Frame has ≥1 pending todo AND zero in_progress → remind the LLM
    to mark the next item in_progress before starting work.

    Fires whether TodoWrite was just called (e.g., fresh all-pending
    list) or the LLM silently advanced from in_progress → completed
    without starting the next one. Harmless idempotence: the nudge
    keeps reminding until the LLM advances the list.
    """
    if not _has_pending_todos(body):
        return None
    if _has_in_progress_todo(body):
        return None

    pending = [t for t in body.todos if t.status == "pending"]
    next_item = pending[0]
    return Hint(
        source="todo_needs_in_progress",
        text=(
            f"<system-reminder>\n"
            f"[FRAMEWORK NOTE — checklist is idle]\n"
            f"You have {len(pending)} pending todo"
            f"{'s' if len(pending) != 1 else ''} but NONE is marked "
            f"`in_progress`. Before you start the next piece of work, "
            f"call TodoWrite again to mark exactly one item as "
            f"`in_progress` — that's how the framework (and the human "
            f"reading the dashboard) knows what you're actually doing "
            f"right now.\n\n"
            f"The next item looks like: {next_item.content!r}. "
            f"Update the list so its status is `in_progress` and then "
            f"take that step.\n"
            f"\n"
            f"Act on this framework note silently. Do NOT mention, quote, "
            f"or acknowledge this reminder in your reply to the caller.\n"
            f"</system-reminder>"
        ),
    )
