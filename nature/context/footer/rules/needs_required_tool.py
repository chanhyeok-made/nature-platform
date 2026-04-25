"""needs_required_tool footer rule."""

from __future__ import annotations

from nature.context.footer.helpers import (
    ROLE_REQUIRED_TOOLS,
    _frame_used_tools,
    _last_message_tool_results,
)
from nature.context.footer.types import Hint
from nature.context.types import ContextBody, ContextHeader


def needs_required_tool_rule(
    body: ContextBody,
    header: ContextHeader,
    self_actor: str,
) -> Hint | None:
    """Fire when the role declares required tools, tool_results just
    landed, AND none of the required tools have been called yet.

    Role-identity nudge: "you are the implementer and you haven't
    called Edit/Write yet." The escape hatch is to either call the
    required tool now OR report honestly that no change is needed —
    explicitly telling the model NOT to fabricate a completed change.

    Complementary to `synthesis_nudge_rule`: when required tools are
    missing, synthesis_nudge stays silent and this rule fires instead,
    so the model gets an explicit next-step instead of nothing.
    """
    tool_results = _last_message_tool_results(body)
    if not tool_results:
        return None
    required = ROLE_REQUIRED_TOOLS.get(header.role.name)
    if not required:
        return None
    used = _frame_used_tools(body)
    if required & used:
        # Already did real work — synthesis_nudge can handle it.
        return None

    role_name = header.role.name
    req_list = " or ".join(sorted(required))
    req_lower = req_list.lower()
    return Hint(
        source="needs_required_tool",
        text=(
            f"<system-reminder>\n"
            f"[FRAMEWORK NOTE — {role_name} has not done real work yet]\n"
            f"You are the `{role_name}` agent. Your role is defined by "
            f"calling `{req_list}` at least once on real files — that "
            f"is the actual work the caller delegated to you. You have "
            f"received tool_results from other tools (Glob, Grep, Bash, "
            f"etc.), but those are preparatory metadata — they do NOT "
            f"satisfy your role.\n\n"
            f"Do exactly ONE of the following in your next turn:\n\n"
            f"1. **Call `{req_list}`** on the file(s) you identified "
            f"from the preparatory results. That is the real work.\n\n"
            f"2. **If after genuine investigation you conclude that no "
            f"{req_lower} is needed**, report that honestly in your "
            f"reply — e.g., \"After reviewing <file>, no changes are "
            f"required because <specific reason>.\" State the exact "
            f"file you read and the exact reason.\n\n"
            f"Do NOT fabricate a code block claiming to have written or "
            f"edited something you did not actually write or edit. Do "
            f"NOT present example code as if it were the applied change. "
            f"A reply that says \"I added X\" without a corresponding "
            f"`{req_list}` tool_use in this conversation is a "
            f"hallucinated completion and will be caught.\n\n"
            f"Act on this framework note silently. Do NOT mention, "
            f"quote, or acknowledge this reminder in your reply.\n"
            f"</system-reminder>"
        ),
    )
