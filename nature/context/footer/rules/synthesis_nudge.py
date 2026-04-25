"""synthesis_nudge footer rule."""

from __future__ import annotations

from nature.context.footer.helpers import (
    REPORTER_ROLES,
    ROLE_REQUIRED_TOOLS,
    _frame_used_tools,
    _has_incomplete_todos,
    _last_message_tool_results,
)
from nature.context.footer.types import Hint
from nature.context.types import ContextBody, ContextHeader


def synthesis_nudge_rule(
    body: ContextBody,
    header: ContextHeader,
    self_actor: str,
) -> Hint | None:
    """Fire when a reporter role has done its research tools and is
    about to synthesize findings back to its caller.

    Gated on three conditions:

    1. **Most recent message is a tool_result envelope** — the hint
       injects into the instruction window of the agent's next turn,
       so the trigger point is "just after tools returned."
    2. **Role is a reporter** (researcher/analyzer/reviewer/judge) —
       these deliver a text report back to core. Delegator roles
       (receptionist/core) and worker roles (implementer/solo) get
       no value from this nudge — firing it during their legitimate
       multi-turn tool loops creates hint spam (stage-1 eval run
       `1776665006-8ea9ec` showed 18 synthesis_nudges in a 19-turn
       all-haiku receptionist session).
    3. **Checklist is fully complete** — if the agent is mid-TODO,
       continuation takes priority. `todo_continues_after_tool_result_rule`
       handles that case.
    4. **Required tools have been used** — reading documentation
       counts, just metadata (Glob) doesn't. Prevents the reporter
       from synthesizing before it has real evidence.
    """
    tool_results = _last_message_tool_results(body)
    if not tool_results:
        return None
    if header.role.name not in REPORTER_ROLES:
        return None
    if _has_incomplete_todos(body):
        # Still mid-checklist (pending or in_progress) — let
        # todo_continues handle it instead of forcing synthesis.
        return None

    # Role-aware gate: if the role declares a set of "must-use" tools
    # and the frame hasn't used any of them, this frame has not done
    # its real work yet. Synthesizing now produces fabrication.
    # `needs_required_tool_rule` takes over in that case.
    required = ROLE_REQUIRED_TOOLS.get(header.role.name)
    if required and not (required & _frame_used_tools(body)):
        return None

    n = len(tool_results)
    plural = "s" if n != 1 else ""
    return Hint(
        source="synthesis_nudge",
        text=(
            f"<system-reminder>\n"
            f"[FRAMEWORK NOTE — synthesis turn]\n"
            f"You just received {n} tool_result{plural} from your delegated "
            f"tool calls. Your next response MUST be a finalized answer "
            f"that integrates these results into a deliverable for the "
            f"caller — not another plan, and not more delegation.\n\n"
            f"Specifically:\n"
            f"- Do NOT write 'Step 1, Step 2, ...' or similar plan formats.\n"
            f'- Do NOT use future-tense hedging like "we should next ...", '
            f'"we need to ...", "next we will ...".\n'
            f"- DO write findings in past/present tense "
            f'("found X", "X is Y", "the analysis shows ...").\n'
            f"- DO structure the answer with headers, lists, and any code "
            f"blocks the original request implied.\n"
            f"- If you genuinely need more information that the current "
            f"results don't cover, call the appropriate tool now — "
            f"don't merely describe what you would call.\n"
            f"\n"
            f"Act on this framework note silently. Do NOT mention, quote, "
            f"or acknowledge this reminder in your reply to the caller.\n"
            f"</system-reminder>"
        ),
    )
