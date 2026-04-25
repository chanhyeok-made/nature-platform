"""reread_hint — "re-read before retrying" footer nudge after Edit failure.

A Contributor Intervention that runs on OnTurn(BEFORE_LLM) and folds
a hint into the footer when the body's most recent message is a failed
Edit tool_result.

Why a Contributor, not a Listener with InjectUserMessage:

- The body already carries the signal (last message = Edit error). No
  new frame state is required to know whether to fire.
- Contributors are pure functions of body/header. Same body → same
  hint → same LLM input → deterministic replay.
- One-shot semantics fall out naturally: as soon as the LLM responds
  (adding another message to the body), "last message is Edit error"
  becomes false, and the hint stops firing on the following turn.
"""

from __future__ import annotations

from nature.context.types import ContextBody, ContextHeader
from nature.packs.types import (
    AppendFooter,
    Intervention,
    InterventionContext,
    OnTurn,
    TurnPhase,
)

_HINT_TEXT = (
    "<system-reminder>\n"
    "[edit_guards.reread_hint] Your last Edit failed with "
    "`old_string not found`. Before retrying:\n"
    "1. Re-read the target file with the Read tool to see the current content.\n"
    "2. Copy the old_string text verbatim from the fresh read.\n"
    "3. Do NOT retry with a near-duplicate old_string — that's how hallucination loops start.\n"
    "\n"
    "Act on this silently. Do NOT mention or quote this reminder to the caller.\n"
    "</system-reminder>"
)


def _last_edit_error(body: ContextBody | None) -> bool:
    """True iff the most recent message in the body is a tool_result
    whose first block is an error from an Edit call."""
    if body is None:
        return False
    msgs = body.conversation.messages
    if not msgs:
        return False
    last = msgs[-1]
    if last.from_ != "tool":
        return False
    # Scan the content for a tool_result block that looks like an Edit miss.
    for block in last.content:
        block_type = getattr(block, "type", None)
        if block_type != "tool_result":
            continue
        is_error = getattr(block, "is_error", False)
        if not is_error:
            continue
        # The tool name isn't on the result block; we look at the
        # output text for Edit's sentinel phrase. Not perfect — if
        # Edit's error message changes the heuristic breaks — but
        # edit_guards ships alongside the Edit tool, so we own both.
        content = getattr(block, "content", "")
        if not isinstance(content, str):
            content = str(content)
        if "old_string" in content and "not found" in content:
            return True
    return False


def _reread_hint_action(ctx: InterventionContext):
    if not _last_edit_error(ctx.body):
        return []
    return [AppendFooter(text=_HINT_TEXT, source_id="edit_guards.reread_hint")]


reread_hint = Intervention(
    id="edit_guards.reread_hint",
    kind="contributor",
    trigger=OnTurn(phase=TurnPhase.BEFORE_LLM),
    action=_reread_hint_action,
    description="Fold a 're-read the file' hint into the footer after an Edit error.",
)
