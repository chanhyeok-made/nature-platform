"""loop_block — pre-Edit gate that refuses calls past the retry threshold.

Runs as a Gate on `OnTool(Edit, PRE)`. For the incoming call:

1. Hash `(file_path, old_string)` → current_hash.
2. Walk the conversation body backward using `loop_detector`'s helper,
   counting consecutive same-hash failed Edit attempts already in history.
3. If `prior_streak + 1 >= THRESHOLD`, return `Block` with a reason
   that tells the LLM to re-read the file and pick a fresh target.

Why body-walking instead of a cached flag:

- Pure function of conversation state. No new frame fields, no
  state-transition events, no mirroring between event store and live
  frame. Whatever reconstruct rebuilds is automatically the same as
  what the live path sees.
- Detector and blocker agree by construction — both read the same
  body with the same helper. There's no "stale block flag" case.
- Self-clears on intervention: if the model changes the old_string
  (fresh hash) or the user sends a new input that shifts direction,
  the streak count resets naturally on the next call.
"""

from __future__ import annotations

from nature.events.types import EventType
from nature.packs.builtin.edit_guards.loop_detector import (
    THRESHOLD,
    count_recent_same_hash_edit_failures,
    hash_edit_input,
)
from nature.packs.types import (
    Block,
    Intervention,
    InterventionContext,
    OnTool,
    ToolPhase,
)

_BLOCK_REASON = (
    "[edit_guards.loop_block] This is the "
    f"{THRESHOLD}th consecutive Edit attempt with the same old_string "
    "and the prior attempts all failed. Do NOT retry this edit. "
    "Call Read on the target file first to see the current content, "
    "then pick a fresh old_string that actually matches."
)


def _loop_block_action(ctx: InterventionContext):
    tc = ctx.tool_call
    if tc is None:
        return []
    frame = ctx.frame
    if frame is None:
        return []

    inp = tc.tool_input or {}
    file_path = inp.get("file_path", "")
    old_string = inp.get("old_string", "")
    if not file_path or not old_string:
        return []

    current_hash = hash_edit_input(file_path, old_string)
    prior_streak = count_recent_same_hash_edit_failures(
        frame.context.body.conversation,
        target_hash=current_hash,
    )
    if prior_streak + 1 >= THRESHOLD:
        return [Block(reason=_BLOCK_REASON, trace_event=EventType.LOOP_BLOCKED)]
    return []


loop_block = Intervention(
    id="edit_guards.loop_block",
    kind="gate",
    trigger=OnTool(tool_name="Edit", phase=ToolPhase.PRE),
    action=_loop_block_action,
    description="Refuse Edit calls whose input has already failed THRESHOLD-1 times consecutively.",
)
