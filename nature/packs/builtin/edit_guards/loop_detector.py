"""loop_detector — POST_EFFECT listener that observes consecutive Edit misses.

Runs after `fuzzy_suggest` on the same trigger (`OnTool(Edit, POST,
error)`) but in the POST_EFFECT phase, so it receives fuzzy_suggest's
effect list via `ctx.primary_effects`. Only fires when fuzzy_suggest
actually emitted EDIT_MISS — a structural guard against false
positives from unrelated Edit errors.

**State source**: the frame's conversation body. nature already syncs
the body as messages land, so `frame.context.body.conversation` is
live for the current turn. Walking backward and pairing each
`assistant(tool_use Edit)` with the following `tool(tool_result
is_error=True)` lets us count consecutive same-hash attempts without
introducing any new frame state or state-transition events.

**Output**: only a trace event (`LOOP_DETECTED`) when the threshold is
reached. No gating, no state mutation. `loop_block` (Gate) is the
component that actually refuses the next Edit call — it reads the
same body and arrives at the same count, so detector + blocker stay
in sync by construction.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from nature.events.payloads import LoopDetectedPayload
from nature.events.types import EventType
from nature.packs.types import (
    EmitEvent,
    Intervention,
    InterventionContext,
    InterventionPhase,
    OnTool,
    ToolPhase,
)

if TYPE_CHECKING:
    from nature.context.conversation import Conversation

THRESHOLD = 3


def hash_edit_input(file_path: str, old_string: str) -> str:
    """Stable short hash of an Edit call's identity-bearing inputs."""
    key = f"{file_path}\x00{old_string[:80]}"
    return hashlib.sha1(key.encode("utf-8", errors="replace")).hexdigest()[:12]


def count_recent_same_hash_edit_failures(
    conversation: "Conversation",
    target_hash: str | None = None,
) -> int:
    """Walk the conversation backward, count consecutive same-hash failed
    Edit attempts ending at the most recent tool_result.

    Pairing rule: an "attempt" is an assistant message with a tool_use
    block for Edit, immediately followed by a tool message whose
    tool_result for that tool_use_id has is_error=True. Consecutive
    attempts with the same (file_path, old_string) hash increment the
    count; the first non-matching pair ends the streak.

    If `target_hash` is None, the most recent failed Edit's hash is
    used (i.e., "how long has THIS error been repeating"). If a hash
    is given, the walker stops counting as soon as an attempt with a
    different hash is seen.
    """
    msgs = list(conversation.messages)
    if not msgs:
        return 0

    # Collect (tool_use_id, hash, is_error) for each Edit attempt in
    # chronological order. An assistant message can carry multiple
    # tool_use blocks; we pair each with its matching tool_result
    # from the immediately-following tool message.
    attempts: list[tuple[str, bool]] = []  # (hash, is_error)
    i = 0
    while i < len(msgs):
        msg = msgs[i]
        if msg.from_ == "tool":
            i += 1
            continue
        # Collect tool_use blocks from this assistant message
        uses: list[tuple[str, str]] = []  # (tool_use_id, hash)
        for block in msg.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            if getattr(block, "name", None) != "Edit":
                continue
            tu_id = getattr(block, "id", "")
            tu_input = getattr(block, "input", {}) or {}
            file_path = tu_input.get("file_path", "")
            old_string = tu_input.get("old_string", "")
            if not file_path or not old_string:
                continue
            uses.append((tu_id, hash_edit_input(file_path, old_string)))
        if not uses:
            i += 1
            continue
        # Look at the immediately-following message for matching tool_results
        if i + 1 < len(msgs) and msgs[i + 1].from_ == "tool":
            results_by_id: dict[str, bool] = {}
            for block in msgs[i + 1].content:
                if getattr(block, "type", None) != "tool_result":
                    continue
                tool_use_id = getattr(block, "tool_use_id", "")
                is_error = bool(getattr(block, "is_error", False))
                results_by_id[tool_use_id] = is_error
            for tu_id, h in uses:
                if tu_id in results_by_id:
                    attempts.append((h, results_by_id[tu_id]))
            i += 2
        else:
            i += 1

    if not attempts:
        return 0

    # Walk backward counting consecutive failed same-hash attempts.
    streak = 0
    anchor_hash = target_hash
    for h, is_error in reversed(attempts):
        if not is_error:
            break
        if anchor_hash is None:
            anchor_hash = h
        if h != anchor_hash:
            break
        streak += 1
    return streak


def _find_edit_miss_effect(primary_effects):
    for eff in primary_effects:
        if isinstance(eff, EmitEvent) and eff.event_type == EventType.EDIT_MISS:
            return eff
    return None


def _loop_detector_action(ctx: InterventionContext):
    tc = ctx.tool_call
    if tc is None or tc.result_is_error is not True:
        return []
    if _find_edit_miss_effect(ctx.primary_effects) is None:
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

    # The current attempt hasn't been appended to the body yet — the
    # manager's _execute_single_tool is still mid-dispatch, emitting
    # events. Count whatever streak the body shows and add 1 for the
    # in-flight attempt.
    prior_streak = count_recent_same_hash_edit_failures(
        frame.context.body.conversation,
        target_hash=current_hash,
    )
    effective_streak = prior_streak + 1

    if effective_streak < THRESHOLD:
        return []

    return [
        EmitEvent(
            event_type=EventType.LOOP_DETECTED,
            payload=LoopDetectedPayload(
                tool="Edit",
                input_hash=current_hash,
                attempts=effective_streak,
            ),
        ),
    ]


loop_detector = Intervention(
    id="edit_guards.loop_detector",
    kind="listener",
    trigger=OnTool(
        tool_name="Edit",
        phase=ToolPhase.POST,
        where=lambda tc: tc.result_is_error is True,
    ),
    phase=InterventionPhase.POST_EFFECT,
    action=_loop_detector_action,
    description="Observe consecutive same-hash Edit misses, emit LOOP_DETECTED at threshold.",
)
