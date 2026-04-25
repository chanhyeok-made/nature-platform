"""reads_budget — cap on Read/Grep/Glob calls per frame.

Two interventions:

- `reads_budget.gate` (Gate, OnTool PRE): counts prior read-family
  calls in the body. Blocks at 100% of the limit.
- `reads_budget.warning` (Contributor, OnTurn BEFORE_LLM): same count
  derivation, fires a footer hint at ≥80% of the limit.

Both pure functions of the conversation body — no separate state.

Edit and Write are intentionally uncapped: the goal is to prevent
analysis paralysis, not limit real work.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nature.events.types import EventType
from nature.packs.types import (
    AppendFooter,
    Block,
    Intervention,
    InterventionContext,
    OnTool,
    OnTurn,
    ToolPhase,
    TurnPhase,
)

if TYPE_CHECKING:
    from nature.context.conversation import Conversation

READ_FAMILY = frozenset({"Read", "Grep", "Glob"})

DEFAULT_LIMIT = 20
WARN_RATIO = 0.8


def count_read_family_calls(conversation: "Conversation") -> int:
    """Count completed Read/Grep/Glob tool calls in the conversation.

    Walks the body looking for assistant messages with tool_use blocks
    whose name is in READ_FAMILY, paired with a successful tool_result.
    Only counts completed pairs — a pending (unpaired) tool_use at the
    tail of the body is not counted.
    """
    count = 0
    msgs = list(conversation.messages)
    i = 0
    while i < len(msgs):
        msg = msgs[i]
        # Only look at assistant messages with tool_use blocks
        if msg.from_ == "tool":
            i += 1
            continue
        uses = []
        for block in msg.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            name = getattr(block, "name", "")
            if name in READ_FAMILY:
                uses.append(getattr(block, "id", ""))
        if not uses:
            i += 1
            continue
        # Pair with the immediately-following tool message
        if i + 1 < len(msgs) and msgs[i + 1].from_ == "tool":
            results_by_id = set()
            for block in msgs[i + 1].content:
                if getattr(block, "type", None) == "tool_result":
                    tu_id = getattr(block, "tool_use_id", "")
                    if tu_id:
                        results_by_id.add(tu_id)
            count += sum(1 for uid in uses if uid in results_by_id)
            i += 2
        else:
            i += 1
    return count


# ── Gate ──────────────────────────────────────────────────────────────


def _reads_gate_action(ctx: InterventionContext):
    tc = ctx.tool_call
    if tc is None:
        return []
    if tc.tool_name not in READ_FAMILY:
        return []
    frame = ctx.frame
    if frame is None:
        return []

    prior = count_read_family_calls(frame.context.body.conversation)
    if prior + 1 > DEFAULT_LIMIT:
        return [
            Block(
                reason=(
                    f"[reads_budget] Read-family call limit reached "
                    f"({prior}/{DEFAULT_LIMIT}). Your analysis budget is "
                    f"exhausted. Next steps:\n"
                    f"1. Proceed to Edit/Write — you have enough context.\n"
                    f"2. Or delegate to a sub-agent (Agent tool) for a "
                    f"fresh budget.\n"
                    f"3. Or raise the cap via the dashboard config editor."
                ),
                trace_event=EventType.BUDGET_BLOCKED,
            )
        ]
    return []


reads_gate = Intervention(
    id="reads_budget.gate",
    kind="gate",
    trigger=OnTool(tool_name=None, phase=ToolPhase.PRE),
    action=_reads_gate_action,
    description=f"Block Read/Grep/Glob calls past {DEFAULT_LIMIT} per frame.",
)


# ── Contributor (80% warning) ────────────────────────────────────────


_WARN_TEXT_TEMPLATE = (
    "<system-reminder>\n"
    "[reads_budget] You have used {used}/{limit} of your read-family "
    "tool budget (Read/Grep/Glob). You're running low.\n"
    "Consider proceeding to Edit/Write with the context you already "
    "have, or delegate to a sub-agent for a fresh budget.\n"
    "Act on this silently. Do NOT mention this reminder.\n"
    "</system-reminder>"
)


def _reads_warning_action(ctx: InterventionContext):
    body = ctx.body
    if body is None:
        return []
    used = count_read_family_calls(body.conversation)
    if used < int(DEFAULT_LIMIT * WARN_RATIO):
        return []
    if used >= DEFAULT_LIMIT:
        # At or past limit — gate handles it, don't duplicate with a hint.
        return []
    return [
        AppendFooter(
            text=_WARN_TEXT_TEMPLATE.format(used=used, limit=DEFAULT_LIMIT),
            source_id="reads_budget.warning",
        )
    ]


reads_warning = Intervention(
    id="reads_budget.warning",
    kind="contributor",
    trigger=OnTurn(phase=TurnPhase.BEFORE_LLM),
    action=_reads_warning_action,
    description="Footer hint when read-family calls reach 80% of per-frame limit.",
)
