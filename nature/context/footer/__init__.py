"""Context footer package — ephemeral framework hints injected at compose time.

Public API re-exported from submodules for backwards compatibility.

M1: footer dispatch runs through the Pack registry. `FOOTER_RULES`
remains exported as a **data declaration** (so `nature.packs.legacy_shim`
can read it on install), but `compute_footer_hints` no longer walks
this list — it asks the registry for whatever Contributor Interventions
are subscribed to `OnTurn(BEFORE_LLM)`. Legacy rules become Contributor
Interventions at registry install time; there is no parallel path.
"""

from __future__ import annotations

import time

from nature.context.footer.helpers import (
    ROLE_REQUIRED_TOOLS,
    _frame_used_tools,
    _has_incomplete_todos,
    _has_in_progress_todo,
    _has_pending_todos,
    _last_message_tool_results,
)
from nature.context.footer.rules.needs_required_tool import needs_required_tool_rule
from nature.context.footer.rules.synthesis_nudge import synthesis_nudge_rule
from nature.context.footer.rules.todo_continues_after_tool_result import (
    todo_continues_after_tool_result_rule,
)
from nature.context.footer.rules.todo_needs_in_progress import todo_needs_in_progress_rule
from nature.context.footer.types import Hint, Rule

# Data declaration consumed by `nature.packs.legacy_shim.install_legacy_rules`
# to build Contributor Interventions. Order determines dispatch order of
# the ported interventions.
FOOTER_RULES: list[Rule] = [
    synthesis_nudge_rule,
    todo_needs_in_progress_rule,
    todo_continues_after_tool_result_rule,
    needs_required_tool_rule,
]


def compute_footer_hints(
    body,
    header,
    self_actor: str,
) -> list[Hint]:
    """Ask the Pack registry for every Contributor-produced footer hint.

    Pure function from the caller's perspective: same inputs → same
    output, no side effects. Internally dispatches through
    `default_registry.dispatch_turn_sync(TurnPhase.BEFORE_LLM, ctx)`
    and filters for `AppendFooter` effects. Legacy rules land here via
    `install_legacy_rules` on first use.
    """
    # Local imports avoid a module-load-time edge with nature.packs
    # (legacy_shim imports from this module to read FOOTER_RULES).
    from nature.packs.legacy_shim import install_legacy_rules, legacy_rules_installed
    from nature.packs.registry import default_registry
    from nature.packs.types import AppendFooter, InterventionContext, TurnPhase

    if not legacy_rules_installed(default_registry):
        install_legacy_rules(default_registry)

    ctx = InterventionContext(
        session_id="",
        now=time.time(),
        registry=default_registry,
        body=body,
        header=header,
        self_actor=self_actor,
    )
    effects = default_registry.dispatch_turn_sync(TurnPhase.BEFORE_LLM, ctx)
    return [
        Hint(source=eff.source_id, text=eff.text)
        for eff in effects
        if isinstance(eff, AppendFooter)
    ]


__all__ = [
    "Hint",
    "Rule",
    "FOOTER_RULES",
    "ROLE_REQUIRED_TOOLS",
    "compute_footer_hints",
    "synthesis_nudge_rule",
    "todo_needs_in_progress_rule",
    "todo_continues_after_tool_result_rule",
    "needs_required_tool_rule",
    "_has_pending_todos",
    "_has_in_progress_todo",
    "_has_incomplete_todos",
    "_frame_used_tools",
    "_last_message_tool_results",
]
