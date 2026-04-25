"""Legacy shim — port existing footer rules into the Pack model.

The footer rules declared in `nature.context.footer.FOOTER_RULES` are
pure `(body, header, self_actor) -> Hint | None` functions. This module
wraps each one as a Contributor Intervention with `trigger=OnTurn(
BEFORE_LLM)` and registers them into the given `PackRegistry`.

The port is faithful: same signature, same exception swallowing, same
output. The only difference is that dispatch now flows through the
registry instead of a hard-coded list walk in `compute_footer_hints`.

This is the "layer unification" step described in pack_architecture.md
§11.1 — legacy rules are not wrapped in a parallel path, they become
first-class Contributor Interventions and legacy footer rendering reads
from the registry like any other contributor source.
"""

from __future__ import annotations

import logging
from typing import Callable

from nature.context.footer.types import Hint, Rule
from nature.packs.registry import PackRegistry
from nature.packs.types import (
    AppendFooter,
    Capability,
    Effect,
    Intervention,
    InterventionContext,
    OnTurn,
    TurnPhase,
)

logger = logging.getLogger(__name__)


_LEGACY_CAPABILITY_NAME = "nature.legacy_footer_rules"


def _wrap_rule_as_action(
    rule: Rule,
    source_id: str,
) -> Callable[[InterventionContext], list[Effect]]:
    """Build a Contributor action that invokes the legacy rule function.

    The returned action reads `body` / `header` / `self_actor` from the
    InterventionContext (populated by `compute_footer_hints`), calls the
    rule, and converts its `Hint | None` return into an `AppendFooter`
    Effect. Exceptions are swallowed to match the legacy behavior in
    `compute_footer_hints`.
    """

    def action(ctx: InterventionContext) -> list[Effect]:
        if ctx.body is None or ctx.header is None:
            return []
        try:
            hint = rule(ctx.body, ctx.header, ctx.self_actor)
        except Exception:
            logger.exception(
                "Legacy footer rule %r raised in action — skipping",
                source_id,
            )
            return []
        if hint is None:
            return []
        return [AppendFooter(text=hint.text, source_id=hint.source or source_id)]

    action.__name__ = f"legacy_rule_action__{source_id}"
    return action


def install_legacy_rules(registry: PackRegistry) -> None:
    """Port every rule in `FOOTER_RULES` into `registry` as a Contributor.

    Safe to call multiple times: we first clear any previously installed
    legacy capability, then re-register. This lets tests swap registries
    without duplicating rules.
    """
    # Local import avoids a circular edge with nature.context.footer.
    from nature.context.footer import FOOTER_RULES

    # Remove prior registration if present — tests expect idempotency.
    existing = registry.capabilities.get(_LEGACY_CAPABILITY_NAME)
    if existing is not None:
        for intv in existing.interventions:
            registry.interventions.pop(intv.id, None)
        for phase_list in registry._by_turn.values():
            phase_list[:] = [
                i for i in phase_list
                if not i.id.startswith("legacy.")
            ]
        registry.capabilities.pop(_LEGACY_CAPABILITY_NAME, None)

    interventions: list[Intervention] = []
    for rule in FOOTER_RULES:
        source_id = f"legacy.{getattr(rule, '__name__', 'rule')}"
        intv = Intervention(
            id=source_id,
            kind="contributor",
            trigger=OnTurn(phase=TurnPhase.BEFORE_LLM),
            action=_wrap_rule_as_action(rule, source_id),
            description=f"Legacy footer rule ported: {rule.__name__}",
        )
        interventions.append(intv)

    cap = Capability(
        name=_LEGACY_CAPABILITY_NAME,
        description="Wraps nature.context.footer.FOOTER_RULES as Contributor Interventions",
        tools=[],
        interventions=interventions,
        event_types=[],
    )
    registry.register_capability(cap)


def legacy_rules_installed(registry: PackRegistry) -> bool:
    return _LEGACY_CAPABILITY_NAME in registry.capabilities


__all__ = [
    "install_legacy_rules",
    "legacy_rules_installed",
]
