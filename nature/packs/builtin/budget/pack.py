"""budget Pack export + registration helper."""

from __future__ import annotations

from nature.events.types import EventType
from nature.packs.builtin.budget.reads_budget import reads_gate, reads_warning
from nature.packs.registry import PackRegistry
from nature.packs.types import Capability, PackMeta

_CAPABILITY_NAME = "reads_budget"

reads_budget_capability = Capability(
    name=_CAPABILITY_NAME,
    description="Per-frame Read/Grep/Glob call budget with 80% warning and 100% block.",
    tools=[],
    interventions=[reads_gate, reads_warning],
    event_types=[EventType.BUDGET_BLOCKED, EventType.BUDGET_WARNING],
)

budget_meta = PackMeta(
    name="nature-budget",
    version="0.1.0",
    description="Phase 3 per-frame budget caps.",
    depends_on=[],
    provides_events=["budget.blocked", "budget.warning"],
)


class _BudgetPack:
    meta = budget_meta
    capabilities = [reads_budget_capability]

    def on_install(self, registry: PackRegistry) -> None:
        registry.register_capability(reads_budget_capability)

    def on_uninstall(self, registry: PackRegistry) -> None:
        cap = registry.capabilities.pop(_CAPABILITY_NAME, None)
        if cap is None:
            return
        for intv in cap.interventions:
            registry.interventions.pop(intv.id, None)
            for index in (
                registry._by_tool_pre,
                registry._by_tool_post,
                registry._by_turn,
            ):
                for lst in index.values():
                    if intv in lst:
                        lst.remove(intv)


budget_pack = _BudgetPack()


def install(registry: PackRegistry) -> None:
    if _CAPABILITY_NAME in registry.capabilities:
        budget_pack.on_uninstall(registry)
    budget_pack.on_install(registry)
