"""edit_guards Pack export + registration helper.

Ships as a built-in Pack (no `pack.json` manifest), so installation is
just a Python call: `install(registry)`. The framework bootstraps all
built-in packs into `default_registry` at AreaManager construction time
via `nature.packs.builtin.install_builtin_packs`.
"""

from __future__ import annotations

from nature.events.types import EventType
from nature.packs.builtin.edit_guards.fuzzy_suggest import fuzzy_suggest
from nature.packs.builtin.edit_guards.loop_block import loop_block
from nature.packs.builtin.edit_guards.loop_detector import loop_detector
from nature.packs.builtin.edit_guards.reread_hint import reread_hint
from nature.packs.registry import PackRegistry
from nature.packs.types import Capability, PackMeta

_CAPABILITY_NAME = "edit_guards"


edit_guards_capability = Capability(
    name=_CAPABILITY_NAME,
    description=(
        "Phase 2 Edit feedback — fuzzy-match suggestions on miss, "
        "re-read reminder, consecutive-miss detection, and post-threshold gating."
    ),
    tools=[],  # augments the existing Edit tool, doesn't add a new one
    interventions=[
        fuzzy_suggest,
        reread_hint,
        loop_detector,
        loop_block,
    ],
    event_types=[
        EventType.EDIT_MISS,
        EventType.LOOP_DETECTED,
        EventType.LOOP_BLOCKED,
    ],
)


edit_guards_meta = PackMeta(
    name="nature-edit-guards",
    version="0.1.0",
    description="Phase 2 Edit feedback Pack — fuzzy + loop detection + re-read hint.",
    depends_on=[],
    provides_events=[
        "edit.miss",
        "loop.detected",
        "loop.blocked",
    ],
)


# Lightweight concrete Pack — the design protocol allows any object with
# the right shape, but a plain dataclass-ish namespace is enough here.
class _EditGuardsPack:
    meta: PackMeta = edit_guards_meta
    capabilities: list[Capability] = [edit_guards_capability]

    def on_install(self, registry: PackRegistry) -> None:
        registry.register_capability(edit_guards_capability)

    def on_uninstall(self, registry: PackRegistry) -> None:
        # Symmetric removal: drop interventions by id and purge the
        # capability from the registry's index. Intentionally minimal —
        # real uninstall with active sessions still referencing the
        # capability will land when third-party packs ship.
        cap = registry.capabilities.pop(_CAPABILITY_NAME, None)
        if cap is None:
            return
        for intv in cap.interventions:
            registry.interventions.pop(intv.id, None)
            for index in (
                registry._by_tool_pre,
                registry._by_tool_post,
                registry._by_event,
                registry._by_turn,
                registry._by_frame,
            ):
                for lst in index.values():
                    if intv in lst:
                        lst.remove(intv)


edit_guards_pack = _EditGuardsPack()


def install(registry: PackRegistry) -> None:
    """Register edit_guards into the given registry. Idempotent-ish:
    re-registering overwrites without duplicating indexes because
    `PackRegistry.register_capability` clears prior entries under the
    same capability name first."""
    if _CAPABILITY_NAME in registry.capabilities:
        edit_guards_pack.on_uninstall(registry)
    edit_guards_pack.on_install(registry)
