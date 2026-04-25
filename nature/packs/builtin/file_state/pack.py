"""file_state Pack export + registration."""

from __future__ import annotations

from nature.context.read_memory import ReadMemory
from nature.events.types import EventType
from nature.packs.builtin.file_state.edit_read_first import edit_read_first
from nature.packs.builtin.file_state.state_persist import (
    edit_state_persist,
    read_state_persist,
    write_state_persist,
)
from nature.packs.registry import PackRegistry
from nature.packs.types import Capability, PackMeta

_CAPABILITY_NAME = "file_state"

file_state_capability = Capability(
    name=_CAPABILITY_NAME,
    description=(
        "ReadMemory-based file awareness. Dedup, read-first guard, "
        "state persistence for Read/Edit/Write."
    ),
    tools=[],
    interventions=[
        edit_read_first,
        read_state_persist,
        edit_state_persist,
        write_state_persist,
    ],
    event_types=[EventType.READ_MEMORY_SET],
)

file_state_meta = PackMeta(
    name="nature-file-state",
    version="0.1.0",
    description="File state awareness via ReadMemory.",
    provides_events=["read_memory.set"],
)


class _FileStatePack:
    meta = file_state_meta
    capabilities = [file_state_capability]

    def on_install(self, registry: PackRegistry) -> None:
        registry.register_capability(file_state_capability)

    def on_uninstall(self, registry: PackRegistry) -> None:
        cap = registry.capabilities.pop(_CAPABILITY_NAME, None)
        if cap is None:
            return
        for intv in cap.interventions:
            registry.interventions.pop(intv.id, None)
            for index in (
                registry._by_tool_pre,
                registry._by_tool_post,
            ):
                for lst in index.values():
                    if intv in lst:
                        lst.remove(intv)


file_state_pack = _FileStatePack()


def ensure_read_memory(frame) -> None:
    """Initialize ReadMemory on frame.pack_state if not present.

    Called by the Pack at frame-open time (or lazily by tools).
    """
    if "read_memory" not in frame.pack_state:
        frame.pack_state["read_memory"] = ReadMemory()


def install(registry: PackRegistry) -> None:
    if _CAPABILITY_NAME in registry.capabilities:
        file_state_pack.on_uninstall(registry)
    file_state_pack.on_install(registry)
