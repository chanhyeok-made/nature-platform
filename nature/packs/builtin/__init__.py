"""Built-in Packs shipped with nature.

Built-in packs skip the `pack.json` manifest entirely — they're Python
modules that expose a `Pack` object directly, registered at framework
startup via `install_builtin_packs(registry)`. Third-party packs that
live under `~/.nature/packs/` will still use the manifest + entry_point
loading story when it lands.
"""

from __future__ import annotations

from nature.packs.registry import PackRegistry


def install_builtin_packs(registry: PackRegistry) -> None:
    """Register every built-in Pack into the given registry.

    Called from `AreaManager.__init__` so any fresh manager starts up
    with the baseline guards in place. Tests that want a clean
    registry can pass their own empty `PackRegistry()` — this function
    is a no-op on manager construction if the caller's registry
    already has the capabilities (see each Pack's own `install` which
    is idempotent).
    """
    # Local imports so that tests and tooling can import this module
    # without eagerly loading every intervention's dependencies.
    from nature.packs.builtin.edit_guards.pack import install as install_edit_guards
    from nature.packs.builtin.budget.pack import install as install_budget
    from nature.packs.builtin.file_state.pack import install as install_file_state

    install_edit_guards(registry)
    install_budget(registry)
    install_file_state(registry)


__all__ = ["install_builtin_packs"]
