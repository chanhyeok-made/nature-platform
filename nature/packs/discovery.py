"""File-based Pack discovery for third-party extensions.

Built-in packs ship in `nature/packs/builtin/` and get registered
through `install_builtin_packs(registry)` at framework startup. This
module is the counterpart for user- and project-layer packs that live
as dropped directories on disk:

    ~/.nature/packs/<pack-name>/
        pack.json     # manifest (see PackManifest below)
        pack.py       # exposes `install(registry: PackRegistry) -> None`
        ...           # (pack internals — imported as a submodule)

    <project>/.nature/packs/<pack-name>/
        ...           # same layout; project layer takes precedence

Discovery is non-fatal: a bad pack logs a warning and is skipped, so
one broken extension cannot take the server down. Errors that are
swallowed:

- Missing / unreadable `pack.json`
- Manifest that fails schema validation
- Missing entry module file
- Python import-time errors in the pack
- `install()` raising at registration time

Callers that want hard-fail semantics (CI pipelines validating a new
pack, say) should load the pack manually with `_load_pack` and handle
the exception themselves.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field

from nature.packs.registry import PackRegistry

logger = logging.getLogger(__name__)


class PackManifest(BaseModel):
    """`pack.json` schema — mirrors `PackMeta` plus an `entry` hint."""

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    description: str = ""
    depends_on: list[str] = Field(default_factory=list)
    provides_events: list[str] = Field(default_factory=list)
    entry: str = "pack"  # importable module name inside the pack dir


# ──────────────────────────────────────────────────────────────────────
# Directory resolution
# ──────────────────────────────────────────────────────────────────────


def _user_dir() -> Path:
    """`~/.nature/packs/`."""
    from nature.config.settings import get_nature_home
    return get_nature_home() / "packs"


def _project_dir(project_dir: Path | str | None) -> Path | None:
    """`<project>/.nature/packs/`. None when no project dir was given."""
    if project_dir is None:
        return None
    return Path(project_dir) / ".nature" / "packs"


def _candidate_dirs(
    project_dir: Path | str | None,
) -> list[Path]:
    """Directories to scan, in ascending priority (user → project).

    Lower-priority entries install first so later layers can override
    a capability of the same name (PackRegistry collapses duplicate
    capabilities on re-registration).
    """
    dirs: list[Path] = [_user_dir()]
    proj = _project_dir(project_dir)
    if proj is not None:
        dirs.append(proj)
    return dirs


# ──────────────────────────────────────────────────────────────────────
# Loader
# ──────────────────────────────────────────────────────────────────────


def _load_pack(pack_dir: Path) -> tuple[PackManifest, object]:
    """Parse the manifest and import the entry module for one pack.

    Returns `(manifest, module)`. The module is expected to expose an
    `install(registry: PackRegistry) -> None` callable; callers verify
    that shape separately. Raises ValueError / OSError / ImportError
    on any failure so callers can distinguish causes during hard-fail
    validation flows.
    """
    manifest_path = pack_dir / "pack.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = PackManifest.model_validate(raw)

    entry_path = pack_dir / f"{manifest.entry}.py"
    if not entry_path.exists():
        raise FileNotFoundError(
            f"pack {pack_dir.name!r} manifest references entry "
            f"{manifest.entry!r} but {entry_path} does not exist"
        )

    # Fully-qualified module name includes the pack dir so imports of
    # neighbouring files inside the pack work without sys.path edits.
    module_name = f"nature._user_packs.{pack_dir.name}.{manifest.entry}"
    spec = importlib.util.spec_from_file_location(
        module_name, entry_path,
        submodule_search_locations=[str(pack_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not create module spec for {entry_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return manifest, module


def discover_packs(
    project_dir: Path | str | None = None,
) -> Iterable[tuple[Path, PackManifest]]:
    """Yield every valid pack directory under the user ∪ project
    layers, paired with its parsed manifest. Packs with unparseable
    manifests are logged and skipped.
    """
    for base in _candidate_dirs(project_dir):
        if not base.exists():
            continue
        for pack_dir in sorted(base.iterdir()):
            if not pack_dir.is_dir():
                continue
            manifest_path = pack_dir / "pack.json"
            if not manifest_path.exists():
                continue
            try:
                raw = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest = PackManifest.model_validate(raw)
            except Exception as exc:
                logger.warning(
                    "discovery: skipping %s — malformed manifest: %s",
                    pack_dir, exc,
                )
                continue
            yield pack_dir, manifest


def install_discovered_packs(
    registry: PackRegistry,
    project_dir: Path | str | None = None,
) -> list[str]:
    """Register every discoverable pack into `registry`.

    Errors (import failures, `install()` exceptions) are logged and
    swallowed so one broken pack cannot block framework startup. The
    return value lists the names of packs that installed successfully,
    which tests and introspection tooling can assert against.
    """
    installed: list[str] = []
    for pack_dir, manifest in discover_packs(project_dir):
        try:
            _, module = _load_pack(pack_dir)
        except Exception as exc:
            logger.warning(
                "discovery: skipping %s — import error: %s",
                pack_dir, exc,
            )
            continue
        install_fn = getattr(module, "install", None)
        if not callable(install_fn):
            logger.warning(
                "discovery: skipping %s — entry module does not expose "
                "`install(registry)` callable",
                pack_dir,
            )
            continue
        try:
            install_fn(registry)
        except Exception as exc:
            logger.warning(
                "discovery: %s install() raised: %s",
                pack_dir, exc,
            )
            continue
        installed.append(manifest.name)
        logger.info("discovery: installed pack %r from %s", manifest.name, pack_dir)
    return installed


__all__ = [
    "PackManifest",
    "discover_packs",
    "install_discovered_packs",
]
