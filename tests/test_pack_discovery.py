"""Tests for file-based Pack discovery (nature.packs.discovery).

Covers the happy path (valid pack → registered) and the defensive
skip-and-log paths (malformed manifest, missing entry, bad install).
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from nature.packs.discovery import (
    PackManifest,
    discover_packs,
    install_discovered_packs,
)
from nature.packs.registry import PackRegistry


# ──────────────────────────────────────────────────────────────────────
# Fixture builders
# ──────────────────────────────────────────────────────────────────────


def _write_pack(
    packs_dir: Path,
    name: str,
    *,
    manifest_body: dict | None = None,
    entry_source: str | None = None,
    entry_filename: str = "pack.py",
) -> Path:
    """Create a `<packs_dir>/<name>/` pack with a manifest + entry.

    Omitting `manifest_body` writes a minimal default manifest;
    omitting `entry_source` writes a no-op install. The defaults give
    a fully valid pack — overriders go in for negative tests.
    """
    pack_dir = packs_dir / name
    pack_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_body if manifest_body is not None else {
        "name": name,
        "version": "0.1.0",
        "description": f"test pack {name}",
    }
    (pack_dir / "pack.json").write_text(
        json.dumps(manifest), encoding="utf-8",
    )
    source = entry_source if entry_source is not None else textwrap.dedent(
        """
        from nature.packs.registry import PackRegistry
        _installed_names = []

        def install(registry: PackRegistry) -> None:
            _installed_names.append(registry)
        """
    )
    (pack_dir / entry_filename).write_text(source, encoding="utf-8")
    return pack_dir


# ──────────────────────────────────────────────────────────────────────
# Manifest parsing
# ──────────────────────────────────────────────────────────────────────


def test_manifest_rejects_extra_fields():
    with pytest.raises(Exception):
        PackManifest.model_validate({
            "name": "x", "version": "0.1.0",
            "bogus_field": 123,
        })


def test_manifest_defaults_entry_to_pack():
    m = PackManifest.model_validate({"name": "x", "version": "0.1.0"})
    assert m.entry == "pack"


# ──────────────────────────────────────────────────────────────────────
# Discovery (scan only, no installation)
# ──────────────────────────────────────────────────────────────────────


def test_discover_empty_when_no_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("NATURE_HOME", str(tmp_path / "no-home"))
    assert list(discover_packs(project_dir=tmp_path / "no-project")) == []


def test_discover_from_user_dir(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("NATURE_HOME", str(home))
    _write_pack(home / "packs", "alpha")

    entries = list(discover_packs(project_dir=None))
    assert len(entries) == 1
    pack_dir, manifest = entries[0]
    assert manifest.name == "alpha"
    assert pack_dir == home / "packs" / "alpha"


def test_discover_project_overrides_user(tmp_path, monkeypatch):
    """Both layers are scanned; project entries come after user so
    install_discovered_packs can collapse a same-named capability."""
    home = tmp_path / "home"
    monkeypatch.setenv("NATURE_HOME", str(home))
    _write_pack(home / "packs", "alpha", manifest_body={
        "name": "alpha", "version": "0.1.0", "description": "user layer",
    })
    project = tmp_path / "proj"
    _write_pack(project / ".nature" / "packs", "alpha", manifest_body={
        "name": "alpha", "version": "0.2.0", "description": "project layer",
    })

    names = [(d, m.description) for d, m in discover_packs(project_dir=project)]
    # Both entries surface; consumer decides precedence.
    descriptions = {desc for _, desc in names}
    assert descriptions == {"user layer", "project layer"}


def test_discover_skips_malformed_manifest(tmp_path, monkeypatch, caplog):
    home = tmp_path / "home"
    monkeypatch.setenv("NATURE_HOME", str(home))
    pack_dir = home / "packs" / "broken"
    pack_dir.mkdir(parents=True)
    (pack_dir / "pack.json").write_text("not: valid: json", encoding="utf-8")
    (pack_dir / "pack.py").write_text("", encoding="utf-8")

    with caplog.at_level("WARNING", logger="nature.packs.discovery"):
        entries = list(discover_packs(project_dir=None))
    assert entries == []
    assert any(
        "broken" in rec.message and "malformed" in rec.message
        for rec in caplog.records
    )


def test_discover_skips_dir_without_manifest(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("NATURE_HOME", str(home))
    stray = home / "packs" / "not-a-pack"
    stray.mkdir(parents=True)
    (stray / "readme.txt").write_text("hi", encoding="utf-8")

    assert list(discover_packs(project_dir=None)) == []


# ──────────────────────────────────────────────────────────────────────
# install_discovered_packs — full flow
# ──────────────────────────────────────────────────────────────────────


_INSTALL_RECORDS_FILE = "installs.log"


def _recorder_source(tag: str, out_path: str) -> str:
    """Entry-module source that appends `tag` to `out_path` on install."""
    return textwrap.dedent(f"""
        from pathlib import Path

        def install(registry):
            Path({out_path!r}).open('a').write('{tag}\\n')
    """)


def test_install_discovered_runs_install_for_each_pack(tmp_path, monkeypatch):
    log = tmp_path / _INSTALL_RECORDS_FILE
    home = tmp_path / "home"
    monkeypatch.setenv("NATURE_HOME", str(home))

    _write_pack(
        home / "packs", "alpha",
        entry_source=_recorder_source("alpha", str(log)),
    )
    _write_pack(
        home / "packs", "beta",
        entry_source=_recorder_source("beta", str(log)),
    )

    reg = PackRegistry()
    installed = install_discovered_packs(reg, project_dir=None)

    assert sorted(installed) == ["alpha", "beta"]
    assert sorted(log.read_text().splitlines()) == ["alpha", "beta"]


def test_install_skips_missing_entry(tmp_path, monkeypatch, caplog):
    home = tmp_path / "home"
    monkeypatch.setenv("NATURE_HOME", str(home))
    pack_dir = home / "packs" / "no-entry"
    pack_dir.mkdir(parents=True)
    (pack_dir / "pack.json").write_text(json.dumps({
        "name": "no-entry", "version": "0.1.0",
    }), encoding="utf-8")
    # no pack.py written

    reg = PackRegistry()
    with caplog.at_level("WARNING", logger="nature.packs.discovery"):
        installed = install_discovered_packs(reg, project_dir=None)
    assert installed == []
    assert any("no-entry" in rec.message for rec in caplog.records)


def test_install_skips_pack_with_raising_install(tmp_path, monkeypatch, caplog):
    home = tmp_path / "home"
    monkeypatch.setenv("NATURE_HOME", str(home))
    _write_pack(
        home / "packs", "explodes",
        entry_source=(
            "def install(registry):\n"
            "    raise RuntimeError('kaboom')\n"
        ),
    )

    reg = PackRegistry()
    with caplog.at_level("WARNING", logger="nature.packs.discovery"):
        installed = install_discovered_packs(reg, project_dir=None)
    assert installed == []
    assert any(
        "explodes" in rec.message and "kaboom" in rec.message
        for rec in caplog.records
    )


def test_install_skips_pack_without_install_callable(tmp_path, monkeypatch, caplog):
    home = tmp_path / "home"
    monkeypatch.setenv("NATURE_HOME", str(home))
    _write_pack(
        home / "packs", "no-install",
        entry_source="# intentionally empty\n",
    )

    reg = PackRegistry()
    with caplog.at_level("WARNING", logger="nature.packs.discovery"):
        installed = install_discovered_packs(reg, project_dir=None)
    assert installed == []
    assert any(
        "no-install" in rec.message and "install" in rec.message
        for rec in caplog.records
    )


def test_install_skips_pack_with_import_error(tmp_path, monkeypatch, caplog):
    home = tmp_path / "home"
    monkeypatch.setenv("NATURE_HOME", str(home))
    _write_pack(
        home / "packs", "bad-import",
        entry_source="import nature_does_not_have_this_module  # noqa\n",
    )

    reg = PackRegistry()
    with caplog.at_level("WARNING", logger="nature.packs.discovery"):
        installed = install_discovered_packs(reg, project_dir=None)
    assert installed == []
    assert any("bad-import" in rec.message for rec in caplog.records)


def test_install_one_broken_pack_does_not_block_others(tmp_path, monkeypatch):
    """A borked pack should not stop a healthy neighbour from installing."""
    log = tmp_path / _INSTALL_RECORDS_FILE
    home = tmp_path / "home"
    monkeypatch.setenv("NATURE_HOME", str(home))

    _write_pack(
        home / "packs", "alpha_ok",
        entry_source=_recorder_source("alpha_ok", str(log)),
    )
    _write_pack(
        home / "packs", "beta_broken",
        entry_source=(
            "def install(registry):\n"
            "    raise RuntimeError('nope')\n"
        ),
    )

    reg = PackRegistry()
    installed = install_discovered_packs(reg, project_dir=None)
    assert installed == ["alpha_ok"]
    assert log.read_text().strip() == "alpha_ok"


def test_install_entry_can_reach_neighbouring_modules(tmp_path, monkeypatch):
    """Pack entries should be able to `import` sibling files via the
    submodule_search_locations the loader sets up."""
    home = tmp_path / "home"
    monkeypatch.setenv("NATURE_HOME", str(home))
    pack_dir = home / "packs" / "multi-module"
    pack_dir.mkdir(parents=True)
    (pack_dir / "pack.json").write_text(json.dumps({
        "name": "multi-module", "version": "0.1.0",
    }), encoding="utf-8")
    (pack_dir / "pack.py").write_text(textwrap.dedent("""
        from .helper import LABEL

        def install(registry):
            registry._installed_label = LABEL
    """), encoding="utf-8")
    (pack_dir / "helper.py").write_text(
        "LABEL = 'neighbour-imported'\n", encoding="utf-8",
    )

    reg = PackRegistry()
    installed = install_discovered_packs(reg, project_dir=None)
    assert installed == ["multi-module"]
    assert getattr(reg, "_installed_label", None) == "neighbour-imported"
