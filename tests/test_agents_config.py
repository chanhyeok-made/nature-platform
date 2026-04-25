"""Tests for the agent registry (nature.agents.config)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nature.agents.config import (
    AgentConfig,
    AgentsRegistry,
    load_agents_registry,
)


# ──────────────────────────────────────────────────────────────────────
# AgentConfig validation
# ──────────────────────────────────────────────────────────────────────


def test_model_requires_host_prefix():
    with pytest.raises(Exception, match="host::model"):
        AgentConfig(model="claude-haiku-4-5", instructions="x.md")


def test_model_rejects_empty_host():
    with pytest.raises(Exception, match="malformed"):
        AgentConfig(model="::claude-haiku-4-5", instructions="x.md")


def test_model_rejects_empty_model():
    with pytest.raises(Exception, match="malformed"):
        AgentConfig(model="anthropic::", instructions="x.md")


def test_accepts_valid_host_qualified_model():
    cfg = AgentConfig(
        model="anthropic::claude-haiku-4-5",
        instructions="x.md",
    )
    assert cfg.model == "anthropic::claude-haiku-4-5"


def test_rejects_unknown_fields():
    # extra="forbid" catches typos like `allowedTools` or stale field names
    with pytest.raises(Exception):
        AgentConfig.model_validate({
            "model": "anthropic::claude-haiku-4-5",
            "instructions": "x.md",
            "allowedTools": ["Read"],  # typo — should be allowed_tools
        })


# ──────────────────────────────────────────────────────────────────────
# Loader — user fixture dir
# ──────────────────────────────────────────────────────────────────────


def _write_agent(dir_path: Path, name: str, json_body: dict, instruction_body: str) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / f"{name}.json").write_text(json.dumps(json_body, indent=2))
    (dir_path / "instructions").mkdir(exist_ok=True)
    (dir_path / "instructions" / f"{name}.md").write_text(instruction_body)


def test_loader_requires_paired_md(tmp_path, monkeypatch):
    # JSON without matching instruction file → hard error.
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "orphan.json").write_text(json.dumps({
        "model": "anthropic::claude-haiku-4-5",
        "instructions": "missing.md",
    }))

    from nature.agents.config import _load_from_dir
    with pytest.raises(ValueError, match="does not exist"):
        _load_from_dir(tmp_path / "agents")


def test_loader_attaches_filename_stem_as_name(tmp_path):
    _write_agent(
        tmp_path, "custom",
        {"model": "anthropic::claude-haiku-4-5", "instructions": "custom.md"},
        "you are custom",
    )
    from nature.agents.config import _load_from_dir
    loaded = _load_from_dir(tmp_path)
    assert loaded["custom"].name == "custom"
    assert loaded["custom"].instructions_text == "you are custom"


def test_loader_missing_dir_returns_empty(tmp_path):
    from nature.agents.config import _load_from_dir
    assert _load_from_dir(tmp_path / "does-not-exist") == {}


# ──────────────────────────────────────────────────────────────────────
# Builtins ship and load
# ──────────────────────────────────────────────────────────────────────


BUILTIN_NAMES = {
    "receptionist", "core", "researcher",
    "analyzer", "implementer", "reviewer", "judge",
}


def test_all_builtins_present(monkeypatch, tmp_path):
    # Redirect user dir so we only see builtins
    monkeypatch.setenv("NATURE_HOME", str(tmp_path))
    reg = load_agents_registry()
    assert BUILTIN_NAMES.issubset(set(reg.names()))


def test_builtins_have_host_qualified_models(monkeypatch, tmp_path):
    monkeypatch.setenv("NATURE_HOME", str(tmp_path))
    reg = load_agents_registry()
    for name in BUILTIN_NAMES:
        cfg = reg.get(name)
        assert cfg is not None
        assert "::" in cfg.model, f"{name}: {cfg.model}"


def test_builtins_have_nonempty_instructions(monkeypatch, tmp_path):
    monkeypatch.setenv("NATURE_HOME", str(tmp_path))
    reg = load_agents_registry()
    for name in BUILTIN_NAMES:
        cfg = reg.get(name)
        assert cfg is not None
        assert len(cfg.instructions_text) > 100, f"{name} instructions too short"


# ──────────────────────────────────────────────────────────────────────
# User overrides builtin
# ──────────────────────────────────────────────────────────────────────


def test_user_fully_replaces_builtin(monkeypatch, tmp_path):
    # Build a fake user dir that overrides `receptionist`
    home = tmp_path / "home-nature"
    home.mkdir()
    monkeypatch.setenv("NATURE_HOME", str(home))

    _write_agent(
        home / "agents", "receptionist",
        {
            "model": "groq::llama-3.3-70b-versatile",
            "allowed_tools": [],
            "instructions": "receptionist.md",
        },
        "overridden",
    )

    reg = load_agents_registry()
    r = reg.get("receptionist")
    assert r is not None
    assert r.model == "groq::llama-3.3-70b-versatile"
    assert r.allowed_tools == []
    assert r.instructions_text == "overridden"
    # Other builtins still present
    assert reg.get("core") is not None


def test_user_can_add_new_agent(monkeypatch, tmp_path):
    home = tmp_path / "home-nature"
    home.mkdir()
    monkeypatch.setenv("NATURE_HOME", str(home))

    _write_agent(
        home / "agents", "scribe",
        {"model": "anthropic::claude-haiku-4-5", "instructions": "scribe.md"},
        "scribe instructions",
    )

    reg = load_agents_registry()
    assert "scribe" in reg
    scribe = reg.get("scribe")
    assert scribe.instructions_text == "scribe instructions"


def test_project_dir_overrides_user(monkeypatch, tmp_path):
    """Project-level agent replaces user-level entry of the same name."""
    home = tmp_path / "home-nature"
    home.mkdir()
    monkeypatch.setenv("NATURE_HOME", str(home))

    _write_agent(
        home / "agents", "shared",
        {"model": "anthropic::claude-haiku-4-5", "instructions": "shared.md"},
        "user version",
    )

    project = tmp_path / "proj"
    _write_agent(
        project / ".nature" / "agents", "shared",
        {"model": "local-ollama::qwen2.5-coder:32b", "instructions": "shared.md"},
        "project version",
    )

    reg = load_agents_registry(project_dir=str(project))
    shared = reg.get("shared")
    assert shared.model == "local-ollama::qwen2.5-coder:32b"
    assert shared.instructions_text == "project version"


def test_project_dir_adds_new_agent(monkeypatch, tmp_path):
    home = tmp_path / "home-nature"
    home.mkdir()
    monkeypatch.setenv("NATURE_HOME", str(home))

    project = tmp_path / "proj"
    _write_agent(
        project / ".nature" / "agents", "project-only",
        {"model": "anthropic::claude-haiku-4-5", "instructions": "project-only.md"},
        "project-only agent",
    )

    reg = load_agents_registry(project_dir=str(project))
    assert "project-only" in reg
