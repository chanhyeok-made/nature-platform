"""Tests for the new preset registry (nature.agents.presets).

Covers schema validation (PresetConfig), the loader (project/user
layering), and validate_preset (the session-creation completeness
check that surfaces dangling references).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nature.agents.config import AgentConfig, AgentsRegistry
from nature.agents.presets import (
    PresetConfig,
    PresetValidationError,
    list_presets,
    load_preset,
    validate_preset,
)
from nature.config.hosts import HostConfig, HostsConfig


# ──────────────────────────────────────────────────────────────────────
# PresetConfig schema
# ──────────────────────────────────────────────────────────────────────


def _minimal_preset(**overrides) -> PresetConfig:
    kwargs = {
        "root_agent": "receptionist",
        "agents": ["receptionist", "core"],
    }
    kwargs.update(overrides)
    return PresetConfig(**kwargs)


def test_preset_requires_agents_nonempty():
    with pytest.raises(Exception, match="non-empty"):
        PresetConfig(root_agent="x", agents=[])


def test_preset_root_must_be_in_agents():
    with pytest.raises(Exception, match="root_agent"):
        PresetConfig(root_agent="missing", agents=["core", "researcher"])


def test_preset_overrides_must_be_listed_agents():
    with pytest.raises(Exception, match="not in agents"):
        PresetConfig(
            root_agent="core",
            agents=["core"],
            model_overrides={"core": "anthropic::c", "other": "anthropic::o"},
        )


def test_preset_overrides_must_be_host_qualified():
    with pytest.raises(Exception, match="host::model"):
        PresetConfig(
            root_agent="core",
            agents=["core"],
            model_overrides={"core": "bare-model-no-host"},
        )


def test_preset_accepts_valid_minimal():
    p = _minimal_preset()
    assert p.root_agent == "receptionist"
    assert p.agents == ["receptionist", "core"]
    assert p.model_overrides == {}


def test_preset_rejects_unknown_fields():
    with pytest.raises(Exception):
        PresetConfig.model_validate({
            "root_agent": "x", "agents": ["x"],
            "extraKey": "oops",
        })


def test_preset_accepts_max_output_tokens_overrides():
    p = _minimal_preset(max_output_tokens_overrides={"core": 4096})
    assert p.max_output_tokens_overrides == {"core": 4096}


def test_preset_rejects_tokens_override_for_unlisted_agent():
    with pytest.raises(Exception, match="max_output_tokens_overrides"):
        PresetConfig(
            root_agent="core",
            agents=["core"],
            max_output_tokens_overrides={"researcher": 2048},
        )


def test_preset_rejects_nonpositive_token_budget():
    with pytest.raises(Exception, match="must be positive"):
        PresetConfig(
            root_agent="core",
            agents=["core"],
            max_output_tokens_overrides={"core": 0},
        )
    with pytest.raises(Exception, match="must be positive"):
        PresetConfig(
            root_agent="core",
            agents=["core"],
            max_output_tokens_overrides={"core": -100},
        )


# ──────────────────────────────────────────────────────────────────────
# Loader — layered (project > user)
# ──────────────────────────────────────────────────────────────────────


def _write_preset(dir_path: Path, name: str, body: dict) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / f"{name}.json").write_text(json.dumps(body))


def test_load_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("NATURE_HOME", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="nonexistent"):
        load_preset("nonexistent", project_dir=str(tmp_path))


def test_load_from_user_dir(tmp_path, monkeypatch):
    home = tmp_path / "home-nature"
    home.mkdir()
    monkeypatch.setenv("NATURE_HOME", str(home))

    _write_preset(home / "presets", "p1", {
        "root_agent": "a", "agents": ["a"],
    })
    p = load_preset("p1", project_dir=str(tmp_path))
    assert p.name == "p1"
    assert p.root_agent == "a"


def test_project_overrides_user(tmp_path, monkeypatch):
    home = tmp_path / "home-nature"
    home.mkdir()
    monkeypatch.setenv("NATURE_HOME", str(home))

    # User version
    _write_preset(home / "presets", "p1", {
        "root_agent": "user-root", "agents": ["user-root"],
    })

    # Project version
    project = tmp_path / "proj"
    _write_preset(project / ".nature" / "presets", "p1", {
        "root_agent": "proj-root", "agents": ["proj-root"],
    })

    p = load_preset("p1", project_dir=str(project))
    assert p.root_agent == "proj-root"


def test_list_presets_merges_dirs(tmp_path, monkeypatch):
    home = tmp_path / "home-nature"
    home.mkdir()
    monkeypatch.setenv("NATURE_HOME", str(home))

    _write_preset(home / "presets", "user-only", {
        "root_agent": "x", "agents": ["x"],
    })
    project = tmp_path / "proj"
    _write_preset(project / ".nature" / "presets", "proj-only", {
        "root_agent": "y", "agents": ["y"],
    })
    _write_preset(project / ".nature" / "presets", "user-only", {
        "root_agent": "z", "agents": ["z"],
    })

    names = list_presets(project_dir=str(project))
    assert "user-only" in names
    assert "proj-only" in names
    # Union, no duplicates
    assert names.count("user-only") == 1


# ──────────────────────────────────────────────────────────────────────
# validate_preset
# ──────────────────────────────────────────────────────────────────────


def _build_agents(names: list[str], model: str = "anthropic::claude-haiku-4-5") -> AgentsRegistry:
    return AgentsRegistry(agents={
        n: AgentConfig(
            model=model,
            instructions=f"{n}.md",
            name=n,
            instructions_text="x",
        )
        for n in names
    })


def _hosts_with(*names: str) -> HostsConfig:
    hosts = {}
    for n in names:
        if n == "anthropic":
            hosts[n] = HostConfig(provider="anthropic", api_key="fake")
        else:
            hosts[n] = HostConfig(provider="openai", base_url="http://x", api_key="fake")
    return HostsConfig(hosts=hosts, default_host="anthropic" if "anthropic" in names else names[0])


def test_validate_ok_when_all_references_resolve():
    agents = _build_agents(["receptionist", "core"])
    hosts = _hosts_with("anthropic")
    preset = PresetConfig(root_agent="receptionist", agents=["receptionist", "core"])
    # Should not raise
    validate_preset(preset, agents, hosts)


def test_validate_rejects_unknown_agent():
    agents = _build_agents(["receptionist"])
    hosts = _hosts_with("anthropic")
    preset = PresetConfig(root_agent="receptionist", agents=["receptionist", "ghost"])
    with pytest.raises(PresetValidationError, match="unknown agents"):
        validate_preset(preset, agents, hosts)


def test_validate_rejects_unknown_host():
    agents = _build_agents(["core"])
    hosts = _hosts_with("anthropic")
    preset = PresetConfig(
        root_agent="core", agents=["core"],
        model_overrides={"core": "mystery-host::any-model"},
    )
    with pytest.raises(PresetValidationError, match="unknown host"):
        validate_preset(preset, agents, hosts)


def test_validate_rejects_agent_model_host_missing():
    agents = AgentsRegistry(agents={
        "x": AgentConfig(
            model="vanished-host::some-model",
            instructions="x.md",
            name="x",
            instructions_text="",
        )
    })
    hosts = _hosts_with("anthropic")
    preset = PresetConfig(root_agent="x", agents=["x"])
    with pytest.raises(PresetValidationError, match="unknown host"):
        validate_preset(preset, agents, hosts)


def test_validate_override_supersedes_agent_default_host():
    # Agent references a missing host, but the preset overrides to a good one → OK
    agents = AgentsRegistry(agents={
        "x": AgentConfig(
            model="missing::m", instructions="x.md", name="x",
            instructions_text="",
        )
    })
    hosts = _hosts_with("anthropic")
    preset = PresetConfig(
        root_agent="x", agents=["x"],
        model_overrides={"x": "anthropic::c"},
    )
    validate_preset(preset, agents, hosts)  # should not raise


def test_validate_api_key_requirement_checked_when_enabled(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    agents = _build_agents(["core"])
    # Remove explicit api_key to force env lookup
    hosts = HostsConfig(
        hosts={"anthropic": HostConfig(
            provider="anthropic", api_key_env="ANTHROPIC_API_KEY",
        )},
        default_host="anthropic",
    )
    preset = PresetConfig(root_agent="core", agents=["core"])
    with pytest.raises(PresetValidationError, match="ANTHROPIC_API_KEY"):
        validate_preset(preset, agents, hosts, require_api_keys=True)


def test_validate_api_key_check_skipped_when_disabled(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    agents = _build_agents(["core"])
    hosts = HostsConfig(
        hosts={"anthropic": HostConfig(
            provider="anthropic", api_key_env="ANTHROPIC_API_KEY",
        )},
        default_host="anthropic",
    )
    preset = PresetConfig(root_agent="core", agents=["core"])
    # should not raise with require_api_keys=False
    validate_preset(preset, agents, hosts, require_api_keys=False)


def test_validate_anonymous_host_no_api_key_ok():
    """Local ollama has api_key_env=None — should pass even without env var."""
    agents = AgentsRegistry(agents={
        "qwen_reader": AgentConfig(
            model="local-ollama::qwen2.5-coder:32b",
            instructions="x.md", name="qwen_reader",
            instructions_text="",
        )
    })
    hosts = HostsConfig(hosts={
        "local-ollama": HostConfig(
            provider="openai",
            base_url="http://localhost:11434/v1",
            api_key_env=None,  # anonymous
        )
    }, default_host="local-ollama")
    preset = PresetConfig(root_agent="qwen_reader", agents=["qwen_reader"])
    validate_preset(preset, agents, hosts, require_api_keys=True)


# ──────────────────────────────────────────────────────────────────────
# prompt_overrides — schema + validate_preset + to_role end-to-end
# ──────────────────────────────────────────────────────────────────────


def test_prompt_overrides_schema_defaults_empty():
    p = PresetConfig(root_agent="core", agents=["core"])
    assert p.prompt_overrides == {}


def test_prompt_overrides_schema_accepts_valid_stems():
    p = PresetConfig(
        root_agent="core", agents=["core", "researcher"],
        prompt_overrides={"researcher": "researcher-stripped"},
    )
    assert p.prompt_overrides == {"researcher": "researcher-stripped"}


def test_prompt_overrides_schema_rejects_unknown_agent_key():
    with pytest.raises(Exception, match="prompt_overrides"):
        PresetConfig(
            root_agent="core", agents=["core"],
            prompt_overrides={"ghost": "researcher-stripped"},
        )


def test_prompt_overrides_schema_rejects_stem_with_slash():
    with pytest.raises(Exception, match="bare filename"):
        PresetConfig(
            root_agent="core", agents=["core"],
            prompt_overrides={"core": "../escape"},
        )


def test_prompt_overrides_schema_rejects_stem_with_extension():
    with pytest.raises(Exception, match="bare filename"):
        PresetConfig(
            root_agent="core", agents=["core"],
            prompt_overrides={"core": "researcher.md"},
        )


def test_prompt_overrides_schema_rejects_empty_stem():
    with pytest.raises(Exception, match="bare filename"):
        PresetConfig(
            root_agent="core", agents=["core"],
            prompt_overrides={"core": ""},
        )


def test_validate_preset_rejects_unresolvable_prompt_stem(tmp_path):
    agents = _build_agents(["core"])
    hosts = _hosts_with("anthropic")
    preset = PresetConfig(
        root_agent="core", agents=["core"],
        prompt_overrides={"core": "does-not-exist"},
    )
    with pytest.raises(PresetValidationError, match="prompt_overrides"):
        validate_preset(preset, agents, hosts, project_dir=str(tmp_path))


def test_validate_preset_accepts_prompt_stem_from_project_layer(
    tmp_path, monkeypatch,
):
    # Plant the variant instruction file in the project layer.
    proj_instr = tmp_path / ".nature" / "agents" / "instructions"
    proj_instr.mkdir(parents=True)
    (proj_instr / "researcher-stripped.md").write_text("stripped body")

    # Empty user home so only project resolves.
    monkeypatch.setenv("NATURE_HOME", str(tmp_path / "empty-home"))

    agents = _build_agents(["core"])
    hosts = _hosts_with("anthropic")
    preset = PresetConfig(
        root_agent="core", agents=["core"],
        prompt_overrides={"core": "researcher-stripped"},
    )
    # Should not raise.
    validate_preset(preset, agents, hosts, project_dir=str(tmp_path))


def test_agent_to_role_with_instructions_override():
    cfg = AgentConfig(
        model="anthropic::claude-haiku-4-5",
        instructions="core.md", name="core",
        instructions_text="ORIGINAL body",
    )
    role = cfg.to_role(instructions_override="STRIPPED body")
    assert role.instructions == "STRIPPED body"
    # Agent's own instructions_text is untouched.
    assert cfg.instructions_text == "ORIGINAL body"


def test_agent_to_role_without_override_uses_own_instructions():
    cfg = AgentConfig(
        model="anthropic::claude-haiku-4-5",
        instructions="core.md", name="core",
        instructions_text="ORIGINAL body",
    )
    role = cfg.to_role()
    assert role.instructions == "ORIGINAL body"


def test_load_agent_instruction_resolves_from_project_over_builtin(
    tmp_path, monkeypatch,
):
    from nature.agents.config import load_agent_instruction

    proj_instr = tmp_path / ".nature" / "agents" / "instructions"
    proj_instr.mkdir(parents=True)
    (proj_instr / "researcher.md").write_text("project-layer body")

    monkeypatch.setenv("NATURE_HOME", str(tmp_path / "empty-home"))

    text = load_agent_instruction("researcher", project_dir=str(tmp_path))
    assert text == "project-layer body"


def test_load_agent_instruction_raises_for_missing(tmp_path, monkeypatch):
    from nature.agents.config import load_agent_instruction

    monkeypatch.setenv("NATURE_HOME", str(tmp_path / "empty-home"))
    with pytest.raises(FileNotFoundError, match="not found"):
        load_agent_instruction(
            "ghost-no-such-stem", project_dir=str(tmp_path),
        )


def test_load_agent_instruction_rejects_path_separators(tmp_path):
    from nature.agents.config import load_agent_instruction

    with pytest.raises(ValueError, match="bare filename"):
        load_agent_instruction("../escape", project_dir=str(tmp_path))
