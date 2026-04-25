"""Tests for the preset-based session flow (Stage A).

Covers the three helpers introduced in SessionRegistry for the new
preset path (`_build_provider_pool`, `_make_provider_resolver`,
`_make_preset_role_resolver`) plus one end-to-end integration test
that drives `create_session` with a preset and asserts the resulting
ServerSession is wired correctly.

These tests never touch the network — provider classes are
constructed but not called, so a dummy `ANTHROPIC_API_KEY` in env is
enough for validate_preset to accept the preset.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nature.agents.config import AgentConfig, AgentsRegistry
from nature.agents.presets import PresetConfig
from nature.config.hosts import HostConfig, HostsConfig
from nature.events.store import FileEventStore
from nature.server.api import CreateSessionRequest
from nature.server.registry import SessionRegistry


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


def _fake_agents_registry() -> AgentsRegistry:
    """Two-agent roster pointing at two different hosts.

    - alpha → anthropic (Claude)
    - beta  → local-ollama (open-ai compat)

    Lets us assert the pool correctly creates one provider per host.
    """
    return AgentsRegistry(agents={
        "alpha": AgentConfig(
            model="anthropic::claude-haiku-4-5",
            instructions="alpha.md",
            name="alpha",
            instructions_text="you are alpha",
        ),
        "beta": AgentConfig(
            model="local-ollama::qwen2.5-coder:32b",
            instructions="beta.md",
            name="beta",
            instructions_text="you are beta",
        ),
    })


def _fake_hosts_config() -> HostsConfig:
    return HostsConfig(
        hosts={
            "anthropic": HostConfig(
                provider="anthropic",
                api_key="fake-anthropic-key",
            ),
            "local-ollama": HostConfig(
                provider="openai",
                base_url="http://localhost:11434/v1",
                api_key_env=None,
            ),
        },
        default_host="anthropic",
    )


@pytest.fixture
def registry(tmp_path, monkeypatch) -> SessionRegistry:
    """A SessionRegistry rooted at an empty temp cwd (no frame.json)."""
    store = FileEventStore(root=tmp_path / "events")
    return SessionRegistry(event_store=store, cwd=str(tmp_path))


# ──────────────────────────────────────────────────────────────────────
# Unit: _build_provider_pool
# ──────────────────────────────────────────────────────────────────────


def test_build_provider_pool_one_per_distinct_host(registry):
    agents = _fake_agents_registry()
    hosts = _fake_hosts_config()
    preset = PresetConfig(
        root_agent="alpha",
        agents=["alpha", "beta"],
    )

    pool = registry._build_provider_pool(preset, agents, hosts)

    assert set(pool.keys()) == {"anthropic", "local-ollama"}
    # Provider classes chosen by host.provider
    from nature.providers.anthropic import AnthropicProvider
    from nature.providers.openai_compat import OpenAICompatProvider
    assert isinstance(pool["anthropic"], AnthropicProvider)
    assert isinstance(pool["local-ollama"], OpenAICompatProvider)


def test_build_provider_pool_override_changes_host(registry):
    """A model_override that points to a different host must extend the pool."""
    agents = _fake_agents_registry()
    hosts = _fake_hosts_config()
    # alpha's default is anthropic; override points it to local-ollama.
    preset = PresetConfig(
        root_agent="alpha",
        agents=["alpha"],
        model_overrides={"alpha": "local-ollama::qwen2.5-coder:32b"},
    )

    pool = registry._build_provider_pool(preset, agents, hosts)
    # Only local-ollama is referenced after override; anthropic is absent.
    assert set(pool.keys()) == {"local-ollama"}


def test_build_provider_pool_dedups_same_host(registry):
    """Two agents on the same host share a single provider instance."""
    agents = AgentsRegistry(agents={
        a: AgentConfig(
            model="anthropic::claude-haiku-4-5",
            instructions=f"{a}.md", name=a, instructions_text="",
        )
        for a in ("alpha", "beta")
    })
    hosts = _fake_hosts_config()
    preset = PresetConfig(root_agent="alpha", agents=["alpha", "beta"])

    pool = registry._build_provider_pool(preset, agents, hosts)
    assert set(pool.keys()) == {"anthropic"}


# ──────────────────────────────────────────────────────────────────────
# Unit: _make_provider_resolver
# ──────────────────────────────────────────────────────────────────────


def test_provider_resolver_routes_on_agent_name(registry):
    agents = _fake_agents_registry()
    hosts = _fake_hosts_config()
    preset = PresetConfig(root_agent="alpha", agents=["alpha", "beta"])
    pool = registry._build_provider_pool(preset, agents, hosts)

    resolve = registry._make_provider_resolver(preset, agents, pool)

    assert resolve("alpha") is pool["anthropic"]
    assert resolve("beta") is pool["local-ollama"]
    # Off-roster → None so AreaManager falls back to default provider
    assert resolve("gamma") is None


def test_provider_resolver_follows_model_override(registry):
    """An override re-routes the agent to a different host's provider."""
    agents = _fake_agents_registry()
    hosts = _fake_hosts_config()
    preset = PresetConfig(
        root_agent="alpha",
        agents=["alpha", "beta"],
        # flip alpha onto the ollama host
        model_overrides={"alpha": "local-ollama::qwen2.5-coder:32b"},
    )
    pool = registry._build_provider_pool(preset, agents, hosts)
    resolve = registry._make_provider_resolver(preset, agents, pool)

    assert resolve("alpha") is pool["local-ollama"]
    assert resolve("beta") is pool["local-ollama"]


# ──────────────────────────────────────────────────────────────────────
# Unit: _make_preset_role_resolver
# ──────────────────────────────────────────────────────────────────────


def test_role_resolver_returns_role_for_roster_member(registry):
    agents = _fake_agents_registry()
    preset = PresetConfig(root_agent="alpha", agents=["alpha", "beta"])

    resolve = registry._make_preset_role_resolver(preset, agents)
    role = resolve("beta")
    assert role is not None
    assert role.name == "beta"
    # bare model — host prefix stripped for host-agnostic rendering
    assert role.model == "qwen2.5-coder:32b"


def test_role_resolver_applies_model_override(registry):
    agents = _fake_agents_registry()
    preset = PresetConfig(
        root_agent="alpha",
        agents=["alpha"],
        model_overrides={"alpha": "anthropic::claude-sonnet-4-6"},
    )
    resolve = registry._make_preset_role_resolver(preset, agents)
    role = resolve("alpha")
    assert role is not None
    assert role.model == "claude-sonnet-4-6"


def test_role_resolver_returns_none_for_off_roster(registry):
    agents = _fake_agents_registry()
    preset = PresetConfig(root_agent="alpha", agents=["alpha"])
    resolve = registry._make_preset_role_resolver(preset, agents)
    assert resolve("beta") is None  # off roster
    assert resolve("ghost") is None  # unknown


# ──────────────────────────────────────────────────────────────────────
# Integration: create_session(preset=...)
# ──────────────────────────────────────────────────────────────────────


def _write_json(path: Path, body: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body), encoding="utf-8")


@pytest.mark.asyncio
async def test_create_session_with_preset_populates_new_fields(
    tmp_path, monkeypatch,
):
    """End-to-end: dispatch → preset branch → ServerSession with
    preset/agents_registry/provider_pool populated."""
    monkeypatch.setenv("NATURE_HOME", str(tmp_path / "home-nature"))
    # validate_preset requires ANTHROPIC_API_KEY (the only host that
    # has api_key_env set in the fixtures).
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")

    project = tmp_path / "proj"
    # Agent fixtures under project/.nature/agents/
    _write_json(project / ".nature" / "agents" / "alpha.json", {
        "model": "anthropic::claude-haiku-4-5",
        "instructions": "alpha.md",
    })
    (project / ".nature" / "agents" / "instructions").mkdir(parents=True)
    (project / ".nature" / "agents" / "instructions" / "alpha.md").write_text(
        "you are alpha"
    )
    _write_json(project / ".nature" / "agents" / "beta.json", {
        "model": "local-ollama::qwen2.5-coder:32b",
        "instructions": "beta.md",
    })
    (project / ".nature" / "agents" / "instructions" / "beta.md").write_text(
        "you are beta"
    )
    # Preset fixture
    _write_json(project / ".nature" / "presets" / "two-agents.json", {
        "root_agent": "alpha",
        "agents": ["alpha", "beta"],
    })

    store = FileEventStore(root=tmp_path / "events")
    reg = SessionRegistry(event_store=store, cwd=str(project))
    session = await reg.create_session(CreateSessionRequest(preset="two-agents"))

    # Root role is alpha, with its bare model (host:: stripped)
    assert session.root_role.name == "alpha"
    assert session.root_model == "claude-haiku-4-5"
    assert session.provider_name == "anthropic"

    # Preset-flow fields populated
    assert session.preset is not None
    assert session.preset.name == "two-agents"
    assert session.agents_registry is not None
    assert {"alpha", "beta"}.issubset(set(session.agents_registry.names()))

    # One provider per distinct host
    assert session.provider_pool is not None
    assert set(session.provider_pool.keys()) == {"anthropic", "local-ollama"}


@pytest.mark.asyncio
async def test_create_session_missing_preset_raises_file_not_found(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("NATURE_HOME", str(tmp_path / "home-nature"))
    store = FileEventStore(root=tmp_path / "events")
    reg = SessionRegistry(event_store=store, cwd=str(tmp_path))
    with pytest.raises(FileNotFoundError, match="does-not-exist"):
        await reg.create_session(CreateSessionRequest(preset="does-not-exist"))


@pytest.mark.asyncio
async def test_create_session_without_preset_uses_builtin_default(
    tmp_path, monkeypatch,
):
    """`CreateSessionRequest()` without an explicit preset resolves to
    the builtin `default.json` shipped with the package."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    monkeypatch.setenv("NATURE_HOME", str(tmp_path / "home-nature"))

    store = FileEventStore(root=tmp_path / "events")
    reg = SessionRegistry(event_store=store, cwd=str(tmp_path))

    session = await reg.create_session(CreateSessionRequest())

    assert session.preset is not None
    assert session.preset.name == "default"
    assert session.root_role.name == "receptionist"
    assert session.agents_registry is not None
    assert session.provider_pool is not None
