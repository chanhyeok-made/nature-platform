"""Tests for the Hosts registry — parsing, resolution, loading, merging."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from nature.config.hosts import (
    BUILTIN_HOSTS,
    HostConfig,
    HostsConfig,
    builtin_hosts_config,
    format_model_ref,
    load_hosts_config,
    parse_model_ref,
)


# ──────────────────────────────────────────────────────────────────────
# parse_model_ref / format_model_ref
# ──────────────────────────────────────────────────────────────────────


def test_parse_unqualified_model():
    assert parse_model_ref("claude-haiku-4-5") == (None, "claude-haiku-4-5")


def test_parse_namespaced_model():
    assert parse_model_ref("groq::llama-3.3-70b-versatile") == (
        "groq", "llama-3.3-70b-versatile"
    )


def test_parse_model_with_colons_in_name():
    """Single `:` in the model segment should stay part of the model name."""
    assert parse_model_ref("local-ollama::qwen2.5-coder:32b") == (
        "local-ollama", "qwen2.5-coder:32b"
    )


def test_parse_model_with_slashes():
    assert parse_model_ref("openrouter::anthropic/claude-sonnet-4") == (
        "openrouter", "anthropic/claude-sonnet-4"
    )


def test_parse_empty_host_keeps_model():
    """`::model` → host None (so default applies), model as given."""
    assert parse_model_ref("::claude-haiku-4-5") == (None, "claude-haiku-4-5")


def test_parse_empty_model_raises():
    with pytest.raises(ValueError):
        parse_model_ref("groq::")


def test_parse_empty_string_raises():
    with pytest.raises(ValueError):
        parse_model_ref("")


def test_format_model_ref_inverse():
    assert format_model_ref("groq", "llama-3.3-70b-versatile") == (
        "groq::llama-3.3-70b-versatile"
    )


# ──────────────────────────────────────────────────────────────────────
# HostConfig — api_key resolution
# ──────────────────────────────────────────────────────────────────────


def test_host_explicit_api_key_wins():
    h = HostConfig(provider="openai", api_key="explicit", api_key_env="SHOULD_BE_IGNORED")
    assert h.resolved_api_key() == "explicit"


def test_host_api_key_env_fallback():
    h = HostConfig(provider="openai", api_key_env="TEST_NATURE_HOSTS_KEY")
    try:
        os.environ["TEST_NATURE_HOSTS_KEY"] = "from-env"
        assert h.resolved_api_key() == "from-env"
    finally:
        os.environ.pop("TEST_NATURE_HOSTS_KEY", None)


def test_host_anonymous_returns_none():
    h = HostConfig(provider="openai", base_url="http://localhost:11434/v1")
    # no api_key, no api_key_env, no matching env
    assert h.resolved_api_key() is None


def test_host_rejects_unknown_fields():
    # extra="forbid" → extra keys fail validation
    with pytest.raises(Exception):
        HostConfig(provider="openai", unexpected="bad")


# ──────────────────────────────────────────────────────────────────────
# HostsConfig.resolve
# ──────────────────────────────────────────────────────────────────────


def _sample_config() -> HostsConfig:
    return HostsConfig(
        hosts={
            "anthropic": HostConfig(provider="anthropic", api_key_env="ANTHROPIC_API_KEY"),
            "local-ollama": HostConfig(
                provider="openai", base_url="http://localhost:11434/v1",
                models=["qwen2.5-coder:32b"],
            ),
        },
        default_host="anthropic",
    )


def test_resolve_namespaced():
    cfg = _sample_config()
    host_name, host_cfg, model = cfg.resolve("local-ollama::qwen2.5-coder:32b")
    assert host_name == "local-ollama"
    assert host_cfg.base_url == "http://localhost:11434/v1"
    assert model == "qwen2.5-coder:32b"


def test_resolve_unqualified_uses_default_host():
    cfg = _sample_config()
    host_name, _, model = cfg.resolve("claude-haiku-4-5")
    assert host_name == "anthropic"
    assert model == "claude-haiku-4-5"


def test_resolve_empty_host_uses_default():
    cfg = _sample_config()
    host_name, _, model = cfg.resolve("::my-model")
    assert host_name == "anthropic"
    assert model == "my-model"


def test_resolve_unknown_host_raises():
    cfg = _sample_config()
    with pytest.raises(KeyError):
        cfg.resolve("mystery-host::some-model")


def test_list_model_refs():
    cfg = _sample_config()
    # anthropic has no models declared, local-ollama has one
    refs = cfg.list_model_refs()
    assert "local-ollama::qwen2.5-coder:32b" in refs


# ──────────────────────────────────────────────────────────────────────
# Built-in defaults
# ──────────────────────────────────────────────────────────────────────


def test_builtin_hosts_include_major_providers():
    assert "anthropic" in BUILTIN_HOSTS
    assert "openai" in BUILTIN_HOSTS
    assert "local-ollama" in BUILTIN_HOSTS
    assert "openrouter" in BUILTIN_HOSTS
    assert "groq" in BUILTIN_HOSTS
    assert "together" in BUILTIN_HOSTS


def test_builtin_ollama_is_anonymous():
    assert BUILTIN_HOSTS["local-ollama"].api_key_env is None
    assert BUILTIN_HOSTS["local-ollama"].base_url == "http://localhost:11434/v1"


def test_builtin_anthropic_has_no_base_url():
    """Anthropic SDK has its own endpoint; base_url stays None."""
    assert BUILTIN_HOSTS["anthropic"].base_url is None


def test_builtin_groq_endpoint():
    assert BUILTIN_HOSTS["groq"].base_url == "https://api.groq.com/openai/v1"
    assert BUILTIN_HOSTS["groq"].api_key_env == "GROQ_API_KEY"


def test_builtin_config_is_fresh_copy():
    """builtin_hosts_config() returns an independent copy so callers can
    safely mutate without poisoning the global table."""
    a = builtin_hosts_config()
    b = builtin_hosts_config()
    a.hosts["local-ollama"].models.append("modified")
    assert "modified" not in b.hosts["local-ollama"].models


# ──────────────────────────────────────────────────────────────────────
# load_hosts_config — layering
# ──────────────────────────────────────────────────────────────────────


def test_load_without_files_returns_builtins(tmp_path, monkeypatch):
    # Redirect ~/.nature to a clean dir
    fake_home = tmp_path / "home-nature"
    fake_home.mkdir()
    monkeypatch.setenv("NATURE_HOME", str(fake_home))

    cfg = load_hosts_config(project_dir=str(tmp_path))
    assert "anthropic" in cfg.hosts
    assert "groq" in cfg.hosts
    assert cfg.default_host == "anthropic"


def test_load_project_overrides_user(tmp_path, monkeypatch):
    fake_home = tmp_path / "home-nature"
    fake_home.mkdir()
    monkeypatch.setenv("NATURE_HOME", str(fake_home))

    # User config adds a custom host
    user_path = fake_home / "hosts.json"
    user_path.write_text(json.dumps({
        "hosts": {
            "user-host": {
                "provider": "openai",
                "base_url": "https://user.example.com/v1",
                "api_key_env": "USER_KEY",
            }
        },
        "default_host": "user-host",
    }))

    # Project config overrides default_host + adds another host
    project_dir = tmp_path / "proj"
    (project_dir / ".nature").mkdir(parents=True)
    project_path = project_dir / ".nature" / "hosts.json"
    project_path.write_text(json.dumps({
        "hosts": {
            "proj-host": {
                "provider": "openai",
                "base_url": "https://proj.example.com/v1",
            }
        },
        "default_host": "proj-host",
    }))

    cfg = load_hosts_config(project_dir=str(project_dir))

    # All three layers merged
    assert "anthropic" in cfg.hosts          # builtin
    assert "user-host" in cfg.hosts          # user layer
    assert "proj-host" in cfg.hosts          # project layer
    # Project wins on default_host
    assert cfg.default_host == "proj-host"


def test_load_default_host_must_exist_after_merge(tmp_path, monkeypatch):
    fake_home = tmp_path / "home-nature"
    fake_home.mkdir()
    monkeypatch.setenv("NATURE_HOME", str(fake_home))

    bad_path = fake_home / "hosts.json"
    bad_path.write_text(json.dumps({
        "hosts": {},
        "default_host": "nonexistent",
    }))

    with pytest.raises(ValueError, match="default_host"):
        load_hosts_config()


def test_load_malformed_json_surfaces_error(tmp_path, monkeypatch):
    fake_home = tmp_path / "home-nature"
    fake_home.mkdir()
    monkeypatch.setenv("NATURE_HOME", str(fake_home))

    bad_path = fake_home / "hosts.json"
    bad_path.write_text("{not valid json")

    with pytest.raises(ValueError, match="Failed to load"):
        load_hosts_config()
