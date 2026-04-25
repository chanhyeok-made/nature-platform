"""Tests for `nature hosts` CLI subcommands.

Uses click's CliRunner + a temp NATURE_HOME so the user's real
~/.nature/hosts.json is never touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from nature.cli import main


@pytest.fixture
def nature_home(tmp_path, monkeypatch):
    """Redirect NATURE_HOME to a clean temp dir for each test."""
    home = tmp_path / "home-nature"
    home.mkdir()
    monkeypatch.setenv("NATURE_HOME", str(home))
    return home


def _user_hosts_file(home: Path) -> Path:
    return home / "hosts.json"


def _read_user_hosts(home: Path) -> dict:
    path = _user_hosts_file(home)
    if not path.exists():
        return {"hosts": {}, "default_host": "anthropic"}
    return json.loads(path.read_text())


# ──────────────────────────────────────────────────────────────────────
# list / show
# ──────────────────────────────────────────────────────────────────────


def test_hosts_list_shows_builtins(nature_home):
    result = CliRunner().invoke(main, ["hosts", "list"])
    assert result.exit_code == 0
    for name in ["anthropic", "openai", "local-ollama", "groq", "together", "openrouter"]:
        assert name in result.output
    assert "* anthropic" in result.output  # default marker


def test_hosts_show_existing(nature_home):
    result = CliRunner().invoke(main, ["hosts", "show", "groq"])
    assert result.exit_code == 0
    assert "groq" in result.output
    assert "api.groq.com" in result.output
    assert "GROQ_API_KEY" in result.output


def test_hosts_show_unknown(nature_home):
    result = CliRunner().invoke(main, ["hosts", "show", "nonexistent"])
    assert result.exit_code != 0
    assert "unknown host" in result.output


# ──────────────────────────────────────────────────────────────────────
# add
# ──────────────────────────────────────────────────────────────────────


def test_hosts_add_writes_user_file(nature_home):
    result = CliRunner().invoke(main, [
        "hosts", "add", "myhost",
        "--provider", "openai",
        "--base-url", "https://example.com/v1",
        "--api-key-env", "MY_KEY",
        "-m", "my-model-1",
        "-m", "my-model-2",
    ])
    assert result.exit_code == 0
    assert "saved host 'myhost'" in result.output

    data = _read_user_hosts(nature_home)
    assert "myhost" in data["hosts"]
    host = data["hosts"]["myhost"]
    assert host["provider"] == "openai"
    assert host["base_url"] == "https://example.com/v1"
    assert host["api_key_env"] == "MY_KEY"
    assert host["models"] == ["my-model-1", "my-model-2"]


def test_hosts_add_duplicate_without_force(nature_home):
    runner = CliRunner()
    runner.invoke(main, [
        "hosts", "add", "myhost",
        "--provider", "openai",
    ])
    result = runner.invoke(main, [
        "hosts", "add", "myhost",
        "--provider", "openai",
    ])
    assert result.exit_code != 0
    assert "already exists" in result.output


def test_hosts_add_with_force_overwrites(nature_home):
    runner = CliRunner()
    runner.invoke(main, [
        "hosts", "add", "myhost",
        "--provider", "openai",
        "-m", "old",
    ])
    result = runner.invoke(main, [
        "hosts", "add", "myhost",
        "--provider", "openai",
        "--base-url", "https://new.example.com/v1",
        "-m", "new",
        "--force",
    ])
    assert result.exit_code == 0

    host = _read_user_hosts(nature_home)["hosts"]["myhost"]
    assert host["base_url"] == "https://new.example.com/v1"
    assert host["models"] == ["new"]


def test_hosts_add_anonymous_is_ok(nature_home):
    """e.g., a second local ollama at a custom port."""
    result = CliRunner().invoke(main, [
        "hosts", "add", "ollama-alt",
        "--provider", "openai",
        "--base-url", "http://localhost:12345/v1",
    ])
    assert result.exit_code == 0
    host = _read_user_hosts(nature_home)["hosts"]["ollama-alt"]
    assert host["api_key_env"] is None


def test_hosts_add_shadowing_builtin_warns(nature_home):
    """Adding a user entry with the same name as a builtin warns but succeeds."""
    result = CliRunner().invoke(main, [
        "hosts", "add", "groq",
        "--provider", "openai",
        "--base-url", "https://my-groq-proxy.example.com/v1",
        "--api-key-env", "MY_PROXY_KEY",
        "--force",
    ])
    assert result.exit_code == 0
    assert "shadows a builtin" in result.output
    host = _read_user_hosts(nature_home)["hosts"]["groq"]
    assert host["base_url"] == "https://my-groq-proxy.example.com/v1"


# ──────────────────────────────────────────────────────────────────────
# remove
# ──────────────────────────────────────────────────────────────────────


def test_hosts_remove_user_entry(nature_home):
    runner = CliRunner()
    runner.invoke(main, ["hosts", "add", "tmp", "--provider", "openai"])
    assert "tmp" in _read_user_hosts(nature_home)["hosts"]

    result = runner.invoke(main, ["hosts", "remove", "tmp"])
    assert result.exit_code == 0
    assert "tmp" not in _read_user_hosts(nature_home)["hosts"]


def test_hosts_remove_builtin_refused(nature_home):
    result = CliRunner().invoke(main, ["hosts", "remove", "anthropic"])
    assert result.exit_code != 0
    assert "builtin" in result.output
    assert "shadow" in result.output


def test_hosts_remove_unknown(nature_home):
    result = CliRunner().invoke(main, ["hosts", "remove", "nonexistent"])
    assert result.exit_code != 0
    assert "not found" in result.output


# ──────────────────────────────────────────────────────────────────────
# set-default
# ──────────────────────────────────────────────────────────────────────


def test_set_default_to_builtin(nature_home):
    result = CliRunner().invoke(main, ["hosts", "set-default", "groq"])
    assert result.exit_code == 0
    data = _read_user_hosts(nature_home)
    assert data["default_host"] == "groq"


def test_set_default_to_user_added(nature_home):
    runner = CliRunner()
    runner.invoke(main, ["hosts", "add", "mine", "--provider", "openai"])
    result = runner.invoke(main, ["hosts", "set-default", "mine"])
    assert result.exit_code == 0
    assert _read_user_hosts(nature_home)["default_host"] == "mine"


def test_set_default_unknown_rejected(nature_home):
    result = CliRunner().invoke(main, ["hosts", "set-default", "nowhere"])
    assert result.exit_code != 0
    assert "unknown host" in result.output


def test_set_default_surfaces_in_list(nature_home):
    runner = CliRunner()
    runner.invoke(main, ["hosts", "set-default", "groq"])
    result = runner.invoke(main, ["hosts", "list"])
    assert "* groq" in result.output
