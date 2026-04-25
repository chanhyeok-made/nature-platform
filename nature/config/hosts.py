"""Host registry — decouple "where does the LLM live" from "which model".

A **Host** is a named endpoint: a provider type (anthropic / openai),
a base_url, an auth convention (api_key env var), and a list of models
the host is known to serve. Users reference models via
`<host>::<model>` (e.g., `local-ollama::qwen2.5-coder:32b`,
`anthropic::claude-haiku-4-5`, `groq::llama-3.3-70b-versatile`).

Layering (first found wins for the same host name, but dicts are
merged rather than overridden across layers):

1. Project — `.nature/hosts.json`
2. User — `~/.nature/hosts.json`
3. Built-in defaults — the `BUILTIN_HOSTS` table below

The builtin layer ships with anthropic, openai, local-ollama,
openrouter, groq, and together preconfigured so the minimum setup is
just exporting the right env var.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field


# ──────────────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────────────


class HostConfig(BaseModel):
    """One host entry in the registry.

    `api_key_env` names the environment variable to read. `api_key`
    overrides it (mostly for tests — real setups leave it null and use
    the env var). `base_url` is None for providers that own their own
    endpoint URL (today: anthropic, since the SDK is pinned to
    api.anthropic.com).
    """

    model_config = ConfigDict(extra="forbid")

    provider: str                                  # "anthropic" | "openai"
    base_url: str | None = None
    api_key_env: str | None = None                 # e.g. "GROQ_API_KEY"; None = anonymous (ollama)
    api_key: str | None = None                     # explicit override — usually leave null
    models: list[str] = Field(default_factory=list)

    def resolved_api_key(self) -> str | None:
        """Look up the API key: explicit value wins, else env var, else None."""
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        return None


class HostsConfig(BaseModel):
    """Top-level hosts file shape."""

    model_config = ConfigDict(extra="forbid")

    hosts: dict[str, HostConfig] = Field(default_factory=dict)
    default_host: str = "anthropic"

    def get_host(self, name: str) -> HostConfig | None:
        return self.hosts.get(name)

    def resolve(self, ref: str) -> tuple[str, HostConfig, str]:
        """Resolve `host::model` or unqualified `model` into
        `(host_name, host_config, model_name)`.

        Rules:
        - `host::model`                  → that exact pair
        - `::model`                      → `default_host::model`
        - `model` (no `::`)              → `default_host::model`
        - `host::` (empty model)         → ValueError
        - Unknown host name              → KeyError
        """
        host_name, model_name = parse_model_ref(ref)
        if host_name is None:
            host_name = self.default_host
        cfg = self.hosts.get(host_name)
        if cfg is None:
            raise KeyError(
                f"Unknown host {host_name!r}. "
                f"Available: {sorted(self.hosts.keys())}"
            )
        return (host_name, cfg, model_name)

    def list_model_refs(self) -> list[str]:
        """Every known `host::model` pair, sorted — for UI dropdowns."""
        out: list[str] = []
        for host_name in sorted(self.hosts.keys()):
            for model in self.hosts[host_name].models:
                out.append(f"{host_name}::{model}")
        return out


# ──────────────────────────────────────────────────────────────────────
# Reference parsing
# ──────────────────────────────────────────────────────────────────────


def parse_model_ref(ref: str) -> tuple[str | None, str]:
    """Split `host::model` or bare `model` into `(host_or_None, model)`.

    The separator `::` is chosen so single `:` in model names
    (e.g., `qwen2.5-coder:32b`) does not conflict.
    """
    if not ref:
        raise ValueError("Empty model reference")
    if "::" in ref:
        host, _, model = ref.partition("::")
        if not model:
            raise ValueError(
                f"Empty model name in reference {ref!r}. "
                f"Expected '<host>::<model>' or '<model>'."
            )
        return (host or None, model)
    return (None, ref)


def format_model_ref(host: str, model: str) -> str:
    """Inverse of parse_model_ref."""
    return f"{host}::{model}"


# ──────────────────────────────────────────────────────────────────────
# Built-in defaults
# ──────────────────────────────────────────────────────────────────────


BUILTIN_HOSTS: dict[str, HostConfig] = {
    "anthropic": HostConfig(
        provider="anthropic",
        api_key_env="ANTHROPIC_API_KEY",
        models=[
            "claude-opus-4-7",
            "claude-sonnet-4-6",
            "claude-haiku-4-5",
        ],
    ),
    "openai": HostConfig(
        provider="openai",
        api_key_env="OPENAI_API_KEY",
        models=[
            "gpt-4o",
            "gpt-4o-mini",
        ],
    ),
    "local-ollama": HostConfig(
        provider="openai",
        base_url="http://localhost:11434/v1",
        api_key_env=None,  # anonymous
        models=[
            "qwen2.5-coder:32b",
            "qwen3:30b",
            "deepseek-r1:32b",
            "llama3.3:70b",
        ],
    ),
    "openrouter": HostConfig(
        provider="openai",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        models=[
            # Claude family (cloud baseline)
            "anthropic/claude-sonnet-4.6",
            "anthropic/claude-haiku-4.5",
            "anthropic/claude-opus-4",
            # GPT family
            "openai/gpt-4o",
            "openai/gpt-4o-mini",
            "openai/gpt-5-mini",
            # Google
            "google/gemini-2.5-pro",
            "google/gemini-2.5-flash",
            # Meta — direct cloud comparison to our local llama3.3:70b
            "meta-llama/llama-3.3-70b-instruct",
            # DeepSeek — cloud comparison to local r1/v3
            "deepseek/deepseek-chat-v3.1",
            "deepseek/deepseek-r1",
            # Mistral
            "mistralai/mistral-large",
            "mistralai/mixtral-8x22b-instruct",
            # Qwen — direct cloud comparison to local qwen2.5:72b
            "qwen/qwen-2.5-72b-instruct",
            "qwen/qwen-2.5-coder-32b-instruct",
            # xAI
            "x-ai/grok-4",
            "x-ai/grok-3-mini",
        ],
    ),
    "groq": HostConfig(
        provider="openai",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        models=[
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "moonshotai/kimi-k2",
            "qwen/qwen3-32b",
        ],
    ),
    "together": HostConfig(
        provider="openai",
        base_url="https://api.together.xyz/v1",
        api_key_env="TOGETHER_API_KEY",
        models=[
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "deepseek-ai/DeepSeek-V3",
            "Qwen/Qwen2.5-Coder-32B-Instruct",
            "mistralai/Mixtral-8x22B-Instruct-v0.1",
        ],
    ),
}


def builtin_hosts_config() -> HostsConfig:
    """Fresh copy of the built-in hosts (callers are free to mutate)."""
    return HostsConfig(
        hosts={k: v.model_copy(deep=True) for k, v in BUILTIN_HOSTS.items()},
        default_host="anthropic",
    )


# ──────────────────────────────────────────────────────────────────────
# Loader
# ──────────────────────────────────────────────────────────────────────


def _candidate_paths(project_dir: Path | str | None) -> Iterable[Path]:
    """Yield candidate hosts.json paths in priority order (highest first)."""
    if project_dir is not None:
        yield Path(project_dir) / ".nature" / "hosts.json"
    # Local import to avoid a config → settings cycle at module import.
    from nature.config.settings import get_nature_home
    yield get_nature_home() / "hosts.json"


def load_hosts_config(
    project_dir: Path | str | None = None,
) -> HostsConfig:
    """Load hosts config.

    Layering, in ascending priority (later layers override earlier):

    1. Built-in defaults
    2. User file at `~/.nature/hosts.json`
    3. Project file at `.nature/hosts.json`

    `hosts` dicts merge (later keys add or replace entries). `default_host`
    is a last-writer-wins scalar. Never returns None — always at least the
    built-in defaults.
    """
    merged = builtin_hosts_config()

    # Collect existing files in ascending priority (user, then project).
    files = list(_candidate_paths(project_dir))[::-1]  # reverse → low→high
    for path in files:
        if not path.exists():
            continue
        try:
            layer = HostsConfig.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(
                f"Failed to load hosts config from {path}: {exc}"
            ) from exc
        merged.hosts.update(layer.hosts)
        if layer.default_host:
            merged.default_host = layer.default_host

    # Validate default_host exists after merge
    if merged.default_host not in merged.hosts:
        raise ValueError(
            f"default_host {merged.default_host!r} is not a registered host. "
            f"Available: {sorted(merged.hosts.keys())}"
        )

    return merged


# ──────────────────────────────────────────────────────────────────────
# User-layer read/write (used by `nature hosts` CLI)
# ──────────────────────────────────────────────────────────────────────


def user_hosts_path() -> Path:
    """Path to the user's personal hosts.json (`~/.nature/hosts.json`)."""
    from nature.config.settings import get_nature_home
    return get_nature_home() / "hosts.json"


def load_user_hosts_config() -> HostsConfig:
    """Load ONLY the user layer (~/.nature/hosts.json).

    Does NOT include builtins or the project layer. Use this for
    write-back: `nature hosts add` / `remove` / `set-default` mutate
    just the user file, preserving the diff-from-builtins shape.

    Returns an empty config (no hosts, default_host='anthropic') if
    the file doesn't exist.
    """
    path = user_hosts_path()
    if not path.exists():
        return HostsConfig(hosts={}, default_host="anthropic")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(
            f"Failed to parse user hosts config at {path}: {exc}"
        ) from exc
    # We validate with model_validate (not model_validate_json) so we
    # can tolerate a missing default_host in user files and fall back
    # to the builtin default.
    raw.setdefault("default_host", "anthropic")
    return HostsConfig.model_validate(raw)


def save_user_hosts_config(cfg: HostsConfig) -> Path:
    """Write the user hosts config to `~/.nature/hosts.json`.

    Creates the parent directory if needed. Uses a pretty-printed
    format for human editability.
    """
    path = user_hosts_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cfg.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def is_builtin(name: str) -> bool:
    return name in BUILTIN_HOSTS


# ──────────────────────────────────────────────────────────────────────
# Admin helpers (used by the HTTP admin API)
# ──────────────────────────────────────────────────────────────────────


def load_hosts_with_origin(
    project_dir: Path | str | None = None,
) -> tuple[list[tuple[str, HostConfig, str]], str]:
    """Every host annotated with `origin` plus the effective
    `default_host`. project > user > builtin per name. Later layers
    overwrite the tuple so the reported origin matches where the
    current value lives.
    """
    merged: dict[str, tuple[HostConfig, str]] = {}
    for name, cfg in BUILTIN_HOSTS.items():
        merged[name] = (cfg.model_copy(deep=True), "builtin")

    default_host = "anthropic"

    # User layer
    user_path = user_hosts_path()
    if user_path.exists():
        try:
            raw = json.loads(user_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(
                f"Failed to parse user hosts config at {user_path}: {exc}",
            ) from exc
        for name, host_body in (raw.get("hosts") or {}).items():
            merged[name] = (HostConfig.model_validate(host_body), "user")
        if raw.get("default_host"):
            default_host = raw["default_host"]

    # Project layer
    if project_dir is not None:
        proj_path = Path(project_dir) / ".nature" / "hosts.json"
        if proj_path.exists():
            try:
                raw = json.loads(proj_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise ValueError(
                    f"Failed to parse project hosts config at {proj_path}: {exc}",
                ) from exc
            for name, host_body in (raw.get("hosts") or {}).items():
                merged[name] = (HostConfig.model_validate(host_body), "project")
            if raw.get("default_host"):
                default_host = raw["default_host"]

    entries = sorted(
        ((name, cfg, origin) for name, (cfg, origin) in merged.items()),
        key=lambda t: t[0],
    )
    return entries, default_host


def delete_user_host(name: str) -> bool:
    """Remove one host entry from the user layer. False if the user
    file doesn't list that name (caller 403's when the name is a
    builtin)."""
    cfg = load_user_hosts_config()
    if name not in cfg.hosts:
        return False
    del cfg.hosts[name]
    save_user_hosts_config(cfg)
    return True


__all__ = [
    "HostConfig",
    "HostsConfig",
    "BUILTIN_HOSTS",
    "builtin_hosts_config",
    "load_hosts_config",
    "load_user_hosts_config",
    "save_user_hosts_config",
    "user_hosts_path",
    "is_builtin",
    "parse_model_ref",
    "format_model_ref",
    "load_hosts_with_origin",
    "delete_user_host",
]
