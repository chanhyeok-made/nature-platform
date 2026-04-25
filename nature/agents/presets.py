"""Preset registry — "which agents to combine + per-preset model overrides".

A preset is a named team composition: a root agent, the full roster of
agents the session may spawn as sub-agents, and optional per-agent model
overrides that apply only while this preset is active.

File layout:
    .nature/presets/<name>.json                 # project-local (wins)
    ~/.nature/presets/<name>.json               # user-level (fallback)

Schema:
    {
      "root_agent": "receptionist",
      "agents": ["receptionist", "core", "researcher", ...],
      "model_overrides": {
        "core": "groq::llama-3.3-70b-versatile"
      }
    }

Semantics (locked 2026-04-17):

- `agents` is REQUIRED and explicit — a preset declares exactly which
  specialists are reachable during its sessions (Q3-a). The AreaManager
  refuses to spawn a child agent whose role name isn't listed here
  (Q3-b — delegated to `validate_preset`).
- `root_agent` MUST appear in `agents`.
- `model_overrides` keys MUST reference names listed in `agents`.
- `model_overrides` values MUST use `host::model` form (same rule as
  `AgentConfig.model` — enforced in resolve-time validation).
- Preset activation is **session-scoped** (Q3-d): no global "active
  preset" state. Session creation injects the preset name; the session
  holds a reference for its lifetime. A default preset file named
  `default.json` is looked up when session creation omits the name.

Validation (the Django `ready()` moment):

Every dangling reference (unknown agent, unknown host, missing
`host::model`, API key absence on a required host) must surface as a
clear error BEFORE any frames open. `validate_preset()` does the full
walk and reports the first problem.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nature.agents.config import AgentsRegistry, load_agent_instruction
from nature.config.hosts import HostsConfig, parse_model_ref


class PresetConfig(BaseModel):
    """One preset definition — parsed from disk by `load_preset`."""

    model_config = ConfigDict(extra="forbid")

    root_agent: str
    agents: list[str]
    model_overrides: dict[str, str] = Field(default_factory=dict)
    # Per-agent instruction-file override: `{agent_name: instruction_stem}`.
    # Resolved via `load_agent_instruction(stem)` at session-build time,
    # so only the prompt body swaps — model, tools, description stay.
    prompt_overrides: dict[str, str] = Field(default_factory=dict)
    # Per-agent policy budget for `max_output_tokens`. Judge/reviewer
    # typically need ~1-2K (verdict + rationale); researcher and analyzer
    # want more headroom for long synthesis; implementer sits in the
    # middle. Composed with the model's physical ceiling as
    # `min(role_budget, model_ceiling)` at LLM call time — so a role
    # budget that exceeds what the backend accepts still lands safely.
    # Missing entry = fall through to the provider default.
    max_output_tokens_overrides: dict[str, int] = Field(default_factory=dict)

    @field_validator("agents")
    @classmethod
    def _agents_nonempty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("preset.agents must be non-empty")
        return v

    @model_validator(mode="after")
    def _root_in_agents(self) -> "PresetConfig":
        if self.root_agent not in self.agents:
            raise ValueError(
                f"root_agent {self.root_agent!r} not in agents list "
                f"{sorted(self.agents)}"
            )
        return self

    @model_validator(mode="after")
    def _overrides_reference_listed_agents(self) -> "PresetConfig":
        extras = set(self.model_overrides) - set(self.agents)
        if extras:
            raise ValueError(
                f"model_overrides keys {sorted(extras)} are not in agents list"
            )
        extras_prompt = set(self.prompt_overrides) - set(self.agents)
        if extras_prompt:
            raise ValueError(
                f"prompt_overrides keys {sorted(extras_prompt)} "
                f"are not in agents list"
            )
        extras_tokens = set(self.max_output_tokens_overrides) - set(self.agents)
        if extras_tokens:
            raise ValueError(
                f"max_output_tokens_overrides keys {sorted(extras_tokens)} "
                f"are not in agents list"
            )
        for agent_name, budget in self.max_output_tokens_overrides.items():
            if budget <= 0:
                raise ValueError(
                    f"max_output_tokens_overrides[{agent_name!r}] = {budget} "
                    f"must be positive"
                )
        return self

    @model_validator(mode="after")
    def _overrides_are_host_qualified(self) -> "PresetConfig":
        for agent_name, model_ref in self.model_overrides.items():
            if "::" not in model_ref:
                raise ValueError(
                    f"model_overrides[{agent_name!r}] = {model_ref!r} "
                    f"must be in host::model form"
                )
        return self

    @model_validator(mode="after")
    def _prompt_overrides_are_bare_stems(self) -> "PresetConfig":
        for agent_name, stem in self.prompt_overrides.items():
            if "/" in stem or "\\" in stem or "." in stem or not stem:
                raise ValueError(
                    f"prompt_overrides[{agent_name!r}] = {stem!r} "
                    f"must be a bare filename stem (no separators, no extension)"
                )
        return self

    # Populated by the loader from the filename; not serialized back.
    name: str = Field(default="", exclude=True)


# ──────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────


class PresetValidationError(ValueError):
    """Raised when a preset references unknown agents/hosts or keys.

    The session-creation layer catches this, surfaces it through the
    HTTP API as a clear `invalid_preset` error, and refuses to start
    the session. Separate exception type so callers can distinguish a
    "your preset is malformed" failure from an IO/schema failure.
    """


def validate_preset(
    preset: PresetConfig,
    agents: AgentsRegistry,
    hosts: HostsConfig,
    *,
    require_api_keys: bool = True,
    project_dir: Path | str | None = None,
) -> None:
    """Check every reference in a preset resolves against the live
    registries. Raises `PresetValidationError` on first missing.

    Checked in order:

    1. Every name in `preset.agents` exists in `agents`.
    2. Every effective model (override or agent default) uses host::model.
    3. Every referenced host exists in `hosts`.
    4. Every host that needs an API key has one (optional — tests pass
       False to skip this in no-env environments).
    5. Every `prompt_overrides` entry resolves to an existing
       `instructions/<stem>.md` file under project > user > builtin.
    """
    # 1. Agent existence
    missing = [name for name in preset.agents if agents.get(name) is None]
    if missing:
        raise PresetValidationError(
            f"preset references unknown agents: {sorted(missing)}. "
            f"Known agents: {agents.names()}"
        )

    # 2+3. Walk each effective model
    for name in preset.agents:
        agent = agents.get(name)
        assert agent is not None  # checked in step 1
        model_ref = preset.model_overrides.get(name, agent.model)

        host_name, model_name = parse_model_ref(model_ref)
        if host_name is None:
            # AgentConfig already enforces host::model; this catches
            # overrides that slipped past PresetConfig._overrides_are_host_qualified
            raise PresetValidationError(
                f"agent {name!r} model {model_ref!r} is not host-qualified"
            )

        host = hosts.get_host(host_name)
        if host is None:
            raise PresetValidationError(
                f"agent {name!r} references unknown host {host_name!r}. "
                f"Known hosts: {sorted(hosts.hosts)}"
            )

        # 4. API key check
        if require_api_keys and host.api_key_env and not host.resolved_api_key():
            raise PresetValidationError(
                f"agent {name!r} uses host {host_name!r} which requires "
                f"${host.api_key_env} (not set in env)"
            )

    # 5. prompt_overrides stems resolve to actual MD files
    for agent_name, stem in preset.prompt_overrides.items():
        try:
            load_agent_instruction(stem, project_dir=project_dir)
        except FileNotFoundError as exc:
            raise PresetValidationError(
                f"prompt_overrides[{agent_name!r}] references {stem!r} "
                f"which does not resolve: {exc}"
            ) from exc


# ──────────────────────────────────────────────────────────────────────
# Loader
# ──────────────────────────────────────────────────────────────────────


def _builtin_dir() -> Path:
    """Directory of shipped preset JSONs (inside the nature package)."""
    return Path(__file__).parent / "builtin" / "presets"


def _user_dir() -> Path:
    from nature.config.settings import get_nature_home
    return get_nature_home() / "presets"


def _project_dir(project_dir: Path | str | None) -> Path | None:
    if project_dir is None:
        return None
    return Path(project_dir) / ".nature" / "presets"


def _candidate_paths(
    name: str, project_dir: Path | str | None
) -> list[Path]:
    """Candidate preset file paths in priority order (highest first).

    project > user > builtin. The builtin layer ships a `default.json`
    so `CreateSessionRequest()` without an explicit preset still has
    something to load on a clean install.
    """
    paths: list[Path] = []
    proj = _project_dir(project_dir)
    if proj is not None:
        paths.append(proj / f"{name}.json")
    paths.append(_user_dir() / f"{name}.json")
    paths.append(_builtin_dir() / f"{name}.json")
    return paths


def load_preset(
    name: str, project_dir: Path | str | None = None,
) -> PresetConfig:
    """Load `<name>.json` from project > user dir.

    Raises FileNotFoundError if no candidate exists.
    """
    for path in _candidate_paths(name, project_dir):
        if not path.exists():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"failed to parse preset at {path}: {exc}") from exc
        preset = PresetConfig.model_validate(raw)
        preset.name = name
        return preset
    candidates = ", ".join(str(p) for p in _candidate_paths(name, project_dir))
    raise FileNotFoundError(
        f"preset {name!r} not found. Looked in: {candidates}"
    )


def list_presets(
    project_dir: Path | str | None = None,
) -> list[str]:
    """Every preset name available from project ∪ user ∪ builtin dirs."""
    names: set[str] = set()
    for dir_path in (_project_dir(project_dir), _user_dir(), _builtin_dir()):
        if dir_path is None or not dir_path.exists():
            continue
        for p in dir_path.glob("*.json"):
            names.add(p.stem)
    return sorted(names)


# ──────────────────────────────────────────────────────────────────────
# Admin helpers (used by the HTTP admin API)
# ──────────────────────────────────────────────────────────────────────


def load_presets_with_origin(
    project_dir: Path | str | None = None,
) -> list[tuple[PresetConfig, str]]:
    """Return every preset merged across builtin/user/project layers,
    annotated with the effective `origin`. Later layers win."""
    ordered: list[tuple[Path, str]] = [
        (_builtin_dir(), "builtin"),
        (_user_dir(), "user"),
    ]
    proj = _project_dir(project_dir)
    if proj is not None:
        ordered.append((proj, "project"))
    merged: dict[str, tuple[PresetConfig, str]] = {}
    for dir_path, origin in ordered:
        if not dir_path.exists():
            continue
        for path in sorted(dir_path.glob("*.json")):
            name = path.stem
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise ValueError(
                    f"failed to parse preset at {path}: {exc}",
                ) from exc
            preset = PresetConfig.model_validate(raw)
            preset.name = name
            merged[name] = (preset, origin)
    return sorted(merged.values(), key=lambda t: t[0].name)


def write_user_preset(name: str, body: dict) -> Path:
    """Write `~/.nature/presets/<name>.json` with validated body.

    Raises ValueError if `body` fails PresetConfig validation so the
    caller can surface it as 400.
    """
    preset = PresetConfig.model_validate(body)  # validate early
    base = _user_dir()
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{name}.json"
    path.write_text(
        json.dumps(preset.model_dump(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def delete_user_preset(name: str) -> bool:
    """Delete `~/.nature/presets/<name>.json`. False when the file
    is absent (caller 403's when a builtin of the same name still
    exists)."""
    path = _user_dir() / f"{name}.json"
    if not path.exists():
        return False
    path.unlink()
    return True


__all__ = [
    "PresetConfig",
    "PresetValidationError",
    "load_preset",
    "list_presets",
    "validate_preset",
    "load_presets_with_origin",
    "write_user_preset",
    "delete_user_preset",
]
