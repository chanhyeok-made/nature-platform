"""Agent registry — one JSON per agent, instructions in paired MD.

The successor to `nature/agents/profiles.py` + `nature/agents/loader.py`
(which read frontmatter-annotated MD files and maintained a hardcoded
table of builtins). Motivation for the split:

- **One agent per file** — easier to diff, easier to override (user
  drops a single JSON into `~/.nature/agents/` to shadow a builtin).
- **Structured metadata in JSON**, prompt prose in MD — no more
  YAML-frontmatter parser coupling inside the loader. Syntax errors in
  one part don't bleed into the other.
- **`host::model` from day one** — agents reference the host registry
  instead of bare model strings, so an agent's model is self-describing
  (you can read an agent JSON and know exactly which endpoint it'll hit).
- **`allowed_interventions`** — Phase 5's per-agent Pack filter lives
  here alongside `allowed_tools`, preserving the (A)-side / (B)-side
  symmetry of the Pack architecture.

File layout:

    nature/agents/builtin/               # shipped with nature
      <name>.json
      instructions/
        <name>.md                        # paired prompt body

    ~/.nature/agents/                    # user overrides (same layout)
      <name>.json
      instructions/
        <name>.md

Override semantics (Q2-c — decided 2026-04-17): a user `<name>.json`
**completely replaces** the builtin of the same name. No field-level
merge. To customize one field, copy the builtin JSON to the user dir
and edit it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AgentConfig(BaseModel):
    """One agent definition, matched to a `<name>.md` instruction file.

    Fields:
    - `model`       required. `host::model` reference (no bare model strings).
    - `allowed_tools`           null = all tools; [] = none; [...] = specific set.
    - `allowed_interventions`   null = all Packs' interventions; [] = none;
                                [...] = explicit allowlist (supports glob-style
                                wildcards like `"edit_guards.*"`).
    - `instructions`            filename in `instructions/<filename>` relative
                                to the agent JSON's directory (Q2-a: always a
                                filename, never inline prose).
    - `description`             optional human-readable blurb (Q2-e: if null,
                                callers should fall back to the agent's name).

    `name` is the filename stem, set by the loader after reading.
    """

    model_config = ConfigDict(extra="forbid")

    model: str
    allowed_tools: list[str] | None = None
    allowed_interventions: list[str] | None = None
    instructions: str
    description: str | None = None

    # Populated by the loader from the filename; not serialized back.
    name: str = Field(default="", exclude=True)
    # Resolved instruction body (from `instructions/<file>`), set by loader.
    instructions_text: str = Field(default="", exclude=True)

    @field_validator("model")
    @classmethod
    def _check_model_is_host_qualified(cls, v: str) -> str:
        if "::" not in v:
            raise ValueError(
                f"model must be in 'host::model' form (got {v!r}). "
                f"Register the host in hosts.json first, then reference it here."
            )
        host, _, model = v.partition("::")
        if not host or not model:
            raise ValueError(f"malformed host::model reference: {v!r}")
        return v

    def to_role(
        self,
        model_override: str | None = None,
        instructions_override: str | None = None,
    ) -> "AgentRole":
        """Build an AgentRole from this config.

        `model_override` takes the preset's per-agent override when the
        preset supplies one. `instructions_override` likewise takes the
        preset's `prompt_overrides` entry — the caller has already
        resolved the override stem to text (see
        `load_agent_instruction`). The agent's own `instructions_text`
        is untouched; only the returned role carries the swapped body.

        The returned AgentRole carries the bare model name (without
        the `host::` prefix) so downstream code that renders the model
        in events/UI stays host-agnostic. The host lookup is done
        separately by the AreaManager via the session's preset +
        agents_registry.
        """
        from nature.context.types import AgentRole

        model_ref = model_override or self.model
        _, bare_model = model_ref.split("::", 1)
        return AgentRole(
            name=self.name,
            description=self.description or "",
            instructions=instructions_override or self.instructions_text,
            allowed_tools=self.allowed_tools,
            model=bare_model,
        )


class AgentsRegistry(BaseModel):
    """In-memory registry of all loaded agents (builtin ∪ user)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    agents: dict[str, AgentConfig] = Field(default_factory=dict)

    def get(self, name: str) -> AgentConfig | None:
        return self.agents.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self.agents

    def names(self) -> list[str]:
        return sorted(self.agents.keys())


# ──────────────────────────────────────────────────────────────────────
# Loader
# ──────────────────────────────────────────────────────────────────────


def _builtin_dir() -> Path:
    """Directory of shipped agent JSONs (inside the nature package)."""
    return Path(__file__).parent / "builtin"


def _user_dir() -> Path:
    """User-level agent directory (`~/.nature/agents/`)."""
    from nature.config.settings import get_nature_home
    return get_nature_home() / "agents"


def _project_dir(project_dir: Path | str | None) -> Path | None:
    """Project-level agent directory (`<project>/.nature/agents/`).

    None when no project dir was supplied — loader skips this layer.
    """
    if project_dir is None:
        return None
    return Path(project_dir) / ".nature" / "agents"


def _load_from_dir(dir_path: Path) -> dict[str, AgentConfig]:
    """Load every `<name>.json` in a directory, pairing with
    `instructions/<name>.md` from the same directory.

    Returns {} if the directory doesn't exist. Agents with a missing
    instruction file raise ValueError — the loader does not silently
    substitute an empty prompt.
    """
    agents: dict[str, AgentConfig] = {}
    if not dir_path.exists():
        return agents
    instructions_dir = dir_path / "instructions"

    for json_path in sorted(dir_path.glob("*.json")):
        name = json_path.stem
        try:
            raw = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"failed to parse {json_path}: {exc}") from exc

        cfg = AgentConfig.model_validate(raw)
        cfg.name = name

        instr_path = instructions_dir / cfg.instructions
        if not instr_path.exists():
            raise ValueError(
                f"agent {name!r} at {json_path} references "
                f"instructions {cfg.instructions!r} but {instr_path} does not exist"
            )
        cfg.instructions_text = instr_path.read_text(encoding="utf-8")
        agents[name] = cfg

    return agents


def load_agents_registry(
    project_dir: Path | str | None = None,
) -> AgentsRegistry:
    """Layered loader: builtin → user → project (ascending priority).

    Later layers completely replace earlier layers for the same agent
    name (Q2-c). To customize just one field of a builtin, copy its
    JSON to the higher-priority directory and edit.
    """
    merged = _load_from_dir(_builtin_dir())
    merged.update(_load_from_dir(_user_dir()))
    proj = _project_dir(project_dir)
    if proj is not None:
        merged.update(_load_from_dir(proj))
    return AgentsRegistry(agents=merged)


def load_agent_instruction(
    stem: str, project_dir: Path | str | None = None,
) -> str:
    """Read `instructions/<stem>.md` from project > user > builtin.

    Used by preset `prompt_overrides` to swap an agent's prompt body
    without touching its other fields (model, tools, description). The
    file stem is validated here — no path separators allowed.

    Raises FileNotFoundError with every candidate path when none exist.
    """
    if "/" in stem or "\\" in stem or "." in stem:
        raise ValueError(
            f"instruction stem {stem!r} must be a bare filename "
            f"(no separators or extensions)"
        )
    candidates: list[Path] = []
    proj = _project_dir(project_dir)
    if proj is not None:
        candidates.append(proj / "instructions" / f"{stem}.md")
    candidates.append(_user_dir() / "instructions" / f"{stem}.md")
    candidates.append(_builtin_dir() / "instructions" / f"{stem}.md")
    for path in candidates:
        if path.exists():
            return path.read_text(encoding="utf-8")
    looked = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"instruction file for stem {stem!r} not found. Looked in: {looked}"
    )


# ──────────────────────────────────────────────────────────────────────
# Admin helpers (used by the HTTP admin API)
# ──────────────────────────────────────────────────────────────────────


def load_agents_with_origin(
    project_dir: Path | str | None = None,
) -> list[tuple[AgentConfig, str]]:
    """Like `load_agents_registry` but annotates each entry with the
    layer it came from. project > user > builtin wins per name."""
    ordered: list[tuple[dict[str, AgentConfig], str]] = [
        (_load_from_dir(_builtin_dir()), "builtin"),
        (_load_from_dir(_user_dir()), "user"),
    ]
    proj = _project_dir(project_dir)
    if proj is not None:
        ordered.append((_load_from_dir(proj), "project"))
    merged: dict[str, tuple[AgentConfig, str]] = {}
    for batch, origin in ordered:
        for name, cfg in batch.items():
            merged[name] = (cfg, origin)
    return sorted(merged.values(), key=lambda t: t[0].name)


def write_user_agent(
    name: str,
    *,
    model: str,
    allowed_tools: list[str] | None,
    allowed_interventions: list[str] | None,
    instructions_text: str,
    description: str | None,
) -> Path:
    """Write `<name>.json` + `instructions/<name>.md` into the user
    layer, creating the directory if needed. Returns the JSON path.

    The instruction file is always named `<name>.md` so the JSON's
    `instructions` field can be generated deterministically — users
    who want multiple instruction variants for the same role should
    create differently-named agents instead.
    """
    cfg = AgentConfig(
        model=model,
        allowed_tools=allowed_tools,
        allowed_interventions=allowed_interventions,
        instructions=f"{name}.md",
        description=description,
    )
    base = _user_dir()
    base.mkdir(parents=True, exist_ok=True)
    (base / "instructions").mkdir(parents=True, exist_ok=True)
    json_path = base / f"{name}.json"
    json_path.write_text(
        json.dumps(
            cfg.model_dump(exclude_none=False), indent=2, ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    (base / "instructions" / f"{name}.md").write_text(
        instructions_text, encoding="utf-8",
    )
    return json_path


def delete_user_agent(name: str) -> bool:
    """Delete `<name>.json` + `instructions/<name>.md` from the user
    layer. Returns False if the JSON file is absent (caller should
    surface this as 404/403 depending on whether a builtin still
    exists); returns True on successful delete.
    """
    base = _user_dir()
    json_path = base / f"{name}.json"
    if not json_path.exists():
        return False
    json_path.unlink()
    md_path = base / "instructions" / f"{name}.md"
    if md_path.exists():
        md_path.unlink()
    return True


__all__ = [
    "AgentConfig",
    "AgentsRegistry",
    "load_agents_registry",
    "load_agents_with_origin",
    "write_user_agent",
    "delete_user_agent",
]
