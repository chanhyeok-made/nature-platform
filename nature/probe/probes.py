"""Probe schema + builtin loader.

Each probe file is a JSON document matching the Probe model below.
Probes are namespaced by id (kebab-case), sit under
`nature/probe/builtin/probes/<id>.json`, and are loaded by
`load_probes()` which layers builtin → user → project the same way
`nature.eval.tasks` and `nature.agents.presets` do.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ──────────────────────────────────────────────────────────────────────
# Workspace (optional — some probes need files laid down before prompt)
# ──────────────────────────────────────────────────────────────────────


class ProbeFile(BaseModel):
    """One file to materialize in the probe's working directory."""

    model_config = ConfigDict(extra="forbid")

    path: str  # relative to workspace root
    content: str


class ProbeWorkspace(BaseModel):
    """Files + env state to set up before handing the prompt to the
    target model. Optional — probes that are pure "emit a tool_use
    with these args" don't need a workspace."""

    model_config = ConfigDict(extra="forbid")

    files: list[ProbeFile] = Field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────
# Success criteria — each probe declares a list; all must pass
# ──────────────────────────────────────────────────────────────────────


class ExpectToolUse(BaseModel):
    """Assert that the Nth tool_use block (0-indexed) matches the
    given name and — optionally — that its input contains the given
    fields / regex patterns. `input_regex` is keyed by JSON field
    name; each regex is searched against the stringified field value.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["tool_use"] = "tool_use"
    at_index: int = 0
    name: str
    input_contains: dict[str, Any] = Field(default_factory=dict)
    input_regex: dict[str, str] = Field(default_factory=dict)


class ExpectFinalText(BaseModel):
    """Assert the model's final assistant text block (after all tool
    loops) matches a regex or substring."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["final_text"] = "final_text"
    regex: str | None = None
    contains: str | None = None


class ExpectFinalJson(BaseModel):
    """Assert the final text is valid JSON and (optionally) contains
    specific keys / values."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["final_json"] = "final_json"
    required_keys: list[str] = Field(default_factory=list)
    key_equals: dict[str, Any] = Field(default_factory=dict)


class ExpectFileState(BaseModel):
    """Assert a workspace file matches the given regex / substring
    after the probe's session completes. Used for Edit / Write
    probes where the test is "did the file actually change."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["file_state"] = "file_state"
    path: str
    regex: str | None = None
    contains: str | None = None
    equals: str | None = None


class ExpectTurnBound(BaseModel):
    """Assert the session completed within N turns without hitting
    the max_turns ceiling. Catches "model looped" failures."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["turn_bound"] = "turn_bound"
    max_turns: int


class ExpectNoToolError(BaseModel):
    """Assert no tool call returned an error result. The model can
    still emit tool_uses; they just can't fail."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["no_tool_error"] = "no_tool_error"


class ExpectToolNotUsed(BaseModel):
    """Assert a specific tool was NOT called during the session.
    Paired with an ExpectToolUse elsewhere, this lets a probe verify
    the model took the delegated path (e.g., emitted Agent) without
    cheating by calling the forbidden tool (e.g., Read) directly."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["tool_not_used"] = "tool_not_used"
    name: str


SuccessCriterion = (
    ExpectToolUse
    | ExpectFinalText
    | ExpectFinalJson
    | ExpectFileState
    | ExpectTurnBound
    | ExpectNoToolError
    | ExpectToolNotUsed
)


# ──────────────────────────────────────────────────────────────────────
# Probe
# ──────────────────────────────────────────────────────────────────────


class Probe(BaseModel):
    """One capability test. Tier + dimensions drive the tier-map;
    workspace + prompt + allowed_tools define what the model sees;
    success criteria define pass/fail."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    tier: int = Field(ge=0, le=9)
    dimensions: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    workspace: ProbeWorkspace | None = None
    system: str | None = None  # optional system prompt; default is a minimal one
    prompt: str
    success: list[SuccessCriterion] = Field(default_factory=list)
    timeout_sec: float = 90.0
    max_turns: int = 5

    @field_validator("id")
    @classmethod
    def _id_is_kebab(cls, v: str) -> str:
        if not v or " " in v or "/" in v:
            raise ValueError(f"probe id {v!r} must be kebab-case, no slashes/spaces")
        return v

    @field_validator("tier")
    @classmethod
    def _tier_range(cls, v: int) -> int:
        if not (0 <= v <= 9):
            raise ValueError("tier must be 0-9")
        return v


# ──────────────────────────────────────────────────────────────────────
# Loader
# ──────────────────────────────────────────────────────────────────────


_BUILTIN_DIR = Path(__file__).parent / "builtin" / "probes"


def _load_dir(directory: Path) -> dict[str, Probe]:
    out: dict[str, Probe] = {}
    if not directory.exists():
        return out
    for p in sorted(directory.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {p}: {exc}") from exc
        # Default id from filename stem so probes don't have to repeat it.
        data.setdefault("id", p.stem)
        probe = Probe.model_validate(data)
        out[probe.id] = probe
    return out


def load_probes(project_dir: Path | None = None) -> dict[str, Probe]:
    """Builtin → user (~/.nature/probes/) → project (<root>/.nature/probes/).

    Later layers override earlier ones by id, matching eval.tasks
    and agents.presets conventions."""
    merged: dict[str, Probe] = {}
    merged.update(_load_dir(_BUILTIN_DIR))

    user_dir = Path.home() / ".nature" / "probes"
    merged.update(_load_dir(user_dir))

    if project_dir is not None:
        merged.update(_load_dir(Path(project_dir) / ".nature" / "probes"))

    return merged


__all__ = [
    "Probe",
    "ProbeFile",
    "ProbeWorkspace",
    "ExpectToolUse",
    "ExpectFinalText",
    "ExpectFinalJson",
    "ExpectFileState",
    "ExpectTurnBound",
    "ExpectNoToolError",
    "ExpectToolNotUsed",
    "SuccessCriterion",
    "load_probes",
]
