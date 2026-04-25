"""Model registry — per-model specs (context window, output reservation).

The runtime needs two pieces of per-model info that aren't captured by
the agent/preset/host triple:

1. `context_window` — the model's hard context limit. Determines the
   autocompact threshold for body compaction.
2. `output_reservation` — how much of the window we reserve for the
   model's reply. The autocompact threshold is computed as
   `context_window - output_reservation - autocompact_buffer`.

These live in `models.json` rather than inline in each agent config
because (a) the same model is used by many agents and we don't want to
repeat the spec, and (b) when a new model is released, adding one
entry to `models.json` makes it available everywhere.

File layout (three-layer, same pattern as agents/hosts/presets):

    <package>/config/builtin/models.json   # shipped
    ~/.nature/models.json                  # user-level override
    <project>/.nature/models.json          # project-level override

Keys are `host::model` strings — the same form used in `hosts.json`,
preset `model_overrides`, and agent `model` fields. Unknown models
fall back to a conservative default (`TokenBudget()` zeros, ~167 k
autocompact).
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from nature.protocols.context import TokenBudget


class ModelSpec(BaseModel):
    """Per-model hard specs consumed by the runtime."""

    model_config = ConfigDict(extra="forbid")

    context_window: int
    output_reservation: int
    family: str = ""
    tier: str = ""
    label: str = ""


class ModelSpecs(BaseModel):
    """Map of `host::model` ref → ModelSpec."""

    model_config = ConfigDict(extra="forbid")

    specs: dict[str, ModelSpec] = Field(default_factory=dict)

    def get(self, model_ref: str) -> ModelSpec | None:
        return self.specs.get(model_ref)


# ──────────────────────────────────────────────────────────────────────
# Loader — layered builtin → user → project
# ──────────────────────────────────────────────────────────────────────


def _builtin_path() -> Path:
    return Path(__file__).parent / "builtin" / "models.json"


def _user_path() -> Path:
    from nature.config.settings import get_nature_home
    return get_nature_home() / "models.json"


def _project_path(project_dir: Path | str | None) -> Path | None:
    if project_dir is None:
        return None
    return Path(project_dir) / ".nature" / "models.json"


def _load_one(path: Path) -> dict[str, ModelSpec]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"failed to parse models config at {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(
            f"{path} must be a JSON object mapping model refs to specs"
        )
    return {k: ModelSpec.model_validate(v) for k, v in raw.items()}


def load_model_specs(project_dir: Path | str | None = None) -> ModelSpecs:
    """Return merged model specs (builtin → user → project).

    Later layers completely replace earlier layers for the same key.
    Missing files at any layer are ignored.
    """
    merged: dict[str, ModelSpec] = {}
    merged.update(_load_one(_builtin_path()))
    merged.update(_load_one(_user_path()))
    proj = _project_path(project_dir)
    if proj is not None:
        merged.update(_load_one(proj))
    return ModelSpecs(specs=merged)


# ──────────────────────────────────────────────────────────────────────
# TokenBudget factory
# ──────────────────────────────────────────────────────────────────────


def resolve_budget(
    model_ref: str, specs: ModelSpecs, *, fallback: TokenBudget | None = None,
) -> TokenBudget:
    """Return a TokenBudget sized for `model_ref`.

    Falls back to `fallback` (or a vanilla `TokenBudget()`) when the
    model is unknown — the caller sees the same shape regardless, so
    the compaction pipeline keeps working on unseen models.
    """
    spec = specs.get(model_ref)
    if spec is None:
        return fallback if fallback is not None else TokenBudget()
    return TokenBudget(
        context_window=spec.context_window,
        output_reservation=spec.output_reservation,
    )


__all__ = [
    "ModelSpec",
    "ModelSpecs",
    "load_model_specs",
    "resolve_budget",
]
