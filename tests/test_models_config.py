"""Tests for the per-model spec registry (nature.config.models).

The registry powers TokenBudget resolution for body compaction: given
a `host::model` ref, return the right context window + output
reservation. Builtin ships pragmatic defaults; user and project layers
can override or add entries.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nature.config.models import (
    ModelSpec,
    ModelSpecs,
    load_model_specs,
    resolve_budget,
)
from nature.protocols.context import TokenBudget


# ──────────────────────────────────────────────────────────────────────
# Schema
# ──────────────────────────────────────────────────────────────────────


def test_model_spec_rejects_unknown_fields():
    with pytest.raises(Exception):
        ModelSpec.model_validate({
            "context_window": 100000,
            "output_reservation": 10000,
            "extra_field": "oops",
        })


def test_model_spec_requires_context_and_reservation():
    with pytest.raises(Exception):
        ModelSpec.model_validate({"context_window": 100000})


# ──────────────────────────────────────────────────────────────────────
# Loader — builtin + layering
# ──────────────────────────────────────────────────────────────────────


def test_builtin_ships_anthropic_and_local_entries():
    specs = load_model_specs()
    assert specs.get("anthropic::claude-haiku-4-5") is not None
    assert specs.get("local-ollama::qwen2.5-coder:32b") is not None


def test_qwen_has_smaller_context_than_haiku():
    specs = load_model_specs()
    haiku = specs.get("anthropic::claude-haiku-4-5")
    qwen = specs.get("local-ollama::qwen2.5-coder:32b")
    assert haiku is not None and qwen is not None
    assert qwen.context_window < haiku.context_window


def _write_json(path: Path, body: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body), encoding="utf-8")


def test_project_layer_overrides_builtin(tmp_path, monkeypatch):
    monkeypatch.setenv("NATURE_HOME", str(tmp_path / "empty-home"))
    _write_json(
        tmp_path / ".nature" / "models.json",
        {
            "anthropic::claude-haiku-4-5": {
                "context_window": 42,
                "output_reservation": 4,
                "label": "overridden",
            },
        },
    )
    specs = load_model_specs(project_dir=str(tmp_path))
    haiku = specs.get("anthropic::claude-haiku-4-5")
    assert haiku is not None
    assert haiku.context_window == 42
    assert haiku.label == "overridden"


def test_user_layer_overrides_builtin_and_is_overridden_by_project(
    tmp_path, monkeypatch,
):
    user_home = tmp_path / "home-nature"
    monkeypatch.setenv("NATURE_HOME", str(user_home))
    _write_json(
        user_home / "models.json",
        {
            "anthropic::claude-haiku-4-5": {
                "context_window": 111, "output_reservation": 1,
            },
        },
    )
    # No project layer — user value should win over builtin.
    specs = load_model_specs(project_dir=None)
    assert specs.get("anthropic::claude-haiku-4-5").context_window == 111

    # Project layer trumps user.
    _write_json(
        tmp_path / "proj" / ".nature" / "models.json",
        {
            "anthropic::claude-haiku-4-5": {
                "context_window": 999, "output_reservation": 9,
            },
        },
    )
    specs2 = load_model_specs(project_dir=str(tmp_path / "proj"))
    assert specs2.get("anthropic::claude-haiku-4-5").context_window == 999


def test_load_models_raises_on_malformed_json(tmp_path, monkeypatch):
    monkeypatch.setenv("NATURE_HOME", str(tmp_path / "home"))
    user_path = tmp_path / "home" / "models.json"
    user_path.parent.mkdir(parents=True)
    user_path.write_text("{ this is not json", encoding="utf-8")
    with pytest.raises(ValueError, match="failed to parse"):
        load_model_specs()


def test_load_models_raises_when_root_is_not_object(tmp_path, monkeypatch):
    monkeypatch.setenv("NATURE_HOME", str(tmp_path / "home"))
    user_path = tmp_path / "home" / "models.json"
    user_path.parent.mkdir(parents=True)
    user_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_model_specs()


# ──────────────────────────────────────────────────────────────────────
# resolve_budget
# ──────────────────────────────────────────────────────────────────────


def test_resolve_budget_known_model_uses_spec():
    specs = ModelSpecs(specs={
        "h::m": ModelSpec(context_window=32000, output_reservation=4000),
    })
    b = resolve_budget("h::m", specs)
    assert b.context_window == 32000
    assert b.output_reservation == 4000


def test_resolve_budget_unknown_model_falls_back_to_default():
    specs = ModelSpecs(specs={})
    b = resolve_budget("ghost::none", specs)
    # Vanilla TokenBudget defaults (200k window, 20k reservation)
    assert b.context_window == 200_000
    assert b.output_reservation == 20_000


def test_resolve_budget_unknown_model_honors_explicit_fallback():
    specs = ModelSpecs(specs={})
    custom = TokenBudget(context_window=12345, output_reservation=123)
    b = resolve_budget("ghost::none", specs, fallback=custom)
    assert b.context_window == 12345
    assert b.output_reservation == 123
