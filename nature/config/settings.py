"""Layered settings system.

5-tier hierarchy (lowest to highest priority):
1. Admin:   /etc/nature/settings.json
2. Flags:   CLI --settings or NATURE_SETTINGS env var
3. Project: .nature/settings.json
4. Local:   .nature/local-settings.json
5. User:    ~/.nature/settings.json

All tiers are merged as raw dicts first, then validated by Pydantic once.
This avoids the problem of partial Pydantic parsing during merge.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class PermissionRules(BaseModel):
    """Permission rules from settings."""
    allow: list[str] = Field(default_factory=list)
    deny: list[str] = Field(default_factory=list)


class NatureSettings(BaseModel):
    """Merged settings from all tiers."""
    # Model
    default_model: str = "claude-sonnet-4-20250514"
    max_output_tokens: int = 8192

    # Per-tier model mapping (agent profiles reference tiers, this maps to actual models)
    model_tiers: dict[str, str] = Field(default_factory=lambda: {
        "heavy": "claude-sonnet-4-20250514",
        "medium": "claude-sonnet-4-20250514",
        "light": "claude-haiku-4-5-20251001",
    })

    # Memory
    auto_memory_enabled: bool = True
    auto_memory_directory: str | None = None

    # Permissions
    permission_mode: str = "default"
    permissions: PermissionRules = Field(default_factory=PermissionRules)

    # Display
    language: str | None = None
    output_style: str | None = None


def _load_json_file(path: Path) -> dict[str, Any]:
    """Load a JSON file, returning empty dict if not found or invalid."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, PermissionError):
        return {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dicts. Override wins for scalar values.
    Lists and dicts are merged recursively.
    """
    result = {**base}
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def get_nature_home() -> Path:
    """Get the nature home directory (~/.nature/)."""
    return Path(os.environ.get("NATURE_HOME", Path.home() / ".nature"))


def load_settings(
    cwd: Path | None = None,
    cli_settings: dict[str, Any] | None = None,
) -> NatureSettings:
    """Load and merge settings from all 5 tiers.

    All tiers are loaded as raw dicts and deep-merged.
    Pydantic validation happens once at the end.
    """
    merged: dict[str, Any] = {}

    # Tier 1: Admin settings
    admin_path = Path("/etc/nature/settings.json")
    merged = _deep_merge(merged, _load_json_file(admin_path))

    # Tier 2: CLI flag / env var settings
    env_settings = os.environ.get("NATURE_SETTINGS")
    if env_settings:
        try:
            merged = _deep_merge(merged, json.loads(env_settings))
        except json.JSONDecodeError:
            env_path = Path(env_settings)
            merged = _deep_merge(merged, _load_json_file(env_path))
    if cli_settings:
        merged = _deep_merge(merged, cli_settings)

    # Tier 3: Project settings
    if cwd:
        project_path = cwd / ".nature" / "settings.json"
        merged = _deep_merge(merged, _load_json_file(project_path))

    # Tier 4: Local settings (not committed)
    if cwd:
        local_path = cwd / ".nature" / "local-settings.json"
        merged = _deep_merge(merged, _load_json_file(local_path))

    # Tier 5: User settings (highest priority)
    user_path = get_nature_home() / "settings.json"
    merged = _deep_merge(merged, _load_json_file(user_path))

    # Validate once — all data is raw dicts at this point
    return NatureSettings.model_validate(merged)
