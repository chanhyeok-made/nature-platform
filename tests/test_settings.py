"""Tests for settings loading."""

import json
from pathlib import Path

from nature.config.settings import NatureSettings, get_nature_home, load_settings


class TestNatureSettings:
    def test_defaults(self):
        settings = NatureSettings()
        assert settings.default_model == "claude-sonnet-4-20250514"
        assert settings.auto_memory_enabled is True
        assert settings.permission_mode == "default"

    def test_load_settings_no_files(self, tmp_path: Path):
        settings = load_settings(cwd=tmp_path)
        assert isinstance(settings, NatureSettings)
        assert settings.default_model == "claude-sonnet-4-20250514"

    def test_load_project_settings(self, tmp_path: Path):
        nature_dir = tmp_path / ".nature"
        nature_dir.mkdir()
        settings_file = nature_dir / "settings.json"
        settings_file.write_text(json.dumps({"default_model": "claude-opus-4-20250514"}))

        settings = load_settings(cwd=tmp_path)
        assert settings.default_model == "claude-opus-4-20250514"

    def test_cli_settings_override(self, tmp_path: Path):
        settings = load_settings(
            cwd=tmp_path,
            cli_settings={"default_model": "claude-haiku-4-5-20251001"},
        )
        # User settings (tier 5) would override, but since there are none:
        assert settings.default_model == "claude-haiku-4-5-20251001"


class TestGetNatureHome:
    def test_default_path(self):
        home = get_nature_home()
        assert home.name == ".nature"

    def test_env_override(self, monkeypatch: object, tmp_path: Path):
        import pytest
        monkeypatch.setenv("NATURE_HOME", str(tmp_path / "custom"))  # type: ignore
        home = get_nature_home()
        assert home == tmp_path / "custom"
