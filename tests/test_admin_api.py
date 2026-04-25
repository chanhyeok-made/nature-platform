"""Tests for the file-based admin API (Stage B.2b).

Verifies each agents/presets/hosts endpoint reads merged builtin ∪
user ∪ project layers (with origin tagged) and that writes land in
the user layer — builtin entries are read-only (403 on delete).
"""

from __future__ import annotations

import json
import socket
import tempfile
from pathlib import Path

import httpx
import pytest

from nature.server.app import ServerApp


def _free_port_pair() -> int:
    """Find a free pair (port, port+1) for HTTP + WS."""
    for candidate in range(18000, 19000, 2):
        try:
            for port in (candidate, candidate + 1):
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind(("localhost", port))
                s.close()
            return candidate
        except OSError:
            continue
    raise RuntimeError("no free port pair in range")


# ──────────────────────────────────────────────────────────────────────
# Agents
# ──────────────────────────────────────────────────────────────────────


async def test_admin_list_agents_includes_builtins(tmp_path, monkeypatch):
    monkeypatch.setenv("NATURE_HOME", str(tmp_path / "home"))
    port = _free_port_pair()
    app = ServerApp(port=port, cwd=str(tmp_path), event_store_dir=tmp_path / "events")
    await app.start()
    try:
        async with httpx.AsyncClient() as http:
            r = await http.get(f"http://localhost:{port}/api/admin/agents")
            assert r.status_code == 200
            data = r.json()
            names = {a["name"] for a in data["agents"]}
            # Every builtin agent shows up with origin=builtin
            assert {
                "receptionist", "core", "researcher",
                "analyzer", "implementer", "reviewer", "judge",
            }.issubset(names)
            for a in data["agents"]:
                assert a["origin"] in ("builtin", "user", "project")
    finally:
        await app.stop()


async def test_admin_put_and_get_user_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("NATURE_HOME", str(tmp_path / "home"))
    port = _free_port_pair()
    app = ServerApp(port=port, cwd=str(tmp_path), event_store_dir=tmp_path / "events")
    await app.start()
    try:
        async with httpx.AsyncClient() as http:
            put = await http.put(
                f"http://localhost:{port}/api/admin/agents/custom",
                json={
                    "model": "anthropic::claude-haiku-4-5",
                    "allowed_tools": ["Read"],
                    "instructions_text": "you are custom",
                    "description": "a custom agent",
                },
            )
            assert put.status_code == 200

            # Files landed in user layer
            home = tmp_path / "home" / "agents"
            assert (home / "custom.json").exists()
            assert (home / "instructions" / "custom.md").exists()
            assert (
                home / "instructions" / "custom.md"
            ).read_text() == "you are custom"

            # GET returns origin=user
            got = await http.get(
                f"http://localhost:{port}/api/admin/agents/custom"
            )
            assert got.status_code == 200
            body = got.json()
            assert body["origin"] == "user"
            assert body["instructions_text"] == "you are custom"
            assert body["allowed_tools"] == ["Read"]
    finally:
        await app.stop()


async def test_admin_put_user_agent_can_override_builtin(tmp_path, monkeypatch):
    monkeypatch.setenv("NATURE_HOME", str(tmp_path / "home"))
    port = _free_port_pair()
    app = ServerApp(port=port, cwd=str(tmp_path), event_store_dir=tmp_path / "events")
    await app.start()
    try:
        async with httpx.AsyncClient() as http:
            r = await http.put(
                f"http://localhost:{port}/api/admin/agents/receptionist",
                json={
                    "model": "anthropic::claude-sonnet-4-6",
                    "instructions_text": "user override",
                },
            )
            assert r.status_code == 200
            got = await http.get(
                f"http://localhost:{port}/api/admin/agents/receptionist"
            )
            assert got.json()["origin"] == "user"
            assert got.json()["model"] == "anthropic::claude-sonnet-4-6"
    finally:
        await app.stop()


async def test_admin_delete_builtin_agent_returns_403(tmp_path, monkeypatch):
    monkeypatch.setenv("NATURE_HOME", str(tmp_path / "home"))
    port = _free_port_pair()
    app = ServerApp(port=port, cwd=str(tmp_path), event_store_dir=tmp_path / "events")
    await app.start()
    try:
        async with httpx.AsyncClient() as http:
            r = await http.delete(
                f"http://localhost:{port}/api/admin/agents/receptionist"
            )
            assert r.status_code == 403
            assert r.json()["error"] == "read_only_layer"
    finally:
        await app.stop()


async def test_admin_delete_user_agent_succeeds(tmp_path, monkeypatch):
    monkeypatch.setenv("NATURE_HOME", str(tmp_path / "home"))
    port = _free_port_pair()
    app = ServerApp(port=port, cwd=str(tmp_path), event_store_dir=tmp_path / "events")
    await app.start()
    try:
        async with httpx.AsyncClient() as http:
            await http.put(
                f"http://localhost:{port}/api/admin/agents/throwaway",
                json={
                    "model": "anthropic::claude-haiku-4-5",
                    "instructions_text": "...",
                },
            )
            r = await http.delete(
                f"http://localhost:{port}/api/admin/agents/throwaway"
            )
            assert r.status_code == 200
            r2 = await http.get(
                f"http://localhost:{port}/api/admin/agents/throwaway"
            )
            assert r2.status_code == 404
    finally:
        await app.stop()


# ──────────────────────────────────────────────────────────────────────
# Presets
# ──────────────────────────────────────────────────────────────────────


async def test_admin_list_presets_includes_builtin_default(tmp_path, monkeypatch):
    monkeypatch.setenv("NATURE_HOME", str(tmp_path / "home"))
    port = _free_port_pair()
    app = ServerApp(port=port, cwd=str(tmp_path), event_store_dir=tmp_path / "events")
    await app.start()
    try:
        async with httpx.AsyncClient() as http:
            r = await http.get(f"http://localhost:{port}/api/admin/presets")
            assert r.status_code == 200
            entries = r.json()["presets"]
            names = {p["name"] for p in entries}
            assert "default" in names
            default = next(p for p in entries if p["name"] == "default")
            assert default["origin"] == "builtin"
    finally:
        await app.stop()


async def test_admin_put_user_preset_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("NATURE_HOME", str(tmp_path / "home"))
    port = _free_port_pair()
    app = ServerApp(port=port, cwd=str(tmp_path), event_store_dir=tmp_path / "events")
    await app.start()
    try:
        async with httpx.AsyncClient() as http:
            r = await http.put(
                f"http://localhost:{port}/api/admin/presets/exp",
                json={
                    "root_agent": "receptionist",
                    "agents": ["receptionist", "core"],
                    "model_overrides": {
                        "core": "anthropic::claude-haiku-4-5",
                    },
                },
            )
            assert r.status_code == 200

            disk = tmp_path / "home" / "presets" / "exp.json"
            assert disk.exists()

            got = await http.get(
                f"http://localhost:{port}/api/admin/presets/exp"
            )
            body = got.json()
            assert body["origin"] == "user"
            assert body["model_overrides"] == {
                "core": "anthropic::claude-haiku-4-5",
            }
    finally:
        await app.stop()


async def test_admin_put_preset_rejects_invalid_body(tmp_path, monkeypatch):
    monkeypatch.setenv("NATURE_HOME", str(tmp_path / "home"))
    port = _free_port_pair()
    app = ServerApp(port=port, cwd=str(tmp_path), event_store_dir=tmp_path / "events")
    await app.start()
    try:
        async with httpx.AsyncClient() as http:
            # root_agent missing from agents list → validator rejects.
            r = await http.put(
                f"http://localhost:{port}/api/admin/presets/bad",
                json={
                    "root_agent": "ghost",
                    "agents": ["receptionist"],
                },
            )
            assert r.status_code == 400
    finally:
        await app.stop()


async def test_admin_delete_builtin_preset_returns_403(tmp_path, monkeypatch):
    monkeypatch.setenv("NATURE_HOME", str(tmp_path / "home"))
    port = _free_port_pair()
    app = ServerApp(port=port, cwd=str(tmp_path), event_store_dir=tmp_path / "events")
    await app.start()
    try:
        async with httpx.AsyncClient() as http:
            r = await http.delete(
                f"http://localhost:{port}/api/admin/presets/default"
            )
            assert r.status_code == 403
    finally:
        await app.stop()


# ──────────────────────────────────────────────────────────────────────
# Hosts
# ──────────────────────────────────────────────────────────────────────


async def test_admin_list_hosts_includes_builtins(tmp_path, monkeypatch):
    monkeypatch.setenv("NATURE_HOME", str(tmp_path / "home"))
    port = _free_port_pair()
    app = ServerApp(port=port, cwd=str(tmp_path), event_store_dir=tmp_path / "events")
    await app.start()
    try:
        async with httpx.AsyncClient() as http:
            r = await http.get(f"http://localhost:{port}/api/admin/hosts")
            assert r.status_code == 200
            data = r.json()
            names = {h["name"] for h in data["hosts"]}
            # Every builtin host is visible
            assert {
                "anthropic", "openai", "local-ollama", "openrouter",
                "groq", "together",
            }.issubset(names)
            assert data["default_host"]  # non-empty
    finally:
        await app.stop()


async def test_admin_put_user_host_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("NATURE_HOME", str(tmp_path / "home"))
    port = _free_port_pair()
    app = ServerApp(port=port, cwd=str(tmp_path), event_store_dir=tmp_path / "events")
    await app.start()
    try:
        async with httpx.AsyncClient() as http:
            r = await http.put(
                f"http://localhost:{port}/api/admin/hosts/myopenai",
                json={
                    "provider": "openai",
                    "base_url": "https://example.com/v1",
                    "api_key_env": "EXAMPLE_KEY",
                    "models": ["gpt-x"],
                },
            )
            assert r.status_code == 200

            listed = (await http.get(
                f"http://localhost:{port}/api/admin/hosts"
            )).json()
            myopenai = next(h for h in listed["hosts"] if h["name"] == "myopenai")
            assert myopenai["origin"] == "user"
            assert myopenai["base_url"] == "https://example.com/v1"
    finally:
        await app.stop()


async def test_admin_delete_builtin_host_returns_403(tmp_path, monkeypatch):
    monkeypatch.setenv("NATURE_HOME", str(tmp_path / "home"))
    port = _free_port_pair()
    app = ServerApp(port=port, cwd=str(tmp_path), event_store_dir=tmp_path / "events")
    await app.start()
    try:
        async with httpx.AsyncClient() as http:
            r = await http.delete(
                f"http://localhost:{port}/api/admin/hosts/anthropic"
            )
            assert r.status_code == 403
    finally:
        await app.stop()


async def test_admin_delete_user_host_succeeds(tmp_path, monkeypatch):
    monkeypatch.setenv("NATURE_HOME", str(tmp_path / "home"))
    port = _free_port_pair()
    app = ServerApp(port=port, cwd=str(tmp_path), event_store_dir=tmp_path / "events")
    await app.start()
    try:
        async with httpx.AsyncClient() as http:
            await http.put(
                f"http://localhost:{port}/api/admin/hosts/scratch",
                json={"provider": "openai", "models": []},
            )
            r = await http.delete(
                f"http://localhost:{port}/api/admin/hosts/scratch"
            )
            assert r.status_code == 200
    finally:
        await app.stop()
