"""End-to-end tests for nature/server + nature/client.

Spins up a real ServerApp on localhost (random ephemeral ports), drives
it via NatureClient, and verifies the wire protocol on both sides. No
real LLM calls — provider construction needs an env API key but tests
patch ANTHROPIC_API_KEY before running.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import tempfile
from pathlib import Path

import pytest

from nature.client import NatureClient, NatureClientError, ServerNotRunning
from nature.events.types import EventType
from nature.server.api import (
    CreateSessionRequest,
    SessionInfo,
)
from nature.server.app import ServerApp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _free_port_pair() -> int:
    """Return an HTTP port whose port and port+1 are both free."""
    for candidate in range(19500, 19700, 2):
        try:
            s1 = socket.socket()
            s2 = socket.socket()
            try:
                s1.bind(("localhost", candidate))
                s2.bind(("localhost", candidate + 1))
                return candidate
            finally:
                s1.close()
                s2.close()
        except OSError:
            continue
    raise RuntimeError("no free port pair")


@pytest.fixture(autouse=True)
def _set_fake_anthropic_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")


# ---------------------------------------------------------------------------
# api models
# ---------------------------------------------------------------------------


def test_create_session_request_defaults():
    req = CreateSessionRequest()
    assert req.preset is None  # None → server looks up `default.json`


def test_session_info_serializes_round_trip():
    info = SessionInfo(
        session_id="s1",
        root_role_name="receptionist",
        root_model="claude-sonnet-4",
        state="active",
        has_active_run=False,
        created_at=1.0,
    )
    data = info.model_dump()
    restored = SessionInfo(**data)
    assert restored == info


# ---------------------------------------------------------------------------
# ServerApp lifecycle
# ---------------------------------------------------------------------------


async def test_server_starts_and_stops_cleanly():
    with tempfile.TemporaryDirectory() as tmp:
        app = ServerApp(
            port=_free_port_pair(),
            cwd=tmp,
            event_store_dir=Path(tmp) / "events",
        )
        ok = await app.start()
        assert ok is True
        assert app.started is True
        await app.stop()
        assert app.started is False


# ---------------------------------------------------------------------------
# REST API end-to-end via NatureClient
# ---------------------------------------------------------------------------


async def test_create_session_via_client_returns_metadata(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    with tempfile.TemporaryDirectory() as tmp:
        port = _free_port_pair()
        app = ServerApp(
            port=port, cwd=tmp,
            event_store_dir=Path(tmp) / "events",
        )
        await app.start()
        try:
            async with NatureClient(port=port) as client:
                assert await client.is_alive()

                # No preset arg → server resolves to builtin default.json
                # (receptionist root on anthropic::claude-haiku-4-5).
                created = await client.create_session()
                assert created.session_id
                assert created.root_role_name == "receptionist"
                assert created.root_model == "claude-haiku-4-5"
                assert created.provider_name == "anthropic"
        finally:
            await app.stop()


async def test_list_sessions_reflects_creates_and_deletes():
    with tempfile.TemporaryDirectory() as tmp:
        port = _free_port_pair()
        app = ServerApp(port=port, cwd=tmp, event_store_dir=Path(tmp) / "events")
        await app.start()
        try:
            async with NatureClient(port=port) as client:
                assert await client.list_sessions() == []

                a = await client.create_session()
                b = await client.create_session()
                listed = await client.list_sessions()
                ids = {s.session_id for s in listed}
                assert {a.session_id, b.session_id} <= ids

                await client.close_session(a.session_id)
                listed = await client.list_sessions()
                ids = {s.session_id for s in listed}
                assert a.session_id not in ids
                assert b.session_id in ids
        finally:
            await app.stop()


async def test_get_session_returns_404_for_unknown_id():
    with tempfile.TemporaryDirectory() as tmp:
        port = _free_port_pair()
        app = ServerApp(port=port, cwd=tmp, event_store_dir=Path(tmp) / "events")
        await app.start()
        try:
            async with NatureClient(port=port) as client:
                with pytest.raises(NatureClientError, match="not_found"):
                    await client.get_session("nonexistent")
        finally:
            await app.stop()


async def test_snapshot_returns_session_started_for_fresh_session():
    with tempfile.TemporaryDirectory() as tmp:
        port = _free_port_pair()
        app = ServerApp(port=port, cwd=tmp, event_store_dir=Path(tmp) / "events")
        await app.start()
        try:
            async with NatureClient(port=port) as client:
                created = await client.create_session()
                events = await client.snapshot(created.session_id)
                # Session creation emits SESSION_STARTED as event #1.
                assert len(events) == 1
                assert events[0].type == EventType.SESSION_STARTED
                assert events[0].payload["preset_name"]
        finally:
            await app.stop()


# ---------------------------------------------------------------------------
# WebSocket event stream
# ---------------------------------------------------------------------------


async def test_stream_events_yields_session_meta_then_filters_it_out():
    """The client filters session_meta from the user-visible stream."""
    with tempfile.TemporaryDirectory() as tmp:
        port = _free_port_pair()
        app = ServerApp(port=port, cwd=tmp, event_store_dir=Path(tmp) / "events")
        await app.start()
        try:
            async with NatureClient(port=port) as client:
                created = await client.create_session()
                sid = created.session_id

                # Inject a fake event directly into the store so we have
                # something to receive without making a real LLM call
                from nature.events.types import Event
                import time as _time
                app.registry.event_store.append(Event(
                    id=0,
                    session_id=sid,
                    frame_id="f1",
                    timestamp=_time.time(),
                    type=EventType.FRAME_OPENED,
                    payload={"purpose": "root", "role_name": "receptionist"},
                ))

                seen = []
                async def consume():
                    async for ev in client.stream_events(sid):
                        seen.append(ev)
                        if ev.type == EventType.FRAME_OPENED:
                            return

                await asyncio.wait_for(consume(), timeout=2.0)
                # Session creation emits SESSION_STARTED (event #1) before
                # the injected FRAME_OPENED lands, so the client sees both.
                assert [ev.type for ev in seen] == [
                    EventType.SESSION_STARTED, EventType.FRAME_OPENED,
                ]
                fo = seen[-1]
                assert fo.payload["role_name"] == "receptionist"
        finally:
            await app.stop()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


async def test_client_raises_server_not_running_when_no_server():
    async with NatureClient(host="localhost", port=19999) as client:
        with pytest.raises(ServerNotRunning):
            await client.list_sessions()


async def test_send_message_to_unknown_session_returns_404():
    with tempfile.TemporaryDirectory() as tmp:
        port = _free_port_pair()
        app = ServerApp(port=port, cwd=tmp, event_store_dir=Path(tmp) / "events")
        await app.start()
        try:
            async with NatureClient(port=port) as client:
                with pytest.raises(NatureClientError, match="not_found"):
                    await client.send_message("ghost", "hi")
        finally:
            await app.stop()


# ---------------------------------------------------------------------------
# Dashboard HTML route
# ---------------------------------------------------------------------------


async def test_get_frame_context_returns_full_context():
    """The new /api/sessions/{id}/frames/{fid}/context endpoint."""
    with tempfile.TemporaryDirectory() as tmp:
        port = _free_port_pair()
        app = ServerApp(port=port, cwd=tmp, event_store_dir=Path(tmp) / "events")
        await app.start()
        try:
            async with NatureClient(port=port) as client:
                created = await client.create_session()
                sid = created.session_id

                # Inject events to simulate a frame.opened with role +
                # one message_appended so reconstruct() has something to rebuild
                from nature.events.types import Event
                import time as _time
                store = app.registry.event_store
                store.append(Event(
                    id=0, session_id=sid, frame_id="frame_test_root",
                    timestamp=_time.time(),
                    type=EventType.FRAME_OPENED,
                    payload={
                        "purpose": "root",
                        "parent_id": None,
                        "role_name": "receptionist",
                        "role_description": "the boss",
                        "instructions": "be brief and helpful",
                        "allowed_tools": None,
                        "model": "fake-model",
                    },
                ))
                store.append(Event(
                    id=0, session_id=sid, frame_id="frame_test_root",
                    timestamp=_time.time(),
                    type=EventType.MESSAGE_APPENDED,
                    payload={
                        "message_id": "msg_1",
                        "from_": "user",
                        "to": "receptionist",
                        "content": [{"type": "text", "text": "hi there"}],
                        "timestamp": _time.time(),
                        "delegations": {},
                    },
                ))

                ctx = await client.get_frame_context(sid, "frame_test_root")
                assert ctx is not None
                assert ctx["frame_id"] == "frame_test_root"
                assert ctx["purpose"] == "root"

                header = ctx["header"]
                assert header["role"]["name"] == "receptionist"
                assert header["role"]["instructions"] == "be brief and helpful"
                assert header["principles"] == []

                messages = ctx["body"]["conversation"]["messages"]
                assert len(messages) == 1
                assert messages[0]["from_"] == "user"
                assert messages[0]["to"] == "receptionist"
                assert messages[0]["content"][0]["text"] == "hi there"
        finally:
            await app.stop()


async def test_get_frame_context_returns_none_for_unknown_frame():
    with tempfile.TemporaryDirectory() as tmp:
        port = _free_port_pair()
        app = ServerApp(port=port, cwd=tmp, event_store_dir=Path(tmp) / "events")
        await app.start()
        try:
            async with NatureClient(port=port) as client:
                created = await client.create_session()
                ctx = await client.get_frame_context(
                    created.session_id, "no_such_frame"
                )
                assert ctx is None
        finally:
            await app.stop()


async def test_archived_session_preview_extracted_from_first_user_message():
    """Archived sessions list returns a single-line preview of the first
    user message, so users can identify sessions in a picker."""
    with tempfile.TemporaryDirectory() as tmp:
        port = _free_port_pair()
        app = ServerApp(port=port, cwd=tmp, event_store_dir=Path(tmp) / "events")
        await app.start()
        try:
            from nature.events.types import Event
            import time as _time
            sid = "abandoned_with_preview"
            store = app.registry.event_store

            store.append(Event(
                id=0, session_id=sid, frame_id="frame_x",
                timestamp=_time.time(),
                type=EventType.FRAME_OPENED,
                payload={"role_name": "receptionist"},
            ))
            store.append(Event(
                id=0, session_id=sid, frame_id="frame_x",
                timestamp=_time.time(),
                type=EventType.MESSAGE_APPENDED,
                payload={
                    "from_": "user", "to": "receptionist",
                    "content": [{
                        "type": "text",
                        "text": "이 프로젝트의 동작 방식을\n시각화해줘 — 시퀀스 다이어그램으로",
                    }],
                    "timestamp": _time.time(),
                },
            ))
            store.append(Event(
                id=0, session_id=sid, frame_id="frame_x",
                timestamp=_time.time(),
                type=EventType.MESSAGE_APPENDED,
                payload={
                    "from_": "receptionist", "to": "user",
                    "content": [{"type": "text", "text": "ok"}],
                    "timestamp": _time.time(),
                },
            ))

            async with NatureClient(port=port) as client:
                archived = await client.list_archived_sessions()
                target = next(a for a in archived if a.session_id == sid)
                # Newlines collapsed to a single space, no truncation here
                assert "이 프로젝트의 동작 방식을" in target.preview
                assert "시각화해줘" in target.preview
                assert "\n" not in target.preview
        finally:
            await app.stop()


async def test_archived_preview_truncates_long_first_message():
    with tempfile.TemporaryDirectory() as tmp:
        port = _free_port_pair()
        app = ServerApp(port=port, cwd=tmp, event_store_dir=Path(tmp) / "events")
        await app.start()
        try:
            from nature.events.types import Event
            import time as _time
            sid = "abandoned_long"
            long_text = "x" * 200
            store = app.registry.event_store
            store.append(Event(
                id=0, session_id=sid, frame_id="f",
                timestamp=_time.time(), type=EventType.FRAME_OPENED,
                payload={"role_name": "r"},
            ))
            store.append(Event(
                id=0, session_id=sid, frame_id="f",
                timestamp=_time.time(), type=EventType.MESSAGE_APPENDED,
                payload={
                    "from_": "user", "to": "r",
                    "content": [{"type": "text", "text": long_text}],
                    "timestamp": _time.time(),
                },
            ))

            async with NatureClient(port=port) as client:
                archived = await client.list_archived_sessions()
                target = next(a for a in archived if a.session_id == sid)
                # Truncated to 80 chars (79 + ellipsis)
                assert len(target.preview) <= 80
                assert target.preview.endswith("…")
        finally:
            await app.stop()


async def test_resume_existing_live_session_returns_existing():
    """Resume a session that's still in the registry → same instance."""
    with tempfile.TemporaryDirectory() as tmp:
        port = _free_port_pair()
        app = ServerApp(port=port, cwd=tmp, event_store_dir=Path(tmp) / "events")
        await app.start()
        try:
            async with NatureClient(port=port) as client:
                created = await client.create_session()
                resumed = await client.resume_session(created.session_id)
                assert resumed.session_id == created.session_id
                assert resumed.root_role_name == created.root_role_name
                # Still registered (resume didn't remove it)
                listed = await client.list_sessions()
                assert any(
                    s.session_id == created.session_id for s in listed
                )
        finally:
            await app.stop()


async def test_resume_archived_session_hydrates_from_event_log(monkeypatch):
    """Sessions whose only trace is the event log can be resumed."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    with tempfile.TemporaryDirectory() as tmp:
        port = _free_port_pair()
        app = ServerApp(port=port, cwd=tmp, event_store_dir=Path(tmp) / "events")
        await app.start()
        try:
            # Pre-seed the event store with a frame that has no live
            # registration. Mimics a session that survived a server restart.
            from nature.events.types import Event
            import time as _time
            archived_sid = "abandoned_session_42"
            store = app.registry.event_store
            store.append(Event(
                id=0, session_id=archived_sid, frame_id="frame_orphan",
                timestamp=_time.time(),
                type=EventType.FRAME_OPENED,
                payload={
                    "purpose": "root",
                    "parent_id": None,
                    "role_name": "receptionist",
                    "role_description": "the boss",
                    "instructions": "be brief",
                    "allowed_tools": None,
                    "model": "claude-sonnet-4-20250514",
                },
            ))
            store.append(Event(
                id=0, session_id=archived_sid, frame_id="frame_orphan",
                timestamp=_time.time(),
                type=EventType.MESSAGE_APPENDED,
                payload={
                    "message_id": "msg_old",
                    "from_": "user",
                    "to": "receptionist",
                    "content": [{"type": "text", "text": "old message"}],
                    "timestamp": _time.time(),
                    "delegations": {},
                },
            ))
            store.append(Event(
                id=0, session_id=archived_sid, frame_id="frame_orphan",
                timestamp=_time.time(),
                type=EventType.FRAME_RESOLVED,
                payload={"bubble_message_id": None},
            ))

            async with NatureClient(port=port) as client:
                # Should appear in archived list (no live entry)
                archived = await client.list_archived_sessions()
                assert any(a.session_id == archived_sid for a in archived)

                # Resume it
                resumed = await client.resume_session(archived_sid)
                assert resumed.session_id == archived_sid
                assert resumed.root_role_name == "receptionist"
                # Resume restamps the root model from the preset
                # (default.json → receptionist → anthropic::claude-haiku-4-5),
                # overriding the historical frame's model on disk.
                assert resumed.root_model == "claude-haiku-4-5"

                # Now in live list
                listed = await client.list_sessions()
                assert any(s.session_id == archived_sid for s in listed)

                # And no longer in archived list (it's live now)
                archived_after = await client.list_archived_sessions()
                assert not any(
                    a.session_id == archived_sid for a in archived_after
                )

                # Snapshot still has the historical events
                snap = await client.snapshot(archived_sid)
                assert len(snap) >= 3
        finally:
            await app.stop()


async def test_resume_with_preset_override_applies(tmp_path, monkeypatch):
    """`client.resume_session(sid, preset="...")` should swap the model
    by re-stamping the preset's root model onto the replayed root frame."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")

    # Project-local preset that pins receptionist onto a haiku override.
    # The preset roster mirrors default so agents exist.
    presets_dir = tmp_path / ".nature" / "presets"
    presets_dir.mkdir(parents=True)
    (presets_dir / "alt.json").write_text(json.dumps({
        "root_agent": "receptionist",
        "agents": [
            "receptionist", "core", "researcher",
            "analyzer", "implementer", "reviewer", "judge",
        ],
        "model_overrides": {
            "receptionist": "anthropic::claude-sonnet-4-6",
        },
    }))

    port = _free_port_pair()
    app = ServerApp(port=port, cwd=str(tmp_path), event_store_dir=tmp_path / "events")
    await app.start()
    try:
        from nature.events.types import Event
        import time as _time
        sid = "resume_override_sid"
        store = app.registry.event_store
        store.append(Event(
            id=0, session_id=sid, frame_id="frame_mx",
            timestamp=_time.time(),
            type=EventType.FRAME_OPENED,
            payload={
                "purpose": "root",
                "parent_id": None,
                "role_name": "receptionist",
                "role_description": "",
                "instructions": "be brief",
                "allowed_tools": None,
                "model": "original-model",
            },
        ))
        store.append(Event(
            id=0, session_id=sid, frame_id="frame_mx",
            timestamp=_time.time(),
            type=EventType.FRAME_RESOLVED,
            payload={"bubble_message_id": None},
        ))

        async with NatureClient(port=port) as client:
            resumed = await client.resume_session(sid, preset="alt")
            # Preset's root_override wins over the historical model.
            assert resumed.root_model == "claude-sonnet-4-6"
            # Role is still the archived one.
            assert resumed.root_role_name == "receptionist"

            # Resume should have emitted FRAME_REOPENED on top of
            # the archived log.
            snap = await client.snapshot(sid)
            reopened = [
                e for e in snap if e.type == EventType.FRAME_REOPENED
            ]
            assert len(reopened) == 1
            assert reopened[0].payload["previous_state"] == "resolved"
    finally:
        await app.stop()


async def test_fork_session_from_live_source_copies_events_and_hydrates():
    """POST /api/sessions/{sid}/fork should copy events 1..at_event_id
    from a live source into a brand-new session, populate fork lineage
    in the response, and the new session should be immediately listed
    as live with parent_session_id surfaced."""
    with tempfile.TemporaryDirectory() as tmp:
        port = _free_port_pair()
        app = ServerApp(port=port, cwd=tmp, event_store_dir=Path(tmp) / "events")
        await app.start()
        try:
            from nature.events.types import Event
            import time as _time

            src_sid = "fork_source_sid"
            fid = "frame_src"
            store = app.registry.event_store

            # 5 historical events on the source
            store.append(Event(
                id=0, session_id=src_sid, frame_id=fid,
                timestamp=_time.time(),
                type=EventType.FRAME_OPENED,
                payload={
                    "purpose": "root",
                    "parent_id": None,
                    "role_name": "receptionist",
                    "role_description": "",
                    "instructions": "be brief",
                    "allowed_tools": None,
                    "model": "claude-sonnet-4-20250514",
                },
            ))
            store.append(Event(
                id=0, session_id=src_sid, frame_id=fid,
                timestamp=_time.time(),
                type=EventType.MESSAGE_APPENDED,
                payload={
                    "message_id": "msg_1",
                    "from_": "user", "to": "receptionist",
                    "content": [{"type": "text", "text": "first"}],
                    "timestamp": _time.time(),
                    "delegations": {},
                },
            ))
            store.append(Event(
                id=0, session_id=src_sid, frame_id=fid,
                timestamp=_time.time(),
                type=EventType.MESSAGE_APPENDED,
                payload={
                    "message_id": "msg_2",
                    "from_": "receptionist", "to": "user",
                    "content": [{"type": "text", "text": "reply"}],
                    "timestamp": _time.time(),
                    "delegations": {},
                },
            ))
            store.append(Event(
                id=0, session_id=src_sid, frame_id=fid,
                timestamp=_time.time(),
                type=EventType.MESSAGE_APPENDED,
                payload={
                    "message_id": "msg_3",
                    "from_": "user", "to": "receptionist",
                    "content": [{"type": "text", "text": "second"}],
                    "timestamp": _time.time(),
                    "delegations": {},
                },
            ))
            store.append(Event(
                id=0, session_id=src_sid, frame_id=fid,
                timestamp=_time.time(),
                type=EventType.FRAME_RESOLVED,
                payload={"bubble_message_id": None},
            ))

            async with NatureClient(port=port) as client:
                # Fork at event 3 — should carry the first two messages
                # but not the second user turn.
                forked = await client.fork_session(
                    src_sid, at_event_id=3,
                )
                assert forked.parent_session_id == src_sid
                assert forked.forked_from_event_id == 3
                new_sid = forked.session_id
                assert new_sid != src_sid

                # New session snapshot has at least events 1..3 +
                # whatever resume emitted on top (FRAME_REOPENED).
                # NOTE: the HTTP /snapshot endpoint doesn't serialize
                # session_id on the wire (it's redundant with the URL),
                # so we verify session_id rewriting via the disk-level
                # event store directly below.
                snap = await client.snapshot(new_sid)
                assert len(snap) >= 3
                assert [e.id for e in snap[:3]] == [1, 2, 3]

                # Disk-level: store.snapshot returns full Event objects
                # including session_id, so this proves the fork copy
                # actually rewrote session_id (not just left the source
                # reference in place).
                disk = store.snapshot(new_sid)
                assert len(disk) >= 3
                assert all(e.session_id == new_sid for e in disk[:3])
                assert [e.id for e in disk[:3]] == [1, 2, 3]

                # Source session snapshot is untouched (all 5 events,
                # original session_id).
                src_snap = await client.snapshot(src_sid)
                assert len(src_snap) == 5
                src_disk = store.snapshot(src_sid)
                assert len(src_disk) == 5
                assert all(e.session_id == src_sid for e in src_disk)

                # Live listing surfaces the fork lineage on the new
                # session.
                live = await client.list_sessions()
                new_live = [s for s in live if s.session_id == new_sid]
                assert len(new_live) == 1
                assert new_live[0].parent_session_id == src_sid
                assert new_live[0].forked_from_event_id == 3
        finally:
            await app.stop()


async def test_fork_session_from_archived_source_works():
    """Source session lives only on disk (never resumed into memory).
    Fork should still find its event log and produce a fresh session."""
    with tempfile.TemporaryDirectory() as tmp:
        port = _free_port_pair()
        app = ServerApp(port=port, cwd=tmp, event_store_dir=Path(tmp) / "events")
        await app.start()
        try:
            from nature.events.types import Event
            import time as _time

            src_sid = "archived_source_sid"
            fid = "frame_arc"
            store = app.registry.event_store
            # Seed a 3-event archived session
            store.append(Event(
                id=0, session_id=src_sid, frame_id=fid,
                timestamp=_time.time(),
                type=EventType.FRAME_OPENED,
                payload={
                    "purpose": "root",
                    "parent_id": None,
                    "role_name": "receptionist",
                    "role_description": "",
                    "instructions": "be brief",
                    "allowed_tools": None,
                    "model": "claude-sonnet-4-20250514",
                },
            ))
            store.append(Event(
                id=0, session_id=src_sid, frame_id=fid,
                timestamp=_time.time(),
                type=EventType.MESSAGE_APPENDED,
                payload={
                    "message_id": "msg_a",
                    "from_": "user", "to": "receptionist",
                    "content": [{"type": "text", "text": "hi"}],
                    "timestamp": _time.time(),
                    "delegations": {},
                },
            ))
            store.append(Event(
                id=0, session_id=src_sid, frame_id=fid,
                timestamp=_time.time(),
                type=EventType.FRAME_RESOLVED,
                payload={"bubble_message_id": None},
            ))

            async with NatureClient(port=port) as client:
                # Source is archived (not in live list)
                live_before = await client.list_sessions()
                assert all(s.session_id != src_sid for s in live_before)

                forked = await client.fork_session(
                    src_sid, at_event_id=2,
                )
                assert forked.parent_session_id == src_sid
                assert forked.forked_from_event_id == 2

                # Fork works even though source was never hydrated.
                new_sid = forked.session_id
                snap = await client.snapshot(new_sid)
                assert len(snap) >= 2
                assert [e.id for e in snap[:2]] == [1, 2]
        finally:
            await app.stop()


async def test_fork_session_rejects_bad_at_event_id():
    with tempfile.TemporaryDirectory() as tmp:
        port = _free_port_pair()
        app = ServerApp(port=port, cwd=tmp, event_store_dir=Path(tmp) / "events")
        await app.start()
        try:
            from nature.events.types import Event
            import time as _time

            src_sid = "bad_fork_sid"
            store = app.registry.event_store
            store.append(Event(
                id=0, session_id=src_sid, frame_id="f",
                timestamp=_time.time(),
                type=EventType.FRAME_OPENED,
                payload={
                    "purpose": "root",
                    "parent_id": None,
                    "role_name": "receptionist",
                    "role_description": "",
                    "instructions": "",
                    "allowed_tools": None,
                    "model": "m",
                },
            ))

            async with NatureClient(port=port) as client:
                # at_event_id=0 should fail validation at the API layer
                with pytest.raises(NatureClientError):
                    await client.fork_session(src_sid, at_event_id=0)

                # Forking from an unknown source returns 404
                with pytest.raises(NatureClientError):
                    await client.fork_session(
                        "no_such_session_exists",
                        at_event_id=1,
                    )
        finally:
            await app.stop()


async def test_fork_session_preset_override_drives_new_branch(tmp_path, monkeypatch):
    """Event-pinned counterfactual primitive: the same source session
    can be forked into branches that resume under different presets.
    The forked session's `root_model` must reflect the override rather
    than the source's original model.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")

    # Project-local preset whose root (receptionist) is pinned to
    # anthropic::claude-sonnet-4-6 — distinct from default (haiku).
    presets_dir = tmp_path / ".nature" / "presets"
    presets_dir.mkdir(parents=True)
    (presets_dir / "fork-alt.json").write_text(json.dumps({
        "root_agent": "receptionist",
        "agents": [
            "receptionist", "core", "researcher",
            "analyzer", "implementer", "reviewer", "judge",
        ],
        "model_overrides": {
            "receptionist": "anthropic::claude-sonnet-4-6",
        },
    }))

    port = _free_port_pair()
    app = ServerApp(port=port, cwd=str(tmp_path), event_store_dir=tmp_path / "events")
    await app.start()
    try:
        from nature.events.types import Event
        import time as _time

        src_sid = "fork_preset_src"
        store = app.registry.event_store
        # Minimal source: FRAME_OPENED + USER_INPUT so there's
        # something to copy.
        store.append(Event(
            id=0, session_id=src_sid, frame_id="frame_src",
            timestamp=_time.time(),
            type=EventType.FRAME_OPENED,
            payload={
                "purpose": "root",
                "parent_id": None,
                "role_name": "receptionist",
                "role_description": "",
                "instructions": "",
                "allowed_tools": None,
                "model": "original-model",
            },
        ))
        store.append(Event(
            id=0, session_id=src_sid, frame_id="frame_src",
            timestamp=_time.time(),
            type=EventType.USER_INPUT,
            payload={"text": "hi", "source": "user"},
        ))

        async with NatureClient(port=port) as client:
            forked = await client.fork_session(
                src_sid, at_event_id=2, preset="fork-alt",
            )
            # Preset's receptionist override wins over the historical
            # model stamped in FRAME_OPENED.
            assert forked.root_role_name == "receptionist"
            assert forked.root_model == "claude-sonnet-4-6"

            # A second branch under the builtin default preset produces
            # a different root_model for the *same* source.
            forked_default = await client.fork_session(
                src_sid, at_event_id=2,  # preset unset → default.json
            )
            assert forked_default.root_model == "claude-haiku-4-5"
            assert forked.session_id != forked_default.session_id
    finally:
        await app.stop()


async def test_get_frame_context_up_to_event_id_slices_replay():
    """Dashboard scrubber endpoint: ?up_to=N returns the frame's body
    as it looked right after event N was applied."""
    with tempfile.TemporaryDirectory() as tmp:
        port = _free_port_pair()
        app = ServerApp(port=port, cwd=tmp, event_store_dir=Path(tmp) / "events")
        await app.start()
        try:
            from nature.events.types import Event
            import time as _time
            sid = "scrubber_sid"
            fid = "frame_scrub"
            store = app.registry.event_store
            # Seed: FRAME_OPENED → HEADER_SNAPSHOT → 3 messages → RESOLVED
            store.append(Event(
                id=0, session_id=sid, frame_id=fid,
                timestamp=_time.time(),
                type=EventType.FRAME_OPENED,
                payload={
                    "purpose": "root",
                    "parent_id": None,
                    "role_name": "receptionist",
                    "role_description": "",
                    "instructions": "go",
                    "allowed_tools": None,
                    "model": "m",
                },
            ))
            store.append(Event(
                id=0, session_id=sid, frame_id=fid,
                timestamp=_time.time(),
                type=EventType.HEADER_SNAPSHOT,
                payload={
                    "role": {
                        "name": "receptionist",
                        "description": "",
                        "instructions": "go",
                        "allowed_tools": None,
                        "model": None,
                    },
                    "principles": [],
                },
            ))
            for i, text in enumerate(("first", "second", "third")):
                store.append(Event(
                    id=0, session_id=sid, frame_id=fid,
                    timestamp=_time.time(),
                    type=EventType.MESSAGE_APPENDED,
                    payload={
                        "message_id": f"m{i}",
                        "from_": "user",
                        "to": "receptionist",
                        "content": [{"type": "text", "text": text}],
                        "timestamp": _time.time(),
                        "delegations": {},
                    },
                ))

            async with NatureClient(port=port) as client:
                # Full replay: all 3 messages
                full = await client.get_frame_context(sid, fid)
                assert full is not None
                assert len(full["body"]["conversation"]["messages"]) == 3

                # Slice at event id 3 (FRAME_OPENED=1, HEADER_SNAPSHOT=2,
                # MESSAGE_APPENDED(first)=3): only 1 message
                partial = await client.get_frame_context(
                    sid, fid, up_to_event_id=3,
                )
                assert partial is not None
                msgs = partial["body"]["conversation"]["messages"]
                assert len(msgs) == 1
                assert msgs[0]["content"][0]["text"] == "first"

                # Slice beyond the end == full
                beyond = await client.get_frame_context(
                    sid, fid, up_to_event_id=9999,
                )
                assert beyond is not None
                assert len(beyond["body"]["conversation"]["messages"]) == 3
        finally:
            await app.stop()


async def test_resume_unknown_session_returns_404():
    with tempfile.TemporaryDirectory() as tmp:
        port = _free_port_pair()
        app = ServerApp(port=port, cwd=tmp, event_store_dir=Path(tmp) / "events")
        await app.start()
        try:
            async with NatureClient(port=port) as client:
                with pytest.raises(NatureClientError, match="not_found"):
                    await client.resume_session("does_not_exist")
        finally:
            await app.stop()


async def test_list_archived_excludes_live_sessions():
    with tempfile.TemporaryDirectory() as tmp:
        port = _free_port_pair()
        app = ServerApp(port=port, cwd=tmp, event_store_dir=Path(tmp) / "events")
        await app.start()
        try:
            async with NatureClient(port=port) as client:
                created = await client.create_session()
                # Need at least one event so list_sessions() picks it up
                from nature.events.types import Event
                import time as _time
                app.registry.event_store.append(Event(
                    id=0, session_id=created.session_id, frame_id="f",
                    timestamp=_time.time(),
                    type=EventType.FRAME_OPENED,
                    payload={"role_name": "r"},
                ))

                archived = await client.list_archived_sessions()
                assert not any(
                    a.session_id == created.session_id for a in archived
                ), "live session should not appear in archived list"
        finally:
            await app.stop()


async def test_dashboard_html_served_with_substituted_ports():
    import httpx

    with tempfile.TemporaryDirectory() as tmp:
        port = _free_port_pair()
        app = ServerApp(port=port, cwd=tmp, event_store_dir=Path(tmp) / "events")
        await app.start()
        try:
            async with httpx.AsyncClient() as http:
                r = await http.get(f"http://localhost:{port}/")
                assert r.status_code == 200
                assert "nature" in r.text
                assert "nature · dashboard" in r.text
                # WS_PORT placeholder substituted with port+1
                assert str(port + 1) in r.text
                assert "__WS_PORT__" not in r.text
                # Dashboard now uses the structured view channel + REST
                # for session discovery. The raw event stream lives on
                # for mobile + tests, but the desktop no longer touches it.
                assert "connectTo" in r.text
                assert "/ws/view/sessions/" in r.text
                assert "/api/sessions" in r.text
        finally:
            await app.stop()


# ---------------------------------------------------------------------------
# Model / tool catalogs
# ---------------------------------------------------------------------------


async def test_http_list_models_returns_catalog():
    import httpx

    with tempfile.TemporaryDirectory() as tmp:
        port = _free_port_pair()
        app = ServerApp(port=port, cwd=tmp, event_store_dir=Path(tmp) / "events")
        await app.start()
        try:
            async with httpx.AsyncClient() as http:
                r = await http.get(f"http://localhost:{port}/api/config/models")
                assert r.status_code == 200
                data = r.json()
                ids = [m["id"] for m in data["models"]]
                assert "claude-opus-4-6" in ids
                assert "claude-sonnet-4-6" in ids
                for m in data["models"]:
                    assert m["tier"] in ("heavy", "medium", "light")
                    assert m["provider"] in ("anthropic", "openai", "openrouter")
        finally:
            await app.stop()


async def test_http_list_tools_matches_registry():
    import httpx

    with tempfile.TemporaryDirectory() as tmp:
        port = _free_port_pair()
        app = ServerApp(port=port, cwd=tmp, event_store_dir=Path(tmp) / "events")
        await app.start()
        try:
            async with httpx.AsyncClient() as http:
                r = await http.get(f"http://localhost:{port}/api/config/tools")
                assert r.status_code == 200
                tools = r.json()["tools"]
                # These tools are always present in the registry
                for expected in ("Read", "Write", "Edit", "Bash", "Agent"):
                    assert expected in tools
        finally:
            await app.stop()


