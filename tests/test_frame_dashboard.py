"""Tests for FrameDashboardServer — HTTP + WS event broadcaster.

These tests drive the server end-to-end with a real WebSocket client
(via the websockets library) and a real FileEventStore, skipping
gracefully if `websockets` isn't installed.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from pathlib import Path

import pytest

from nature.events import Event, EventType, FileEventStore


def _has_websockets() -> bool:
    try:
        import websockets  # noqa: F401
        return True
    except ImportError:
        return False


requires_ws = pytest.mark.skipif(
    not _has_websockets(),
    reason="websockets package not installed",
)


def _find_free_port_pair() -> int:
    """Return an HTTP port such that port and port+1 are both free."""
    import socket
    for candidate in range(18000, 18200, 2):
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
    raise RuntimeError("no free port pair found")


def _make_event(
    session_id: str = "s1",
    frame_id: str | None = "f1",
    event_type: EventType = EventType.FRAME_OPENED,
    payload: dict | None = None,
) -> Event:
    return Event(
        id=0,
        session_id=session_id,
        frame_id=frame_id,
        timestamp=time.time(),
        type=event_type,
        payload=payload or {},
    )


@requires_ws
async def test_dashboard_imports_and_instantiates():
    from nature.ui.frame_dashboard import FrameDashboardServer

    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        server = FrameDashboardServer(
            event_store=store,
            session_id="s1",
            port=_find_free_port_pair(),
            model="test-model",
        )
        assert server.url.startswith("http://localhost:")
        assert server._started is False


@requires_ws
async def test_dashboard_start_stop():
    from nature.ui.frame_dashboard import FrameDashboardServer

    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        server = FrameDashboardServer(
            event_store=store,
            session_id="s1",
            port=_find_free_port_pair(),
        )
        started = await server.start(open_browser=False)
        assert started is True
        assert server._started is True
        await server.stop()
        assert server._started is False


@requires_ws
async def test_dashboard_serves_html_with_ws_port_substitution():
    """GET / should return the dashboard HTML with WS_PORT resolved."""
    from nature.ui.frame_dashboard import FrameDashboardServer
    import websockets

    http_port = _find_free_port_pair()

    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        server = FrameDashboardServer(
            event_store=store,
            session_id="sid12345",
            port=http_port,
            model="some-model",
        )
        await server.start(open_browser=False)

        try:
            # Fetch HTML via a raw asyncio client
            reader, writer = await asyncio.open_connection("localhost", http_port)
            writer.write(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
            await writer.drain()
            body = await reader.read(-1)
            writer.close()
            await writer.wait_closed()

            text = body.decode("utf-8", errors="replace")
            assert "200 OK" in text
            assert "nature" in text
            assert "frame dashboard" in text
            # Substitutions happened
            assert str(http_port + 1) in text  # WS_PORT
            assert "sid12345" in text           # SESSION_ID
            assert "some-model" in text          # MODEL
            # Placeholders are gone
            assert "__WS_PORT__" not in text
            assert "__SESSION_ID__" not in text
        finally:
            await server.stop()


@requires_ws
async def test_dashboard_ws_delivers_historical_then_live_events():
    """A WS client should receive session_meta first, then snapshot, then live."""
    from nature.ui.frame_dashboard import FrameDashboardServer
    import websockets

    http_port = _find_free_port_pair()

    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        # Pre-write one historical event
        store.append(_make_event(
            event_type=EventType.FRAME_OPENED,
            payload={"purpose": "root", "role_name": "receptionist"},
        ))

        server = FrameDashboardServer(
            event_store=store,
            session_id="s1",
            port=http_port,
            model="m",
        )
        await server.start(open_browser=False)

        received: list[dict] = []
        try:
            async with websockets.connect(
                f"ws://localhost:{http_port + 1}/"
            ) as ws:
                async def collect(n: int) -> None:
                    for _ in range(n):
                        msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                        received.append(json.loads(msg))

                # Expect: session_meta + historical FRAME_OPENED
                await collect(2)
                assert received[0].get("__kind__") == "session_meta"
                assert received[0]["session_id"] == "s1"
                assert received[1]["type"] == "frame.opened"
                assert received[1]["payload"]["role_name"] == "receptionist"

                # Emit a live event — client should see it
                store.append(_make_event(
                    event_type=EventType.MESSAGE_APPENDED,
                    payload={
                        "message_id": "m1",
                        "from_": "user",
                        "to": "receptionist",
                        "content": [{"type": "text", "text": "hi"}],
                        "timestamp": time.time(),
                        "delegations": {},
                    },
                ))
                await collect(1)
                assert received[2]["type"] == "message.appended"
                assert received[2]["payload"]["from_"] == "user"
        finally:
            await server.stop()


@requires_ws
async def test_dashboard_multiple_clients_each_get_full_history():
    """Late-connecting clients should still receive all events."""
    from nature.ui.frame_dashboard import FrameDashboardServer
    import websockets

    http_port = _find_free_port_pair()

    with tempfile.TemporaryDirectory() as tmp:
        store = FileEventStore(Path(tmp))
        for i in range(3):
            store.append(_make_event(
                event_type=EventType.FRAME_OPENED,
                payload={"purpose": f"p{i}", "role_name": f"role{i}"},
            ))

        server = FrameDashboardServer(
            event_store=store,
            session_id="s1",
            port=http_port,
        )
        await server.start(open_browser=False)

        try:
            async def collect_n(ws, n):
                out = []
                for _ in range(n):
                    msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    out.append(json.loads(msg))
                return out

            # Client A connects first
            ws_url = f"ws://localhost:{http_port + 1}/"
            async with websockets.connect(ws_url) as ws_a:
                events_a = await collect_n(ws_a, 4)  # meta + 3 frames
                assert events_a[0].get("__kind__") == "session_meta"
                assert [e["payload"]["role_name"] for e in events_a[1:]] == [
                    "role0", "role1", "role2"
                ]

                # Client B connects late
                async with websockets.connect(ws_url) as ws_b:
                    events_b = await collect_n(ws_b, 4)
                    assert events_b[0].get("__kind__") == "session_meta"
                    assert [e["payload"]["role_name"] for e in events_b[1:]] == [
                        "role0", "role1", "role2"
                    ]
        finally:
            await server.stop()


def test_frame_tui_accepts_open_browser_flag_without_crashing():
    """Smoke: the constructor takes the client-mode kwargs without error."""
    from nature.ui.frame_tui import FrameTUI
    tui = FrameTUI(host="localhost", port=12345, open_browser=True)
    assert tui._open_browser is True
    assert tui._client is None  # not built until on_mount
    assert tui._host == "localhost"
    assert tui._port == 12345


def test_format_elapsed_and_tokens_helpers():
    """Status-bar pulse helpers format elapsed time and token counts for
    the three size buckets the TUI actually encounters."""
    from nature.ui.frame_tui import _format_elapsed, _format_tokens

    # Elapsed — sub-minute shows decimals, minute/hour strip them
    assert _format_elapsed(0.3) == "0.3s"
    assert _format_elapsed(5.27) == "5.3s"
    assert _format_elapsed(59.9) == "59.9s"
    assert _format_elapsed(60) == "1m 00s"
    assert _format_elapsed(125) == "2m 05s"
    assert _format_elapsed(3600) == "1h 00m"
    assert _format_elapsed(3725) == "1h 02m"

    # Tokens — exact → 1.Xk → whole k
    assert _format_tokens(0) == "0"
    assert _format_tokens(42) == "42"
    assert _format_tokens(999) == "999"
    assert _format_tokens(1000) == "1.0k"
    assert _format_tokens(2847) == "2.8k"
    assert _format_tokens(9999) == "10.0k"   # pre-rollover
    assert _format_tokens(10_000) == "10k"
    assert _format_tokens(15_432) == "15k"


def test_frame_tui_run_lifecycle_state():
    """`_start_run` / `_end_run` flip the reactive state the status-bar
    tick reads from — verify they don't touch unrelated fields."""
    from nature.ui.frame_tui import FrameTUI

    tui = FrameTUI(host="localhost", port=12345)
    # Initial
    assert tui._run_start_ts is None
    assert tui._run_input_tokens == 0
    assert tui._run_output_tokens == 0
    assert tui._run_activity == "thinking"

    # Mid-run pollution that should survive a restart
    tui._run_input_tokens = 1234
    tui._run_output_tokens = 567
    tui._run_activity = "running Read"

    tui._start_run()
    assert tui._run_start_ts is not None
    assert tui._run_input_tokens == 0       # reset
    assert tui._run_output_tokens == 0      # reset
    assert tui._run_activity == "thinking"  # reset

    # Simulate the run accumulating
    tui._run_input_tokens = 3000
    tui._run_output_tokens = 150
    tui._run_activity = "running Glob"

    tui._end_run()
    assert tui._run_start_ts is None
    # Tokens are not reset by _end_run — they remain for introspection
    # until the next _start_run clears them.
    assert tui._run_input_tokens == 3000
