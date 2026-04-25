"""FrameDashboardServer — a pure event-store consumer dashboard.

An HTTP + WebSocket server that reads from EventStore.live_tail and
broadcasts events to browser clients. Zero coupling with execution:
can run alongside a live session or against a saved session snapshot.

Usage (from code):
    server = FrameDashboardServer(
        event_store=store,
        session_id=sid,
        model="claude-sonnet-4",
    )
    await server.start()
    # ... session runs, events flow into store, dashboard updates live
    await server.stop()

Layout: HTTP on `port`, WebSocket on `port + 1`. Each connecting
client gets its own live_tail subscription so late connects still
see full history.
"""

from __future__ import annotations

import asyncio
import json
import logging
import webbrowser
from typing import Any

from nature.events import Event, EventStore

logger = logging.getLogger(__name__)

DEFAULT_PORT = 7777


class FrameDashboardServer:
    """HTTP + WebSocket dashboard for frame-mode sessions."""

    def __init__(
        self,
        *,
        event_store: EventStore,
        session_id: str,
        port: int = DEFAULT_PORT,
        model: str = "",
    ) -> None:
        self._store = event_store
        self._session_id = session_id
        self._port = port
        self._ws_port = port + 1
        self._model = model
        self._ws_server: Any = None
        self._http_server: asyncio.base_events.Server | None = None
        self._ws_handler_tasks: set[asyncio.Task] = set()
        self._started = False

    @property
    def url(self) -> str:
        return f"http://localhost:{self._port}"

    async def start(self, *, open_browser: bool = True) -> bool:
        """Start the HTTP + WebSocket servers. Returns True on success."""
        try:
            import websockets  # noqa: F401
            import websockets.asyncio.server
        except ImportError:
            logger.warning(
                "websockets package not installed — dashboard disabled. "
                "Install with: pip install 'nature[dashboard]'"
            )
            return False

        # Try base port, then fall back by +2 if the first attempt fails
        last_err: Exception | None = None
        for attempt_port in (self._port, self._port + 2):
            try:
                await self._start_on_ports(
                    http_port=attempt_port,
                    ws_port=attempt_port + 1,
                )
                self._port = attempt_port
                self._ws_port = attempt_port + 1
                self._started = True
                break
            except OSError as exc:
                last_err = exc
                logger.debug("port %d unavailable: %s", attempt_port, exc)
                continue

        if not self._started:
            logger.warning("dashboard failed to bind ports: %s", last_err)
            return False

        logger.info("Frame dashboard running at %s", self.url)
        if open_browser:
            try:
                webbrowser.open(self.url)
            except Exception:
                pass
        return True

    async def _start_on_ports(self, *, http_port: int, ws_port: int) -> None:
        import websockets.asyncio.server

        html = (
            _DASHBOARD_HTML
            .replace("__WS_PORT__", str(ws_port))
            .replace("__SESSION_ID__", self._session_id)
            .replace("__MODEL__", self._model or "")
        )

        async def http_handler(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            try:
                data = await reader.readuntil(b"\r\n\r\n")
                first_line = data.split(b"\r\n", 1)[0].decode(
                    "latin-1", errors="replace"
                )
                if first_line.startswith("GET "):
                    body = html.encode("utf-8")
                    headers = (
                        b"HTTP/1.1 200 OK\r\n"
                        b"Content-Type: text/html; charset=utf-8\r\n"
                        + f"Content-Length: {len(body)}\r\n".encode()
                        + b"Connection: close\r\n"
                        b"\r\n"
                    )
                    writer.write(headers + body)
                else:
                    writer.write(b"HTTP/1.1 405 Method Not Allowed\r\n\r\n")
                await writer.drain()
            except Exception as exc:
                logger.debug("http handler error: %s", exc)
            finally:
                try:
                    writer.close()
                    await writer.wait_closed()
                except Exception:
                    pass

        async def ws_handler(websocket: Any) -> None:
            task = asyncio.current_task()
            if task is not None:
                self._ws_handler_tasks.add(task)
            try:
                # Session metadata first — dashboard uses it for the header
                await websocket.send(
                    json.dumps({
                        "__kind__": "session_meta",
                        "session_id": self._session_id,
                        "model": self._model,
                    })
                )
                # live_tail yields snapshot then live — full replay + tail
                async for event in self._store.live_tail(self._session_id):
                    try:
                        await websocket.send(
                            json.dumps(self._event_to_json(event))
                        )
                    except Exception:
                        break
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.debug("ws client disconnected: %s", exc)
            finally:
                if task is not None:
                    self._ws_handler_tasks.discard(task)

        self._http_server = await asyncio.start_server(
            http_handler, "localhost", http_port
        )
        self._ws_server = await websockets.asyncio.server.serve(
            ws_handler, "localhost", ws_port
        )

    async def stop(self) -> None:
        # Cancel all active ws handlers. Each is blocked on queue.get()
        # inside live_tail, so it won't notice the server shutdown on
        # its own — explicit cancellation triggers the finally blocks
        # (which remove subscribers from the store).
        for task in list(self._ws_handler_tasks):
            if not task.done():
                task.cancel()
        # Give cancelled tasks a chance to run their finally blocks
        if self._ws_handler_tasks:
            await asyncio.gather(
                *list(self._ws_handler_tasks), return_exceptions=True
            )
        self._ws_handler_tasks.clear()

        if self._ws_server is not None:
            self._ws_server.close()
            try:
                await self._ws_server.wait_closed()
            except Exception:
                pass
            self._ws_server = None
        if self._http_server is not None:
            self._http_server.close()
            try:
                await self._http_server.wait_closed()
            except Exception:
                pass
            self._http_server = None
        self._started = False

    @staticmethod
    def _event_to_json(event: Event) -> dict[str, Any]:
        return {
            "id": event.id,
            "type": event.type.value,
            "frame_id": event.frame_id,
            "timestamp": event.timestamp,
            "payload": event.payload,
        }


# ---------------------------------------------------------------------------
# HTML — inlined so the dashboard is a single-file server
# ---------------------------------------------------------------------------


_DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>nature · frame dashboard</title>
  <style>
    :root {
      --bg: #0a0e14;
      --bg-elev: #0f141c;
      --panel: #141a22;
      --panel-hover: #1a2230;
      --border: #2a3340;
      --border-soft: #1e2530;
      --text: #e6edf3;
      --text-dim: #7d8590;
      --text-dimer: #58606a;
      --accent: #58a6ff;
      --success: #3fb950;
      --warning: #d29922;
      --error: #f85149;
      --purple: #bc8cff;
      --pink: #ff7b72;
      --cyan: #79c0ff;
      --lilac: #d2a8ff;
      --depth-0: var(--accent);
      --depth-1: var(--purple);
      --depth-2: var(--cyan);
      --depth-3: var(--pink);
      --depth-4: var(--lilac);
      --font-mono: "JetBrains Mono", "SF Mono", Menlo, Consolas, monospace;
      --font-ui: -apple-system, BlinkMacSystemFont, "SF Pro Text",
                 "Segoe UI", Roboto, sans-serif;
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: var(--font-ui);
      font-size: 14px;
      line-height: 1.55;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
    }
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 5px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--text-dimer); }

    /* Top bar */
    header.top {
      position: sticky; top: 0; z-index: 10;
      background: var(--panel);
      border-bottom: 1px solid var(--border);
      padding: 14px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      backdrop-filter: blur(8px);
    }
    header.top .brand {
      font-family: var(--font-mono);
      font-size: 14px;
      font-weight: 600;
      letter-spacing: -0.01em;
      display: flex;
      align-items: baseline;
      gap: 8px;
    }
    header.top .brand .dot {
      color: var(--accent);
      font-size: 20px;
      line-height: 0;
    }
    header.top .brand .name {
      color: var(--text);
    }
    header.top .brand .sub {
      color: var(--text-dim);
      font-weight: 400;
      font-size: 12px;
    }
    header.top .meta {
      color: var(--text-dim);
      font-family: var(--font-mono);
      font-size: 12px;
      flex: 1;
      text-align: center;
    }
    header.top .status {
      display: flex;
      align-items: center;
      gap: 6px;
      font-family: var(--font-mono);
      font-size: 11px;
      color: var(--text-dim);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    header.top .status::before {
      content: "";
      width: 8px; height: 8px;
      border-radius: 50%;
      background: var(--text-dimer);
      transition: background 0.2s, box-shadow 0.2s;
    }
    header.top .status.connected::before {
      background: var(--success);
      box-shadow: 0 0 10px var(--success);
    }
    header.top .status.error::before {
      background: var(--error);
      box-shadow: 0 0 10px var(--error);
    }

    /* Pulse bar — shown only while a run is in flight. Spinner ticks
       in JS (text content rotates), opacity breathes via CSS keyframes,
       elapsed time + token counters update on the same JS tick.
       This is the "something is happening" indicator the TUI grew in
       commit 7c9b0f0; the dashboard mirrors it for browser users. */
    #pulse-bar {
      position: sticky;
      top: 50px;
      z-index: 9;
      background: var(--panel);
      border-bottom: 1px solid var(--border);
      padding: 6px 24px;
      font-family: var(--font-mono);
      font-size: 11px;
      color: var(--text-dim);
      display: none;
      align-items: center;
      gap: 10px;
    }
    #pulse-bar.active {
      display: flex;
      animation: pulseFade 1.6s ease-in-out infinite;
    }
    #pulse-bar .spin {
      color: var(--accent);
      font-size: 13px;
      width: 12px;
      text-align: center;
    }
    #pulse-bar .activity {
      color: var(--text);
      font-weight: 500;
    }
    #pulse-bar .meta {
      color: var(--text-dim);
      margin-left: auto;
    }
    @keyframes pulseFade {
      0%   { opacity: 0.55; }
      50%  { opacity: 1.0; }
      100% { opacity: 0.55; }
    }

    /* Main */
    main {
      padding: 16px 20px;
      max-width: 1800px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 16px;
      align-items: start;
    }
    #frames-col, #contexts-col {
      min-width: 0;
    }
    .col-label {
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--text-dimer);
      padding: 4px 8px;
      margin-bottom: 4px;
      border-bottom: 1px solid var(--border-soft);
      position: sticky;
      top: 0;
      background: var(--bg);
      z-index: 5;
    }
    @media (max-width: 1100px) {
      main {
        grid-template-columns: 1fr;
      }
    }

    /* Empty state */
    .empty {
      text-align: center;
      padding: 120px 20px;
      color: var(--text-dimer);
    }
    .empty .big {
      font-family: var(--font-mono);
      font-size: 16px;
      margin-bottom: 8px;
    }
    .empty .hint {
      font-size: 12px;
    }

    /* Frame card */
    .frame {
      margin: 14px 0;
      border-left: 3px solid var(--depth-0);
      border-radius: 6px;
      background: var(--panel);
      overflow: hidden;
      transition: opacity 0.3s;
    }
    .frame.depth-1 { border-left-color: var(--depth-1); }
    .frame.depth-2 { border-left-color: var(--depth-2); }
    .frame.depth-3 { border-left-color: var(--depth-3); }
    .frame.depth-4 { border-left-color: var(--depth-4); }
    .frame.closed { opacity: 0.55; }
    .frame.error-state { border-left-color: var(--error); }

    .frame-header {
      padding: 12px 16px;
      display: flex;
      align-items: center;
      gap: 12px;
      border-bottom: 1px solid var(--border-soft);
      background: linear-gradient(90deg, var(--panel) 0%, var(--bg-elev) 100%);
      font-family: var(--font-mono);
      font-size: 12px;
    }
    .frame-header .role {
      font-weight: 600;
      color: var(--depth-0);
    }
    .frame.depth-1 .frame-header .role { color: var(--depth-1); }
    .frame.depth-2 .frame-header .role { color: var(--depth-2); }
    .frame.depth-3 .frame-header .role { color: var(--depth-3); }
    .frame.depth-4 .frame-header .role { color: var(--depth-4); }
    .frame-header .purpose {
      color: var(--text-dim);
      font-style: italic;
      max-width: 40%;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .frame-header .spacer { flex: 1; }
    .frame-header .model {
      color: var(--text-dimer);
      font-size: 11px;
    }
    .frame-header .state {
      color: var(--text-dimer);
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      padding: 2px 8px;
      border: 1px solid var(--border);
      border-radius: 10px;
    }
    .frame-header .state.resolved {
      color: var(--success);
      border-color: rgba(63, 185, 80, 0.4);
    }
    .frame-header .state.error {
      color: var(--error);
      border-color: rgba(248, 81, 73, 0.4);
    }
    .frame-body {
      padding: 12px 16px;
    }
    .frame .frame {
      /* nested sub-frames — slightly smaller margins + different bg */
      background: var(--bg-elev);
      margin: 10px 0;
    }

    /* Message bubble */
    .message {
      margin: 8px 0;
      padding: 10px 14px;
      border-radius: 6px;
      background: var(--bg-elev);
      border: 1px solid var(--border-soft);
      border-left: 3px solid var(--text-dimer);
      transition: border-color 0.2s;
    }
    .message.user { border-left-color: var(--accent); }
    .message.assistant { border-left-color: var(--success); }
    .message.tool { border-left-color: var(--warning); }
    .message .from {
      font-family: var(--font-mono);
      font-size: 11px;
      color: var(--text-dim);
      margin-bottom: 4px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .message .from .actor { color: var(--text); font-weight: 500; }
    .message .from .arrow { color: var(--text-dimer); margin: 0 4px; }
    .message .content {
      color: var(--text);
      white-space: pre-wrap;
      word-wrap: break-word;
      font-size: 13px;
      max-height: 480px;
      overflow-y: auto;
      padding-right: 4px;
    }
    .message .content code {
      font-family: var(--font-mono);
      font-size: 12px;
      background: var(--panel);
      padding: 1px 5px;
      border-radius: 3px;
    }

    /* Tool card */
    .tool {
      margin: 6px 0;
      padding: 8px 12px;
      border-radius: 6px;
      background: var(--bg-elev);
      border: 1px solid var(--border-soft);
      font-family: var(--font-mono);
      font-size: 12px;
      transition: border-color 0.25s;
    }
    .tool.running {
      border-color: var(--warning);
      animation: pulseBorder 1.2s ease-in-out infinite;
    }
    @keyframes pulseBorder {
      0%, 100% { border-color: var(--warning); }
      50% { border-color: rgba(210, 153, 34, 0.35); }
    }
    .tool.success { border-left: 3px solid var(--success); }
    .tool.error { border-left: 3px solid var(--error); }
    .tool .head {
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .tool .dot {
      color: var(--warning);
      font-size: 14px;
      line-height: 0;
    }
    .tool.success .dot { color: var(--success); }
    .tool.error .dot { color: var(--error); }
    .tool .name { color: var(--text); font-weight: 600; }
    .tool .args {
      color: var(--text-dim);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      max-width: 700px;
    }
    .tool .result {
      margin-top: 6px;
      padding-top: 6px;
      border-top: 1px dashed var(--border-soft);
      color: var(--text-dim);
      white-space: pre-wrap;
      word-wrap: break-word;
      max-height: 200px;
      overflow-y: auto;
    }
    .tool .duration {
      margin-left: auto;
      color: var(--text-dimer);
      font-size: 11px;
    }

    /* Annotation badges under a message */
    .badges {
      display: flex;
      flex-wrap: wrap;
      gap: 5px;
      margin-top: 8px;
      font-family: var(--font-mono);
      font-size: 10px;
    }
    .badge {
      padding: 1px 7px;
      background: var(--panel);
      border: 1px solid var(--border-soft);
      border-radius: 10px;
      color: var(--text-dim);
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }
    .badge.tok { color: var(--text); }
    .badge.think { color: var(--purple); border-color: rgba(188, 140, 255, 0.3); }
    .badge.stop { color: var(--cyan); border-color: rgba(121, 192, 255, 0.3); }

    /* Error banner */
    .error-banner {
      margin: 10px 0;
      padding: 10px 14px;
      border-radius: 6px;
      background: rgba(248, 81, 73, 0.08);
      border: 1px solid rgba(248, 81, 73, 0.35);
      color: var(--error);
      font-family: var(--font-mono);
      font-size: 12px;
    }

    /* Expandable cards — click a message / tool / frame header to reveal details */
    .expandable {
      cursor: pointer;
      position: relative;
    }
    .expandable:hover {
      background: var(--panel-hover);
    }
    .expandable .caret {
      display: inline-block;
      color: var(--text-dimer);
      font-size: 10px;
      transition: transform 0.15s ease-out;
      margin-left: 6px;
      user-select: none;
    }
    .expandable.expanded .caret {
      transform: rotate(90deg);
    }
    .frame-header.expandable:hover {
      background: linear-gradient(90deg, var(--panel-hover) 0%, var(--bg-elev) 100%);
    }

    .details {
      display: none;
      margin-top: 10px;
      padding-top: 10px;
      border-top: 1px dashed var(--border);
      font-family: var(--font-mono);
      font-size: 11px;
      color: var(--text-dim);
      line-height: 1.5;
      cursor: default;
    }
    .expanded > .details {
      display: block;
    }
    .details .section-title {
      color: var(--text-dimer);
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin: 6px 0 4px 0;
    }
    .details .kv {
      display: flex;
      gap: 10px;
      margin: 2px 0;
    }
    .details .kv .k {
      color: var(--text-dimer);
      min-width: 90px;
      flex-shrink: 0;
    }
    .details .kv .v {
      color: var(--text);
      word-break: break-word;
      overflow-wrap: anywhere;
    }
    .details pre {
      margin: 4px 0;
      padding: 8px 10px;
      background: var(--bg);
      border: 1px solid var(--border-soft);
      border-radius: 4px;
      overflow-x: auto;
      white-space: pre-wrap;
      word-break: break-word;
      color: var(--text);
      font-size: 11px;
      line-height: 1.45;
      max-height: 400px;
      overflow-y: auto;
    }
    .details .block {
      margin: 4px 0;
      padding: 6px 10px;
      background: var(--bg);
      border-left: 2px solid var(--border);
      border-radius: 3px;
      font-size: 11px;
    }
    .details .block-type {
      color: var(--purple);
      text-transform: uppercase;
      font-size: 9px;
      letter-spacing: 0.05em;
      margin-right: 6px;
    }
    .details .block-type.text { color: var(--success); }
    .details .block-type.tool_use { color: var(--warning); }
    .details .block-type.tool_result { color: var(--cyan); }
    .details .block-type.thinking { color: var(--purple); }

    /* Right-column per-frame context card — synced with left frames */
    .context-card {
      margin: 14px 0;
      border-radius: 6px;
      background: var(--panel);
      border-left: 3px solid var(--depth-0);
      overflow: hidden;
      transition: opacity 0.3s;
    }
    .context-card.depth-1 { border-left-color: var(--depth-1); margin-left: 16px; }
    .context-card.depth-2 { border-left-color: var(--depth-2); margin-left: 32px; }
    .context-card.depth-3 { border-left-color: var(--depth-3); margin-left: 48px; }
    .context-card.depth-4 { border-left-color: var(--depth-4); margin-left: 64px; }
    .context-card.closed { opacity: 0.55; }
    .context-card .ctx-head {
      padding: 10px 14px;
      display: flex;
      align-items: center;
      gap: 10px;
      border-bottom: 1px solid var(--border-soft);
      background: linear-gradient(90deg, var(--panel) 0%, var(--bg-elev) 100%);
      font-family: var(--font-mono);
      font-size: 11px;
    }
    .context-card .ctx-head .role {
      font-weight: 600;
      color: var(--depth-0);
    }
    .context-card.depth-1 .ctx-head .role { color: var(--depth-1); }
    .context-card.depth-2 .ctx-head .role { color: var(--depth-2); }
    .context-card.depth-3 .ctx-head .role { color: var(--depth-3); }
    .context-card.depth-4 .ctx-head .role { color: var(--depth-4); }
    .context-card .ctx-head .spacer { flex: 1; }
    .context-card .ctx-head .state {
      color: var(--text-dimer);
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      padding: 2px 8px;
      border: 1px solid var(--border);
      border-radius: 10px;
    }
    .context-card .ctx-head .state.resolved {
      color: var(--success);
      border-color: rgba(63, 185, 80, 0.4);
    }
    .context-card .ctx-body {
      padding: 10px 14px;
    }
    .context-card .ctx-section {
      color: var(--text-dimer);
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin: 8px 0 4px 0;
      padding-bottom: 2px;
      border-bottom: 1px dashed var(--border-soft);
    }
    .context-card .ctx-section:first-child { margin-top: 0; }
    .context-card .ctx-role-instr {
      padding: 6px 10px;
      background: var(--bg);
      border: 1px solid var(--border-soft);
      border-radius: 4px;
      font-size: 11px;
      color: var(--text-dim);
      white-space: pre-wrap;
      word-break: break-word;
      max-height: 140px;
      overflow-y: auto;
      margin: 4px 0;
    }
    .context-card .ctx-msg {
      margin: 6px 0;
      padding: 6px 8px;
      background: var(--bg);
      border: 1px solid var(--border-soft);
      border-radius: 4px;
      font-size: 11px;
    }
    .context-card .ctx-msg-hdr {
      color: var(--text-dimer);
      font-size: 10px;
      margin-bottom: 4px;
      padding-bottom: 4px;
      border-bottom: 1px dashed var(--border-soft);
    }
    .context-card .ctx-msg-hdr .from { color: var(--success); font-weight: 600; }
    .context-card .ctx-msg-hdr .to { color: var(--cyan); font-weight: 600; }
    .context-card .ctx-msg-hdr .time { float: right; color: var(--text-dimer); }
    .context-card .ctx-msg-hdr .badges { margin-top: 2px; }
    .context-card .ctx-msg-hdr .badges .bg {
      display: inline-block;
      padding: 0 5px;
      margin-right: 4px;
      border-radius: 3px;
      font-size: 9px;
      background: var(--bg-elev);
      color: var(--text-dim);
      border: 1px solid var(--border-soft);
    }
    .context-card .ctx-msg-text {
      color: var(--text);
      white-space: pre-wrap;
      word-break: break-word;
      max-height: 260px;
      overflow-y: auto;
    }
    .context-card .ctx-msg-tool {
      color: var(--warning);
      font-family: var(--font-mono);
      font-size: 10px;
      margin: 2px 0;
      padding: 2px 6px;
      background: rgba(210, 153, 34, 0.06);
      border-radius: 3px;
      max-height: 140px;
      overflow-y: auto;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .context-card .ctx-msg-tool.result {
      color: var(--cyan);
      background: rgba(90, 190, 220, 0.05);
    }
    .context-card .ctx-msg-tool.result.error {
      color: var(--error);
      background: rgba(248, 81, 73, 0.06);
    }
    .context-card .ctx-msg-thinking {
      color: var(--purple);
      font-style: italic;
      font-size: 10px;
      margin: 2px 0;
      padding: 2px 6px;
      background: rgba(163, 113, 247, 0.06);
      border-radius: 3px;
      max-height: 100px;
      overflow-y: auto;
      white-space: pre-wrap;
      word-break: break-word;
    }

    /* Context view: header / body bands */
    .details .ctx-band {
      margin: 10px 0 4px 0;
      padding: 4px 10px;
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-weight: 600;
      border-radius: 3px;
    }
    .details .ctx-band.header-band {
      color: var(--warning);
      background: rgba(250, 180, 70, 0.07);
      border-left: 3px solid var(--warning);
    }
    .details .ctx-band.body-band {
      color: var(--cyan);
      background: rgba(90, 190, 220, 0.07);
      border-left: 3px solid var(--cyan);
    }
    .details .ctx-group {
      margin: 0 0 6px 8px;
      padding: 4px 6px 4px 10px;
      border-left: 1px solid var(--border-soft);
    }
    .details .msg-card {
      margin: 6px 0;
      padding: 6px 8px;
      background: var(--bg);
      border: 1px solid var(--border-soft);
      border-radius: 4px;
    }
    .details .msg-hdr {
      color: var(--text-dimer);
      font-size: 10px;
      margin-bottom: 4px;
      padding-bottom: 4px;
      border-bottom: 1px dashed var(--border-soft);
    }
    .details .msg-hdr .msg-from { color: var(--success); font-weight: 600; }
    .details .msg-hdr .msg-to { color: var(--cyan); font-weight: 600; }
    .details .msg-hdr .msg-time { float: right; color: var(--text-dimer); }

    /* Fade-in for new content */
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(4px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .message, .tool, .frame, .error-banner {
      animation: fadeIn 0.25s ease-out;
    }
  </style>
</head>
<body>
  <header class="top">
    <div class="brand">
      <span class="dot">●</span>
      <span class="name">nature</span>
      <span class="sub">frame dashboard</span>
    </div>
    <div class="meta" id="meta">connecting...</div>
    <div class="status" id="status">connecting</div>
  </header>
  <div id="pulse-bar">
    <span class="spin">⠋</span>
    <span class="activity">thinking</span>
    <span class="meta">0.0s</span>
  </div>
  <main id="main">
    <div id="frames-col">
      <div class="col-label">events · left</div>
      <div class="empty" id="empty">
        <div class="big">waiting for events</div>
        <div class="hint">this pane will populate as the session runs</div>
      </div>
    </div>
    <div id="contexts-col">
      <div class="col-label">context · per frame</div>
    </div>
  </main>

  <script>
    const WS_PORT = __WS_PORT__;
    const SESSION_ID = "__SESSION_ID__";
    const MODEL = "__MODEL__";

    const frames = new Map();   // frame_id -> { el, bodyEl, depth }
    const messages = new Map(); // message_id -> DOM el
    const tools = new Map();    // tool_use_id -> DOM el

    // Full payloads for the details panels (built lazily on expand)
    const frameData = new Map();         // frame_id -> full frame.opened payload
    const messagePayloads = new Map();   // message_id -> full message.appended payload
    const messageAnnotations = new Map();// message_id -> annotation.stored payload
    const toolStartPayloads = new Map(); // tool_use_id -> tool.started payload
    const toolEndPayloads = new Map();   // tool_use_id -> tool.completed payload

    function attachToggle(triggerEl, cardEl, detailsBuilder) {
      const caret = document.createElement('span');
      caret.className = 'caret';
      caret.textContent = '›';
      triggerEl.appendChild(caret);
      triggerEl.classList.add('expandable');

      triggerEl.addEventListener('click', (ev) => {
        // Clicks inside the details panel itself shouldn't collapse it
        if (ev.target.closest && ev.target.closest('.details')) return;
        ev.stopPropagation();
        const willExpand = !cardEl.classList.contains('expanded');
        cardEl.classList.toggle('expanded');
        if (willExpand) {
          const detailsEl = cardEl.querySelector(':scope > .details');
          if (detailsEl) {
            detailsEl.innerHTML = detailsBuilder();
          }
        }
      });
    }

    function refreshDetailsIfExpanded(cardEl, detailsBuilder) {
      if (cardEl && cardEl.classList.contains('expanded')) {
        const detailsEl = cardEl.querySelector(':scope > .details');
        if (detailsEl) detailsEl.innerHTML = detailsBuilder();
      }
    }

    function kv(k, v) {
      return `<div class="kv"><span class="k">${escapeHtml(k)}</span><span class="v">${escapeHtml(String(v))}</span></div>`;
    }
    function kvPre(k, v) {
      return `<div class="kv"><span class="k">${escapeHtml(k)}</span></div><pre>${escapeHtml(String(v))}</pre>`;
    }
    function sectionTitle(t) {
      return `<div class="section-title">${escapeHtml(t)}</div>`;
    }
    function escapeHtml(s) {
      return String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
      }[c]));
    }

    function formatTs(ts) {
      if (!ts) return '-';
      try {
        const d = new Date(Number(ts) * 1000);
        return d.toISOString().replace('T', ' ').replace('Z', '');
      } catch { return String(ts); }
    }

    function renderContentBlock(block) {
      const type = block && block.type;
      if (type === 'text') {
        return `<div class="block"><span class="block-type text">text</span><pre>${escapeHtml(block.text || '')}</pre></div>`;
      }
      if (type === 'tool_use') {
        const input = JSON.stringify(block.input || {}, null, 2);
        return `<div class="block"><span class="block-type tool_use">tool_use</span><div>${escapeHtml(block.name || '')} (id: ${escapeHtml(block.id || '')})</div><pre>${escapeHtml(input)}</pre></div>`;
      }
      if (type === 'tool_result') {
        const c = typeof block.content === 'string' ? block.content : JSON.stringify(block.content || '', null, 2);
        return `<div class="block"><span class="block-type tool_result">tool_result</span><div>id: ${escapeHtml(block.tool_use_id || '')}${block.is_error ? ' · <span style="color:var(--error)">error</span>' : ''}</div><pre>${escapeHtml(c)}</pre></div>`;
      }
      if (type === 'thinking') {
        return `<div class="block"><span class="block-type thinking">thinking</span><pre>${escapeHtml(block.thinking || '')}</pre></div>`;
      }
      return `<div class="block"><span class="block-type">${escapeHtml(type || '?')}</span><pre>${escapeHtml(JSON.stringify(block, null, 2))}</pre></div>`;
    }

    function buildFrameDetails(frameId) {
      const d = frameData.get(frameId);
      if (!d) return '<div class="kv"><span class="k">no data</span></div>';
      const p = d.payload || {};
      let html = '';
      html += kv('frame_id', frameId);
      html += kv('purpose', p.purpose || '-');
      html += kv('role', p.role_name || '-');
      html += kv('model', p.model || '(inherited)');
      html += kv('parent_id', p.parent_id || '(root)');
      html += kv('allowed_tools', p.allowed_tools == null ? 'all' : JSON.stringify(p.allowed_tools));
      html += kv('opened_at', formatTs(d.timestamp));
      if (p.role_description) {
        html += sectionTitle('description');
        html += `<pre>${escapeHtml(p.role_description)}</pre>`;
      }
      if (p.instructions) {
        html += sectionTitle('instructions');
        html += `<pre>${escapeHtml(p.instructions)}</pre>`;
      }
      // Placeholder for the live context fetch — the async load runs
      // right after this returns and replaces the inner HTML.
      html += `<div class="ctx-slot" data-frame-id="${escapeHtml(frameId)}">${sectionTitle('live context')}<div class="kv"><span class="k">loading...</span></div></div>`;
      // Schedule the fetch (microtask) so the details element is in the DOM
      setTimeout(() => fetchAndRenderContext(frameId), 0);
      return html;
    }

    function fetchAndRenderContext(frameId) {
      if (!CURRENT_SESSION_ID) return;
      // Same-origin fetch — uses the dashboard page's host:port automatically.
      const url = `/api/sessions/${encodeURIComponent(CURRENT_SESSION_ID)}/frames/${encodeURIComponent(frameId)}/context`;
      fetch(url)
        .then(r => r.ok ? r.json() : null)
        .then(ctx => {
          if (!ctx) return;
          const f = frames.get(frameId);
          if (!f) return;
          // Find the matching ctx-slot inside this frame's details
          const slot = f.el.querySelector(`:scope > .details .ctx-slot[data-frame-id="${cssEscape(frameId)}"]`);
          if (!slot) return;
          slot.innerHTML = renderContextView(ctx);
        })
        .catch(err => {
          console.warn('context fetch failed', err);
        });
    }

    function cssEscape(s) {
      // Minimal CSS attr escape — safe for our frame ids (alphanumeric + _-)
      return String(s).replace(/[^A-Za-z0-9_-]/g, '\\$&');
    }

    function renderContextView(ctx) {
      const header = ctx.header || {};
      const role = header.role || {};
      const principles = header.principles || [];
      const messages = ((ctx.body || {}).conversation || {}).messages || [];

      let html = '';

      // ── HEADER ───────────────────────────────────────────────
      html += `<div class="ctx-band header-band">── header ──</div>`;
      html += `<div class="ctx-group">`;
      html += sectionTitle('role');
      html += kv('name', role.name || '?');
      html += kv('model', role.model || '(inherited)');
      html += kv('allowed_tools', role.allowed_tools == null ? 'all' : JSON.stringify(role.allowed_tools));
      if (role.description) {
        html += `<div class="kv"><span class="k">description</span></div>`;
        html += `<pre>${escapeHtml(role.description)}</pre>`;
      }
      if (role.instructions) {
        html += `<div class="kv"><span class="k">instructions</span></div>`;
        html += `<pre>${escapeHtml(role.instructions)}</pre>`;
      }
      html += `</div>`;

      html += `<div class="ctx-group">`;
      html += sectionTitle(`principles (${principles.length})`);
      if (principles.length === 0) {
        html += '<div class="block">(none)</div>';
      } else {
        for (const bp of principles) {
          const src = bp.source || 'principle';
          const prio = bp.priority != null ? ` · priority ${bp.priority}` : '';
          html += `<div class="block"><span class="block-type">${escapeHtml(src)}${escapeHtml(prio)}</span><pre>${escapeHtml(bp.text || '')}</pre></div>`;
        }
      }
      html += `</div>`;

      // ── BODY ─────────────────────────────────────────────────
      html += `<div class="ctx-band body-band">── body ──</div>`;
      html += `<div class="ctx-group">`;
      html += sectionTitle(`conversation (${messages.length} messages)`);
      if (messages.length === 0) {
        html += '<div class="block">(empty)</div>';
      } else {
        for (let i = 0; i < messages.length; i++) {
          const m = messages[i];
          const from_ = m.from_ || '?';
          const to = m.to || '?';
          const blocks = m.content || [];
          html += `<div class="msg-card">`;
          html += `<div class="msg-hdr">#${i + 1} <span class="msg-from">${escapeHtml(from_)}</span> → <span class="msg-to">${escapeHtml(to)}</span> <span class="msg-time">${blocks.length} block${blocks.length === 1 ? '' : 's'} · ${formatTs(m.timestamp)}</span></div>`;
          for (const block of blocks) {
            html += renderContentBlock(block);
          }
          html += `</div>`;
        }
      }
      html += `</div>`;
      return html;
    }

    function buildMessageDetails(messageId) {
      const d = messagePayloads.get(messageId);
      const ann = messageAnnotations.get(messageId);
      let html = '';
      if (d) {
        const p = d.payload || {};
        html += kv('message_id', messageId);
        html += kv('from → to', `${p.from_ || '?'} → ${p.to || '?'}`);
        html += kv('timestamp', formatTs(p.timestamp || d.timestamp));
        const blocks = p.content || [];
        html += sectionTitle(`content blocks (${blocks.length})`);
        if (blocks.length === 0) {
          html += '<div class="block">(empty)</div>';
        } else {
          for (const block of blocks) {
            html += renderContentBlock(block);
          }
        }
      }
      if (ann) {
        const a = ann.payload || {};
        html += sectionTitle('annotation');
        if (a.stop_reason) html += kv('stop_reason', a.stop_reason);
        if (a.llm_request_id) html += kv('request_id', a.llm_request_id);
        if (a.duration_ms != null) html += kv('duration_ms', a.duration_ms);
        if (a.usage) {
          html += `<div class="kv"><span class="k">usage</span></div>`;
          html += `<pre>${escapeHtml(JSON.stringify(a.usage, null, 2))}</pre>`;
        }
        if (a.thinking && a.thinking.length) {
          html += sectionTitle(`thinking blocks (${a.thinking.length})`);
          for (const t of a.thinking) {
            html += `<div class="block"><span class="block-type thinking">thinking</span><pre>${escapeHtml(t)}</pre></div>`;
          }
        }
      }
      if (!d && !ann) {
        html += '<div class="kv"><span class="k">no data</span></div>';
      }
      return html;
    }

    function buildToolDetails(toolUseId) {
      const s = toolStartPayloads.get(toolUseId);
      const e = toolEndPayloads.get(toolUseId);
      let html = '';
      html += kv('tool_use_id', toolUseId);
      if (s) {
        const p = s.payload || {};
        html += kv('tool_name', p.tool_name || '-');
        html += kv('started_at', formatTs(s.timestamp));
        html += sectionTitle('tool_input');
        html += `<pre>${escapeHtml(JSON.stringify(p.tool_input || {}, null, 2))}</pre>`;
      }
      if (e) {
        const p = e.payload || {};
        html += kv('is_error', String(p.is_error));
        if (p.duration_ms != null) html += kv('duration_ms', p.duration_ms);
        html += kv('completed_at', formatTs(e.timestamp));
        html += sectionTitle('output (full, no truncation)');
        const output = p.output == null ? '(empty)' : String(p.output);
        html += `<pre>${escapeHtml(output)}</pre>`;
      }
      if (!s && !e) {
        html += '<div class="kv"><span class="k">no data</span></div>';
      }
      return html;
    }

    const mainEl = document.getElementById('main');
    const framesCol = document.getElementById('frames-col');
    const contextsCol = document.getElementById('contexts-col');
    let emptyEl = document.getElementById('empty');
    const statusEl = document.getElementById('status');
    const metaEl = document.getElementById('meta');

    // frame_id -> { el, bodyEl, lastMsgCard } for the right-column context card
    const contextCards = new Map();

    // Full session id (set from session_meta on WS connect).
    // SESSION_ID is the server-substituted placeholder, may be empty
    // when the dashboard is loaded standalone (auto-discovery picks one).
    let CURRENT_SESSION_ID = SESSION_ID || '';

    function setMeta(sessionId, model) {
      if (sessionId) CURRENT_SESSION_ID = sessionId;
      const sid = (sessionId || SESSION_ID || '').slice(0, 8);
      const m = model || MODEL || '';
      metaEl.textContent = `${sid}${m ? ' · ' + m : ''}`;
    }
    setMeta(SESSION_ID, MODEL);

    function removeEmpty() {
      if (emptyEl && emptyEl.parentNode) {
        emptyEl.parentNode.removeChild(emptyEl);
        emptyEl = null;
      }
    }

    function connect() {
      const ws = new WebSocket(`ws://${location.hostname}:${WS_PORT}/`);
      ws.onopen = () => {
        statusEl.textContent = 'live';
        statusEl.className = 'status connected';
      };
      ws.onclose = () => {
        statusEl.textContent = 'disconnected';
        statusEl.className = 'status error';
        setTimeout(connect, 2000);
      };
      ws.onerror = () => {
        statusEl.className = 'status error';
      };
      ws.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data);
          if (data.__kind__ === 'session_meta') {
            setMeta(data.session_id, data.model);
            return;
          }
          dispatch(data);
        } catch (e) {
          console.error('parse error', e);
        }
      };
    }

    function dispatch(event) {
      removeEmpty();
      try {
        switch (event.type) {
          case 'frame.opened':     onFrameOpened(event); break;
          case 'frame.resolved':   onFrameResolved(event); break;
          case 'frame.closed':     onFrameClosed(event); break;
          case 'frame.errored':    onFrameErrored(event); break;
          case 'frame.reopened':   pulseEnd(); break;
          case 'user.input':       break;
          case 'message.appended': onMessageAppended(event); break;
          case 'annotation.stored':onAnnotation(event); break;
          case 'header.snapshot':  break;
          case 'body.compacted':   break;
          case 'tool.started':     onToolStarted(event); break;
          case 'tool.completed':   onToolCompleted(event); break;
          case 'llm.request':      break;
          case 'llm.response':     break;
          case 'llm.error':        onError(event); break;
          case 'error':            onError(event); break;
          case 'principle.added':  break;
          case 'role.changed':     break;
          default: break;
        }
      } catch (e) {
        console.error('dispatch error for', event, e);
      }
    }

    function depthFor(parent_id) {
      if (!parent_id) return 0;
      const parent = frames.get(parent_id);
      return parent ? parent.depth + 1 : 1;
    }

    // ── Right column: per-frame context cards ─────────────────────
    function createContextCard(frameId, payload, depth) {
      const card = document.createElement('section');
      card.className = `context-card depth-${Math.min(depth, 4)}`;
      card.innerHTML = `
        <div class="ctx-head">
          <span class="role"></span>
          <span class="purpose"></span>
          <span class="spacer"></span>
          <span class="state">active</span>
        </div>
        <div class="ctx-body"></div>
      `;
      card.querySelector('.role').textContent = payload.role_name || '?';
      card.querySelector('.purpose').textContent = payload.purpose || '';
      const bodyEl = card.querySelector('.ctx-body');

      // ── header section: role + instructions ──
      const hdrSec = document.createElement('div');
      hdrSec.className = 'ctx-section';
      hdrSec.textContent = 'header';
      bodyEl.appendChild(hdrSec);

      const roleLine = document.createElement('div');
      roleLine.style.fontSize = '10px';
      roleLine.style.color = 'var(--text-dimer)';
      const toolsPart = payload.allowed_tools == null
        ? 'all tools'
        : `tools: ${payload.allowed_tools.length}`;
      roleLine.textContent = `${payload.role_name || '?'} · ${payload.model || '(inherited)'} · ${toolsPart}`;
      bodyEl.appendChild(roleLine);

      if (payload.role_description) {
        const descEl = document.createElement('div');
        descEl.className = 'ctx-role-instr';
        descEl.textContent = payload.role_description;
        bodyEl.appendChild(descEl);
      }
      if (payload.instructions) {
        const instrEl = document.createElement('div');
        instrEl.className = 'ctx-role-instr';
        instrEl.textContent = payload.instructions;
        bodyEl.appendChild(instrEl);
      }

      // ── body section: messages will be appended here ──
      const bodySec = document.createElement('div');
      bodySec.className = 'ctx-section';
      bodySec.textContent = 'body · conversation';
      bodyEl.appendChild(bodySec);

      const msgsWrap = document.createElement('div');
      msgsWrap.className = 'ctx-msgs';
      bodyEl.appendChild(msgsWrap);

      contextsCol.appendChild(card);
      contextCards.set(frameId, { el: card, msgsWrap, lastMsgEl: null, lastMsgId: null });
      card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    function contextAppendMessage(frameId, payload) {
      const cc = contextCards.get(frameId);
      if (!cc) return;
      const blocks = payload.content || [];
      const msgEl = document.createElement('div');
      msgEl.className = 'ctx-msg';
      const hdr = document.createElement('div');
      hdr.className = 'ctx-msg-hdr';
      hdr.innerHTML = `<span class="from">${escapeHtml(payload.from_ || '?')}</span> → <span class="to">${escapeHtml(payload.to || '?')}</span><span class="time">${formatTs(payload.timestamp)}</span>`;
      msgEl.appendChild(hdr);

      for (const block of blocks) {
        const bType = block && block.type;
        if (bType === 'text' && block.text) {
          const t = document.createElement('div');
          t.className = 'ctx-msg-text';
          t.textContent = block.text;
          msgEl.appendChild(t);
        } else if (bType === 'tool_use') {
          const t = document.createElement('div');
          t.className = 'ctx-msg-tool';
          const input = JSON.stringify(block.input || {}, null, 2);
          t.textContent = `▶ ${block.name || '?'} ${input}`;
          msgEl.appendChild(t);
        } else if (bType === 'tool_result') {
          const t = document.createElement('div');
          t.className = `ctx-msg-tool result${block.is_error ? ' error' : ''}`;
          const c = typeof block.content === 'string'
            ? block.content
            : JSON.stringify(block.content || '', null, 2);
          t.textContent = `⎿ ${c}`;
          msgEl.appendChild(t);
        } else if (bType === 'thinking') {
          const t = document.createElement('div');
          t.className = 'ctx-msg-thinking';
          t.textContent = block.thinking || '';
          msgEl.appendChild(t);
        }
      }

      cc.msgsWrap.appendChild(msgEl);
      cc.lastMsgEl = msgEl;
      cc.lastMsgId = payload.message_id || null;
      cc.el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    function contextAnnotateMessage(frameId, p) {
      const cc = contextCards.get(frameId);
      if (!cc || !cc.lastMsgEl) return;
      // Only annotate the matching message
      if (p.message_id && cc.lastMsgId && p.message_id !== cc.lastMsgId) return;
      const hdr = cc.lastMsgEl.querySelector('.ctx-msg-hdr');
      if (!hdr) return;
      let badgesEl = hdr.querySelector('.badges');
      if (!badgesEl) {
        badgesEl = document.createElement('div');
        badgesEl.className = 'badges';
        hdr.appendChild(badgesEl);
      }
      badgesEl.innerHTML = '';
      if (p.stop_reason) {
        const b = document.createElement('span');
        b.className = 'bg';
        b.textContent = `stop: ${p.stop_reason}`;
        badgesEl.appendChild(b);
      }
      if (p.usage) {
        const b = document.createElement('span');
        b.className = 'bg';
        const inp = p.usage.input_tokens || 0;
        const out = p.usage.output_tokens || 0;
        b.textContent = `↓${inp.toLocaleString()} ↑${out.toLocaleString()}`;
        badgesEl.appendChild(b);
      }
      if (p.duration_ms != null) {
        const b = document.createElement('span');
        b.className = 'bg';
        b.textContent = p.duration_ms < 1000 ? `${p.duration_ms}ms` : `${(p.duration_ms / 1000).toFixed(1)}s`;
        badgesEl.appendChild(b);
      }
    }

    function contextSetState(frameId, state) {
      const cc = contextCards.get(frameId);
      if (!cc) return;
      const stateEl = cc.el.querySelector('.ctx-head .state');
      if (stateEl) {
        stateEl.textContent = state;
        stateEl.className = `state ${state}`;
      }
      if (state === 'closed') cc.el.classList.add('closed');
    }

    function onFrameOpened(event) {
      const p = event.payload || {};
      const depth = depthFor(p.parent_id);
      // Sub-frame opened → root run is delegating
      if (depth > 0 && pulseState.startTs != null) {
        pulseState.activity = 'delegating → ' + (p.role_name || 'sub-agent');
      }
      const el = document.createElement('section');
      el.className = `frame depth-${Math.min(depth, 4)}`;
      el.innerHTML = `
        <div class="frame-header">
          <span class="role"></span>
          <span class="purpose"></span>
          <span class="spacer"></span>
          <span class="model"></span>
          <span class="state">active</span>
        </div>
        <div class="details"></div>
        <div class="frame-body"></div>
      `;
      el.querySelector('.role').textContent = p.role_name || '?';
      el.querySelector('.purpose').textContent = p.purpose || '';
      el.querySelector('.model').textContent = p.model || '';

      const bodyEl = el.querySelector('.frame-body');

      // Attach to parent's body or to the left column
      let target = framesCol;
      if (p.parent_id) {
        const parent = frames.get(p.parent_id);
        if (parent) target = parent.bodyEl;
      }
      target.appendChild(el);

      frames.set(event.frame_id, { el, bodyEl, depth });

      // Mirror on the right column as a context card (flat list, indented)
      createContextCard(event.frame_id, p, depth);

      // Store full payload + wire toggle on the header
      frameData.set(event.frame_id, { payload: p, timestamp: event.timestamp });
      const frameId = event.frame_id;
      attachToggle(
        el.querySelector('.frame-header'),
        el,
        () => buildFrameDetails(frameId),
      );

      el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    function onFrameResolved(event) {
      const f = frames.get(event.frame_id);
      if (f) {
        const state = f.el.querySelector('.frame-header .state');
        if (state) {
          state.textContent = 'resolved';
          state.className = 'state resolved';
        }
      }
      contextSetState(event.frame_id, 'resolved');
      // Root frame resolved → end the pulse. Sub-agent resolves
      // bubble back up; the next event will retag activity.
      if (event.frame_id === pulseRootFrameId) {
        pulseEnd();
      } else if (pulseState.startTs != null) {
        pulseState.activity = 'thinking';
      }
    }

    function onFrameClosed(event) {
      const f = frames.get(event.frame_id);
      if (f) f.el.classList.add('closed');
      contextSetState(event.frame_id, 'closed');
    }

    function onFrameErrored(event) {
      const f = frames.get(event.frame_id);
      if (f) {
        const state = f.el.querySelector('.frame-header .state');
        if (state) {
          state.textContent = 'error';
          state.className = 'state error';
        }
      }
      contextSetState(event.frame_id, 'error');
      // Root frame errored → pulse stops; sub-frame errors leave the
      // pulse running because the parent will surface its own state.
      if (event.frame_id === pulseRootFrameId) {
        pulseEnd();
      }
    }

    function extractText(content) {
      let text = '';
      for (const block of (content || [])) {
        if (block && block.type === 'text') text += block.text || '';
      }
      return text;
    }

    function onMessageAppended(event) {
      const p = event.payload || {};
      // Mirror EVERY message (including tool-only) on the right context panel
      contextAppendMessage(event.frame_id, p);

      const f = frames.get(event.frame_id);
      if (!f) return;
      const from = p.from_ || '?';
      const to = p.to || '?';

      // User input on a root frame → start the pulse if not running
      if (from === 'user' && f.depth === 0 && pulseState.startTs == null) {
        pulseStart(event.frame_id);
      }

      const text = extractText(p.content);
      // Skip tool-result messages with no text content (tool cards render them)
      if (from === 'tool' && !text) return;

      const cls = from === 'user' ? 'user' :
                  from === 'tool' ? 'tool' : 'assistant';

      const el = document.createElement('div');
      el.className = `message ${cls}`;
      el.innerHTML = `
        <div class="from">
          <span class="actor"></span>
          <span class="arrow">→</span>
          <span class="actor to"></span>
        </div>
        <div class="content"></div>
        <div class="details"></div>
      `;
      el.querySelector('.actor').textContent = from;
      el.querySelector('.actor.to').textContent = to;
      el.querySelector('.content').textContent = text;

      f.bodyEl.appendChild(el);
      if (p.message_id) {
        messages.set(p.message_id, el);
        messagePayloads.set(p.message_id, { payload: p, timestamp: event.timestamp });
        const messageId = p.message_id;
        attachToggle(el, el, () => buildMessageDetails(messageId));
      }
      el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    function onAnnotation(event) {
      const p = event.payload || {};
      // Mirror annotations on the right context panel's last message
      contextAnnotateMessage(event.frame_id, p);

      // Feed the pulse meta — accumulate input/output across every
      // LLM call in the current run (root + sub-agents).
      if (pulseState.startTs != null && p.usage) {
        pulseState.inputTokens += p.usage.input_tokens || 0;
        pulseState.outputTokens += p.usage.output_tokens || 0;
      }

      const msgEl = messages.get(p.message_id);
      if (!msgEl) return;

      // Store full annotation payload + refresh details if already expanded
      messageAnnotations.set(p.message_id, { payload: p, timestamp: event.timestamp });
      const messageId = p.message_id;
      refreshDetailsIfExpanded(msgEl, () => buildMessageDetails(messageId));

      const badges = [];
      if (p.stop_reason) badges.push({ text: `stop: ${p.stop_reason}`, cls: 'stop' });
      if (p.usage) {
        const u = p.usage;
        const inp = u.input_tokens || 0;
        const out = u.output_tokens || 0;
        const cache = u.cache_read_input_tokens || 0;
        let tok = `↓${inp.toLocaleString()} ↑${out.toLocaleString()}`;
        if (cache) tok += ` ◈${cache.toLocaleString()}`;
        badges.push({ text: tok, cls: 'tok' });
      }
      if (p.thinking && p.thinking.length) {
        badges.push({ text: `thinking ×${p.thinking.length}`, cls: 'think' });
      }
      if (p.duration_ms != null) {
        const d = p.duration_ms < 1000
          ? `${p.duration_ms}ms`
          : `${(p.duration_ms / 1000).toFixed(1)}s`;
        badges.push({ text: d, cls: '' });
      }

      if (!badges.length) return;

      // Remove any existing badges row to support re-annotation
      const existing = msgEl.querySelector('.badges');
      if (existing) existing.remove();

      const row = document.createElement('div');
      row.className = 'badges';
      for (const b of badges) {
        const span = document.createElement('span');
        span.className = `badge ${b.cls}`;
        span.textContent = b.text;
        row.appendChild(span);
      }
      msgEl.appendChild(row);
    }

    function formatArgs(obj) {
      if (!obj || typeof obj !== 'object') return '';
      const parts = [];
      for (const [k, v] of Object.entries(obj)) {
        let val = typeof v === 'string' ? v : JSON.stringify(v);
        if (val && val.length > 70) val = val.slice(0, 67) + '...';
        parts.push(`${k}: ${val}`);
      }
      return parts.join(', ');
    }

    function onToolStarted(event) {
      const f = frames.get(event.frame_id);
      if (!f) return;
      const p = event.payload || {};

      if (pulseState.startTs != null) {
        pulseState.activity = 'running ' + (p.tool_name || 'tool');
      }

      const el = document.createElement('div');
      el.className = 'tool running';
      el.innerHTML = `
        <div class="head">
          <span class="dot">●</span>
          <span class="name"></span>
          <span class="args"></span>
          <span class="duration"></span>
        </div>
        <div class="details"></div>
      `;
      el.querySelector('.name').textContent = p.tool_name || '?';
      el.querySelector('.args').textContent = `(${formatArgs(p.tool_input)})`;
      el.querySelector('.duration').textContent = '…';

      f.bodyEl.appendChild(el);
      tools.set(p.tool_use_id, el);

      toolStartPayloads.set(p.tool_use_id, { payload: p, timestamp: event.timestamp });
      const toolUseId = p.tool_use_id;
      attachToggle(el, el, () => buildToolDetails(toolUseId));

      el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    function onToolCompleted(event) {
      const p = event.payload || {};

      if (pulseState.startTs != null) {
        pulseState.activity = 'thinking';
      }

      const el = tools.get(p.tool_use_id);
      if (!el) return;

      el.classList.remove('running');
      el.classList.add(p.is_error ? 'error' : 'success');

      const duration = p.duration_ms < 1000
        ? `${p.duration_ms || 0}ms`
        : `${((p.duration_ms || 0) / 1000).toFixed(1)}s`;
      const durEl = el.querySelector('.duration');
      if (durEl) durEl.textContent = duration;

      const output = (p.output || '').toString();
      if (output) {
        // Insert the summary result block BEFORE the details element so
        // the expand panel still appears at the bottom of the card
        const detailsEl = el.querySelector(':scope > .details');
        const resEl = document.createElement('div');
        resEl.className = 'result';
        resEl.textContent = `⎿ ${output.length > 600 ? output.slice(0, 600) + '…' : output}`;
        if (detailsEl) {
          el.insertBefore(resEl, detailsEl);
        } else {
          el.appendChild(resEl);
        }
      }

      // Stash full payload + refresh details if currently expanded
      toolEndPayloads.set(p.tool_use_id, { payload: p, timestamp: event.timestamp });
      const toolUseId = p.tool_use_id;
      refreshDetailsIfExpanded(el, () => buildToolDetails(toolUseId));
    }

    function onError(event) {
      const p = event.payload || {};
      const f = frames.get(event.frame_id);
      const target = f ? f.bodyEl : framesCol;
      const el = document.createElement('div');
      el.className = 'error-banner';
      el.textContent = `✗ ${p.error_type || 'Error'}: ${p.message || ''}`;
      target.appendChild(el);
      pulseEnd();
    }

    // ------------------------------------------------------------------
    // Pulse indicator — keeps the dashboard header alive during long
    // turns. Mirrors the TUI status-bar pulse (commit 7c9b0f0).
    // ------------------------------------------------------------------
    const PULSE_FRAMES = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"];
    let pulseState = {
      startTs: null,        // ms since epoch when run started
      activity: 'thinking',
      inputTokens: 0,
      outputTokens: 0,
      spinIdx: 0,
    };
    const pulseBar = document.getElementById('pulse-bar');
    const pulseSpinEl = pulseBar.querySelector('.spin');
    const pulseActivityEl = pulseBar.querySelector('.activity');
    const pulseMetaEl = pulseBar.querySelector('.meta');
    let pulseRootFrameId = null;

    function pulseStart(rootFrameId) {
      pulseState.startTs = Date.now();
      pulseState.activity = 'thinking';
      pulseState.inputTokens = 0;
      pulseState.outputTokens = 0;
      pulseState.spinIdx = 0;
      pulseRootFrameId = rootFrameId;
      pulseBar.classList.add('active');
    }
    function pulseEnd() {
      pulseState.startTs = null;
      pulseRootFrameId = null;
      pulseBar.classList.remove('active');
    }
    function fmtElapsed(seconds) {
      if (seconds < 60) return seconds.toFixed(1) + 's';
      if (seconds < 3600) {
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        return m + 'm ' + (s < 10 ? '0' + s : s) + 's';
      }
      const h = Math.floor(seconds / 3600);
      const m = Math.floor((seconds % 3600) / 60);
      return h + 'h ' + (m < 10 ? '0' + m : m) + 'm';
    }
    function fmtTokens(n) {
      if (n < 1000) return String(n);
      if (n < 10000) return (n / 1000).toFixed(1) + 'k';
      return Math.floor(n / 1000) + 'k';
    }
    function pulseTick() {
      if (pulseState.startTs == null) return;
      pulseState.spinIdx = (pulseState.spinIdx + 1) % PULSE_FRAMES.length;
      pulseSpinEl.textContent = PULSE_FRAMES[pulseState.spinIdx];
      pulseActivityEl.textContent = pulseState.activity || 'thinking';
      const elapsed = (Date.now() - pulseState.startTs) / 1000;
      const bits = [fmtElapsed(elapsed)];
      const tokBits = [];
      if (pulseState.inputTokens) tokBits.push('↓ ' + fmtTokens(pulseState.inputTokens));
      if (pulseState.outputTokens) tokBits.push('↑ ' + fmtTokens(pulseState.outputTokens));
      if (tokBits.length) bits.push(tokBits.join(' · '));
      pulseMetaEl.textContent = '(' + bits.join(' · ') + ')';
    }
    setInterval(pulseTick, 150);

    connect();
  </script>
</body>
</html>
"""
