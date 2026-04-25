"""ServerApp — HTTP + WebSocket server hosting nature execution.

Layout: HTTP on `port`, WebSocket on `port + 1`. Same pattern as the
old FrameDashboardServer, but now the HTTP side serves a real REST API
plus the dashboard HTML, and the WS side is per-session.

Routes:

    GET    /                               → dashboard HTML
    GET    /m                              → mobile dashboard HTML
    GET    /api/sessions                   → list sessions
    POST   /api/sessions                   → create session
    GET    /api/sessions/{id}              → session detail
    POST   /api/sessions/{id}/messages     → send user input
    POST   /api/sessions/{id}/cancel       → cancel current run
    DELETE /api/sessions/{id}              → close session
    GET    /api/sessions/{id}/snapshot     → events snapshot
    WS     /ws/sessions/{id}               → live raw event stream
    WS     /ws/view/sessions/{id}          → live structured SessionViewDto
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any, Awaitable, Callable

from nature.agents.presets import PresetValidationError
from nature.events.store import FileEventStore
from nature.events.types import Event
from nature.server.api import (
    AgentAdminInfo,
    AgentPutRequest,
    ArchivedSessionInfo,
    CreateSessionRequest,
    CreateSessionResponse,
    ErrorResponse,
    ForkSessionRequest,
    HostAdminInfo,
    HostPutRequest,
    ListAgentsAdminResponse,
    ListArchivedSessionsResponse,
    ListHostsAdminResponse,
    ListModelsResponse,
    ListPresetsAdminResponse,
    ListSessionsResponse,
    ListToolsResponse,
    ModelCatalogEntry,
    OkResponse,
    PresetAdminInfo,
    PresetPutRequest,
    SendMessageRequest,
    SessionInfo,
)
from nature.server.registry import ServerSession, SessionRegistry

logger = logging.getLogger(__name__)

DEFAULT_PORT = 7777
DEFAULT_HOST = "localhost"


def _parse_up_to(query_string: str) -> int | None:
    """Parse `up_to=N` out of a query string. Returns None if absent or
    unparseable. N must be a non-negative integer."""
    if not query_string:
        return None
    from urllib.parse import parse_qs
    try:
        params = parse_qs(query_string, keep_blank_values=False)
    except ValueError:
        return None
    raw = params.get("up_to")
    if not raw:
        return None
    try:
        val = int(raw[0])
    except (ValueError, TypeError):
        return None
    return val if val >= 0 else None


class ServerApp:
    """Long-running daemon hosting nature execution behind HTTP + WS."""

    def __init__(
        self,
        *,
        port: int = DEFAULT_PORT,
        host: str = DEFAULT_HOST,
        cwd: str | None = None,
        event_store_dir: Path | None = None,
    ) -> None:
        from nature.config.settings import get_nature_home

        store_dir = event_store_dir or (get_nature_home() / "events")
        store_dir.mkdir(parents=True, exist_ok=True)

        self._host = host
        self._port = port
        self._ws_port = port + 1
        self._cwd = cwd
        self._store = FileEventStore(store_dir)
        self._registry = SessionRegistry(event_store=self._store, cwd=cwd)

        self._http_server: asyncio.base_events.Server | None = None
        self._ws_server: Any = None
        self._ws_handler_tasks: set[asyncio.Task] = set()
        self._started = False

    @property
    def http_url(self) -> str:
        return f"http://{self._display_host}:{self._port}"

    @property
    def ws_url(self) -> str:
        return f"ws://{self._display_host}:{self._ws_port}"

    @property
    def _display_host(self) -> str:
        # Show a clickable URL even when bound to all interfaces.
        return "localhost" if self._host in ("", "0.0.0.0", "::") else self._host

    @property
    def registry(self) -> SessionRegistry:
        return self._registry

    @property
    def started(self) -> bool:
        return self._started

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> bool:
        try:
            import websockets  # noqa: F401
            import websockets.asyncio.server
        except ImportError:
            logger.error(
                "websockets package not installed. Install with: "
                "pip install 'nature[dashboard]'"
            )
            return False

        try:
            self._http_server = await asyncio.start_server(
                self._handle_http_connection, self._host, self._port
            )
            self._ws_server = await websockets.asyncio.server.serve(
                self._handle_ws_connection, self._host, self._ws_port
            )
        except OSError as exc:
            logger.error("failed to bind server ports: %s", exc)
            await self.stop()
            return False

        self._started = True
        logger.info("nature server listening at %s (WS %s)", self.http_url, self.ws_url)
        self._warn_missing_tool_config()
        return True

    def _warn_missing_tool_config(self) -> None:
        """One-shot at boot: scan the builtin agent registry for tools
        that depend on env vars, and log a warning for each missing one.
        The tools themselves still surface an error at call-time — this
        just gives the operator a heads-up before the first session."""
        import os
        checks = [
            ("WebSearch", "BRAVE_SEARCH_API_KEY", "https://api.search.brave.com"),
        ]
        try:
            from nature.agents.config import load_agents_registry
            reg = load_agents_registry(project_dir=Path(self._registry.cwd))
            exposed: set[str] = set()
            for agent in reg.agents.values():
                exposed.update(agent.allowed_tools or [])
        except Exception:  # noqa: BLE001 — boot-path probe must not kill startup
            return
        for tool_name, env_var, provider_url in checks:
            if tool_name in exposed and not os.environ.get(env_var, "").strip():
                logger.warning(
                    "%s tool is exposed to at least one agent but %s is unset; "
                    "tool calls will fail with a config error. Set the env var "
                    "(key at %s) or remove the tool from allowed_tools.",
                    tool_name, env_var, provider_url,
                )

    async def stop(self) -> None:
        # Cancel any in-flight ws handlers — they're blocked on live_tail
        for task in list(self._ws_handler_tasks):
            if not task.done():
                task.cancel()
        if self._ws_handler_tasks:
            await asyncio.gather(
                *list(self._ws_handler_tasks), return_exceptions=True
            )
        self._ws_handler_tasks.clear()

        await self._registry.close_all()

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

    # ------------------------------------------------------------------
    # HTTP handling
    # ------------------------------------------------------------------

    async def _handle_http_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            parsed = await self._parse_http_request(reader)
            if parsed is None:
                writer.write(b"HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n")
                await writer.drain()
                return
            method, path, body = parsed
            response_bytes = await self._route(method, path, body)
            writer.write(response_bytes)
            await writer.drain()
        except Exception as exc:
            logger.exception("http handler error: %s", exc)
            try:
                writer.write(
                    b"HTTP/1.1 500 Internal Server Error\r\nConnection: close\r\n\r\n"
                )
                await writer.drain()
            except Exception:
                pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    @staticmethod
    async def _parse_http_request(
        reader: asyncio.StreamReader,
    ) -> tuple[str, str, bytes] | None:
        try:
            request_line_bytes = await reader.readline()
            if not request_line_bytes:
                return None
            request_line = request_line_bytes.decode("latin-1").strip()
            parts = request_line.split(" ", 2)
            if len(parts) != 3:
                return None
            method, path, _version = parts

            headers: dict[str, str] = {}
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b"\n", b""):
                    break
                name, _, value = line.decode("latin-1").partition(":")
                headers[name.strip().lower()] = value.strip()

            body = b""
            content_length = int(headers.get("content-length", "0") or "0")
            if content_length > 0:
                body = await reader.readexactly(content_length)

            return (method.upper(), path, body)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    _ROUTE_TABLE: list[tuple[str, re.Pattern[str], str]] = [
        ("GET",    re.compile(r"^/$"),                                                                   "_route_dashboard"),
        ("GET",    re.compile(r"^/m$"),                                                                  "_route_mobile"),
        ("GET",    re.compile(r"^/eval$"),                                                               "_route_eval_page"),
        ("GET",    re.compile(r"^/api/eval/runs$"),                                                      "_route_eval_list_runs"),
        ("GET",    re.compile(r"^/api/eval/runs/(?P<run_id>[A-Za-z0-9_-]+)$"),                           "_route_eval_get_run"),
        ("GET",    re.compile(r"^/api/eval/runs/(?P<run_id>[A-Za-z0-9_-]+)/cells/(?P<task_id>[A-Za-z0-9_.-]+)/(?P<preset>[A-Za-z0-9_.-]+)$"), "_route_eval_get_cell"),
        ("GET",    re.compile(r"^/api/eval/runs/(?P<run_id>[A-Za-z0-9_-]+)/cells/(?P<task_id>[A-Za-z0-9_.-]+)/(?P<preset>[A-Za-z0-9_.-]+)/turns/(?P<frame_id>[A-Za-z0-9_]+)/(?P<turn_index>\d+)/request$"), "_route_eval_get_turn_request"),
        ("GET",    re.compile(r"^/api/eval/diff$"),                                                      "_route_eval_diff"),
        ("GET",    re.compile(r"^/probe$"),                                                               "_route_probe_page"),
        ("GET",    re.compile(r"^/api/probe/runs$"),                                                      "_route_probe_list_runs"),
        ("GET",    re.compile(r"^/api/probe/runs/(?P<run_id>[A-Za-z0-9_-]+)$"),                           "_route_probe_get_run"),
        ("GET",    re.compile(r"^/api/probe/probes$"),                                                    "_route_probe_list_probes"),
        ("GET",    re.compile(r"^/api/probe/combined$"),                                                  "_route_probe_combined"),
        ("GET",    re.compile(r"^/api/sessions$"),                                                       "_route_list_sessions"),
        ("GET",    re.compile(r"^/api/sessions/archived$"),                                              "_route_list_archived"),
        ("POST",   re.compile(r"^/api/sessions$"),                                                       "_route_create_session"),
        ("GET",    re.compile(r"^/api/sessions/(?P<sid>[A-Za-z0-9_-]+)$"),                               "_route_get_session"),
        ("POST",   re.compile(r"^/api/sessions/(?P<sid>[A-Za-z0-9_-]+)/messages$"),                      "_route_send_message"),
        ("POST",   re.compile(r"^/api/sessions/(?P<sid>[A-Za-z0-9_-]+)/cancel$"),                        "_route_cancel"),
        ("POST",   re.compile(r"^/api/sessions/(?P<sid>[A-Za-z0-9_-]+)/resume$"),                        "_route_resume"),
        ("POST",   re.compile(r"^/api/sessions/(?P<sid>[A-Za-z0-9_-]+)/fork$"),                          "_route_fork"),
        ("GET",    re.compile(r"^/api/config/models$"),                                                  "_route_list_models"),
        ("GET",    re.compile(r"^/api/config/tools$"),                                                   "_route_list_tools"),
        ("GET",    re.compile(r"^/api/admin/agents$"),                                                   "_route_admin_list_agents"),
        ("GET",    re.compile(r"^/api/admin/agents/(?P<name>[A-Za-z0-9_-]+)$"),                          "_route_admin_get_agent"),
        ("PUT",    re.compile(r"^/api/admin/agents/(?P<name>[A-Za-z0-9_-]+)$"),                          "_route_admin_put_agent"),
        ("DELETE", re.compile(r"^/api/admin/agents/(?P<name>[A-Za-z0-9_-]+)$"),                          "_route_admin_delete_agent"),
        ("GET",    re.compile(r"^/api/admin/presets$"),                                                  "_route_admin_list_presets"),
        ("GET",    re.compile(r"^/api/admin/presets/(?P<name>[A-Za-z0-9_-]+)$"),                         "_route_admin_get_preset"),
        ("PUT",    re.compile(r"^/api/admin/presets/(?P<name>[A-Za-z0-9_-]+)$"),                         "_route_admin_put_preset"),
        ("DELETE", re.compile(r"^/api/admin/presets/(?P<name>[A-Za-z0-9_-]+)$"),                         "_route_admin_delete_preset"),
        ("GET",    re.compile(r"^/api/admin/hosts$"),                                                    "_route_admin_list_hosts"),
        ("PUT",    re.compile(r"^/api/admin/hosts/(?P<name>[A-Za-z0-9_-]+)$"),                           "_route_admin_put_host"),
        ("DELETE", re.compile(r"^/api/admin/hosts/(?P<name>[A-Za-z0-9_-]+)$"),                           "_route_admin_delete_host"),
        ("DELETE", re.compile(r"^/api/sessions/(?P<sid>[A-Za-z0-9_-]+)$"),                               "_route_close_session"),
        ("GET",    re.compile(r"^/api/sessions/(?P<sid>[A-Za-z0-9_-]+)/snapshot$"),                      "_route_snapshot"),
        ("GET",    re.compile(r"^/api/sessions/(?P<sid>[A-Za-z0-9_-]+)/frames/(?P<fid>[A-Za-z0-9_-]+)/context$"), "_route_frame_context"),
    ]

    async def _route(self, method: str, path: str, body: bytes) -> bytes:
        # Strip query string before regex matching so every handler
        # sees a clean path. Handlers that want query params declare
        # a `query_string` keyword argument — we thread it in via
        # signature inspection so existing handlers don't need changes.
        import inspect as _inspect
        path_only, _, query_string = path.partition("?")
        for route_method, regex, handler_name in self._ROUTE_TABLE:
            if route_method != method:
                continue
            match = regex.match(path_only)
            if match:
                handler: Callable[..., Awaitable[bytes]] = getattr(self, handler_name)
                kwargs: dict[str, Any] = dict(match.groupdict())
                if "query_string" in _inspect.signature(handler).parameters:
                    kwargs["query_string"] = query_string
                try:
                    return await handler(body, **kwargs)
                except Exception as exc:
                    logger.exception("route handler error in %s", handler_name)
                    return self._http_json_response(
                        500, ErrorResponse(error="internal", detail=str(exc)).model_dump()
                    )
        return self._http_json_response(
            404, ErrorResponse(error="not_found", detail=path).model_dump()
        )

    # ------------------------------------------------------------------
    # Route handlers
    # ------------------------------------------------------------------

    _STATIC_DIR = Path(__file__).parent / "static"

    def _read_static_html(self, name: str) -> bytes:
        """Read a static HTML template, substitute __WS_PORT__, return UTF-8 bytes.

        Read-on-each-request so UI edits land on browser refresh — the
        template strings are small (tens of KB) and the OS page cache
        makes this effectively free.
        """
        text = (self._STATIC_DIR / name).read_text(encoding="utf-8")
        return text.replace("__WS_PORT__", str(self._ws_port)).encode("utf-8")

    async def _route_dashboard(self, body: bytes) -> bytes:
        return self._http_response(
            200, "text/html; charset=utf-8", self._read_static_html("dashboard.html")
        )

    async def _route_mobile(self, body: bytes) -> bytes:
        return self._http_response(
            200, "text/html; charset=utf-8", self._read_static_html("mobile.html")
        )

    async def _route_eval_page(self, body: bytes) -> bytes:
        return self._http_response(
            200, "text/html; charset=utf-8", self._read_static_html("eval.html")
        )

    # ------------------------------------------------------------------
    # Eval API — read-only surface over .nature/eval/results
    # ------------------------------------------------------------------
    #
    # Reads the JSON run records the CLI writes; does not drive new
    # runs (those stay CLI-driven for now). The dashboard uses these
    # to render the runs list, matrix view, and diff comparison.

    async def _route_eval_list_runs(self, body: bytes) -> bytes:
        from nature.eval.results import list_runs
        paths = list_runs(project_dir=Path(self._registry.cwd))
        summaries: list[dict] = []
        for path in paths:
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            cells = doc.get("cells", [])
            summaries.append({
                "run_id": doc.get("run_id", path.stem),
                "started_at": doc.get("started_at"),
                "finished_at": doc.get("finished_at"),
                "repo_git_sha": doc.get("repo_git_sha"),
                "task_ids": doc.get("task_ids", []),
                "preset_names": doc.get("preset_names", []),
                "cell_count": len(cells),
                "pass_count": sum(1 for c in cells if c.get("passed")),
                "error_count": sum(1 for c in cells if c.get("error")),
            })
        return self._http_json_response(200, {"runs": summaries})

    async def _route_eval_get_run(self, body: bytes, run_id: str) -> bytes:
        from nature.eval.results import load_run
        try:
            doc = load_run(run_id, project_dir=Path(self._registry.cwd))
        except FileNotFoundError:
            return self._http_json_response(
                404,
                ErrorResponse(error="run_not_found", detail=run_id).model_dump(),
            )
        return self._http_json_response(200, doc)

    async def _route_eval_get_cell(
        self, body: bytes, run_id: str, task_id: str, preset: str,
    ) -> bytes:
        """Return one cell's full context for in-place inspection.

        Payload:
        - `cell`     — CellResult dict as stored in the run record.
        - `edits`    — list of {tool, file_path, old, new, event_id}
                       extracted from Edit/Write tool_started events,
                       ready for diff rendering.
        - `frames`   — frame tree + turn decomposition per frame (see
                       `nature.eval.inspect.build_cell_inspection`).
                       This is the primary structure the dashboard
                       renders — flat timeline below is retained only
                       as a raw-debug fallback.
        - `timeline` — compact per-event summary in logical order.
        """
        from nature.eval.results import load_run
        try:
            doc = load_run(run_id, project_dir=Path(self._registry.cwd))
        except FileNotFoundError:
            return self._http_json_response(
                404,
                ErrorResponse(
                    error="run_not_found", detail=run_id,
                ).model_dump(),
            )
        cell = next(
            (
                c for c in doc.get("cells", [])
                if c.get("task_id") == task_id and c.get("preset") == preset
            ),
            None,
        )
        if cell is None:
            return self._http_json_response(
                404,
                ErrorResponse(
                    error="cell_not_found",
                    detail=f"{task_id} × {preset}",
                ).model_dump(),
            )

        log_path_str = cell.get("event_log_path") or ""
        edits: list[dict] = []
        timeline: list[dict] = []
        role_by_frame: dict[str, str] = {}
        frames_structure: dict = {"root_frame_id": None, "frames": []}
        if log_path_str:
            from pathlib import Path as _P
            log_path = _P(log_path_str)
            if log_path.exists():
                try:
                    from nature.eval.inspect import build_cell_inspection
                    frames_structure = build_cell_inspection(log_path)
                except Exception:  # noqa: BLE001
                    logger.exception("cell inspection failed")
                try:
                    for line in log_path.read_text(encoding="utf-8").splitlines():
                        try:
                            ev = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        t = ev.get("type")
                        p = ev.get("payload") or {}
                        fid = ev.get("frame_id")
                        role = role_by_frame.get(fid or "", "?")

                        if t == "frame.opened":
                            name = p.get("role_name") or "?"
                            role_by_frame[fid or ""] = name
                            timeline.append({
                                "id": ev.get("id"), "type": t,
                                "role": name,
                                "model": p.get("model"),
                                "parent_id": p.get("parent_id"),
                            })
                        elif t == "frame.resolved":
                            timeline.append({
                                "id": ev.get("id"), "type": t, "role": role,
                            })
                        elif t == "frame.errored":
                            timeline.append({
                                "id": ev.get("id"), "type": t, "role": role,
                                "error_type": p.get("error_type"),
                                "message": p.get("message", "")[:500],
                            })
                        elif t == "llm.response":
                            timeline.append({
                                "id": ev.get("id"), "type": t, "role": role,
                                "stop_reason": p.get("stop_reason"),
                            })
                        elif t == "llm.error":
                            timeline.append({
                                "id": ev.get("id"), "type": t, "role": role,
                                "error_type": p.get("error_type"),
                                "message": p.get("message", "")[:400],
                            })
                        elif t == "tool.started":
                            ti = p.get("tool_input") or {}
                            name = p.get("tool_name")
                            timeline.append({
                                "id": ev.get("id"), "type": t, "role": role,
                                "tool_name": name,
                                "tool_input": ti,
                            })
                            if name in ("Edit", "Write"):
                                edits.append({
                                    "event_id": ev.get("id"),
                                    "tool": name,
                                    "role": role,
                                    "file_path": ti.get("file_path", ""),
                                    "old": ti.get("old_string", ""),
                                    "new": (
                                        ti.get("new_string", "")
                                        if name == "Edit"
                                        else ti.get("content", "")
                                    ),
                                })
                        elif t == "tool.completed":
                            timeline.append({
                                "id": ev.get("id"), "type": t, "role": role,
                                "tool_name": p.get("tool_name"),
                                "is_error": bool(p.get("is_error")),
                                "output": str(p.get("output", ""))[:400],
                            })
                        elif t == "hint.injected":
                            hints = p.get("hints") or []
                            timeline.append({
                                "id": ev.get("id"), "type": t, "role": role,
                                "hints": [
                                    {
                                        "source": h.get("source") if isinstance(h, dict) else "?",
                                        "text": (
                                            h.get("text", "")[:200]
                                            if isinstance(h, dict) else ""
                                        ),
                                    }
                                    for h in hints
                                ],
                            })
                        elif t == "body.compacted":
                            timeline.append({
                                "id": ev.get("id"), "type": t, "role": role,
                                "strategy": p.get("strategy"),
                                "tokens_before": p.get("tokens_before"),
                                "tokens_after": p.get("tokens_after"),
                            })
                except Exception as exc:  # noqa: BLE001
                    logger.exception("eval cell timeline parse failed")

        return self._http_json_response(200, {
            "cell": cell,
            "edits": edits,
            "frames": frames_structure.get("frames", []),
            "root_frame_id": frames_structure.get("root_frame_id"),
            "timeline": timeline,
        })

    async def _route_eval_get_turn_request(
        self,
        body: bytes,
        run_id: str,
        task_id: str,
        preset: str,
        frame_id: str,
        turn_index: str,
    ) -> bytes:
        """Return the *full* LLM request body reconstructed from events
        for (run, cell, frame, turn). Heavier than the cell-inspection
        payload, so clients lazy-fetch it on "full request" click."""
        from nature.eval.results import load_run
        try:
            doc = load_run(run_id, project_dir=Path(self._registry.cwd))
        except FileNotFoundError:
            return self._http_json_response(
                404, ErrorResponse(error="run_not_found", detail=run_id).model_dump(),
            )
        cell = next(
            (
                c for c in doc.get("cells", [])
                if c.get("task_id") == task_id and c.get("preset") == preset
            ),
            None,
        )
        if cell is None:
            return self._http_json_response(
                404, ErrorResponse(
                    error="cell_not_found", detail=f"{task_id} × {preset}",
                ).model_dump(),
            )
        log_path_str = cell.get("event_log_path") or ""
        if not log_path_str:
            return self._http_json_response(
                404, ErrorResponse(error="no_event_log", detail="").model_dump(),
            )
        from pathlib import Path as _P
        from nature.eval.inspect import build_turn_request
        try:
            turn_idx = int(turn_index)
        except ValueError:
            return self._http_json_response(
                400, ErrorResponse(
                    error="bad_turn_index", detail=turn_index,
                ).model_dump(),
            )
        try:
            payload = build_turn_request(_P(log_path_str), frame_id, turn_idx)
        except Exception:  # noqa: BLE001
            logger.exception("turn request reconstruction failed")
            return self._http_json_response(
                500, ErrorResponse(
                    error="reconstruct_failed", detail="",
                ).model_dump(),
            )
        if payload is None:
            return self._http_json_response(
                404, ErrorResponse(
                    error="turn_not_found",
                    detail=f"{frame_id} #{turn_index}",
                ).model_dump(),
            )
        return self._http_json_response(200, payload)

    async def _route_eval_diff(self, body: bytes, query_string: str = "") -> bytes:
        from urllib.parse import parse_qs
        from nature.eval.diff import diff_runs
        from nature.eval.results import load_run
        params = parse_qs(query_string or "")
        a = (params.get("a") or [""])[0]
        b = (params.get("b") or [""])[0]
        if not a or not b:
            return self._http_json_response(
                400,
                ErrorResponse(
                    error="bad_request",
                    detail="diff requires both ?a= and ?b= run ids",
                ).model_dump(),
            )
        try:
            run_a = load_run(a, project_dir=Path(self._registry.cwd))
            run_b = load_run(b, project_dir=Path(self._registry.cwd))
        except FileNotFoundError as exc:
            return self._http_json_response(
                404,
                ErrorResponse(error="run_not_found", detail=str(exc)).model_dump(),
            )
        return self._http_json_response(200, {
            "a": a, "b": b,
            "markdown": diff_runs(run_a, run_b),
        })

    # ------------------------------------------------------------------
    # Probe API — read-only surface over .nature/probe/results
    # ------------------------------------------------------------------

    async def _route_probe_page(self, body: bytes) -> bytes:
        return self._http_response(
            200, "text/html; charset=utf-8", self._read_static_html("probe.html")
        )

    async def _route_probe_list_runs(self, body: bytes) -> bytes:
        from nature.probe.results import list_runs
        runs = list_runs(Path(self._registry.cwd))
        # list_runs returns lightweight metadata already, but we
        # augment with pass counts per run so the sidebar can render
        # "27/29" style summaries without a second fetch per row.
        from nature.probe.results import runs_dir
        enriched: list[dict] = []
        for r in runs:
            rid = r.get("run_id")
            if not rid:
                continue
            try:
                doc = json.loads(
                    (runs_dir(Path(self._registry.cwd)) / f"{rid}.json").read_text(encoding="utf-8")
                )
            except Exception:  # noqa: BLE001
                enriched.append(r)
                continue
            cells = doc.get("cells", [])
            enriched.append({
                **r,
                "pass_count": sum(1 for c in cells if c.get("passed")),
                "models": doc.get("models", []),
            })
        return self._http_json_response(200, {"runs": enriched})

    async def _route_probe_get_run(self, body: bytes, run_id: str) -> bytes:
        from nature.probe.results import runs_dir
        path = runs_dir(Path(self._registry.cwd)) / f"{run_id}.json"
        if not path.exists():
            return self._http_json_response(
                404,
                ErrorResponse(error="run_not_found", detail=run_id).model_dump(),
            )
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            return self._http_json_response(
                500,
                ErrorResponse(error="run_parse_failed", detail=str(exc)).model_dump(),
            )
        return self._http_json_response(200, doc)

    async def _route_probe_list_probes(self, body: bytes) -> bytes:
        """Return the probe registry (tier + dimensions lookup for UI)."""
        from nature.probe.probes import load_probes
        probes = load_probes(project_dir=Path(self._registry.cwd))
        out = [
            {
                "id": p.id,
                "title": p.title,
                "tier": p.tier,
                "dimensions": p.dimensions,
                "allowed_tools": p.allowed_tools,
                "prompt": p.prompt,
            }
            for p in sorted(probes.values(), key=lambda x: (x.tier, x.id))
        ]
        return self._http_json_response(200, {"probes": out})

    async def _route_probe_combined(self, body: bytes) -> bytes:
        """Stitch cells across all probe runs with a latest-wins rule
        per (probe_id, model_ref). Lets the dashboard render a
        single table that reflects the best data we have for each
        cell, even when recent runs only covered a subset of models."""
        from nature.probe.results import list_runs, runs_dir
        cells_by_key: dict[tuple[str, str], dict] = {}
        runs = list_runs(Path(self._registry.cwd))
        run_ids_used = []
        for r in runs:
            rid = r.get("run_id")
            if not rid:
                continue
            path = runs_dir(Path(self._registry.cwd)) / f"{rid}.json"
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            run_ids_used.append(rid)
            for c in doc.get("cells", []):
                key = (c.get("probe_id"), c.get("model_ref"))
                if not all(key):
                    continue
                prev = cells_by_key.get(key)
                # Later run wins; within a run, just keep what's there.
                if prev is None or doc.get("started_at", 0) > prev.get("_started_at", 0):
                    cells_by_key[key] = {**c, "_started_at": doc.get("started_at", 0)}
        cells = [
            {k: v for k, v in c.items() if k != "_started_at"}
            for c in cells_by_key.values()
        ]
        models = sorted({c["model_ref"] for c in cells})
        probe_ids = sorted({c["probe_id"] for c in cells})
        return self._http_json_response(200, {
            "cells": cells,
            "models": models,
            "probe_ids": probe_ids,
            "source_runs": run_ids_used,
        })

    async def _route_list_sessions(self, body: bytes) -> bytes:
        infos = [self._session_to_info(s) for s in self._registry.list()]
        return self._http_json_response(
            200, ListSessionsResponse(sessions=infos).model_dump()
        )

    async def _route_create_session(self, body: bytes) -> bytes:
        try:
            payload = json.loads(body.decode("utf-8") or "{}")
        except json.JSONDecodeError as exc:
            return self._http_json_response(
                400, ErrorResponse(error="bad_json", detail=str(exc)).model_dump()
            )
        try:
            req = CreateSessionRequest(**payload)
        except Exception as exc:
            return self._http_json_response(
                400, ErrorResponse(error="bad_request", detail=str(exc)).model_dump()
            )
        try:
            session = await self._registry.create_session(req)
        except FileNotFoundError as exc:
            # Most commonly: requested preset name does not exist.
            return self._http_json_response(
                404,
                ErrorResponse(error="preset_not_found", detail=str(exc)).model_dump(),
            )
        except PresetValidationError as exc:
            # Dangling ref in the preset (unknown agent / host / API key).
            return self._http_json_response(
                400,
                ErrorResponse(error="invalid_preset", detail=str(exc)).model_dump(),
            )
        except ValueError as exc:
            return self._http_json_response(
                400, ErrorResponse(error="invalid_config", detail=str(exc)).model_dump()
            )
        except Exception as exc:
            logger.exception("session creation failed")
            return self._http_json_response(
                500, ErrorResponse(error="create_failed", detail=str(exc)).model_dump()
            )
        return self._http_json_response(
            200,
            CreateSessionResponse(
                session_id=session.session_id,
                root_role_name=session.root_role.name,
                root_model=session.root_model,
                provider_name=session.provider_name,
                base_url=session.base_url,
                created_at=session.created_at,
            ).model_dump(),
        )

    async def _route_get_session(self, body: bytes, sid: str) -> bytes:
        session = self._registry.get(sid)
        if session is None:
            return self._http_json_response(
                404, ErrorResponse(error="not_found", detail=sid).model_dump()
            )
        return self._http_json_response(200, self._session_to_info(session).model_dump())

    async def _route_send_message(self, body: bytes, sid: str) -> bytes:
        try:
            payload = json.loads(body.decode("utf-8") or "{}")
            req = SendMessageRequest(**payload)
        except Exception as exc:
            return self._http_json_response(
                400, ErrorResponse(error="bad_request", detail=str(exc)).model_dump()
            )
        try:
            await self._registry.send_message(sid, req.text)
        except KeyError:
            return self._http_json_response(
                404, ErrorResponse(error="not_found", detail=sid).model_dump()
            )
        except RuntimeError as exc:
            return self._http_json_response(
                409, ErrorResponse(error="conflict", detail=str(exc)).model_dump()
            )
        return self._http_json_response(202, OkResponse().model_dump())

    async def _route_cancel(self, body: bytes, sid: str) -> bytes:
        try:
            await self._registry.cancel(sid)
        except KeyError:
            return self._http_json_response(
                404, ErrorResponse(error="not_found", detail=sid).model_dump()
            )
        return self._http_json_response(200, OkResponse().model_dump())

    async def _route_close_session(self, body: bytes, sid: str) -> bytes:
        await self._registry.close_session(sid)
        return self._http_json_response(200, OkResponse().model_dump())

    async def _route_snapshot(self, body: bytes, sid: str) -> bytes:
        events = self._store.snapshot(sid)
        payload = {"events": [self._event_to_json(e) for e in events]}
        return self._http_json_response(200, payload)

    async def _route_frame_context(
        self,
        body: bytes,
        sid: str,
        fid: str,
        query_string: str = "",
    ) -> bytes:
        # Optional ?up_to=N query param for time-travel replay. The
        # dashboard's scrubber drops this in; passing it through to
        # the registry means the reconstruction is sliced at the
        # event layer (no "full rebuild then filter" overhead).
        up_to = _parse_up_to(query_string)
        ctx = self._registry.get_frame_context(
            sid, fid, up_to_event_id=up_to,
        )
        if ctx is None:
            return self._http_json_response(
                404,
                ErrorResponse(
                    error="frame_not_found", detail=f"{sid}/{fid}"
                ).model_dump(),
            )
        return self._http_json_response(200, ctx)

    async def _route_list_archived(self, body: bytes) -> bytes:
        from nature.server.registry import archived_preview_from_events

        archived = []
        for meta in self._registry.list_archived():
            preview = archived_preview_from_events(
                self._store.snapshot(meta.session_id)
            )
            archived.append(ArchivedSessionInfo(
                session_id=meta.session_id,
                event_count=meta.event_count,
                created_at=meta.created_at,
                last_event_at=meta.last_event_at,
                preview=preview,
                parent_session_id=meta.parent_session_id,
                forked_from_event_id=meta.forked_from_event_id,
            ))
        # Most recent first
        archived.sort(key=lambda a: a.last_event_at, reverse=True)
        return self._http_json_response(
            200, ListArchivedSessionsResponse(sessions=archived).model_dump()
        )

    async def _route_resume(self, body: bytes, sid: str) -> bytes:
        # Optional override body — same shape as CreateSessionRequest.
        # Empty body → no overrides (keeps the prior behavior).
        req: CreateSessionRequest | None = None
        if body:
            try:
                payload = json.loads(body.decode("utf-8") or "{}")
                req = CreateSessionRequest(**payload)
            except json.JSONDecodeError as exc:
                return self._http_json_response(
                    400,
                    ErrorResponse(error="bad_json", detail=str(exc)).model_dump(),
                )
            except Exception as exc:
                return self._http_json_response(
                    400,
                    ErrorResponse(
                        error="bad_request", detail=str(exc)
                    ).model_dump(),
                )
        try:
            session = await self._registry.resume_session(sid, req)
        except KeyError:
            return self._http_json_response(
                404,
                ErrorResponse(
                    error="not_found",
                    detail=f"no events for session {sid}",
                ).model_dump(),
            )
        except FileNotFoundError as exc:
            return self._http_json_response(
                404,
                ErrorResponse(
                    error="preset_not_found", detail=str(exc),
                ).model_dump(),
            )
        except PresetValidationError as exc:
            return self._http_json_response(
                400,
                ErrorResponse(
                    error="invalid_preset", detail=str(exc),
                ).model_dump(),
            )
        except ValueError as exc:
            return self._http_json_response(
                400,
                ErrorResponse(
                    error="invalid_config", detail=str(exc)
                ).model_dump(),
            )
        except Exception as exc:
            logger.exception("resume failed for %s", sid)
            return self._http_json_response(
                500,
                ErrorResponse(
                    error="resume_failed", detail=str(exc)
                ).model_dump(),
            )
        return self._http_json_response(
            200,
            CreateSessionResponse(
                session_id=session.session_id,
                root_role_name=session.root_role.name,
                root_model=session.root_model,
                provider_name=session.provider_name,
                base_url=session.base_url,
                created_at=session.created_at,
            ).model_dump(),
        )

    async def _route_fork(self, body: bytes, sid: str) -> bytes:
        """POST /api/sessions/{sid}/fork — branch a new session at an event.

        Body: `{"at_event_id": <int>}`. Copies events 1..at_event_id
        from the source session into a fresh session with a new id,
        writes the fork lineage sidecar, and hydrates the new session
        live via the existing resume path so it's immediately usable.
        """
        try:
            payload = json.loads(body.decode("utf-8") or "{}")
            req = ForkSessionRequest(**payload)
        except json.JSONDecodeError as exc:
            return self._http_json_response(
                400,
                ErrorResponse(error="bad_json", detail=str(exc)).model_dump(),
            )
        except Exception as exc:
            return self._http_json_response(
                400,
                ErrorResponse(
                    error="bad_request", detail=str(exc)
                ).model_dump(),
            )
        try:
            session = await self._registry.fork_session(
                sid, at_event_id=req.at_event_id, preset=req.preset,
            )
        except KeyError:
            return self._http_json_response(
                404,
                ErrorResponse(
                    error="not_found",
                    detail=f"no events for source session {sid}",
                ).model_dump(),
            )
        except ValueError as exc:
            return self._http_json_response(
                400,
                ErrorResponse(
                    error="invalid_fork", detail=str(exc)
                ).model_dump(),
            )
        except Exception as exc:
            logger.exception("fork failed for %s", sid)
            return self._http_json_response(
                500,
                ErrorResponse(
                    error="fork_failed", detail=str(exc)
                ).model_dump(),
            )
        return self._http_json_response(
            200,
            CreateSessionResponse(
                session_id=session.session_id,
                root_role_name=session.root_role.name,
                root_model=session.root_model,
                provider_name=session.provider_name,
                base_url=session.base_url,
                created_at=session.created_at,
                parent_session_id=sid,
                forked_from_event_id=req.at_event_id,
            ).model_dump(),
        )

    # ------------------------------------------------------------------
    # Model / tool catalogs (used by future admin UI dropdowns)
    # ------------------------------------------------------------------

    async def _route_list_models(self, body: bytes) -> bytes:
        """GET /api/config/models — curated dropdown entries."""
        from nature.config.constants import MODEL_CATALOG
        entries = [ModelCatalogEntry(**m) for m in MODEL_CATALOG]
        return self._http_json_response(
            200,
            ListModelsResponse(models=entries).model_dump(),
        )

    async def _route_list_tools(self, body: bytes) -> bytes:
        """GET /api/config/tools — tool names from the live registry."""
        names = self._registry.list_available_tool_names()
        return self._http_json_response(
            200,
            ListToolsResponse(tools=names).model_dump(),
        )

    # ------------------------------------------------------------------
    # Admin API — file-based edits of agents / presets / hosts
    # ------------------------------------------------------------------
    #
    # Reads merge builtin/user/project (project wins) and surface the
    # effective origin. Writes always land in the user layer
    # (`~/.nature/`); builtin entries are read-only and deletion
    # returns 403 when a same-named builtin exists.

    async def _route_admin_list_agents(self, body: bytes) -> bytes:
        from nature.agents.config import load_agents_with_origin

        agents = load_agents_with_origin(project_dir=Path(self._registry.cwd))
        infos = [
            AgentAdminInfo(
                name=cfg.name,
                model=cfg.model,
                allowed_tools=cfg.allowed_tools,
                allowed_interventions=cfg.allowed_interventions,
                instructions_text=cfg.instructions_text,
                description=cfg.description,
                origin=origin,
            )
            for cfg, origin in agents
        ]
        return self._http_json_response(
            200,
            ListAgentsAdminResponse(agents=infos).model_dump(),
        )

    async def _route_admin_get_agent(self, body: bytes, name: str) -> bytes:
        from nature.agents.config import load_agents_with_origin

        for cfg, origin in load_agents_with_origin(project_dir=Path(self._registry.cwd)):
            if cfg.name == name:
                return self._http_json_response(
                    200,
                    AgentAdminInfo(
                        name=cfg.name,
                        model=cfg.model,
                        allowed_tools=cfg.allowed_tools,
                        allowed_interventions=cfg.allowed_interventions,
                        instructions_text=cfg.instructions_text,
                        description=cfg.description,
                        origin=origin,
                    ).model_dump(),
                )
        return self._http_json_response(
            404,
            ErrorResponse(error="not_found", detail=name).model_dump(),
        )

    async def _route_admin_put_agent(self, body: bytes, name: str) -> bytes:
        from nature.agents.config import write_user_agent

        try:
            payload = json.loads(body.decode("utf-8") or "{}")
            req = AgentPutRequest.model_validate(payload)
        except json.JSONDecodeError as exc:
            return self._http_json_response(
                400,
                ErrorResponse(error="bad_json", detail=str(exc)).model_dump(),
            )
        except Exception as exc:
            return self._http_json_response(
                400,
                ErrorResponse(error="bad_request", detail=str(exc)).model_dump(),
            )
        try:
            write_user_agent(
                name,
                model=req.model,
                allowed_tools=req.allowed_tools,
                allowed_interventions=req.allowed_interventions,
                instructions_text=req.instructions_text,
                description=req.description,
            )
        except Exception as exc:
            logger.exception("write_user_agent failed for %s", name)
            return self._http_json_response(
                500,
                ErrorResponse(error="write_failed", detail=str(exc)).model_dump(),
            )
        return self._http_json_response(200, OkResponse().model_dump())

    async def _route_admin_delete_agent(self, body: bytes, name: str) -> bytes:
        from nature.agents.config import delete_user_agent

        if not delete_user_agent(name):
            # Nothing to delete in the user layer. If a builtin or
            # project entry by this name still exists, say so with 403;
            # otherwise the name is simply unknown → 404.
            from nature.agents.config import load_agents_with_origin
            for cfg, origin in load_agents_with_origin(
                project_dir=Path(self._registry.cwd),
            ):
                if cfg.name == name:
                    return self._http_json_response(
                        403,
                        ErrorResponse(
                            error="read_only_layer",
                            detail=(
                                f"agent {name!r} is defined in the {origin} "
                                f"layer and cannot be deleted via the admin API"
                            ),
                        ).model_dump(),
                    )
            return self._http_json_response(
                404,
                ErrorResponse(error="not_found", detail=name).model_dump(),
            )
        return self._http_json_response(200, OkResponse().model_dump())

    async def _route_admin_list_presets(self, body: bytes) -> bytes:
        from nature.agents.presets import load_presets_with_origin

        try:
            entries = load_presets_with_origin(project_dir=Path(self._registry.cwd))
        except ValueError as exc:
            return self._http_json_response(
                500,
                ErrorResponse(error="preset_corrupt", detail=str(exc)).model_dump(),
            )
        infos = [
            PresetAdminInfo(
                name=p.name,
                root_agent=p.root_agent,
                agents=p.agents,
                model_overrides=p.model_overrides,
                origin=origin,
            )
            for p, origin in entries
        ]
        return self._http_json_response(
            200,
            ListPresetsAdminResponse(presets=infos).model_dump(),
        )

    async def _route_admin_get_preset(self, body: bytes, name: str) -> bytes:
        from nature.agents.presets import load_presets_with_origin

        try:
            entries = load_presets_with_origin(project_dir=Path(self._registry.cwd))
        except ValueError as exc:
            return self._http_json_response(
                500,
                ErrorResponse(error="preset_corrupt", detail=str(exc)).model_dump(),
            )
        for p, origin in entries:
            if p.name == name:
                return self._http_json_response(
                    200,
                    PresetAdminInfo(
                        name=p.name,
                        root_agent=p.root_agent,
                        agents=p.agents,
                        model_overrides=p.model_overrides,
                        origin=origin,
                    ).model_dump(),
                )
        return self._http_json_response(
            404,
            ErrorResponse(error="not_found", detail=name).model_dump(),
        )

    async def _route_admin_put_preset(self, body: bytes, name: str) -> bytes:
        from nature.agents.presets import write_user_preset

        try:
            payload = json.loads(body.decode("utf-8") or "{}")
            req = PresetPutRequest.model_validate(payload)
        except json.JSONDecodeError as exc:
            return self._http_json_response(
                400,
                ErrorResponse(error="bad_json", detail=str(exc)).model_dump(),
            )
        except Exception as exc:
            return self._http_json_response(
                400,
                ErrorResponse(error="bad_request", detail=str(exc)).model_dump(),
            )
        try:
            write_user_preset(name, req.model_dump())
        except ValueError as exc:
            return self._http_json_response(
                400,
                ErrorResponse(error="invalid_preset", detail=str(exc)).model_dump(),
            )
        except Exception as exc:
            logger.exception("write_user_preset failed for %s", name)
            return self._http_json_response(
                500,
                ErrorResponse(error="write_failed", detail=str(exc)).model_dump(),
            )
        return self._http_json_response(200, OkResponse().model_dump())

    async def _route_admin_delete_preset(self, body: bytes, name: str) -> bytes:
        from nature.agents.presets import (
            delete_user_preset, load_presets_with_origin,
        )

        if not delete_user_preset(name):
            for p, origin in load_presets_with_origin(
                project_dir=Path(self._registry.cwd),
            ):
                if p.name == name:
                    return self._http_json_response(
                        403,
                        ErrorResponse(
                            error="read_only_layer",
                            detail=(
                                f"preset {name!r} is defined in the {origin} "
                                f"layer and cannot be deleted via the admin API"
                            ),
                        ).model_dump(),
                    )
            return self._http_json_response(
                404,
                ErrorResponse(error="not_found", detail=name).model_dump(),
            )
        return self._http_json_response(200, OkResponse().model_dump())

    async def _route_admin_list_hosts(self, body: bytes) -> bytes:
        from nature.config.hosts import load_hosts_with_origin

        try:
            entries, default_host = load_hosts_with_origin(
                project_dir=Path(self._registry.cwd),
            )
        except ValueError as exc:
            return self._http_json_response(
                500,
                ErrorResponse(error="hosts_corrupt", detail=str(exc)).model_dump(),
            )
        hosts = [
            HostAdminInfo(
                name=n, provider=c.provider, base_url=c.base_url,
                api_key_env=c.api_key_env, models=list(c.models), origin=origin,
            )
            for n, c, origin in entries
        ]
        return self._http_json_response(
            200,
            ListHostsAdminResponse(
                hosts=hosts, default_host=default_host,
            ).model_dump(),
        )

    async def _route_admin_put_host(self, body: bytes, name: str) -> bytes:
        from nature.config.hosts import (
            HostConfig, load_user_hosts_config, save_user_hosts_config,
        )

        try:
            payload = json.loads(body.decode("utf-8") or "{}")
            req = HostPutRequest.model_validate(payload)
        except json.JSONDecodeError as exc:
            return self._http_json_response(
                400,
                ErrorResponse(error="bad_json", detail=str(exc)).model_dump(),
            )
        except Exception as exc:
            return self._http_json_response(
                400,
                ErrorResponse(error="bad_request", detail=str(exc)).model_dump(),
            )
        try:
            user_cfg = load_user_hosts_config()
            user_cfg.hosts[name] = HostConfig(
                provider=req.provider,
                base_url=req.base_url,
                api_key_env=req.api_key_env,
                models=list(req.models),
            )
            save_user_hosts_config(user_cfg)
        except Exception as exc:
            logger.exception("put host failed for %s", name)
            return self._http_json_response(
                500,
                ErrorResponse(error="write_failed", detail=str(exc)).model_dump(),
            )
        return self._http_json_response(200, OkResponse().model_dump())

    async def _route_admin_delete_host(self, body: bytes, name: str) -> bytes:
        from nature.config.hosts import (
            delete_user_host, is_builtin, load_hosts_with_origin,
        )

        if not delete_user_host(name):
            # Nothing in the user layer — decide between 403 (builtin
            # or project shadows it) and 404 (unknown name).
            if is_builtin(name):
                return self._http_json_response(
                    403,
                    ErrorResponse(
                        error="read_only_layer",
                        detail=(
                            f"host {name!r} is a builtin and cannot be "
                            f"deleted via the admin API"
                        ),
                    ).model_dump(),
                )
            for hname, _cfg, origin in load_hosts_with_origin(
                project_dir=Path(self._registry.cwd),
            )[0]:
                if hname == name:
                    return self._http_json_response(
                        403,
                        ErrorResponse(
                            error="read_only_layer",
                            detail=(
                                f"host {name!r} is defined in the {origin} "
                                f"layer and cannot be deleted via the admin API"
                            ),
                        ).model_dump(),
                    )
            return self._http_json_response(
                404,
                ErrorResponse(error="not_found", detail=name).model_dump(),
            )
        return self._http_json_response(200, OkResponse().model_dump())

    # ------------------------------------------------------------------
    # WebSocket
    # ------------------------------------------------------------------

    _WS_PATH_RE = re.compile(r"^/ws/sessions/(?P<sid>[A-Za-z0-9_-]+)$")
    _WS_VIEW_PATH_RE = re.compile(r"^/ws/view/sessions/(?P<sid>[A-Za-z0-9_-]+)$")

    async def _handle_ws_connection(self, websocket: Any) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._ws_handler_tasks.add(task)
        try:
            path = self._ws_request_path(websocket) or ""
            view_match = self._WS_VIEW_PATH_RE.match(path)
            if view_match:
                await self._handle_ws_view(websocket, view_match.group("sid"))
                return
            raw_match = self._WS_PATH_RE.match(path)
            if raw_match:
                await self._handle_ws_raw(websocket, raw_match.group("sid"))
                return
            await websocket.close(code=4404, reason="not found")
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.debug("ws handler error: %s", exc)
        finally:
            if task is not None:
                self._ws_handler_tasks.discard(task)

    async def _handle_ws_raw(self, websocket: Any, sid: str) -> None:
        """Legacy raw-event stream. Mobile dashboard + tests still consume
        this — desktop has moved to the structured view channel."""
        session = self._registry.get(sid)
        if session is not None:
            meta = {
                "__kind__": "session_meta",
                "session_id": sid,
                "model": session.root_model,
                "role_name": session.root_role.name,
                "provider": session.provider_name,
            }
        else:
            meta = {
                "__kind__": "session_meta",
                "session_id": sid,
                "model": "",
                "role_name": "",
                "provider": "",
            }
        await websocket.send(json.dumps(meta))

        async for event in self._store.live_tail(sid):
            try:
                await websocket.send(json.dumps(self._event_to_json(event)))
            except Exception:
                break

    async def _handle_ws_view(self, websocket: Any, sid: str) -> None:
        """Structured session-view channel.

        On every appended event we rebuild the full SessionViewDto from
        the accumulated log and push it. Stateless frontend: it just
        renders the last DTO it saw. The per-event rebuild cost is
        O(events) in Python, but for realistic session sizes the DTO
        is small enough that this is cheaper than maintaining a diffing
        layer — we can revisit if a single session grows pathological.
        """
        from nature.server.view import build_session_view

        session = self._registry.get(sid)
        role_name = session.root_role.name if session is not None else ""
        model = session.root_model if session is not None else ""
        provider = session.provider_name if session is not None else ""

        events: list[Event] = []
        async for event in self._store.live_tail(sid):
            events.append(event)
            view = build_session_view(
                events,
                session_id=sid,
                role_name=role_name,
                model=model,
                provider=provider,
            )
            try:
                await websocket.send(view.model_dump_json())
            except Exception:
                break

    @staticmethod
    def _ws_request_path(websocket: Any) -> str | None:
        # websockets ≥ 13
        request = getattr(websocket, "request", None)
        if request is not None:
            return getattr(request, "path", None)
        # older versions
        return getattr(websocket, "path", None)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _session_to_info(self, session: ServerSession) -> SessionInfo:
        meta = self._registry.event_store.get_session_meta(session.session_id)
        parent = meta.parent_session_id if meta else None
        forked_from = meta.forked_from_event_id if meta else None
        return SessionInfo(
            session_id=session.session_id,
            root_role_name=session.root_role.name,
            root_model=session.root_model,
            state=session.state,
            has_active_run=session.has_active_run,
            created_at=session.created_at,
            preview=session.preview,
            parent_session_id=parent,
            forked_from_event_id=forked_from,
        )

    @staticmethod
    def _event_to_json(event: Event) -> dict[str, Any]:
        return {
            "id": event.id,
            "type": event.type.value,
            "frame_id": event.frame_id,
            "timestamp": event.timestamp,
            "payload": event.payload,
        }

    @staticmethod
    def _http_response(
        status: int, content_type: str, body: bytes
    ) -> bytes:
        status_text = {
            200: "OK", 202: "Accepted", 204: "No Content",
            400: "Bad Request", 404: "Not Found", 409: "Conflict",
            500: "Internal Server Error",
        }.get(status, "OK")
        return (
            f"HTTP/1.1 {status} {status_text}\r\n".encode("latin-1")
            + f"Content-Type: {content_type}\r\n".encode("latin-1")
            + f"Content-Length: {len(body)}\r\n".encode("latin-1")
            + b"Connection: close\r\n"
            + b"Access-Control-Allow-Origin: *\r\n"
            + b"\r\n"
            + body
        )

    @classmethod
    def _http_json_response(cls, status: int, payload: dict) -> bytes:
        body = json.dumps(payload, default=str, ensure_ascii=False).encode("utf-8")
        return cls._http_response(status, "application/json; charset=utf-8", body)
