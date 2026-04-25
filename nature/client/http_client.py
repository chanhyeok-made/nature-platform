"""NatureClient — HTTP + WebSocket client for the nature server.

Thin wrapper around httpx (HTTP) and websockets (events). Reuses the
shared API models from nature.server.api so request/response shapes
are single-source.

Connection model: client doesn't own any background tasks except the
async iterator returned by stream_events(). The caller drives both
HTTP calls and event iteration explicitly.
"""

from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from nature.events.types import Event, EventType
from nature.server.api import (
    ArchivedSessionInfo,
    CreateSessionRequest,
    CreateSessionResponse,
    ErrorResponse,
    ListArchivedSessionsResponse,
    ListSessionsResponse,
    SendMessageRequest,
    SessionInfo,
)


class NatureClientError(Exception):
    """Base exception for client / server interaction failures."""


class ServerNotRunning(NatureClientError):
    """Raised when the server is unreachable at the configured URL."""


class NatureClient:
    """Async client for the nature server."""

    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = 7777,
        timeout: float = 30.0,
    ) -> None:
        self._host = host
        self._port = port
        self._http_url = f"http://{host}:{port}"
        self._ws_url = f"ws://{host}:{port + 1}"
        self._http = httpx.AsyncClient(base_url=self._http_url, timeout=timeout)

    @property
    def http_url(self) -> str:
        return self._http_url

    @property
    def ws_url(self) -> str:
        return self._ws_url

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "NatureClient":
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def is_alive(self) -> bool:
        """Quick check: does the server respond on /api/sessions?"""
        try:
            r = await self._http.get("/api/sessions", timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    async def list_sessions(self) -> list[SessionInfo]:
        r = await self._get("/api/sessions")
        return ListSessionsResponse(**r).sessions

    async def list_archived_sessions(self) -> list[ArchivedSessionInfo]:
        """Sessions on disk but not currently in the live registry."""
        r = await self._get("/api/sessions/archived")
        return ListArchivedSessionsResponse(**r).sessions

    async def resume_session(
        self,
        session_id: str,
        *,
        preset: str | None = None,
    ) -> CreateSessionResponse:
        """Reattach to a live session, or hydrate one from the event log.

        The role and conversation history are always taken from the
        replayed event log. `preset` names the preset to drive the new
        run (provider pool, per-agent routing). Leave it None to use
        `default.json`.

        Returns the same shape as create_session so the caller can use
        them interchangeably.
        """
        body = {"preset": preset} if preset is not None else {}
        data = await self._post(
            f"/api/sessions/{session_id}/resume",
            body,
            expected_status=(200,),
        )
        return CreateSessionResponse(**data)

    async def create_session(
        self,
        *,
        preset: str | None = None,
    ) -> CreateSessionResponse:
        req = CreateSessionRequest(preset=preset)
        data = await self._post("/api/sessions", req.model_dump(exclude_none=False))
        return CreateSessionResponse(**data)

    async def fork_session(
        self,
        source_session_id: str,
        *,
        at_event_id: int,
        preset: str | None = None,
    ) -> CreateSessionResponse:
        """Fork a session at a specific event id.

        Copies events 1..at_event_id from `source_session_id` into a
        fresh session (new id, original event ids preserved), writes
        fork lineage sidecar metadata, and hydrates the new session
        via the same resume path create/resume use.

        `preset` lets the fork continue under a different configuration
        than the source — the primitive behind event-pinned
        counterfactual experiments. Omitting it continues under
        `default.json`, same as resume.
        """
        body: dict = {"at_event_id": at_event_id}
        if preset is not None:
            body["preset"] = preset
        data = await self._post(
            f"/api/sessions/{source_session_id}/fork",
            body,
            expected_status=(200,),
        )
        return CreateSessionResponse(**data)

    async def get_session(self, session_id: str) -> SessionInfo:
        data = await self._get(f"/api/sessions/{session_id}")
        return SessionInfo(**data)

    async def send_message(self, session_id: str, text: str) -> None:
        req = SendMessageRequest(text=text)
        await self._post(
            f"/api/sessions/{session_id}/messages",
            req.model_dump(),
            expected_status=(200, 202),
        )

    async def cancel(self, session_id: str) -> None:
        await self._post(f"/api/sessions/{session_id}/cancel", {})

    async def close_session(self, session_id: str) -> None:
        await self._delete(f"/api/sessions/{session_id}")

    async def snapshot(self, session_id: str) -> list[Event]:
        data = await self._get(f"/api/sessions/{session_id}/snapshot")
        return [self._event_from_json(e) for e in data.get("events", [])]

    async def get_frame_context(
        self,
        session_id: str,
        frame_id: str,
        *,
        up_to_event_id: int | None = None,
    ) -> dict | None:
        """Fetch the full Context (header + body) for a frame.

        Returns a dict with header.role, header.principles, and
        body.conversation.messages — or None if the frame doesn't exist.
        Server replays events to rebuild the frame, so closed child
        frames also work.

        `up_to_event_id` slices the replay: the returned context
        reflects the frame as it looked right after the event with
        that id was applied. Useful for time-travel UIs (dashboard
        scrubber).
        """
        path = f"/api/sessions/{session_id}/frames/{frame_id}/context"
        if up_to_event_id is not None:
            path += f"?up_to={up_to_event_id}"
        try:
            return await self._get(path)
        except NatureClientError as exc:
            if "frame_not_found" in str(exc):
                return None
            raise

    # ------------------------------------------------------------------
    # Event streaming
    # ------------------------------------------------------------------

    async def stream_events(
        self, session_id: str
    ) -> AsyncIterator[Event]:
        """Subscribe to the session's WebSocket and yield Event objects.

        The first message from the server is a session_meta envelope
        (`{"__kind__": "session_meta", ...}`) which is filtered out
        here — callers only see real Events. If you need the metadata,
        call `get_session()` separately.
        """
        try:
            import websockets
        except ImportError as exc:
            raise NatureClientError(
                "websockets package not installed. "
                "Install with: pip install 'nature[dashboard]'"
            ) from exc

        url = f"{self._ws_url}/ws/sessions/{session_id}"
        try:
            async with websockets.connect(url) as ws:
                async for raw in ws:
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(data, dict) and data.get("__kind__") == "session_meta":
                        continue
                    yield self._event_from_json(data)
        except OSError as exc:
            raise ServerNotRunning(
                f"could not connect to {url}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Internal HTTP helpers
    # ------------------------------------------------------------------

    async def _get(self, path: str) -> dict:
        return await self._request("GET", path, None, (200,))

    async def _post(
        self,
        path: str,
        payload: dict,
        expected_status: tuple[int, ...] = (200,),
    ) -> dict:
        return await self._request("POST", path, payload, expected_status)

    async def _delete(self, path: str) -> dict:
        return await self._request("DELETE", path, None, (200,))

    async def _request(
        self,
        method: str,
        path: str,
        payload: dict | None,
        expected_status: tuple[int, ...],
    ) -> dict:
        try:
            r = await self._http.request(method, path, json=payload)
        except httpx.ConnectError as exc:
            raise ServerNotRunning(
                f"could not connect to {self._http_url}: {exc}. "
                "Start it with `nature server start`."
            ) from exc
        if r.status_code not in expected_status:
            try:
                err = ErrorResponse(**r.json())
                raise NatureClientError(
                    f"{method} {path} → {r.status_code}: "
                    f"{err.error}{(': ' + err.detail) if err.detail else ''}"
                )
            except (ValueError, KeyError):
                raise NatureClientError(
                    f"{method} {path} → {r.status_code}: {r.text[:200]}"
                )
        try:
            return r.json()
        except ValueError:
            return {}

    @staticmethod
    def _event_from_json(data: dict) -> Event:
        type_value = data.get("type", "")
        try:
            event_type = EventType(type_value)
        except ValueError:
            event_type = EventType.ERROR
        return Event(
            id=data.get("id", 0),
            session_id=data.get("session_id", ""),
            frame_id=data.get("frame_id"),
            timestamp=data.get("timestamp", 0.0),
            type=event_type,
            payload=data.get("payload") or {},
        )
