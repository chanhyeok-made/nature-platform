"""nature server — daemon hosting execution behind an HTTP + WebSocket API.

Architecture:
- ServerApp owns a SessionRegistry + FileEventStore.
- Each Session has its own AreaManager + Frame tree, running in the
  server process. Execution lives here.
- HTTP API for control (create / send / cancel / list).
- WebSocket /ws/sessions/{id} for live event streaming.
- Dashboard HTML served at /.

The point of this layer is **process isolation**: UI clients (TUI,
browser, remote) crash without affecting execution. Multiple clients
can attach to the same session simultaneously, and the session
survives client restarts.
"""

from nature.server.app import ServerApp
from nature.server.registry import ServerSession, SessionRegistry

__all__ = [
    "ServerApp",
    "ServerSession",
    "SessionRegistry",
]
