"""nature client — talks to nature server over HTTP + WebSocket.

The client is the read/write counterpart to nature/server. UIs (TUI,
REPL, browser) use NatureClient to drive sessions without holding any
execution state in-process. UI bugs cannot affect execution because
execution lives in a separate process.

Public surface:
- NatureClient: HTTP API client (create_session, send_message, ...)
- stream_events: async iterator over the WebSocket event stream
- NatureClientError: base exception for connection / API errors
"""

from nature.client.http_client import (
    NatureClient,
    NatureClientError,
    ServerNotRunning,
)

__all__ = [
    "NatureClient",
    "NatureClientError",
    "ServerNotRunning",
]
