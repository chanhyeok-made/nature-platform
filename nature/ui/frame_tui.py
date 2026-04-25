"""FrameTUI — Textual client for the nature server.

This is a PURE CLIENT. It holds no execution state, no providers, no
EventStore, no SessionRunner. It talks to a separately-running
`nature server` over HTTP + WebSocket via NatureClient. Crashes in
this UI process cannot affect execution, because execution lives in
a different process.

Use:
    nature server start             # in one terminal
    nature chat --frame             # in another terminal — connects to localhost:7777
    nature chat --frame --port 8888 # different server port
    nature chat --frame --host remote.example.com

If the server isn't running, the TUI shows a clear error and exits.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import ScrollableContainer
from textual.widgets import Input, Static

logger = logging.getLogger(__name__)

from nature.client import NatureClient, NatureClientError, ServerNotRunning
from nature.events.types import Event, EventType
from nature.ui.event_consumer import EventConsumer

CSS = """
Screen {
    background: $surface;
    layout: vertical;
}
#header-bar { height: 1; padding: 0 1; background: $panel; }
#chat-area { height: 1fr; padding: 0 1; scrollbar-size: 1 1; }
.user-msg { color: $text; margin: 1 0 0 0; }
.assistant-msg { margin: 0; }
.tool-panel { margin: 0 0 0 2; padding: 0 1; color: $text-muted; }
.child-banner { margin: 0 0 0 2; color: $accent; }
.error-msg { color: $error; }
#status-bar { height: 1; padding: 0 1; }
#input-sep { height: 1; padding: 0 1; background: $panel; }
#prompt-input { border: none; background: $surface; }
#prompt-input:focus { border: none; }
"""


def _esc(s: object) -> str:
    """Escape Textual / Rich markup characters in arbitrary strings.

    Tool outputs and message text frequently contain literal `[` / `]`
    (markdown links, JSON arrays, log brackets, etc). Without escaping,
    Textual's markup parser interprets them as tags and either swallows
    content or raises MarkupError mid-render, killing the consumer task.
    """
    from rich.markup import escape

    return escape("" if s is None else str(s))


def _format_tool_args(tool_input: dict) -> str:
    args = []
    for key, value in tool_input.items():
        val = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        if len(val) > 70:
            val = val[:67] + "..."
        args.append(f"{key}: {val}")
    return ", ".join(args)


def _format_tool_call(name: str, tool_input: dict) -> str:
    """Build a tool-call header string with escaped user data."""
    args_str = _esc(_format_tool_args(tool_input))
    safe_name = _esc(name)
    return f"⏺ [bold]{safe_name}[/]({args_str})" if args_str else f"⏺ [bold]{safe_name}[/]()"


def _extract_text(content_blocks: list[dict]) -> str:
    return "".join(
        block.get("text", "")
        for block in content_blocks
        if block.get("type") == "text"
    )


# ---------------------------------------------------------------------------
# Progress indicator helpers — used by the status bar while a turn is running
# ---------------------------------------------------------------------------

# 10-frame braille spinner — a full cycle at 150ms tick = 1.5s
_SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

# Color "breathing" cycle for the activity label. Rich color names only —
# dim/bold adjectives are combined at render time.
_PULSE_STYLES = [
    "dim white",
    "white",
    "bold white",
    "bold bright_white",
    "bold white",
    "white",
]


def _format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s:02d}s"
    h, rem = divmod(int(seconds), 3600)
    m, _ = divmod(rem, 60)
    return f"{h}h {m:02d}m"


def _format_tokens(n: int) -> str:
    if n < 1000:
        return f"{n}"
    if n < 10_000:
        return f"{n/1000:.1f}k"
    return f"{n//1000}k"


class FrameTUI(App):
    CSS = CSS
    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=True),
        Binding("ctrl+c", "quit_app", "Quit", show=False),
    ]

    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = 7777,
        open_browser: bool = False,
        resume_session_id: str | None = None,
    ) -> None:
        super().__init__()
        # Server connection
        self._host = host
        self._port = port
        self._open_browser = open_browser
        self._resume_session_id = resume_session_id

        # Live state
        self._client: NatureClient | None = None
        self._session_id: str = ""
        self._root_role_name: str = ""
        self._root_model: str = ""
        self._stream_task: asyncio.Task | None = None
        self._send_task: asyncio.Task | None = None

        # Per-event rendering state
        self._tool_widgets: dict[str, Static] = {}
        self._tool_headers: dict[str, str] = {}
        self._frame_depth: dict[str, int] = {}

        # Pulse indicator state — updated on every run event, read by the
        # 150ms status-bar tick. When `_run_start_ts is None`, the tick
        # leaves the status bar alone and the bar shows whatever the
        # last concrete state (ready / cancelled / error) put there.
        self._run_start_ts: float | None = None
        self._run_input_tokens: int = 0
        self._run_output_tokens: int = 0
        self._run_activity: str = "thinking"
        self._run_spinner_idx: int = 0
        self._run_pulse_idx: int = 0
        # FIFO of user-input texts we already echoed at submit time —
        # _on_message_appended dedupes against this so submit-echo and
        # event-driven render don't double up. Historical user messages
        # (resume) flow through with no pending entries and render directly.
        self._pending_user_echoes: list[str] = []

    def compose(self) -> ComposeResult:
        yield Static("[bold green]nature[/] [dim](frame · loading...)[/]", id="header-bar")
        yield ScrollableContainer(id="chat-area")
        yield Static("", id="status-bar")
        yield Static("[dim]───[/]", id="input-sep")
        yield Input(
            placeholder="Type a message... (Esc to cancel, Ctrl+C to quit)",
            id="prompt-input",
        )

    # ------------------------------------------------------------------
    # Mount / setup
    # ------------------------------------------------------------------

    async def on_mount(self) -> None:
        status = self.query_one("#status-bar", Static)

        # Build the client and verify the server is reachable
        self._client = NatureClient(host=self._host, port=self._port)
        if not await self._client.is_alive():
            status.update(
                f"[red]nature server not running at {_esc(self._client.http_url)}. "
                "Start it with `nature server start`.[/]"
            )
            return

        # Create OR resume a session on the server. The server reads
        # frame.json, builds the provider, resolves the root role, etc.
        try:
            if self._resume_session_id:
                created = await self._client.resume_session(
                    self._resume_session_id
                )
            else:
                # Config is server-side: hosts.json + frame.json + presets.
                # Use dashboard ⚙ config or `nature hosts` to tune.
                created = await self._client.create_session()
        except NatureClientError as exc:
            verb = "resume" if self._resume_session_id else "create"
            status.update(f"[red]{verb} session failed: {_esc(exc)}[/]")
            return

        self._session_id = created.session_id
        self._root_role_name = created.root_role_name
        self._root_model = created.root_model

        # Open a streaming task that pulls events from the server WS
        self._stream_task = asyncio.create_task(self._consume_events())

        from nature import __version__
        self.query_one("#header-bar", Static).update(
            f"[bold green]nature[/] [dim]v{__version__} · frame client · "
            f"{_esc(self._root_model)} · {self._session_id[:8]} · "
            f"{_esc(self._client.http_url)}[/]"
        )
        status.update("[dim]ready[/]")
        self.query_one("#prompt-input", Input).focus()

        # Start the status-bar tick — cycles spinner chars and pulses
        # color/intensity while a run is active. When no run is in
        # flight, the tick is a cheap no-op.
        self.set_interval(0.15, self._tick_status_bar)

        # Optional: open the dashboard URL in the user's browser
        if self._open_browser:
            try:
                import webbrowser
                webbrowser.open(self._client.http_url)
            except Exception as exc:
                logger.debug("open browser failed: %s", exc)

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    @on(Input.Submitted, "#prompt-input")
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return

        from nature.config.constants import EXIT_COMMANDS
        if text.lower() in EXIT_COMMANDS:
            self.exit()
            return

        if not self._client or not self._session_id:
            self.query_one("#status-bar", Static).update(
                "[red]not connected to server[/]"
            )
            return

        if self._send_task and not self._send_task.done():
            self.query_one("#status-bar", Static).update(
                "[yellow]busy — wait for current turn to finish[/]"
            )
            return

        chat = self.query_one("#chat-area", ScrollableContainer)
        await chat.mount(
            Static(f"[bold blue]>[/] {_esc(text)}", classes="user-msg")
        )
        chat.scroll_end(animate=False)
        # Track the optimistic echo so the matching message.appended
        # event doesn't render a duplicate when the server emits it.
        self._pending_user_echoes.append(text)

        # Kick off the pulse indicator — ticks begin on the next
        # timer fire. Cleared in _on_frame_resolved / action_cancel.
        self._start_run()

        self._send_task = asyncio.create_task(self._send_to_server(text))

    async def _send_to_server(self, user_input: str) -> None:
        """POST the message to the server. Run progress arrives via the
        WebSocket event stream — this method only enqueues input."""
        status = self.query_one("#status-bar", Static)
        assert self._client is not None
        if not self._session_id:
            status.update("[red]no session — restart the TUI[/]")
            return
        try:
            # The 150ms tick will overwrite this immediately with the
            # animated pulse. We set a baseline string so there's no
            # visual gap on slow terminals.
            status.update("[dim]⋯ starting…[/]")
            await self._client.send_message(self._session_id, user_input)
        except ServerNotRunning as exc:
            status.update(f"[red]server gone: {_esc(exc)}[/]")
        except NatureClientError as exc:
            status.update(f"[red]send failed: {_esc(exc)}[/]")
        except asyncio.CancelledError:
            status.update("[yellow]cancelled[/]")
            raise

    # ------------------------------------------------------------------
    # Event consumer — pulls events from the server WebSocket
    # ------------------------------------------------------------------

    async def _consume_events(self) -> None:
        consumer = (
            EventConsumer()
            .on(EventType.FRAME_OPENED, self._on_frame_opened)
            .on(EventType.MESSAGE_APPENDED, self._on_message_appended)
            .on(EventType.TOOL_STARTED, self._on_tool_started)
            .on(EventType.TOOL_COMPLETED, self._on_tool_completed)
            .on(EventType.ANNOTATION_STORED, self._on_annotation)
            .on(EventType.LLM_ERROR, self._on_error)
            .on(EventType.FRAME_ERRORED, self._on_frame_errored)
            .on(EventType.FRAME_RESOLVED, self._on_frame_resolved)
        )
        assert self._client is not None
        try:
            async for event in self._client.stream_events(self._session_id):
                await consumer.dispatch(event)
        except asyncio.CancelledError:
            return
        except ServerNotRunning as exc:
            try:
                self.query_one("#status-bar", Static).update(
                    f"[red]server disconnected: {_esc(exc)}[/]"
                )
            except Exception:
                pass
        except Exception as exc:
            logger.warning("event stream ended: %s", exc)

    # ------------------------------------------------------------------
    # Pulse indicator — keeps the status bar alive during long turns
    # ------------------------------------------------------------------

    def _start_run(self) -> None:
        """Mark the start of a new run cycle (new user input submitted)."""
        import time as _time
        self._run_start_ts = _time.monotonic()
        self._run_input_tokens = 0
        self._run_output_tokens = 0
        self._run_activity = "thinking"
        self._run_spinner_idx = 0
        self._run_pulse_idx = 0

    def _end_run(self) -> None:
        """Mark the current run as finished; next tick will show 'ready'."""
        self._run_start_ts = None
        self._run_activity = ""

    def _tick_status_bar(self) -> None:
        """Called every ~150ms. Refreshes the status bar if a run is active.

        Goals, per the user's UX request:
        - Spinner character cycles → something is moving (not frozen)
        - Activity label breathes through dim→bright→dim → color variation
        - Elapsed time ticks at sub-second resolution
        - Token counters reflect the latest annotation.stored
        """
        if self._run_start_ts is None:
            return  # no active run — leave the bar alone
        try:
            bar = self.query_one("#status-bar", Static)
        except Exception:
            return  # DOM not ready (e.g., during early mount)

        import time as _time
        elapsed = _time.monotonic() - self._run_start_ts

        self._run_spinner_idx = (self._run_spinner_idx + 1) % len(_SPINNER_FRAMES)
        self._run_pulse_idx = (self._run_pulse_idx + 1) % len(_PULSE_STYLES)

        spinner = _SPINNER_FRAMES[self._run_spinner_idx]
        pulse_style = _PULSE_STYLES[self._run_pulse_idx]

        elapsed_str = _format_elapsed(elapsed)
        tok_parts = []
        if self._run_input_tokens:
            tok_parts.append(f"↓ {_format_tokens(self._run_input_tokens)}")
        if self._run_output_tokens:
            tok_parts.append(f"↑ {_format_tokens(self._run_output_tokens)}")
        tok_str = " · ".join(tok_parts)
        activity = _esc(self._run_activity or "thinking")

        meta_bits = [elapsed_str]
        if tok_str:
            meta_bits.append(tok_str)
        meta = " · ".join(meta_bits)

        bar.update(
            f"[bright_cyan]{spinner}[/] "
            f"[{pulse_style}]{activity}[/] "
            f"[dim]({meta})[/]"
        )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _on_frame_opened(self, event: Event) -> None:
        # Track frame depth so other handlers can filter sub-agent
        # chatter, but DO NOT render anything visible for sub-frames in
        # the TUI. The dashboard is the place to see the full tree.
        frame_id = event.frame_id or ""
        parent_id = event.payload.get("parent_id")
        if parent_id is None:
            self._frame_depth[frame_id] = 0
            return
        parent_depth = self._frame_depth.get(parent_id, 0)
        self._frame_depth[frame_id] = parent_depth + 1
        # Spawned a sub-frame? The TUI only shows root chatter, but
        # the pulse should announce the delegation so the user sees
        # movement during long core/specialist runs.
        if parent_depth == 0:
            role = event.payload.get("role_name") or "sub-agent"
            self._run_activity = f"delegating → {role}"

    async def _on_message_appended(self, event: Event) -> None:
        payload = event.payload
        from_actor = payload.get("from_", "")
        content = payload.get("content", [])
        frame_id = event.frame_id or ""
        depth = self._frame_depth.get(frame_id, 0)

        # TUI shows ONLY the root frame's user-facing conversation.
        # Sub-agent chatter (delegation prompts, intermediate tools,
        # researcher output, etc.) lives in the dashboard, not here.
        if depth > 0:
            return

        text = _extract_text(content)
        chat = self.query_one("#chat-area", ScrollableContainer)

        if from_actor == "user":
            # Dedupe against optimistic echo from the submit handler.
            # Historical user messages on resume have no pending entries
            # and render directly.
            if (
                self._pending_user_echoes
                and self._pending_user_echoes[0] == text
            ):
                self._pending_user_echoes.pop(0)
                return
            if not text:
                return
            await chat.mount(Static(
                f"[bold blue]>[/] {_esc(text)}", classes="user-msg"
            ))
            chat.scroll_end(animate=False)
            return

        # Tool results render through tool cards
        if from_actor == "tool":
            return

        if not text:
            # Assistant turn that was only tool_use — rendered by tool cards
            return

        safe_text = _esc(text)
        safe_from = _esc(from_actor)
        await chat.mount(Static(
            f"[bold green]{safe_from}[/] {safe_text}", classes="assistant-msg"
        ))
        chat.scroll_end(animate=False)

    async def _on_tool_started(self, event: Event) -> None:
        payload = event.payload
        tool_name = payload.get("tool_name", "?")
        frame_id = event.frame_id or ""
        depth = self._frame_depth.get(frame_id, 0)

        # Every tool start feeds the pulse indicator so the user sees
        # "running Glob" / "running Read" for sub-agents too — that's
        # the whole point of this feedback loop. Rendering still
        # filters by depth so the chat area only shows root-level tool
        # cards, but the status bar surfaces sub-agent activity.
        self._run_activity = f"running {tool_name}"

        if depth > 0:
            return

        tool_use_id = payload.get("tool_use_id", "")
        tool_input = payload.get("tool_input", {})
        header = _format_tool_call(tool_name, tool_input)

        chat = self.query_one("#chat-area", ScrollableContainer)
        widget = Static(f"{header} [dim](running…)[/]", classes="tool-panel")
        self._tool_widgets[tool_use_id] = widget
        self._tool_headers[tool_use_id] = header
        await chat.mount(widget)
        chat.scroll_end(animate=False)

    async def _on_tool_completed(self, event: Event) -> None:
        payload = event.payload
        tool_use_id = payload.get("tool_use_id", "")
        is_error = payload.get("is_error", False)
        duration_ms = payload.get("duration_ms", 0)
        output = payload.get("output") or ""

        # Tool finished — the next LLM call is probably about to fire,
        # so the activity label goes back to "thinking" until the next
        # tool.started or frame.resolved event arrives.
        if self._run_start_ts is not None:
            self._run_activity = "thinking"

        widget = self._tool_widgets.pop(tool_use_id, None)
        header = self._tool_headers.pop(tool_use_id, "")
        if widget is None:
            return

        dur = (
            f"{duration_ms}ms"
            if duration_ms < 1000
            else f"{duration_ms / 1000:.1f}s"
        )
        if is_error:
            preview = _esc(output[:120]) if output else "failed"
            widget.update(
                f"{header}\n  [red]⎿ {preview}[/] [dim]({_esc(dur)})[/]"
            )
        else:
            first_line = _esc(output.split("\n")[0][:100]) if output else "done"
            widget.update(
                f"{header}\n  [dim]⎿ {first_line} ({_esc(dur)})[/]"
            )

    async def _on_annotation(self, event: Event) -> None:
        usage = event.payload.get("usage")
        if not usage:
            return
        inp = usage.get("input_tokens", 0) or 0
        out = usage.get("output_tokens", 0) or 0
        # Accumulate across all LLM calls in the current run — includes
        # sub-agent turns. The pulse tick picks these up on the next
        # frame render.
        self._run_input_tokens += inp
        self._run_output_tokens += out

    async def _on_error(self, event: Event) -> None:
        msg = _esc(event.payload.get("message", ""))
        err_type = _esc(event.payload.get("error_type", "Error"))
        chat = self.query_one("#chat-area", ScrollableContainer)
        await chat.mount(Static(
            f"[red]✗ {err_type}: {msg}[/]", classes="error-msg"
        ))
        chat.scroll_end(animate=False)
        self._end_run()
        self.query_one("#status-bar", Static).update(f"[red]{err_type}[/]")

    async def _on_frame_errored(self, event: Event) -> None:
        # FRAME_ERRORED is the state-transition twin of LLM_ERROR — end
        # the pulse if the root frame died. Sub-frame errors are caller
        # problems we let bubble up through the tool_result path.
        frame_id = event.frame_id or ""
        if self._frame_depth.get(frame_id, 0) == 0:
            self._end_run()

    async def _on_frame_resolved(self, event: Event) -> None:
        frame_id = event.frame_id or ""
        depth = self._frame_depth.get(frame_id, 0)
        if depth == 0:
            # Root-level turn done — end the pulse and show ready.
            self._end_run()
            self.query_one("#status-bar", Static).update("[dim]ready[/]")
        else:
            # Sub-agent finished — parent is about to resume, show that
            # in the pulse label instead of the specialist's name.
            if self._run_start_ts is not None:
                self._run_activity = "thinking"

    # ------------------------------------------------------------------
    # Actions / lifecycle
    # ------------------------------------------------------------------

    def action_cancel(self) -> None:
        # Cancel local send task (the server-side run keeps going — use
        # the cancel API endpoint via _send_task_cancel below if you
        # actually want to abort execution server-side)
        if self._send_task and not self._send_task.done():
            self._send_task.cancel()
        if self._client and self._session_id:
            asyncio.create_task(self._client.cancel(self._session_id))
        self._end_run()
        self.query_one("#status-bar", Static).update("[yellow]cancelled[/]")

    def action_quit_app(self) -> None:
        self.exit()

    async def on_unmount(self) -> None:
        # Stop the event stream first
        if self._stream_task and not self._stream_task.done():
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
        if self._send_task and not self._send_task.done():
            self._send_task.cancel()
            try:
                await self._send_task
            except asyncio.CancelledError:
                pass
        # Close the HTTP client. The session itself stays alive on the
        # server so a future TUI can attach to it via --resume.
        if self._client is not None:
            try:
                await self._client.close()
            except Exception:
                pass
            self._client = None


def run_frame_tui(
    *,
    host: str = "localhost",
    port: int = 7777,
    open_browser: bool = False,
    resume_session_id: str | None = None,
    force_new: bool = False,
) -> None:
    """Launch FrameTUI as a standalone app, connecting to a nature server.

    When `resume_session_id` is unset and `force_new` is False, first
    run the SessionPickerApp so the user can choose between new, live,
    and archived sessions.

    Provider / model / endpoint / auth settings all come from
    hosts.json + frame.json + env vars, managed via `nature hosts`
    (no per-invocation CLI flags — use the dashboard ⚙ config for
    live tuning, or edit `.nature/presets/*.json` for named shapes).
    """
    if resume_session_id is None and not force_new:
        from nature.ui.session_picker import CANCEL_SENTINEL, pick_session_blocking

        picked = pick_session_blocking(host=host, port=port)
        if picked == CANCEL_SENTINEL:
            return
        resume_session_id = picked  # None → new, "<sid>" → resume

    app = FrameTUI(
        host=host,
        port=port,
        open_browser=open_browser,
        resume_session_id=resume_session_id,
    )
    app.run()


if __name__ == "__main__":
    run_frame_tui()
