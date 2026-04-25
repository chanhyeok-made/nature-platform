"""SessionPickerApp — Textual screen for choosing a session before chat.

Run sequence:
    1. nature chat --frame  (no --resume)
    2. SessionPickerApp.run()  → user picks new / live / archived
    3. picker.selected_session_id is set:
         None       → new session
         "<sid>"    → resume that session
         "<cancel>" → user pressed Esc, exit without launching chat
    4. caller launches FrameTUI accordingly

Pure client of NatureClient. No execution state.
"""

from __future__ import annotations

import logging
from datetime import datetime

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, ListItem, ListView, Static

from nature.client import NatureClient, ServerNotRunning

logger = logging.getLogger(__name__)

CANCEL_SENTINEL = "__cancelled__"
NEW_SENTINEL = "__new__"


def _esc(s: object) -> str:
    """Escape Textual markup characters."""
    from rich.markup import escape
    return escape("" if s is None else str(s))


def _format_time(ts: float) -> str:
    if not ts:
        return ""
    try:
        return datetime.fromtimestamp(ts).strftime("%m-%d %H:%M")
    except Exception:
        return ""


CSS = """
Screen {
    background: $surface;
    layout: vertical;
}
#picker-header {
    height: 1;
    padding: 0 1;
    background: $panel;
    color: $text;
}
#picker-status {
    height: 1;
    padding: 0 1;
    color: $text-muted;
}
#picker-list {
    height: 1fr;
    padding: 0 1;
}
ListItem {
    padding: 0 1;
}
ListItem.-highlight {
    background: $accent 30%;
}
.picker-row {
    layout: horizontal;
}
.picker-marker {
    width: 3;
    color: $text-muted;
}
.picker-marker.new { color: $success; }
.picker-marker.live { color: $accent; }
.picker-marker.archived { color: $text-muted; }
.picker-id {
    width: 14;
    color: $text-muted;
}
.picker-meta {
    width: 22;
    color: $text-muted;
}
.picker-preview {
    width: 1fr;
    color: $text;
}
"""


class SessionPickerApp(App):
    """Pre-chat screen — select a session (live, archived, or new)."""

    CSS = CSS
    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=True),
        Binding("ctrl+c", "cancel", "Cancel", show=False),
        Binding("enter", "select", "Select", show=True),
        Binding("n", "select_new", "New", show=True),
        Binding("r", "refresh", "Refresh", show=True),
    ]

    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = 7777,
    ) -> None:
        super().__init__()
        self._host = host
        self._port = port
        self.selected_session_id: str | None = CANCEL_SENTINEL
        self._client: NatureClient | None = None
        # Keys correspond to ListItem `id` attributes; values are
        # session_id or NEW_SENTINEL for the "start new session" row.
        self._items: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        yield Static(
            "[bold green]nature[/] [dim]· session picker[/] "
            "[dim](enter=select · n=new · r=refresh · esc=cancel)[/]",
            id="picker-header",
        )
        yield Static("loading...", id="picker-status")
        yield ListView(id="picker-list")
        yield Footer()

    async def on_mount(self) -> None:
        self._client = NatureClient(host=self._host, port=self._port)
        await self._refresh()
        list_view = self.query_one("#picker-list", ListView)
        list_view.focus()

    async def _refresh(self) -> None:
        status = self.query_one("#picker-status", Static)
        list_view = self.query_one("#picker-list", ListView)
        await list_view.clear()
        self._items.clear()

        if self._client is None:
            return

        if not await self._client.is_alive():
            status.update(
                f"[red]server not running at http://{self._host}:{self._port} — "
                "start with `nature server start`[/]"
            )
            await self._add_item(
                "new",
                "+",
                "(new)",
                "",
                "start a new session",
                "new",
                NEW_SENTINEL,
            )
            return

        try:
            live = await self._client.list_sessions()
            archived = await self._client.list_archived_sessions()
        except ServerNotRunning as exc:
            status.update(f"[red]{_esc(exc)}[/]")
            return
        except Exception as exc:
            logger.exception("session list failed")
            status.update(f"[red]list failed: {_esc(exc)}[/]")
            return

        # Always include the new-session option at the top
        await self._add_item(
            "new",
            "+",
            "(new)",
            "",
            "start a new session",
            "new",
            NEW_SENTINEL,
        )

        # Live sessions next
        for s in sorted(live, key=lambda x: -x.created_at):
            await self._add_item(
                f"live-{s.session_id}",
                "●",
                s.session_id[:12],
                f"{s.state} · {s.root_role_name}",
                s.preview or "(no input yet)",
                "live",
                s.session_id,
            )

        # Archived
        for s in archived:
            await self._add_item(
                f"arch-{s.session_id}",
                "○",
                s.session_id[:12],
                _format_time(s.last_event_at),
                s.preview or f"{s.event_count} events",
                "archived",
                s.session_id,
            )

        n_live = len(live)
        n_arch = len(archived)
        status.update(
            f"[dim]{n_live} live · {n_arch} archived  "
            f"(↑/↓ navigate, enter to pick)[/]"
        )

        # Default cursor on the first session if any, else on "new"
        try:
            list_view.index = 1 if (n_live or n_arch) else 0
        except Exception:
            pass

    async def _add_item(
        self,
        item_id: str,
        marker: str,
        sid_short: str,
        meta: str,
        preview: str,
        kind: str,
        session_id: str,
    ) -> None:
        list_view = self.query_one("#picker-list", ListView)
        label_markup = (
            f"[{kind}]{_esc(marker)}[/]  "
            f"[dim]{_esc(sid_short):<12}[/]  "
            f"[dim]{_esc(meta):<20}[/]  "
            f"{_esc(preview)}"
        )
        # Map our color classes to rich tags
        label_markup = (
            label_markup
            .replace("[new]", "[bold green]")
            .replace("[live]", "[bold cyan]")
            .replace("[archived]", "[dim]")
        )
        item = ListItem(Static(label_markup), id=item_id)
        await list_view.append(item)
        self._items[item_id] = session_id

    def action_select(self) -> None:
        list_view = self.query_one("#picker-list", ListView)
        item = list_view.highlighted_child
        if item is None or item.id is None:
            return
        chosen = self._items.get(item.id)
        if chosen == NEW_SENTINEL:
            self.selected_session_id = None
        else:
            self.selected_session_id = chosen
        self.exit()

    def action_select_new(self) -> None:
        self.selected_session_id = None
        self.exit()

    @on(ListView.Selected, "#picker-list")
    def on_list_selected(self, event: ListView.Selected) -> None:
        # Mouse / enter via ListView
        self.action_select()

    def action_cancel(self) -> None:
        self.selected_session_id = CANCEL_SENTINEL
        self.exit()

    async def action_refresh(self) -> None:
        await self._refresh()

    async def on_unmount(self) -> None:
        if self._client is not None:
            try:
                await self._client.close()
            except Exception:
                pass
            self._client = None


def pick_session_blocking(
    *,
    host: str = "localhost",
    port: int = 7777,
) -> str | None:
    """Run the picker app and return the selection.

    Returns:
        - None  → user wants a NEW session
        - "<sid>" → resume that session
        - CANCEL_SENTINEL → user cancelled (caller should exit)
    """
    app = SessionPickerApp(host=host, port=port)
    app.run()
    return app.selected_session_id
