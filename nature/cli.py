"""CLI entry point for nature."""

import asyncio
import os

import click

from nature import __version__


# Common options shared between `nature` and `nature chat`. The CLI is
# a pure client of the nature server (run `nature server start` first).
#
# Deprecated-and-removed flags (now config-driven, see README "Hosts"):
#   -p/--provider, -m/--model, --api-key, --base-url
# Model/provider settings live in `hosts.json` and are managed via
# `nature hosts`. Secrets come from environment variables referenced
# by the host's `api_key_env`. Per-agent models stay in frame.json.
_common_options = [
    click.option("--resume", "-r", default=None, help="Resume session by ID"),
    click.option("--new", "new_session", is_flag=True, default=False, help="Skip session picker, start a fresh session"),
    click.option("--dashboard", "-d", is_flag=True, default=False, help="Open event dashboard URL in browser"),
    click.option("--host", default="localhost", help="Server host to connect to (default: localhost)"),
    click.option("--port", default=7777, type=int, help="Server port to connect to (default: 7777)"),
]


def _add_options(options):
    def decorator(f):
        for opt in reversed(options):
            f = opt(f)
        return f
    return decorator


def _run_chat(**kwargs):
    """Launch the Frame+Event TUI, talking to the nature server over HTTP/WS.

    The server must already be running: `nature server start` in a
    separate terminal. Crashes on the client side never touch execution.
    """
    from nature.ui.frame_tui import run_frame_tui
    run_frame_tui(
        host=kwargs.get("host") or "localhost",
        port=kwargs.get("port") or 7777,
        open_browser=kwargs.get("dashboard", False),
        resume_session_id=kwargs.get("resume"),
        force_new=kwargs.get("new_session", False),
    )


@click.group(invoke_without_command=True)
@click.version_option(version=__version__, prog_name="nature")
@_add_options(_common_options)
@click.pass_context
def main(ctx: click.Context, **kwargs) -> None:
    """Nature — LLM agent orchestration framework."""
    if ctx.invoked_subcommand is None:
        _run_chat(**kwargs)


# Attach `nature eval` as a subcommand group. Import is local so a
# missing pydantic/click in the eval package can't break the rest of
# the CLI at import time (the eval group's own imports stay inside
# the group).
from nature.eval.cli import register as _register_eval
_register_eval(main)

from nature.probe.cli import register as _register_probe
_register_probe(main)


@main.command()
@_add_options(_common_options)
def chat(**kwargs) -> None:
    """Start an interactive chat session (Frame+Event TUI).

    Requires `nature server start` to be running.

    Examples:
        nature                              # session picker, then TUI
        nature --new                        # skip picker, new session
        nature -r <session_id>              # resume a specific session
        nature -d                           # open dashboard alongside TUI
    """
    _run_chat(**kwargs)


@main.group()
def server() -> None:
    """Manage the nature server daemon."""


@server.command("start")
@click.option("--cwd", default=None, help="Project directory (default: current)")
def server_start(cwd: str | None) -> None:
    """Start the server in the foreground (logs to stdout, Ctrl+C to stop).

    Run this in its own terminal/tmux pane. The server holds the
    SessionRegistry, AreaManager, and EventStore — UI clients connect
    over HTTP/WS and crashes on the client side cannot affect execution.

    Bind address/port come from environment variables (Docker/systemd
    friendly):

        NATURE_SERVER_HOST   bind address (default: localhost).
                             Use 0.0.0.0 to expose on all interfaces.
        NATURE_SERVER_PORT   HTTP port (default: 7777). WS uses port+1.
    """
    host = os.environ.get("NATURE_SERVER_HOST", "localhost")
    port = int(os.environ.get("NATURE_SERVER_PORT", "7777"))
    import logging
    import signal

    from nature.config.settings import get_nature_home
    from nature.server.app import ServerApp

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    pid_file = get_nature_home() / "server.pid"
    if pid_file.exists():
        try:
            existing_pid = int(pid_file.read_text().strip())
            os.kill(existing_pid, 0)  # signal 0 = exists?
            click.echo(
                f"[error] server already running (pid {existing_pid}, "
                f"pidfile {pid_file})",
                err=True,
            )
            click.echo("Stop it with `nature server stop` first.", err=True)
            raise SystemExit(1)
        except (ProcessLookupError, ValueError):
            # Stale pidfile
            pid_file.unlink(missing_ok=True)
        except PermissionError:
            click.echo(
                f"[warning] pidfile {pid_file} points at a process we can't signal",
                err=True,
            )

    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))

    app = ServerApp(port=port, host=host, cwd=cwd)

    async def _run() -> None:
        ok = await app.start()
        if not ok:
            click.echo("[error] server failed to start", err=True)
            raise SystemExit(1)
        click.echo(f"nature server listening at {app.http_url}")
        click.echo(f"  ws:        {app.ws_url}")
        click.echo(f"  pid file:  {pid_file}")
        click.echo(f"  events:    {get_nature_home() / 'events'}")
        click.echo("Ctrl+C to stop.")

        stop_event = asyncio.Event()

        def _signal_handler(*_args) -> None:
            stop_event.set()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _signal_handler)
            except NotImplementedError:
                signal.signal(sig, _signal_handler)

        try:
            await stop_event.wait()
        finally:
            click.echo("\nshutting down...")
            await app.stop()

    try:
        asyncio.run(_run())
    finally:
        pid_file.unlink(missing_ok=True)
        click.echo("server stopped.")


@server.command("status")
def server_status() -> None:
    """Show whether the server is running."""
    import os

    from nature.config.settings import get_nature_home

    pid_file = get_nature_home() / "server.pid"
    if not pid_file.exists():
        click.echo("server: not running (no pidfile)")
        return
    try:
        pid = int(pid_file.read_text().strip())
    except ValueError:
        click.echo(f"server: pidfile {pid_file} is corrupt")
        return
    try:
        os.kill(pid, 0)
        click.echo(f"server: running (pid {pid}, pidfile {pid_file})")
    except ProcessLookupError:
        click.echo(f"server: stale pidfile (pid {pid} not running)")
    except PermissionError:
        click.echo(f"server: pidfile points at pid {pid} (no signal permission)")


@main.group()
def hosts() -> None:
    """Manage the LLM host registry (hosts.json).

    A host is a named endpoint: provider type + base_url + auth env var +
    a list of known models. Users reference models as `host::model`
    (e.g., `groq::llama-3.3-70b-versatile`,
    `local-ollama::qwen2.5-coder:32b`). Builtin hosts include anthropic,
    openai, local-ollama, openrouter, groq, together — add your own with
    `nature hosts add`.
    """


@hosts.command("list")
def hosts_list_cmd() -> None:
    """List every registered host (builtin + user) with its default marker."""
    from nature.config.hosts import is_builtin, load_hosts_config

    cfg = load_hosts_config(project_dir=".")
    rows: list[tuple[str, str, str, str, str, int]] = []
    for name in sorted(cfg.hosts.keys()):
        host = cfg.hosts[name]
        base = host.base_url or "(sdk default)"
        key_label = "(anonymous)"
        if host.api_key_env:
            status = "set" if os.environ.get(host.api_key_env) else "unset"
            key_label = f"{host.api_key_env} ({status})"
        flag = "builtin" if is_builtin(name) else "user"
        default_mark = "*" if name == cfg.default_host else " "
        rows.append((default_mark, name, host.provider, base, key_label, len(host.models)))

    click.echo(f"{'':<2}{'NAME':<18}{'PROVIDER':<12}{'BASE_URL':<46}{'API KEY':<34}MODELS")
    for mark, name, provider, base, key, n in rows:
        click.echo(f"{mark:<2}{name:<18}{provider:<12}{base:<46}{key:<34}{n}")
    click.echo(f"\n* = default host  (set via `nature hosts set-default <name>`)")


@hosts.command("show")
@click.argument("name")
def hosts_show_cmd(name: str) -> None:
    """Print a single host's full config (models, auth, endpoint)."""
    import os as _os

    from nature.config.hosts import is_builtin, load_hosts_config

    cfg = load_hosts_config(project_dir=".")
    host = cfg.get_host(name)
    if host is None:
        click.echo(f"[error] unknown host: {name}", err=True)
        click.echo(f"  available: {', '.join(sorted(cfg.hosts.keys()))}", err=True)
        raise SystemExit(1)

    click.echo(f"Host:     {name}{'  (builtin)' if is_builtin(name) else '  (user)'}")
    click.echo(f"Provider: {host.provider}")
    click.echo(f"Base URL: {host.base_url or '(SDK default)'}")
    if host.api_key_env:
        status = "set" if _os.environ.get(host.api_key_env) else "NOT SET"
        click.echo(f"API key:  {host.api_key_env} ({status})")
    else:
        click.echo(f"API key:  (anonymous — no auth)")
    click.echo(f"Default:  {'yes' if name == cfg.default_host else 'no'}")
    if host.models:
        click.echo(f"\nKnown models ({len(host.models)}):")
        for m in host.models:
            click.echo(f"  - {name}::{m}")
    else:
        click.echo(f"\nKnown models: (none listed)")


@hosts.command("add")
@click.argument("name")
@click.option("--provider", required=True, type=click.Choice(["anthropic", "openai"]),
              help="Provider SDK type (anthropic or openai-compat)")
@click.option("--base-url", default=None, help="Endpoint URL (omit for anthropic SDK default)")
@click.option("--api-key-env", default=None,
              help="Environment variable name for the API key. Omit for anonymous (e.g. local ollama).")
@click.option("--model", "-m", "models", multiple=True,
              help="Known model name (repeat for multiple).")
@click.option("--force", is_flag=True, default=False,
              help="Overwrite if the host name already exists in the user file.")
def hosts_add_cmd(name: str, provider: str, base_url: str | None,
                  api_key_env: str | None, models: tuple[str, ...],
                  force: bool) -> None:
    """Register a new host in ~/.nature/hosts.json.

    Example:
        nature hosts add mygroq --provider openai \\
            --base-url https://api.groq.com/openai/v1 \\
            --api-key-env GROQ_API_KEY \\
            -m llama-3.3-70b-versatile -m moonshotai/kimi-k2
    """
    from nature.config.hosts import (
        HostConfig, is_builtin, load_user_hosts_config, save_user_hosts_config,
        user_hosts_path,
    )

    user_cfg = load_user_hosts_config()
    if name in user_cfg.hosts and not force:
        click.echo(f"[error] host {name!r} already exists in user config. Use --force to overwrite.", err=True)
        raise SystemExit(1)
    if is_builtin(name):
        click.echo(f"[warning] {name!r} shadows a builtin host — user entry will take precedence.")

    user_cfg.hosts[name] = HostConfig(
        provider=provider,
        base_url=base_url,
        api_key_env=api_key_env,
        models=list(models),
    )
    path = save_user_hosts_config(user_cfg)
    click.echo(f"saved host {name!r} to {path}")
    if not models:
        click.echo("  (no models registered — use `nature hosts add --model <name>` to add later)")


@hosts.command("remove")
@click.argument("name")
def hosts_remove_cmd(name: str) -> None:
    """Remove a user-added host from ~/.nature/hosts.json.

    Builtin hosts can't be removed directly — use `nature hosts add <name> --force`
    with the same name to shadow the builtin with your own settings.
    """
    from nature.config.hosts import (
        is_builtin, load_user_hosts_config, save_user_hosts_config,
    )

    user_cfg = load_user_hosts_config()
    if name not in user_cfg.hosts:
        if is_builtin(name):
            click.echo(
                f"[error] {name!r} is a builtin host and can't be removed. "
                f"Use `nature hosts add {name} --force ...` to shadow it with a user entry.",
                err=True,
            )
        else:
            click.echo(f"[error] host {name!r} not found in user config", err=True)
        raise SystemExit(1)

    del user_cfg.hosts[name]
    path = save_user_hosts_config(user_cfg)
    click.echo(f"removed host {name!r} from {path}")


@hosts.command("set-default")
@click.argument("name")
def hosts_set_default_cmd(name: str) -> None:
    """Set the default host (used when a model is referenced without `host::` prefix)."""
    from nature.config.hosts import (
        load_hosts_config, load_user_hosts_config, save_user_hosts_config,
    )

    # Validate against the merged view so builtin + user hosts both count
    merged = load_hosts_config(project_dir=".")
    if name not in merged.hosts:
        click.echo(f"[error] unknown host: {name}", err=True)
        click.echo(f"  available: {', '.join(sorted(merged.hosts.keys()))}", err=True)
        raise SystemExit(1)

    user_cfg = load_user_hosts_config()
    user_cfg.default_host = name
    path = save_user_hosts_config(user_cfg)
    click.echo(f"default host set to {name!r} in {path}")


@main.group()
def sessions() -> None:
    """Manage nature sessions held by the running server."""


@sessions.command("list")
@click.option("--host", default="localhost", help="Server host")
@click.option("--port", default=7777, type=int, help="Server port")
@click.option("--limit", default=20, type=int, help="Max archived sessions to show")
def sessions_list(host: str, port: int, limit: int) -> None:
    """List live + archived sessions on the server."""
    import asyncio
    from datetime import datetime

    from nature.client import NatureClient, ServerNotRunning

    async def _run() -> None:
        async with NatureClient(host=host, port=port) as client:
            try:
                live = await client.list_sessions()
                archived = await client.list_archived_sessions()
            except ServerNotRunning as exc:
                click.echo(f"[error] {exc}", err=True)
                raise SystemExit(1)

            click.echo("Live sessions:")
            if not live:
                click.echo("  (none)")
            for s in live:
                click.echo(
                    f"  {s.session_id[:12]}  {s.state:<14}  "
                    f"{s.root_role_name:<14}  {s.root_model}"
                )

            click.echo("\nArchived sessions (most recent first):")
            if not archived:
                click.echo("  (none)")
            for s in archived[:limit]:
                last = datetime.fromtimestamp(s.last_event_at).strftime("%Y-%m-%d %H:%M")
                click.echo(
                    f"  {s.session_id[:12]}  {s.event_count:>4} events  last: {last}"
                )
            if len(archived) > limit:
                click.echo(f"  ... ({len(archived) - limit} more)")

    asyncio.run(_run())


@sessions.command("resume")
@click.argument("session_id")
@click.option("--host", default="localhost", help="Server host")
@click.option("--port", default=7777, type=int, help="Server port")
def sessions_resume(session_id: str, host: str, port: int) -> None:
    """Hydrate an archived session into the live registry without launching the TUI.

    Useful for warming up a session before a UI client connects.
    The TUI's `nature chat --resume <id>` does this automatically.
    """
    import asyncio
    from nature.client import NatureClient, NatureClientError, ServerNotRunning

    async def _run() -> None:
        async with NatureClient(host=host, port=port) as client:
            try:
                resumed = await client.resume_session(session_id)
            except ServerNotRunning as exc:
                click.echo(f"[error] {exc}", err=True)
                raise SystemExit(1)
            except NatureClientError as exc:
                click.echo(f"[error] {exc}", err=True)
                raise SystemExit(1)
            click.echo(f"resumed: {resumed.session_id}")
            click.echo(f"  role:     {resumed.root_role_name}")
            click.echo(f"  model:    {resumed.root_model}")
            click.echo(f"  provider: {resumed.provider_name}")

    asyncio.run(_run())


@sessions.command("fork")
@click.argument("source_session_id")
@click.option(
    "--at",
    "at_event_id",
    required=True,
    type=int,
    help="Copy events 1..AT from the source session into the fork",
)
@click.option("--host", default="localhost", help="Server host")
@click.option("--port", default=7777, type=int, help="Server port")
def sessions_fork(
    source_session_id: str,
    at_event_id: int,
    host: str,
    port: int,
) -> None:
    """Fork a session at a specific event id.

    Creates a new session by copying events 1..AT from SOURCE_SESSION_ID,
    rewriting session_id in each copy and preserving original event ids.
    New events appended to the forked session continue from AT+1. The
    lineage (parent_session_id + forked_from_event_id) is stored as
    sidecar metadata so the dashboard can render the fork tree.

    Works for both live and archived source sessions — as long as the
    source has an event log on disk, it can be forked.
    """
    import asyncio
    from nature.client import NatureClient, NatureClientError, ServerNotRunning

    async def _run() -> None:
        async with NatureClient(host=host, port=port) as client:
            try:
                forked = await client.fork_session(
                    source_session_id, at_event_id=at_event_id,
                )
            except ServerNotRunning as exc:
                click.echo(f"[error] {exc}", err=True)
                raise SystemExit(1)
            except NatureClientError as exc:
                click.echo(f"[error] {exc}", err=True)
                raise SystemExit(1)
            click.echo(f"forked: {forked.session_id}")
            click.echo(f"  from:     {forked.parent_session_id}@{forked.forked_from_event_id}")
            click.echo(f"  role:     {forked.root_role_name}")
            click.echo(f"  model:    {forked.root_model}")
            click.echo(f"  provider: {forked.provider_name}")

    asyncio.run(_run())


@server.command("stop")
def server_stop() -> None:
    """Stop a running server (sends SIGTERM)."""
    _send_stop_signal(verbose=True)


@server.command("restart")
@click.option("--cwd", default=None, help="Project directory (default: current)")
@click.pass_context
def server_restart(ctx: click.Context, cwd: str | None) -> None:
    """Stop the running server (if any) and start a fresh one.

    Use this after pulling new code — the server loads route handlers
    and HTML templates once at startup, so source changes don't take
    effect until the process is restarted.

    Bind address/port come from NATURE_SERVER_HOST / NATURE_SERVER_PORT
    env vars, same as `server start`.
    """
    import os
    import time as _time

    from nature.config.settings import get_nature_home

    pid_file = get_nature_home() / "server.pid"
    if pid_file.exists():
        _send_stop_signal(verbose=True)
        # Wait up to 3s for the old process to exit
        for _ in range(30):
            if not pid_file.exists():
                break
            try:
                pid = int(pid_file.read_text().strip())
                os.kill(pid, 0)
            except (ProcessLookupError, ValueError, FileNotFoundError):
                pid_file.unlink(missing_ok=True)
                break
            _time.sleep(0.1)
        else:
            click.echo(
                "[warning] previous server still alive after 3s — "
                "starting anyway may fail to bind",
                err=True,
            )

    ctx.invoke(server_start, cwd=cwd)


def _send_stop_signal(*, verbose: bool) -> bool:
    import os
    import signal

    from nature.config.settings import get_nature_home

    pid_file = get_nature_home() / "server.pid"
    if not pid_file.exists():
        if verbose:
            click.echo("server: not running (no pidfile)")
        return False
    try:
        pid = int(pid_file.read_text().strip())
    except ValueError:
        click.echo(f"server: pidfile {pid_file} is corrupt — removing")
        pid_file.unlink(missing_ok=True)
        return False
    try:
        os.kill(pid, signal.SIGTERM)
        if verbose:
            click.echo(f"sent SIGTERM to pid {pid}")
        return True
    except ProcessLookupError:
        click.echo(f"server: pid {pid} not running, removing stale pidfile")
        pid_file.unlink(missing_ok=True)
        return False
    except PermissionError:
        click.echo(f"[error] cannot signal pid {pid}", err=True)
        raise SystemExit(1)
