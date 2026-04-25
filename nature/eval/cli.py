"""`nature eval` CLI surface — run cells, list tasks/runs, render reports.

Wires the runner / tasks / results / report modules behind a small
click subcommand group. The main CLI entry point in `nature/cli.py`
imports `register(main)` from here and attaches it.
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import click

from nature.eval.diff import diff_runs
from nature.eval.results import (
    RunRecord, list_runs, load_latest_run, load_run, logs_dir,
    new_run_id, runs_dir, write_run,
)
from nature.eval.report import render_markdown
from nature.eval.runner import CellResult, run_cell
from nature.eval.tasks import load_tasks_registry


def _repo_git_sha(repo_root: Path) -> str | None:
    try:
        import subprocess
        r = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() or None
    except Exception:
        return None


@click.group(name="eval")
def eval_group() -> None:
    """Benchmark nature presets against tasks.

    \b
        nature eval run                         # every task × every preset
        nature eval run --task n1 --preset default
        nature eval tasks list
        nature eval runs list
        nature eval report --run <run-id>
    """


# ──────────────────────────────────────────────────────────────────────
# run
# ──────────────────────────────────────────────────────────────────────


@eval_group.command("run")
@click.option("--task", "tasks", multiple=True, help="Task id (repeatable). Omit to run all known tasks.")
@click.option("--preset", "presets", multiple=True, help="Preset name (repeatable). Omit to use 'default' only.")
@click.option("--seeds", default=1, type=int, show_default=True,
              help="Number of runs per (task, preset) cell. >1 enables variance reporting.")
@click.option("--concurrency", default=1, type=int, show_default=True,
              help="Number of cells to run in parallel. Default 1 (serial). "
              "Remote-provider cells scale with this; local Ollama cells need "
              "OLLAMA_NUM_PARALLEL set on the Ollama server to benefit.")
@click.option("--project-dir", default=None, help="Project dir (default: cwd).")
@click.option("--notes", default="", help="Free-form notes stored on the run record.")
@click.option("--timeout-override", default=None, type=float,
              help="Override the session watchdog timeout (seconds). "
              "Decouples the agent's wall-clock budget from the task's "
              "acceptance.timeout_sec. Use for hypothesis tests where "
              "the default gate would kill a slow but still-progressing "
              "session. Acceptance-test timeout is unaffected.")
def run_cmd(
    tasks: tuple[str, ...],
    presets: tuple[str, ...],
    seeds: int,
    concurrency: int,
    project_dir: str | None,
    notes: str,
    timeout_override: float | None,
) -> None:
    """Run every selected (task × preset) cell and write a run record.

    With `--seeds N > 1`, each (task, preset) pair is executed `N`
    times; the run record stores one CellResult per execution with
    the `seed` field recording which run it was. Aggregation (mean /
    std / pass rate) happens at report / diff time so raw data is
    preserved for later re-analysis.

    With `--concurrency N > 1`, up to N cells run in parallel via a
    thread pool. Each cell spawns its own server subprocess with a
    free port and isolated NATURE_HOME, so state does not cross
    boundaries. The caller is responsible for ensuring the backends
    (cloud rate limits, `OLLAMA_NUM_PARALLEL`) can handle N
    concurrent sessions.
    """
    if seeds < 1:
        raise click.ClickException("--seeds must be >= 1")
    if concurrency < 1:
        raise click.ClickException("--concurrency must be >= 1")

    root = Path(project_dir or os.getcwd()).resolve()
    registry = load_tasks_registry(project_dir=root)
    if not registry:
        raise click.ClickException("no tasks found (builtin + user + project).")

    selected_ids = list(tasks) or sorted(registry.keys())
    for tid in selected_ids:
        if tid not in registry:
            raise click.ClickException(
                f"unknown task id {tid!r}. Known: {sorted(registry)}",
            )
    selected_tasks = [registry[tid] for tid in selected_ids]
    selected_presets = list(presets) or ["default"]

    total_cells = len(selected_ids) * len(selected_presets) * seeds

    run = RunRecord(
        run_id=new_run_id(),
        started_at=time.time(),
        repo_git_sha=_repo_git_sha(root),
        task_ids=selected_ids,
        preset_names=selected_presets,
        notes=notes,
    )
    click.echo(
        f"[eval] run {run.run_id}: "
        f"{len(selected_ids)} task × {len(selected_presets)} preset "
        f"× {seeds} seed = {total_cells} cells"
        + (f" (concurrency={concurrency})" if concurrency > 1 else "")
    )

    # Build the full cell schedule as a flat list.
    schedule = [
        (task, preset, seed)
        for task in selected_tasks
        for preset in selected_presets
        for seed in range(seeds)
    ]

    def _execute(task, preset, seed):
        return run_cell(
            task, preset,
            seed=seed if seeds > 1 else None,
            repo_root=root,
            logs_dir=logs_dir(root),
            timeout_override=timeout_override,
        )

    def _label(task, preset, seed) -> str:
        label = f"{task.id} × {preset}"
        if seeds > 1:
            label += f" [seed {seed}]"
        return label

    def _format_result(cell) -> str:
        verdict = (
            "ERR" if cell.error
            else ("PASS" if cell.passed else "FAIL")
        )
        cost = f"${cell.cost_usd:.4f}" if cell.cost_usd is not None else "?"
        lat = f"{cell.latency_sec:.0f}s" if cell.latency_sec is not None else "?"
        return f"{verdict}  {cost}  {lat}"

    if concurrency == 1:
        # Serial path — preserves the original start/result inline format.
        for task, preset, seed in schedule:
            click.echo(f"[eval] • {_label(task, preset, seed)} …", nl=False)
            cell = _execute(task, preset, seed)
            run.cells.append(cell)
            click.echo(f" {_format_result(cell)}")
            if cell.error:
                click.echo(f"    error: {cell.error}", err=True)
    else:
        # Parallel path — cells interleave, so print one line per
        # completion prefixed with a progress counter.
        lock = threading.Lock()
        done = 0
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            future_to_label = {
                pool.submit(_execute, task, preset, seed):
                    _label(task, preset, seed)
                for task, preset, seed in schedule
            }
            for fut in as_completed(future_to_label):
                label = future_to_label[fut]
                try:
                    cell = fut.result()
                except Exception as exc:
                    # Surface but keep going so one pathological cell
                    # doesn't abort the whole matrix.
                    click.echo(
                        f"[eval] ✗ {label} — exception {exc!r}", err=True,
                    )
                    continue
                with lock:
                    run.cells.append(cell)
                    done += 1
                    click.echo(
                        f"[eval] [{done}/{total_cells}] "
                        f"{label} {_format_result(cell)}"
                    )
                    if cell.error:
                        click.echo(
                            f"    error: {cell.error}", err=True,
                        )

    run.finished_at = time.time()
    path = write_run(run, project_dir=root)
    click.echo(f"[eval] wrote {path}")
    click.echo("")
    click.echo(render_markdown(run.to_dict()))


# ──────────────────────────────────────────────────────────────────────
# tasks list
# ──────────────────────────────────────────────────────────────────────


@eval_group.group("tasks")
def tasks_group() -> None:
    """Task definition queries."""


@tasks_group.command("list")
@click.option("--project-dir", default=None)
def tasks_list_cmd(project_dir: str | None) -> None:
    """List every known task with one-line summary."""
    root = Path(project_dir or os.getcwd()).resolve()
    registry = load_tasks_registry(project_dir=root)
    if not registry:
        click.echo("(no tasks found)")
        return
    for tid in sorted(registry):
        t = registry[tid]
        click.echo(
            f"  {tid:30s} {t.category:10s} {t.size:8s} "
            f"setup={t.setup.type:25s} acc={t.acceptance.type}"
        )


# ──────────────────────────────────────────────────────────────────────
# runs list
# ──────────────────────────────────────────────────────────────────────


@eval_group.group("runs")
def runs_group() -> None:
    """Past run records."""


@runs_group.command("list")
@click.option("--project-dir", default=None)
@click.option("--limit", type=int, default=10)
def runs_list_cmd(project_dir: str | None, limit: int) -> None:
    """List the most recent run files (newest first)."""
    root = Path(project_dir or os.getcwd()).resolve()
    paths = list_runs(project_dir=root)[:limit]
    if not paths:
        click.echo("(no runs yet)")
        return
    for path in paths:
        import json
        doc = json.loads(path.read_text(encoding="utf-8"))
        cells = len(doc.get("cells", []))
        passed = sum(1 for c in doc["cells"] if c.get("passed"))
        click.echo(
            f"  {doc['run_id']:30s} cells={cells} pass={passed} "
            f"tasks={len(doc.get('task_ids', []))} "
            f"presets={len(doc.get('preset_names', []))}"
        )


# ──────────────────────────────────────────────────────────────────────
# report
# ──────────────────────────────────────────────────────────────────────


@eval_group.command("report")
@click.option("--run", "run_id", default=None, help="Run id (omit for the latest).")
@click.option("--project-dir", default=None)
def report_cmd(run_id: str | None, project_dir: str | None) -> None:
    """Render a markdown report for one run."""
    root = Path(project_dir or os.getcwd()).resolve()
    if run_id is None:
        run = load_latest_run(project_dir=root)
        if run is None:
            raise click.ClickException("no runs recorded yet.")
    else:
        run = load_run(run_id, project_dir=root)
    click.echo(render_markdown(run))


# ──────────────────────────────────────────────────────────────────────
# diff
# ──────────────────────────────────────────────────────────────────────


@eval_group.command("rebuild-metrics")
@click.option("--run", "run_id", default=None,
              help="Run id to rebuild (omit for the latest).")
@click.option("--project-dir", default=None)
def rebuild_metrics_cmd(run_id: str | None, project_dir: str | None) -> None:
    """Re-extract event-derived metrics from every cell's archived log
    and stamp them back onto the run record.

    The event log is the canonical source; CellResult fields are a
    cache. When a new metric lands in `_extract_metrics`, this command
    backfills it for historical runs without re-executing any sessions.
    Metric-only fields are touched; acceptance verdicts, errors, and
    run-level metadata stay exactly as stored.
    """
    import json
    from nature.eval.runner import _copy_metrics_onto_cell, _extract_metrics

    root = Path(project_dir or os.getcwd()).resolve()
    if run_id is None:
        runs = list_runs(project_dir=root)
        if not runs:
            raise click.ClickException("no runs recorded yet.")
        run_path = runs[0]
    else:
        run_path = runs_dir(project_dir=root) / f"{run_id}.json"
        if not run_path.exists():
            raise click.ClickException(f"run {run_id!r} not found")

    doc = json.loads(run_path.read_text(encoding="utf-8"))
    rebuilt = 0
    skipped = 0
    for cell in doc.get("cells", []):
        log_path = cell.get("event_log_path")
        if not log_path or not Path(log_path).exists():
            skipped += 1
            continue
        try:
            metrics = _extract_metrics(Path(log_path), root)
        except Exception as exc:
            click.echo(
                f"  skip {cell.get('task_id')}×{cell.get('preset')}: "
                f"extract failed ({exc})", err=True,
            )
            skipped += 1
            continue

        cell_obj = CellResult(
            task_id=cell.get("task_id", ""),
            preset=cell.get("preset", ""),
            started_at=cell.get("started_at", 0.0),
        )
        _copy_metrics_onto_cell(cell_obj, metrics)
        # Overlay only the cached-metric fields — leave verdict,
        # error, timing, acceptance output, and identifiers alone.
        for key in (
            "cost_usd", "tokens_in", "tokens_out", "cache_read_tokens",
            "turn_count", "tool_call_count", "sub_frame_count",
            "max_delegation_depth", "agents_used",
            "body_compactions", "cost_by_agent", "provider_errors",
            "tool_error_count", "cache_hit_rate", "avg_turn_latency_ms",
            "source_session_id", "forked_from_event_id",
        ):
            cell[key] = getattr(cell_obj, key)
        rebuilt += 1

    run_path.write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    click.echo(
        f"[eval] rebuilt {rebuilt} cell metrics ({skipped} skipped) → {run_path}"
    )


@eval_group.command("diff")
@click.argument("run_a")
@click.argument("run_b")
@click.option("--project-dir", default=None)
def diff_cmd(run_a: str, run_b: str, project_dir: str | None) -> None:
    """Compare two runs side by side.

    Pairs cells by `(task_id, preset)`. Shows verdict transitions,
    signed percent deltas on cost/latency, and turn/tool counts
    before-and-after. Regressions and fixes are called out in
    dedicated sections at the bottom.
    """
    root = Path(project_dir or os.getcwd()).resolve()
    a = load_run(run_a, project_dir=root)
    b = load_run(run_b, project_dir=root)
    click.echo(diff_runs(a, b))


def register(main_group: click.Group) -> None:
    """Attach `nature eval …` to the top-level CLI."""
    main_group.add_command(eval_group)
