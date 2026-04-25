"""CLI for `nature probe ...`.

    nature probe list                                # list probes
    nature probe run --model <host::model> [--probe ID]
    nature probe runs list
    nature probe report [--run RUN_ID]

Intentionally parallels `nature eval ...` so the commands are
guessable by anyone familiar with the eval group.
"""

from __future__ import annotations

import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import click

from nature.probe.probes import load_probes
from nature.probe.results import (
    ProbeRunRecord,
    _cell_dict,
    list_runs,
    load_run_record,
    new_run_id,
    write_run_record,
)
from nature.probe.runner import run_probe


@click.group("probe")
def probe_group() -> None:
    """Probe target models against tiered capability tests.

    \b
        nature probe list                         # enumerate probes
        nature probe run --model MODEL            # run all probes × one model
        nature probe report --run RUN_ID          # pivot table
    """


# ──────────────────────────────────────────────────────────────────────
# list probes
# ──────────────────────────────────────────────────────────────────────


@probe_group.command("list")
@click.option("--project-dir", default=None)
def probe_list_cmd(project_dir: str | None) -> None:
    root = Path(project_dir or os.getcwd()).resolve()
    probes = load_probes(project_dir=root)
    if not probes:
        click.echo("(no probes)")
        return
    click.echo(f"{len(probes)} probes:")
    for p in sorted(probes.values(), key=lambda x: (x.tier, x.id)):
        dims = ",".join(p.dimensions) if p.dimensions else "-"
        click.echo(
            f"  T{p.tier}  {p.id:40}  [{dims}]  tools={','.join(p.allowed_tools) or '-'}"
        )


# ──────────────────────────────────────────────────────────────────────
# run
# ──────────────────────────────────────────────────────────────────────


@probe_group.command("run")
@click.option("--probe", "probe_ids", multiple=True,
              help="Probe id (repeatable). Omit to run every known probe.")
@click.option("--model", "models", multiple=True, required=True,
              help="host::model reference (repeatable).")
@click.option("--tier-max", default=None, type=int,
              help="Skip probes with tier > this value.")
@click.option("--project-dir", default=None)
@click.option("--notes", default="", help="Free-form notes stored on the run record.")
def probe_run_cmd(
    probe_ids: tuple[str, ...],
    models: tuple[str, ...],
    tier_max: int | None,
    project_dir: str | None,
    notes: str,
) -> None:
    """Run the selected probes against each model, write a run record."""
    root = Path(project_dir or os.getcwd()).resolve()
    probes = load_probes(project_dir=root)
    if not probes:
        raise click.ClickException("no probes found (builtin + user + project).")

    if probe_ids:
        unknown = [pid for pid in probe_ids if pid not in probes]
        if unknown:
            raise click.ClickException(
                f"unknown probe ids: {unknown}. Known: {sorted(probes)}"
            )
        selected = [probes[pid] for pid in probe_ids]
    else:
        selected = sorted(probes.values(), key=lambda x: (x.tier, x.id))

    if tier_max is not None:
        selected = [p for p in selected if p.tier <= tier_max]

    if not selected:
        raise click.ClickException("no probes matched the filter.")

    total = len(selected) * len(models)
    record = ProbeRunRecord(
        run_id=new_run_id(),
        started_at=time.time(),
        repo_git_sha=_repo_git_sha(root),
        probe_ids=[p.id for p in selected],
        models=list(models),
        notes=notes,
    )
    click.echo(
        f"[probe] run {record.run_id}: "
        f"{len(selected)} probes × {len(models)} models = {total} cells"
    )

    # Outer=model, inner=probe so local (Ollama) models stay loaded
    # across the whole 29-probe sweep instead of being swapped in/out
    # per probe. Each Ollama swap is 10-120s of idle time depending on
    # model size, so the loop ordering dominates total wall clock.
    done = 0
    for model in models:
        for probe in selected:
            done += 1
            click.echo(
                f"[probe] [{done}/{total}] T{probe.tier} {probe.id} × {model} … ",
                nl=False,
            )
            try:
                result = run_probe(probe, model, project_dir=root)
            except Exception as exc:  # noqa: BLE001
                click.echo(f"HARNESS ERR  {type(exc).__name__}: {exc}")
                continue
            verdict = "PASS" if result.outcome.passed else "FAIL"
            reason = (
                ""
                if result.outcome.passed
                else f"  [{result.outcome.fail_category}]"
            )
            click.echo(
                f"{verdict}  {result.latency_ms}ms  "
                f"in:{result.tokens_in} out:{result.tokens_out}"
                f"{reason}"
            )
            record.cells.append(_cell_dict(result))

    record.finished_at = time.time()
    target = write_run_record(root, record)
    click.echo(f"[probe] wrote {target}")
    _print_summary(record)


# ──────────────────────────────────────────────────────────────────────
# runs list
# ──────────────────────────────────────────────────────────────────────


@probe_group.group("runs")
def runs_group() -> None:
    """List or inspect past probe runs."""


@runs_group.command("list")
@click.option("--project-dir", default=None)
@click.option("--limit", default=20, type=int)
def probe_runs_list_cmd(project_dir: str | None, limit: int) -> None:
    root = Path(project_dir or os.getcwd()).resolve()
    runs = list_runs(root)
    if not runs:
        click.echo("(no probe runs yet)")
        return
    for r in runs[-limit:][::-1]:
        started = (
            time.strftime("%Y-%m-%d %H:%M", time.localtime(r["started_at"]))
            if r.get("started_at") else "?"
        )
        click.echo(
            f"  {r['run_id']}  {started}  probes={r['probes']}  "
            f"models={r['models']}  cells={r['cells']}  {r['notes'][:40]}"
        )


# ──────────────────────────────────────────────────────────────────────
# report
# ──────────────────────────────────────────────────────────────────────


@probe_group.command("report")
@click.option("--run", "run_id", default=None, help="Run id (default: latest).")
@click.option("--project-dir", default=None)
def probe_report_cmd(run_id: str | None, project_dir: str | None) -> None:
    """Pivot table: probe × model with pass/fail glyph; per-model tier
    ceiling at the bottom."""
    root = Path(project_dir or os.getcwd()).resolve()
    runs = list_runs(root)
    if not runs:
        click.echo("(no runs)")
        return
    if run_id is None:
        run_id = runs[-1]["run_id"]
    doc = load_run_record(root, run_id)
    _print_pivot(doc)


# ──────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────


def _repo_git_sha(root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root, capture_output=True, text=True, timeout=5,
        )
    except Exception:  # noqa: BLE001
        return None
    return out.stdout.strip() or None


def _print_summary(record: ProbeRunRecord) -> None:
    total = len(record.cells)
    passed = sum(1 for c in record.cells if c["passed"])
    click.echo(f"\n[probe] summary: {passed}/{total} passed")
    per_model: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
    for c in record.cells:
        p, t = per_model[c["model_ref"]]
        per_model[c["model_ref"]] = (p + (1 if c["passed"] else 0), t + 1)
    for model, (p, t) in per_model.items():
        click.echo(f"  {model}: {p}/{t}")


def _print_pivot(doc: dict) -> None:
    models = doc.get("models") or []
    probe_ids = doc.get("probe_ids") or []
    cells = doc.get("cells") or []
    # Index by (probe_id, model_ref)
    by_key: dict[tuple[str, str], dict] = {
        (c["probe_id"], c["model_ref"]): c for c in cells
    }
    # Probe tier lookup from the loaded registry (optional, if
    # probes.json have tier info — we fetch fresh so the report is
    # deterministic even if probe files changed since the run).
    from nature.probe.probes import load_probes
    probe_reg = load_probes(project_dir=Path("."))

    click.echo(f"\n# probe run {doc.get('run_id')}")
    click.echo(f"- started: {time.strftime('%Y-%m-%d %H:%M', time.localtime(doc.get('started_at', 0)))}")
    click.echo(f"- models: {models}")
    click.echo()
    # Pivot header
    header = "| probe | tier |"
    for m in models:
        header += f" {m} |"
    click.echo(header)
    sep = "|---|---|" + "|".join(["---" for _ in models]) + "|"
    click.echo(sep)
    for pid in probe_ids:
        probe = probe_reg.get(pid)
        tier = probe.tier if probe else "?"
        row = f"| `{pid}` | T{tier} |"
        for m in models:
            c = by_key.get((pid, m))
            if c is None:
                row += " - |"
            elif c["passed"]:
                row += " ✓ |"
            else:
                row += f" ✗ ({c.get('fail_category') or '?'}) |"
        click.echo(row)
    # Per-model tier profile. Three columns:
    #   pass_ceiling  — highest tier where pass_rate ≥ 0.8 (contiguous
    #                   from T0). Conservative "can reliably do."
    #   attempt_ceiling — highest tier where pass_rate ≥ 0.5 (not
    #                     required contiguous). "Can sometimes do."
    #   top_any       — highest tier where the model passed at least
    #                   one probe. "Best attempted."
    # Useful together because contiguous-ceiling is brittle when a
    # single probe has a bug or edge case.
    click.echo("\n## Tier profile per model")
    click.echo(f"  {'model':40}  {'pass':>5} {'attempt':>7} {'top':>4}  per-tier rates")
    for m in models:
        model_cells = [c for c in cells if c["model_ref"] == m]
        per_tier: dict[int, list[bool]] = defaultdict(list)
        for c in model_cells:
            probe = probe_reg.get(c["probe_id"])
            tier = probe.tier if probe else 0
            per_tier[tier].append(c["passed"])

        pass_ceiling = -1
        for tier in sorted(per_tier):
            rate = sum(per_tier[tier]) / max(1, len(per_tier[tier]))
            if rate >= 0.8:
                pass_ceiling = tier
            else:
                break
        attempt_ceiling = -1
        for tier in sorted(per_tier):
            rate = sum(per_tier[tier]) / max(1, len(per_tier[tier]))
            if rate >= 0.5:
                attempt_ceiling = max(attempt_ceiling, tier)
        top_any = -1
        for tier, passes in per_tier.items():
            if any(passes):
                top_any = max(top_any, tier)

        rates = " ".join(
            f"T{t}:{sum(per_tier[t])}/{len(per_tier[t])}"
            for t in sorted(per_tier)
        )
        click.echo(
            f"  {m.split('::')[-1][:40]:40}  "
            f"T{pass_ceiling:>3}  T{attempt_ceiling:>5}  T{top_any:>2}  {rates}"
        )


def register(main_group: click.Group) -> None:
    main_group.add_command(probe_group)


__all__ = ["register"]
