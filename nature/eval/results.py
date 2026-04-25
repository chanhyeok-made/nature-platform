"""Run / result persistence — JSON files under `.nature/eval/results/`.

Layout:
    .nature/eval/results/
        runs/
            <run-id>.json    # one file per `nature eval run` invocation
        logs/
            <cell-tag>.jsonl # raw session event logs, tagged by
                             # (task, preset, timestamp); kept to let
                             # the dashboard replay any cell later.

`<run-id>` is `<timestamp>-<short-hex>` so runs sort chronologically
and concurrent runs don't collide. One run file holds the whole
matrix (every cell executed in that invocation) plus run-level
metadata: nature git sha, CLI args, total wall time, summary
counters. That keeps diffing two runs a single-file-vs-single-file
operation.
"""

from __future__ import annotations

import json
import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from nature.eval.runner import CellResult


# ──────────────────────────────────────────────────────────────────────
# RunRecord — the on-disk shape
# ──────────────────────────────────────────────────────────────────────


@dataclass
class RunRecord:
    """One invocation of `nature eval run`, possibly across many cells."""

    run_id: str
    started_at: float
    finished_at: float | None = None
    repo_git_sha: str | None = None
    task_ids: list[str] = field(default_factory=list)
    preset_names: list[str] = field(default_factory=list)
    cells: list[CellResult] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "repo_git_sha": self.repo_git_sha,
            "task_ids": list(self.task_ids),
            "preset_names": list(self.preset_names),
            "cells": [c.to_dict() for c in self.cells],
            "notes": self.notes,
        }


# ──────────────────────────────────────────────────────────────────────
# Directory layout
# ──────────────────────────────────────────────────────────────────────


def results_root(project_dir: Path | str | None = None) -> Path:
    base = Path(project_dir) if project_dir else Path(os.getcwd())
    return base / ".nature" / "eval" / "results"


def runs_dir(project_dir: Path | str | None = None) -> Path:
    d = results_root(project_dir) / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def logs_dir(project_dir: Path | str | None = None) -> Path:
    d = results_root(project_dir) / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ──────────────────────────────────────────────────────────────────────
# Read / write
# ──────────────────────────────────────────────────────────────────────


def new_run_id() -> str:
    """`<ts>-<6hex>` — sortable and collision-safe for parallel runs."""
    return f"{int(time.time())}-{secrets.token_hex(3)}"


def write_run(run: RunRecord, project_dir: Path | str | None = None) -> Path:
    path = runs_dir(project_dir) / f"{run.run_id}.json"
    path.write_text(
        json.dumps(run.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def list_runs(project_dir: Path | str | None = None) -> list[Path]:
    """Run files newest first."""
    d = runs_dir(project_dir)
    return sorted(d.glob("*.json"), reverse=True)


def load_run(run_id: str, project_dir: Path | str | None = None) -> dict:
    """Load a run's raw JSON by id."""
    path = runs_dir(project_dir) / f"{run_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"run {run_id!r} not found at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_latest_run(project_dir: Path | str | None = None) -> dict | None:
    """Convenience: load the most recent run, or None if there are none."""
    runs = list_runs(project_dir)
    if not runs:
        return None
    return json.loads(runs[0].read_text(encoding="utf-8"))


# ──────────────────────────────────────────────────────────────────────
# Summary helpers used by the report layer
# ──────────────────────────────────────────────────────────────────────


def summarize_by_preset(run: dict) -> dict[str, dict]:
    """Collapse cells by preset into roll-up counters.

    Returned keys per preset:
        pass, total, error              — counts
        cost_sum, latency_sum           — USD / seconds
        turn_sum, tool_sum              — LLM turn + tool-call totals
                                          (Phase 2a; absent on older
                                          run files → just 0)
    """
    agg: dict[str, dict] = {}
    for cell in run["cells"]:
        slot = agg.setdefault(
            cell["preset"],
            {
                "pass": 0, "total": 0, "error": 0,
                "cost_sum": 0.0, "latency_sum": 0.0,
                "turn_sum": 0, "tool_sum": 0,
            },
        )
        slot["total"] += 1
        if cell.get("passed"):
            slot["pass"] += 1
        if cell.get("cost_usd") is not None:
            slot["cost_sum"] += float(cell["cost_usd"])
        if cell.get("latency_sec") is not None:
            slot["latency_sum"] += float(cell["latency_sec"])
        if cell.get("turn_count") is not None:
            slot["turn_sum"] += int(cell["turn_count"])
        if cell.get("tool_call_count") is not None:
            slot["tool_sum"] += int(cell["tool_call_count"])
        if cell.get("error"):
            slot["error"] += 1
    return agg


def cells_by_task(cells: Iterable[dict]) -> dict[str, dict[str, dict]]:
    """Pivot cells into `{task_id: {preset: cell-or-aggregate}}`.

    Phase 2a was one cell per (task, preset). With multi-seed, one
    (task, preset) may appear N times — this helper aggregates them
    into a synthetic "cell" whose metric fields are means and whose
    extra fields (`pass_count`, `seed_count`, `*_std`) describe the
    variance. Single-seed cells surface unchanged.
    """
    groups: dict[tuple[str, str], list[dict]] = {}
    for cell in cells:
        groups.setdefault((cell["task_id"], cell["preset"]), []).append(cell)

    out: dict[str, dict[str, dict]] = {}
    for (task_id, preset), group in groups.items():
        out.setdefault(task_id, {})[preset] = (
            group[0] if len(group) == 1 else _aggregate_group(group)
        )
    return out


def _aggregate_group(group: list[dict]) -> dict:
    """Collapse a multi-seed cell pool into a single aggregate dict.

    Numeric fields collapse to means; `*_std` fields carry sample
    standard deviation. `pass_count` / `seed_count` let report code
    render `pass_count / seed_count` instead of a single verdict.
    Error cells are excluded from numeric aggregation but still
    reported as `error_count`.
    """
    import statistics as _stats

    n = len(group)
    passed_count = sum(1 for c in group if c.get("passed"))
    error_count = sum(1 for c in group if c.get("error"))

    def _mean_and_std(key: str) -> tuple[float | None, float | None]:
        vals = [c[key] for c in group if c.get(key) is not None]
        if not vals:
            return None, None
        mean = sum(vals) / len(vals)
        std = _stats.pstdev(vals) if len(vals) > 1 else 0.0
        return mean, std

    cost_mean, cost_std = _mean_and_std("cost_usd")
    lat_mean, lat_std = _mean_and_std("latency_sec")
    turn_mean, turn_std = _mean_and_std("turn_count")
    tool_mean, tool_std = _mean_and_std("tool_call_count")

    first = group[0]
    agg = {
        "task_id": first["task_id"],
        "preset": first["preset"],
        "seed_count": n,
        "pass_count": passed_count,
        "error_count": error_count,
        # Display-time expects the old keys too; treat means as the
        # authoritative numeric value for single-cell rendering.
        "cost_usd": cost_mean,
        "latency_sec": lat_mean,
        "turn_count": turn_mean,
        "tool_call_count": tool_mean,
        # `passed` is kept as "majority passed" for quick glance; the
        # honest signal is `pass_count / seed_count`.
        "passed": passed_count > n / 2,
        # `error` is truthy only if *every* seed errored.
        "error": group[0].get("error") if error_count == n else None,
        # Std fields (Phase 2b introduces these; absent → unknown).
        "cost_std": cost_std,
        "latency_std": lat_std,
        "turn_std": turn_std,
        "tool_std": tool_std,
    }
    return agg
