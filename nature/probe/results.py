"""Persistence for probe runs.

Layout:
    .nature/probe/results/
        runs/
            <run-id>.json    # one file per `nature probe run` invocation

`<run-id>` format matches eval: `<timestamp>-<short-hex>`. One run
aggregates every (probe × model) cell executed in that invocation.
"""

from __future__ import annotations

import json
import os
import secrets
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from nature.probe.runner import ProbeRunResult


@dataclass
class ProbeRunRecord:
    """One `nature probe run` invocation's aggregate record."""

    run_id: str
    started_at: float
    finished_at: float | None = None
    repo_git_sha: str | None = None
    probe_ids: list[str] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    cells: list[dict] = field(default_factory=list)  # one per ProbeRunResult
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "repo_git_sha": self.repo_git_sha,
            "probe_ids": list(self.probe_ids),
            "models": list(self.models),
            "cells": list(self.cells),
            "notes": self.notes,
        }


def _cell_dict(r: ProbeRunResult) -> dict:
    return {
        "probe_id": r.probe_id,
        "model_ref": r.model_ref,
        "passed": r.outcome.passed,
        "fail_category": r.outcome.fail_category,
        "criteria": [
            {"kind": c.kind, "passed": c.passed, "reason": c.reason}
            for c in r.outcome.criteria
        ],
        "latency_ms": r.latency_ms,
        "tokens_in": r.tokens_in,
        "tokens_out": r.tokens_out,
        "runner_errors": list(r.runner_errors),
        "trace": {
            "turn_count": r.trace.turn_count,
            "hit_max_turns": r.trace.hit_max_turns,
            "final_text": r.trace.final_text[:2000],
            "tool_uses": [
                {
                    "index": tu.index,
                    "name": tu.name,
                    "input": tu.input,
                    "result_is_error": tu.result_is_error,
                    "result_preview": (tu.result_text or "")[:400],
                }
                for tu in r.trace.tool_uses
            ],
        },
    }


def new_run_id() -> str:
    return f"{int(time.time())}-{secrets.token_hex(3)}"


def results_dir(project_dir: Path) -> Path:
    d = project_dir / ".nature" / "probe" / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def runs_dir(project_dir: Path) -> Path:
    d = results_dir(project_dir) / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_run_record(project_dir: Path, record: ProbeRunRecord) -> Path:
    target = runs_dir(project_dir) / f"{record.run_id}.json"
    target.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
    return target


def load_run_record(project_dir: Path, run_id: str) -> dict:
    return json.loads((runs_dir(project_dir) / f"{run_id}.json").read_text(encoding="utf-8"))


def list_runs(project_dir: Path) -> list[dict]:
    """Return recent runs as lightweight metadata (no cells)."""
    out: list[dict] = []
    for p in sorted(runs_dir(project_dir).glob("*.json")):
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        out.append({
            "run_id": doc.get("run_id"),
            "started_at": doc.get("started_at"),
            "finished_at": doc.get("finished_at"),
            "probes": len(doc.get("probe_ids") or []),
            "models": len(doc.get("models") or []),
            "cells": len(doc.get("cells") or []),
            "notes": doc.get("notes", ""),
        })
    return out


__all__ = [
    "ProbeRunRecord",
    "new_run_id",
    "write_run_record",
    "load_run_record",
    "list_runs",
    "results_dir",
    "runs_dir",
    "_cell_dict",
]
