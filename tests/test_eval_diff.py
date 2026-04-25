"""Tests for `nature.eval.diff` — two-run comparison renderer."""

from __future__ import annotations

from nature.eval.diff import diff_runs


def _cell(
    task_id: str, preset: str, *, passed: bool = True,
    cost: float | None = 0.01, latency: float | None = 10.0,
    turns: int | None = 3, tools: int | None = 1,
    error: str | None = None,
) -> dict:
    return {
        "task_id": task_id,
        "preset": preset,
        "passed": passed,
        "error": error,
        "cost_usd": cost,
        "latency_sec": latency,
        "turn_count": turns,
        "tool_call_count": tools,
    }


def _run(run_id: str, cells: list[dict]) -> dict:
    return {"run_id": run_id, "cells": cells}


def test_diff_renders_common_cells():
    a = _run("A", [
        _cell("t1", "default", cost=0.10, latency=20, turns=5, tools=2),
    ])
    b = _run("B", [
        _cell("t1", "default", cost=0.05, latency=10, turns=4, tools=1),
    ])
    md = diff_runs(a, b)
    assert "eval diff — A → B" in md
    assert "`t1`" in md
    assert "$0.1000 → $0.0500" in md
    # percent change visible
    assert "-50%" in md


def test_diff_flags_fix_and_regression():
    a = _run("A", [
        _cell("t1", "default", passed=False),
        _cell("t2", "default", passed=True),
    ])
    b = _run("B", [
        _cell("t1", "default", passed=True),
        _cell("t2", "default", passed=False),
    ])
    md = diff_runs(a, b)

    assert "↑ FIX" in md
    assert "↓ BREAK" in md
    assert "## Regressions (PASS → FAIL)" in md
    assert "## Fixes (FAIL → PASS)" in md
    # Each bucket lists the right task.
    reg_section = md.split("## Regressions")[1].split("##")[0]
    fix_section = md.split("## Fixes")[1].split("##")[0]
    assert "`t2`" in reg_section
    assert "`t1`" in fix_section


def test_diff_lists_unmatched_cells():
    a = _run("A", [
        _cell("t1", "default"),
        _cell("t2", "default"),
    ])
    b = _run("B", [
        _cell("t2", "default"),
        _cell("t3", "default"),
    ])
    md = diff_runs(a, b)

    assert "## Unmatched cells" in md
    assert "Only in A" in md and "`t1`" in md
    assert "Only in B" in md and "`t3`" in md


def test_diff_handles_no_common_cells():
    a = _run("A", [_cell("t1", "default")])
    b = _run("B", [_cell("t2", "default")])
    md = diff_runs(a, b)
    assert "No overlapping cells to diff" in md
    assert "## Unmatched cells" in md


def test_diff_tolerates_missing_metrics():
    """Old run files (pre-Phase-2a) don't record turn/tool counts.
    Diff output should fall back to em-dashes instead of crashing."""
    a = _run("A", [{
        "task_id": "t1", "preset": "default", "passed": True, "error": None,
        "cost_usd": 0.01, "latency_sec": 10.0,
        # turn_count / tool_call_count absent
    }])
    b = _run("B", [_cell("t1", "default", turns=3, tools=2)])
    md = diff_runs(a, b)
    assert "`t1`" in md
    assert "— → 3" in md  # turn metric unknown on A side


def test_diff_error_cells_show_err_verdict():
    a = _run("A", [_cell("t1", "default", error="timeout")])
    b = _run("B", [_cell("t1", "default", passed=True)])
    md = diff_runs(a, b)
    # Transition row mentions ERR
    assert "ERR" in md or "ERR→PASS" in md
