"""Tests for nature.eval metric extraction + summary roll-up.

The extractor reads a session jsonl and computes every number the
runner surfaces. We fabricate tiny event sequences here (without
starting a real server) so each metric has at least one test.

Multi-seed aggregation (Phase 2b) adds a second dimension:
`cells_by_task` collapses same-(task,preset) cell groups into an
aggregate dict when `seed_count > 1`; single cells surface unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

from nature.eval.results import cells_by_task, summarize_by_preset
from nature.eval.runner import _extract_metrics


REPO_ROOT = Path(__file__).resolve().parents[1]


# ──────────────────────────────────────────────────────────────────────
# jsonl event builders
# ──────────────────────────────────────────────────────────────────────


def _make_event(
    *, id: int, frame_id: str, type: str, payload: dict,
) -> dict:
    """One event in the shape the store writes."""
    return {
        "id": id,
        "session_id": "s1",
        "frame_id": frame_id,
        "timestamp": 0.0,
        "type": type,
        "payload": payload,
    }


def _write_events(tmp_path: Path, events: list[dict]) -> Path:
    """Serialize events as one-event-per-line jsonl."""
    path = tmp_path / "s1.jsonl"
    path.write_text(
        "\n".join(json.dumps(e) for e in events) + "\n",
        encoding="utf-8",
    )
    return path


# ──────────────────────────────────────────────────────────────────────
# Metric extractor
# ──────────────────────────────────────────────────────────────────────


def test_extract_metrics_counts_turns_and_tokens(tmp_path):
    events = [
        _make_event(
            id=1, frame_id="root",
            type="frame.opened",
            payload={"parent_id": None, "role_name": "receptionist"},
        ),
        _make_event(
            id=2, frame_id="root",
            type="llm.request",
            payload={"request_id": "r1", "model": "claude-haiku-4-5"},
        ),
        _make_event(
            id=3, frame_id="root",
            type="llm.response",
            payload={
                "request_id": "r1",
                "usage": {
                    "input_tokens": 1000,
                    "output_tokens": 200,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
            },
        ),
        _make_event(
            id=4, frame_id="root",
            type="llm.request",
            payload={"request_id": "r2", "model": "claude-haiku-4-5"},
        ),
        _make_event(
            id=5, frame_id="root",
            type="llm.response",
            payload={
                "request_id": "r2",
                "usage": {
                    "input_tokens": 500,
                    "output_tokens": 50,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 800,
                },
            },
        ),
    ]
    jsonl = _write_events(tmp_path, events)

    m = _extract_metrics(jsonl, REPO_ROOT)
    assert m.turn_count == 2
    assert m.tokens_in == 1500
    assert m.tokens_out == 250
    assert m.cache_read_tokens == 800
    assert m.cost_usd > 0
    assert m.agents_used == ["receptionist"]


def test_extract_metrics_counts_tool_calls(tmp_path):
    events = [
        _make_event(
            id=1, frame_id="root",
            type="frame.opened",
            payload={"parent_id": None, "role_name": "receptionist"},
        ),
        _make_event(
            id=2, frame_id="root",
            type="tool.completed",
            payload={"tool_name": "Read", "is_error": False},
        ),
        _make_event(
            id=3, frame_id="root",
            type="tool.completed",
            payload={"tool_name": "Grep", "is_error": False},
        ),
        _make_event(
            id=4, frame_id="root",
            type="tool.completed",
            payload={"tool_name": "Edit", "is_error": True},
        ),
    ]
    jsonl = _write_events(tmp_path, events)

    m = _extract_metrics(jsonl, REPO_ROOT)
    assert m.tool_call_count == 3


def test_extract_metrics_tracks_delegation_depth(tmp_path):
    """root → core → researcher → analyzer, so max depth = 3."""
    events = [
        _make_event(
            id=1, frame_id="root",
            type="frame.opened",
            payload={"parent_id": None, "role_name": "receptionist"},
        ),
        _make_event(
            id=2, frame_id="core",
            type="frame.opened",
            payload={"parent_id": "root", "role_name": "core"},
        ),
        _make_event(
            id=3, frame_id="researcher",
            type="frame.opened",
            payload={"parent_id": "core", "role_name": "researcher"},
        ),
        _make_event(
            id=4, frame_id="analyzer",
            type="frame.opened",
            payload={"parent_id": "core", "role_name": "analyzer"},
        ),
        _make_event(
            id=5, frame_id="deep",
            type="frame.opened",
            payload={"parent_id": "researcher", "role_name": "judge"},
        ),
    ]
    jsonl = _write_events(tmp_path, events)

    m = _extract_metrics(jsonl, REPO_ROOT)
    assert m.sub_frame_count == 4
    assert m.max_delegation_depth == 3  # root → core → researcher → deep
    # first-seen order preserved
    assert m.agents_used == ["receptionist", "core", "researcher", "analyzer", "judge"]


def test_extract_metrics_missing_file_returns_empty(tmp_path):
    m = _extract_metrics(tmp_path / "no-such.jsonl", REPO_ROOT)
    assert m.turn_count == 0
    assert m.tokens_in == 0
    assert m.agents_used == []


def test_extract_metrics_tolerates_malformed_lines(tmp_path):
    path = tmp_path / "s1.jsonl"
    path.write_text(
        "\n".join([
            json.dumps(_make_event(
                id=1, frame_id="root", type="frame.opened",
                payload={"parent_id": None, "role_name": "receptionist"},
            )),
            "{{ not json",
            json.dumps(_make_event(
                id=2, frame_id="root", type="llm.response",
                payload={
                    "request_id": "r1",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )),
        ]) + "\n",
        encoding="utf-8",
    )
    m = _extract_metrics(path, REPO_ROOT)
    assert m.turn_count == 1
    assert m.agents_used == ["receptionist"]


# ──────────────────────────────────────────────────────────────────────
# Summary roll-up
# ──────────────────────────────────────────────────────────────────────


def test_summarize_by_preset_rolls_up_counters():
    run = {
        "cells": [
            {
                "preset": "default", "passed": True, "error": None,
                "cost_usd": 0.05, "latency_sec": 10.0,
                "turn_count": 4, "tool_call_count": 2,
            },
            {
                "preset": "default", "passed": False, "error": None,
                "cost_usd": 0.10, "latency_sec": 20.0,
                "turn_count": 8, "tool_call_count": 5,
            },
            {
                "preset": "all-haiku", "passed": True, "error": None,
                "cost_usd": 0.02, "latency_sec": 8.0,
                "turn_count": 3, "tool_call_count": 1,
            },
            {
                "preset": "all-haiku", "passed": False,
                "error": "timeout",
                "cost_usd": None, "latency_sec": None,
                "turn_count": None, "tool_call_count": None,
            },
        ],
    }
    agg = summarize_by_preset(run)

    assert agg["default"]["pass"] == 1
    assert agg["default"]["total"] == 2
    assert agg["default"]["turn_sum"] == 12
    assert agg["default"]["tool_sum"] == 7
    assert abs(agg["default"]["cost_sum"] - 0.15) < 1e-9
    assert agg["default"]["latency_sum"] == 30.0
    assert agg["default"]["error"] == 0

    assert agg["all-haiku"]["pass"] == 1
    assert agg["all-haiku"]["total"] == 2
    assert agg["all-haiku"]["turn_sum"] == 3  # None ignored
    assert agg["all-haiku"]["error"] == 1


# ──────────────────────────────────────────────────────────────────────
# Multi-seed aggregation (cells_by_task)
# ──────────────────────────────────────────────────────────────────────


def _seed_cell(
    task_id: str, preset: str, seed: int,
    *, passed: bool = True, cost: float | None = 0.1,
    latency: float | None = 10.0, turns: int | None = 5,
    tools: int | None = 3, error: str | None = None,
) -> dict:
    return {
        "task_id": task_id, "preset": preset, "seed": seed,
        "passed": passed, "error": error,
        "cost_usd": cost, "latency_sec": latency,
        "turn_count": turns, "tool_call_count": tools,
    }


def test_cells_by_task_passes_through_single_cells_unchanged():
    """A (task, preset) group with one cell should surface the cell
    as-is, no aggregate fields added."""
    cells = [_seed_cell("t1", "default", 0)]
    out = cells_by_task(cells)
    assert out["t1"]["default"]["passed"] is True
    # Single-cell path: no aggregate markers leaked in.
    assert "seed_count" not in out["t1"]["default"]


def test_cells_by_task_aggregates_multi_seed_group():
    cells = [
        _seed_cell("t1", "default", 0, cost=0.10, latency=10, turns=6, tools=3),
        _seed_cell("t1", "default", 1, cost=0.12, latency=14, turns=8, tools=4),
        _seed_cell("t1", "default", 2, cost=0.08, latency=12, turns=7, tools=3,
                   passed=False),
    ]
    out = cells_by_task(cells)
    agg = out["t1"]["default"]
    assert agg["seed_count"] == 3
    assert agg["pass_count"] == 2
    # Means — allow float tolerance.
    assert abs(agg["cost_usd"] - 0.10) < 1e-9
    assert abs(agg["latency_sec"] - 12.0) < 1e-9
    assert abs(agg["turn_count"] - 7.0) < 1e-9
    # Std fields populated (non-zero since values vary).
    assert agg["cost_std"] > 0
    assert agg["latency_std"] > 0


def test_cells_by_task_ignores_none_fields_in_means():
    """Cells where a metric is None (e.g., cost tally failed) should
    be excluded from that metric's mean, not coerced to 0."""
    cells = [
        _seed_cell("t1", "default", 0, cost=0.10),
        _seed_cell("t1", "default", 1, cost=0.20),
        _seed_cell("t1", "default", 2, cost=None),  # missing
    ]
    out = cells_by_task(cells)
    agg = out["t1"]["default"]
    assert abs(agg["cost_usd"] - 0.15) < 1e-9  # mean of the two present


def test_cells_by_task_error_propagation():
    """Error cell is still counted (error_count increments) but
    excluded from numeric aggregation. `error` on the aggregate is
    set only when every seed errored."""
    # Mixed: one error, two OK → aggregate error is None
    cells_mixed = [
        _seed_cell("t1", "default", 0, cost=0.10, passed=True),
        _seed_cell("t1", "default", 1, cost=None, passed=False, error="timeout"),
        _seed_cell("t1", "default", 2, cost=0.12, passed=True),
    ]
    mixed = cells_by_task(cells_mixed)["t1"]["default"]
    assert mixed["seed_count"] == 3
    assert mixed["pass_count"] == 2
    assert mixed["error_count"] == 1
    assert mixed["error"] is None  # not all seeds errored

    # All-error: aggregate error retained
    cells_all_err = [
        _seed_cell("t1", "default", 0, cost=None, passed=False, error="timeout"),
        _seed_cell("t1", "default", 1, cost=None, passed=False, error="timeout"),
    ]
    all_err = cells_by_task(cells_all_err)["t1"]["default"]
    assert all_err["error"] == "timeout"
    assert all_err["error_count"] == 2


def test_cells_by_task_aggregate_passed_majority_rule():
    """Aggregate `passed` is 'majority passed' for one-glance display;
    the honest signal is pass_count / seed_count."""
    # 2/3 pass → aggregate.passed = True
    cells = [
        _seed_cell("t1", "default", 0, passed=True),
        _seed_cell("t1", "default", 1, passed=True),
        _seed_cell("t1", "default", 2, passed=False),
    ]
    out = cells_by_task(cells)["t1"]["default"]
    assert out["passed"] is True
    # 1/3 pass → aggregate.passed = False
    cells = [
        _seed_cell("t1", "default", 0, passed=True),
        _seed_cell("t1", "default", 1, passed=False),
        _seed_cell("t1", "default", 2, passed=False),
    ]
    out = cells_by_task(cells)["t1"]["default"]
    assert out["passed"] is False
