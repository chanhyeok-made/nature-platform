"""Diff two run records — side-by-side delta for every common cell.

Phase 2b: pure markdown output. Given two run JSONs, line up cells
by `(task_id, preset)` and emit a row per pair with before/after
verdict + metric deltas. Asymmetric keys (cell in only one run)
surface in an "unmatched" tail so the user notices missing coverage.
"""

from __future__ import annotations

from nature.eval.results import cells_by_task


# ──────────────────────────────────────────────────────────────────────
# Shape helpers
# ──────────────────────────────────────────────────────────────────────


def _key(cell: dict) -> tuple[str, str]:
    return (cell["task_id"], cell["preset"])


def _verdict(cell: dict) -> str:
    if cell.get("error"):
        return "ERR"
    return "PASS" if cell.get("passed") else "FAIL"


def _verdict_arrow(before: dict, after: dict) -> str:
    """Small glyph summarising the verdict transition."""
    b = _verdict(before)
    a = _verdict(after)
    if b == a:
        return b  # unchanged
    if b == "FAIL" and a == "PASS":
        return "↑ FIX"
    if b == "PASS" and a == "FAIL":
        return "↓ BREAK"
    return f"{b}→{a}"


def _pct(before: float | None, after: float | None) -> str:
    """Signed percent change, or an em-dash when either side is missing."""
    if before is None or after is None or before == 0:
        return "—"
    change = (after - before) / before * 100.0
    sign = "+" if change >= 0 else ""
    return f"{sign}{change:.0f}%"


def _fmt(value: float | int | None, fmt: str = "") -> str:
    if value is None:
        return "—"
    if fmt:
        return f"{value:{fmt}}"
    return str(value)


# ──────────────────────────────────────────────────────────────────────
# Diff builders
# ──────────────────────────────────────────────────────────────────────


def _row_for_pair(before: dict, after: dict) -> list[str]:
    """One markdown row — metrics before / after / delta."""
    return [
        f"`{before['task_id']}`",
        before["preset"],
        _verdict_arrow(before, after),
        (
            f"${_fmt(before.get('cost_usd'), '.4f')}"
            f" → "
            f"${_fmt(after.get('cost_usd'), '.4f')}"
            f" ({_pct(before.get('cost_usd'), after.get('cost_usd'))})"
        ),
        (
            f"{_fmt(before.get('latency_sec'), '.0f')}s"
            f" → "
            f"{_fmt(after.get('latency_sec'), '.0f')}s"
            f" ({_pct(before.get('latency_sec'), after.get('latency_sec'))})"
        ),
        (
            f"{_fmt(before.get('turn_count'))}"
            f" → "
            f"{_fmt(after.get('turn_count'))}"
        ),
        (
            f"{_fmt(before.get('tool_call_count'))}"
            f" → "
            f"{_fmt(after.get('tool_call_count'))}"
        ),
    ]


def diff_runs(run_a: dict, run_b: dict) -> str:
    """Render a markdown diff between two run records.

    Pairs cells by `(task_id, preset)`. Cells present in only one
    run are collected into an "unmatched" section. Rows are sorted
    task-major, preset-minor so the table reads top-down.
    """
    a_cells = {_key(c): c for c in run_a.get("cells", [])}
    b_cells = {_key(c): c for c in run_b.get("cells", [])}
    common = sorted(set(a_cells) & set(b_cells))
    only_a = sorted(set(a_cells) - set(b_cells))
    only_b = sorted(set(b_cells) - set(a_cells))

    out: list[str] = []
    out.append(f"# eval diff — {run_a['run_id']} → {run_b['run_id']}")
    out.append("")
    out.append(
        f"- cells matched: {len(common)} · only in A: {len(only_a)} "
        f"· only in B: {len(only_b)}"
    )
    out.append("")

    if not common:
        out.append("_No overlapping cells to diff._")
    else:
        header = (
            "| task | preset | verdict | cost | latency | "
            "turns | tool calls |"
        )
        sep = "|------|--------|---------|------|---------|------:|------------:|"
        out.append(header)
        out.append(sep)
        for key in common:
            row = _row_for_pair(a_cells[key], b_cells[key])
            out.append("| " + " | ".join(row) + " |")
        out.append("")

        # Headline regressions / fixes.
        regressions = [
            k for k in common
            if _verdict(a_cells[k]) == "PASS" and _verdict(b_cells[k]) == "FAIL"
        ]
        fixes = [
            k for k in common
            if _verdict(a_cells[k]) == "FAIL" and _verdict(b_cells[k]) == "PASS"
        ]
        if regressions:
            out.append("## Regressions (PASS → FAIL)")
            out.append("")
            for task_id, preset in regressions:
                out.append(f"- `{task_id}` × `{preset}`")
            out.append("")
        if fixes:
            out.append("## Fixes (FAIL → PASS)")
            out.append("")
            for task_id, preset in fixes:
                out.append(f"- `{task_id}` × `{preset}`")
            out.append("")

    if only_a or only_b:
        out.append("## Unmatched cells")
        out.append("")
        if only_a:
            out.append("**Only in A:**")
            for task_id, preset in only_a:
                out.append(f"- `{task_id}` × `{preset}`")
            out.append("")
        if only_b:
            out.append("**Only in B:**")
            for task_id, preset in only_b:
                out.append(f"- `{task_id}` × `{preset}`")
            out.append("")

    return "\n".join(out)
