"""Report renderers — turn a RunRecord into human-readable output.

Phase 1 ships a markdown renderer only. `nature eval report` just
calls `render_markdown(run)` and prints. Future formats (HTML, CSV
for spreadsheet import) plug in here behind the same signature.
"""

from __future__ import annotations

from nature.eval.results import cells_by_task, summarize_by_preset


def _cell_cell(cell: dict | None) -> str:
    """One cell's pass / cost / latency / turns compact string.

    Turns is shown only when the run file recorded it — earlier
    (pre-Phase-2a) cells have `turn_count` absent, in which case it
    gets dropped from the display rather than shown as `?`.

    For aggregated cells (multi-seed), the verdict becomes
    `pass_count/seed_count` and the numeric fields show the mean.
    """
    if cell is None:
        return "—"
    seed_count = cell.get("seed_count")
    if cell.get("error") and not seed_count:
        return f"ERR  {cell['error'][:40]}"

    if seed_count:
        # Aggregated view — show pass fraction + mean metrics.
        verdict = f"{cell.get('pass_count', 0)}/{seed_count}"
    else:
        verdict = "PASS" if cell.get("passed") else "FAIL"

    cost = cell.get("cost_usd")
    latency = cell.get("latency_sec")
    cost_s = f"${cost:.4f}" if cost is not None else "?"
    lat_s = f"{latency:.0f}s" if latency is not None else "?"
    turns = cell.get("turn_count")
    if turns is not None:
        return f"{verdict}  {cost_s}  {lat_s}  {turns:.0f}t"
    return f"{verdict}  {cost_s}  {lat_s}"


def render_markdown(run: dict) -> str:
    """Produce a markdown summary for one run record."""
    presets = run.get("preset_names") or sorted(
        {c["preset"] for c in run["cells"]}
    )
    task_ids = run.get("task_ids") or sorted(
        {c["task_id"] for c in run["cells"]}
    )
    pivot = cells_by_task(run["cells"])

    out: list[str] = []
    out.append(f"# nature eval — {run['run_id']}")
    out.append("")
    if run.get("repo_git_sha"):
        out.append(f"- repo sha: `{run['repo_git_sha']}`")
    out.append(f"- tasks: {len(task_ids)} · presets: {len(presets)}")
    wall = None
    if run.get("finished_at") and run.get("started_at"):
        wall = run["finished_at"] - run["started_at"]
        out.append(f"- wall clock: {wall:.1f}s")
    out.append("")

    # Matrix
    header = "| task |" + "".join(f" {p} |" for p in presets)
    sep = "|------|" + "".join("------|" for _ in presets)
    out.append(header)
    out.append(sep)
    for tid in task_ids:
        row = [f"`{tid}`"]
        for p in presets:
            row.append(_cell_cell(pivot.get(tid, {}).get(p)))
        out.append("| " + " | ".join(row) + " |")
    out.append("")

    # Per-preset summary
    agg = summarize_by_preset(run)
    out.append("## Summary by preset")
    out.append("")
    out.append(
        "| preset | pass | total | pass-rate | Σ cost | "
        "avg latency | Σ turns | Σ tool calls | errors |"
    )
    out.append(
        "|--------|-----:|-----:|---------:|------:|"
        "-----------:|-------:|------------:|------:|"
    )
    for p in presets:
        s = agg.get(p, {})
        total = s.get("total", 0)
        rate = (s.get("pass", 0) / total * 100.0) if total else 0.0
        avg_lat = (s.get("latency_sum", 0.0) / total) if total else 0.0
        turns = s.get("turn_sum", 0)
        tools = s.get("tool_sum", 0)
        out.append(
            f"| {p} | {s.get('pass', 0)} | {total} | {rate:.0f}% | "
            f"${s.get('cost_sum', 0.0):.4f} | {avg_lat:.1f}s | "
            f"{turns} | {tools} | {s.get('error', 0)} |"
        )
    out.append("")
    return "\n".join(out)
