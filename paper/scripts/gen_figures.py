"""Generate publication figures for paper §5 and §6.

Outputs PNGs to paper/figures/. Data sourced from eval run records.
"""

import json
import glob
import os
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


OUT = Path("paper/figures")
OUT.mkdir(parents=True, exist_ok=True)


# ────────────────────────────────────────────────────────────────────
# Load all preset benchmark data from eval runs since 2026-04-23
# ────────────────────────────────────────────────────────────────────
def load_all_cells():
    runs = sorted(glob.glob(".nature/eval/results/runs/*.json"))
    all_cells = []
    for fp in runs:
        if os.path.getmtime(fp) < 1776960000:
            continue
        with open(fp) as f:
            r = json.load(f)
        for c in r["cells"]:
            all_cells.append(c)
    # De-duplicate by (preset, task_id, seed) keeping most recent.
    # Phase A's 360s-watchdog n1 cells are superseded by the 600s redo.
    latest = {}
    for c in all_cells:
        key = (c.get("preset"), c.get("task_id"), c.get("seed"))
        prev = latest.get(key)
        if prev is None or (c.get("started_at", 0) > prev.get("started_at", 0)):
            latest[key] = c
    # Drop the smoke cell which has seed=None
    return [c for c in latest.values() if c.get("seed") is not None]


CELLS = load_all_cells()
PRESET_ORDER = [
    "default", "direct-core", "haiku-phi4",
    "direct-haiku-phi4", "local-role-optimized",
]
TASK_ORDER = ["s1-csv-parser", "s3-json-pointer", "s4-async-refactor",
              "x1-pluggy-remove-plugin", "n1-pack-discovery"]


def aggregate_pass_rate():
    """Returns {preset: {task: [passed:bool, ...]}}."""
    out = {p: {t: [] for t in TASK_ORDER} for p in PRESET_ORDER}
    for c in CELLS:
        p, t = c.get("preset"), c.get("task_id")
        if p in out and t in out[p]:
            out[p][t].append(bool(c.get("passed")))
    return out


def aggregate_cost_latency():
    """Returns {preset: {cost: [..], latency: [..]}} only PASS cells."""
    out = {p: {"cost": [], "latency": []} for p in PRESET_ORDER}
    for c in CELLS:
        p = c.get("preset")
        if p not in out:
            continue
        if not c.get("passed"):
            continue
        cost = c.get("cost_usd") or 0
        lat = c.get("latency_sec") or 0
        if cost > 0 and lat > 0:
            out[p]["cost"].append(cost)
            out[p]["latency"].append(lat)
    return out


# ────────────────────────────────────────────────────────────────────
# Figure 1: Preset × task heatmap
# ────────────────────────────────────────────────────────────────────
def fig_heatmap():
    data = aggregate_pass_rate()
    rows = []
    annotations = []
    for p in PRESET_ORDER:
        row = []
        ann_row = []
        for t in TASK_ORDER:
            cells = data[p][t]
            if not cells:
                row.append(np.nan)
                ann_row.append("—")
            else:
                pass_rate = sum(cells) / len(cells)
                row.append(pass_rate)
                ann_row.append(f"{sum(cells)}/{len(cells)}")
        rows.append(row)
        annotations.append(ann_row)
    arr = np.array(rows, dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    # Custom colormap: red -> yellow -> green
    cmap = plt.cm.RdYlGn
    im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=0, vmax=1.0)
    ax.set_xticks(range(len(TASK_ORDER)))
    ax.set_xticklabels([t.split("-", 1)[1] if "-" in t else t for t in TASK_ORDER],
                       rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(len(PRESET_ORDER)))
    ax.set_yticklabels(PRESET_ORDER, fontsize=9)
    ax.set_xlabel("task", fontsize=10)
    ax.set_ylabel("preset", fontsize=10)
    for i in range(len(PRESET_ORDER)):
        for j in range(len(TASK_ORDER)):
            txt = annotations[i][j]
            # Choose text color for contrast
            val = arr[i][j]
            color = "white" if (not np.isnan(val) and val < 0.35) else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    color=color, fontsize=10, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("pass rate", fontsize=9)
    ax.set_title("Preset × task pass-rate matrix (5 tasks × 5 presets × 2 seeds)",
                 fontsize=11, pad=12)
    fig.tight_layout()
    out = OUT / "fig_preset_task_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ────────────────────────────────────────────────────────────────────
# Figure 2: Cost × latency Pareto scatter
# ────────────────────────────────────────────────────────────────────
def fig_pareto():
    data = aggregate_cost_latency()
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    colors = plt.cm.tab10(np.linspace(0, 1, len(PRESET_ORDER)))
    # Manual label offsets to avoid overlap at similar coordinates
    label_offsets = {
        "default": (10, 10),
        "direct-core": (10, -15),
        "haiku-phi4": (-75, 8),
        "direct-haiku-phi4": (10, 5),
        "local-role-optimized": (-110, 10),
    }
    for i, p in enumerate(PRESET_ORDER):
        costs = data[p]["cost"]
        lats = data[p]["latency"]
        if not costs:
            continue
        avg_cost = statistics.mean(costs)
        avg_lat = statistics.mean(lats)
        n = len(costs)
        ax.scatter([avg_cost], [avg_lat], s=180, c=[colors[i]],
                   label=f"{p} (n={n})", edgecolors="black", linewidth=0.8,
                   zorder=3)
        # Annotate with per-preset offset
        off = label_offsets.get(p, (8, 5))
        ax.annotate(p, (avg_cost, avg_lat), fontsize=9,
                    xytext=off, textcoords="offset points",
                    color=colors[i], fontweight="bold")
        # Show individual cells with low alpha
        ax.scatter(costs, lats, s=30, c=[colors[i]], alpha=0.3, zorder=2)
    ax.set_xlabel("average cost per passed cell (USD)", fontsize=10)
    ax.set_ylabel("average latency per passed cell (s)", fontsize=10)
    ax.set_title("Cost vs latency Pareto (average of passed cells per preset)",
                 fontsize=11, pad=10)
    ax.grid(alpha=0.3, zorder=0)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.95)
    fig.tight_layout()
    out = OUT / "fig_cost_latency_pareto.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ────────────────────────────────────────────────────────────────────
# Figure 3: Per-task cost bars (grouped)
# ────────────────────────────────────────────────────────────────────
def fig_cost_bars():
    # For each task, plot average cost per preset (only PASS cells)
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    width = 0.15
    xs = np.arange(len(TASK_ORDER))
    colors = plt.cm.tab10(np.linspace(0, 1, len(PRESET_ORDER)))
    for i, p in enumerate(PRESET_ORDER):
        costs_per_task = []
        for t in TASK_ORDER:
            costs = [c.get("cost_usd") or 0 for c in CELLS
                     if c.get("preset") == p and c.get("task_id") == t
                     and c.get("passed")]
            avg = statistics.mean(costs) if costs else 0
            costs_per_task.append(avg)
        offset = (i - len(PRESET_ORDER) / 2) * width + width / 2
        bars = ax.bar(xs + offset, costs_per_task, width, label=p,
                      color=colors[i], edgecolor="black", linewidth=0.4)
        # Annotate zero bars with "fail"
        for j, cpt in enumerate(costs_per_task):
            if cpt == 0:
                ax.text(xs[j] + offset, 0.01, "FAIL",
                        ha="center", va="bottom", fontsize=7, rotation=90,
                        color="#b00")
    ax.set_xticks(xs)
    ax.set_xticklabels([t.split("-", 1)[1] if "-" in t else t for t in TASK_ORDER],
                       rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("average cost per passed cell (USD)", fontsize=10)
    ax.set_title("Per-task cost (passed cells only — FAIL marks indicate all seeds failed)",
                 fontsize=11, pad=10)
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = OUT / "fig_per_task_cost.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ────────────────────────────────────────────────────────────────────
# Figure 4: 2x2 factorial interaction plot
# ────────────────────────────────────────────────────────────────────
def fig_factorial():
    # topology × model-class with pass rates
    # all-haiku: default (80%) vs direct-core (100%)
    # hybrid:     haiku-phi4 (100%) vs direct-haiku-phi4 (80%)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = [0, 1]  # with/without receptionist
    all_haiku = [80, 100]
    hybrid = [100, 80]
    ax.plot(x, all_haiku, "o-", linewidth=2, markersize=12,
            label="all-haiku downstream", color="#1f77b4")
    ax.plot(x, hybrid, "s-", linewidth=2, markersize=12,
            label="haiku+phi4 hybrid downstream", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(["with receptionist\n(default / haiku-phi4)",
                        "without receptionist\n(direct-core / direct-haiku-phi4)"],
                       fontsize=10)
    ax.set_ylabel("pass rate over 10 cells (%)", fontsize=10)
    ax.set_ylim(60, 110)
    ax.set_yticks([60, 70, 80, 90, 100])
    for i, v in enumerate(all_haiku):
        ax.annotate(f"{v}%", (x[i], v), fontsize=10,
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", color="#1f77b4")
    for i, v in enumerate(hybrid):
        ax.annotate(f"{v}%", (x[i], v), fontsize=10,
                    xytext=(0, -18), textcoords="offset points",
                    ha="center", color="#ff7f0e")
    ax.set_title("Receptionist × downstream-model interaction effect\n"
                 "(paths cross — effect of receptionist depends on downstream)",
                 fontsize=11, pad=10)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = OUT / "fig_factorial_interaction.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ────────────────────────────────────────────────────────────────────
# Figure 5: Ablation (event-pinned fork) comparison
# ────────────────────────────────────────────────────────────────────
def fig_ablation():
    # From /tmp/nature-eval/ablation_multi.json — the model-swap seeds
    with open("/tmp/nature-eval/ablation_multi.json") as f:
        amd = json.load(f)
    trials = amd["trials"]
    # Seed 0,1,2 are model-swap; trial 3 is prompt-ablation
    ms = [t for t in trials if "model-swap" in t["trial_name"]]
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    xs = np.arange(len(ms))
    a_dur = [t["A"]["duration_s"] for t in ms]
    b_dur = [t["B"]["duration_s"] for t in ms]
    a_len = [t["A"]["last_text_length"] for t in ms]
    b_len = [t["B"]["last_text_length"] for t in ms]

    w = 0.35
    ax.bar(xs - w / 2, a_dur, w, label="Branch A (direct-core)",
           color="#1f77b4", edgecolor="black", linewidth=0.4)
    ax.bar(xs + w / 2, b_dur, w, label="Branch B (haiku-phi4)",
           color="#ff7f0e", edgecolor="black", linewidth=0.4)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"seed {i}\nfork@event 34" for i in range(len(ms))],
                       fontsize=9)
    ax.set_ylabel("post-fork duration (s)", fontsize=10)
    ax.set_title("Event-pinned model-swap ablation\n"
                 "post-fork delta attributable solely to sub-agent model choice",
                 fontsize=11, pad=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    # Annotate bars with response length
    for i in range(len(ms)):
        ax.text(xs[i] - w / 2, a_dur[i] + 0.2, f"{a_len[i]}ch",
                ha="center", fontsize=8, color="#1f77b4")
        ax.text(xs[i] + w / 2, b_dur[i] + 0.2, f"{b_len[i]}ch",
                ha="center", fontsize=8, color="#ff7f0e")
    fig.tight_layout()
    out = OUT / "fig_ablation_model_swap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ────────────────────────────────────────────────────────────────────
# Figure 6: Fork lineage diagram (schematic)
# ────────────────────────────────────────────────────────────────────
def fig_fork_diagram():
    fig, ax = plt.subplots(figsize=(9.0, 3.5))
    # Baseline timeline
    ax.plot([0, 8], [2, 2], "k-", linewidth=2)
    ax.scatter([0, 2, 4, 6, 8], [2]*5, s=60, c="#555",
               edgecolors="black", zorder=3)
    # Fork at event 4
    ax.plot([8, 12], [2.7, 3.2], color="#1f77b4", linewidth=2)
    ax.plot([8, 12], [1.3, 0.8], color="#ff7f0e", linewidth=2)
    ax.scatter([10, 11.5], [3.0, 3.2], s=60, c="#1f77b4",
               edgecolors="black", zorder=3)
    ax.scatter([10, 11.5], [1.0, 0.8], s=60, c="#ff7f0e",
               edgecolors="black", zorder=3)
    # Labels
    ax.annotate("baseline\n(preset P)", (0, 2.3), fontsize=10, ha="left")
    ax.annotate("fork point\n(last event)", (8, 2), fontsize=9,
                xytext=(8, 1.7), ha="center",
                arrowprops=dict(arrowstyle="->", color="black", lw=1))
    ax.annotate("Branch A: preset P\n(control)", (12, 3.2), fontsize=10,
                color="#1f77b4", ha="left")
    ax.annotate("Branch B: preset Q\n(treatment)", (12, 0.8), fontsize=10,
                color="#ff7f0e", ha="left")
    # Shared-prefix shading
    rect = patches.Rectangle((0, 1.7), 8, 0.6, linewidth=0,
                             facecolor="#dddddd", alpha=0.5, zorder=1)
    ax.add_patch(rect)
    ax.text(4, 2.0, "shared byte-identical prefix — events 1..N",
            ha="center", fontsize=9, style="italic", color="#333",
            zorder=2)
    ax.text(11, 4.0, "post-fork delta attributable only to preset",
            fontsize=9, style="italic", color="#333",
            ha="center")

    ax.set_xlim(-1, 14)
    ax.set_ylim(0, 4.5)
    ax.axis("off")
    ax.set_title("Event-pinned counterfactual fork", fontsize=12, pad=8)
    fig.tight_layout()
    out = OUT / "fig_fork_schematic.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    print("Generating paper figures...")
    fig_heatmap()
    fig_pareto()
    fig_cost_bars()
    fig_factorial()
    fig_ablation()
    fig_fork_diagram()
    print("Done.")
