"""Generate an architecture schematic for paper §3.

Box/arrow diagram showing the six-layer decomposition and the
compose-and-dispatch relationship between them.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

OUT = Path("paper/figures")
OUT.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 6.5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis("off")

# Color palette
PRIMARY = "#1f77b4"
SECONDARY = "#ff7f0e"
ACCENT = "#2ca02c"
MUTED = "#888"
PACK = "#9467bd"

def draw_box(ax, x, y, w, h, label, sub=None, color="white", edge="black",
             lw=1.0, fontsize=10, sub_fontsize=8):
    rect = patches.Rectangle((x, y), w, h, linewidth=lw,
                             facecolor=color, edgecolor=edge, zorder=2)
    ax.add_patch(rect)
    if sub:
        ax.text(x + w / 2, y + h * 0.62, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", zorder=3)
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center",
                fontsize=sub_fontsize, color="#333", style="italic", zorder=3)
    else:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", zorder=3)

def draw_arrow(ax, x1, y1, x2, y2, color="black", lw=1.2, style="->"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw),
                zorder=2)

# ─── Top layer: Preset ─────────────────────────────────────────────
draw_box(ax, 5.0, 8.2, 4.0, 1.1, "Preset",
         sub="root_agent + agents + overrides", color="#e3f2fd", edge=PRIMARY,
         fontsize=12, sub_fontsize=8)

# ─── Middle layer: Agents (3 cards) ─────────────────────────────────
draw_box(ax, 1.0, 6.0, 2.6, 1.4, "Agent",
         sub="JSON + instructions.md", color="#fff3e0", edge=SECONDARY,
         fontsize=11, sub_fontsize=8)
draw_box(ax, 5.7, 6.0, 2.6, 1.4, "Agent",
         sub="role contract", color="#fff3e0", edge=SECONDARY,
         fontsize=11, sub_fontsize=8)
draw_box(ax, 10.4, 6.0, 2.6, 1.4, "Agent",
         sub="tool allowlist", color="#fff3e0", edge=SECONDARY,
         fontsize=11, sub_fontsize=8)

# Preset → Agents
draw_arrow(ax, 6.2, 8.2, 2.3, 7.4, color=PRIMARY, lw=1.3)
draw_arrow(ax, 7.0, 8.2, 7.0, 7.4, color=PRIMARY, lw=1.3)
draw_arrow(ax, 7.8, 8.2, 11.7, 7.4, color=PRIMARY, lw=1.3)

# ─── Host layer ─────────────────────────────────────────────────────
draw_box(ax, 1.0, 4.0, 4.0, 1.1, "Host",
         sub="endpoint + model catalog", color="#e8f5e9", edge=ACCENT,
         fontsize=11, sub_fontsize=8)
draw_box(ax, 9.0, 4.0, 4.0, 1.1, "Host",
         sub="Anthropic / Ollama / OpenRouter", color="#e8f5e9", edge=ACCENT,
         fontsize=11, sub_fontsize=8)

# Agents → Hosts
draw_arrow(ax, 2.3, 6.0, 3.0, 5.1, color=ACCENT, lw=1.0, style="->")
draw_arrow(ax, 7.0, 6.0, 4.5, 5.1, color=ACCENT, lw=1.0, style="->")
draw_arrow(ax, 7.0, 6.0, 10.5, 5.1, color=ACCENT, lw=1.0, style="->")
draw_arrow(ax, 11.7, 6.0, 11.0, 5.1, color=ACCENT, lw=1.0, style="->")

# ─── Execution: AreaManager ─────────────────────────────────────────
draw_box(ax, 1.0, 2.0, 12.0, 1.4,
         "AreaManager  (event-sourced execution loop)",
         sub="open_root / run / dispatch / emit → EventStore",
         color="#f5f5f5", edge="black", lw=1.4,
         fontsize=12, sub_fontsize=9)

# Preset → manager (configures)
draw_arrow(ax, 7.0, 8.2, 7.0, 3.4, color=MUTED, lw=0.8, style="-|>")
ax.text(7.15, 6.5, "configures", fontsize=7.5, color=MUTED,
        style="italic", ha="left", rotation=0)

# ─── Side: Pack (gates/listeners/contributors) ──────────────────────
draw_box(ax, 0.2, 0.2, 3.5, 1.2, "Pack",
         sub="Gate / Listener / Contributor",
         color="#f3e5f5", edge=PACK,
         fontsize=11, sub_fontsize=8)
draw_arrow(ax, 2.0, 1.4, 4.5, 2.0, color=PACK, lw=1.0, style="->")
ax.text(3.5, 1.55, "installs hooks", fontsize=7.5, color=PACK,
        style="italic", rotation=12)

# ─── Side: EventStore (feeds replay + fork) ─────────────────────────
draw_box(ax, 10.3, 0.2, 3.5, 1.2, "EventStore",
         sub="JSONL per session → replay / fork",
         color="#fffde7", edge="#f9a825",
         fontsize=11, sub_fontsize=8)
draw_arrow(ax, 11.0, 2.0, 11.8, 1.4, color="#f9a825", lw=1.0, style="->")
ax.text(10.6, 1.55, "appends", fontsize=7.5, color="#b8860b",
        style="italic", rotation=-12)

# ─── Body compaction (sits inline with manager) ─────────────────────
draw_box(ax, 5.0, 0.4, 4.0, 1.0,
         "BodyCompactionPipeline",
         sub="pre-LLM context trim",
         color="#fafafa", edge=MUTED, lw=0.8,
         fontsize=10, sub_fontsize=7)
draw_arrow(ax, 7.0, 2.0, 7.0, 1.4, color=MUTED, lw=0.8, style="->")
ax.text(7.15, 1.6, "gates LLM call", fontsize=7, color=MUTED,
        style="italic")

# Title annotation
ax.text(7.0, 9.8,
        "nature — six-layer composition around an event-sourced execution loop",
        ha="center", fontsize=12, fontweight="bold")

plt.tight_layout()
out = OUT / "fig_architecture.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out}")
