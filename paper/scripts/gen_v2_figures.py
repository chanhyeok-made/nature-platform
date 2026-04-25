"""Generate paper figures from Plan v2 data.

Outputs:
  paper/figures/fig_paired_jaccard_regime.png  — 3-prompt paired vs unpaired Jaccard
  paper/figures/fig_ablation_model_swap.png    — replaces prior 3-seed fig with 15-seed P1
"""

import json
import math
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path("paper/figures")
OUT.mkdir(parents=True, exist_ok=True)


STOPWORDS = {
    "a","about","above","after","again","against","ain","all","am","an","and","any","are","aren",
    "aren't","as","at","be","because","been","before","being","below","between","both","but","by",
    "can","couldn","couldn't","d","did","didn","didn't","do","does","doesn","doesn't","doing","don",
    "don't","down","during","each","few","for","from","further","had","hadn","hadn't","has","hasn",
    "hasn't","have","haven","haven't","having","he","her","here","hers","herself","him","himself",
    "his","how","i","if","in","into","is","isn","isn't","it","it's","its","itself","just","ll",
    "m","ma","me","mightn","mightn't","more","most","mustn","mustn't","my","myself","needn","needn't",
    "no","nor","not","now","o","of","off","on","once","only","or","other","our","ours","ourselves",
    "out","over","own","re","s","same","shan","shan't","she","she's","should","should've","shouldn",
    "shouldn't","so","some","such","t","than","that","that'll","the","their","theirs","them",
    "themselves","then","there","these","they","this","those","through","to","too","under","until",
    "up","ve","very","was","wasn","wasn't","we","were","weren","weren't","what","when","where",
    "which","while","who","whom","why","will","with","won","won't","wouldn","wouldn't","y","you",
    "you'd","you'll","you're","you've","your","yours","yourself","yourselves",
}


def tokens(text):
    if not text:
        return set()
    words = re.findall(r"\b[a-z_][a-z_0-9]{2,}\b", text.lower())
    return {w for w in words if w not in STOPWORDS}


def jaccard(a, b):
    if not a and not b: return 1.0
    inter = len(a & b); union = len(a | b)
    return inter / union if union else 0.0


def fig_paired_jaccard_regime():
    """3-prompt paired vs unpaired Jaccard comparison (Block B).

    Shows the regime dependence: P1 strong effect, P2/P3 null.
    """
    with open("/tmp/nature-eval/trajectory_multiprompt.json") as f:
        d = json.load(f)
    trials = d["trials"]
    by_p = {}
    for t in trials:
        by_p.setdefault(t["prompt_id"], []).append(t)

    # For each prompt, compute A vs B paired (same seed) and unpaired (cross)
    prompt_order = ["P1_risk_analysis", "P2_diff_comparison", "P3_code_summary"]
    short_labels = {
        "P1_risk_analysis":     "P1: open-ended\n(risk analysis)",
        "P2_diff_comparison":   "P2: constrained\n(diff comparison)",
        "P3_code_summary":      "P3: selection\n(code summary)",
    }

    paired_means, paired_sds = [], []
    unp_means, unp_sds = [], []
    ratios, zstats = [], []
    for pid in prompt_order:
        ts = by_p[pid]
        A_toks = [tokens(t["branches"].get("A_control", {}).get("last_text", "")) for t in ts]
        B_toks = [tokens(t["branches"].get("B_accept",  {}).get("last_text", "")) for t in ts]
        n = len(ts)
        paired = [jaccard(A_toks[i], B_toks[i]) for i in range(n)]
        unpaired = [jaccard(A_toks[i], B_toks[j]) for i in range(n) for j in range(n) if i != j]
        pm = sum(paired)/len(paired)
        ps = (sum((x-pm)**2 for x in paired)/max(1,len(paired)-1))**0.5
        um = sum(unpaired)/len(unpaired)
        us = (sum((x-um)**2 for x in unpaired)/max(1,len(unpaired)-1))**0.5
        se = math.sqrt(ps**2/len(paired) + us**2/len(unpaired))
        z = (pm - um)/se if se else 0
        paired_means.append(pm); paired_sds.append(ps)
        unp_means.append(um); unp_sds.append(us)
        ratios.append(pm/um if um else 0)
        zstats.append(z)

    # Bar chart
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    x = np.arange(len(prompt_order))
    w = 0.35
    paired_sem = [s/math.sqrt(len(by_p[pid])) for s, pid in zip(paired_sds, prompt_order)]
    unp_sem = [s/math.sqrt(len(by_p[pid])*(len(by_p[pid])-1)) for s, pid in zip(unp_sds, prompt_order)]

    bars1 = ax.bar(x - w/2, paired_means, w, yerr=paired_sem,
                   label="paired (same baseline, $n=15$)",
                   color="#1f77b4", edgecolor="black", linewidth=0.4, capsize=3)
    bars2 = ax.bar(x + w/2, unp_means, w, yerr=unp_sem,
                   label="unpaired (cross baseline, $n=210$)",
                   color="#aed0ea", edgecolor="black", linewidth=0.4, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels([short_labels[p] for p in prompt_order], fontsize=9)
    ax.set_ylabel("Jaccard agreement A vs B", fontsize=10)
    ax.set_title("Paired analysis tightens Jaccard CI only in open-ended regime\n"
                 "(direct-core baseline → A=direct-core vs B=haiku-phi4)",
                 fontsize=11, pad=10)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # Annotate ratio and z above paired bar
    for i, (r, z) in enumerate(zip(ratios, zstats)):
        h = max(paired_means[i], unp_means[i]) + max(paired_sds[i], unp_sds[i]) * 0.3
        label = f"ratio={r:.2f}\n$z={z:.2f}$"
        color = "#1f4e79" if z > 2.0 else "#888"
        ax.text(i, h + 0.008, label, ha="center", va="bottom", fontsize=8.5,
                color=color, fontweight="bold" if z > 2.0 else "normal")

    # Annotate H1 pass/fail
    threshold = "z > 2.0 required for H1 pass"
    ax.text(0.98, 0.02, threshold, transform=ax.transAxes, fontsize=8,
            ha="right", va="bottom", color="#555", style="italic")

    ax.set_ylim(0, max(max(paired_means), max(unp_means)) * 1.5)
    fig.tight_layout()
    out = OUT / "fig_paired_jaccard_regime.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def fig_ablation_model_swap_v2():
    """Replace prior 3-seed model-swap figure with Block B P1 15-seed data.

    Per-seed A (direct-core) vs B (haiku-phi4) final response length.
    """
    with open("/tmp/nature-eval/trajectory_multiprompt.json") as f:
        d = json.load(f)
    trials = [t for t in d["trials"] if t["prompt_id"] == "P1_risk_analysis"]
    trials.sort(key=lambda t: t["seed"])

    fig, ax = plt.subplots(figsize=(10.0, 4.5))
    xs = np.arange(len(trials))
    a_len = [t["branches"].get("A_control", {}).get("last_text_length", 0) for t in trials]
    b_len = [t["branches"].get("B_accept",  {}).get("last_text_length", 0) for t in trials]

    w = 0.35
    ax.bar(xs - w/2, a_len, w, label="Branch A (direct-core, haiku)",
           color="#1f77b4", edgecolor="black", linewidth=0.4)
    ax.bar(xs + w/2, b_len, w, label="Branch B (haiku-phi4, phi4 analyzer/reviewer/judge)",
           color="#ff7f0e", edgecolor="black", linewidth=0.4)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"seed {t['seed']}" for t in trials], fontsize=8, rotation=0)
    ax.set_ylabel("post-fork response length (chars)", fontsize=10)
    ax.set_title("Event-pinned model-swap ablation (P1 open-ended regime, $n=15$)\n"
                 "Same direct-core baseline forked into A/B; response-length delta $t=-2.23$, $p<0.05$",
                 fontsize=10.5, pad=10)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # Horizontal line for mean of each branch
    ax.axhline(sum(a_len)/len(a_len), color="#1f77b4", linestyle=":", alpha=0.4, linewidth=1)
    ax.axhline(sum(b_len)/len(b_len), color="#ff7f0e", linestyle=":", alpha=0.4, linewidth=1)

    fig.tight_layout()
    out = OUT / "fig_ablation_model_swap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    print("Generating v2 figures...")
    fig_paired_jaccard_regime()
    fig_ablation_model_swap_v2()
    print("Done.")
