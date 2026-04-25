"""Plan v2 analysis — paired vs unpaired Jaccard/length across both blocks.

Standardized stop-word list (NLTK English, 179 words, documented in paper).
No prompt-specific tuning. Any template word that appears in all responses
contributes equally to paired and unpaired Jaccard, so the RATIO is
unaffected by template matching.

Reads:
  /tmp/nature-eval/fork_ci_v2_n30.json           (Block A, 2-branch N=30)
  /tmp/nature-eval/trajectory_multiprompt.json   (Block B, 3 prompts × 3 branches × 15 seeds)

Writes:
  /tmp/nature-eval/v2_analysis.json
"""
import json
import math
import re
from pathlib import Path


# NLTK English stopwords (v3.9.1 list), 179 words
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "ain", "all", "am",
    "an", "and", "any", "are", "aren", "aren't", "as", "at", "be", "because",
    "been", "before", "being", "below", "between", "both", "but", "by", "can",
    "couldn", "couldn't", "d", "did", "didn", "didn't", "do", "does", "doesn",
    "doesn't", "doing", "don", "don't", "down", "during", "each", "few",
    "for", "from", "further", "had", "hadn", "hadn't", "has", "hasn",
    "hasn't", "have", "haven", "haven't", "having", "he", "her", "here",
    "hers", "herself", "him", "himself", "his", "how", "i", "if", "in",
    "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just",
    "ll", "m", "ma", "me", "mightn", "mightn't", "more", "most", "mustn",
    "mustn't", "my", "myself", "needn", "needn't", "no", "nor", "not", "now",
    "o", "of", "off", "on", "once", "only", "or", "other", "our", "ours",
    "ourselves", "out", "over", "own", "re", "s", "same", "shan", "shan't",
    "she", "she's", "should", "should've", "shouldn", "shouldn't", "so",
    "some", "such", "t", "than", "that", "that'll", "the", "their", "theirs",
    "them", "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "ve", "very", "was",
    "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "when",
    "where", "which", "while", "who", "whom", "why", "will", "with", "won",
    "won't", "wouldn", "wouldn't", "y", "you", "you'd", "you'll", "you're",
    "you've", "your", "yours", "yourself", "yourselves",
}


def tokens(text):
    if not text:
        return set()
    words = re.findall(r"\b[a-z_][a-z_0-9]{2,}\b", text.lower())
    return {w for w in words if w not in STOPWORDS}


def jaccard(a, b):
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def mean(xs): return sum(xs) / len(xs) if xs else 0
def std(xs):
    m = mean(xs)
    return (sum((x-m)**2 for x in xs)/max(1,len(xs)-1))**0.5 if xs else 0


def ci_halfwidth(xs, z=1.96):
    return z * std(xs) / math.sqrt(len(xs)) if len(xs) > 1 else 0


def paired_unpaired_stats(A_toks, B_toks):
    """Given lists of token sets from A/B branches (same-indexed), compute
    paired Jaccard stats and unpaired (cross-index) stats."""
    n = len(A_toks)
    paired = [jaccard(A_toks[i], B_toks[i]) for i in range(n)]
    unpaired = []
    for i in range(n):
        for j in range(n):
            if i != j:
                unpaired.append(jaccard(A_toks[i], B_toks[j]))
    pm, ps = mean(paired), std(paired)
    um, us = mean(unpaired), std(unpaired)
    se = math.sqrt(ps**2/len(paired) + us**2/len(unpaired)) if us else 0
    z = (pm - um) / se if se else 0
    return {
        "paired_mean": pm, "paired_sd": ps, "paired_n": len(paired),
        "unpaired_mean": um, "unpaired_sd": us, "unpaired_n": len(unpaired),
        "ratio": pm / um if um else None,
        "z_stat": z,
    }


# ─── Block A analysis ──────────────────────────────────────────────
def analyze_block_a():
    path = Path("/tmp/nature-eval/fork_ci_v2_n30.json")
    if not path.exists():
        return {"error": "not yet run"}
    with open(path) as f:
        d = json.load(f)
    trials = d["trials"]

    # Length stats
    A_lens = [t["A_text_length"] for t in trials]
    B_lens = [t["B_text_length"] for t in trials]
    deltas = [t["delta_length_B_minus_A"] for t in trials]

    length_stats = {
        "n": len(trials),
        "A_mean": mean(A_lens), "A_sd": std(A_lens),
        "B_mean": mean(B_lens), "B_sd": std(B_lens),
        "delta_mean": mean(deltas), "delta_sd": std(deltas),
        "unpaired_95ci_halfwidth": 1.96 * math.sqrt(std(A_lens)**2/len(A_lens) + std(B_lens)**2/len(B_lens)),
        "paired_95ci_halfwidth": ci_halfwidth(deltas),
    }
    length_stats["ratio_unpaired_over_paired"] = (
        length_stats["unpaired_95ci_halfwidth"] / length_stats["paired_95ci_halfwidth"]
        if length_stats["paired_95ci_halfwidth"] > 0 else None
    )

    # Jaccard stats (using texts from trials)
    A_toks = [tokens(t.get("A_text", "")) for t in trials]
    B_toks = [tokens(t.get("B_text", "")) for t in trials]
    jaccard_stats = paired_unpaired_stats(A_toks, B_toks)

    return {
        "n_trials": len(trials),
        "length": length_stats,
        "jaccard": jaccard_stats,
    }


# ─── Block B analysis ──────────────────────────────────────────────
def analyze_block_b():
    path = Path("/tmp/nature-eval/trajectory_multiprompt.json")
    if not path.exists():
        return {"error": "not yet run"}
    with open(path) as f:
        d = json.load(f)
    trials = d["trials"]

    # Group by prompt_id
    by_prompt = {}
    for t in trials:
        by_prompt.setdefault(t["prompt_id"], []).append(t)

    branch_labels = [b["label"] for b in d["branches"]]

    per_prompt = {}
    all_A_toks_agg, all_B_toks_agg, all_C_toks_agg = [], [], []
    for pid, ts in by_prompt.items():
        A_texts = [t["branches"].get("A_control", {}).get("last_text", "") for t in ts]
        B_texts = [t["branches"].get("B_accept",  {}).get("last_text", "") for t in ts]
        C_texts = [t["branches"].get("C_reject",  {}).get("last_text", "") for t in ts]

        A_toks = [tokens(x) for x in A_texts]
        B_toks = [tokens(x) for x in B_texts]
        C_toks = [tokens(x) for x in C_texts]

        A_lens = [len(x) for x in A_texts]
        B_lens = [len(x) for x in B_texts]
        C_lens = [len(x) for x in C_texts]

        per_prompt[pid] = {
            "n_seeds": len(ts),
            "length": {
                "A_mean": mean(A_lens), "A_sd": std(A_lens),
                "B_mean": mean(B_lens), "B_sd": std(B_lens),
                "C_mean": mean(C_lens), "C_sd": std(C_lens),
                "B_minus_A_delta_mean": mean([b-a for a,b in zip(A_lens, B_lens)]),
                "B_minus_A_delta_sd":   std([b-a for a,b in zip(A_lens, B_lens)]),
                "C_minus_A_delta_mean": mean([c-a for a,c in zip(A_lens, C_lens)]),
                "C_minus_A_delta_sd":   std([c-a for a,c in zip(A_lens, C_lens)]),
            },
            "jaccard": {
                "A_vs_B": paired_unpaired_stats(A_toks, B_toks),
                "A_vs_C": paired_unpaired_stats(A_toks, C_toks),
                "B_vs_C": paired_unpaired_stats(B_toks, C_toks),
            },
        }

        all_A_toks_agg.extend(A_toks)
        all_B_toks_agg.extend(B_toks)
        all_C_toks_agg.extend(C_toks)

    # Aggregate across prompts (pool all tokens)
    aggregate_jaccard = {
        "A_vs_B": paired_unpaired_stats(all_A_toks_agg, all_B_toks_agg),
        "A_vs_C": paired_unpaired_stats(all_A_toks_agg, all_C_toks_agg),
        "B_vs_C": paired_unpaired_stats(all_B_toks_agg, all_C_toks_agg),
    }

    # H1/H2/H3 evaluation
    h1_per_prompt_z = [per_prompt[p]["jaccard"]["A_vs_B"]["z_stat"] for p in per_prompt]
    h1_agg_z = aggregate_jaccard["A_vs_B"]["z_stat"]
    h1_pass = h1_agg_z > 2.0

    # H2: length ratio ≈ 1
    h2_length_ratios = []
    for p, d in per_prompt.items():
        la = d["length"]
        unpaired_hw = 1.96 * math.sqrt(la["A_sd"]**2/d["n_seeds"] + la["B_sd"]**2/d["n_seeds"])
        paired_hw = 1.96 * la["B_minus_A_delta_sd"] / math.sqrt(d["n_seeds"])
        ratio = unpaired_hw / paired_hw if paired_hw > 0 else None
        h2_length_ratios.append((p, ratio))
    h2_in_range = all(r is not None and 0.90 <= r <= 1.10 for _, r in h2_length_ratios)

    # H3: B ratio > 1, C ratio < 1 in ≥2/3 prompts
    h3_b_gt1 = sum(1 for p in per_prompt if per_prompt[p]["jaccard"]["A_vs_B"]["ratio"] > 1.0)
    h3_c_lt1 = sum(1 for p in per_prompt if per_prompt[p]["jaccard"]["A_vs_C"]["ratio"] < 1.0)
    h3_pass = h3_b_gt1 >= 2 and h3_c_lt1 >= 2

    return {
        "per_prompt": per_prompt,
        "aggregate_jaccard": aggregate_jaccard,
        "hypotheses": {
            "H1_paired_gt_unpaired_aggregate": {
                "aggregate_z": h1_agg_z,
                "per_prompt_z": h1_per_prompt_z,
                "pass_threshold_z_gt_2": h1_pass,
            },
            "H2_length_ratio_near_unity": {
                "per_prompt_ratios": h2_length_ratios,
                "all_in_0.90_1.10": h2_in_range,
            },
            "H3_mutation_magnitude_reflected": {
                "B_ratio_gt1_count": h3_b_gt1,
                "C_ratio_lt1_count": h3_c_lt1,
                "pass_ge_2_of_3_both": h3_pass,
            },
        },
    }


def main():
    a = analyze_block_a()
    b = analyze_block_b()
    out = {"block_a": a, "block_b": b}
    with open("/tmp/nature-eval/v2_analysis.json", "w") as f:
        json.dump(out, f, indent=2)

    # Print
    print("=" * 80)
    print("BLOCK A — fork_ci v2 N=30")
    print("=" * 80)
    if "error" in a:
        print(f"  [{a['error']}]")
    else:
        print(f"  n_trials: {a['n_trials']}")
        ls = a["length"]
        print(f"  Length: A mean={ls['A_mean']:.1f}±{ls['A_sd']:.1f}, B mean={ls['B_mean']:.1f}±{ls['B_sd']:.1f}")
        print(f"          delta {ls['delta_mean']:.1f}±{ls['delta_sd']:.1f}")
        print(f"          CI halfwidth: unpaired={ls['unpaired_95ci_halfwidth']:.2f}, paired={ls['paired_95ci_halfwidth']:.2f}")
        print(f"          ratio (unp/paired) = {ls['ratio_unpaired_over_paired']:.4f}  (H2: expect ≈ 1.0)")
        js = a["jaccard"]
        print(f"  Jaccard paired: {js['paired_mean']:.4f}±{js['paired_sd']:.4f} (n={js['paired_n']})")
        print(f"  Jaccard unpaired: {js['unpaired_mean']:.4f}±{js['unpaired_sd']:.4f} (n={js['unpaired_n']})")
        print(f"  Ratio: {js['ratio']:.3f}, z={js['z_stat']:.2f}  (H1: expect z > 2.0)")

    print()
    print("=" * 80)
    print("BLOCK B — trajectory multiprompt (3 prompts × 15 seeds)")
    print("=" * 80)
    if "error" in b:
        print(f"  [{b['error']}]")
    else:
        for pid, pd in b["per_prompt"].items():
            print(f"\n  Prompt {pid}: n={pd['n_seeds']}")
            for pair in ("A_vs_B", "A_vs_C", "B_vs_C"):
                j = pd["jaccard"][pair]
                print(f"    {pair}: paired={j['paired_mean']:.3f} unpaired={j['unpaired_mean']:.3f} ratio={j['ratio']:.2f} z={j['z_stat']:.2f}")
            ls = pd["length"]
            print(f"    length: B-A delta={ls['B_minus_A_delta_mean']:.0f}±{ls['B_minus_A_delta_sd']:.0f}, "
                  f"C-A delta={ls['C_minus_A_delta_mean']:.0f}±{ls['C_minus_A_delta_sd']:.0f}")
        print(f"\n  AGGREGATE (all 3 prompts pooled):")
        for pair in ("A_vs_B", "A_vs_C", "B_vs_C"):
            j = b["aggregate_jaccard"][pair]
            print(f"    {pair}: paired={j['paired_mean']:.3f} unpaired={j['unpaired_mean']:.3f} ratio={j['ratio']:.2f} z={j['z_stat']:.2f}")

        print()
        print("  HYPOTHESIS TESTS")
        h = b["hypotheses"]
        print(f"    H1 (aggregate z > 2.0): z={h['H1_paired_gt_unpaired_aggregate']['aggregate_z']:.2f} → {'PASS' if h['H1_paired_gt_unpaired_aggregate']['pass_threshold_z_gt_2'] else 'FAIL'}")
        print(f"    H2 (length ratio in [0.90,1.10] all prompts): {'PASS' if h['H2_length_ratio_near_unity']['all_in_0.90_1.10'] else 'FAIL'}")
        print(f"       ratios: {h['H2_length_ratio_near_unity']['per_prompt_ratios']}")
        print(f"    H3 (B>1 in ≥2/3 AND C<1 in ≥2/3): B_count={h['H3_mutation_magnitude_reflected']['B_ratio_gt1_count']}/3, C_count={h['H3_mutation_magnitude_reflected']['C_ratio_lt1_count']}/3 → {'PASS' if h['H3_mutation_magnitude_reflected']['pass_ge_2_of_3_both'] else 'FAIL'}")

    print(f"\nwrote /tmp/nature-eval/v2_analysis.json")


if __name__ == "__main__":
    main()
