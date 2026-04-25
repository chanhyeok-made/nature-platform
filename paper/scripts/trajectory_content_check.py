"""Content (Jaccard) analysis for ablation_trajectory.py output.

For each paired comparison (B vs A, C vs A, C vs B):
  paired: Jaccard(A_i, B_i) with same baseline_i
  unpaired: Jaccard(A_i, B_j) with i != j (cross-baseline)
Report paired/unpaired ratio — >1 means shared baseline structures
correlate the branches' content.

Also: detect "which risk was identified" — extract the core risk phrase
from each response and check ACCEPT/REJECT narrative qualitatively.
"""

import json
import math
import re


def tokens(text):
    if not text:
        return set()
    words = re.findall(r"\b[a-z_]{3,}\b", text.lower())
    stop = {"the", "and", "for", "that", "this", "with", "not", "are",
            "from", "into", "could", "would", "than", "also", "will",
            "have", "has", "more", "each", "can", "but", "one", "how",
            "what", "any", "when", "where", "which", "these", "those",
            "such", "then", "now", "only", "just", "very", "most",
            "main", "them", "there", "been", "their", "here", "all",
            "because", "concrete", "answer", "given", "analysis",
            "above", "mitigate", "design", "sentences", "bullet",
            "points", "risk", "response", "your", "you"}
    return {w for w in words if w not in stop}


def jaccard(a, b):
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def mean(xs): return sum(xs) / len(xs) if xs else 0
def std(xs):
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1)) ** 0.5 if xs else 0


def main():
    with open("/tmp/nature-eval/ablation_trajectory.json") as f:
        d = json.load(f)
    seeds = d["seeds"]

    # Build text maps per branch
    branch_labels = ["A_control", "B_accept", "C_reject"]
    texts = {lab: [] for lab in branch_labels}
    for s in seeds:
        for lab in branch_labels:
            br = s["branches"].get(lab, {})
            texts[lab].append(br.get("last_text", ""))

    toks = {lab: [tokens(t) for t in texts[lab]] for lab in branch_labels}

    results = {}
    for x, y in [("A_control", "B_accept"),
                 ("A_control", "C_reject"),
                 ("B_accept", "C_reject")]:
        paired = [jaccard(toks[x][i], toks[y][i]) for i in range(len(seeds))]
        unpaired = []
        for i in range(len(seeds)):
            for j in range(len(seeds)):
                if i != j:
                    unpaired.append(jaccard(toks[x][i], toks[y][j]))
        pm, ps = mean(paired), std(paired)
        um, us = mean(unpaired), std(unpaired)
        se = math.sqrt(ps**2/len(paired) + us**2/len(unpaired)) if us else 0
        z = (pm - um) / se if se else 0
        results[f"{x}_vs_{y}"] = {
            "paired_mean": pm, "paired_sd": ps, "paired_n": len(paired),
            "unpaired_mean": um, "unpaired_sd": us, "unpaired_n": len(unpaired),
            "ratio": pm / um if um else None,
            "z_stat": z,
        }

    # Print concisely
    print("=" * 70)
    print("Trajectory content Jaccard (paired same-baseline vs unpaired cross-baseline)")
    print("=" * 70)
    for k, v in results.items():
        print(f"\n{k}:")
        print(f"  paired   (same baseline):  mean={v['paired_mean']:.3f} ±{v['paired_sd']:.3f} n={v['paired_n']}")
        print(f"  unpaired (cross baseline): mean={v['unpaired_mean']:.3f} ±{v['unpaired_sd']:.3f} n={v['unpaired_n']}")
        print(f"  ratio: {v['ratio']:.2f}  z={v['z_stat']:.2f}")

    # Qualitative: print first 200 chars of each text
    print("\n" + "=" * 70)
    print("Qualitative: first 300ch per branch per seed")
    print("=" * 70)
    for i in range(len(seeds)):
        print(f"\n--- seed {i} ---")
        for lab in branch_labels:
            t = texts[lab][i][:300].replace("\n", " ")
            print(f"  [{lab}] {t}")

    out = {"pairwise_jaccard": results,
           "texts": {lab: texts[lab] for lab in branch_labels}}
    with open("/tmp/nature-eval/trajectory_content.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote /tmp/nature-eval/trajectory_content.json")


if __name__ == "__main__":
    main()
