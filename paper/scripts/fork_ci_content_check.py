"""Re-fetch A and B branch texts from the fork CI experiment and
compute content-agreement metrics to see whether pairing helps on
baseline-dependent (content) metrics even though it didn't help on
baseline-independent surface metrics (length, duration).

Paired stat: for each pair i, measure A_i vs B_i agreement.
Unpaired: cross-pair A_i vs B_j (i!=j) agreement.
If fork's shared prefix matters, paired > unpaired.
"""

import asyncio
import json
import math
import re
import sys

from nature.client import NatureClient


TRIAL_FILE = "/tmp/nature-eval/fork_ci_experiment.json"


def tokens(text):
    """Simple word set, lowercased, dropping tiny/common words."""
    if not text:
        return set()
    words = re.findall(r"\b[a-z_]{3,}\b", text.lower())
    stop = {"the", "and", "for", "that", "this", "with", "not", "are",
            "from", "into", "could", "would", "than", "also", "will",
            "have", "has", "more", "each", "can", "but", "one", "how",
            "what", "any", "when", "where", "which", "these", "those",
            "such", "then", "now", "only", "just", "very", "most",
            "main", "them", "there", "been", "their", "here", "all",
            "all,", "because", "concrete", "answer", "given", "analysis",
            "above", "mitigate", "design", "sentences", "bullet",
            "points", "risk", "response"}
    return set(w for w in words if w not in stop)


def jaccard(a, b):
    if not a and not b: return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


async def fetch_last_text(client, session_id):
    events = await client.snapshot(session_id)
    for e in reversed(events):
        if e.type.value == "message.appended":
            content = e.payload.get("content") or []
            for b in content:
                if isinstance(b, dict) and b.get("type") == "text" and b.get("text"):
                    return b["text"]
    return ""


async def main():
    with open(TRIAL_FILE) as f:
        data = json.load(f)
    trials = data["trials"]

    async with NatureClient() as client:
        # Fetch A and B texts per trial by session id
        A_texts, B_texts = [], []
        for i, t in enumerate(trials):
            # The session IDs weren't saved for A/B branches — only baseline_sid.
            # Fallback: find sessions by listing current sessions on the server.
            pass
        sessions = await client.list_sessions()
        # list_sessions returns only live; also try archived
        try:
            archived = await client.list_archived_sessions()
        except Exception:
            archived = []
        all_ids = {s.session_id for s in sessions}
        all_ids |= {s.session_id for s in archived}

        # We know trials have baseline_sid and fork_event.
        # We can find A/B sessions by scanning for sessions whose
        # parent_session_id matches baseline_sid.
        # SessionInfo.parent_session_id exists.
        parent_to_children = {}
        for s in sessions:
            p = s.parent_session_id
            if p:
                parent_to_children.setdefault(p, []).append(s)
        for s in archived:
            p = getattr(s, "parent_session_id", None)
            if p:
                parent_to_children.setdefault(p, []).append(s)

        retrieved = 0
        for i, t in enumerate(trials):
            children = parent_to_children.get(t["baseline_sid"], [])
            if len(children) < 2:
                print(f"  trial {i}: only {len(children)} children found, skipping")
                continue
            # Sort by creation time — A was forked first
            children.sort(key=lambda s: getattr(s, "created_at", 0))
            a_sid = children[0].session_id
            b_sid = children[1].session_id if len(children) > 1 else None
            a_text = await fetch_last_text(client, a_sid)
            b_text = await fetch_last_text(client, b_sid) if b_sid else ""
            A_texts.append(a_text)
            B_texts.append(b_text)
            retrieved += 1
            print(f"  trial {i}: A_len={len(a_text)}, B_len={len(b_text)}")

        print(f"\nretrieved {retrieved} pairs\n")

        A_toks = [tokens(t) for t in A_texts]
        B_toks = [tokens(t) for t in B_texts]

        # Paired agreement: J(A_i, B_i)
        paired = [jaccard(a, b) for a, b in zip(A_toks, B_toks)]
        # Unpaired: J(A_i, B_j) for all i != j
        unpaired = []
        for i, a in enumerate(A_toks):
            for j, b in enumerate(B_toks):
                if i != j:
                    unpaired.append(jaccard(a, b))

        def mean(xs): return sum(xs) / len(xs) if xs else 0
        def std(xs):
            m = mean(xs)
            return (sum((x-m)**2 for x in xs)/max(1,len(xs)-1))**0.5 if xs else 0

        pm, ps = mean(paired), std(paired)
        um, us = mean(unpaired), std(unpaired)
        p_half = 1.96 * ps / math.sqrt(len(paired)) if paired else 0
        u_half = 1.96 * us / math.sqrt(len(unpaired)) if unpaired else 0

        print("=" * 60)
        print("Jaccard agreement (post-fork A vs B token overlap)")
        print("=" * 60)
        print(f"  paired (same baseline):   mean={pm:.3f} sd={ps:.3f} n={len(paired)}")
        print(f"  unpaired (cross baseline): mean={um:.3f} sd={us:.3f} n={len(unpaired)}")
        print(f"  paired/unpaired ratio:    {pm/um:.2f} (>1 = shared baseline helps)")
        # Two-sample z-test
        if ps and us:
            se = math.sqrt(ps**2/len(paired) + us**2/len(unpaired))
            z = (pm - um) / se if se else 0
            print(f"  z-stat (paired vs unpaired): {z:.2f}")

        # Save
        out = {
            "paired_jaccard": paired,
            "unpaired_jaccard_samples": unpaired[:50],
            "summary": {
                "paired_mean": pm, "paired_sd": ps, "paired_n": len(paired),
                "unpaired_mean": um, "unpaired_sd": us, "unpaired_n": len(unpaired),
                "ratio": pm / um if um else None,
            },
            "A_texts": A_texts,
            "B_texts": B_texts,
        }
        with open("/tmp/nature-eval/fork_ci_content.json", "w") as f:
            json.dump(out, f, indent=2)
        print("\nwrote /tmp/nature-eval/fork_ci_content.json")


if __name__ == "__main__":
    asyncio.run(main())
