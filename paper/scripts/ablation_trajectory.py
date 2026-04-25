"""Workflow-trajectory ablation for paper §7 flagship.

Extends the §8 model-swap ablation (ablation_multi.py) with:
  - 5 seeds (up from 3), tightening §8's sample-size claim
  - A third branch (direct-haiku-phi4) framed as a REJECT candidate,
    demonstrating the ACCEPT/REJECT decision structure of the tuning
    workflow rather than a one-shot swap.

Design per seed i ∈ {0..4}:
  baseline_i = direct-core, BASELINE_PROMPT → run to idle
  fork baseline_i at last event into three branches, each with
  FOLLOWUP_PROMPT:
    A_i = direct-core                (control: same preset)
    B_i = haiku-phi4                 (ACCEPT candidate: analyzer/reviewer/judge → phi4)
    C_i = direct-haiku-phi4          (REJECT candidate: no receptionist + phi4 subagents;
                                      §5 notes silent-failure risk on n1)

Writes /tmp/nature-eval/ablation_trajectory.json.
Probe-prior justification (for the paper prose in §7):
  phi4:14b scores 25/29 on probe v8, placing it in the local-top tier
  together with qwen2.5:72b. That prior motivates routing
  analyzer/reviewer/judge to phi4 (B), while §5's direct-haiku-phi4
  cost-latency point (fastest and cheapest but with observed silent
  failure on n1-pack-discovery) motivates testing it as a REJECT
  candidate in the same trajectory.
"""

import asyncio
import json
import math
import sys
import time

from nature.client import NatureClient


N_SEEDS = 5

BASELINE_PROMPT = (
    "Read `nature/frame/manager.py`. In 3 bullet points, list the key "
    "responsibilities of the AreaManager class. Keep each bullet under "
    "20 words. Do not delegate — answer directly."
)

FOLLOWUP_PROMPT = (
    "Given your analysis above, what is ONE concrete risk in the current "
    "design, and how would you mitigate it? Answer in 2-3 sentences."
)

BRANCHES = [
    ("direct-core",         "A_control"),
    ("haiku-phi4",          "B_accept"),
    ("direct-haiku-phi4",   "C_reject"),
]


async def wait_idle(client, sid, timeout=180):
    start = time.time()
    phase1_deadline = start + 15
    while time.time() < phase1_deadline:
        info = await client.get_session(sid)
        if info.has_active_run:
            break
        await asyncio.sleep(0.5)
    stable_since = None
    while time.time() - start < timeout:
        info = await client.get_session(sid)
        if info.has_active_run:
            stable_since = None
            await asyncio.sleep(2)
            continue
        if stable_since is None:
            stable_since = time.time()
        elif time.time() - stable_since >= 5:
            return time.time() - start
        await asyncio.sleep(1)
    raise TimeoutError(f"session {sid} still active after {timeout}s")


async def summarize(client, sid):
    events = await client.snapshot(sid)
    turns = sum(1 for e in events if e.type.value == "llm.response")
    tool_calls = sum(1 for e in events if e.type.value == "tool.started")
    last_text = ""
    for e in reversed(events):
        if e.type.value == "message.appended":
            content = e.payload.get("content") or []
            for b in content:
                if isinstance(b, dict) and b.get("type") == "text" and b.get("text"):
                    last_text = b["text"]
                    break
            if last_text:
                break
    return {
        "session_id": sid,
        "total_events": len(events),
        "turns": turns,
        "tool_calls": tool_calls,
        "last_text": last_text,
        "last_text_length": len(last_text),
    }


async def run_seed(client, seed):
    print(f"\n=== seed {seed}/{N_SEEDS-1} ===", flush=True)
    s = await client.create_session(preset="direct-core")
    sid = s.session_id
    print(f"  baseline sid={sid[:8]} preset=direct-core")
    await client.send_message(sid, BASELINE_PROMPT)
    b_dur = await wait_idle(client, sid)
    b_sum = await summarize(client, sid)
    print(f"  baseline done {b_dur:.1f}s events={b_sum['total_events']}")

    events = await client.snapshot(sid)
    fork_at = max(e.id or 0 for e in events)

    branch_results = {}
    for preset, label in BRANCHES:
        try:
            f = await client.fork_session(sid, at_event_id=fork_at, preset=preset)
            fsid = f.session_id
            print(f"  [{label}] fork sid={fsid[:8]} preset={preset}")
            await client.send_message(fsid, FOLLOWUP_PROMPT)
            dur = await wait_idle(client, fsid)
            summ = await summarize(client, fsid)
            branch_results[label] = {
                "preset": preset, "duration_s": dur, **summ,
            }
            print(f"  [{label}] done {dur:.1f}s textlen={summ['last_text_length']}")
        except Exception as exc:
            print(f"  [{label}] FAILED: {exc}")
            branch_results[label] = {"preset": preset, "error": str(exc)}

    return {
        "seed": seed,
        "baseline_sid": sid,
        "baseline_duration_s": b_dur,
        "baseline_summary": b_sum,
        "fork_event": fork_at,
        "branches": branch_results,
    }


def stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0}
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1)) ** 0.5
    return {"n": len(xs), "mean": m, "sd": sd,
            "ci_halfwidth_95": 1.96 * sd / math.sqrt(len(xs)) if len(xs) > 0 else 0}


async def main():
    async with NatureClient() as client:
        if not await client.is_alive():
            print("ERROR: server not running", file=sys.stderr)
            sys.exit(1)

        seeds = []
        for i in range(N_SEEDS):
            try:
                r = await run_seed(client, i)
                seeds.append(r)
            except Exception as exc:
                print(f"  SEED {i} FAILED: {exc}", flush=True)

        # Summary per branch
        summary = {}
        for _, label in BRANCHES:
            lens = []
            durs = []
            for s in seeds:
                b = s["branches"].get(label, {})
                if "last_text_length" in b:
                    lens.append(b["last_text_length"])
                if "duration_s" in b:
                    durs.append(b["duration_s"])
            summary[label] = {
                "text_length": stats(lens),
                "duration_s": stats(durs),
            }

        # Paired deltas: B-A and C-A per seed
        def paired_delta(label_x, label_y, field):
            ds = []
            for s in seeds:
                x = s["branches"].get(label_x, {}).get(field)
                y = s["branches"].get(label_y, {}).get(field)
                if x is not None and y is not None:
                    ds.append(x - y)
            return stats(ds) if ds else {"n": 0}

        pairs = {}
        for tgt_label in ("B_accept", "C_reject"):
            pairs[f"{tgt_label}_minus_A_control"] = {
                "text_length": paired_delta(tgt_label, "A_control", "last_text_length"),
                "duration_s": paired_delta(tgt_label, "A_control", "duration_s"),
            }

        out = {
            "n_seeds": len(seeds),
            "branches": [{"preset": p, "label": l} for p, l in BRANCHES],
            "baseline_prompt": BASELINE_PROMPT,
            "followup_prompt": FOLLOWUP_PROMPT,
            "seeds": seeds,
            "per_branch_summary": summary,
            "paired_deltas": pairs,
        }
        with open("/tmp/nature-eval/ablation_trajectory.json", "w") as f:
            json.dump(out, f, indent=2)

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for label, s in summary.items():
            print(f"{label}: textlen n={s['text_length']['n']}, "
                  f"mean={s['text_length'].get('mean', 0):.0f}, "
                  f"sd={s['text_length'].get('sd', 0):.0f}")
        for k, v in pairs.items():
            d = v["text_length"]
            print(f"{k} text_length: mean_delta={d.get('mean', 0):.1f}, "
                  f"±{d.get('ci_halfwidth_95', 0):.1f}")
        print(f"\nwrote /tmp/nature-eval/ablation_trajectory.json")


if __name__ == "__main__":
    asyncio.run(main())
