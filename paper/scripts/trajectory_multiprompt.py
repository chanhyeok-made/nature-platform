"""Plan v2 Block B — multi-prompt trajectory sweep.

3 prompt pairs × 15 seeds × 4 branches (baseline + A_control + B_accept + C_reject).
Writes /tmp/nature-eval/trajectory_multiprompt.json.

Goal: establish paired/unpaired Jaccard ratio pattern across 3 different
analytical prompt types, to support the H1-H3 pre-registered hypotheses
for paper §4.2 and §7.

Each prompt pair asks the model to (a) read/describe something and
(b) answer a concrete derived question. Baselines use direct-core; each
seed forks into:
  A_control  = direct-core       (same preset)
  B_accept   = haiku-phi4        (analyzer/reviewer/judge → phi4, light mutation)
  C_reject   = direct-haiku-phi4 (no receptionist + phi4 subagents, heavy mutation)
"""

import asyncio
import json
import math
import sys
import time

from nature.client import NatureClient


N_SEEDS = 15

PROMPTS = [
    {
        "id": "P1_risk_analysis",
        "baseline": (
            "Read `nature/frame/manager.py`. In 3 bullet points, list the "
            "key responsibilities of the AreaManager class. Keep each bullet "
            "under 20 words. Do not delegate — answer directly."
        ),
        "followup": (
            "Given your analysis above, what is ONE concrete risk in the "
            "current design, and how would you mitigate it? Answer in 2-3 "
            "sentences."
        ),
    },
    {
        "id": "P2_diff_comparison",
        "baseline": (
            "Read `nature/agents/builtin/presets/default.json` and "
            "`nature/agents/builtin/presets/direct-core.json`. In 3 bullet "
            "points, describe the functional differences between them. Keep "
            "each bullet under 20 words. Do not delegate — answer directly."
        ),
        "followup": (
            "Given the differences above, what is ONE concrete consequence "
            "for runtime behavior when comparing these two presets? Answer "
            "in 2-3 sentences."
        ),
    },
    {
        "id": "P3_code_summary",
        "baseline": (
            "Read `nature/client/http_client.py`. In 3 bullet points, "
            "summarize what this module does. Keep each bullet under 20 "
            "words. Do not delegate — answer directly."
        ),
        "followup": (
            "Given your summary above, what is ONE potential failure mode "
            "that could occur when this client is used in production? "
            "Answer in 2-3 sentences."
        ),
    },
]

BRANCHES = [
    ("direct-core",       "A_control"),
    ("haiku-phi4",        "B_accept"),
    ("direct-haiku-phi4", "C_reject"),
]


async def wait_idle(client, sid, timeout=180):
    start = time.time()
    phase1 = start + 15
    while time.time() < phase1:
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
        "last_text": last_text,
        "last_text_length": len(last_text),
    }


async def run_trial(client, prompt, seed):
    s = await client.create_session(preset="direct-core")
    sid = s.session_id
    await client.send_message(sid, prompt["baseline"])
    b_dur = await wait_idle(client, sid)
    b_sum = await summarize(client, sid)
    events = await client.snapshot(sid)
    fork_at = max(e.id or 0 for e in events)
    print(f"    [{prompt['id']} s{seed}] baseline {sid[:8]} {b_dur:.0f}s events={b_sum['total_events']}")

    branches = {}
    for preset, label in BRANCHES:
        try:
            f = await client.fork_session(sid, at_event_id=fork_at, preset=preset)
            fsid = f.session_id
            await client.send_message(fsid, prompt["followup"])
            dur = await wait_idle(client, fsid)
            summ = await summarize(client, fsid)
            branches[label] = {"preset": preset, "duration_s": dur, **summ}
            print(f"    [{prompt['id']} s{seed}] {label:9} {fsid[:8]} {dur:.0f}s len={summ['last_text_length']}")
        except Exception as exc:
            print(f"    [{prompt['id']} s{seed}] {label} FAILED: {exc}")
            branches[label] = {"preset": preset, "error": str(exc)}

    return {
        "prompt_id": prompt["id"],
        "seed": seed,
        "baseline_sid": sid,
        "baseline_duration_s": b_dur,
        "baseline_summary": b_sum,
        "fork_event": fork_at,
        "branches": branches,
    }


async def main():
    async with NatureClient() as client:
        if not await client.is_alive():
            print("ERROR: server not running", file=sys.stderr); sys.exit(1)

        trials = []
        total_trials = N_SEEDS * len(PROMPTS)
        done = 0
        t0 = time.time()
        for prompt in PROMPTS:
            print(f"\n=== Prompt {prompt['id']} ===", flush=True)
            for seed in range(N_SEEDS):
                try:
                    r = await run_trial(client, prompt, seed)
                    trials.append(r)
                    done += 1
                    elapsed = time.time() - t0
                    eta = elapsed / done * (total_trials - done)
                    print(f"  [{done}/{total_trials}] elapsed {elapsed/60:.1f}m ETA {eta/60:.1f}m", flush=True)
                except Exception as exc:
                    print(f"  TRIAL failed: {exc}")

        # Minimal summary
        out = {
            "n_seeds": N_SEEDS,
            "prompts": PROMPTS,
            "branches": [{"preset": p, "label": l} for p, l in BRANCHES],
            "trials": trials,
        }
        with open("/tmp/nature-eval/trajectory_multiprompt.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote /tmp/nature-eval/trajectory_multiprompt.json ({len(trials)} trials)")


if __name__ == "__main__":
    asyncio.run(main())
