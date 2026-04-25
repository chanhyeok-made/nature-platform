"""Fork paired-sample vs unpaired CI experiment for §7.

Hypothesis: event-pinned forks let the same specific baseline be
continued under both preset A and preset B, producing a paired
delta per baseline. Analyzed as paired samples, the standard error
of the delta is smaller than the naive unpaired analysis that
treats the 10 A-runs and 10 B-runs as independent samples.

Design:
  - Run 10 independent baseline sessions under `direct-core`
  - For each baseline, fork twice at the last event:
      branch A → continue under `direct-core`  (control)
      branch B → continue under `haiku-phi4`   (treatment)
  - Measure response length on the follow-up prompt
  - Compare:
      paired CI:    CI of (len_B_i − len_A_i) across i=1..10
      unpaired CI:  CI of mean(len_B) − mean(len_A)
  - Expect paired SE < unpaired SE because the baseline-specific
    component of variance cancels in the paired delta.

Writes /tmp/nature-eval/fork_ci_experiment.json.
"""

import asyncio
import json
import math
import sys
import time

from nature.client import NatureClient


N_BASELINES = 10

BASELINE_PROMPT = (
    "Read `nature/frame/manager.py`. In 3 bullet points, list the key "
    "responsibilities of the AreaManager class. Keep each bullet under "
    "20 words. Do not delegate — answer directly."
)

FOLLOWUP_PROMPT = (
    "Given your analysis above, what is ONE concrete risk in the current "
    "design, and how would you mitigate it? Answer in 2-3 sentences."
)


async def wait_idle(client, session_id, timeout=180):
    start = time.time()
    phase1_deadline = start + 15
    while time.time() < phase1_deadline:
        info = await client.get_session(session_id)
        if info.has_active_run:
            break
        await asyncio.sleep(0.5)
    stable_since = None
    while time.time() - start < timeout:
        info = await client.get_session(session_id)
        if info.has_active_run:
            stable_since = None
            await asyncio.sleep(2)
            continue
        if stable_since is None:
            stable_since = time.time()
        elif time.time() - stable_since >= 5:
            return time.time() - start
        await asyncio.sleep(1)
    raise TimeoutError(f"session {session_id} still active after {timeout}s")


async def last_assistant_text(client, session_id):
    events = await client.snapshot(session_id)
    for e in reversed(events):
        if e.type.value == "message.appended":
            content = e.payload.get("content") or []
            for b in content:
                if isinstance(b, dict) and b.get("type") == "text" and b.get("text"):
                    return b.get("text", "")
    return ""


async def run_baseline(client, preset="direct-core"):
    s = await client.create_session(preset=preset)
    sid = s.session_id
    await client.send_message(sid, BASELINE_PROMPT)
    dur = await wait_idle(client, sid)
    events = await client.snapshot(sid)
    last_event = max(e.id or 0 for e in events)
    return sid, last_event, dur


async def continue_fork(client, baseline_sid, fork_event, preset):
    s = await client.fork_session(
        baseline_sid, at_event_id=fork_event, preset=preset,
    )
    sid = s.session_id
    await client.send_message(sid, FOLLOWUP_PROMPT)
    dur = await wait_idle(client, sid)
    text = await last_assistant_text(client, sid)
    return sid, dur, text, len(text or "")


def mean(xs):
    return sum(xs) / len(xs)


def std(xs):
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1)) ** 0.5


def ci_half_width(xs, z=1.96):
    """Half-width of the 95% CI for the mean of xs."""
    return z * std(xs) / math.sqrt(len(xs))


async def main():
    async with NatureClient() as client:
        if not await client.is_alive():
            print("ERROR: server not running", file=sys.stderr)
            sys.exit(1)

        trials = []
        for i in range(N_BASELINES):
            print(f"\n=== baseline {i+1}/{N_BASELINES} ===", flush=True)
            try:
                b_sid, fork_event, b_dur = await run_baseline(client)
                print(f"  baseline {b_sid[:8]} done in {b_dur:.1f}s "
                      f"(fork at event {fork_event})")
                a_sid, a_dur, a_text, a_len = await continue_fork(
                    client, b_sid, fork_event, "direct-core",
                )
                print(f"  A (direct-core) {a_sid[:8]}: {a_dur:.1f}s, {a_len}ch")
                bb_sid, b2_dur, b_text, b_len = await continue_fork(
                    client, b_sid, fork_event, "haiku-phi4",
                )
                print(f"  B (haiku-phi4)  {bb_sid[:8]}: {b2_dur:.1f}s, {b_len}ch")
                trials.append({
                    "index": i,
                    "baseline_sid": b_sid,
                    "baseline_duration": b_dur,
                    "fork_event": fork_event,
                    "A_duration": a_dur,
                    "A_text_length": a_len,
                    "B_duration": b2_dur,
                    "B_text_length": b_len,
                    "delta_length_B_minus_A": b_len - a_len,
                })
            except Exception as exc:
                print(f"  TRIAL {i} FAILED: {exc}")

        # Analyze
        A_lens = [t["A_text_length"] for t in trials]
        B_lens = [t["B_text_length"] for t in trials]
        deltas = [t["delta_length_B_minus_A"] for t in trials]

        a_mean, b_mean = mean(A_lens), mean(B_lens)
        a_sd, b_sd = std(A_lens), std(B_lens)

        # Unpaired SE of the mean difference
        n_a, n_b = len(A_lens), len(B_lens)
        unpaired_se = math.sqrt(a_sd**2 / n_a + b_sd**2 / n_b)
        unpaired_halfwidth = 1.96 * unpaired_se

        # Paired SE of the mean difference
        paired_halfwidth = ci_half_width(deltas)

        summary = {
            "n_trials": len(trials),
            "A_text_length_mean": a_mean,
            "A_text_length_sd": a_sd,
            "B_text_length_mean": b_mean,
            "B_text_length_sd": b_sd,
            "mean_delta": mean(deltas),
            "delta_sd": std(deltas),
            "unpaired_95ci_halfwidth": unpaired_halfwidth,
            "paired_95ci_halfwidth": paired_halfwidth,
            "se_reduction_ratio": (
                unpaired_halfwidth / paired_halfwidth
                if paired_halfwidth > 0 else None
            ),
        }

        out = {"trials": trials, "summary": summary}
        with open("/tmp/nature-eval/fork_ci_experiment.json", "w") as f:
            json.dump(out, f, indent=2)

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print(f"\nwrote /tmp/nature-eval/fork_ci_experiment.json")


if __name__ == "__main__":
    asyncio.run(main())
