"""Multi-seed event-pinned ablations for paper §6.

Runs:
- §6.1 prompt ablation: baseline direct-core, fork to direct-core (A)
  vs direct-core-scout (B, researcher uses scout variant)
- §6.2 model-swap ablation: 3 seeds of direct-core baseline forked
  into direct-core (A) vs haiku-phi4 (B)

Writes all runs to /tmp/nature-eval/ablation_multi.json for analysis.
"""

import asyncio
import json
import sys
import time

from nature.client import NatureClient


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
    """Wait for run to become active, then idle with 5s stability."""
    start = time.time()
    phase1_deadline = start + 15
    became_active = False
    while time.time() < phase1_deadline:
        info = await client.get_session(session_id)
        if info.has_active_run:
            became_active = True
            break
        await asyncio.sleep(0.5)
    if not became_active:
        pass  # run may have finished synchronously
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


async def summarize(client, session_id):
    events = await client.snapshot(session_id)
    turns = sum(1 for e in events if e.type.value == "llm.response")
    tool_calls = sum(1 for e in events if e.type.value == "tool.started")
    messages = sum(1 for e in events if e.type.value == "message.appended")
    last_text = None
    for e in reversed(events):
        if e.type.value == "message.appended":
            try:
                content = e.payload.get("content") or []
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "text" and b.get("text"):
                        last_text = b.get("text", "")
                        break
                if last_text:
                    break
            except Exception:
                continue
    return {
        "session_id": session_id,
        "total_events": len(events),
        "turns": turns,
        "tool_calls": tool_calls,
        "messages": messages,
        "last_text": last_text,
        "last_text_length": len(last_text or ""),
    }


async def run_trial(client, trial_name, base_preset, a_preset, b_preset):
    """Run: baseline(base_preset) → fork → A(a_preset) vs B(b_preset).
    Returns {baseline, A, B} metrics."""
    print(f"\n=== {trial_name} ===")
    # Baseline
    s = await client.create_session(preset=base_preset)
    sid = s.session_id
    print(f"  baseline session={sid[:8]} preset={base_preset}")
    t0 = time.time()
    await client.send_message(sid, BASELINE_PROMPT)
    b_dur = await wait_idle(client, sid)
    print(f"  baseline done in {b_dur:.1f}s")
    b_sum = await summarize(client, sid)
    events = await client.snapshot(sid)
    fork_at = max(e.id or 0 for e in events)
    print(f"  fork point: event {fork_at}")

    # Branch A
    a_session = await client.fork_session(sid, at_event_id=fork_at, preset=a_preset)
    a_sid = a_session.session_id
    print(f"  A session={a_sid[:8]} preset={a_preset}")
    await client.send_message(a_sid, FOLLOWUP_PROMPT)
    a_dur = await wait_idle(client, a_sid)
    print(f"  A done in {a_dur:.1f}s")
    a_sum = await summarize(client, a_sid)

    # Branch B
    b_session = await client.fork_session(sid, at_event_id=fork_at, preset=b_preset)
    b_sid2 = b_session.session_id
    print(f"  B session={b_sid2[:8]} preset={b_preset}")
    await client.send_message(b_sid2, FOLLOWUP_PROMPT)
    b_dur2 = await wait_idle(client, b_sid2)
    print(f"  B done in {b_dur2:.1f}s")
    b_sum = await summarize(client, b_sid2)

    return {
        "trial_name": trial_name,
        "fork_event": fork_at,
        "baseline_preset": base_preset,
        "a_preset": a_preset,
        "b_preset": b_preset,
        "baseline": {**b_sum, "duration_s": b_dur},
        "A": {**a_sum, "duration_s": a_dur},
        "B": {**b_sum, "duration_s": b_dur2},
    }


async def main():
    async with NatureClient() as client:
        if not await client.is_alive():
            print("ERROR: server not running", file=sys.stderr)
            sys.exit(1)

        trials = []
        # §6.2 model-swap ablation, 3 seeds
        for seed in range(3):
            t = await run_trial(
                client,
                f"model-swap seed={seed}",
                base_preset="direct-core",
                a_preset="direct-core",
                b_preset="haiku-phi4",
            )
            trials.append(t)

        # §6.1 prompt ablation, 1 seed
        t = await run_trial(
            client,
            "prompt-ablation researcher-default-vs-scout",
            base_preset="direct-core",
            a_preset="direct-core",
            b_preset="direct-core-scout",
        )
        trials.append(t)

        out = {"trials": trials}
        with open("/tmp/nature-eval/ablation_multi.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote /tmp/nature-eval/ablation_multi.json ({len(trials)} trials)")


if __name__ == "__main__":
    asyncio.run(main())
