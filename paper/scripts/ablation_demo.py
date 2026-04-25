"""Event-pinned ablation demo for paper §6.

Runs one baseline code-analysis prompt under `direct-core`, forks at a
chosen event id into two branches (direct-core vs haiku-phi4), sends
the same follow-up prompt to each, and compares post-fork metrics.

This is the minimal worked example the paper needs: same prefix,
different post-fork preset, clean metric attribution.
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

# Fork at the end of the baseline run. Picking a mid-frame event
# orphans a `tool_use` and triggers an Anthropic 400 when the forked
# branch's next LLM call tries to include unbalanced history.
# FORK_AT_EVENT is resolved dynamically to the baseline's last event.
FORK_AT_EVENT = None


async def run_branch(client, preset, label, baseline_session_id=None, at_event=None):
    """Either start a baseline (baseline_session_id=None) or fork one."""
    if baseline_session_id is None:
        print(f"[{label}] creating baseline session (preset={preset})")
        s = await client.create_session(preset=preset)
    else:
        print(f"[{label}] forking {baseline_session_id[:8]}@{at_event} (preset={preset})")
        s = await client.fork_session(
            baseline_session_id, at_event_id=at_event, preset=preset,
        )
    sid = s.session_id
    print(f"[{label}] session={sid[:8]}")
    return sid


async def wait_idle(client, session_id, timeout=180):
    """Wait for: (1) run_task to become active, (2) then become idle
    with a 5s stability window. Guards against the 'send_message
    returned before run_task existed' race."""
    start = time.time()
    # Phase 1: wait for run to become active (up to 15s)
    phase1_deadline = start + 15
    became_active = False
    while time.time() < phase1_deadline:
        info = await client.get_session(session_id)
        if info.has_active_run:
            became_active = True
            break
        await asyncio.sleep(0.5)
    if not became_active:
        # Run may have completed synchronously very fast — fall through
        print(f"  (warn) session {session_id[:8]} never showed active run")
    # Phase 2: wait for run to go idle, then stable 5s
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


async def summarize(client, session_id, label):
    events = await client.snapshot(session_id)
    turns = sum(1 for e in events if e.type.value == "llm.response")
    tool_calls = sum(1 for e in events if e.type.value == "tool.started")
    messages = sum(1 for e in events if e.type.value == "message.appended")
    # last assistant message text
    last_text = None
    for e in reversed(events):
        if e.type.value == "message.appended":
            try:
                content = e.payload.get("content", [])
                for b in content:
                    if b.get("type") == "text":
                        last_text = b.get("text", "")
                        break
                if last_text: break
            except Exception:
                continue
    print(f"[{label}] total_events={len(events)} turns={turns} tool_calls={tool_calls} messages={messages}")
    print(f"[{label}] last_assistant_text[:300]={(last_text or '')[:300]!r}")
    return {
        "session_id": session_id,
        "total_events": len(events),
        "turns": turns,
        "tool_calls": tool_calls,
        "messages": messages,
        "last_text": last_text,
    }


async def main():
    async with NatureClient() as client:
        if not await client.is_alive():
            print("ERROR: server not running", file=sys.stderr)
            sys.exit(1)

        # ─── Baseline ────────────────────────────────────────────────
        baseline_sid = await run_branch(client, "direct-core", "baseline")
        t0 = time.time()
        await client.send_message(baseline_sid, BASELINE_PROMPT)
        b_wait = await wait_idle(client, baseline_sid)
        print(f"[baseline] baseline turn complete in {b_wait:.1f}s")
        baseline_result = await summarize(client, baseline_sid, "baseline")

        # Resolve fork event to the baseline's last event id (post-run)
        baseline_events = await client.snapshot(baseline_sid)
        fork_at = max(e.id or 0 for e in baseline_events)
        print(f"[fork] forking at event {fork_at} (end of baseline)")

        # ─── Branch A: continue under direct-core (control) ─────────
        a_sid = await run_branch(
            client, "direct-core", "A-direct-core",
            baseline_session_id=baseline_sid, at_event=fork_at,
        )
        await client.send_message(a_sid, FOLLOWUP_PROMPT)
        a_wait = await wait_idle(client, a_sid)
        print(f"[A] follow-up complete in {a_wait:.1f}s")
        a_result = await summarize(client, a_sid, "A-direct-core")

        # ─── Branch B: continue under haiku-phi4 (treatment) ────────
        b_sid = await run_branch(
            client, "haiku-phi4", "B-haiku-phi4",
            baseline_session_id=baseline_sid, at_event=fork_at,
        )
        await client.send_message(b_sid, FOLLOWUP_PROMPT)
        b_wait2 = await wait_idle(client, b_sid)
        print(f"[B] follow-up complete in {b_wait2:.1f}s")
        b_result = await summarize(client, b_sid, "B-haiku-phi4")

        out = {
            "fork_event": FORK_AT_EVENT,
            "baseline_prompt": BASELINE_PROMPT,
            "followup_prompt": FOLLOWUP_PROMPT,
            "baseline": {**baseline_result, "duration_s": b_wait},
            "branch_A_direct_core": {**a_result, "duration_s": a_wait},
            "branch_B_haiku_phi4": {**b_result, "duration_s": b_wait2},
        }
        with open("/tmp/nature-eval/ablation_result.json", "w") as f:
            json.dump(out, f, indent=2)
        print("wrote /tmp/nature-eval/ablation_result.json")


if __name__ == "__main__":
    asyncio.run(main())
