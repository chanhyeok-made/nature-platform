"""Event-log introspection — frame tree + per-turn slicing.

The eval dashboard needs a higher-level view of a cell's session than
a flat event stream. The conceptual layers are:

    Cell → Frame(s) → Turn(s) → phase {input, pre, call, post}

This module walks a session's jsonl once, groups events by frame,
then partitions each frame's events into turns keyed on the
`llm.request → llm.response` pair(s). Each turn is further split
into phases so the UI can render lanes:

    input  — `tool_result` / `user.input` / `hint.injected`
             that accumulated since the prior turn finished
    pre    — `body.compacted` (framework-side mutation right before
             the call)
    call   — `llm.request` / `llm.response` / `llm.error`. Retries
             (overloaded, rate-limit) land here as multiple attempts
             of the SAME turn, not separate turns.
    post   — `tool.started` / `tool.completed` the agent emitted,
             plus any `spawned_frame_id` pairing for Agent tool calls
             that created child frames.

Returns a JSON-serializable dict that the server route forwards
verbatim to the /eval dashboard.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Cap on how much of a tool_result's output we embed per event.
# Large outputs (huge Read results, long Bash stdout) would bloat the
# cell payload; the full text is still reconstructable from the raw
# jsonl via `Raw events` fallback.
_TOOL_OUTPUT_PREVIEW = 2000
_MESSAGE_TEXT_PREVIEW = 2000


def _event_summary(ev: dict, role: str) -> dict:
    """Lightweight projection used both for raw-timeline and as
    fallback data when a slot doesn't need the full turn structure."""
    t = ev.get("type")
    p = ev.get("payload") or {}
    base = {"id": ev.get("id"), "type": t, "role": role}
    if t == "frame.opened":
        base.update({
            "model": p.get("model"),
            "parent_id": p.get("parent_id"),
            "purpose": p.get("purpose"),
        })
    elif t == "frame.errored":
        base.update({
            "error_type": p.get("error_type"),
            "message": str(p.get("message", ""))[:500],
        })
    elif t == "llm.response":
        base.update({"stop_reason": p.get("stop_reason")})
    elif t == "llm.error":
        base.update({
            "error_type": p.get("error_type"),
            "message": str(p.get("message", ""))[:400],
        })
    elif t == "tool.started":
        base.update({
            "tool_name": p.get("tool_name"),
            "tool_input": p.get("tool_input") or {},
        })
    elif t == "tool.completed":
        base.update({
            "tool_name": p.get("tool_name"),
            "is_error": bool(p.get("is_error")),
            "output": str(p.get("output", ""))[:_TOOL_OUTPUT_PREVIEW],
        })
    elif t == "hint.injected":
        hints = p.get("hints") or []
        base.update({
            "hints": [
                {
                    "source": h.get("source", "?") if isinstance(h, dict) else "?",
                    "text": str(h.get("text", ""))[:400] if isinstance(h, dict) else "",
                }
                for h in hints
            ],
        })
    elif t == "body.compacted":
        base.update({
            "strategy": p.get("strategy"),
            "tokens_before": p.get("tokens_before"),
            "tokens_after": p.get("tokens_after"),
        })
    return base


def _extract_assistant_text(messages: list[dict], llm_response_event_id: int) -> str:
    """Find the `message.appended` that carries this response's
    assistant content and return its concatenated text blocks.

    Agent-loop convention: the assistant message gets appended
    immediately after the corresponding `llm.response`, always with
    `from_ == role_name` (i.e., the frame's self_actor). We pick the
    first such message following the response id.
    """
    for msg in messages:
        if msg.get("after_event_id", -1) <= llm_response_event_id:
            continue
        parts = []
        for block in msg.get("content", []):
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        joined = "".join(parts)[:_MESSAGE_TEXT_PREVIEW]
        return joined
    return ""


def _extract_input_messages(events: list[dict], start_idx: int, end_idx: int) -> list[dict]:
    """Summarize message-like events between start_idx (inclusive)
    and end_idx (exclusive), used to describe this turn's input."""
    items: list[dict] = []
    for ev in events[start_idx:end_idx]:
        t = ev.get("type")
        p = ev.get("payload") or {}
        if t == "user.input":
            items.append({
                "kind": "user_input",
                "event_id": ev.get("id"),
                "text": str(p.get("text", ""))[:_MESSAGE_TEXT_PREVIEW],
            })
        elif t == "message.appended":
            # Only surface tool_result envelopes here — the agent's
            # own messages are not "input" to the agent.
            if p.get("from_") == "tool":
                tool_results = []
                for block in p.get("content", []) or []:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        content = block.get("content", "")
                        if isinstance(content, list):
                            content = "".join(
                                c.get("text", "")
                                for c in content
                                if isinstance(c, dict) and c.get("type") == "text"
                            )
                        tool_results.append({
                            "tool_use_id": block.get("tool_use_id"),
                            "is_error": bool(block.get("is_error")),
                            "content": str(content)[:_TOOL_OUTPUT_PREVIEW],
                        })
                if tool_results:
                    items.append({
                        "kind": "tool_results",
                        "event_id": ev.get("id"),
                        "results": tool_results,
                    })
        elif t == "hint.injected":
            items.append({
                "kind": "hint",
                "event_id": ev.get("id"),
                "hints": [
                    {
                        "source": h.get("source", "?") if isinstance(h, dict) else "?",
                        "text": str(h.get("text", ""))[:400] if isinstance(h, dict) else "",
                    }
                    for h in (p.get("hints") or [])
                ],
            })
    return items


def _tool_call_post(
    events: list[dict],
    call_end_idx: int,
    turn_end_idx: int,
    frame_by_parent_tool: dict[str, str],
) -> list[dict]:
    """Summarize the agent's post-call actions — tool_uses emitted
    and their matching tool.completed events, plus child-frame spawn
    pointers when the tool was `Agent`."""
    tool_uses: dict[str, dict] = {}
    for ev in events[call_end_idx:turn_end_idx]:
        t = ev.get("type")
        p = ev.get("payload") or {}
        if t == "tool.started":
            tid = p.get("tool_use_id") or ""
            tool_uses[tid] = {
                "tool_use_id": tid,
                "event_id": ev.get("id"),
                "name": p.get("tool_name"),
                "input": p.get("tool_input") or {},
                "completed_event_id": None,
                "is_error": None,
                "output": "",
                "spawned_frame_id": frame_by_parent_tool.get(tid),
            }
        elif t == "tool.completed":
            tid = p.get("tool_use_id") or ""
            if tid in tool_uses:
                tool_uses[tid]["completed_event_id"] = ev.get("id")
                tool_uses[tid]["is_error"] = bool(p.get("is_error"))
                tool_uses[tid]["output"] = str(p.get("output", ""))[:_TOOL_OUTPUT_PREVIEW]
    return list(tool_uses.values())


def build_cell_inspection(log_path: Path) -> dict[str, Any]:
    """Walk a session jsonl once, produce a frame-tree + per-turn
    decomposition the eval UI can render without further processing.

    Returns:
        {
          "root_frame_id": "frame_xxx" | None,
          "frames": [
            {
              "frame_id": str,
              "role": str,
              "model": str,
              "parent_id": str | None,
              "parent_tool_use_id": str | None,
              "children_ids": [str, ...],
              "status": "resolved" | "errored" | "closed" | "active",
              "turns": [
                {
                  "turn_index": int,
                  "input":  [ <items> ],
                  "pre":    [ <summaries> ],
                  "call":   {
                    "model": str,
                    "attempts": [ { request_id, ok, stop_reason?, error_type?, message?, response_event_id? } ],
                    "usage": { ... },
                    "cost_usd": float,       # empty when unknown (local models)
                    "latency_ms": int,
                    "response_text": str,
                  },
                  "post":   {
                    "tool_uses": [ ... ],
                  },
                }
              ],
            }
          ],
        }
    """
    # Sentinel shape so callers can render "no data" uniformly even
    # when the log file is absent (eval cell failed before any events
    # wrote, for example).
    empty: dict[str, Any] = {"root_frame_id": None, "frames": []}
    if not log_path.exists():
        return empty

    try:
        raw = log_path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("cell log read failed: %s", exc)
        return empty

    events: list[dict] = []
    for line in raw.splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    # ── Pass 1: frame metadata + message annotations index ─────────
    frames: dict[str, dict] = {}
    # frame_id → role name, used when an event's frame_id isn't
    # FRAME_OPENED but we need the actor.
    frame_of_event: dict[int, str | None] = {}
    parent_tool_use_to_child: dict[str, str] = {}
    message_appended: list[dict] = []
    role_by_frame: dict[str, str] = {}

    for ev in events:
        fid = ev.get("frame_id")
        frame_of_event[ev.get("id") or -1] = fid
        t = ev.get("type")
        p = ev.get("payload") or {}
        if t == "frame.opened":
            spawn = p.get("spawned_by_tool_use_id")
            frames[fid] = {
                "frame_id": fid,
                "role": p.get("role_name") or "?",
                "model": p.get("model") or "",
                "parent_id": p.get("parent_id"),
                "parent_tool_use_id": spawn,
                "children_ids": [],
                "status": "active",
                "_opened_at": ev.get("id"),
            }
            role_by_frame[fid] = p.get("role_name") or "?"
            if spawn:
                parent_tool_use_to_child[spawn] = fid
            if p.get("parent_id") and p.get("parent_id") in frames:
                frames[p.get("parent_id")]["children_ids"].append(fid)
        elif t == "frame.resolved":
            if fid in frames:
                frames[fid]["status"] = "resolved"
        elif t == "frame.errored":
            if fid in frames:
                frames[fid]["status"] = "errored"
                frames[fid]["error"] = {
                    "type": p.get("error_type"),
                    "message": str(p.get("message", ""))[:500],
                }
        elif t == "frame.closed":
            if fid in frames and frames[fid]["status"] == "active":
                frames[fid]["status"] = "closed"
        elif t == "message.appended":
            message_appended.append({
                "after_event_id": ev.get("id"),
                "frame_id": fid,
                "from_": p.get("from_"),
                "content": p.get("content") or [],
            })

    # ── Pass 2: per-frame turn slicing ─────────────────────────────
    events_by_frame: dict[str, list[tuple[int, dict]]] = {fid: [] for fid in frames}
    for idx, ev in enumerate(events):
        fid = ev.get("frame_id")
        if fid in events_by_frame:
            events_by_frame[fid].append((idx, ev))

    # Pricing (same table the runner uses). Swallow import errors so
    # the inspection code stays independent.
    try:
        from nature.utils.cost import _get_pricing
    except Exception:  # noqa: BLE001
        def _get_pricing(model):  # type: ignore[misc]
            return {"input": 0, "output": 0, "cache_write": 0, "cache_read": 0}

    for fid, frame in frames.items():
        f_events = events_by_frame.get(fid, [])
        if not f_events:
            frame["turns"] = []
            continue

        # Frame-level mission: the first delegation / task seed that
        # lands before this frame's initial llm.request. For root
        # frames it's the user's prompt; for child frames it's the
        # parent's Agent delegation text. Always-visible in the UI so
        # a reader doesn't have to open "show full request" just to
        # remember what this agent is trying to do.
        mission_text: str | None = None
        mission_from: str | None = None
        mission_event_id: int | None = None
        for _, ev in f_events:
            if ev.get("type") == "llm.request":
                break
            if ev.get("type") != "message.appended":
                continue
            p = ev.get("payload") or {}
            from_ = p.get("from_") or ""
            if not from_ or from_ == "tool" or from_ == frame.get("role"):
                continue
            text_parts = [
                str(b.get("text", ""))
                for b in (p.get("content") or [])
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            if text_parts:
                mission_text = "".join(text_parts)[:_MESSAGE_TEXT_PREVIEW * 6]
                mission_from = from_
                mission_event_id = ev.get("id")
                break
        if mission_text:
            frame["mission"] = {
                "text": mission_text,
                "from_": mission_from,
                "event_id": mission_event_id,
            }

        # Partition points — every llm.request begins a new turn.
        # We then grow each turn forward until the next llm.request,
        # collecting input/pre/call/post along the way.
        turn_boundaries: list[int] = []
        for i, (_, ev) in enumerate(f_events):
            if ev.get("type") == "llm.request":
                turn_boundaries.append(i)

        turns: list[dict] = []
        for ti, start in enumerate(turn_boundaries):
            end = (
                turn_boundaries[ti + 1]
                if ti + 1 < len(turn_boundaries)
                else len(f_events)
            )
            # Classify events within [start, end).
            attempts: list[dict] = []
            current_attempt: dict | None = None
            pre_items: list[dict] = []
            call_usage: dict | None = None
            call_latency_ms: int | None = None
            call_cost_usd: float = 0.0
            response_event_id: int | None = None
            response_text = ""
            tool_uses_in_turn: dict[str, dict] = {}

            # Gather prior-boundary input (from previous turn's end
            # up to THIS turn's start). For turn 0, that's from the
            # beginning of the frame's events to `start`.
            prev_end = turn_boundaries[ti - 1] + 1 if ti > 0 else 0
            # Find where previous llm.response landed so input
            # begins AFTER it (not including the response itself).
            if ti > 0:
                for j in range(prev_end - 1, len(f_events)):
                    if j >= start:
                        break
                    if f_events[j][1].get("type") == "llm.response":
                        prev_end = j + 1
                        break
            # Everything in [prev_end, start) is the input window for
            # this turn. We render the subset that's meaningful.
            input_items: list[dict] = []
            for j in range(prev_end, start):
                ev = f_events[j][1]
                ev_t = ev.get("type")
                p = ev.get("payload") or {}
                if ev_t == "user.input":
                    input_items.append({
                        "kind": "user_input",
                        "event_id": ev.get("id"),
                        "text": str(p.get("text", ""))[:_MESSAGE_TEXT_PREVIEW],
                    })
                elif ev_t == "message.appended":
                    from_ = p.get("from_") or ""
                    blocks = p.get("content") or []
                    if from_ == "tool":
                        tool_results = []
                        for block in blocks:
                            if (
                                isinstance(block, dict)
                                and block.get("type") == "tool_result"
                            ):
                                content = block.get("content", "")
                                if isinstance(content, list):
                                    content = "".join(
                                        c.get("text", "")
                                        for c in content
                                        if isinstance(c, dict) and c.get("type") == "text"
                                    )
                                tool_results.append({
                                    "tool_use_id": block.get("tool_use_id"),
                                    "is_error": bool(block.get("is_error")),
                                    "content": str(content)[:_TOOL_OUTPUT_PREVIEW],
                                })
                        if tool_results:
                            input_items.append({
                                "kind": "tool_results",
                                "event_id": ev.get("id"),
                                "results": tool_results,
                            })
                    elif from_ and from_ != "user" and from_ != frame.get("role"):
                        # Non-tool, non-user, non-self message arriving
                        # between turns is bootstrap / delegation context
                        # seeded by a parent/sibling frame (e.g. the root
                        # task description injected by receptionist before
                        # core's first llm.request). Previously dropped
                        # silently — now surfaced so T#0's IN shows the
                        # actual mission instead of appearing empty.
                        text_parts: list[str] = []
                        for block in blocks:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text_parts.append(str(block.get("text", "")))
                        text = "".join(text_parts)
                        if text:
                            input_items.append({
                                "kind": "bootstrap",
                                "event_id": ev.get("id"),
                                "from_": from_,
                                "text": text[:_MESSAGE_TEXT_PREVIEW * 4],
                            })
                elif ev_t == "hint.injected":
                    input_items.append({
                        "kind": "hint",
                        "event_id": ev.get("id"),
                        "hints": [
                            {
                                "source": (
                                    h.get("source", "?")
                                    if isinstance(h, dict) else "?"
                                ),
                                "text": (
                                    str(h.get("text", ""))[:400]
                                    if isinstance(h, dict) else ""
                                ),
                            }
                            for h in (p.get("hints") or [])
                        ],
                    })
                elif ev_t == "body.compacted":
                    pre_items.append({
                        "kind": "body_compacted",
                        "event_id": ev.get("id"),
                        "strategy": p.get("strategy"),
                        "before": p.get("tokens_before"),
                        "after": p.get("tokens_after"),
                        "summary": p.get("summary"),
                    })

            # Walk the CALL range — may include retries (multiple
            # llm.request → llm.error/response pairs). The previous
            # retry lands AFTER an llm.error and BEFORE the next
            # llm.request. Since we sliced on llm.request boundaries,
            # retries stay within the same turn as long as there's
            # no intervening tool.started (which there shouldn't be
            # — retry loops live entirely in the provider layer).
            post_start_idx = start  # will advance to after the final response
            hit_response = False
            for j in range(start, end):
                ev = f_events[j][1]
                ev_t = ev.get("type")
                p = ev.get("payload") or {}
                if ev_t == "llm.request":
                    current_attempt = {
                        "request_id": p.get("request_id"),
                        "model": p.get("model"),
                    }
                elif ev_t == "llm.error":
                    if current_attempt is not None:
                        current_attempt.update({
                            "ok": False,
                            "error_type": p.get("error_type"),
                            "message": str(p.get("message", ""))[:300],
                        })
                        attempts.append(current_attempt)
                        current_attempt = None
                elif ev_t == "llm.response":
                    if current_attempt is not None:
                        current_attempt.update({
                            "ok": True,
                            "stop_reason": p.get("stop_reason"),
                            "response_event_id": ev.get("id"),
                        })
                        attempts.append(current_attempt)
                        current_attempt = None
                    call_usage = p.get("usage") or {}
                    response_event_id = ev.get("id")
                    hit_response = True
                    post_start_idx = j + 1
                elif ev_t == "annotation.stored":
                    if p.get("duration_ms") is not None:
                        call_latency_ms = int(p.get("duration_ms") or 0)
                elif ev_t == "tool.started" and hit_response:
                    tid = p.get("tool_use_id") or ""
                    tool_uses_in_turn[tid] = {
                        "tool_use_id": tid,
                        "event_id": ev.get("id"),
                        "name": p.get("tool_name"),
                        "input": p.get("tool_input") or {},
                        "completed_event_id": None,
                        "is_error": None,
                        "output": "",
                        "spawned_frame_id": parent_tool_use_to_child.get(tid),
                    }
                elif ev_t == "tool.completed" and hit_response:
                    tid = p.get("tool_use_id") or ""
                    if tid in tool_uses_in_turn:
                        tool_uses_in_turn[tid]["completed_event_id"] = ev.get("id")
                        tool_uses_in_turn[tid]["is_error"] = bool(p.get("is_error"))
                        tool_uses_in_turn[tid]["output"] = (
                            str(p.get("output", ""))[:_TOOL_OUTPUT_PREVIEW]
                        )

            # Compute cost for this call (final response's usage × model pricing)
            if call_usage and attempts:
                # Use the successful attempt's model for pricing; fall
                # back to the last attempt's if all failed.
                model_for_cost = (
                    next(
                        (a.get("model") for a in attempts if a.get("ok")),
                        attempts[-1].get("model") if attempts else "",
                    )
                    or ""
                )
                pricing = _get_pricing(model_for_cost)
                regular_in = int(call_usage.get("input_tokens", 0) or 0)
                out = int(call_usage.get("output_tokens", 0) or 0)
                cache_write = int(call_usage.get("cache_creation_input_tokens", 0) or 0)
                cache_read = int(call_usage.get("cache_read_input_tokens", 0) or 0)
                call_cost_usd = (
                    (regular_in / 1_000_000) * pricing["input"]
                    + (out / 1_000_000) * pricing["output"]
                    + (cache_write / 1_000_000) * pricing["cache_write"]
                    + (cache_read / 1_000_000) * pricing["cache_read"]
                )

            # Pick up the agent's text response from message_appended
            # (the first assistant message after the llm.response).
            if response_event_id is not None:
                for msg in message_appended:
                    if msg.get("frame_id") != fid:
                        continue
                    if msg.get("after_event_id", -1) <= response_event_id:
                        continue
                    if msg.get("from_") != role_by_frame.get(fid):
                        # Assistant messages carry from_ == self_actor.
                        # Anything else (e.g. tool-result envelopes)
                        # belongs to a later turn's input.
                        continue
                    parts = []
                    for block in msg.get("content", []):
                        if isinstance(block, dict) and block.get("type") == "text":
                            parts.append(str(block.get("text", "")))
                    response_text = "".join(parts)[:_MESSAGE_TEXT_PREVIEW]
                    break

            turns.append({
                "turn_index": ti,
                "input": input_items,
                "pre": pre_items,
                "call": {
                    "model": (
                        attempts[0].get("model") if attempts else frame.get("model")
                    ),
                    "attempts": attempts,
                    "usage": call_usage or {},
                    "cost_usd": round(call_cost_usd, 6),
                    "latency_ms": call_latency_ms,
                    "response_text": response_text,
                    "response_event_id": response_event_id,
                },
                "post": {
                    "tool_uses": list(tool_uses_in_turn.values()),
                },
            })

        # Also capture trailing input items that landed AFTER the
        # final llm.response but before the frame closed — these
        # belong to no turn's call but are the post of the final
        # turn. Rare but can include hint.injected between the last
        # response and frame.resolved.
        if turn_boundaries:
            final_response_idx = None
            last_turn_start = turn_boundaries[-1]
            for j in range(last_turn_start, len(f_events)):
                if f_events[j][1].get("type") == "llm.response":
                    final_response_idx = j
            if final_response_idx is not None and turns:
                # Any further tool calls / hints are already folded
                # into the last turn's post or trailing-input (the
                # else branch). Nothing extra to add here.
                pass

        # Strip internal scratch field
        frame.pop("_opened_at", None)
        frame["turns"] = turns

    root_frame_id = next(
        (f["frame_id"] for f in frames.values() if f.get("parent_id") is None),
        None,
    )

    return {
        "root_frame_id": root_frame_id,
        "frames": list(frames.values()),
    }


def build_turn_request(
    log_path: Path,
    frame_id: str,
    turn_index: int,
) -> dict[str, Any] | None:
    """Reconstruct the *full* LLM request that was sent for one turn.

    The `llm.request` event itself only records counts (to keep the
    log compact), so the request body has to be rebuilt from the
    preceding state-transition events:

    - `frame.opened`  + `header.snapshot` → role + principles (≈ system)
    - `principle.added` / `role.changed`  → header mutations between
                                            opens and this turn's call
    - `message.appended`                   → every user / assistant /
                                            tool message in order
    - `body.compacted`                     → body truncation events
                                            whose `new_messages` RESET
                                            the accumulated list (same
                                            rule `reconstruct.py` uses)

    Returns a dict or `None` if the turn cannot be located (frame or
    turn index out of range). The dict is JSON-serializable:

        {
          "frame_id": ...,
          "turn_index": ...,
          "model": ...,
          "request_event_id": int,
          "system": {
            "role_name": str,
            "role_description": str,
            "principles": [ { "source": str, "text": str }, ... ]
          },
          "messages": [
            {
              "index": int, "event_id": int,
              "from_": str,                # "user" / "tool" / role_name
              "to": str,
              "blocks": [ <raw content blocks> ],
              "blocks_summary": str        # short one-liner for quick scan
            },
            ...
          ],
          "compactions": [
            { "event_id": int, "strategy": str, "before": int, "after": int,
              "summary": str | None,
              "kept_messages": int }       # len(new_messages) after reset
          ]
        }
    """
    if not log_path.exists():
        return None
    try:
        raw = log_path.read_text(encoding="utf-8")
    except OSError:
        return None

    events: list[dict] = []
    for line in raw.splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    # Collect events for the requested frame, in order.
    frame_events: list[dict] = [e for e in events if e.get("frame_id") == frame_id]
    if not frame_events:
        return None

    # Locate the target turn's llm.request.
    requests = [e for e in frame_events if e.get("type") == "llm.request"]
    if turn_index < 0 or turn_index >= len(requests):
        return None
    target_request = requests[turn_index]
    target_id = target_request.get("id")

    # Pass over the frame's events up to (but not including) the
    # target llm.request, materializing:
    #   - role_name / role_description / principles
    #   - a sliding messages list (reset on body.compacted)
    #   - the list of compactions that happened pre-request
    #   - provenance state so each message can point back to its
    #     originating event (tool.started / llm.response / user.input /
    #     body.compacted / bootstrap delegation)
    role_name = "?"
    role_description = ""
    principles: list[dict] = []
    messages: list[dict] = []
    compactions: list[dict] = []

    # tool_use_id → { event_id, name, originating_frame_id }. Used
    # so a tool_result message can point back at the tool_use that
    # triggered it — either this frame's own Agent call or an
    # upstream event in a parent/sibling frame.
    tool_started_by_id: dict[str, dict] = {}
    last_llm_response_id: int | None = None
    last_user_input_id: int | None = None
    last_hint_injected_id: int | None = None

    def _principle_dict(pobj: Any) -> dict:
        if not isinstance(pobj, dict):
            return {"source": "?", "text": str(pobj)[:400]}
        return {
            "source": str(pobj.get("source", "?") or "?"),
            "text": str(pobj.get("text", "") or "")[:400],
        }

    def _summarize_blocks(blocks: list) -> str:
        parts: list[str] = []
        for b in blocks or []:
            if not isinstance(b, dict):
                continue
            bt = b.get("type")
            if bt == "text":
                txt = str(b.get("text", ""))
                parts.append(txt[:80].replace("\n", " "))
            elif bt == "tool_use":
                parts.append(f"→{b.get('name', '?')}")
            elif bt == "tool_result":
                out = b.get("content", "")
                if isinstance(out, list):
                    out = "".join(
                        c.get("text", "")
                        for c in out
                        if isinstance(c, dict) and c.get("type") == "text"
                    )
                parts.append(f"←tool_result {str(out)[:60]}".replace("\n", " "))
            elif bt == "image":
                parts.append("[img]")
        return " | ".join(parts)[:240]

    def _classify(
        ev: dict,
        payload: dict,
        role: str,
    ) -> dict:
        """Attach provenance to a message.appended event so the UI can
        explain where each piece came from. The `blocks` array is the
        authoritative content — we read its types to decide kind."""
        blocks = payload.get("content") or []
        from_ = payload.get("from_") or "?"
        has_tool_use = any(
            isinstance(b, dict) and b.get("type") == "tool_use" for b in blocks
        )
        tool_result_links: list[dict] = []
        for b in blocks:
            if isinstance(b, dict) and b.get("type") == "tool_result":
                tuid = b.get("tool_use_id") or ""
                started = tool_started_by_id.get(tuid)
                tool_result_links.append({
                    "tool_use_id": tuid,
                    "source_event_id": started.get("event_id") if started else None,
                    "tool_name": started.get("name") if started else None,
                    "is_error": bool(b.get("is_error")),
                })
        has_tool_result = bool(tool_result_links)

        kind: str
        detail: dict[str, Any] = {}
        if from_ == "tool" or has_tool_result:
            kind = "tool_result"
            detail["links"] = tool_result_links
        elif has_tool_use:
            kind = "assistant_tool_call"
            detail["llm_response_event_id"] = last_llm_response_id
        elif from_ == "user":
            kind = "user_input"
            detail["user_input_event_id"] = last_user_input_id
        elif from_ == role:
            # self-actor text — agent's reply right after llm.response
            kind = "assistant_text"
            detail["llm_response_event_id"] = last_llm_response_id
        else:
            # Anything else arriving with `from_ == <other role>` is
            # bootstrap context a parent/sibling frame injected. The
            # most common shape is receptionist → core when a root
            # Agent call spawns a worker frame with a seeded prompt.
            kind = "bootstrap"
            detail["bootstrap_from"] = from_
        if last_hint_injected_id is not None and kind == "assistant_text":
            # Footer hints are attached to the NEXT llm.request; the
            # assistant reply following that response consumed them.
            # Surface the hint event id so a reader can jump to what
            # was whispered in.
            detail["consumed_hint_event_id"] = last_hint_injected_id
        return {"kind": kind, **detail}

    # Track at parent level (visible inside _classify).
    for ev in frame_events:
        if ev.get("id") == target_id:
            break
        t = ev.get("type")
        p = ev.get("payload") or {}
        if t == "frame.opened":
            role_name = p.get("role_name") or role_name
        elif t == "header.snapshot":
            r = p.get("role") or {}
            if isinstance(r, dict):
                role_name = r.get("name") or role_name
                role_description = str(r.get("description") or "")
            principles = [_principle_dict(p2) for p2 in (p.get("principles") or [])]
        elif t == "role.changed":
            new_role = p.get("new_role") or {}
            if isinstance(new_role, dict):
                role_name = new_role.get("name") or role_name
                role_description = str(new_role.get("description") or role_description)
        elif t == "principle.added":
            principles.append(_principle_dict(p))
        elif t == "tool.started":
            tuid = p.get("tool_use_id") or ""
            if tuid:
                tool_started_by_id[tuid] = {
                    "event_id": ev.get("id"),
                    "name": p.get("tool_name"),
                }
        elif t == "llm.response":
            last_llm_response_id = ev.get("id")
            last_hint_injected_id = None  # consumed by the response
        elif t == "hint.injected":
            last_hint_injected_id = ev.get("id")
        elif t == "user.input":
            last_user_input_id = ev.get("id")
        elif t == "message.appended":
            blocks = p.get("content") or []
            messages.append({
                "index": len(messages),
                "event_id": ev.get("id"),
                "from_": p.get("from_") or "?",
                "to": p.get("to") or "",
                "blocks": blocks,
                "blocks_summary": _summarize_blocks(blocks),
                "provenance": _classify(ev, p, role_name),
            })
        elif t == "body.compacted":
            new_msgs = p.get("new_messages") or []
            # body.compacted is the canonical reset signal — the
            # server trims the in-memory message list down to exactly
            # `new_messages`, then subsequent message.appended events
            # stack on top of that.
            reset: list[dict] = []
            for i, m in enumerate(new_msgs):
                blocks = m.get("content") if isinstance(m, dict) else []
                reset.append({
                    "index": i,
                    "event_id": ev.get("id"),
                    "from_": m.get("from_") if isinstance(m, dict) else "?",
                    "to": m.get("to") if isinstance(m, dict) else "",
                    "blocks": blocks or [],
                    "blocks_summary": _summarize_blocks(blocks or []),
                    "synthetic_compaction": True,
                    "provenance": {
                        "kind": "compacted_synthetic",
                        "compaction_event_id": ev.get("id"),
                        "strategy": p.get("strategy"),
                    },
                })
            messages = reset
            compactions.append({
                "event_id": ev.get("id"),
                "strategy": p.get("strategy"),
                "before": p.get("tokens_before"),
                "after": p.get("tokens_after"),
                "summary": p.get("summary"),
                "kept_messages": len(new_msgs),
            })

    request_payload = target_request.get("payload") or {}

    # Rendered view: replay through `reconstruct()` and feed the
    # resulting Frame to `ContextComposer` to get the exact
    # LLMRequest that was sent over the wire — system sections
    # (header), messages (body), footer hints, tool definitions.
    rendered = _render_exact_request(
        log_path, frame_id, target_id, request_payload.get("model") or "",
    )

    return {
        "frame_id": frame_id,
        "turn_index": turn_index,
        "model": request_payload.get("model"),
        "request_event_id": target_id,
        "message_count_reported": request_payload.get("message_count"),
        "tool_count_reported": request_payload.get("tool_count"),
        "message_count_reconstructed": len(messages),
        # Raw event-replay view (each piece has provenance).
        "system": {
            "role_name": role_name,
            "role_description": role_description,
            "principles": principles,
        },
        "messages": messages,
        "compactions": compactions,
        # Authoritative view — exactly what went over the wire.
        "rendered": rendered,
    }


def _render_exact_request(
    log_path: Path,
    frame_id: str,
    target_request_id: int | None,
    request_model: str,
) -> dict[str, Any]:
    """Replay events → Frame → ContextComposer.compose() to capture
    the *exact* LLMRequest the provider received for this turn.

    Returns a JSON-serializable dict with:
      system_sections  — list[str], concatenated to form the `system`
                         parameter. First element is `role.instructions`
                         (HEADER); remaining are principles sorted by
                         priority descending.
      api_messages     — list of { role, content, tail_footer_note }
                         — the `messages` array. BODY comes first, then
                         footer hints (if any fired) are appended as a
                         final tail user message tagged [FRAMEWORK NOTE].
      footer_hints     — list of { source, text } so a reader can tell
                         which rule produced each framework note.
      tools            — list of { name, description, input_schema }
                         filtered by role.allowed_tools.
      error            — present only if reconstruction failed; holds
                         an explanation string.
    """
    try:
        from nature.events.reconstruct import reconstruct
        from nature.events.store import FileEventStore
        from nature.context.composer import ContextComposer
        from nature.tools.registry import get_default_tools
        from nature.frame.agent_tool import AgentTool
    except Exception as exc:  # noqa: BLE001
        return {"error": f"import failed: {exc}"}

    try:
        store = FileEventStore(log_path.parent)
        session_id = log_path.stem
        replay = reconstruct(
            session_id,
            store,
            up_to_event_id=(target_request_id - 1) if target_request_id else None,
        )
        frame = replay.frames.get(frame_id)
        if frame is None:
            return {"error": f"frame {frame_id} not found after replay"}
        # AreaManager registers AgentTool dynamically per frame when
        # `allowed_tools` includes "Agent" — get_default_tools() only
        # covers filesystem tools, so re-add Agent here so the rendered
        # `tools` count matches what actually shipped.
        tool_pool = get_default_tools() + [AgentTool()]
        composed = ContextComposer().compose(
            frame.context,
            self_actor=frame.self_actor,
            tool_registry=tool_pool,
            model=request_model or frame.model,
            source="eval_inspection",
        )
        req = composed.request

        api_messages: list[dict] = []
        footer_note_indices: set[int] = set()
        # Mark the last message as the footer note if hints fired —
        # composer appends exactly one tail user message with the
        # concatenated [FRAMEWORK NOTE] text.
        if composed.hints:
            footer_note_indices.add(len(req.messages) - 1)
        for i, m in enumerate(req.messages):
            api_messages.append({
                "index": i,
                "role": getattr(m.role, "value", str(m.role)),
                "content": _serialize_content(m.content),
                "is_footer_note": i in footer_note_indices,
            })

        tools = [
            {
                "name": t.name,
                "description": (t.description or "")[:1200],
                "input_schema": t.input_schema,
                "deferred": bool(t.deferred),
            }
            for t in req.tools
        ]

        hints = [
            {
                "source": h.source,
                "text": (h.text or "")[:2000],
            }
            for h in composed.hints
        ]

        # `api_payload` is the *wire-accurate* request body — we run
        # the composed LLMRequest through the exact same helpers the
        # AnthropicProvider uses (`_message_to_api`,
        # `_build_system_blocks`, `_tool_def_to_api`), so the
        # returned shape matches what `client.messages.stream(**kw)`
        # received, down to cache_control markers on tools / last
        # system block / last real message.
        api_payload = _build_wire_payload(req)

        return {
            "system_sections": list(req.system),
            "api_messages": api_messages,
            "footer_hints": hints,
            "tools": tools,
            "api_payload": api_payload,
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("rendered request reconstruction failed")
        return {"error": f"{type(exc).__name__}: {exc}"}


def _build_wire_payload(req: Any) -> dict[str, Any]:
    """Produce the *actual* Anthropic API request body for a composed
    LLMRequest — same transformations the provider performs in
    `AnthropicProvider.stream()` (system → content blocks with cache
    breakpoint, tools → defer_loading + cache on last, last real
    message → cache breakpoint).

    Caveats surfaced via the return value's `_notes` field so a
    reader can tell what's inferred vs read directly:
      - cache_control: defaults to ephemeral (matches the runner's
        default — if a session ran with caching disabled, the
        displayed markers are an over-approximation).
      - max_tokens: not stored in the event log; shown as null.
      - temperature / extra_body: likewise absent.
    """
    try:
        from nature.providers.anthropic import (
            _message_to_api,
            _build_system_blocks,
            _tool_def_to_api,
            _cache_control_dict,
            _mark_last_block_cacheable,
            _pick_cache_anchor_index,
        )
        from nature.protocols.provider import CacheControl
    except Exception as exc:  # noqa: BLE001
        return {
            "_notes": [f"provider import failed: {exc}"],
            "model": req.model,
            "system": list(req.system),
            "messages": [],
            "tools": [],
        }

    cache_control = CacheControl(type="ephemeral")

    api_messages = [_message_to_api(m) for m in req.messages]
    system_blocks = _build_system_blocks(list(req.system), cache_control)

    kwargs: dict[str, Any] = {
        "model": req.model,
        "messages": api_messages,
        "max_tokens": None,  # provider-config-dependent; not in events
    }
    if system_blocks:
        kwargs["system"] = system_blocks

    tools_out: list[dict] = []
    if req.tools:
        tools_out = [_tool_def_to_api(t) for t in req.tools]
        if tools_out:
            tools_out[-1]["cache_control"] = _cache_control_dict(cache_control)
        kwargs["tools"] = tools_out

    if api_messages:
        anchor = _pick_cache_anchor_index(api_messages)
        if anchor is not None:
            _mark_last_block_cacheable(
                api_messages[anchor].get("content", []),
                cache_control,
            )

    kwargs["_notes"] = [
        "cache_control markers shown assume ephemeral caching "
        "(runner default). A session that ran with caching "
        "disabled would send this body without them.",
        "max_tokens / temperature / extra_body come from the "
        "provider config at run time and are not in the event log.",
    ]
    return kwargs


def _serialize_content(content: list) -> list[dict]:
    """Convert LLMMessage content blocks to plain dicts.

    `content` is a list of protocol message content blocks (TextContent,
    ToolUseContent, ToolResultContent, ImageContent). Each has a
    Pydantic `model_dump()` since they're BaseModels. If a block is
    already a dict (rare), pass through.
    """
    result: list[dict] = []
    for block in content or []:
        if isinstance(block, dict):
            result.append(block)
        elif hasattr(block, "model_dump"):
            result.append(block.model_dump())
        else:
            # Last-resort string coercion so we never crash the whole
            # reconstruction on one oddball block.
            result.append({"type": "unknown", "repr": str(block)[:500]})
    return result


__all__ = ["build_cell_inspection", "build_turn_request"]
