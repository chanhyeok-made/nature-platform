from __future__ import annotations

from collections import defaultdict
from typing import Any

from nature.events.types import Event, EventType
from nature.protocols.todo import TodoItem
from nature.server.view.messages import AnnotationDto, HintDto, MessageDto
from nature.server.view.session import PulseDto, SessionViewDto
from nature.server.view.steps import StepDto, SubAgentDto
from nature.server.view.tools import ReceivedDto, ToolDto, ToolResultBlockDto
from nature.server.view.turns import TurnDto, TurnSummaryDto


def _extract_text(content_blocks: list[dict[str, Any]]) -> str:
    """Pull plain text out of a content-blocks list."""
    parts = []
    for b in content_blocks or []:
        if isinstance(b, dict) and b.get("type") == "text" and b.get("text"):
            parts.append(b["text"])
    return "\n\n".join(parts).strip()


def _message_dto_from_event(ev: Event) -> MessageDto:
    p = ev.payload or {}
    return MessageDto(
        message_id=p.get("message_id", f"msg_{ev.id}"),
        from_=p.get("from_", ""),
        to=p.get("to", ""),
        text=_extract_text(p.get("content", [])),
        content=list(p.get("content", [])),
        timestamp=p.get("timestamp", ev.timestamp),
    )


def _latest_todos_from_events(events: list[Event]) -> list[TodoItem]:
    """Walk a frame's events and return the current todo list as of the
    latest TODO_WRITTEN event. Full-list-overwrite semantics means we
    only care about the last one; everything before it is superseded.
    Returns [] when no TodoWrite has ever run on this frame.
    """
    latest_payload: list[dict[str, Any]] | None = None
    for ev in events:
        if ev.type == EventType.TODO_WRITTEN:
            raw = (ev.payload or {}).get("todos") or []
            if isinstance(raw, list):
                latest_payload = raw
    if latest_payload is None:
        return []
    out: list[TodoItem] = []
    for raw_item in latest_payload:
        if not isinstance(raw_item, dict):
            continue
        try:
            out.append(TodoItem.model_validate(raw_item))
        except Exception:
            continue
    return out


def _frame_state_from_events(events: list[Event]) -> tuple[str, float | None]:
    """Return (state, ended_at) for a frame given its events."""
    state = "active"
    ended_at: float | None = None
    for ev in events:
        if ev.type == EventType.FRAME_RESOLVED:
            state = "resolved"
            ended_at = ev.timestamp
        elif ev.type == EventType.FRAME_CLOSED:
            state = "closed"
            ended_at = ev.timestamp
        elif ev.type == EventType.FRAME_ERRORED:
            state = "error"
            ended_at = ev.timestamp
        elif ev.type == EventType.FRAME_REOPENED:
            state = "active"
            ended_at = None
    return state, ended_at


def _compute_turn_summary(turn: TurnDto) -> TurnSummaryDto:
    tool_count = 0
    sub_count = 0
    received_count = 0
    for step in turn.steps:
        if step.kind == "tool":
            tool_count += 1
        elif step.kind == "sub_agent":
            sub_count += 1
        elif step.kind == "received":
            received_count += 1
    duration_ms: int | None = None
    if turn.ended_at is not None:
        duration_ms = int((turn.ended_at - turn.started_at) * 1000)
    return TurnSummaryDto(
        step_count=len(turn.steps),
        tool_count=tool_count,
        sub_agent_count=sub_count,
        received_count=received_count,
        duration_ms=duration_ms,
    )


def _aggregate_sub_agent_summary(turns: list[TurnDto]) -> TurnSummaryDto:
    """Roll up turn summaries across all of a sub-agent's turns."""
    s = TurnSummaryDto()
    total_ms = 0
    any_duration = False
    for t in turns:
        s.step_count += t.summary.step_count
        s.tool_count += t.summary.tool_count
        s.sub_agent_count += t.summary.sub_agent_count
        s.received_count += t.summary.received_count
        if t.summary.duration_ms is not None:
            total_ms += t.summary.duration_ms
            any_duration = True
    if any_duration:
        s.duration_ms = total_ms
    return s


def _derive_pulse(events: list[Event], root_frame_id: str | None) -> PulseDto:
    """Walk the tail of the event log (from the last user message on the
    root frame forward) and count in-flight llm/tool work."""

    if root_frame_id is None:
        return PulseDto(active=False)

    # Find the index of the last user message on the root frame
    last_user_idx = -1
    for i, ev in enumerate(events):
        if (
            ev.type == EventType.MESSAGE_APPENDED
            and ev.frame_id == root_frame_id
            and (ev.payload or {}).get("from_") == "user"
        ):
            last_user_idx = i
    if last_user_idx < 0:
        return PulseDto(active=False)

    tail = events[last_user_idx:]
    pending_llm = 0
    pending_tools = 0
    activity: str | None = None
    input_tokens = 0
    output_tokens = 0
    started_at = tail[0].timestamp

    for ev in tail:
        t = ev.type
        p = ev.payload or {}
        if t == EventType.LLM_REQUEST:
            pending_llm += 1
            if pending_tools == 0:
                activity = "thinking"
        elif t == EventType.LLM_RESPONSE or t == EventType.LLM_ERROR:
            pending_llm = max(0, pending_llm - 1)
        elif t == EventType.TOOL_STARTED:
            pending_tools += 1
            activity = "running " + (p.get("tool_name") or "tool")
        elif t == EventType.TOOL_COMPLETED:
            pending_tools = max(0, pending_tools - 1)
            if pending_tools == 0 and pending_llm > 0:
                activity = "thinking"
        elif t == EventType.ANNOTATION_STORED:
            u = (p.get("usage") or {})
            input_tokens += u.get("input_tokens", 0) or 0
            output_tokens += u.get("output_tokens", 0) or 0
        elif t == EventType.FRAME_ERRORED and ev.frame_id == root_frame_id:
            pending_llm = 0
            pending_tools = 0

    active = pending_llm > 0 or pending_tools > 0
    return PulseDto(
        active=active,
        activity=activity if active else None,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        started_at=started_at if active else None,
    )


def _build_turns_for_frame(
    frame_id: str,
    events_by_frame: dict[str, list[Event]],
    frame_meta: dict[str, dict[str, Any]],
    spawned_children: dict[str, str],
) -> list[TurnDto]:
    """Walk one frame's events and produce its list of TurnDto.

    Sub-frame steps are created by recursion — each child frame's own
    events live under `events_by_frame[child_id]` and are transformed
    into a `SubAgentDto` that wraps the child's turns.

    Turn-boundary rule: any "incoming" message to this frame's actor
    starts a new turn. On a root frame that's the user; on a child
    frame that's the parent agent's actor (delegation prompt). Tool
    result messages (from_="tool") are NOT boundaries — they're
    intermediate plumbing already represented by the tool steps.
    """

    events = events_by_frame.get(frame_id, [])
    self_role = frame_meta.get(frame_id, {}).get("role_name", "")
    turns: list[TurnDto] = []
    current: TurnDto | None = None

    # Scratch state for attaching tool completions + annotations back to
    # the objects we've already added to the current turn.
    tool_by_use_id: dict[str, ToolDto] = {}
    msg_by_id: dict[str, MessageDto] = {}
    # The most recent received envelope's tool_use_ids that haven't been
    # claimed by a downstream self-actor message yet. The next assistant
    # message gets these attached as `regenerated_from`, then the
    # pointer clears so the *next* received envelope is associated with
    # the *next* assistant message after it.
    pending_received_ids: list[str] = []
    # Footer hints attached to the most recent LLM_REQUEST that haven't
    # been claimed by the assistant message responding to that request.
    # Same pattern as pending_received_ids — claimed and cleared by the
    # next self-actor MESSAGE_APPENDED.
    pending_hints: list[HintDto] = []

    for ev in events:
        t = ev.type
        p = ev.payload or {}

        # Bump the current turn's last_event_id for EVERY event we see
        # while that turn is open, so the dashboard's fork button can
        # target the freshest session-monotonic event id on each turn.
        # The boundary-opening branch below resets `current` to a new
        # turn anchored at `ev.id`, so even the opening user message
        # lands with first_event_id == last_event_id == ev.id.
        if current is not None and ev.id > current.last_event_id:
            current.last_event_id = ev.id

        if t == EventType.MESSAGE_APPENDED:
            from_ = p.get("from_", "")

            # Incoming message (from someone other than this frame's
            # own actor or the tool bus) → new turn boundary.
            is_incoming = from_ != "tool" and from_ != self_role
            if is_incoming:
                if current is not None:
                    turns.append(current)
                msg = _message_dto_from_event(ev)
                current = TurnDto(
                    id=f"turn_{ev.id}",
                    state="running",
                    started_at=ev.timestamp,
                    user_message=msg,
                    first_event_id=ev.id,
                    last_event_id=ev.id,
                )
                msg_by_id[msg.message_id] = msg
                # Reset per-turn scratch (tool/message lookups are
                # scoped to the current turn)
                tool_by_use_id = {}
                pending_received_ids = []
                continue

            if current is None:
                # Orphan message before any incoming — rare, skip safely
                continue

            # Tool result message (from_="tool") → the bundled
            # tool_result envelope delivered back to this frame's
            # agent after its tool_use calls finished. Render as a
            # "received" step so the aggregation point is visible in
            # the timeline, not hidden.
            if from_ == "tool":
                results: list[ToolResultBlockDto] = []
                for block in p.get("content", []) or []:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") != "tool_result":
                        continue
                    raw_content_field = block.get("content", "")
                    if isinstance(raw_content_field, str):
                        text = raw_content_field
                    else:
                        # Structured list — concatenate text blocks
                        text_parts = []
                        for sub in (raw_content_field or []):
                            if isinstance(sub, dict) and sub.get("type") == "text":
                                text_parts.append(sub.get("text", ""))
                        text = "\n".join(text_parts)
                    results.append(ToolResultBlockDto(
                        tool_use_id=block.get("tool_use_id", "") or "",
                        content=text,
                        is_error=bool(block.get("is_error", False)),
                        raw_content=raw_content_field,
                    ))
                received = ReceivedDto(
                    message_id=p.get("message_id", f"msg_{ev.id}"),
                    timestamp=p.get("timestamp", ev.timestamp),
                    results=results,
                )
                current.steps.append(StepDto(
                    kind="received",
                    id=f"s_{ev.id}",
                    timestamp=ev.timestamp,
                    received=received,
                ))
                # Pin these tool_use_ids so the next self-actor message
                # gets tagged as "regenerated_from" them — that's the
                # synthesis-after-tool-results pattern.
                pending_received_ids = [r.tool_use_id for r in results if r.tool_use_id]
                continue

            # Self-actor message (assistant reply on this frame) — add
            # as a step; post-processing promotes the last one to final.
            msg = _message_dto_from_event(ev)
            if pending_received_ids:
                msg.regenerated_from = pending_received_ids
                pending_received_ids = []
            if pending_hints:
                msg.injected_hints = pending_hints
                pending_hints = []
            msg_by_id[msg.message_id] = msg
            current.steps.append(StepDto(
                kind="message",
                id=f"s_{ev.id}",
                timestamp=ev.timestamp,
                message=msg,
            ))
            continue

        if current is None:
            # Only frame lifecycle / header events land here before any
            # user message — safe to skip.
            continue

        if t == EventType.TOOL_STARTED:
            tu_id = p.get("tool_use_id", "")
            # Did this tool spawn a child frame? → render as sub_agent step
            child_frame_id = spawned_children.get(tu_id)
            if child_frame_id:
                child_meta = frame_meta.get(child_frame_id, {})
                sub_turns = _build_turns_for_frame(
                    child_frame_id, events_by_frame, frame_meta, spawned_children,
                )
                sub_summary = _aggregate_sub_agent_summary(sub_turns)

                # What the child handed back to the parent: the last
                # turn's final_message. This IS the tool_result the
                # parent's from_=tool message would have carried, so
                # exposing it here makes "what did the sub-agent
                # return?" a first-class field instead of something
                # the client has to drill into nested turns to find.
                returned_text: str | None = None
                returned_message_id: str | None = None
                if sub_turns and sub_turns[-1].final_message is not None:
                    fm = sub_turns[-1].final_message
                    returned_text = fm.text or None
                    returned_message_id = fm.message_id

                sub = SubAgentDto(
                    frame_id=child_frame_id,
                    role_name=child_meta.get("role_name", "?"),
                    purpose=child_meta.get("purpose", ""),
                    state=child_meta.get("state", "active"),
                    spawned_by_tool_use_id=tu_id,
                    turns=sub_turns,
                    summary=sub_summary,
                    started_at=child_meta.get("opened_at", ev.timestamp),
                    ended_at=child_meta.get("ended_at"),
                    returned_text=returned_text,
                    returned_message_id=returned_message_id,
                    todos=list(child_meta.get("todos") or []),
                )
                current.steps.append(StepDto(
                    kind="sub_agent",
                    id=f"s_{ev.id}",
                    timestamp=ev.timestamp,
                    sub_agent=sub,
                    parallel_group_id=ev.parallel_group_id,
                ))
            else:
                tool_dto = ToolDto(
                    tool_use_id=tu_id,
                    tool_name=p.get("tool_name", "?"),
                    tool_input=dict(p.get("tool_input") or {}),
                    started_at=ev.timestamp,
                )
                tool_by_use_id[tu_id] = tool_dto
                current.steps.append(StepDto(
                    kind="tool",
                    id=f"s_{ev.id}",
                    timestamp=ev.timestamp,
                    tool=tool_dto,
                    parallel_group_id=ev.parallel_group_id,
                ))
            continue

        if t == EventType.TOOL_COMPLETED:
            tu_id = p.get("tool_use_id", "")
            tool_dto = tool_by_use_id.get(tu_id)
            if tool_dto is not None:
                tool_dto.output = p.get("output", "") or ""
                tool_dto.is_error = bool(p.get("is_error", False))
                tool_dto.duration_ms = p.get("duration_ms")
                tool_dto.completed_at = ev.timestamp
            continue

        if t == EventType.ANNOTATION_STORED:
            msg_id = p.get("message_id", "")
            msg_dto = msg_by_id.get(msg_id)
            if msg_dto is not None:
                msg_dto.annotation = AnnotationDto(
                    stop_reason=p.get("stop_reason"),
                    usage=p.get("usage"),
                    duration_ms=p.get("duration_ms"),
                    thinking=p.get("thinking"),
                    llm_request_id=p.get("llm_request_id"),
                )
            continue

        if t == EventType.HINT_INJECTED:
            # Framework whispered a [FRAMEWORK NOTE] into the next
            # LLM_REQUEST. Record the source(s) so the assistant
            # message that this LLM_REQUEST produces can carry them.
            for h in p.get("hints", []) or []:
                if not isinstance(h, dict):
                    continue
                pending_hints.append(HintDto(
                    source=h.get("source", "") or "",
                    text=h.get("text", "") or "",
                ))
            continue

        if t in (EventType.FRAME_RESOLVED, EventType.FRAME_CLOSED):
            current.state = "resolved" if t == EventType.FRAME_RESOLVED else "closed"
            current.ended_at = ev.timestamp
            continue

        if t == EventType.FRAME_ERRORED:
            current.state = "error"
            current.ended_at = ev.timestamp
            continue

        # llm.request / llm.response / header.snapshot etc. — not needed
        # here (pulse derivation handles llm state, header info is pulled
        # from frame_meta).

    if current is not None:
        turns.append(current)

    # Post-process each turn: promote the last message step to final_message
    # and compute the summary block.
    for turn in turns:
        for i in range(len(turn.steps) - 1, -1, -1):
            step = turn.steps[i]
            if step.kind == "message" and step.message is not None:
                turn.final_message = step.message
                turn.steps.pop(i)
                break
        turn.summary = _compute_turn_summary(turn)

    return turns


def build_session_view(
    events: list[Event],
    *,
    session_id: str,
    role_name: str = "",
    model: str = "",
    provider: str = "",
) -> SessionViewDto:
    """Transform a session's full event log into a renderable DTO.

    `session_id`/`role_name`/`model`/`provider` come from the caller
    (the server knows these independently — either from the live
    ServerSession or from the first frame.opened event). Passing them
    in keeps this function a pure function of (events, meta).
    """

    if not events:
        return SessionViewDto(
            session_id=session_id,
            role_name=role_name,
            model=model,
            provider=provider,
            state="active",
        )

    # Pass 1: bucket events by frame, collect frame metadata, find root.
    events_by_frame: dict[str, list[Event]] = defaultdict(list)
    frame_meta: dict[str, dict[str, Any]] = {}
    root_frame_id: str | None = None

    for ev in events:
        fid = ev.frame_id
        if fid is None:
            continue
        events_by_frame[fid].append(ev)

        if ev.type == EventType.FRAME_OPENED:
            p = ev.payload or {}
            frame_meta[fid] = {
                "role_name": p.get("role_name", ""),
                "purpose": p.get("purpose", ""),
                "parent_id": p.get("parent_id"),
                "model": p.get("model", ""),
                "spawned_by_tool_use_id": p.get("spawned_by_tool_use_id"),
                "opened_at": ev.timestamp,
                "state": "active",
                "ended_at": None,
            }
            if not p.get("parent_id") and root_frame_id is None:
                root_frame_id = fid

    # Fill in per-frame state from lifecycle events (state, ended_at)
    for fid, fevents in events_by_frame.items():
        if fid in frame_meta:
            state, ended_at = _frame_state_from_events(fevents)
            frame_meta[fid]["state"] = state
            frame_meta[fid]["ended_at"] = ended_at
            frame_meta[fid]["todos"] = _latest_todos_from_events(fevents)

    # Pass 2: spawned_by_tool_use_id → child frame id, so sub-agent steps
    # can attach on the parent side.
    spawned_children: dict[str, str] = {}
    for fid, meta in frame_meta.items():
        tu_id = meta.get("spawned_by_tool_use_id")
        if tu_id and meta.get("parent_id"):
            spawned_children[tu_id] = fid

    # Pass 3: build the root's turns (recurses into children)
    turns: list[TurnDto] = []
    if root_frame_id is not None:
        turns = _build_turns_for_frame(
            root_frame_id, events_by_frame, frame_meta, spawned_children,
        )

    # Session-level state mirrors the root frame's state.
    root_state = "active"
    if root_frame_id is not None:
        root_state = frame_meta[root_frame_id]["state"]

    # If role/model weren't provided, try to recover them from the root's
    # frame.opened payload (useful for snapshots built without a live
    # ServerSession to consult).
    if not role_name and root_frame_id is not None:
        role_name = frame_meta[root_frame_id].get("role_name") or ""
    if not model and root_frame_id is not None:
        model = frame_meta[root_frame_id].get("model") or ""

    pulse = _derive_pulse(events, root_frame_id)

    root_todos: list[TodoItem] = []
    if root_frame_id is not None:
        root_todos = list(frame_meta[root_frame_id].get("todos") or [])

    return SessionViewDto(
        session_id=session_id,
        role_name=role_name,
        model=model,
        provider=provider,
        state=root_state,
        turns=turns,
        pulse=pulse,
        root_todos=root_todos,
    )
