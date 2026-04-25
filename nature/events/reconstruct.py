"""Replay events to reconstruct Frame state.

Reconstruction is a deterministic function of an event stream: given the
same sequence of events, it produces the same Frame tree every time.
Because llm_agent is pure and the event log captures every state
transition, this is the foundation of resume / replay / time-travel.

Reconstruction only looks at STATE_TRANSITION events. TRACE events
(LLM_*, TOOL_*, USER_INPUT, ERROR) are skipped entirely — this is the
invariant that makes "trace telemetry can be dropped without data loss"
hold.

State-transition event types handled here:
- FRAME_OPENED       → create Frame with role / context shell
- FRAME_RESOLVED     → state → RESOLVED
- FRAME_CLOSED       → state → CLOSED
- FRAME_ERRORED      → state → ERROR
- MESSAGE_APPENDED   → append to frame's conversation
- ANNOTATION_STORED  → collect into annotations-by-message map
- ROLE_CHANGED       → swap frame's role
- PRINCIPLE_ADDED    → append to frame's principles
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nature.context.conversation import (
    Conversation,
    Message,
    MessageAnnotation,
)
from nature.context.types import (
    AgentRole,
    BasePrinciple,
    BasePrincipleSource,
    Context,
    ContextBody,
    ContextHeader,
)
from nature.events.payloads import (
    AgentModelSwappedPayload,
    AnnotationStoredPayload,
    BodyCompactedPayload,
    BudgetConsumedPayload,
    FrameOpenedPayload,
    HeaderSnapshotPayload,
    LedgerApproachRejectedPayload,
    ReadMemorySetPayload,
    LedgerRuleSetPayload,
    LedgerSymbolConfirmedPayload,
    LedgerTestExecutedPayload,
    MessageAppendedPayload,
    PrincipleAddedPayload,
    RoleChangedPayload,
    TodoWrittenPayload,
    load_payload,
)
from nature.events.store import EventStore
from nature.events.types import Event, EventType, is_state_transition
from nature.frame.frame import Frame, FrameState
from nature.protocols.message import Usage


@dataclass
class IncompleteSpan:
    """A trace-level operation that started but never completed.

    Produced by `reconstruct()` when it sees, inside the replayed
    window:

    - `LLM_REQUEST` with a given `request_id` but no matching
      `LLM_RESPONSE` or `LLM_ERROR`  (kind="llm_request")
    - `TOOL_STARTED` with a given `tool_use_id` but no matching
      `TOOL_COMPLETED`                (kind="tool_use")

    Typical causes:
    - Server crashed mid-turn (most common — the LLM was streaming when
      the process died; no RESPONSE landed on disk).
    - Tool call in-flight at crash time.
    - Replay was truncated via `up_to_event_id` to a point inside an
      otherwise complete span; in that case the "incompleteness" is an
      artifact of the slice, not a real crash.

    Callers that want to drive recovery (e.g. roll back the last turn
    before resume) should read `ReplayResult.incomplete_spans` after
    `reconstruct()` returns.
    """

    kind: str  # "llm_request" | "tool_use"
    frame_id: str | None
    identifier: str  # request_id or tool_use_id
    started_event_id: int
    started_timestamp: float
    # Full payload dict of the started event, in case the caller wants
    # details (model, tool_name, tool_input, ...).
    payload: dict = field(default_factory=dict)


@dataclass
class ReplayResult:
    """The reconstructed state after replaying a session's events."""

    frames: dict[str, Frame] = field(default_factory=dict)
    annotations: dict[str, list[MessageAnnotation]] = field(default_factory=dict)
    # parent tool_use_id → spawned child frame id. Built from FRAME_OPENED
    # events that carry `spawned_by_tool_use_id`, cross-checked against
    # MessageAppendedPayload.delegations on parent tool_result messages.
    spawn_by_tool_use: dict[str, str] = field(default_factory=dict)
    # child frame id → (parent_frame_id, parent_message_id, tool_use_id)
    spawn_origin: dict[str, tuple[str, str, str]] = field(default_factory=dict)
    # LLM_REQUEST / TOOL_STARTED spans that didn't see a matching close
    # event. Sorted by started_event_id for determinism.
    incomplete_spans: list[IncompleteSpan] = field(default_factory=list)

    @property
    def root_frames(self) -> list[Frame]:
        return [f for f in self.frames.values() if f.is_root]

    def child_of(self, tool_use_id: str) -> Frame | None:
        """Return the child frame spawned by a given parent tool_use_id."""
        child_id = self.spawn_by_tool_use.get(tool_use_id)
        return self.frames.get(child_id) if child_id else None


def reconstruct(
    session_id: str,
    store: EventStore,
    *,
    up_to_event_id: int | None = None,
) -> ReplayResult:
    """Replay events for a session into frames + annotations.

    `up_to_event_id` — if given, replay only events whose id is ≤ the
    argument and stop. Event ids are session-monotonic starting at 1,
    so this lets callers "rewind" a session to any historical point
    (e.g. a dashboard scrubber, or a test that wants to inspect the
    body right before a specific tool call fired).

    Passing `None` or an id ≥ the last event is equivalent to a full
    replay. Passing `0` or a negative id returns an empty ReplayResult.
    Events are processed in store order; slicing happens at dispatch
    time, so trace events inside the allowed window are still skipped
    via the usual state-transition filter.

    Side effect: `result.incomplete_spans` is populated from any
    `LLM_REQUEST` or `TOOL_STARTED` event inside the window that has
    no matching close event. See `IncompleteSpan` for details.
    """
    result = ReplayResult()
    open_llm: dict[str, Event] = {}      # request_id → started event
    open_tool: dict[str, Event] = {}     # tool_use_id → started event

    for event in store.snapshot(session_id):
        if up_to_event_id is not None and event.id > up_to_event_id:
            break
        # State-transition events drive the frame tree
        if is_state_transition(event.type):
            _apply_event(event, result)
            continue
        # Trace events feed the span tracker
        _track_span(event, open_llm, open_tool)

    # Whatever's still open at the end of the window is incomplete
    for ev in open_llm.values():
        result.incomplete_spans.append(IncompleteSpan(
            kind="llm_request",
            frame_id=ev.frame_id,
            identifier=str(ev.payload.get("request_id", "")),
            started_event_id=ev.id,
            started_timestamp=ev.timestamp,
            payload=dict(ev.payload),
        ))
    for ev in open_tool.values():
        result.incomplete_spans.append(IncompleteSpan(
            kind="tool_use",
            frame_id=ev.frame_id,
            identifier=str(ev.payload.get("tool_use_id", "")),
            started_event_id=ev.id,
            started_timestamp=ev.timestamp,
            payload=dict(ev.payload),
        ))
    result.incomplete_spans.sort(key=lambda s: s.started_event_id)
    return result


def snapshot_events(
    session_id: str,
    store: EventStore,
    *,
    up_to_event_id: int | None = None,
) -> list[Event]:
    """Thin passthrough for callers that only want the raw events.

    Honors `up_to_event_id` the same way `reconstruct` does.
    """
    events = store.snapshot(session_id)
    if up_to_event_id is None:
        return events
    return [e for e in events if e.id <= up_to_event_id]


# ---------------------------------------------------------------------------
# Private: single-event dispatch
# ---------------------------------------------------------------------------


def _apply_event(event: Event, result: ReplayResult) -> None:
    # Hard invariant: reconstruct ONLY touches state-transition events.
    # Trace events may show up in the stream but must never mutate state.
    if not is_state_transition(event.type):
        return
    handler = _HANDLERS.get(event.type)
    if handler is not None:
        handler(event, result)


def _track_span(
    event: Event,
    open_llm: dict[str, Event],
    open_tool: dict[str, Event],
) -> None:
    """Maintain the "span still open" maps used for incomplete detection."""
    if event.type == EventType.LLM_REQUEST:
        req_id = event.payload.get("request_id")
        if req_id:
            open_llm[str(req_id)] = event
    elif event.type in (EventType.LLM_RESPONSE, EventType.LLM_ERROR):
        req_id = event.payload.get("request_id")
        if req_id:
            open_llm.pop(str(req_id), None)
    elif event.type == EventType.TOOL_STARTED:
        tool_use_id = event.payload.get("tool_use_id")
        if tool_use_id:
            open_tool[str(tool_use_id)] = event
    elif event.type == EventType.TOOL_COMPLETED:
        tool_use_id = event.payload.get("tool_use_id")
        if tool_use_id:
            open_tool.pop(str(tool_use_id), None)


def _handle_frame_opened(event: Event, result: ReplayResult) -> None:
    p = load_payload(event)
    assert isinstance(p, FrameOpenedPayload)
    role = AgentRole(
        name=p.role_name,
        description=p.role_description,
        instructions=p.instructions,
        allowed_tools=p.allowed_tools,
        model=p.role_model,
    )
    # Counterparty: root frames reply TO "user"; child frames reply TO
    # their parent's self_actor. The original runtime stamps this at
    # open time (frame/manager.py); replay needs to do the same so a
    # reconstructed child frame matches the live frame byte-for-byte.
    counterparty = "user"
    if p.parent_id:
        parent = result.frames.get(p.parent_id)
        if parent is not None:
            counterparty = parent.self_actor
    frame = Frame(
        id=event.frame_id or "",
        session_id=event.session_id,
        purpose=p.purpose,
        context=Context(
            header=ContextHeader(role=role),
            body=ContextBody(conversation=Conversation()),
        ),
        model=p.model,
        parent_id=p.parent_id,
        counterparty=counterparty,
    )
    result.frames[frame.id] = frame
    if p.parent_id and p.parent_id in result.frames:
        result.frames[p.parent_id].children_ids.append(frame.id)

    # Record the spawn edge so callers can resolve parent→child without
    # scanning the event stream.
    if p.spawned_by_tool_use_id and p.parent_id:
        result.spawn_by_tool_use[p.spawned_by_tool_use_id] = frame.id
        result.spawn_origin[frame.id] = (
            p.parent_id,
            p.spawned_from_message_id or "",
            p.spawned_by_tool_use_id,
        )


def _handle_header_snapshot(event: Event, result: ReplayResult) -> None:
    frame = result.frames.get(event.frame_id or "")
    if frame is None:
        return
    p = load_payload(event)
    assert isinstance(p, HeaderSnapshotPayload)
    frame.context = Context(
        header=ContextHeader(
            role=p.role,
            principles=list(p.principles),
        ),
        body=frame.context.body,
    )


def _handle_message_appended(event: Event, result: ReplayResult) -> None:
    frame = result.frames.get(event.frame_id or "")
    if frame is None:
        return
    p = load_payload(event)
    assert isinstance(p, MessageAppendedPayload)
    frame.context.body.conversation.append(
        Message(
            id=p.message_id,
            from_=p.from_,
            to=p.to,
            content=list(p.content),
            timestamp=p.timestamp or event.timestamp,
        )
    )


def _handle_annotation_stored(event: Event, result: ReplayResult) -> None:
    p = load_payload(event)
    assert isinstance(p, AnnotationStoredPayload)
    if not p.message_id:
        return
    annotation = MessageAnnotation(
        message_id=p.message_id,
        thinking=p.thinking,
        usage=Usage(**p.usage) if p.usage else None,
        stop_reason=p.stop_reason,
        llm_request_id=p.llm_request_id,
        duration_ms=p.duration_ms,
    )
    result.annotations.setdefault(p.message_id, []).append(annotation)


def _handle_role_changed(event: Event, result: ReplayResult) -> None:
    frame = result.frames.get(event.frame_id or "")
    if frame is None:
        return
    p = load_payload(event)
    assert isinstance(p, RoleChangedPayload)
    new_role = AgentRole(
        name=p.role_name,
        description=p.role_description,
        instructions=p.instructions,
        allowed_tools=p.allowed_tools,
        model=p.model,
    )
    frame.context = frame.context.with_role(new_role)


def _handle_todo_written(event: Event, result: ReplayResult) -> None:
    """Replace the frame's todo list with the full new payload.

    Full-list-overwrite semantics: the payload always carries the
    complete intended state, so we set rather than diff. Replay-safe
    by construction.
    """
    frame = result.frames.get(event.frame_id or "")
    if frame is None:
        return
    p = load_payload(event)
    assert isinstance(p, TodoWrittenPayload)
    frame.context.body.todos = list(p.todos)


def _handle_principle_added(event: Event, result: ReplayResult) -> None:
    frame = result.frames.get(event.frame_id or "")
    if frame is None:
        return
    p = load_payload(event)
    assert isinstance(p, PrincipleAddedPayload)
    try:
        source = BasePrincipleSource(p.source)
    except ValueError:
        source = BasePrincipleSource.RUNTIME
    bp = BasePrinciple(
        text=p.text,
        source=source,
        priority=p.priority,
    )
    frame.context = frame.context.with_principle(bp)


def _handle_body_compacted(event: Event, result: ReplayResult) -> None:
    frame = result.frames.get(event.frame_id or "")
    if frame is None:
        return
    p = load_payload(event)
    assert isinstance(p, BodyCompactedPayload)
    frame.context = Context(
        header=frame.context.header,
        body=ContextBody(
            conversation=Conversation(messages=list(p.new_messages)),
        ),
    )


def _handle_frame_resolved(event: Event, result: ReplayResult) -> None:
    frame = result.frames.get(event.frame_id or "")
    if frame is not None:
        frame.state = FrameState.RESOLVED


def _handle_frame_closed(event: Event, result: ReplayResult) -> None:
    frame = result.frames.get(event.frame_id or "")
    if frame is not None:
        frame.state = FrameState.CLOSED


def _handle_frame_errored(event: Event, result: ReplayResult) -> None:
    frame = result.frames.get(event.frame_id or "")
    if frame is not None:
        frame.state = FrameState.ERROR


def _handle_frame_reopened(event: Event, result: ReplayResult) -> None:
    frame = result.frames.get(event.frame_id or "")
    if frame is not None:
        frame.state = FrameState.ACTIVE


def _handle_budget_consumed(event: Event, result: ReplayResult) -> None:
    frame = result.frames.get(event.frame_id or "")
    if frame is None:
        return
    p = load_payload(event)
    assert isinstance(p, BudgetConsumedPayload)
    frame.budget_counts[p.kind] = frame.budget_counts.get(p.kind, 0) + p.delta


def _handle_read_memory_set(event: Event, result: ReplayResult) -> None:
    """Restore a ReadMemory entry as expired (segments=None).

    Events carry metadata only — no content. The reconstructed entry
    is born expired. On next live Read, the tool will re-read from disk
    and fill segments if mtime matches. This is by design: the event log
    stays small, and the body already has the historical Read output.
    """
    frame = result.frames.get(event.frame_id or "")
    if frame is None:
        return
    p = load_payload(event)
    assert isinstance(p, ReadMemorySetPayload)

    from nature.context.read_memory import ReadMemory, ReadMemoryEntry

    read_memory = frame.pack_state.get("read_memory")
    if read_memory is None:
        read_memory = ReadMemory()
        frame.pack_state["read_memory"] = read_memory

    read_memory.set(p.path, ReadMemoryEntry(
        path=p.path,
        mtime_ns=p.mtime_ns,
        total_lines=p.lines,
        segments=None,  # expired — content from disk on next live Read
        depth=p.depth,
    ))


def _handle_ledger_symbol_confirmed(event: Event, result: ReplayResult) -> None:
    frame = result.frames.get(event.frame_id or "")
    if frame is None:
        return
    p = load_payload(event)
    assert isinstance(p, LedgerSymbolConfirmedPayload)
    frame.ledger.symbols_confirmed.append({
        "name": p.name,
        "file": p.file,
        "line_start": p.line_start,
        "line_end": p.line_end,
    })


def _handle_ledger_approach_rejected(event: Event, result: ReplayResult) -> None:
    frame = result.frames.get(event.frame_id or "")
    if frame is None:
        return
    p = load_payload(event)
    assert isinstance(p, LedgerApproachRejectedPayload)
    frame.ledger.approaches_rejected.append({
        "tool": p.tool,
        "input_hash": p.input_hash,
        "reason": p.reason,
    })


def _handle_ledger_test_executed(event: Event, result: ReplayResult) -> None:
    frame = result.frames.get(event.frame_id or "")
    if frame is None:
        return
    p = load_payload(event)
    assert isinstance(p, LedgerTestExecutedPayload)
    frame.ledger.tests_executed.append({
        "command": p.command,
        "status": p.status,
        "summary": p.summary,
    })


def _handle_ledger_rule_set(event: Event, result: ReplayResult) -> None:
    frame = result.frames.get(event.frame_id or "")
    if frame is None:
        return
    p = load_payload(event)
    assert isinstance(p, LedgerRuleSetPayload)
    frame.ledger.rules.append({"rule": p.rule, "source": p.source})


def _handle_agent_model_swapped(event: Event, result: ReplayResult) -> None:
    frame = result.frames.get(event.frame_id or "")
    if frame is None:
        return
    p = load_payload(event)
    assert isinstance(p, AgentModelSwappedPayload)
    frame.model = p.new_model


_HANDLERS = {
    EventType.FRAME_OPENED: _handle_frame_opened,
    EventType.HEADER_SNAPSHOT: _handle_header_snapshot,
    EventType.MESSAGE_APPENDED: _handle_message_appended,
    EventType.ANNOTATION_STORED: _handle_annotation_stored,
    EventType.ROLE_CHANGED: _handle_role_changed,
    EventType.PRINCIPLE_ADDED: _handle_principle_added,
    EventType.BODY_COMPACTED: _handle_body_compacted,
    EventType.TODO_WRITTEN: _handle_todo_written,
    EventType.FRAME_RESOLVED: _handle_frame_resolved,
    EventType.FRAME_CLOSED: _handle_frame_closed,
    EventType.FRAME_ERRORED: _handle_frame_errored,
    EventType.FRAME_REOPENED: _handle_frame_reopened,
    EventType.BUDGET_CONSUMED: _handle_budget_consumed,
    EventType.READ_MEMORY_SET: _handle_read_memory_set,
    EventType.LEDGER_SYMBOL_CONFIRMED: _handle_ledger_symbol_confirmed,
    EventType.LEDGER_APPROACH_REJECTED: _handle_ledger_approach_rejected,
    EventType.LEDGER_TEST_EXECUTED: _handle_ledger_test_executed,
    EventType.LEDGER_RULE_SET: _handle_ledger_rule_set,
    EventType.AGENT_MODEL_SWAPPED: _handle_agent_model_swapped,
}
