"""Event types and the Event envelope.

Each event is an append-only record. Events are the ONLY way execution
communicates with the outside world — UIs, persistence, replay, and
reconstruct() all read from events.

Events fall into two explicit categories:

- STATE_TRANSITION — drives reconstruct(). Removing one changes the
  rebuilt Frame state; never omit or reorder without care. Replay
  applies these deterministically in event order.
- TRACE — observability only. UIs and debuggers read them; reconstruct
  ignores them. Skipping or dropping a trace event MUST NOT change the
  reconstructed state. This is the invariant Phase 2 locks in.

Design rules:
- Append-only: events are never modified once written
- Self-contained: each payload has everything needed to apply it
- Type-specific payloads: see nature.events.payloads for the models
- Replay-deterministic on state-transition events alone
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """All event types emitted by the execution layer.

    Each value is a `{domain}.{verb}` string for easy grep/filter. Extending
    this enum is the only way to add new event kinds — do not reuse existing
    types for new meanings.
    """

    # ── Session lifecycle (state-transition) ───────────────────────────
    # Emitted once per session, before any frame. Self-describing:
    # carries the full preset composition, the resolved per-agent
    # definition (JSON + instructions text), the hosts and models
    # referenced, and fork lineage when applicable. Makes each event
    # log a provenance record independent of the filesystem.
    SESSION_STARTED = "session.started"

    # ── Frame lifecycle (state-transition) ─────────────────────────────
    FRAME_OPENED = "frame.opened"
    FRAME_RESOLVED = "frame.resolved"
    FRAME_CLOSED = "frame.closed"
    FRAME_ERRORED = "frame.errored"
    # Emitted when resume_session reactivates a terminal frame so the
    # state transition RESOLVED/CLOSED/ERROR → ACTIVE is explicit in
    # the log instead of implied by a MESSAGE_APPENDED landing on a
    # terminal frame.
    FRAME_REOPENED = "frame.reopened"

    # ── Conversation / body (state-transition) ─────────────────────────
    MESSAGE_APPENDED = "message.appended"
    ANNOTATION_STORED = "annotation.stored"

    # ── Header (state-transition) ──────────────────────────────────────
    # Emitted immediately after FRAME_OPENED; carries the full role +
    # initial principles. Incremental ROLE_CHANGED / PRINCIPLE_ADDED
    # events apply on top of the snapshot.
    HEADER_SNAPSHOT = "header.snapshot"
    PRINCIPLE_ADDED = "principle.added"
    ROLE_CHANGED = "role.changed"

    # ── Body compaction (state-transition) ────────────────────────────
    # Replaces the frame's body with a post-compaction snapshot so
    # resume lands on the same trimmed conversation a live run saw.
    BODY_COMPACTED = "body.compacted"

    # ── Todos (state-transition) ───────────────────────────────────────
    # Replaces the frame's todo list with the full new list provided by
    # a TodoWrite call. The full-list-overwrite semantics keeps replay
    # trivial (no per-item diffing) and matches Claude Code's
    # TodoWrite contract.
    TODO_WRITTEN = "todo.written"

    # ── LLM interactions (trace) ───────────────────────────────────────
    LLM_REQUEST = "llm.request"
    LLM_RESPONSE = "llm.response"
    LLM_ERROR = "llm.error"

    # ── Tool execution (trace) ─────────────────────────────────────────
    TOOL_STARTED = "tool.started"
    TOOL_COMPLETED = "tool.completed"

    # ── Session-level (trace) ──────────────────────────────────────────
    USER_INPUT = "user.input"
    ERROR = "error"

    # ── Framework hint injection (trace) ───────────────────────────────
    # Emitted whenever a footer rule fires and a hint message is
    # appended to the upcoming LLM request. Pure observability — does
    # not affect frame state, the body is not mutated.
    HINT_INJECTED = "hint.injected"

    # ── Parallel execution brackets (trace) ────────────────────────────
    # Emitted around a batch of concurrency-safe tool calls that run
    # in parallel via `asyncio.gather` inside a single action
    # dispatch. All events emitted between the STARTED / COMPLETED
    # pair carry the same `parallel_group_id` on their envelope, so
    # downstream consumers (session fork validation, dashboard
    # timeline) can recognize the group as an atomic unit: forking
    # at an event strictly inside a bracket is ambiguous (events
    # 43..48 in a parallel run have no total order), so the fork
    # API rejects such ids and points callers at the nearest
    # boundary (STARTED or COMPLETED) instead.
    PARALLEL_GROUP_STARTED = "parallel.started"
    PARALLEL_GROUP_COMPLETED = "parallel.completed"

    # ── Budget tracking (Phase 3 prerequisite) ─────────────────────────
    # `budget.consumed` is the only state-transition event of the three:
    # it carries the counter increment that reconstruct replays into
    # `Frame.budget_counts`. 80%-threshold warnings and 100%-block
    # refusals are derivable from the counter at check time, so they
    # ship as pure trace events for observability.
    BUDGET_CONSUMED = "budget.consumed"
    BUDGET_WARNING = "budget.warning"
    BUDGET_BLOCKED = "budget.blocked"

    # ── Edit / loop / path / parse guards (Phase 2 + 4 prerequisites) ──
    # All trace-only. Their downstream effects (a failed Edit, a
    # refused tool call, a reparsed tool block) already show up in the
    # usual TOOL_COMPLETED / MESSAGE_APPENDED stream; these events
    # exist so dashboards and retroactive analysis can explain *why*.
    LOOP_DETECTED = "loop.detected"
    LOOP_BLOCKED = "loop.blocked"
    EDIT_MISS = "edit.miss"
    PATH_INVALID = "path.invalid"
    PARSE_RETRY = "parse.retry"

    # ── Memory Ledger (v2 §6) ──────────────────────────────────────────
    # State-transition: reconstruct replays every ledger event into
    # `Frame.ledger`. These are the confirmed-fact store that survives
    # turn boundaries so small local models don't re-discover what a
    # previous turn already verified.
    READ_MEMORY_SET = "read_memory.set"
    LEDGER_SYMBOL_CONFIRMED = "ledger.symbol_confirmed"
    LEDGER_APPROACH_REJECTED = "ledger.approach_rejected"
    LEDGER_TEST_EXECUTED = "ledger.test_executed"
    LEDGER_RULE_SET = "ledger.rule_set"

    # ── Agent runtime model swap (v2 §7) ───────────────────────────────
    # State-transition: escalation (or de-escalation) flips which model
    # a frame uses mid-session without spawning a new frame. Replay
    # must see the swap so the rebuilt frame's `model` matches live.
    AGENT_MODEL_SWAPPED = "agent.model_swapped"


class EventCategory(str, Enum):
    """Classification of events into state-driving vs observability.

    reconstruct() applies STATE_TRANSITION events and ignores TRACE.
    This makes the invariant explicit and testable: dropping TRACE
    events must not change the rebuilt frame tree.
    """

    STATE_TRANSITION = "state_transition"
    TRACE = "trace"


EVENT_CATEGORIES: dict["EventType", EventCategory] = {
    # State transitions
    EventType.SESSION_STARTED: EventCategory.STATE_TRANSITION,
    EventType.FRAME_OPENED: EventCategory.STATE_TRANSITION,
    EventType.FRAME_RESOLVED: EventCategory.STATE_TRANSITION,
    EventType.FRAME_CLOSED: EventCategory.STATE_TRANSITION,
    EventType.FRAME_ERRORED: EventCategory.STATE_TRANSITION,
    EventType.FRAME_REOPENED: EventCategory.STATE_TRANSITION,
    EventType.MESSAGE_APPENDED: EventCategory.STATE_TRANSITION,
    EventType.ANNOTATION_STORED: EventCategory.STATE_TRANSITION,
    EventType.HEADER_SNAPSHOT: EventCategory.STATE_TRANSITION,
    EventType.PRINCIPLE_ADDED: EventCategory.STATE_TRANSITION,
    EventType.ROLE_CHANGED: EventCategory.STATE_TRANSITION,
    EventType.BODY_COMPACTED: EventCategory.STATE_TRANSITION,
    EventType.TODO_WRITTEN: EventCategory.STATE_TRANSITION,
    EventType.BUDGET_CONSUMED: EventCategory.STATE_TRANSITION,
    EventType.READ_MEMORY_SET: EventCategory.STATE_TRANSITION,
    EventType.LEDGER_SYMBOL_CONFIRMED: EventCategory.STATE_TRANSITION,
    EventType.LEDGER_APPROACH_REJECTED: EventCategory.STATE_TRANSITION,
    EventType.LEDGER_TEST_EXECUTED: EventCategory.STATE_TRANSITION,
    EventType.LEDGER_RULE_SET: EventCategory.STATE_TRANSITION,
    EventType.AGENT_MODEL_SWAPPED: EventCategory.STATE_TRANSITION,
    # Trace (observability only)
    EventType.LLM_REQUEST: EventCategory.TRACE,
    EventType.LLM_RESPONSE: EventCategory.TRACE,
    EventType.LLM_ERROR: EventCategory.TRACE,
    EventType.TOOL_STARTED: EventCategory.TRACE,
    EventType.TOOL_COMPLETED: EventCategory.TRACE,
    EventType.USER_INPUT: EventCategory.TRACE,
    EventType.ERROR: EventCategory.TRACE,
    EventType.HINT_INJECTED: EventCategory.TRACE,
    EventType.PARALLEL_GROUP_STARTED: EventCategory.TRACE,
    EventType.PARALLEL_GROUP_COMPLETED: EventCategory.TRACE,
    EventType.BUDGET_WARNING: EventCategory.TRACE,
    EventType.BUDGET_BLOCKED: EventCategory.TRACE,
    EventType.LOOP_DETECTED: EventCategory.TRACE,
    EventType.LOOP_BLOCKED: EventCategory.TRACE,
    EventType.EDIT_MISS: EventCategory.TRACE,
    EventType.PATH_INVALID: EventCategory.TRACE,
    EventType.PARSE_RETRY: EventCategory.TRACE,
}


def category_of(event_type: "EventType") -> EventCategory:
    """Return the category for an event type. Unknown types default to TRACE."""
    return EVENT_CATEGORIES.get(event_type, EventCategory.TRACE)


def is_state_transition(event_type: "EventType") -> bool:
    return category_of(event_type) is EventCategory.STATE_TRANSITION


class Event(BaseModel):
    """An append-only event in the execution log.

    The `id` field is monotonic within a session and is assigned by the
    store at append time — callers should pass `id=0` and trust the store.

    `parallel_group_id` tags events that were emitted as part of a
    parallel-execution batch. All events sharing the same id (plus the
    bracketing PARALLEL_GROUP_STARTED/COMPLETED) form one atomic
    group; forking at an event strictly inside that group is ambiguous
    (the inner events have no total order relative to each other), so
    the fork API rejects it and points callers at the bracket
    boundaries. `None` means "not part of a parallel batch" and is the
    default for organic, sequential events.
    """

    model_config = {"frozen": True}

    id: int
    session_id: str
    frame_id: str | None = None
    timestamp: float
    type: EventType
    payload: dict[str, Any] = Field(default_factory=dict)
    parallel_group_id: str | None = None
    schema_version: int = 1

    @property
    def category(self) -> EventCategory:
        return category_of(self.type)

    @property
    def is_state_transition(self) -> bool:
        return is_state_transition(self.type)
