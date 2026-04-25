"""Typed payload models, one per EventType.

`Event.payload` remains a plain dict on disk (for JSONL storage and
schema-evolution tolerance), but producers and consumers go through the
typed models defined here. This is the single place that knows the
exact shape of each event's payload.

Usage:
    # producer (AreaManager)
    self._emit(frame, EventType.FRAME_OPENED, FrameOpenedPayload(...))

    # consumer (reconstruct)
    p = load_payload(event)        # dispatches on event.type
    if isinstance(p, FrameOpenedPayload):
        ...
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from nature.context.conversation import Message
from nature.context.types import AgentRole, BasePrinciple
from nature.events.types import Event, EventType
from nature.protocols.message import ContentBlock
from nature.protocols.todo import TodoItem


class _PayloadBase(BaseModel):
    """Common config: allow arbitrary types (ContentBlock is a union)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)


# ── Session lifecycle ────────────────────────────────────────────────


class SessionStartedPayload(_PayloadBase):
    """Provenance snapshot emitted once per session.

    Makes the event log self-describing: every variable that shapes
    the session's behaviour is captured at session start so later
    analyses can reconstruct "what was run" without touching the
    filesystem.

    Fields:
    - `preset_name` — convenience reference, points into `preset`.
    - `preset` — full preset JSON (roster, model_overrides,
      prompt_overrides, root_agent).
    - `agents_resolved` — per-agent snapshot of effective definition:
      `{name: {model, allowed_tools, allowed_interventions,
      instructions, description}}`. Uses the *resolved* model
      (preset override applied) and *resolved* instructions
      (prompt_overrides applied) so this is what actually ran, not
      just what was configured.
    - `hosts_used` — distinct host names pulled from agents_resolved.
    - `models_used` — distinct bare model strings.
    - `model_budgets` — per-`host::model` TokenBudget in use for
      this session (context_window + output_reservation).
    - `parent_session_id` / `forked_from_event_id` — fork lineage;
      both None for organically created sessions.
    - `repo_git_sha` — optional, stamped by eval runner for
      provenance; empty for non-eval sessions.
    """

    preset_name: str
    preset: dict[str, Any] = Field(default_factory=dict)
    agents_resolved: dict[str, dict[str, Any]] = Field(default_factory=dict)
    hosts_used: list[str] = Field(default_factory=list)
    models_used: list[str] = Field(default_factory=list)
    model_budgets: dict[str, dict[str, int]] = Field(default_factory=dict)
    parent_session_id: str | None = None
    forked_from_event_id: int | None = None
    repo_git_sha: str = ""


# ── Frame lifecycle ──────────────────────────────────────────────────


class FrameOpenedPayload(_PayloadBase):
    purpose: str
    parent_id: str | None = None
    role_name: str
    role_description: str = ""
    instructions: str = ""
    allowed_tools: list[str] | None = None
    model: str = ""
    role_model: str | None = None
    # Populated when this frame was spawned by a parent's Agent tool call.
    # Kept optional in Phase 1 so the field ships without changing producers.
    spawned_from_message_id: str | None = None
    spawned_by_tool_use_id: str | None = None


class FrameResolvedPayload(_PayloadBase):
    bubble_message_id: str | None = None


class FrameClosedPayload(_PayloadBase):
    pass


class FrameReopenedPayload(_PayloadBase):
    """Terminal frame reactivated (RESOLVED/CLOSED/ERROR → ACTIVE).

    Emitted by SessionRegistry.resume_session after reconstruct() so
    replay sees the transition explicitly. `previous_state` is recorded
    for observability; reconstruct itself only needs the type of the
    event to reset state.
    """

    previous_state: str = ""
    reason: str = "resume"


class FrameErroredPayload(_PayloadBase):
    """The state-transition twin of LLM_ERROR.

    LLM_ERROR (trace) carries the raw exception details. FRAME_ERRORED
    is what reconstruct watches — it says "this frame's state is now
    ERROR". Keeping them separate means reconstruct never reads trace
    events, and a frame can enter ERROR for non-LLM reasons later.
    """

    error_type: str = ""
    message: str = ""
    # Optional cross-ref back to the LLM_ERROR trace event id.
    trace_event_id: int | None = None


# ── Conversation (body) ──────────────────────────────────────────────


class MessageAppendedPayload(_PayloadBase):
    message_id: str
    from_: str
    to: str
    content: list[ContentBlock] = Field(default_factory=list)
    timestamp: float
    # Tool-use id → child frame id, for tool_result messages whose blocks
    # came from Agent delegations. Empty on every non-delegation message.
    # Consumers that want to drill into a child frame look up
    # ReplayResult.frames[child_id] and render from there.
    delegations: dict[str, str] = Field(default_factory=dict)


class AnnotationStoredPayload(_PayloadBase):
    message_id: str
    thinking: list[str] | None = None
    usage: dict[str, Any] | None = None
    stop_reason: str | None = None
    llm_request_id: str | None = None
    duration_ms: int | None = None


# ── LLM interactions ─────────────────────────────────────────────────


class LLMRequestPayload(_PayloadBase):
    request_id: str
    model: str
    message_count: int
    tool_count: int


class LLMResponsePayload(_PayloadBase):
    request_id: str | None = None
    stop_reason: str | None = None
    usage: dict[str, Any] | None = None


class LLMErrorPayload(_PayloadBase):
    request_id: str | None = None
    error_type: str
    message: str


# ── Tool execution ───────────────────────────────────────────────────


class ToolStartedPayload(_PayloadBase):
    tool_use_id: str
    tool_name: str
    tool_input: dict[str, Any] = Field(default_factory=dict)


class ToolCompletedPayload(_PayloadBase):
    tool_use_id: str
    tool_name: str
    output: str
    is_error: bool = False
    duration_ms: int = 0


# ── Context mutations (header) ───────────────────────────────────────


class BodyCompactedPayload(_PayloadBase):
    """Full post-compaction body snapshot.

    Carrying the complete trimmed conversation (not a diff) keeps
    replay simple: reconstruct() replaces the frame's body with
    `new_messages`, and subsequent MESSAGE_APPENDED events naturally
    append on top of that trimmed list.
    """

    strategy: str
    tokens_before: int = 0
    tokens_after: int = 0
    new_messages: list[Message] = Field(default_factory=list)
    summary: str | None = None


class HeaderSnapshotPayload(_PayloadBase):
    """Full header dump — emitted right after FRAME_OPENED.

    Separating this from FRAME_OPENED makes "header is the static
    cacheable half of context" an enforced contract at the event level:
    a resume reads a single snapshot event to reconstitute the role +
    principles, and cache boundary calculations can target exactly this
    event id.
    """

    role: AgentRole
    principles: list[BasePrinciple] = Field(default_factory=list)


class PrincipleAddedPayload(_PayloadBase):
    source: str
    text: str
    priority: int = 0


class RoleChangedPayload(_PayloadBase):
    role_name: str
    role_description: str = ""
    instructions: str = ""
    allowed_tools: list[str] | None = None
    model: str | None = None


# ── Session-level ────────────────────────────────────────────────────


class UserInputPayload(_PayloadBase):
    text: str
    source: str = "user"


class ErrorPayload(_PayloadBase):
    error_type: str
    message: str


class TodoWrittenPayload(_PayloadBase):
    """Full overwrite of a frame's todo list.

    Carrying the entire new list (rather than a diff) keeps replay
    trivial: reconstruct() just sets `frame.context.body.todos`. The
    `source` field records who emitted the write — usually
    `"todo_write_tool"`, but reserved for future cases where the
    framework synthesizes a clear (e.g., on frame.closed).
    """

    todos: list[TodoItem] = Field(default_factory=list)
    source: str = "todo_write_tool"


class HintInjectedPayload(_PayloadBase):
    """Footer hints appended to an LLM request by ContextComposer rules.

    Each hint corresponds to one footer rule that fired (e.g.,
    `synthesis_nudge` after a tool_result message). Trace-only — the
    body is not mutated; this event exists purely so dashboards can
    show "the framework whispered something to the LLM here".
    """

    request_id: str | None = None
    hints: list[dict[str, Any]] = Field(default_factory=list)


class ParallelGroupStartedPayload(_PayloadBase):
    """Marker event opening a parallel-tool-execution bracket.

    Emitted by `AreaManager._execute_and_apply` right before a batch
    of concurrency-safe tools is handed to `asyncio.gather`. The
    `group_id` identifies the batch and appears on every inner
    TOOL_STARTED / TOOL_COMPLETED event's envelope
    (`parallel_group_id`). `tool_count` lets consumers size their
    UI up-front without scanning the inner events.
    """

    group_id: str
    tool_count: int = 0


class ParallelGroupCompletedPayload(_PayloadBase):
    """Marker event closing a parallel-tool-execution bracket.

    Emitted right after `asyncio.gather` resolves, regardless of
    per-tool success. `duration_ms` is the wall-clock cost of the
    whole batch (not the sum of inner tool durations), so UIs can
    show the actual parallelism speedup.
    """

    group_id: str
    duration_ms: int | None = None


# ── Budget tracking (state-transition: BUDGET_CONSUMED only) ─────────


class BudgetConsumedPayload(_PayloadBase):
    """One unit of a tracked budget was used.

    State-transition: reconstruct adds `delta` to
    `Frame.budget_counts[kind]`. `limit` is informational (the live
    value at emission time) and is intentionally not load-bearing —
    caps are checked against the currently-configured limit at the
    call site, not against the number baked into past events.
    """

    kind: str
    delta: int = 1
    used: int = 0
    limit: int | None = None


class BudgetWarningPayload(_PayloadBase):
    """Budget crossed its soft threshold (typically 80%)."""

    kind: str
    used: int
    limit: int


class BudgetBlockedPayload(_PayloadBase):
    """A tracked-tool call was refused because the hard cap was reached."""

    kind: str
    tool: str
    reason: str = ""


# ── Edit / loop / path / parse guards (trace) ────────────────────────


class LoopDetectedPayload(_PayloadBase):
    """N near-identical tool_use bodies observed in a row.

    `input_hash` is a short digest of the tool input (see the loop
    detector implementation in Phase 2.2 for the exact hashing rule).
    """

    tool: str
    input_hash: str
    attempts: int


class LoopBlockedPayload(_PayloadBase):
    """The framework refused the next retry after LOOP_DETECTED fired."""

    tool: str
    input_hash: str


class EditMissPayload(_PayloadBase):
    """Edit's `old_string` was not found in the target file.

    `fuzzy_match` carries the closest window the fuzzy suggester
    surfaced (if any), along with its line range. Both fields are
    optional so producers that skip fuzzy matching still emit a
    well-formed event.
    """

    file: str
    fuzzy_match: str | None = None
    lineno: int | None = None


class PathInvalidPayload(_PayloadBase):
    """A Read/Edit/Write call targeted a path that does not exist."""

    path: str
    tool: str = ""
    suggestions: list[str] = Field(default_factory=list)


class ParseRetryPayload(_PayloadBase):
    """The text tool parser retried a fallback parse attempt."""

    attempt: int
    reason: str = ""


# ── Memory Ledger (state-transition) ────────────────────────────────


class ReadMemorySetPayload(_PayloadBase):
    """A file was read and its state recorded in ReadMemory.

    Persists metadata only — content lives in the live ReadMemory
    segments, not in the event store. On reconstruct, entries are
    restored as expired (segments=None) and re-read from disk as needed.

    Renamed from LedgerFileConfirmedPayload (Phase 1).
    """

    path: str
    content_hash: str
    mtime_ns: int = 0
    lines: int = 0
    offset: int | None = None
    limit: int | None = None
    depth: int = 0


class LedgerSymbolConfirmedPayload(_PayloadBase):
    """A named symbol (function / class / const) was located in a file."""

    name: str
    file: str
    line_start: int = 0
    line_end: int = 0


class LedgerApproachRejectedPayload(_PayloadBase):
    """An approach was tried and failed — don't retry with the same inputs.

    `input_hash` is the same tool-input digest used by loop detection;
    callers that store both can dedupe.
    """

    tool: str
    input_hash: str
    reason: str = ""


class LedgerTestExecutedPayload(_PayloadBase):
    """A test command was run; the outcome is now a confirmed fact."""

    command: str
    status: str
    summary: str = ""


class LedgerRuleSetPayload(_PayloadBase):
    """A guardrail or constraint declared by the user / judge.

    Rules promote across frame boundaries (see v2 §6.4) because they
    describe policy, not transient state.
    """

    rule: str
    source: str = ""


# ── Agent runtime model swap (state-transition) ─────────────────────


class AgentModelSwappedPayload(_PayloadBase):
    """The active LLM model for this frame was replaced at runtime.

    Used by the escalation flow (v2 §7) and by dashboard-driven
    overrides. `previous_model` is recorded so a future de-escalation
    can roll back without the framework having to scan history.
    """

    previous_model: str = ""
    new_model: str
    reason: str = ""


# ── Dispatch table ───────────────────────────────────────────────────


PAYLOAD_MODELS: dict[EventType, type[_PayloadBase]] = {
    EventType.SESSION_STARTED: SessionStartedPayload,
    EventType.FRAME_OPENED: FrameOpenedPayload,
    EventType.FRAME_RESOLVED: FrameResolvedPayload,
    EventType.FRAME_CLOSED: FrameClosedPayload,
    EventType.FRAME_ERRORED: FrameErroredPayload,
    EventType.FRAME_REOPENED: FrameReopenedPayload,
    EventType.MESSAGE_APPENDED: MessageAppendedPayload,
    EventType.ANNOTATION_STORED: AnnotationStoredPayload,
    EventType.HEADER_SNAPSHOT: HeaderSnapshotPayload,
    EventType.BODY_COMPACTED: BodyCompactedPayload,
    EventType.LLM_REQUEST: LLMRequestPayload,
    EventType.LLM_RESPONSE: LLMResponsePayload,
    EventType.LLM_ERROR: LLMErrorPayload,
    EventType.TOOL_STARTED: ToolStartedPayload,
    EventType.TOOL_COMPLETED: ToolCompletedPayload,
    EventType.PRINCIPLE_ADDED: PrincipleAddedPayload,
    EventType.ROLE_CHANGED: RoleChangedPayload,
    EventType.USER_INPUT: UserInputPayload,
    EventType.ERROR: ErrorPayload,
    EventType.HINT_INJECTED: HintInjectedPayload,
    EventType.TODO_WRITTEN: TodoWrittenPayload,
    EventType.PARALLEL_GROUP_STARTED: ParallelGroupStartedPayload,
    EventType.PARALLEL_GROUP_COMPLETED: ParallelGroupCompletedPayload,
    EventType.BUDGET_CONSUMED: BudgetConsumedPayload,
    EventType.BUDGET_WARNING: BudgetWarningPayload,
    EventType.BUDGET_BLOCKED: BudgetBlockedPayload,
    EventType.LOOP_DETECTED: LoopDetectedPayload,
    EventType.LOOP_BLOCKED: LoopBlockedPayload,
    EventType.EDIT_MISS: EditMissPayload,
    EventType.PATH_INVALID: PathInvalidPayload,
    EventType.PARSE_RETRY: ParseRetryPayload,
    EventType.READ_MEMORY_SET: ReadMemorySetPayload,
    EventType.LEDGER_SYMBOL_CONFIRMED: LedgerSymbolConfirmedPayload,
    EventType.LEDGER_APPROACH_REJECTED: LedgerApproachRejectedPayload,
    EventType.LEDGER_TEST_EXECUTED: LedgerTestExecutedPayload,
    EventType.LEDGER_RULE_SET: LedgerRuleSetPayload,
    EventType.AGENT_MODEL_SWAPPED: AgentModelSwappedPayload,
}


def load_payload(event: Event) -> _PayloadBase:
    """Validate event.payload into its declared typed model.

    Unknown event types fall back to an empty model — callers should
    use `isinstance` checks on the result, not blindly trust the return.
    """
    cls = PAYLOAD_MODELS.get(event.type)
    if cls is None:
        return _PayloadBase()
    return cls.model_validate(event.payload)


def dump_payload(payload: _PayloadBase | dict[str, Any]) -> dict[str, Any]:
    """Producer-side helper: normalize a typed payload into a JSONL-safe dict.

    Accepts plain dicts too so callers can still emit ad-hoc payloads
    during the migration — phase 1 flips every producer but this keeps
    the `_emit` helper forgiving while the switch lands.
    """
    if isinstance(payload, _PayloadBase):
        return payload.model_dump(mode="json")
    return payload


__all__ = [
    "SessionStartedPayload",
    "FrameOpenedPayload",
    "FrameResolvedPayload",
    "FrameClosedPayload",
    "FrameErroredPayload",
    "FrameReopenedPayload",
    "MessageAppendedPayload",
    "AnnotationStoredPayload",
    "HeaderSnapshotPayload",
    "BodyCompactedPayload",
    "LLMRequestPayload",
    "LLMResponsePayload",
    "LLMErrorPayload",
    "ToolStartedPayload",
    "ToolCompletedPayload",
    "PrincipleAddedPayload",
    "RoleChangedPayload",
    "UserInputPayload",
    "ErrorPayload",
    "HintInjectedPayload",
    "TodoWrittenPayload",
    "ParallelGroupStartedPayload",
    "ParallelGroupCompletedPayload",
    "BudgetConsumedPayload",
    "BudgetWarningPayload",
    "BudgetBlockedPayload",
    "LoopDetectedPayload",
    "LoopBlockedPayload",
    "EditMissPayload",
    "PathInvalidPayload",
    "ParseRetryPayload",
    "ReadMemorySetPayload",
    "LedgerSymbolConfirmedPayload",
    "LedgerApproachRejectedPayload",
    "LedgerTestExecutedPayload",
    "LedgerRuleSetPayload",
    "AgentModelSwappedPayload",
    "PAYLOAD_MODELS",
    "load_payload",
    "dump_payload",
]
