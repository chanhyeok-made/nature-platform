"""Frame — the mutable container around a Context.

A Frame holds exactly one Context, plus metadata about its place in the
frame tree (parent, children) and lifecycle state. Frames are mutated
by the AreaManager as execution proceeds; the Context inside a frame is
the agent's view of "what has happened and who I am".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from nature.context.types import Context


@dataclass
class FrameLedger:
    """Confirmed facts the frame has learned through tool execution.

    Kept separate from `context.body` because body mixes LLM output
    (unreliable) with tool results (reliable). Ledger is the
    tool-result-only zone: reconstruct populates it from `ledger.*`
    events, never from message content. See v2 §6 for rationale.

    File-state knowledge moved out of this dataclass: what used to be
    `files_confirmed` now lives in `Frame.pack_state["read_memory"]`
    (owned by the `file_state` Pack) with richer segment/range
    tracking. The remaining four lists are Phase 1 scaffolding for the
    v2-P6 Memory Ledger MVP — no producers wired yet.
    """

    symbols_confirmed: list[dict[str, Any]] = field(default_factory=list)
    approaches_rejected: list[dict[str, Any]] = field(default_factory=list)
    tests_executed: list[dict[str, Any]] = field(default_factory=list)
    rules: list[dict[str, Any]] = field(default_factory=list)


class FrameState(str, Enum):
    """Lifecycle states of a Frame.

    ACTIVE:         currently running or ready to run
    AWAITING_USER:  paused for user input (NEEDS_USER signal)
    RESOLVED:       completed successfully, bubble message ready
    ERROR:          hit an unrecoverable error
    CLOSED:         popped from the manager, memory released
    """

    ACTIVE = "active"
    AWAITING_USER = "awaiting_user"
    RESOLVED = "resolved"
    ERROR = "error"
    CLOSED = "closed"


@dataclass
class Frame:
    """A scope that owns a Context and tracks lifecycle.

    Frames are created by AreaManager.open_root / open_child and mutated
    during AreaManager.run as the agent function is invoked repeatedly.
    Do not mutate Frames directly outside the manager.

    `counterparty` is the actor this frame's agent replies TO when it
    speaks. Stamped at open time and never changes — this prevents the
    counterparty from drifting to "tool" after a tool result arrives.
    Root frames reply to "user"; child frames reply to their parent's
    actor (e.g. core for a researcher spawned by core).
    """

    id: str
    session_id: str
    purpose: str
    context: Context
    model: str
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    state: FrameState = FrameState.ACTIVE
    counterparty: str = "user"
    # Frame-local budget counters ({kind: used}). Incremented by
    # `budget.consumed` events during replay; checked live by the
    # `max_*_per_frame` caps enforced by the budget Pack.
    budget_counts: dict[str, int] = field(default_factory=dict)
    # Confirmed-fact store, see FrameLedger docstring.
    ledger: FrameLedger = field(default_factory=FrameLedger)
    # Generic Pack-owned state bag. Packs write to their namespace key
    # at installation time; tools access via ToolContext.pack_state.
    pack_state: dict[str, Any] = field(default_factory=dict)

    @property
    def self_actor(self) -> str:
        """The actor name this frame's agent uses when it speaks."""
        return self.context.header.role.name

    @property
    def is_root(self) -> bool:
        return self.parent_id is None
