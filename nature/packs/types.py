"""Pack architecture type definitions.

Conventions (see `pack_architecture.md`):

- Interventions are the (B) side of the framework: framework-initiated
  behavior that reacts to conditions. They come in three flavors: Gate,
  Listener, Contributor.
- Each Intervention returns a list of Effect objects. Registry applies
  the effects; Interventions never mutate state directly.
- Tools are the (A) side — agent-invoked. This module does not redefine
  `Tool`; we re-export `nature.protocols.tool.Tool` for consistency.
- A Capability groups Tools and Interventions under one named unit.
- A Pack is the installable outer container; a Pack holds one or more
  Capabilities.

M1 deviations from the design doc (intentional, documented in §13):

- `InterventionContext.frame` is `Frame | None` rather than always
  deepcopy'd. Contributors dispatched from `ContextComposer.compose()`
  don't have a Frame (compose takes a bare Context), so for contributor
  triggers the fields `body` / `header` / `self_actor` are populated
  directly instead. Tool-call dispatch sites (where a Frame is present)
  will pass a real Frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Literal, Protocol, Union

from nature.events.types import Event, EventType
from nature.protocols.tool import Tool as Tool  # re-export for pack authors

if TYPE_CHECKING:
    from nature.context.types import ContextBody, ContextHeader
    from nature.frame.frame import Frame
    from nature.packs.registry import PackRegistry


# ──────────────────────────────────────────────────────────────────────
# Enums — phases that triggers can key on
# ──────────────────────────────────────────────────────────────────────


class ToolPhase(str, Enum):
    PRE = "pre"
    POST = "post"


class LLMPhase(str, Enum):
    PRE = "pre"
    POST = "post"


class TurnPhase(str, Enum):
    BEFORE_LLM = "before_llm"
    AFTER_LLM = "after_llm"


class FramePhase(str, Enum):
    OPENED = "opened"
    RESOLVING = "resolving"
    RESOLVED = "resolved"
    ERRORED = "errored"
    CLOSED = "closed"


InterventionKind = Literal["gate", "listener", "contributor"]


class InterventionPhase(int, Enum):
    """Listener execution phase (see pack_architecture.md §4.5).

    Same-trigger listeners are layered into 2 explicit phases instead of
    cross-referencing each other. PRIMARY listeners react directly to
    the trigger; POST_EFFECT listeners receive the PRIMARY listeners'
    effect list via `ctx.primary_effects` and may emit further effects
    based on what PRIMARY did.

    No third phase. Cycles are structurally impossible because
    POST_EFFECT cannot trigger a re-entry into PRIMARY within the same
    dispatch. Cross-trigger cascades happen on the next dispatch cycle.

    Gate and Contributor interventions ignore this field — they run in
    a single pass.
    """

    PRIMARY = 0
    POST_EFFECT = 1


# ──────────────────────────────────────────────────────────────────────
# Triggers — when an Intervention fires
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class OnTool:
    tool_name: str | None = None          # None = any tool
    phase: ToolPhase = ToolPhase.POST
    where: Callable[["ToolCallInfo"], bool] | None = None


@dataclass(frozen=True)
class OnLLM:
    phase: LLMPhase


@dataclass(frozen=True)
class OnEvent:
    event_type: EventType


@dataclass(frozen=True)
class OnTurn:
    phase: TurnPhase


@dataclass(frozen=True)
class OnFrame:
    phase: FramePhase


@dataclass(frozen=True)
class OnCondition:
    # Evaluated at every dispatch tick; keep the predicate cheap.
    predicate: Callable[["InterventionContext"], bool]


TriggerSpec = Union[OnTool, OnLLM, OnEvent, OnTurn, OnFrame, OnCondition]


# ──────────────────────────────────────────────────────────────────────
# Effects — what an Intervention's action may return
# ──────────────────────────────────────────────────────────────────────


@dataclass
class Block:
    """Refuse the upcoming action. Gate-only.

    A registry that sees any `Block` in a pre-hook result must
    short-circuit: skip execution and treat the action as an error whose
    message is `reason`. `trace_event` is an optional TRACE event to
    emit so dashboards can explain the block.
    """

    reason: str
    trace_event: EventType | None = None


@dataclass
class ModifyToolInput:
    """Merge `patch` into the pending tool_input dict before execution."""

    patch: dict[str, Any]


@dataclass
class ModifyToolResult:
    """Adjust the tool result after execution but before TOOL_COMPLETED.

    Either overrides the output/is_error entirely, or appends a hint
    string to the existing error message (edit_guards.fuzzy_suggest
    use case).
    """

    output: str | None = None
    is_error: bool | None = None
    append_hint: str | None = None


@dataclass
class AppendFooter:
    """Contribute a footer hint for the upcoming LLM request."""

    text: str
    source_id: str


@dataclass
class AppendInstructions:
    """Contribute extra instructions to the role's system prompt."""

    text: str
    source_id: str


@dataclass
class InjectUserMessage:
    """One-shot user message injected on the next turn only.

    `ttl` is how many turns this hint survives before being retired.
    Default `1` = "next turn only", matching the typical
    "after-Edit-failure, re-read before retrying" pattern.
    """

    text: str
    source_id: str
    ttl: int = 1


@dataclass
class SwapModel:
    """Replace the frame's active model. Used for escalation/de-escalation."""

    new_model: str
    reason: str = ""


@dataclass
class UpdateFrameField:
    """Mutate a field on the frame.

    `path` is a dot-separated accessor (e.g. `ledger.files_confirmed`).
    `mode` picks the semantics: overwrite, append to list, or merge
    into a dict.
    """

    path: str
    value: Any
    mode: Literal["set", "append", "merge"] = "set"


@dataclass
class EmitEvent:
    """Append an Event to the store. Payload is a Pydantic model instance."""

    event_type: EventType
    payload: Any


Effect = Union[
    Block,
    ModifyToolInput,
    ModifyToolResult,
    AppendFooter,
    AppendInstructions,
    InjectUserMessage,
    SwapModel,
    UpdateFrameField,
    EmitEvent,
]


# ──────────────────────────────────────────────────────────────────────
# Context objects passed to Intervention actions
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ToolCallInfo:
    """A snapshot of one tool call at pre- or post-phase."""

    tool_name: str
    tool_use_id: str
    tool_input: dict[str, Any]
    phase: ToolPhase
    result_output: str | None = None
    result_is_error: bool | None = None


@dataclass
class InterventionContext:
    """All context an Intervention action may read.

    Which fields are populated depends on the trigger kind (see the
    field comments). Interventions should treat `frame`, if present, as
    a read-only snapshot — mutations are only legitimate via returned
    Effects. Python can't enforce immutability, so mutating the object
    still propagates, but no part of the framework depends on that path.
    """

    session_id: str
    now: float
    registry: "PackRegistry | None" = None
    # frame-aware dispatch (tool calls, frame lifecycle, on_event)
    frame: "Frame | None" = None
    event: Event | None = None
    tool_call: ToolCallInfo | None = None
    # POST_EFFECT phase listeners only — populated by the registry with
    # the list of effects PRIMARY phase listeners returned for the same
    # trigger. PRIMARY phase listeners always see an empty list here.
    # See pack_architecture.md §4.5.
    primary_effects: list["Effect"] = field(default_factory=list)
    # compose-time dispatch (contributor fired from ContextComposer).
    # Populated when the caller only has a bare Context, no Frame.
    body: "ContextBody | None" = None
    header: "ContextHeader | None" = None
    self_actor: str = ""


# ──────────────────────────────────────────────────────────────────────
# Intervention — the (B) side
# ──────────────────────────────────────────────────────────────────────


InterventionAction = Callable[
    [InterventionContext],
    "Awaitable[list[Effect]] | list[Effect]",
]


@dataclass
class Intervention:
    """A framework behavior that reacts to a specific trigger.

    The action is a (sync or async) function that takes an
    `InterventionContext` and returns a list of `Effect` objects. The
    registry owns effect application; interventions are pure in the
    sense that "same input → same effects".

    `phase` is meaningful only for `kind="listener"` — see
    InterventionPhase. Gates and Contributors ignore it.
    """

    id: str                              # globally unique: "edit_guards.fuzzy_suggest"
    kind: InterventionKind
    trigger: TriggerSpec
    action: InterventionAction
    phase: InterventionPhase = InterventionPhase.PRIMARY
    description: str = ""
    default_enabled: bool = True


# ──────────────────────────────────────────────────────────────────────
# Capability — the grouping unit
# ──────────────────────────────────────────────────────────────────────


@dataclass
class Capability:
    """A coherent feature bundling Tools and Interventions."""

    name: str
    description: str = ""
    tools: list[Tool] = field(default_factory=list)
    interventions: list[Intervention] = field(default_factory=list)
    event_types: list[EventType] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────
# Pack — the installable outer container
# ──────────────────────────────────────────────────────────────────────


@dataclass
class PackMeta:
    name: str
    version: str
    description: str = ""
    depends_on: list[str] = field(default_factory=list)
    provides_events: list[str] = field(default_factory=list)


class Pack(Protocol):
    """An installable unit providing one or more capabilities."""

    meta: PackMeta
    capabilities: list[Capability]

    def on_install(self, registry: "PackRegistry") -> None: ...

    def on_uninstall(self, registry: "PackRegistry") -> None: ...


__all__ = [
    # phases / enums
    "ToolPhase",
    "LLMPhase",
    "TurnPhase",
    "FramePhase",
    "InterventionKind",
    "InterventionPhase",
    # triggers
    "OnTool",
    "OnLLM",
    "OnEvent",
    "OnTurn",
    "OnFrame",
    "OnCondition",
    "TriggerSpec",
    # effects
    "Block",
    "ModifyToolInput",
    "ModifyToolResult",
    "AppendFooter",
    "AppendInstructions",
    "InjectUserMessage",
    "SwapModel",
    "UpdateFrameField",
    "EmitEvent",
    "Effect",
    # context
    "ToolCallInfo",
    "InterventionContext",
    # objects
    "Intervention",
    "InterventionAction",
    "Capability",
    "PackMeta",
    "Pack",
    "Tool",
]
