"""Session view model — the rendering DTO for dashboard clients.

The frontend should never rebuild session state from raw events. It
receives a `SessionViewDto` and just renders it. This module owns the
transformation from the append-only event log into a structured turn
tree, so the shape of what the UI paints is a backend concern.

Conceptual model (the "git squash merge" framing the user asked for):

- The main conversation with the root frame is a sequence of **turns**.
- A turn is one user message + everything that happened on the root
  frame until the next user message (or "now" if still running).
- Each turn has one **final reply** — the last assistant message in the
  turn. Everything else (intermediate assistant messages, tool calls,
  sub-agent delegations) is a **step** inside the turn.
- Sub-agent delegations are the recursive bit: when a tool call spawns
  a child frame, that child's own turns get nested inside the parent's
  step list as a `sub_agent` step. Clients render them the same way
  they render top-level turns — collapsed summary by default, expand
  for details.
- Pulse (live activity indicator) is derived once on the backend, not
  reconstructed by the client.
"""

from __future__ import annotations

# Import in dependency order so model_rebuild() sees all forward refs.
from nature.server.view.messages import AnnotationDto, HintDto, MessageDto
from nature.server.view.tools import ReceivedDto, ToolDto, ToolResultBlockDto
from nature.server.view.turns import TurnDto, TurnSummaryDto
from nature.server.view.steps import StepDto, SubAgentDto
from nature.server.view.session import PulseDto, SessionViewDto
from nature.server.view.build import (
    build_session_view,
    _extract_text,
    _message_dto_from_event,
    _latest_todos_from_events,
    _frame_state_from_events,
    _compute_turn_summary,
    _aggregate_sub_agent_summary,
    _derive_pulse,
    _build_turns_for_frame,
)

# Resolve forward references now that all classes are in scope.
# SubAgentDto.turns: list[TurnDto]  — defined in steps.py, resolved here.
# StepDto.sub_agent: SubAgentDto | None — same module, still needs rebuild
# after TurnDto is concrete.
ToolResultBlockDto.model_rebuild()
ReceivedDto.model_rebuild()
SubAgentDto.model_rebuild()
StepDto.model_rebuild()
TurnDto.model_rebuild()

__all__ = [
    # messages
    "AnnotationDto",
    "HintDto",
    "MessageDto",
    # tools
    "ToolDto",
    "ToolResultBlockDto",
    "ReceivedDto",
    # steps
    "SubAgentDto",
    "StepDto",
    # turns
    "TurnSummaryDto",
    "TurnDto",
    # session
    "PulseDto",
    "SessionViewDto",
    # build
    "build_session_view",
    "_extract_text",
    "_message_dto_from_event",
    "_latest_todos_from_events",
    "_frame_state_from_events",
    "_compute_turn_summary",
    "_aggregate_sub_agent_summary",
    "_derive_pulse",
    "_build_turns_for_frame",
]
