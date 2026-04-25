from __future__ import annotations

from pydantic import BaseModel, Field

from nature.server.view.messages import MessageDto
from nature.server.view.steps import StepDto


class TurnSummaryDto(BaseModel):
    """Pre-computed one-liner numbers for the collapsed turn state."""

    step_count: int = 0
    tool_count: int = 0
    sub_agent_count: int = 0
    received_count: int = 0
    duration_ms: int | None = None


class TurnDto(BaseModel):
    """One turn on a frame's main conversation.

    `user_message` is the opening user input, `final_message` is the
    last assistant reply in the turn (may be null while running),
    `steps` is everything in between in chronological order.

    `first_event_id` / `last_event_id` expose the session-monotonic
    event id range this turn spans — used by the dashboard's fork
    button to call `POST /api/sessions/{sid}/fork` with a concrete
    `at_event_id` without the client having to scan the raw event
    log. `first_event_id` is the id of the incoming user message
    that opened the turn; `last_event_id` grows as new events land
    on the turn (any `ev.id` greater than the current max updates
    it). For still-running turns both ids stay live — the next
    event that lands will bump `last_event_id`.
    """

    id: str  # derived from the opening user message event id
    state: str  # "running" | "resolved" | "error"
    started_at: float
    ended_at: float | None = None
    user_message: MessageDto
    final_message: MessageDto | None = None
    steps: list[StepDto] = Field(default_factory=list)
    summary: TurnSummaryDto = Field(default_factory=TurnSummaryDto)
    first_event_id: int = 0
    last_event_id: int = 0
