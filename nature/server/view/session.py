from __future__ import annotations

from pydantic import BaseModel, Field

from nature.protocols.todo import TodoItem
from nature.server.view.turns import TurnDto


class PulseDto(BaseModel):
    """Live activity indicator state, derived on the backend.

    `active` flips true as soon as a user message triggers an llm/tool
    and back to false when everything in-flight settles.
    """

    active: bool = False
    activity: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    started_at: float | None = None


class SessionViewDto(BaseModel):
    """Top-level view of a session — the payload the dashboard renders."""

    session_id: str
    role_name: str = ""
    model: str = ""
    provider: str = ""
    state: str = "active"  # "active" | "resolved" | "closed" | "error"
    turns: list[TurnDto] = Field(default_factory=list)
    pulse: PulseDto = Field(default_factory=PulseDto)
    # Root frame's current TodoWrite list (as of the latest TODO_WRITTEN
    # event on the root frame). Child frames' todo lists live on their
    # SubAgentDto. Empty when the root agent never called TodoWrite.
    root_todos: list[TodoItem] = Field(default_factory=list)
