from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolDto(BaseModel):
    """One tool call — started + completed collapsed into a single record."""

    tool_use_id: str
    tool_name: str
    tool_input: dict[str, Any] = Field(default_factory=dict)
    started_at: float
    # Null while the tool is still running
    output: str | None = None
    is_error: bool | None = None
    duration_ms: int | None = None
    completed_at: float | None = None

    @property
    def is_done(self) -> bool:
        return self.completed_at is not None


class ToolResultBlockDto(BaseModel):
    """One tool_result content block — what a single tool call produced,
    plucked out of the bundled tool_result message for easy rendering."""

    tool_use_id: str
    content: str  # Text form; raw structured content kept in raw_content
    is_error: bool = False
    raw_content: Any = None


class ReceivedDto(BaseModel):
    """The bundled tool_result message (from_="tool") that delivers one
    or more tool_result blocks back to a parent agent after its
    tool_use calls complete.

    This is the timeline marker where a parent receives the results of
    the tools it dispatched — both plain tool calls and Agent
    delegations. Surfacing it as a first-class step makes "where
    aggregation starts" visually explicit instead of hiding it behind
    the `from_="tool"` filter.
    """

    message_id: str
    timestamp: float
    results: list[ToolResultBlockDto] = Field(default_factory=list)
