from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from nature.protocols.todo import TodoItem
from nature.server.view.messages import MessageDto
from nature.server.view.tools import ReceivedDto, ToolDto

if TYPE_CHECKING:
    from nature.server.view.turns import TurnDto, TurnSummaryDto


class SubAgentDto(BaseModel):
    """A sub-agent delegation — the child frame's work, wrapped so a
    parent turn can include it as a single step.

    Children are recursive: their own `turns` may contain more
    `sub_agent` steps for grand-children.

    `returned_text` / `returned_message_id` mirror the child frame's
    last assistant reply — i.e., what this delegation handed back to
    the parent as the Agent tool's result. Surfacing it on the
    sub_agent DTO itself lets the UI show a "⎿ <preview>" row on the
    card without the client having to dig into the nested turns.

    `todos` is the child frame's current TodoWrite list (as of the
    most recent TODO_WRITTEN event on that frame). Empty when the
    sub-agent never called TodoWrite.
    """

    frame_id: str
    role_name: str
    purpose: str = ""
    state: str  # "active" | "resolved" | "closed" | "error"
    spawned_by_tool_use_id: str | None = None
    turns: list[TurnDto] = Field(default_factory=list)
    summary: TurnSummaryDto
    started_at: float
    ended_at: float | None = None
    returned_text: str | None = None
    returned_message_id: str | None = None
    todos: list[TodoItem] = Field(default_factory=list)


class StepDto(BaseModel):
    """One unit of work inside a turn. Discriminated on `kind`.

    - `kind="message"`: an intermediate assistant message (not the final
      reply — the final is promoted to `turn.final_message`).
    - `kind="tool"`: a regular (non-delegating) tool call.
    - `kind="sub_agent"`: an Agent delegation that spawned a child frame.
      Its detail lives in `sub_agent`; the raw tool call on the parent
      is not rendered separately.
    - `kind="received"`: the bundled tool_result message delivering
      results back to this frame's agent. Marks the "aggregation
      point" in the timeline — right before the agent's next LLM call
      synthesizes a response from the collected results.

    `parallel_group_id` is populated (non-null) when this step was
    executed as part of a parallel batch. Two consecutive steps with
    the same `parallel_group_id` ran concurrently via
    `asyncio.gather`; the dashboard uses this to render them with a
    shared visual marker ("∥" left border + accent color). `None`
    means the step ran on its own — the overwhelming majority of
    steps in a typical session.
    """

    kind: Literal["message", "tool", "sub_agent", "received"]
    id: str
    timestamp: float

    message: MessageDto | None = None
    tool: ToolDto | None = None
    sub_agent: SubAgentDto | None = None
    received: ReceivedDto | None = None
    parallel_group_id: str | None = None
