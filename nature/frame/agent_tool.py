"""AgentTool — the delegation surface for the LLM.

From the LLM's perspective this looks like a normal tool: "Agent(name,
prompt)". The AreaManager intercepts calls to this tool and handles
them via open_child + run, so the tool's own `run` method is never
actually invoked in the new execution path. If it IS called, that's a
bug (AreaManager didn't intercept) and the tool returns an error.

This is the new-architecture replacement for nature/multi/agent_tool.py,
which still exists for the legacy query() path during the refactor.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from nature.protocols.tool import ToolContext, ToolResult
from nature.tools.base import BaseTool


class AgentInput(BaseModel):
    """Arguments the LLM provides when delegating.

    Both `prompt` and `name` are REQUIRED. `name` used to default to
    `"core"`, but that silently routed unspecified delegations into
    self-loops when the calling agent was already core (session
    `8ed7d997`). The framework now requires the caller to be explicit
    about which specialist they want — pick from the table the role
    instructions provide.
    """

    prompt: str = Field(description="Task for the sub-agent to perform")
    name: str = Field(
        description=(
            "Which agent profile to delegate to. REQUIRED — no default. "
            "Resolved against the profile registry (researcher / analyzer "
            "/ implementer / reviewer / judge / core / receptionist). "
            "You cannot delegate to yourself; pick a specialist that "
            "actually advances the task."
        ),
    )


class AgentTool(BaseTool):
    """Delegation schema holder — calls are intercepted by AreaManager."""

    input_model = AgentInput

    @property
    def name(self) -> str:
        return "Agent"

    @property
    def description(self) -> str:
        return (
            "Delegate a task to a specialized sub-agent. The sub-agent "
            "starts with a fresh context scoped to this task, uses only "
            "the tools its role allows, and returns its final answer. "
            "Use for: focused research, heavy analysis, implementation "
            "work that would pollute the current context."
        )

    def is_read_only(self, input: dict[str, Any]) -> bool:
        return False

    def is_concurrency_safe(self, input: dict[str, Any]) -> bool:
        return False

    async def run(self, params: AgentInput, context: ToolContext) -> ToolResult:
        return ToolResult(
            output=(
                "Agent tool was not intercepted by AreaManager. This is a "
                "bug: delegation should never reach the tool's execute path."
            ),
            is_error=True,
        )
