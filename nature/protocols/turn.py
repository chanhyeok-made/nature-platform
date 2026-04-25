"""AgentTurn — the structured output of one LLM interaction.

Every LLM response is parsed into this canonical form.
This is what agents produce and consume — the universal language
for agent-to-agent communication.

An LLM response naturally contains:
1. Observations — reasoning, findings, commentary (text)
2. Actions — things to do (tool calls, delegation, user questions)

Both can occur multiple times and in any order. The parser
extracts them into a structured, inspectable object.

Example LLM output:
    "Let me check the code structure."        → Observation
    [tool_use: Glob(pattern="**/*.py")]        → Action(tool_call)
    [tool_use: Read(file_path="main.py")]      → Action(tool_call)
    "Found the bug on line 42. Fixing now."    → Observation
    [tool_use: Edit(file_path="main.py", ...)] → Action(tool_call)
    "The fix is complete."                     → Observation

Parsed into:
    AgentTurn(
        observations=["Let me check...", "Found the bug...", "The fix is complete."],
        actions=[Glob(...), Read(...), Edit(...)],
        is_complete=True,
    )
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from nature.config.types import ToolInput


class ActionType(str, Enum):
    """Types of actions an agent can take."""
    TOOL_CALL = "tool_call"          # Execute a tool
    DELEGATE = "delegate"            # Ask another agent to do something
    ASK_USER = "ask_user"            # Need user input to proceed
    COMPLETE = "complete"            # Task is done


class Observation(BaseModel):
    """A piece of reasoning, finding, or commentary.

    Preserved in order — the sequence of observations tells
    the story of the agent's thought process.
    """
    text: str
    index: int = 0  # Position in the original response


class Action(BaseModel):
    """Something the agent wants to do.

    Actions are extracted from tool_use blocks, but the interface
    is general enough for delegation and user questions.
    """
    type: ActionType
    tool_name: str | None = None
    tool_input: ToolInput = Field(default_factory=dict)
    tool_use_id: str | None = None

    # For delegation
    target_agent: str | None = None
    prompt: str | None = None

    # For ask_user
    question: str | None = None

    # Result (filled after execution)
    result: str | None = None
    is_error: bool = False
    executed: bool = False


class AgentTurn(BaseModel):
    """The structured output of one LLM interaction.

    This is the canonical interface that agents produce and consume.
    Other agents inspect this, not raw text or tool_use blocks.
    """
    # Content — what the agent communicated
    observations: list[Observation] = Field(default_factory=list)
    actions: list[Action] = Field(default_factory=list)

    # Status
    is_complete: bool = False  # No more actions needed
    needs_input: bool = False  # Waiting for user/agent input

    # Metadata
    summary: str | None = None  # Brief summary for parent agents
    turn_number: int = 0
    request_id: str | None = None

    @property
    def has_actions(self) -> bool:
        return len(self.actions) > 0

    @property
    def tool_calls(self) -> list[Action]:
        return [a for a in self.actions if a.type == ActionType.TOOL_CALL]

    @property
    def pending_actions(self) -> list[Action]:
        return [a for a in self.actions if not a.executed]

    @property
    def full_text(self) -> str:
        """All observations concatenated."""
        return "\n".join(o.text for o in self.observations)

    def add_result(self, tool_use_id: str, result: str, is_error: bool = False) -> None:
        """Record the result of an action."""
        for action in self.actions:
            if action.tool_use_id == tool_use_id:
                action.result = result
                action.is_error = is_error
                action.executed = True
                break
