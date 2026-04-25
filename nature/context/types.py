"""Static context primitives — role, principles, header/body split.

The header/body split is the single biggest simplification of this
refactor: header (role + principles) is static/cacheable identity,
body (conversation) is the growing dialogue. Cache boundaries follow
this split cleanly, replacing the old DYNAMIC_BOUNDARY string marker.

See conversation.py for Message, Conversation, MessageAnnotation.
See composer.py for Context → LLMRequest serialization.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from nature.context.conversation import Conversation
from nature.protocols.todo import TodoItem


class AgentRole(BaseModel):
    """Identity and capabilities of an agent.

    The static part of context: who you are, how you're told to behave,
    which tools you're allowed to touch, and optionally which model
    should run you. Roles are swappable at runtime to change an
    agent's identity without discarding its conversation.

    `allowed_tools` is the single source of truth for tool gating —
    both what gets prompted to the LLM and what's executable. None = all
    tools in the registry; a list = filter to those names.

    `model` is an optional per-role model override. When set,
    AreaManager uses it for child frames delegated into this role.
    When None, the child inherits the parent frame's model (or the
    runner default).
    """

    name: str
    description: str = ""
    instructions: str
    allowed_tools: list[str] | None = None
    model: str | None = None


class BasePrincipleSource(str, Enum):
    """Where a principle came from — used for debugging and policy."""

    FRAMEWORK = "framework"   # Hardcoded by nature
    PROJECT = "project"       # From .nature/principles/*.yaml
    USER = "user"             # From ~/.nature/principles/*.yaml
    RUNTIME = "runtime"       # Added during execution


class BasePrinciple(BaseModel):
    """A behavior rule governing all of an agent's actions.

    Each principle records its source and priority so conflicts can be
    resolved and debugging can trace "why is this in the prompt?".
    """

    text: str
    source: BasePrincipleSource = BasePrincipleSource.RUNTIME
    priority: int = 0  # Higher = stronger


class ContextHeader(BaseModel):
    """Static, cacheable identity half of a Context."""

    role: AgentRole
    principles: list[BasePrinciple] = Field(default_factory=list)


class ContextBody(BaseModel):
    """Growing, uncacheable dialogue half of a Context.

    Holds the conversation log plus any other per-frame mutable state
    that the agent has accumulated — currently a todo list (set by
    the TodoWrite tool, replayed from TODO_WRITTEN events). New
    per-frame state types should land here so reconstruct() picks
    them up automatically.
    """

    conversation: Conversation = Field(default_factory=Conversation)
    todos: list[TodoItem] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}


class Context(BaseModel):
    """Full context an agent sees when called.

    Header holds who-and-how (static, cacheable). Body holds what-happened
    (dynamic, uncacheable). Compression strategies may touch body only;
    header is inviolable.
    """

    header: ContextHeader
    body: ContextBody = Field(default_factory=ContextBody)

    model_config = {"arbitrary_types_allowed": True}

    def with_role(self, role: AgentRole) -> "Context":
        """Return a copy with a swapped role — same conversation, new identity."""
        new_header = self.header.model_copy(update={"role": role})
        return self.model_copy(update={"header": new_header})

    def with_principle(self, principle: BasePrinciple) -> "Context":
        """Return a copy with an added principle."""
        new_principles = [*self.header.principles, principle]
        new_header = self.header.model_copy(update={"principles": new_principles})
        return self.model_copy(update={"header": new_header})
