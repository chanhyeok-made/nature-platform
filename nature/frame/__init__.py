"""Frame — the stateful container that holds a Context and drives execution.

An agent is a pure function. A Frame is the scope that owns a Context,
calls the agent, applies its output, executes tools, and decides when
to resolve. Frames nest: a parent frame opens a child frame when its
agent wants to delegate, and the child bubbles its final message back
to the parent on resolution.

Execution writes to an EventStore only — it never touches UI code. UIs
consume from the store through live_tail / snapshot.
"""

from nature.frame.agent_tool import AgentInput, AgentTool
from nature.frame.frame import Frame, FrameState
from nature.frame.manager import AreaManager, RoleResolver

__all__ = [
    "AgentInput",
    "AgentTool",
    "AreaManager",
    "Frame",
    "FrameState",
    "RoleResolver",
]
