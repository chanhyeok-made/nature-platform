"""AgentOutput and Signal — the return types of llm_agent().

These types describe what a single agent invocation produced, without any
reference to state or side effects. The caller (AreaManager in Step 4)
decides what to do next based on `signal`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from nature.context.conversation import Message, MessageAnnotation
from nature.context.footer import Hint
from nature.protocols.llm import LLMRequest
from nature.protocols.message import Usage
from nature.protocols.turn import Action


class Signal(str, Enum):
    """Control signal from an agent call — tells the caller what to do next.

    CONTINUE:   output requires follow-up (tool calls pending, or token
                limit hit). Caller should execute pending actions and
                re-invoke llm_agent.
    RESOLVED:   agent considers the current task done. Caller should
                close the frame and bubble result upward.
    NEEDS_USER: agent is waiting for user input. Caller should yield
                control back to whoever drives the session.
    DELEGATE:   agent wants to spawn a sub-frame. Caller should open a
                child frame with the delegation request. (Used in Step 5
                once AreaManager exists.)
    ERROR:      unrecoverable error surfaced from the provider.
    """

    CONTINUE = "continue"
    RESOLVED = "resolved"
    NEEDS_USER = "needs_user"
    DELEGATE = "delegate"
    ERROR = "error"


@dataclass
class AgentOutput:
    """Everything a single llm_agent() invocation produces.

    Pure data — no agent state references, no side effects. The caller
    applies these changes to a Frame's Context as it sees fit.
    """

    new_messages: list[Message]
    actions: list[Action]
    annotations: list[MessageAnnotation] = field(default_factory=list)
    signal: Signal = Signal.CONTINUE
    stop_reason: str | None = None
    usage: Usage | None = None
    raw_request: LLMRequest | None = None  # kept for replay / debugging
    # Footer hints that were appended to the LLM request just sent.
    # The caller (FrameManager) emits a HINT_INJECTED trace event with
    # the same request_id so the dashboard can show "the framework
    # whispered something here". Empty when no rule fired.
    hints: list[Hint] = field(default_factory=list)
