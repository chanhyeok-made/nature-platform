"""Domain conversation primitives — pure I/O records.

A Conversation is an ordered list of Messages, each a directed I/O event
between two actors. Actors are named strings ("user", "receptionist",
"core", "tool:Bash", ...). All auxiliary info (thinking, token usage,
timing) lives in a parallel MessageAnnotation keyed by message_id so the
conversation itself stays a pure log of who-said-what.

Naming note: `Message` here is the DOMAIN message (from_/to). It is
DISTINCT from `nature.protocols.message.Message`, which is the LLM-API
boundary type (role/content). Files should import one or the other, and
use `ContextComposer` as the translation boundary when both perspectives
are needed.

Why a separate type: the API Message uses `role: user|assistant`, which
is a relative concept (user vs assistant depends on whose turn it is).
The domain Message uses absolute actor identity, which is what multi-
agent tracking, replay, and debugging actually need.
"""

from __future__ import annotations

from uuid import uuid4

from pydantic import BaseModel, Field

from nature.protocols.message import ContentBlock, Usage


class Message(BaseModel):
    """A single directed I/O event in a Conversation.

    `from_` / `to` are actor names. `content` is a list of ContentBlocks
    reused from protocols.message so text/tool_use/tool_result/image/
    thinking primitives stay canonical.

    Sub-agent delegation is modeled at the event level via
    `MessageAppendedPayload.delegations` (tool_use_id → child frame id)
    rather than an embedded conversation — keeping Message a pure I/O
    record and routing drill-down through ReplayResult.
    """

    id: str = Field(default_factory=lambda: f"msg_{uuid4().hex[:16]}")
    from_: str
    to: str
    content: list[ContentBlock] = Field(default_factory=list)
    timestamp: float

    model_config = {"arbitrary_types_allowed": True}


class Conversation(BaseModel):
    """An ordered list of I/O messages. Pure log — no metadata."""

    messages: list[Message] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

    def append(self, msg: Message) -> None:
        self.messages.append(msg)

    def extend(self, msgs: list[Message]) -> None:
        self.messages.extend(msgs)

    def __len__(self) -> int:
        return len(self.messages)


class MessageAnnotation(BaseModel):
    """Auxiliary data attached to a message by id.

    Stored outside Conversation so the log itself remains pure I/O.
    Populated from llm.response / tool.completed / annotation.stored
    events during replay.
    """

    message_id: str
    thinking: list[str] | None = None
    usage: Usage | None = None
    stop_reason: str | None = None
    llm_request_id: str | None = None
    duration_ms: int | None = None


