"""Context domain — header/body primitives + composer.

Public surface:
- AgentRole, BasePrinciple, BasePrincipleSource, ContextHeader, ContextBody, Context
- Message, Conversation, MessageAnnotation  (domain types — distinct from LLM API Message)
- ContextComposer  (Context → LLMRequest boundary)

Body-side compaction lives in `nature.context.body_compaction`. The
token estimator in `nature.context.estimator` is a pure utility used
by the compaction pipeline.
"""

from nature.context.composer import ContextComposer
from nature.context.conversation import (
    Conversation,
    Message,
    MessageAnnotation,
)
from nature.context.types import (
    AgentRole,
    BasePrinciple,
    BasePrincipleSource,
    Context,
    ContextBody,
    ContextHeader,
)

__all__ = [
    "AgentRole",
    "BasePrinciple",
    "BasePrincipleSource",
    "Context",
    "ContextBody",
    "ContextComposer",
    "ContextHeader",
    "Conversation",
    "Message",
    "MessageAnnotation",
]
