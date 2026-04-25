"""Event sourcing layer — the single source of truth for all execution.

Execution writes events here and nowhere else. UIs and replay read from here
and nothing else. This module has ZERO dependency on UI concerns.
"""

from nature.events.payloads import (
    AnnotationStoredPayload,
    BodyCompactedPayload,
    ErrorPayload,
    FrameClosedPayload,
    FrameErroredPayload,
    FrameOpenedPayload,
    FrameReopenedPayload,
    FrameResolvedPayload,
    HeaderSnapshotPayload,
    LLMErrorPayload,
    LLMRequestPayload,
    LLMResponsePayload,
    MessageAppendedPayload,
    PrincipleAddedPayload,
    RoleChangedPayload,
    ToolCompletedPayload,
    ToolStartedPayload,
    UserInputPayload,
    load_payload,
)
from nature.events.store import EventStore, FileEventStore, SessionMeta
from nature.events.types import (
    EVENT_CATEGORIES,
    Event,
    EventCategory,
    EventType,
    category_of,
    is_state_transition,
)

__all__ = [
    "Event",
    "EventType",
    "EventCategory",
    "EVENT_CATEGORIES",
    "category_of",
    "is_state_transition",
    "EventStore",
    "FileEventStore",
    "SessionMeta",
    # Typed payloads
    "FrameOpenedPayload",
    "FrameResolvedPayload",
    "FrameClosedPayload",
    "FrameErroredPayload",
    "FrameReopenedPayload",
    "MessageAppendedPayload",
    "HeaderSnapshotPayload",
    "BodyCompactedPayload",
    "AnnotationStoredPayload",
    "LLMRequestPayload",
    "LLMResponsePayload",
    "LLMErrorPayload",
    "ToolStartedPayload",
    "ToolCompletedPayload",
    "PrincipleAddedPayload",
    "RoleChangedPayload",
    "UserInputPayload",
    "ErrorPayload",
    "load_payload",
]
