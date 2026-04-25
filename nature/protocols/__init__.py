"""Core protocols (ABCs) and data models for nature."""

from nature.protocols.message import (
    ContentBlock,
    ImageContent,
    Message,
    Role,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolResultContent,
    ToolUseContent,
    Usage,
)
from nature.protocols.llm import LLMRequest, LLMResponse
from nature.protocols.turn import Action, ActionType, AgentTurn, Observation
from nature.protocols.provider import LLMProvider, ProviderConfig
from nature.protocols.tool import (
    PermissionResult,
    Tool,
    ToolContext,
    ToolDefinition,
    ToolResult,
)

__all__ = [
    "ContentBlock",
    "ImageContent",
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "Message",
    "PermissionResult",
    "ProviderConfig",
    "Role",
    "StreamEvent",
    "StreamEventType",
    "TextContent",
    "Tool",
    "ToolContext",
    "ToolDefinition",
    "ToolResult",
    "ToolResultContent",
    "ToolUseContent",
    "Usage",
]
