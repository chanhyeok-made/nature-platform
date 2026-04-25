"""Core message types for the nature framework.

These Pydantic models define the universal message format used across all providers,
tools, and the agent loop. Internal code never touches provider-specific types directly.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal, Union
from uuid import uuid4

from nature.config.types import ToolInput

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Role(str, Enum):
    """Message role in the conversation."""
    USER = "user"
    ASSISTANT = "assistant"


class StreamEventType(str, Enum):
    """Types of streaming events from the LLM provider."""
    MESSAGE_START = "message_start"
    CONTENT_BLOCK_START = "content_block_start"
    CONTENT_BLOCK_DELTA = "content_block_delta"
    CONTENT_BLOCK_STOP = "content_block_stop"
    MESSAGE_DELTA = "message_delta"
    MESSAGE_STOP = "message_stop"


# ---------------------------------------------------------------------------
# Content blocks
# ---------------------------------------------------------------------------

class TextContent(BaseModel):
    """A text content block."""
    type: Literal["text"] = "text"
    text: str


class ToolUseContent(BaseModel):
    """A tool_use content block — the model is requesting a tool call."""
    type: Literal["tool_use"] = "tool_use"
    id: str = Field(default_factory=lambda: f"toolu_{uuid4().hex[:24]}")
    name: str
    input: ToolInput = Field(default_factory=dict)


class ToolResultContent(BaseModel):
    """A tool_result content block — the result of a tool call."""
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str | list[TextContent | ImageContent] = ""
    is_error: bool = False


class ImageContent(BaseModel):
    """An image content block."""
    type: Literal["image"] = "image"
    source: ImageSource


class ImageSource(BaseModel):
    """Image source data."""
    type: Literal["base64"] = "base64"
    media_type: str  # e.g. "image/png"
    data: str


class ThinkingContent(BaseModel):
    """Extended thinking content block."""
    type: Literal["thinking"] = "thinking"
    thinking: str


ContentBlock = Annotated[
    Union[TextContent, ToolUseContent, ToolResultContent, ImageContent, ThinkingContent],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------

class Message(BaseModel):
    """A single message in the conversation.

    This is the universal message format. Provider adapters convert their
    native types to/from this format at the boundary.
    """
    role: Role
    content: list[ContentBlock] = Field(default_factory=list)

    @classmethod
    def user(cls, text: str) -> Message:
        """Create a user message with text content."""
        return cls(role=Role.USER, content=[TextContent(text=text)])

    @classmethod
    def assistant(cls, text: str) -> Message:
        """Create an assistant message with text content."""
        return cls(role=Role.ASSISTANT, content=[TextContent(text=text)])

    @classmethod
    def tool_result(cls, tool_use_id: str, content: str, is_error: bool = False) -> Message:
        """Create a user message containing a tool result."""
        return cls(
            role=Role.USER,
            content=[ToolResultContent(
                tool_use_id=tool_use_id,
                content=content,
                is_error=is_error,
            )],
        )

    @property
    def text(self) -> str:
        """Extract concatenated text from all text blocks."""
        return "".join(
            block.text for block in self.content if isinstance(block, TextContent)
        )

    @property
    def tool_use_blocks(self) -> list[ToolUseContent]:
        """Extract all tool_use blocks."""
        return [b for b in self.content if isinstance(b, ToolUseContent)]

    @property
    def has_tool_use(self) -> bool:
        """Whether this message contains tool_use blocks."""
        return any(isinstance(b, ToolUseContent) for b in self.content)


# ---------------------------------------------------------------------------
# Token usage
# ---------------------------------------------------------------------------

class Usage(BaseModel):
    """Token usage from an API response."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# ---------------------------------------------------------------------------
# Stream events
# ---------------------------------------------------------------------------

class StreamEvent(BaseModel):
    """A streaming event from the LLM provider.

    The agent loop consumes these to progressively build up the assistant
    message and detect tool_use blocks for concurrent execution.
    """
    type: StreamEventType
    index: int | None = None  # content block index
    delta_text: str | None = None  # for content_block_delta
    delta_tool_input: str | None = None  # for tool input streaming
    content_block: ContentBlock | None = None  # for content_block_start
    usage: Usage | None = None  # for message_delta/message_stop
    stop_reason: str | None = None  # for message_delta
    message: Message | None = None  # for message_start (full initial message)
