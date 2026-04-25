"""LLM Request/Response — the canonical interface for all LLM interactions.

Every LLM call is a Request → Response pair. Both are first-class objects.
This enables:
- Agent-to-agent communication inspection and control
- Hook interception of requests/responses
- Dashboard full request/response visibility
- Logging, replay, and debugging
- Orchestrator routing between agents
"""

from __future__ import annotations

from dataclasses import field
from typing import Any

from pydantic import BaseModel, Field

from nature.protocols.message import Message, Usage
from nature.protocols.tool import ToolDefinition


class LLMRequest(BaseModel):
    """Complete request going to an LLM.

    This is the single object that captures everything about
    an LLM call. Providers convert this to their native format.
    """
    # Conversation
    messages: list[Message]
    system: list[str] = Field(default_factory=list)
    tools: list[ToolDefinition] | None = None

    # Generation config
    model: str | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None

    # Caching
    cache_control: dict[str, str] | None = None  # {"type": "ephemeral", "ttl": "5m"}

    # Metadata (for tracking, not sent to API)
    request_id: str | None = None
    source: str | None = None  # "main_loop", "sub_agent", "autocompact", etc.
    turn_number: int | None = None

    @property
    def last_user_message(self) -> str:
        """Extract the last user message text (for logging)."""
        for msg in reversed(self.messages):
            if msg.role.value == "user" and msg.text:
                return msg.text[:200]
        return ""

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def tool_names(self) -> list[str]:
        return [t.name for t in self.tools] if self.tools else []


class LLMResponse(BaseModel):
    """Complete response from an LLM.

    Built from accumulated stream events. Captures the full
    result of a single LLM call.
    """
    # Content
    message: Message  # The assistant message (text + tool_use blocks)
    text: str = ""  # Extracted text content (convenience)

    # Metadata
    usage: Usage = Field(default_factory=Usage)
    stop_reason: str | None = None
    model: str | None = None

    # Tracking
    request_id: str | None = None  # Matches the request
    duration_ms: int | None = None

    @property
    def has_tool_use(self) -> bool:
        return self.message.has_tool_use

    @property
    def tool_use_blocks(self):
        return self.message.tool_use_blocks
