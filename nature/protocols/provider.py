"""LLM Provider protocol — the port/adapter boundary for LLM APIs.

Every provider implements this ABC. The canonical interface uses
LLMRequest/LLMResponse for all interactions. Providers convert
these to their native format at the boundary.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncGenerator

from pydantic import BaseModel, Field

from nature.protocols.message import Message, StreamEvent, Usage
from nature.protocols.tool import ToolDefinition


class CacheControl(BaseModel):
    """Prompt caching configuration for a system prompt block."""
    type: str = "ephemeral"
    ttl: str | None = None


class ProviderConfig(BaseModel):
    """Configuration for an LLM provider."""
    model: str = "claude-sonnet-4-20250514"
    api_key: str | None = None
    base_url: str | None = None
    max_output_tokens: int = 8192
    temperature: float | None = None
    top_p: float | None = None
    extra_headers: dict[str, str] = Field(default_factory=dict)
    extra_body: dict = Field(default_factory=dict)
    # Short host identifier (e.g. "openrouter", "anthropic",
    # "local-ollama"). When set, the provider composes
    # `host_name::model` for capability lookups and clips outgoing
    # max_tokens to the model's physical ceiling. None = trust the
    # caller / provider default. Callers that know the host set this
    # so model-specific caps take effect without extra plumbing.
    host_name: str | None = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    Two ways to call:
    1. stream_request(LLMRequest) — canonical, used by agent loop
    2. stream(messages, system, ...) — convenience, delegates to stream_request
    """

    async def stream_request(
        self,
        request: "LLMRequest",
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream a response from an LLMRequest.

        Default implementation unpacks the request and calls stream(),
        passing `request.model` through so per-request model selection
        (e.g., per-agent models in frame mode) actually takes effect.
        Override this only if you need custom request handling.
        """
        from nature.protocols.llm import LLMRequest

        cache = None
        if request.cache_control:
            cache = CacheControl(**request.cache_control)

        async for event in self.stream(
            messages=request.messages,
            system=request.system,
            tools=request.tools,
            model=request.model,
            max_output_tokens=request.max_output_tokens,
            cache_control=cache,
        ):
            yield event

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        system: list[str],
        tools: list[ToolDefinition] | None = None,
        *,
        model: str | None = None,
        max_output_tokens: int | None = None,
        cache_control: CacheControl | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream a response from the LLM.

        `model` (when set) overrides the model the provider was
        constructed with. This enables per-request / per-agent model
        selection without swapping provider instances.
        """
        ...
        if False:
            yield  # type: ignore[misc]

    async def count_tokens_for_request(self, request: "LLMRequest") -> int:
        """Count tokens for an LLMRequest."""
        return await self.count_tokens(
            messages=request.messages,
            system=request.system,
            tools=request.tools,
        )

    @abstractmethod
    async def count_tokens(
        self,
        messages: list[Message],
        system: list[str],
        tools: list[ToolDefinition] | None = None,
    ) -> int:
        """Count tokens (exact or estimated)."""
        ...

    @property
    @abstractmethod
    def model_id(self) -> str:
        ...

    @property
    def context_window(self) -> int:
        return 200_000

    @property
    def supports_caching(self) -> bool:
        return False

    async def close(self) -> None:
        """Clean up provider resources."""
