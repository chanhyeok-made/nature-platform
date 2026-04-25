"""Shared test fakes — kept out of conftest.py so imports are explicit."""

from __future__ import annotations

from typing import Any, AsyncGenerator

from nature.protocols.message import (
    Message as LLMMessage,
    StreamEvent,
)
from nature.protocols.provider import CacheControl, LLMProvider
from nature.protocols.tool import (
    Tool,
    ToolContext,
    ToolDefinition,
    ToolResult,
)


class FakeProvider(LLMProvider):
    """Yields a fixed or scripted sequence of StreamEvents.

    Two modes:
    - Fixed: pass a `list[StreamEvent]` — every call replays the same
      events.
    - Scripted: pass a `list[list[StreamEvent]]` — each call yields the
      next sequence. When calls exceed scripted turns, the last sequence
      is replayed.

    `last_request` / `requests` capture LLMRequests so tests can assert
    on composition and per-turn behavior.
    """

    def __init__(
        self,
        events: list[StreamEvent] | list[list[StreamEvent]],
    ) -> None:
        if not events:
            self._sequences: list[list[StreamEvent]] = [[]]
        elif isinstance(events[0], StreamEvent):
            self._sequences = [events]  # type: ignore[list-item]
        else:
            self._sequences = list(events)  # type: ignore[arg-type]
        self._call_count = 0
        self.requests: list = []
        self.last_request = None

    async def stream_request(self, request):  # type: ignore[override]
        self.requests.append(request)
        self.last_request = request
        seq_idx = min(self._call_count, len(self._sequences) - 1)
        self._call_count += 1
        for event in self._sequences[seq_idx]:
            yield event

    async def stream(
        self,
        messages: list[LLMMessage],
        system: list[str],
        tools: list[ToolDefinition] | None = None,
        *,
        model: str | None = None,
        max_output_tokens: int | None = None,
        cache_control: CacheControl | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        seq_idx = min(self._call_count, len(self._sequences) - 1)
        self._call_count += 1
        for event in self._sequences[seq_idx]:
            yield event

    async def count_tokens(self, messages, system, tools=None) -> int:
        return 0

    @property
    def model_id(self) -> str:
        return "fake-model"


class FakeTool(Tool):
    """A tool whose `execute` returns a pre-configured output string."""

    def __init__(self, name: str, output: str = "", is_error: bool = False) -> None:
        self._name = name
        self._output = output
        self._is_error = is_error

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"fake {self._name}"

    @property
    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(
        self, input: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        return ToolResult(output=self._output, is_error=self._is_error)

    def to_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._name,
            description=self.description,
            input_schema=self.input_schema,
        )
