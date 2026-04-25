"""TextToolAdapterProvider — decorate a provider so models that
cannot consume the `tools=` field (Ollama's `deepseek-r1:*` returns
HTTP 400 "does not support tools") or don't reliably emit structured
tool_use blocks (qwen-coder tunes drift toward prose-embedded JSON
tool calls) still look tool-capable to consumers.

The wrapper:

1. Forwards calls unchanged when `tools` is empty — no-op path.
2. Attempts the inner provider with the original `tools`. If it
   raises an "does not support tools" style 400, retries the same
   request with `tools=None` and a text catalog of the tools appended
   to the system prompt.
3. Streams the inner provider's events through; accumulates any
   emitted TextContent. At stream end, if NO structured
   ToolUseContent was observed, runs the accumulated text through
   `nature.agent.text_tool_parser.extract_tool_calls_from_text` and —
   when tool calls are recovered — synthesizes CONTENT_BLOCK_START /
   CONTENT_BLOCK_STOP events for each so downstream consumers (agent
   loop, probe runner) see the same shape whether the model emitted
   structured blocks or free-form JSON text.

Consumers get a clean tool-using provider interface. The wrapper's
only observable effect is "this model acts tool-capable now."
"""

from __future__ import annotations

import json as _json
from typing import AsyncGenerator

from nature.protocols.message import (
    ContentBlock,
    Message,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolUseContent,
)
from nature.protocols.provider import CacheControl, LLMProvider
from nature.protocols.tool import ToolDefinition


def _catalog_text(tools: list[ToolDefinition]) -> str:
    """Render the tool list as a system-prompt addendum that matches
    the shape `text_tool_parser` can parse back."""
    # The parser normalizes case when matching tool names, so we don't
    # tell the model "case-sensitive" — it would be a lie. We kept the
    # synthesis-enforcement line tested in v7b: it didn't measurably help
    # the one model that needed it (gemma2:27b), and added noise on
    # single-turn probes; erring on less prompt is better than more.
    lines = [
        "You can invoke a tool by emitting a JSON object in a fenced",
        "code block with this exact shape:",
        "",
        '```json',
        '{"name": "<tool-name>", "arguments": {...args...}}',
        '```',
        "",
        "Emit ONE tool call per message if a tool is needed. Do not",
        "narrate the call — just the JSON block. Available tools:",
        "",
    ]
    for t in tools:
        lines.append(f"- {t.name}: {t.description}")
        schema = t.input_schema if isinstance(t.input_schema, dict) else {}
        lines.append(f"  input_schema: {_json.dumps(schema)}")
    return "\n".join(lines)


def _is_no_tools_error(exc: Exception) -> bool:
    """Match backend errors that mean "this model has no tool-calling
    endpoint available" — retry the same request in text-tool mode.

    Known patterns:
    - Ollama: HTTP 400 with "does not support tools"
    - OpenRouter: HTTP 404 with "No endpoints found that support tool use"
      (raised when none of OpenRouter's upstream providers advertise
      tool use for the requested model, e.g. qwen-2.5-coder-32b)
    """
    msg = str(exc)
    if "does not support tools" in msg:
        return True
    if "No endpoints found that support tool use" in msg:
        return True
    # A 400 that mentions tools is also a strong signal (some backends
    # phrase it differently) — match conservatively.
    if "API error 400" in msg and "tools" in msg.lower():
        return True
    return False


class TextToolAdapterProvider(LLMProvider):
    """Wrap an inner provider so tool-capability is synthesized when
    the backend model can't do it natively."""

    def __init__(self, inner: LLMProvider) -> None:
        self._inner = inner

    # ── Forward the remaining LLMProvider abstract surface to inner ──
    # LLMProvider has a handful of abstract methods besides stream();
    # implement them as thin passthroughs so instantiation succeeds
    # and callers that introspect config/model_id keep working.

    async def count_tokens(self, messages, system=None, tools=None):  # noqa: ANN001
        return await self._inner.count_tokens(
            messages=messages, system=system, tools=tools,
        )

    def model_id(self) -> str:
        return self._inner.model_id()

    def context_window(self) -> int:
        return self._inner.context_window()

    def supports_caching(self) -> bool:
        return self._inner.supports_caching()

    async def close(self) -> None:
        await self._inner.close()

    # Catch-all passthrough for ad-hoc attributes the inner provider
    # exposes (e.g. `provider.config`). `__getattr__` only fires on
    # misses so the explicitly-defined methods above take precedence.
    def __getattr__(self, name: str):  # noqa: D401
        return getattr(self._inner, name)

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
        # Fast path: no tools → no wrapping needed.
        if not tools:
            async for ev in self._inner.stream(
                messages=messages, system=system, tools=tools,
                model=model, max_output_tokens=max_output_tokens,
                cache_control=cache_control,
            ):
                yield ev
            return

        text_mode = False
        effective_system = list(system)
        effective_tools: list[ToolDefinition] | None = list(tools)

        # First attempt: structured tools. If the backend rejects,
        # flip to text mode and retry. Flipping mid-stream is
        # intractable (would double-emit MESSAGE_START), so the retry
        # starts a fresh stream.
        attempt = 0
        while True:
            attempt += 1
            if text_mode:
                effective_tools = None
                effective_system = list(system) + [_catalog_text(tools)]
            try:
                async for ev in self._wrap_stream(
                    tools,
                    self._inner.stream(
                        messages=messages, system=effective_system,
                        tools=effective_tools, model=model,
                        max_output_tokens=max_output_tokens,
                        cache_control=cache_control,
                    ),
                ):
                    yield ev
                return
            except Exception as exc:  # noqa: BLE001
                if not text_mode and _is_no_tools_error(exc) and attempt < 2:
                    text_mode = True
                    continue
                raise

    async def _wrap_stream(
        self,
        tools: list[ToolDefinition],
        inner_stream: AsyncGenerator[StreamEvent, None],
    ) -> AsyncGenerator[StreamEvent, None]:
        """Pass through inner events; track whether any structured
        tool_use block fired. If not, parse the accumulated text at
        stream end and synthesize tool_use events so downstream code
        sees the same shape it would from a native tool provider.

        Important: synthetic events are emitted BEFORE MESSAGE_STOP
        so consumers that stop iterating on MESSAGE_STOP (which the
        probe runner does not, but agent loops typically do) still
        observe them.
        """
        known_names = {t.name for t in tools}
        text_buf: list[str] = []
        saw_structured_tool = False
        pending_message_stop: StreamEvent | None = None

        async for ev in inner_stream:
            if ev.type == StreamEventType.CONTENT_BLOCK_START:
                if (
                    ev.content_block is not None
                    and isinstance(ev.content_block, ToolUseContent)
                ):
                    saw_structured_tool = True
            elif ev.type == StreamEventType.CONTENT_BLOCK_DELTA:
                if ev.delta_text is not None:
                    text_buf.append(ev.delta_text)
            elif ev.type == StreamEventType.MESSAGE_STOP:
                pending_message_stop = ev
                continue  # hold until after synthetic emissions
            yield ev

        # Emit synthetic tool_use events if the model drifted to
        # JSON-in-text.
        if not saw_structured_tool and known_names and text_buf:
            from nature.agent.text_tool_parser import extract_tool_calls_from_text
            _, recovered = extract_tool_calls_from_text(
                "".join(text_buf), known_names,
            )
            # Use high index numbers to avoid collision with any
            # indexes the inner stream already used for text blocks.
            for i, tu in enumerate(recovered, start=1000):
                yield StreamEvent(
                    type=StreamEventType.CONTENT_BLOCK_START,
                    index=i,
                    content_block=tu,
                )
                # Skip deltas — the ToolUseContent already carries the
                # parsed input directly.
                yield StreamEvent(
                    type=StreamEventType.CONTENT_BLOCK_STOP,
                    index=i,
                    content_block=tu,
                )

        if pending_message_stop is not None:
            yield pending_message_stop


__all__ = ["TextToolAdapterProvider"]
