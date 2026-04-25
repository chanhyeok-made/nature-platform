"""Anthropic provider — Claude API streaming with prompt caching.

This is the port/adapter boundary. Internal code never touches
anthropic SDK types directly; everything is converted to nature's
universal Message/StreamEvent format here.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator

from nature.protocols.message import (
    ContentBlock,
    ImageContent,
    ImageSource,
    Message,
    Role,
    StreamEvent,
    StreamEventType,
    TextContent,
    ThinkingContent,
    ToolResultContent,
    ToolUseContent,
    Usage,
)
from nature.protocols.provider import CacheControl, ProviderConfig
from nature.protocols.tool import ToolDefinition
from nature.providers.base import BaseLLMProvider
from nature.utils.tokens import estimate_tokens_for_text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Message conversion: nature -> Anthropic API format
# ---------------------------------------------------------------------------

def _content_block_to_api(block: ContentBlock) -> dict[str, Any]:
    """Convert a nature ContentBlock to Anthropic API format."""
    if isinstance(block, TextContent):
        return {"type": "text", "text": block.text}
    if isinstance(block, ToolUseContent):
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    if isinstance(block, ToolResultContent):
        return {
            "type": "tool_result",
            "tool_use_id": block.tool_use_id,
            "content": block.content if isinstance(block.content, str) else [
                _content_block_to_api(b) for b in block.content
            ],
            **({"is_error": True} if block.is_error else {}),
        }
    if isinstance(block, ImageContent):
        return {
            "type": "image",
            "source": {
                "type": block.source.type,
                "media_type": block.source.media_type,
                "data": block.source.data,
            },
        }
    if isinstance(block, ThinkingContent):
        return {"type": "thinking", "thinking": block.thinking}
    raise ValueError(f"Unknown content block type: {type(block)}")


def _message_to_api(msg: Message) -> dict[str, Any]:
    """Convert a nature Message to Anthropic API format."""
    return {
        "role": msg.role.value,
        "content": [_content_block_to_api(b) for b in msg.content],
    }


def _tool_def_to_api(tool: ToolDefinition) -> dict[str, Any]:
    """Convert a nature ToolDefinition to Anthropic API format."""
    result: dict[str, Any] = {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
    }
    if tool.deferred:
        result["defer_loading"] = True
    return result


def _mark_last_block_cacheable(
    content: list[dict[str, Any]],
    cache_control: CacheControl,
) -> None:
    """Stamp a cache_control breakpoint on the LAST content block in
    `content`, in-place. Used to set a cache breakpoint at the end
    of the messages array so the entire conversation prefix becomes
    a cacheable suffix on the next turn.

    Forwards the caller's CacheControl (including `ttl`) so long-TTL
    choices actually reach the API. No-op when `content` is empty.
    Existing `cache_control` fields are overwritten — the most
    recent placement wins.
    """
    if not content:
        return
    last = content[-1]
    if isinstance(last, dict):
        last["cache_control"] = _cache_control_dict(cache_control)


def _is_footer_hint_message(msg: dict[str, Any]) -> bool:
    """Heuristic: detect a synthetic footer-hint user message.

    The composer appends footer hints (synthesis_nudge,
    todo_continues, etc.) as a trailing user-role message whose
    text starts with `<system-reminder>`. These messages change
    every turn (different rule fires, different counts), so if we
    place a cache breakpoint on a hint message, the next turn's
    request has a different last block at that position and the
    cache prefix doesn't match → no cache hit.

    By detecting hint messages we can skip them when picking the
    cache anchor, which lets the cache prefix end at the last
    *real* conversation message — a position that's stable across
    turns and produces actual cache hits on the next request.
    """
    if msg.get("role") != "user":
        return False
    content = msg.get("content") or []
    if not content:
        return False
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text = (block.get("text") or "").lstrip()
            if text.startswith("<system-reminder>"):
                return True
    return False


def _pick_cache_anchor_index(api_messages: list[dict[str, Any]]) -> int | None:
    """Return the index of the message that should carry the cache
    breakpoint. Walks back from the end skipping footer-hint
    messages so the breakpoint lands on a stable conversation
    message instead of the per-turn synthetic hint."""
    if not api_messages:
        return None
    for i in range(len(api_messages) - 1, -1, -1):
        if not _is_footer_hint_message(api_messages[i]):
            return i
    # All messages look like hints — fall back to last (better than nothing).
    return len(api_messages) - 1


# ---------------------------------------------------------------------------
# System prompt building with cache boundaries
# ---------------------------------------------------------------------------

from nature.config.defaults import DYNAMIC_BOUNDARY


def _cache_control_dict(cc: CacheControl) -> dict[str, str]:
    """Convert a CacheControl dataclass to the Anthropic API dict.

    Forwards the optional `ttl` field ("5m" or "1h"). Longer TTLs
    cost 2x to create instead of 1.25x, but survive long multi-frame
    sessions — worth it when the same cached prefix is reused by
    more than ~3 LLM calls over >5 minutes.
    """
    d: dict[str, str] = {"type": cc.type}
    if cc.ttl:
        d["ttl"] = cc.ttl
    return d


def _build_system_blocks(
    system: list[str],
    cache_control: CacheControl | None = None,
) -> list[dict[str, Any]]:
    """Build system prompt blocks with cache control markers.

    Splits on DYNAMIC_BOUNDARY: blocks before it get cache_control,
    blocks after it don't (session-specific, not cacheable).
    """
    if not system:
        return []

    joined = "\n\n".join(system)

    if DYNAMIC_BOUNDARY in joined:
        parts = joined.split(DYNAMIC_BOUNDARY, 1)
        static_part = parts[0].strip()
        dynamic_part = parts[1].strip() if len(parts) > 1 else ""

        blocks: list[dict[str, Any]] = []
        if static_part:
            block: dict[str, Any] = {"type": "text", "text": static_part}
            if cache_control:
                block["cache_control"] = _cache_control_dict(cache_control)
            blocks.append(block)
        if dynamic_part:
            blocks.append({"type": "text", "text": dynamic_part})
        return blocks

    # No boundary marker: single block with optional cache control
    block = {"type": "text", "text": joined}
    if cache_control:
        block["cache_control"] = _cache_control_dict(cache_control)
    return [block]


# ---------------------------------------------------------------------------
# AnthropicProvider
# ---------------------------------------------------------------------------

class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API provider with streaming and prompt caching.

    Wraps the anthropic Python SDK and converts all types at the boundary.
    """

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic package required. Install with: pip install 'nature[anthropic]'"
            ) from e

        kwargs: dict[str, Any] = {}
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url
        if config.extra_headers:
            kwargs["default_headers"] = config.extra_headers

        self._client = anthropic.AsyncAnthropic(**kwargs)

    @property
    def context_window(self) -> int:
        model = self._config.model.lower()
        if "1m" in model or "1000k" in model:
            return 1_000_000
        if "opus" in model:
            return 200_000
        if "sonnet" in model:
            return 200_000
        if "haiku" in model:
            return 200_000
        return 200_000

    @property
    def supports_caching(self) -> bool:
        return True

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
        """Stream a response from Claude.

        Converts Anthropic SDK streaming events to nature StreamEvents.
        Per-request `model` (when set) overrides the provider's config
        model — enables per-agent model selection in frame mode.
        """
        # Build request
        api_messages = [_message_to_api(m) for m in messages]
        system_blocks = _build_system_blocks(system, cache_control)

        effective_model = model or self._config.model
        from nature.providers.model_capabilities import clip_to_ceiling
        effective_max = clip_to_ceiling(
            max_output_tokens or self._config.max_output_tokens,
            self._config.host_name, effective_model,
        )
        request_kwargs: dict[str, Any] = {
            "model": effective_model,
            "messages": api_messages,
            "max_tokens": effective_max,
        }
        if system_blocks:
            request_kwargs["system"] = system_blocks
        if tools:
            tool_defs = [_tool_def_to_api(t) for t in tools]
            # Cache the tools list. Anthropic's prompt cache is a
            # prefix cache, so marking the LAST tool definition stamps
            # the system+tools prefix as cacheable. Tools change
            # rarely (only when role.allowed_tools changes), so this
            # breakpoint typically holds across the whole session.
            if cache_control and tool_defs:
                tool_defs[-1]["cache_control"] = _cache_control_dict(cache_control)
            request_kwargs["tools"] = tool_defs

        # Cache the conversation history. Stamp a breakpoint on the
        # last *real* message, skipping any trailing footer-hint
        # message — hints change every turn (different rule, different
        # counts) and would break cache continuity if marked. Pinning
        # the breakpoint on the prior turn's actual content lets the
        # next request hit the cached prefix even though its hint is
        # different. Anthropic caches everything up to and including
        # the marked block, so on the next turn the system + tools +
        # all prior real messages are a cache hit and only the new
        # tool_result + new hint count as fresh input. Without this,
        # every turn re-pays the full input cost (session 409b958e:
        # 1.84M billed input over 85 turns with 0 cache reads, ~$6).
        if cache_control and api_messages:
            anchor = _pick_cache_anchor_index(api_messages)
            if anchor is not None:
                _mark_last_block_cacheable(
                    api_messages[anchor].get("content", []),
                    cache_control,
                )
        if self._config.temperature is not None:
            request_kwargs["temperature"] = self._config.temperature
        if self._config.extra_body:
            request_kwargs.update(self._config.extra_body)

        # Stream response — retry the initial request on transient
        # provider errors (overloaded, rate-limit, connection drop,
        # timeout). Retry only before the first StreamEvent has
        # yielded; once the caller has seen a MESSAGE_START, re-
        # issuing the request would double-emit into their iterator.
        from nature.providers.retry import (
            DEFAULT_MAX_ATTEMPTS,
            is_retryable_anthropic_error,
            sleep_backoff,
        )
        for attempt in range(DEFAULT_MAX_ATTEMPTS):
            gen = self._stream_one_attempt(request_kwargs)
            try:
                first_event = await gen.__anext__()
            except StopAsyncIteration:
                return
            except Exception as exc:  # noqa: BLE001
                if (
                    is_retryable_anthropic_error(exc)
                    and attempt < DEFAULT_MAX_ATTEMPTS - 1
                ):
                    logger.warning(
                        "anthropic stream attempt %d/%d failed "
                        "(retryable): %s — backing off",
                        attempt + 1, DEFAULT_MAX_ATTEMPTS, exc,
                    )
                    await sleep_backoff(attempt)
                    continue
                raise
            # Committed: yield the first event and the rest inline.
            yield first_event
            async for event in gen:
                yield event
            return

    async def _stream_one_attempt(
        self, request_kwargs: dict[str, Any],
    ) -> AsyncGenerator[StreamEvent, None]:
        """One attempt at streaming. Yields StreamEvents in order.

        Factored out of `stream()` so the retry wrapper can safely
        discard and re-invoke before the first event has crossed the
        boundary into the caller.
        """
        async with self._client.messages.stream(**request_kwargs) as stream:
            # Yield message_start
            yield StreamEvent(type=StreamEventType.MESSAGE_START)

            current_block_index = 0
            current_tool_name: str | None = None
            current_tool_id: str | None = None
            tool_input_json = ""

            async for event in stream:
                event_type = event.type

                if event_type == "content_block_start":
                    block = event.content_block
                    current_block_index = event.index

                    if block.type == "text":
                        yield StreamEvent(
                            type=StreamEventType.CONTENT_BLOCK_START,
                            index=current_block_index,
                            content_block=TextContent(text=""),
                        )
                    elif block.type == "tool_use":
                        current_tool_name = block.name
                        current_tool_id = block.id
                        tool_input_json = ""
                        yield StreamEvent(
                            type=StreamEventType.CONTENT_BLOCK_START,
                            index=current_block_index,
                            content_block=ToolUseContent(
                                id=block.id, name=block.name, input={}
                            ),
                        )
                    elif block.type == "thinking":
                        yield StreamEvent(
                            type=StreamEventType.CONTENT_BLOCK_START,
                            index=current_block_index,
                            content_block=ThinkingContent(thinking=""),
                        )

                elif event_type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        yield StreamEvent(
                            type=StreamEventType.CONTENT_BLOCK_DELTA,
                            index=current_block_index,
                            delta_text=delta.text,
                        )
                    elif delta.type == "input_json_delta":
                        tool_input_json += delta.partial_json
                        yield StreamEvent(
                            type=StreamEventType.CONTENT_BLOCK_DELTA,
                            index=current_block_index,
                            delta_tool_input=delta.partial_json,
                        )
                    elif delta.type == "thinking_delta":
                        yield StreamEvent(
                            type=StreamEventType.CONTENT_BLOCK_DELTA,
                            index=current_block_index,
                            delta_text=delta.thinking,
                        )

                elif event_type == "content_block_stop":
                    # If this was a tool_use block, parse the accumulated JSON
                    if current_tool_name is not None:
                        parsed_input = {}
                        if tool_input_json:
                            try:
                                parsed_input = json.loads(tool_input_json)
                            except json.JSONDecodeError:
                                parsed_input = {}
                        yield StreamEvent(
                            type=StreamEventType.CONTENT_BLOCK_STOP,
                            index=current_block_index,
                            content_block=ToolUseContent(
                                id=current_tool_id or "",
                                name=current_tool_name,
                                input=parsed_input,
                            ),
                        )
                        current_tool_name = None
                        current_tool_id = None
                        tool_input_json = ""
                    else:
                        yield StreamEvent(
                            type=StreamEventType.CONTENT_BLOCK_STOP,
                            index=current_block_index,
                        )

                elif event_type == "message_delta":
                    usage_data = getattr(event, "usage", None)
                    usage = None
                    if usage_data:
                        usage = Usage(
                            output_tokens=getattr(usage_data, "output_tokens", 0),
                        )
                    yield StreamEvent(
                        type=StreamEventType.MESSAGE_DELTA,
                        stop_reason=getattr(event.delta, "stop_reason", None),
                        usage=usage,
                    )

            # After stream completes, get the final message for full usage
            final_message = await stream.get_final_message()
            final_usage = Usage(
                input_tokens=final_message.usage.input_tokens,
                output_tokens=final_message.usage.output_tokens,
                cache_creation_input_tokens=getattr(
                    final_message.usage, "cache_creation_input_tokens", 0
                ) or 0,
                cache_read_input_tokens=getattr(
                    final_message.usage, "cache_read_input_tokens", 0
                ) or 0,
            )
            self._accumulate_usage(final_usage)

            yield StreamEvent(
                type=StreamEventType.MESSAGE_STOP,
                usage=final_usage,
                stop_reason=final_message.stop_reason,
            )

    async def count_tokens(
        self,
        messages: list[Message],
        system: list[str],
        tools: list[ToolDefinition] | None = None,
    ) -> int:
        """Count tokens using the Anthropic API (exact)."""
        try:
            api_messages = [_message_to_api(m) for m in messages]
            system_blocks = _build_system_blocks(system)

            kwargs: dict[str, Any] = {
                "model": self._config.model,
                "messages": api_messages,
            }
            if system_blocks:
                kwargs["system"] = system_blocks
            if tools:
                kwargs["tools"] = [_tool_def_to_api(t) for t in tools]

            result = await self._client.beta.messages.count_tokens(**kwargs)
            return result.input_tokens
        except Exception as e:
            logger.debug("API token counting failed, using estimation: %s", e)
            # Fallback to byte-based estimation
            total = 0
            for s in system:
                total += estimate_tokens_for_text(s)
            for m in messages:
                total += estimate_tokens_for_text(m.model_dump_json())
            if tools:
                for t in tools:
                    total += estimate_tokens_for_text(json.dumps(t.input_schema))
            return total

    async def close(self) -> None:
        """Close the underlying httpx client."""
        await self._client.close()
