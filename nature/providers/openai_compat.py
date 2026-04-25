"""OpenAI-compatible provider — works with OpenRouter, Gemini, Ollama, vLLM, etc.

Any service that speaks the OpenAI Chat Completions API can be used:
- OpenRouter (free models: google/gemini-2.5-flash, meta-llama/llama-4-scout, etc.)
- Google Gemini via OpenAI compat endpoint
- Ollama (local models)
- vLLM, LiteLLM, Together AI, Groq, etc.

Usage:
    nature chat --provider openai --model google/gemini-2.5-flash-preview
    OPENAI_BASE_URL=https://openrouter.ai/api/v1 OPENAI_API_KEY=... nature chat
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator

import httpx

from nature.protocols.message import (
    ContentBlock,
    Message,
    Role,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolResultContent,
    ToolUseContent,
    Usage,
)
from nature.protocols.provider import CacheControl, ProviderConfig
from nature.protocols.tool import ToolDefinition
from nature.providers.base import BaseLLMProvider
from nature.utils.tokens import estimate_tokens_for_text

logger = logging.getLogger(__name__)

# Default free models via OpenRouter
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_FREE_MODEL = "google/gemini-2.5-flash-preview"


# ---------------------------------------------------------------------------
# Message conversion: nature -> OpenAI format
# ---------------------------------------------------------------------------

def _content_block_to_openai(block: ContentBlock) -> dict[str, Any] | str:
    """Convert a nature ContentBlock to OpenAI message content."""
    if isinstance(block, TextContent):
        return {"type": "text", "text": block.text}
    if isinstance(block, ToolUseContent):
        # Tool calls are handled separately in OpenAI format
        return {"type": "text", "text": ""}
    if isinstance(block, ToolResultContent):
        content = block.content if isinstance(block.content, str) else str(block.content)
        return {"type": "text", "text": content}
    return {"type": "text", "text": str(block)}


def _messages_to_openai(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert nature messages to OpenAI Chat format."""
    result: list[dict[str, Any]] = []

    for msg in messages:
        # Check for tool_use blocks (assistant with tool calls)
        tool_use_blocks = [b for b in msg.content if isinstance(b, ToolUseContent)]
        tool_result_blocks = [b for b in msg.content if isinstance(b, ToolResultContent)]
        text_blocks = [b for b in msg.content if isinstance(b, TextContent)]

        if msg.role == Role.ASSISTANT and tool_use_blocks:
            # Assistant message with tool calls
            text_content = "".join(b.text for b in text_blocks)
            tool_calls = [
                {
                    "id": tu.id,
                    "type": "function",
                    "function": {
                        "name": tu.name,
                        "arguments": json.dumps(tu.input),
                    },
                }
                for tu in tool_use_blocks
            ]
            entry: dict[str, Any] = {
                "role": "assistant",
                "tool_calls": tool_calls,
            }
            if text_content:
                entry["content"] = text_content
            result.append(entry)

        elif msg.role == Role.USER and tool_result_blocks:
            # Tool results → OpenAI "tool" role messages
            for tr in tool_result_blocks:
                content = tr.content if isinstance(tr.content, str) else str(tr.content)
                result.append({
                    "role": "tool",
                    "tool_call_id": tr.tool_use_id,
                    "content": content,
                })

        else:
            # Regular text message
            text = "".join(
                b.text for b in msg.content if isinstance(b, TextContent)
            )
            if text:
                result.append({
                    "role": msg.role.value,
                    "content": text,
                })

    return result


def _tool_def_to_openai(tool: ToolDefinition) -> dict[str, Any]:
    """Convert a nature ToolDefinition to OpenAI function format."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


# ---------------------------------------------------------------------------
# OpenAICompatProvider
# ---------------------------------------------------------------------------

class OpenAICompatProvider(BaseLLMProvider):
    """OpenAI-compatible provider for OpenRouter, Gemini, Ollama, etc.

    Uses raw httpx for streaming to avoid requiring the openai package.
    """

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._base_url = (config.base_url or OPENROUTER_BASE_URL).rstrip("/")
        self._api_key = config.api_key or ""
        # HTTP timeout — 120s works fine for cloud OpenAI-compat endpoints
        # (OpenRouter, Gemini), but local ollama running a 30B+ model on
        # consumer HW can take 1–5 minutes for prefill on long context +
        # another 30–60s for generation. Read from NATURE_OPENAI_TIMEOUT
        # (seconds) so a local run can extend it without touching code.
        # Default 600s covers typical local workloads.
        import os as _os
        timeout = float(_os.environ.get("NATURE_OPENAI_TIMEOUT", "600"))
        self._client = httpx.AsyncClient(timeout=timeout)

    @property
    def context_window(self) -> int:
        model = self._config.model.lower()
        if "gemini" in model:
            return 1_000_000
        if "llama" in model:
            return 128_000
        if "qwen" in model:
            return 128_000
        return 128_000

    @property
    def supports_caching(self) -> bool:
        return False  # Most OpenAI-compat endpoints don't support Anthropic-style caching

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
        """Stream a response via OpenAI Chat Completions API.

        Per-request `model` (when set) overrides the provider's config
        model — enables per-agent model selection in frame mode.
        """

        # Build request
        api_messages = _messages_to_openai(messages)

        # Prepend system message
        system_text = "\n\n".join(system)
        if system_text:
            api_messages.insert(0, {"role": "system", "content": system_text})

        effective_model = model or self._config.model
        from nature.providers.model_capabilities import clip_to_ceiling
        effective_max = clip_to_ceiling(
            max_output_tokens or self._config.max_output_tokens,
            self._config.host_name, effective_model,
        )
        body: dict[str, Any] = {
            "model": effective_model,
            "messages": api_messages,
            "stream": True,
            "max_tokens": effective_max,
        }
        if tools:
            body["tools"] = [_tool_def_to_openai(t) for t in tools]
        if self._config.temperature is not None:
            body["temperature"] = self._config.temperature

        # Merge extra_body (Ollama options: num_ctx, num_batch, num_gpu_layers, etc.)
        if self._config.extra_body:
            body.update(self._config.extra_body)

        headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        # OpenRouter-specific headers
        if "openrouter" in self._base_url:
            headers["HTTP-Referer"] = "https://github.com/nature-agent"
            headers["X-Title"] = "nature"
        headers.update(self._config.extra_headers)

        yield StreamEvent(type=StreamEventType.MESSAGE_START)

        # Stream SSE
        input_estimate = estimate_tokens_for_text(json.dumps(body))
        output_tokens = 0
        current_tool_calls: dict[int, dict[str, str]] = {}  # index -> {id, name, arguments}
        stop_reason: str | None = None

        async with self._client.stream(
            "POST",
            f"{self._base_url}/chat/completions",
            json=body,
            headers=headers,
        ) as response:
            if response.status_code != 200:
                error_body = await response.aread()
                error_text = error_body.decode("utf-8", errors="replace")
                raise Exception(
                    f"API error {response.status_code}: {error_text}"
                )

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break

                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                finish_reason = choices[0].get("finish_reason")

                # Text content
                content = delta.get("content")
                if content:
                    output_tokens += len(content) // 4  # rough estimate
                    yield StreamEvent(
                        type=StreamEventType.CONTENT_BLOCK_DELTA,
                        delta_text=content,
                    )

                # Tool calls
                tool_calls = delta.get("tool_calls", [])
                for tc in tool_calls:
                    idx = tc.get("index", 0)
                    if idx not in current_tool_calls:
                        current_tool_calls[idx] = {
                            "id": tc.get("id", f"call_{idx}"),
                            "name": tc.get("function", {}).get("name", ""),
                            "arguments": "",
                        }
                        # Emit start
                        yield StreamEvent(
                            type=StreamEventType.CONTENT_BLOCK_START,
                            index=idx,
                            content_block=ToolUseContent(
                                id=current_tool_calls[idx]["id"],
                                name=current_tool_calls[idx]["name"],
                                input={},
                            ),
                        )

                    # Accumulate arguments
                    args_delta = tc.get("function", {}).get("arguments", "")
                    if args_delta:
                        current_tool_calls[idx]["arguments"] += args_delta
                        yield StreamEvent(
                            type=StreamEventType.CONTENT_BLOCK_DELTA,
                            index=idx,
                            delta_tool_input=args_delta,
                        )

                if finish_reason:
                    stop_reason = finish_reason

        # Emit tool_use block stops with parsed input
        for idx, tc_data in current_tool_calls.items():
            try:
                parsed_input = json.loads(tc_data["arguments"]) if tc_data["arguments"] else {}
            except json.JSONDecodeError:
                parsed_input = {}
            yield StreamEvent(
                type=StreamEventType.CONTENT_BLOCK_STOP,
                index=idx,
                content_block=ToolUseContent(
                    id=tc_data["id"],
                    name=tc_data["name"],
                    input=parsed_input,
                ),
            )

        # Final usage
        final_usage = Usage(
            input_tokens=input_estimate,
            output_tokens=output_tokens,
        )
        self._accumulate_usage(final_usage)

        yield StreamEvent(
            type=StreamEventType.MESSAGE_STOP,
            usage=final_usage,
            stop_reason=stop_reason or "stop",
        )

    async def count_tokens(
        self,
        messages: list[Message],
        system: list[str],
        tools: list[ToolDefinition] | None = None,
    ) -> int:
        """Estimate tokens (no exact counting API for OpenAI-compat)."""
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
        await self._client.aclose()
