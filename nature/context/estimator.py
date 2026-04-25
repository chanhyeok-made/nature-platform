"""Token estimator — dual strategy (API exact + byte fallback).

Uses the provider's count_tokens() when available, falls back to
byte-based estimation.
"""

from __future__ import annotations

from nature.protocols.message import Message
from nature.protocols.provider import LLMProvider
from nature.protocols.tool import ToolDefinition
from nature.utils.tokens import estimate_tokens_for_text


async def estimate_messages_tokens(
    messages: list[Message],
    system: list[str],
    tools: list[ToolDefinition] | None = None,
    provider: LLMProvider | None = None,
) -> int:
    """Estimate total input tokens for a request.

    Tries API-based counting first, falls back to byte estimation.
    """
    if provider:
        try:
            return await provider.count_tokens(messages, system, tools)
        except Exception:
            pass

    # Byte-based fallback
    total = 0
    for s in system:
        total += estimate_tokens_for_text(s)
    for m in messages:
        total += estimate_tokens_for_text(m.model_dump_json(), is_json=True)
    if tools:
        for t in tools:
            total += estimate_tokens_for_text(str(t.input_schema), is_json=True)
    return total
