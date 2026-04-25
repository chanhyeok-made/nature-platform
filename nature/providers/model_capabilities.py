"""Per-model capability flags.

Some local models advertised via Ollama don't support the structured
`tools=` channel, or are known to drift toward JSON-in-text tool calls
even when `tools=` is accepted. Rather than sprinkle model names
through the agent loop or the probe runner, capture those facts here
once and have callers wrap the provider based on this table.

The design keeps the decision small and test-local — there's no
user-configurable layer yet because the set of adaptations is still
small (two or three models) and changes slowly. Add a YAML/JSON
overlay later if the list starts to grow.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelCapabilities:
    """What the *backend* can accept / how the *model* tends to reply.

    text_tool_adaptation — True if the model should be wrapped in
        `TextToolAdapterProvider` to (a) catch provider-layer tool-
        rejection and retry with a text tool catalog, and (b) parse
        tool calls out of free-form text output. Set for models whose
        Ollama adapter lacks tool_calls plumbing (deepseek-r1 line)
        and for coder tunes that emit JSON-in-text too often to leave
        unhandled (qwen coder tunes).

    stream_timeout_multiplier — per-call timeout scale applied by the
        probe runner. 70B-class local models spend most of their first
        budget window on cold-load TTFT; reasoning models spend it on
        `<think>` traces. A multiplier here lets probes declare
        sensible tier-based budgets while still reaching these slow
        paths. Default 1.0 = no change.

    max_output_ceiling — upper bound the *backend* will accept for a
        single response's max_tokens. When set, the provider clips
        the outgoing `max_tokens` to this value regardless of what
        the caller requested, preventing silent API rejection on
        models whose output window is smaller than nature's 8192
        default. None = trust the caller (or the provider default).
        This is a *physical* cap; per-role policy budgets are a
        separate concern and live in PresetConfig.
    """

    text_tool_adaptation: bool = False
    stream_timeout_multiplier: float = 1.0
    max_output_ceiling: int | None = None


# `host::model` patterns → capabilities. Leftmost match wins, so put
# more-specific patterns before wildcards. Patterns use fnmatch
# (`*` = wildcard, `?` = single char).
#
# Ceiling values below come from provider docs (Anthropic / OpenAI /
# Google model cards, llama.cpp / Ollama config defaults). When in
# doubt, leave None — an explicit 8192 default is fine for most modern
# models and we only need to *lower* when the provider is stricter.
_PATTERNS: tuple[tuple[str, ModelCapabilities], ...] = (
    # DeepSeek reasoning distills: Ollama returns HTTP 400
    # "does not support tools" regardless of the underlying model's
    # capability. Adapt so the probe runner / agent loop can still
    # reach them. Reasoning trace is long — give 3× the default
    # timeout window so probes aren't all cold-clipped.
    ("local-ollama::deepseek-r1*", ModelCapabilities(
        text_tool_adaptation=True, stream_timeout_multiplier=3.0,
    )),
    ("local-ollama::deepseek-v3*", ModelCapabilities(text_tool_adaptation=True)),
    # Qwen coder tunes accept `tools=` but frequently drift: the
    # model emits `{"name":"Read",...}` inside an assistant text
    # block instead of the structured tool_calls channel. Adapting
    # here means consumers see tool_use blocks either way.
    ("local-ollama::qwen2.5-coder*", ModelCapabilities(text_tool_adaptation=True)),
    # Ollama models that don't declare tools support at the HTTP
    # layer — same 400 handling as deepseek-r1, just in narrower
    # model families.
    ("local-ollama::phi4*", ModelCapabilities(text_tool_adaptation=True)),
    ("local-ollama::gemma2*", ModelCapabilities(text_tool_adaptation=True)),
    ("local-ollama::gemma3*", ModelCapabilities(text_tool_adaptation=True)),
    # Cohere command-r: supports tools via Ollama but with a Cohere-
    # specific action format that our Anthropic-shape extractor
    # doesn't match. Falling back to text adaptation at least gets
    # the model through the probe rather than 0/29 across the
    # board on schema-shape divergence.
    ("local-ollama::command-r*", ModelCapabilities(text_tool_adaptation=True)),
    # 70B local models: cold TTFT + weight-load exceeds typical
    # probe budgets on consumer hardware. Give 4× headroom so first-
    # touch probes aren't all reported as timeouts.
    ("local-ollama::llama3.3:70b*", ModelCapabilities(stream_timeout_multiplier=4.0)),
    ("local-ollama::qwen2.5:72b*", ModelCapabilities(stream_timeout_multiplier=2.0)),

    # Anthropic — Claude 4-series output windows per Anthropic model
    # cards (2025): Sonnet 4.6=64K, Haiku 4.5=8K, Opus 4=32K. Request
    # beyond ceiling is rejected with 400.
    ("anthropic::claude-sonnet-4*", ModelCapabilities(max_output_ceiling=64_000)),
    ("anthropic::claude-opus-4*", ModelCapabilities(max_output_ceiling=32_000)),
    ("anthropic::claude-haiku-4*", ModelCapabilities(max_output_ceiling=8_192)),

    # OpenRouter — reasoning-flavored models produce long <think>
    # traces, plus ceilings per official docs where known.
    ("openrouter::deepseek/deepseek-r1*", ModelCapabilities(
        text_tool_adaptation=True, stream_timeout_multiplier=2.0,
        max_output_ceiling=8_192,
    )),
    # Gemini 2.5 Pro/Flash thinking mode: 64K output per Google docs.
    ("openrouter::google/gemini-2.5-pro*", ModelCapabilities(
        stream_timeout_multiplier=2.0, max_output_ceiling=64_000,
    )),
    ("openrouter::google/gemini-2.5-flash*", ModelCapabilities(
        max_output_ceiling=64_000,
    )),
    # Grok 4 reasoning; ceiling conservatively per xAI docs.
    ("openrouter::x-ai/grok-4*", ModelCapabilities(
        stream_timeout_multiplier=2.0, max_output_ceiling=32_000,
    )),
    # Claude via OpenRouter: same physical ceilings as direct.
    ("openrouter::anthropic/claude-sonnet-4*", ModelCapabilities(
        max_output_ceiling=64_000,
    )),
    ("openrouter::anthropic/claude-opus-4*", ModelCapabilities(
        max_output_ceiling=32_000,
    )),
    ("openrouter::anthropic/claude-haiku-4*", ModelCapabilities(
        max_output_ceiling=8_192,
    )),
    # OpenAI chat/reasoning: GPT-4o 16K, GPT-5-mini 32K per docs.
    ("openrouter::openai/gpt-4o*", ModelCapabilities(max_output_ceiling=16_384)),
    ("openrouter::openai/gpt-5*", ModelCapabilities(max_output_ceiling=32_000)),
    # Llama 3.x via OpenRouter: 8K output. Mixtral/Mistral Large same.
    ("openrouter::meta-llama/llama-3*", ModelCapabilities(max_output_ceiling=8_192)),
    ("openrouter::mistralai/*", ModelCapabilities(max_output_ceiling=8_192)),
    # Qwen 2.5 coder via OpenRouter has no tool-use endpoint (OpenRouter
    # returns HTTP 404 "No endpoints found that support tool use" for
    # every tool call). Adapt to text-tool mode the same way we do for
    # local-ollama qwen-coder tunes. Must come BEFORE the wildcard
    # openrouter::qwen/qwen-2.5* pattern below so the more specific
    # rule wins.
    ("openrouter::qwen/qwen-2.5-coder*", ModelCapabilities(
        text_tool_adaptation=True, max_output_ceiling=8_192,
    )),
    # Qwen 2.5 cloud: 8K output.
    ("openrouter::qwen/qwen-2.5*", ModelCapabilities(max_output_ceiling=8_192)),
)


def lookup(model_ref: str) -> ModelCapabilities:
    """Return the capability record for `host::model`. Models not
    in the table get the default (no adaptation)."""
    for pattern, caps in _PATTERNS:
        if fnmatch.fnmatchcase(model_ref, pattern):
            return caps
    return ModelCapabilities()


def clip_to_ceiling(
    requested_max_tokens: int,
    host_name: str | None,
    model: str,
) -> int:
    """Return `min(requested, model_ceiling)`. When host_name is None
    or the model has no entry in the table, returns `requested`
    unchanged — the provider's own default still applies.

    Called by providers at the `max_tokens` synthesis point so a
    preset budget or config default that exceeds the model's
    physical ceiling gets silently lowered instead of being rejected
    with an HTTP 400.
    """
    if host_name is None or requested_max_tokens <= 0:
        return requested_max_tokens
    caps = lookup(f"{host_name}::{model}")
    if caps.max_output_ceiling is None:
        return requested_max_tokens
    return min(requested_max_tokens, caps.max_output_ceiling)


__all__ = ["ModelCapabilities", "lookup", "clip_to_ceiling"]
