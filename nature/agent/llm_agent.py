"""llm_agent — the pure function at the core of agent execution.

This is the stateless unit: given a Context, a role, a model, and a
provider, make ONE LLM call and return structured output. No loops,
no state mutation, no tool execution — those belong to the caller
(AreaManager in Step 4).

The signature is what makes multi-agent orchestration tractable: every
agent is literally this function, called with different contexts and
roles. What changes between "receptionist", "core", and "researcher"
is only the data flowing in — the function is identical.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from uuid import uuid4

from nature.agent.output import AgentOutput, Signal
from nature.config.constants import StopReason
from nature.context.composer import ContextComposer
from nature.context.conversation import Message, MessageAnnotation
from nature.context.types import Context
from nature.protocols.message import (
    ContentBlock,
    StreamEventType,
    TextContent,
    ThinkingContent,
    ToolUseContent,
    Usage,
)
from nature.protocols.provider import LLMProvider
from nature.protocols.tool import Tool
from nature.protocols.turn import Action, ActionType


# Per-call hard timeout on LLM streaming. A local Ollama model that
# hangs mid-response (observed on qwen2.5-coder:32b with certain long
# tool_result inputs) will otherwise freeze execution indefinitely.
# Override via the `NATURE_LLM_TIMEOUT` env var; set to `0` or a
# negative value to disable the wrapper entirely.
DEFAULT_LLM_TIMEOUT_SECONDS = float(os.environ.get("NATURE_LLM_TIMEOUT", "300"))


class LLMCallTimeout(Exception):
    """An LLM streaming call exceeded its hard timeout.

    Raised when the provider's event stream hasn't produced a
    MESSAGE_STOP (or any event) within the configured window.
    AreaManager's exception handler translates this into both an
    `LLM_ERROR` trace event and a `FRAME_ERRORED` state event, and
    the frame transitions to `ERROR` — the session's next user input
    starts a fresh turn on top of the existing history.
    """


async def llm_agent(
    context: Context,
    *,
    self_actor: str,
    counterparty: str,
    model: str,
    provider: LLMProvider,
    tool_registry: list[Tool],
    composer: ContextComposer | None = None,
    cache_control: dict[str, str] | None = None,
    max_output_tokens: int | None = None,
) -> AgentOutput:
    """Make one LLM call and return structured AgentOutput.

    Pure async function: same inputs → same output (given a deterministic
    provider). Does NOT execute tools, does NOT loop, does NOT mutate
    context. The caller is responsible for everything state-related.

    Args:
        context: Full context the agent should see (header + body).
        self_actor: The agent's own actor name (e.g., "receptionist").
            Used by the composer to decide which messages are "mine"
            (role=assistant) vs incoming (role=user).
        counterparty: Who this agent is replying TO. Becomes the `to`
            field on any new Message produced.
        model: Model identifier to call.
        provider: Concrete LLMProvider to stream from.
        tool_registry: Full set of available tools. `role.allowed_tools`
            filters this at the composer boundary.
        composer: Optional ContextComposer override. Defaults to a fresh
            stock composer.
        cache_control: Optional cache policy forwarded to the provider.
        max_output_tokens: Optional output token limit.

    Returns:
        AgentOutput with exactly one new Message (the assistant response),
        any pending Actions (tool calls), annotations, and a Signal.
    """
    composer = composer or ContextComposer()

    composed = composer.compose(
        context,
        self_actor=self_actor,
        tool_registry=tool_registry,
        model=model,
        cache_control=cache_control,
        max_output_tokens=max_output_tokens,
        request_id=f"req_{uuid4().hex[:16]}",
    )
    request = composed.request

    # Accumulate the stream into raw primitives
    text_chunks: list[str] = []
    tool_uses: list[ToolUseContent] = []
    thinking_chunks: list[str] = []
    final_usage: Usage | None = None
    stop_reason: str | None = None

    # Per-call hard timeout around the provider stream. A
    # `nullcontext()` when disabled keeps the happy-path identical to
    # the legacy behavior. TimeoutError is converted to a domain
    # exception so AreaManager's handler can surface it as LLM_ERROR +
    # FRAME_ERRORED with a recognizable error_type.
    if DEFAULT_LLM_TIMEOUT_SECONDS > 0:
        timeout_ctx: contextlib.AbstractAsyncContextManager = asyncio.timeout(
            DEFAULT_LLM_TIMEOUT_SECONDS
        )
    else:
        timeout_ctx = contextlib.nullcontext()

    stream_started_at = time.time()
    try:
        async with timeout_ctx:
            async for event in provider.stream_request(request):
                if event.type == StreamEventType.CONTENT_BLOCK_DELTA:
                    if event.delta_text:
                        text_chunks.append(event.delta_text)
                elif event.type == StreamEventType.CONTENT_BLOCK_STOP and event.content_block:
                    block = event.content_block
                    if isinstance(block, ToolUseContent):
                        tool_uses.append(block)
                    elif isinstance(block, ThinkingContent) and block.thinking:
                        thinking_chunks.append(block.thinking)
                elif event.type == StreamEventType.MESSAGE_STOP:
                    if event.usage is not None:
                        final_usage = event.usage
                    if event.stop_reason is not None:
                        stop_reason = event.stop_reason
                elif event.type == StreamEventType.MESSAGE_DELTA:
                    if event.usage is not None:
                        final_usage = event.usage
                    if event.stop_reason is not None:
                        stop_reason = event.stop_reason
    except asyncio.TimeoutError as exc:
        raise LLMCallTimeout(
            f"LLM streaming exceeded {DEFAULT_LLM_TIMEOUT_SECONDS}s "
            f"(model={model}, request_id={request.request_id}, "
            f"self_actor={self_actor})"
        ) from exc

    text = "".join(text_chunks)

    # Fallback: some local / OpenAI-compat models emit tool calls as
    # text (`Agent(name="core", ...)` or fenced JSON) instead of using
    # the function-calling API. If we got no proper tool_use blocks but
    # the text contains tool-call-like content, extract it. The text
    # parser is role-aware — it only matches names this role is allowed
    # to call, so it can't surface tools the role isn't supposed to use.
    if not tool_uses and text:
        from nature.agent.text_tool_parser import extract_tool_calls_from_text

        allowed = context.header.role.allowed_tools
        if allowed is None:
            known_names = {t.name for t in tool_registry}
        else:
            known_names = set(allowed)

        if known_names:
            remaining_text, extracted = extract_tool_calls_from_text(
                text, known_names
            )
            if extracted:
                tool_uses = extracted
                text = remaining_text.strip()

    # Assemble the domain Message for this turn
    content: list[ContentBlock] = []
    if text:
        content.append(TextContent(text=text))
    content.extend(tool_uses)

    new_message = Message(
        from_=self_actor,
        to=counterparty,
        content=content,
        timestamp=time.time(),
    )

    annotation = MessageAnnotation(
        message_id=new_message.id,
        thinking=thinking_chunks or None,
        usage=final_usage,
        stop_reason=stop_reason,
        llm_request_id=request.request_id,
        duration_ms=int((time.time() - stream_started_at) * 1000),
    )

    actions = [
        Action(
            type=ActionType.TOOL_CALL,
            tool_name=tu.name,
            tool_input=tu.input,
            tool_use_id=tu.id,
        )
        for tu in tool_uses
    ]

    signal = _determine_signal(stop_reason, has_tool_uses=bool(tool_uses))

    return AgentOutput(
        new_messages=[new_message],
        actions=actions,
        annotations=[annotation],
        signal=signal,
        stop_reason=stop_reason,
        usage=final_usage,
        raw_request=request,
        hints=list(composed.hints),
    )


def _determine_signal(stop_reason: str | None, *, has_tool_uses: bool) -> Signal:
    """Translate (stop_reason, has_tool_uses) into a control Signal.

    Rules:
    - pending tool_uses              → CONTINUE (caller executes then re-calls)
    - MAX_TOKENS **with** tool_uses  → CONTINUE (tool cycle still in flight)
    - MAX_TOKENS **without** tool_use → RESOLVED (truncated synthesis; see
      note below — re-issuing the request with no follow-up user message
      would make Anthropic reject on "conversation must end with user
      message" and kill the frame)
    - END_TURN / STOP                → RESOLVED
    - anything else                  → CONTINUE (caller decides)

    The MAX_TOKENS-without-tool branch was previously CONTINUE "for
    recovery", but no continuation mechanism was ever wired — the
    agent loop simply re-invoked the LLM on the same message stack,
    which now ends with an assistant message (the truncated text).
    Stage 1 v3 reproduced the `invalid_request_error: This model does
    not support assistant message prefill. The conversation must end
    with a user message.` failure that followed every such case,
    killing the whole session. Treating truncation as a resolution
    hands the truncated synthesis to the caller, which can retry with
    tighter instructions if needed — much safer than looping into a
    guaranteed 400.
    """
    if has_tool_uses:
        return Signal.CONTINUE
    if stop_reason in (
        StopReason.END_TURN, StopReason.STOP, StopReason.MAX_TOKENS, None,
    ):
        return Signal.RESOLVED
    return Signal.CONTINUE
