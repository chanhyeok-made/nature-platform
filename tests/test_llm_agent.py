"""Tests for llm_agent — the pure agent function (Step 3 of the refactor)."""

from __future__ import annotations

from typing import Any, AsyncGenerator

import pytest

from nature.agent.llm_agent import llm_agent
from nature.agent.output import AgentOutput, Signal
from nature.config.constants import StopReason
from nature.context import (
    AgentRole,
    BasePrinciple,
    Context,
    ContextHeader,
    Message as DomainMessage,
)
from nature.protocols.message import (
    Role,
    StreamEvent,
    StreamEventType,
    TextContent,
    ThinkingContent,
    ToolUseContent,
    Usage,
)
from nature.protocols.turn import ActionType

from tests._fakes import FakeProvider, FakeTool as _FakeTool


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _text_response(text: str, stop_reason: str = StopReason.END_TURN) -> list[StreamEvent]:
    """A stream sequence for a plain text response."""
    return [
        StreamEvent(type=StreamEventType.MESSAGE_START),
        StreamEvent(
            type=StreamEventType.CONTENT_BLOCK_START,
            index=0,
            content_block=TextContent(text=""),
        ),
        StreamEvent(
            type=StreamEventType.CONTENT_BLOCK_DELTA,
            index=0,
            delta_text=text,
        ),
        StreamEvent(type=StreamEventType.CONTENT_BLOCK_STOP, index=0),
        StreamEvent(
            type=StreamEventType.MESSAGE_STOP,
            usage=Usage(input_tokens=10, output_tokens=5),
            stop_reason=stop_reason,
        ),
    ]


def _tool_use_response(
    tool_name: str,
    tool_input: dict,
    tool_id: str = "toolu_test1",
    leading_text: str = "",
) -> list[StreamEvent]:
    """A stream sequence ending in a tool_use block."""
    events: list[StreamEvent] = [StreamEvent(type=StreamEventType.MESSAGE_START)]
    if leading_text:
        events += [
            StreamEvent(
                type=StreamEventType.CONTENT_BLOCK_START,
                index=0,
                content_block=TextContent(text=""),
            ),
            StreamEvent(
                type=StreamEventType.CONTENT_BLOCK_DELTA,
                index=0,
                delta_text=leading_text,
            ),
            StreamEvent(type=StreamEventType.CONTENT_BLOCK_STOP, index=0),
        ]
    tool_block = ToolUseContent(id=tool_id, name=tool_name, input=tool_input)
    events += [
        StreamEvent(
            type=StreamEventType.CONTENT_BLOCK_START,
            index=1,
            content_block=tool_block,
        ),
        StreamEvent(
            type=StreamEventType.CONTENT_BLOCK_STOP,
            index=1,
            content_block=tool_block,
        ),
        StreamEvent(
            type=StreamEventType.MESSAGE_STOP,
            usage=Usage(input_tokens=20, output_tokens=10),
            stop_reason=StopReason.TOOL_USE,
        ),
    ]
    return events


def _make_context(
    role_name: str = "receptionist",
    instructions: str = "be helpful",
    allowed_tools: list[str] | None = None,
    principles: list[BasePrinciple] | None = None,
    initial_user_text: str | None = "hello",
) -> Context:
    ctx = Context(
        header=ContextHeader(
            role=AgentRole(
                name=role_name,
                instructions=instructions,
                allowed_tools=allowed_tools,
            ),
            principles=principles or [],
        )
    )
    if initial_user_text:
        ctx.body.conversation.append(
            DomainMessage(
                from_="user",
                to=role_name,
                content=[TextContent(text=initial_user_text)],
                timestamp=1.0,
            )
        )
    return ctx


# ---------------------------------------------------------------------------
# Pure text response
# ---------------------------------------------------------------------------


async def test_text_response_produces_one_message_and_resolved_signal():
    provider = FakeProvider(_text_response("hi there"))
    ctx = _make_context()

    out = await llm_agent(
        ctx,
        self_actor="receptionist",
        counterparty="user",
        model="fake",
        provider=provider,
        tool_registry=[],
    )

    assert isinstance(out, AgentOutput)
    assert out.signal == Signal.RESOLVED
    assert len(out.new_messages) == 1

    msg = out.new_messages[0]
    assert msg.from_ == "receptionist"
    assert msg.to == "user"
    assert len(msg.content) == 1
    assert isinstance(msg.content[0], TextContent)
    assert msg.content[0].text == "hi there"
    assert out.actions == []


async def test_text_response_annotation_captures_usage_and_stop_reason():
    provider = FakeProvider(_text_response("ok"))
    ctx = _make_context()

    out = await llm_agent(
        ctx,
        self_actor="r",
        counterparty="user",
        model="fake",
        provider=provider,
        tool_registry=[],
    )

    assert len(out.annotations) == 1
    ann = out.annotations[0]
    assert ann.message_id == out.new_messages[0].id
    assert ann.stop_reason == StopReason.END_TURN
    assert ann.usage is not None
    assert ann.usage.input_tokens == 10
    assert ann.usage.output_tokens == 5
    assert ann.llm_request_id is not None


# ---------------------------------------------------------------------------
# Tool use response
# ---------------------------------------------------------------------------


async def test_tool_use_produces_actions_and_continue_signal():
    provider = FakeProvider(
        _tool_use_response("Read", {"file_path": "/tmp/x"}, tool_id="toolu_1")
    )
    ctx = _make_context()

    out = await llm_agent(
        ctx,
        self_actor="r",
        counterparty="user",
        model="fake",
        provider=provider,
        tool_registry=[_FakeTool("Read")],
    )

    assert out.signal == Signal.CONTINUE
    assert len(out.actions) == 1
    action = out.actions[0]
    assert action.type == ActionType.TOOL_CALL
    assert action.tool_name == "Read"
    assert action.tool_input == {"file_path": "/tmp/x"}
    assert action.tool_use_id == "toolu_1"
    assert action.executed is False


async def test_tool_use_message_has_text_then_tool_use_blocks():
    provider = FakeProvider(
        _tool_use_response(
            "Read", {"path": "x"}, tool_id="toolu_2", leading_text="looking up x"
        )
    )
    ctx = _make_context()

    out = await llm_agent(
        ctx,
        self_actor="r",
        counterparty="user",
        model="fake",
        provider=provider,
        tool_registry=[_FakeTool("Read")],
    )

    msg = out.new_messages[0]
    assert len(msg.content) == 2
    assert isinstance(msg.content[0], TextContent)
    assert msg.content[0].text == "looking up x"
    assert isinstance(msg.content[1], ToolUseContent)
    assert msg.content[1].name == "Read"


# ---------------------------------------------------------------------------
# Signal translation
# ---------------------------------------------------------------------------


async def test_max_tokens_without_tool_use_is_resolved():
    """Truncated text-only responses must not loop back into the LLM.

    Regression: stage 1 v3 n2×all-sonnet hit max_tokens mid-synthesis
    (researcher output 8192 tokens, stopping on a bare assistant
    message). The previous rule classified this as CONTINUE for
    'recovery', but nothing in the loop appends a user message before
    re-invoking the provider — Anthropic then rejected the request
    with `invalid_request_error: conversation must end with a user
    message` and the whole frame errored. RESOLVED hands the
    truncated synthesis to the caller, which is a safe degradation.
    """
    provider = FakeProvider(_text_response("partial", stop_reason=StopReason.MAX_TOKENS))
    ctx = _make_context()

    out = await llm_agent(
        ctx, self_actor="r", counterparty="user",
        model="fake", provider=provider, tool_registry=[],
    )

    assert out.signal == Signal.RESOLVED
    assert out.stop_reason == StopReason.MAX_TOKENS


async def test_end_turn_is_resolved():
    provider = FakeProvider(_text_response("done", stop_reason=StopReason.END_TURN))
    ctx = _make_context()

    out = await llm_agent(
        ctx, self_actor="r", counterparty="user",
        model="fake", provider=provider, tool_registry=[],
    )

    assert out.signal == Signal.RESOLVED


async def test_missing_stop_reason_defaults_to_resolved_when_no_tools():
    events = [
        StreamEvent(type=StreamEventType.MESSAGE_START),
        StreamEvent(
            type=StreamEventType.CONTENT_BLOCK_START,
            index=0,
            content_block=TextContent(text=""),
        ),
        StreamEvent(type=StreamEventType.CONTENT_BLOCK_DELTA, index=0, delta_text="ok"),
        StreamEvent(type=StreamEventType.CONTENT_BLOCK_STOP, index=0),
        StreamEvent(type=StreamEventType.MESSAGE_STOP),  # no stop_reason
    ]
    provider = FakeProvider(events)
    ctx = _make_context()

    out = await llm_agent(
        ctx, self_actor="r", counterparty="user",
        model="fake", provider=provider, tool_registry=[],
    )

    assert out.signal == Signal.RESOLVED
    assert out.stop_reason is None


# ---------------------------------------------------------------------------
# Thinking blocks
# ---------------------------------------------------------------------------


async def test_thinking_blocks_captured_in_annotation():
    thinking_block = ThinkingContent(thinking="let me think...")
    events = [
        StreamEvent(type=StreamEventType.MESSAGE_START),
        StreamEvent(
            type=StreamEventType.CONTENT_BLOCK_START,
            index=0,
            content_block=thinking_block,
        ),
        StreamEvent(
            type=StreamEventType.CONTENT_BLOCK_STOP,
            index=0,
            content_block=thinking_block,
        ),
        StreamEvent(
            type=StreamEventType.CONTENT_BLOCK_START,
            index=1,
            content_block=TextContent(text=""),
        ),
        StreamEvent(type=StreamEventType.CONTENT_BLOCK_DELTA, index=1, delta_text="done"),
        StreamEvent(type=StreamEventType.CONTENT_BLOCK_STOP, index=1),
        StreamEvent(
            type=StreamEventType.MESSAGE_STOP,
            usage=Usage(input_tokens=1, output_tokens=1),
            stop_reason=StopReason.END_TURN,
        ),
    ]
    provider = FakeProvider(events)
    ctx = _make_context()

    out = await llm_agent(
        ctx, self_actor="r", counterparty="user",
        model="fake", provider=provider, tool_registry=[],
    )

    assert out.annotations[0].thinking == ["let me think..."]


# ---------------------------------------------------------------------------
# Request construction (composer integration)
# ---------------------------------------------------------------------------


async def test_request_includes_system_from_role_and_principles():
    ctx = _make_context(
        instructions="core role",
        principles=[
            BasePrinciple(text="low", priority=1),
            BasePrinciple(text="high", priority=10),
        ],
    )
    provider = FakeProvider(_text_response("ok"))

    await llm_agent(
        ctx, self_actor="core", counterparty="user",
        model="fake", provider=provider, tool_registry=[],
    )

    assert provider.last_request is not None
    assert provider.last_request.system == ["core role", "high", "low"]


async def test_request_tools_filtered_by_role_allowed_tools():
    ctx = _make_context(allowed_tools=["Read"])
    provider = FakeProvider(_text_response("ok"))
    registry = [_FakeTool("Read"), _FakeTool("Bash"), _FakeTool("Write")]

    await llm_agent(
        ctx, self_actor="r", counterparty="user",
        model="fake", provider=provider, tool_registry=registry,
    )

    req = provider.last_request
    assert req is not None
    assert req.tools is not None
    assert {t.name for t in req.tools} == {"Read"}


async def test_conversation_mapped_as_user_role_when_not_from_self():
    provider = FakeProvider(_text_response("ok"))
    ctx = _make_context(initial_user_text="the initial question")

    await llm_agent(
        ctx, self_actor="receptionist", counterparty="user",
        model="fake", provider=provider, tool_registry=[],
    )

    req = provider.last_request
    assert req is not None
    assert len(req.messages) == 1
    assert req.messages[0].role == Role.USER
    assert req.messages[0].content[0].text == "the initial question"  # type: ignore[attr-defined]


async def test_request_passes_model_and_cache_control():
    provider = FakeProvider(_text_response("ok"))
    ctx = _make_context()

    await llm_agent(
        ctx, self_actor="r", counterparty="user",
        model="claude-sonnet-4", provider=provider, tool_registry=[],
        cache_control={"type": "ephemeral"},
        max_output_tokens=4096,
    )

    req = provider.last_request
    assert req.model == "claude-sonnet-4"
    assert req.cache_control == {"type": "ephemeral"}
    assert req.max_output_tokens == 4096
    assert req.request_id is not None
    assert req.request_id.startswith("req_")


async def test_new_message_counterparty_controls_to_field():
    provider = FakeProvider(_text_response("result"))
    ctx = _make_context(role_name="core")

    out = await llm_agent(
        ctx, self_actor="core", counterparty="receptionist",
        model="fake", provider=provider, tool_registry=[],
    )

    assert out.new_messages[0].to == "receptionist"
    assert out.new_messages[0].from_ == "core"


# ---------------------------------------------------------------------------
# Text-tool-call fallback for local / OpenAI-compat models
# ---------------------------------------------------------------------------


async def test_text_tool_call_fallback_extracts_python_call_syntax():
    """Model emits `Agent(name="core", prompt="...")` as text → extract it."""
    text = 'I will delegate. Agent(name="core", prompt="analyze the loop")'
    provider = FakeProvider(_text_response(text))
    ctx = _make_context(allowed_tools=None)

    out = await llm_agent(
        ctx, self_actor="receptionist", counterparty="user",
        model="fake", provider=provider,
        tool_registry=[_FakeTool("Agent")],
    )

    assert out.signal == Signal.CONTINUE
    assert len(out.actions) == 1
    assert out.actions[0].tool_name == "Agent"
    assert out.actions[0].tool_input.get("name") == "core"
    assert out.actions[0].tool_input.get("prompt") == "analyze the loop"


async def test_text_tool_call_fallback_extracts_json_in_fenced_block():
    """Model emits a fenced JSON tool call → extract it."""
    text = '```json\n{"name": "Agent", "arguments": {"name": "core", "prompt": "go"}}\n```'
    provider = FakeProvider(_text_response(text))
    ctx = _make_context(allowed_tools=None)

    out = await llm_agent(
        ctx, self_actor="receptionist", counterparty="user",
        model="fake", provider=provider,
        tool_registry=[_FakeTool("Agent")],
    )

    assert out.signal == Signal.CONTINUE
    assert len(out.actions) == 1
    assert out.actions[0].tool_name == "Agent"


async def test_text_tool_call_fallback_respects_role_allowed_tools():
    """Fallback only matches names in role.allowed_tools — security boundary."""
    text = 'Bash(command="ls /")'
    provider = FakeProvider(_text_response(text))
    ctx = _make_context(allowed_tools=["Read"])

    out = await llm_agent(
        ctx, self_actor="receptionist", counterparty="user",
        model="fake", provider=provider,
        tool_registry=[_FakeTool("Bash"), _FakeTool("Read")],
    )

    # Bash text not extracted because the role doesn't allow it
    assert out.signal == Signal.RESOLVED
    assert len(out.actions) == 0
    assert "Bash" in out.new_messages[0].content[0].text  # type: ignore[attr-defined]


async def test_text_tool_call_fallback_loose_agent_prose_with_args_dict():
    """qwen-style: descriptive prose + raw args dict (no Agent(...) wrapper).

    Real example from a session:
        '-Agent tool call with the following parameters: {"name": "core",
        "prompt": "위 프로젝트를 철저히 분석..."}'
    """
    text = (
        '-Agent tool call with the following parameters: '
        '{"name": "core", "prompt": "analyze the project thoroughly"}'
    )
    provider = FakeProvider(_text_response(text))
    ctx = _make_context(allowed_tools=None)

    out = await llm_agent(
        ctx, self_actor="receptionist", counterparty="user",
        model="fake", provider=provider,
        tool_registry=[_FakeTool("Agent")],
    )

    assert out.signal == Signal.CONTINUE
    assert len(out.actions) == 1
    action = out.actions[0]
    assert action.tool_name == "Agent"
    assert action.tool_input.get("name") == "core"
    assert action.tool_input.get("prompt") == "analyze the project thoroughly"


async def test_text_tool_call_fallback_loose_agent_without_name_is_skipped():
    """Loose Agent fallback used to default `name="core"` when missing,
    which silently produced self-loops in multi-specialist setups
    (session 8ed7d997). The fallback now skips nameless calls so the
    framework's `_handle_delegation` validator can force the model
    to retry with an explicit specialist name.
    """
    text = 'I will delegate. Calling Agent: {"prompt": "do the thing"}'
    provider = FakeProvider(_text_response(text))
    ctx = _make_context(allowed_tools=None)

    out = await llm_agent(
        ctx, self_actor="receptionist", counterparty="user",
        model="fake", provider=provider,
        tool_registry=[_FakeTool("Agent")],
    )

    # No Agent action should be emitted; the fallback skipped the
    # nameless call and the response resolves with no tool uses.
    assert out.actions == []


async def test_text_tool_call_fallback_loose_agent_requires_agent_in_text():
    """Random JSON with `prompt` key without `Agent` mention is NOT extracted."""
    text = 'Here is some config: {"prompt": "engineering", "name": "alice"}'
    provider = FakeProvider(_text_response(text))
    ctx = _make_context(allowed_tools=None)

    out = await llm_agent(
        ctx, self_actor="receptionist", counterparty="user",
        model="fake", provider=provider,
        tool_registry=[_FakeTool("Agent")],
    )

    # Not extracted — no Agent context in the text
    assert out.signal == Signal.RESOLVED
    assert out.actions == []


async def test_text_tool_call_fallback_skipped_when_real_tool_use_present():
    """If the model used the proper tool_use API, fallback shouldn't double-trigger."""
    events = [
        StreamEvent(type=StreamEventType.MESSAGE_START),
        StreamEvent(
            type=StreamEventType.CONTENT_BLOCK_START,
            index=0,
            content_block=TextContent(text=""),
        ),
        StreamEvent(
            type=StreamEventType.CONTENT_BLOCK_DELTA,
            index=0,
            delta_text='Agent(name="core", prompt="x")',  # text mentions Agent
        ),
        StreamEvent(type=StreamEventType.CONTENT_BLOCK_STOP, index=0),
        StreamEvent(
            type=StreamEventType.CONTENT_BLOCK_START,
            index=1,
            content_block=ToolUseContent(
                id="toolu_real", name="Read", input={"path": "/x"}
            ),
        ),
        StreamEvent(
            type=StreamEventType.CONTENT_BLOCK_STOP,
            index=1,
            content_block=ToolUseContent(
                id="toolu_real", name="Read", input={"path": "/x"}
            ),
        ),
        StreamEvent(
            type=StreamEventType.MESSAGE_STOP,
            usage=Usage(input_tokens=1, output_tokens=1),
            stop_reason=StopReason.TOOL_USE,
        ),
    ]
    provider = FakeProvider(events)
    ctx = _make_context(allowed_tools=None)

    out = await llm_agent(
        ctx, self_actor="r", counterparty="user",
        model="fake", provider=provider,
        tool_registry=[_FakeTool("Agent"), _FakeTool("Read")],
    )

    # Only the real tool_use survives — text-extracted Agent is ignored
    assert len(out.actions) == 1
    assert out.actions[0].tool_name == "Read"
    assert out.actions[0].tool_use_id == "toolu_real"


# ---------------------------------------------------------------------------
# Timeout / self-heal path (#4)
# ---------------------------------------------------------------------------


class _SlowProvider(FakeProvider):
    """Fake provider whose stream takes longer than any reasonable
    timeout. Used to verify `asyncio.timeout` kicks in and llm_agent
    raises LLMCallTimeout."""

    async def stream_request(self, request):  # type: ignore[override]
        self.requests.append(request)
        self.last_request = request
        import asyncio as _asyncio
        # Yield MESSAGE_START so the consumer sees *something* before
        # the stall, then block forever — mimics Ollama hanging mid-stream
        yield StreamEvent(type=StreamEventType.MESSAGE_START)
        await _asyncio.sleep(60)
        yield StreamEvent(  # pragma: no cover — never reached
            type=StreamEventType.MESSAGE_STOP,
            usage=Usage(input_tokens=1, output_tokens=0),
            stop_reason=StopReason.END_TURN,
        )


async def test_llm_agent_raises_llmcalltimeout_on_stream_stall(monkeypatch):
    """A provider that stalls mid-stream must surface as LLMCallTimeout
    after the configured timeout window."""
    import nature.agent.llm_agent as la

    monkeypatch.setattr(la, "DEFAULT_LLM_TIMEOUT_SECONDS", 0.15)

    provider = _SlowProvider(_text_response("unused"))
    ctx = _make_context()

    with pytest.raises(la.LLMCallTimeout) as exc_info:
        await la.llm_agent(
            ctx, self_actor="r", counterparty="user",
            model="fake-slow", provider=provider,
            tool_registry=[],
        )
    msg = str(exc_info.value)
    assert "0.15s" in msg
    assert "fake-slow" in msg


async def test_llm_agent_disables_timeout_when_set_to_zero(monkeypatch):
    """A `NATURE_LLM_TIMEOUT=0` disables the wrapper entirely — the
    happy-path clean stream still returns normally."""
    import nature.agent.llm_agent as la

    monkeypatch.setattr(la, "DEFAULT_LLM_TIMEOUT_SECONDS", 0)

    provider = FakeProvider(_text_response("hello"))
    ctx = _make_context()

    out = await la.llm_agent(
        ctx, self_actor="r", counterparty="user",
        model="fake", provider=provider,
        tool_registry=[],
    )
    assert out.signal is Signal.RESOLVED
    assert out.new_messages[0].content[0].text == "hello"  # type: ignore[attr-defined]
