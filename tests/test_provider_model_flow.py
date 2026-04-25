"""Regression test: per-request model flows from LLMRequest through
stream_request → stream → concrete provider API call.

This was a latent bug in the refactor — LLMRequest.model was populated
by ContextComposer but silently ignored by both AnthropicProvider and
OpenAICompatProvider, which baked the model in at construction time.
"""

from __future__ import annotations

from typing import Any, AsyncGenerator

from nature.protocols.llm import LLMRequest
from nature.protocols.message import (
    Message as LLMMessage,
    StreamEvent,
    StreamEventType,
)
from nature.protocols.provider import CacheControl, LLMProvider
from nature.protocols.tool import ToolDefinition


class CapturingProvider(LLMProvider):
    """Minimal provider that records the model it was called with."""

    def __init__(self) -> None:
        self.captured_model: str | None = None
        self.call_count = 0

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
        self.captured_model = model
        self.call_count += 1
        yield StreamEvent(type=StreamEventType.MESSAGE_STOP)

    async def count_tokens(self, messages, system, tools=None) -> int:
        return 0

    @property
    def model_id(self) -> str:
        return "capturing"


async def test_default_stream_request_passes_request_model_to_stream():
    """ABC default stream_request must hand request.model to stream()."""
    provider = CapturingProvider()
    request = LLMRequest(
        messages=[],
        system=[],
        model="per-agent-model",
    )

    async for _ in provider.stream_request(request):
        pass

    assert provider.call_count == 1
    assert provider.captured_model == "per-agent-model"


async def test_default_stream_request_with_none_model_passes_none():
    """When request.model is None, stream() receives None — provider
    then falls back to its config.model. Legacy behavior preserved."""
    provider = CapturingProvider()
    request = LLMRequest(messages=[], system=[], model=None)

    async for _ in provider.stream_request(request):
        pass

    assert provider.captured_model is None


async def test_anthropic_provider_uses_request_model_when_set():
    """AnthropicProvider.stream must respect the model kwarg.

    We can't actually call the Anthropic API in a unit test, so this
    verifies the code path by inspecting the internal request_kwargs
    construction via monkeypatching the SDK client.
    """
    try:
        import anthropic  # noqa: F401
    except ImportError:
        import pytest
        pytest.skip("anthropic package not installed")

    from nature.protocols.provider import ProviderConfig
    from nature.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider(
        ProviderConfig(model="claude-sonnet-4-20250514", api_key="sk-ant-fake"),
    )

    captured_kwargs: dict = {}

    class _FakeStreamCtx:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def get_final_message(self):
            class _U:
                input_tokens = 0
                output_tokens = 0
                cache_creation_input_tokens = 0
                cache_read_input_tokens = 0

            class _M:
                usage = _U()
                stop_reason = "end_turn"

            return _M()

    def _fake_stream(**kwargs):
        captured_kwargs.update(kwargs)
        return _FakeStreamCtx()

    provider._client.messages.stream = _fake_stream  # type: ignore[assignment]

    # Call stream() directly with an override model
    async for _ in provider.stream(
        messages=[],
        system=[],
        tools=None,
        model="per-agent-override",
    ):
        pass

    assert captured_kwargs.get("model") == "per-agent-override"


async def test_anthropic_provider_falls_back_to_config_model_when_no_override():
    """When model=None, AnthropicProvider uses its constructed config."""
    try:
        import anthropic  # noqa: F401
    except ImportError:
        import pytest
        pytest.skip("anthropic package not installed")

    from nature.protocols.provider import ProviderConfig
    from nature.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider(
        ProviderConfig(model="claude-sonnet-4-20250514", api_key="sk-ant-fake"),
    )

    captured_kwargs: dict = {}

    class _FakeStreamCtx:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def get_final_message(self):
            class _U:
                input_tokens = 0
                output_tokens = 0
                cache_creation_input_tokens = 0
                cache_read_input_tokens = 0

            class _M:
                usage = _U()
                stop_reason = "end_turn"

            return _M()

    def _fake_stream(**kwargs):
        captured_kwargs.update(kwargs)
        return _FakeStreamCtx()

    provider._client.messages.stream = _fake_stream  # type: ignore[assignment]

    async for _ in provider.stream(messages=[], system=[], tools=None, model=None):
        pass

    assert captured_kwargs.get("model") == "claude-sonnet-4-20250514"
