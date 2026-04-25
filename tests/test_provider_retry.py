"""Provider-level retry on transient failures.

Stage 1 v3 eval run surfaced `APIStatusError: Overloaded` killing
sessions mid-matrix with no retry — one `s3 × all-sonnet` cell
registered FAIL purely because Anthropic's servers were overloaded
for a few seconds. Tests here pin the retry contract: classifier
picks the right error families, and the stream wrapper retries up to
`DEFAULT_MAX_ATTEMPTS` times before surfacing failure.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from nature.providers.retry import (
    DEFAULT_MAX_ATTEMPTS,
    is_retryable_anthropic_error,
)


# ──────────────────────────────────────────────────────────────────────
# Classifier — which exception types are retried?
# ──────────────────────────────────────────────────────────────────────


def test_overloaded_error_is_retryable():
    from anthropic import APIStatusError
    # APIStatusError needs a response object; the classifier only
    # reads `status_code` and `body`, so a minimal stand-in works.
    exc = APIStatusError.__new__(APIStatusError)
    exc.status_code = 529
    exc.body = {"error": {"type": "overloaded_error", "message": "Overloaded"}}
    assert is_retryable_anthropic_error(exc)


def test_rate_limit_error_is_retryable():
    from anthropic import APIStatusError
    exc = APIStatusError.__new__(APIStatusError)
    exc.status_code = 429
    exc.body = {"error": {"type": "rate_limit_error", "message": "Too many requests"}}
    assert is_retryable_anthropic_error(exc)


def test_server_5xx_is_retryable():
    from anthropic import APIStatusError
    exc = APIStatusError.__new__(APIStatusError)
    exc.status_code = 503
    exc.body = {"error": {"type": "api_error", "message": "Service unavailable"}}
    assert is_retryable_anthropic_error(exc)


def test_timeout_error_is_retryable():
    from anthropic import APITimeoutError
    # APITimeoutError.__init__ needs a Request; fake via __new__.
    exc = APITimeoutError.__new__(APITimeoutError)
    assert is_retryable_anthropic_error(exc)


def test_400_bad_request_is_not_retryable():
    from anthropic import APIStatusError
    exc = APIStatusError.__new__(APIStatusError)
    exc.status_code = 400
    exc.body = {"error": {"type": "invalid_request_error", "message": "Bad request"}}
    assert not is_retryable_anthropic_error(exc)


def test_401_auth_is_not_retryable():
    from anthropic import APIStatusError
    exc = APIStatusError.__new__(APIStatusError)
    exc.status_code = 401
    exc.body = {"error": {"type": "authentication_error", "message": "Invalid API key"}}
    assert not is_retryable_anthropic_error(exc)


def test_unrelated_exception_is_not_retryable():
    assert not is_retryable_anthropic_error(ValueError("nope"))


# ──────────────────────────────────────────────────────────────────────
# Stream wrapper — transient error triggers re-attempt
# ──────────────────────────────────────────────────────────────────────


class _OverloadThenSucceed:
    """Stand-in for `self._client.messages.stream(...)` that returns a
    context manager which raises `APIStatusError(overloaded)` on the
    first N `__aenter__` calls, then yields a normal stream on the
    next one."""

    def __init__(self, fail_first_n: int):
        self.fail_first_n = fail_first_n
        self.call_count = 0

    def __call__(self, **request_kwargs):
        self.call_count += 1
        return _StreamCtx(self, self.call_count)


class _StreamCtx:
    def __init__(self, parent: _OverloadThenSucceed, call_number: int):
        self.parent = parent
        self.call_number = call_number

    async def __aenter__(self):
        if self.call_number <= self.parent.fail_first_n:
            from anthropic import APIStatusError
            exc = APIStatusError.__new__(APIStatusError)
            exc.status_code = 529
            exc.body = {"error": {"type": "overloaded_error", "message": "Overloaded"}}
            raise exc
        return _SuccessStream()

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _SuccessStream:
    """Minimal async stream that yields zero SDK events and returns a
    final_message with the right shape. The nature provider will emit
    MESSAGE_START, no content, and MESSAGE_STOP."""

    def __aiter__(self):
        async def _empty():
            if False:
                yield None  # make this a generator
        return _empty()

    async def get_final_message(self):
        msg = MagicMock()
        msg.usage.input_tokens = 0
        msg.usage.output_tokens = 0
        msg.usage.cache_creation_input_tokens = 0
        msg.usage.cache_read_input_tokens = 0
        msg.stop_reason = "end_turn"
        return msg


@pytest.fixture
def provider_with_fake_client():
    from nature.providers.anthropic import AnthropicProvider
    from nature.protocols.provider import ProviderConfig

    cfg = ProviderConfig(model="claude-haiku-4-5", api_key="fake")
    p = AnthropicProvider(cfg)
    # Replace the client with a MagicMock so we control .messages.stream
    p._client = MagicMock()
    return p


async def test_stream_retries_until_success(provider_with_fake_client):
    """Two overloaded_error failures then success → three attempts,
    one final clean stream."""
    p = provider_with_fake_client
    fake = _OverloadThenSucceed(fail_first_n=2)
    p._client.messages.stream = fake

    events = []
    async for ev in p.stream(messages=[], system=[], tools=None):
        events.append(ev)

    assert fake.call_count == 3
    # At minimum MESSAGE_START + MESSAGE_STOP yielded.
    types = [e.type for e in events]
    from nature.protocols.message import StreamEventType
    assert StreamEventType.MESSAGE_START in types
    assert StreamEventType.MESSAGE_STOP in types


async def test_stream_gives_up_after_max_attempts(provider_with_fake_client):
    """All DEFAULT_MAX_ATTEMPTS fail → exception surfaces to caller."""
    p = provider_with_fake_client
    fake = _OverloadThenSucceed(fail_first_n=DEFAULT_MAX_ATTEMPTS)
    p._client.messages.stream = fake

    from anthropic import APIStatusError
    with pytest.raises(APIStatusError):
        async for _ in p.stream(messages=[], system=[], tools=None):
            pass

    assert fake.call_count == DEFAULT_MAX_ATTEMPTS


async def test_stream_passes_through_non_retryable_immediately(
    provider_with_fake_client,
):
    """A 400 / invalid_request_error must NOT be retried — it would
    fail identically on every retry and just waste time."""
    from anthropic import APIStatusError

    p = provider_with_fake_client
    call_count = 0

    class _BadRequestCtx:
        def __init__(self):
            pass
        async def __aenter__(self):
            nonlocal call_count
            call_count += 1
            exc = APIStatusError.__new__(APIStatusError)
            exc.status_code = 400
            exc.body = {"error": {"type": "invalid_request_error", "message": "bad"}}
            raise exc
        async def __aexit__(self, *a):
            return False

    p._client.messages.stream = MagicMock(return_value=_BadRequestCtx())

    with pytest.raises(APIStatusError):
        async for _ in p.stream(messages=[], system=[], tools=None):
            pass

    # Called exactly once — not retried.
    assert call_count == 1
