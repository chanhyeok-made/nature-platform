"""Retry helpers for transient provider failures.

Anthropic's overloaded_error (HTTP 529), rate-limit responses, 5xx
server errors, connection drops and timeouts are all transient:
observed once in a stage-1 eval run as `APIStatusError: Overloaded`
that killed an entire session just before any content streamed.
Re-issuing the same request a few seconds later almost always
succeeds — so the provider layer catches the classifiable transient
errors and retries with exponential backoff.

Retry only triggers BEFORE the first stream event yields. Once the
model has started emitting tokens, retrying would double-yield into
the caller's async iterator, so we don't attempt it.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_INITIAL_BACKOFF_SECONDS = 2.0


def is_retryable_anthropic_error(exc: BaseException) -> bool:
    """Classify an Anthropic-SDK exception as transient (should retry).

    Retryable:
    - APITimeoutError / APIConnectionError (network + SDK timeout)
    - APIStatusError with status 429 (rate limit), 529 (overloaded),
      5xx server errors
    - APIStatusError whose body `error.type` is `overloaded_error`,
      `api_error`, or `rate_limit_error` (fallback when status code
      parsing doesn't set the attribute as expected)

    Not retryable:
    - 4xx client errors other than 429 (malformed request, auth, etc.)
    - Anything unrelated to the provider layer.
    """
    try:
        from anthropic import (
            APIConnectionError,
            APIStatusError,
            APITimeoutError,
        )
    except ImportError:  # pragma: no cover - anthropic always installed
        return False

    if isinstance(exc, (APITimeoutError, APIConnectionError)):
        return True
    if isinstance(exc, APIStatusError):
        status = getattr(exc, "status_code", None)
        if status in (429, 529):
            return True
        if isinstance(status, int) and 500 <= status < 600:
            return True
        body = getattr(exc, "body", None) or {}
        err = body.get("error") if isinstance(body, dict) else None
        err_type = err.get("type") if isinstance(err, dict) else None
        if err_type in ("overloaded_error", "api_error", "rate_limit_error"):
            return True
    return False


def is_retryable_openai_error(exc: BaseException) -> bool:
    """Classify an OpenAI-SDK exception as transient.

    Covers the OpenAI-compat path used by local Ollama and proxied
    OpenAI endpoints. The structure mirrors Anthropic's but the
    class names differ.
    """
    try:
        from openai import (  # type: ignore
            APIConnectionError,
            APIStatusError,
            APITimeoutError,
        )
    except ImportError:  # pragma: no cover
        return False

    if isinstance(exc, (APITimeoutError, APIConnectionError)):
        return True
    if isinstance(exc, APIStatusError):
        status = getattr(exc, "status_code", None)
        if status in (429, 529):
            return True
        if isinstance(status, int) and 500 <= status < 600:
            return True
    return False


async def sleep_backoff(attempt: int, initial: float = DEFAULT_INITIAL_BACKOFF_SECONDS) -> None:
    """Exponential backoff: 2s, 4s, 8s for attempt=0,1,2."""
    await asyncio.sleep(initial * (2 ** attempt))
