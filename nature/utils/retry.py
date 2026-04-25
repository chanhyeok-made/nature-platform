"""Retry utilities with exponential backoff."""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# HTTP status codes that indicate transient failures
TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 529}


class RetryableError(Exception):
    """An error that can be retried."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def is_transient_error(error: Exception) -> bool:
    """Check if an error is transient and should be retried."""
    if isinstance(error, RetryableError):
        return True
    # Check for common HTTP library errors
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    if isinstance(status, int) and status in TRANSIENT_STATUS_CODES:
        return True
    # anthropic SDK overloaded error
    type_name = type(error).__name__
    if type_name in ("RateLimitError", "InternalServerError", "APIConnectionError"):
        return True
    return False


async def retry_with_backoff(
    fn: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
) -> T:
    """Execute an async function with exponential backoff on transient errors.

    Args:
        fn: Async callable to retry.
        max_retries: Maximum number of retries (0 = no retries).
        base_delay: Initial delay in seconds.
        max_delay: Maximum delay in seconds.
        backoff_factor: Multiply delay by this each retry.

    Returns:
        Result of fn().

    Raises:
        The last exception if all retries fail.
    """
    last_error: Exception | None = None
    delay = base_delay

    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except Exception as e:
            last_error = e
            if attempt >= max_retries or not is_transient_error(e):
                raise
            logger.warning(
                "Retry %d/%d after %s: %.1fs delay",
                attempt + 1,
                max_retries,
                type(e).__name__,
                delay,
            )
            await asyncio.sleep(delay)
            delay = min(delay * backoff_factor, max_delay)

    raise last_error  # type: ignore[misc]  # unreachable but satisfies type checker
