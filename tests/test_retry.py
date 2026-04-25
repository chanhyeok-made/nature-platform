"""Tests for retry utilities."""

import pytest

from nature.utils.retry import RetryableError, is_transient_error, retry_with_backoff


class TestIsTransientError:
    def test_retryable_error(self):
        assert is_transient_error(RetryableError("test")) is True

    def test_rate_limit_by_name(self):
        class RateLimitError(Exception):
            pass

        assert is_transient_error(RateLimitError()) is True

    def test_status_code_429(self):
        class HttpError(Exception):
            status_code = 429

        assert is_transient_error(HttpError()) is True

    def test_non_transient(self):
        assert is_transient_error(ValueError("bad input")) is False


class TestRetryWithBackoff:
    @pytest.mark.asyncio
    async def test_success_no_retry(self):
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await retry_with_backoff(fn, max_retries=3, base_delay=0.01)
        assert result == "ok"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_then_success(self):
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RetryableError("transient")
            return "ok"

        result = await retry_with_backoff(fn, max_retries=3, base_delay=0.01)
        assert result == "ok"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_non_retryable_raises_immediately(self):
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            raise ValueError("permanent")

        with pytest.raises(ValueError, match="permanent"):
            await retry_with_backoff(fn, max_retries=3, base_delay=0.01)
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_exhausted_retries(self):
        async def fn():
            raise RetryableError("always fails")

        with pytest.raises(RetryableError):
            await retry_with_backoff(fn, max_retries=2, base_delay=0.01)
