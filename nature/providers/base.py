"""Base provider with shared logic for all LLM providers."""

from __future__ import annotations

from nature.protocols.message import Usage
from nature.protocols.provider import LLMProvider, ProviderConfig


class BaseLLMProvider(LLMProvider):
    """Base class providing shared functionality for providers.

    Handles config storage, usage accumulation, and cost tracking.
    Subclasses implement stream() and count_tokens().
    """

    def __init__(self, config: ProviderConfig) -> None:
        self._config = config
        self._total_usage = Usage()

    @property
    def config(self) -> ProviderConfig:
        return self._config

    @property
    def model_id(self) -> str:
        return self._config.model

    @property
    def total_usage(self) -> Usage:
        """Accumulated usage across all calls."""
        return self._total_usage

    def _accumulate_usage(self, usage: Usage) -> None:
        """Add usage from a single call to the running total."""
        self._total_usage = Usage(
            input_tokens=self._total_usage.input_tokens + usage.input_tokens,
            output_tokens=self._total_usage.output_tokens + usage.output_tokens,
            cache_creation_input_tokens=(
                self._total_usage.cache_creation_input_tokens
                + usage.cache_creation_input_tokens
            ),
            cache_read_input_tokens=(
                self._total_usage.cache_read_input_tokens + usage.cache_read_input_tokens
            ),
        )
