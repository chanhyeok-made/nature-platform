"""Context management protocols — compression strategies and context orchestration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

from pydantic import BaseModel

from nature.protocols.message import Message


# ---------------------------------------------------------------------------
# Token budget
# ---------------------------------------------------------------------------

class TokenWarningState(str, Enum):
    """Progressive warning states as context fills up."""
    OK = "ok"
    AUTOCOMPACT = "autocompact"  # Approaching limit, auto-compress
    WARNING = "warning"  # Near limit, warn user
    ERROR = "error"  # At limit, block further input


class TokenBudget(BaseModel):
    """Token budget configuration and tracking."""
    context_window: int = 200_000
    output_reservation: int = 20_000
    autocompact_buffer: int = 13_000
    warning_buffer: int = 20_000
    block_buffer: int = 3_000

    @property
    def effective_window(self) -> int:
        return self.context_window - self.output_reservation

    @property
    def autocompact_threshold(self) -> int:
        return self.effective_window - self.autocompact_buffer

    @property
    def warning_threshold(self) -> int:
        return self.effective_window - self.warning_buffer

    @property
    def block_threshold(self) -> int:
        return self.effective_window - self.block_buffer

    def get_warning_state(self, current_tokens: int) -> TokenWarningState:
        # Check from highest threshold down
        if current_tokens >= self.block_threshold:
            return TokenWarningState.ERROR
        if current_tokens >= self.autocompact_threshold:
            return TokenWarningState.AUTOCOMPACT
        if current_tokens >= self.warning_threshold:
            return TokenWarningState.WARNING
        return TokenWarningState.OK


# ---------------------------------------------------------------------------
# Compression strategy protocol
# ---------------------------------------------------------------------------

class CompressionResult(BaseModel):
    """Result of a compression operation."""
    messages: list[Message]
    tokens_before: int
    tokens_after: int
    strategy_name: str
    summary: str | None = None


class CompressionStrategy(ABC):
    """Abstract base class for context compression strategies.

    Four strategies compose orthogonally:
    1. Microcompact — clear old tool outputs (free)
    2. Snip — remove old message pairs (cheap)
    3. Autocompact — LLM summarization (expensive)
    4. Collapse — disk archive + re-project (heavy)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy identifier."""
        ...

    @abstractmethod
    async def compress(
        self,
        messages: list[Message],
        budget: TokenBudget,
        current_tokens: int,
    ) -> CompressionResult:
        """Apply this compression strategy.

        Args:
            messages: Current conversation messages.
            budget: Token budget configuration.
            current_tokens: Current total token count.

        Returns:
            CompressionResult with the (possibly shorter) message list.
        """
        ...


# ---------------------------------------------------------------------------
# Context manager protocol
# ---------------------------------------------------------------------------

class ContextManager(ABC):
    """Orchestrates multiple compression strategies.

    Runs strategies in priority order (cheapest first) until the
    token count is below the autocompact threshold.
    """

    @abstractmethod
    async def prepare(
        self,
        messages: list[Message],
        budget: TokenBudget,
        current_tokens: int,
    ) -> list[Message]:
        """Prepare messages for the next API call.

        Applies compression strategies as needed to stay within budget.

        Returns:
            Messages ready for the API call (possibly compressed).
        """
        ...

    @abstractmethod
    async def estimate_tokens(self, messages: list[Message]) -> int:
        """Estimate total tokens for a message list."""
        ...
