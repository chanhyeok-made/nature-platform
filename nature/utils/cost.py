"""Cost tracking — per-model pricing and usage accumulation."""

from __future__ import annotations

from dataclasses import dataclass, field

from nature.protocols.message import Usage

# Pricing per 1M tokens (USD) as of 2025
# Source: https://docs.anthropic.com/en/docs/about-claude/models
MODEL_PRICING: dict[str, dict[str, float]] = {
    # Claude 4.6 family
    "claude-opus-4-6": {"input": 15.0, "output": 75.0, "cache_write": 18.75, "cache_read": 1.50},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0, "cache_write": 3.75, "cache_read": 0.30},
    # Claude 4.5 family
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.0, "cache_write": 1.0, "cache_read": 0.08},
    # Claude 4 family
    "claude-opus-4-20250514": {"input": 15.0, "output": 75.0, "cache_write": 18.75, "cache_read": 1.50},
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0, "cache_write": 3.75, "cache_read": 0.30},
}

# Fallback pricing for unknown models
_DEFAULT_PRICING = {"input": 3.0, "output": 15.0, "cache_write": 3.75, "cache_read": 0.30}


def _get_pricing(model: str) -> dict[str, float]:
    """Get pricing for a model, with fuzzy matching fallback."""
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]
    # Try prefix matching (e.g., "claude-sonnet-4-6" matches any variant)
    model_lower = model.lower()
    for key, pricing in MODEL_PRICING.items():
        if model_lower.startswith(key.split("-202")[0]):
            return pricing
    return _DEFAULT_PRICING


def calculate_cost(usage: Usage, model: str) -> float:
    """Calculate cost in USD for a single API call.

    Args:
        usage: Token usage from the API response.
        model: Model identifier.

    Returns:
        Cost in USD.
    """
    pricing = _get_pricing(model)
    cost = 0.0

    # Regular input tokens (exclude cached)
    regular_input = usage.input_tokens - usage.cache_read_input_tokens
    cost += (regular_input / 1_000_000) * pricing["input"]

    # Output tokens
    cost += (usage.output_tokens / 1_000_000) * pricing["output"]

    # Cache write tokens
    cost += (usage.cache_creation_input_tokens / 1_000_000) * pricing["cache_write"]

    # Cache read tokens (discounted)
    cost += (usage.cache_read_input_tokens / 1_000_000) * pricing["cache_read"]

    return cost


@dataclass
class CostTracker:
    """Tracks cumulative costs across a session."""

    total_cost_usd: float = 0.0
    total_usage: Usage = field(default_factory=Usage)
    call_count: int = 0
    _per_model: dict[str, Usage] = field(default_factory=dict)

    def add(self, usage: Usage, model: str) -> float:
        """Record usage from an API call. Returns the call's cost."""
        cost = calculate_cost(usage, model)
        self.total_cost_usd += cost
        self.call_count += 1

        # Accumulate total usage
        self.total_usage = Usage(
            input_tokens=self.total_usage.input_tokens + usage.input_tokens,
            output_tokens=self.total_usage.output_tokens + usage.output_tokens,
            cache_creation_input_tokens=(
                self.total_usage.cache_creation_input_tokens
                + usage.cache_creation_input_tokens
            ),
            cache_read_input_tokens=(
                self.total_usage.cache_read_input_tokens + usage.cache_read_input_tokens
            ),
        )

        # Per-model breakdown
        if model not in self._per_model:
            self._per_model[model] = Usage()
        prev = self._per_model[model]
        self._per_model[model] = Usage(
            input_tokens=prev.input_tokens + usage.input_tokens,
            output_tokens=prev.output_tokens + usage.output_tokens,
            cache_creation_input_tokens=(
                prev.cache_creation_input_tokens + usage.cache_creation_input_tokens
            ),
            cache_read_input_tokens=(
                prev.cache_read_input_tokens + usage.cache_read_input_tokens
            ),
        )

        return cost

    @property
    def cache_savings_pct(self) -> float:
        """Percentage of input tokens served from cache."""
        total_input = self.total_usage.input_tokens
        if total_input == 0:
            return 0.0
        return (self.total_usage.cache_read_input_tokens / total_input) * 100

    def format_summary(self) -> str:
        """Format a human-readable cost summary."""
        u = self.total_usage
        lines = [
            f"Cost: ${self.total_cost_usd:.4f} ({self.call_count} calls)",
            f"Tokens: {u.input_tokens:,} in / {u.output_tokens:,} out",
        ]
        if u.cache_read_input_tokens > 0:
            lines.append(
                f"Cache: {u.cache_read_input_tokens:,} read / "
                f"{u.cache_creation_input_tokens:,} write "
                f"({self.cache_savings_pct:.0f}% hit)"
            )
        return " | ".join(lines)
