"""Tests for cost tracking."""

from nature.protocols.message import Usage
from nature.utils.cost import CostTracker, calculate_cost


class TestCalculateCost:
    def test_basic_cost(self):
        usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = calculate_cost(usage, "claude-sonnet-4-20250514")
        # Sonnet: $3/M input + $15/M output = $18
        assert abs(cost - 18.0) < 0.01

    def test_cached_cost(self):
        usage = Usage(
            input_tokens=1_000_000,
            output_tokens=100_000,
            cache_creation_input_tokens=500_000,
            cache_read_input_tokens=400_000,
        )
        cost = calculate_cost(usage, "claude-sonnet-4-20250514")
        # Regular input: (1M - 400K cache_read) = 600K * $3/M = $1.80
        # Output: 100K * $15/M = $1.50
        # Cache write: 500K * $3.75/M = $1.875
        # Cache read: 400K * $0.30/M = $0.12
        expected = 1.80 + 1.50 + 1.875 + 0.12
        assert abs(cost - expected) < 0.01

    def test_unknown_model_uses_default(self):
        usage = Usage(input_tokens=1000, output_tokens=1000)
        cost = calculate_cost(usage, "some-unknown-model")
        assert cost > 0


class TestCostTracker:
    def test_empty_tracker(self):
        tracker = CostTracker()
        assert tracker.total_cost_usd == 0.0
        assert tracker.call_count == 0

    def test_add_usage(self):
        tracker = CostTracker()
        usage = Usage(input_tokens=10_000, output_tokens=5_000)
        cost = tracker.add(usage, "claude-sonnet-4-20250514")
        assert cost > 0
        assert tracker.call_count == 1
        assert tracker.total_usage.input_tokens == 10_000

    def test_multiple_calls(self):
        tracker = CostTracker()
        tracker.add(Usage(input_tokens=10_000, output_tokens=5_000), "claude-sonnet-4-20250514")
        tracker.add(Usage(input_tokens=20_000, output_tokens=10_000), "claude-sonnet-4-20250514")
        assert tracker.call_count == 2
        assert tracker.total_usage.input_tokens == 30_000
        assert tracker.total_usage.output_tokens == 15_000

    def test_cache_savings(self):
        tracker = CostTracker()
        tracker.add(
            Usage(input_tokens=10_000, cache_read_input_tokens=8_000),
            "claude-sonnet-4-20250514",
        )
        assert tracker.cache_savings_pct == 80.0

    def test_format_summary(self):
        tracker = CostTracker()
        tracker.add(
            Usage(input_tokens=10_000, output_tokens=5_000),
            "claude-sonnet-4-20250514",
        )
        summary = tracker.format_summary()
        assert "Cost:" in summary
        assert "Tokens:" in summary
