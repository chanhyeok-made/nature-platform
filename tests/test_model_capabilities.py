"""Tests for nature.providers.model_capabilities.

Covers the two-layer max_output_tokens design:
- Layer 1: `clip_to_ceiling` enforces the model's physical cap at
  provider send time, regardless of caller intent.
- Layer 2 (preset-driven role budgets) is tested elsewhere in the
  preset validation suite; this file focuses on the provider-level
  ceiling gate so that lookup table additions don't silently regress.
"""

from __future__ import annotations

from nature.providers.model_capabilities import (
    ModelCapabilities,
    clip_to_ceiling,
    lookup,
)


def test_unknown_model_returns_default_caps():
    caps = lookup("openrouter::unknown-vendor/unknown-model")
    assert caps.max_output_ceiling is None
    assert caps.text_tool_adaptation is False
    assert caps.stream_timeout_multiplier == 1.0


def test_clip_passes_through_when_host_is_none():
    assert clip_to_ceiling(16_000, None, "anything") == 16_000


def test_clip_passes_through_when_model_has_no_ceiling():
    # local-ollama::phi4 has text_tool_adaptation but no ceiling set.
    assert clip_to_ceiling(8_192, "local-ollama", "phi4:latest") == 8_192


def test_clip_lowers_to_ceiling():
    # Haiku 4.5 ceiling is 8192; a 32K request gets clipped.
    assert clip_to_ceiling(32_000, "anthropic", "claude-haiku-4-5") == 8_192


def test_clip_leaves_below_ceiling_alone():
    # Sonnet 4.6 ceiling is 64K; asking for 4K returns 4K.
    assert clip_to_ceiling(4_000, "anthropic", "claude-sonnet-4-6") == 4_000


def test_openrouter_sonnet_matches_direct_anthropic():
    direct = clip_to_ceiling(100_000, "anthropic", "claude-sonnet-4-6")
    router = clip_to_ceiling(100_000, "openrouter", "anthropic/claude-sonnet-4.6")
    assert direct == router == 64_000


def test_deepseek_r1_via_openrouter_has_ceiling():
    # Reasoning model with 8K output cap.
    assert clip_to_ceiling(16_000, "openrouter", "deepseek/deepseek-r1") == 8_192


def test_nonpositive_request_passes_through():
    # Degenerate inputs should not trigger a lookup or return negative.
    assert clip_to_ceiling(0, "anthropic", "claude-haiku-4-5") == 0
    assert clip_to_ceiling(-1, "anthropic", "claude-haiku-4-5") == -1


def test_modelcapabilities_frozen():
    caps = ModelCapabilities(max_output_ceiling=1234)
    # dataclass frozen → attribute assignment raises.
    try:
        caps.max_output_ceiling = 999  # type: ignore[misc]
    except Exception as exc:
        assert "frozen" in str(exc).lower() or "cannot assign" in str(exc).lower()
    else:
        raise AssertionError("expected frozen dataclass to reject mutation")
