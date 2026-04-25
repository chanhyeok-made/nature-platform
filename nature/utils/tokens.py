"""Token estimation utilities.

Dual strategy: API-accurate counting when available, byte-based fallback otherwise.
"""

from __future__ import annotations

import json
from typing import Any

from nature.config.defaults import (
    DEFAULT_BYTES_PER_TOKEN,
    IMAGE_TOKEN_ESTIMATE,
    JSON_BYTES_PER_TOKEN,
)


def estimate_tokens_for_text(text: str, is_json: bool = False) -> int:
    """Estimate token count for a text string.

    Uses byte-based estimation (4 bytes/token for text, 2 bytes/token for JSON).
    """
    byte_count = len(text.encode("utf-8"))
    bpt = JSON_BYTES_PER_TOKEN if is_json else DEFAULT_BYTES_PER_TOKEN
    return max(1, byte_count // bpt)


def estimate_tokens_for_value(value: Any) -> int:
    """Estimate tokens for an arbitrary value (serialized to JSON)."""
    if value is None:
        return 0
    if isinstance(value, str):
        return estimate_tokens_for_text(value)
    try:
        serialized = json.dumps(value, ensure_ascii=False)
        return estimate_tokens_for_text(serialized, is_json=True)
    except (TypeError, ValueError):
        return estimate_tokens_for_text(str(value))


def estimate_tokens_for_image() -> int:
    """Conservative token estimate for an image (resized 2000x2000)."""
    return IMAGE_TOKEN_ESTIMATE
