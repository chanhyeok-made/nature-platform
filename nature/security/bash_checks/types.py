"""Types for bash safety checks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BashSafetyResult:
    """Result of a bash safety check."""
    safe: bool
    check_id: int = 0
    reason: str = ""
