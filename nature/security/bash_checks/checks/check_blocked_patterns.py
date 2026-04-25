"""Check 1: Absolutely blocked command patterns."""

from __future__ import annotations

from nature.security.bash_checks.types import BashSafetyResult
from nature.security.patterns import DANGEROUS_COMMANDS


def check_blocked_patterns(cmd: str) -> BashSafetyResult:
    """Check 1: Absolutely blocked command patterns."""
    for pattern in DANGEROUS_COMMANDS:
        if pattern in cmd:
            return BashSafetyResult(safe=False, reason=f"Blocked pattern: {pattern}")
    return BashSafetyResult(safe=True)
