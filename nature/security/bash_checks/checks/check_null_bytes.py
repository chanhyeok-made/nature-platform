"""Check 5: Null bytes and control characters."""

from __future__ import annotations

from nature.security.bash_checks.types import BashSafetyResult


def check_null_bytes(cmd: str) -> BashSafetyResult:
    """Check 5: Null bytes and control characters."""
    if "\x00" in cmd:
        return BashSafetyResult(safe=False, reason="Null byte in command")
    for c in cmd:
        if ord(c) < 32 and c not in ("\n", "\t", "\r"):
            return BashSafetyResult(safe=False, reason=f"Control character U+{ord(c):04X}")
    return BashSafetyResult(safe=True)
