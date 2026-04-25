"""Check 4: Suspicious command substitution in dangerous contexts."""

from __future__ import annotations

from nature.security.bash_checks.types import BashSafetyResult


def check_command_substitution(cmd: str) -> BashSafetyResult:
    """Check 4: Suspicious command substitution in dangerous contexts."""
    if "$(rm " in cmd or "$(curl " in cmd or "$(wget " in cmd:
        return BashSafetyResult(safe=False, reason="Dangerous command in substitution")
    return BashSafetyResult(safe=True)
