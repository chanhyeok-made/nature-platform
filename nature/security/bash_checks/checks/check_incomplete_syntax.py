"""Check 6: Unclosed quotes or obvious syntax errors."""

from __future__ import annotations

import shlex

from nature.security.bash_checks.types import BashSafetyResult


def check_incomplete_syntax(cmd: str) -> BashSafetyResult:
    """Check 6: Unclosed quotes or obvious syntax errors."""
    try:
        shlex.split(cmd)
    except ValueError:
        return BashSafetyResult(safe=False, reason="Unclosed quotes or syntax error")
    return BashSafetyResult(safe=True)
