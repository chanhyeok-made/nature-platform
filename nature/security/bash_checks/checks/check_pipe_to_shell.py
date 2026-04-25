"""Check 7: Piping external content to shell interpreters."""

from __future__ import annotations

import re

from nature.security.bash_checks.types import BashSafetyResult


def check_pipe_to_shell(cmd: str) -> BashSafetyResult:
    """Check 7: Piping external content to shell interpreters."""
    if re.search(r"\b(curl|wget)\b.*\|\s*(sh|bash|zsh|fish)\b", cmd, re.IGNORECASE):
        return BashSafetyResult(safe=False, reason="Pipe to shell interpreter")
    return BashSafetyResult(safe=True)
