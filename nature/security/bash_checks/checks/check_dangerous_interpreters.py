"""Check 2: Arbitrary code execution via interpreters."""

from __future__ import annotations

from nature.security.bash_checks.types import BashSafetyResult
from nature.security.patterns import DANGEROUS_INTERPRETERS


def check_dangerous_interpreters(cmd: str) -> BashSafetyResult:
    """Check 2: Arbitrary code execution via interpreters.

    Blocks: python -c, node -e, eval, exec, etc.
    Allows: python script.py, node app.js (running existing files)
    """
    parts = cmd.strip().split()
    if not parts:
        return BashSafetyResult(safe=True)

    first = parts[0]
    if first in ("sudo",) and len(parts) > 1:
        first = parts[1]

    if first in DANGEROUS_INTERPRETERS:
        danger_flags = {"-c", "-e", "--eval", "--exec", "-i"}
        if any(f in parts for f in danger_flags):
            return BashSafetyResult(
                safe=False,
                reason=f"Arbitrary code execution via {first} with inline code flag",
            )
    return BashSafetyResult(safe=True)
