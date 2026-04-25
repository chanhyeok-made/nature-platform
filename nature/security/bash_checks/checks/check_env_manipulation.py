"""Check 3: Dangerous environment variable manipulation."""

from __future__ import annotations

from nature.security.bash_checks.types import BashSafetyResult
from nature.security.patterns import DANGEROUS_ENV_VARS


def check_env_manipulation(cmd: str) -> BashSafetyResult:
    """Check 3: Dangerous environment variable manipulation."""
    for var in DANGEROUS_ENV_VARS:
        var_name = var.lstrip("$")
        if f"{var_name}=" in cmd or f"export {var_name}" in cmd or f"unset {var_name}" in cmd:
            return BashSafetyResult(safe=False, reason=f"Manipulation of {var}")
    return BashSafetyResult(safe=True)
