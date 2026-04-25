"""Check 9: Access to sensitive system files."""

from __future__ import annotations

from nature.security.bash_checks.types import BashSafetyResult


def check_sensitive_files(cmd: str) -> BashSafetyResult:
    """Check 9: Access to sensitive system files."""
    sensitive = [
        "/etc/shadow", "/etc/passwd",
        "~/.ssh/id_", "~/.ssh/authorized_keys",
        "~/.aws/credentials", "~/.gnupg/",
    ]
    for path in sensitive:
        if path in cmd:
            if any(w in cmd for w in ("rm ", "mv ", "> ", ">> ", "chmod ", "tee ")):
                return BashSafetyResult(safe=False, reason=f"Write to sensitive file: {path}")
    return BashSafetyResult(safe=True)
