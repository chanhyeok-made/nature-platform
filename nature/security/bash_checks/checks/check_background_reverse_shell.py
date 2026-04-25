"""Check 8: Reverse shell patterns."""

from __future__ import annotations

from nature.security.bash_checks.types import BashSafetyResult


def check_background_reverse_shell(cmd: str) -> BashSafetyResult:
    """Check 8: Reverse shell patterns."""
    patterns = [
        "/dev/tcp/", "/dev/udp/",
        "nc -e", "ncat -e",
        "mkfifo", "telnet",
    ]
    for p in patterns:
        if p in cmd:
            return BashSafetyResult(safe=False, reason=f"Potential reverse shell: {p}")
    return BashSafetyResult(safe=True)
