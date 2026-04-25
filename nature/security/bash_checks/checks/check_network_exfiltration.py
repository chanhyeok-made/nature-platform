"""Check 10: Data exfiltration patterns."""

from __future__ import annotations

from nature.security.bash_checks.types import BashSafetyResult


def check_network_exfiltration(cmd: str) -> BashSafetyResult:
    """Check 10: Data exfiltration patterns."""
    if "curl -d" in cmd or "curl --data" in cmd or "curl -X POST" in cmd:
        if any(f in cmd for f in ("@/", "< /", "$(cat")):
            return BashSafetyResult(safe=False, reason="Potential data exfiltration via curl POST")
    return BashSafetyResult(safe=True)
