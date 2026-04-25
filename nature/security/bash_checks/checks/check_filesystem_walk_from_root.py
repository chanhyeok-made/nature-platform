"""Check 11: Walking the entire filesystem from `/`."""

from __future__ import annotations

import re

from nature.security.bash_checks.types import BashSafetyResult


def check_filesystem_walk_from_root(cmd: str) -> BashSafetyResult:
    """Check 11: Walking the entire filesystem from `/`.

    A model that's "looking for files in the project" sometimes
    decides to scan from root — `find / -name '*.py'`, `ls -R /`,
    etc. On a typical macOS box this hangs the agent for hours,
    pegs CPU, and exposes paths far outside the project. Session
    `8ca51065` burned 8 minutes on `find / -maxdepth 4 ...` before
    we noticed and killed it.

    These commands are almost always accidents. Block them so the
    model gets an `is_error` tool_result and retries with a path
    scoped to the project.
    """
    patterns = [
        (r"\bfind\s+/(?=[\s\-]|$)", "`find /` walks the entire filesystem"),
        (r"\bls\s+-[A-Za-z]*R[A-Za-z]*\s+/(?=[\s]|$)",
         "`ls -R /` recurses the entire filesystem"),
        (r"\bdu\s+-[A-Za-z]+\s+/(?=[\s]|$)",
         "`du /` scans the entire filesystem"),
        (r"\btree\s+/(?=[\s]|$)",
         "`tree /` scans the entire filesystem"),
        (r"\bgrep\s+-[A-Za-z]*[rR][A-Za-z]*\s+\S*\s+/(?=[\s]|$)",
         "`grep -r ... /` recurses from filesystem root"),
        (r"\brg\s+(?:-[A-Za-z]+\s+)*\S+\s+/(?=[\s]|$)",
         "`rg ... /` searches from filesystem root"),
    ]
    for pat, reason in patterns:
        if re.search(pat, cmd):
            return BashSafetyResult(
                safe=False,
                reason=(
                    f"{reason}. Use a path inside the project's working "
                    f"directory instead (e.g., `.` or a relative subpath)."
                ),
            )
    return BashSafetyResult(safe=True)
