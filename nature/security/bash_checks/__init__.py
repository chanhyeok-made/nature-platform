"""Bash command safety analysis — 11 core checks.

Each check returns a BashSafetyResult. If any check fails,
the command is blocked with an explanation.
"""

from __future__ import annotations

from nature.security.bash_checks.types import BashSafetyResult
from nature.security.bash_checks.checks.check_blocked_patterns import check_blocked_patterns
from nature.security.bash_checks.checks.check_dangerous_interpreters import check_dangerous_interpreters
from nature.security.bash_checks.checks.check_env_manipulation import check_env_manipulation
from nature.security.bash_checks.checks.check_command_substitution import check_command_substitution
from nature.security.bash_checks.checks.check_null_bytes import check_null_bytes
from nature.security.bash_checks.checks.check_incomplete_syntax import check_incomplete_syntax
from nature.security.bash_checks.checks.check_pipe_to_shell import check_pipe_to_shell
from nature.security.bash_checks.checks.check_background_reverse_shell import check_background_reverse_shell
from nature.security.bash_checks.checks.check_sensitive_files import check_sensitive_files
from nature.security.bash_checks.checks.check_network_exfiltration import check_network_exfiltration
from nature.security.bash_checks.checks.check_filesystem_walk_from_root import check_filesystem_walk_from_root

__all__ = ["check_bash_command", "BashSafetyResult"]


def check_bash_command(command: str) -> BashSafetyResult:
    """Run all safety checks on a bash command.

    Returns the first failing check, or safe=True if all pass.
    """
    checks = [
        check_blocked_patterns,
        check_dangerous_interpreters,
        check_env_manipulation,
        check_command_substitution,
        check_null_bytes,
        check_incomplete_syntax,
        check_pipe_to_shell,
        check_background_reverse_shell,
        check_sensitive_files,
        check_network_exfiltration,
        check_filesystem_walk_from_root,
    ]

    for i, check_fn in enumerate(checks, 1):
        result = check_fn(command)
        if not result.safe:
            result.check_id = i
            return result

    return BashSafetyResult(safe=True)
