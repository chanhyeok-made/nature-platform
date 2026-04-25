"""BashTool — shell command execution with safety checks."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from pydantic import BaseModel, Field

from nature.protocols.tool import ToolContext, ToolResult
from nature.tools.base import BaseTool

# Commands that are always blocked
BLOCKED_PATTERNS = [
    "rm -rf /",
    "rm -rf /*",
    "mkfs.",
    ":(){:|:&};:",  # fork bomb
    "> /dev/sda",
]

# Dangerous interpreters that need explicit permission
DANGEROUS_INTERPRETERS = {
    "eval", "exec", "sudo su", "sudo -i", "sudo bash", "sudo sh",
}


class BashInput(BaseModel):
    command: str = Field(description="The shell command to execute")
    timeout: int | None = Field(default=120, description="Timeout in seconds")


class BashTool(BaseTool):
    input_model = BashInput

    @property
    def name(self) -> str:
        return "Bash"

    @property
    def description(self) -> str:
        return (
            "Execute a shell command. Use for running scripts, installing packages, "
            "git operations, and other system commands."
        )

    def is_read_only(self, input: dict[str, Any]) -> bool:
        cmd = input.get("command", "").strip()
        read_only_prefixes = (
            "ls", "cat", "head", "tail", "grep", "rg", "find", "which",
            "echo", "pwd", "date", "wc", "sort", "uniq", "diff",
            "git status", "git log", "git diff", "git branch",
        )
        return any(cmd.startswith(p) for p in read_only_prefixes)

    def is_concurrency_safe(self, input: dict[str, Any]) -> bool:
        return self.is_read_only(input)

    async def validate_input(self, input: dict[str, Any], context: ToolContext) -> str | None:
        cmd = input.get("command", "").strip()
        if not cmd:
            return "Command cannot be empty."
        if context.is_read_only:
            return "Bash commands are not allowed in read-only mode."
        # Run security checks (10 checks)
        from nature.security.bash_checks import check_bash_command
        result = check_bash_command(cmd)
        if not result.safe:
            return f"Blocked (check #{result.check_id}): {result.reason}"
        return None

    async def run(self, params: BashInput, context: ToolContext) -> ToolResult:
        timeout = params.timeout or 120
        try:
            proc = await asyncio.create_subprocess_shell(
                params.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=context.cwd,
                env={**os.environ, "TERM": "dumb"},
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )

            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")
            exit_code = proc.returncode or 0

            # Build output
            parts = []
            if stdout_str.strip():
                parts.append(stdout_str.rstrip())
            if stderr_str.strip():
                parts.append(f"STDERR:\n{stderr_str.rstrip()}")
            parts.append(f"Exit code: {exit_code}")

            output = "\n".join(parts)
            return ToolResult(
                output=output,
                is_error=exit_code != 0,
                metadata={"exit_code": exit_code},
            )

        except asyncio.TimeoutError:
            return ToolResult(
                output=f"Command timed out after {timeout}s: {params.command}",
                is_error=True,
            )
        except Exception as e:
            return ToolResult(output=f"Execution error: {e}", is_error=True)
