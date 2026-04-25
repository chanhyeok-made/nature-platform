"""GrepTool — content search with regex support."""

from __future__ import annotations

import os
import re
import subprocess
from typing import Any

from pydantic import BaseModel, Field

from nature.protocols.tool import ToolContext, ToolResult
from nature.tools.base import BaseTool

MAX_RESULTS = 250


class GrepInput(BaseModel):
    pattern: str = Field(description="Regex pattern to search for")
    path: str | None = Field(default=None, description="File or directory to search in (default: cwd)")
    glob: str | None = Field(default=None, description='File glob filter (e.g. "*.py")')
    context: int | None = Field(default=None, description="Lines of context around matches")
    case_insensitive: bool = Field(default=False, description="Case insensitive search")


class GrepTool(BaseTool):
    input_model = GrepInput

    @property
    def name(self) -> str:
        return "Grep"

    @property
    def description(self) -> str:
        return "Search file contents using regex. Uses ripgrep (rg) if available, falls back to Python re."

    def is_read_only(self, input: dict[str, Any]) -> bool:
        return True

    def is_concurrency_safe(self, input: dict[str, Any]) -> bool:
        return True

    async def run(self, params: GrepInput, context: ToolContext) -> ToolResult:
        search_path = params.path or context.cwd
        if not os.path.isabs(search_path):
            search_path = os.path.join(context.cwd, search_path)

        # Try ripgrep first (much faster)
        rg = _find_rg()
        if rg:
            return await self._run_rg(rg, params, search_path)
        return await self._run_python(params, search_path)

    async def _run_rg(self, rg_path: str, params: GrepInput, search_path: str) -> ToolResult:
        cmd = [rg_path, "--no-heading", "--line-number", "-n"]
        if params.case_insensitive:
            cmd.append("-i")
        if params.context:
            cmd.extend(["-C", str(params.context)])
        if params.glob:
            cmd.extend(["--glob", params.glob])
        cmd.extend(["--max-count", str(MAX_RESULTS)])
        cmd.append(params.pattern)
        cmd.append(search_path)

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30,
            )
            output = proc.stdout.strip()
            if not output:
                return ToolResult(output=f"No matches for '{params.pattern}'")
            return ToolResult(output=output)
        except subprocess.TimeoutExpired:
            return ToolResult(output="Search timed out after 30s", is_error=True)
        except Exception as e:
            return ToolResult(output=f"rg error: {e}", is_error=True)

    async def _run_python(self, params: GrepInput, search_path: str) -> ToolResult:
        """Fallback: Python-based grep."""
        flags = re.IGNORECASE if params.case_insensitive else 0
        try:
            regex = re.compile(params.pattern, flags)
        except re.error as e:
            return ToolResult(output=f"Invalid regex: {e}", is_error=True)

        results: list[str] = []

        if os.path.isfile(search_path):
            files = [search_path]
        else:
            files = []
            for root, _, filenames in os.walk(search_path):
                # Skip hidden dirs and common ignores
                if any(p.startswith(".") for p in root.split(os.sep)):
                    continue
                for fn in filenames:
                    if params.glob:
                        import fnmatch
                        if not fnmatch.fnmatch(fn, params.glob):
                            continue
                    files.append(os.path.join(root, fn))

        for filepath in files[:1000]:  # Cap file scanning
            try:
                with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                    for line_no, line in enumerate(f, 1):
                        if regex.search(line):
                            results.append(f"{filepath}:{line_no}:{line.rstrip()}")
                            if len(results) >= MAX_RESULTS:
                                break
            except (PermissionError, IsADirectoryError, OSError):
                continue

            if len(results) >= MAX_RESULTS:
                break

        if not results:
            return ToolResult(output=f"No matches for '{params.pattern}'")

        output = "\n".join(results)
        if len(results) >= MAX_RESULTS:
            output += f"\n\n... (results capped at {MAX_RESULTS})"
        return ToolResult(output=output)


def _find_rg() -> str | None:
    """Find ripgrep binary."""
    import shutil
    return shutil.which("rg")
