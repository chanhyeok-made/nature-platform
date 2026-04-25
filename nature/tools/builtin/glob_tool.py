"""GlobTool — fast file pattern matching."""

from __future__ import annotations

import glob as glob_mod
import os
from typing import Any

from pydantic import BaseModel, Field

from nature.protocols.tool import ToolContext, ToolResult
from nature.tools.base import BaseTool

MAX_RESULTS = 500


class GlobInput(BaseModel):
    pattern: str = Field(description='Glob pattern (e.g. "**/*.py", "src/**/*.ts")')
    path: str | None = Field(default=None, description="Directory to search in (default: cwd)")


class GlobTool(BaseTool):
    input_model = GlobInput

    @property
    def name(self) -> str:
        return "Glob"

    @property
    def description(self) -> str:
        return "Find files matching a glob pattern. Returns file paths sorted by modification time."

    def is_read_only(self, input: dict[str, Any]) -> bool:
        return True

    def is_concurrency_safe(self, input: dict[str, Any]) -> bool:
        return True

    async def run(self, params: GlobInput, context: ToolContext) -> ToolResult:
        # Reject absolute patterns up-front. `os.path.join(cwd, "/etc/*")`
        # silently ignores `cwd` when the right side is absolute, which
        # would let a model bypass the search-dir boundary by tucking
        # the escape into `pattern` instead of `path`.
        if os.path.isabs(params.pattern):
            return ToolResult(
                output=(
                    f"Glob pattern must be relative, got absolute path "
                    f"'{params.pattern}'. Use a relative pattern like "
                    f"'nature/server/**/*.py' so the search stays inside "
                    f"the project's working directory."
                ),
                is_error=True,
            )

        search_dir = params.path or context.cwd
        if not os.path.isabs(search_dir):
            search_dir = os.path.join(context.cwd, search_dir)

        # Working-directory boundary: refuse to scan anything that
        # resolves outside the project's cwd. Catches the obvious
        # `path="/"` escape (session 8ca51065 tried to walk the entire
        # macOS filesystem under that), and also normalizes `..`,
        # symlinks, etc., via realpath. We compare canonical paths so
        # an attacker can't dodge with `/private/var/..` style tricks.
        search_dir_resolved = os.path.realpath(search_dir)
        cwd_resolved = os.path.realpath(context.cwd)
        if not (
            search_dir_resolved == cwd_resolved
            or search_dir_resolved.startswith(cwd_resolved + os.sep)
        ):
            return ToolResult(
                output=(
                    f"Glob path '{params.path}' resolves to "
                    f"'{search_dir_resolved}' which is outside the project's "
                    f"working directory '{cwd_resolved}'. Glob is restricted "
                    f"to the working directory — use a path inside it, or "
                    f"omit `path` to default to the working directory."
                ),
                is_error=True,
            )

        if not os.path.isdir(search_dir):
            return ToolResult(output=f"Directory not found: {search_dir}", is_error=True)

        full_pattern = os.path.join(search_dir, params.pattern)

        try:
            matches = glob_mod.glob(full_pattern, recursive=True)
        except Exception as e:
            return ToolResult(output=f"Glob error: {e}", is_error=True)

        # Sort by mtime (newest first)
        matches_with_mtime = []
        for m in matches:
            try:
                mtime = os.path.getmtime(m)
                matches_with_mtime.append((m, mtime))
            except OSError:
                matches_with_mtime.append((m, 0))

        matches_with_mtime.sort(key=lambda x: x[1], reverse=True)

        if not matches_with_mtime:
            return ToolResult(output=f"No files matching '{params.pattern}' in {search_dir}")

        truncated = len(matches_with_mtime) > MAX_RESULTS
        results = [m for m, _ in matches_with_mtime[:MAX_RESULTS]]

        output = "\n".join(results)
        if truncated:
            output += f"\n\n... ({len(matches_with_mtime) - MAX_RESULTS} more files)"

        return ToolResult(
            output=output,
            metadata={"total_matches": len(matches_with_mtime)},
        )
