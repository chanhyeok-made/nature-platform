"""EditTool — surgical text replacement. State update after success."""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field

from nature.protocols.tool import ToolContext, ToolResult
from nature.tools.base import BaseTool


class EditInput(BaseModel):
    file_path: str = Field(description="Absolute path to the file to edit")
    old_string: str = Field(description="The exact text to find and replace")
    new_string: str = Field(description="The replacement text")
    replace_all: bool = Field(default=False, description="Replace all occurrences")


class EditTool(BaseTool):
    input_model = EditInput

    @property
    def name(self) -> str:
        return "Edit"

    @property
    def description(self) -> str:
        return (
            "Replace exact text in a file. The old_string must match exactly "
            "(including whitespace/indentation). Use replace_all for multiple occurrences."
        )

    def is_read_only(self, input: dict[str, Any]) -> bool:
        return False

    async def validate_input(self, input: dict[str, Any], context: ToolContext) -> str | None:
        # Pure input validation only. Contextual guards (read-first,
        # is_read_only) belong in Pack Gates.
        old = input.get("old_string", "")
        new = input.get("new_string", "")
        if old == new:
            return "old_string and new_string are identical."
        if not old:
            return "old_string cannot be empty."
        return None

    async def run(self, params: EditInput, context: ToolContext) -> ToolResult:
        path = params.file_path
        if not os.path.isabs(path):
            path = os.path.join(context.cwd, path)

        if not os.path.exists(path):
            return ToolResult(output=f"File not found: {path}", is_error=True)

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            return ToolResult(output=f"Error reading {path}: {e}", is_error=True)

        if params.old_string not in content:
            return ToolResult(
                output=f"old_string not found in {path}. Ensure exact match including whitespace.",
                is_error=True,
            )

        if not params.replace_all:
            count = content.count(params.old_string)
            if count > 1:
                return ToolResult(
                    output=f"old_string found {count} times. Use replace_all=true or provide more context.",
                    is_error=True,
                )

        if params.replace_all:
            new_content = content.replace(params.old_string, params.new_string)
            count = content.count(params.old_string)
        else:
            new_content = content.replace(params.old_string, params.new_string, 1)
            count = 1

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_content)
        except Exception as e:
            return ToolResult(output=f"Error writing {path}: {e}", is_error=True)

        # ── Update ReadMemory after successful edit ──────────────────
        read_memory = (getattr(context, "pack_state", None) or {}).get("read_memory")
        if read_memory is not None:
            from nature.context.read_memory import ReadMemoryEntry, ReadSegment

            try:
                new_stat = os.stat(path)
                new_lines = new_content.splitlines()
                read_memory.set(path, ReadMemoryEntry(
                    path=path,
                    mtime_ns=new_stat.st_mtime_ns,
                    total_lines=len(new_lines),
                    segments=[ReadSegment(0, len(new_lines), "\n".join(new_lines))],
                    depth=0,
                ))
            except OSError:
                # Best effort — if stat fails, invalidate to force re-read
                read_memory.invalidate(path)

        return ToolResult(output=f"Replaced {count} occurrence(s) in {path}")
