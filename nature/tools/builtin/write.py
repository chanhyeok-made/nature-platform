"""WriteTool — file creation/overwrite."""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field

from nature.protocols.tool import ToolContext, ToolResult
from nature.tools.base import BaseTool


class WriteInput(BaseModel):
    file_path: str = Field(description="Absolute path to the file to write")
    content: str = Field(description="Content to write to the file")


class WriteTool(BaseTool):
    input_model = WriteInput

    @property
    def name(self) -> str:
        return "Write"

    @property
    def description(self) -> str:
        return (
            "Write content to a file. Creates the file if it doesn't exist, "
            "or overwrites if it does. Creates parent directories as needed."
        )

    def is_read_only(self, input: dict[str, Any]) -> bool:
        return False

    def is_destructive(self, input: dict[str, Any]) -> bool:
        return True  # Overwrites existing content

    async def validate_input(self, input: dict[str, Any], context: ToolContext) -> str | None:
        if context.is_read_only:
            return "Write is not allowed in read-only mode."
        return None

    async def run(self, params: WriteInput, context: ToolContext) -> ToolResult:
        path = params.file_path
        if not os.path.isabs(path):
            path = os.path.join(context.cwd, path)

        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(params.content)
        except PermissionError:
            return ToolResult(output=f"Permission denied: {path}", is_error=True)
        except Exception as e:
            return ToolResult(output=f"Error writing {path}: {e}", is_error=True)

        # ── Update ReadMemory after successful write ─────────────────
        read_memory = (getattr(context, "pack_state", None) or {}).get("read_memory")
        if read_memory is not None:
            from nature.context.read_memory import ReadMemoryEntry, ReadSegment

            try:
                new_stat = os.stat(path)
                lines = params.content.splitlines()
                read_memory.set(path, ReadMemoryEntry(
                    path=path,
                    mtime_ns=new_stat.st_mtime_ns,
                    total_lines=len(lines),
                    segments=[ReadSegment(0, len(lines), "\n".join(lines))],
                    depth=0,
                ))
            except OSError:
                read_memory.invalidate(path)

        return ToolResult(output=f"Successfully wrote {len(params.content)} chars to {path}")
