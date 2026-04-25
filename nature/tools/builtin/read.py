"""ReadTool — file reading with ReadMemory integration.

When `context.pack_state["read_memory"]` is available:
- **Dedup**: fully-covered range → return stub
- **Cache serve**: different range of a cached file → serve from segments
- **State set**: after reading, add segment to ReadMemory
- **Size cap**: files over MAX_FILE_BYTES rejected on full-read

When read_memory is absent (tests, no Pack), behavior is unchanged.
Uses read_line_range() instead of readlines() for memory safety.
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field

from nature.protocols.tool import ToolContext, ToolResult
from nature.tools.base import BaseTool

MAX_READ_LINES = 2000
MAX_FILE_BYTES = 256 * 1024  # 256KB — full-read requests only


def _format_lines(lines: list[str], offset: int, total_lines: int | None = None) -> str:
    numbered = []
    for i, line in enumerate(lines, start=offset + 1):
        numbered.append(f"{i}\t{line}")
    output = "\n".join(numbered)
    if total_lines is not None and offset + len(lines) < total_lines:
        output += f"\n\n... ({total_lines - offset - len(lines)} more lines)"
    return output


class ReadInput(BaseModel):
    file_path: str = Field(description="Absolute path to the file to read")
    offset: int | None = Field(default=None, description="Line number to start from (0-based)")
    limit: int | None = Field(default=None, description="Max number of lines to read")


class ReadTool(BaseTool):
    input_model = ReadInput

    @property
    def name(self) -> str:
        return "Read"

    @property
    def description(self) -> str:
        return (
            "Read a file's contents. Returns lines with line numbers. "
            "Use offset/limit for large files."
        )

    def is_read_only(self, input: dict[str, Any]) -> bool:
        return True

    def is_concurrency_safe(self, input: dict[str, Any]) -> bool:
        return True

    async def run(self, params: ReadInput, context: ToolContext) -> ToolResult:
        path = params.file_path
        if not os.path.isabs(path):
            path = os.path.join(context.cwd, path)

        if not os.path.exists(path):
            return ToolResult(output=f"File not found: {path}", is_error=True)

        if os.path.isdir(path):
            return ToolResult(
                output=f"{path} is a directory. Use Bash 'ls' to list contents.",
                is_error=True,
            )

        # ── Size cap (full-read only) ────────────────────────────────
        try:
            stat = os.stat(path)
        except OSError as e:
            return ToolResult(output=f"Cannot stat {path}: {e}", is_error=True)

        offset = params.offset or 0
        limit = params.limit or MAX_READ_LINES
        is_full_read = params.limit is None and (params.offset is None or params.offset == 0)

        if is_full_read and stat.st_size > MAX_FILE_BYTES:
            return ToolResult(
                output=(
                    f"File is {stat.st_size:,} bytes ({stat.st_size // 1024}KB), "
                    f"exceeds the {MAX_FILE_BYTES // 1024}KB limit for full reads.\n"
                    f"1. Use offset and limit to read specific portions.\n"
                    f"2. Use Grep to search for specific content.\n"
                    f"3. Delegate to a sub-agent (Agent tool) for focused reading."
                ),
                is_error=True,
            )

        # ── ReadMemory check ─────────────────────────────────────────
        read_memory = (getattr(context, "pack_state", None) or {}).get("read_memory")
        end = offset + limit

        if read_memory is not None:
            entry = read_memory.get(path)

            if entry is not None and not entry.expired:
                # Cap requested end at total_lines — if file has 8 lines
                # and limit is 2000, effective end is 8, not 2000.
                effective_end = min(end, entry.total_lines) if entry.total_lines > 0 else end
                if entry.covers(offset, effective_end):
                    return ToolResult(
                        output=(
                            f"File unchanged since last read ({path}). "
                            f"Lines {offset+1}–{end} were already read in this "
                            f"session — refer to the earlier Read output."
                        ),
                        metadata={"deduped": True, "total_lines": entry.total_lines},
                    )

                # Cached but different range → serve from segments
                cached = entry.get_lines(offset, end)
                if cached is not None:
                    from nature.context.read_memory import ReadSegment

                    entry.add_segment(ReadSegment(offset, offset + len(cached), "\n".join(cached)))
                    entry.hit_count += 1
                    return ToolResult(
                        output=_format_lines(cached, offset, entry.total_lines),
                        metadata={"from_cache": True, "total_lines": entry.total_lines},
                    )

        # ── Actual disk read (line-by-line, no readlines) ────────────
        from nature.context.read_memory import read_line_range

        try:
            selected, total_lines = read_line_range(path, offset, limit)
        except PermissionError:
            return ToolResult(output=f"Permission denied: {path}", is_error=True)
        except Exception as e:
            return ToolResult(output=f"Error reading {path}: {e}", is_error=True)

        # ── State set ────────────────────────────────────────────────
        if read_memory is not None:
            import hashlib
            from nature.context.read_memory import ReadMemoryEntry, ReadSegment

            actual_end = min(offset + limit, total_lines)
            seg = ReadSegment(offset, actual_end, "\n".join(selected))
            content_for_hash = "\n".join(selected)

            existing = read_memory.get(path)
            if existing is not None and not existing.expired:
                existing.add_segment(seg)
                existing.total_lines = total_lines
            else:
                read_memory.set(path, ReadMemoryEntry(
                    path=path,
                    mtime_ns=stat.st_mtime_ns,
                    total_lines=total_lines,
                    segments=[seg],
                    depth=0,
                ))

        output = _format_lines(selected, offset, total_lines)
        return ToolResult(output=output, metadata={"total_lines": total_lines})
