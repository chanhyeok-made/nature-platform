"""state_persist — Listeners that emit READ_MEMORY_SET events.

These fire POST tool-call on Read/Edit/Write success. The live
ReadMemory is already updated by the tool itself (dual-write pattern);
these Listeners handle the event-store side for persistence/reconstruct.
"""

from __future__ import annotations

import hashlib
import os

from nature.events.payloads import ReadMemorySetPayload
from nature.events.types import EventType
from nature.packs.types import (
    EmitEvent,
    Intervention,
    InterventionContext,
    OnTool,
    ToolPhase,
)


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def _read_persist_action(ctx: InterventionContext):
    """After successful Read, emit READ_MEMORY_SET for persistence."""
    tc = ctx.tool_call
    if tc is None or tc.result_is_error:
        return []
    # Skip deduped reads (no new data to persist)
    if tc.result_output and "already read" in tc.result_output:
        return []
    frame = ctx.frame
    if frame is None:
        return []
    read_memory = frame.pack_state.get("read_memory")
    if read_memory is None:
        return []

    path = tc.tool_input.get("file_path", "")
    if not path:
        return []

    entry = read_memory.get(path)
    if entry is None or entry.expired:
        return []

    return [EmitEvent(
        event_type=EventType.READ_MEMORY_SET,
        payload=ReadMemorySetPayload(
            path=entry.path,
            content_hash=_content_hash(entry.content or ""),
            mtime_ns=entry.mtime_ns,
            lines=entry.total_lines,
            offset=tc.tool_input.get("offset"),
            limit=tc.tool_input.get("limit"),
            depth=entry.depth,
        ),
    )]


def _write_persist_action(ctx: InterventionContext):
    """After successful Edit or Write, emit READ_MEMORY_SET."""
    tc = ctx.tool_call
    if tc is None or tc.result_is_error:
        return []
    frame = ctx.frame
    if frame is None:
        return []
    read_memory = frame.pack_state.get("read_memory")
    if read_memory is None:
        return []

    path = tc.tool_input.get("file_path", "")
    if not path:
        return []

    entry = read_memory.get(path)
    if entry is None:
        return []

    return [EmitEvent(
        event_type=EventType.READ_MEMORY_SET,
        payload=ReadMemorySetPayload(
            path=entry.path,
            content_hash=_content_hash(entry.content or ""),
            mtime_ns=entry.mtime_ns,
            lines=entry.total_lines,
            depth=entry.depth,
        ),
    )]


read_state_persist = Intervention(
    id="file_state.read_persist",
    kind="listener",
    trigger=OnTool(
        tool_name="Read",
        phase=ToolPhase.POST,
        where=lambda tc: tc.result_is_error is not True,
    ),
    action=_read_persist_action,
    description="Emit READ_MEMORY_SET after successful Read.",
)

edit_state_persist = Intervention(
    id="file_state.edit_persist",
    kind="listener",
    trigger=OnTool(
        tool_name="Edit",
        phase=ToolPhase.POST,
        where=lambda tc: tc.result_is_error is not True,
    ),
    action=_write_persist_action,
    description="Emit READ_MEMORY_SET after successful Edit.",
)

write_state_persist = Intervention(
    id="file_state.write_persist",
    kind="listener",
    trigger=OnTool(
        tool_name="Write",
        phase=ToolPhase.POST,
        where=lambda tc: tc.result_is_error is not True,
    ),
    action=_write_persist_action,
    description="Emit READ_MEMORY_SET after successful Write.",
)
