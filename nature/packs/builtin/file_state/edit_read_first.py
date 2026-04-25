"""edit_read_first — strict Gate: Edit blocked unless file was Read.

Checks `read_memory` for the target path. Two levels:

1. **No entry at all** → "Read this file first" (never seen it).
2. **Entry exists, not expired, but old_string not in content** →
   "Target text not in any region you've read" (saw the file but
   not this part). Forces a Read of the relevant section.
3. **Expired entry** → allow (the file WAS read at some point;
   if it's stale, Edit.run will catch the mismatch and
   edit_guards.fuzzy_suggest will help recover).
"""

from __future__ import annotations

import os

from nature.events.types import EventType
from nature.packs.types import (
    Block,
    Intervention,
    InterventionContext,
    OnTool,
    ToolPhase,
)


def _edit_read_first_action(ctx: InterventionContext):
    tc = ctx.tool_call
    if tc is None:
        return []
    frame = ctx.frame
    if frame is None:
        return []

    read_memory = frame.pack_state.get("read_memory")
    if read_memory is None:
        return []  # Pack not fully initialized — passthrough

    path = tc.tool_input.get("file_path", "")
    if not path:
        return []
    if not os.path.isabs(path):
        # Best-effort normalization; the tool will also normalize
        return []

    entry = read_memory.get(path)

    if entry is None:
        return [Block(
            reason=(
                "You must Read this file before editing it. "
                "Call Read on the file first so you can see the "
                "current content, then retry the Edit with an "
                "old_string copied from the Read output."
            ),
            trace_event=EventType.PATH_INVALID,
        )]

    # Expired entry → file was read at some point → allow
    if entry.expired:
        return []

    # Strict: old_string must appear in read content
    old_string = tc.tool_input.get("old_string", "")
    if old_string and entry.content is not None:
        if old_string not in entry.content:
            return [Block(
                reason=(
                    "The target text is not in any region you've read. "
                    "Read the relevant section of the file first, "
                    "then retry the Edit."
                ),
                trace_event=EventType.EDIT_MISS,
            )]

    return []


edit_read_first = Intervention(
    id="file_state.edit_read_first",
    kind="gate",
    trigger=OnTool(tool_name="Edit", phase=ToolPhase.PRE),
    action=_edit_read_first_action,
    description="Block Edit if target file/text not in ReadMemory.",
)
