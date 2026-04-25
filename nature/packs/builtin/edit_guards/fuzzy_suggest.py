"""fuzzy_suggest — Edit "old_string not found" → closest-match hint.

Fires on a POST tool_call for Edit with `result_is_error=True`. Reads
the target file from disk, walks windowed slices around the length of
the missed `old_string`, and picks the best `difflib` match. Returns
two effects:

- `ModifyToolResult(append_hint=...)` — appends the suggestion to the
  tool error so the next LLM turn sees it alongside the original error.
- `EmitEvent(EDIT_MISS, payload)` — structured record for dashboards
  and the downstream loop_detector (which is POST_EFFECT and reads
  primary_effects).

Scope notes:

- Silent on any read/encoding/difflib failure — logs and skips. A bad
  fuzzy suggestion is worse than none.
- Length window: we sample slices of the file the same line count as
  the intended old_string, then fuzzy-compare. Cheaper than full-file
  ratio and produces more actionable suggestions.
- No I/O budget — callers should set a reasonable max file size
  elsewhere (M3 ships without that guard; Phase 3 budget handles it).
"""

from __future__ import annotations

import difflib
import logging
from pathlib import Path

from nature.events.payloads import EditMissPayload
from nature.events.types import EventType
from nature.packs.types import (
    EmitEvent,
    Intervention,
    InterventionContext,
    ModifyToolResult,
    OnTool,
    ToolPhase,
)

logger = logging.getLogger(__name__)

_MAX_SAMPLE_LENGTH = 2000   # cutoff for the difflib similarity ratio call
_TOP_N_WINDOWS = 1          # how many suggestions to surface


def _find_closest_window(
    file_path: str,
    old_string: str,
) -> tuple[str, int] | None:
    """Return (matched_text, 1-based line number) for the best window,
    or None on any failure. Best-effort — logs but never raises."""
    try:
        text = Path(file_path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        logger.debug("fuzzy_suggest: cannot read %r", file_path, exc_info=True)
        return None

    target = old_string.strip()
    if not target:
        return None
    target_lines = max(1, target.count("\n") + 1)

    file_lines = text.splitlines(keepends=True)
    if not file_lines:
        return None

    best_ratio = 0.0
    best_window = ""
    best_lineno = 1

    # Slide a target_lines-sized window over the file, cheap ratio check.
    for start in range(len(file_lines) - target_lines + 1):
        window = "".join(file_lines[start : start + target_lines])
        if not window.strip():
            continue
        # Truncate the comparison to avoid pathological O(n^2) on huge windows.
        a = target[:_MAX_SAMPLE_LENGTH]
        b = window[:_MAX_SAMPLE_LENGTH]
        ratio = difflib.SequenceMatcher(None, a, b).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_window = window
            best_lineno = start + 1

    if best_ratio < 0.5 or not best_window:
        return None
    return (best_window.rstrip(), best_lineno)


def _format_hint(match_text: str, lineno: int) -> str:
    snippet = match_text[:600]
    if len(match_text) > 600:
        snippet += "\n    …"
    return (
        "[edit_guards.fuzzy_suggest] old_string not found. "
        f"Closest match near line {lineno}:\n\n"
        f"{snippet}\n\n"
        "Re-read the file before retrying — your old_string may be stale."
    )


def _fuzzy_suggest_action(ctx: InterventionContext):
    tc = ctx.tool_call
    if tc is None or tc.result_is_error is not True:
        return []
    inp = tc.tool_input or {}
    old_string = inp.get("old_string", "")
    file_path = inp.get("file_path", "")
    if not old_string or not file_path:
        return []

    match = _find_closest_window(file_path, old_string)
    edit_miss = EmitEvent(
        event_type=EventType.EDIT_MISS,
        payload=EditMissPayload(
            file=file_path,
            fuzzy_match=match[0] if match else None,
            lineno=match[1] if match else None,
        ),
    )
    if match is None:
        return [edit_miss]
    return [
        ModifyToolResult(append_hint=_format_hint(match[0], match[1])),
        edit_miss,
    ]


fuzzy_suggest = Intervention(
    id="edit_guards.fuzzy_suggest",
    kind="listener",
    trigger=OnTool(
        tool_name="Edit",
        phase=ToolPhase.POST,
        where=lambda tc: tc.result_is_error is True,
    ),
    action=_fuzzy_suggest_action,
    description="On Edit old_string miss, append closest-match hint.",
)
