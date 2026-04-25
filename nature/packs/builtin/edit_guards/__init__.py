"""edit_guards — Phase 2 Edit feedback Pack.

Four interventions that together address the 2026-04-15 benchmark's
headline failure: qwen2.5-coder:32b made 10 consecutive Edit attempts
on a hallucinated `old_string`, every one rejected, zero framework
intervention. Small models forget file contents across turns; Edit's
default "old_string not found" error is too terse to help them
re-anchor; and no loop detection stops the retry storm.

This Pack fixes all three:

- `fuzzy_suggest` (listener, PRIMARY): when Edit fails with a miss,
  run `difflib.get_close_matches` against the file's content windows
  and attach the closest match to the tool error. Also emits EDIT_MISS.
- `reread_hint` (contributor): when the most recent message in the
  body is a failed Edit tool_result, fold a "re-read before retrying"
  note into the footer. No new state required — the body itself is
  the signal.
- `loop_detector` (listener, POST_EFFECT): watches fuzzy_suggest's
  EDIT_MISS emission, tracks consecutive same-input-hash failures in
  `frame.budget_counts`, and emits LOOP_DETECTED when the streak hits
  the threshold. Also flips `frame.budget_counts["edit_loop_blocked"]`
  so the next Edit pre-hook refuses execution.
- `loop_block` (gate, on Edit PRE): reads
  `frame.budget_counts["edit_loop_blocked"]` and returns Block when
  set. Clears itself when the user intervenes or a successful Edit
  lands (future enhancement — M3 ships the baseline guard).

See pack_architecture.md §12 for the original pilot sketch.
"""

from nature.packs.builtin.edit_guards.pack import (
    edit_guards_capability,
    edit_guards_pack,
    install,
)

__all__ = ["edit_guards_capability", "edit_guards_pack", "install"]
