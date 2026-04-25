"""budget — per-frame code-level caps to prevent analysis paralysis.

Code-level enforcement of `max_*_per_frame` caps. Motivating incident:
a session that burnt 36 Reads and 0 Edits in 240 seconds — a
prompt-based "don't read everything" rule was not enforced.

Current capabilities:

- `reads_budget` (Phase 3.1): counts Read + Grep + Glob calls in the
  conversation body. At 80% of the limit, a Contributor injects a
  "start editing" footer hint. At 100%, a Gate blocks further read-
  family calls with a suggestion to proceed to Edit or delegate.

Planned but not yet shipped:

- `turns_budget` (Phase 3.2): hard ceiling on turns per frame.
- `tools_per_turn` (Phase 3.3): max tool calls per single LLM turn.

All budgets derive their current count from the conversation body —
same pure-function-of-state pattern as edit_guards.
"""

from nature.packs.builtin.budget.pack import budget_pack, install

__all__ = ["budget_pack", "install"]
