"""nature.eval — preset benchmarking framework.

Drives nature preset configurations through a battery of tasks,
collects pass/fail + cost + latency metrics per (task × preset) cell,
and stores results for cross-run comparison.

The package is intentionally self-contained: the eval CLI lives at
`nature eval` (see `nature/cli.py`), task definitions layer the same
way agents/presets/hosts do (builtin → user → project), and the
runner uses the existing admin/session HTTP API rather than reaching
into internals.

Phase 1 ships with three acceptance evaluators (pytest today, LLM
judge and shell-command stubs declared for extensibility) and four
setup strategies (local git, git clone at ref, patch file, none).
"""

from __future__ import annotations

from nature.eval.tasks import (
    AcceptancePytest,
    AcceptanceSpec,
    SetupApplyDiffFromCommit,
    SetupApplyPatchFile,
    SetupGitCloneAtRef,
    SetupNone,
    SetupSpec,
    Task,
    TaskTarget,
    load_task,
    load_tasks_registry,
)

__all__ = [
    "Task",
    "TaskTarget",
    "SetupSpec",
    "SetupApplyDiffFromCommit",
    "SetupApplyPatchFile",
    "SetupGitCloneAtRef",
    "SetupNone",
    "AcceptanceSpec",
    "AcceptancePytest",
    "load_task",
    "load_tasks_registry",
]
