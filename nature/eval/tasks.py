"""Task schema + loader for the nature eval system.

A task defines one benchmark case: a prompt, the workspace setup that
must exist before the agent is invoked, and the acceptance check that
decides pass/fail. The loader layers builtin → user → project (same
pattern as agents/presets/hosts) so projects can ship their own cases
alongside the ones nature bundles.

Schema (pydantic-validated, stored as `<id>.json`):

    {
      "id": "n1-pack-discovery",
      "title": "Pack discovery loader",
      "category": "feature",
      "tags": ["packs", "new-module"],
      "size": "medium",
      "prompt": "Add a file-based pack discovery loader…",

      "target": {                          // where the work lands
        "repo": "local",                   // "local" or a git URL
        "ref": "b6b8ff5^"                  // commit / branch / tag
      },

      "setup": {                           // workspace prep
        "type": "apply_diff_from_commit",  // discriminator
        "apply_commit": "b6b8ff5",
        "diff_scope": ["tests/test_pack_discovery.py"]
      },

      "acceptance": {                      // pass/fail check
        "type": "pytest",                  // discriminator
        "node_ids": [
          "tests/test_pack_discovery.py::test_install_discovered_runs_install_for_each_pack"
        ],
        "timeout_sec": 240
      }
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, Field


# ──────────────────────────────────────────────────────────────────────
# Target — the repo + ref the task runs against
# ──────────────────────────────────────────────────────────────────────


class TaskTarget(BaseModel):
    """Which repo and ref the task workspace starts from.

    `repo == "local"` → use the current nature checkout (tasks that
    benchmark nature itself against a past commit of nature).
    Otherwise `repo` is a git URL that the runner clones into a tmp
    dir before applying setup.
    """

    model_config = ConfigDict(extra="forbid")

    repo: str = "local"
    ref: str  # commit sha, branch, tag, or `<sha>^` style ref


# ──────────────────────────────────────────────────────────────────────
# Setup strategies — how to prepare the workspace
# ──────────────────────────────────────────────────────────────────────


class SetupApplyDiffFromCommit(BaseModel):
    """Check out `target.ref`, then apply the *files-only* diff of
    `apply_commit` restricted to `diff_scope`. Used to plant
    acceptance tests on top of a past workspace while the
    implementation under test is still absent."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["apply_diff_from_commit"] = "apply_diff_from_commit"
    apply_commit: str
    diff_scope: list[str] = Field(default_factory=list)


class SetupApplyPatchFile(BaseModel):
    """Check out `target.ref`, then apply a bundled patch file.
    Useful for synthetic cases where no git commit carries the
    seed — the patch file sits next to the task JSON."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["apply_patch_file"] = "apply_patch_file"
    patch_path: str  # relative to the task's directory


class SetupGitCloneAtRef(BaseModel):
    """Clone an external repo and check out `target.ref`, then
    optionally apply a test diff the same way as
    `apply_diff_from_commit`."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["git_clone_at_ref"] = "git_clone_at_ref"
    # Tests added on top — same shape as apply_diff_from_commit's
    # fields so runners can share code paths.
    apply_commit: str | None = None
    diff_scope: list[str] = Field(default_factory=list)


class SetupNone(BaseModel):
    """Use `target.ref` as-is — no diff or patch applied. Usually
    paired with a synthetic bundle that already contains everything
    the agent needs to see."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["none"] = "none"


SetupSpec = Union[
    SetupApplyDiffFromCommit,
    SetupApplyPatchFile,
    SetupGitCloneAtRef,
    SetupNone,
]


# ──────────────────────────────────────────────────────────────────────
# Acceptance evaluators — how to score pass/fail
# ──────────────────────────────────────────────────────────────────────


class AcceptancePytest(BaseModel):
    """Run pytest on the given node ids inside the task workspace.
    Pass iff pytest exits 0."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["pytest"] = "pytest"
    node_ids: list[str]
    timeout_sec: int = 180
    # Optional extra pytest args (e.g. ["-x", "--tb=short"]).
    extra_args: list[str] = Field(default_factory=list)
    # Paths (relative to the workspace root) prepended to PYTHONPATH
    # before pytest runs. Essential for repos that use a `src/` layout
    # (e.g. pluggy, attrs) so the workspace's edits are imported
    # instead of whatever copy happens to live in the caller's venv.
    pythonpath: list[str] = Field(default_factory=list)


# Placeholder shapes declared so task authors can write forward-
# compatible JSON. Phase 1 runner rejects them with a clear error.
class AcceptanceShellCommand(BaseModel):
    """Run an arbitrary shell command in the workspace; non-zero exit
    is failure. Not implemented in Phase 1."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["shell_command"] = "shell_command"
    command: str
    timeout_sec: int = 180


class AcceptanceLLMJudge(BaseModel):
    """Send the agent's final response + workspace diff to a judge LLM
    with a rubric. Not implemented in Phase 1."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["llm_judge"] = "llm_judge"
    judge_preset: str  # preset name for the judge session
    rubric: str
    timeout_sec: int = 120


AcceptanceSpec = Union[
    AcceptancePytest,
    AcceptanceShellCommand,
    AcceptanceLLMJudge,
]


# ──────────────────────────────────────────────────────────────────────
# Task — the top-level unit
# ──────────────────────────────────────────────────────────────────────


class Task(BaseModel):
    """One benchmark case loaded from `<id>.json`."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str = ""
    category: str = "bug-fix"      # "bug-fix" | "feature" | "refactor" | …
    tags: list[str] = Field(default_factory=list)
    size: str = "medium"           # "trivial" | "small" | "medium" | "large"
    prompt: str

    target: TaskTarget
    setup: SetupSpec = Field(discriminator="type")
    acceptance: AcceptanceSpec = Field(discriminator="type")

    # Directory the task was loaded from. Set by the loader, not
    # written to disk, so setups that reference relative paths
    # (e.g. `patch_path`) can resolve them.
    source_dir: Path = Field(default_factory=Path, exclude=True)


# ──────────────────────────────────────────────────────────────────────
# Loader
# ──────────────────────────────────────────────────────────────────────


def _builtin_dir() -> Path:
    """Directory of shipped task JSONs (inside the nature package)."""
    return Path(__file__).parent / "builtin" / "cases"


def _user_dir() -> Path:
    """User-level task directory (`~/.nature/eval/cases/`)."""
    from nature.config.settings import get_nature_home
    return get_nature_home() / "eval" / "cases"


def _project_dir(project_dir: Path | str | None) -> Path | None:
    """Project-level task directory (`<project>/.nature/eval/cases/`)."""
    if project_dir is None:
        return None
    return Path(project_dir) / ".nature" / "eval" / "cases"


def _load_from_dir(dir_path: Path) -> dict[str, Task]:
    """Scan `<dir>/<id>/task.json` or `<dir>/<id>.json` entries and
    return them keyed by id. Both layouts are supported so self-
    contained tasks (bundled patch + helper files) can live in their
    own directory while simple tasks stay as one file."""
    tasks: dict[str, Task] = {}
    if not dir_path.exists():
        return tasks

    # Flat `<id>.json` files.
    for json_path in sorted(dir_path.glob("*.json")):
        task = _load_task_file(json_path, json_path.parent)
        tasks[task.id] = task

    # Sub-directories holding `<dir>/task.json`.
    for sub in sorted(p for p in dir_path.iterdir() if p.is_dir()):
        task_file = sub / "task.json"
        if not task_file.exists():
            continue
        task = _load_task_file(task_file, sub)
        tasks[task.id] = task

    return tasks


def _load_task_file(json_path: Path, source_dir: Path) -> Task:
    try:
        raw = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"failed to parse task at {json_path}: {exc}") from exc
    task = Task.model_validate(raw)
    task.source_dir = source_dir
    return task


def load_tasks_registry(
    project_dir: Path | str | None = None,
) -> dict[str, Task]:
    """Merge builtin → user → project, later layers wins on id clash."""
    merged = _load_from_dir(_builtin_dir())
    merged.update(_load_from_dir(_user_dir()))
    proj = _project_dir(project_dir)
    if proj is not None:
        merged.update(_load_from_dir(proj))
    return merged


def load_task(
    task_id: str,
    project_dir: Path | str | None = None,
) -> Task:
    """Load a single task by id. Raises KeyError when unknown."""
    registry = load_tasks_registry(project_dir=project_dir)
    if task_id not in registry:
        raise KeyError(
            f"unknown task {task_id!r}. Known: {sorted(registry)}"
        )
    return registry[task_id]
