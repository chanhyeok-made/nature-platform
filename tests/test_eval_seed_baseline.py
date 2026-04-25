"""Focused unit tests for `_commit_seed_baseline` — the helper that
folds every file the setup stage planted into a HEAD commit so the
llm_judge acceptance path (which diffs against HEAD) sees only the
agent's changes.

Without this step, seed files stay untracked and `git diff HEAD`
returns empty, causing the judge to see no agent work regardless of
what the agent actually did.

Double duty: this file pins the spec for the
`n3-seed-baseline-commit` benchmark case. At the baseline the helper
does not exist, so every test fails at collection time with
`ImportError`; a correct re-implementation makes all tests pass.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from nature.eval.runner import _commit_seed_baseline


def _init_repo(workspace: Path) -> Path:
    """Init an empty git repo at `workspace` with one initial commit
    so HEAD is a valid ref."""
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    subprocess.run(
        [
            "git", "-C", str(workspace),
            "-c", "user.email=t@t", "-c", "user.name=t",
            "-c", "commit.gpgsign=false",
            "commit", "--quiet", "--allow-empty", "-m", "init",
        ],
        check=True,
    )
    return workspace


def _rev_parse_head(workspace: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(workspace), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def test_commits_seed_files_into_head(tmp_path):
    ws = _init_repo(tmp_path)
    (ws / "seed.py").write_text("# seed body\n", encoding="utf-8")
    (ws / "tests").mkdir()
    (ws / "tests" / "test_seed.py").write_text(
        "def test_ok(): assert True\n", encoding="utf-8",
    )
    _commit_seed_baseline(ws)
    porcelain = subprocess.run(
        ["git", "-C", str(ws), "status", "--porcelain"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert porcelain == "", (
        f"expected a clean working tree after seed baseline commit, got:\n{porcelain}"
    )


def test_agent_edit_appears_in_diff_head_after_seed_commit(tmp_path):
    ws = _init_repo(tmp_path)
    (ws / "target.py").write_text("# seed\n", encoding="utf-8")
    _commit_seed_baseline(ws)
    # Simulate the agent editing the seeded file.
    (ws / "target.py").write_text("# seed\n# agent edit\n", encoding="utf-8")
    diff = subprocess.run(
        ["git", "-C", str(ws), "diff", "HEAD", "--", "target.py"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert "+# agent edit" in diff
    # Seed content must not appear as an addition — only the agent edit.
    added_lines = [
        ln for ln in diff.splitlines()
        if ln.startswith("+") and not ln.startswith("+++")
    ]
    assert added_lines == ["+# agent edit"]


def test_advances_head_even_when_no_seed_changes(tmp_path):
    """With no seed-stage files to add, the helper must still produce
    a commit (via `--allow-empty`). This guarantees the downstream
    `git diff HEAD` compares against a post-setup baseline even for
    tasks whose setup is effectively a no-op."""
    ws = _init_repo(tmp_path)
    head_before = _rev_parse_head(ws)
    _commit_seed_baseline(ws)
    head_after = _rev_parse_head(ws)
    assert head_before != head_after
