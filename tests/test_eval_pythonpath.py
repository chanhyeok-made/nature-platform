"""Focused unit tests for `AcceptancePytest.pythonpath` and its
propagation into the pytest subprocess env.

These tests pin the contract consumed by every external-repo task
with a `src/` layout (pluggy, attrs, …): without the right directory
on `PYTHONPATH`, pytest imports whatever copy of the package lives in
the ambient venv instead of the workspace edits, and the agent's
work is silently invisible to the acceptance check.

Double duty: this file also acts as the spec for the
`n2-pytest-pythonpath` benchmark case. At the task's baseline
(before the 2026-04-18 fix) all four tests fail; a correct
re-implementation makes them pass.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

from nature.eval.runner import _run_pytest_acceptance
from nature.eval.tasks import AcceptancePytest


def test_pythonpath_field_defaults_to_empty_list():
    spec = AcceptancePytest(node_ids=["tests/test_foo.py"])
    assert spec.pythonpath == []


def test_pythonpath_field_accepts_list_of_paths():
    spec = AcceptancePytest(
        node_ids=["tests/test_foo.py"],
        pythonpath=["src", "lib"],
    )
    assert spec.pythonpath == ["src", "lib"]


def test_runner_prepends_resolved_pythonpath_to_env(tmp_path):
    spec = AcceptancePytest(
        node_ids=["-k", "never_matches"],
        pythonpath=["src"],
        timeout_sec=30,
    )
    fake = MagicMock(returncode=0, stdout="", stderr="")
    # Clear ambient PYTHONPATH so the eval-supplied entry is the only
    # value on the subprocess's PYTHONPATH.
    clean_env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    with patch.dict(os.environ, clean_env, clear=True):
        with patch(
            "nature.eval.runner.subprocess.run", return_value=fake,
        ) as run_mock:
            _run_pytest_acceptance(tmp_path, spec)
    env = run_mock.call_args.kwargs["env"]
    expected = str((tmp_path / "src").resolve())
    assert env["PYTHONPATH"] == expected


def test_runner_preserves_ambient_pythonpath_after_eval_entry(tmp_path):
    spec = AcceptancePytest(
        node_ids=["-k", "never_matches"],
        pythonpath=["src"],
        timeout_sec=30,
    )
    fake = MagicMock(returncode=0, stdout="", stderr="")
    ambient = "/some/ambient/path"
    with patch.dict(os.environ, {"PYTHONPATH": ambient}):
        with patch(
            "nature.eval.runner.subprocess.run", return_value=fake,
        ) as run_mock:
            _run_pytest_acceptance(tmp_path, spec)
    env = run_mock.call_args.kwargs["env"]
    parts = env["PYTHONPATH"].split(os.pathsep)
    assert parts[0] == str((tmp_path / "src").resolve())
    assert ambient in parts
