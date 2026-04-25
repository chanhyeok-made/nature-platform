"""Evaluate probe success criteria against a run trace.

The probe runner produces a `ProbeTrace` — a flat record of the
tool_uses the model emitted, the tool_result each one returned,
the final assistant text, the turn count, and the workspace state
at the end. `evaluate(probe, trace)` walks the probe's declared
success criteria and returns a list of per-criterion outcomes.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nature.probe.probes import (
    ExpectFileState,
    ExpectFinalJson,
    ExpectFinalText,
    ExpectNoToolError,
    ExpectToolNotUsed,
    ExpectToolUse,
    ExpectTurnBound,
    Probe,
    SuccessCriterion,
)


@dataclass
class ToolUseRecord:
    """One tool_use emitted by the target model + its observed result."""

    index: int
    name: str
    input: dict[str, Any]
    result_text: str | None = None
    result_is_error: bool = False


@dataclass
class ProbeTrace:
    """What the runner collected during one probe execution. Used as
    the input to `evaluate()`."""

    tool_uses: list[ToolUseRecord]
    final_text: str
    turn_count: int
    hit_max_turns: bool
    workspace_root: Path | None = None
    # Human-readable error categories emitted by the runner layer
    # (not by success criteria) — e.g., "timeout", "llm_error",
    # "provider_error", "parse_error". Empty tuple when the model
    # simply produced an incomplete or wrong answer.
    runner_errors: tuple[str, ...] = ()


@dataclass
class CriterionOutcome:
    """Pass/fail for one criterion, plus a reason string for
    debugging. The reason is printed in the `nature probe report`
    output and helps classify failure modes across models."""

    passed: bool
    kind: str
    reason: str


@dataclass
class ProbeOutcome:
    """All criteria outcomes for one probe run, plus the bottom-line
    pass/fail (all-must-pass) and a fail category that summarizes
    the first failing criterion for easy aggregation."""

    passed: bool
    criteria: list[CriterionOutcome]
    fail_category: str | None  # "tool_use", "final_text", "file_state", etc.


# ──────────────────────────────────────────────────────────────────────


def _check_tool_use(c: ExpectToolUse, trace: ProbeTrace) -> CriterionOutcome:
    if c.at_index >= len(trace.tool_uses):
        return CriterionOutcome(
            False, "tool_use",
            f"expected tool_use at index {c.at_index} but only "
            f"{len(trace.tool_uses)} emitted",
        )
    tu = trace.tool_uses[c.at_index]
    if tu.name != c.name:
        return CriterionOutcome(
            False, "tool_use",
            f"expected tool_use[{c.at_index}].name={c.name!r}, got {tu.name!r}",
        )
    for field, expected in c.input_contains.items():
        actual = tu.input.get(field)
        if actual != expected:
            return CriterionOutcome(
                False, "tool_use",
                f"tool_use[{c.at_index}].input[{field!r}] = {actual!r}, "
                f"expected {expected!r}",
            )
    for field, pattern in c.input_regex.items():
        actual = str(tu.input.get(field, ""))
        # Probe authors: use inline flags to loosen matching, e.g.
        # `(?i)apply the patch` for case-insensitive or `(?s)` to
        # let `.` cross newlines. DOTALL is on by default.
        if not re.search(pattern, actual, re.DOTALL):
            return CriterionOutcome(
                False, "tool_use",
                f"tool_use[{c.at_index}].input[{field!r}] did not match "
                f"regex {pattern!r}; got {actual[:120]!r}",
            )
    return CriterionOutcome(True, "tool_use", "ok")


def _check_final_text(c: ExpectFinalText, trace: ProbeTrace) -> CriterionOutcome:
    text = trace.final_text or ""
    if c.regex is not None:
        if not re.search(c.regex, text, re.DOTALL):
            return CriterionOutcome(
                False, "final_text",
                f"final text did not match regex {c.regex!r}; "
                f"first 160 chars: {text[:160]!r}",
            )
    if c.contains is not None:
        if c.contains not in text:
            return CriterionOutcome(
                False, "final_text",
                f"final text did not contain {c.contains!r}; "
                f"first 160 chars: {text[:160]!r}",
            )
    return CriterionOutcome(True, "final_text", "ok")


def _check_final_json(c: ExpectFinalJson, trace: ProbeTrace) -> CriterionOutcome:
    text = (trace.final_text or "").strip()
    # Be forgiving of fenced JSON code blocks — the model may wrap
    # its answer in ```json ... ```. Strip the fence if present.
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2 and lines[-1].strip().startswith("```"):
            text = "\n".join(lines[1:-1])
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        return CriterionOutcome(
            False, "final_json",
            f"final text not valid JSON: {exc}; first 160 chars: {text[:160]!r}",
        )
    if not isinstance(obj, dict):
        return CriterionOutcome(
            False, "final_json",
            f"final JSON is not an object (type={type(obj).__name__})",
        )
    for k in c.required_keys:
        if k not in obj:
            return CriterionOutcome(
                False, "final_json",
                f"final JSON missing required key {k!r}; keys={list(obj.keys())}",
            )
    for k, expected in c.key_equals.items():
        if obj.get(k) != expected:
            return CriterionOutcome(
                False, "final_json",
                f"final JSON[{k!r}] = {obj.get(k)!r}, expected {expected!r}",
            )
    return CriterionOutcome(True, "final_json", "ok")


def _check_file_state(c: ExpectFileState, trace: ProbeTrace) -> CriterionOutcome:
    if trace.workspace_root is None:
        return CriterionOutcome(
            False, "file_state",
            "probe has file_state criterion but no workspace was provisioned",
        )
    path = trace.workspace_root / c.path
    if not path.exists():
        return CriterionOutcome(
            False, "file_state",
            f"file {c.path!r} does not exist after session",
        )
    body = path.read_text(encoding="utf-8", errors="replace")
    if c.equals is not None:
        if body != c.equals:
            return CriterionOutcome(
                False, "file_state",
                f"file {c.path!r} does not match expected content exactly "
                f"(got {len(body)} chars, expected {len(c.equals)} chars)",
            )
    if c.contains is not None:
        if c.contains not in body:
            return CriterionOutcome(
                False, "file_state",
                f"file {c.path!r} does not contain {c.contains!r}",
            )
    if c.regex is not None:
        if not re.search(c.regex, body, re.DOTALL):
            return CriterionOutcome(
                False, "file_state",
                f"file {c.path!r} did not match regex {c.regex!r}",
            )
    return CriterionOutcome(True, "file_state", "ok")


def _check_turn_bound(c: ExpectTurnBound, trace: ProbeTrace) -> CriterionOutcome:
    if trace.hit_max_turns:
        return CriterionOutcome(
            False, "turn_bound",
            f"model hit max_turns ceiling (expected ≤{c.max_turns})",
        )
    if trace.turn_count > c.max_turns:
        return CriterionOutcome(
            False, "turn_bound",
            f"used {trace.turn_count} turns, expected ≤{c.max_turns}",
        )
    return CriterionOutcome(True, "turn_bound", "ok")


def _check_no_tool_error(_c: ExpectNoToolError, trace: ProbeTrace) -> CriterionOutcome:
    bad = [tu for tu in trace.tool_uses if tu.result_is_error]
    if bad:
        return CriterionOutcome(
            False, "no_tool_error",
            f"{len(bad)} tool call(s) errored: "
            + ", ".join(f"{tu.name}(idx={tu.index})" for tu in bad[:3]),
        )
    return CriterionOutcome(True, "no_tool_error", "ok")


def _check_tool_not_used(c: ExpectToolNotUsed, trace: ProbeTrace) -> CriterionOutcome:
    hits = [tu for tu in trace.tool_uses if tu.name == c.name]
    if hits:
        return CriterionOutcome(
            False, "tool_not_used",
            f"{c.name} was used {len(hits)}× but probe forbids it",
        )
    return CriterionOutcome(True, "tool_not_used", "ok")


_CHECKERS = {
    "tool_use": _check_tool_use,
    "final_text": _check_final_text,
    "final_json": _check_final_json,
    "file_state": _check_file_state,
    "turn_bound": _check_turn_bound,
    "no_tool_error": _check_no_tool_error,
    "tool_not_used": _check_tool_not_used,
}


def evaluate(probe: Probe, trace: ProbeTrace) -> ProbeOutcome:
    """Walk every criterion; all must pass for probe to pass. The
    runner-layer errors short-circuit before criterion checks so a
    timeout doesn't get re-labelled as e.g. a tool_use failure."""
    if trace.runner_errors:
        return ProbeOutcome(
            passed=False,
            criteria=[CriterionOutcome(False, "runner", err) for err in trace.runner_errors],
            fail_category="runner:" + trace.runner_errors[0],
        )

    outcomes: list[CriterionOutcome] = []
    for c in probe.success:
        checker = _CHECKERS[c.kind]
        outcomes.append(checker(c, trace))

    fail = next((o for o in outcomes if not o.passed), None)
    return ProbeOutcome(
        passed=fail is None,
        criteria=outcomes,
        fail_category=fail.kind if fail else None,
    )


__all__ = ["ToolUseRecord", "ProbeTrace", "CriterionOutcome", "ProbeOutcome", "evaluate"]
