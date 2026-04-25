"""Tests for the llm_judge acceptance helpers.

The judge evaluator itself runs end-to-end against a live server, so
the parts worth unit-testing are the deterministic bits:

- Prompt builder: rubric + diff + optional final message, with size
  caps so a giant diff doesn't blow past the judge's context.
- Response parser: extract the last `{"verdict": …}` JSON block with
  a loose PASS/FAIL keyword fallback when the judge forgot to obey
  the JSON template.
- Event-log pivot: pick the last assistant→user message from a
  snapshot's event list.
"""

from __future__ import annotations

import pytest

from nature.eval.runner import (
    _build_judge_prompt,
    _extract_final_user_text,
    _parse_judge_verdict,
)


# ──────────────────────────────────────────────────────────────────────
# Prompt builder
# ──────────────────────────────────────────────────────────────────────


def test_build_judge_prompt_includes_rubric_and_diff():
    prompt = _build_judge_prompt(
        rubric="Must pass all existing tests.",
        diff="diff --git a/x b/x\n@@ foo\n- bar\n+ baz",
        final_message="Done.",
    )
    assert "Must pass all existing tests." in prompt
    assert "diff --git" in prompt
    assert "Done." in prompt
    assert "verdict" in prompt  # response template in the preamble


def test_build_judge_prompt_handles_empty_diff_and_final():
    prompt = _build_judge_prompt(rubric="do the thing", diff="", final_message=None)
    assert "no changes" in prompt
    assert "no final response" in prompt


def test_build_judge_prompt_truncates_large_diff():
    big = "x" * 60_000
    prompt = _build_judge_prompt(rubric="r", diff=big, final_message=None)
    # Cap is 40k per the builder; template itself may have a stray 'x'
    # or two ("exactly"), so give 100 chars of slack.
    assert "[truncated]" in prompt
    assert prompt.count("x") <= 40_100


# ──────────────────────────────────────────────────────────────────────
# Response parser
# ──────────────────────────────────────────────────────────────────────


def test_parse_verdict_extracts_last_json_block():
    text = (
        "I looked at the diff and the tests pass.\n"
        '{"verdict": "pass", "reason": "tests green, diff looks correct"}'
    )
    passed, reason = _parse_judge_verdict(text)
    assert passed is True
    assert "tests green" in reason


def test_parse_verdict_fail_path():
    text = '{"verdict": "fail", "reason": "didnt implement required API"}'
    passed, reason = _parse_judge_verdict(text)
    assert passed is False
    assert "required API" in reason


def test_parse_verdict_prefers_last_json_when_multiple():
    """If the judge prototypes then corrects itself, the last JSON is
    the one we should trust."""
    text = (
        '{"verdict": "pass", "reason": "first pass"}\n'
        "Actually on reflection:\n"
        '{"verdict": "fail", "reason": "spotted missing case"}'
    )
    passed, reason = _parse_judge_verdict(text)
    assert passed is False
    assert "missing case" in reason


def test_parse_verdict_plain_text_fallback_pass():
    passed, reason = _parse_judge_verdict(
        "Looks good. Everything clean. PASS",
    )
    assert passed is True
    assert "plain text" in reason


def test_parse_verdict_plain_text_fallback_fail():
    passed, reason = _parse_judge_verdict(
        "No. Missing half the spec. FAIL.",
    )
    assert passed is False


def test_parse_verdict_raises_on_ambiguous_text():
    with pytest.raises(ValueError):
        _parse_judge_verdict("Hmm, maybe PASS, maybe FAIL — I can't decide.")
    with pytest.raises(ValueError):
        _parse_judge_verdict("")


def test_parse_verdict_ignores_invalid_json():
    text = (
        '{"verdict": "pass", "reason": broken-json}\n'
        '{"verdict": "fail", "reason": "the previous one was malformed"}'
    )
    passed, _ = _parse_judge_verdict(text)
    assert passed is False


# ──────────────────────────────────────────────────────────────────────
# Snapshot → final message
# ──────────────────────────────────────────────────────────────────────


def test_extract_final_user_text_picks_last_to_user():
    events = [
        {
            "type": "message.appended",
            "payload": {
                "to": "user",
                "content": [{"type": "text", "text": "first"}],
            },
        },
        {
            "type": "tool.completed",
            "payload": {"output": "noise"},
        },
        {
            "type": "message.appended",
            "payload": {
                "to": "user",
                "content": [
                    {"type": "text", "text": "second line 1"},
                    {"type": "text", "text": "second line 2"},
                ],
            },
        },
        {
            "type": "message.appended",
            "payload": {
                "to": "researcher",   # not to user
                "content": [{"type": "text", "text": "delegate prompt"}],
            },
        },
    ]
    text = _extract_final_user_text(events)
    assert "second line 1" in text
    assert "second line 2" in text
    assert "first" not in text


def test_extract_final_user_text_empty_when_none():
    assert _extract_final_user_text([]) == ""
    assert _extract_final_user_text([
        {"type": "tool.completed", "payload": {"output": "x"}},
    ]) == ""
