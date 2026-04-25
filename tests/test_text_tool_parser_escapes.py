"""Regression — local models (qwen-family) emit JSON strings with
shell-style escape sequences like `\\:` that the strict JSON spec
rejects. `extract_tool_calls_from_text` must tolerate these via
relaxed re-parse so a valid-shaped Agent call isn't dropped by a
spurious backslash inside a prompt string.
"""

from __future__ import annotations

from nature.agent.text_tool_parser import (
    _relax_json_escapes,
    _try_parse_tool_call,
    extract_tool_calls_from_text,
)


KNOWN = {"Agent", "Read", "Bash", "Edit", "Write"}


def test_relax_strips_invalid_single_char_escapes():
    # `\:` is not a valid JSON escape; relax drops the backslash.
    out = _relax_json_escapes(r'{"s": "a\:b"}')
    assert out == '{"s": "a:b"}'


def test_relax_strips_invalid_paren_escapes():
    # `\(` / `\)` are common shell-style escapes local models emit.
    out = _relax_json_escapes(r'{"s": "a\(b\)"}')
    assert out == '{"s": "a(b)"}'


def test_relax_preserves_legal_escapes():
    # `\n`, `\"`, `\\`, `\t`, `\u1234`, `\/`, `\b`, `\f`, `\r` all
    # stay — only truly invalid escapes get touched.
    src = r'{"s": "a\nb\"c\\d\te\u0041f"}'
    assert _relax_json_escapes(src) == src


def test_try_parse_tool_call_retries_on_invalid_escape():
    json_str = r'{"name": "Agent", "arguments": {"name": "core", "prompt": "use \: sparingly"}}'
    parsed = _try_parse_tool_call(json_str, KNOWN)
    assert parsed is not None
    assert parsed.name == "Agent"
    assert parsed.input["name"] == "core"
    assert parsed.input["prompt"] == "use : sparingly"  # \\: relaxed to :


def test_extract_tool_calls_parses_qwen_emission_with_invalid_escape():
    """End-to-end regression — stage-1 v3 reproducer. Qwen emitted a
    tool call as text (no native tool_use block) with `\\:` inside
    the prompt. The old parser failed JSON strict mode and returned
    zero tool calls; frame resolved with no delegation and x2 cell
    FAILed even though qwen structurally knew what to do."""
    text = (
        r'{"name": "Agent", "arguments": '
        r'{"name": "core", "prompt": "escape `:` via `\:` only in '
        r'`_describe` path"}}'
        '\n\nAdditional prose explaining.'
    )
    remaining, tool_uses = extract_tool_calls_from_text(text, KNOWN)
    assert len(tool_uses) == 1
    tu = tool_uses[0]
    assert tu.name == "Agent"
    assert tu.input.get("name") == "core"
    assert "escape" in tu.input.get("prompt", "")
    # Only the JSON region should be consumed; prose survives.
    assert "Additional prose" in remaining
