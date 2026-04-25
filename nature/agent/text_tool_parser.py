"""Parse tool calls from plain text output.

Some models (especially local ones via Ollama) don't use the tool_calls API
properly and instead output tool calls as JSON in their text response.

This module detects and parses such patterns, converting them into proper
ToolUseContent blocks so the agent loop can execute them.

Supported patterns:
  {"name": "ToolName", "arguments": {...}}
  {"name": "ToolName", "input": {...}}
  ```json\n{"name": "ToolName", ...}\n```
"""

from __future__ import annotations

import json
import re
from typing import Any

from nature.protocols.message import ToolUseContent
from nature.utils.ids import generate_tool_use_id


def _find_json_objects(text: str) -> list[tuple[str, int, int]]:
    """Find top-level JSON objects in text by brace matching.

    Returns list of (json_str, start_pos, end_pos).
    Handles nested braces correctly.
    """
    results = []
    i = 0
    while i < len(text):
        if text[i] == "{":
            depth = 0
            start = i
            in_string = False
            escape = False
            for j in range(i, len(text)):
                c = text[j]
                if escape:
                    escape = False
                    continue
                if c == "\\":
                    escape = True
                    continue
                if c == '"' and not escape:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        results.append((text[start : j + 1], start, j + 1))
                        i = j + 1
                        break
            else:
                break
        else:
            i += 1
    return results


# Fenced code block pattern
_FENCED_RE = re.compile(r"```(?:json)?\s*\n(\{.+?\})\s*\n```", re.DOTALL)


def extract_tool_calls_from_text(
    text: str,
    known_tool_names: set[str],
) -> tuple[str, list[ToolUseContent]]:
    """Extract tool calls from text output.

    Returns (remaining_text, tool_use_blocks).
    """
    if not text or not known_tool_names:
        return text, []

    tool_uses: list[ToolUseContent] = []
    regions_to_remove: list[tuple[int, int]] = []

    # 1. Fenced code blocks first
    for match in _FENCED_RE.finditer(text):
        json_str = match.group(1)
        parsed = _try_parse_tool_call(json_str, known_tool_names)
        if parsed:
            tool_uses.append(parsed)
            regions_to_remove.append((match.start(), match.end()))

    # 2. Inline JSON objects (brace-matched)
    for json_str, start, end in _find_json_objects(text):
        # Skip if already found in a fenced block
        if any(s <= start and end <= e for s, e in regions_to_remove):
            continue
        parsed = _try_parse_tool_call(json_str, known_tool_names)
        if parsed:
            tool_uses.append(parsed)
            regions_to_remove.append((start, end))

    # 3. Bare tool calls: "ToolName {json}" or "ToolName description... {json}"
    if not tool_uses:
        for json_str, start, end in _find_json_objects(text):
            # Look backwards from the JSON for a known tool name
            prefix = text[:start].rstrip()
            for name in known_tool_names:
                # Match "ToolName" or "ToolName some description text"
                if prefix == name or prefix.startswith(name + " "):
                    try:
                        obj = json.loads(json_str)
                        if isinstance(obj, dict):
                            tool_uses.append(ToolUseContent(
                                id=generate_tool_use_id(),
                                name=name,
                                input=obj,
                            ))
                            # Remove from tool name to end of JSON
                            line_start = text.rfind("\n", 0, start) + 1
                            regions_to_remove.append((line_start, end))
                            break
                    except json.JSONDecodeError:
                        pass

    # 4. Python call syntax: "ToolName(key="value", key2=123)"
    if not tool_uses:
        _PY_CALL_RE = re.compile(
            r'\b(' + '|'.join(re.escape(n) for n in known_tool_names) + r')\((.+?)\)',
            re.DOTALL | re.IGNORECASE,
        )
        _canonical_map = {n.lower(): n for n in known_tool_names}
        for match in _PY_CALL_RE.finditer(text):
            name = _canonical_map.get(match.group(1).lower(), match.group(1))
            args_str = match.group(2)
            try:
                # Parse Python-style kwargs: key="value", key2=123
                parsed_input = {}
                for kv in re.finditer(r'(\w+)\s*=\s*(".*?"|\'.*?\'|[\w.*/-]+)', args_str):
                    key = kv.group(1)
                    val = kv.group(2).strip("\"'")
                    parsed_input[key] = val
                if parsed_input:
                    tool_uses.append(ToolUseContent(
                        id=generate_tool_use_id(),
                        name=name,
                        input=parsed_input,
                    ))
                    regions_to_remove.append((match.start(), match.end()))
            except Exception:
                pass

    # 5. Last-resort Agent fallback: models sometimes emit the Agent
    #    invocation as descriptive prose + the *args dict directly*,
    #    e.g. `-Agent tool call with parameters: {"name":"core","prompt":"…"}`.
    #    The earlier patterns expect either `Agent(...)`, fenced JSON
    #    with a top-level `name` of a known tool, or a bare prefix.
    #    None of them catches this loose shape, so we add an
    #    Agent-specific recognizer.
    if not tool_uses and "Agent" in known_tool_names:
        agent_uses = _extract_loose_agent_calls(text)
        for tu, start, end in agent_uses:
            tool_uses.append(tu)
            regions_to_remove.append((start, end))

    if not tool_uses:
        return text, []

    # Remove matched regions (reverse order to preserve positions)
    remaining = text
    for start, end in sorted(regions_to_remove, reverse=True):
        remaining = remaining[:start] + remaining[end:]
    remaining = remaining.strip()

    return remaining, tool_uses


_VALID_JSON_ESCAPES = set('"\\/bfnrtu')
# Match either a legal `\\` pair (two backslashes = one backslash, the
# JSON escape for backslash), OR a `\X` where X is not one of the JSON-
# legal escape characters. The alternation preserves `\\` intact and
# only strips the backslash from truly invalid sequences. Without the
# first branch, `\\d` (legal JSON for `\d`) would be miscounted as
# `\` + `d` and collapsed to `d`, breaking legal input.
_INVALID_ESCAPE_RE = re.compile(r'\\\\|\\([^"\\/bfnrtu])')


def _relax_json_escapes(json_str: str) -> str:
    """Strip invalid backslash-escapes so `json.loads` accepts the string.

    Qwen-style local models frequently emit JSON string values with
    shell-style escape sequences like `\\:` or `\\(` — valid in the
    model's world, but not in the JSON spec. The standard parser then
    dies with `invalid \\escape` and the whole tool call is dropped,
    which stage-1 eval `x2-click-zsh-colon × all-qwen-coder-32b`
    hit (qwen emitted a valid-shaped Agent call containing `\\:`
    inside the prompt string and we silently parsed zero tools).

    Replaces `\\X` (where X is not one of the JSON-legal escape
    characters) with just `X`. Double backslashes are preserved so
    legal `\\<char>` strings still parse correctly.
    """
    return _INVALID_ESCAPE_RE.sub(
        lambda m: m.group(0) if m.group(1) is None else m.group(1),
        json_str,
    )


def _try_parse_tool_call(json_str: str, known_names: set[str]) -> ToolUseContent | None:
    """Try to parse a JSON string as a tool call.

    First attempt is strict `json.loads`. On failure, retry once with
    `_relax_json_escapes` so tolerant recovery covers the common
    local-model emission pattern without masking genuinely broken
    structure (brace mismatch, truncation, etc.).
    """
    try:
        obj = json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        try:
            obj = json.loads(_relax_json_escapes(json_str))
        except (json.JSONDecodeError, TypeError):
            return None

    if not isinstance(obj, dict) or "name" not in obj:
        return None

    name = obj["name"]
    if not isinstance(name, str):
        return None
    if name not in known_names:
        # Case-insensitive fallback — phi4 and others occasionally emit
        # lowercase tool names ("grep", "bash") even when the catalog
        # uses canonical casing. Normalize before rejecting.
        canonical = {n.lower(): n for n in known_names}.get(name.lower())
        if canonical is None:
            return None
        name = canonical

    tool_input = obj.get("arguments") or obj.get("input") or {}

    return ToolUseContent(
        id=generate_tool_use_id(),
        name=name,
        input=tool_input,
    )


def _extract_loose_agent_calls(
    text: str,
) -> list[tuple[ToolUseContent, int, int]]:
    """Extract Agent invocations whose JSON body is the *args dict directly*.

    Triggered when the text mentions "Agent" and contains a JSON object
    with a `prompt` key (the required field of AgentInput). Matches the
    common qwen / deepseek pattern:

        -Agent tool call with the following parameters: {"name": "core",
                                                          "prompt": "…"}
        Calling Agent: {"prompt": "…"}
        Use the Agent tool: {"name": "core", "prompt": "…"}

    Returns a list of (ToolUseContent, start, end) so the caller can
    strip the matched region from the remaining text.
    """
    if "agent" not in text.lower():
        return []

    found: list[tuple[ToolUseContent, int, int]] = []
    for json_str, start, end in _find_json_objects(text):
        try:
            obj = json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(obj, dict):
            continue
        # Required: "prompt" key (AgentInput's required field)
        if "prompt" not in obj:
            continue
        # Skip if it already looks like a structured tool envelope —
        # the earlier parsers handle those.
        if "arguments" in obj:
            continue
        # Skip when "name" looks like a TOOL NAME rather than an agent
        # name (e.g. {"name": "Bash", ...} → not an Agent delegation).
        if isinstance(obj.get("name"), str) and obj["name"] in {
            "Bash", "Read", "Write", "Edit", "Glob", "Grep",
        }:
            continue
        # Verify "Agent" appears within ~120 chars before the JSON,
        # to avoid grabbing arbitrary {prompt:…} objects with no
        # delegation context.
        prefix_window = text[max(0, start - 120):start]
        if "Agent" not in prefix_window and "agent" not in prefix_window:
            continue

        # Skip when `name` is missing — the framework rejects nameless
        # Agent calls now (session `8ed7d997` exposed the silent
        # default-to-core failure mode). Forcing the model to emit
        # `name` explicitly lets `_handle_delegation` enforce its
        # self-delegation guard properly.
        if not obj.get("name"):
            continue
        agent_input = {
            "prompt": obj["prompt"],
            "name": obj["name"],
        }
        found.append((
            ToolUseContent(
                id=generate_tool_use_id(),
                name="Agent",
                input=agent_input,
            ),
            start,
            end,
        ))
    return found
