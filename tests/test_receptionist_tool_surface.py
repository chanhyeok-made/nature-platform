"""The builtin receptionist agent must delegate everything — not
touch files, run shells, or call sub-agents other than via the Agent
tool. Stage 1 eval runs showed that an `allowed_tools=null` (full
surface) receptionist happily did direct work on simple tasks and
skipped delegation entirely, which collapsed preset-level comparisons
for presets that only differ in sub-agent composition (e.g.
haiku-qwen-reader vs all-haiku).
"""

from __future__ import annotations

from pathlib import Path

from nature.agents.config import load_agents_registry


def test_receptionist_allowed_tools_is_agent_only():
    agents = load_agents_registry()
    receptionist = agents.get("receptionist")
    assert receptionist is not None, "receptionist agent must ship as a builtin"
    assert receptionist.allowed_tools == ["Agent"], (
        "receptionist must be restricted to the Agent tool so it "
        "cannot short-circuit delegation by doing direct file work"
    )


def test_receptionist_instructions_are_always_delegate():
    """The prose instruction must match the tool surface — telling
    the agent it has Read/Bash when it doesn't would waste turns on
    tool calls that the permission layer would reject."""
    agents = load_agents_registry()
    receptionist = agents.get("receptionist")
    assert receptionist is not None
    body = receptionist.instructions_text.lower()
    # Any mention of "handle directly" or "trivial ... directly" would
    # conflict with the Agent-only tool surface.
    for forbidden in ("handle directly", "handle a trivial", "use your own tools"):
        assert forbidden not in body, (
            f"receptionist instructions still say {forbidden!r}; that "
            f"contradicts allowed_tools=['Agent']"
        )
