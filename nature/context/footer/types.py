"""Shared types for the footer package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from nature.context.types import ContextBody, ContextHeader


@dataclass(frozen=True)
class Hint:
    """One footer hint emitted by a rule.

    `source` is the rule's name (for trace events + dashboard display).
    `text` is the natural-language note that lands in the LLM prompt.
    """

    source: str
    text: str


# Type of a footer rule: pure function of (body, header, self_actor)
# returning either a Hint or None. Rules MUST NOT mutate inputs.
Rule = Callable[[ContextBody, ContextHeader, str], "Hint | None"]
