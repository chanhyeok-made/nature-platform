"""TodoItem — the dual-form unit a TodoWrite tool produces.

The dual-form schema (`content` + `activeForm`) is borrowed from
Claude Code's TodoWrite tool. It's a deceptively small detail with
outsized effect: forcing the LLM to author both the *imperative*
form ("Run tests") and the *continuous/active* form ("Running
tests") nudges it from "describing what should happen" into "doing
the thing right now". A model that fills in the activeForm has
already mentally committed to executing the step.

Status moves pending → in_progress → completed. Items in flight
should always have exactly one in_progress entry — the framework's
footer rules use that as a deterministic signal of "is this agent
actually working or just listing things".
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

TodoStatus = Literal["pending", "in_progress", "completed"]


class TodoItem(BaseModel):
    """One item in an agent's externalized todo list.

    Both forms are required so a buggy/lazy LLM can't sneak in a
    plan-only entry. `content` is what the LLM intends to do;
    `activeForm` is what the dashboard shows while it's running.
    """

    content: str = Field(
        ...,
        min_length=1,
        description="Imperative form of the task — what to do. e.g., 'Run tests'.",
    )
    activeForm: str = Field(
        ...,
        min_length=1,
        description=(
            "Continuous/active form shown while the task is running. "
            "e.g., 'Running tests'. Required even for pending items so "
            "the LLM is forced to commit to execution wording up front."
        ),
    )
    status: TodoStatus = Field(
        default="pending",
        description="One of pending, in_progress, completed.",
    )
