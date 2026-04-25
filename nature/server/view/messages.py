from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AnnotationDto(BaseModel):
    """Per-message metadata — thinking, tokens, stop reason, duration."""

    stop_reason: str | None = None
    usage: dict[str, Any] | None = None
    duration_ms: int | None = None
    thinking: list[str] | None = None
    llm_request_id: str | None = None


class HintDto(BaseModel):
    """One footer-rule nudge that the composer injected into the LLM
    request this message was produced from. The `source` is the rule
    name (e.g. `synthesis_nudge`, `todo_continues_after_tool_result`);
    `text` is the full `[FRAMEWORK NOTE]` body the LLM actually saw.
    Surfaced on the message so the dashboard can mark "the framework
    whispered something before the model spoke this reply" — useful
    for debugging why a turn ended / continued when it did.
    """

    source: str
    text: str


class MessageDto(BaseModel):
    """One message in a conversation, with its raw content blocks kept
    alongside the extracted text so the drawer can render tool_use /
    tool_result / thinking blocks without the client needing to re-parse.

    `regenerated_from` lists the tool_use_ids whose tool_results were
    ingested by the LLM call that produced this message — i.e., this
    message's text was *synthesized* from those tool outputs, not just
    a passthrough. Empty/None means the LLM call had no tool_results
    to ingest (a regular reply).

    `injected_hints` lists the footer-rule nudges that the composer
    appended to the LLM request that produced this message (empty on
    every regular reply). See `nature/context/footer.py` for the rule
    catalog.
    """

    message_id: str
    from_: str = Field(alias="from_")
    to: str
    text: str
    content: list[dict[str, Any]] = Field(default_factory=list)
    timestamp: float
    annotation: AnnotationDto | None = None
    regenerated_from: list[str] = Field(default_factory=list)
    injected_hints: list[HintDto] = Field(default_factory=list)

    model_config = {"populate_by_name": True}
