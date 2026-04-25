"""Body compaction — domain-oriented compression for `ContextBody`.

The Frame architecture enforces a sharp header/body split:

- Header (role + principles) is static, cacheable identity. Compaction
  NEVER touches it.
- Body (conversation) is the growing dialogue. Compaction operates here.

Strategies receive the body plus a read-only header view and return a
new, possibly shorter body. The pipeline runs strategies in priority
order until the token estimate drops below the autocompact threshold.

Replay fidelity is preserved by emitting a `BODY_COMPACTED` state-
transition event from the surrounding execution loop whenever a
strategy mutates the body. reconstruct() applies the event by replacing
the frame's body with the payload's post-compaction snapshot, so resume
after compaction lands on the exact same trimmed body the live run saw.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from nature.context.composer import ContextComposer
from nature.context.conversation import Conversation, Message
from nature.context.estimator import estimate_messages_tokens
from nature.context.types import Context, ContextBody, ContextHeader
from nature.protocols.context import TokenBudget
from nature.protocols.message import (
    Message as APIMessage,
    StreamEventType,
    TextContent,
    ToolResultContent,
    ToolUseContent,
)
from nature.protocols.provider import LLMProvider
from nature.protocols.tool import Tool
from nature.utils.tokens import estimate_tokens_for_text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy protocol
# ---------------------------------------------------------------------------


@dataclass
class BodyCompactionResult:
    """Outcome of running a single body strategy."""

    body: ContextBody
    tokens_before: int
    tokens_after: int
    strategy_name: str
    summary: str | None = None

    @property
    def changed(self) -> bool:
        return self.tokens_after < self.tokens_before


class BodyCompactionStrategy(ABC):
    """Domain-oriented body compressor.

    Strategies are pure functions of (body, header, budget). They must
    not touch the header. Returning the same body object unchanged is
    valid — the pipeline detects that as a no-op.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    async def compact(
        self,
        body: ContextBody,
        *,
        header: ContextHeader,
        budget: TokenBudget,
        current_tokens: int,
        provider: LLMProvider | None = None,
    ) -> BodyCompactionResult:
        ...


# ---------------------------------------------------------------------------
# Microcompact — clear old tool_result blocks (zero-cost)
# ---------------------------------------------------------------------------


MICROCOMPACT_CLEARED = "[Old tool result cleared]"
_DEFAULT_PRESERVE_TURNS = 4


class MicrocompactBodyStrategy(BodyCompactionStrategy):
    """Replace tool_result content older than N turns with a placeholder.

    "Turn" is defined as a message produced by the frame's self actor
    (the assistant side of the conversation). We keep the last N turns
    intact and compact tool_result blocks in every earlier message.
    """

    def __init__(self, preserve_turns: int = _DEFAULT_PRESERVE_TURNS) -> None:
        self._preserve_turns = preserve_turns

    @property
    def name(self) -> str:
        return "microcompact"

    async def compact(
        self,
        body: ContextBody,
        *,
        header: ContextHeader,
        budget: TokenBudget,
        current_tokens: int,
        provider: LLMProvider | None = None,
    ) -> BodyCompactionResult:
        messages = body.conversation.messages
        if not messages:
            return BodyCompactionResult(
                body=body,
                tokens_before=current_tokens,
                tokens_after=current_tokens,
                strategy_name=self.name,
            )

        self_actor = header.role.name
        turn_indices = [
            i for i, m in enumerate(messages) if m.from_ == self_actor
        ]
        if len(turn_indices) <= self._preserve_turns:
            return BodyCompactionResult(
                body=body,
                tokens_before=current_tokens,
                tokens_after=current_tokens,
                strategy_name=self.name,
            )

        cutoff = turn_indices[-self._preserve_turns]
        new_messages: list[Message] = []
        cleared = 0
        for i, msg in enumerate(messages):
            if i >= cutoff:
                new_messages.append(msg)
                continue
            new_blocks = []
            touched = False
            for block in msg.content:
                if isinstance(block, ToolResultContent):
                    new_blocks.append(
                        ToolResultContent(
                            tool_use_id=block.tool_use_id,
                            content=MICROCOMPACT_CLEARED,
                            is_error=block.is_error,
                        )
                    )
                    touched = True
                    cleared += 1
                else:
                    new_blocks.append(block)
            if touched:
                new_messages.append(msg.model_copy(update={"content": new_blocks}))
            else:
                new_messages.append(msg)

        if cleared == 0:
            return BodyCompactionResult(
                body=body,
                tokens_before=current_tokens,
                tokens_after=current_tokens,
                strategy_name=self.name,
            )

        new_body = ContextBody(conversation=Conversation(messages=new_messages))
        tokens_after = _estimate_body_bytes(new_body)
        return BodyCompactionResult(
            body=new_body,
            tokens_before=current_tokens,
            tokens_after=min(tokens_after, current_tokens),
            strategy_name=self.name,
            summary=f"cleared {cleared} tool_result blocks",
        )


# ---------------------------------------------------------------------------
# Dreamer — LLM-based semantic summarization of older turns
# ---------------------------------------------------------------------------


DREAMER_SUMMARY_PROMPT = (
    "You are a conversation summarizer for an AI agent system. "
    "You will receive a chronological transcript of a working session "
    "between agents and their tools. Produce a tight summary covering: "
    "(1) what was asked / the current goal, "
    "(2) concrete facts discovered — file paths, function names, "
    "decisions, verified observations, "
    "(3) what was tried and what happened, "
    "(4) what remains open and the natural next step. "
    "No preamble, no apologies. Markdown allowed. "
    "Target under 400 words. The agent reading your summary will use "
    "it in place of the transcript to continue the task, so keep "
    "every load-bearing detail and drop every pleasantry."
)


def _render_message_for_summary(msg: Message) -> str:
    """Flatten one domain Message to a text block the summarizer can read.

    Tool inputs and tool results are shown as labelled blocks so the
    summary can reason about them; text is passed through verbatim.
    """
    parts: list[str] = [f"[{msg.from_} → {msg.to}]"]
    for block in msg.content:
        if isinstance(block, TextContent):
            parts.append(block.text)
        elif isinstance(block, ToolUseContent):
            parts.append(
                f"  (tool_use name={block.name!r} input={block.input!r})"
            )
        elif isinstance(block, ToolResultContent):
            content = block.content
            if isinstance(content, list):
                rendered = "".join(
                    getattr(b, "text", "") for b in content
                    if isinstance(b, TextContent)
                )
            else:
                rendered = str(content)
            prefix = "tool_error" if block.is_error else "tool_result"
            parts.append(f"  ({prefix} {block.tool_use_id}): {rendered}")
        else:
            parts.append(f"  <{type(block).__name__}>")
    return "\n".join(parts)


async def _call_summarizer(
    provider: LLMProvider,
    rendered_prefix: str,
    *,
    model: str | None,
    max_output_tokens: int = 1024,
) -> str:
    """Drive one non-streaming summary completion and return the text.

    The caller passes the prefix rendered via `_render_message_for_summary`
    joined by blank lines. Errors bubble up so the strategy treats a
    summary failure the same way as any other failed compaction step.
    """
    api_request = [APIMessage.user(rendered_prefix)]
    out: list[str] = []
    async for event in provider.stream(
        messages=api_request,
        system=[DREAMER_SUMMARY_PROMPT],
        tools=None,
        model=model,
        max_output_tokens=max_output_tokens,
    ):
        if event.type == StreamEventType.CONTENT_BLOCK_DELTA and event.delta_text:
            out.append(event.delta_text)
    return "".join(out).strip()


class DreamerBodyStrategy(BodyCompactionStrategy):
    """LLM-based semantic compactor.

    Replaces the body's pre-window prefix with a single system summary
    message plus an optional pointer to a long-term memory file that
    preserves the raw transcript for later recall.

    Fires only when the body exceeds the budget's autocompact threshold
    (the pipeline enforces that gate upstream). The strategy itself
    will return a no-op when there aren't enough self-actor turns to
    leave the preserve window intact, or when the provider is absent.

    The compaction point is chosen to keep the `preserve_recent_turns`
    latest self-actor turns (plus everything after them) untouched.
    Messages before that point are rendered to text, fed to the
    provider with a summarization system prompt, and the result is
    stamped into a `system → <self_actor>` message at the head of the
    new body.

    LTM persistence is best-effort: if `ltm_dir` and `session_id` are
    configured the raw slice is also dumped to
    `<ltm_dir>/<session_id>/<frame_id>-<timestamp>.md` and the
    summary message carries a reference line so the agent can Read it
    back if it needs detail. Failures to write LTM files do not fail
    the compaction — they just drop the reference.
    """

    def __init__(
        self,
        *,
        preserve_recent_turns: int = 6,
        session_id: str = "",
        ltm_dir: Path | str | None = None,
        summarizer_model: str | None = None,
        summarizer_max_tokens: int = 1024,
    ) -> None:
        self._preserve_recent_turns = preserve_recent_turns
        self._session_id = session_id
        self._ltm_dir = Path(ltm_dir) if ltm_dir is not None else None
        self._summarizer_model = summarizer_model
        self._summarizer_max_tokens = summarizer_max_tokens

    @property
    def name(self) -> str:
        return "dreamer"

    async def compact(
        self,
        body: ContextBody,
        *,
        header: ContextHeader,
        budget: TokenBudget,
        current_tokens: int,
        provider: LLMProvider | None = None,
    ) -> BodyCompactionResult:
        messages = body.conversation.messages
        self_actor = header.role.name

        self_indices = [
            i for i, m in enumerate(messages) if m.from_ == self_actor
        ]
        if len(self_indices) <= self._preserve_recent_turns:
            return BodyCompactionResult(
                body=body,
                tokens_before=current_tokens,
                tokens_after=current_tokens,
                strategy_name=self.name,
            )

        cutoff = self_indices[-self._preserve_recent_turns]
        if cutoff <= 0:
            return BodyCompactionResult(
                body=body,
                tokens_before=current_tokens,
                tokens_after=current_tokens,
                strategy_name=self.name,
            )

        prefix = messages[:cutoff]
        tail = messages[cutoff:]

        if provider is None:
            logger.warning(
                "dreamer strategy invoked without a provider; skipping"
            )
            return BodyCompactionResult(
                body=body,
                tokens_before=current_tokens,
                tokens_after=current_tokens,
                strategy_name=self.name,
            )

        rendered = "\n\n".join(
            _render_message_for_summary(m) for m in prefix
        )
        try:
            summary_text = await _call_summarizer(
                provider, rendered,
                model=self._summarizer_model,
                max_output_tokens=self._summarizer_max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("dreamer summarization failed: %s", exc)
            return BodyCompactionResult(
                body=body,
                tokens_before=current_tokens,
                tokens_after=current_tokens,
                strategy_name=self.name,
            )

        if not summary_text:
            return BodyCompactionResult(
                body=body,
                tokens_before=current_tokens,
                tokens_after=current_tokens,
                strategy_name=self.name,
            )

        ltm_ref = self._persist_ltm(
            self_actor, prefix, rendered, summary_text,
        )

        summary_body = summary_text
        if ltm_ref is not None:
            summary_body = (
                f"[Earlier turns compacted — full transcript at `{ltm_ref}`]\n\n"
                f"{summary_text}"
            )
        else:
            summary_body = f"[Earlier turns compacted]\n\n{summary_text}"

        summary_msg = Message(
            from_="system",
            to=self_actor,
            content=[TextContent(text=summary_body)],
            timestamp=time.time(),
        )
        new_body = ContextBody(
            conversation=Conversation(messages=[summary_msg] + tail),
        )
        tokens_after = _estimate_body_bytes(new_body)

        return BodyCompactionResult(
            body=new_body,
            tokens_before=current_tokens,
            tokens_after=min(tokens_after, current_tokens),
            strategy_name=self.name,
            summary=(
                f"compacted {len(prefix)} messages → 1 summary"
                + (f" (ltm: {ltm_ref})" if ltm_ref else "")
            ),
        )

    # ── LTM persistence ──────────────────────────────────────────

    def _persist_ltm(
        self,
        frame_tag: str,
        prefix: list[Message],
        rendered_prefix: str,
        summary_text: str,
    ) -> str | None:
        """Write the raw prefix + summary to a dated LTM file.

        `frame_tag` is derived from the header's role name at compact
        time, so a root agent (`receptionist`) writes
        `receptionist-<ts>.md` while a researcher sub-frame under the
        same session writes `researcher-<ts>.md` — the session dir
        scopes both.

        Returns the path (absolute-string) of the created file, or None
        when persistence is disabled or the write fails. The strategy
        never raises on persistence failure — compaction must proceed
        even when the LTM directory is read-only or missing.
        """
        if self._ltm_dir is None or not self._session_id:
            return None
        try:
            session_dir = self._ltm_dir / self._session_id
            session_dir.mkdir(parents=True, exist_ok=True)
            ts = int(time.time())
            tag = frame_tag or "frame"
            path = session_dir / f"{tag}-{ts}.md"
            body_lines = [
                f"# LTM — {self._session_id} / {tag}",
                f"- generated_at: {ts}",
                f"- messages_covered: {len(prefix)}",
                "",
                "## Summary",
                summary_text,
                "",
                "## Raw transcript",
                rendered_prefix,
                "",
            ]
            path.write_text("\n".join(body_lines), encoding="utf-8")
            return str(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("dreamer LTM persistence failed: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Pipeline — runs strategies in order until budget is satisfied
# ---------------------------------------------------------------------------


@dataclass
class PipelineRunResult:
    """Outcome of a single pipeline pass."""

    steps: list[BodyCompactionResult]
    final_body: ContextBody
    tokens_before: int
    tokens_after: int

    @property
    def changed(self) -> bool:
        return any(s.changed for s in self.steps)


class BodyCompactionPipeline:
    """Runs a list of strategies in order until the budget is met.

    The pipeline is stateless and reusable — it holds the strategy
    list, the budget, and a composer for token estimation. Callers
    invoke `run(frame_context, ...)` before each LLM call; the pipeline
    decides whether any strategy needs to fire and produces a list of
    step results so the surrounding loop can emit one event per step.
    """

    def __init__(
        self,
        *,
        strategies: Sequence[BodyCompactionStrategy],
        budget: TokenBudget | None = None,
        composer: ContextComposer | None = None,
    ) -> None:
        self._strategies = list(strategies)
        self._budget = budget or TokenBudget()
        self._composer = composer or ContextComposer()

    @property
    def strategies(self) -> list[BodyCompactionStrategy]:
        return list(self._strategies)

    @property
    def budget(self) -> TokenBudget:
        return self._budget

    async def estimate_tokens(
        self,
        context: Context,
        *,
        self_actor: str,
        tool_registry: list[Tool],
        model: str,
        provider: LLMProvider | None = None,
    ) -> int:
        """Rough token count for the composed request a role would see."""
        composed = self._composer.compose(
            context,
            self_actor=self_actor,
            tool_registry=tool_registry,
            model=model,
        )
        request = composed.request
        return await estimate_messages_tokens(
            request.messages, request.system, request.tools, provider
        )

    async def run(
        self,
        context: Context,
        *,
        self_actor: str,
        tool_registry: list[Tool],
        model: str,
        provider: LLMProvider | None = None,
    ) -> PipelineRunResult:
        """Apply strategies until the estimate is under the autocompact
        threshold or we run out of strategies."""
        current = context.body
        tokens = await self.estimate_tokens(
            context,
            self_actor=self_actor,
            tool_registry=tool_registry,
            model=model,
            provider=provider,
        )
        initial = tokens

        steps: list[BodyCompactionResult] = []
        if tokens < self._budget.autocompact_threshold or not self._strategies:
            return PipelineRunResult(
                steps=steps,
                final_body=current,
                tokens_before=initial,
                tokens_after=tokens,
            )

        for strategy in self._strategies:
            if tokens < self._budget.autocompact_threshold:
                break
            try:
                result = await strategy.compact(
                    current,
                    header=context.header,
                    budget=self._budget,
                    current_tokens=tokens,
                    provider=provider,
                )
            except Exception as exc:
                logger.error(
                    "body compaction strategy %s failed: %s",
                    strategy.name, exc,
                )
                continue
            if result.body is current:
                continue
            current = result.body
            tokens = result.tokens_after
            steps.append(result)

        return PipelineRunResult(
            steps=steps,
            final_body=current,
            tokens_before=initial,
            tokens_after=tokens,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _estimate_body_bytes(body: ContextBody) -> int:
    """Byte-based fallback estimator — used when a strategy wants an
    immediate number without a full composer round-trip."""
    total = 0
    for m in body.conversation.messages:
        total += estimate_tokens_for_text(m.model_dump_json(), is_json=True)
    return total


__all__ = [
    "BodyCompactionStrategy",
    "BodyCompactionResult",
    "PipelineRunResult",
    "BodyCompactionPipeline",
    "MicrocompactBodyStrategy",
    "DreamerBodyStrategy",
    "DREAMER_SUMMARY_PROMPT",
    "MICROCOMPACT_CLEARED",
]
