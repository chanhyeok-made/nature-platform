"""ContextComposer — Context → LLMRequest serialization boundary.

The only place that knows how to translate the domain model (Context
with from_/to Messages) into provider-ready API types (LLMRequest with
role-based LLMMessages). Providers (AnthropicProvider, OpenAICompat)
stay unchanged.

In addition to the header → system / body → messages translation,
the composer runs the *footer* pipeline: a set of pure-function
rules that look at the current body + header state and may emit
ephemeral hints to append at the tail of the LLM message list. The
footer is NOT stored in Context — it's computed on every compose().
See `nature/context/footer.py` for the rule definitions.
"""

from __future__ import annotations

from nature.context.conversation import Conversation
from nature.context.footer import Hint, compute_footer_hints
from nature.context.types import Context, ContextHeader
from nature.protocols.llm import LLMRequest
from nature.protocols.message import Message as LLMMessage
from nature.protocols.message import Role, TextContent
from nature.protocols.tool import Tool


class ContextComposer:
    """Build an LLMRequest from a Context for a specific actor."""

    def compose(
        self,
        context: Context,
        *,
        self_actor: str,
        tool_registry: list[Tool],
        model: str,
        cache_control: dict[str, str] | None = None,
        request_id: str | None = None,
        max_output_tokens: int | None = None,
        source: str = "context_composer",
    ) -> "ComposedRequest":
        """Flatten context → LLMRequest for the given actor's perspective.

        `self_actor` determines how domain messages map to API roles:
        a message FROM self becomes `role=assistant`, any other message
        becomes `role=user`.

        Returns a `ComposedRequest` wrapping the LLMRequest plus the
        list of footer hints that fired (so the caller can emit a
        HINT_INJECTED trace event with the same request_id).
        """
        selected_tools = self._filter_tools(
            tool_registry, context.header.role.allowed_tools
        )

        messages = self._convert_conversation(
            context.body.conversation, self_actor
        )

        # Footer pipeline: ask each rule, collect hints, append them as
        # one synthetic user-role tail message tagged [FRAMEWORK NOTE].
        # Body is NOT mutated — the hints exist only in the rendered
        # LLM request and the trace event the caller will emit.
        hints = compute_footer_hints(
            context.body,
            context.header,
            self_actor,
        )
        if hints:
            messages.append(_render_hints_as_message(hints))

        request = LLMRequest(
            messages=messages,
            system=self._build_system(context.header),
            tools=[t.to_definition() for t in selected_tools],
            model=model,
            max_output_tokens=max_output_tokens,
            cache_control=cache_control,
            request_id=request_id,
            source=source,
        )
        return ComposedRequest(request=request, hints=hints)

    @staticmethod
    def _build_system(header: ContextHeader) -> list[str]:
        """Order: role instructions first, then principles by priority (desc)."""
        sections: list[str] = []
        if header.role.instructions:
            sections.append(header.role.instructions)
        for bp in sorted(header.principles, key=lambda p: -p.priority):
            if bp.text:
                sections.append(bp.text)
        return sections

    @staticmethod
    def _filter_tools(
        registry: list[Tool], allowed: list[str] | None
    ) -> list[Tool]:
        """Role policy filter — `allowed=None` means all registered tools."""
        if allowed is None:
            return list(registry)
        allowed_set = set(allowed)
        return [t for t in registry if t.name in allowed_set]

    @staticmethod
    def _convert_conversation(
        conv: Conversation, self_actor: str
    ) -> list[LLMMessage]:
        """Map domain Messages to LLMMessages via actor → role rule.

        FROM self  → role=assistant
        TO self    → role=user
        others     → role=user (counterparty perspective collapses to user)
        """
        result: list[LLMMessage] = []
        for msg in conv.messages:
            role = Role.ASSISTANT if msg.from_ == self_actor else Role.USER
            result.append(LLMMessage(role=role, content=list(msg.content)))
        return result


# ---------------------------------------------------------------------------
# Composed request bundle
# ---------------------------------------------------------------------------


class ComposedRequest:
    """A composed LLM request together with any footer hints that fired.

    Lightweight wrapper so the caller (LLMAgent) can both send the
    request and emit a HINT_INJECTED trace event with the same
    request_id. Not a Pydantic model on purpose — it never crosses the
    serialization boundary, just travels between Composer and caller.

    Transparently proxies attribute access to the underlying
    LLMRequest so legacy callers and tests that wrote `req.system` /
    `req.messages` keep working without changes; new callers that
    care about hints reach for `composed.hints` and `composed.request`
    explicitly.
    """

    __slots__ = ("request", "hints")

    def __init__(self, request: LLMRequest, hints: list[Hint]) -> None:
        self.request = request
        self.hints = hints

    def __getattr__(self, name: str):  # noqa: D401
        # __getattr__ only fires when normal lookup misses, so the
        # explicit __slots__ above (`request`, `hints`) take precedence.
        # Anything else falls through to the wrapped LLMRequest.
        return getattr(self.request, name)


def _render_hints_as_message(hints: list[Hint]) -> LLMMessage:
    """Render footer hints as a single user-role tail message.

    All providers handle additional user-role messages at the tail of
    the conversation, which makes this the most portable injection
    point. The text is prefixed with `[FRAMEWORK NOTE]` so a future
    reader (and the model itself) can tell it isn't a real user turn.
    """
    parts: list[str] = []
    for hint in hints:
        # Each hint already starts with its own [FRAMEWORK NOTE …] tag
        # in its text; we just join with blank lines between rules.
        parts.append(hint.text.rstrip())
    body = "\n\n".join(parts)
    return LLMMessage(role=Role.USER, content=[TextContent(text=body)])
