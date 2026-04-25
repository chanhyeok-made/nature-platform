"""Tests for the Step 2 context refactor: new domain types + composer.

Covers:
- AgentRole / BasePrinciple / ContextHeader / ContextBody / Context
- Conversation / Message (domain, from_/to) / MessageAnnotation
- ContextComposer: system assembly, tool filtering, role mapping
- Immutability and copy semantics (with_role / with_principle)
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from nature.context import (
    AgentRole,
    BasePrinciple,
    BasePrincipleSource,
    Context,
    ContextBody,
    ContextComposer,
    ContextHeader,
    Conversation,
    Message,
    MessageAnnotation,
)
from nature.protocols.message import Role, TextContent, ToolResultContent, ToolUseContent
from nature.protocols.tool import PermissionResult, Tool, ToolContext, ToolDefinition, ToolResult


# ---------------------------------------------------------------------------
# Test fixtures / helpers
# ---------------------------------------------------------------------------


class _FakeTool(Tool):
    """Minimal Tool implementation for composer filter tests."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"fake tool {self._name}"

    @property
    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        return ToolResult(output="")

    def to_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._name,
            description=self.description,
            input_schema=self.input_schema,
        )


def _make_message(
    from_: str = "user",
    to: str = "receptionist",
    text: str = "hello",
) -> Message:
    return Message(
        from_=from_,
        to=to,
        content=[TextContent(text=text)],
        timestamp=time.time(),
    )


# ---------------------------------------------------------------------------
# AgentRole
# ---------------------------------------------------------------------------


def test_agent_role_minimal():
    role = AgentRole(name="receptionist", instructions="be helpful")
    assert role.name == "receptionist"
    assert role.instructions == "be helpful"
    assert role.allowed_tools is None
    assert role.description == ""


def test_agent_role_with_allowed_tools():
    role = AgentRole(
        name="researcher",
        instructions="explore",
        allowed_tools=["Read", "Glob", "Grep"],
    )
    assert role.allowed_tools == ["Read", "Glob", "Grep"]


# ---------------------------------------------------------------------------
# BasePrinciple
# ---------------------------------------------------------------------------


def test_base_principle_defaults():
    bp = BasePrinciple(text="do not narrate")
    assert bp.source == BasePrincipleSource.RUNTIME
    assert bp.priority == 0


def test_base_principle_all_sources():
    for source in BasePrincipleSource:
        bp = BasePrinciple(text="rule", source=source)
        assert bp.source == source


def test_base_principle_priority():
    bp = BasePrinciple(text="critical", priority=100)
    assert bp.priority == 100


# ---------------------------------------------------------------------------
# Context structure
# ---------------------------------------------------------------------------


def test_context_header_body_split():
    role = AgentRole(name="r", instructions="x")
    ctx = Context(header=ContextHeader(role=role))
    assert ctx.header.role.name == "r"
    assert len(ctx.body.conversation) == 0


def test_context_body_default():
    role = AgentRole(name="r", instructions="x")
    ctx = Context(header=ContextHeader(role=role))
    assert isinstance(ctx.body, ContextBody)
    assert isinstance(ctx.body.conversation, Conversation)


def test_context_with_role_swap_preserves_body():
    role1 = AgentRole(name="receptionist", instructions="friendly")
    role2 = AgentRole(name="core", instructions="analytical")
    ctx = Context(header=ContextHeader(role=role1))
    ctx.body.conversation.append(_make_message(text="first"))

    swapped = ctx.with_role(role2)

    assert swapped.header.role.name == "core"
    # Body is copied so original stays intact
    assert len(swapped.body.conversation) == 1
    assert ctx.header.role.name == "receptionist"  # original unchanged


def test_context_with_principle_adds_without_mutating():
    role = AgentRole(name="r", instructions="x")
    ctx = Context(header=ContextHeader(role=role))
    bp = BasePrinciple(text="rule 1", source=BasePrincipleSource.USER)

    updated = ctx.with_principle(bp)

    assert len(updated.header.principles) == 1
    assert updated.header.principles[0].text == "rule 1"
    assert len(ctx.header.principles) == 0  # original unchanged


# ---------------------------------------------------------------------------
# Conversation / Message (domain)
# ---------------------------------------------------------------------------


def test_message_auto_id():
    msg = _make_message()
    assert msg.id.startswith("msg_")
    assert len(msg.id) > 4


def test_message_from_to_content():
    msg = Message(
        from_="user",
        to="receptionist",
        content=[TextContent(text="hi")],
        timestamp=1.0,
    )
    assert msg.from_ == "user"
    assert msg.to == "receptionist"
    assert len(msg.content) == 1


def test_conversation_append_and_extend():
    conv = Conversation()
    assert len(conv) == 0

    conv.append(_make_message(text="one"))
    assert len(conv) == 1

    conv.extend([_make_message(text="two"), _make_message(text="three")])
    assert len(conv) == 3


def test_message_annotation_fields():
    ann = MessageAnnotation(
        message_id="msg_abc",
        thinking=["step 1", "step 2"],
        stop_reason="end_turn",
        duration_ms=500,
    )
    assert ann.message_id == "msg_abc"
    assert ann.thinking == ["step 1", "step 2"]
    assert ann.duration_ms == 500
    assert ann.usage is None  # optional


# ---------------------------------------------------------------------------
# ContextComposer — system assembly
# ---------------------------------------------------------------------------


def test_composer_system_starts_with_role_instructions():
    role = AgentRole(name="r", instructions="you are a researcher")
    ctx = Context(header=ContextHeader(role=role))
    composer = ContextComposer()

    req = composer.compose(
        ctx, self_actor="r", tool_registry=[], model="claude-sonnet-4"
    )

    assert req.system[0] == "you are a researcher"


def test_composer_principles_ordered_by_priority_desc():
    role = AgentRole(name="r", instructions="role")
    ctx = Context(
        header=ContextHeader(
            role=role,
            principles=[
                BasePrinciple(text="low", priority=1),
                BasePrinciple(text="high", priority=10),
                BasePrinciple(text="mid", priority=5),
            ],
        )
    )
    composer = ContextComposer()
    req = composer.compose(
        ctx, self_actor="r", tool_registry=[], model="m"
    )

    # First system block is role, then principles in priority-desc order
    assert req.system == ["role", "high", "mid", "low"]


def test_composer_empty_role_instructions_skipped():
    role = AgentRole(name="r", instructions="")
    ctx = Context(
        header=ContextHeader(
            role=role, principles=[BasePrinciple(text="just this")]
        )
    )
    composer = ContextComposer()
    req = composer.compose(ctx, self_actor="r", tool_registry=[], model="m")

    assert req.system == ["just this"]


# ---------------------------------------------------------------------------
# ContextComposer — tool filtering by role policy
# ---------------------------------------------------------------------------


def test_composer_allowed_tools_none_includes_all():
    role = AgentRole(name="r", instructions="x", allowed_tools=None)
    ctx = Context(header=ContextHeader(role=role))
    registry = [_FakeTool("Read"), _FakeTool("Bash"), _FakeTool("Agent")]
    composer = ContextComposer()

    req = composer.compose(
        ctx, self_actor="r", tool_registry=registry, model="m"
    )

    assert req.tools is not None
    assert {t.name for t in req.tools} == {"Read", "Bash", "Agent"}


def test_composer_allowed_tools_list_filters_registry():
    role = AgentRole(
        name="researcher",
        instructions="x",
        allowed_tools=["Read", "Glob"],
    )
    ctx = Context(header=ContextHeader(role=role))
    registry = [
        _FakeTool("Read"),
        _FakeTool("Glob"),
        _FakeTool("Bash"),
        _FakeTool("Agent"),
    ]
    composer = ContextComposer()

    req = composer.compose(
        ctx, self_actor="researcher", tool_registry=registry, model="m"
    )

    assert req.tools is not None
    assert {t.name for t in req.tools} == {"Read", "Glob"}


def test_composer_allowed_tools_empty_list_yields_no_tools():
    role = AgentRole(name="r", instructions="x", allowed_tools=[])
    ctx = Context(header=ContextHeader(role=role))
    registry = [_FakeTool("Read"), _FakeTool("Bash")]
    composer = ContextComposer()

    req = composer.compose(
        ctx, self_actor="r", tool_registry=registry, model="m"
    )

    assert req.tools == []


# ---------------------------------------------------------------------------
# ContextComposer — message role mapping
# ---------------------------------------------------------------------------


def test_composer_maps_from_self_to_assistant():
    role = AgentRole(name="receptionist", instructions="x")
    ctx = Context(header=ContextHeader(role=role))
    ctx.body.conversation.extend([
        Message(
            from_="user", to="receptionist",
            content=[TextContent(text="hi")], timestamp=1.0,
        ),
        Message(
            from_="receptionist", to="user",
            content=[TextContent(text="hello")], timestamp=2.0,
        ),
    ])

    composer = ContextComposer()
    req = composer.compose(
        ctx, self_actor="receptionist", tool_registry=[], model="m"
    )

    assert len(req.messages) == 2
    assert req.messages[0].role == Role.USER
    assert req.messages[1].role == Role.ASSISTANT


def test_composer_preserves_content_blocks():
    role = AgentRole(name="r", instructions="x")
    ctx = Context(header=ContextHeader(role=role))
    tool_use = ToolUseContent(id="tu_1", name="Read", input={"path": "x"})
    ctx.body.conversation.append(
        Message(
            from_="r",
            to="user",
            content=[TextContent(text="let me read"), tool_use],
            timestamp=1.0,
        )
    )

    composer = ContextComposer()
    req = composer.compose(
        ctx, self_actor="r", tool_registry=[], model="m"
    )

    assert len(req.messages) == 1
    msg = req.messages[0]
    assert msg.role == Role.ASSISTANT
    assert len(msg.content) == 2
    assert isinstance(msg.content[0], TextContent)
    assert isinstance(msg.content[1], ToolUseContent)


def test_composer_tool_result_message_mapped_as_user():
    role = AgentRole(name="r", instructions="x")
    ctx = Context(header=ContextHeader(role=role))
    ctx.body.conversation.append(
        Message(
            from_="tool:Read",
            to="r",
            content=[ToolResultContent(tool_use_id="tu_1", content="file body")],
            timestamp=1.0,
        )
    )

    composer = ContextComposer()
    req = composer.compose(
        ctx, self_actor="r", tool_registry=[], model="m"
    )

    assert req.messages[0].role == Role.USER
    assert isinstance(req.messages[0].content[0], ToolResultContent)


# ---------------------------------------------------------------------------
# ContextComposer — passthrough metadata
# ---------------------------------------------------------------------------


def test_composer_passes_model_and_request_id():
    role = AgentRole(name="r", instructions="x")
    ctx = Context(header=ContextHeader(role=role))
    composer = ContextComposer()

    req = composer.compose(
        ctx,
        self_actor="r",
        tool_registry=[],
        model="claude-sonnet-4",
        request_id="req_abc",
        cache_control={"type": "ephemeral"},
        max_output_tokens=8192,
    )

    assert req.model == "claude-sonnet-4"
    assert req.request_id == "req_abc"
    assert req.cache_control == {"type": "ephemeral"}
    assert req.max_output_tokens == 8192
    assert req.source == "context_composer"


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


def test_context_json_round_trip():
    role = AgentRole(
        name="core",
        instructions="plan",
        allowed_tools=["Agent"],
    )
    ctx = Context(
        header=ContextHeader(
            role=role,
            principles=[BasePrinciple(text="rule", priority=5)],
        )
    )
    ctx.body.conversation.append(_make_message(from_="user", to="core"))

    json_str = ctx.model_dump_json()
    restored = Context.model_validate_json(json_str)

    assert restored.header.role.name == "core"
    assert restored.header.role.allowed_tools == ["Agent"]
    assert len(restored.header.principles) == 1
    assert restored.header.principles[0].priority == 5
    assert len(restored.body.conversation) == 1
