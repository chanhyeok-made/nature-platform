"""Tests for the Pack layer scaffolding (M1).

Covers:
- Type-level smoke tests (Intervention/Capability construction)
- Registry trigger indexing
- Registry dispatch (sync contributors, async listeners/gates)
- Legacy shim: FOOTER_RULES round-trip through Contributor Interventions
- compute_footer_hints output parity with the pre-unification behavior
"""

from __future__ import annotations

import asyncio
import time

import pytest

from nature.events.types import EventType
from nature.packs.legacy_shim import (
    _LEGACY_CAPABILITY_NAME,
    install_legacy_rules,
    legacy_rules_installed,
)
from nature.packs.registry import PackRegistry, default_registry
from nature.packs.types import (
    AppendFooter,
    Block,
    Capability,
    EmitEvent,
    Intervention,
    InterventionContext,
    InterventionPhase,
    LLMPhase,
    OnEvent,
    OnLLM,
    OnTool,
    OnTurn,
    ToolCallInfo,
    ToolPhase,
    TurnPhase,
)


# ──────────────────────────────────────────────────────────────────────
# Type / construction smoke tests
# ──────────────────────────────────────────────────────────────────────


def test_intervention_construction():
    def action(ctx):
        return []

    intv = Intervention(
        id="test.noop",
        kind="listener",
        trigger=OnEvent(event_type=EventType.EDIT_MISS),
        action=action,
    )
    assert intv.id == "test.noop"
    assert intv.default_enabled is True
    assert intv.description == ""


def test_capability_bundles_tools_and_interventions():
    cap = Capability(
        name="test_cap",
        description="bundle test",
        interventions=[
            Intervention(
                id="test_cap.one",
                kind="listener",
                trigger=OnEvent(event_type=EventType.EDIT_MISS),
                action=lambda ctx: [],
            ),
        ],
        event_types=[EventType.EDIT_MISS],
    )
    assert len(cap.interventions) == 1
    assert cap.event_types == [EventType.EDIT_MISS]
    assert cap.tools == []


# ──────────────────────────────────────────────────────────────────────
# Registry indexing
# ──────────────────────────────────────────────────────────────────────


def _make_listener(id: str, trigger):
    return Intervention(
        id=id,
        kind="listener",
        trigger=trigger,
        action=lambda ctx: [],
    )


def test_register_tool_call_indexes_correctly():
    reg = PackRegistry()
    pre = _make_listener(
        "x.pre", OnTool(tool_name="Edit", phase=ToolPhase.PRE)
    )
    post_all = _make_listener(
        "x.post_wild", OnTool(tool_name=None, phase=ToolPhase.POST)
    )
    post_edit = _make_listener(
        "x.post_edit", OnTool(tool_name="Edit", phase=ToolPhase.POST)
    )
    reg.register_intervention(pre)
    reg.register_intervention(post_all)
    reg.register_intervention(post_edit)

    assert reg._by_tool_pre["Edit"] == [pre]
    assert reg._by_tool_post[None] == [post_all]
    assert reg._by_tool_post["Edit"] == [post_edit]


def test_register_event_and_turn_and_llm():
    reg = PackRegistry()
    ev = _make_listener("x.ev", OnEvent(event_type=EventType.LOOP_DETECTED))
    turn = _make_listener("x.turn", OnTurn(phase=TurnPhase.BEFORE_LLM))
    llm_pre = _make_listener("x.llmpre", OnLLM(phase=LLMPhase.PRE))
    llm_post = _make_listener("x.llmpost", OnLLM(phase=LLMPhase.POST))

    reg.register_intervention(ev)
    reg.register_intervention(turn)
    reg.register_intervention(llm_pre)
    reg.register_intervention(llm_post)

    assert reg._by_event[EventType.LOOP_DETECTED] == [ev]
    assert reg._by_turn[TurnPhase.BEFORE_LLM] == [turn]
    assert reg._by_llm_pre == [llm_pre]
    assert reg._by_llm_post == [llm_post]


def test_register_capability_flattens_tools_and_interventions():
    reg = PackRegistry()
    intv1 = _make_listener("cap.one", OnEvent(event_type=EventType.EDIT_MISS))
    intv2 = _make_listener("cap.two", OnTurn(phase=TurnPhase.BEFORE_LLM))
    cap = Capability(name="demo", interventions=[intv1, intv2])
    reg.register_capability(cap)

    assert "demo" in reg.capabilities
    assert reg.interventions["cap.one"] is intv1
    assert reg.interventions["cap.two"] is intv2
    assert reg._by_event[EventType.EDIT_MISS] == [intv1]
    assert reg._by_turn[TurnPhase.BEFORE_LLM] == [intv2]


def test_clear_empties_all_indexes():
    reg = PackRegistry()
    reg.register_intervention(
        _make_listener("x.ev", OnEvent(event_type=EventType.EDIT_MISS))
    )
    reg.clear()
    assert reg.interventions == {}
    assert reg._by_event == {}


# ──────────────────────────────────────────────────────────────────────
# Dispatch — sync contributors
# ──────────────────────────────────────────────────────────────────────


def test_dispatch_turn_sync_calls_actions_in_order():
    reg = PackRegistry()
    trace: list[str] = []

    def action_a(ctx):
        trace.append("a")
        return [AppendFooter(text="A", source_id="a")]

    def action_b(ctx):
        trace.append("b")
        return [AppendFooter(text="B", source_id="b")]

    reg.register_intervention(Intervention(
        id="t.a",
        kind="contributor",
        trigger=OnTurn(phase=TurnPhase.BEFORE_LLM),
        action=action_a,
    ))
    reg.register_intervention(Intervention(
        id="t.b",
        kind="contributor",
        trigger=OnTurn(phase=TurnPhase.BEFORE_LLM),
        action=action_b,
    ))

    ctx = InterventionContext(session_id="s", now=time.time(), registry=reg)
    effects = reg.dispatch_turn_sync(TurnPhase.BEFORE_LLM, ctx)

    assert trace == ["a", "b"]   # declaration order
    assert len(effects) == 2
    assert all(isinstance(e, AppendFooter) for e in effects)
    assert [e.source_id for e in effects] == ["a", "b"]


def test_dispatch_turn_sync_skips_async_action():
    reg = PackRegistry()

    async def async_action(ctx):
        return [AppendFooter(text="oops", source_id="async")]

    def sync_action(ctx):
        return [AppendFooter(text="ok", source_id="sync")]

    reg.register_intervention(Intervention(
        id="t.async",
        kind="contributor",
        trigger=OnTurn(phase=TurnPhase.BEFORE_LLM),
        action=async_action,
    ))
    reg.register_intervention(Intervention(
        id="t.sync",
        kind="contributor",
        trigger=OnTurn(phase=TurnPhase.BEFORE_LLM),
        action=sync_action,
    ))

    ctx = InterventionContext(session_id="s", now=time.time(), registry=reg)
    effects = reg.dispatch_turn_sync(TurnPhase.BEFORE_LLM, ctx)

    # Only the sync action contributed; async was logged+skipped.
    assert len(effects) == 1
    assert effects[0].source_id == "sync"


def test_dispatch_turn_sync_swallows_exceptions():
    reg = PackRegistry()

    def boom(ctx):
        raise ValueError("kaboom")

    def ok(ctx):
        return [AppendFooter(text="still running", source_id="ok")]

    reg.register_intervention(Intervention(
        id="t.boom",
        kind="contributor",
        trigger=OnTurn(phase=TurnPhase.BEFORE_LLM),
        action=boom,
    ))
    reg.register_intervention(Intervention(
        id="t.ok",
        kind="contributor",
        trigger=OnTurn(phase=TurnPhase.BEFORE_LLM),
        action=ok,
    ))

    ctx = InterventionContext(session_id="s", now=time.time(), registry=reg)
    effects = reg.dispatch_turn_sync(TurnPhase.BEFORE_LLM, ctx)

    assert len(effects) == 1
    assert effects[0].source_id == "ok"


# ──────────────────────────────────────────────────────────────────────
# dispatch_event_sync — used by AreaManager._emit
# ──────────────────────────────────────────────────────────────────────


def test_dispatch_event_sync_fires_listeners_in_order():
    reg = PackRegistry()
    log: list[str] = []

    reg.register_intervention(Intervention(
        id="ev.a",
        kind="listener",
        trigger=OnEvent(event_type=EventType.EDIT_MISS),
        action=lambda ctx: log.append("a") or [],
    ))
    reg.register_intervention(Intervention(
        id="ev.b",
        kind="listener",
        trigger=OnEvent(event_type=EventType.EDIT_MISS),
        action=lambda ctx: log.append("b") or [],
    ))

    ctx = InterventionContext(session_id="s", now=time.time(), registry=reg)
    reg.dispatch_event_sync(EventType.EDIT_MISS, ctx)

    assert log == ["a", "b"]


def test_dispatch_event_sync_skips_async_action():
    reg = PackRegistry()

    async def async_action(ctx):
        return [AppendFooter(text="async", source_id="a")]

    def sync_action(ctx):
        return [AppendFooter(text="sync", source_id="s")]

    reg.register_intervention(Intervention(
        id="ev.async",
        kind="listener",
        trigger=OnEvent(event_type=EventType.EDIT_MISS),
        action=async_action,
    ))
    reg.register_intervention(Intervention(
        id="ev.sync",
        kind="listener",
        trigger=OnEvent(event_type=EventType.EDIT_MISS),
        action=sync_action,
    ))

    ctx = InterventionContext(session_id="s", now=time.time(), registry=reg)
    effects = reg.dispatch_event_sync(EventType.EDIT_MISS, ctx)

    assert len(effects) == 1
    assert effects[0].source_id == "s"


def test_dispatch_event_sync_swallows_exceptions():
    reg = PackRegistry()

    def boom(ctx):
        raise ValueError("kaboom")

    def ok(ctx):
        return [AppendFooter(text="still fires", source_id="ok")]

    reg.register_intervention(Intervention(
        id="ev.boom",
        kind="listener",
        trigger=OnEvent(event_type=EventType.EDIT_MISS),
        action=boom,
    ))
    reg.register_intervention(Intervention(
        id="ev.ok",
        kind="listener",
        trigger=OnEvent(event_type=EventType.EDIT_MISS),
        action=ok,
    ))

    ctx = InterventionContext(session_id="s", now=time.time(), registry=reg)
    effects = reg.dispatch_event_sync(EventType.EDIT_MISS, ctx)

    assert len(effects) == 1
    assert effects[0].source_id == "ok"


def test_dispatch_event_sync_honors_primary_post_effect_split():
    """POST_EFFECT listeners see PRIMARY's effects via ctx.primary_effects."""
    reg = PackRegistry()
    seen: list[list] = []

    def primary(ctx):
        return [EmitEvent(event_type=EventType.EDIT_MISS, payload={"tag": "p"})]

    def post(ctx):
        seen.append(list(ctx.primary_effects or []))
        return []

    reg.register_intervention(Intervention(
        id="ev.primary",
        kind="listener",
        trigger=OnEvent(event_type=EventType.LOOP_DETECTED),
        phase=InterventionPhase.PRIMARY,
        action=primary,
    ))
    reg.register_intervention(Intervention(
        id="ev.post",
        kind="listener",
        trigger=OnEvent(event_type=EventType.LOOP_DETECTED),
        phase=InterventionPhase.POST_EFFECT,
        action=post,
    ))

    ctx = InterventionContext(session_id="s", now=time.time(), registry=reg)
    effects = reg.dispatch_event_sync(EventType.LOOP_DETECTED, ctx)

    # PRIMARY's EmitEvent surfaces in the returned effects …
    assert any(
        isinstance(e, EmitEvent) and e.payload.get("tag") == "p"
        for e in effects
    )
    # … and POST_EFFECT saw it via primary_effects.
    assert len(seen) == 1
    assert len(seen[0]) == 1
    assert isinstance(seen[0][0], EmitEvent)


# ──────────────────────────────────────────────────────────────────────
# Dispatch — async tool call + where filter
# ──────────────────────────────────────────────────────────────────────


def test_dispatch_tool_walks_wildcard_and_specific():
    reg = PackRegistry()

    def specific(ctx):
        return [EmitEvent(event_type=EventType.EDIT_MISS, payload={"tag": "edit"})]

    def wild(ctx):
        return [EmitEvent(event_type=EventType.EDIT_MISS, payload={"tag": "wild"})]

    reg.register_intervention(Intervention(
        id="t.edit",
        kind="listener",
        trigger=OnTool(tool_name="Edit", phase=ToolPhase.POST),
        action=specific,
    ))
    reg.register_intervention(Intervention(
        id="t.wild",
        kind="listener",
        trigger=OnTool(tool_name=None, phase=ToolPhase.POST),
        action=wild,
    ))

    ctx = InterventionContext(
        session_id="s", now=time.time(), registry=reg,
        tool_call=ToolCallInfo(
            tool_name="Edit",
            tool_use_id="tu1",
            tool_input={"file_path": "foo.py"},
            phase=ToolPhase.POST,
            result_is_error=True,
        ),
    )
    effects = asyncio.run(reg.dispatch_tool(ToolPhase.POST, ctx))

    tags = [e.payload["tag"] for e in effects if isinstance(e, EmitEvent)]
    assert tags == ["edit", "wild"]


def test_where_filter_rejects_nonmatching_tool_call():
    reg = PackRegistry()

    def action(ctx):
        return [EmitEvent(event_type=EventType.EDIT_MISS, payload={})]

    reg.register_intervention(Intervention(
        id="t.guard",
        kind="listener",
        trigger=OnTool(
            tool_name="Edit",
            phase=ToolPhase.POST,
            where=lambda tc: tc.result_is_error is True,
        ),
        action=action,
    ))

    ctx_ok = InterventionContext(
        session_id="s", now=time.time(), registry=reg,
        tool_call=ToolCallInfo(
            tool_name="Edit",
            tool_use_id="tu1",
            tool_input={},
            phase=ToolPhase.POST,
            result_is_error=True,
        ),
    )
    ctx_fail = InterventionContext(
        session_id="s", now=time.time(), registry=reg,
        tool_call=ToolCallInfo(
            tool_name="Edit",
            tool_use_id="tu2",
            tool_input={},
            phase=ToolPhase.POST,
            result_is_error=False,
        ),
    )

    out_ok = asyncio.run(reg.dispatch_tool(ToolPhase.POST, ctx_ok))
    out_fail = asyncio.run(reg.dispatch_tool(ToolPhase.POST, ctx_fail))

    assert len(out_ok) == 1
    assert len(out_fail) == 0


def test_dispatch_tool_requires_tool_call_in_ctx():
    reg = PackRegistry()
    ctx = InterventionContext(session_id="s", now=time.time(), registry=reg)
    with pytest.raises(ValueError):
        asyncio.run(reg.dispatch_tool(ToolPhase.PRE, ctx))


# ──────────────────────────────────────────────────────────────────────
# Dispatch — async listener that returns Block
# ──────────────────────────────────────────────────────────────────────


def test_gate_returns_block_effect():
    reg = PackRegistry()

    async def gate_action(ctx):
        return [Block(reason="over budget", trace_event=EventType.BUDGET_BLOCKED)]

    reg.register_intervention(Intervention(
        id="t.gate",
        kind="gate",
        trigger=OnTool(tool_name="Read", phase=ToolPhase.PRE),
        action=gate_action,
    ))

    ctx = InterventionContext(
        session_id="s", now=time.time(), registry=reg,
        tool_call=ToolCallInfo(
            tool_name="Read",
            tool_use_id="tu1",
            tool_input={"file_path": "foo.py"},
            phase=ToolPhase.PRE,
        ),
    )
    effects = asyncio.run(reg.dispatch_tool(ToolPhase.PRE, ctx))
    assert len(effects) == 1
    assert isinstance(effects[0], Block)
    assert effects[0].trace_event == EventType.BUDGET_BLOCKED


# ──────────────────────────────────────────────────────────────────────
# Legacy shim — FOOTER_RULES port
# ──────────────────────────────────────────────────────────────────────


def test_install_legacy_rules_registers_contributor_per_rule():
    from nature.context.footer import FOOTER_RULES

    reg = PackRegistry()
    install_legacy_rules(reg)

    assert legacy_rules_installed(reg)
    assert _LEGACY_CAPABILITY_NAME in reg.capabilities
    assert len(reg.capabilities[_LEGACY_CAPABILITY_NAME].interventions) == len(FOOTER_RULES)

    # Every legacy intervention subscribes to OnTurn(BEFORE_LLM)
    turn_list = reg._by_turn[TurnPhase.BEFORE_LLM]
    assert len(turn_list) == len(FOOTER_RULES)


def test_install_legacy_rules_is_idempotent():
    reg = PackRegistry()
    install_legacy_rules(reg)
    install_legacy_rules(reg)  # Second call should replace, not duplicate

    from nature.context.footer import FOOTER_RULES
    assert len(reg.capabilities[_LEGACY_CAPABILITY_NAME].interventions) == len(FOOTER_RULES)
    assert len(reg._by_turn[TurnPhase.BEFORE_LLM]) == len(FOOTER_RULES)


# ──────────────────────────────────────────────────────────────────────
# compute_footer_hints parity — same output as pre-unification
# ──────────────────────────────────────────────────────────────────────


def test_compute_footer_hints_parity_with_legacy_walk():
    """compute_footer_hints(body, header, actor) should produce the exact
    same Hint list that walking FOOTER_RULES directly would produce.
    The new path runs through the registry; the old path is simulated
    here by calling each rule function directly."""
    from nature.context.footer import FOOTER_RULES, compute_footer_hints
    from nature.context.conversation import Conversation, Message
    from nature.context.types import ContextBody, ContextHeader, AgentRole
    from nature.protocols.message import TextContent, ToolResultContent

    # Build a body that would trigger synthesis_nudge_rule:
    # - last message is from="tool" with tool_result blocks
    # - no incomplete todos
    body = ContextBody(
        conversation=Conversation(messages=[
            Message(
                id="m1", from_="self", to="tool",
                content=[TextContent(text="calling tool")],
                timestamp=0.0,
            ),
            Message(
                id="m2", from_="tool", to="self",
                content=[
                    ToolResultContent(
                        tool_use_id="tu1",
                        content="result text",
                        is_error=False,
                    ),
                ],
                timestamp=0.0,
            ),
        ]),
    )
    header = ContextHeader(
        role=AgentRole(name="receptionist", instructions=""),
    )
    self_actor = "self"

    # Reference: walk FOOTER_RULES directly
    reference_hints = []
    for rule in FOOTER_RULES:
        h = rule(body, header, self_actor)
        if h is not None:
            reference_hints.append(h)

    # Actual: go through registry-backed compute_footer_hints
    actual_hints = compute_footer_hints(body, header, self_actor)

    assert len(actual_hints) == len(reference_hints)
    for a, r in zip(actual_hints, reference_hints):
        assert a.text == r.text


def test_compute_footer_hints_empty_when_rules_silent():
    """With an empty body, legacy rules produce no hints; the registry
    path must agree."""
    from nature.context.footer import compute_footer_hints
    from nature.context.conversation import Conversation
    from nature.context.types import ContextBody, ContextHeader, AgentRole

    body = ContextBody(conversation=Conversation(messages=[]))
    header = ContextHeader(role=AgentRole(name="x", instructions=""))
    assert compute_footer_hints(body, header, "x") == []


# ──────────────────────────────────────────────────────────────────────
# Default registry — idempotent lazy install
# ──────────────────────────────────────────────────────────────────────


def test_default_registry_lazy_installs_legacy_rules():
    """First call to compute_footer_hints via default_registry should
    auto-install legacy rules. This test is order-sensitive with other
    tests that clear default_registry; we defensively check idempotency
    rather than assuming a clean slate."""
    from nature.context.footer import compute_footer_hints
    from nature.context.conversation import Conversation
    from nature.context.types import ContextBody, ContextHeader, AgentRole

    # Trigger a call — may no-op if rules already installed
    compute_footer_hints(
        ContextBody(conversation=Conversation(messages=[])),
        ContextHeader(role=AgentRole(name="x", instructions="")),
        "x",
    )
    assert legacy_rules_installed(default_registry)


# ──────────────────────────────────────────────────────────────────────
# M2 — AreaManager tool-call dispatch hooks
# ──────────────────────────────────────────────────────────────────────


def _build_manager_with_pack_registry(pack_registry, *, tools=None, provider=None):
    """Wire up a fresh AreaManager + tmpdir EventStore for hook tests."""
    import tempfile
    from pathlib import Path

    from nature.events import FileEventStore
    from nature.frame import AreaManager

    tmp = tempfile.TemporaryDirectory()
    store = FileEventStore(Path(tmp.name))
    manager = AreaManager(
        store=store,
        provider=provider,
        tool_registry=tools or [],
        cwd="/tmp",
        pack_registry=pack_registry,
    )
    return manager, store, tmp


def _scripted_tool_then_text_provider(tool_name: str, tool_input: dict):
    """Yield one tool_use turn, then a text turn — matches test_area_manager
    pattern but kept inline so this file doesn't depend on test helpers."""
    from nature.config.constants import StopReason
    from nature.protocols.message import (
        StreamEvent,
        StreamEventType,
        TextContent,
        ToolUseContent,
        Usage,
    )
    from tests._fakes import FakeProvider

    tool_block = ToolUseContent(
        id="toolu_X", name=tool_name, input=tool_input
    )
    tool_turn = [
        StreamEvent(type=StreamEventType.MESSAGE_START),
        StreamEvent(
            type=StreamEventType.CONTENT_BLOCK_START,
            index=0,
            content_block=tool_block,
        ),
        StreamEvent(
            type=StreamEventType.CONTENT_BLOCK_STOP,
            index=0,
            content_block=tool_block,
        ),
        StreamEvent(
            type=StreamEventType.MESSAGE_STOP,
            usage=Usage(input_tokens=10, output_tokens=5),
            stop_reason=StopReason.TOOL_USE,
        ),
    ]
    text_turn = [
        StreamEvent(type=StreamEventType.MESSAGE_START),
        StreamEvent(
            type=StreamEventType.CONTENT_BLOCK_START,
            index=0,
            content_block=TextContent(text=""),
        ),
        StreamEvent(
            type=StreamEventType.CONTENT_BLOCK_DELTA,
            index=0,
            delta_text="done",
        ),
        StreamEvent(type=StreamEventType.CONTENT_BLOCK_STOP, index=0),
        StreamEvent(
            type=StreamEventType.MESSAGE_STOP,
            usage=Usage(input_tokens=8, output_tokens=2),
            stop_reason=StopReason.END_TURN,
        ),
    ]
    return FakeProvider([tool_turn, text_turn])


def _open_root_and_run(manager, *, allowed_tools):
    from nature.context.types import AgentRole

    role = AgentRole(name="receptionist", instructions="t", allowed_tools=allowed_tools)
    frame = manager.open_root(
        session_id="s1",
        role=role,
        model="fake",
        initial_user_input="please run the tool",
    )
    import asyncio
    asyncio.run(manager.run(frame))
    return frame


def test_tool_dispatch_with_empty_pack_registry_is_no_op():
    """Empty PackRegistry → behavior identical to pre-Pack manager:
    TOOL_STARTED + TOOL_COMPLETED both fire, output unchanged."""
    from nature.events.types import EventType
    from tests._fakes import FakeTool

    reg = PackRegistry()
    tools = [FakeTool("FakeRead", output="vanilla output")]
    manager, store, tmp = _build_manager_with_pack_registry(
        reg,
        tools=tools,
        provider=_scripted_tool_then_text_provider("FakeRead", {"path": "x"}),
    )
    try:
        _open_root_and_run(manager, allowed_tools=["FakeRead"])
        events = store.snapshot("s1")
        types = [e.type for e in events]
        assert EventType.TOOL_STARTED in types
        assert EventType.TOOL_COMPLETED in types
        completed = next(e for e in events if e.type == EventType.TOOL_COMPLETED)
        assert completed.payload["output"] == "vanilla output"
        assert completed.payload["is_error"] is False
    finally:
        tmp.cleanup()


def test_pre_hook_block_short_circuits_execution():
    """A Gate that returns Block must skip TOOL_STARTED + tool exec, and
    surface a synthetic error block via a single TOOL_COMPLETED event."""
    from nature.events.types import EventType
    from tests._fakes import FakeTool

    reg = PackRegistry()

    async def gate_action(ctx):
        return [Block(reason="forbidden", trace_event=EventType.BUDGET_BLOCKED)]

    reg.register_intervention(Intervention(
        id="test.gate.always_block",
        kind="gate",
        trigger=OnTool(tool_name="FakeRead", phase=ToolPhase.PRE),
        action=gate_action,
    ))

    tools = [FakeTool("FakeRead", output="should never appear")]
    manager, store, tmp = _build_manager_with_pack_registry(
        reg,
        tools=tools,
        provider=_scripted_tool_then_text_provider("FakeRead", {"path": "x"}),
    )
    try:
        _open_root_and_run(manager, allowed_tools=["FakeRead"])
        events = store.snapshot("s1")
        types = [e.type for e in events]

        # No TOOL_STARTED for the blocked call
        tool_started_count = sum(1 for t in types if t == EventType.TOOL_STARTED)
        assert tool_started_count == 0, "Block should suppress TOOL_STARTED"

        # Synthetic TOOL_COMPLETED with the block reason
        completed = next(e for e in events if e.type == EventType.TOOL_COMPLETED)
        assert completed.payload["is_error"] is True
        assert "forbidden" in completed.payload["output"]

        # The trace_event was emitted
        assert EventType.BUDGET_BLOCKED in types
    finally:
        tmp.cleanup()


def test_pre_hook_modify_tool_input_patches_input_before_execution():
    """ModifyToolInput from a Gate must merge into the tool_input dict
    that the tool actually sees."""
    from nature.events.types import EventType
    from nature.packs.types import ModifyToolInput
    from nature.protocols.tool import Tool, ToolDefinition, ToolResult

    reg = PackRegistry()

    captured: dict = {}

    class CapturingTool(Tool):
        @property
        def name(self) -> str: return "CaptureTool"
        @property
        def description(self) -> str: return "captures input"
        @property
        def input_schema(self) -> dict: return {"type": "object", "properties": {}}
        async def execute(self, input, context) -> ToolResult:
            captured.update(input)
            return ToolResult(output=f"saw {sorted(input.items())}", is_error=False)
        def to_definition(self) -> ToolDefinition:
            return ToolDefinition(name=self.name, description=self.description, input_schema=self.input_schema)

    def patcher(ctx):
        return [ModifyToolInput(patch={"injected": "yes"})]

    reg.register_intervention(Intervention(
        id="test.gate.patch_input",
        kind="gate",
        trigger=OnTool(tool_name="CaptureTool", phase=ToolPhase.PRE),
        action=patcher,
    ))

    manager, store, tmp = _build_manager_with_pack_registry(
        reg,
        tools=[CapturingTool()],
        provider=_scripted_tool_then_text_provider("CaptureTool", {"original": "value"}),
    )
    try:
        _open_root_and_run(manager, allowed_tools=["CaptureTool"])
        assert captured.get("original") == "value"
        assert captured.get("injected") == "yes"
        # TOOL_STARTED should reflect the patched input
        events = store.snapshot("s1")
        started = next(e for e in events if e.type == EventType.TOOL_STARTED)
        assert started.payload["tool_input"]["injected"] == "yes"
    finally:
        tmp.cleanup()


def test_post_hook_modify_tool_result_overrides_output():
    """A POST listener returning ModifyToolResult must update the
    TOOL_COMPLETED event's output and the returned block."""
    from nature.events.types import EventType
    from nature.packs.types import ModifyToolResult
    from tests._fakes import FakeTool

    reg = PackRegistry()

    def overrider(ctx):
        return [ModifyToolResult(output="<<rewritten>>", is_error=False)]

    reg.register_intervention(Intervention(
        id="test.listener.override_result",
        kind="listener",
        trigger=OnTool(tool_name="FakeRead", phase=ToolPhase.POST),
        action=overrider,
    ))

    manager, store, tmp = _build_manager_with_pack_registry(
        reg,
        tools=[FakeTool("FakeRead", output="original")],
        provider=_scripted_tool_then_text_provider("FakeRead", {"path": "x"}),
    )
    try:
        _open_root_and_run(manager, allowed_tools=["FakeRead"])
        events = store.snapshot("s1")
        completed = next(e for e in events if e.type == EventType.TOOL_COMPLETED)
        assert completed.payload["output"] == "<<rewritten>>"
    finally:
        tmp.cleanup()


def test_post_hook_modify_tool_result_appends_hint():
    """append_hint should concatenate after the existing output."""
    from nature.events.types import EventType
    from nature.packs.types import ModifyToolResult
    from tests._fakes import FakeTool

    reg = PackRegistry()

    def appender(ctx):
        return [ModifyToolResult(append_hint="(framework hint: try X)")]

    reg.register_intervention(Intervention(
        id="test.listener.append_hint",
        kind="listener",
        trigger=OnTool(tool_name="FakeRead", phase=ToolPhase.POST),
        action=appender,
    ))

    manager, store, tmp = _build_manager_with_pack_registry(
        reg,
        tools=[FakeTool("FakeRead", output="base")],
        provider=_scripted_tool_then_text_provider("FakeRead", {"path": "x"}),
    )
    try:
        _open_root_and_run(manager, allowed_tools=["FakeRead"])
        events = store.snapshot("s1")
        completed = next(e for e in events if e.type == EventType.TOOL_COMPLETED)
        assert completed.payload["output"].startswith("base")
        assert "framework hint: try X" in completed.payload["output"]
    finally:
        tmp.cleanup()


def test_post_effect_phase_sees_primary_effects():
    """POST_EFFECT listeners must receive PRIMARY listeners' effects via
    ctx.primary_effects, while PRIMARY listeners always see an empty list."""
    from nature.events.types import EventType
    from nature.events.payloads import EditMissPayload

    reg = PackRegistry()

    primary_calls: list[str] = []
    post_calls: list[list] = []

    def primary_action(ctx):
        # PRIMARY phase MUST see an empty primary_effects.
        primary_calls.append(f"primary_saw_{len(ctx.primary_effects)}")
        return [EmitEvent(
            event_type=EventType.EDIT_MISS,
            payload=EditMissPayload(file="a.py"),
        )]

    def post_action(ctx):
        # POST_EFFECT phase MUST receive primary's effects.
        post_calls.append([type(e).__name__ for e in ctx.primary_effects])
        return []

    reg.register_intervention(Intervention(
        id="t.primary",
        kind="listener",
        trigger=OnTool(tool_name="X", phase=ToolPhase.POST),
        phase=InterventionPhase.PRIMARY,
        action=primary_action,
    ))
    reg.register_intervention(Intervention(
        id="t.post",
        kind="listener",
        trigger=OnTool(tool_name="X", phase=ToolPhase.POST),
        phase=InterventionPhase.POST_EFFECT,
        action=post_action,
    ))

    ctx = InterventionContext(
        session_id="s", now=time.time(), registry=reg,
        tool_call=ToolCallInfo(
            tool_name="X", tool_use_id="tu", tool_input={},
            phase=ToolPhase.POST, result_is_error=False,
        ),
    )
    effects = asyncio.run(reg.dispatch_tool(ToolPhase.POST, ctx))

    assert primary_calls == ["primary_saw_0"]
    assert post_calls == [["EmitEvent"]]
    assert len(effects) == 1   # only primary's EmitEvent (post returned [])


def test_post_effect_phase_does_not_run_when_no_post_listeners():
    """Optimization sanity check: with zero POST_EFFECT listeners, the
    second pass is skipped and PRIMARY effects are returned directly."""
    reg = PackRegistry()
    reg.register_intervention(Intervention(
        id="t.only_primary",
        kind="listener",
        trigger=OnTool(tool_name="Y", phase=ToolPhase.POST),
        action=lambda ctx: [],
    ))
    ctx = InterventionContext(
        session_id="s", now=time.time(), registry=reg,
        tool_call=ToolCallInfo(
            tool_name="Y", tool_use_id="tu", tool_input={},
            phase=ToolPhase.POST,
        ),
    )
    effects = asyncio.run(reg.dispatch_tool(ToolPhase.POST, ctx))
    assert effects == []


def test_post_hook_emit_event_lands_in_store():
    """A POST listener returning EmitEvent must produce an event in the store
    BEFORE the TOOL_COMPLETED event for the same call."""
    from nature.events.types import EventType
    from nature.events.payloads import EditMissPayload
    from tests._fakes import FakeTool

    reg = PackRegistry()

    def emitter(ctx):
        return [
            EmitEvent(
                event_type=EventType.EDIT_MISS,
                payload=EditMissPayload(file="foo.py", fuzzy_match="def foo():", lineno=10),
            ),
        ]

    reg.register_intervention(Intervention(
        id="test.listener.emit_edit_miss",
        kind="listener",
        trigger=OnTool(tool_name="FakeRead", phase=ToolPhase.POST),
        action=emitter,
    ))

    manager, store, tmp = _build_manager_with_pack_registry(
        reg,
        tools=[FakeTool("FakeRead", output="ok")],
        provider=_scripted_tool_then_text_provider("FakeRead", {"path": "x"}),
    )
    try:
        _open_root_and_run(manager, allowed_tools=["FakeRead"])
        events = store.snapshot("s1")
        types = [e.type for e in events]
        # EDIT_MISS landed
        assert EventType.EDIT_MISS in types
        # And it was emitted BEFORE TOOL_COMPLETED for the call
        edit_idx = next(i for i, e in enumerate(events) if e.type == EventType.EDIT_MISS)
        completed_idx = next(
            i for i, e in enumerate(events)
            if e.type == EventType.TOOL_COMPLETED
        )
        assert edit_idx < completed_idx
        # Payload round-tripped
        edit = events[edit_idx]
        assert edit.payload["file"] == "foo.py"
        assert edit.payload["fuzzy_match"] == "def foo():"
        assert edit.payload["lineno"] == 10
    finally:
        tmp.cleanup()
