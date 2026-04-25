"""PackRegistry — the dispatch hub for Interventions.

The registry owns three responsibilities:

1. **Registration**: Packs hand over their Capabilities, which contain
   Tools and Interventions. Tools land in a flat name→tool dict.
   Interventions land in a trigger-indexed structure so dispatch is
   O(1) lookup per trigger firing.

2. **Dispatch**: When the framework hits a hook point (pre/post tool
   call, contributor build, event emission, ...), it asks the registry
   for the list of Interventions that react to that trigger. The
   registry invokes each action, collects the returned Effects, and
   returns them to the caller. Effect application is the caller's
   job — see `nature.packs.effects.apply_effects`.

3. **Default instance**: a module-level `default_registry` exists so
   legacy code paths (`ContextComposer`, `AreaManager`) can find "the"
   registry without threading an argument through every call. Tests
   spin up their own instances and avoid the default.

Scope note (M1):
    The registry supports Gate, Listener, and Contributor intervention
    kinds. Depth limits, cross-pack dependency validation, and hot
    uninstall are not yet implemented — see pack_architecture.md §13.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass, field

from dataclasses import replace as dc_replace

from nature.events.types import EventType
from nature.packs.types import (
    Capability,
    Effect,
    Intervention,
    InterventionContext,
    InterventionPhase,
    OnCondition,
    OnEvent,
    OnFrame,
    OnLLM,
    OnTool,
    OnTurn,
    Tool,
    ToolPhase,
    TriggerSpec,
    TurnPhase,
    FramePhase,
    LLMPhase,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# PackRegistry
# ──────────────────────────────────────────────────────────────────────


@dataclass
class PackRegistry:
    """Trigger-indexed store of Capabilities.

    All indexes are built incrementally as capabilities are registered.
    Lookup on dispatch is a single dict access + linear walk of the
    (small) reacting-intervention list, so the hot path stays flat.
    """

    capabilities: dict[str, Capability] = field(default_factory=dict)
    tools: dict[str, Tool] = field(default_factory=dict)
    interventions: dict[str, Intervention] = field(default_factory=dict)

    # Trigger indexes
    _by_tool_pre: dict[str | None, list[Intervention]] = field(default_factory=dict)
    _by_tool_post: dict[str | None, list[Intervention]] = field(default_factory=dict)
    _by_event: dict[EventType, list[Intervention]] = field(default_factory=dict)
    _by_llm_pre: list[Intervention] = field(default_factory=list)
    _by_llm_post: list[Intervention] = field(default_factory=list)
    _by_turn: dict[TurnPhase, list[Intervention]] = field(default_factory=dict)
    _by_frame: dict[FramePhase, list[Intervention]] = field(default_factory=dict)
    _by_condition: list[Intervention] = field(default_factory=list)

    # ── registration ────────────────────────────────────────────────

    def register_capability(self, cap: Capability) -> None:
        """Install one Capability's tools + interventions into the indexes."""
        if cap.name in self.capabilities:
            logger.warning("Capability %r already registered; overwriting", cap.name)
        self.capabilities[cap.name] = cap
        for tool in cap.tools:
            self.tools[tool.name] = tool
        for intv in cap.interventions:
            self._register_intervention(intv)

    def register_intervention(self, intv: Intervention) -> None:
        """Add a single free-standing Intervention (e.g., for tests)."""
        self._register_intervention(intv)

    def _register_intervention(self, intv: Intervention) -> None:
        if intv.id in self.interventions:
            logger.warning("Intervention id %r already registered; overwriting", intv.id)
        self.interventions[intv.id] = intv

        trigger = intv.trigger
        if isinstance(trigger, OnTool):
            idx = self._by_tool_pre if trigger.phase == ToolPhase.PRE else self._by_tool_post
            idx.setdefault(trigger.tool_name, []).append(intv)
        elif isinstance(trigger, OnEvent):
            self._by_event.setdefault(trigger.event_type, []).append(intv)
        elif isinstance(trigger, OnLLM):
            target = self._by_llm_pre if trigger.phase == LLMPhase.PRE else self._by_llm_post
            target.append(intv)
        elif isinstance(trigger, OnTurn):
            self._by_turn.setdefault(trigger.phase, []).append(intv)
        elif isinstance(trigger, OnFrame):
            self._by_frame.setdefault(trigger.phase, []).append(intv)
        elif isinstance(trigger, OnCondition):
            self._by_condition.append(intv)
        else:  # pragma: no cover — guarded by type system
            raise TypeError(f"Unknown trigger type: {type(trigger).__name__}")

    def clear(self) -> None:
        """Empty the registry — primarily for test setup/teardown."""
        self.capabilities.clear()
        self.tools.clear()
        self.interventions.clear()
        self._by_tool_pre.clear()
        self._by_tool_post.clear()
        self._by_event.clear()
        self._by_llm_pre.clear()
        self._by_llm_post.clear()
        self._by_turn.clear()
        self._by_frame.clear()
        self._by_condition.clear()

    # ── dispatch ────────────────────────────────────────────────────

    async def dispatch_tool(
        self,
        phase: ToolPhase,
        ctx: InterventionContext,
    ) -> list[Effect]:
        """Run every intervention subscribed to this tool phase.

        Looks up both the tool-specific bucket and the wildcard bucket
        (`tool_name=None`), runs `where` filters, awaits each action,
        and concatenates the returned Effects.

        Listener interventions are split into two phases:
        PRIMARY runs first, POST_EFFECT runs after with PRIMARY's
        effect list available via `ctx.primary_effects` (see §4.5).
        Gates have no phase concept — they all run together.
        """
        if ctx.tool_call is None:
            raise ValueError("dispatch_tool requires ctx.tool_call")
        idx = self._by_tool_pre if phase == ToolPhase.PRE else self._by_tool_post
        candidates: list[Intervention] = []
        candidates.extend(idx.get(ctx.tool_call.tool_name, ()))
        candidates.extend(idx.get(None, ()))
        return await self._run_phased(
            candidates, ctx, tool_where_filter=True,
        )

    async def dispatch_event(
        self,
        event_type: EventType,
        ctx: InterventionContext,
    ) -> list[Effect]:
        """Run every listener subscribed to an EventType.

        Same PRIMARY → POST_EFFECT phase split as `dispatch_tool`.
        """
        candidates = list(self._by_event.get(event_type, ()))
        return await self._run_phased(candidates, ctx)

    def dispatch_event_sync(
        self,
        event_type: EventType,
        ctx: InterventionContext,
    ) -> list[Effect]:
        """Synchronous sibling of `dispatch_event`.

        Used by `AreaManager._emit`, which is itself sync and cannot
        await. Listener actions must be sync — async actions are
        closed and a warning is logged (same policy as
        `dispatch_turn_sync`). PRIMARY vs POST_EFFECT ordering is
        still honoured via the `InterventionPhase` field.
        """
        candidates = list(self._by_event.get(event_type, ()))
        primary: list[Intervention] = []
        post: list[Intervention] = []
        for intv in candidates:
            if intv.kind == "listener" and intv.phase == InterventionPhase.POST_EFFECT:
                post.append(intv)
            else:
                primary.append(intv)

        def _run(batch: list[Intervention]) -> list[Effect]:
            out: list[Effect] = []
            for intv in batch:
                if not intv.default_enabled:
                    continue
                try:
                    result = intv.action(ctx)
                    if inspect.isawaitable(result):
                        close = getattr(result, "close", None)
                        if callable(close):
                            try:
                                close()
                            except Exception:
                                pass
                        logger.error(
                            "Listener %r returned an awaitable in sync "
                            "event dispatch — skipping. Convert to sync "
                            "or use dispatch_event().",
                            intv.id,
                        )
                        continue
                except Exception:
                    logger.exception(
                        "Listener %r raised on event %s — skipping",
                        intv.id, event_type,
                    )
                    continue
                if result:
                    out.extend(result)
            return out

        primary_effects = _run(primary)
        ctx = dc_replace(ctx, primary_effects=list(primary_effects))
        return primary_effects + _run(post)

    async def dispatch_turn(
        self,
        phase: TurnPhase,
        ctx: InterventionContext,
    ) -> list[Effect]:
        """Run every intervention subscribed to `OnTurn(phase)`.

        Mostly Contributors today (footer hints + legacy rules), but
        any kind with an OnTurn trigger lands here. Actions may be
        sync or async.
        """
        candidates = list(self._by_turn.get(phase, ()))
        return await self._run_candidates(candidates, ctx)

    def dispatch_turn_sync(
        self,
        phase: TurnPhase,
        ctx: InterventionContext,
    ) -> list[Effect]:
        """Synchronous sibling of `dispatch_turn`.

        Used by callers that cannot await — notably
        `ContextComposer.compose()`, which runs inside `llm_agent` just
        before the provider call and has no obvious `await` point.
        Contributor actions here must be synchronous; an action that
        returns an awaitable is logged and skipped.
        """
        candidates = list(self._by_turn.get(phase, ()))
        out: list[Effect] = []
        for intv in candidates:
            if not intv.default_enabled:
                continue
            try:
                result = intv.action(ctx)
                if inspect.isawaitable(result):
                    # Close the coroutine/awaitable so Python doesn't
                    # log a RuntimeWarning about an un-awaited coroutine.
                    close = getattr(result, "close", None)
                    if callable(close):
                        try:
                            close()
                        except Exception:
                            pass
                    logger.error(
                        "Contributor %r returned an awaitable in sync "
                        "dispatch — skipping. Use dispatch_turn() "
                        "(async) for async contributors.",
                        intv.id,
                    )
                    continue
            except Exception:
                logger.exception(
                    "Contributor %r raised in action — skipping effects",
                    intv.id,
                )
                continue
            if result:
                out.extend(result)
        return out

    async def dispatch_frame(
        self,
        phase: FramePhase,
        ctx: InterventionContext,
    ) -> list[Effect]:
        """Run every intervention subscribed to `OnFrame(phase)`."""
        candidates = list(self._by_frame.get(phase, ()))
        return await self._run_candidates(candidates, ctx)

    def dispatch_frame_sync(
        self,
        phase: FramePhase,
        ctx: InterventionContext,
    ) -> list[Effect]:
        """Synchronous sibling of `dispatch_frame` for lifecycle call
        sites that can't await (`open_root`, `open_child`, `close`, and
        the sync branches of `run()`'s state transitions). Same async-
        action skip + exception-swallow policy as the other _sync
        dispatchers."""
        candidates = list(self._by_frame.get(phase, ()))
        out: list[Effect] = []
        for intv in candidates:
            if not intv.default_enabled:
                continue
            try:
                result = intv.action(ctx)
                if inspect.isawaitable(result):
                    close = getattr(result, "close", None)
                    if callable(close):
                        try:
                            close()
                        except Exception:
                            pass
                    logger.error(
                        "Listener %r returned an awaitable in sync "
                        "frame dispatch — skipping.",
                        intv.id,
                    )
                    continue
            except Exception:
                logger.exception(
                    "Listener %r raised on frame phase %s — skipping",
                    intv.id, phase,
                )
                continue
            if result:
                out.extend(result)
        return out

    async def dispatch_llm(
        self,
        phase: LLMPhase,
        ctx: InterventionContext,
    ) -> list[Effect]:
        """Run interventions subscribed to LLM call pre/post phases."""
        candidates = (
            self._by_llm_pre if phase == LLMPhase.PRE else self._by_llm_post
        )
        return await self._run_candidates(list(candidates), ctx)

    # ── introspection ───────────────────────────────────────────────

    def list_intervention_ids(self) -> list[str]:
        return sorted(self.interventions.keys())

    def get_tool(self, name: str) -> Tool | None:
        return self.tools.get(name)

    # ── internals ───────────────────────────────────────────────────

    async def _run_phased(
        self,
        candidates: list[Intervention],
        ctx: InterventionContext,
        *,
        tool_where_filter: bool = False,
    ) -> list[Effect]:
        """Run candidates with PRIMARY → POST_EFFECT phase ordering.

        Gates and Contributors are not phase-aware: they live in PRIMARY.
        Listeners are split by their `phase` field. Within a phase,
        actions run in declaration order with no visibility into each
        other's results.
        """
        primary_list: list[Intervention] = []
        post_list: list[Intervention] = []
        for intv in candidates:
            if intv.kind == "listener" and intv.phase == InterventionPhase.POST_EFFECT:
                post_list.append(intv)
            else:
                primary_list.append(intv)

        primary_effects = await self._run_candidates(
            primary_list, ctx, tool_where_filter=tool_where_filter,
        )
        if not post_list:
            return primary_effects

        # POST_EFFECT receives PRIMARY's output via ctx.primary_effects.
        post_ctx = dc_replace(ctx, primary_effects=list(primary_effects))
        post_effects = await self._run_candidates(
            post_list, post_ctx, tool_where_filter=tool_where_filter,
        )
        return primary_effects + post_effects

    async def _run_candidates(
        self,
        candidates: list[Intervention],
        ctx: InterventionContext,
        *,
        tool_where_filter: bool = False,
    ) -> list[Effect]:
        out: list[Effect] = []
        for intv in candidates:
            if not intv.default_enabled:
                # Per-agent allowed_interventions filtering will land in
                # a later phase; for M1 the default_enabled flag is the
                # only gate.
                continue
            if tool_where_filter and isinstance(intv.trigger, OnTool):
                where = intv.trigger.where
                if where is not None and ctx.tool_call is not None:
                    try:
                        if not where(ctx.tool_call):
                            continue
                    except Exception:
                        logger.exception(
                            "where() for intervention %r raised — skipping",
                            intv.id,
                        )
                        continue
            try:
                result = intv.action(ctx)
                if inspect.isawaitable(result):
                    result = await result
            except Exception:
                logger.exception(
                    "Intervention %r raised in action — skipping effects",
                    intv.id,
                )
                continue
            if result:
                out.extend(result)
        return out


# ──────────────────────────────────────────────────────────────────────
# Default module-level registry
# ──────────────────────────────────────────────────────────────────────


default_registry = PackRegistry()
"""Process-wide registry used by legacy callers (ContextComposer,
AreaManager) that can't thread an instance through their call chain.
Tests should instantiate a private `PackRegistry()` instead of mutating
this one."""


__all__ = [
    "PackRegistry",
    "default_registry",
]
