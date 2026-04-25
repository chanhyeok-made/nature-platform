"""AreaManager — owns the execution loop that drives Frames to resolution.

The AreaManager is the orchestrator in the new architecture. It:

- Creates Frames (open_root / open_child)
- Calls llm_agent as a pure function, applies its output
- Executes pending tool actions between agent calls
- Writes EVERY state transition to the EventStore (single sink)
- Never touches UI — UIs consume from the store independently

This class absorbs what DefaultOrchestrator + query()'s loop used to
own, minus the middleware chain (UIs are now pure EventStore consumers).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Callable
from uuid import uuid4

from nature.agent.llm_agent import llm_agent
from nature.agent.output import AgentOutput, Signal
from nature.agent.executor import execute_tools
from nature.context.body_compaction import BodyCompactionPipeline
from nature.context.composer import ContextComposer
from nature.context.conversation import Conversation, Message, MessageAnnotation
from nature.context.types import AgentRole, Context, ContextBody, ContextHeader
from nature.events.payloads import (
    AnnotationStoredPayload,
    BodyCompactedPayload,
    FrameClosedPayload,
    FrameErroredPayload,
    FrameOpenedPayload,
    FrameReopenedPayload,
    FrameResolvedPayload,
    HeaderSnapshotPayload,
    HintInjectedPayload,
    LLMErrorPayload,
    LLMRequestPayload,
    LLMResponsePayload,
    MessageAppendedPayload,
    TodoWrittenPayload,
    ToolCompletedPayload,
    ToolStartedPayload,
    UserInputPayload,
    dump_payload,
)
from nature.events.store import EventStore
from nature.events.types import Event, EventType
from nature.events.payloads import _PayloadBase
from nature.frame.frame import Frame, FrameState
from nature.packs.registry import PackRegistry, default_registry
from nature.packs.types import (
    Block,
    EmitEvent,
    FramePhase,
    InterventionContext,
    LLMPhase,
    ModifyToolInput,
    ModifyToolResult,
    ToolCallInfo,
    ToolPhase,
)
from nature.protocols.message import (
    ContentBlock,
    TextContent,
    ToolResultContent,
    ToolUseContent,
)
from nature.protocols.provider import LLMProvider
from nature.protocols.tool import Tool, ToolContext
from nature.protocols.turn import Action
from nature.tools.builtin.todowrite import TODO_WRITE_TOOL_NAME, TodoWriteInput
from nature.utils.ids import generate_tool_use_id

AGENT_TOOL_NAME = "Agent"

# Callable signature: name → AgentRole (or None to fall through to the profile system).
RoleResolver = Callable[[str], "AgentRole | None"]

# Callable signature: agent_name → LLMProvider for that agent (or None to
# fall through to the manager's default provider). Supplied by the preset
# flow so sub-agents on a different host than the root land on the right
# endpoint; legacy frame.json sessions leave this None and keep using the
# single provider passed to the manager.
ProviderResolver = Callable[[str], "LLMProvider | None"]

# Callable signature: agent_name → role policy budget for max_output_tokens
# (or None to defer to the provider). Supplied by the preset flow so each
# role gets an output budget that matches its task shape (judges short,
# analyzers long). The provider still clips to the model's physical
# ceiling, so a resolver returning None is equivalent to "use provider
# default"; returning a large value is safe because the ceiling gate
# handles the actual model limits.
MaxOutputResolver = Callable[[str], int | None]

logger = logging.getLogger(__name__)


class AreaManager:
    """Drives Frames to resolution. Writes all events to the store."""

    def __init__(
        self,
        *,
        store: EventStore,
        provider: LLMProvider,
        tool_registry: list[Tool],
        cwd: str | None = None,
        composer: ContextComposer | None = None,
        role_resolver: RoleResolver | None = None,
        compaction_pipeline: BodyCompactionPipeline | None = None,
        pack_registry: PackRegistry | None = None,
        provider_resolver: ProviderResolver | None = None,
        max_output_resolver: MaxOutputResolver | None = None,
    ) -> None:
        self._store = store
        self._provider = provider
        self._tool_registry = tool_registry
        self._cwd = cwd or os.getcwd()
        self._composer = composer or ContextComposer()
        self._role_resolver = role_resolver
        self._compaction_pipeline = compaction_pipeline
        # Optional per-agent provider lookup. When set, every LLM call
        # site consults it with the active frame's self_actor, falling
        # back to `self._provider` when the resolver returns None (so
        # an unknown agent still gets a usable provider). Legacy
        # sessions leave this unset and always hit `self._provider`.
        self._provider_resolver = provider_resolver
        # Optional per-agent output-token budget lookup; preset flow
        # uses this, legacy leaves it None (provider default applies).
        self._max_output_resolver = max_output_resolver
        # Pack-architecture dispatch hub. None → process-wide singleton so
        # default callers keep working without threading an instance through.
        # Tests pass a private PackRegistry() to keep state isolated.
        self._pack_registry = pack_registry or default_registry
        # Cascade counter for OnEvent Listener → EmitEvent → _emit
        # recursion; bounded by `_EMIT_MAX_DEPTH`.
        self._emit_depth = 0
        # Auto-install built-in Packs + file-based user/project Packs
        # into the default registry. Tests that pass their own empty
        # PackRegistry opt out — they get a clean registry and can
        # register only what they need.
        if pack_registry is None:
            from nature.packs.builtin import install_builtin_packs
            from nature.packs.discovery import install_discovered_packs
            install_builtin_packs(self._pack_registry)
            install_discovered_packs(self._pack_registry, project_dir=self._cwd)

    # ------------------------------------------------------------------
    # Frame lifecycle
    # ------------------------------------------------------------------

    def open_root(
        self,
        *,
        session_id: str,
        role: AgentRole,
        model: str,
        initial_user_input: str,
    ) -> Frame:
        """Create a root frame and seed it with the user's first message."""
        frame = Frame(
            id=f"frame_{uuid4().hex[:16]}",
            session_id=session_id,
            purpose="root",
            context=Context(
                header=ContextHeader(role=role),
                body=ContextBody(conversation=Conversation()),
            ),
            model=model,
            counterparty="user",  # root frames always reply to the user
        )
        self._init_pack_state(frame)

        self._emit(frame, EventType.FRAME_OPENED, FrameOpenedPayload(
            purpose=frame.purpose,
            parent_id=None,
            role_name=role.name,
            role_description=role.description,
            instructions=role.instructions,
            allowed_tools=role.allowed_tools,
            model=model,
            role_model=role.model,
        ))
        self._emit_header_snapshot(frame)
        self._dispatch_frame_lifecycle(frame, FramePhase.OPENED)

        user_msg = Message(
            from_="user",
            to=role.name,
            content=[TextContent(text=initial_user_input)],
            timestamp=time.time(),
        )

        self._emit(frame, EventType.USER_INPUT, UserInputPayload(
            text=initial_user_input,
            source="user",
        ))
        self._append_message(frame, user_msg)
        return frame

    def open_child(
        self,
        *,
        parent: Frame,
        role: AgentRole,
        initial_input: str,
        model: str | None = None,
        purpose: str = "delegation",
        spawned_from_message_id: str | None = None,
        spawned_by_tool_use_id: str | None = None,
    ) -> Frame:
        """Open a child frame for sub-agent delegation.

        The child gets a fresh Context (new header with the given role,
        empty body) — parent context is NEVER inherited. The delegation
        prompt is seeded as the first message, flowing from the parent's
        actor to the child's actor. The child's counterparty is locked
        to the parent's actor at open time so it doesn't drift to "tool"
        when tool results land in the child's conversation.

        `spawned_from_message_id` / `spawned_by_tool_use_id` record the
        parent-side tool_use that caused the spawn, so replay can wire
        parent tool_result → child frame lookups without timing-based
        inference.
        """
        frame = Frame(
            id=f"frame_{uuid4().hex[:16]}",
            session_id=parent.session_id,
            purpose=purpose,
            context=Context(
                header=ContextHeader(role=role),
                body=ContextBody(conversation=Conversation()),
            ),
            model=model or parent.model,
            parent_id=parent.id,
            counterparty=parent.self_actor,
        )
        parent.children_ids.append(frame.id)
        self._init_pack_state(frame)

        self._emit(frame, EventType.FRAME_OPENED, FrameOpenedPayload(
            purpose=purpose,
            parent_id=parent.id,
            role_name=role.name,
            role_description=role.description,
            instructions=role.instructions,
            allowed_tools=role.allowed_tools,
            model=frame.model,
            role_model=role.model,
            spawned_from_message_id=spawned_from_message_id,
            spawned_by_tool_use_id=spawned_by_tool_use_id,
        ))
        self._emit_header_snapshot(frame)
        self._dispatch_frame_lifecycle(frame, FramePhase.OPENED)

        delegation_msg = Message(
            from_=parent.self_actor,
            to=role.name,
            content=[TextContent(text=initial_input)],
            timestamp=time.time(),
        )
        self._append_message(frame, delegation_msg)
        return frame

    def append_user_input(self, frame: Frame, text: str) -> None:
        """Append a new user message to a (usually awaiting) frame.

        After this call the frame is back in ACTIVE state and the caller
        can invoke run() again to continue the conversation.
        """
        if frame.state == FrameState.CLOSED:
            raise ValueError(f"Cannot append to closed frame {frame.id}")

        self._emit(frame, EventType.USER_INPUT, UserInputPayload(
            text=text,
            source="user",
        ))

        user_msg = Message(
            from_="user",
            to=frame.self_actor,
            content=[TextContent(text=text)],
            timestamp=time.time(),
        )
        self._append_message(frame, user_msg)
        frame.state = FrameState.ACTIVE

    def close(self, frame: Frame) -> None:
        """Emit frame.closed. Caller should drop references to the frame."""
        frame.state = FrameState.CLOSED
        self._emit(frame, EventType.FRAME_CLOSED, FrameClosedPayload())
        self._dispatch_frame_lifecycle(frame, FramePhase.CLOSED)

    def reopen(self, frame: Frame, *, reason: str = "resume") -> None:
        """Reactivate a terminal frame so it can accept new input.

        Called by SessionRegistry.resume_session after reconstruct()
        rebuilds an archived frame from the event log. Emits
        FRAME_REOPENED so the state transition RESOLVED/CLOSED/ERROR →
        ACTIVE is explicit in the log — otherwise a reader would have
        to infer "this frame was reopened" from a MESSAGE_APPENDED
        landing on a frame whose last state event was terminal.

        No-op when the frame is already ACTIVE or AWAITING_USER.
        """
        terminal = (FrameState.RESOLVED, FrameState.CLOSED, FrameState.ERROR)
        if frame.state not in terminal:
            return
        previous = frame.state.value
        frame.state = FrameState.ACTIVE
        self._emit(frame, EventType.FRAME_REOPENED, FrameReopenedPayload(
            previous_state=previous,
            reason=reason,
        ))

    # ------------------------------------------------------------------
    # Execution loop
    # ------------------------------------------------------------------

    def _provider_for(self, frame: Frame) -> LLMProvider:
        """Pick the LLMProvider that should drive `frame`.

        With a `provider_resolver` wired in (preset flow), ask it for
        the frame's self_actor; fall back to `self._provider` when
        the resolver returns None (unknown agent) or no resolver is
        configured (legacy frame.json flow).
        """
        if self._provider_resolver is None:
            return self._provider
        resolved = self._provider_resolver(frame.self_actor)
        return resolved if resolved is not None else self._provider

    async def run(self, frame: Frame) -> Frame:
        """Drive the agent loop until the frame resolves or pauses.

        The loop terminates when the agent signals RESOLVED / NEEDS_USER
        / ERROR, or on an exception from the provider. Returns the same
        Frame with its `state` field updated.
        """
        while frame.state == FrameState.ACTIVE:
            await self._maybe_compact(frame)
            provider = self._provider_for(frame)
            max_out = (
                self._max_output_resolver(frame.self_actor)
                if self._max_output_resolver is not None else None
            )
            await self._dispatch_llm(frame, LLMPhase.PRE)
            try:
                output = await llm_agent(
                    frame.context,
                    self_actor=frame.self_actor,
                    counterparty=self._counterparty_for(frame),
                    model=frame.model,
                    provider=provider,
                    tool_registry=self._tool_registry,
                    composer=self._composer,
                    max_output_tokens=max_out,
                    # Always opt in to prompt caching. The provider
                    # silently ignores this when the underlying LLM
                    # (e.g., openai-compatible local) doesn't support
                    # it, so the override is safe to apply globally.
                    # Without this, every turn re-pays the full input
                    # token cost for system + tools + history — which
                    # is what made session 409b958e burn ~$6 on a
                    # five-line code change (1.84M billed input,
                    # 0 cache reads).
                    #
                    # TTL="1h": multi-frame sessions can run 10+ min
                    # (session 29324910 refactor took 9.7 min), and
                    # the default 5m TTL would expire mid-session,
                    # forcing re-creation of the system+tools prefix
                    # at 1.25x cost per recreation. 1h TTL creates
                    # at 2x but survives the whole session, so a
                    # prefix that's reused >3 times across specialist
                    # frames wins — which is the common case in core-
                    # orchestrated multi-step work.
                    cache_control={"type": "ephemeral", "ttl": "1h"},
                )
            except Exception as exc:
                logger.exception("llm_agent call failed in frame %s", frame.id)
                self._emit_llm_error(frame, exc)
                self._emit_frame_errored(frame, exc)
                self._dispatch_frame_lifecycle(frame, FramePhase.ERRORED)
                frame.state = FrameState.ERROR
                break

            self._emit_llm_request(frame, output)
            self._emit_llm_response(frame, output)
            await self._dispatch_llm(frame, LLMPhase.POST)

            for msg in output.new_messages:
                self._append_message(frame, msg)
            for ann in output.annotations:
                self._emit_annotation(frame, ann)

            if output.actions:
                await self._execute_and_apply(frame, output.actions)

            if output.signal == Signal.RESOLVED:
                frame.state = FrameState.RESOLVED
                break
            if output.signal == Signal.NEEDS_USER:
                frame.state = FrameState.AWAITING_USER
                break
            if output.signal == Signal.ERROR:
                frame.state = FrameState.ERROR
                self._dispatch_frame_lifecycle(frame, FramePhase.ERRORED)
                break
            # CONTINUE (or DELEGATE, which is handled via Agent tool interception)
            # loop and call llm_agent again

        if frame.state == FrameState.RESOLVED:
            bubble = self._last_outgoing_message(frame)
            self._emit(frame, EventType.FRAME_RESOLVED, FrameResolvedPayload(
                bubble_message_id=bubble.id if bubble else None,
            ))
            self._dispatch_frame_lifecycle(frame, FramePhase.RESOLVED)
        return frame

    # ------------------------------------------------------------------
    # Body compaction hook
    # ------------------------------------------------------------------

    async def _maybe_compact(self, frame: Frame) -> None:
        """Run the body compaction pipeline before the next LLM call.

        No-op when no pipeline is configured or the token estimate is
        below the budget's autocompact threshold. For each strategy that
        actually mutates the body, a BODY_COMPACTED state-transition
        event is emitted so replay lands on the exact same trimmed body.
        """
        if self._compaction_pipeline is None:
            return
        try:
            result = await self._compaction_pipeline.run(
                frame.context,
                self_actor=frame.self_actor,
                tool_registry=self._filter_allowed_tools(frame),
                model=frame.model,
                provider=self._provider_for(frame),
            )
        except Exception as exc:
            logger.warning(
                "body compaction pipeline failed in frame %s: %s", frame.id, exc
            )
            return

        if not result.steps:
            return

        for step in result.steps:
            frame.context = Context(
                header=frame.context.header,
                body=step.body,
            )
            self._emit(frame, EventType.BODY_COMPACTED, BodyCompactedPayload(
                strategy=step.strategy_name,
                tokens_before=step.tokens_before,
                tokens_after=step.tokens_after,
                new_messages=list(step.body.conversation.messages),
                summary=step.summary,
            ))

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    async def _execute_and_apply(
        self, frame: Frame, actions: list[Action]
    ) -> None:
        """Handle pending actions — delegations via open_child, rest via tools.

        Partitions actions into consecutive runs by kind:
          - `AGENT_TOOL_NAME` runs execute sequentially via
            `_handle_delegation` (phase A keeps delegations serial;
            phase B will parallelize them).
          - Regular tool runs are further split into consecutive
            concurrency-safe vs unsafe sub-runs. Safe runs of size ≥ 2
            are executed as a single `execute_tools(...)` batch so the
            executor's asyncio.gather path actually fires, bracketed by
            PARALLEL_GROUP_STARTED / PARALLEL_GROUP_COMPLETED events.
            Unsafe runs (and size-1 safe runs) keep the existing
            single-tool path so timing events fall on their own ids.

        All inner TOOL_STARTED / TOOL_COMPLETED events inside a
        parallel batch carry the same `parallel_group_id` on their
        envelope — the fork API reads this to reject at_event_ids
        that fall strictly inside a batch bracket (inner events have
        no total order relative to each other, so forking at one of
        them is ambiguous).

        Produces one tool_result message that combines both delegation
        results and regular tool outputs, in action order. Delegations
        are tracked in a tool_use_id → child frame id map that rides
        along on the tool_result MESSAGE_APPENDED event.
        """
        if not actions:
            return

        # The assistant turn that bears the pending tool_use blocks is
        # the last message currently on the frame (appended just before
        # this call). Capture its id so child frames can record their
        # spawn origin.
        msgs = frame.context.body.conversation.messages
        parent_msg_id = msgs[-1].id if msgs else None

        # Walk actions in order. Two kinds of batching run here:
        #
        #   1. Regular concurrency-safe tools — coalesced consecutive
        #      runs go through `execute_tools(tool_uses=[...all...])`
        #      which hits the executor's asyncio.gather path.
        #   2. Agent delegations — consecutive Agent calls are
        #      batched via `asyncio.gather([_handle_delegation, ...])`
        #      so a turn that emits
        #      `Agent(researcher) + Agent(analyzer) + Agent(reviewer)`
        #      runs all three child frames concurrently.
        #
        # Both batch kinds are bracketed by PARALLEL_GROUP_STARTED /
        # PARALLEL_GROUP_COMPLETED events, so fork validation treats
        # them as transactional units. Switching kinds (safe→agent,
        # agent→safe) flushes the pending batch to preserve ordering.
        result_by_index: dict[int, ContentBlock] = {}
        delegations: dict[str, str] = {}

        pending_safe: list[tuple[int, Action]] = []
        pending_agents: list[tuple[int, Action]] = []

        async def flush_safe_batch() -> None:
            if not pending_safe:
                return
            if len(pending_safe) == 1:
                idx, action = pending_safe[0]
                block = await self._execute_single_tool(frame, action)
                if block is not None:
                    result_by_index[idx] = block
            else:
                blocks_in_order = await self._execute_parallel_tool_batch(
                    frame, [a for _, a in pending_safe]
                )
                for (idx, _), block in zip(pending_safe, blocks_in_order):
                    if block is not None:
                        result_by_index[idx] = block
            pending_safe.clear()

        async def flush_agent_batch() -> None:
            if not pending_agents:
                return
            if len(pending_agents) == 1:
                idx, action = pending_agents[0]
                block, child_id = await self._handle_delegation(
                    frame, action, parent_msg_id=parent_msg_id,
                )
                if block is not None:
                    result_by_index[idx] = block
                if block is not None and child_id:
                    delegations[block.tool_use_id] = child_id
            else:
                pairs_in_order = await self._execute_parallel_delegation_batch(
                    frame,
                    [a for _, a in pending_agents],
                    parent_msg_id=parent_msg_id,
                )
                for (idx, _), (block, child_id) in zip(
                    pending_agents, pairs_in_order
                ):
                    if block is not None:
                        result_by_index[idx] = block
                    if block is not None and child_id:
                        delegations[block.tool_use_id] = child_id
            pending_agents.clear()

        # Precompute which actions are safe for batching so we don't
        # ask the tool twice. Unknown tool → treat as unsafe so the
        # executor's single-tool path reports the error uniformly.
        tool_by_name = {t.name: t for t in self._tool_registry}

        def is_safe_for_batch(action: Action) -> bool:
            if action.tool_name == AGENT_TOOL_NAME:
                return False
            tool = tool_by_name.get(action.tool_name or "")
            if tool is None:
                return False
            try:
                return bool(tool.is_concurrency_safe(dict(action.tool_input)))
            except Exception:
                return False

        for i, action in enumerate(actions):
            if not action.tool_name:
                continue

            if action.tool_name == AGENT_TOOL_NAME:
                # Switching to an agent run — flush any pending safe
                # tool batch first to keep the relative order of
                # tool_result blocks intact.
                await flush_safe_batch()
                pending_agents.append((i, action))
                continue

            # Regular tool path. Any pending agent batch flushes
            # first (same ordering reason).
            await flush_agent_batch()

            if is_safe_for_batch(action):
                pending_safe.append((i, action))
                continue

            # Non-safe regular tool — flush any pending safe batch
            # first, then run this tool on its own.
            await flush_safe_batch()
            block = await self._execute_single_tool(frame, action)
            if block is not None:
                result_by_index[i] = block

        # Tail: flush whichever batch type still has pending work.
        await flush_safe_batch()
        await flush_agent_batch()

        # Reassemble result blocks in original action order so the
        # model sees each tool_result in the position matching its
        # tool_use block.
        result_blocks: list[ContentBlock] = [
            result_by_index[i] for i in sorted(result_by_index.keys())
        ]
        if not result_blocks:
            return

        tool_result_msg = Message(
            from_="tool",
            to=frame.self_actor,
            content=result_blocks,
            timestamp=time.time(),
        )
        self._append_message(frame, tool_result_msg, delegations=delegations)

    async def _execute_parallel_tool_batch(
        self, frame: Frame, actions: list[Action]
    ) -> list[ToolResultContent | None]:
        """Run 2+ concurrency-safe tools in one asyncio.gather batch.

        Bracketed by PARALLEL_GROUP_STARTED / PARALLEL_GROUP_COMPLETED
        events, and every inner TOOL_STARTED / TOOL_COMPLETED tagged
        with the shared `parallel_group_id` so session-fork can treat
        the block as atomic.

        Preserves order: the returned list lines up 1:1 with `actions`
        so the caller can write tool_results back into the combined
        tool_result message in the original position.
        """
        if len(actions) < 2:
            raise AssertionError(
                "_execute_parallel_tool_batch called with <2 actions; "
                "the caller should have used _execute_single_tool instead."
            )

        from nature.events.payloads import (
            ParallelGroupCompletedPayload,
            ParallelGroupStartedPayload,
        )
        from nature.utils.ids import generate_session_id  # monotonic enough

        group_id = "pg_" + generate_session_id()[:12]

        self._emit(
            frame,
            EventType.PARALLEL_GROUP_STARTED,
            ParallelGroupStartedPayload(
                group_id=group_id,
                tool_count=len(actions),
            ),
            parallel_group_id=group_id,
        )

        # Build tool_uses and emit TOOL_STARTED for each BEFORE the
        # gather fires. They share the same "start" moment from the
        # event log's perspective, which is fine for grouping — real
        # wall-clock start of each inner coroutine is within
        # microseconds of each other.
        tool_uses: list[ToolUseContent] = []
        started_ats: list[float] = []
        for action in actions:
            tu = ToolUseContent(
                id=action.tool_use_id or generate_tool_use_id(),
                name=action.tool_name or "",
                input=dict(action.tool_input),
            )
            tool_uses.append(tu)
            started_ats.append(time.time())
            self._emit(
                frame,
                EventType.TOOL_STARTED,
                ToolStartedPayload(
                    tool_use_id=tu.id,
                    tool_name=tu.name,
                    tool_input=dict(tu.input),
                ),
                parallel_group_id=group_id,
            )

        effective_tools = self._filter_allowed_tools(frame)
        tool_context = ToolContext(
            cwd=self._cwd,
            session_id=frame.session_id,
            agent_id=frame.id,
            pack_state=frame.pack_state,
        )

        batch_started_at = time.time()
        result_messages = await execute_tools(
            tool_uses=tool_uses,
            tools=effective_tools,
            context=tool_context,
        )
        batch_duration_ms = int((time.time() - batch_started_at) * 1000)

        # Map results back to tool_use_id → ToolResultContent so we can
        # assemble in the input order (execute_tools already returns
        # them in input order, but an explicit lookup insulates us
        # from future ordering changes).
        result_by_id: dict[str, ToolResultContent] = {}
        for result_msg in result_messages:
            for block in result_msg.content:
                if isinstance(block, ToolResultContent):
                    result_by_id[block.tool_use_id] = block

        ordered: list[ToolResultContent | None] = []
        for tu, started_at in zip(tool_uses, started_ats):
            block = result_by_id.get(tu.id)
            duration_ms = int((time.time() - started_at) * 1000)
            output_text = ""
            is_error = False
            if block is not None:
                raw = block.content
                output_text = raw if isinstance(raw, str) else json.dumps(raw)
                is_error = bool(block.is_error)
            self._emit(
                frame,
                EventType.TOOL_COMPLETED,
                ToolCompletedPayload(
                    tool_use_id=tu.id,
                    tool_name=tu.name,
                    output=output_text,
                    is_error=is_error,
                    duration_ms=duration_ms,
                ),
                parallel_group_id=group_id,
            )
            ordered.append(block)

        self._emit(
            frame,
            EventType.PARALLEL_GROUP_COMPLETED,
            ParallelGroupCompletedPayload(
                group_id=group_id,
                duration_ms=batch_duration_ms,
            ),
            parallel_group_id=group_id,
        )

        return ordered

    async def _execute_parallel_delegation_batch(
        self,
        parent: Frame,
        actions: list[Action],
        *,
        parent_msg_id: str | None,
    ) -> list[tuple[ToolResultContent | None, str | None]]:
        """Run 2+ Agent delegations concurrently via asyncio.gather.

        Opens N child frames, runs each to completion in its own
        coroutine, and returns the (result_block, child_frame_id)
        pairs in the SAME ORDER as `actions` so the caller can
        rebuild the tool_result envelope without reshuffling.

        Bracketed by PARALLEL_GROUP_STARTED / PARALLEL_GROUP_COMPLETED
        around the whole gather. Each delegation's own TOOL_STARTED
        / TOOL_COMPLETED (emitted inside `_handle_delegation`) gets
        the bracket's group_id via the `parallel_group_id` keyword,
        so fork validation sees the whole batch (plus every inner
        child-frame event that landed between STARTED and COMPLETED
        ids, by virtue of the id range check alone) as atomic.

        Concurrency safety: the event store's `append` is
        synchronous (no awaits inside), and the frame manager's
        in-memory frames dict is mutated from the event loop's
        current coroutine only between awaits — which means the
        gathered _handle_delegation coroutines can interleave
        freely without corrupting state. Each child frame has its
        own frame_id and its own Context, so there's no shared
        mutable state between siblings.
        """
        if len(actions) < 2:
            raise AssertionError(
                "_execute_parallel_delegation_batch called with <2 actions"
            )

        from nature.events.payloads import (
            ParallelGroupCompletedPayload,
            ParallelGroupStartedPayload,
        )
        from nature.utils.ids import generate_session_id

        group_id = "pg_" + generate_session_id()[:12]

        self._emit(
            parent,
            EventType.PARALLEL_GROUP_STARTED,
            ParallelGroupStartedPayload(
                group_id=group_id,
                tool_count=len(actions),
            ),
            parallel_group_id=group_id,
        )

        batch_started_at = time.time()

        # asyncio.gather preserves argument order on its return, so
        # the resulting list lines up with `actions` 1:1 without any
        # explicit bookkeeping. `return_exceptions=False` means the
        # whole batch raises if any child frame throws — that's the
        # same semantics as the sequential `_handle_delegation` path
        # and keeps error handling uniform with the regular tool
        # executor.
        results = await asyncio.gather(
            *(
                self._handle_delegation(
                    parent,
                    action,
                    parent_msg_id=parent_msg_id,
                    parallel_group_id=group_id,
                )
                for action in actions
            ),
        )

        batch_duration_ms = int((time.time() - batch_started_at) * 1000)

        self._emit(
            parent,
            EventType.PARALLEL_GROUP_COMPLETED,
            ParallelGroupCompletedPayload(
                group_id=group_id,
                duration_ms=batch_duration_ms,
            ),
            parallel_group_id=group_id,
        )

        return list(results)

    async def _execute_single_tool(
        self, frame: Frame, action: Action
    ) -> ToolResultContent | None:
        """Execute one non-delegation tool call, emit events, return result block.

        Two Pack-architecture dispatch hooks bracket the actual execution:

        - **Pre-hook** (`OnTool(phase=PRE)`): Gates run here. A `Block`
          effect short-circuits the call — TOOL_STARTED is never emitted, a
          synthetic TOOL_COMPLETED records the block reason, and the
          synthetic error block is returned. `ModifyToolInput` effects
          patch `tu.input` before execution.
        - **Post-hook** (`OnTool(phase=POST)`): Listeners run here, with
          PRIMARY → POST_EFFECT phase ordering. Effects can modify the
          result block or emit follow-up events before TOOL_COMPLETED lands.

        With no Pack interventions registered (the M2 default), both hooks
        return empty effect lists and the path is byte-equivalent to the
        pre-Pack manager.
        """
        tu = ToolUseContent(
            id=action.tool_use_id or generate_tool_use_id(),
            name=action.tool_name or "",
            input=dict(action.tool_input),
        )

        # ── Pack pre-hook ────────────────────────────────────────────
        pre_ctx = InterventionContext(
            session_id=frame.session_id,
            now=time.time(),
            registry=self._pack_registry,
            frame=frame,
            tool_call=ToolCallInfo(
                tool_name=tu.name,
                tool_use_id=tu.id,
                tool_input=dict(tu.input),
                phase=ToolPhase.PRE,
            ),
        )
        pre_effects = await self._pack_registry.dispatch_tool(
            ToolPhase.PRE, pre_ctx
        )
        for eff in pre_effects:
            if isinstance(eff, Block):
                # Short-circuit: synthesize an error result, skip execution.
                if eff.trace_event is not None:
                    # Best-effort trace event with an empty payload — Pack
                    # authors that need a richer payload should attach it
                    # via a separate EmitEvent effect.
                    self._emit(frame, eff.trace_event, _PayloadBase())
                self._emit(frame, EventType.TOOL_COMPLETED, ToolCompletedPayload(
                    tool_use_id=tu.id,
                    tool_name=tu.name,
                    output=f"Tool call blocked: {eff.reason}",
                    is_error=True,
                    duration_ms=0,
                ))
                return ToolResultContent(
                    tool_use_id=tu.id,
                    content=f"Tool call blocked: {eff.reason}",
                    is_error=True,
                )
            if isinstance(eff, ModifyToolInput):
                tu.input.update(eff.patch)

        self._emit(frame, EventType.TOOL_STARTED, ToolStartedPayload(
            tool_use_id=tu.id,
            tool_name=tu.name,
            tool_input=dict(tu.input),
        ))

        started_at = time.time()
        effective_tools = self._filter_allowed_tools(frame)
        tool_context = ToolContext(
            cwd=self._cwd,
            session_id=frame.session_id,
            agent_id=frame.id,
            pack_state=frame.pack_state,
        )

        result_messages = await execute_tools(
            tool_uses=[tu],
            tools=effective_tools,
            context=tool_context,
        )

        duration_ms = int((time.time() - started_at) * 1000)
        for result_msg in result_messages:
            for block in result_msg.content:
                if isinstance(block, ToolResultContent):
                    output_text = (
                        block.content
                        if isinstance(block.content, str)
                        else str(block.content)
                    )
                    is_error = bool(block.is_error)

                    # ── Pack post-hook ────────────────────────────
                    post_ctx = InterventionContext(
                        session_id=frame.session_id,
                        now=time.time(),
                        registry=self._pack_registry,
                        frame=frame,
                        tool_call=ToolCallInfo(
                            tool_name=tu.name,
                            tool_use_id=tu.id,
                            tool_input=dict(tu.input),
                            phase=ToolPhase.POST,
                            result_output=output_text,
                            result_is_error=is_error,
                        ),
                    )
                    post_effects = await self._pack_registry.dispatch_tool(
                        ToolPhase.POST, post_ctx
                    )
                    for eff in post_effects:
                        if isinstance(eff, ModifyToolResult):
                            if eff.output is not None:
                                output_text = eff.output
                            if eff.is_error is not None:
                                is_error = eff.is_error
                            if eff.append_hint:
                                output_text = (
                                    f"{output_text}\n\n{eff.append_hint}"
                                    if output_text
                                    else eff.append_hint
                                )
                        elif isinstance(eff, EmitEvent):
                            self._emit(frame, eff.event_type, eff.payload)
                        # Other effect types are intentionally no-ops in
                        # the tool-call dispatch site; they belong to
                        # other dispatch surfaces (compose, frame
                        # lifecycle, etc.) and will be wired in later
                        # phases as those surfaces gain hooks.

                    if is_error != bool(block.is_error) or output_text != (
                        block.content if isinstance(block.content, str) else str(block.content)
                    ):
                        block = ToolResultContent(
                            tool_use_id=block.tool_use_id,
                            content=output_text,
                            is_error=is_error,
                        )

                    # TodoWrite is the first built-in tool that mutates
                    # per-frame state. The tool itself stays pure (just
                    # validates + returns a confirmation); the framework
                    # is the one allowed to touch the event store, so
                    # we emit TODO_WRITTEN here on the same frame after
                    # a successful run. Order matters: the
                    # state-transition event lands BEFORE the
                    # TOOL_COMPLETED trace event, matching how
                    # message.appended lands before annotation.stored
                    # in the assistant-reply flow.
                    if (
                        not is_error
                        and tu.name == TODO_WRITE_TOOL_NAME
                    ):
                        self._emit_todo_written(frame, dict(tu.input))
                    self._emit(frame, EventType.TOOL_COMPLETED, ToolCompletedPayload(
                        tool_use_id=block.tool_use_id,
                        tool_name=tu.name,
                        output=output_text,
                        is_error=is_error,
                        duration_ms=duration_ms,
                    ))
                    return block
        return None

    def _emit_todo_written(self, frame: Frame, tool_input: dict) -> None:
        """Re-validate TodoWrite input and emit TODO_WRITTEN on the frame.

        Parsing twice (once in the tool, once here) is intentional:
        the tool's parse may have mutated/normalized things in ways we
        don't want to trust, and going through the Pydantic model
        again makes "what landed in the event log" identical to "what
        the tool would accept", regardless of how the LLM serialized
        the input.
        """
        try:
            params = TodoWriteInput.model_validate(tool_input)
        except Exception as exc:
            logger.warning(
                "TodoWrite input failed validation on emit (frame=%s): %s",
                frame.id, exc,
            )
            return
        self._emit(frame, EventType.TODO_WRITTEN, TodoWrittenPayload(
            todos=list(params.todos),
            source="todo_write_tool",
        ))
        # Mirror the reconstruct handler — keep the in-memory body in
        # sync so the next compose() sees the updated todos and the
        # footer rule pipeline routes to todo_continues / needs_in_progress
        # instead of falling through to synthesis_nudge.
        frame.context.body.todos = list(params.todos)

    async def _handle_delegation(
        self,
        parent: Frame,
        action: Action,
        *,
        parent_msg_id: str | None,
        parallel_group_id: str | None = None,
    ) -> tuple[ToolResultContent | None, str | None]:
        """Open a child frame for delegation, run it, wrap result as tool block.

        Returns `(result_block, child_frame_id)` so the caller can record
        the parent→child link on the surrounding tool_result message.

        `parallel_group_id` is threaded onto this delegation's TOOL_STARTED
        and TOOL_COMPLETED events when the caller is running the
        delegation as part of a parallel batch. It does NOT propagate
        into the child frame's own events — the child is its own
        frame_id and runs its own sub-sequence; fork validation
        doesn't need per-event tags because it already scans the
        (session-global, monotonic) event log for the bracket STARTED
        and COMPLETED ids and rejects everything strictly between them.
        """
        params = dict(action.tool_input)
        child_role_name = params.get("name")
        prompt = params.get("prompt", "")
        tool_use_id = action.tool_use_id or generate_tool_use_id()

        self._emit(
            parent,
            EventType.TOOL_STARTED,
            ToolStartedPayload(
                tool_use_id=tool_use_id,
                tool_name=AGENT_TOOL_NAME,
                tool_input=params,
            ),
            parallel_group_id=parallel_group_id,
        )

        # Validate the delegation target. Two structural failure modes
        # we now reject as is_error tool_results so the parent agent
        # sees its own mistake and self-corrects on the next turn:
        #   1. `name` missing entirely → caller forgot to pick a
        #      specialist. Used to default to "core" which silently
        #      created self-loops when the caller WAS core.
        #   2. `name` equals the parent's own role name → self-
        #      delegation, which is always a no-op. Observed in
        #      session `8ed7d997` where core delegated to "core" twice
        #      and produced fabricated output.
        parent_role_name = parent.context.header.role.name
        rejection_reason: str | None = None
        if not child_role_name:
            rejection_reason = (
                "Agent call rejected: required field `name` is missing. "
                "You must specify which specialist to delegate to (e.g., "
                "researcher / analyzer / implementer / reviewer / judge). "
                "Re-issue the call with an explicit `name` field."
            )
        elif child_role_name == parent_role_name:
            rejection_reason = (
                f"Agent call rejected: cannot delegate to '{child_role_name}' "
                f"from '{parent_role_name}' — that's self-delegation and is "
                f"always a no-op. Pick a different specialist (e.g., "
                f"researcher / analyzer / implementer / reviewer / judge) "
                f"that actually advances the task."
            )

        if rejection_reason is not None:
            self._emit(
                parent,
                EventType.TOOL_COMPLETED,
                ToolCompletedPayload(
                    tool_use_id=tool_use_id,
                    tool_name=AGENT_TOOL_NAME,
                    output=rejection_reason,
                    is_error=True,
                    duration_ms=0,
                ),
                parallel_group_id=parallel_group_id,
            )
            return (
                ToolResultContent(
                    tool_use_id=tool_use_id,
                    content=rejection_reason,
                    is_error=True,
                ),
                None,
            )

        child_role = self._resolve_child_role(child_role_name)
        started_at = time.time()

        # Per-agent model from role > parent's model. CLI --model override
        # flows in by being stamped on the root role at runner construction.
        child_model = child_role.model or parent.model

        child = self.open_child(
            parent=parent,
            role=child_role,
            initial_input=prompt,
            model=child_model,
            purpose=f"delegation:{child_role_name}",
            spawned_from_message_id=parent_msg_id,
            spawned_by_tool_use_id=tool_use_id,
        )
        await self.run(child)

        # ── Propagate ReadMemory child → parent ──────────────────────
        # Child's read_memory bubbles up. Child is already being closed,
        # so evict first (expire oversized content), then merge into
        # parent with depth+1.
        self._merge_pack_state_on_resolve(child, parent)

        bubble_text = self._extract_bubble_text(child)
        is_error = child.state == FrameState.ERROR
        self.close(child)

        duration_ms = int((time.time() - started_at) * 1000)
        self._emit(
            parent,
            EventType.TOOL_COMPLETED,
            ToolCompletedPayload(
                tool_use_id=tool_use_id,
                tool_name=AGENT_TOOL_NAME,
                output=bubble_text,
                is_error=is_error,
                duration_ms=duration_ms,
            ),
            parallel_group_id=parallel_group_id,
        )

        return (
            ToolResultContent(
                tool_use_id=tool_use_id,
                content=bubble_text or "(sub-agent produced no output)",
                is_error=is_error,
            ),
            child.id,
        )

    def _resolve_child_role(self, name: str) -> AgentRole:
        """Resolve a role name → AgentRole.

        When an injected `role_resolver` (preset-driven) is present and
        returns a non-None result, use it. Otherwise — or when the
        resolver returns None for an off-roster name — produce a
        minimal placeholder role so the frame can still open with
        enough context for the agent to report back that it wasn't
        configured. Teams that want richer fallbacks should list the
        agent in their preset.
        """
        if self._role_resolver is not None:
            role = self._role_resolver(name)
            if role is not None:
                return role

        return AgentRole(
            name=name,
            instructions=(
                f"You are the {name} agent. Complete the task you've "
                "been given and report back your findings."
            ),
            allowed_tools=None,
        )

    @staticmethod
    def _extract_bubble_text(child: Frame) -> str:
        """Return the child's last outgoing text message."""
        self_actor = child.self_actor
        for msg in reversed(child.context.body.conversation.messages):
            if msg.from_ == self_actor:
                return "".join(
                    block.text
                    for block in msg.content
                    if isinstance(block, TextContent)
                )
        return ""

    def _filter_allowed_tools(self, frame: Frame) -> list[Tool]:
        allowed = frame.context.header.role.allowed_tools
        if allowed is None:
            return list(self._tool_registry)
        allowed_set = set(allowed)
        return [t for t in self._tool_registry if t.name in allowed_set]

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _init_pack_state(self, frame: Frame) -> None:
        """Initialize pack_state for a fresh frame.

        Called from open_root / open_child BEFORE any events are emitted,
        so interventions that fire on FRAME_OPENED can already read state.
        """
        from nature.packs.builtin.file_state.pack import ensure_read_memory
        ensure_read_memory(frame)

    def _merge_pack_state_on_resolve(self, child: Frame, parent: Frame) -> None:
        """Merge child's Pack-owned state into parent on child resolve.

        Currently only merges read_memory. As more Packs gain
        frame-tree-propagating state, extend this method.
        Also evicts child's content to budget before merge so huge
        reads don't explode parent memory.
        """
        child_rm = child.pack_state.get("read_memory")
        parent_rm = parent.pack_state.get("read_memory")
        if child_rm is None or parent_rm is None:
            return
        child_rm.evict_to_budget()
        parent_rm.merge(child_rm, depth_increment=1)

    def _counterparty_for(self, frame: Frame) -> str:
        """The actor this frame replies to. Stamped on the Frame at open()."""
        return frame.counterparty

    def _last_outgoing_message(self, frame: Frame) -> Message | None:
        self_actor = frame.self_actor
        for msg in reversed(frame.context.body.conversation.messages):
            if msg.from_ == self_actor:
                return msg
        return None

    def _append_message(
        self,
        frame: Frame,
        msg: Message,
        *,
        delegations: dict[str, str] | None = None,
    ) -> None:
        frame.context.body.conversation.append(msg)
        # Single logical clock: the event's timestamp is the message's
        # timestamp. reconstruct() relies on this — Message.timestamp
        # from the payload must equal Event.timestamp.
        self._emit(
            frame,
            EventType.MESSAGE_APPENDED,
            MessageAppendedPayload(
                message_id=msg.id,
                from_=msg.from_,
                to=msg.to,
                content=list(msg.content),
                timestamp=msg.timestamp,
                delegations=dict(delegations or {}),
            ),
            timestamp=msg.timestamp,
        )

    # ------------------------------------------------------------------
    # Event emission helpers
    # ------------------------------------------------------------------

    #: Hard cap on nested `_emit` → OnEvent listener → EmitEvent → _emit
    #: recursion. Four levels is more than any observed Pack needs; past
    #: that we log and stop the cascade.
    _EMIT_MAX_DEPTH = 4

    async def _dispatch_llm(self, frame: Frame, phase: LLMPhase) -> None:
        """Fire OnLLM Listeners around the llm_agent call.

        `PRE` fires just before the provider request goes out, after
        body compaction and per-agent provider selection. `POST` fires
        after the response has been recorded via `_emit_llm_request` /
        `_emit_llm_response` but before new messages are appended to
        the frame conversation. Listeners see the frame as it stood
        heading into the LLM call.

        This hook is observational in the current wiring — Listeners
        that return `EmitEvent` have those emissions applied via
        `_emit`, but there's no effect type today for mutating the
        LLM request / response itself. A future `ModifyLLMRequest`
        effect can slot in here without changing the hook placement.
        """
        ctx = InterventionContext(
            session_id=frame.session_id,
            now=time.time(),
            registry=self._pack_registry,
            frame=frame,
        )
        effects = await self._pack_registry.dispatch_llm(phase, ctx)
        for eff in effects:
            if isinstance(eff, EmitEvent):
                self._emit(frame, eff.event_type, eff.payload)

    def _dispatch_frame_lifecycle(self, frame: Frame, phase: FramePhase) -> None:
        """Fire OnFrame Listeners for a lifecycle phase transition.

        Called at each of the five phase transitions (`OPENED` after
        open_root/open_child, `RESOLVED` after the run loop reaches
        RESOLVED, `ERRORED` on provider exceptions + ERROR-signalled
        turns, `CLOSED` on explicit close()). `EmitEvent` effects
        returned by Listeners flow through `_emit` so the same
        depth-guarded cascade applies. Other effect types are no-ops
        at this site — the lifecycle hook is strictly observational
        for now.
        """
        ctx = InterventionContext(
            session_id=frame.session_id,
            now=time.time(),
            registry=self._pack_registry,
            frame=frame,
        )
        effects = self._pack_registry.dispatch_frame_sync(phase, ctx)
        for eff in effects:
            if isinstance(eff, EmitEvent):
                self._emit(frame, eff.event_type, eff.payload)

    def _emit(
        self,
        frame: Frame,
        event_type: EventType,
        payload: _PayloadBase | dict,
        *,
        timestamp: float | None = None,
        parallel_group_id: str | None = None,
    ) -> None:
        """Append one event, then fire OnEvent Listeners synchronously.

        `timestamp` is optional so callers that already took a single
        `time.time()` snapshot (e.g. `_append_message`, which stamps both
        the Message and its event from the same reading) can pass it
        through. When omitted, a fresh reading is taken here.

        `parallel_group_id` tags the event as part of a parallel-
        execution batch. Set by `_execute_and_apply` when dispatching
        a concurrency-safe tool batch through `asyncio.gather`; every
        TOOL_STARTED / TOOL_COMPLETED inside the batch (plus the
        bracket PARALLEL_GROUP_* events) carries the same id so the
        fork API can reject at_event_ids that fall strictly inside
        the bracket.

        Listener dispatch policy:
        - Runs synchronously after the store write.
        - Only `EmitEvent` effects are applied (each recurses through
          `_emit`, bounded by `_EMIT_MAX_DEPTH`). Other effect types
          returned here are intentionally no-ops because this path has
          no tool/frame-mutation context to apply them to.
        - Listener exceptions are caught in the registry — one broken
          Listener cannot block further event emission.
        """
        self._store.append(
            Event(
                id=0,  # store assigns
                session_id=frame.session_id,
                frame_id=frame.id,
                timestamp=timestamp if timestamp is not None else time.time(),
                type=event_type,
                payload=dump_payload(payload),
                parallel_group_id=parallel_group_id,
            )
        )
        # Cascade guard. A Listener firing EmitEvent recurses through
        # _emit which re-enters this block; beyond MAX_DEPTH we bail
        # out to keep a misbehaving Pack from stalling the run loop.
        if self._emit_depth >= self._EMIT_MAX_DEPTH:
            logger.warning(
                "_emit: depth cap %d hit on %s — skipping OnEvent dispatch",
                self._EMIT_MAX_DEPTH, event_type,
            )
            return
        self._emit_depth += 1
        try:
            ctx = InterventionContext(
                session_id=frame.session_id,
                now=time.time(),
                registry=self._pack_registry,
                frame=frame,
            )
            effects = self._pack_registry.dispatch_event_sync(event_type, ctx)
            for eff in effects:
                if isinstance(eff, EmitEvent):
                    self._emit(frame, eff.event_type, eff.payload)
                # Other effect types are no-ops here (see docstring).
        finally:
            self._emit_depth -= 1

    def _emit_header_snapshot(self, frame: Frame) -> None:
        """Dump the frame's full header as a dedicated state event.

        Called immediately after FRAME_OPENED. The snapshot is the
        source of truth for role + principles during replay; incremental
        ROLE_CHANGED / PRINCIPLE_ADDED events apply on top of it.
        """
        header = frame.context.header
        self._emit(frame, EventType.HEADER_SNAPSHOT, HeaderSnapshotPayload(
            role=header.role,
            principles=list(header.principles),
        ))

    def _emit_llm_request(self, frame: Frame, output: AgentOutput) -> None:
        req = output.raw_request
        if req is None:
            return
        self._emit(frame, EventType.LLM_REQUEST, LLMRequestPayload(
            request_id=req.request_id,
            model=req.model,
            message_count=req.message_count,
            tool_count=len(req.tools) if req.tools else 0,
        ))
        # Footer hints fired by the composer's rule pipeline. Trace-only
        # — the body wasn't mutated, this is purely so the dashboard
        # (and any future debugger) can see "the framework whispered
        # something into this LLM call".
        if output.hints:
            self._emit(frame, EventType.HINT_INJECTED, HintInjectedPayload(
                request_id=req.request_id,
                hints=[
                    {"source": h.source, "text": h.text}
                    for h in output.hints
                ],
            ))

    def _emit_llm_response(self, frame: Frame, output: AgentOutput) -> None:
        req_id = output.raw_request.request_id if output.raw_request else None
        self._emit(frame, EventType.LLM_RESPONSE, LLMResponsePayload(
            request_id=req_id,
            stop_reason=output.stop_reason,
            usage=output.usage.model_dump() if output.usage else None,
        ))

    def _emit_llm_error(self, frame: Frame, exc: Exception) -> None:
        self._emit(frame, EventType.LLM_ERROR, LLMErrorPayload(
            request_id=None,
            error_type=type(exc).__name__,
            message=str(exc)[:500],
        ))

    def _emit_frame_errored(self, frame: Frame, exc: Exception) -> None:
        """State-transition counterpart of LLM_ERROR.

        LLM_ERROR is trace (what went wrong at the provider level);
        FRAME_ERRORED is what reconstruct() watches. Emitting both
        keeps the replay contract clean: trace events can be stripped
        without losing the terminal state.
        """
        self._emit(frame, EventType.FRAME_ERRORED, FrameErroredPayload(
            error_type=type(exc).__name__,
            message=str(exc)[:500],
        ))

    def _emit_annotation(
        self, frame: Frame, ann: MessageAnnotation
    ) -> None:
        self._emit(frame, EventType.ANNOTATION_STORED, AnnotationStoredPayload(
            message_id=ann.message_id,
            thinking=ann.thinking,
            usage=ann.usage.model_dump() if ann.usage else None,
            stop_reason=ann.stop_reason,
            llm_request_id=ann.llm_request_id,
            duration_ms=ann.duration_ms,
        ))
