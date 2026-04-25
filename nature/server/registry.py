"""SessionRegistry — server-side ownership of live AreaManagers + Frames.

One ServerSession per nature session. Each holds its own provider,
SessionRunner, root role, and (after the first message) the running
Frame. Background asyncio tasks drive llm_agent calls; the registry
doesn't block on them, so the HTTP layer stays responsive.

This module is the single place that knows how to:
- load frame.json
- build a provider with CLI/config/env precedence
- assemble the tool registry
- resolve the root role from config or md profiles
- spawn / cancel run tasks per session

The HTTP route handlers (server/app.py) just call into the registry.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from nature.agents.config import (
    AgentsRegistry,
    load_agent_instruction,
    load_agents_registry,
)
from nature.agents.presets import PresetConfig, load_preset, validate_preset
from nature.config.hosts import HostConfig, HostsConfig, load_hosts_config
from nature.config.models import load_model_specs, resolve_budget
from nature.context.body_compaction import (
    BodyCompactionPipeline,
    DreamerBodyStrategy,
    MicrocompactBodyStrategy,
)
from nature.context.types import AgentRole
from nature.events.store import EventStore, SessionMeta
from nature.frame.agent_tool import AgentTool
from nature.frame.frame import Frame
from nature.protocols.provider import LLMProvider, ProviderConfig
from nature.protocols.tool import Tool
from nature.server.api import CreateSessionRequest
from nature.session.runner import SessionRunner
from nature.utils.ids import generate_session_id

logger = logging.getLogger(__name__)

PREVIEW_MAX_CHARS = 80


def _truncate_preview(text: str) -> str:
    """Single-line, ~80-char preview of a user message."""
    if not text:
        return ""
    flat = " ".join(text.split())  # collapse all whitespace runs to one space
    if len(flat) <= PREVIEW_MAX_CHARS:
        return flat
    return flat[: PREVIEW_MAX_CHARS - 1] + "…"


def archived_preview_from_events(events_iter) -> str:
    """Extract the first user message text from an event iterable.

    Returns "" if no user message is found. Used by the archived-session
    listing path so we don't have to keep the original Frame in memory.
    """
    for ev in events_iter:
        if ev.type.value != "message.appended":
            continue
        payload = ev.payload or {}
        if payload.get("from_") != "user":
            continue
        for block in payload.get("content", []):
            if block.get("type") == "text" and block.get("text"):
                return _truncate_preview(block["text"])
    return ""


@dataclass
class ServerSession:
    """A live session held by the server process."""

    session_id: str
    root_role: AgentRole
    root_model: str
    provider_name: str
    base_url: str | None
    provider: LLMProvider
    runner: SessionRunner
    created_at: float
    frame: Frame | None = None
    run_task: asyncio.Task | None = None
    closed: bool = False

    # --- New preset-based fields (Stage A additive skeleton) ---------
    # Populated only when the session was created through the preset
    # flow (`CreateSessionRequest.preset` set). Legacy sessions created
    # via the deprecated override path leave these as None and keep
    # using the single `provider` above for every LLM call.
    #
    # - `preset`           the resolved PresetConfig naming root_agent,
    #                      roster, and per-agent model overrides.
    # - `agents_registry`  the loaded AgentsRegistry snapshot used to
    #                      build role objects for sub-agent spawns.
    # - `provider_pool`    host_name → LLMProvider, one per distinct
    #                      host referenced by the preset's effective
    #                      model set. AreaManager picks the right
    #                      provider per agent at LLM call time.
    preset: PresetConfig | None = None
    agents_registry: AgentsRegistry | None = None
    provider_pool: dict[str, LLMProvider] | None = None

    @property
    def state(self) -> str:
        if self.closed:
            return "closed"
        if self.frame is None:
            return "active"
        return self.frame.state.value if hasattr(self.frame.state, "value") else str(self.frame.state)

    @property
    def has_active_run(self) -> bool:
        return self.run_task is not None and not self.run_task.done()

    @property
    def preview(self) -> str:
        """First user message text from the live conversation, truncated."""
        if self.frame is None:
            return ""
        from nature.context.conversation import Message
        from nature.protocols.message import TextContent
        for msg in self.frame.context.body.conversation.messages:
            if msg.from_ != "user":
                continue
            for block in msg.content:
                if isinstance(block, TextContent) and block.text:
                    return _truncate_preview(block.text)
        return ""


class SessionRegistry:
    """Owns all live sessions inside the server process."""

    def __init__(
        self,
        *,
        event_store: EventStore,
        cwd: str | None = None,
    ) -> None:
        self._store = event_store
        self._cwd = cwd or os.getcwd()
        self._sessions: dict[str, ServerSession] = {}

    @property
    def event_store(self) -> EventStore:
        return self._store

    @property
    def cwd(self) -> str:
        return self._cwd

    def list_available_tool_names(self) -> list[str]:
        """Tool names available to sessions on this server.

        Read straight off the default tool registry — the live
        tool_registry a runner gets is a fresh list built by
        `_build_tool_registry`, so this matches what a new session
        would see when compiling its role's allowed_tools filter.
        """
        tools = self._build_tool_registry()
        return sorted(t.name for t in tools)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, session_id: str) -> ServerSession | None:
        return self._sessions.get(session_id)

    def list(self) -> list[ServerSession]:
        return list(self._sessions.values())

    def get_frame_context(
        self,
        session_id: str,
        frame_id: str,
        *,
        up_to_event_id: int | None = None,
    ) -> dict | None:
        """Return the Context (header + body) for a frame, by replaying
        the session's events.

        Works for both live root frames and closed child frames — the
        rebuilt Frame from reconstruct() carries the full conversation
        history accumulated in the event log, regardless of whether the
        original Frame object is still in memory.

        `up_to_event_id` slices the replay at a historical event id.
        Dashboards use this for time-travel scrubbers: "show the frame
        as it looked right before event N fired".
        """
        from nature.events.reconstruct import reconstruct

        result = reconstruct(session_id, self._store, up_to_event_id=up_to_event_id)
        frame = result.frames.get(frame_id)
        if frame is None:
            return None

        ctx = frame.context
        return {
            "frame_id": frame.id,
            "session_id": frame.session_id,
            "purpose": frame.purpose,
            "model": frame.model,
            "parent_id": frame.parent_id,
            "state": frame.state.value if hasattr(frame.state, "value") else str(frame.state),
            "header": {
                "role": ctx.header.role.model_dump(),
                "principles": [p.model_dump() for p in ctx.header.principles],
            },
            "body": {
                "conversation": {
                    "messages": [m.model_dump() for m in ctx.body.conversation.messages],
                },
            },
        }

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def _write_session_started(
        self,
        session: ServerSession,
        preset: PresetConfig,
        agents: AgentsRegistry,
        *,
        parent_session_id: str | None = None,
        forked_from_event_id: int | None = None,
    ) -> None:
        """Emit SESSION_STARTED to the new session's event stream.

        Captures the preset composition + every roster agent's resolved
        definition (model, tools, instructions text) + the model
        budgets in use, so the event log alone is a provenance record.
        For forked sessions the parent/event lineage lands in the same
        payload, making the fork boundary visible without consulting
        the store's sidecar metadata.
        """
        from nature.events.payloads import SessionStartedPayload
        from nature.events.types import Event, EventType

        agents_resolved: dict[str, dict[str, object]] = {}
        hosts_used: set[str] = set()
        models_used: set[str] = set()
        for name in preset.agents:
            agent_cfg = agents.get(name)
            if agent_cfg is None:
                continue
            model_ref = preset.model_overrides.get(name, agent_cfg.model)
            prompt_stem = preset.prompt_overrides.get(name)
            if prompt_stem is not None:
                try:
                    instr_text = load_agent_instruction(
                        prompt_stem, project_dir=self._cwd,
                    )
                except Exception:
                    instr_text = agent_cfg.instructions_text
            else:
                instr_text = agent_cfg.instructions_text
            agents_resolved[name] = {
                "model": model_ref,
                "allowed_tools": agent_cfg.allowed_tools,
                "allowed_interventions": agent_cfg.allowed_interventions,
                "instructions": instr_text,
                "description": agent_cfg.description,
            }
            host_name = model_ref.split("::", 1)[0]
            hosts_used.add(host_name)
            models_used.add(model_ref)

        specs = load_model_specs(project_dir=self._cwd)
        budgets: dict[str, dict[str, int]] = {}
        for model_ref in models_used:
            b = resolve_budget(model_ref, specs)
            budgets[model_ref] = {
                "context_window": b.context_window,
                "output_reservation": b.output_reservation,
            }

        repo_git_sha = ""
        try:
            import subprocess
            r = subprocess.run(
                ["git", "-C", str(self._cwd), "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                repo_git_sha = r.stdout.strip()
        except Exception:  # noqa: BLE001
            repo_git_sha = ""

        payload = SessionStartedPayload(
            preset_name=preset.name,
            preset={
                "root_agent": preset.root_agent,
                "agents": list(preset.agents),
                "model_overrides": dict(preset.model_overrides),
                "prompt_overrides": dict(preset.prompt_overrides),
            },
            agents_resolved=agents_resolved,
            hosts_used=sorted(hosts_used),
            models_used=sorted(models_used),
            model_budgets=budgets,
            parent_session_id=parent_session_id,
            forked_from_event_id=forked_from_event_id,
            repo_git_sha=repo_git_sha,
        )

        event = Event(
            id=0,
            session_id=session.session_id,
            frame_id=None,
            timestamp=time.time(),
            type=EventType.SESSION_STARTED,
            payload=payload.model_dump(),
        )
        self._store.append(event)

    async def create_session(self, req: CreateSessionRequest) -> ServerSession:
        """Build a fresh preset-driven session.

        Resolves `req.preset` (or `"default"` when None), loads the
        agents + hosts registries, runs validate_preset, builds one
        LLMProvider per distinct host referenced by the preset, and
        constructs the ServerSession with its preset/agents_registry/
        provider_pool fields populated.
        """
        preset, agents, hosts = self._load_preset_bundle(req.preset)
        session = self._assemble_preset_session(preset, agents, hosts)
        # Fresh session: emit the provenance snapshot at event #1 so the
        # event log is self-describing without needing the filesystem.
        self._write_session_started(session, preset, agents)
        return session

    def _load_preset_bundle(
        self, preset_name: str | None,
    ) -> tuple[PresetConfig, AgentsRegistry, HostsConfig]:
        """Resolve (preset, agents, hosts) for a session.

        `preset_name` of None falls back to `"default"` so clients that
        pass an empty CreateSessionRequest still get a working session
        when a default.json preset is on disk. `validate_preset` runs
        before returning so every downstream ref is known-good.
        """
        effective_name = preset_name or "default"
        preset = load_preset(effective_name, project_dir=self._cwd)
        agents = load_agents_registry(project_dir=self._cwd)
        hosts = load_hosts_config(project_dir=self._cwd)
        # Ready-moment validation: raises PresetValidationError on any
        # dangling reference (unknown agent, unknown host, missing env
        # API key). Bubbled up to the HTTP layer as a 400.
        validate_preset(preset, agents, hosts, project_dir=self._cwd)
        return preset, agents, hosts

    def _assemble_preset_session(
        self,
        preset: PresetConfig,
        agents: AgentsRegistry,
        hosts: HostsConfig,
        *,
        session_id: str | None = None,
        frame: Frame | None = None,
        created_at: float | None = None,
    ) -> ServerSession:
        """Construct the ServerSession wiring from a validated preset.

        Shared by `create_session` (fresh session) and `resume_session`
        (replayed session — `frame` + `session_id` are supplied). The
        resume path stamps the preset's root model onto the replayed
        frame so a post-resume LLM call reflects the current preset
        rather than the historical model.
        """
        provider_pool = self._build_provider_pool(preset, agents, hosts)

        root_agent_cfg = agents.get(preset.root_agent)
        assert root_agent_cfg is not None  # checked by validate_preset
        root_override = preset.model_overrides.get(preset.root_agent)
        root_prompt_stem = preset.prompt_overrides.get(preset.root_agent)
        root_prompt_text = (
            load_agent_instruction(root_prompt_stem, project_dir=self._cwd)
            if root_prompt_stem is not None else None
        )
        root_role = root_agent_cfg.to_role(
            model_override=root_override,
            instructions_override=root_prompt_text,
        )
        root_model_ref = root_override or root_agent_cfg.model
        root_host_name, root_bare_model = root_model_ref.split("::", 1)
        root_host = hosts.get_host(root_host_name)
        assert root_host is not None  # checked by validate_preset
        root_provider = provider_pool[root_host_name]

        provider_resolver = self._make_provider_resolver(
            preset, agents, provider_pool,
        )
        role_resolver = self._make_preset_role_resolver(preset, agents)
        max_output_resolver = self._make_max_output_resolver(preset)

        sid = session_id or generate_session_id()
        compaction_pipeline = self._build_compaction_pipeline(
            root_model_ref=root_model_ref, session_id=sid,
        )

        tools = self._build_tool_registry()
        runner = SessionRunner(
            provider=root_provider,
            tool_registry=tools,
            event_store=self._store,
            cwd=self._cwd,
            role_resolver=role_resolver,
            provider_resolver=provider_resolver,
            compaction_pipeline=compaction_pipeline,
            max_output_resolver=max_output_resolver,
        )

        if frame is not None:
            frame.model = root_bare_model

        session = ServerSession(
            session_id=sid,
            root_role=root_role,
            root_model=root_bare_model,
            provider_name=root_host.provider,
            base_url=root_host.base_url,
            provider=root_provider,
            runner=runner,
            created_at=created_at if created_at is not None else time.time(),
            frame=frame,
            preset=preset,
            agents_registry=agents,
            provider_pool=provider_pool,
        )
        self._sessions[session.session_id] = session
        logger.info(
            "session %s %s via preset %r "
            "(root=%s, root_model=%s::%s, hosts=%s)",
            session.session_id,
            "resumed" if frame is not None else "created",
            preset.name, root_role.name,
            root_host_name, root_bare_model, sorted(provider_pool.keys()),
        )
        return session

    # ------------------------------------------------------------------
    # Run / cancel
    # ------------------------------------------------------------------

    async def send_message(self, session_id: str, text: str) -> None:
        """Queue a user input on the session's run task.

        Spawns a background task so HTTP returns immediately. Events flow
        into the EventStore as the run progresses; clients see them via
        the WebSocket subscription.
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        if session.closed:
            raise RuntimeError(f"session {session_id} is closed")
        if session.has_active_run:
            raise RuntimeError(f"session {session_id} is busy — current run still in progress")

        session.run_task = asyncio.create_task(self._drive_run(session, text))

    async def cancel(self, session_id: str) -> None:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        if session.run_task and not session.run_task.done():
            session.run_task.cancel()

    async def close_session(self, session_id: str) -> None:
        """Cancel any in-flight run, mark closed, drop provider client."""
        session = self._sessions.pop(session_id, None)
        if session is None:
            return
        if session.run_task and not session.run_task.done():
            session.run_task.cancel()
            try:
                await session.run_task
            except (asyncio.CancelledError, Exception):
                pass
        session.closed = True
        try:
            await session.provider.close()
        except Exception as exc:
            logger.debug("provider close failed for session %s: %s", session_id, exc)

    async def close_all(self) -> None:
        """Server shutdown — close every live session."""
        for session_id in list(self._sessions.keys()):
            await self.close_session(session_id)

    # ------------------------------------------------------------------
    # Resume / archived sessions
    # ------------------------------------------------------------------

    def list_archived(self) -> list[SessionMeta]:
        """Sessions on disk that are NOT currently in the live registry."""
        live_ids = set(self._sessions.keys())
        return [
            meta for meta in self._store.list_sessions()
            if meta.session_id not in live_ids
        ]

    async def resume_session(
        self,
        session_id: str,
        req: CreateSessionRequest | None = None,
    ) -> ServerSession:
        """Resume a session — reattach to a live one, or rebuild from the
        event log on disk if not currently in the registry.

        The replayed conversation history + header role come from the
        event log as-is. `req.preset` (optional) names the preset that
        drives the resumed run's provider pool and per-agent routing;
        leaving it None falls through to `default.json`. Re-built
        sessions are explicitly reopened via `AreaManager.reopen()`,
        which emits FRAME_REOPENED so the state transition is visible
        in the event log.
        """
        existing = self._sessions.get(session_id)
        if existing is not None:
            return existing

        # Reconstruct frame tree from event log.
        from nature.events.reconstruct import reconstruct
        result = reconstruct(session_id, self._store)
        if not result.frames:
            raise KeyError(session_id)
        root_frames = [f for f in result.frames.values() if f.is_root]
        if not root_frames:
            raise KeyError(session_id)
        root_frame = root_frames[0]

        preset_name = req.preset if req is not None else None
        preset, agents, hosts = self._load_preset_bundle(preset_name)
        session = self._assemble_preset_session(
            preset, agents, hosts,
            session_id=session_id,
            frame=root_frame,
        )
        # Reopen for input regardless of previous terminal state. The
        # manager emits FRAME_REOPENED so replay sees the transition
        # explicitly instead of inferring it from the next message.
        session.runner.manager.reopen(root_frame)
        return session

    # ------------------------------------------------------------------
    # Fork (event-level branching)
    # ------------------------------------------------------------------

    async def fork_session(
        self,
        source_session_id: str,
        *,
        at_event_id: int,
        preset: str | None = None,
    ) -> ServerSession:
        """Create a new session by branching off `source_session_id`
        at `at_event_id`.

        Events 1..at_event_id are copied into a fresh session file
        (with `session_id` rewritten and original event ids preserved),
        a lineage sidecar is written, and the new session is hydrated
        live via the same `reopen()` path that `resume_session` uses.

        `preset` is the hook for event-pinned counterfactual experiments
        (see paper §4.2): the forked branch may continue under a
        different preset than the source ran with, attributing any
        post-fork delta cleanly to that configuration change. When
        `preset` is None the forked session resumes under `default.json`
        — the same fallback `resume_session` applies.

        Works uniformly for live and archived source sessions — as
        long as the source's event log exists on disk, the fork will
        read from it regardless of whether the source is currently in
        the in-memory registry.
        """
        new_sid = generate_session_id()

        # Copy events + write sidecar. `EventStore.fork` raises KeyError
        # if the source doesn't exist and ValueError for id range or
        # collision issues — let those bubble up to the HTTP layer.
        copied = self._store.fork(
            source_session_id,
            at_event_id=at_event_id,
            new_session_id=new_sid,
        )
        logger.info(
            "session %s forked from %s at event %d "
            "(%d events copied, preset=%s)",
            new_sid, source_session_id, at_event_id, copied,
            preset or "default",
        )

        # Resume the forked session through the same path as a normal
        # archived-session resume. When a caller supplied `preset`,
        # that preset drives the runner's provider pool / role
        # resolver; otherwise `default.json` applies via the shared
        # preset bundle loader inside `resume_session`.
        req = CreateSessionRequest(preset=preset) if preset else None
        session = await self.resume_session(new_sid, req)

        # Mark the fork boundary in the forked session's event stream:
        # appended after the copied prefix so the next reader sees
        # "from event `copied + 1` onward, this preset is active".
        # The payload carries the preset (possibly overridden) plus
        # the parent/event lineage that otherwise only lived in the
        # store's sidecar metadata.
        fork_preset, fork_agents, _ = self._load_preset_bundle(preset)
        self._write_session_started(
            session, fork_preset, fork_agents,
            parent_session_id=source_session_id,
            forked_from_event_id=at_event_id,
        )
        return session

    async def _drive_run(self, session: ServerSession, text: str) -> None:
        try:
            if session.frame is None:
                result = await session.runner.run(
                    session_id=session.session_id,
                    role=session.root_role,
                    model=session.root_model,
                    user_input=text,
                )
                session.frame = result.frame
            else:
                await session.runner.continue_session(
                    frame=session.frame,
                    user_input=text,
                )
        except asyncio.CancelledError:
            logger.info("session %s run cancelled", session.session_id)
            raise
        except Exception:
            logger.exception("session %s run failed", session.session_id)

    # ------------------------------------------------------------------
    # Internal: provider / tools / role construction
    # ------------------------------------------------------------------

    def _make_preset_role_resolver(
        self,
        preset: PresetConfig,
        agents: AgentsRegistry,
    ) -> Callable[[str], AgentRole | None]:
        """Build the agent_name → AgentRole callback for preset mode.

        For roster members: returns the AgentRole built from
        `agents_registry` with the preset's per-agent model override
        and prompt override applied (or the agent's defaults when
        neither override is set).

        For off-roster names: returns None. AreaManager then falls
        back to a minimal placeholder role — presets should list
        every agent they want the session to delegate to.
        """
        cwd = self._cwd

        def resolve(name: str) -> AgentRole | None:
            if name not in preset.agents:
                return None
            agent_cfg = agents.get(name)
            if agent_cfg is None:
                return None
            override = preset.model_overrides.get(name)
            prompt_stem = preset.prompt_overrides.get(name)
            prompt_text = (
                load_agent_instruction(prompt_stem, project_dir=cwd)
                if prompt_stem is not None else None
            )
            return agent_cfg.to_role(
                model_override=override,
                instructions_override=prompt_text,
            )
        return resolve

    @staticmethod
    def _make_max_output_resolver(
        preset: PresetConfig,
    ) -> Callable[[str], int | None]:
        """Build the agent_name → max_output_tokens callback.

        Reads from the preset's `max_output_tokens_overrides` map only;
        the model's physical ceiling is applied separately inside the
        provider, so this resolver carries just the role-level policy
        budget. Missing entry returns None — the provider then falls
        back to `ProviderConfig.max_output_tokens` (currently 8192).
        """
        budgets = dict(preset.max_output_tokens_overrides)

        def resolve(agent_name: str) -> int | None:
            return budgets.get(agent_name)

        return resolve

    @staticmethod
    def _make_provider_resolver(
        preset: PresetConfig,
        agents: AgentsRegistry,
        pool: dict[str, LLMProvider],
    ) -> Callable[[str], LLMProvider | None]:
        """Build the agent_name → LLMProvider callback for AreaManager.

        Closes over preset + agents + pool so AreaManager can stay
        agnostic to the preset schema. Returns None for agents that
        aren't in the preset roster — AreaManager then falls back to
        the root provider rather than failing, which preserves the
        legacy-resolver path for sub-agents spawned outside the
        preset (until Stage A.4 tightens delegation).
        """
        def resolve(agent_name: str) -> LLMProvider | None:
            if agent_name not in preset.agents:
                return None
            agent_cfg = agents.get(agent_name)
            if agent_cfg is None:
                return None
            model_ref = preset.model_overrides.get(agent_name, agent_cfg.model)
            host_name, _ = model_ref.split("::", 1)
            return pool.get(host_name)
        return resolve

    def _build_provider_pool(
        self,
        preset: PresetConfig,
        agents: AgentsRegistry,
        hosts: HostsConfig,
    ) -> dict[str, LLMProvider]:
        """One LLMProvider per distinct host referenced by the preset.

        Walks the preset's effective models (agent default or
        per-agent override), collects distinct host names, and
        instantiates one provider per host. Each provider is
        constructed with a bootstrap model (the first agent-on-that-
        host's bare model) — the per-call model is overridden at
        stream-time through `LLMRequest.model`, so the bootstrap only
        matters for providers that read it as a fallback.

        Callers must have already run `validate_preset` so every
        referenced host exists and has an API key when needed.
        """
        # First pass — group bare model refs by host so the provider
        # builder can check capability flags for every model it will
        # be asked to serve, not just the bootstrap.
        models_by_host: dict[str, list[str]] = {}
        bootstraps: dict[str, str] = {}
        for agent_name in preset.agents:
            agent_cfg = agents.get(agent_name)
            assert agent_cfg is not None  # validate_preset precondition
            model_ref = preset.model_overrides.get(agent_name, agent_cfg.model)
            host_name, bare_model = model_ref.split("::", 1)
            models_by_host.setdefault(host_name, []).append(bare_model)
            bootstraps.setdefault(host_name, bare_model)

        pool: dict[str, LLMProvider] = {}
        for host_name, models in models_by_host.items():
            host_cfg = hosts.get_host(host_name)
            assert host_cfg is not None  # validate_preset precondition
            pool[host_name] = self._build_provider_for_host(
                host_cfg,
                bootstrap_model=bootstraps[host_name],
                host_name=host_name,
                served_models=models,
            )
        return pool

    def _build_provider_for_host(
        self,
        host: HostConfig,
        *,
        bootstrap_model: str,
        host_name: str = "",
        served_models: list[str] | None = None,
    ) -> LLMProvider:
        """Instantiate the right provider class for a host.

        - provider=="anthropic"  → AnthropicProvider
        - otherwise              → OpenAICompatProvider (covers openai,
                                   local-ollama, openrouter, groq,
                                   together — everything that speaks
                                   the OpenAI-compatible API).

        `bootstrap_model` is the model the ProviderConfig is initialized
        with. It's just a default; real per-call models arrive via
        `LLMRequest.model` at stream time.

        When `served_models` includes any model flagged
        `text_tool_adaptation=True` in `model_capabilities`, wrap the
        inner provider with `TextToolAdapterProvider` so no-native-tools
        models (phi4, gemma2, coder tunes, etc.) don't hard-fail with
        an Ollama 400. The wrapper is transparent for capable models.
        """
        api_key = host.resolved_api_key() or ""
        config = ProviderConfig(
            model=bootstrap_model,
            api_key=api_key,
            base_url=host.base_url,
            host_name=host_name,
        )
        if host.provider == "anthropic":
            from nature.providers.anthropic import AnthropicProvider
            return AnthropicProvider(config)
        from nature.providers.openai_compat import OpenAICompatProvider
        provider: LLMProvider = OpenAICompatProvider(config)

        if served_models and host_name:
            from nature.providers import model_capabilities as _mc
            wants_wrap = any(
                _mc.lookup(f"{host_name}::{m}").text_tool_adaptation
                for m in served_models
            )
            if wants_wrap:
                from nature.providers.text_tool_wrapper import (
                    TextToolAdapterProvider,
                )
                provider = TextToolAdapterProvider(provider)
        return provider

    def _build_tool_registry(self) -> list[Tool]:
        from nature.tools.registry import get_default_tools
        tools = get_default_tools()
        tools.append(AgentTool())
        return tools

    def _build_compaction_pipeline(
        self, *, root_model_ref: str, session_id: str,
    ) -> BodyCompactionPipeline:
        """Assemble the body-compaction pipeline for this session.

        Strategies fire in priority order until the body estimate
        drops below the budget's autocompact threshold:
        1. `MicrocompactBodyStrategy` — clears old tool_result blocks
           (zero-cost; lossy but cheap).
        2. `DreamerBodyStrategy` — LLM-summarizes the prefix, writes
           the raw slice to `<cwd>/.nature/ltm/<session_id>/…` so the
           agent can Read it back if it needs detail.

        Budget is resolved from `models.json` against the root
        agent's model ref. Sub-frames share this budget — a known
        limitation when the root and a delegate use very different
        context windows; the pipeline still fires per frame, just
        against the same threshold.
        """
        specs = load_model_specs(project_dir=self._cwd)
        budget = resolve_budget(root_model_ref, specs)
        ltm_dir = Path(self._cwd) / ".nature" / "ltm"
        strategies = [
            MicrocompactBodyStrategy(preserve_turns=4),
            DreamerBodyStrategy(
                preserve_recent_turns=6,
                session_id=session_id,
                ltm_dir=ltm_dir,
            ),
        ]
        return BodyCompactionPipeline(strategies=strategies, budget=budget)

