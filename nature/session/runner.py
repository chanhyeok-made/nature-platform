"""SessionRunner — the entry point for the new Frame + Event architecture.

This is the single place where execution is kicked off. It owns an
AreaManager and exposes the minimal public surface needed by CLI
drivers: `run` to start a new session, `continue_session` to add a
follow-up user input to an existing frame, `close` to dispose.

SessionRunner is deliberately UI-agnostic. It writes events to the
store and returns the live Frame for anyone who wants to introspect
it (tests, integration drivers). UIs consume from the store separately
via `nature.ui.event_consumer.EventConsumer`.
"""

from __future__ import annotations

from dataclasses import dataclass

from nature.context.body_compaction import BodyCompactionPipeline
from nature.context.types import AgentRole
from nature.events.store import EventStore
from nature.frame.frame import Frame, FrameState
from nature.frame.manager import (
    AreaManager,
    MaxOutputResolver,
    ProviderResolver,
    RoleResolver,
)
from nature.protocols.provider import LLMProvider
from nature.protocols.tool import Tool


@dataclass
class RunResult:
    """Outcome of running a session to a pause point or resolution."""

    frame: Frame
    session_id: str

    @property
    def is_resolved(self) -> bool:
        return self.frame.state == FrameState.RESOLVED

    @property
    def is_awaiting_user(self) -> bool:
        return self.frame.state == FrameState.AWAITING_USER


class SessionRunner:
    """Drives a single conversation session end-to-end.

    One runner instance ↔ one AreaManager ↔ many sessions. Each session
    gets its own root Frame identified by `session_id`.
    """

    def __init__(
        self,
        *,
        provider: LLMProvider,
        tool_registry: list[Tool],
        event_store: EventStore,
        cwd: str | None = None,
        role_resolver: RoleResolver | None = None,
        compaction_pipeline: BodyCompactionPipeline | None = None,
        provider_resolver: ProviderResolver | None = None,
        max_output_resolver: MaxOutputResolver | None = None,
    ) -> None:
        self._provider = provider
        self._tool_registry = tool_registry
        self._event_store = event_store
        self._manager = AreaManager(
            store=event_store,
            provider=provider,
            tool_registry=tool_registry,
            cwd=cwd,
            role_resolver=role_resolver,
            compaction_pipeline=compaction_pipeline,
            provider_resolver=provider_resolver,
            max_output_resolver=max_output_resolver,
        )

    @property
    def manager(self) -> AreaManager:
        return self._manager

    @property
    def store(self) -> EventStore:
        return self._event_store

    async def run(
        self,
        *,
        session_id: str,
        role: AgentRole,
        model: str,
        user_input: str,
    ) -> RunResult:
        """Open a fresh root frame and drive it to pause or resolution."""
        frame = self._manager.open_root(
            session_id=session_id,
            role=role,
            model=model,
            initial_user_input=user_input,
        )
        await self._manager.run(frame)
        return RunResult(frame=frame, session_id=session_id)

    async def continue_session(
        self,
        *,
        frame: Frame,
        user_input: str,
    ) -> RunResult:
        """Resume an existing frame with new user input."""
        self._manager.append_user_input(frame, user_input)
        await self._manager.run(frame)
        return RunResult(frame=frame, session_id=frame.session_id)

    def close(self, frame: Frame) -> None:
        """Close a frame and emit frame.closed to the store."""
        self._manager.close(frame)
