"""HTTP API request/response models for the nature server.

Shared between server (server/app.py route handlers) and client
(nature/client/http_client.py) so the wire format is single-source.
All models are pure pydantic — no business logic here.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Session creation
# ---------------------------------------------------------------------------


class CreateSessionRequest(BaseModel):
    """Session creation inputs.

    `preset` names a preset file under
    `.nature/presets/<name>.json` (project-local, wins) or
    `~/.nature/presets/<name>.json` (user). The preset declares
    root_agent + agents roster + optional per-agent model overrides,
    all wiring through the Host registry.

    When `preset` is None, the server looks up `default.json` from the
    same project/user directories (raising 404 if neither file exists).
    """

    preset: str | None = None


class CreateSessionResponse(BaseModel):
    session_id: str
    root_role_name: str
    root_model: str
    provider_name: str
    base_url: str | None = None
    created_at: float
    parent_session_id: str | None = None
    forked_from_event_id: int | None = None


# ---------------------------------------------------------------------------
# Session forking (event-level branching)
# ---------------------------------------------------------------------------


class ModelCatalogEntry(BaseModel):
    """One entry in the curated model dropdown for the settings UI."""

    provider: str  # "anthropic" | "openai" | ...
    id: str  # the model id to send to the provider
    label: str  # human-readable display name
    tier: str = "medium"  # "heavy" | "medium" | "light"


class ListModelsResponse(BaseModel):
    models: list[ModelCatalogEntry] = Field(default_factory=list)


class ListToolsResponse(BaseModel):
    tools: list[str] = Field(default_factory=list)


class ForkSessionRequest(BaseModel):
    """Body for `POST /api/sessions/{source_sid}/fork`.

    Creates a new session by copying events 1..at_event_id from the
    source session, rewriting each event's `session_id` to the new id
    and preserving original event ids in the copy. New events on the
    forked session continue from `at_event_id + 1`.

    `preset` lets the forked branch resume under a different preset
    than the source ran with — the primitive behind event-pinned
    counterfactual experiments (cf. nature-eval). When omitted, the
    fork falls through to `default.json` the same way `resume_session`
    does.

    `parent_session_id` / `forked_from_event_id` are persisted as
    sidecar metadata so UIs can render a fork tree.
    """

    at_event_id: int = Field(
        ge=1,
        description="Copy events 1..at_event_id from the source session",
    )
    preset: str | None = Field(
        default=None,
        description=(
            "Preset name to drive the forked session's runner. "
            "Omit to continue under `default.json`."
        ),
    )


# ---------------------------------------------------------------------------
# Sending input
# ---------------------------------------------------------------------------


class SendMessageRequest(BaseModel):
    text: str


# ---------------------------------------------------------------------------
# Listing / inspection
# ---------------------------------------------------------------------------


class SessionInfo(BaseModel):
    session_id: str
    root_role_name: str
    root_model: str
    state: str  # "active" | "awaiting_user" | "resolved" | "error" | "closed"
    has_active_run: bool
    created_at: float
    preview: str = ""  # first user message text (truncated, single-line)
    parent_session_id: str | None = None
    forked_from_event_id: int | None = None


class ListSessionsResponse(BaseModel):
    sessions: list[SessionInfo] = Field(default_factory=list)


class ArchivedSessionInfo(BaseModel):
    """A session present on disk but not currently in the live registry.

    Server can hydrate one of these into a live ServerSession via
    POST /api/sessions/{id}/resume.
    """

    session_id: str
    event_count: int
    created_at: float
    last_event_at: float
    preview: str = ""  # first user message text (truncated, single-line)
    parent_session_id: str | None = None
    forked_from_event_id: int | None = None


class ListArchivedSessionsResponse(BaseModel):
    sessions: list[ArchivedSessionInfo] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Admin API — file-based edits of agents / presets / hosts
# ---------------------------------------------------------------------------
#
# Read endpoints merge builtin ∪ user ∪ project (project wins) and
# tag each entry with its effective `origin`. Writes always go to the
# user layer (`~/.nature/…`); builtin entries are read-only and return
# 403 on delete. Project-layer entries are read-only over HTTP too —
# project configs are typically committed to git and edited in-repo.


class AgentAdminInfo(BaseModel):
    """One agent as surfaced by `GET /api/admin/agents[/{name}]`."""

    name: str
    model: str  # host::model reference
    allowed_tools: list[str] | None = None
    allowed_interventions: list[str] | None = None
    instructions_text: str = ""
    description: str | None = None
    origin: str  # "builtin" | "user" | "project"


class ListAgentsAdminResponse(BaseModel):
    agents: list[AgentAdminInfo] = Field(default_factory=list)


class AgentPutRequest(BaseModel):
    """Body for `PUT /api/admin/agents/{name}` — writes a new agent
    (or overrides a builtin) into the user layer."""

    model: str
    allowed_tools: list[str] | None = None
    allowed_interventions: list[str] | None = None
    instructions_text: str = ""
    description: str | None = None


class PresetAdminInfo(BaseModel):
    name: str
    root_agent: str
    agents: list[str]
    model_overrides: dict[str, str] = Field(default_factory=dict)
    origin: str  # "builtin" | "user" | "project"


class ListPresetsAdminResponse(BaseModel):
    presets: list[PresetAdminInfo] = Field(default_factory=list)


class PresetPutRequest(BaseModel):
    root_agent: str
    agents: list[str]
    model_overrides: dict[str, str] = Field(default_factory=dict)


class HostAdminInfo(BaseModel):
    name: str
    provider: str
    base_url: str | None = None
    api_key_env: str | None = None
    models: list[str] = Field(default_factory=list)
    origin: str  # "builtin" | "user" | "project"


class ListHostsAdminResponse(BaseModel):
    hosts: list[HostAdminInfo] = Field(default_factory=list)
    default_host: str


class HostPutRequest(BaseModel):
    provider: str
    base_url: str | None = None
    api_key_env: str | None = None
    models: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Generic responses
# ---------------------------------------------------------------------------


class OkResponse(BaseModel):
    ok: bool = True


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
