# Architecture

> **Audience.** Engineers, researchers, and paper readers who want to
> understand how nature is built — and, specifically, how each axis of
> an LLM agent system is made into a composable, isolatable unit.
>
> **Scope.** This document describes the system as currently
> implemented. Companion design notes (`pack_architecture.md`) are
> kept for historical reference; if they diverge from this file,
> this file wins.

---

## 1. Design goal

nature is an **experimentation platform** for LLM agent systems. The
design assumes three premises:

1. Production agent configurations are combinatorial — prompts, tools,
   models, and orchestration topology all interact.
2. Useful research and engineering progress requires *isolating* each
   axis: changing one variable while holding the rest fixed.
3. Reproducibility requires *byte-level* determinism of prior state,
   so post-hoc analysis and counterfactual re-execution are possible.

Every major abstraction in the codebase is justified by the isolation
goal above. The sections below map each abstraction to the axis it
makes swappable.

---

## 2. Core abstractions

The system revolves around five units. Each one is a file-based
artifact (no hidden state, no database) and each one is discovered via
the same three-layer scheme (§3).

```
Host        — endpoint registry (where models live)
Agent       — role definition (prompt + allowed tools + instructions)
Preset      — composition (root agent, roster, per-agent model override)
Pack        — intervention bundle (gates, listeners, contributors + tools)
Frame       — the runtime execution unit, event-sourced
```

### 2.1 Host

> `nature/config/hosts.py`, manifest at `nature/agents/builtin/hosts` /
> `~/.nature/hosts.json` / `<project>/.nature/hosts.json`.

A **Host** is a named LLM endpoint: provider type (`anthropic`,
`openai`), base URL, authentication convention (`api_key_env`), and a
list of known models. References elsewhere in the system use
`<host>::<model>` syntax (e.g. `anthropic::claude-haiku-4-5`,
`local-ollama::qwen2.5-coder:32b`).

```json
{
  "provider": "openai",
  "base_url": "http://localhost:11434/v1",
  "api_key_env": null,
  "models": ["qwen2.5-coder:32b", "deepseek-r1:32b"]
}
```

**Axis isolated.** Where any given model call actually executes —
cloud vs local, which vendor — without leaking into role definitions.
A preset may reference `anthropic::claude-haiku-4-5` for one agent and
`local-ollama::qwen2.5-coder:32b` for another; the router in
§5 picks the right endpoint per call.

### 2.2 Agent

> `nature/agents/config.py::AgentConfig`, layout under
> `nature/agents/builtin/` or `~/.nature/agents/` or
> `<project>/.nature/agents/`.

An **Agent** is a role definition: the instruction prompt, the tools
the role is allowed to call, the default `host::model` reference, and
optional description. Stored as paired files:

```
<agents-dir>/
    <agent-name>.json          # AgentConfig: model, allowed_tools,
                               #              allowed_interventions,
                               #              instructions, description
    instructions/
        <agent-name>.md        # the prompt body (referenced by JSON)
```

Splitting the JSON and the prompt body keeps each axis cleanly
swappable:

- Rewriting an agent's prompt is a one-file edit that doesn't touch
  model or tool configuration.
- Restricting or expanding an agent's tool set is a one-field edit.
- Swapping an agent's default model updates a single string.

**Axis isolated.** The role contract (prompt, tool permission,
description) distinct from the model behind it.

### 2.3 Preset

> `nature/agents/presets.py::PresetConfig`, manifests at
> `nature/agents/builtin/presets/<name>.json` /
> `~/.nature/presets/<name>.json` /
> `<project>/.nature/presets/<name>.json`.

A **Preset** is the unit at which whole configurations are compared.
It declares:

```json
{
  "root_agent": "receptionist",
  "agents": ["receptionist", "core", "researcher", "analyzer",
             "implementer", "reviewer", "judge"],
  "model_overrides": {
    "researcher": "local-ollama::qwen2.5-coder:32b"
  }
}
```

- `root_agent` — the agent the session opens with.
- `agents` — the roster the session is allowed to spawn sub-agents
  from (off-roster delegations fall through to a minimal placeholder
  role).
- `model_overrides` — optional per-agent model swap that supersedes
  the agent's default for the duration of this preset.

Session creation takes a preset name (falling back to `default.json`
when omitted); the preset becomes the session-scoped source of truth
for which agents are reachable and which hosts to talk to. There is no
cross-session global "active preset" — two sessions can run under
different presets simultaneously.

**Axis isolated.** The composition of agents + their model mapping,
as a single switch. A preset is the natural unit at which "production
configurations" are compared.

### 2.4 Pack

> `nature/packs/registry.py::PackRegistry`, `nature/packs/discovery.py`,
> builtin packs at `nature/packs/builtin/`.

A **Pack** is a bundle of interventions and tools registered with the
runtime dispatch hub. An **Intervention** takes the form
`(kind, trigger, action)`:

```
kind         ∈ {gate, listener, contributor}
trigger      ∈ OnTool | OnLLM | OnEvent | OnTurn | OnFrame | OnCondition
action       callable: InterventionContext → list[Effect]
```

The three kinds occupy distinct roles in execution:

- **Gate** — runs before an action (tool call today). May return
  `Block(reason)` to short-circuit and surface an error to the LLM.
- **Listener** — runs after an action; may emit follow-up events,
  modify results (post-tool), or record state.
- **Contributor** — pure function returning footer hints or appended
  instructions, invoked during context composition.

Triggers dispatch through parallel indexes in `PackRegistry`
(`_by_tool_pre`, `_by_tool_post`, `_by_event`, `_by_turn`, `_by_frame`,
`_by_llm_pre`, `_by_llm_post`, `_by_condition`). Every dispatcher is
named `dispatch_<target>` or its `_sync` sibling for callers that
cannot await (see §5 for how dispatch wires into the run loop).

Packs are discovered at `AreaManager` construction via
`install_discovered_packs()` scanning:

```
nature/packs/builtin/             # shipped with nature
~/.nature/packs/<name>/           # user layer, file-based
<project>/.nature/packs/<name>/   # project layer
```

Each discovered pack exposes `install(registry: PackRegistry) -> None`
and is imported via an explicit module spec so neighbour files within
the pack directory resolve without `sys.path` edits. Errors during
discovery (malformed manifest, missing entry, install crash) are
logged and skipped — one broken pack cannot block the framework.

**Axis isolated.** Behaviour policy (guardrails, observability,
context augmentation) distinct from roles and models.

### 2.5 Frame + event sourcing

> `nature/frame/frame.py::Frame`, `nature/frame/manager.py::AreaManager`,
> `nature/events/store.py`, `nature/events/reconstruct.py`.

A **Frame** is the runtime unit of execution. Every session opens at
least one root frame; delegation (the `Agent` tool) opens a child
frame whose parent is recorded explicitly, producing a frame tree
rooted at the session's receptionist.

Every state-changing decision is an appended event — `FRAME_OPENED`,
`MESSAGE_APPENDED`, `LLM_REQUEST`, `LLM_RESPONSE`, `TOOL_STARTED`,
`TOOL_COMPLETED`, `FRAME_RESOLVED`, and so on. Events live in a
per-session jsonl file under `~/.nature/events/<session_id>.jsonl`.

Three properties fall out of this log-driven design:

1. **Reconstructability.** `events.reconstruct.reconstruct(session_id,
   store)` rebuilds the complete frame tree, conversation history,
   pack state, and ledger from the jsonl alone, with no runtime
   dependency on in-memory objects. Resume, fork, and dashboard replay
   all rely on this.
2. **Addressability.** Events carry monotonic `id`s per session.
   Any event can be referenced as `(session_id, event_id)`, which
   makes the log a flat index of decision points.
3. **Counterfactual execution.** `EventStore.fork(source_sid,
   at_event_id, new_sid)` copies events `1..at_event_id` into a new
   session, preserving original event ids. The forked session then
   resumes under an independently-chosen preset (see §5), continuing
   from `at_event_id + 1` under whatever configuration the caller
   chose. Because the prefix is byte-for-byte identical across
   branches, the post-fork delta in cost / latency / outcome
   attributes cleanly to the single variable that changed — the
   "event-pinned counterfactual" lever that motivates the paper.

**Axis isolated.** Prior state, via reconstruction, so any single
post-fork variable can be swept independently of the session history.

---

## 3. Discovery and layering

Hosts, agents, presets, packs, and eval tasks all use the same
three-layer resolution:

```
project   <cwd>/.nature/<kind>/        # highest priority, per-project
user      ~/.nature/<kind>/            # user-wide overrides
builtin   <nature-package>/builtin/    # shipped with nature
```

Later layers override by **whole entry** — there is no field-level
merge. To customize one setting of a builtin, copy the JSON to the
higher-priority directory and edit. This rule keeps the resolution
mental model trivial and makes diff-able "what changed" queries
feasible (compare the user-layer file against the builtin).

**Variable-isolation implication.** Because every artifact is a file,
every experimental configuration is diff-able, version-controllable,
and shareable. A researcher can publish a pack / agent / preset / task
by sharing a directory.

---

## 4. Session lifecycle

```
create_session(preset)
    │
    ├─ load_preset_bundle(preset_name)
    │       ├─ load_preset()      → PresetConfig
    │       ├─ load_agents()      → AgentsRegistry
    │       ├─ load_hosts()       → HostsConfig
    │       └─ validate_preset()  ← fails fast on dangling references
    │
    ├─ _build_provider_pool()        one LLMProvider per distinct host
    │
    └─ _assemble_preset_session()
            ├─ root AgentRole from preset.root_agent
            ├─ SessionRunner with:
            │     role_resolver       (agent_name → AgentRole)
            │     provider_resolver   (agent_name → LLMProvider)
            │     tool_registry
            └─ ServerSession attached
                │
send_message(sid, text)
    │
    └─ _drive_run (async background task)
            │
            AreaManager.run(frame):
                loop while frame.state == ACTIVE:
                    maybe_compact(frame)
                    dispatch_llm(PRE,  ctx)   ←  OnLLM(PRE)   listeners
                    output = await llm_agent(...)
                    emit LLM_REQUEST + LLM_RESPONSE
                    dispatch_llm(POST, ctx)   ←  OnLLM(POST)  listeners
                    append output messages
                    execute_tools(actions):
                        for each tool use:
                            dispatch_tool(PRE,  ctx)   ←  OnTool(PRE)  gates + listeners
                            run tool
                            dispatch_tool(POST, ctx)   ←  OnTool(POST) listeners
                    apply effects
                    if terminal signal: break
                dispatch_frame(RESOLVED|ERRORED, ctx)   ←  OnFrame(phase) listeners
                (in background: _emit → dispatch_event(type, ctx)  ←  OnEvent listeners)
```

The dispatch hooks are how Packs observe and steer execution. The
framework calls them at fixed points in the run loop; pack authors
register interventions against the trigger they care about and return
`Effect`s (emit event, modify input, block, append footer, …).

### Fork and resume

```
resume_session(sid, req)
    reconstruct(sid)  → frame tree + history from jsonl
    _load_preset_bundle(req.preset)   ← caller may pass a different preset
    _assemble_preset_session(..., frame=reconstructed_root)
    runner.manager.reopen(frame)      ← emits FRAME_REOPENED

fork_session(source_sid, at_event_id, preset=None)
    EventStore.fork(source_sid, at_event_id, new_sid)
         ├─ copies events 1..at_event_id with session_id rewritten
         └─ writes sidecar: {parent_session_id, forked_from_event_id}
    resume_session(new_sid, CreateSessionRequest(preset=preset))
```

The `preset` parameter on `fork_session` is the event-pinned
counterfactual primitive: two forks of the same source at the same
event id, one per preset, diverge from byte-identical prefixes.

---

## 5. Provider routing

> `nature/server/registry.py::SessionRegistry._build_provider_pool`
> and `_make_provider_resolver`.

When a preset references multiple hosts (e.g., `haiku-qwen-reader`
puts the researcher on `local-ollama` and everyone else on
`anthropic`), the registry builds a **provider pool** — one
`LLMProvider` instance per distinct host — and hands the
`AreaManager` a resolver `agent_name → LLMProvider`.

Each `llm_agent` call inside the run loop consults the resolver with
the current frame's `self_actor` (the agent's name). The resolver:

1. Checks the preset roster. Off-roster names return `None`; the
   manager falls back to the single `provider` passed at construction
   (for legacy / single-host callers).
2. Computes the effective model: `preset.model_overrides[name]` if
   present, else `agents[name].model`.
3. Parses `host::model`, returns `pool[host]`.

The `model` field on each outgoing `LLMRequest` carries the bare
model name (without `host::` prefix), so the provider uses its
endpoint with per-call model selection — e.g., one Anthropic
provider serves both `claude-haiku-4-5` and `claude-sonnet-4-6`
calls.

**Axis isolated.** Where each role's LLM traffic goes, per-call, as a
function of the preset. A single configuration change in the preset
file reroutes traffic without touching code.

---

## 6. Dispatch catalogue

| Trigger | Dispatched from | Sync sibling |
|---|---|---|
| `OnTool(tool_name, phase)` | `AreaManager._execute_and_apply` (pre / post tool) | — |
| `OnLLM(phase)` | `AreaManager.run` around `llm_agent` | — |
| `OnEvent(event_type)` | `AreaManager._emit` after the store append | `dispatch_event_sync` |
| `OnFrame(phase)` | `open_root`, `open_child`, `close`, `run` resolution / error | `dispatch_frame_sync` |
| `OnTurn(phase)` | `ContextComposer` during footer/contributor build | `dispatch_turn_sync` |
| `OnCondition(predicate)` | (not routed automatically; intended for future cross-cutting use) | — |

Listeners that return `EmitEvent(event_type, payload)` feed back into
`_emit`, which is the same entry point the framework uses directly —
so an event emitted by a listener is indistinguishable from a
framework-emitted event to downstream listeners. A
`_EMIT_MAX_DEPTH = 4` guard on the runtime counter prevents pathological
`Listener → EmitEvent → _emit → Listener …` cascades from stalling
the loop.

---

## 7. File layout reference

```
nature/
    agents/
        builtin/                          # shipped agents
            <agent>.json
            instructions/<agent>.md
            presets/<name>.json           # shipped presets (default.json)
        config.py                         # AgentConfig + 3-layer loader
        presets.py                        # PresetConfig + 3-layer loader
    config/
        hosts.py                          # HostConfig + 3-layer loader
        models.py                         # ModelSpec + 3-layer loader
        builtin/models.json               # per-`host::model` budget specs
    packs/
        builtin/<pack-name>/              # shipped packs
        discovery.py                      # file-based user/project loader
        registry.py                       # dispatch hub
        types.py                          # trigger, effect, intervention
    frame/
        frame.py                          # Frame dataclass + pack_state
        manager.py                        # AreaManager, run loop, dispatch
        agent_tool.py                     # Agent delegation tool
    context/
        body_compaction.py                # Pipeline + Microcompact + Dreamer
    events/
        store.py                          # FileEventStore + fork()
        reconstruct.py                    # log → frame tree
    server/
        app.py                            # HTTP + WS surface
        registry.py                       # SessionRegistry (orchestration)
    eval/
        builtin/cases/<task-id>/          # shipped eval tasks
        cli.py                            # `nature eval …`
        runner.py                         # single-cell execution
        results.py                        # RunRecord + aggregation
        report.py                         # markdown renderer
        diff.py                           # two-run comparison

~/.nature/
    agents/<agent>.json + instructions/   # user-layer agents
    presets/<name>.json                   # user-layer presets
    packs/<pack-name>/                    # user-layer packs
    hosts.json                            # user-layer hosts
    events/<session-id>.jsonl             # persistent event log
    server.pid                            # live server pidfile

<project>/.nature/
    (same subdirectory layout as ~/.nature/, project-local)
    eval/
        results/runs/<run-id>.json        # eval run records
        results/logs/<cell-tag>.jsonl     # per-cell event log copies
```

---

## 8. Invariants worth knowing

- **Event log is the source of truth.** Any in-memory state can be
  reconstructed from the jsonl. Tests and tools rely on this; adding a
  live state that cannot be replayed from events is a bug.
- **Preset validation runs at session creation.** Dangling references
  (unknown agent in roster, unknown host in override, missing API key
  env var) surface as `PresetValidationError` before any frame opens.
- **Framework never guesses identities.** Role resolution, tool
  permission, and provider routing are all explicit — either through a
  preset, a resolver callback, or a minimal placeholder. There is no
  implicit "default agent" outside the preset system.
- **File-layered discovery is append-only.** No single layer can
  remove entries from a lower layer; only the name-level override
  rule applies. This keeps discovery behaviour deterministic across
  environments.
- **Pack discovery is fail-soft.** A malformed user pack logs and is
  skipped; the framework always boots. Hard-fail validation is
  available via `_load_pack` for tooling that needs it (CI, lint).

---

## 9. What this architecture deliberately does *not* do

To keep the design defensible, some choices are out of scope and
should stay that way unless a strong reason surfaces:

- **No global mutable config at session scope.** Live config editing
  (mutating an active session's roster or model routing mid-run) was
  deleted in Stage B.2a. Experiments swap configurations by spawning
  new sessions — either fresh or forked.
- **No built-in RPC to other nature instances.** Each session is a
  local runtime; cross-machine coordination is a higher-level concern.
- **No hidden fallbacks for unknown triggers.** `OnCondition` and
  some combinations are declared in `types.py` but are not yet wired
  into dispatch; adding them requires an explicit wiring commit, not
  runtime heuristics.
- **No framework-level authentication / authorization.** Providers
  handle their own credentials (via `api_key_env`). nature does not
  secure its HTTP surface — production deployments that expose it
  must put their own proxy in front.
