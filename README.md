# nature

An LLM agent orchestration framework built around **event sourcing**, a
strict **header/body context split**, and a **Pack-based extension model**
that lets framework behavior (gates, nudges, memory, guards) be composed
out of reusable interventions instead of hard-coded into tools. Every
state transition is captured as a typed event; replay, resume, and
time-travel are first-class, not afterthoughts.

- **Client-server by default** — execution runs in a long-lived daemon;
  the TUI, dashboard, and any other UI are pure read-only clients that
  subscribe to an append-only event log.
- **Context as data, not text** — `Context = ContextHeader + ContextBody`.
  Header (role, principles) is static and cacheable; body (conversation)
  is the only thing compaction is allowed to touch.
- **Replay-deterministic** — state-transition events fully reconstruct
  every frame's context, including post-compaction body snapshots. Trace
  events (LLM/tool/user-input) can be stripped without changing the
  rebuilt state — and there's a test enforcing that invariant.
- **Multi-agent via sub-frames** — `Agent(name, prompt)` spawns a child
  frame whose spawn origin (`parent_frame_id`, `parent_tool_use_id`,
  `parent_message_id`) is recorded on the child's `FRAME_OPENED` event,
  so replay can resolve parent↔child links without timing inference.
- **Session fork (event-level branching)** — `nature sessions fork <sid>
  --at <event_id>` (or the `⑂ fork` button on any turn in the
  dashboard) copies the prefix of an existing session into a new
  session with parent-pointer metadata. Lets you try "what if the
  model had branched differently at turn 5?" without touching the
  base session. See [Sessions](#sessions) below.
- **Prompt caching baked in** — the Anthropic provider stamps cache
  breakpoints on the system prompt, the tools list, and the last
  real conversation message (skipping synthetic footer hints) and
  runs with a 1h TTL by default, so long multi-frame sessions
  serve 80%+ of their input tokens from cache. See
  [Prompt caching (Anthropic)](#prompt-caching-anthropic).
- **Pack architecture** — framework behaviors (gates, nudges, state
  caches) are declared as `Intervention`s inside installable `Pack`s,
  each scoped to a named `Capability`. Three built-in packs ship
  today: `edit_guards` (fuzzy-match + loop detection + reread hint),
  `reads_budget` (per-frame Read/Grep/Glob cap), and `file_state`
  (dedup + Edit read-first guard via `ReadMemory`). See
  [Pack architecture](#pack-architecture).
- **Code-level safety guards** — strict tool-input validation,
  self-delegation rejection, cwd-boundary on Glob, and a bash
  root-walk block each come from a real incident and run in code
  (not just prompt discipline). See
  [Framework safety guards](#framework-safety-guards).
- **Parallel tool + delegation execution** — when a turn emits
  multiple concurrency-safe tools (`Read × 3`) or multiple Agent
  delegations (`core` → `researcher + analyzer + reviewer` at once),
  the manager partitions them into batches and runs each batch
  through `asyncio.gather`. Each batch is bracketed by
  `PARALLEL_GROUP_STARTED` / `PARALLEL_GROUP_COMPLETED` events with
  a shared `parallel_group_id` stamped on every inner event, so the
  batch reads as one transactional unit — session fork refuses to
  cut through the middle of one. The dashboard surfaces parallel
  batches with an accent-colored left border + `∥` marker on every
  step that ran inside the batch. See
  [Parallel execution](#parallel-execution).

## Quickstart

```bash
# Install
pip install -e ".[anthropic,dashboard]"

# Or, for local models via Ollama:
pip install -e ".[openai,dashboard]"

# 1. Start the server in a dedicated terminal
nature server start

# Or bind to all interfaces so another device (iPhone on Tailscale, ...)
# can reach the dashboard:
nature server start --host 0.0.0.0

# 2. In another terminal, launch the TUI (picks up sessions from the server)
export ANTHROPIC_API_KEY=sk-ant-...
nature

# 3. Or open the browser dashboard alongside the TUI
nature -d                           # desktop  — http://localhost:7777/
#                                      mobile   — http://localhost:7777/m
```

The server owns the `AreaManager`, `EventStore`, and provider connections.
The TUI talks to it over HTTP + WebSocket — crashes on the client side
never touch execution.

## Providers

nature ships two concrete providers, and any OpenAI-compatible API works
through `openai_compat`.

```bash
# Anthropic (default)
export ANTHROPIC_API_KEY=sk-ant-...
nature

# OpenRouter
export OPENROUTER_API_KEY=sk-or-...
nature -p openrouter -m anthropic/claude-sonnet-4

# Ollama (local) — anything OpenAI-compatible works via -p openai
nature -p openai --base-url http://localhost:11434/v1 -m qwen2.5-coder:32b

# Local models need longer HTTP timeouts than cloud endpoints because
# prefill on a 30B+ model with full context can run 1–5 minutes. The
# OpenAI-compat provider reads NATURE_OPENAI_TIMEOUT (seconds, default 600):
export NATURE_OPENAI_TIMEOUT=900
```

Per-project provider/model settings live in `.nature/frame.json` so you
don't need CLI flags after the first run. CLI flags always win over the
config file.

### Prompt caching (Anthropic)

The Anthropic provider enables **prompt caching by default** — every
`llm_agent` call is sent with `cache_control={"type": "ephemeral",
"ttl": "1h"}` and the provider stamps breakpoints at three positions:

1. **End of the system prompt** — caches the role's instructions and
   principles.
2. **End of the tools list** — caches the tool schema, which is
   usually identical across turns in the same role.
3. **Last real message** (the footer-hint message is explicitly
   skipped via `_pick_cache_anchor_index`, because the hint changes
   every turn and would break prefix continuity if marked).

The 1h TTL is a deliberate choice over the default 5m: multi-frame
sessions routinely run 5–10+ minutes, and at 2x create / 0.1x read
pricing, the break-even is ~3 reads per created prefix — which is
essentially guaranteed in any core-orchestrated chain.

Empirical impact measured on a
"split `bash_checks.py` into a per-check package" self-refactor:

| Setup | Events | LLM calls | Cost | Duration | Cache hit rate |
|-------|-------:|----------:|-----:|---------:|---------------:|
| Pre-caching (Sonnet throughout)            | 994 | 110 | $5.96 | 8 min   | 0%    |
| Caching + footer-rule fix (Sonnet)         | 273 |  31 | $0.79 | 3 min   | 58.7% |
| + anchor skips hint + Haiku on non-core    | 415 |  48 | $0.66 | 4.7 min | 87.2% |
| Trivial task (add one `/health` endpoint)  |  51 |   6 | $0.04 | 30 sec  | —     |

`openai_compat` (used for Ollama and other OpenAI-compatible backends)
accepts the same `cache_control` parameter and silently ignores it, so
the override is safe to apply globally.

## CLI

```
nature                             # launch TUI (session picker → chat)
nature --new                       # skip picker, start a fresh session
nature -r <session_id>             # resume a specific archived session
nature -d                          # open dashboard alongside TUI

nature server start                # run the daemon in the foreground
nature server start --host 0.0.0.0 # bind non-loopback (mobile / LAN / Tailscale)
nature server stop                 # SIGTERM a running daemon
nature server status               # show running/stopped + pid
nature server restart              # stop + start (use after code changes)

nature sessions list               # live + archived sessions on the server
nature sessions resume <id>        # hydrate an archived session without TUI
nature sessions fork <id> --at <n> # branch a session at event n into a new session
```

Global options: `-p/--provider`, `-m/--model`, `--api-key`, `--base-url`,
`--host`, `--port`, `-r/--resume`, `--new`, `-d/--dashboard`.

## Frame + Event architecture

The execution model is three layers of progressively higher-level
abstractions, each with a single responsibility:

```
Event Layer                — append-only log, reconstruct(), typed payloads
  ↑
Frame Layer                — AreaManager drives one Frame to resolution
  ↑
Session Layer              — SessionRunner (one session ↔ one root frame tree)
```

### Frame

A `Frame` is a scope that owns exactly one `Context`, its role, its model,
and its place in the frame tree (parent, children, state). The
`AreaManager` mutates frames as execution progresses — every mutation is
also written to the `EventStore` so replay produces the same frame tree
without running the LLM again.

```python
@dataclass
class Frame:
    id: str
    session_id: str
    purpose: str
    context: Context                   # header + body
    model: str
    parent_id: str | None
    children_ids: list[str]
    state: FrameState                  # ACTIVE | AWAITING_USER | RESOLVED | ERROR | CLOSED
    counterparty: str                  # who this frame replies TO (stamped at open)
    budget_counts: dict[str, int]      # Pack-tracked counters (reads used, etc.)
    ledger: FrameLedger                # Phase 1 scaffolding (mostly unused today)
    pack_state: dict[str, Any]         # Pack-owned state bag — see Pack architecture
```

`pack_state` is where Packs put their per-frame state. `file_state` Pack
stores `ReadMemory` at `frame.pack_state["read_memory"]`; future Packs
add their own namespace keys. Keeps Frame minimal — feature-specific
state belongs to the Pack that owns the feature.

### Context = Header + Body

The header/body split is the single most load-bearing invariant in the
codebase. Compaction touches body only; cache boundaries follow the same
line. Both halves are Pydantic models, not strings.

```python
class Context(BaseModel):
    header: ContextHeader              # static identity (cacheable half)
    body: ContextBody                  # growing dialogue (uncacheable half)

class ContextHeader(BaseModel):
    role: AgentRole                    # name, instructions, allowed_tools, model
    principles: list[BasePrinciple]    # framework/project/user/runtime rules

class ContextBody(BaseModel):
    conversation: Conversation         # ordered list of from_/to Messages
    todos: list[TodoItem]              # per-frame checklist (see TodoWrite)
```

`ContextComposer.compose()` is the single boundary that serializes a
`Context` into an `LLMRequest`. No other code flattens context to prompt
text. On top of the header → system / body → messages mapping, the
composer also runs a **footer rule pipeline**: a set of pure functions
that inspect the current body state and may append ephemeral
`[FRAMEWORK NOTE]` hints as a tail user-role message — used today to
push the LLM past plan-shape stops and to keep it aligned with its
own todo list (see *Footer hints*, below).

### Events — the single source of truth

Every state transition is captured as a typed event. Producers never
touch the store directly; they build a `*Payload` model and hand it to
`AreaManager._emit`, which dumps it into the `Event` envelope.

Events split into two categories:

- **STATE_TRANSITION** — drives `reconstruct()`. Dropping or reordering
  one changes the rebuilt Frame. Replay applies these deterministically
  in event order.
  - `FRAME_OPENED`, `FRAME_RESOLVED`, `FRAME_CLOSED`, `FRAME_ERRORED`
  - `FRAME_REOPENED` (resume reactivates a terminal frame)
  - `HEADER_SNAPSHOT` (role + initial principles, emitted after FRAME_OPENED)
  - `MESSAGE_APPENDED`, `ANNOTATION_STORED`
  - `PRINCIPLE_ADDED`, `ROLE_CHANGED`
  - `BODY_COMPACTED` (post-compaction body snapshot)
  - `TODO_WRITTEN` (full-list overwrite of the frame's todo checklist,
    emitted by the framework after a successful `TodoWrite` tool call)
  - `BUDGET_CONSUMED` (per-tracked-tool counter increment; Phase 3.1)
  - `READ_MEMORY_SET` (ReadMemory entry written; metadata-only, no content)
  - `LEDGER_SYMBOL_CONFIRMED`, `LEDGER_APPROACH_REJECTED`,
    `LEDGER_TEST_EXECUTED`, `LEDGER_RULE_SET` (Phase 1 scaffolding for
    the future Memory Ledger — events defined, producers TBD)
  - `AGENT_MODEL_SWAPPED` (runtime escalation — defined, producers TBD)
- **TRACE** — observability only. UIs and debuggers read them; reconstruct
  ignores them. Dropping a trace event must not change the rebuilt state
  — this invariant is locked in by a property test.
  - `LLM_REQUEST`, `LLM_RESPONSE`, `LLM_ERROR`
  - `TOOL_STARTED`, `TOOL_COMPLETED`
  - `USER_INPUT`, `ERROR`
  - `HINT_INJECTED` (records which footer rules / Contributor
    Interventions fired on a given `LLM_REQUEST` and what text
    they injected)
  - `PARALLEL_GROUP_STARTED`, `PARALLEL_GROUP_COMPLETED` (bracket
    events around a parallel tool/delegation batch; every inner
    `TOOL_*` event carries the same `parallel_group_id` so session
    fork can reject at_event_ids that fall strictly inside)
  - `EDIT_MISS`, `LOOP_DETECTED`, `LOOP_BLOCKED`, `PATH_INVALID`,
    `PARSE_RETRY` (Pack guard observations — emitted by Interventions
    in `edit_guards` / `file_state` for dashboard visibility)
  - `BUDGET_WARNING`, `BUDGET_BLOCKED` (budget crossing observations)

Events are stored in JSONL per session under `~/.nature/events/` with a
session-monotonic `id`. `FileEventStore.live_tail()` gives UIs a subscribe
stream that starts from the historical snapshot and then yields live
events as they land.

### Time-travel replay

`reconstruct(session_id, store, up_to_event_id=N)` slices the event
stream so the rebuilt Frame tree reflects the state right after event
`N` was applied. Use cases:

- Dashboard scrubber — show a frame as it looked right before a
  specific tool fired (`GET /api/sessions/{sid}/frames/{fid}/context?up_to=N`)
- Resume safety — inspect what state the next user input will land on
- Tests — assert mid-run invariants without driving a second session

The slicer is O(events_up_to_N) — no full rebuild + filter.

### Crash recovery — incomplete spans

`reconstruct()` also surfaces in-flight LLM and tool calls that never
saw a matching close event. `ReplayResult.incomplete_spans` is a
`list[IncompleteSpan]`, populated by scanning trace events for
unmatched `LLM_REQUEST` / `TOOL_STARTED` records. A resumed session
can check this list and decide to roll back the orphaned turn or
warn the user before sending new input on top of it.

### Multi-agent delegation via sub-frames

When a frame calls `Agent(name, prompt)`, `AreaManager` opens a **child
frame** with a fresh context (parent context is never inherited) and
records the spawn origin on the child's `FRAME_OPENED` event:

```
child FRAME_OPENED payload:
  parent_id: frame_xxx
  spawned_from_message_id: msg_yyy     (parent's assistant turn that contained the Agent call)
  spawned_by_tool_use_id: toolu_zzz    (the specific Agent tool_use id)
```

The parent's tool_result `MESSAGE_APPENDED` carries a `delegations` map
(`tool_use_id → child_frame_id`). After replay, `ReplayResult.child_of()`
resolves parent → child in O(1) without scanning.

### Body compaction

Compaction runs as a pipeline of strategies against `ContextBody`,
with read-only access to the header. Each strategy that actually mutates
the body produces a `BODY_COMPACTED` event carrying the full
post-compaction message list — so resume after compaction lands on the
exact same trimmed body the live run saw.

```python
pipeline = BodyCompactionPipeline(
    strategies=[MicrocompactBodyStrategy(preserve_turns=4)],
    budget=TokenBudget(context_window=200_000, ...),
)
manager = AreaManager(..., compaction_pipeline=pipeline)
```

### Footer hints — nudging the LLM at compose time

`ContextComposer.compose()` runs a footer rule pipeline on top of the
header/body → LLMRequest mapping. Each rule is a pure function of
`(ContextBody, ContextHeader, self_actor) → Hint | None`, and each
rule now lives in its own file under `nature/context/footer/rules/`
with shared helpers in `footer/helpers.py`. The composer aggregates
the non-None hints and appends them as a single tail user-role
message wrapping each hint in `<system-reminder>` tags (mirroring
Claude Code's reminder protocol — the model treats them as framework
signals instead of user content, and each hint explicitly tells the
model to act silently rather than narrate the reminder back). The
body is never mutated; hints exist only in the rendered prompt and
in the `HINT_INJECTED` trace event the framework emits alongside
`LLM_REQUEST`, so replay stays deterministic.

Four rules ship today, all focused on preventing plan-shape stops,
keeping the LLM aligned with its own `TodoWrite` checklist, and
ensuring role-required real work actually happens before synthesis:

| Rule | Fires when | Nudge |
|------|-----------|------|
| `synthesis_nudge` | `from_=tool` tail, no incomplete todos, AND the role's required tools (if any) have been used | "you just received tool_results, write the final answer now — don't re-plan" |
| `todo_continues_after_tool_result` | same tail shape, but unfinished todos remain (pending OR in_progress) | "this tool_result is a checkpoint, not a finish line — keep going down the list" |
| `todo_needs_in_progress` | pending todos exist but none is in_progress | "pick the next item and mark it in_progress before you start working" |
| `needs_required_tool` | `from_=tool` tail AND the current role has a required-tool set AND none of those tools were called yet in this frame | "you are the `<role>` and you haven't called `Edit`/`Write`/etc. yet — either call one now or honestly report that no change is needed (don't fabricate a completion)" |

`synthesis_nudge` and `todo_continues_after_tool_result` are mutually
exclusive: the first yields whenever the second would fire. `synthesis_nudge`
additionally yields to `needs_required_tool` — a specialist that hasn't
done its real work (e.g., `implementer` without an `Edit`/`Write`) gets
pushed to do the work, not to synthesize over nothing. The
`ROLE_REQUIRED_TOOLS` catalog maps each specialist role to its minimum
viable real-work tool set (`implementer → {Edit, Write}`,
`researcher → {Read}`, etc.) — this is the code-level check that stops
the "ran one Glob, fabricated a file tree" failure mode that weak models
produce when they're only nudged by prompts.

The `_last_message_tool_results` helper that drives all four rules
explicitly **filters out `TodoWrite` tool_results** — the agent's
own bookkeeping doesn't count as "real work just completed", and
without that filter the todo-oriented rules fired on every `TodoWrite`,
the agent called `TodoWrite` again to comply, and the rule fired
again ad infinitum (one production session saw 79 consecutive
identical `TodoWrite` calls before the filter was added).

Because the rules are pure functions of the body, `reconstruct()`
rebuilds the same state that fed them — the same hints would be
derived at the same event ids, so time-travel resume stays
byte-for-byte stable.

A fourth nudge — the Claude-Code-style "you just marked 3+ todos as
completed without scheduling a verification step" warning — is
embedded in the `TodoWrite` tool's own `tool_result` text rather than
in the footer. This matches Claude Code's approach (the model sees
the warning in the same turn as the write) and the text lands in the
`TOOL_COMPLETED` event log verbatim, so resume replays it byte-for-byte.

## Multi-agent topology — `frame.json`

A multi-agent system is declared in a single file at
`.nature/frame.json` (project) or `~/.nature/frame.json` (user). No
Python registration, no decorators.

```jsonc
{
  "provider": {
    "name": "openai",
    "base_url": "http://localhost:11434/v1",
    "model": "qwen2.5:72b-instruct-q4_K_M"
  },
  "root_agent": "receptionist",
  "agents": {
    "receptionist": {
      "description": "User-facing agent",
      "allowed_tools": null,              // null = all registered tools
      "instructions": "# Role: Receptionist\n..."
    },
    "core": {
      "description": "Planner / delegator",
      "model": "qwen2.5:72b-instruct-q4_K_M",
      "allowed_tools": ["Agent"],         // delegation-only
      "instructions": "# Role: Core\n..."
    },
    "researcher": {
      "description": "Codebase explorer",
      "model": "qwen2.5-coder:32b",
      "allowed_tools": ["Read", "Glob", "Grep", "Bash"],
      "instructions": "# Role: Researcher\n..."
    }
  }
}
```

`frame.json` gets loaded by `SessionRegistry` at session creation. When
a frame's role is `core` and it calls `Agent(name="researcher", ...)`,
the manager spawns a child frame with the `researcher` role's model,
tools, and instructions — all derived from `frame.json`.

The seven built-in agent profiles
(`receptionist`, `core`, `researcher`, `analyzer`, `implementer`,
`reviewer`, `judge`) live in `nature/agents/builtin/` and serve as the
default topology if `frame.json` is absent.

## Dashboard

```bash
nature -d                           # opens the dashboard in your browser
# Desktop:  http://localhost:7777/
# Mobile:   http://localhost:7777/m   (simplified UI, touch targets, stop button)
```

Both dashboards are pure event-store consumers. They open a WebSocket
to `/ws/view/sessions/{id}` which carries a **structured
`SessionViewDto`** — a turn-tree already shaped for rendering
(turns, steps, tool calls, sub-agent cards, todos panels), so the HTML
side just reconciles DTO → DOM with keyed updates. The raw event
stream at `/ws/sessions/{id}` is still exposed for debuggers.

- **Turn tree** — the main stream reads like a git log: each top-level
  turn is a squash-merged node showing the user input and the final
  assistant reply, with a ▸ toggle that expands to the per-step tool
  calls and delegations underneath. Sub-agent frames render as nested
  cards with their own mini turn-tree.
- **Todos panel** — sticky block pinned above the turn when the
  frame's `body.todos` is non-empty. Shows a `2 / 3 COMPLETED · 1 IN
  PROGRESS` summary, uses the `activeForm` copy for the currently
  in-progress item (so the UI matches what the LLM is narrating to
  itself), and the `content` copy for pending / completed items.
  Updated live as each `TODO_WRITTEN` event lands.
- **Badges** — per-turn markers showing whether the final assistant
  message was **regenerated from tool_result** (`✨ SYNTHESIS · N
  SOURCES`), the stop reason (`STOP: …`), and cumulative token usage
  (`↓in ↑out`).
- **Pulse bar** — sticky breathing indicator that appears under the
  header while a run is in flight, showing spinner + activity
  ("running Glob", "delegating → researcher", "thinking") + elapsed
  time + cumulative input/output tokens. Mirrors the TUI pulse so
  long delegation chains don't feel frozen. The mobile dashboard
  replaces the three-dot menu with a dedicated **Stop** button that
  cancels the in-flight run directly.
- **Context panel / drawer** — click any frame card (desktop) or
  turn (mobile) to see its full Context: HEADER (role instructions +
  principles + allowed_tools) and BODY (every `from_ → to` message
  in order, plus the latest todos). Supports `?up_to=N` query param
  for time-travel rewind.
- **Session picker** — toggle at the top to switch between live and
  archived sessions without restarting.
- **Input bar** — floating textarea at the bottom. Sends a new user
  message to the currently viewed session, or creates a fresh session if
  none is selected. Mostly used for browser-driven smoke testing.

The TUI gets the same pulse indicator in its status bar
(`nature/ui/frame_tui.py`), driven by a 150ms `set_interval` tick over
the same reactive run state.

### Live agent config — the **⚙ config** panel

Both dashboards expose a **⚙ config** button in the header that opens an
editor for the in-memory `FrameConfig`. You can change an agent's model,
toggle its tool allowlist, rewrite its instructions, save the current
shape as a named preset, load a preset, or delete one — all without
touching `.nature/frame.json` or restarting the server.

- **What "live" means** — the editor mutates the registry's
  `current_config`, which is what role resolution reads every time a
  new session is created or a parent agent spawns a sub-agent. Running
  frames keep the config they were already given; the next `Agent(...)`
  delegation picks up the new shape. That makes it safe to A/B a
  receptionist on haiku vs opus while a long-running session keeps
  going.
- **Per-agent knobs** — pick the model from the curated `MODEL_CATALOG`
  dropdown (`custom…` lets you type any id), tick an explicit
  `allowed_tools` list or leave it as `all`, and expand the
  instructions textarea to edit the system prompt. Validation on the
  server side rejects unknown tool names so you can't save a typo.
- **Presets** — save the current config to `.nature/presets/<name>.json`
  and reload it later. Handy for keeping a `"fast-qwen-local"`,
  `"mixed-tier-anthropic"`, and `"all-opus-for-hard-stuff"` side by
  side and flipping between them for per-task tuning. Preset names
  must match `[A-Za-z0-9_-]{1,64}` (no path traversal).
- **Provider is read-only** — the provider block (`name`, `model`,
  `base_url`) is surfaced for reference but can't be changed from the
  UI; switch providers by restarting with a different flag or
  `frame.json` until there's a story for hot-swapping keys.

REST surface (used by both dashboards, also handy from curl):

| Method & path | Purpose |
|---|---|
| `GET /api/config` | Current `FrameConfig` JSON (or `{}` if none active) |
| `PUT /api/config` | Replace the whole config (full pydantic validation) |
| `PATCH /api/config/agents/{name}` | Partial update of one agent (model, allowed_tools, instructions, description) |
| `GET /api/config/models` | Curated dropdown entries with `tier` metadata |
| `GET /api/config/tools` | Tool names the server's registry currently exposes |
| `GET /api/presets` | List saved preset filenames |
| `POST /api/presets/{name}` | Save the **current** in-memory config under that name |
| `POST /api/config/apply-preset/{name}` | Load a preset into `current_config` |
| `DELETE /api/presets/{name}` | Remove a preset file from disk |

### Preset benchmarking — the **`nature-eval`** Claude skill

`.claude/skills/nature-eval/` is a reusable harness that benchmarks presets
against real past commits from this repo. For each `(preset × task)`
cell it checks out `commit^` in a throwaway git worktree, applies the
golden test-only diff as acceptance criteria, spins up an isolated
nature server there (with `NATURE_HOME` redirected so the eval never
touches `~/.nature/events/`), drives it with a curated prompt, then
runs pytest to score pass/fail. Cost + latency come from the session's
event log.

- **5 curated tasks** picked from recent commits spanning trivial
  bug-fix → large multi-file feature (see
  `.claude/skills/nature-eval/tasks.json`).
- **`run_one.py`** runs one cell end-to-end and prints a single-line
  JSON result. `render_table.py` aggregates multiple cells into a
  markdown table with cost deltas and a decision signal.
- **Eval convention**: an invariant preamble is prepended to every
  task prompt forbidding the AI-under-test from running pytest itself.
  Otherwise cheap-model sessions loop in `pip install` / `venv`
  bootstrap phases instead of making the actual edit — the harness is
  responsible for verification, not the model.
- **Per-run isolation**: worktree + `NATURE_HOME` both scoped to a
  unique tmp dir per cell. Parallel cells can run without port
  conflicts.
- **Timeout scaling**: `--timeout-scale N` multiplies task budgets for
  slow presets (e.g., `--timeout-scale 5` for local 32B models that
  need 15+ minutes per task).

Example measurement from 2026-04-15 (cloud matrix, see
`.claude/skills/nature-eval/results/cloud-matrix-final.json`):

| preset | pass rate | total cost | avg latency |
|---|---|---|---|
| `current` (mixed sonnet+haiku) | 5/5 | $2.92 | 342s |
| `all-haiku` | 4/5 | $1.01 | 133s |

Invoke from any Claude Code session in this repo with `/nature-eval`.

## Built-in tools

| Tool | Description |
|------|-------------|
| **Bash** | Shell command execution with 11 safety checks (see `nature/security/bash_checks/`) |
| **Read** | File reading with line numbers, offset/limit |
| **Write** | File creation/overwrite with auto mkdir |
| **Edit** | Surgical text replacement (exact match) |
| **Glob** | File pattern matching, sorted by mtime |
| **Grep** | Content search (ripgrep if available, Python fallback) |
| **TodoWrite** | Externalized per-frame todo checklist with dual-form items (`content` + `activeForm`). The tool itself just validates; the framework emits `TODO_WRITTEN` and drives the footer hints above. |
| **Agent** | Sub-frame delegation (intercepted by AreaManager, not executed as a normal tool) |

Each role's `allowed_tools` acts as a filter in `ContextComposer`: both
the prompt schema shown to the LLM and the executable registry are
derived from the same list.

### Framework safety guards

The framework rejects malformed or dangerous agent behavior at the
tool boundary, not in the prompt. Each of these came from a real
production failure — the fix lives in code so the failure can't
recur even if the prompt language is changed or a different model
is swapped in.

- **Strict tool input validation** — `TodoWriteInput.todos` and
  `AgentInput.name` are required fields with no default. A model
  that confused schemas (e.g., sending `{"name":"researcher"}` to
  `TodoWrite`) used to write an empty todo list silently; now the
  Pydantic validator rejects it, and the model sees an
  `is_error=true` tool_result and self-corrects on the next turn.
- **Self-delegation rejection** — `_handle_delegation` refuses to
  open a child frame whose role name matches the parent's role, and
  refuses delegations without an explicit `name` field. Previously,
  an `Agent()` call without `name` defaulted to `"core"`, which
  silently produced `core → core → core` self-loops when the caller
  *was* core.
- **Glob cwd boundary** — `GlobTool` canonicalizes `path` via
  `os.path.realpath` and rejects any target that escapes the
  project's working directory. Absolute patterns like `/etc/*` are
  also rejected up-front. Without this, one Sonnet session decided
  to "look for files in the project" by issuing `Glob(pattern="**/*",
  path="/")` and spent 8 minutes walking a macOS filesystem before
  being killed.
- **Bash root-walk block** — a new `check_filesystem_walk_from_root`
  in `bash_checks/` rejects `find /`, `ls -R /`, `du -a /`, `tree /`,
  `grep -r ... /`, `rg ... /` and similar obvious root-walk patterns.
  Specific absolute paths under root (`find /Users/me/project`,
  `tree /tmp/build`) still pass — only the bare `/` traversal is
  blocked.

All of these return `is_error` tool_results rather than raising
exceptions, so the model can observe its own mistake and retry —
the same self-correction pattern Claude Code uses for
`InputValidationError`.

## Pack architecture

Framework behavior splits into two modes: things an **agent invokes**
(tools — the `(A)` side) and things the **framework does to the agent**
(gates, nudges, state caches — the `(B)` side). The Pack architecture
gives `(B)` the same first-class status that tools have always had:
every framework behavior is an `Intervention` inside a `Capability`
inside a `Pack`, and the dispatch layer routes triggers → interventions
→ effects → manager-level application.

```
Pack (installable unit)
 └── Capability (coherent feature name)
      ├── Tool           — (A) agent-invoked
      └── Intervention   — (B) framework-initiated, 3 kinds:
            ├── Gate         — pre-action: allow / Block / ModifyToolInput
            ├── Listener     — post-event: EmitEvent, ModifyToolResult, InjectUserMessage
            └── Contributor  — context-build: AppendFooter, AppendInstructions
```

**Listener phases** — same-trigger listeners don't cross-reference each
other; they split into `PRIMARY` (react to trigger) and `POST_EFFECT`
(react to PRIMARY's effect list via `ctx.primary_effects`). Two phases
max, no cascade, no depth limit needed — cycles are structurally
impossible. See `pack_architecture.md` §4.5.

**Effects** — Interventions don't mutate state directly; they return a
list of effects (`Block`, `ModifyToolInput`, `ModifyToolResult`,
`AppendFooter`, `InjectUserMessage`, `EmitEvent`, `UpdateFrameField`,
`SwapModel`). The `AreaManager` applies them, emitting events along
the way. This keeps interventions pure and trivially testable.

**Registry** — `nature/packs/registry.py` holds a trigger-indexed map
of interventions. `AreaManager.__init__` auto-installs the built-in
packs into the process-wide `default_registry` when no explicit
registry is passed (tests pass their own empty registry).

### Built-in Packs

| Pack | What it does | Interventions |
|------|---|---|
| **`edit_guards`** | Recovery layer for Edit failures. Pure body-walking (no new state). | `fuzzy_suggest` (Listener PRIMARY): on Edit miss, attach `difflib` closest match to the error. `reread_hint` (Contributor): footer hint after Edit failure in last body message. `loop_detector` (Listener POST_EFFECT): observes `EDIT_MISS` emissions, emits `LOOP_DETECTED` at threshold. `loop_block` (Gate): refuses Edit after 3 consecutive same-hash failures. |
| **`reads_budget`** | Per-frame Read/Grep/Glob cap. Body-walk derives the count; no separate counter. | `reads_budget.gate` (Gate): `Block` when prior reads + 1 > 20. `reads_budget.warning` (Contributor): footer nudge at 80% of the limit. |
| **`file_state`** | Dedup + Edit read-first guard via `ReadMemory`. | `edit_read_first` (Gate): strict — the `old_string` must appear in a read segment. `read_persist` / `edit_persist` / `write_persist` (Listeners): emit `READ_MEMORY_SET` so state survives resume. |
| **legacy footer shim** | Existing footer rules ported to Contributor Interventions on first touch. | `legacy.synthesis_nudge_rule`, `legacy.todo_needs_in_progress_rule`, `legacy.todo_continues_after_tool_result_rule`, `legacy.needs_required_tool_rule` |

### ReadMemory — the file-state ledger

`nature/context/read_memory.py`. Not a file cache — a structured record
of "which file, which line ranges, at what mtime" the model has
observed. Stored at `frame.pack_state["read_memory"]` by the
`file_state` Pack.

```python
@dataclass
class ReadSegment:
    start: int          # inclusive line
    end: int            # exclusive
    text: str           # actual line content

@dataclass
class ReadMemoryEntry:
    path: str
    mtime_ns: int
    total_lines: int
    segments: list[ReadSegment] | None   # None = expired (content gone, meta kept)
    depth: int = 0                        # 0=self, 1=child, 2=grandchild...
    hit_count: int = 0
```

Key behaviors:

- **Dedup**: `Read` checks `read_memory.get(path)` first. If the
  requested range is covered by existing segments and mtime hasn't
  changed, returns a stub ("already read in this session"). Saves
  context tokens.
- **Cache-serve**: different range of a cached file → served from
  segments without disk I/O. Segments overlap-merge on insert.
- **Read-first guard**: `file_state.edit_read_first` Gate blocks
  Edit if the path isn't in `read_memory` OR if the `old_string`
  doesn't appear in any read segment (strict — model can't Edit a
  region it hasn't seen).
- **Stale vs expired**: mtime mismatch on active entry → delete
  (live stale). Budget eviction or resume reconstruct → expire
  (segments=None, metadata kept). Expired entries skip mtime check
  and signal "file was read at some point, content unknown."
- **Frame-tree propagation**: when a child frame resolves, its
  `read_memory` merges into the parent's with `depth += 1`. Parallel
  sub-agents reading different files naturally aggregate into the
  parent's view without a separate scatter-gather mechanism. Lower
  depth (more direct knowledge) wins on path collision.
- **Event-sourced**: `READ_MEMORY_SET` events persist metadata only
  (path, mtime, hash, offset, limit, depth) — no content. Reconstruct
  rebuilds entries as expired; content re-fills on next live Read.

### Writing a new Pack

The shortest path to add framework behavior (say, a gate that blocks
`Bash` calls containing `curl` in read-only mode):

```python
# nature/packs/builtin/example/block_curl.py
from nature.packs.types import (
    Block, Intervention, InterventionContext, OnTool, ToolPhase,
)
from nature.events.types import EventType

def _action(ctx: InterventionContext):
    tc = ctx.tool_call
    if tc is None or not ctx.frame:
        return []
    cmd = tc.tool_input.get("command", "")
    if "curl" in cmd and ctx.frame.state == "read_only":
        return [Block(reason="curl blocked in read-only mode",
                      trace_event=EventType.PATH_INVALID)]
    return []

block_curl = Intervention(
    id="example.block_curl",
    kind="gate",
    trigger=OnTool(tool_name="Bash", phase=ToolPhase.PRE),
    action=_action,
)
```

Then wrap it in a `Capability` + `Pack` export and register from
`nature/packs/builtin/__init__.py`'s `install_builtin_packs`. Copy
`nature/packs/builtin/edit_guards/` for a complete template with four
interventions across all three kinds.

### Parallel execution

`AreaManager._execute_and_apply` partitions a turn's pending
actions into consecutive runs and batches each run that contains
2+ items of the same kind into a single `asyncio.gather` call:

- **Regular tool batches** — any run of concurrency-safe tools
  (`Read`, `Glob`, `Grep`, `Bash` with safe commands) is handed
  to `execute_tools(tool_uses=[...all...])` so the executor's
  per-tool partition + gather logic actually fires. Three
  consecutive `Read`s in one assistant turn run simultaneously.
- **Delegation batches** — any run of `Agent` tool_uses is handed
  to `asyncio.gather([_handle_delegation, ...])`. Three concurrent
  child frames — `researcher + analyzer + reviewer` — open at the
  same moment, stream their own LLM traffic in parallel, and
  return their bubble text in input order.

Each batch is bracketed by a pair of trace events:

```
PARALLEL_GROUP_STARTED (group_id=pg_abc, tool_count=3)
TOOL_STARTED   A    (parallel_group_id=pg_abc)
TOOL_STARTED   B    (parallel_group_id=pg_abc)
TOOL_STARTED   C    (parallel_group_id=pg_abc)
... parallel child / tool work streams in here ...
TOOL_COMPLETED A    (parallel_group_id=pg_abc)
TOOL_COMPLETED B    (parallel_group_id=pg_abc)
TOOL_COMPLETED C    (parallel_group_id=pg_abc)
PARALLEL_GROUP_COMPLETED (group_id=pg_abc, duration_ms=423)
```

The `parallel_group_id` is stamped on every inner event's
**envelope** (not just its payload), so downstream consumers can
recognize the group as one atomic unit:

- `EventStore.fork` refuses to branch at an `at_event_id` that
  falls strictly between a `PARALLEL_GROUP_STARTED` and its
  matching `PARALLEL_GROUP_COMPLETED`. Inner events have no total
  order relative to each other, so forking at one of them would
  produce an ambiguous "state right after event N" — the store
  rejects it and points the caller at the bracket ids instead
  ("fork right before the batch, or right after the join").
- The dashboard's `StepDto.parallel_group_id` surfaces the tag
  all the way to the rendered step card, where CSS gives it an
  accent-colored left border and a `∥` glyph so parallel-batched
  work visually reads as one block.
- `reconstruct()` ignores the bracket events entirely — they're
  `TRACE`, not `STATE_TRANSITION`, so replay rebuilds the same
  frame state whether or not the bracket was preserved. The
  parallelism is an execution detail, not a state fact.

The bracket pattern is intentionally transaction-shaped: open,
do work under shared identity, commit. That analogy is what
makes the fork semantics clean — a fork is "time travel to a
historical state", which is well-defined only at event ids where
the event log has a total order, and brackets are exactly the
intervals where it doesn't.

## Sessions

Every run creates or resumes a **session**, identified by a
session-scoped id. All of its events live in
`~/.nature/events/<session_id>.jsonl`.

- **Create** — `nature --new` (or just `nature` and pick "new" in the
  picker) opens a fresh session. The server wires up a
  `SessionRunner` + root `Frame` and returns the id.
- **Resume** — `nature -r <id>` or the session picker's "resume" entry.
  Archived sessions are rebuilt purely from their event log; no
  intermediate state files are needed. After resume the root frame is
  reopened for input — `AreaManager.reopen()` flips the state and
  emits a `FRAME_REOPENED` event so replay sees the transition
  explicitly. New events append on top of the existing log with the
  id counter staying monotonic.
- **Resume with overrides** — `client.resume_session(sid, model=...,
  provider=..., api_key=..., base_url=...)` swaps the LLM backend
  without touching the role or conversation history. Same shape as
  `create_session`, useful for "continue this conversation against a
  different model" or rotating an expired API key.
- **Archive** — when the daemon exits, every session's events stay on
  disk. `nature sessions list` shows them under "archived".
- **Fork (event-level branching)** — `nature sessions fork <sid> --at
  <event_id>` creates a new session whose event log starts with a copy
  of events 1..event_id from the source. Original event ids are
  preserved in the copy; new events appended to the fork continue from
  `event_id + 1`. The fork's `parent_session_id` and
  `forked_from_event_id` land in a sidecar metadata file next to the
  event log, and the dashboard renders them as a `⑂ parent@event`
  badge in the session picker that jumps to the parent on click.
  Works uniformly for live and archived source sessions.

  In the dashboard, each turn carries a `⑂ fork` button next to its
  state pill. Clicking it forks the current session at that turn's
  `last_event_id` (the highest session-monotonic event id within the
  turn) and navigates to the new session on success. The button
  stops click-propagation so it doesn't also toggle the turn's
  expansion. Mobile shows the same button as a compact `⑂` pill. The
  button's target id is pulled from the `TurnDto.first_event_id` /
  `last_event_id` pair that the view builder now surfaces on every
  turn, so running turns pick up newly-landed events on each DTO
  push — you can fork from a turn that's still streaming.

  Fork is the right primitive for "what if the model had taken a
  different decision at turn 5?" and similar time-travel experiments:
  a) it doesn't touch the base session, so repeated forks all stay
  comparable; b) the tree metadata makes the lineage explicit so
  dashboards can render a real branch tree; c) resume semantics are
  reused as-is — hydrating the forked session goes through the same
  `reopen()` path that emits a `FRAME_REOPENED` event, so the
  state-transition log is still a complete record of what happened.

  HTTP: `POST /api/sessions/{source_sid}/fork` with body
  `{"at_event_id": <int>}` — the response shape is the usual
  `CreateSessionResponse` plus `parent_session_id` and
  `forked_from_event_id`.

Resume + continuation has been verified end-to-end: a resumed session
correctly reconstructs its full frame tree (including deeply nested
sub-agents) from events alone, and subsequent delegations append new
events with id continuity.

### Hang protection

Every LLM streaming call is wrapped in `asyncio.timeout()` inside
`llm_agent`. If a provider hangs mid-stream (observed on Ollama
under specific tool_use payloads), the wrapper raises
`LLMCallTimeout` after the configured window. `AreaManager` catches
the exception, emits both `LLM_ERROR` (trace) and `FRAME_ERRORED`
(state-transition), and the frame transitions to `ERROR` — the
session's next user input starts a fresh turn on top of the existing
event log.

Configuration:

- `NATURE_LLM_TIMEOUT` environment variable (seconds, default `300`)
- `0` or negative disables the wrapper entirely
- The TUI / dashboard pulse indicators show the elapsed time so users
  can see when a turn is approaching the timeout

## Development

```bash
git clone <repo> && cd nature
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,anthropic,dashboard]"

pytest -q                          # 492 tests
ruff check nature tests            # lint
```

Test layout:

```
tests/
  test_events.py                   # event types (36), categories, store, replay round-trips
  test_area_manager.py             # Frame lifecycle + replay invariants
  test_delegation.py               # sub-frame linkage + spawn index
  test_body_compaction.py          # compaction pipeline + BODY_COMPACTED
  test_footer.py                   # footer rule pipeline (synthesis / todo nudges)
  test_todowrite_tool.py           # TodoWrite tool + TODO_WRITTEN wiring
  test_packs.py                    # Pack types, registry dispatch, phase ordering, legacy shim
  test_edit_guards.py              # fuzzy_suggest, reread_hint, loop_detector, loop_block
  test_budget.py                   # reads_budget Gate + Contributor
  test_read_memory.py              # ReadMemory segments, merge, evict, stale/expired
  test_view.py                     # SessionViewDto turn-tree builder
  test_frame.py                    # Frame dataclass
  test_frame_config.py             # frame.json parsing
  test_frame_dashboard.py          # dashboard WS consumer
  test_server_client.py            # HTTP server + NatureClient
  test_session_runner.py           # SessionRunner end-to-end
  test_llm_agent.py                # pure llm_agent() function
  test_context_refactor.py         # Context / Header / Body types
  test_provider.py                 # LLMProvider ABC
  test_provider_model_flow.py      # CLI --model override flow
  test_agent_profiles.py           # md profile loader
  test_tools.py                    # Bash, Read, Write, Edit, Glob, Grep
  test_text_tool_parser.py         # text-based tool call extraction
  test_tool_protocol.py            # Tool ABC
  test_messages.py                 # Message / ContentBlock
  test_tokens.py                   # token estimation
  test_cost.py                     # cost calculation
  test_retry.py                    # retry logic
  test_security.py                 # Bash safety checks
  test_settings.py                 # 5-tier settings merge
  ...
```

## Project structure

```
nature/
  agent/                     — pure llm_agent() + tool executor + text tool parser
  agents/                    — built-in role profiles (receptionist, core, researcher, ...)
  client/                    — HTTP client library (NatureClient)
  config/                    — settings, constants, domain types
  context/                   — Header/Body primitives, Composer, body compaction, ReadMemory
  context/footer/            — footer rule package: per-rule modules under rules/,
                               shared helpers in helpers.py, public API in __init__.py
                               (rules are auto-ported to Contributor Interventions via legacy_shim)
  events/                    — Event envelope, typed payloads (36 types), FileEventStore, reconstruct()
  frame/                     — Frame (with pack_state), AreaManager (execution loop), frame.json config, AgentTool
  packs/                     — Pack architecture: types, registry, legacy_shim
  packs/builtin/edit_guards/ — Phase 2 Edit recovery: fuzzy_suggest, reread_hint, loop_detector, loop_block
  packs/builtin/budget/      — Phase 3.1 reads budget: Gate + Contributor
  packs/builtin/file_state/  — ReadMemory integration: edit_read_first Gate + persist Listeners
  providers/                 — Anthropic (w/ prompt caching), OpenAI-compat (OpenRouter, Ollama, ...)
  protocols/                 — ABCs + data models: LLMRequest, LLMResponse, Tool, ToolContext, Message, Turn, TodoItem
  server/                    — HTTP + WebSocket daemon (ServerApp, SessionRegistry, routes, view DTOs)
  server/static/             — dashboard.html (desktop) + mobile.html (touch / Stop button)
  session/                   — SessionRunner (kicks off execution end-to-end)
  tools/                     — builtin tools (Bash, Read, Write, Edit, Glob, Grep, TodoWrite) + registry
                               (Read/Edit/Write integrate with ReadMemory via ToolContext.pack_state)
  security/bash_checks/      — per-check package: 11 checks under checks/, BashSafetyResult in types.py
  utils/                     — cost, tokens, ids, retry
  ui/                        — Frame TUI, session picker, dashboard server
  cli.py                     — Click entry point
```

## Design documents

Several design documents capture the evolution of the framework. Read
them when you need the "why" behind a particular subsystem:

| Document | Status | When to read |
|---|---|---|
| `README.md` | Current | Getting started; overview of everything that works today. |
| `ARCHITECTURE.md` | Current | High-level decomposition (Pack / Host / Agent / Preset / Frame). |
| `EXPERIMENTATION.md` | Current | How to run an experiment matrix and analyse the eval-run records. |
| `pack_architecture.md` | **M1–M3 implemented**, §17 scatter-gather is design only | Before writing a new Pack or understanding the Gate/Listener/Contributor model + phase ordering. |
| `paper/` | The published platform paper + reproducibility scripts. |

## License

This project is licensed under the Apache License 2.0 — see
[`LICENSE`](LICENSE) for details.

## Citation

If you use nature in academic work, please cite the companion paper.
The repository version of the PDF is at `paper/paper.pdf`; arXiv
identifier will be added here once posted.
