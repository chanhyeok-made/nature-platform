# Experimentation

> **Audience.** Anyone running controlled experiments with nature —
> paper authors, framework developers, or teams choosing a production
> configuration.
>
> **Prerequisite.** Read [ARCHITECTURE.md](ARCHITECTURE.md) first; this
> document assumes the core abstractions (Host, Agent, Preset, Pack,
> Frame) are familiar.

nature is built so that every major axis of an LLM agent system is a
swappable, file-based artifact. This document is the practical guide
to *isolating* each axis and running reproducible comparisons.

---

## 1. The variables you can isolate

| Variable | Artifact that holds it | How to swap it |
|---|---|---|
| **Model choice per role** | `Preset.model_overrides[agent]` | Point the override at a different `host::model`. |
| **Endpoint / provider** | `hosts.json` | Add or redirect a Host, reference it via `host::model`. |
| **Prompt per role** | `Preset.prompt_overrides[agent]` + `agents/instructions/<stem>.md` | Point the override at a different instruction stem; the agent's other fields (model, tools) stay. |
| **Tool permission per role** | `agents/<agent>.json::allowed_tools` | Same as above — new agent, new preset. |
| **Tool surface (framework-wide)** | Packs (each Pack may contribute tools) | Install / remove a Pack; `allowed_interventions` on agents can narrow further. |
| **Intervention (gate / listener)** | Packs | Install / uninstall the Pack. |
| **Orchestration topology** | `Preset.root_agent`, `Preset.agents` | New preset with different roster. |
| **Prior state (counterfactual)** | Event log | Fork at any `event_id` (§4). |
| **Context compaction policy** | `BodyCompactionPipeline` strategies + `models.json` budgets | Add / reorder strategies on the pipeline; per-model budgets come from `models.json`. |

Every entry above is a **file** — no code change required to vary it.

---

## 2. Recipes

### 2.1 Compare whole configurations (preset-level benchmarking)

Run each `(task × preset)` pair from a shared task catalogue:

```bash
nature eval run \
    --task t1 --task t2 --task t3 \
    --preset default --preset haiku-qwen-reader --preset all-haiku \
    --seeds 3
```

Each cell is executed `seeds` times; aggregation (pass rate, mean
cost, mean latency, stdev) happens at report time so raw cells are
preserved for re-analysis.

### 2.2 Swap one model for one role

Copy the preset, flip one `model_overrides` entry:

```jsonc
// .nature/presets/haiku-qwen-reader.json
{
  "root_agent": "receptionist",
  "agents": [...],
  "model_overrides": {
    "researcher": "local-ollama::qwen2.5-coder:32b"
  }
}

// .nature/presets/haiku-deepseek-reader.json
{
  "root_agent": "receptionist",
  "agents": [...],
  "model_overrides": {
    "researcher": "local-ollama::deepseek-r1:32b"
  }
}
```

Run both presets on the same task catalogue. The post-matrix diff
shows the effect of the single swap.

### 2.3 Swap one agent's prompt

Drop the variant instruction file and point at it from a new preset:

```
~/.nature/agents/instructions/researcher-stripped.md    # the new prompt body
```

```jsonc
// ~/.nature/presets/haiku-qwen-reader-stripped.json
{
  "root_agent": "receptionist",
  "agents": ["receptionist", "core", "researcher", "analyzer", ...],
  "model_overrides": { "researcher": "local-ollama::qwen2.5-coder:32b" },
  "prompt_overrides": { "researcher": "researcher-stripped" }
}
```

At session-build time the researcher's model stays on qwen, tools
stay as defined in `researcher.json`, but the prompt body is
`researcher-stripped.md` instead of the default. Running the original
vs stripped preset on the same tasks isolates the prompt change.

The stem must be a bare filename (no path separators, no `.md`
extension); resolution is project → user → builtin, same as agent
JSON loading.

### 2.4 Swap a tool surface

Tools live inside Packs. Add a custom pack to the user layer:

```
~/.nature/packs/my-guards/
    pack.json      # manifest
    pack.py        # exposes install(registry)
```

`install_discovered_packs` picks up the new pack at `AreaManager`
construction. To run an experiment with vs without the pack, pair two
invocations — one using a NATURE_HOME that has the pack directory,
one without. (The same pattern also cleanly isolates builtin pack
ablations: pass your own empty `PackRegistry` to `AreaManager` and
install only the packs under test.)

### 2.5 Event-pinned counterfactual (§ 4.2 of the paper)

The key lever. Steps:

```python
# 1. Record a baseline session normally.
baseline = await client.create_session(preset="default")
await client.send_message(baseline.session_id, "user prompt")

# 2. Inspect the event log; pick a decision point.
snap = await client.snapshot(baseline.session_id)
# e.g., right after event 12 — core delegates to researcher.
fork_event_id = 12

# 3. Fork into as many branches as you like, each under a different preset.
branch_a = await client.fork_session(
    baseline.session_id, at_event_id=fork_event_id,
    preset="haiku-qwen-reader",
)
branch_b = await client.fork_session(
    baseline.session_id, at_event_id=fork_event_id,
    preset="all-haiku",
)

# 4. Let each branch continue to completion; measure post-fork delta.
```

Both branches share the byte-identical prefix `events[1..12]`; the
only variable that differs is the preset after event 12. Any
difference in final cost / latency / verdict attributes to that
single change — no confounded pre-fork noise.

> **Practical note.** Pick a fork event that happens *before* the
> role whose behaviour you're probing makes its first LLM call.
> Forking mid-response of an agent you're also swapping produces
> ambiguous results.

### 2.6 Replay an old session for analysis (no mutation)

```python
from nature.events.reconstruct import reconstruct
from nature.events.store import FileEventStore

store = FileEventStore(root=Path.home() / ".nature" / "events")
replay = reconstruct("<session-id>", store)

# replay.frames — dict of frame_id → Frame, with conversation history
# replay.frames[root_frame_id].ledger — confirmed facts, ReadMemory, etc.
```

Used by the dashboard's session view, by `nature eval runs list`, and
for paper-time metric mining.

### 2.7 Vary the context-compaction policy

Every session's `AreaManager` holds a `BodyCompactionPipeline` that
runs before each LLM call. The pipeline checks the body's token
estimate against the frame's model budget (resolved from
`nature/config/builtin/models.json` → per-`host::model` context
window and output reservation) and invokes strategies in order until
the estimate drops back under the autocompact threshold.

Default strategies (installed by `SessionRegistry._build_compaction_pipeline`):

1. `MicrocompactBodyStrategy` — zero-LLM, replaces old `tool_result`
   blocks with a placeholder. Lossy but cheap.
2. `DreamerBodyStrategy` — LLM-summarizes the prefix (everything
   older than the last six self-actor turns). The raw slice is also
   dumped to `<cwd>/.nature/ltm/<session_id>/<role>-<ts>.md` so the
   agent can `Read` it back on demand.

To experiment, swap the strategy list in a custom SessionRegistry or
drop in a new `BodyCompactionStrategy` subclass that replaces one of
the default two — the pipeline sees every strategy via the same
protocol, so no framework change is required. Budgets for new models
land in `models.json`; unknown models fall back to a conservative
200 k / 20 k default.

Every compaction emits a `BODY_COMPACTED` event with the full post-
compaction body snapshot, so replay lands on the same trimmed body
and paper-time analysis can attribute cost savings to specific
compactions.

---

## 3. Measurement

### 3.1 Event log is the canonical source

Every analytical surface — cost, latency, topology, compaction,
provenance — is derived from the session's append-only event log.
`CellResult` fields are a **cache** of what the log says, not the
source of truth. Two invariants fall out:

1. Every metric on `CellResult` has a corresponding extractor in
   `nature.eval.runner._extract_metrics`. Adding a new metric means
   extending that extractor, not instrumenting the runtime.
2. Historical runs can be re-analyzed without re-executing any
   session via `nature eval rebuild-metrics --run <id>`, which
   walks each cell's archived `.jsonl` log and re-stamps the
   metric fields.

Practical corollary: if a fact isn't in the event log, it isn't
analytically recoverable. The one exception is the acceptance
verdict (pytest exit or judge result) — acceptance runs after the
agent session has closed and stays on the `CellResult` alongside
the logs.

The event `SESSION_STARTED` carries the provenance snapshot
(preset composition, per-agent resolved definitions including
instruction text, hosts, models, token budgets, repo sha, fork
lineage). So every session log is self-describing: a reader with
just the `.jsonl` file can reconstruct what configuration produced it.

### 3.2 Extracted metrics

`_extract_metrics(session_jsonl)` walks an event log once and
emits the full metric set the platform records per cell:

| Metric | Source events |
|---|---|
| `cost_usd` | sum over `llm.response.usage` × per-model pricing |
| `cost_by_agent` | same, attributed via the frame → role map from `frame.opened` |
| `tokens_in` / `tokens_out` / `cache_read_tokens` | `llm.response.usage` fields |
| `cache_hit_rate` | `cache_read_tokens / (tokens_in + cache_read_tokens)` |
| `turn_count` | count of `llm.response` |
| `tool_call_count` | count of `tool.completed` |
| `tool_error_count` | count of `tool.completed` with `is_error=true` |
| `provider_errors` | count of `llm.error` |
| `body_compactions` | count of `body.compacted` |
| `avg_turn_latency_ms` | mean of `annotation.stored.duration_ms` |
| `sub_frame_count` | count of `frame.opened` with non-null `parent_id` |
| `max_delegation_depth` | longest chain via `frame_opened.parent_id` |
| `agents_used` | distinct `role_name` from `frame.opened` |
| `source_session_id` / `forked_from_event_id` | `session.started` payload |

Everything is post-hoc — no runtime hook is required to compute
these, so a session created by any client (CLI, HTTP, test harness)
is analyzable under the same metric definitions.

---

## 4. Reporting and comparison

### 4.1 Single run

```bash
nature eval report --run <run-id>
```

Produces a markdown matrix (tasks × presets), a per-preset summary
(pass rate, Σ cost, avg latency, Σ turns, Σ tool-calls, error count),
and inline cell markers for multi-seed aggregation
(`pass_count / seed_count  mean_cost  mean_latency  mean_turns`).

### 4.2 Two runs (diff)

```bash
nature eval diff <run-a-id> <run-b-id>
```

Pairs cells by `(task_id, preset)` and prints signed percent deltas on
cost / latency, turn and tool counts before-and-after, and dedicated
sections for regressions (PASS → FAIL) and fixes (FAIL → PASS). Cells
present in only one run surface as "unmatched".

### 4.3 Dashboard UI

`/eval` (served by the local `nature server start`) renders the same
data interactively: run list on the left, matrix + summary on the
right, per-cell link into the session replay view, and a dropdown to
diff two runs inline.

---

## 5. Sharing results

Every artifact involved in an experiment is file-based:

- **Task definitions**: `nature/eval/builtin/cases/<id>/` or user /
  project layer. Ship with `task.json` + any `seed.patch` / fixture
  files the setup strategy needs.
- **Presets, agents, packs, hosts**: same three-layer discovery. Drop
  a JSON or a directory, done.
- **Run records**: `.nature/eval/results/runs/<run-id>.json` —
  self-contained (repo sha, task ids, preset names, all cell metrics,
  notes). No external DB required to replay or compare.
- **Session logs**: `<project>/.nature/eval/results/logs/<cell-tag>.jsonl`
  plus `~/.nature/events/<sid>.jsonl` (hydrated on each run so the
  dashboard can open any cell's session after the fact).

To publish a result set: commit the run JSON + the cell logs + the
task and preset directories referenced in the run. A reader can
reconstruct every session in the matrix byte-for-byte.

---

## 6. Worked example: a single ablation

Suppose we want to know whether switching the `researcher` role from
haiku to local qwen changes cost more than it changes pass rate, on a
specific task set.

1. **Establish the baseline.** Preset `all-haiku` exists and is
   stable; pick a task catalogue (e.g., `s1-csv-parser`,
   `x1-pluggy-remove-plugin`, `n1-pack-discovery`).

2. **Create the variant.** A single-file change:

   ```jsonc
   // .nature/presets/haiku-qwen-reader.json
   {
     "root_agent": "receptionist",
     "agents": ["receptionist", "core", "researcher", "analyzer",
                "implementer", "reviewer", "judge"],
     "model_overrides": {
       "researcher": "local-ollama::qwen2.5-coder:32b"
     }
   }
   ```

3. **Run both presets.**

   ```bash
   nature eval run \
       --task s1-csv-parser --task x1-pluggy-remove-plugin \
       --task n1-pack-discovery \
       --preset all-haiku --preset haiku-qwen-reader \
       --seeds 3
   ```

4. **Render and compare.**

   ```bash
   nature eval report --run <latest>
   nature eval diff <baseline-run> <variant-run>
   ```

5. **Inspect notable cells.** `nature eval runs list` shows the
   run id; the `/eval` dashboard exposes per-cell session replay for
   any row that surprises. Every conversation, tool call, and
   intervention is inspectable from the event log.

6. **Optional: event-pinned follow-up.** If the matrix shows a large
   delta on one task, fork the relevant session at the point where
   the researcher was dispatched, under both presets, and confirm the
   delta persists in the clean counterfactual. This separates "the
   researcher swap changed *this specific decision*" from "the rest of
   the orchestration happened to run differently by chance".

---

## 7. What nature explicitly does *not* help with

- **Prompt engineering in isolation.** nature can swap a role's
  prompt file and measure the effect, but doesn't optimize prompts
  for you (no auto-tuning loop). If you want systematic prompt search,
  build that on top.
- **Model training or fine-tuning.** nature benchmarks configurations
  of existing models; it doesn't train, fine-tune, or distill.
- **Multi-user or multi-tenant isolation.** A session is assumed to
  belong to a single operator. There is no ACL layer.
- **Cross-machine distributed runs.** The current runner spawns a
  local server per cell; distributing across machines requires an
  external scheduler.

These boundaries are deliberate and keep the experimentation model
well-defined.
