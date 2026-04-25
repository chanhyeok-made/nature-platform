# Paper outline

**Working title:** nature: A platform for systematic experimentation
with LLM agent systems

**Target venue:** arXiv — primary `cs.SE`, cross-list `cs.LG`.
(Primary `cs.SE` avoids the endorsement requirement that `cs.LG`-primary
submissions face for first-time arXiv authors.)

**Length target:** 10-14 pages (NeurIPS workshop template, LaTeX).

---

## 1. Introduction (1-1.5 pages)

- LLM agent systems are production realities, not research toys.
- Configuration space is huge: prompts × tools × models × orchestration
  topology. Every axis interacts with every other.
- Motivating example: a "researcher" role on haiku vs qwen-local is not
  a model comparison — it's a multi-host routing decision that shifts
  the cost/latency frontier in ways that single-model benchmarks cannot
  detect.
- Thesis: progress in this space needs a platform where each variable
  can be isolated, swapped, and measured — not just an application
  framework, not just a static benchmark.
- Contributions:
  1. An architecture that treats each configuration axis as a
     first-class, composable, file-based artifact.
  2. Event-sourced execution that makes every run reproducible,
     post-hoc analyzable, **and — uniquely — supports event-pinned
     counterfactuals: forking any recorded session at a chosen
     decision point, continuing under a mutated configuration, and
     attributing the resulting delta cleanly to the post-fork
     variable.**
  3. Illustrative experiments — a preset-level benchmark plus targeted
     ablations (two of them event-pinned) — that show the platform in
     use.

## 2. Related work (1-1.5 pages)

Group by what they optimize for:

- **Application frameworks** — LangChain, LlamaIndex, CrewAI,
  AutoGen. Optimized for building apps; not for systematic variation.
- **Production monitoring / eval** — LangSmith, Arize, Weights &
  Biases (prompts). Focused on live traces and pipeline-level eval, not
  on isolated-variable experiments.
- **Static benchmarks** — SWE-bench, AgentBench, HumanEval,
  MLE-bench. Compare models on a fixed task set; don't measure
  configuration variance.
- **Experiment tracking (ML)** — MLflow, W&B. Track parameters +
  metrics but are model-level, not agent-topology-aware.
- **Adjacent: MCP / tool ecosystems** — standardize the interface but
  don't address experimentation.

Gap: *composable, file-based, reproducible experimentation across every
axis of an LLM agent system* — nature's claimed niche.

## 3. nature architecture (2-3 pages)

The paper's technical core. Each subsection frames the design decision
in terms of **what variable it isolates**.

- **3.1 Pack** — interventions (Gate / Listener / Contributor) bundled
  as a file-based, discoverable unit. Isolates *behavior policy* as a
  swappable variable.
- **3.2 Host** — model endpoint + model catalog decoupled from the
  role that uses them. Isolates *model choice* as a swappable variable.
- **3.3 Agent** — role definition (JSON + instructions MD). Isolates
  *prompt / tool permission / role contract* as a swappable variable.
- **3.4 Preset** — the declarative composition of Agents + model
  overrides + roster. The unit at which whole configurations are
  compared.
- **3.5 Event-sourced frames** — every decision is an appended event,
  every run is replayable, and metrics are extracted from the log
  rather than from runtime introspection.
- **3.6 Memory compaction pipeline** — `BodyCompactionPipeline` runs
  before every LLM call, gated by a per-model budget resolved from
  `models.json`. Strategies (Microcompact, Dreamer) are swappable
  `BodyCompactionStrategy` instances; every compaction emits a
  `BODY_COMPACTED` event whose payload is the post-compaction body
  snapshot, so replay is byte-identical. Dreamer additionally
  persists the raw pre-compaction slice to a session-scoped
  long-term-memory directory and references it from the summary
  message, keeping detail recallable. Isolates *context compaction
  policy* as a swappable variable.

A single architecture diagram ties these together.

## 4. Experimentation primitives (1.5-2 pages)

How the platform supports the isolation story in practice.

### 4.1 Swap-and-run (whole-session comparison)

Replace one Pack / Host / Agent / Preset and re-execute from scratch.
Cheap, but every source of per-run variance (sampling, tool state,
external call ordering) is folded into the reported delta.
Appropriate for broad surveys like the preset matrix in §5.

### 4.2 Event-pinned counterfactual divergence

Every run's decisions are appended events, so a recorded session can
be **forked at any event id**; nature then replays the prefix into a
new session and continues from the fork point under a mutated
configuration (different Preset, model host, or Pack set). Because
the pre-fork state is byte-for-byte shared across branches, the post-
fork metric delta attributes to the single variable that changed.

This is the key methodological lever the event-sourced design
unlocks. Counterfactuals like "what if we had swapped the researcher's
model at turn 3 of *this* specific task execution" become a single
API call rather than a confounded two-session comparison.

### 4.3 Replay with modification

`reconstruct()` rebuilds any past frame tree from its jsonl log.
Useful for post-hoc metric extraction, tool behavior analysis, and
regression testing of framework-level changes against an archive of
historical runs.

### 4.4 Shareable artifacts

Pack, Agent, Preset, Host, and task definitions are file-based with
strict schemas. Experiments, their recipes, and their results can be
version-controlled and shared verbatim — the file layout doubles as
an exchange format.

## 4.5 Probes: component-level capability artifacts (new)

Added in M4.4–M4.11 to complement preset-level benchmarking. Probes
are JSON-defined single-axis capability tests (tiered T0-T9, with a
parallel structured-output track T7). Each probe isolates one
dimension (`tool.emit`, `edit.discipline`, `multi_turn_state`, etc.)
and uses structural success criteria (`tool_use`, `tool_not_used`,
`file_state`, `final_json`) rather than LLM-judged pass/fail, so
runs are deterministic and replayable. The probe runner composes
with Host + model selection the same way `nature eval` does — a
probe is just a smaller, narrower task.

See §5.5 for the measurement-fidelity case studies this enables.

## 5. Case study: preset-level benchmark (2 pages)

- Setup: `N` tasks × `M` presets × 3 seeds.
- Task taxonomy: nature-local (commit-based), external repo
  (pluggy-class), synthetic bundles.
- Presets: `default`, `all-haiku`, `all-sonnet`, `haiku-qwen-reader`,
  `all-qwen-coder-32b`, `solo-haiku`. (Final list subject to change.)
- Metrics: pass rate, cost, latency, turn count, tool-call count,
  max delegation depth, cache-read share, agents-used trace.
- Results: matrix heatmap + per-preset summary. Cost-latency Pareto
  scatter. Per-task breakdown where variance tells a story.
- Headline: configuration choice produces cost-latency variance that
  dominates the model-choice component of the same decision.

## 5.5 Case study: component-level probes across 15 local models (new)

See `section_probe_measurement.md` for the full draft. Key content:

- **Framework** (§5.5.2): tier ladder, dimension vocabulary, six
  success-criterion types, workspace templating.
- **Measurement fidelity cases** (§5.5.3): 4 models whose initial
  scores were infrastructure artifacts, with root-cause + fix for
  each. phi4:14b 3→26, coder:32b 4→21, llama3.3:70b 3→10+,
  command-r7b unchanged (disqualified for hallucinated tool calls).
- **Cross-cutting findings** (§5.5.4): prompt ≠ capability
  (gemma2 synthesis failure), coder fine-tunes moderately weaker,
  non-monotonic size effect in qwen family, reasoning label ≠
  tool-use success.
- **Methodology lessons** (§5.5.5): dimension audit (long_context
  retraction), success-criteria audit (3 false-positive fixes),
  orthogonality note for T7.
- **Role routing** (§5.5.6): weighted-sum per-dimension scoring
  translates probe data into per-role recommendations. Ships with
  three probe-data-grounded presets (`all-local-phi4`,
  `haiku-phi4`, `all-local-qwen72b`).

This is the paper's strongest methodological claim: **infrastructure
configuration is a measurement variable**. Benchmark reports that
don't disclose the adapter / wrapper stack they ran under are
incomplete.

## 5.6 Case study: cloud cross-provider probes across 11 models (new)

See `section_probe_cloud.md` for the full draft. Key content:

- **Setup** (§5.6.1): 11 cloud models × 32 probes + multi-seed on 12
  borderline probes via OpenRouter, $2.56 total.
- **Universal ceiling** (§5.6.3): `t3-32k` + `t4-128k` fail 11/11 —
  cloud models share the same repetitive-long-context retrieval
  ceiling as local 70B models.
- **Provider-family traits** (§5.6.4): Gemini paraphrase,
  Mixtral path fabrication, gpt-4o conservative pagination,
  llama schema-adherence instability, 30-40× verbosity spread,
  reasoning-tax latency invisible in token counts.
- **Methodological finding** (§5.6.5): probe regex over JSON text
  is vulnerable to dict-key ordering non-determinism; ~2% flake
  floor attributable to serialization variance alone.
- **Platform friction** (§5.6.6): `qwen-2.5-coder-32b` via OpenRouter
  routes through Cloudflare and is non-functional (404 on tools,
  mid-stream 500 on text). Diagnosis + capability-table fix promoted
  to a first-class observation rather than a silent drop.
- **Contrast with local sweep** (§5.6.7): cloud infrastructure
  absorbs the heterogeneous-backend problem; cloud distribution
  is narrower; stability vs capacity are different failure axes.

Paired with §5.5, this gives the paper a two-track measurement story:
local sweep exposes infrastructure-configuration sensitivity, cloud
sweep exposes provider-family qualitative traits and cross-seed
stability.

## 6. Illustrative ablations (1.5-2 pages)

Each ablation is a capability demonstration rather than an exhaustive
study. Two of the three use event-pinning (§4.2) so the reported
delta is isolated to the variable that changed.

### 6.1 Prompt ablation (event-pinned)

- Setup: run a baseline session for a fixed task/preset up to the
  point where the researcher agent receives its delegation prompt.
- Fork at that event id; in each branch, substitute a different
  researcher prompt variant; continue under identical model/tool/preset.
- Compare post-fork metrics. The delta is attributable to the prompt
  change alone.

### 6.2 Model swap ablation (event-pinned)

- Setup: baseline session up to the pre-LLM point of a chosen frame.
- Fork; branch A keeps the original model, branch B swaps in a
  different host (e.g., haiku → qwen-local) for that frame going
  forward.
- Compare completion quality and cost post-fork.

### 6.3 Solo vs multi-agent (whole-session)

- Because single-agent and multi-agent configurations diverge from
  the first event, event-pinning does not apply; §4.1's swap-and-run
  comparison is used instead.
- Useful as a contrast: same task, same tooling, radically different
  orchestration topology. Explicitly framed as the limit case where
  event-pinning is structurally unavailable.

## 7. Discussion (1 page)

- What the case study does *not* claim: no winning preset, no model
  ranking, no generalization beyond the measured subset.
- Threats to validity: prompt-model coupling, task bias, seed count,
  acceptance-evaluator stochasticity.
- Implications for practitioners: default framework thinking should
  shift from "which model" to "which configuration, given costs."
- Community angle: file-based artifacts invite contributed Packs,
  Presets, Agents, and task catalogs; nature's layout is designed to
  support that accumulation.

## 8. Limitations + future work (0.5-1 page)

- Tool coverage and prompt catalogue are currently nature-supplied;
  third-party prompts/tools are a future axis.
- Benchmark scale is proof-of-concept; a larger community-curated task
  catalog is the next milestone.
- Judge-based acceptance (llm_judge) needs its own validation study.
- Community features (result registry, public dashboards) are outlined
  but not implemented in this paper's scope.

## 9. Conclusion (0.25 page)

- The combinatorial problem is real and currently under-served.
- Variable isolation + file-based artifacts + event sourcing is one
  coherent answer.
- nature is open-source; invitation to contribute.

---

## Figures to produce

1. **Architecture diagram** — Pack / Host / Agent / Preset +
   event-sourced runner, with arrows showing compose and dispatch.
2. **Matrix heatmap** — tasks (rows) × presets (columns), cell color
   by pass rate.
3. **Pareto scatter** — average cost (x) vs avg latency (y), one dot
   per preset, labeled.
4. **Per-task breakdown bars** — cost per preset per task, ordered.
5. **Ablation comparisons** — 2-3 small paired bar charts.
6. **Event-pinned fork diagram** — timeline of a baseline session
   with a fork at event N, two branches diverging under different
   configurations, and the post-fork metric delta annotated. Visually
   anchors §4.2 and the event-pinned ablations in §6.

## References target

~25-40 entries. Anchor citations:

- SWE-bench, AgentBench, HELM, HumanEval, MLE-bench.
- LangChain, LlamaIndex, CrewAI, AutoGen.
- LangSmith, Arize, MLflow, W&B.
- Event sourcing (Fowler), CQRS / DDD references as needed.
- MCP spec, tool-use Anthropic papers.
- Classic software middleware (Rack, Express, Redux middleware) for the
  Pack taxonomy lineage paragraph.
