# Probe v8 — Final Analysis (15 models × 29 probes)

Run: `1776872683-80e3c9` (completed 2026-04-23)
Surgical re-run (v8s): `1776915003-7e766b` for 5 models × 3 patched probes

> This analysis supersedes v6b/v7/v7b takeaways for headline claims.

## 1. Final ranking (v8 baseline + v8s corrections applied)

| # | Model | Final | v8 raw | prior | Δ | tier |
|---|---|---|---|---|---|---|
| 1 | claude-sonnet-4-6 | **28/29** | 27 | 28 | 0 | cloud-top |
| 1 | **claude-haiku-4-5** | **28/29** | 26 | 27 | +1 | cloud-top (ties sonnet!) |
| 3 | phi4:14b | 25/29 | 25 | 26 | −1 | local-top |
| 3 | qwen2.5:72b-q4 | 25/29 | 25 | 24 | +1 | local-top |
| 5 | qwen2.5:7b | 23/29 | 23 | 19 | +4 | local-strong |
| 6 | **llama3.3:70b** | **22/29** | 22 | **3** | **+19 🚨** | local-strong (fixed by timeout mult) |
| 7 | qwen2.5-coder:32b | 21/29 | 21 | 4 | +17 | local-middle (recovered) |
| 8 | qwen2.5-coder:7b | 20/29 | 20 | 5 | +15 | local-middle (recovered) |
| 9 | qwen2.5:14b | 17/29 | 17 | 17 | 0 | local-middle |
| 9 | gemma3:27b (new) | 17/29 | 17 | — | — | local-middle |
| 9 | gemma2:27b | 17/29 | 15 | 17 | 0 | local-middle |
| 12 | llama3.2:3b | 14/29 | 14 | 13 | +1 | local-weak |
| 13 | mistral-nemo:12b | 11/29 | 11 | 13 | −2 | local-weak |
| 14 | deepseek-r1:32b | 4/29 | 4 | 1 | +3 | disqualified (reasoning trace timeout) |
| 15 | command-r7b | 2/29 | 2 | 3 | −1 | disqualified (Cohere format + hallucination) |

**Top-tier tie** (cloud) — sonnet 28/29 = haiku 28/29. First time in the sweep haiku matches sonnet. Both models lose only 2 probes:
- Sonnet: `t1-grep-specific-def` (no_tool_error — tool ran into runner error), `t3-read-long-file` (retracted probe noise — got 429 overloaded, unrelated)
- Haiku: (v8s passed everything that was failing from buggy patches; 2 genuine fails remain)

**Local champions** — phi4:14b (25/29, 86%), qwen2.5:72b (25/29, 86%) tied. qwen2.5:7b (23/29, 79%). llama3.3:70b (22/29, 76%).

### Actually changed by v8s surgical rerun
5 cells flipped FAIL→PASS:
- sonnet: t6-locate-then-edit-then-verify
- haiku: t2-todowrite-simple + t6-locate-then-edit-then-verify
- gemma2: t2-todowrite-simple + t6-todowrite-progress

phi4 and gemma3 already passed these probes in v8 (not affected by the over-tight patches).

**Disqualified** — deepseek-r1:32b and command-r7b. First from reasoning-trace timeouts even at 3×, second from Cohere action format + hallucinated tool results.

## 2. Per-dimension pass rate (across all 15 models)

| Dimension | Pass rate | Notes |
|---|---|---|
| rule.follow | 88% | Mostly T0/T7 text-only tests |
| text_baseline | 86% | Simple math "7×6" |
| structured_json | 86% | T7 JSON verdicts, orthogonal skill |
| delegation.format | 64% | Half of models struggle |
| tool.emit | 61% | Widespread bottleneck |
| reasoning | 56% | |
| conditional | 56% | |
| edit.discipline | 48% | Writing files precisely is hard |
| tool.result.consume | 46% | Synthesizing tool output is hard |
| multi_turn_state | 43% | Weakest — genuinely hard across the board |

**Tool emission is the first gate** (61% pass). Models that can't emit structured tool calls fail everything downstream. The gap between emission (61%) and consumption (46%) means even models that *can* call tools often can't use the results properly.

## 3. Tier-wise difficulty (across all models)

| Tier | Pass rate | Character |
|---|---|---|
| T0 | 80% | Echo one tool call |
| T1 | 71% | Echo + slight arg reasoning |
| T2 | 42% | **Biggest drop** — chain two tools |
| T3 | 58% | Edits (file_state closes the loop) |
| T4 | 56% | Conditional based on tool result |
| T5 | 64% | Agent delegation (surprisingly doable) |
| T6 | 51% | Multi-step state |
| T7 | **100%** | Structured JSON (no tools) — orthogonal |
| T8 | 33% | **Hardest** — bug fix with edit precision |
| T9 | 50% | Multi-file refactor |

**T2 is the make-or-break tier.** Weak models can echo one tool (T0-T1) but can't chain (T2). This is the capability cliff.

**T7 at 100%** confirms JSON-structured output is orthogonal to tool-use capability — even the weakest model passes both T7 probes. Separate capability axis; ranking should track separately.

**T8 at 33%** is the genuine bug-fix tier. Even strong models often mis-edit. This is where `edit.discipline` × `reasoning` actually matter.

## 4. Hardest probes (failure distribution)

| Probe | Pass | Dominant failures |
|---|---|---|
| t2-grep-then-read | 4/15 | tool_use:7 (chain breaks) |
| t8-add-missing-import | 4/15 | file_state:10 (edit precision) |
| t3-read-long-file | 5/15 | final_text:8 (retracted probe) |
| t6-locate-then-edit-then-verify | 6/15* | tool_use:7 (inflated by v8 patch bug; v8s fixes) |
| t2-glob-then-read | 6/15 | tool_use:5 + final_text:4 |
| t8-swap-operator | 6/15 | file_state:8 |
| t9-two-step-refactor | 7/15 | file_state:8 |

**T8 dominates file_state failures** — models can read, delegate, JSON-format, but can't precisely edit. This is the most stable weakness signal in the data.

## 5. Cross-cutting findings

### 5.1 llama3.3:70b was a pure infrastructure artifact
Initial v6b: 3/29. Post-4× timeout multiplier: 22/29 (+19). The "70B worse than 3B at agent loops" narrative was **entirely** a per-probe timeout budget too small for cold-load TTFT on consumer hardware. With the multiplier:
- T0-T1: 8/8 pass
- T2: 1/3 pass (still some chains time out at 360s)
- T3+: mixed (genuine capability there)

This is the cleanest infrastructure-as-variable example in the set.

### 5.2 Coder fine-tune penalty is real but smaller than reported
qwen2.5:72b-instruct (25) vs qwen2.5-coder:32b (21) = **4-probe gap (14%)**, not the 6× gap from v6b (4 vs 24). The earlier chasm was infrastructure: wrapper catalog format + parser case-sensitivity catastrophically combined to suppress coder output.

Post-fix, coder variants land in the middle tier — modestly weaker than base, but usable.

### 5.3 deepseek-r1:32b is legitimately disqualified
With 3× timeout multiplier, still 4/29 passes. The failure pattern:
- 13 timeouts even at extended budgets
- T9 probes hit 27-53 minute runtimes (reasoning trace accumulates across turns)
- When it *does* produce output, it's often correct — but budget is the killer

Reasoning-trace-centric models don't fit current tool-use timeouts. A different evaluation regime would be needed (per-turn timeouts rather than per-probe).

### 5.4 command-r7b disqualified for safety reasons
2/29, unchanged across iterations. Emits Cohere-specific action format that no parser handles. **Critically — hallucinates tool results without emitting tool calls.** Verified on t0-read-exact and t1-bash-simple: model narrated "contents of hello.txt: line 1..." without ever calling Read, with fabricated content. This is strictly worse than "clean failure" — user sees cooperative output that is actually lies.

### 5.5 Gemma 2 → 3 modest improvement
gemma2 (15/29) vs gemma3 (18/29 after v8s) = +3. New family iteration helps, but the core "tool-call-as-final-answer" pattern persists in both. Gemma family trained with weak tool-use signal; structural.

### 5.6 Provider abstraction as measurement layer
Four models had score jumps of +15 or more after capability-config changes:
- phi4 3 → 26 (wrapper pattern added)
- coder:32b 4 → 21 (case-insensitive parser)
- coder:7b 5 → 20 (same)
- llama3.3:70b 3 → 22 (timeout multiplier)

Any published benchmark that doesn't disclose its adapter stack per model is incomplete. This is the paper's headline methodological claim.

### 5.7 Query pass-through failure (newly observed)
Separate from probes but paper-worthy: Claude Sonnet 4.6 (the model itself) rewrote user's query "Claude Sonnet 4.6 pricing" to "Claude Sonnet 4.5 pricing" before delegation, because its training cutoff predates its own release and it "knew" 4.6 didn't exist. Training-data authority overriding user authority — a distinct failure mode from hallucination. Mitigated by adding explicit pass-through guards to core.md + researcher.md.

## 6. Measurement fidelity — fix history

| Fix | Models affected | ΔPass | Root cause |
|---|---|---|---|
| TextToolAdapter wrapper (v7) | phi4, gemma2 | +21, +14 | Ollama 400 error unhandled |
| Case-insensitive parser (v7b) | coder:32b, coder:7b, phi4 | +17, +15, +2 | Model emits lowercase tool names |
| Stream timeout multiplier (v8) | llama3.3:70b, r1 | +19, +3 | TTFT > probe budget |
| Case-insensitive todowrite regex (v8s) | haiku, gemma2, gemma3 | +1-2 each | Over-tightened criterion |
| Removed Grep requirement on t6 (v8s) | sonnet, haiku, gemma2, gemma3 | +1 each | Over-tightened criterion (Read was also valid) |
| Probe schema tightening (v7b) | all models | −1-2 each | Caught genuine false positives |

## 7. Recommendations for nature routing

**Cloud-tier roles (core, implementer):** sonnet (28/29). Haiku (28/29) is cost-competitive; drop in as default for cost-sensitive variants.

**Local-tier roles with network access:** haiku (cloud) + phi4:14b (local read/analyze) hybrid preset `haiku-phi4`.

**Fully-offline fallback:** `all-local-phi4` (root = core, all roles phi4 except implementer which needs precision → qwen2.5:72b). phi4's 100% tool.emit makes it the single best universal-tool local model.

**Researcher-scout / researcher-reader:** Already routed to phi4:14b (best local tool.emit + tool.result.consume). Validated.

**Disqualified for any role:** deepseek-r1:32b, command-r7b, mistral-nemo:12b (11/29 is too weak). gemma variants ok for structured_json-heavy roles but weak for multi-turn.

## 8. Paper implications

Headline findings, in priority order:

1. **"Infrastructure configuration is a measurement variable"** — 4 models with ≥+15 score change from capability-config alone. Published benchmarks that don't disclose adapter stacks are incomplete. (§5.5 of paper)
2. **"Coder fine-tunes are moderately weaker, not catastrophic"** — revised from earlier "6× gap" to "14% gap after controlling for infrastructure". The original narrative was an artifact. (§5.5 case 2)
3. **"Agent capability ≠ model size"** — with llama3.3:70b now at 22/29 (post-timeout-fix), the clean example becomes qwen2.5:7b (23/29) > qwen2.5:14b (17/29). Non-monotonic within a family. (§5.5 case 3)
4. **"Prompt engineering cannot override training-level defaults"** — gemma2 synthesis-enforcement prompt had zero effect. (§5.5 synthesis prompt footnote)
5. **"Training cutoff authority vs user authority"** — Claude 4.6 self-rewrites to 4.5 when asked about its own version. New failure mode surfaced in live session, distinct from standard hallucination. (§5.5 case 5 — new)
6. **T2 is the capability cliff** — chain-two-tools is where weak models fail. T7 (JSON) is orthogonal and 100% passed. Tier ladder has two axes, not one.

## 9. Remaining work

- Surgical v8s results (5 × 3 cells) — in progress, updates this doc
- Paper §5 (preset-level benchmark) — still placeholder
- Paper §6 (ablations) — today's Korean session = event-pinned fork demo material (default→direct-core, then 4.6-rewrite fix)
- Paper §7-9 (discussion, limits, conclusion) — not yet drafted
- Abstract update — current one is 2026-04-19, predates all v8 findings
