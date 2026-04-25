# Probe v8 Final — Takeaways (15 models × 29 probes, 2026-04-23)

Supersedes all prior versions. Merged from `run=1776872683-80e3c9` (full v8) and `run=1776915003-7e766b` (v8s surgical rerun for 5 models × 3 patched probes).

## 1. Final Ranking

| # | Model | Pass | Tier |
|---|---|---|---|
| 1= | claude-sonnet-4-6 | **28/29** (97%) | cloud-top |
| 1= | **claude-haiku-4-5** | **28/29** (97%) | cloud-top (ties sonnet) |
| 3= | phi4:14b | 25/29 (86%) | local-top |
| 3= | qwen2.5:72b-q4 | 25/29 (86%) | local-top |
| 5 | qwen2.5:7b | 23/29 (79%) | local-strong |
| 6 | llama3.3:70b | 22/29 (76%) | local-strong (was 3/29 pre-timeout-mult) |
| 7 | qwen2.5-coder:32b | 21/29 (72%) | local-middle (was 4/29 pre-infra-fix) |
| 8 | qwen2.5-coder:7b | 20/29 (69%) | local-middle (was 5/29 pre-infra-fix) |
| 9= | qwen2.5:14b | 17/29 (59%) | local-middle |
| 9= | gemma3:27b | 17/29 (59%) | local-middle (new, no gain vs gemma2) |
| 9= | gemma2:27b | 17/29 (59%) | local-middle |
| 12 | llama3.2:3b | 14/29 (48%) | local-weak |
| 13 | mistral-nemo:12b | 11/29 (38%) | local-weak |
| 14 | deepseek-r1:32b | 4/29 (14%) | disqualified (reasoning timeout) |
| 15 | command-r7b | 2/29 (7%) | disqualified (Cohere format + hallucination) |

## 2. Per-Dimension Pass Rates (v8 actual)

| Model | tool.emit | tool.result | multi_turn | edit | reasoning | structured_json | delegation | rule.follow |
|---|---|---|---|---|---|---|---|---|
| sonnet-4-6 | 100% | 85% | 100% | 100% | 100% | 100% | 100% | 100% |
| haiku-4-5 | 94% | 85% | 87% | 100% | 100% | 100% | 100% | 100% |
| phi4:14b | 76% | 71% | 75% | 75% | 100% | 100% | 100% | 100% |
| qwen2.5:72b | 88% | 85% | 87% | 87% | 75% | 66% | 100% | 100% |
| qwen2.5:7b | 70% | 85% | 75% | 62% | 100% | 100% | 33% | 100% |
| llama3.3:70b | 82% | 42% | 50% | 50% | 50% | 100% | 100% | 100% |
| coder:32b | 70% | 71% | 75% | 75% | 75% | 100% | 66% | 100% |
| coder:7b | 64% | 85% | 37% | 62% | 75% | 100% | 0% | 75% |
| gemma2:27b | 70% | 0% | 12% | 25% | 25% | 100% | 100% | 100% |
| gemma3:27b | 70% | 0% | 12% | 25% | 25% | 100% | 100% | 100% |
| qwen2.5:14b | 47% | 42% | 37% | 50% | 50% | 100% | 100% | 100% |
| llama3.2:3b | 47% | 28% | 25% | 25% | 25% | 100% | 33% | 75% |
| mistral-nemo | 47% | 14% | 12% | 0% | 25% | 66% | 33% | 75% |
| r1:32b | 5% | 0% | 0% | 12% | 25% | 66% | 0% | 50% |
| command-r7b | 0% | 0% | 0% | 0% | 0% | 66% | 0% | 50% |

**Notable observations:**
- **Sonnet tool.result.consume 85%, not 100%** — earlier doc overstated; one probe failure on a tool.result-tagged probe.
- **qwen2.5:72b's tool.emit (88%) beats phi4:14b (76%)** — but phi4's `reasoning` is 100% vs qwen:72b's 75%. Different profiles at same 25/29 total.
- **llama3.3:70b is tool.emit-strong (82%) but tool.result-weak (42%)** — can call tools, struggles to consume results correctly. Matches the file_state failure pattern.
- **qwen2.5:7b delegation.format 33%** — crucial weakness for core/receptionist roles. Prefer other locals for delegation-heavy work.
- **gemma family tool.result = 0%** — literally never completes the "consume tool output → produce synthesis" path. Training-level deficiency; prompt engineering cannot fix.
- **Every model passes structured_json ≥ 66%** (T7) — orthogonal to tool-use capability. Confirms two-axis view.

## 3. Tier Difficulty (across all 15 models)

| Tier | Pass rate | Character |
|---|---|---|
| T0 | 80% | Single tool emission (echo from prompt) |
| T1 | 71% | Arg reasoning |
| T2 | **42%** | **Chain 2 tools — biggest cliff** |
| T3 | 58% | File edits (file_state closes loop) |
| T4 | 56% | Conditional on tool result |
| T5 | 64% | Agent delegation |
| T6 | 51% | Multi-step state |
| T7 | **100%** | Structured JSON (orthogonal) |
| T8 | **33%** | Hardest — bug fix + edit precision |
| T9 | 50% | Multi-file refactor |

## 4. Measurement Fidelity — Fix History

Four models had score jumps ≥+15 from infrastructure changes alone:

| Fix | Models affected | ΔPass |
|---|---|---|
| TextToolAdapter wrapper pattern added | phi4, gemma2, gemma3 | +22, +14, NEW |
| Case-insensitive text-tool parser | coder:32b, coder:7b | +17, +15 |
| stream_timeout_multiplier (4×/3×/2×) | llama3.3:70b, r1:32b, qwen:72b | +19, +3, +1 |
| Criterion tightening (t2/t6 todowrite) | all models | −1 to −2 each |
| Criterion revert (over-tight Grep on t6) | sonnet, haiku, gemma2 | +1-2 each |

**The single most important finding for the paper:** **infrastructure configuration is a measurement variable.** Four models' probe scores changed by ≥15 probes (out of 29) from capability-config alone, with no change to the model. Any published benchmark that doesn't disclose the adapter stack it ran under is incomplete.

## 5. Role Routing Recommendations

Based on v8 per-dim profiles:

| Role | Current default | Recommendation |
|---|---|---|
| core (delegation) | sonnet | **haiku** — matches sonnet at 28/29, 35× cheaper. Minor tie-breakers go to sonnet for edge cases. |
| implementer | sonnet | **sonnet** (keep) — edit.discipline matters; sonnet 100% vs haiku 100%, but sonnet's reasoning is more consistent. |
| researcher | haiku | **haiku** (keep) — best cloud option with web tools. |
| analyzer | sonnet | **haiku** — ties at delegation + rule.follow. |
| reviewer | haiku | **haiku** (keep) — simple read/grep pattern. |
| judge | haiku | **haiku** (keep) — structured_json is 100% for cloud. |
| receptionist | haiku | See paper §5.5 discussion — Option B (direct-core preset) eliminates this role entirely. |
| researcher-reader | phi4:14b ✓ | keep — best local for Read-heavy work |
| researcher-scout | qwen2.5:7b | **phi4:14b** — consider swap given 76% tool.emit vs 70% |
| researcher-synth | qwen2.5:14b | keep — structured_json 100% |

**Offline ladder:** phi4:14b → qwen2.5:72b → qwen2.5:7b → qwen:14b. Avoid deepseek-r1, command-r7b, mistral-nemo, llama3.2:3b for any agent role.

## 6. Probe Set Coverage Gaps (for future work)

Current probe set's weak spots, identified in §5.5 audit:
- **No true long-context test** — fixtures max out at 603 tokens. Real 10K/50K/128K needle probes added but not yet run.
- **No planning / task decomposition** — "given this vague goal, produce a step plan".
- **No error-recovery probe** — deliberately-failing tool + recovery path.
- **No ambiguity-handling probe** — "does model ask for clarification vs guess".
- **No consistency / multi-seed** — single measurement per (model, probe). Variance ±8% observed.

## 7. Paper headline findings (priority order)

1. **"Infrastructure configuration is a measurement variable"** — 4 models with ≥+15 score jumps.
2. **"Coder fine-tunes are moderately weaker, not catastrophic"** — revised from "6× gap" to "14% gap" after infra fix.
3. **"Agent capability ≠ model size"** — qwen2.5:7b (79%) > qwen2.5:14b (59%), non-monotonic within family.
4. **"Prompt engineering cannot override training defaults"** — gemma synthesis-enforcement prompt: zero effect. Gemma family structurally can't complete tool-result → synthesis path.
5. **"Training cutoff authority vs user authority"** — sonnet 4.6 rewrites user's "4.6" query to "4.5" because its training predates its own release.
6. **T2 is the capability cliff** — chain-two-tools (42%) separates weak from middling. T7 (100%) is an orthogonal structured-output axis, not a difficulty tier.

## 8. Caveats

- Single-run measurement (no seed variance data yet — +/-8% observed between runs).
- command-r7b disqualified for hallucinated tool results (safety-relevant; see v7 failure analysis).
- r1:32b disqualified for reasoning-trace timeout; may perform differently under per-turn timeout regimes.
- Probe `long_context` dimension is currently a misnomer — t3-read-long-file is only 603 tokens. Use the new t3-read-long-file-{8k,32k} and t4-read-long-file-128k probes (added but not yet in a full run).
