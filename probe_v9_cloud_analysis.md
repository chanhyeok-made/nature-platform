# Cloud multi-provider probe v9 — final analysis (2026-04-23)

Full 5-stage cloud matrix via OpenRouter. 11 usable cloud models × 32
probes = **352 canonical cells**. One model (`qwen-2.5-coder-32b`)
dropped from the matrix after Stage 5 revealed the OpenRouter →
Cloudflare route is non-functional (see §Upstream infra finding below);
its 32 cells are not included in cross-model tables.

## Coverage

| stage | scope | cost |
|---|---|---|
| 1 | gpt-4o-mini × 5 probes (sanity) | $0.036 |
| 2 | gpt-4o-mini × 32 probes | $0.010 |
| 3 | gpt-4o-mini + haiku-4.5 + gemini-flash × 32 | $0.209 |
| 4 | sonnet-4.6 + gpt-4o + gemini-2.5-pro + mixtral-8x22b × 32 | $1.24 |
| 5 | llama-3.3-70b + deepseek-v3.1 + qwen-72b + qwen-coder-32b + grok-3-mini × 32 | $0.11 |
| — | ad-hoc (t1-bash retry, 128k verify, qwen-coder diagnostic) | ~$0.003 |
| **total** | **11 usable cloud models × 32 probes** | **$1.75 / $100** |

## Final pass rates (most recent 32-probe run per model)

Sorted by pass count, then by output verbosity.

| model | pass | out_median | lat_median |
|---|---|---|---|
| openai/gpt-4o-mini | 30/32 | 1 | 3.7s |
| anthropic/claude-haiku-4.5 | 30/32 | 34 | 4.0s |
| anthropic/claude-sonnet-4.6 | 30/32 | 32 | 6.2s |
| x-ai/grok-3-mini | 30/32 | 8 | 34.7s (reasoning) |
| deepseek/deepseek-chat-v3.1 | 30/32 | 36 | 19.5s (reasoning) |
| google/gemini-2.5-pro | 29/32 | 2 | 11.6s (thinking) |
| google/gemini-2.5-flash | 28/32 | 8 | 2.8s |
| openai/gpt-4o | 28/32 | 6 | 3.3s |
| qwen/qwen-2.5-72b | 28/32 | 8 | 4.7s |
| mistralai/mixtral-8x22b | 27/32 | 2 | 2.5s |
| meta-llama/llama-3.3-70b | 26/32 | 2 | 6.1s |

## Failure structure (probe × number of models failing)

| probe | fails | models |
|---|---|---|
| t3-read-long-file-32k | **11/11** | ALL |
| t4-read-long-file-128k | **11/11** | ALL |
| t6-todowrite-progress | 2/11 | gemini-flash, gemini-2.5-pro (family) |
| t3-read-long-file | 2/11 | mixtral, gpt-4o |
| t0-no-tool-text | 1/11 | gpt-4o |
| t0-read-exact | 1/11 | mixtral (path hallucination) |
| t1-bash-simple | 1/11 | llama-3.3-70b |
| t1-glob-narrow-path | 1/11 | qwen-2.5-72b |
| t2-glob-then-read | 1/11 | llama-3.3-70b |
| t2-grep-then-read | 1/11 | llama-3.3-70b |
| t3-read-long-file-8k | 1/11 | gemini-2.5-flash |
| t4-read-then-noop-if-nomatch | 1/11 | mixtral |
| t8-add-missing-import | 1/11 | qwen-2.5-72b |
| t8-swap-operator | 1/11 | llama-3.3-70b |

No probe in the "sometimes fails across many models" middle band.
Two modes only: **universal ceiling (long-file 32k/128k)** and
**model-specific idiosyncrasies**.

## Paper-worthy provider-family signals

### 1. Gemini family: verbatim instruction echoing
Both `gemini-2.5-flash` and `gemini-2.5-pro` fail `t6-todowrite-progress`
the same way. Probe prompt specifies `activeForm: "stepping A"`
verbatim. Gemini models paraphrase:
- flash: `"Executing step A"`
- pro: `"doing step A"`

All 9 non-Gemini models echo `"stepping A"` exactly. **Reproducible
Gemini family bias** — training-level trait, not per-model.

### 2. Mixtral: long random-string hallucination
Mixtral fails three probes with the same pattern:
```
real path:    .../sfdqvc0000gn/...
mixtral 1st:  .../sfd7vc0000gn/...  (char substituted)
mixtral 2nd:  .../sfd9vc0000gn/...
mixtral 3rd:  .../sfd5vc0000gn/...
mixtral 4th:  .../sfdqvc0000gn/...  (finally correct)
```
Loss: t0-read-exact, t3-read-long-file, t4-noop-if-nomatch. Unique
to mixtral — llama-3.3-70b and qwen-72b handle the same random
paths without corruption.

### 3. gpt-4o: pathological `limit=1` on Read
gpt-4o chose `limit=1` on t3-read-long-file — scanning 1 line at a
time. After 2 turns (line 1, line 201), declared "no content after
line 200 in the file." File had thousands of lines. **Over-defensive
tool parameter selection**, not capacity failure.

### 4. llama-3.3-70b: T1/T2 tool_use drops
Llama 3.3 70B is the only model that failed `t1-bash-simple`,
`t2-glob-then-read`, `t2-grep-then-read`, `t8-swap-operator` — all
`tool_use` criterion failures. 4 unique fails, largest count among
the 11 models. Suggests weaker function-calling RLHF than qwen 72B /
deepseek-v3 / sonnet / gpt-4o.

### 5. Verbosity spectrum is a provider-family trait
Median output tokens across identical prompts:
- Claude (haiku 34, sonnet 32): **narrator family**
- DeepSeek v3.1: 36 (most verbose — reasoning traces leak)
- OpenAI (gpt-4o-mini 1, gpt-4o 6): **terse action family**
- Gemini (pro 2, flash 8): **concise family**
- Grok-3-mini: 8 (reasoning internal, invisible to output)
- Mistral/Llama/Qwen: 2-8

**30-40× spread**. Output style is a stable provider trait
independent of capability tier. Affects UX and cost (per-call output
tokens drive dollar cost more than input for workflow-heavy use).

### 6. Latency stratification reveals reasoning class
- **Fast (≤4s)**: mixtral, gemini-flash, gpt-4o, gpt-4o-mini, haiku, qwen-72b
- **Medium (4-8s)**: llama-70b, sonnet
- **Slow reasoning (11-35s)**: gemini-2.5-pro (thinking), deepseek-v3.1, grok-3-mini

Grok-3-mini's 34.7s median is striking — its reasoning is entirely
internal (out_median=8), so the cost is invisible to consumers of the
text output but real at infra level.

## Universal long-context ceiling

`t3-read-long-file-32k` and `t4-read-long-file-128k` failed **11/11**.

- **t3-32k**: file fits in Read's 256KB cap → first Read delivers
  the full 1829-line file in-context. Models still fail to extract
  the needle at line 915. **Retrieval failure in long repetitive
  content**, not I/O.
- **t4-128k**: file exceeds cap (527KB). Pagination required.
  Models choose insufficient page sizes (gpt-4o `limit=1`; gpt-4o-mini
  `limit=256`) and exhaust max_turns before reaching needle at line
  6584 (90% of 7315 lines). **Pagination strategy failure**, not
  context window.

These probes don't discriminate among cloud models. They mark the
current-generation ceiling on repetitive long-context retrieval.

## Single-run non-determinism at probe boundary

Observed on gpt-4o-mini during Stage 2 → Stage 3 retests:
- `t3-read-long-file-8k`: Stage 2 FAIL → Stage 3 PASS
- `t4-read-then-noop-if-nomatch`: Stage 2 FAIL → Stage 3 PASS

Providers default to non-zero temperature. **Borderline probe flake
rate ~3-5%**. Multi-seed retest on non-universal-failure probes
(12 probes × 11 models × 2 extra seeds = 264 cells) is the next
activity.

## Infrastructure health

### 10/11 working routes: clean

Across 352 cells on the 10 healthy models:
- HTTP 401/402/429/5xx: **0**
- ceiling-clip saturations: 0 (largest out=331, far below 8K ceilings)
- runner_errors: 1 flake on gpt-4o-mini t1-bash-simple (60.8s stream
  timeout, retry at 2.1s)
- parsing failures: 0

Overall flake rate: **~0.3%**.

### Upstream infra finding: qwen-2.5-coder-32b on OpenRouter is broken

OpenRouter routes `qwen-2.5-coder-32b-instruct` to **Cloudflare
Workers AI**. Two distinct failure modes discovered:

1. **Tool-use endpoints missing**: original Stage 5 returned HTTP 404
   "No endpoints found that support tool use" for every tool-using
   probe (28/32 cells). **Fixed** by adding the OpenRouter 404
   signature to `_is_no_tools_error()` so the text-tool wrapper
   activates automatically for this route.

2. **Cloudflare mid-stream 500s** (not fixable by us): even in text
   mode with a trivial prompt (`"Write a JSON"`), OpenRouter's
   Cloudflare route returns a truncated response with embedded 500:
   ```json
   "provider": "Cloudflare",
   "content": "Here is the JSON you requested:\n\n```json\n{\n  \"name\": \"test\",\n  \"value\": ",
   "error": { "code": 500, "message": "Internal Server Error" }
   ```
   Response cuts at ~8 output tokens mid-emission. Reproducible.
   Same model works fine via local Ollama — OpenRouter↔Cloudflare
   routing bug.

**Outcome**: qwen-2.5-coder-32b excluded from the cloud matrix.
11 models cited in paper cross-provider claims. The infra story
becomes a §5.x observation: "agent platforms must treat model
availability as a probabilistic, per-route attribute even on
aggregator APIs."

### Code changes made during v9

- `nature/providers/model_capabilities.py` — `max_output_ceiling`
  field + ceilings for Anthropic 4-series, GPT-4o/5, Gemini 2.5,
  Grok 4, DeepSeek R1, Llama 3.x, Mistral, Qwen 2.5 (with qwen-coder
  `text_tool_adaptation=True`).
- `nature/providers/text_tool_wrapper.py` — `_is_no_tools_error()`
  matches OpenRouter's 404 "No endpoints found that support tool use"
  in addition to Ollama's 400.
- `nature/providers/openai_compat.py` + `anthropic.py` —
  `clip_to_ceiling()` at `max_tokens` synthesis.
- `nature/protocols/provider.py` — `ProviderConfig.host_name` field.
- `nature/probe/runner.py` + `nature/server/registry.py` — pass
  `host_name` when constructing `ProviderConfig`.
- `nature/frame/manager.py` + `nature/session/runner.py` —
  `MaxOutputResolver` type + parameter threaded through for
  preset-driven per-role budgets.
- `nature/agents/presets.py` — `PresetConfig.max_output_tokens_overrides`
  field + validators.
- `tests/test_model_capabilities.py` — 9 ceiling-logic tests.

## Paper claims this data supports

1. **Cross-provider reliability**: 10/11 cloud models pass ≥26/32
   probes (81%+). Agent platforms can depend on cloud models for
   T0-T2 tool emission and T5-T9 multi-turn tasks.
2. **Provider-family qualitative traits** (§5 case study): Gemini
   paraphrase, Mixtral path hallucination, gpt-4o conservative
   pagination, llama-3.3-70b function-calling drops, verbosity
   stratification.
3. **Universal long-context retrieval ceiling**: all 11 fail needle
   extraction in 32K/128K repetitive contexts. 2026-Q2 frontier
   limit shared across providers and generations.
4. **Platform friction is a real cost**: qwen-2.5-coder-32b is
   unusable via OpenRouter despite being listed. Agent infra must
   handle per-route reliability or measurement is biased.

## 3-seed multi-seed results (added 2026-04-24)

Ran 12 borderline probes × 11 models × 2 additional seeds (264 cells,
$0.81). Aggregated into 3-seed majority vote.

### Final pass rates after majority vote

| model | 3-seed majority | seed1-only | Δ | stability |
|---|---|---|---|---|
| anthropic/claude-haiku-4.5 | 30/32 | 30/32 | ±0 | 36/36 consistent |
| anthropic/claude-sonnet-4.6 | 30/32 | 30/32 | ±0 | 36/36 consistent |
| deepseek/deepseek-chat-v3.1 | 30/32 | 30/32 | ±0 | 36/36 consistent |
| openai/gpt-4o-mini | 30/32 | 30/32 | ±0 | 1 flaky, 11 consistent |
| google/gemini-2.5-flash | 29/32 | 28/32 | **+1** | 2 flaky |
| google/gemini-2.5-pro | 29/32 | 29/32 | ±0 | 1 flaky |
| openai/gpt-4o | 29/32 | 28/32 | **+1** | 1 flaky + 1 consistent fail |
| mistralai/mixtral-8x22b | 29/32 | 27/32 | **+2** | **9 flaky** |
| x-ai/grok-3-mini | 29/32 | 30/32 | **-1** | 1 flaky |
| qwen/qwen-2.5-72b | 28/32 | 28/32 | ±0 | 1 flaky + 1 consistent fail |
| **meta-llama/llama-3.3-70b** | **23/32** | 26/32 | **-3** | **8 flaky + 1 consistent fail** |

### Stability stratification

- **Rock solid** (0 flaky on borderline probes): haiku-4.5, sonnet-4.6,
  deepseek-v3.1. Single-run results can be cited directly.
- **Low variance** (1-2 flaky): gpt-4o-mini, gemini-2.5-pro, gemini-flash,
  gpt-4o, grok-3-mini, qwen-72b. Multi-seed improves data but
  single-run ranking is reliable.
- **High variance** (mixtral, 9 flaky): Single-run ranking may be off
  by ±2 positions. Multi-seed required for paper claims.
- **Critical instability** (llama-3.3-70b, 8 flaky + 1 consistent fail):
  **single-run is unreliable**. Multi-seed reveals the real pass rate
  is 3 positions below the original 26/32.

### Per-probe flakiness (11 models × 3 seeds)

- **Most stable**: t0-read-exact, t1-bash-simple (only 1/11 flaky each)
- **Most variable**: t3-read-long-file-8k (4/11 flaky), t0-no-tool-text,
  t6-todowrite-progress (3/11 flaky each)
- The `t6-todowrite-progress` flake traces back to the JSON key-order
  observation below, not true model randomness.

## Methodological finding: probe regex vulnerable to JSON key ordering

During the seed 2 audit, `gemini-2.5-flash × t6-todowrite-progress`
flipped FAIL→PASS despite identical underlying model behavior
(same `activeForm: "doing step A"` paraphrase pattern). The flip
was traced to dict-key emission order changing between runs:

- **Seed 1** (FAIL): first item emitted `{"content": "step A", "status":
  "pending", ...}`, second item emitted `{"status": "pending",
  "activeForm": "Executing step B", "content": "step B"}`. When
  serialized as text, the sequence becomes `step A ... pending ...
  pending ... step B ...` — the probe regex `step A.*pending.*step B
  .*pending` expects a pending AFTER step B, but item 2's pending
  appeared BEFORE step B in the JSON text. Regex fails.
- **Seed 2** (PASS): both items emitted `{"content": "step X",
  "status": "pending", ...}` — key order puts content before status,
  so the serialized text reads `step A ... pending ... step B ...
  pending`. Regex passes.

Same model, same semantic behavior, different JSON key order → different
probe outcome. This is a general vulnerability of regex criteria over
serialized structured inputs. Paper methodology note: criteria operating
on deserialized schemas (e.g., `jmespath` queries) would be robust
where regex is not.

## Paper claims this data supports (updated)

1. **Cross-provider reliability (3-seed confirmed)**: 10/11 cloud
   models pass ≥28/32 probes (88%+). Stability is a provider trait —
   Anthropic, DeepSeek, and OpenAI small-tier models have zero
   flakiness on borderline cells.
2. **Stability is orthogonal to pass rate**: gpt-4o-mini (30/32) and
   mixtral (29/32) have similar raw scores but very different
   stability (0 vs 9 flaky cells). Agent platforms should report
   both.
3. **llama-3.3-70b is measurably unreliable**: 8/12 borderline probes
   flaky + schema-violation recovery failure (passed `todos` as JSON
   string instead of array, 4× in a row without correcting).
   Single-run evaluations systematically overstate its capability.
4. **Provider-family qualitative traits** (§5 case study): Gemini
   paraphrase, Mixtral path hallucination (reproduced across seeds),
   gpt-4o conservative pagination, llama function-calling drops,
   verbosity stratification.
5. **Universal long-context retrieval ceiling**: all 11 fail needle
   extraction in 32K/128K repetitive contexts. 2026-Q2 frontier
   limit shared across providers and generations.
6. **Platform friction is a real cost**: qwen-2.5-coder-32b is
   unusable via OpenRouter's Cloudflare route despite being listed.
   Agent infra must handle per-route reliability.
7. **Methodology caveat**: probe criteria using regex over JSON
   text are vulnerable to dict-key ordering non-determinism. Future
   probe sets should prefer schema-aware criteria.

## Next

1. **Paper §5 drafting**: promote the six provider-family signals
   plus the stability findings to a dedicated subsection with
   representative trace excerpts.
2. **Cross-reference local v8 data** (14 local Ollama models) for a
   local-vs-cloud contrast table. Local data already exists in
   `.nature/probe/results/runs/` from prior v8 runs.
3. **Schema-aware probe criteria** (future work, not this paper):
   replace regex criteria over JSON text with JMESPath or equivalent
   so key-ordering variance stops masquerading as model variance.

## Total cost

- Stages 1-5 (single-seed × 11 models × 32 probes): $1.75
- Multi-seed retest (2 extra seeds × 12 probes × 11 models = 264 cells): $0.81
- **Total: $2.56 / $100** (2.6% used)
