# Experiment plan (M3)

This is the **design artifact** for the paper's empirical work. It locks
the task catalogue, preset list, main matrix shape, and the three aux
experiments before any cells are collected in M4.

---

## 1. Task catalogue (10 total)

| id | source | category | size | acceptance | skill probed | status |
|---|---|---|---|---|---|---|
| `n1-pack-discovery` | nature-local | feature | medium | pytest | new module (pydantic + dynamic import + wiring) | ✅ built |
| `n2-pytest-pythonpath` | nature-local | feature | medium | pytest | schema extension + subprocess env wiring | ✅ built |
| `n3-seed-baseline-commit` | nature-local | feature | medium | pytest | git wiring inside eval runner (untracked→HEAD) | ✅ built |
| `s1-csv-parser` | synthetic | bug-fix | small | pytest | string parsing edge cases | ✅ built |
| `s2-docstring-judge` | synthetic | doc | trivial | llm_judge | non-code output + judge rubric | ✅ built |
| `s3-json-pointer` | synthetic | bug-fix | small | pytest | RFC 6901 edge cases, 3 bugs | ✅ built |
| `s4-async-refactor` | synthetic | refactor | medium | pytest | callback→async/await rewrite | ✅ built |
| `x1-pluggy-remove-plugin` | external | bug-fix | medium | pytest | multi-file change in upstream repo | ✅ built |
| `x2-click-zsh-colon` | external | bug-fix | medium | pytest | Zsh completion escape behavior in upstream click | ✅ built |
| `x3-httpx-noproxy-ip` | external | bug-fix | medium | pytest | NO_PROXY IPv4/IPv6/CIDR/localhost handling in upstream httpx | ✅ built |

Distribution: nature-local 3 / synthetic 4 / external 3. Bug-fix 5 /
feature 3 / refactor 1 / doc 1. Pytest 9 / llm_judge 1.

Acceptance-type note: `s4` was originally slated for llm_judge but
flipped to pytest with structural checks (`inspect.iscoroutinefunction`,
signature-parameter absence, end-to-end `asyncio.run`) — more
deterministic and less judge-variance noise. `s2` remains the sole
llm_judge exemplar (sufficient as a proof-of-capability for the paper;
second llm_judge data point adds noise without new information).

The commit-based pattern (`n?`, `x?`) treats a known-good commit's tests
as the oracle and rolls back to the parent as baseline. `n2`/`n3` are
now wired to dedicated spec-test commits (`616c5f3`, `7ff3615`) that
add focused unit tests on top of the implementation commits; the
task's workspace is `implementation^` and the diff-scoped test file
is extracted from the spec-test commit.

---

## 2. New task specs

### 2.1 n2-pytest-pythonpath — **built**

- **Baseline**: `1667c6d^`.
- **Spec commit**: `616c5f3` (adds `tests/test_eval_pythonpath.py`).
- **Dry-run verified**: 4/4 tests fail at baseline with distinct
  failure modes (AttributeError for missing field, pydantic
  ValidationError for extra input, and KeyError on `env` kwarg when
  the runner doesn't pass `env=`).

### 2.2 n3-seed-baseline-commit — **built**

- **Baseline**: `41bb39f^`.
- **Spec commit**: `7ff3615` (adds `tests/test_eval_seed_baseline.py`).
- **Dry-run verified**: collection fails at baseline with
  `ImportError: cannot import name '_commit_seed_baseline'` (all
  three tests effectively fail).

### 2.3 s3-json-pointer — **built**

- **Seed**: `seed.patch` creates `jsonptr.py` with a buggy
  `resolve(doc, pointer) -> Any` implementation plus
  `tests/test_jsonptr.py` (4 tests).
- **Three bugs visible on the buggy code** (verified — 3 fail / 1
  pass on a clean apply):
  1. Empty pointer `""` should return the whole document; current
     asserts `pointer.startswith("/")` and raises.
  2. Array indexing — tokens are always treated as dict keys, so
     `resolve([10, 20, 30], "/1")` raises `TypeError` instead of
     returning `20`.
  3. Escape order — `~0` is decoded before `~1`, so the pointer
     `"/~01"` meant for key `"~1"` decodes to `"/"` instead of `"~1"`.
- **Acceptance**: 4 tests in `tests/test_jsonptr.py` pass.
- **Constraints**: standard-library only; `jsonptr.py` kept compact.

### 2.4 s4-async-refactor — **built**

- **Seed**: `seed.patch` creates `notify.py` (callback-style
  `send_async(sub, msg, callback)` and `send_all(subs, msg, on_done)`)
  plus `tests/test_notify.py` (4 structural + behavioral tests).
- **Task**: rewrite both functions to `async def`, drop the callback
  parameters entirely, return awaited values, fan out with
  `asyncio.gather` (or equivalent).
- **Acceptance** (pytest, 4 tests, all 4 fail on the seed):
  1. `inspect.iscoroutinefunction(send_async)` is `True`.
  2. `inspect.iscoroutinefunction(send_all)` is `True`.
  3. `"callback"` not in `send_async` params and `"on_done"` not in
     `send_all` params.
  4. `asyncio.run(send_all(["a","b","c"], "hello"))` returns results
     ordered by subscriber.
- **Constraints**: stdlib only; no nested `asyncio.run` inside the
  async functions.

### 2.5 x2-click-zsh-colon — **built**

- **Upstream**: `pallets/click` commit `a1235aa` (PR #2846 / issue
  #2703) — `ZshComplete.format_completion` didn't escape `:` in
  completion values when help text was present, corrupting Zsh
  completion for command/option names containing a colon.
- **Shape**: `git_clone_at_ref` setup, baseline `a1235aa^`, diff
  scope `tests/test_shell_completion.py`. Acceptance:
  `test_zsh_full_complete_with_colons` (3 parametrized cases).
- **Dry-run verified**: 3/3 tests fail at baseline.
- **pythonpath**: `["src"]` (click uses src-layout).

### 2.6 x3-httpx-noproxy-ip — **built**

- **Upstream**: `encode/httpx` commit `15d09a3` (PR #2659) —
  `get_environment_proxies` unconditionally prefixed every NO_PROXY
  entry with `"all://*{hostname}"`, which broke IPv4 literals,
  IPv6 literals, CIDR ranges, and `localhost`.
- **Shape**: `git_clone_at_ref` setup, baseline `15d09a3^`, diff
  scope `tests/test_utils.py`. Acceptance: 4 parametrized cases of
  `test_get_environment_proxies` (env5-8, the new IP/CIDR/localhost
  rows).
- **Dry-run verified**: 4/4 fail at baseline when run in isolation,
  all pass at post-fix.
- **Pytest flags** (in `extra_args`): `--noconftest -p no:cacheprovider
  -o filterwarnings=`. Required because httpx's `conftest.py` imports
  heavy test-only deps (`trustme`, `uvicorn`, `cryptography`) and its
  `setup.cfg` has warning filters referencing `trio`. None of these
  are relevant to the targeted utility test; bypassing them keeps the
  eval env footprint minimal.

---

## 3. Presets for the main matrix

Six presets, intentionally covering the cost/capability frontier:

| preset | roster highlight | purpose |
|---|---|---|
| `default` | sonnet core/analyzer/implementer, haiku researcher | current shipping default |
| `all-haiku` | every role on haiku | low-cost cloud baseline |
| `all-sonnet` | every role on sonnet | high-capability cloud ceiling |
| `haiku-qwen-reader` | haiku default, researcher on local qwen2.5-coder:32b | validated recommendation (2026-04-18 benchmark) |
| `all-qwen-coder-32b` | every role on local qwen | fully-local capability probe |
| `solo-haiku` | single-agent on haiku, no delegation | topology control for §6.3 |

These names must match files under `.nature/presets/` at M4 time.
All six now exist (`default` is the builtin; `all-haiku`,
`all-sonnet`, `haiku-qwen-reader`, `all-qwen-coder-32b`, `solo-haiku`
live under `.nature/presets/`).

---

## 4. Main matrix shape and budget

- **Cells**: 10 tasks × 6 presets × 3 seeds = **180 cells**.
- **Expected cost range** (rough, based on 2026-04-18 measurements):
  - `haiku-qwen-reader` avg ~$0.09/cell → 30 cells ≈ $3
  - `all-haiku` avg ~$0.25/cell → 30 cells ≈ $8
  - `default` avg ~$0.26/cell → 30 cells ≈ $8
  - `all-sonnet` avg ~$0.80–$1.50/cell (sonnet is pricey) → 30 cells ≈ $30–$45
  - `all-qwen-coder-32b` $0 out-of-pocket (local) → 30 cells ≈ $0
  - `solo-haiku` avg ~$0.10/cell → 30 cells ≈ $3
  - **Total estimate: ~$55–$70.** Well under the $100 cap.
- Timeouts at 360s per cell; total wall time ~6–10 hours for 180 cells
  in serial (parallelism is a separate M4 question).

---

## 5. Aux experiments

Three illustrative ablations. Two use event-pinning (§4.2 of paper
outline), one is a structural limit case where event-pinning does not
apply.

### 5.1 Aux-A — Prompt ablation (event-pinned) — **unblocked (M3.4)**

- **Base task**: `n1-pack-discovery` (long-enough session with a clear
  researcher-delegation point).
- **Base preset**: `haiku-qwen-reader`.
- **Baseline run**: execute to completion, capture event log.
- **Fork point**: the `frame.opened` event of the researcher's first
  sub-frame (before researcher's first LLM call).
- **Branches** (two, each forked from the same event id):
  - Branch P0: resume under `haiku-qwen-reader` (original researcher
    prompt).
  - Branch P1: resume under `haiku-qwen-reader-stripped`
    (`prompt_overrides = {"researcher": "researcher-stripped"}`,
    otherwise identical model/tool/roster).
- **Variant artifacts** (both live under `.nature/`):
  - `instructions/researcher-stripped.md` — stripped prompt body
    (no "precision over coverage" scaffolding, no output-format
    template, no tool-selection guidance).
  - `haiku-qwen-reader-stripped.json` — the variant preset with
    `prompt_overrides`.
- **Framework support** (M3.4, this session): `PresetConfig` gained
  a `prompt_overrides: dict[str, str]` field; `AgentConfig.to_role`
  takes an optional `instructions_override`; `load_agent_instruction`
  resolves bare stems across project > user > builtin layers. The
  agent's own `instructions_text` is never mutated — only the
  emitted `AgentRole` carries the swapped body.
- **Metrics compared post-fork**: cost, turn count, tool-call count,
  pass.
- **Claim form**: "the prompt change caused a Δ-cost of X%, Δ-pass of
  Y pp, under otherwise byte-identical prefix."

### 5.2 Aux-B — Model swap (event-pinned)

- **Base task**: `x1-pluggy-remove-plugin` (external repo, researcher
  is heavily exercised for file reading).
- **Base preset**: `default` (sonnet core/analyzer/implementer, haiku
  researcher).
- **Baseline run**: execute to completion, capture event log.
- **Fork point**: `frame.opened` for the researcher sub-frame,
  pre-LLM.
- **Branches**:
  - Branch M0: continue under `default` (haiku researcher).
  - Branch M1: continue under a variant preset
    `default-qwen-researcher.json` that sets
    `model_overrides.researcher = "local-ollama::qwen2.5-coder:32b"`.
- **Metrics compared post-fork**: cost, latency, recall signal
  (tool-call count during researcher phase), pass.
- **Claim form**: "switching the researcher model mid-session changed
  Δ-cost and Δ-latency cleanly, absent pre-fork noise."

### 5.3 Aux-C — Solo vs multi (whole-session, limit case)

- **Base task**: `s1-csv-parser` (small enough that a solo agent is a
  fair comparison).
- **Comparison presets**: `default` (7-agent) vs `solo-haiku` (single
  agent, no delegation).
- **Why event-pinning does not apply**: the first emitted event for
  the two presets is already different (`default` opens a
  receptionist frame that never exists in `solo-haiku`), so there is
  no byte-identical prefix to share. This is explicitly framed in the
  paper as the structural limit of event-pinning.
- **Metric**: swap-and-run §4.1 comparison with 5 seeds each to
  dampen variance.

### 5.4 Protocol summary

Each aux experiment records:
- Base-run event log path (source of truth for the fork).
- Fork event id.
- Branch preset files and agent variant files (all version-controlled).
- Per-branch final metrics (cost, latency, turns, tool calls, pass).
- Metric deltas and a short narrative paragraph for the paper.

Aux outputs live in `paper/aux/<name>/` as JSON + markdown narrative.

---

## 6. Prerequisites before M4 can start

**Done in M3 (this session)**
- [x] `s3-json-pointer` seed.patch (3 fail / 1 pass on seed).
- [x] `s4-async-refactor` seed.patch (4 fail on seed).
- [x] `n2-pytest-pythonpath` spec-test commit `616c5f3` + task.json.
- [x] `n3-seed-baseline-commit` spec-test commit `7ff3615` + task.json.
- [x] `x2-click-zsh-colon` task.json (upstream `a1235aa`, 3 fail on baseline).
- [x] `x3-httpx-noproxy-ip` task.json (upstream `15d09a3`, 4 fail on baseline).
- [x] Matrix presets: `all-sonnet.json`, `solo-haiku.json` (built).
  `all-qwen-coder-32b.json`, `solo-qwen-coder-32b.json`,
  `haiku-qwen-reader.json`, `all-haiku.json` already existed.
- [x] Aux-B preset: `default-qwen-researcher.json`.

**Remaining for M3→M4 handoff**
- [x] Aux-A unblocked via `Preset.prompt_overrides` (M3.4 this session).
- [x] Per-task smoke dry-runs on `haiku-qwen-reader` for six new
      tasks. Run `1776619924-b2c0c5`, 6/6 PASS, total $0.6032,
      wall time ~11 min (range per cell: 14s–290s; cost per cell
      $0.02–$0.24). External tasks (x2, x3) dominate both axes as
      expected.
- [ ] Decide parallelism for the 180-cell matrix (serial vs. bounded
      worker pool; default serial keeps cost attribution clean).
