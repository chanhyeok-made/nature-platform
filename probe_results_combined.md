# Probe results — combined (v2 + v4 + v5)

_Generated from the latest (probe × model) cell across all runs.
v2 = original 9-model matrix; v4 = after `{workspace}` templating fix;
v5 = coder tunes + r1 through TextToolAdapter wrapper._

## Per-model totals

| model | pass | rate |
|---|---:|---:|
| `claude-sonnet-4-6` | 28/29 | 97% |
| `claude-haiku-4-5` | 27/29 | 93% |
| `qwen2.5:72b` | 24/29 | 83% |
| `qwen2.5:7b` | 19/29 | 66% |
| `qwen2.5:14b` | 17/29 | 59% |
| `llama3.2:3b` | 13/29 | 45% |
| `qwen2.5-coder:7b` | 5/29 | 17% |
| `qwen2.5-coder:32b` | 4/29 | 14% |
| `deepseek-r1:32b` | 1/29 | 3% |

## 3-axis tier profile per model

- **pass_ceil**  = highest T with ≥80% pass (contiguous from T0)
- **attempt_ceil** = highest T with ≥50% pass (not required contiguous)
- **top_any**   = highest T where at least one probe passed

| model | pass_ceil | attempt_ceil | top_any | per-tier |
|---|:---:|:---:|:---:|---|
| `claude-sonnet-4-6` | T4 | T9 | T9 | T0:4/4 T1:4/4 T2:3/3 T3:4/4 T4:2/2 T5:2/3 T6:3/3 T7:2/2 T8:2/2 T9:2/2 |
| `claude-haiku-4-5` | T4 | T9 | T9 | T0:4/4 T1:4/4 T2:3/3 T3:4/4 T4:2/2 T5:1/3 T6:3/3 T7:2/2 T8:2/2 T9:2/2 |
| `qwen2.5:72b` | T-1 | T9 | T9 | T0:3/4 T1:3/4 T2:3/3 T3:3/4 T4:2/2 T5:1/3 T6:3/3 T7:2/2 T8:2/2 T9:2/2 |
| `qwen2.5:7b` | T-1 | T9 | T9 | T0:3/4 T1:4/4 T2:2/3 T3:1/4 T4:2/2 T5:1/3 T6:2/3 T7:2/2 T8:1/2 T9:1/2 |
| `qwen2.5:14b` | T0 | T9 | T9 | T0:4/4 T1:2/4 T2:2/3 T3:1/4 T4:2/2 T5:1/3 T6:1/3 T7:2/2 T8:1/2 T9:1/2 |
| `llama3.2:3b` | T-1 | T8 | T8 | T0:3/4 T1:4/4 T2:1/3 T3:1/4 T4:0/2 T5:0/3 T6:1/3 T7:2/2 T8:1/2 T9:0/2 |
| `qwen2.5-coder:7b` | T-1 | T7 | T7 | T0:1/4 T1:0/4 T2:1/3 T3:0/4 T4:0/2 T5:0/3 T6:1/3 T7:2/2 T8:0/2 T9:0/2 |
| `qwen2.5-coder:32b` | T-1 | T7 | T7 | T0:1/4 T1:0/4 T2:1/3 T3:0/4 T4:0/2 T5:0/3 T6:0/3 T7:2/2 T8:0/2 T9:0/2 |
| `deepseek-r1:32b` | T-1 | T7 | T7 | T0:0/4 T1:0/4 T2:0/3 T3:0/4 T4:0/2 T5:0/3 T6:0/3 T7:1/2 T8:0/2 T9:0/2 |

## Full pivot (probe × model)

| probe | tier | claude-sonnet- | claude-haiku-4 | qwen2.5:72b | qwen2.5:7b | qwen2.5:14b | llama3.2:3b | qwen2.5-coder: | qwen2.5-coder: | deepseek-r1:32 |
|---|:---:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `t0-glob-basic` | T0 | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ |
| `t0-grep-literal` | T0 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| `t0-no-tool-text` | T0 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| `t0-read-exact` | T0 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| `t1-bash-simple` | T1 | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| `t1-glob-narrow-path` | T1 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ |
| `t1-grep-specific-def` | T1 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ |
| `t1-read-then-answer` | T1 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| `t2-glob-then-read` | T2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| `t2-grep-then-read` | T2 | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `t2-todowrite-simple` | T2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| `t3-edit-exact-bytes` | T3 | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `t3-edit-preserve-indent` | T3 | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `t3-read-long-file` | T3 | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `t3-write-new-file` | T3 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| `t4-read-then-edit-if-match` | T4 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| `t4-read-then-noop-if-nomatch` | T4 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| `t5-agent-delegation-basic` | T5 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| `t5-agent-delegation-rule-follow` | T5 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `t5-agent-delegation-structured-prompt` | T5 | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `t6-locate-then-edit-then-verify` | T6 | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `t6-read-three-and-sum` | T6 | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `t6-todowrite-progress` | T6 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| `t7-json-verdict` | T7 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `t7-json-verdict-fail-case` | T7 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| `t8-add-missing-import` | T8 | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ |
| `t8-swap-operator` | T8 | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `t9-locate-and-fix` | T9 | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `t9-two-step-refactor` | T9 | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ |

## Failure categories (per model)

| model | file_state | final_text | no_tool_error | timeout | tool_use |
|---|:-:|:-:|:-:|:-:|:-:|
| `claude-sonnet-4-6` | · | · | 1 | · | · |
| `claude-haiku-4-5` | · | · | 1 | · | 1 |
| `qwen2.5:72b` | · | 1 | 1 | 1 | 2 |
| `qwen2.5:7b` | 3 | 1 | 1 | · | 5 |
| `qwen2.5:14b` | 1 | 2 | 1 | 2 | 6 |
| `llama3.2:3b` | 6 | 6 | · | · | 4 |
| `qwen2.5-coder:7b` | 8 | 4 | · | · | 12 |
| `qwen2.5-coder:32b` | 8 | 6 | · | · | 11 |
| `deepseek-r1:32b` | · | · | · | 28 | · |