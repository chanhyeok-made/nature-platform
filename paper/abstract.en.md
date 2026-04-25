# Abstract (English)

> Status: draft v1. Subject to revision as matrix data lands and framing
> tightens. Placeholders `N`, `M` are filled once the benchmark matrix is
> collected (Milestone M4).

LLM agent systems involve a combinatorial explosion of variables — prompts,
tools, models, orchestration configurations — making systematic
experimentation and fair comparison difficult. Existing tooling addresses
adjacent needs (application frameworks, production monitoring, static
benchmarks) but none is designed for isolated-variable experimentation
across the agent configuration space. We present **nature**, a platform that
treats each variable as a composable, isolatable unit through its Pack /
Host / Agent / Preset architecture and event-sourced execution, enabling
reproducible experiments and shareable research artifacts. Beyond
whole-session comparison, nature's event-sourced runtime supports
**event-pinned counterfactuals**: a recorded session can be forked at any
decision point and continued under a mutated configuration, cleanly
isolating the post-fork variable's contribution to the observed delta. We
demonstrate the platform's utility with (a) a preset-level benchmark across
`N` tasks × `M` configurations × 3 seeds, showing configuration-driven
cost-latency variance substantially exceeding model-choice effects, and (b)
event-pinned ablations isolating prompt and model-routing dimensions from
otherwise-identical histories. By grounding artifact
sharing in file-based schemas, nature aims to enable community-driven
accumulation of knowledge about effective agent configurations.
