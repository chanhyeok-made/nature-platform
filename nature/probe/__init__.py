"""nature-probe — model capability probing suite.

Probes are low-level capability tests that isolate one primitive
(tool emission, structured output, multi-turn state, edit discipline,
etc.) and ask a single model to handle it. Each probe has a
graduated tier (T0 trivial → T9 autonomous) and a list of
`dimensions` it exercises, so running a probe set against a model
produces a tier-ceiling + dimension-coverage map we can route by.

Unlike `nature.eval` which runs a full agent graph end-to-end
against a preset, `nature.probe` talks to the target model directly
(provider + tools only, no frame manager / delegation machinery).
That makes each probe cheap to run and attributes the result to the
model's own capability, not to framework-level orchestration quirks.
"""
