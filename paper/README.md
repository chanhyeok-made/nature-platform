# Paper sources

This directory holds the paper *nature: A platform for systematic
experimentation with LLM agent systems* — LaTeX source, rendered
PDF, figures, and reproducibility scripts.

## Files

| path | role |
|---|---|
| `paper.tex` | English LaTeX source (single file, self-contained) |
| `paper_ko.tex` | Korean translation (uses `kotex` package) |
| `paper.pdf` | compiled English PDF (~13 pages, ~700 KB) |
| `paper_ko.pdf` | compiled Korean PDF (~12 pages, ~5.7 MB; embeds Nanum fonts) |
| `figures/` | 8 PNGs referenced by both tex sources |
| `scripts/` | experiment harnesses + figure generators (see below) |
| `outline.md` | the original section-by-section plan that became the paper |
| `experiment_plan.md` | task catalogue + preset list locked before §5 measurement |
| `abstract.en.md` / `abstract.ko.md` | short-form abstracts for registrations |
| `arxiv_prep.sh` | bundles `paper.tex` + `figures/` into `arxiv_bundle.tar.gz` |

## Rebuild

### Requirements

- TeX Live 2024+ (or any pdflatex with the standard package set —
  `geometry`, `graphicx`, `booktabs`, `hyperref`, `amsmath`,
  `amssymb`, `xcolor`, `listings`, `microtype`, `url`, `enumitem`,
  `float`).
- For `paper_ko.tex`: `kotex` package + Nanum fonts (TeX Live ships
  these in the `kotex` collection).
- No bibtex run needed — references use inline `thebibliography`.

### Commands

```sh
cd paper
pdflatex paper.tex
pdflatex paper.tex      # second pass resolves cross-references

pdflatex paper_ko.tex
pdflatex paper_ko.tex
```

The second pass lets `hyperref` / label cross-refs settle.

## Figures

Six of the eight figures are generated from eval / experiment data by
Python scripts in `paper/scripts/`:

```sh
# Run from repo root
python3 paper/scripts/gen_figures.py          # §5 figures from eval runs
python3 paper/scripts/gen_architecture_fig.py # §3 schematic
python3 paper/scripts/gen_v2_figures.py       # §4.2 paired-CI + §7 ablation
```

Data sources:

- `fig_preset_task_heatmap.png`, `fig_cost_latency_pareto.png`,
  `fig_per_task_cost.png`, `fig_factorial_interaction.png` —
  computed from `.nature/eval/results/runs/*.json` (preset-benchmark
  records; reproducible via `nature eval run` with the same presets,
  tasks, and seeds documented in §5.1).
- `fig_ablation_model_swap.png` — generated from
  `/tmp/nature-eval/trajectory_multiprompt.json` (Block B P1 slice,
  produced by `paper/scripts/trajectory_multiprompt.py` against a
  running `nature server`).
- `fig_paired_jaccard_regime.png` — same data file (3-prompt aggregate),
  also produced by `paper/scripts/gen_v2_figures.py`.
- `fig_fork_schematic.png`, `fig_architecture.png` — hand-drawn
  box/arrow diagrams, no live data dependency.

## Reproducibility scripts

In `paper/scripts/`:

| script | purpose |
|---|---|
| `fork_ci_experiment.py` | Block A: $N=30$ paired CI experiment for §4.2 |
| `fork_ci_content_check.py` | Jaccard analysis on Block A |
| `trajectory_multiprompt.py` | Block B: 3 prompt × 15 seed × 3 branch sweep |
| `trajectory_content_check.py` | Jaccard analysis on Block B |
| `ablation_multi.py` | original 3-seed model-swap pilot (legacy) |
| `ablation_demo.py` | single-trial demo of the fork primitive |
| `ablation_trajectory.py` | 5-seed × 3-branch trajectory pilot (legacy) |
| `v2_analysis.py` | unified analysis over Block A + B with NLTK stopwords |
| `gen_figures.py` | renders §5 figures from eval runs |
| `gen_v2_figures.py` | renders §4.2 + §7 figures from Block A+B output |
| `gen_architecture_fig.py` | hand-drawn §3 schematic generator |

To rerun the empirical pipeline behind the §4.2 / §7 claims:

```sh
nature server start
python3 paper/scripts/fork_ci_experiment.py
python3 paper/scripts/fork_ci_content_check.py
python3 paper/scripts/trajectory_multiprompt.py
python3 paper/scripts/v2_analysis.py
python3 paper/scripts/gen_v2_figures.py
```

## Source of paper numbers

Every quantitative claim in `paper.tex` traces back to either a
`.nature/eval/results/runs/*.json` file (§5 preset benchmark) or a
`.nature/probe/results/*` file (§6 probes), or to the
`/tmp/nature-eval/*.json` outputs of the §4.2 / §7 scripts above.
The probe-companion analysis docs at the repo root
(`probe_v8_analysis.md`, `probe_v9_cloud_analysis.md`, etc.) show
the per-probe derivation; `paper.tex` cites only the headline numbers.

Notable note: the preset-benchmark matrix de-duplicates by
`(preset, task_id, seed)` keeping the most recent cell — the Phase A
n1 cells that originally ran under a 360 s watchdog are superseded by
the 600 s re-run. See §5.2 for the full account of this confound.

## arXiv submission

`arxiv_prep.sh` produces a self-contained `arxiv_bundle.tar.gz`
containing `paper.tex` and `figures/`. arXiv's build system runs
`pdflatex` twice by default, matching local build behavior.

```sh
cd paper
./arxiv_prep.sh
# upload paper/arxiv_bundle.tar.gz at https://arxiv.org/submit
```

The submission targets primary category `cs.SE` with cross-list
`cs.LG`; primary `cs.SE` avoids the endorsement requirement that
first-time `cs.LG`-primary submitters face.

## Known limitations (§9 of paper)

- Per-cell $n=2$ in §5; per-prompt $n=15$ paired in §7.
- No cross-framework (LangChain etc.) baseline.
- Paired-analysis CI tightening is prompt-regime conditional
  (open-ended analytical: yes; constrained factual / variety-seeking: no).
- Task set skewed to pytest-accepted Python.
- Judge acceptance not separately validated.
