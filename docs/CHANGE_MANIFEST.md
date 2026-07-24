# CHANGE MANIFEST — everything modified this session

Drop each file at the repo path shown (left column). Paths are relative to project root.
All Python files compile. Docs at the end are reference only — they don't go in the repo
unless you want them there.

## Code — these replace existing files

| repo path                                         | what changed                                                                                                                                                                                                                                                     | must re-run after           |
| ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| `src/viz/style.py`                                | `RED`/`RED_DEEP` canonical (`ROSE` aliased); `set_background()` for transparent/tinted poster figures                                                                                                                                                            | —                           |
| `src/viz/figures.py`                              | most figure functions rewritten (see FIGURE_REVIEW_SUMMARY); new: `error_gain_map`, `construct_gain_bars`, `paired_gain_distribution`, `speed_accuracy_multi`, `construct_confusions`, `lba_recovery`, `speed_accuracy_tradeoff`, `_wilson`, `_model_constructs` | figure scripts              |
| `src/viz/recovery.py`                             | NEW — recovery panel grids + cross-method summaries                                                                                                                                                                                                              | `make_recovery_figures.py`  |
| `src/viz/labels.py`                               | NEW — strict R-label→GRIN-class parser + `labels_from_amortized`                                                                                                                                                                                                 | comparison scripts          |
| `src/viz/generation.py`                           | NEW — house-style prior-coverage figures                                                                                                                                                                                                                         | `generate_data.py --report` |
| `src/data/generator.py`                           | `_plot_coverage` now delegates to `viz.generation`                                                                                                                                                                                                               | `generate_data.py --report` |
| `src/inference/mle.py`                            | adds `fit_selected`; multi-start (`fit_*_multistart`); penalised (`fit_*_penalised`)                                                                                                                                                                             | comparison + poster scripts |
| `scripts/make_figures.py`                         | uses `recovery_panels` + comparison-head labels; drops `infer_class`; batched+single timing                                                                                                                                                                      | run to regen core suite     |
| `scripts/make_figures_rt.py`                      | writes to `results/figures/rt/`; parity confusions; architecture/LBA via shared fns; MLE parity                                                                                                                                                                  | run to regen RT suite       |
| `scripts/make_recovery_figures.py`                | NEW — the recovery family; showcase figure NOT duplicated here                                                                                                                                                                                                   | run after R fits            |
| `scripts/compare_to_r.py`                         | rebuilt (speed/convergence/accuracy/agreement); **MLE-full removed**, selected only                                                                                                                                                                              | run after R fits            |
| `scripts/sweeps.py`                               | figure plots all four sweep axes; placeholder for missing JSON                                                                                                                                                                                                   | run after sweeps            |
| `scripts/check_mle_health.py`                     | NEW — diagnoses separation vs optimiser failure                                                                                                                                                                                                                  | standalone, run once        |
| `scripts/R/fit_baselines.R`                       | **grtools bound sign fix** (+ mdsdt full-fit columns)                                                                                                                                                                                                            | **R fits MUST be re-run**   |
| `presentations/sbi_poster/make_poster_figures.py` | compact 3→2 panel recovery; coverage-only calibration; `accuracy_crossover`; trade-off figure; single+multi-start+penalised MLE; transparent bg                                                                                                                  | run to regen poster figures |

## The one hard dependency

`scripts/R/fit_baselines.R` changed the grtools parameter sign. Every `zx`/`zy` grtools
produced before is wrong. **Re-run the R fits before any comparison figure.** One-line
check after: `grtools_zx_0` should come out negative.

## Run order for real numbers

```
python scripts/generate_data.py --report          # coverage figures
python scripts/export_for_r.py --n 600             # shared test matrices
Rscript scripts/R/fit_baselines.R                  # baselines (re-run — sign fix)
python scripts/make_figures.py                     # counts core suite
python scripts/make_figures_rt.py                  # RT suite
python scripts/compare_to_r.py                     # comparison figure + printed table
python scripts/make_recovery_figures.py            # recovery family
python scripts/sweeps.py                           # after sweeps have run
python presentations/sbi_poster/make_poster_figures.py   # poster figs + \chk{} numbers
python scripts/check_mle_health.py                 # optional: the separation diagnostic
```

## Standing caveats

- **No number I put in any figure is real** — all synthetic test renders for layout. Read
  accuracy/speed off each script's own printed output; never transplant a number between
  regimes. The scripts score all methods on one shared set, so their printed numbers are
  the fair ones.
- **`poster.tex` needs two edits** (see POSTER_NOTES): the recovery `\includegraphics`
  currently resolves to nothing; the ~3 µs latency claim should quote the single-matrix
  time the poster script now prints.
- **Don't quote an MLE accuracy gap yet** — `check_mle_health.py` shows MLE is undefined
  (separation) on most matrices; report in a realistic regime with the penalised baseline,
  stated plainly.
- Unverified against a trained model / live R; a few figures were checked structurally, not
  by eye, when the image viewer failed. Named in FIGURE_REVIEW_SUMMARY.

## Reference docs (not repo files unless you want them)

FIGURE_REVIEW_SUMMARY.md · FIGURE_EXPLAINER.md (3-level explainer of every figure) ·
POSTER_NOTES.md · RECOVERY_FIGURES_NOTES.md · MISSING_FUNCTIONS_NOTES.md
