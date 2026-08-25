# The GRIN pipeline

What runs, in what order, what each stage writes, and where those outputs end up.
This is the map to follow when cutting a new version. The mechanics of getting a
release out the door are in [`RELEASE.md`](RELEASE.md); this document is about
what produces the things that release ships.

Everything runs **from the project root**. Paths in `src/config.py` are absolute,
so outputs land in the same place regardless of where you invoke from.

## The dependency chain, and why it is the whole problem

```
  config.py  ──┐
               ▼
  1. generate_data.py ──► data/simulated/grt_dataset.npz        [tier 3]
               │
               ▼
  2. train.py ─────────► results/models/npe_model.pt            [tier 3]
               │           (embeds a src/provenance.py manifest)
               │
     ┌─────────┼─────────────────┬──────────────────┐
     ▼         ▼                 ▼                  ▼
  3. validation/run_all.py   4. evaluate.py     5. make_figures.py   6. export_onnx.py
     results/validation/*       results/           results/figures/     web/assets/models/
     [tier 2]                   mle_fits/          [tier 3]             [tier 1]
                                [2: *.csv]              │
                                                        ▼
                                              results/manuscript/
                                              figures + backing CSVs
                                              [tier 2]
```

Regenerating **any** stage invalidates everything below it. That is the core
constraint: there is no such thing as a cheap partial rerun, and it is why
artifacts are tiered by *role* rather than by size or by "can I regenerate it"
(see `RELEASE.md` for the tier table). A run is only meaningful as a whole, which
is why one `results/run_manifest.json` describes the whole run.

Two independent provenance records chain through this:

- **`src/provenance.py`** runs at *train* time and embeds a manifest inside the
  `.pt` (dataset sha256, prior, architecture, optimiser settings). It exists
  because the July 2026 shipped checkpoint could not be matched to the dataset
  that trained it. `verify_manifest()` checks a checkpoint against the current
  dataset.
- **`scripts/release_provenance.py`** runs at *release* time and records the
  whole run (commit, machine, config, sha256 of every artifact including bulk).
  `export_onnx.py --install` reads the checkpoint manifest and carries it into
  the site manifest, so a shipped `.onnx` traces back to its training dataset.

## Counts-only — the release pipeline

This is the pipeline behind the shipped model and the manuscript's headline
numbers. It is independent of the RT work below.

| # | command | writes | tier |
|---|---|---|---|
| 1 | `python scripts/generate_data.py --report` | `data/simulated/grt_dataset.npz` | 3 |
| 2 | `python scripts/train.py` | `results/models/npe_model.pt`, `results/training_history/` | 3 |
| 3 | `python validation/run_all.py` | `results/validation/v01–v16.json`, `SUMMARY.md` | 2 |
| 4 | `python scripts/evaluate.py` | `results/mle_fits/` (`*.csv` tracked) | 2/3 |
| 5 | `python scripts/make_figures.py` | `results/figures/` | 3 |
| 6 | `python scripts/export_onnx.py --install --version X` | `web/assets/models/cm/` | 1 |

Optional, as needed:

- `python scripts/sweeps.py` — one-factor-at-a-time scope sweeps → `results/validation/sweeps/` **[tier 2]**
- `python scripts/export_for_r.py --n 600` → `Rscript scripts/R/fit_baselines.R` → `python scripts/make_recovery_figures.py` — the recovery-figure family against the R baselines
- `Rscript scripts/R/fit_real_data.R` → `python scripts/compare_real_data.py` — GRIN vs mdsdt on real matrices
- `python -m scripts.check_mle_health` — is the MLE baseline failing, or is ML itself badly behaved here
- `python scripts/export_torchscript.py` — TorchScript for native R inference

**Step 5.5, and do not skip it:** copy the figures that actually appear in the
paper, *plus the CSVs they are drawn from*, into `results/manuscript/`. That
directory is tracked; `results/figures/` is not. This is what lets the manuscript
be rebuilt on a laptop with no GPU, and what gives a reviewer's "where does this
number come from" a `git log` answer.

## Validation, and what its PASS actually means

`validation/run_all.py` is a **development/CI regression suite**. Every check
trains its own small fresh network; none load the released checkpoint. Its
`SUMMARY.md` distinguishes `GATE` rows (a real quantitative threshold was met)
from `REPORT` rows (the check ran and produced output — "PASS" there means only
that). The manuscript's headline numbers come from separate, larger,
production-checkpoint evaluations: `results/validation/manuscript_recovery.json`,
`results/rt_metrics.json`, `results/mle_fits/baseline_fits.csv`.

As of the last commit, `SUMMARY.md` carries two `FAIL`/`ERROR` rows — `v05`
(speed) and `v11` (amortized comparison). Clear them or account for them before
submitting.

## Response times — read this before touching anything RT

**No RT model in this repository is currently release-ready, and the RT pipeline
above is the superseded one.** Full status, with the evidence, is in
[`dynamic_grt_rt_design.md` §0](dynamic_grt_rt_design.md). The short version:

- `src/data/rt_lba_generator.py` + `scripts/train_rt.py` are the **retired**
  ballistic model. `train_rt.py` now requires `--allow-legacy-ballistic` to run.
  It produced the currently-shipped `cmrt` weights and the 84.6% five-way
  architecture result. Per the design record, neither may be used as evidence for
  the replacement.
- `src/data/rt_dynamic_grt.py` (scalar reference) and
  `rt_dynamic_grt_vectorized.py` (fast generator) are the **replacement**.
  Gates 1, 3 and 4 are satisfied, gate 2 misses its stated threshold in one
  condition, and **gates 5–8 have not been run**. No dynamic-GRT network has ever
  been trained — there is no training script for it yet.

So the RT column of the pipeline table above currently reads: generate
(`generate_data.py --rt`) → train (`train_rt.py --allow-legacy-ballistic`) →
evaluate (`evaluate_rt.py`) → figures (`make_figures_rt.py`) → export
(`export_onnx.py --rt`), and every stage of it belongs to the retired model.

## Cutting a new version

Once the stages above have run, [`RELEASE.md`](RELEASE.md) takes over: export
with a version, record the run manifest, archive the bulk, push, and verify the
live site. The deploy workflow will refuse to ship weights whose sha256 disagrees
with their manifest, whose filename is unversioned, or whose provenance is still
a placeholder.
