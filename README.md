# GRIN — General Recognition Inference Network

Fast, amortized, uncertainty-calibrated inference of General Recognition Theory (GRT)
perceptual representations from a 2×2 identification confusion matrix. A neural network,
trained once on simulated data, replaces per-participant maximum-likelihood fitting:
milliseconds instead of seconds, with a calibrated approximate posterior and a
model-class inference, fast enough to run inside the trial loop for adaptive testing.

- **What the numbers mean:** `docs/interpreting.md`
- **The maths (parameterisation, constraints, prior):** `docs/GRT_model_spec.md`
- **What each validation check establishes:** `validation/README_validation.md`
- **Getting data from PsychoPy / jsPsych / an online platform into GRIN:** `docs/data_collection.md`
- **How the stages fit together, and what each one writes:** `docs/PIPELINE.md`
- **Cutting a release (models to the site, numbers to the manuscript):** `docs/RELEASE.md`
- **Status of the response-time extension:** `docs/dynamic_grt_rt_design.md` §0

Everything runs **from the project root**. Paths in `src/config.py` are absolute, so outputs
land in `data/` and `results/` regardless of where you invoke from.

## Packages

This repo trains GRIN and produces the two things people actually install. Both wrap
the same trained weights (numerically verified to agree) and are otherwise independent
of each other and of the training pipeline above:

- **[`packages/grintools/`](packages/grintools/)** — Python, `pip install grintools`. Torch-free at
  runtime (ONNX + `onnxruntime` only).
- **[`packages/grin/`](packages/grin/)** — R, native inference via the `torch` package
  (libtorch) — no Python required. `remotes::install_github("MurraySBennett/grin", subdir = "packages/grin")`.

Everything else in the repo (`src/`, `scripts/`, `data/`, `results/`, `validation/`, `web/`) is
how the model is trained and validated, not something an end user needs to touch.

---

## 0. One-time setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate       macOS/Linux:  source .venv/bin/activate
pip install -e .
```

GPU (optional but recommended for training):

```bash
python -c "import torch; print(torch.cuda.is_available())"   # want True
```

If `False`, reinstall torch with the CUDA build from pytorch.org (the default Windows wheel
is CPU-only).

**Before any real run, set `TRIAL_RANGE` in `src/config.py`** to bracket the trial count your
experiment collects. This matters more than any other setting — the network is calibrated for
the range it trains on. `TRIAL_IMBALANCE` (default 0.35) controls how uneven the four
per-stimulus counts within a matrix may be; leave it unless your data are unusually balanced
or unbalanced.

---

## 1. Which model do you need?

| Your data                            | Generator                      | Recovers                                                  |
| ------------------------------------ | ------------------------------ | --------------------------------------------------------- |
| Confusion matrix only (incl. legacy) | `src/data/generator.py`        | GRT params, PS, PI                                        |
| + response times                     | `src/data/rt_lba_generator.py` | the above **plus** processing architecture and LBA params |

Train whichever you need, or both and dispatch on what the participant supplies.

---

## 2. The complete pipeline, in order

Each block is independent given its inputs. The **counts pipeline (2.1–2.2)** is the core;
everything else builds on the trained model it produces.

Regenerating any stage invalidates everything downstream of it, so a partial rerun is rarely
partial in practice. `docs/PIPELINE.md` draws the dependency graph, says which tier each
stage's output belongs to (git, or archive-only), and explains the two provenance records
that chain through it. To ship what a run produced, follow `docs/RELEASE.md`.

### 2.1 — Generate data and train the counts model

```bash
python scripts/generate_data.py --report     # -> data/simulated/grt_dataset.npz + coverage figures
python scripts/train.py                       # -> results/models/npe_model.pt
python scripts/evaluate.py                    # recovery, calibration, model-ID, speed vs MLE
```

`--report` also writes `results/figures/coverage_report.png` and the per-panel breakdown in
`results/figures/generation/`.

### 2.2 — Render the counts figure suite

```bash
python scripts/make_figures.py                # -> results/figures/  (8 core figures)
```

### 2.3 — (Optional) RT pipeline — **superseded, reproduction only**

> **This pipeline trains the retired ballistic RT model.** It is kept so the earlier
> developmental analysis stays reproducible, and `train_rt.py` requires an explicit
> `--allow-legacy-ballistic` flag to prevent an expensive accidental retrain. Its
> outputs — including the 84.6% five-way architecture-recovery result — must not be
> used as evidence for the replacement model.
>
> The replacement (`src/data/rt_dynamic_grt.py`, a genuine stochastic-evidence model)
> has passed validation gates 1, 3 and 4; gate 2 misses its threshold in one condition
> and gates 5–8 have not been run, so **no dynamic-GRT network has been trained yet**
> and there is no training script for it. Full status, with the evidence and the open
> decisions: `docs/dynamic_grt_rt_design.md` §0.

Only if your data include response times. Same prior, same trial range — it additionally
simulates the RTs those trials produced.

```bash
python scripts/generate_data.py --rt                    # -> data/simulated/grt_rt_dataset.npz
python scripts/train_rt.py --allow-legacy-ballistic     # -> results/models/npe_rt_model.pt
python scripts/evaluate_rt.py                            # recovery, architecture, neglect, LBA
python scripts/make_figures_rt.py                       # -> results/figures/rt/  + results/rt_metrics.json
```

`make_figures_rt.py` writes a full parity suite (every counts figure has an `rt_` twin) plus
RT-specific figures, and exports `results/rt_metrics.json` — the single source of RT-model
timing/accuracy that the comparison and poster scripts read. Add `--rt-only` to
`generate_data.py` to skip regenerating the counts dataset.

### 2.4 — Compare against the R gold standards (grtools / mdsdt)

```bash
python scripts/export_for_r.py --n 600         # -> data/simulated/test_set_for_R.csv (stratified)
Rscript scripts/R/fit_baselines.R              # -> results/mle_fits/baseline_fits.csv
python scripts/compare_to_r.py                 # -> results/figures/comparison_to_r.png + printed table
```

**The R fits must be run (or re-run) whenever `fit_baselines.R` changes.** A sanity check
after: in `baseline_fits.csv`, `grtools_zx_0` should be **negative** (A1 sits below the
bound). If it is positive, the grtools sign convention has regressed — see the note at the
top of `fit_baselines.R`.

Why a 600-matrix sample and not the whole dataset: R MLE is ~0.1–0.5 s/matrix, so fitting a
million would take weeks and add nothing — the comparison is a statistical claim, and a few
hundred stratified matrices give tight intervals. Stratified by trial count, model class, and
effect size, so the interesting structure is represented rather than averaged away.

### 2.5 — The recovery figure family (per-method deep dive)

```bash
python scripts/make_recovery_figures.py        # -> results/figures/recovery/
```

Requires the R fits from 2.4. Produces per-method recovery grids (GRIN / mdsdt / grtools /
Python-MLE) on identical matrices, plus cross-method summaries. `--no-mle` skips the slow
Python-MLE reference; `--mle-select` adds its AIC/BIC labels to the classification figures.

### 2.6 — (Optional) Robustness sweeps

```bash
python scripts/sweeps.py                        # -> results/figures/sweeps.png
```

Run the sweeps first (they write the JSON the figure reads); missing sweeps show as labelled
placeholders rather than blank panels.

### 2.7 — (Optional) Real-data check

```bash
Rscript scripts/R/fit_real_data.R              # -> data/real/real_matrices.csv + mdsdt fits
python scripts/compare_real_data.py            # -> GRIN on the same matrices, vs mdsdt
```

mdsdt ships five real 2×2 matrices (`thomas01a/b`, `silbert09a/b`, `silbert12`). No ground
truth, so the check is agreement with the published gold standard — does GRIN reach the same
conclusions (PI, separability) as mdsdt's AIC model selection, in microseconds rather than
seconds — plus each matrix's envelope/reconstruction deviance, since a high value there means
the network's own fitted parameters don't reproduce the observed matrix well, most likely
because it falls outside the region the training prior populated, and the estimate should be
treated with caution (this is not a test of whether the GRT-Gaussian family itself could fit
the matrix — some parameter vector almost always can; see `src/inference/ood.py`). Distinct
from `compare_to_r.py` (2.4): that script scores simulated data against ground truth; this
one has none, so it scores agreement instead.

### 2.8 — Poster figures

```bash
python presentations/sbi_poster/make_poster_figures.py
```

Writes poster-scaled figures to `presentations/sbi_poster/figures/` and prints the numbers to
confirm against the `\chk{}` placeholders in `poster.tex`. If `results/rt_metrics.json` exists
(from 2.3), the speed-accuracy and crossover figures gain an indicative +RT overlay
automatically. `accuracy_crossover` is the slow figure (generator runs + MLE fits); pass
`crossover=False` to `main()` while iterating on layout.

**Two edits `poster.tex` still needs** (see `docs/POSTER_NOTES.md`, if retained): the recovery
`\includegraphics` filename, and quoting the single-matrix (not batched) latency.

### 2.9 — (Optional) Diagnostics

```bash
python scripts/check_mle_health.py             # is the MLE baseline separation-limited?
```

Standalone. Explains why MLE loses to GRIN at low trial counts (empty confusion-matrix cells
make the likelihood unbounded — a property of the data, not a bad optimiser). Run once if you
need to defend the accuracy comparison.

### 2.10 — Deploy the browser tools (static, no backend)

Export, install into `web/`, and stamp the site manifest in one step:

```bash
python scripts/export_onnx.py       --install --version 1.0.0   # -> web/assets/models/cm/
python scripts/export_onnx.py --rt  --install --version 1.0.0   # -> web/assets/models/cmrt/
```

`--install` writes a **versioned** filename (`npe_model.v1.0.0.onnx`) and the artifact's
real sha256 into `manifest.json`. Both matter: the deploy workflow serves `.onnx` with a
one-year immutable cache header, so overwriting a fixed filename would leave returning
visitors on the old weights indefinitely, and the hash is what CI checks before uploading.
Without `--install` the script just writes to `results/models/` as before.

Pushing `main` deploys `web/` to S3 and invalidates CloudFront
(`.github/workflows/deploy.yaml`). Confirm the CDN actually serves the new weights:

```bash
python scripts/verify_deploy.py                # hashes what the live site returns
```

The site is static — inference runs in the visitor's browser and nothing is uploaded — so
`web/` can equally be copied to any static host. Full procedure: `docs/RELEASE.md`.

---

## 3. Use it in your own code

```python
from grin import infer
result = infer(confusion_matrix, trials)   # trials optional (defaults to row sums)
result.summary()
result.as_dict()          # {param: {estimate, sd, ci90}}
result.model_class        # inferred GRT model
result.envelope_deviance  # is this matrix outside the region GRIN was trained on?
```

Adaptive / real-time:

```python
from src.adaptive.engine import AdaptiveSession
sess = AdaptiveSession(model)
sess.add_trial(stimulus, response)         # inside your trial loop
theta = sess.estimate()                    # microseconds — fits any ISI
sess.uncertainty()                         # stop when this crosses your threshold
```

---

## 4. What lands where

Outputs are tiered by role, which decides whether they go in git (see `docs/RELEASE.md`):
**[1]** ships to the website, **[2]** backs a manuscript number, **[3]** bulk, archived
once and never committed.

```
data/simulated/     grt_dataset.npz, grt_rt_dataset.npz, test_set_for_R.csv    [3]
results/models/     npe_model.pt, npe_rt_model.pt, *.onnx                      [3]
results/mle_fits/   baseline_fits.csv                                          [2]  (*.csv only)
results/figures/    core suite                                                 [3]
        /generation/  prior-coverage panels
        /recovery/    per-method comparison
        /rt/          RT suite
results/manuscript/ final paper figures + the CSVs behind them                 [2]
results/validation/ v01–v16.json, SUMMARY.md, sweeps/                          [2]
results/*.json      rt_metrics, dynamic_grt_gates, ...                         [2]
results/run_manifest.json   one record per release run: commit, machine,       [2]
                            config, sha256 of every artifact incl. bulk
web/assets/models/  cm/ and cmrt/: versioned .onnx + manifest.json             [1]
```

---

## Troubleshooting

**`GRTDataGenerator.__init__() got an unexpected keyword argument 'imbalance'`** — your
`generator.py` predates the `imbalance` parameter. Use the current `src/data/generator.py`.

**`Missing key(s) in state_dict` on load** — the architecture you built differs from the
checkpoint. Use `from src.api import load_model` (reads the architecture from the checkpoint)
rather than constructing `NPEModel` by hand.

**`grtools_zx_0` is positive in `baseline_fits.csv`** — the grtools bound sign has regressed;
re-check `extract_grtools_params()` in `fit_baselines.R`.

**Figures look weak** — confirm you trained on the full dataset (not a smoke-test subset) and
that `TRIAL_RANGE` matches your target regime.

**Don't hand-mix numbers across scripts.** Each comparison script scores all its methods on
one shared set and prints its own fair table; read each number off the script that computed
it, never transplant between scripts or evaluation regimes.

```

```
