# GRIN — BayesFlow port (parallel branch)

A side-by-side reimplementation of GRIN's inference machinery on **BayesFlow v2**,
kept in its own directory so it runs against the current bespoke PyTorch stack on an
**identical generative model** and an **identical frozen test set**, scored by an
**identical ruler**. The goal is an honest pros/cons comparison between frameworks,
after which we fold back to a hybrid if the numbers justify one.

## What is shared vs what differs

Shared (the science): the generative model. `bf_simulator.py` reuses
`src/grt_model.py` directly — same identified 12-parameter forward model, same
class-specific prior (`|z| ~ U(0,3)`, `rho ~ U(-0.9,0.9)`), same log-uniform
per-stimulus trial regime, same `featurize` (16 row-proportions ++ 4 log10 trial
counts). Nothing about GRT changes.

Differs (the plumbing): everything downstream. The bespoke stack's hand-rolled
Gaussian/Cholesky NPE head + custom training loop + custom SBC is replaced by
BayesFlow's adapter → inference network (FlowMatching _or_ CouplingFlow) →
`BasicWorkflow`, with BayesFlow's built-in standardisation, training-history
tracking, and diagnostic suite (`compute_default_diagnostics` /
`plot_default_diagnostics`).

## Directory layout

```
bayesflow_port/
├── README.md
├── bf_simulator.py               # shared GRT simulator (single matrix)
├── bf_workflow.py                # single-matrix adapter + workflow
├── multiparticipant_workflow.py  # milestone-2 (stage 1) session simulator + SetTransformer workflow
├── attention_workflow.py         # milestone-2, stage 2: per-participant attention scalars
├── shared_eval.py                # framework-agnostic metrics for compare_frameworks.py
├── compare_frameworks.py         # frozen test set -> both frameworks -> one table
├── train_and_diagnose.py         # entry point: single-matrix pilot + diagnostics + report
├── train_multiparticipant.py     # entry point: milestone-2 stage-1 pilot + diagnostics + report
├── train_attention.py            # entry point: milestone-2 stage-2 pilot + diagnostics + report
├── grt_figures.py                # custom classic-GRT figures (group/individual/attention)
├── scripts/                      # amortized-workflow skill helpers
│   ├── inspect_training.py
│   └── check_diagnostics.py
├── .gitignore                    # keeps metrics.csv/history.json/report.md, drops checkpoints/figures
└── results/                      # all run output, one <slug>/ per analysis (never source)
    ├── single-matrix-base/       # default --results-dir for train_and_diagnose.py
    │   ├── checkpoints/model.keras
    │   ├── loss.png, recovery.png, calibration_ecdf.png, coverage.png, z_score_contraction.png
    │   ├── history.json, metrics.csv, report.md
    ├── multiparticipant-base/    # default --results-dir for train_multiparticipant.py
    ├── attention-base/           # default --results-dir for train_attention.py
    └── compare-run/              # default --results-dir for compare_frameworks.py
```

The shared classic-GRT perceptual-space primitive itself (ellipses, decision
bounds, marginal-density strips) lives OUTSIDE this directory, at
`src/viz/grt_space.py` — it draws whatever 12-parameter template it is given,
regardless of which framework produced it, and is also the Python/matplotlib
counterpart to the web app's `web/assets/js/grt-plot.js`. `grt_figures.py` here
imports it and adds the port-specific composites (group/individual/attention).

Source files (the port's code) stay flat at the top level; every generated artifact —
model checkpoints, figures, metrics, reports — lives under `results/<slug>/`, one
directory per analysis, per `references/reporting.md`'s naming convention. Checkpoints
nest under `results/<slug>/checkpoints/model.keras` (not the results root) so a `git
status` in a slug directory shows evidence (`metrics.csv`, `history.json`, `report.md`)
next to, but visually separated from, the multi-MB binary. `results/*/checkpoints/`
and `results/*/*.png` are gitignored — regenerable from a re-run — while `metrics.csv`,
`history.json`, and `report.md` are the actual diagnostic evidence and stay tracked,
mirroring the root `.gitignore`'s `results/models|figures` vs. `results/validation`
split.

## Files

- `bf_simulator.py` — shared GRT simulator, BayesFlow-shaped output dict (single matrix).
- `bf_workflow.py` — single-matrix adapter, inference network factory, `BasicWorkflow`, sample.
- `train_and_diagnose.py` — runnable offline-pilot pipeline for the single-matrix port:
  fits with `validation_data`, saves `history.json`, runs the standard diagnostic suite,
  writes a `report.md`. **The primary entry point for the single-matrix port.** Writes to
  `results/single-matrix-base/` by default.
- `multiparticipant_workflow.py` — milestone-2 stage-1 session simulator + `SetTransformer`
  pooling adapter/workflow (see below).
- `train_multiparticipant.py` — same pipeline as `train_and_diagnose.py`, for the
  multi-participant pooling workflow. **The primary entry point for milestone 2, stage 1.**
  Writes to `results/multiparticipant-base/` by default.
- `attention_workflow.py` — milestone-2 stage 2: recovers per-participant `log(k_A)`,
  `log(k_B)` conditioned on the group template (plug-in-true during training); also
  provides `sample_attention_propagated` for application-time inference that pools
  across stage-1 posterior draws (see below).
- `train_attention.py` — same pipeline as `train_and_diagnose.py`, for the attention-scalar
  workflow. **The primary entry point for milestone 2, stage 2.** Writes to
  `results/attention-base/` by default.
- `shared_eval.py` — framework-agnostic metrics (recovery r/MAE, SBC coverage, ranks),
  used specifically by `compare_frameworks.py` for the cross-framework table (the
  bespoke model has no `BasicWorkflow`, so this hand-rolled ruler is the only way to
  score both sides identically — everywhere else, prefer
  `workflow.compute_default_diagnostics`).
- `compare_frameworks.py` — frozen test set → both frameworks → one table. Writes to
  `results/compare-run/` by default.
- `grt_figures.py` — custom classic-GRT figures on top of the standard report (see
  "Custom figures" below).
- `scripts/inspect_training.py`, `scripts/check_diagnostics.py` — training-convergence
  and diagnostic-interpretation helpers (from the amortized-workflow skill).

Run (from the repo root, with `bayesflow` installed — `conda activate grin_venv`):

```bash
KERAS_BACKEND=torch python bayesflow_port/train_and_diagnose.py --train 20000 --epochs 100
KERAS_BACKEND=torch python bayesflow_port/train_multiparticipant.py --train 20000 --epochs 100
KERAS_BACKEND=torch python bayesflow_port/train_attention.py --train 20000 --epochs 100
```

## Custom figures

Beyond the mandatory standard report (loss/recovery/calibration/coverage/
z-score-contraction, produced by every `train_*.py`), `grt_figures.py` adds
classic-GRT-style figures for the hierarchical group/individual structure this
port introduces — requested explicitly, since these read the way a GRT paper's
figures normally do, which the generic diagnostic suite has no reason to
reproduce on its own:

| Function | Shows |
|---|---|
| `group_template_figure` | Hierarchical level: the shared group template, optionally true-vs-recovered. |
| `individual_vs_group_figure` | ONE participant's own (attention-scaled) perceptual space against the group template, annotated with their `k_A`/`k_B`. |
| `individual_grid_figure` | Small multiples: every participant in a session against the group template, on one shared axis scale. |
| `attention_scalar_forest` | Per-participant `k_A`/`k_B` posteriors (forest plot), reference line at `k=1` (group-typical). |
| `attention_recovery_figure` | In-silico recovery scatter for `log(k_A)`, `log(k_B)`, matching `src/viz/figures.py`'s `parameter_recovery` conventions. |
| `pooled_vs_unpooled_rho_figure` | Milestone 2's actual hypothesis — does pooling participants recover rho better than one matrix alone? Reuses `src.viz.figures.rt_vs_counts_dumbbell` (generalised with `before_label`/`after_label`/`xlim` so its RT-poster defaults don't silently mislabel or clip a non-accuracy metric like NRMSE) rather than inventing a new comparison figure. |

Run `python bayesflow_port/grt_figures.py` for a runnable demo of all six on
purely simulated data (no trained checkpoint required) — see the module's
`__main__` block for exactly which arrays to swap in once real
`multiparticipant_workflow.py` / `attention_workflow.py` checkpoints and
`sample_attention_propagated` output exist. The underlying ellipse/decision-bound
primitive (`perceptual_space`, `perceptual_space_figure`) lives in
`src/viz/grt_space.py`, framework-agnostic and shared with the web app's
`grt-plot.js` drawing conventions (solid = reference structure, dashed =
comparison overlay).

## Decisions logged

1. **Backend = torch**, to match the existing stack (Keras 3, `KERAS_BACKEND=torch`).
2. **rho unconstraining**: BayesFlow's idiomatic `constrain(lower=-1, upper=1)`
   (logit-style), _not_ the bespoke stack's `atanh` (Fisher-z). Equivalent
   unconstraining maps; using each framework's natural choice keeps the comparison
   honest rather than forcing parity that neither would ship. Noted as a known
   coordinate difference.
3. **No summary network in the single-matrix port.** The condition is already a fixed
   20-d featurised vector with meaningful element order — the "simple vector" case in
   the amortized-workflow skill's conditioning decision table — so it enters directly
   as `inference_conditions`. The `SetTransformer` summary network is reserved for the
   multi-participant extension (`multiparticipant_workflow.py`) — the main reason to be
   on BayesFlow at all.
4. **Coupling transform = spline (rational-quadratic), not affine.** See the bug note.
5. **`bf.BasicWorkflow`, not a hand-assembled `ContinuousApproximator`.** Gives
   `workflow.simulate`, `fit_offline(..., validation_data=...)`,
   `compute_default_diagnostics`/`plot_default_diagnostics`, and `workflow.sample`
   (returning original parameter names) for free.
6. **Network capacity pinned to named tiers** from the amortized-workflow skill's
   `model-sizes.md` (Base by default in both workflows). `FlowMatching`/Base is the
   default inference network per the skill's current preference for free-form nets over
   coupling flows; `CouplingFlow` (spline) stays available via `kind="coupling"` since
   it's the transform already validated against the affine NaN bug below, with cheaper
   sampling latency relevant to the adaptive-loop use case.
7. **Dtype conversion lives in the adapter** (`.convert_dtype("float64","float32")`),
   not hand-cast in the simulator — the adapter owns every train/inference-time
   transform so the two paths cannot silently diverge.

## Bug found and fixed (affine → spline)

The first coupling-flow build used the default **affine** transform. In short CPU
runs it intermittently emitted near-`float32` overflow values (~3e38), producing
NaN posterior draws in ~0.15% of samples but touching ~34% of test matrices — enough
to poison recovery correlations to NaN. Switching to the **spline** transform
(bounded-derivative, the modern SBI default per Durkan et al. 2019) gave 0 NaNs at
the same budget. Spline is now the default; pass `transform='affine'` to reproduce
the instability. This was a real defect in the first cut, not a tuning nicety.

## Bug found and fixed (broken `sys.path` — the port could not import at all)

`bf_simulator.py` originally did `sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "grin"))`,
which resolves to a nonexistent `grin/grin` sibling directory — `bayesflow_port/`
already lives inside the `grin` repo root. `import src.grt_model` failed immediately.
Fixed to insert the repo root itself (one level up from `bayesflow_port/`). Verified:
`simulate()` now runs end-to-end, proportions land in `[0,1]`, `rho` in `(-0.9,0.9)`.

## Findings so far (tiny CPU smoke runs — NOT final numbers)

These come from 4–15 epoch CPU runs on 6k–20k simulations, purely to prove the
pipeline and surface early signal. Real numbers need the full budget — now
trainable end-to-end via `train_and_diagnose.py` / `train_multiparticipant.py`.

- **Recovery**: z-score `r ≈ 0.96` already at ~8 epochs (bespoke target ~0.97).
  Correlation `r ≈ 0.29–0.35` — well below the ~0.61 target, but rho is the
  information-constrained parameter and these runs are badly undertrained; expected
  to climb with the real budget.
- **Calibration**: essentially on target out of the box — nominal 90/95% central
  intervals gave empirical ~90.5/94.8% coverage. This is BayesFlow's strong suit and
  it shows.
- **Latency** (the crux for the adaptive loop), CPU, spline coupling flow:
  - single matrix, 1 draw: **~81 ms** — almost entirely _fixed per-call overhead_
    (Python + Keras/torch dispatch + adapter + standardisation).
  - single matrix, 200 draws: ~101 ms (the draws add only ~20 ms; overhead dominates).
  - batched 200 matrices: ~27 ms/matrix (batching amortises the fixed overhead).
  - **bespoke Gaussian head: ~0.003 ms/call.**

  Honest reading: a flow is ~4 orders of magnitude slower than the Gaussian head for
  a naive one-at-a-time call, **but** most of that is fixed dispatch overhead, not
  flow math — so GPU + batching + a compiled inference path would cut it a lot. This
  does _not_ prove "flows are too slow for the loop"; it says the loop economics need
  batching, a GPU, or a distilled fast head, whereas the Gaussian head is already
  loop-cheap. This is the central input to the hybrid decision, and it is now
  measured rather than assumed.

## Milestone 2 — multi-participant pooling (in progress)

The single-matrix port mostly reproduces what the bespoke stack already does (a flow
instead of a Gaussian, with better calibration but far higher per-call cost). The
payoff BayesFlow uniquely provides is the **permutation-invariant summary network**
(`SetTransformer`) over a _set_ of participants' confusion matrices — the amortised
analogue to GRT-wIND, pooling across participants to attack DS/rho identifiability.

`multiparticipant_workflow.py` implements this: a session simulator where N
participants share one canonical 12-parameter group template but vary individually in
dimensional attention (`k_A`, `k_B` — reusing the drift-scaling idea already in
`src/data/rt_lba_generator.py`), pooled via a `SetTransformer` (Base tier,
`summary_dim=36`) to recover the shared template. Per-participant attention scalars are
simulated by this stage but inferred by **stage 2** (`attention_workflow.py`, below).

Run `train_multiparticipant.py` to train and diagnose this; it also cross-references
`results/single-matrix-base/metrics.csv` (if present) so the pooled-vs-unpooled rho
comparison — the actual point of this milestone — is a direct read from `report.md`.

### Milestone 2, stage 2 — per-participant attention scalars

`attention_workflow.py` recovers each participant's `log(k_A)`, `log(k_B)` conditioned on
the group template (`x_participant` ++ `group_theta`, still a fixed-length "simple
vector" condition — no summary network needed here). Design decisions locked in with the
user before building (see the module's docstring for the full rationale):

1. **Training conditions on the TRUE simulated group template**, not a stage-1 posterior
   draw — reuses `multiparticipant_workflow.simulate_sessions(n_participants=1, ...)`
   verbatim (squeezed to drop the length-1 participant axis), so no new generative-model
   code exists for this stage.
2. **Targets are `log(k_A)`, `log(k_B)`**, matching the generative
   `k = exp(Normal(0, attention_sd))` draw exactly — no `.constrain()` needed.
3. **Application-time uncertainty propagation is Monte Carlo mixture**, not a
   point-estimate plug-in or noise-injected retraining:
   `attention_workflow.sample_attention_propagated` runs stage 2 once per stage-1
   posterior draw (e.g. 50) and pools the resulting samples. This is a **"cut"** in the
   Bayesian-workflow sense — individual-level fit cannot feed back and correct the
   group-level estimate — an accepted limitation of the plug-in two-stage design, not an
   oversight.
4. Network capacity reuses `bf_workflow.build_inference_network`'s named tiers directly
   — one source of truth for Base/Large capacity across the whole port.

Run `train_attention.py` to train and diagnose stage 2 in isolation (conditioned on the
true group template, so its own diagnostics are not confounded with stage-1 error — see
`report.md`'s "Two-stage composition" section for that caveat spelled out). This pass is
simulation-only per the user's request; real data (trial-by-trial with response times,
reducible to the same confusion-matrix format used throughout this port, or usable at
the raw trial level) is a deliberately out-of-scope next step — raw-trial-level
conditioning would need a set/time-series summary network, a different architecture from
the fixed-length vector used here.

## Open questions / TODO

- Wire `run_bespoke()` in `compare_frameworks.py` on a machine with the trained
  `npe_model.pt` + `src` package so the comparison table has both rows (hook + interface
  documented in `compare_frameworks.py`).
- Clean GPU latency benchmark (fixed-overhead vs per-draw, single vs batched) — the
  CPU numbers above are order-of-magnitude only.
- Full-budget training run (`train_and_diagnose.py`, `train_multiparticipant.py`,
  `train_attention.py`) to judge correlation recovery and the MLE crossover fairly.
- Decide whether atanh-vs-constrain coordinate choice materially affects rho recovery.
- Wire real trial-by-trial data (with response times) into the port, in both the
  reduced confusion-matrix format (backwards-comparable with historical GRT analyses)
  and, eventually, a raw-trial-level representation — the latter needs a summary network
  over trials, not yet designed.
- Full three-stage real-data pipeline: `multiparticipant_workflow.py` on a real session
  -> `attention_workflow.sample_attention_propagated` on that session's participants ->
  report per-participant attention estimates with group uncertainty propagated.
