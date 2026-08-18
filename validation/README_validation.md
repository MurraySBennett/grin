# GRIN validation suite

Every check, control, and mini-exploration run during development — re-runnable, so any
claim in `DESIGN_RECORD.md` can be reproduced or interrogated (lab meeting, conference
question, reviewer request).

**Scope, read before citing a row below as evidence for a manuscript claim:** every check
in this suite (`validation/checks.py`, `validation/checks_rt.py`) trains its own small,
fresh network for that run (`_fit_model()`: 2,500 examples/class, 20 epochs by default;
v06 uses 1,200/12) -- none of them load the actual released production checkpoint
(`results/models/npe_model.pt`). That makes this suite a **development/CI regression
suite**: it establishes that the training recipe behaves sensibly and stays stable
commit-to-commit, at a size cheap enough to run often. It is not where the manuscript's
headline numbers come from -- those are separate, larger, production-checkpoint
evaluations (`results/validation/manuscript_recovery.json`, `results/rt_construct_metrics.json`,
`results/mle_fits/baseline_fits.csv`, `manuscript/independence_accuracy_account.md`), each
cross-referenced from the manuscript section that cites it. Also note that "PASS" below
means different things row to row: some checks (v01, v02, v03, v04, v05, v06, v09, v14,
v15, v16) test a stated quantitative threshold; others (v07, v08, v10, v11, v12, v13) have
no pass/fail criterion at all and report a number unconditionally -- "PASS" there means
"ran and produced output," not "met a bar." `results/validation/SUMMARY.md` marks the
difference explicitly.

```bash
python validation/run_all.py                 # everything (slow)
python validation/run_all.py --quick         # fast smoke version
python validation/run_all.py --only v03      # a single check
```

Results land in `results/validation/` (a JSON per check + a summary table).

| id  | check                    | what it establishes                    | claim it supports                       |
| --- | ------------------------ | -------------------------------------- | --------------------------------------- |
| v01 | forward-model exactness  | bivariate-normal CDF vs scipy          | the simulator is exact, not approximate |
| v02 | prior coverage           | induced accuracy/bias/asymmetry spread | coverage is verified, not engineered    |
| v03 | parameter recovery       | MAE / r per parameter                  | GRIN recovers the representation        |
| v04 | calibration (SBC)        | rank uniformity + interval coverage    | the posterior is trustworthy            |
| v05 | speed vs MLE             | wall-clock head-to-head                | the amortization claim                  |
| v06 | ensemble stability (dev-scale) | agreement across training seeds, small dev networks | the dev recipe isn't an artifact of one run -- not a production-checkpoint ensemble claim |
| v07 | trial-count sweep        | recovery + calibration vs n            | reliability across regimes (report only, no threshold) |
| v08 | identifiability frontier | PI accuracy vs effect size             | the PI identifiability limit (report only, no threshold) |
| v09 | training-envelope diagnostic | in-envelope vs.\ out-of-envelope deviance (reversed mapping) | flags inputs outside what the network was trained on -- not a model-family/misspecification test, see `src/inference/ood.py`'s module docstring |
| v10 | graceful degradation     | lapse contamination                    | fails smoothly, not catastrophically (report only, no threshold) |
| v11 | amortized comparison     | learned heads vs AIC/BIC, dev-scale    | where amortized model comparison stands relative to AIC/BIC -- currently behind it on correlation structure (report only, no threshold; production-scale head-to-head is a manuscript `\pending{}` item, not this check) |
| v12 | RT collinearity probe    | corr(RT, rho) vs corr(acc, rho)        | WHY RTs don't fix PI (report only, no threshold; purely generative, no trained network) |
| v13 | RT gain                  | counts vs +RT quantiles                | what RTs actually buy (report only, no threshold) |
| v14 | architecture recovery    | 5-way SFT confusion, self-terminating-subset mean | 5-way accuracy against chance; the subset mean is architecture recall, not evidence of stable dimension neglect |
| v15 | speed-confound control   | RT-matched architectures               | architecture is shape, not speed        |
| v16 | LBA recovery + confound  | LBA params; GRT unaffected             | extra params don't cost identifiability |

## Rejected approaches (kept deliberately — the reasons are instructive)

- `r01_two_rt_prototype` — the two-RT model whose gain rested on `corr(RT_x, RT_y)`, a
  quantity **not observable** in a standard task. Large apparent gain, invalid premise.
- `r02_accuracy_binned_prior` — the original rejection-sampling generator; made the prior
  implicit and distorted, and binned on a 1-D summary blind to structure.

## Sweeps — establishing SCOPE

```bash
python validation/sweeps.py                 # all four axes
python validation/sweeps.py --only rt_speed # one axis
python validation/sweeps.py --quick         # fast version
```

One-factor-at-a-time from a documented baseline (a full grid is thousands of runs and mostly
redundant). Outputs `results/validation/sweeps/` + `results/figures/sweeps.png`.

| axis       | range                                              | what it establishes                                       |
| ---------- | -------------------------------------------------- | --------------------------------------------------------- |
| `trials`   | 5 → 2000 per stimulus (log), at 3 prior settings   | recovery + calibration across every realistic regime      |
| `prior`    | `z_max` ∈ {2,3,4} × `r_max` ∈ {0.7,0.9,0.95}       | not tuned to one arbitrary parameter envelope             |
| `rt_speed` | median RT ~0.4s → ~5s, plus a slow/noisy condition | the RT/architecture results are invariant to speed regime |
| `capacity` | hidden {128×3, 256×3} × seeds {0,1,2}              | the result is stable, not an artifact of one run          |

**Not swept:** batch size, learning rate, dropout, epochs — these affect convergence, not
conclusions.

**Important:** the sweeps are _not_ a hyperparameter search. `TRIAL_RANGE`, `Z_MAX`, and
`R_MAX` are scope claims to be set from experimental reality; the sweeps then evidence that
the results hold across them. Selecting the best-scoring values would tune the settings to
the evaluation.
