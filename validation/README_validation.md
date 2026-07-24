# GRIN validation suite

Every check, control, and mini-exploration run during development — re-runnable, so any
claim in `DESIGN_RECORD.md` can be reproduced or interrogated (lab meeting, conference
question, reviewer request).

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
| v06 | ensemble stability       | agreement across training seeds        | not an artifact of one run              |
| v07 | trial-count sweep        | recovery + calibration vs n            | reliability across regimes              |
| v08 | identifiability frontier | PI accuracy vs effect size             | the honest limit on PI                  |
| v09 | OOD detection            | in-family vs out-of-family deviance    | the safety net                          |
| v10 | graceful degradation     | lapse contamination                    | fails smoothly, not catastrophically    |
| v11 | amortized comparison     | learned heads vs AIC/BIC               | model comparison in one pass            |
| v12 | RT collinearity probe    | corr(RT, rho) vs corr(acc, rho)        | WHY RTs don't fix PI                    |
| v13 | RT gain                  | counts vs +RT quantiles                | what RTs actually buy                   |
| v14 | architecture recovery    | 5-way SFT confusion                    | dimension-neglect detection             |
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
