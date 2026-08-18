# GRIN canonical GRT model specification

The single source of truth for the GRT parameterization, model-class constraints,
identifiability, prior, and validation gates that the whole GRIN pipeline
(simulator, network heads, losses, evaluation) shares. Implemented in
`src/grt_model.py`; validated by the checks in `validation/` (see
`validation/README_validation.md`).

---

## 1. The identifiability result that anchors everything

A single 4x4 confusion matrix contains exactly **12 independent numbers** (4
stimulus rows, each a distribution over 4 responses = 3 free per row). Under
decisional separability (DS) — which every model in this family assumes — the GRT
identification model, expressed in its *identified* coordinates, has at most 12
free parameters. So the fully unconstrained model is *exactly saturated*, and
every restricted model is *over*-identified. **There is no structural null space
to average over.** This is the coordinate system mdsdt fits, which is why mdsdt
can fit the "full" model to one matrix while a raw means+variances+bounds
parameterization cannot (that raw form is redundant/over-parameterized).

The only GRT non-identifiability — separating perceptual from decisional sources
(Silbert & Thomas, 2013) — is resolved here by the DS convention: bounds fixed,
marginal position attributed to the perceptual mean. It re-appears only if DS is
relaxed; that is out of scope for the current family, and is exactly where the
design-based identification thread (RTs, multiple bound conditions) would be
needed.

---

## 2. Canonical parameterization (identified coordinates)

Stimulus order (matches mdsdt and grtools): s0=A1B1, s1=A1B2, s2=A2B1, s3=A2B2,
with dimension A = x, dimension B = y.

The canonical 12-vector the network predicts:

    [ zx_0..zx_3 ,  zy_0..zy_3 ,  rho_0..rho_3 ]

- `zx_i`, `zy_i`: per-stimulus marginal **z-scores** (sensitivities), i.e. the
  standardized distance of stimulus i's perceptual mean from the decision bound on
  each dimension (bound fixed at 0, unit variance). These are d'-like quantities.
- `rho_i`: within-stimulus perceptual correlation.

**Sign convention (design-consistent):** dimension level 1 sits below its bound
(negative z), level 2 above (positive z). Response BIAS is captured by *asymmetric
magnitudes* about the fixed bound, not by sign flips — so nothing is lost by
fixing signs to the design. This is a documented prior choice, revisitable.

Why this and not raw means/covariances/criteria: those 26 raw numbers are a
redundant over-parameterization of the same 12 identified quantities. Predicting
them would mean predicting non-identified combinations. The identified coordinates
are minimal, psychophysically interpretable, and mdsdt-native.

---

## 3. Model-class constraints (complete and exact)

Correlation structure:

| tag    | constraint                                  | # corr params |
|--------|---------------------------------------------|---------------|
| PI     | all rho = 0                                 | 0             |
| RHO1   | one shared (equal) correlation              | 1             |
| (free) | correlations differ across stimuli          | 4             |

Perceptual separability (ties z-scores; both dimensions have unit variance so PS
reduces exactly to equality of the relevant z-scores):

| tag    | constraint                                        | effect         |
|--------|---------------------------------------------------|----------------|
| PS(A)  | x-sensitivity invariant across B: zx_0=zx_1, zx_2=zx_3 | 4 -> 2 x-params |
| PS(B)  | y-sensitivity invariant across A: zy_0=zy_2, zy_1=zy_3 | 4 -> 2 y-params |
| PS     | PS(A) and PS(B)                                    | both           |

The 12 model names are the crossing of {PI, RHO1, free} with {PS, PS(A), PS(B),
none}, all under DS. Free-parameter counts (data df = 12):

| model        | free | identified |
|--------------|------|------------|
| pi_ps_ds     | 4    | yes        |
| rho1_ps_ds   | 5    | yes        |
| pi_psa_ds    | 6    | yes        |
| pi_psb_ds    | 6    | yes        |
| rho1_psa_ds  | 7    | yes        |
| rho1_psb_ds  | 7    | yes        |
| pi_ds        | 8    | yes        |
| ps_ds        | 8    | yes        |
| rho1_ds      | 9    | yes        |
| psa_ds       | 10   | yes        |
| psb_ds       | 10   | yes        |
| ds           | 12   | yes (saturated) |

Every class is identifiable from a single matrix. No accident-rejection is used:
a RHO1 model with rho ~ 0 is a genuine near-PI point, and we WANT the continuum so
the PI-vs-weak-RHO1 boundary (research question 1) is learned with calibrated
uncertainty rather than artificially separated.

---

## 4. Prior (Option 1: explicit, correct-by-construction)

Fix bounds = 0 and unit variances (WLOG under DS), then sample the identified
parameters directly:

- z-score magnitudes ~ U(0, z_max), z_max ~ 3 (chance to near-ceiling per
  dimension), signed by design; ties applied for PS classes.
- correlations ~ U(-r_max, r_max), r_max ~ 0.9, including the near-zero band.

No rejection loop, no accuracy calibration: draw params, run the forward map,
sample multinomial counts. The induced distribution over confusion matrices is
correct by construction. **Coverage is then verified, not engineered** — plot the
induced distributions over accuracy AND structural features (per-stimulus
accuracy, bias/asymmetry indices); if a region of interest is thin, reshape the
*prior* explicitly, never with a post-hoc filter. (Verified in testing: the
default prior already spans ~31%-99% accuracy across classes.)

Ranges are set from psychophysical plausibility so the amortized network covers
any plausible experiment; real grtools/mdsdt fits are used only as a sanity check
that the envelope brackets observed data, never to define it (that would make the
prior experiment-specific).

---

## 5. Cross-reference transforms (implemented in `grt_model.py`)

- **mdsdt**: identical coordinates and order. Reports (mu_r, sd_r, mu_c, sd_c, rho)
  with sd=1, so mu_r=zx, mu_c=zy. Transform is the identity (verified from source).
- **grtools**: different but equivalent frame — fixes the reference stimulus s0 at
  the origin and reports decision bounds (a1,a2) plus other means/covariances.
  Rigid-translation map: means_i = (zx_i-zx_0, zy_i-zy_0); bounds = (-zx_0,-zy_0);
  cov_i = [[1,rho_i],[rho_i,1]]. Means/bounds map is exact and round-trips to
  machine precision. The covariance-reference detail (whether grtools pins s0's
  covariance) as implemented in `to_grtools()` here is PROVISIONAL and remains
  unvalidated against grtools' own output -- **but this specific utility function
  is not what any reported number depends on.** The manuscript's actual
  grtools comparisons (`scripts/R/fit_baselines.R`'s `extract_grtools_params()`,
  feeding `results/mle_fits/baseline_fits.csv` and every `\rhoGrin`/`\rhoAicbic`-
  style number in the paper) read `rho` directly off grtools' own
  `best_model$covmat[[i]][1,2]` in a live R session, confirmed against grtools'
  source and a live run (see the RESOLVED note in `fit_baselines.R`'s header) --
  an independent extraction path that does not go through this transform. If a
  future use needs `to_grtools()` itself (e.g. converting a GRIN fit into
  grtools' own plotting functions), validate it first; nothing currently
  published relies on it being correct.

---

## 6. Inference target and uncertainty (A+B)

GRIN predicts the 12 identified parameters and derives the model-class inference
from their structure (regression-first). Because the coordinates are identified,
the posterior uncertainty is honest **sampling noise** (tight with many trials,
wide with few) rather than structural null-space blur — and it is controlled by
trial count, which the adaptive loop manipulates.

Uncertainty representation — decision flagged for Phase 3: prefer **neural
posterior estimation** (network outputs a distribution over parameters directly)
over point-regression-plus-MC-dropout, since dropout uncertainty tends to be
cruder and often miscalibrated. Calibration is a hard gate (Section 7), so the
representation that makes "posterior" literally true is preferred.

Identification vs adaptation (for the adaptive layer): you can only *steer* a
representation along directions you can *measure*. Active stimulus selection that
maximizes information gain expands the measurable subspace and thus the trainable
subspace; the two objectives (reduce uncertainty about theta vs drive theta toward
an expert target) are blended, not sequential, with an explore-early/exploit-later
schedule. A lower-risk sibling application: on-the-fly stimulus **calibration /
multidimensional staircasing** for standard GRT experiments (steer z-scores toward
a target separation to speed pilot testing).

---

## 7. Validation gates (hard requirements before trust)

1. Parameter recovery on simulations (truth known): per-parameter MAE, bias,
   correlation on held-out sims; target <= grtools/mdsdt recovery error.
2. Model identification: agreement between class inferred from predicted params and
   ground truth; 12-class confusion matrix.
3. Uncertainty calibration: empirical coverage of credible intervals (nominal 90%
   contains truth ~90% of the time).
4. Head-to-head vs grtools and mdsdt on shared matrices: accuracy AND wall-clock.
5. Training-envelope / posterior-predictive checks on real data; flag and defer
   cases outside the trained envelope to MLE. (This is an input-support
   diagnostic, not a test of the Gaussian-GRT model family itself -- see
   `src/inference/ood.py`'s module docstring and the manuscript's
   identifiability-frontier study for why that distinction is load-bearing.)

Gates 1-3 are hard requirements; 4 quantifies the payoff; 5 is the safety net.

---

## 8. Status

All of Sections 1-7 are implemented and validated, not aspirational: the identified
parameterization, prior, and transforms live in `src/grt_model.py`; the generator in
`src/data/generator.py`; the NPE distributional head and model-class inference in
`src/models/`; the Section-7 gates as the checks in `validation/` (16/16 passing --
see `validation/README_validation.md`). Built since this spec was first written and
not covered above: a response-time extension (processing architecture and LBA
parameters alongside the GRT representation, `src/data/rt_lba_generator.py`) and an
adaptive/online inference layer (`src/adaptive/`). The still-open items from the
original roadmap are in Section 9.

---

## 9. Open items

- **Non-DS models.** This spec is DS-only throughout (Section 1); relaxing
  decisional separability needs design-based identification (RTs, multiple bound
  conditions) and is out of scope for the current model family.
- **The `to_grtools()` covariance-reference map** (Section 5) is flagged provisional
  there, pending validation against grtools' own output on shared matrices --
  narrowly scoped to that one utility function, which nothing published currently
  depends on; the manuscript's own grtools comparisons use an independent,
  already-verified extraction path (see Section 5).
- **Design-consistent sign convention** for z-scores (Section 2) is a documented
  prior choice, not a derived necessity -- revisitable if a use case needs it.

Resolved since this spec was first written: NPE (not MC Dropout) is the uncertainty
representation (Section 6); `z_max`/`r_max` prior coverage (Section 4) is verified,
not just assumed (`validation/README_validation.md`, check v02).