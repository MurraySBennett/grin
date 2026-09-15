# GRIN Presentation Module Spine

## Core Claim

GRIN turns Gaussian GRT fitting from a slow, fragile optimization workflow into a
fast, uncertainty-aware measurement tool. The contribution is not a new theory of
perception. It is a practical inference system for an established model class,
with explicit validation and explicit limits.

## The Talk In One Sentence

GRT can distinguish different sources of multidimensional identification errors,
but existing single-observer fitting is too slow and brittle for repeated,
uncertainty-aware, or adaptive workflows; GRIN amortizes that inference while
making the limits of the design visible.

## The Shared Narrative

1. **Why GRT matters**
   Accuracy alone can collapse distinct psychological explanations. A 2x2
   identification matrix can reflect sensitivity loss, perceptual separability
   failures, perceptual independence failures, decisional effects, or attention
   changes. GRT gives these distinctions a formal language.

2. **Why the current workflow is limiting**
   Existing fitting tools are useful and established, but they rely on numerical
   optimization. Point estimates are slow enough to matter when repeated many
   times; intervals require still more fitting; sparse matrices can fail or
   produce unstable estimates.

3. **What GRIN does**
   GRIN trains a neural network offline on simulated GRT parameter/matrix pairs.
   At use time, one forward pass maps a confusion matrix to an approximate
   posterior over the 12 identified Gaussian GRT parameters and construct
   probabilities for PI and PS.

4. **What had to be validated**
   A fast inverse is only useful if it recovers parameters, reports honest
   uncertainty, agrees with established methods where comparison is possible, and
   fails in interpretable ways when the design has weak information.

5. **What the validation says**
   Sensitivities recover strongly. Correlations recover more weakly because the
   matrix contains less information about them. Intervals need calibration checks:
   sensitivities are conservative and correlations overconfident before
   correction. GRIN is much faster than optimization and has the largest practical
   advantage in sparse-data regimes.

6. **What this enables**
   Because inference is cheap, GRIN makes repeated analyses practical:
   simulation sweeps, quick software comparisons, precision-based adaptive
   stopping, and exploratory checks of whether new matrices resemble the
   training distribution.

7. **What it does not solve**
   GRIN inherits the model and design limits of Gaussian GRT: decisional
   separability is assumed, correlations can be weakly identified, construct
   probabilities depend on the training prior, and a single Gaussian posterior
   head cannot represent every posterior geometry.

8. **Why that is still useful**
   The tool makes a mature cognitive model easier to use, easier to validate, and
   easier to embed in design workflows. It also makes the boundaries of the
   measurement problem more visible rather than hiding them behind optimizer
   behavior.

## Reusable Modules

### M1. Opening Motivation: Same Accuracy, Different Mechanisms

Purpose: make the problem real before introducing the method.

Best figure: `results/figures/vignette.png`.

Core point: four observers can have matched accuracy while requiring different
scientific interpretations and follow-up actions.

Audience variants:

- Lay: same score, different reasons for mistakes.
- Student: confusion matrices contain structure beyond overall accuracy.
- Expert: identical marginal performance can hide PS, PI, DS, or attention
  differences.

### M2. GRT Refresher

Purpose: define the constructs only as much as the audience needs.

Core point: GRT separates perceptual interactions from decisional interactions in
multidimensional identification.

Audience variants:

- Lay: a model of what people perceived versus how they decided.
- Student: perceptual separability, perceptual independence, decisional
  separability.
- Expert: Gaussian GRT, 2x2 identification, 12 identified continuous parameters,
  constrained model classes.

### M3. Bottleneck: Numerical Fitting

Purpose: justify why an inference tool is worth building.

Core point: optimization works, but repeated fitting, intervals, sparse data, and
adaptive workflows make speed and convergence matter.

Evidence:

- Existing baselines are `mdsdt`, `grtools`, and a direct Python MLE.
- Sparse matrices are where fitting is hardest and where dropping cases is most
  consequential.

### M4. GRIN Architecture At The Right Level

Purpose: explain amortization without overclaiming "AI".

Core point: train an inverse of the simulator once; use it many times.

Essential details:

- Inputs: canonical 4x4 confusion matrix and trial totals.
- Outputs: Gaussian approximate posterior over 12 parameters; construct
  probabilities for PI and PS.
- Training data: simulated Gaussian GRT observers across 12 model classes.

Expert-only details:

- Parameters are represented in an unconstrained space.
- Regression head returns full covariance via Cholesky factor.
- Construct heads are amortized model comparison under the training prior.

### M5. Parameter Recovery

Purpose: show that the inverse learned the main structure.

Best figure: `results/figures/recovery/recovery_grin.png`.

Headline:

- Marginal sensitivities: recovered vs true `r = 0.91`.
- Correlations: recovered vs true `r = 0.59`.

Interpretation: this asymmetry is part estimator and part design; correlations
have a weaker signature in the confusion matrix.

### M6. Calibration And Uncertainty

Purpose: prevent the talk from sounding like "the network is fast, therefore it
is right."

Best figure: `results/figures/calibration_breakdown.png`.

Headline:

- Nominal 90% intervals cover 94.5% for sensitivities.
- Nominal 90% intervals cover 84.1% for correlations.
- A held-out scale correction improves coverage and ships as an opt-in package
  feature.

Key framing: the posterior is useful because it is checked, not because neural
networks automatically produce calibrated uncertainty.

### M7. Comparison To Existing Software

Purpose: position GRIN relative to familiar tools.

Best figure: `results/figures/comparison_to_r.png`.

Headline:

- Single-matrix GRIN inference is about 0.97 ms.
- In the sparse 5-20 trial-per-stimulus regime, GRIN MAE is 0.38 versus 1.23 for
  the direct Python MLE.
- GRIN returns an estimate for every matrix.

Important caveat: complete-case comparisons select matrices every method fitted.

### M8. Design Envelope

Purpose: show that GRIN is not only an estimator; it can map when the design is
informative.

Best figure: `results/figures/accuracy_stratified.png`.

Core point: PI and PS are best recovered in different accuracy regions; the
classic near-60% overall target is a compromise, not a universal truth.

### M9. Published Data Compatibility

Purpose: show contact with real examples without overstating ground truth.

Best figure: `results/figures/real_data_spaces.png` or
`results/figures/real_data_params.png`.

Core point: on five published example datasets, GRIN falls within the range
spanned by established tools.

Caveat: this is a compatibility check, not posterior predictive adequacy.

### M10. Adaptive Stopping

Purpose: show the new workflow enabled by cheap inference.

Best figure: `results/figures/adaptive_stopping.png`.

Headline:

- Precision-based adaptive stopping reduced mean trial requirements by 73.5% in
  simulation under the tested criterion.

Caveat: do not generalize to all populations or all stopping criteria.

### M11. DS Non-identifiability Teaching Branch

Purpose: explain a known limitation when teaching or answering a reviewer.

Best figure: `results/figures/robustness_addendum.png`.

Core point: DS failures can be absorbed by transforming perceptual space; a
single confusion matrix cannot distinguish the decisional and perceptual
explanations.

Use only in:

- Math Psych long version.
- Lab meeting.
- Teaching.
- Reviewer response.

Do not use as a main manuscript result unless requested.

### M12. Limitations And Future Work

Purpose: close honestly and set up the next research arc.

Core points:

- DS requires design-based identification or additional observables.
- Correlations remain harder than sensitivities.
- Prior sensitivity matters for construct probabilities.
- A richer posterior family could improve non-Gaussian posterior shapes.
- Group-level extensions and RT models are complementary next steps.

### M13. Job Talk Bridge

Purpose: connect GRIN to a broader research identity.

Core point: this project is an example of building real-time measurement tools
for adaptive perceptual-expertise training. GRIN is the inference engine that
lets a training system estimate a learner's perceptual representation quickly
enough to adapt the next stimulus, difficulty level, or feedback target.

Likely bridge:

- Past: formal models of perception/decision.
- Present: amortized inference and validation.
- Future: adaptive expertise training systems that identify what a learner is
  currently seeing, compare it with expert perceptual structure, and select
  stimuli that should move the learner toward that expert representation.

Example domains:

- Melanoma identification: real-time estimates suggest which diagnostic
  dimensions a trainee is insensitive to, confounding, or using decisionally.
- Any perceptual-expertise domain: radiology, pathology, face/expression
  perception, speech perception, auditory categorization, visual search, or
  other multidimensional classification settings.

Training logic:

- If a learner struggles with a dimension, reduce difficulty or isolate that
  dimension.
- If a learner is doing well, increase difficulty or present more diagnostic
  near-boundary stimuli.
- If the learner's representation differs systematically from experts, choose
  stimuli that should reshape the relevant perceptual relation.
- If the model is uncertain, collect targeted trials rather than continuing a
  fixed curriculum.

## Suggested Core Deck Spine

The default 15-minute non-DS project talk should probably be:

1. Title / one-sentence claim.
2. Same accuracy, different mechanisms.
3. Minimal GRT refresher.
4. Why fitting is the bottleneck.
5. GRIN: learn the inverse once.
6. What the network returns.
7. Recovery result.
8. Calibration result.
9. Software comparison.
10. Design envelope.
11. Adaptive stopping.
12. Limits.
13. Takeaway.

Short talks should remove detail, not simply speed through this list.
