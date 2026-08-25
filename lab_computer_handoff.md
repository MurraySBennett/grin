# Handoff: dynamic-GRT RT model, validation gates 4-8

> **SUPERSEDED — historical record of a 2026-08-14 handoff, not current status.**
> Work has moved past it: the gate-2 high-n rerun this document queued as "first
> thing to do" has been run, gate 4 has been completed analytically, and the
> vectorised generator has been built and tested. For where the RT extension
> actually stands, read **`docs/dynamic_grt_rt_design.md` §0**, which is kept
> current against the gate artifacts. Do not plan from the task list below.

Paste this whole message into the new Claude Code session on the lab computer
(same repo, `OneDrive - The Ohio State University\projects\grin` -- should already
be synced via OneDrive, so no files need copying, just context).

## Decision made (this session, laptop, 2026-08-14)

We're retiring the five-recipe LBA-inspired RT generator
(`src/data/rt_lba_generator.py`, the 84.6% five-way architecture-recovery result
currently in the manuscript) as the scientific basis for the RT extension. Reason:
self-terminating processing's "guess the unprocessed dimension" coin flip is a
dimension-neglect mixture bolted onto an identification task, not real SFT
self-termination, and it made a manuscript claim ("architecture has no signature in
response proportions") that was flagged as false by external review. Coactive
processing was also just two summed rates, not a derived joint decision model.

Replacement: a genuine stochastic-evidence model (Ashby 2000, Smith 2019) --
bivariate-normal trial-level drift feeds two independent Wiener first-passage
processes; serial-exhaustive and parallel-exhaustive architectures share EXACTLY
the same response by construction (verified, see below) and differ only in how the
two finishing times combine (sum vs max). Self-terminating and coactive are
dropped from version 1 rather than patched.

Full design spec: **`docs/dynamic_grt_rt_design.md`** -- read this first, it has the
generative model, the 8 validation gates, and the manuscript-migration rule
(current RT section stays "historical draft material, not a release claim" until
the gates pass). Historical correction recorded in `docs/DESIGN_RECORD.md` (search
"CORRECTION (2026-08-14") and `manuscript/SUBMISSION_PLAN.md`.

**Manuscript scope for this BRM submission is explicitly NOT decided yet** -- asked
and the answer was "don't touch the manuscript's RT framing, finish the validation
gates first, decide with real evidence." Don't revise 60_extensions.tex /
90_discussion.tex / supplement.tex RT content without raising that question again.

## What's already done (laptop, zero/low compute)

- `src/data/rt_dynamic_grt.py` -- scalar reference simulator. Euler-Maruyama
  first-passage, explicit censoring (no RT clipping), serial/parallel architectures.
- `tests/test_rt_dynamic_grt.py` -- 11/11 passing. Notably
  `test_serial_and_parallel_share_responses_but_combine_time_differently` directly
  proves the by-construction response-equivalence claim above.
- `scripts/check_dynamic_grt_gates.py` -- implements design-doc **gate 2**
  (discretisation convergence) and **gate 3** (prior-predictive plausibility).
  Results in `results/dynamic_grt_gates.json`. Status:
  - Gate 3 (prior-predictive): clean. Zero censoring over 300 draws across the
    full GRT+nuisance prior, RT range 0.26-2.13s, 100% right-skewed, 84% of draws
    show slower error RTs than correct RTs (expected pattern), both
    speed-accuracy correlations positive as required. No flags.
  - Gate 2 (convergence): 3/4 conditions clearly pass (moderate drift, strong
    drift, weak-drift/large-boundary). **near-zero-drift is borderline**: 99.45%
    response agreement vs a 99.5% threshold I set, but the hitting-time
    distribution itself is fine (median diff 0.6ms, p95 34ms). Read as the
    expected hard case (path lingers near the boundary when drift~0, so ~0.5% of
    trials have a step-size-sensitive tie on which side they cross), not
    necessarily a bug -- but NOT independently confirmed at larger n. **First
    thing to do here**: rerun `scripts/check_dynamic_grt_gates.py` with a much
    larger `n` (the lab machine can afford it -- try n=200_000+ for the
    near-zero-drift condition specifically) to see whether 99.45% holds steady or
    drifts toward/away from the threshold as Monte Carlo noise shrinks.

## What needs the lab computer (design doc S5, gates 4-8)

Read `docs/dynamic_grt_rt_design.md` section 5 for the exact gate definitions.
In order:

1. **Confirm gate 2's near-zero-drift edge case** at high n (above).
2. **Gate 4, static-dynamic bridge**: quantify how dynamic response probabilities
   (from `simulate_dynamic_grt_trials`, large n, marginalized to just the 4-way
   response) differ from the static GRT orthant probabilities (`src/grt_model.py`)
   over a grid of `(z, rho, boundary)`. No vectorised generator needed yet -- the
   scalar simulator at large n is enough for a grid sweep, but it's slow per-point
   (Python loop over time steps), so this is the first place raw CPU helps a lot.
3. **Vectorised/GPU generator**: only build this once gates 1-4 are solid, per the
   design doc's own ordering ("must not be trained until all of the following
   pass"). This is training-adjacent infrastructure -- appropriate for the lab
   machine, not the laptop.
4. **Gate 5, identifiability**: small recovery pilot (design doc says "before
   committing the lab computer to the full network") -- z, rho, boundary, rate, t0
   recovery, stratified by trial count/accuracy/entropy/boundary/rate/correlation
   magnitude.
5. **Gate 6, architecture evidence**: serial-vs-parallel calibration/confusion.
   Counts alone MUST be at chance here (by construction, per the response-equivalence
   proof) -- if a trained network somehow beats chance on architecture from counts
   alone, something is leaking and needs to be found before going further.
6. **Gate 7, misspecification sensitivity**: evaluate the trained estimator on data
   generated by a different channel process -- e.g. the legacy
   `rt_lba_generator.py` ballistic model, or a conventional LBA with start-point
   variability. High confidence under the wrong process = failure, even with
   excellent within-simulator recovery.
7. **Gate 8, empirical check**: if real 2x2 identification RT data are available
   (check `data/real/`), compare observed vs prior/posterior-predictive
   response-conditional RT distributions.

Only after gates 4-8 pass: retrain, regenerate figures, and THEN revisit the
manuscript-scope question that's currently deferred.

## Also still queued from before this pivot (unrelated, can interleave)

From `docs/DESIGN_RECORD.md` / memory `grin-manuscript-validation-numbers`: all
remaining `\pending{}` production numbers for the COUNT-ONLY model (Study 2 clean
speed/AIC-BIC, Study 3 frontier calibration/stratified recovery, envelope-warning
operating characteristics), the adaptive-savings compute-or-drop decision, and
final count-only GRIN retraining + ONNX export (this one's independent of the RT
pivot and was already recommended to proceed regardless).
