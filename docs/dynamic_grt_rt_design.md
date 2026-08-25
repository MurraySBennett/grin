# Dynamic GRT response-time model: design record

Status: **accepted direction; reference implementation validated to gate 4;
gates 5-8 not started; no network trained**

This document replaces the LBA-inspired simulator as the scientific design for
the response-time extension. It does not change the count-only GRIN model. The
existing `rt_lba_generator.py`, checkpoint, figures, and reported recovery
numbers are retained only to reproduce the earlier developmental analysis until
the replacement has passed the gates below.

## 0. Where this actually stands

*Last verified 2026-08-25 by re-running the tests and re-reading the gate
artifacts, not from memory. Update this section whenever a gate moves.*

**Nothing RT-related in this repository is release-ready.** The `cmrt` model on
the website and the RT numbers in the manuscript draft both come from the retired
ballistic generator. Section 6 forbids using either as evidence for the
replacement.

### The two implementations

| | retired | replacement |
|---|---|---|
| generator | `src/data/rt_lba_generator.py` | `src/data/rt_dynamic_grt.py` (scalar reference)<br>`src/data/rt_dynamic_grt_vectorized.py` (fast) |
| training | `scripts/train_rt.py` (requires `--allow-legacy-ballistic`) | **none written yet** |
| architectures | 5-way: serial/parallel x exhaustive/self-terminating, + coactive | 2-way: serial-exhaustive, parallel-exhaustive |
| status | superseded 2026-08-14 | under validation |

The retirement reason, recorded at the time: self-terminating processing's "guess
the unprocessed dimension" step was a dimension-neglect mixture bolted onto an
identification task rather than real SFT self-termination, and coactive was two
summed rates rather than a derived joint decision model. The 84.6% five-way
architecture-recovery figure and the claim that "architecture has no signature in
response proportions" came from that generator; the latter was flagged as false by
external review. Self-terminating and coactive are dropped from version 1 of the
replacement rather than patched.

### Gate status

| gate | what it demands | status | evidence |
|---|---|---|---|
| 1 | reference correctness | **PASS** | `tests/test_rt_dynamic_grt.py` + `..._vectorized.py`, 16/16 passing |
| 2 | discretisation convergence | **FAILS AS SPECIFIED** (1 of 4 conditions) | `results/dynamic_grt_gates.json`, `results/dynamic_grt_gate2_nearzero_highn.json` |
| 3 | prior-predictive plausibility | **PASS**, zero flags | `results/dynamic_grt_gates.json` |
| 4 | static-dynamic bridge | **DONE** (quantified, as the gate asks) | `results/dynamic_grt_gate4_bridge.json` |
| 5 | identifiability | **not started** | — |
| 6 | architecture evidence | **not started** | — |
| 7 | misspecification sensitivity | **not started** | — |
| 8 | empirical check | **not started** | — |

**Gate 1.** 16/16, including
`test_serial_and_parallel_share_responses_but_combine_time_differently`, which
directly proves the by-construction response-equivalence claim that gate 6 later
depends on, and
`test_vectorized_matches_scalar_response_and_rt_distribution`, which ties the fast
generator to the scalar reference. The vectorised generator exists and is verified
— that was the design doc's "only build this once gates 1-4 are solid" step, and
it has been done.

**Gate 2 — the one open question.** Three of four conditions pass cleanly. The
near-zero-drift condition does not, and the high-n rerun that was queued to settle
whether this was Monte Carlo noise **has been run and settled it against the
threshold**: at n = 2,000,000 (vs the original 8,000), response agreement is
0.99419 with a 95% CI that `ci_confirms_below_threshold: true` — it is genuinely
below the 0.995 line, not noise around it.

But the hitting-time criteria pass comfortably in the same run: median difference
0.64 ms against a 10 ms limit, p95 37 ms against a 50 ms limit. So the *timing*
is converged and only the *side of the boundary* disagrees, on ~0.6% of trials, in
the one condition where drift is exactly zero and the crossing is close to a coin
flip by construction. That is a plausible property of the process rather than a
defect in the integrator.

This is a threshold-choice decision, and it is not mine to make. Either:

- justify relaxing the response-agreement threshold for the near-zero-drift
  condition, citing the converged hitting-time distribution as the substantive
  evidence, and record the justification here; **or**
- reduce `dt` further and re-run to show agreement climbing toward 0.995, which
  tests whether the disagreement really is irreducible.

Until one of those happens, gate 2 is failed as written and the design doc's own
rule ("must not be trained until all of the following pass") blocks training.

**Gate 4.** Done analytically rather than by simulation — 2D Gauss-Hermite
quadrature of the two-barrier first-passage probability, cross-checked against
adaptive quadrature to 4.9e-6 and spot-checked against the simulator to 0.013.
Over a 1,715-point `(z, rho, boundary)` grid, the dynamic-vs-static total-variation
distance has median 0.045 and max 0.273. The discrepancy is quantified, which is
what the gate asks for; whether a median TV of 0.045 is acceptable for the
manuscript's interpretation is a separate scientific judgement that has not been
recorded anywhere.

### What has to happen next, in order

1. Resolve gate 2 (relax with justification, or reduce `dt`).
2. Gate 5, identifiability: recovery pilot for `z`, `rho`, `a`, `g`, `t0`,
   stratified as section 5 specifies.
3. Gate 6, architecture evidence: serial-vs-parallel calibration and confusion.
   Counts alone **must** be at chance; if a trained network beats chance on
   architecture from counts alone, something is leaking and must be found first.
4. Gate 7, misspecification: evaluate against the legacy ballistic generator or a
   conventional LBA. High confidence under the wrong process is a failure.
5. Gate 8, empirical: check `data/real/` for trial-level 2x2 identification RT
   data. Without this the extension stays a simulation proof of concept.
6. Only then: write a training script for the dynamic generator, train, evaluate,
   regenerate figures, and revisit the manuscript-scope question.

### Consequences for the current release

- Export `cmrt` with `--status preview`, not `--status release`. The deploy gate
  permits `preview` and prints a note; it refuses the legacy placeholder values.
- Keep the RT section out of the submission, per section 6. That was the standing
  instruction and no gate result has changed it.
- `results/validation/SUMMARY.md` rows `v14`-`v16` describe the **retired**
  five-way model (`v14` is literally "architecture recovery (5-way SFT)"). They
  are regression checks on superseded code, not evidence for the replacement, and
  should not be cited as if they were.

## 1. Scientific target

The target is a joint model of four-way identification responses and response
times in which the familiar Gaussian GRT parameters describe the perceptual
representation and a stochastic evidence process describes how that
representation produces a decision over time.

The first model supports two architectures that are well defined for a 2 x 2
identification task:

- **serial exhaustive**: both dimensional decisions are completed in sequence;
- **parallel exhaustive**: both dimensional decisions begin together and the
  response waits for the slower channel.

Both architectures identify both dimensions. Consequently, for the same latent
drifts and channel paths they produce exactly the same four-way response and
differ only in total decision time. This is intentional: an architecture
classifier must use latency rather than a guessing-induced accuracy cue.

Self-terminating processing is not included. One completed dimensional decision
cannot uniquely determine a response that must name both dimensions. Guessing
the unprocessed dimension is a dimension-neglect mixture, not the
self-termination rule used in logical-rule categorisation.

Coactive processing is also not included in version 1. A defensible candidate
is a circular diffusion with the boundary partitioned into four response arcs,
following Smith's bivariate-drift formulation, but that is a distinct
four-response decision model and must be derived and tested rather than
represented by adding two already-computed rates.

## 2. Generative model

For stimulus cell s, let the trial-level drift vector be

    V_s ~ Normal_2(mu_s, Sigma_s),

where

    mu_s = (z_xs, z_ys)

and, under the same unit-marginal-variance identification used by count-only
GRIN,

    Sigma_s = [[1, rho_s], [rho_s, 1]].

Thus the twelve familiar GRT parameters describe the means and correlations of
the across-trial drift distribution. This follows the dynamic-GRT idea that a
multivariate perceptual representation supplies drift to an evidence process,
and specifically the bivariate-normal drift representation used by Smith
(2019). It is not a claim that the resulting response probabilities equal the
static GRT orthant probabilities.

Conditional on V_s, each dimension has an independent standard Wiener evidence
process in internal time u:

    dE_d(u) = V_sd du + dW_d(u),             d in {x, y}.

Each process starts at zero and terminates on first reaching either +a or -a:

    T*_d = inf{u > 0 : |E_d(u)| >= a}.

The sign of the crossed boundary supplies the response on that dimension. The
two signs give the ordinary four-way response code:

    R = 2 I(E_x(T*_x) > 0) + I(E_y(T*_y) > 0).

A participant-level processing rate g converts internal decision time to
seconds. With nondecision time t0,

    RT_serial   = t0 + (T*_x + T*_y) / g,
    RT_parallel = t0 + max(T*_x, T*_y) / g.

The first nuisance-parameter set is therefore `(t0, boundary, rate)`. The
within-trial diffusion coefficient is fixed to one, as are the marginal
variances of the GRT drift distributions. Both constraints are needed to fix
scale. Starting points are fixed at zero and the two boundaries are symmetric;
response bias continues to live in the GRT mean locations rather than in an
additional response-specific threshold.

The current ballistic generator is related to this model only as a limiting
case: when within-trial diffusion noise becomes negligible, the response tends
to the sign of the trial-level drift and a channel's time tends to a constant
divided by the drift magnitude. The new model retains that useful connection
while allowing the evidence process itself to generate correct and incorrect
decisions.

## 3. Interpretation and non-equivalence to count-only GRIN

The dynamic model and count-only GRIN share a GRT-shaped latent representation,
but their numerical parameters are not automatically interchangeable:

- count-only GRIN maps one Gaussian percept directly to a response quadrant;
- dynamic GRIN maps a Gaussian drift through a first-passage process.

The decision boundary changes the mapping from `(z, rho)` to response
probabilities, and response time helps identify that mapping. A bridge study
must quantify when estimates from the static and dynamic models agree, how they
differ, and whether the same construct thresholds remain calibrated. Until that
study is complete, the RT estimator must be described as estimating a dynamic
GRT representation rather than as merely adding precision to the existing
static estimator.

The correlation rho is across-trial dependence between the two drift
components. Conditional Wiener noise is independent across channels in version
1. This separates perceptual dependence from moment-to-moment process crosstalk;
the latter is future model expansion, not silently absorbed into rho.

## 4. Numerical rules

The reference simulator uses Euler-Maruyama steps and records first passage by
interpolating the terminal step. The step size is a numerical approximation,
not a model parameter, and must pass a convergence check.

There is no RT clipping. A finite simulation horizon is an implementation guard,
not an observation model. Trials that fail to terminate by that horizon are
returned explicitly as censored (`response = -1`, `rt = NaN`) so that prior
choices can be revised or a censoring model added. They must never be converted
into an architecture-specific point mass at the horizon.

## 5. Validation gates before training

The full vectorised generator and neural posterior must not be trained until all
of the following pass:

1. **Reference correctness**: deterministic tests, response coding, exact
   serial/parallel response agreement, and explicit censoring.
2. **Discretisation convergence**: response probabilities and RT quantiles are
   stable across at least two smaller integration steps.
3. **Prior-predictive plausibility**: RT range, leading edge, skew, conditional
   correct/error distributions, speed-accuracy behaviour, and censoring rate are
   plausible across the complete GRT and nuisance priors.
4. **Static-dynamic bridge**: quantify the discrepancy between dynamic response
   probabilities and static GRT orthant probabilities over `(z, rho, a)`.
5. **Identifiability**: recovery of `z`, `rho`, `a`, `g`, and `t0`, stratified by
   trial count, accuracy/entropy, boundary, rate, and correlation magnitude.
6. **Architecture evidence**: serial-versus-parallel calibration and confusion,
   including regimes in which the two are practically indistinguishable. Counts
   alone must be at chance because the architectures share responses.
7. **Misspecification sensitivity**: evaluate the trained estimator on data from
   at least one alternative channel process (for example a conventional LBA or
   the legacy distance-to-bound ballistic model). High confidence under the wrong
   process is a failure, even if within-simulator recovery is excellent.
8. **Empirical check**: where trial-level 2 x 2 identification RT data are
   available, compare observed and prior/posterior-predictive response-conditional
   RT distributions. Without this check, the extension remains a simulation
   proof of concept.

## 6. Repository and manuscript migration

- The count-only packages, checkpoint, and final retraining are unaffected.
- Do not use the current RT checkpoint or its 84.6% five-way result as evidence
  for the replacement model.
- The legacy generator and results remain reproducible but must be marked
  superseded and excluded from default training commands.
- Manuscript and website claims should be migrated only after the replacement
  passes the gates above. Until then, the current RT section is historical draft
  material rather than a release claim.

## References anchoring the design

- Ashby (2000), *A stochastic version of General Recognition Theory*.
- Townsend, Houpt, and Silbert (2012), *General Recognition Theory extended to
  include response times: Predictions for a class of parallel systems*.
- Fific, Little, and Nosofsky (2010), *Logical-rule models of classification
  response times*.
- Smith (2019), *Linking the diffusion model and General Recognition Theory:
  Circular diffusion with bivariate-normally distributed drift rates*.

