# Dynamic GRT response-time model: design record

Status: **accepted direction; reference implementation under validation**

This document replaces the LBA-inspired simulator as the scientific design for
the response-time extension. It does not change the count-only GRIN model. The
existing `rt_lba_generator.py`, checkpoint, figures, and reported recovery
numbers are retained only to reproduce the earlier developmental analysis until
the replacement has passed the gates below.

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

