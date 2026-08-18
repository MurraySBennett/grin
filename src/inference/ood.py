"""
ood.py — training-envelope / input-support diagnostics.

WHAT THIS DOES NOT DO, stated first because the name of this module invites the
wrong reading: neither function here tests whether the observed matrix is
consistent with the Gaussian-GRT model family in the abstract. It can't. A single
stimulus's response distribution is three numbers (two marginals, one joint cell),
and the unconstrained twelve-parameter model spends exactly three free parameters
per stimulus, so for essentially any response-proportion table -- however it was
generated: a lapse mixture, a mid-block criterion shift, a non-Gaussian process, a
reversed response mapping -- SOME choice of the twelve parameters reproduces it
exactly (Frechet-Hoeffding + IVT on the per-stimulus tetrachoric correlation; see
`docs/GRT_model_spec.md` / the manuscript's Introduction for the full argument).
There is no "no representation in the model family" case to detect at the level of
response proportions alone.

WHAT THIS ACTUALLY DOES: both functions score agreement between the observed
matrix and the TRAINED NETWORK's own fit to it (its posterior mean, or replicates
drawn from its posterior) -- not agreement with the best-fitting parameters an
exhaustive search would find. A large value means the network's own answer for
this matrix doesn't reproduce the matrix well, which can happen for several
distinct reasons that this score does not itself distinguish:
  - the matrix genuinely falls outside the training prior's support (extreme z/rho
    magnitude, reversed sign/orientation, an unusual trial-count regime) and the
    network extrapolated poorly;
  - ordinary network approximation error, unrelated to support;
  - posterior shrinkage pulling the point estimate away from a good fit;
  - low trial counts inflating the deviance's own sampling variance;
  - model-class averaging in the point estimate blurring a matrix that any single
    model class would fit well.
Read a large value as "this input warrants caution, inspect it" -- an envelope /
input-support warning -- not as a hypothesis test result about which GRT
assumption, if any, the data violate. `flag_envelope_warning`/`envelope_deviance`
are the names to use going forward; `flag_ood`/`ood_deviance` remain as aliases
below for existing callers, not because "OOD" is still the right description.
"""
import numpy as np

from .predict import predict_posterior

try:
    from src import grt_model as gm
except ImportError:
    import grt_model as gm


def _vec_multinomial(probs, T, rng):
    """Vectorised multinomial replicates. probs (S,4,4), T (4,) -> counts (S,4,4)."""
    S = probs.shape[0]
    out = np.zeros((S, 4, 4), dtype=np.int64)
    for st in range(4):
        remaining = np.full(S, int(T[st]))
        rem_p = np.ones(S)
        for r in range(3):
            cp = np.clip(probs[:, st, r] / np.maximum(rem_p, 1e-12), 0.0, 1.0)
            c = rng.binomial(remaining, cp)
            out[:, st, r] = c; remaining -= c; rem_p -= probs[:, st, r]
        out[:, st, 3] = remaining
    return out


def posterior_predictive_pvalue(model, counts, trials, n_post=300, rng=None):
    """counts (N,16), trials (N,4) -> p-values (N,). Low p => the observed matrix is
    atypical of what the network's OWN posterior predicts for it -- an
    input-support/envelope signal, not a model-family test (see module docstring)."""
    rng = np.random.default_rng() if rng is None else rng
    counts = np.asarray(counts); trials = np.asarray(trials)
    N = counts.shape[0]
    samples = predict_posterior(model, counts, trials, n_samples=n_post)["samples"].numpy()
    counts4 = counts.reshape(N, 4, 4).astype(float)
    pvals = np.zeros(N)
    for i in range(N):
        zx, zy, rho = gm.unpack(samples[:, i, :])
        probs = np.clip(gm.forward_probabilities(zx, zy, rho), 1e-12, 1.0)     # (S,4,4)
        ll_obs = (counts4[i][None] * np.log(probs)).sum((1, 2))                # (S,)
        rep = _vec_multinomial(probs, trials[i], rng)
        ll_rep = (rep * np.log(probs)).sum((1, 2))                            # (S,)
        pvals[i] = np.mean(ll_rep <= ll_obs)      # obs in the lower tail => small p
    return pvals


def flag_envelope_warning(pvalues, alpha=0.05):
    return np.asarray(pvalues) < alpha


flag_ood = flag_envelope_warning  # deprecated alias -- see module docstring


def _deviance(counts4, probs):
    """Reconstruction deviance: 2*(LL_saturated - LL_model). ~0 => the network's own fitted
    parameters reproduce the matrix; this is NOT a test of whether some other,
    unfound parameter vector could -- the saturated model can basically always
    find one (see module docstring)."""
    T = counts4.sum(2, keepdims=True)
    phat = np.clip(counts4 / np.maximum(T, 1), 1e-12, 1.0)
    p = np.clip(probs, 1e-12, 1.0)
    return 2.0 * np.sum(counts4 * (np.log(phat) - np.log(p)), axis=(1, 2))


def envelope_deviance(model, counts, trials):
    """
    Fast input-support/envelope score: the posterior-mean reconstruction deviance of the
    network's own posterior-mean fit against the observed matrix. ~0 => the
    network's fitted parameters reproduce the matrix; large => the network's
    fitted parameters do not, which most often means this matrix falls outside
    the region its training prior populated (unusual sign/orientation, an
    extreme z or rho magnitude, an atypical trial-count regime) -- see the
    module docstring for the other possible causes and why this is not a test
    of the abstract model family. counts (N,16), trials (N,4) -> deviance (N,).
    """
    from .predict import predict_point
    counts = np.asarray(counts)
    pred = predict_point(model, counts, trials).numpy()
    zx, zy, rho = gm.unpack(pred)
    probs = gm.forward_probabilities(zx, zy, rho)          # (N,4,4)
    return _deviance(counts.reshape(-1, 4, 4).astype(float), probs)


ood_deviance = envelope_deviance  # deprecated alias -- see module docstring
