"""
ood.py — out-of-distribution / misspecification detection via posterior-predictive
checks.

Given an observed confusion matrix, we infer the posterior, then for each posterior
sample simulate a replicate matrix at the same trial counts and compare how well the
OBSERVED matrix fits (multinomial log-likelihood) against how well the REPLICATES
fit. If the observed matrix fits far worse than the model's own replicates, it lies
outside what the model family can produce — flag it.

posterior_predictive_pvalue returns, per matrix, a value in [0,1]:
  ~uniform for in-distribution data; near 0 for misspecified/OOD data.
Flag when p < alpha (e.g. 0.05) and route those cases to MLE or human review.
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
    """counts (N,16), trials (N,4) -> p-values (N,). Low p => OOD / misspecified."""
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


def flag_ood(pvalues, alpha=0.05):
    return np.asarray(pvalues) < alpha


def _deviance(counts4, probs):
    """GOF deviance: 2*(LL_saturated - LL_model). ~0 => model reproduces the matrix."""
    T = counts4.sum(2, keepdims=True)
    phat = np.clip(counts4 / np.maximum(T, 1), 1e-12, 1.0)
    p = np.clip(probs, 1e-12, 1.0)
    return 2.0 * np.sum(counts4 * (np.log(phat) - np.log(p)), axis=(1, 2))


def ood_deviance(model, counts, trials):
    """
    Fast out-of-family score: the goodness-of-fit deviance of the network's own
    posterior-mean fit. ~0 => the GRT-Gaussian family reproduces the matrix (in
    distribution); large => the matrix has structure (e.g. non-normal within-
    stimulus dependence) that NO parameter setting can produce => genuinely OOD.

    Preferred over the posterior-predictive p-value here, because the full model is
    saturated: it fits almost any single matrix, so only truly out-of-family data
    (off the achievable manifold) produces a large deviance. counts (N,16),
    trials (N,4) -> deviance (N,).
    """
    from .predict import predict_point
    counts = np.asarray(counts)
    pred = predict_point(model, counts, trials).numpy()
    zx, zy, rho = gm.unpack(pred)
    probs = gm.forward_probabilities(zx, zy, rho)          # (N,4,4)
    return _deviance(counts.reshape(-1, 4, 4).astype(float), probs)
