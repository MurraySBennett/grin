"""Exact multinomial likelihood and multistart full-model MLE for 3x3 GRIN.

The base 3x3 models have an evaluable likelihood; simulation-based inference is
used for amortization, not because the likelihood is unavailable.  These fits
provide deterministic reference estimates for validating neural posteriors.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import norm

from src import grt_model_3x3 as unit_model
from src import grt_model_3x3_hetero as free_model


def multinomial_log_likelihood(counts, probabilities, include_constant=False):
    counts = np.asarray(counts, dtype=float).reshape(9, 9)
    probabilities = np.asarray(probabilities, dtype=float).reshape(9, 9)
    if np.any(counts < 0) or not np.allclose(counts, np.round(counts)):
        raise ValueError("counts must be nonnegative integers")
    if np.any(probabilities < 0) or not np.allclose(probabilities.sum(axis=1), 1.0):
        raise ValueError("each probability row must be nonnegative and sum to one")
    value = np.sum(counts * np.log(np.clip(probabilities, 1e-300, 1.0)))
    if include_constant:
        totals = counts.sum(axis=1)
        value += np.sum(gammaln(totals + 1) - gammaln(counts + 1).sum(axis=1))
    return float(value)


def _response_marginals(counts):
    table = np.asarray(counts, dtype=float).reshape(9, 3, 3)
    # Jeffreys-like smoothing only constructs a finite optimizer starting point;
    # it is not added to the likelihood.
    table = table + 0.5
    probs = table / table.sum(axis=(1, 2), keepdims=True)
    return probs.sum(axis=2), probs.sum(axis=1)


def initial_unit(counts):
    px, py = _response_marginals(counts)
    zx1 = norm.ppf(np.clip(px[:, 0], 1e-6, 1 - 1e-6))
    zy1 = norm.ppf(np.clip(py[:, 0], 1e-6, 1 - 1e-6))
    mux, muy = -zx1, -zy1
    zx2 = norm.ppf(np.clip(px[:, :2].sum(axis=1), 1e-6, 1 - 1e-6))
    zy2 = norm.ppf(np.clip(py[:, :2].sum(axis=1), 1e-6, 1 - 1e-6))
    bx = np.clip(np.median(zx2 + mux), 0.05, 10.0)
    by = np.clip(np.median(zy2 + muy), 0.05, 10.0)
    return np.concatenate([mux, muy, np.zeros(9), np.log([bx, by])])


def initial_free(counts):
    px, py = _response_marginals(counts)

    def solve(marginal):
        z1 = norm.ppf(np.clip(marginal[:, 0], 1e-6, 1 - 1e-6))
        z2 = norm.ppf(np.clip(marginal[:, :2].sum(axis=1), 1e-6, 1 - 1e-6))
        sd = np.clip(1.0 / np.maximum(z2 - z1, 0.05), 0.05, 10.0)
        mu = -z1 * sd
        return mu, sd

    mux, sdx = solve(px)
    muy, sdy = solve(py)
    return np.concatenate([mux, muy, np.log(sdx), np.log(sdy), np.zeros(9)])


def _unit_from_unconstrained(values):
    values = np.asarray(values)
    return values[:9], values[9:18], np.tanh(values[18:27]), *np.exp(values[27:29])


def _free_from_unconstrained(values):
    values = np.asarray(values)
    return (values[:9], values[9:18], np.exp(values[18:27]),
            np.exp(values[27:36]), np.tanh(values[36:45]))


def _objective(values, counts, variance_model):
    if variance_model == "unit":
        theta = _unit_from_unconstrained(values)
        probabilities = unit_model.forward_probabilities(*theta)
    else:
        theta = _free_from_unconstrained(values)
        probabilities = free_model.forward_probabilities(*theta)
    return -multinomial_log_likelihood(counts, probabilities)


def fit_full(counts, variance_model="unit", n_restarts=8, jitter=0.35, seed=0,
             maxiter=1500):
    """Fit the 29- or 45-parameter unconstrained structural class by exact MLE."""
    counts = np.asarray(counts, dtype=float).reshape(9, 9)
    if variance_model == "unit":
        start = initial_unit(counts)
        bounds = [(-8, 8)] * 18 + [(-3.8, 3.8)] * 9 + [(np.log(.05), np.log(10))] * 2
        decode, model = _unit_from_unconstrained, unit_model
    elif variance_model == "free":
        start = initial_free(counts)
        bounds = ([(-8, 8)] * 18 + [(np.log(.05), np.log(10))] * 18
                  + [(-3.8, 3.8)] * 9)
        decode, model = _free_from_unconstrained, free_model
    else:
        raise ValueError("variance_model must be 'unit' or 'free'")

    rng = np.random.default_rng(seed)
    results = []
    for restart in range(max(1, int(n_restarts))):
        x0 = start if restart == 0 else np.clip(
            start + rng.normal(0, jitter, start.shape),
            [item[0] for item in bounds], [item[1] for item in bounds],
        )
        result = minimize(_objective, x0, args=(counts, variance_model), method="L-BFGS-B",
                          bounds=bounds, options={"maxiter": int(maxiter), "ftol": 1e-10})
        results.append(result)
    best = min(results, key=lambda result: result.fun)
    theta = decode(best.x)
    packed = model.pack(*theta)
    n_free = model.n_free_params("ds")
    n_trials = int(counts.sum())
    log_likelihood = -float(best.fun)
    return {
        "variance_model": variance_model,
        "params": packed,
        "theta": theta,
        "log_likelihood": log_likelihood,
        "aic": 2 * n_free - 2 * log_likelihood,
        "bic": np.log(n_trials) * n_free - 2 * log_likelihood,
        "success": bool(best.success),
        "message": str(best.message),
        "n_iterations": int(best.nit),
        "n_restarts": len(results),
        "restart_objectives": np.array([result.fun for result in results]),
    }
