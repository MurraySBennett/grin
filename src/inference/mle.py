"""
mle.py — maximum-likelihood GRT fitter on our exact forward model.

This is the same estimator mdsdt implements (identified z-score + correlation
coordinates, multinomial likelihood), written in Python against grt_model so it is
runnable and testable here. It serves two roles:
  1. the head-to-head MLE baseline for the evaluation gates (accuracy AND speed);
  2. proper AIC/BIC model selection (fit each class, select), which replaces the
     Phase-2 heuristic in model_selection.py.

Fitting is done in an unconstrained "train space" (Fisher-z for correlations) and
warm-started from the confusion-matrix marginals (the mdsdt initialisation), so it
converges quickly and reliably.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

try:
    from src import grt_model as gm
except ImportError:
    import grt_model as gm


def _expand(model_name, free):
    """Expand a class's free parameter vector to full (zx, zy, rho)."""
    corr, ps_x, ps_y = gm.MODEL_SPECS[model_name]
    i = 0
    if ps_x:
        zx = np.array([free[i], free[i], free[i + 1], free[i + 1]]); i += 2
    else:
        zx = np.asarray(free[i:i + 4]); i += 4
    if ps_y:
        zy = np.array([free[i], free[i + 1], free[i], free[i + 1]]); i += 2
    else:
        zy = np.asarray(free[i:i + 4]); i += 4
    if corr == "pi":
        rho = np.zeros(4)
    elif corr == "rho1":
        rho = np.tanh(free[i]) * np.ones(4); i += 1
    else:
        rho = np.tanh(np.asarray(free[i:i + 4])); i += 4
    return zx, zy, rho


def _nll(free, model_name, counts4x4):
    zx, zy, rho = _expand(model_name, free)
    probs = gm.forward_probabilities(zx[None], zy[None], rho[None])[0]
    probs = np.clip(probs, 1e-12, 1.0)
    return -np.sum(counts4x4 * np.log(probs))


def _init_from_data(model_name, counts4x4, trials):
    """Warm start: z-scores from marginal response proportions (mdsdt-style)."""
    T = np.maximum(trials, 1)
    p_x1 = np.clip((counts4x4[:, 0] + counts4x4[:, 1]) / T, 1e-3, 1 - 1e-3)  # respond a1 on x
    p_y1 = np.clip((counts4x4[:, 0] + counts4x4[:, 2]) / T, 1e-3, 1 - 1e-3)  # respond b1 on y
    zx = -norm.ppf(p_x1)
    zy = -norm.ppf(p_y1)
    corr, ps_x, ps_y = gm.MODEL_SPECS[model_name]
    init = []
    init += ([0.5 * (zx[0] + zx[1]), 0.5 * (zx[2] + zx[3])] if ps_x else list(zx))
    init += ([0.5 * (zy[0] + zy[2]), 0.5 * (zy[1] + zy[3])] if ps_y else list(zy))
    init += ([] if corr == "pi" else ([0.0] if corr == "rho1" else [0.0, 0.0, 0.0, 0.0]))
    return np.array(init, dtype=float)


def fit_class(counts, trials, model_name, init=None):
    counts4x4 = np.asarray(counts, float).reshape(4, 4)
    trials = np.asarray(trials, float).reshape(4)
    if init is None:
        init = _init_from_data(model_name, counts4x4, trials)
    res = minimize(_nll, init, args=(model_name, counts4x4), method="L-BFGS-B")
    ll = -res.fun
    k = gm.n_free_params(model_name)
    n = counts4x4.sum()
    zx, zy, rho = _expand(model_name, res.x)
    return {"model": model_name, "params": gm.pack(zx, zy, rho), "nll": res.fun,
            "loglik": ll, "k": k, "aic": 2 * k - 2 * ll, "bic": k * np.log(n) - 2 * ll}


def fit_full(counts, trials, init=None):
    """Fit the saturated (ds) model = the 12 identified parameters."""
    return fit_class(counts, trials, "ds", init=init)


def fit_and_select(counts, trials, criterion="bic"):
    """Fit every class, select by AIC/BIC. Returns (best_fit, all_fits)."""
    fits = [fit_class(counts, trials, m) for m in gm.MODEL_NAMES]
    best = min(fits, key=lambda f: f[criterion])
    return best, fits


# ---------------------------------------------------------------------------
# Multi-start fitting.
#
# The single-start fits above warm-start from the confusion-matrix marginals, which is
# what mdsdt does and is usually fine. But "usually" is doing work there: on a sizeable
# fraction of matrices L-BFGS-B walks into a local optimum or a bound and stops, and a fit
# that stops on a bound still returns a finite number, so nothing downstream notices. That
# makes the MLE baseline look worse than maximum likelihood actually is, which is exactly
# the wrong direction for an error when MLE is the thing you are claiming to beat.
#
# The standard remedy is multi-start: refit from several jittered starting points and keep
# the best likelihood. It is what a careful practitioner does, and it costs n_restarts
# times as long — a real cost that belongs in the speed comparison rather than being
# quietly excluded from it. Hence both variants are exposed: fit a single start for the
# cheap baseline, fit multi-start for the fair one, and report the timing of each.
# ---------------------------------------------------------------------------

def fit_class_multistart(counts, trials, model_name, n_restarts=20, jitter=0.75, seed=0):
    """Best-of-n_restarts fit for one model class, selected on likelihood.

    Start 1 is the data-driven warm start (identical to fit_class); the remaining
    n_restarts - 1 are Gaussian perturbations of it in the unconstrained train space.
    Jitter is applied in train space, so it perturbs Fisher-z correlations on the same
    footing as z-scores. `seed` is per-call, so results are reproducible.
    """
    counts4x4 = np.asarray(counts, float).reshape(4, 4)
    trials_arr = np.asarray(trials, float).reshape(4)
    base = _init_from_data(model_name, counts4x4, trials_arr)
    rng = np.random.default_rng(seed)

    best = fit_class(counts, trials, model_name, init=base)
    n_improved = 0
    for _ in range(max(int(n_restarts) - 1, 0)):
        cand = fit_class(counts, trials, model_name,
                         init=base + rng.normal(0.0, jitter, size=base.shape))
        if cand["nll"] < best["nll"]:
            best = cand
            n_improved += 1
    best = dict(best)
    best["restarts"] = int(n_restarts)
    best["n_improved"] = n_improved      # >0 means the single start was NOT the best fit
    return best


def fit_full_multistart(counts, trials, n_restarts=20, jitter=0.75, seed=0):
    """Saturated (ds) fit, best of n_restarts jittered starts."""
    return fit_class_multistart(counts, trials, "ds", n_restarts=n_restarts,
                                jitter=jitter, seed=seed)


def fit_and_select_multistart(counts, trials, criterion="bic", n_restarts=20, jitter=0.75,
                              seed=0):
    """Fit EVERY class multi-start, then select by AIC/BIC. Returns (best_fit, all_fits).

    This is the expensive, defensible workflow: 12 classes x n_restarts optimisations per
    matrix. Its cost is the honest cost of the comparison a careful analyst would run.
    """
    fits = [fit_class_multistart(counts, trials, m, n_restarts=n_restarts, jitter=jitter,
                                 seed=seed + i)
            for i, m in enumerate(gm.MODEL_NAMES)]
    best = min(fits, key=lambda f: f[criterion])
    return best, fits


def fit_selected_multistart(counts, trials, criterion="bic", n_restarts=20, jitter=0.75,
                            seed=0):
    """As fit_selected, but every class is fitted multi-start."""
    best, _ = fit_and_select_multistart(counts, trials, criterion=criterion,
                                        n_restarts=n_restarts, jitter=jitter, seed=seed)
    return best


def fit_selected(counts, trials, criterion="bic"):
    """The realistic MLE workflow: fit every class, keep the AIC/BIC winner.

    `fit_full` fits only the saturated `ds` model -- fast, but not what a practitioner
    reports. This returns the selected fit in exactly `fit_class`'s dict shape, so
    `["params"]` is the same packed 12-vector `fit_full` yields and the two are directly
    comparable in a recovery figure. `["model"]` carries the selected class name.
    """
    best, _all = fit_and_select(counts, trials, criterion=criterion)
    return best


# ---------------------------------------------------------------------------
# Penalised likelihood.
#
# On a 4x4 identification matrix, cells are routinely EMPTY: a sensitivity of 3 implies a
# 99.87% response rate, so at 200 trials the complementary cell is empty about three times
# in four. An empty cell means the likelihood for that parameter increases without bound as
# it is pushed outward -- the same separation problem that makes logistic regression blow
# up when a predictor perfectly separates the outcome. There is no interior maximum, so
# multi-start does not help: every start walks out along the same flat asymptote.
#
# The standard remedy is to add a pseudo-count to every cell, which is equivalent to a
# Dirichlet prior on the response probabilities and gives a proper, finite maximum.
# `pseudo=0.5` is the Jeffreys prior and is the usual default; `pseudo=1.0` is Laplace.
#
# NOTE ON SMALL PSEUDO-COUNTS: a value like 0.001 is too weak to do the job. The penalty it
# contributes is pseudo * log(p); at z = 5 that is about 0.001 * -15 = -0.015 of
# log-likelihood, which will not stop an optimiser that gains more than that by continuing
# outward. If you want the estimate bounded, the pseudo-count has to be an appreciable
# fraction of a trial.
# ---------------------------------------------------------------------------

def _nll_penalised(free, model_name, counts4x4, pseudo):
    zx, zy, rho = _expand(model_name, free)
    probs = gm.forward_probabilities(zx[None], zy[None], rho[None])[0]
    probs = np.clip(probs, 1e-12, 1.0)
    return -np.sum((counts4x4 + pseudo) * np.log(probs))


def fit_class_penalised(counts, trials, model_name, pseudo=0.5, init=None):
    """Penalised ML for one class: every cell gets `pseudo` added before fitting.

    Returns the same dict shape as fit_class. `nll`, `aic` and `bic` are computed on the
    UNPENALISED likelihood at the penalised optimum, so information criteria remain
    comparable with the unpenalised fits.
    """
    counts4x4 = np.asarray(counts, float).reshape(4, 4)
    trials_arr = np.asarray(trials, float).reshape(4)
    if init is None:
        init = _init_from_data(model_name, counts4x4, trials_arr)
    res = minimize(_nll_penalised, init, args=(model_name, counts4x4, pseudo),
                   method="L-BFGS-B")
    zx, zy, rho = _expand(model_name, res.x)
    nll = _nll(res.x, model_name, counts4x4)      # unpenalised, for AIC/BIC comparability
    ll = -nll
    k = gm.n_free_params(model_name)
    n = counts4x4.sum()
    return {"model": model_name, "params": gm.pack(zx, zy, rho), "nll": nll, "loglik": ll,
            "k": k, "aic": 2 * k - 2 * ll, "bic": k * np.log(n) - 2 * ll,
            "pseudo": pseudo, "penalised_nll": res.fun}


def fit_full_penalised(counts, trials, pseudo=0.5, init=None):
    """Saturated (ds) penalised fit -- the separation-safe MLE baseline."""
    return fit_class_penalised(counts, trials, "ds", pseudo=pseudo, init=init)


def fit_and_select_penalised(counts, trials, criterion="bic", pseudo=0.5):
    fits = [fit_class_penalised(counts, trials, m, pseudo=pseudo) for m in gm.MODEL_NAMES]
    best = min(fits, key=lambda f: f[criterion])
    return best, fits


def fit_selected_penalised(counts, trials, criterion="bic", pseudo=0.5):
    best, _ = fit_and_select_penalised(counts, trials, criterion=criterion, pseudo=pseudo)
    return best
