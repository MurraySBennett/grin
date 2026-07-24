"""
compare.py — head-to-head: our amortized NPE vs a maximum-likelihood baseline,
on shared simulated matrices where the truth is known. Scores BOTH accuracy
(recovery vs truth) and speed (wall-clock).
"""
import time
import numpy as np

try:
    from src import grt_model as gm
except ImportError:
    import grt_model as gm
from .mle import fit_full


def _acc(pred, true):
    err = np.abs(np.asarray(pred) - np.asarray(true))
    return {"zscore_mae": err[:, 0:8].mean(), "corr_mae": err[:, 8:12].mean(),
            "overall_mae": err.mean()}


def head_to_head(npe_predict_fn, X, X_trials, true_params, n_mle=None):
    """
    npe_predict_fn: callable(X, X_trials) -> (pred_mean array (N,12), seconds)
    Fits MLE on the same matrices (or the first n_mle of them) and tabulates.
    """
    true = np.asarray(true_params)
    N = X.shape[0]
    idx = np.arange(N if n_mle is None else min(n_mle, N))

    npe_pred, npe_time = npe_predict_fn(X[idx], X_trials[idx])
    npe_pred = np.asarray(npe_pred)

    t0 = time.time()
    mle_pred = np.array([fit_full(X[i], X_trials[i])["params"] for i in idx])
    mle_time = time.time() - t0

    npe_acc = _acc(npe_pred, true[idx])
    mle_acc = _acc(mle_pred, true[idx])
    return {
        "n_matrices": len(idx),
        "npe": {"accuracy": npe_acc, "seconds": npe_time,
                "ms_per_matrix": 1e3 * npe_time / len(idx)},
        "mle": {"accuracy": mle_acc, "seconds": mle_time,
                "ms_per_matrix": 1e3 * mle_time / len(idx)},
        "speedup": mle_time / max(npe_time, 1e-9),
    }
