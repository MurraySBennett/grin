"""
shared_eval.py — framework-agnostic evaluation, so the BayesFlow branch and the
bespoke branch are judged by identical metrics on an identical frozen test set.

Every function takes plain numpy:
    posterior_samples : (N, S, 12)  draws per test matrix, ORIGINAL param space
    true_params       : (N, 12)     ground truth [zx0..3, zy0..3, rho0..3]

Nothing here is BayesFlow-specific. The bespoke predictor just has to return
posterior_samples of the same shape (its predict_posterior already does), and it
gets scored by the same code. That is the whole point of the parallel-directory
design: one ruler, two methods.
"""
import numpy as np

Z_SLICE = slice(0, 8)
RHO_SLICE = slice(8, 12)


def _pearson(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 2:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def recovery_metrics(posterior_samples, true_params):
    """Point-estimate recovery from the posterior mean: Pearson r and MAE, split
    into the z-score group and the correlation group (they behave very differently:
    z is well-identified, rho is information-constrained)."""
    pm = np.nanmean(posterior_samples, axis=1)          # (N,12)
    err = np.abs(pm - true_params)
    nan_frac = float(np.isnan(posterior_samples).mean())
    return {
        "z_r":     _pearson(pm[:, Z_SLICE],   true_params[:, Z_SLICE]),
        "rho_r":   _pearson(pm[:, RHO_SLICE], true_params[:, RHO_SLICE]),
        "z_mae":   float(np.nanmean(err[:, Z_SLICE])),
        "rho_mae": float(np.nanmean(err[:, RHO_SLICE])),
        "overall_mae": float(np.nanmean(err)),
        "nan_fraction": nan_frac,
    }


def sbc_coverage(posterior_samples, true_params, quantiles=(0.5, 0.8, 0.9, 0.95)):
    """Empirical central-credible-interval coverage per nominal level, averaged
    over all 12 parameters. Well-calibrated inference lands each empirical value
    on its nominal level. This is the diagnostic GRIN's whole value rests on, so it
    must be computed identically for both frameworks."""
    N, S, P = posterior_samples.shape
    out = {}
    for q in quantiles:
        lo = np.nanquantile(posterior_samples, (1 - q) / 2, axis=1)   # (N,P)
        hi = np.nanquantile(posterior_samples, 1 - (1 - q) / 2, axis=1)
        inside = (true_params >= lo) & (true_params <= hi)            # (N,P)
        out[q] = float(np.nanmean(inside))
    return out


def sbc_rank_stats(posterior_samples, true_params):
    """SBC rank statistic per parameter: the rank of the true value among the
    posterior draws, normalised to [0,1]. Under calibration these are Uniform(0,1).
    Returns mean and std of ranks per parameter (both should be ~0.5 and ~0.29)."""
    ranks = (posterior_samples < true_params[:, None, :]).mean(axis=1)  # (N,P)
    return {"rank_mean": np.nanmean(ranks, axis=0),
            "rank_std":  np.nanstd(ranks, axis=0)}


def summarise(name, posterior_samples, true_params, seconds_per_call=None):
    """One tidy dict per framework for side-by-side tabulation."""
    rec = recovery_metrics(posterior_samples, true_params)
    cov = sbc_coverage(posterior_samples, true_params)
    row = {"framework": name, **rec,
           "cov50": cov[0.5], "cov90": cov[0.9], "cov95": cov[0.95]}
    if seconds_per_call is not None:
        row["ms_per_call"] = 1e3 * seconds_per_call
    return row


def print_table(rows):
    keys = ["framework", "z_r", "rho_r", "z_mae", "rho_mae",
            "cov90", "cov95", "nan_fraction", "ms_per_call"]
    keys = [k for k in keys if any(k in r for r in rows)]
    widths = {k: max(len(k), *(len(_fmt(r.get(k))) for r in rows)) for k in keys}
    print(" | ".join(k.ljust(widths[k]) for k in keys))
    print("-+-".join("-" * widths[k] for k in keys))
    for r in rows:
        print(" | ".join(_fmt(r.get(k)).ljust(widths[k]) for k in keys))


def _fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 100 else f"{v:.1f}"
    return str(v)
