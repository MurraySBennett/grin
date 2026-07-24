"""
evaluate.py — the simulated-data validation gates (truth known).

  recovery_metrics      per-parameter MAE / bias / RMSE / Pearson r
  classification_metrics model-ID accuracy + 12x12 confusion matrix
  sbc_ranks / coverage  uncertainty calibration (simulation-based calibration)
  plot_calibration      SBC rank histograms + a nominal-vs-empirical coverage curve
"""
import numpy as np

try:
    from src import grt_model as gm
except ImportError:
    import grt_model as gm


def recovery_metrics(pred_mean, true):
    pred_mean = np.asarray(pred_mean); true = np.asarray(true)
    err = pred_mean - true
    out = {}
    for j, name in enumerate(gm.PARAM_NAMES):
        t, p = true[:, j], pred_mean[:, j]
        r = np.corrcoef(t, p)[0, 1] if t.std() > 0 else np.nan
        out[name] = {"mae": np.abs(err[:, j]).mean(), "bias": err[:, j].mean(),
                     "rmse": np.sqrt((err[:, j] ** 2).mean()), "r": r}
    out["_aggregate"] = {
        "zscore_mae": np.abs(err[:, 0:8]).mean(),
        "corr_mae": np.abs(err[:, 8:12]).mean(),
        "overall_mae": np.abs(err).mean(),
    }
    return out


def classification_metrics(pred_labels, true_labels):
    names = gm.MODEL_NAMES
    idx = {n: i for i, n in enumerate(names)}
    y_t = np.array([idx[l] for l in true_labels])
    y_p = np.array([idx[l] for l in pred_labels])
    K = len(names)
    cm = np.zeros((K, K), dtype=int)
    for t, p in zip(y_t, y_p):
        cm[t, p] += 1
    acc = (y_t == y_p).mean()
    return {"accuracy": acc, "confusion": cm, "labels": names}


def sbc_ranks(samples, true):
    """samples (S,N,P), true (N,P) -> normalised ranks (N,P) in [0,1] and a KS stat."""
    samples = np.asarray(samples); true = np.asarray(true)
    S = samples.shape[0]
    ranks = (samples < true[None, :, :]).sum(0) / S           # (N,P)
    # KS distance of the pooled ranks from Uniform(0,1), per parameter
    ks = {}
    for j, name in enumerate(gm.PARAM_NAMES):
        r = np.sort(ranks[:, j])
        emp = np.arange(1, len(r) + 1) / len(r)
        ks[name] = np.max(np.abs(emp - r))
    return ranks, ks


def coverage_curve(samples, true, levels=(0.5, 0.7, 0.8, 0.9, 0.95)):
    """Empirical coverage of central credible intervals at each nominal level."""
    samples = np.asarray(samples); true = np.asarray(true)
    emp = {}
    for lvl in levels:
        lo = np.quantile(samples, (1 - lvl) / 2, axis=0)
        hi = np.quantile(samples, (1 + lvl) / 2, axis=0)
        emp[lvl] = ((true >= lo) & (true <= hi)).mean()
    return emp


def plot_calibration(samples, true, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ranks, _ = sbc_ranks(samples, true)
    levels = (0.5, 0.7, 0.8, 0.9, 0.95)
    emp = coverage_curve(samples, true, levels)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].hist(ranks.ravel(), bins=25, color="#4C72B0", density=True)
    ax[0].axhline(1.0, ls="--", c="grey")
    ax[0].set_title("SBC rank histogram\n(flat = calibrated)")
    ax[0].set_xlabel("normalised rank of truth"); ax[0].set_ylabel("density")
    ax[1].plot([0, 1], [0, 1], ls="--", c="grey")
    ax[1].plot(list(levels), [emp[l] for l in levels], "o-", color="#C44E52")
    ax[1].set_title("Coverage calibration")
    ax[1].set_xlabel("nominal credible level"); ax[1].set_ylabel("empirical coverage")
    ax[1].set_xlim(0, 1); ax[1].set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    