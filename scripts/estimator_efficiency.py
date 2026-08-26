"""
Is GRIN at the information limit, or is it under-trained where the prior is thin?

This is the question that decides whether retraining on a reshaped prior would help.
The training prior draws z uniformly, which puts most of its mass near ceiling and
leaves the 60-80% accuracy band -- where identification studies are actually run --
thinly sampled. GRIN nonetheless performs BEST in that band, but that does not settle
the matter: performance there could be good because the design is informative there
while still being worse than it would be with more training coverage. Those two
explanations are confounded in any measurement of error alone.

The Cramer-Rao bound separates them. It says how well ANY unbiased estimator could do
from a given matrix, so the ratio of GRIN's error to that bound is an efficiency:

  * efficiency near 1 across accuracy  -> GRIN is at the information limit; the
    remaining error is the design's, and more training data cannot remove it.
  * efficiency systematically worse where the prior is thin -> GRIN is under-trained
    there, and reshaping the prior would help.

For the unconstrained model each stimulus contributes 3 free response probabilities and
carries exactly 3 parameters (zx_i, zy_i, rho_i), so the per-stimulus Fisher information
is a 3x3 matrix and the bound on rho_i follows from inverting it. Derivatives are taken
by central differences on the exact forward model.

Writes results/validation/estimator_efficiency.json.

    python scripts/estimator_efficiency.py
"""
import json, os
import numpy as np

from src.config import MODEL_FILE, R_MAX
from src.data.generator import GRTDataGenerator
from src.inference.predict import predict_posterior
from src.api import load_model
import src.grt_model as gm

OUT = os.path.join("results", "validation", "estimator_efficiency.json")
ACC_EDGES = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
EPS = 1e-5
FLOOR = 1e-12


def _stim_probs(zx, zy, rho):
    """The 4 response probabilities for ONE stimulus with these parameters."""
    P = gm.forward_probabilities(np.array([zx] * 4), np.array([zy] * 4),
                                 np.array([rho] * 4))
    return np.asarray(P)[0]


def crlb_rho(zx, zy, rho, n):
    """Cramer-Rao lower bound on the variance of rho-hat for one stimulus at n trials.

    Multinomial Fisher information: I = n * sum_r (1/p_r) g_r g_r^T, with
    g_r = d p_r / d theta and theta = (zx, zy, rho). Returns the (rho, rho) element of
    the inverse, i.e. the bound accounting for the other two parameters being estimated
    rather than known.
    """
    base = np.array([zx, zy, rho], float)
    p0 = _stim_probs(*base)
    G = np.zeros((4, 3))
    for j in range(3):
        hi = base.copy(); lo = base.copy()
        hi[j] += EPS; lo[j] -= EPS
        if j == 2:                                  # keep rho strictly inside (-1, 1)
            hi[2] = min(hi[2], 0.999); lo[2] = max(lo[2], -0.999)
        G[:, j] = (_stim_probs(*hi) - _stim_probs(*lo)) / (hi[j] - lo[j])
    p = np.clip(p0, FLOOR, 1.0)
    I = n * (G.T @ (G / p[:, None]))
    try:
        return float(np.linalg.inv(I)[2, 2])
    except np.linalg.LinAlgError:
        return np.nan


def main(n_per_class=400, seed=4242, n_samples=400):
    model = load_model(MODEL_FILE)
    Xs, ys, Xts = [], [], []
    for k, zc in enumerate((0.6, 1.0, 1.6, 2.2, 3.0)):
        g = GRTDataGenerator(n_per_class=n_per_class, z_max=zc, r_max=R_MAX, seed=seed + k)
        X, y, Xt, _, _ = g.generate_all_model_cms()
        Xs.append(X); ys.append(y); Xts.append(Xt)
    X = np.concatenate(Xs); y = np.concatenate(ys); Xt = np.concatenate(Xts)

    # only free correlations: under PI the true rho is pinned at 0 and the bound is
    # not the relevant comparison
    # Restrict to matrices whose four correlations genuinely differ. Under the
    # shared-correlation class the network can pool information across stimuli, so it can
    # beat a PER-STIMULUS bound without that saying anything about efficiency; under PI
    # the true value is pinned at 0 and the bound is not the relevant comparison at all.
    rho = y[:, 8:12]
    free = (np.abs(rho).max(1) > 1e-9) & (np.ptp(rho, axis=1) > 1e-6)
    X, y, Xt = X[free], y[free], Xt[free]
    print(f"{len(X)} matrices with genuinely free (unequal) correlations")

    mean = predict_posterior(model, X, Xt, n_samples=n_samples)["mean"].numpy()

    # per-stimulus true accuracy, error, and bound
    accs, errs, bounds = [], [], []
    for i in range(len(X)):
        for sdx in range(4):
            zx, zy, rho = y[i, sdx], y[i, 4 + sdx], y[i, 8 + sdx]
            P = _stim_probs(zx, zy, rho)
            a = P[0] + P[1] if zx < 0 else P[2] + P[3]
            b = P[0] + P[2] if zy < 0 else P[1] + P[3]
            v = crlb_rho(zx, zy, rho, Xt[i, sdx])
            if not np.isfinite(v) or v <= 0:
                continue
            accs.append(0.5 * (a + b))
            errs.append(mean[i, 8 + sdx] - rho)
            bounds.append(np.sqrt(v))
    accs = np.asarray(accs); errs = np.asarray(errs); bounds = np.asarray(bounds)
    print(f"{len(accs)} stimulus-level comparisons\n")

    out = dict(meta=dict(n_matrices=int(len(X)), n_stimuli=int(len(accs)),
                         seed=seed, acc_edges=ACC_EDGES), by_accuracy=[])
    print(f"{'accuracy band':>14} {'n':>6} {'RMSE':>7} {'bias':>7} {'SD':>7} "
          f"{'CR bound':>9} {'RMSE/CR':>8} {'SD/CR':>7}")
    for lo, hi in zip(ACC_EDGES[:-1], ACC_EDGES[1:]):
        m = (accs >= lo) & (accs < hi)
        if m.sum() < 200:
            continue
        rmse = float(np.sqrt(np.mean(errs[m] ** 2)))
        # RMSE against an UNBIASED bound is not a clean efficiency measure for a
        # prior-regularised estimator: shrinkage buys lower variance at the cost of bias,
        # and RMSE charges for both. Split them. If the excess over the bound is bias,
        # it is the prior working as intended; if it is variance, it is slack that more
        # training data could take up.
        bias = float(np.mean(errs[m]))
        sd = float(np.std(errs[m]))
        # the achievable RMSE if every estimate sat exactly at its own bound
        bound = float(np.sqrt(np.mean(bounds[m] ** 2)))
        row = dict(lo=lo, hi=hi, n=int(m.sum()), rmse=rmse, cr_bound=bound,
                   bias=bias, sd=sd,
                   ratio=rmse / bound if bound > 0 else np.nan,
                   sd_ratio=sd / bound if bound > 0 else np.nan)
        out["by_accuracy"].append(row)
        print(f"{lo:.2f}-{hi:.2f}".rjust(14),
              f"{row['n']:>6} {rmse:>7.3f} {bias:>7.3f} {sd:>7.3f} "
              f"{bound:>9.3f} {row['ratio']:>8.2f} {row['sd_ratio']:>7.2f}")

    rr = [r["ratio"] for r in out["by_accuracy"]]
    print(f"\nratio range {min(rr):.2f} to {max(rr):.2f}")
    print("A ratio flat across accuracy means the estimator is equally close to the")
    print("information limit everywhere, so extra training density would not help.")
    print("A ratio that worsens where the prior is thin means it is under-trained there.")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
