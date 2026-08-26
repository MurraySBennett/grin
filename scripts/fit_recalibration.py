"""
Fit an optional post-hoc recalibration of GRIN's posterior scale.

Section "Recovery and calibration" reports that the two parameter families are
miscalibrated in opposite directions: the marginal sensitivities' intervals are wider
than they need to be, and the within-stimulus correlations' are narrower than they
claim. This fits a correction for that and validates it on data not used to fit it.

METHOD. For a calibrated Gaussian posterior the standardised residual

    u_i = (theta_true,i - mu_i) / sigma_i

is standard normal, so SD(u) = 1. If SD(u) = s, the posterior is too narrow by exactly
that factor and multiplying sigma by s restores it. One factor is fitted per parameter
family, because that is where the miscalibration differs; a trial-count-dependent
variant is fitted too and reported alongside, so the added complexity can be judged
against what it buys rather than assumed to be worth it.

WHAT THIS IS NOT. The correction is estimated under the training prior, so it inherits
that prior's coverage. For observers in regions the prior samples thinly it may not
transfer, and it is therefore opt-in rather than applied by default: `infer()` returns
the network's own posterior unless recalibration is explicitly requested. The rescaled
interval is a calibrated interval derived from the posterior, not the posterior itself,
and the two are kept distinguishable in the API and in what is reported.

Correlations that are exactly zero by construction (the perceptual-independence classes)
are excluded from the correlation fit: the residual is not informative about interval
width when the true value sits on the boundary of the parameter space.

Writes results/models/recalibration.json.

    python scripts/fit_recalibration.py
"""
import json, os
import numpy as np

from src.config import MODEL_FILE, Z_MAX, R_MAX
from src.data.generator import GRTDataGenerator
from src.inference.predict import predict_posterior
from src.api import load_model

OUT = os.path.join("results", "models", "recalibration.json")
TPS_EDGES = [0, 20, 50, 200, 10_000]
LEVELS = [0.5, 0.7, 0.8, 0.9, 0.95]
FAMS = {"z": slice(0, 8), "rho": slice(8, 12)}


def _standardised(mean, sd, truth, keep):
    u = (truth - mean) / np.maximum(sd, 1e-9)
    return u[keep]


def _coverage(mean, sd, truth, keep, level, scale=1.0):
    from scipy.stats import norm
    h = norm.ppf(0.5 + level / 2) * sd * scale
    return float((((truth >= mean - h) & (truth <= mean + h))[keep]).mean())


def _draw(n_per_class, seed):
    gen = GRTDataGenerator(n_per_class=n_per_class, z_max=Z_MAX, r_max=R_MAX, seed=seed)
    X, y, Xt, _, _ = gen.generate_all_model_cms()
    return X, y, Xt


def _posterior(model, X, Xt):
    p = predict_posterior(model, X, Xt, n_samples=800)
    s = p["samples"].numpy()
    return p["mean"].numpy(), s.std(0)


def main(n_fit=1500, n_val=1500, seed_fit=1234, seed_val=5678):
    model = load_model(MODEL_FILE)

    print("fitting set ...")
    Xf, yf, Xtf = _draw(n_fit, seed_fit)
    mf, sf = _posterior(model, Xf, Xtf)
    print("validation set ...")
    Xv, yv, Xtv = _draw(n_val, seed_val)
    mv, sv = _posterior(model, Xv, Xtv)

    keep_f = np.ones_like(yf, bool); keep_f[:, 8:12] = yf[:, 8:12] != 0.0
    keep_v = np.ones_like(yv, bool); keep_v[:, 8:12] = yv[:, 8:12] != 0.0

    out = dict(meta=dict(model=str(MODEL_FILE), n_fit=int(len(Xf)), n_val=int(len(Xv)),
                         seed_fit=seed_fit, seed_val=seed_val, levels=LEVELS,
                         method="per-family scale on the posterior SD, s = SD(standardised residual)"),
               global_scale={}, by_trials={}, validation={})

    # ---- global per-family scale ------------------------------------------
    print("\nfamily   scale   (1.00 = already calibrated)")
    for fam, sl in FAMS.items():
        u = _standardised(mf[:, sl], sf[:, sl], yf[:, sl], keep_f[:, sl])
        s = float(np.std(u))
        out["global_scale"][fam] = s
        print(f"  {fam:>4}   {s:5.3f}")

    # ---- trial-count-dependent variant, to see whether it earns its keep ---
    tps_f = Xtf.mean(1)
    for fam, sl in FAMS.items():
        out["by_trials"][fam] = {}
        for lo, hi in zip(TPS_EDGES[:-1], TPS_EDGES[1:]):
            m = (tps_f >= lo) & (tps_f < hi)
            if m.sum() < 100:
                continue
            u = _standardised(mf[m][:, sl], sf[m][:, sl], yf[m][:, sl], keep_f[m][:, sl])
            out["by_trials"][fam][f"{lo}-{hi}"] = float(np.std(u))

    # ---- validate on the held-out set -------------------------------------
    print(f"\ncoverage on {len(Xv)} held-out observers not used to fit the correction")
    print(f"{'level':>7} {'family':>6} {'raw':>7} {'global':>8} {'by-trials':>10}")
    tps_v = Xtv.mean(1)
    for lv in LEVELS:
        for fam, sl in FAMS.items():
            raw = _coverage(mv[:, sl], sv[:, sl], yv[:, sl], keep_v[:, sl], lv)
            glo = _coverage(mv[:, sl], sv[:, sl], yv[:, sl], keep_v[:, sl], lv,
                            out["global_scale"][fam])
            # per-trial-band scaling applied row-wise
            sc = np.ones(len(Xv))
            for k, v in out["by_trials"][fam].items():
                lo, hi = (float(x) for x in k.split("-"))
                sc[(tps_v >= lo) & (tps_v < hi)] = v
            byt = _coverage(mv[:, sl], sv[:, sl] * sc[:, None], yv[:, sl],
                            keep_v[:, sl], lv)
            out["validation"].setdefault(f"{lv}", {})[fam] = dict(
                raw=raw, global_scaled=glo, by_trials_scaled=byt)
            print(f"{lv:>7.2f} {fam:>6} {raw:>7.3f} {glo:>8.3f} {byt:>10.3f}")

    # is the extra complexity worth it?
    err_g = np.mean([abs(out["validation"][f"{l}"][f]["global_scaled"] - l)
                     for l in LEVELS for f in FAMS])
    err_t = np.mean([abs(out["validation"][f"{l}"][f]["by_trials_scaled"] - l)
                     for l in LEVELS for f in FAMS])
    err_r = np.mean([abs(out["validation"][f"{l}"][f]["raw"] - l)
                     for l in LEVELS for f in FAMS])
    out["mean_abs_coverage_error"] = dict(raw=err_r, global_scaled=err_g,
                                          by_trials_scaled=err_t)
    print(f"\nmean |empirical - nominal| across levels and families:")
    print(f"  raw            {err_r:.4f}")
    print(f"  global scale   {err_g:.4f}")
    print(f"  by trial band  {err_t:.4f}")
    out["recommended"] = "global" if err_g <= err_t + 0.002 else "by_trials"
    print(f"\nrecommended: {out['recommended']} "
          f"({'the trial-band variant does not earn its complexity' if out['recommended']=='global' else 'the trial-band variant is worth it'})")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT}")
    return out


if __name__ == "__main__":
    main()
