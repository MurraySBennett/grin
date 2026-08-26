"""
Disaggregated simulation-based calibration for the counts-only GRIN model.

The manuscript previously reported one pooled interval-coverage number across all
twelve parameters, all twelve model classes and the whole trial-count range. Pooling
lets miscalibrations in opposite directions cancel (src/viz/figures.py:calibration
says as much in its own docstring), so this script reports coverage and SBC rank
uniformity broken down three ways:

    * by parameter family      -- the 8 marginal z-scores vs the 4 correlations
    * by trials-per-stimulus   -- the same 9 bands the speed/accuracy comparison uses
    * by generating model class

Structurally-degenerate correlations (true rho == 0 exactly, under the perceptual
independence classes) are excluded from the rho arm throughout: the SBC rank is not a
calibration statistic when the true value sits on the boundary of the parameter space.
The count of exclusions is reported.

Writes results/validation/calibration_breakdown.json and
results/figures/calibration_breakdown.png.

    python scripts/calibration_breakdown.py [--n-per-class 2000] [--seed 999]
"""
import argparse, json, os
import numpy as np
from scipy.stats import binom, chisquare

from src.config import MODEL_FILE, FIGURES_DIR, Z_MAX, R_MAX
from src.data.generator import GRTDataGenerator
from src.inference.predict import predict_posterior
from src.api import load_model
import src.grt_model as gm

OUT_JSON = os.path.join("results", "validation", "calibration_breakdown.json")
TPS_EDGES = [5, 10, 15, 20, 30, 50, 75, 100, 200, 500]
LEVELS = [0.5, 0.7, 0.8, 0.9, 0.95]
Z_SL, RHO_SL = slice(0, 8), slice(8, 12)


def _coverage(samples, true, keep, level):
    """Empirical coverage of the central `level` interval, over kept entries only."""
    lo = np.quantile(samples, (1 - level) / 2, axis=0)
    hi = np.quantile(samples, (1 + level) / 2, axis=0)
    inside = (true >= lo) & (true <= hi)
    n = int(keep.sum())
    if n == 0:
        return dict(coverage=None, n=0, mc_se=None)
    c = float(inside[keep].mean())
    return dict(coverage=c, n=n, mc_se=float(np.sqrt(c * (1 - c) / n)))


def _rank_uniformity(ranks, keep, n_bins=20):
    """Chi-square test that SBC ranks are uniform, plus a signed shape summary.

    shape > 0  => U-shaped   (too much mass at both ends: posteriors too NARROW)
    shape < 0  => hump-shaped (mass in the middle: posteriors too WIDE)
    """
    r = ranks[keep].ravel()
    if r.size < 100:
        return dict(n=int(r.size), chi2=None, p=None, shape=None)
    counts, _ = np.histogram(r, bins=n_bins, range=(0, 1))
    chi2, p = chisquare(counts)
    expected = r.size / n_bins
    k = max(1, n_bins // 10)
    ends = counts[:k].sum() + counts[-k:].sum()
    mid = counts[n_bins // 2 - k: n_bins // 2 + k].sum()
    return dict(n=int(r.size), chi2=float(chi2), p=float(p),
                shape=float((ends - mid) / expected / (2 * k)))


def main(n_per_class=2000, seed=999, n_samples=800):
    model = load_model(MODEL_FILE)
    gen = GRTDataGenerator(n_per_class=n_per_class, z_max=Z_MAX, r_max=R_MAX, seed=seed)
    X, y, Xt, _, labels = gen.generate_all_model_cms()
    labels = np.asarray(labels)
    post = predict_posterior(model, X, Xt, n_samples=n_samples)
    samples = post["samples"].numpy()                       # (S, N, 12)
    S, N, _ = samples.shape
    ranks = (samples < y[None]).sum(0) / S                  # (N, 12)

    # rho == 0 exactly under the PI classes -> rank is not a calibration statistic
    keep = np.ones_like(y, dtype=bool)
    keep[:, RHO_SL] = y[:, RHO_SL] != 0.0
    n_drop = int((~keep[:, RHO_SL]).sum())

    tps = Xt.mean(1)                                        # mean trials per stimulus
    out = dict(
        meta=dict(n_observers=int(N), n_posterior_samples=int(S), seed=int(seed),
                  n_per_class=int(n_per_class), model_file=str(MODEL_FILE),
                  rho_excluded_structural_zero=n_drop, levels=LEVELS),
        pooled={}, by_family={}, by_trial_band={}, by_model_class={}, rank_tests={},
    )

    # ---- pooled (what the manuscript currently reports) -----------------------
    for lv in LEVELS:
        out["pooled"][str(lv)] = _coverage(samples, y, keep, lv)

    # ---- by parameter family -------------------------------------------------
    for name, sl in (("z", Z_SL), ("rho", RHO_SL)):
        out["by_family"][name] = {
            str(lv): _coverage(samples[:, :, sl], y[:, sl], keep[:, sl], lv) for lv in LEVELS
        }
        out["rank_tests"][name] = _rank_uniformity(ranks[:, sl], keep[:, sl])

    # ---- by trial band x family ---------------------------------------------
    for lo, hi in zip(TPS_EDGES[:-1], TPS_EDGES[1:]):
        m = (tps >= lo) & (tps < hi)
        if m.sum() < 50:
            continue
        band = {}
        for name, sl in (("z", Z_SL), ("rho", RHO_SL)):
            band[name] = _coverage(samples[:, m][:, :, sl], y[m][:, sl], keep[m][:, sl], 0.9)
        band["n_observers"] = int(m.sum())
        out["by_trial_band"][f"{lo}-{hi}"] = band

    # ---- by model class x family --------------------------------------------
    for cls in sorted(set(labels.tolist())):
        m = labels == cls
        if m.sum() < 50:
            continue
        row = {}
        for name, sl in (("z", Z_SL), ("rho", RHO_SL)):
            row[name] = _coverage(samples[:, m][:, :, sl], y[m][:, sl], keep[m][:, sl], 0.9)
        row["n_observers"] = int(m.sum())
        out["by_model_class"][str(cls)] = row

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    from src.viz.calibration_panel import calibration_breakdown as _fig
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig_path = os.path.join(FIGURES_DIR, "calibration_breakdown.png")
    _fig(out, ranks=ranks, keep=keep, path=fig_path)
    print(f"wrote {fig_path}")

    # ---------------------------------------------------------------- report
    p = out["pooled"]["0.9"]
    print(f"\nN = {N} observers, {S} posterior draws, seed {seed}")
    print(f"rho entries excluded as structural zeros: {n_drop}\n")
    print(f"POOLED 90% coverage      {p['coverage']:.4f}  (MC SE {p['mc_se']:.4f}, n={p['n']})")
    for name in ("z", "rho"):
        c = out["by_family"][name]["0.9"]
        t = out["rank_tests"][name]
        direction = ("too NARROW / overconfident" if t["shape"] > 0.15 else
                     "too WIDE / conservative" if t["shape"] < -0.15 else "≈ calibrated")
        print(f"  {name:>4} 90% coverage       {c['coverage']:.4f}  (SE {c['mc_se']:.4f}, "
              f"n={c['n']})   SBC shape {t['shape']:+.2f} -> {direction}")
    print("\ncoverage curve (nominal -> empirical):")
    for lv in LEVELS:
        z, r = out["by_family"]["z"][str(lv)], out["by_family"]["rho"][str(lv)]
        print(f"  {lv:.2f}   z {z['coverage']:.3f}   rho {r['coverage']:.3f}")
    print("\n90% coverage by trials/stimulus:")
    for k, v in out["by_trial_band"].items():
        print(f"  {k:>8}  n={v['n_observers']:>5}   z {v['z']['coverage']:.3f}   "
              f"rho {v['rho']['coverage'] if v['rho']['coverage'] is not None else float('nan'):.3f}")
    print(f"\nwrote {OUT_JSON}")
    return out


# ---------------------------------------------------------------------------
# Construct-probability calibration: does a construct reported at p turn out to be
# true about p of the time? This is distinct from the parameter calibration above and
# from classification accuracy, which says nothing about whether the stated confidence
# is warranted. Reported as a reliability curve plus expected calibration error.
# ---------------------------------------------------------------------------
def construct_calibration(n_per_class=2000, seed=999, n_bins=10):
    import numpy as np
    from src.inference.model_posterior import amortized_compare

    model = load_model(MODEL_FILE)
    gen = GRTDataGenerator(n_per_class=n_per_class, z_max=Z_MAX, r_max=R_MAX, seed=seed)
    X, y, Xt, _, labels = gen.generate_all_model_cms()
    ac = amortized_compare(model, X, Xt, parsimony=0.0)

    specs = [gm.MODEL_SPECS[l] for l in labels]
    truth = {
        "perceptual independence": np.array([s[0] == "pi" for s in specs]),
        "separability on A":       np.array([bool(s[1]) for s in specs]),
        "separability on B":       np.array([bool(s[2]) for s in specs]),
    }
    probs = {"perceptual independence": ac["p_PI"],
             "separability on A": ac["p_sep_A"],
             "separability on B": ac["p_sep_B"]}

    out = {}
    edges = np.linspace(0, 1, n_bins + 1)
    for k in truth:
        p, t = probs[k], truth[k]
        idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
        curve, ece = [], 0.0
        for b in range(n_bins):
            m = idx == b
            if m.sum() < 20:
                curve.append(dict(bin=b, n=int(m.sum()), mean_p=None, freq=None))
                continue
            mp, fr = float(p[m].mean()), float(t[m].mean())
            curve.append(dict(bin=b, n=int(m.sum()), mean_p=mp, freq=fr))
            ece += (m.sum() / len(p)) * abs(mp - fr)
        out[k] = dict(ece=float(ece), accuracy=float(((p > 0.5) == t).mean()),
                      base_rate=float(t.mean()), curve=curve)
        print(f"{k:26s} accuracy {out[k]['accuracy']:.3f}   ECE {ece:.3f}   "
              f"base rate {out[k]['base_rate']:.3f}")

    path = os.path.join("results", "validation", "construct_calibration.json")
    with open(path, "w") as f:
        json.dump(dict(meta=dict(n=int(len(X)), seed=seed, n_bins=n_bins), constructs=out),
                  f, indent=2)
    print(f"wrote {path}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--n-samples", type=int, default=800)
    ap.add_argument("--constructs-only", action="store_true")
    a = ap.parse_args()
    if a.constructs_only:
        construct_calibration(a.n_per_class, a.seed)
    else:
        main(a.n_per_class, a.seed, a.n_samples)
        construct_calibration(a.n_per_class, a.seed)