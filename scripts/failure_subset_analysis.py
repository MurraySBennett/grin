"""
How good is GRIN's answer on the matrices where the baselines give up?

"GRIN returns an estimate on every matrix" is only a virtue if those estimates are
worth having. The speed/accuracy comparison scores every method on the common
subset where all four converged -- necessarily the easier matrices. This script
scores GRIN on the complement: the matrices where grtools reports a non-zero optim()
convergence code, and those where mdsdt raises an R-level error.

Ground truth is known (the test set is simulated), so this is a direct check rather
than an agreement check. The comparison that matters is against GRIN's own error in
the same trial-count band, since the failure matrices are concentrated at the sparse
end and would look worse than the pooled average for that reason alone.

Writes results/validation/failure_subset.json.

    python scripts/failure_subset_analysis.py
"""
import json, os
import numpy as np
import pandas as pd

from src.config import MODEL_FILE, MLE_FITS_DIR
from src.api import load_model
from src.inference.predict import predict_posterior
from src.inference.model_posterior import amortized_compare

TEST = os.path.join("data", "simulated", "test_set_for_R.csv")
FITS = os.path.join(MLE_FITS_DIR, "baseline_fits.csv")
OUT = os.path.join("results", "validation", "failure_subset.json")
TPS_EDGES = [5, 10, 15, 20, 30, 50, 75, 100, 200, 500]
PARAMS = [f"z{d}_{i}" for d in "xy" for i in range(4)] + [f"rho_{i}" for i in range(4)]


def _band(tps):
    for lo, hi in zip(TPS_EDGES[:-1], TPS_EDGES[1:]):
        if lo <= tps < hi:
            return f"{lo}-{hi}"
    return f"{TPS_EDGES[-2]}-{TPS_EDGES[-1]}"


def _summarise(abs_error, mask, label, tps, bands):
    """Family MAE on `mask`, plus trial-band-matched reference MAE."""
    if mask.sum() == 0:
        return None
    sub = abs_error[mask]
    # band-matched reference: GRIN's own MAE on NON-masked matrices, reweighted to the
    # band composition of the masked set, so sparsity alone cannot explain a gap.
    comp = pd.Series([_band(t) for t in tps[mask]]).value_counts(normalize=True)
    ref_parts = {"z": [], "rho": []}
    ref_w = []
    for b, w in comp.items():
        other = (~mask) & np.array([_band(t) == b for t in tps])
        if other.sum() >= 5:
            ref_parts["z"].append(abs_error[other, :8].mean())
            ref_parts["rho"].append(abs_error[other, 8:].mean())
            ref_w.append(w)
    ref = ({fam: float(np.average(values, weights=ref_w))
            for fam, values in ref_parts.items()} if ref_w else None)
    return dict(label=label, n=int(mask.sum()),
                mae_z=float(sub[:, :8].mean()), mae_rho=float(sub[:, 8:].mean()),
                mae_band_matched_reference=ref,
                median_tps=float(np.median(tps[mask])),
                band_composition={str(k): float(v) for k, v in comp.items()})


def main():
    df = pd.read_csv(TEST)
    fits = pd.read_csv(FITS)
    d = df.merge(fits, on="row_id", how="inner")
    print(f"{len(d)} matrices with both ground truth and baseline fit records")

    cm = d[[f"cm_{s}{r}" for s in range(4) for r in range(4)]].to_numpy(float)
    Xt = d[[f"trials_{i}" for i in range(4)]].to_numpy(float)
    truth = d[[f"z{d_}_{i}" for d_ in "xy" for i in range(4)] +
              [f"rho_{i}" for i in range(4)]].to_numpy(float)
    tps = d["tps"].to_numpy(float)

    model = load_model(MODEL_FILE)
    post = predict_posterior(model, cm, Xt, n_samples=800)
    mean = post["mean"].numpy()
    samples = post["samples"].numpy()
    abs_error = np.abs(mean - truth)

    # 90% interval coverage per matrix, pooled over the 12 params
    lo = np.quantile(samples, 0.05, axis=0)
    hi = np.quantile(samples, 0.95, axis=0)
    covered = (truth >= lo) & (truth <= hi)

    ok = lambda c: d[c].astype(str).str.upper().isin(["TRUE", "1"]).to_numpy()
    grt_ok, mds_ok = ok("grtools_ok"), ok("mdsdt_ok")

    masks = {
        "grtools_failed": ~grt_ok,
        "mdsdt_failed": ~mds_ok,
        "either_failed": ~(grt_ok & mds_ok),
        "both_failed": ~grt_ok & ~mds_ok,
        "all_converged": grt_ok & mds_ok,
    }

    out = dict(meta=dict(n=len(d), source_test=TEST, source_fits=FITS,
                         grtools_failure_rate=float((~grt_ok).mean()),
                         mdsdt_failure_rate=float((~mds_ok).mean())),
               subsets={}, coverage={})
    for k, m in masks.items():
        s = _summarise(abs_error, m, k, tps, TPS_EDGES)
        if s:
            s["coverage_90"] = {
                "z": float(covered[m, :8].mean()),
                "rho": float(covered[m, 8:].mean()),
            }
            out["subsets"][k] = s

    print()
    for k, s in out["subsets"].items():
        ref = s["mae_band_matched_reference"]
        refs = (f"z {ref['z']:.3f}, rho {ref['rho']:.3f}" if ref is not None
                else "n/a")
        print(f"{k:>16}  n={s['n']:>4}  median tps={s['median_tps']:>6.1f}  "
              f"GRIN MAE z {s['mae_z']:.3f}, rho {s['mae_rho']:.3f}  "
              f"(band-matched ref {refs})  90% coverage "
              f"z {s['coverage_90']['z']:.3f}, rho {s['coverage_90']['rho']:.3f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT}")
    return out


if __name__ == "__main__":
    main()
