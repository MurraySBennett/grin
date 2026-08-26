"""
Is the maximum-likelihood baseline being compared fairly?

grt_hm_fit() searches from n_reps random starting points per model (default 10) and
keeps the best-likelihood one. The Python maximum-likelihood baseline in compare_to_r.py
calls fit_selected(), which is SINGLE-START. Comparing a 10-restart search against a
1-restart search confounds "different method" with "different search budget", and it
does so in the direction that flatters the amortised estimator, because a single start
on a flat likelihood is exactly where an optimiser does worst.

This script refits the Python baseline at matched restart counts and reports parameter
MAE by trial band, so the manuscript can quote a comparison at equal search budget.

Writes results/validation/restart_parity.json.

    python scripts/restart_parity_check.py [--n-restarts 10] [--subset common]
"""
import argparse, json, os, time
import numpy as np
import pandas as pd

from src.config import MLE_FITS_DIR
from src.inference.mle import fit_selected, fit_selected_multistart

TEST = os.path.join("data", "simulated", "test_set_for_R.csv")
FITS = os.path.join(MLE_FITS_DIR, "baseline_fits.csv")
OUT = os.path.join("results", "validation", "restart_parity.json")
EDGES = [5, 10, 15, 20, 30, 50, 75, 100, 200, 500]
PCOLS = ([f"zx_{i}" for i in range(4)] + [f"zy_{i}" for i in range(4)]
         + [f"rho_{i}" for i in range(4)])


def _band(t):
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        if lo <= t < hi:
            return f"{lo}-{hi}"
    return f"{EDGES[-2]}-{EDGES[-1]}"


def main(n_restarts=10, subset="common"):
    df = pd.read_csv(TEST)
    if subset == "common":
        fits = pd.read_csv(FITS)
        ok = lambda c: fits[c].astype(str).str.upper().isin(["TRUE", "1"]).to_numpy()
        keep = fits.loc[ok("grtools_ok") & ok("mdsdt_ok"), "row_id"]
        df = df[df["row_id"].isin(keep)].reset_index(drop=True)
    print(f"{len(df)} matrices ({subset} subset)")

    cm = df[[f"cm_{s}{r}" for s in range(4) for r in range(4)]].to_numpy(float)
    Xt = df[[f"trials_{i}" for i in range(4)]].to_numpy(float)
    truth = df[PCOLS].to_numpy(float)
    tps = df["tps"].to_numpy(float)
    bands = np.array([_band(t) for t in tps])

    runs = {}
    for label, fn in (("single_start", lambda c, t: fit_selected(c, t, criterion="aic")),
                      (f"restarts_{n_restarts}",
                       lambda c, t: fit_selected_multistart(c, t, criterion="aic",
                                                            n_restarts=n_restarts))):
        est, secs = [], []
        t0 = time.time()
        for i in range(len(df)):
            s0 = time.time()
            try:
                f = fn(cm[i].reshape(4, 4), Xt[i])
                th = np.asarray(f["params"], float)
            except Exception:
                th = np.full(12, np.nan)
            est.append(th); secs.append(1e3 * (time.time() - s0))
        est = np.asarray(est)
        mae_row = np.abs(est - truth).mean(1)
        by_band = {b: float(np.nanmean(mae_row[bands == b])) for b in sorted(set(bands),
                   key=lambda x: EDGES.index(int(x.split("-")[0])))}
        low = np.isin(bands, ["5-10", "10-15", "15-20"])
        high = np.isin(bands, ["75-100", "100-200", "200-500"])
        runs[label] = dict(
            overall_mae=float(np.nanmean(mae_row)),
            sparsest_three=float(np.nanmean(mae_row[low])),
            densest_three=float(np.nanmean(mae_row[high])),
            by_band=by_band,
            median_ms=float(np.median(secs)),
            total_s=time.time() - t0)
        r = runs[label]
        print(f"\n{label}:  overall {r['overall_mae']:.3f}   "
              f"sparsest-3 {r['sparsest_three']:.3f}   densest-3 {r['densest_three']:.3f}   "
              f"{r['median_ms']:.0f} ms/matrix")
        for b, v in r["by_band"].items():
            print(f"    {b:>8}  {v:.3f}")

    a, b = runs["single_start"], runs[f"restarts_{n_restarts}"]
    print(f"\nsparsest-three MAE: {a['sparsest_three']:.3f} (1 start) -> "
          f"{b['sparsest_three']:.3f} ({n_restarts} restarts)")
    print(f"cost: {a['median_ms']:.0f} ms -> {b['median_ms']:.0f} ms per matrix")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(dict(n=len(df), subset=subset, n_restarts=n_restarts, runs=runs), f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-restarts", type=int, default=10)
    ap.add_argument("--subset", default="common", choices=["common", "all"])
    main(**vars(ap.parse_args()))
