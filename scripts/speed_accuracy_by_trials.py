"""
speed_accuracy_by_trials.py — speed and accuracy as a function of trial count, WITH
uncertainty, replacing the old aggregate-only Table 3 (speed) and Table 4 (MAE by
trial band).

Why this exists rather than reusing compare_to_r.py's numbers directly: that script's
`ms[...]` values are each a single aggregate (total elapsed / N), which is fine for a
headline "X ms per matrix" claim but cannot support error bars and cannot show whether
speed depends on trial count the way accuracy does. This script times GRIN and the
Python-MLE baseline PER MATRIX instead of in one big block, and pulls mdsdt/grtools'
already-per-matrix `_secs` columns out of baseline_fits.csv rather than only their mean.

    python scripts/speed_accuracy_by_trials.py [--refit]

Writes results/speed_accuracy_by_trials.json and results/figures/speed_accuracy.png.
Accuracy is reported separately for the eight marginal sensitivities and four
correlations. The JSON also contains an all-matrix GRIN--Python-MLE comparison;
the four-method curves remain a secondary complete-case analysis because the R
packages do not return scorable estimates for every matrix.
The Python-MLE per-matrix timing loop is the slow step here (~12 model-class fits per
matrix via L-BFGS-B) -- cached in results/mle_fits/speed_accuracy_mle_cache.csv after
the first run, same idiom as compare_to_r.py's MLE_CACHE.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import SIMULATED_DATA_DIR, MLE_FITS_DIR, FIGURES_DIR, RESULTS_DIR
from src.api import load_model
from src.inference.predict import predict_point
from src.inference.mle import fit_selected
from scripts.export_for_r import TRIAL_BIN_LABELS, N_TRIAL_BINS
import src.grt_model as gm

CSV = os.path.join(SIMULATED_DATA_DIR, "test_set_for_R.csv")
RFITS = os.path.join(MLE_FITS_DIR, "baseline_fits.csv")
MLE_CACHE = os.path.join(MLE_FITS_DIR, "speed_accuracy_mle_cache.csv")
OUT_JSON = os.path.join(RESULTS_DIR, "speed_accuracy_by_trials.json")
FAMILIES = {"z": slice(0, 8), "rho": slice(8, 12)}


def _error_summary(estimates, truth, mask):
    """Family-specific MAE, with pooled MAE retained only as a diagnostic."""
    ae = np.abs(estimates[mask] - truth[mask])
    return {
        "n": int(mask.sum()),
        "mae_z": float(ae[:, FAMILIES["z"]].mean()),
        "mae_rho": float(ae[:, FAMILIES["rho"]].mean()),
        "mae_pooled_diagnostic": float(ae.mean()),
    }


def main():
    refit = "--refit" in sys.argv
    df = pd.read_csv(CSV)
    cm_cols = [f"cm_{s}{r}" for s in range(4) for r in range(4)]
    X = df[cm_cols].to_numpy()
    Xt = df[[f"trials_{s}" for s in range(4)]].to_numpy()
    truth = df[gm.PARAM_NAMES].to_numpy(dtype=float)
    trial_bin = df["trial_bin"].to_numpy()
    N = len(df)

    model = load_model()

    # ---- GRIN: per-matrix single-call timing (fast; always fresh) --------------
    grin_ms = np.empty(N)
    grin_params = np.empty((N, 12))
    for i in range(N):
        t0 = time.perf_counter()
        p = predict_point(model, X[i:i + 1], Xt[i:i + 1]).numpy()
        grin_ms[i] = 1e3 * (time.perf_counter() - t0)
        grin_params[i] = p[0]

    # ---- Python-MLE: per-matrix timing (slow; cached after first run) ----------
    cached = (not refit) and os.path.exists(MLE_CACHE)
    if cached:
        cdf = pd.read_csv(MLE_CACHE)
        cached = len(cdf) == N
    if cached:
        mle_ms = cdf["ms"].to_numpy(float)
        mle_params = cdf[[f"p{k}" for k in range(12)]].to_numpy(float)
        print(f"(Python-MLE per-matrix timing loaded from cache {MLE_CACHE})")
    else:
        mle_ms = np.empty(N)
        mle_params = np.empty((N, 12))
        for i in range(N):
            t0 = time.perf_counter()
            try:
                mle_params[i] = fit_selected(X[i], Xt[i])["params"]
            except Exception:
                mle_params[i] = np.nan
            mle_ms[i] = 1e3 * (time.perf_counter() - t0)
            if i % 50 == 0:
                print(f"  Python-MLE {i}/{N}", flush=True)
        cdf = pd.DataFrame(mle_params, columns=[f"p{k}" for k in range(12)])
        cdf["ms"] = mle_ms
        os.makedirs(os.path.dirname(MLE_CACHE), exist_ok=True)
        cdf.to_csv(MLE_CACHE, index=False)
        print(f"wrote Python-MLE per-matrix cache -> {MLE_CACHE}")

    # ---- mdsdt / grtools: already per-matrix in baseline_fits.csv --------------
    r = pd.read_csv(RFITS).set_index("row_id")
    j = df.set_index("row_id").join(r).reset_index()

    def _params(prefix):
        cols = [f"{prefix}_{n}" for n in gm.PARAM_NAMES]
        return j[cols].to_numpy(dtype=float) if all(c in j.columns for c in cols) else None

    methods_ms = {"GRIN": grin_ms, "Python-MLE": mle_ms,
                  "mdsdt": j["mdsdt_secs"].to_numpy(float) * 1e3,
                  "grtools": j["grtools_secs"].to_numpy(float) * 1e3}
    methods_params = {"GRIN": grin_params, "Python-MLE": mle_params,
                      "mdsdt": _params("mdsdt"), "grtools": _params("grtools")}
    methods_ok = {
        "GRIN": np.ones(N, bool),
        "Python-MLE": np.isfinite(mle_params).all(1),
        "mdsdt": j["mdsdt_ok"].fillna(False).to_numpy(bool) & np.isfinite(methods_params["mdsdt"]).all(1),
        "grtools": j["grtools_ok"].fillna(False).to_numpy(bool) & np.isfinite(methods_params["grtools"]).all(1),
    }
    common = np.ones(N, bool)
    for m in methods_ok:
        common &= methods_ok[m]

    # ---- bin everything by trial count: mean + percentile-bootstrap 95% CI -----
    # Same _boot_ci convention as fig:recovery (src/viz/recovery.py), not +-1 SD: SD on
    # a log-scale, right-skewed latency distribution produced nonsensical bands (the
    # first version of this figure had GRIN's speed band reaching down to 1e-4ms after
    # clipping a negative lower bound -- an artefact of the summary statistic, not a
    # real property of the timing data).
    from src.viz.recovery import _boot_ci
    out = {
        "meta": {
            "n_total": int(N),
            "n_complete_case": int(common.sum()),
            "parameter_families": {"z": [0, 8], "rho": [8, 12]},
            "accuracy_note": (
                "Four-method curves use identical complete cases. The GRIN--Python-MLE "
                "summary uses every matrix scorable by that pair. Sensitivity and "
                "correlation MAE are primary; pooled MAE is diagnostic only."
            ),
        },
        "bands": TRIAL_BIN_LABELS,
        "n_common": int(common.sum()),  # compatibility with earlier readers
        "methods": {},
        "pairwise_all_matrices": {},
        "complete_case_summary": {},
    }
    for m in methods_ms:
        speed_mean, speed_lo, speed_hi = [], [], []
        family = {
            fam: {"mae_mean": [], "mae_lo": [], "mae_hi": []}
            for fam in FAMILIES
        }
        # Legacy pooled values are kept so old notebooks can still read the artifact;
        # they are no longer plotted or used as the primary manuscript metric.
        mae_mean, mae_lo, mae_hi, n_band = [], [], [], []
        for b in range(N_TRIAL_BINS):
            mask_speed = trial_bin == b   # speed: every matrix this method actually ran on
            mask_acc = common & (trial_bin == b)   # accuracy: paired complete-case, as before
            s = methods_ms[m][mask_speed]
            s = s[np.isfinite(s)]
            mu, lo, hi = _boot_ci(s) if s.size else (None, None, None)
            speed_mean.append(mu); speed_lo.append(lo); speed_hi.append(hi)
            e = (np.abs(methods_params[m][mask_acc] - truth[mask_acc])
                 if mask_acc.any() else np.empty((0, 12)))
            pooled = e.mean(1) if e.size else np.array([])
            mu, lo, hi = _boot_ci(pooled) if pooled.size else (None, None, None)
            mae_mean.append(mu); mae_lo.append(lo); mae_hi.append(hi)
            for fam, sl in FAMILIES.items():
                ef = e[:, sl].mean(1) if e.size else np.array([])
                fmu, flo, fhi = _boot_ci(ef) if ef.size else (None, None, None)
                family[fam]["mae_mean"].append(fmu)
                family[fam]["mae_lo"].append(flo)
                family[fam]["mae_hi"].append(fhi)
            n_band.append(int(mask_acc.sum()))
        out["methods"][m] = dict(speed_ms_mean=speed_mean, speed_ms_lo=speed_lo, speed_ms_hi=speed_hi,
                                 mae_mean=mae_mean, mae_lo=mae_lo, mae_hi=mae_hi,
                                 family_mae=family, n_common_band=n_band)
        out["complete_case_summary"][m] = _error_summary(
            methods_params[m], truth, common)

    pair_ok = methods_ok["GRIN"] & methods_ok["Python-MLE"]
    sparse = pair_ok & np.isin(trial_bin, [0, 1, 2])
    dense = pair_ok & np.isin(trial_bin, [6, 7, 8])
    for m in ("GRIN", "Python-MLE"):
        out["pairwise_all_matrices"][m] = {
            "all": _error_summary(methods_params[m], truth, pair_ok),
            "sparse_5_20": _error_summary(methods_params[m], truth, sparse),
            "dense_75_500": _error_summary(methods_params[m], truth, dense),
        }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {OUT_JSON}")

    from src.viz.speed_accuracy_panel import speed_accuracy_figure
    speed_accuracy_figure(out, os.path.join(FIGURES_DIR, "speed_accuracy.png"))


if __name__ == "__main__":
    main()
