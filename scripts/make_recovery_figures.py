"""
make_recovery_figures.py — build the whole parameter-recovery family.

    python scripts/export_for_r.py --n 600        # 1. stratified matrices for R
    Rscript scripts/R/fit_baselines.R             # 2. fit them with mdsdt + grtools
    python scripts/make_recovery_figures.py       # 3. figures  <- you are here

Writes to results/figures/recovery/:

  (The GRIN showcase figure — 200 trials/stimulus, balanced — is NOT produced here. It is
  results/figures/recovery.png, written by scripts/make_figures.py through the same
  recovery_panels() code. Generating it twice would give two files of the same thing.)

  recovery_<method>.png           GRIN / mdsdt / grtools / Python-MLE on the SAME stratified
                                  matrices. Identical geometry and axes so they can be laid
                                  side by side. Hue by construct, shape by whether that
                                  method's own model selection got THAT construct right.

  recovery_<method>_12way.png     Same, but shaped by 12-way exact-class correctness.
                                  Exploratory: a point can be marked wrong for a failure in
                                  a construct the panel is not about, so these are kept out
                                  of the poster set.

  summary_recovery.png            MAE by parameter family, by trial regime, and rho MAE by
                                  true |rho| — the identifiability frontier, per method.
  summary_classification.png      Per-construct and 12-way accuracy, plus the agreement
                                  decomposition (both correct / one only / both wrong).

TWO THINGS THIS SCRIPT REFUSES TO DO QUIETLY
  1. Unparseable baseline labels raise (see src/viz/labels.py). A silent parse failure would
     mark every baseline point wrong and make GRIN look perfect.
  2. Headline numbers are computed on the subset where EVERY method converged. Dropping each
     method's own failures would score each method only on the matrices it found easy, which
     systematically flatters whichever method fails most. The excluded count and the
     per-method failure rate are printed next to the results, not hidden.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.export_for_r import TRIAL_BIN_LABELS
from src.config import (SIMULATED_DATA_DIR, MLE_FITS_DIR, FIGURES_DIR, MODEL_FILE,
                        Z_MAX, R_MAX)
from src.api import load_model
from src.inference.predict import predict_point
from src.inference.model_posterior import amortized_compare
from src.viz import recovery as R
from src.viz.labels import to_grin_labels, labels_from_amortized
import src.grt_model as gm

CSV = os.path.join(SIMULATED_DATA_DIR, "test_set_for_R.csv")
RFITS = os.path.join(MLE_FITS_DIR, "baseline_fits.csv")
OUT = os.path.join(FIGURES_DIR, "recovery")

# Imported, not restated: the export writes 9 trials/stimulus bins and a local
# ("low", "mid", "high") here scored only the first three of them (see
# export_for_r.TPS_EDGES for the full note).
TRIAL_BIN_NAMES = TRIAL_BIN_LABELS
RHO_BIN_NAMES = ("PI", "weak", "mod", "strong")


# ------------------------------------------------------------------ helpers
def _params_from(df, prefix):
    """Pull a (N, 12) parameter block written as <prefix>_zx_0 ... <prefix>_rho_3."""
    cols = [f"{prefix}_{n}" for n in gm.PARAM_NAMES]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{prefix}: missing parameter columns {missing[:4]}"
                       f"{'...' if len(missing) > 4 else ''} — re-run fit_baselines.R "
                       "(the version that saves parameters).")
    return df[cols].to_numpy(dtype=float)


def _grin_labels(model, X, Xt, parsimony=0.0):
    """GRIN's 12-way class = argmax of each construct from the comparison head, composed."""
    ac = amortized_compare(model, X, Xt, parsimony=parsimony)
    return labels_from_amortized(ac), ac


def _construct_correct(pred_labels, true_labels):
    """-> dict of per-construct boolean arrays + the 12-way exact array."""
    from src.viz.labels import constructs_from_labels
    pc, px, py = constructs_from_labels(pred_labels)
    tc, tx, ty = constructs_from_labels(true_labels)
    exact = np.array([(p is not None) and (p == t)
                      for p, t in zip(pred_labels, true_labels)])
    return {"corr": pc == tc, "ps_x": px == tx, "ps_y": py == ty}, exact


# ------------------------------------------------------------------ comparison family
def comparison(model, with_mle=True, mle_select=False, parsimony=0.0):
    df = pd.read_csv(CSV)
    cm_cols = [f"cm_{s}{r}" for s in range(4) for r in range(4)]
    X = df[cm_cols].to_numpy()
    Xt = df[[f"trials_{s}" for s in range(4)]].to_numpy()
    truth = df[gm.PARAM_NAMES].to_numpy(dtype=float)
    true_labels = df["model_label"].to_numpy(dtype=object)
    trial_bin = df["trial_bin"].to_numpy()
    rho_bin = df["rho_bin"].to_numpy()
    N = len(df)

    params, labels, converged = {}, {}, {}

    # --- GRIN -------------------------------------------------------------
    params["GRIN"] = predict_point(model, X, Xt).numpy()
    labels["GRIN"], _ = _grin_labels(model, X, Xt, parsimony=parsimony)
    converged["GRIN"] = np.ones(N, bool)          # always returns an answer

    # --- R baselines ------------------------------------------------------
    if not os.path.exists(RFITS):
        raise SystemExit(f"no R fits at {RFITS} — run: Rscript scripts/R/fit_baselines.R")
    r = pd.read_csv(RFITS).set_index("row_id")
    j = df.set_index("row_id").join(r).reset_index()
    for pkg, disp in (("mdsdt", "mdsdt"), ("grtools", "grtools")):
        params[disp] = _params_from(j, pkg)
        labels[disp] = to_grin_labels(j[f"{pkg}_model"].to_numpy(dtype=object))
        converged[disp] = (j[f"{pkg}_ok"].fillna(False).to_numpy(dtype=bool)
                           & np.isfinite(params[disp]).all(1))

    # --- Python-MLE -------------------------------------------------------
    if with_mle:
        from src.inference.mle import fit_full, fit_and_select
        print("  fitting Python-MLE ...", flush=True)
        p = np.full((N, 12), np.nan)
        lab = np.empty(N, dtype=object)
        for i in range(N):
            try:
                p[i] = fit_full(X[i], Xt[i])["params"]
            except Exception:
                pass
            if mle_select:
                try:
                    best, _all = fit_and_select(X[i], Xt[i], criterion="bic")
                    lab[i] = best.get("model") or best.get("model_name")
                except Exception:
                    lab[i] = None
            else:
                lab[i] = None
        params["Python-MLE"] = p
        labels["Python-MLE"] = lab
        converged["Python-MLE"] = np.isfinite(p).all(1)

    methods = list(params)

    # --- the common-convergence subset ------------------------------------
    common = np.ones(N, bool)
    for m in methods:
        common &= converged[m]
    print(f"\n  convergence on {N} matrices")
    for m in methods:
        print(f"    {m:12s} {converged[m].sum():4d} ok  "
              f"({100 * (1 - converged[m].mean()):5.1f}% failure)")
    print(f"    {'COMMON':12s} {common.sum():4d} matrices scored "
          f"({N - common.sum()} excluded so every method is scored on the same data)\n")
    if common.sum() < 30:
        raise SystemExit("fewer than 30 matrices converged for every method — "
                         "the comparison would not be meaningful. Check the R fits.")

    regime = f"stratified set — n={common.sum()} scored by all methods"

    # --- per-method panel grids -------------------------------------------
    for m in methods:
        per_construct, exact = _construct_correct(labels[m][common], true_labels[common])
        has_labels = any(l is not None for l in labels[m][common])
        R.recovery_panels(truth[common], params[m][common],
                          os.path.join(OUT, f"recovery_{m.lower().replace('-', '_')}.png"),
                          class_names=true_labels[common],
                          correct=per_construct if has_labels else None,
                          method=m, regime=regime, z_max=Z_MAX, r_max=R_MAX)
        if has_labels:
            R.recovery_panels(truth[common], params[m][common],
                              os.path.join(OUT,
                                           f"recovery_{m.lower().replace('-', '_')}_12way.png"),
                              class_names=true_labels[common], correct=exact,
                              method=m, regime=regime + " — 12-way exact shading",
                              z_max=Z_MAX, r_max=R_MAX)
        print(f"  recovery_{m.lower().replace('-', '_')}.png")

    # --- summaries ---------------------------------------------------------
    R.summary_recovery(
        {m: {"true": truth[common], "pred": params[m][common],
             "trial_bin": trial_bin[common], "rho_bin": rho_bin[common]} for m in methods},
        os.path.join(OUT, "summary_recovery.png"),
        trial_bin_names=TRIAL_BIN_NAMES, rho_bin_names=RHO_BIN_NAMES)
    print("  summary_recovery.png")

    cls_methods = {m: labels[m][common] for m in methods
                   if any(l is not None for l in labels[m][common])}
    R.summary_classification(true_labels[common], cls_methods,
                             os.path.join(OUT, "summary_classification.png"),
                             reference="GRIN")
    print("  summary_classification.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-mle", action="store_true",
                    help="skip the Python-MLE reference method (it is the slow one)")
    ap.add_argument("--mle-select", action="store_true",
                    help="also run AIC/BIC selection for Python-MLE (fits 12 classes per "
                         "matrix — slow; without it Python-MLE appears in the recovery "
                         "figures but not the classification ones)")
    ap.add_argument("--parsimony", type=float, default=0.0,
                    help="Occam prior on GRIN's correlation logits (see amortized_compare)")
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    model = load_model(MODEL_FILE)
    print(f"writing to {OUT}")
    comparison(model, with_mle=not a.no_mle, mle_select=a.mle_select,
               parsimony=a.parsimony)
    print("\ndone")


if __name__ == "__main__":
    main()
