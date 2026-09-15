"""
model_class_components.py — decompose 12-way model-class agreement into its three
underlying components (correlation type: pi/rho1/free, separability on A, separability
on B), on the same export/fits compare_to_r.py already uses.

Motivation: Figure 4B reports pooled/per-band agreement on the FULL 12-way class, and
that number can sit near 50% in the sparse bands. Read on its own, "50% agreement"
invites "the baseline gets nothing right half the time" -- but the 12-way class is a
conjunction of three independent calls, and missing the full class usually means
missing exactly ONE of the three, not all three. This script reports, per method and
per trial-count band: accuracy on each component separately, and the distribution of
how many of the three components were correct (0/1/2/3).

EXPLORATORY -- not wired into the manuscript. Run it, read results/validation/
model_class_components.json and the printed summary, and decide whether it adds
anything Figure 4B doesn't already say before building a figure around it.

Reuses the exact same export CSV and cached fits compare_to_r.py uses, so the ground
truth and per-method predictions are identical to Figure 4 -- this is a different view
of that comparison, not a new one.

    python scripts/model_class_components.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import SIMULATED_DATA_DIR, MLE_FITS_DIR, RESULTS_DIR
from src.api import load_model
from src.inference.predict import predict_point
from src.inference.mle import fit_selected
from src.inference.model_posterior import amortized_compare
from src.viz.labels import to_grin_labels, labels_from_amortized, constructs_from_labels
from scripts.export_for_r import TRIAL_BIN_LABELS, N_TRIAL_BINS
import src.grt_model as gm

CSV = os.path.join(SIMULATED_DATA_DIR, "test_set_for_R.csv")
RFITS = os.path.join(MLE_FITS_DIR, "baseline_fits.csv")
MLE_CACHE = os.path.join(MLE_FITS_DIR, "compare_to_r_python_mle_cache.csv")
OUT_JSON = os.path.join(RESULTS_DIR, "validation", "model_class_components.json")


def _load():
    """Same data assembly as compare_to_r.py:main(), reading only cached fits -- this
    script never re-fits anything itself. Run compare_to_r.py first if either cache is
    missing.
    """
    df = pd.read_csv(CSV)
    cm_cols = [f"cm_{s}{r}" for s in range(4) for r in range(4)]
    X = df[cm_cols].to_numpy()
    Xt = df[[f"trials_{s}" for s in range(4)]].to_numpy()
    true_labels = df["model_label"].to_numpy(dtype=object)
    trial_bin = df["trial_bin"].to_numpy()
    N = len(df)

    model = load_model()
    labels, ok = {}, {}
    labels["GRIN"] = labels_from_amortized(amortized_compare(model, X, Xt))
    ok["GRIN"] = np.ones(N, bool)

    if os.path.exists(MLE_CACHE):
        cdf = pd.read_csv(MLE_CACHE)
        if len(cdf) == N:
            labels["Python-MLE"] = cdf["model"].to_numpy(dtype=object)
            ok["Python-MLE"] = np.isfinite(
                cdf[[f"p{k}" for k in range(12)]].to_numpy(dtype=float)).all(1)
    if "Python-MLE" not in labels:
        print(f"(no usable Python-MLE cache at {MLE_CACHE} -- fitting fresh, this is slow)")
        sel = [fit_selected(X[i], Xt[i]) for i in range(N)]
        labels["Python-MLE"] = np.array([f["model"] for f in sel], dtype=object)
        ok["Python-MLE"] = np.isfinite(
            np.array([f["params"] for f in sel], dtype=float)).all(1)

    if not os.path.exists(RFITS):
        raise SystemExit(f"no R fits at {RFITS} -- run: Rscript scripts/R/fit_baselines.R")
    r = pd.read_csv(RFITS).set_index("row_id")
    j = df.set_index("row_id").join(r).reset_index()
    for pkg in ("mdsdt", "grtools"):
        if f"{pkg}_model" not in j.columns:
            print(f"!! {pkg}: no fits in {RFITS} -- skipped")
            continue
        labels[pkg] = to_grin_labels(j[f"{pkg}_model"].to_numpy(dtype=object))
        ok[pkg] = j[f"{pkg}_ok"].fillna(False).to_numpy(dtype=bool)

    methods = list(labels)
    common = np.ones(N, bool)
    for m in methods:
        common &= ok[m]
    return methods, labels, true_labels, trial_bin, common, N


def _breakdown(pred_labels, true_labels, mask):
    tc, tx, ty = constructs_from_labels(true_labels[mask])
    pc, px, py = constructs_from_labels(pred_labels[mask])
    corr_ok = pc == tc
    a_ok = px == tx
    b_ok = py == ty
    n_correct = corr_ok.astype(int) + a_ok.astype(int) + b_ok.astype(int)
    n = int(mask.sum())
    if n == 0:
        return None
    return dict(
        n=n,
        corr_acc=float(corr_ok.mean()),
        sepA_acc=float(a_ok.mean()),
        sepB_acc=float(b_ok.mean()),
        full_class_acc=float((n_correct == 3).mean()),
        zero_correct=float((n_correct == 0).mean()),
        at_least_one_correct=float((n_correct >= 1).mean()),
        at_least_two_correct=float((n_correct >= 2).mean()),
        mean_n_correct=float(n_correct.mean()),
    )


def main():
    methods, labels, true_labels, trial_bin, common, N = _load()
    order = [m for m in ("GRIN", "Python-MLE", "mdsdt", "grtools") if m in methods]

    out = dict(meta=dict(n_total=int(N), n_common=int(common.sum())),
               pooled={}, by_trial_band={m: {} for m in order})

    print(f"N = {N} matrices, {int(common.sum())} common to every method\n")
    print("=== POOLED (matrices every method fitted) ===")
    header = f"{'method':12s} {'full':>6} {'corr':>6} {'sepA':>6} {'sepB':>6} " \
             f"{'>=1':>6} {'>=2':>6} {'mean#':>6} {'0/3':>6}"
    print(header)
    for m in order:
        b = _breakdown(labels[m], true_labels, common)
        out["pooled"][m] = b
        print(f"{m:12s} {b['full_class_acc']:6.3f} {b['corr_acc']:6.3f} "
              f"{b['sepA_acc']:6.3f} {b['sepB_acc']:6.3f} "
              f"{b['at_least_one_correct']:6.3f} {b['at_least_two_correct']:6.3f} "
              f"{b['mean_n_correct']:6.2f} {b['zero_correct']:6.3f}")

    print("\n=== BY TRIAL-COUNT BAND (mean_n_correct out of 3, full-class acc) ===")
    print(f"{'band':>8}  " + "  ".join(f"{m:>22s}" for m in order))
    for bi in range(N_TRIAL_BINS):
        bmask = common & (trial_bin == bi)
        row = [TRIAL_BIN_LABELS[bi]]
        for m in order:
            b = _breakdown(labels[m], true_labels, bmask)
            out["by_trial_band"][m][TRIAL_BIN_LABELS[bi]] = b
            row.append("n/a" if b is None else
                        f"mean={b['mean_n_correct']:.2f} full={b['full_class_acc']:.2f}")
        print(f"{row[0]:>8}  " + "  ".join(f"{c:>22s}" for c in row[1:]))

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    import json
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT_JSON}")
    return out


if __name__ == "__main__":
    main()
