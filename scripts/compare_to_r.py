"""
Compare GRIN against the R baselines (grtools / mdsdt) on the SAME matrices.

    python scripts/export_for_r.py --n 600      # 1. export a stratified sample
    Rscript scripts/R/fit_baselines.R           # 2. fit them in R
    python scripts/compare_to_r.py              # 3. compare  <- you are here

Writes results/figures/comparison_to_r.png, four panels:

  SPEED        wall-clock ms/matrix for every method. GRIN is reported BATCHED and
               PER-MATRIX, because batched throughput against serial per-matrix fitting is
               a throughput claim, not a latency one, and the honest comparator for "one
               participant walks in" is the single-matrix number.
  CONVERGENCE  failure rate per method. GRIN always returns a calibrated posterior; the MLE
               implementations do not always return anything. This is a real practical
               advantage and the panel the previous version promised in its docstring but
               never actually drew.
  ACCURACY     parameter MAE by trial regime, every method, on the subset where EVERY
               method converged.
  AGREEMENT    GRIN vs each baseline on model class, DECOMPOSED BY WHO WAS RIGHT. Bare
               agreement is the wrong statistic on simulated data: "we agree 60% of the
               time" is worthless if both are wrong in most of those cases, and here the
               ground truth is known. Bare agreement belongs on real data, where it is the
               only check available.

Deep parameter-level recovery comparison lives in scripts/make_recovery_figures.py
(results/figures/recovery/); this script deliberately does not duplicate it.
"""
import os
import time

import numpy as np
import pandas as pd

from src.config import SIMULATED_DATA_DIR, MLE_FITS_DIR, FIGURES_DIR, RESULTS_DIR
from src.api import load_model
from src.inference.predict import predict_point
from src.inference.mle import fit_selected
from src.inference.model_posterior import amortized_compare
from src.viz.labels import to_grin_labels, labels_from_amortized
import src.grt_model as gm

CSV = os.path.join(SIMULATED_DATA_DIR, "test_set_for_R.csv")
RFITS = os.path.join(MLE_FITS_DIR, "baseline_fits.csv")
TRIAL_BIN_NAMES = ("low", "mid", "high")


def _params(df, prefix):
    cols = [f"{prefix}_{n}" for n in gm.PARAM_NAMES]
    if any(c not in df.columns for c in cols):
        return None
    return df[cols].to_numpy(dtype=float)


def main(n_single=50):
    df = pd.read_csv(CSV)
    cm_cols = [f"cm_{s}{r}" for s in range(4) for r in range(4)]
    X = df[cm_cols].to_numpy()
    Xt = df[[f"trials_{s}" for s in range(4)]].to_numpy()
    truth = df[gm.PARAM_NAMES].to_numpy(dtype=float)
    true_labels = df["model_label"].to_numpy(dtype=object)
    trial_bin = df["trial_bin"].to_numpy()
    N = len(df)

    model = load_model()
    params, labels, ok, ms = {}, {}, {}, {}

    t0 = time.time(); grin = predict_point(model, X, Xt).numpy()
    ms["GRIN (batched)"] = 1e3 * (time.time() - t0) / N
    t0 = time.time()
    for i in range(min(n_single, N)):
        predict_point(model, X[i:i + 1], Xt[i:i + 1])
    ms["GRIN (1 matrix)"] = 1e3 * (time.time() - t0) / min(n_single, N)
    params["GRIN"] = grin
    labels["GRIN"] = labels_from_amortized(amortized_compare(model, X, Xt))
    ok["GRIN"] = np.ones(N, bool)

    # Only the SELECTED workflow — nobody reports the saturated (full) fit, so it is not a
    # baseline anyone competes against. fit_selected returns the AIC/BIC winner's packed
    # params AND its class name, so it drives both the accuracy and the agreement panels.
    t0 = time.time()
    sel = [fit_selected(X[i], Xt[i]) for i in range(N)]
    ms["Python-MLE"] = 1e3 * (time.time() - t0) / N
    params["Python-MLE"] = np.array([f["params"] for f in sel], dtype=float)
    labels["Python-MLE"] = np.array([f["model"] for f in sel], dtype=object)
    ok["Python-MLE"] = np.isfinite(params["Python-MLE"]).all(1)

    if not os.path.exists(RFITS):
        raise SystemExit(f"no R fits at {RFITS} — run: Rscript scripts/R/fit_baselines.R")
    r = pd.read_csv(RFITS).set_index("row_id")
    j = df.set_index("row_id").join(r).reset_index()
    for pkg in ("mdsdt", "grtools"):
        p = _params(j, pkg)
        if p is None:
            print(f"!! {pkg}: no parameter columns in {RFITS} — re-run fit_baselines.R")
            continue
        params[pkg] = p
        labels[pkg] = to_grin_labels(j[f"{pkg}_model"].to_numpy(dtype=object))
        ok[pkg] = j[f"{pkg}_ok"].fillna(False).to_numpy(dtype=bool) & np.isfinite(p).all(1)
        ms[pkg] = 1e3 * float(np.nanmean(j[f"{pkg}_secs"].to_numpy(dtype=float)))
    if "grtools_1rep_secs" in j.columns:
        ms["grtools (1 rep)"] = 1e3 * float(np.nanmean(
            j["grtools_1rep_secs"].to_numpy(dtype=float)))

    # OPTIONAL: fold in the +RT model's timing if make_figures_rt.py has exported it.
    # Speed only — the RT model was evaluated on its own held-out set, so its accuracy is
    # NOT comparable on these exact matrices and is deliberately left out of panel 3.
    import json as _json
    rt_json = os.path.join(RESULTS_DIR, "rt_metrics.json")
    if os.path.exists(rt_json):
        _rt = _json.load(open(rt_json))
        ms["+RT (1 matrix)"] = _rt["rt_model"]["single_ms"]
        print(f"(+RT model timing folded in from {rt_json})")

    methods = list(params)
    common = np.ones(N, bool)
    for m in methods:
        common &= ok[m]

    print("=== CONVERGENCE ===")
    for m in methods:
        print(f"   {m:14s} {ok[m].sum():4d}/{N} ok   ({100 * (1 - ok[m].mean()):5.1f}% failure)")
    print(f"   COMMON         {common.sum():4d}/{N} scored by every method\n")
    print("=== SPEED (ms/matrix) ===")
    for k, v in ms.items():
        print(f"   {k:24s} {v:10.4f}")
    print()

    _figure(methods, params, labels, ok, ms, truth, true_labels, trial_bin, common, N)


def _figure(methods, params, labels, ok, ms, truth, true_labels, trial_bin, common, N):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.viz.style import set_style, BLUE, BLUE_DEEP, RED, RED_DEEP, MUTE, INK
    from src.viz.figures import _wilson

    set_style()
    colour = {"GRIN": BLUE_DEEP, "Python-MLE": MUTE, "mdsdt": BLUE, "grtools": RED_DEEP}
    fig, ax = plt.subplots(1, 4, figsize=(21, 5))

    # --- 1. speed ---
    keys = list(ms)
    vals = [ms[k] for k in keys]
    cols = [BLUE_DEEP if k.startswith("GRIN") else
            (MUTE if k.startswith("Python") else (BLUE if "mdsdt" in k else RED_DEEP))
            for k in keys]
    ax[0].bar(range(len(keys)), vals, color=cols, width=0.65)
    ax[0].set_yscale("log"); ax[0].set_ylabel("ms per matrix (log)")
    ax[0].set_xticks(range(len(keys)))
    ax[0].set_xticklabels(keys, rotation=35, ha="right", fontsize=8.5)
    for i, v in enumerate(vals):
        ax[0].text(i, v, f" {v:.3g}", ha="center", va="bottom", fontsize=8)
    baseline_ms = {k: v for k, v in ms.items() if not k.startswith("GRIN")}
    slow = max(baseline_ms.values())
    slow_name = max(baseline_ms, key=baseline_ms.get)
    ax[0].set_title(f"Speed — {slow / ms['GRIN (1 matrix)']:,.0f}× faster than {slow_name}\n"
                    f"(per matrix; GRIN batched is {slow / ms['GRIN (batched)']:,.0f}×)")

    # --- 2. convergence ---
    fr = [100 * (1 - ok[m].mean()) for m in methods]
    err = np.array([[100 * (1 - h), 100 * (1 - l)] for m in methods
                    for l, h in [_wilson(int(ok[m].sum()), N)]])
    e = np.vstack([np.clip(np.array(fr) - err[:, 0], 0, None),
                   np.clip(err[:, 1] - np.array(fr), 0, None)])
    ax[1].bar(methods, fr, color=[colour.get(m, MUTE) for m in methods], width=0.6,
              yerr=e, error_kw=dict(ecolor=INK, elinewidth=0.9, capsize=3, alpha=0.7))
    ax[1].set_ylabel("fit failure rate (%)")
    ax[1].set_title("Convergence — GRIN always returns an answer")
    ax[1].tick_params(axis="x", labelrotation=25, labelsize=9)
    for i, v in enumerate(fr):
        ax[1].text(i, v, f" {v:.1f}%", ha="center", va="bottom", fontsize=8.5)

    # --- 3. accuracy by trial regime, common-convergence subset ---
    x = np.arange(len(TRIAL_BIN_NAMES)); w = 0.8 / len(methods)
    for i, m in enumerate(methods):
        v = [np.nanmean(np.abs(params[m][common & (trial_bin == b)] -
                               truth[common & (trial_bin == b)]))
             for b in range(len(TRIAL_BIN_NAMES))]
        ax[2].bar(x + (i - (len(methods) - 1) / 2) * w, v, w * 0.92,
                  color=colour.get(m, MUTE), label=m)
    ax[2].set_xticks(x); ax[2].set_xticklabels([f"{n}\ntrials" for n in TRIAL_BIN_NAMES])
    ax[2].set_ylabel("parameter MAE")
    ax[2].set_title(f"Accuracy by data regime (n={int(common.sum())})")
    ax[2].legend(fontsize=8.5)

    # --- 4. agreement, decomposed by who was right ---
    ref = "GRIN"
    ref_ex = np.array([a is not None and a == b
                       for a, b in zip(labels[ref][common], true_labels[common])])
    others = [m for m in methods if m != ref and m in labels]
    cats = [("both correct", BLUE_DEEP), (f"{ref} only", BLUE),
            ("baseline only", RED), ("both wrong", MUTE)]
    bottom = np.zeros(len(others))
    for ci, (lab, col) in enumerate(cats):
        h = []
        for m in others:
            o_ex = np.array([a is not None and a == b
                             for a, b in zip(labels[m][common], true_labels[common])])
            h.append([(ref_ex & o_ex), (ref_ex & ~o_ex),
                      (~ref_ex & o_ex), (~ref_ex & ~o_ex)][ci].mean())
        h = np.asarray(h)
        ax[3].bar(np.arange(len(others)), h, 0.6, bottom=bottom, color=col, label=lab)
        for xi, (hh, bb) in enumerate(zip(h, bottom)):
            if hh > 0.06:
                ax[3].text(xi, bb + hh / 2, f"{hh:.0%}", ha="center", va="center",
                           fontsize=8.5, color="white" if col in (BLUE_DEEP, MUTE) else INK)
        bottom += h
    ax[3].set_xticks(np.arange(len(others))); ax[3].set_xticklabels(others, fontsize=9)
    ax[3].set_ylim(0, 1); ax[3].set_ylabel("fraction of matrices")
    ax[3].set_title("Agreement on model class — and who was right")
    ax[3].legend(fontsize=8.5)

    fig.suptitle("GRIN vs the R baselines, same matrices", x=0.02, ha="left",
                 fontweight="bold", fontsize=15, color=INK)
    fig.text(0.02, 0.945,
             f"stratified simulated set, n={N}; accuracy scored on the "
             f"{int(common.sum())} matrices where every method converged",
             ha="left", va="top", fontsize=10, color=MUTE)
    p = os.path.join(FIGURES_DIR, "comparison_to_r.png")
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(p); plt.close(fig)
    print(f"figure -> {p}")


if __name__ == "__main__":
    main()
    