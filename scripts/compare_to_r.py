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
from scripts.export_for_r import TRIAL_BIN_LABELS, N_TRIAL_BINS
import src.grt_model as gm

CSV = os.path.join(SIMULATED_DATA_DIR, "test_set_for_R.csv")
RFITS = os.path.join(MLE_FITS_DIR, "baseline_fits.csv")


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
    # Prefer a dedicated repeated-measures timing run over the single in-script
    # measurements above. Those are taken while this process is also fitting the MLE
    # baseline and building figures, so they are a throughput estimate under load, not
    # a clean latency measurement -- and the manuscript quotes the clean one. Same
    # fold-in idiom as the +RT timing below. Delete results/timing_laptop.json to fall
    # back to whatever this run measured.
    t_json = os.path.join(RESULTS_DIR, "timing_laptop.json")
    if os.path.exists(t_json):
        _t = _json.load(open(t_json))
        ms["GRIN (batched)"]  = _t["grin_batched"]["median_ms"]
        ms["GRIN (1 matrix)"] = _t["grin_single"]["median_ms"]
        ms["Python-MLE"]      = _t["python_mle"]["median_ms"]
        print(f"(GRIN/Python-MLE timing taken from {t_json} -- medians of "
              f"{_t['grin_batched']['reps']}/{_t['python_mle']['reps']} dedicated reps, "
              f"not this run's single in-script measurement)")

    # The response-time model is NOT folded into this figure. It is trained on a
    # generator that docs/dynamic_grt_rt_design.md retired on 2026-08-14, and that
    # document forbids using it as evidence until the replacement passes its gates.
    # Set GRIN_INCLUDE_RT=1 only for developmental comparisons, never for the manuscript.
    if os.environ.get("GRIN_INCLUDE_RT") == "1":
        rt_json = os.path.join(RESULTS_DIR, "rt_metrics.json")
        if os.path.exists(rt_json):
            _rt = _json.load(open(rt_json))
            ms["+RT (1 matrix)"] = _rt["rt_model"]["single_ms"]
            print(f"(+RT timing folded in from {rt_json} -- DEVELOPMENTAL ONLY)")

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
    # Dots on a log axis, not bars: a bar on a log scale encodes nothing meaningful in
    # its length (the baseline is arbitrary), and the level is the whole message. A stem
    # to the axis floor keeps each point anchored to its label without the ink of a bar.
    floor = min(vals) / 6
    ax[0].vlines(range(len(keys)), floor, vals, color=cols, lw=1.1, alpha=0.45)
    ax[0].plot(range(len(keys)), vals, "o", ms=8, ls="none",
               markerfacecolor="none", markeredgewidth=0)
    for i, (v, c) in enumerate(zip(vals, cols)):
        ax[0].plot([i], [v], "o", color=c, ms=7.5, zorder=3)
    ax[0].set_yscale("log"); ax[0].set_ylabel("ms per matrix (log)")
    ax[0].set_ylim(bottom=floor)
    ax[0].set_xlim(-0.6, len(keys) - 0.4)
    ax[0].set_xticks(range(len(keys)))
    ax[0].set_xticklabels(keys, rotation=35, ha="right", fontsize=8.5)
    for i, v in enumerate(vals):
        ax[0].text(i, v * 1.35, f"{v:.3g}", ha="center", va="bottom", fontsize=8)
    baseline_ms = {k: v for k, v in ms.items() if not k.startswith("GRIN")}
    slow = max(baseline_ms.values())
    slow_name = max(baseline_ms, key=baseline_ms.get)
    # Headline the within-machine comparison (GRIN vs the Python MLE, identical forward
    # model, same CPU). The R timings come from a different machine; see the manuscript's
    # Speed subsection. Quoting a cross-machine ratio as the headline would overstate it.
    pymle = ms.get("Python-MLE")
    ax[0].set_title(f"Speed — {pymle / ms['GRIN (1 matrix)']:,.0f}× faster than the\n"
                    f"same-model MLE ({pymle / ms['GRIN (batched)']:,.0f}× batched)")

    # --- 2. convergence ---
    fr = [100 * (1 - ok[m].mean()) for m in methods]
    err = np.array([[100 * (1 - h), 100 * (1 - l)] for m in methods
                    for l, h in [_wilson(int(ok[m].sum()), N)]])
    e = np.vstack([np.clip(np.array(fr) - err[:, 0], 0, None),
                   np.clip(err[:, 1] - np.array(fr), 0, None)])
    xi = np.arange(len(methods))
    for i, m in enumerate(methods):
        c = colour.get(m, MUTE)
        ax[1].vlines(i, err[i, 0], err[i, 1], color=c, lw=1.5, alpha=0.85, zorder=2)
        ax[1].plot([i], [fr[i]], "o", color=c, ms=7.5, zorder=3)
    ax[1].axhline(0, color=MUTE, lw=0.9, ls=(0, (4, 3)), zorder=0)
    ax[1].set_xticks(xi); ax[1].set_xticklabels(methods)
    ax[1].set_xlim(-0.6, len(methods) - 0.4)
    ax[1].set_ylabel("fit failure rate (%)")
    ax[1].set_title("Convergence — GRIN always returns an answer")
    ax[1].tick_params(axis="x", labelrotation=25, labelsize=9)
    for i, v in enumerate(fr):
        ax[1].text(i + 0.12, v, f"{v:.1f}%", ha="left", va="center", fontsize=8.5)

    # --- 3. accuracy by trial regime, common-convergence subset ---
    # Bins come from export_for_r.TRIAL_BIN_LABELS, so this scores EVERY band the export
    # wrote. It previously walked range(len("low","mid","high")) against a 9-bin export
    # and silently dropped bins 3-8 -- the bar drawn as "high" was 15-20 trials/stimulus.
    # Lines rather than grouped bars: 9 bins x 4 methods is 36 bars, and MAE against
    # trials per stimulus is a curve, which is what the manuscript actually claims.
    x = np.arange(N_TRIAL_BINS)
    counts = [int((common & (trial_bin == b)).sum()) for b in range(N_TRIAL_BINS)]
    for m in methods:
        v = [np.nanmean(np.abs(params[m][common & (trial_bin == b)] -
                               truth[common & (trial_bin == b)]))
             if counts[b] else np.nan
             for b in range(N_TRIAL_BINS)]
        ax[2].plot(x, v, marker="o", ms=4, lw=1.6, color=colour.get(m, MUTE), label=m)
    ax[2].set_xticks(x)
    ax[2].set_xticklabels([f"{l}\n(n={c})" for l, c in zip(TRIAL_BIN_LABELS, counts)],
                          fontsize=6.5)
    ax[2].set_xlabel("trials per stimulus")
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
    