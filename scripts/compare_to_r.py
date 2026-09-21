"""
Compare GRIN against the R baselines (grtools / mdsdt) on the SAME matrices.

    python scripts/export_for_r.py --n 600      # 1. export a stratified sample
    Rscript scripts/R/fit_baselines.R           # 2. fit them in R
    python scripts/compare_to_r.py              # 3. compare  <- you are here

Writes results/figures/comparison_to_r.png, two panels:

  CONVERGENCE  failure rate per method, BY TRIAL COUNT (not one pooled point per method
               as the first version of this figure had -- a flat rate hides that
               failures concentrate exactly where the identifiability frontier
               analysis says information is thinnest, which is the more useful fact).
  AGREEMENT    GRIN's assembled componentwise label vs each baseline's selected class,
               DECOMPOSED BY WHO WAS RIGHT. Bare
               agreement is the wrong statistic on simulated data: "we agree 60% of the
               time" is worthless if both are wrong in most of those cases, and here the
               ground truth is known. Bare agreement belongs on real data, where it is the
               only check available.

Speed and parameter-MAE-by-trial-count used to live here as two more panels, but both
are now richer standalone content in scripts/speed_accuracy_by_trials.py (WITH
uncertainty, which these single-run numbers never had) -- kept here would just be a
strictly worse duplicate.

Deep parameter-level recovery comparison lives in scripts/make_recovery_figures.py
(results/figures/recovery/); this script deliberately does not duplicate it.
"""
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
from src.inference.model_posterior import amortized_compare
from src.viz.labels import to_grin_labels, labels_from_amortized
from scripts.export_for_r import TRIAL_BIN_LABELS, N_TRIAL_BINS
import src.grt_model as gm

CSV = os.path.join(SIMULATED_DATA_DIR, "test_set_for_R.csv")
RFITS = os.path.join(MLE_FITS_DIR, "baseline_fits.csv")
# The Python-MLE fit_selected loop below is the one genuinely slow step in this script
# (~12 model-class fits x N matrices via L-BFGS-B) -- multiple hours on this machine.
# Everything downstream of it (both R baselines already came pre-fit from RFITS, and the
# GRIN forward pass is sub-second) is cheap. Caching just this loop means a plotting-only
# change (font size, legend placement, panel layout) never has to pay that cost again;
# delete the cache file or pass --refit to force a genuine re-fit.
MLE_CACHE = os.path.join(MLE_FITS_DIR, "compare_to_r_python_mle_cache.csv")


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
    refit = "--refit" in sys.argv
    cached = (not refit) and os.path.exists(MLE_CACHE)
    if cached:
        cdf = pd.read_csv(MLE_CACHE)
        cached = len(cdf) == N   # invalidate silently if the export set changed size
    if cached:
        params["Python-MLE"] = cdf[[f"p{k}" for k in range(12)]].to_numpy(dtype=float)
        labels["Python-MLE"] = cdf["model"].to_numpy(dtype=object)
        ms["Python-MLE"] = float(cdf["_ms_per_matrix"].iloc[0])
        print(f"(Python-MLE fits loaded from cache {MLE_CACHE} -- pass --refit to redo them)")
    else:
        t0 = time.time()
        sel = [fit_selected(X[i], Xt[i]) for i in range(N)]
        ms["Python-MLE"] = 1e3 * (time.time() - t0) / N
        params["Python-MLE"] = np.array([f["params"] for f in sel], dtype=float)
        labels["Python-MLE"] = np.array([f["model"] for f in sel], dtype=object)
        cdf = pd.DataFrame(params["Python-MLE"], columns=[f"p{k}" for k in range(12)])
        cdf["model"] = labels["Python-MLE"]
        cdf["_ms_per_matrix"] = ms["Python-MLE"]
        os.makedirs(os.path.dirname(MLE_CACHE), exist_ok=True)
        cdf.to_csv(MLE_CACHE, index=False)
        print(f"wrote Python-MLE fit cache -> {MLE_CACHE}")
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
    _table(methods, params, truth, trial_bin, common)


def _table(methods, params, truth, trial_bin, common):
    """Dump the manuscript's tab:mae-by-trials rows plus the pooled-band sentences that
    accompany it, computed from the exact same `common`/`trial_bin` arrays the "Accuracy
    by data regime" figure panel uses -- so the table, the figure, and the prose can never
    silently drift apart the way they had (n=60/71 in prose vs n=44/62 in the table, an
    n_common of 206 in the caption vs 181 summed from the table's own rows) before this
    function existed.
    """
    order = ["GRIN", "Python-MLE", "mdsdt", "grtools"]
    order = [m for m in order if m in methods]
    lines = ["Trials/stimulus & $n$ & " + " & ".join(order) + r" \\"]
    band_n, band_mae = [], {m: [] for m in order}
    for b in range(N_TRIAL_BINS):
        mask = common & (trial_bin == b)
        n = int(mask.sum())
        band_n.append(n)
        row = [TRIAL_BIN_LABELS[b], str(n)]
        for m in order:
            mae = np.nanmean(np.abs(params[m][mask] - truth[mask])) if n else np.nan
            band_mae[m].append(mae)
            row.append(f"{mae:.2f}")
        lines.append(" & ".join(row) + r" \\")
    out = os.path.join(RESULTS_DIR, "compare_to_r_table.txt")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n\n")
        f.write(f"n_common (all bands) = {int(common.sum())}\n\n")
        # pooled bands: whatever contiguous prefix/suffix the manuscript text quotes
        for label, idx in [("5-20 (bands 0-2)", range(0, 3)),
                            ("75-500 (bands 6-8)", range(6, 9))]:
            n = sum(band_n[i] for i in idx)
            f.write(f"pooled {label}: n={n}\n")
            for m in order:
                mask = common & np.isin(trial_bin, list(idx))
                mae = np.nanmean(np.abs(params[m][mask] - truth[mask]))
                f.write(f"   {m:12s} MAE={mae:.3f}\n")
    print(f"wrote table/pooled numbers -> {out}")


PRINT_W = 6.5   # \textwidth in GRIN_combined_edited.tex


def _panel_convergence(ax, methods, ok, trial_bin, colour, scale, letter="A"):
    from src.viz.style import floating_trial_bands
    from src.viz.figures import _wilson
    series, err = {}, {}
    for m in methods:
        fr, ci = [], []
        for b in range(N_TRIAL_BINS):
            mask = trial_bin == b
            n = int(mask.sum())
            if n == 0:
                fr.append(np.nan); ci.append(None); continue
            k = int(ok[m][mask].sum())          # successes = converged
            rate = 100 * (1 - k / n)
            lo, hi = _wilson(n - k, n)           # CI on the FAILURE rate
            fr.append(rate); ci.append((100 * lo, 100 * hi))
        series[m] = (fr, colour.get(m, "0.5"))
        err[m] = ci
    floating_trial_bands(ax, TRIAL_BIN_LABELS, series, offset_span=0.14, err=err)
    ax.tick_params(axis="x", labelsize=7.5 * scale)
    ax.set_xlabel("trials per stimulus")
    ax.set_ylabel("fit failure rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"{letter}   Convergence by trial count" if letter
                 else "Convergence by trial count")
    ax.legend(fontsize=7.5 * scale, loc="upper right", frameon=False, handlelength=1.4)


def _panel_agreement(ax, methods, labels, true_labels, common, colour, scale, letter="B"):
    ref = "GRIN"
    ref_ex = np.array([a is not None and a == b
                       for a, b in zip(labels[ref][common], true_labels[common])])
    others = [m for m in methods if m != ref and m in labels]
    cats = [("both correct", colour["GRIN"]), (f"{ref} only", "#5AA9E6"),
            ("baseline only", "#F2A5C0"), ("both wrong", "0.65")]
    bottom = np.zeros(len(others))
    for ci, (lab, col) in enumerate(cats):
        h = []
        for m in others:
            o_ex = np.array([a is not None and a == b
                             for a, b in zip(labels[m][common], true_labels[common])])
            h.append([(ref_ex & o_ex), (ref_ex & ~o_ex),
                      (~ref_ex & o_ex), (~ref_ex & ~o_ex)][ci].mean())
        h = np.asarray(h)
        ax.bar(np.arange(len(others)), h, 0.6, bottom=bottom, color=col, label=lab)
        for xi, (hh, bb) in enumerate(zip(h, bottom)):
            if hh > 0.06:
                ax.text(xi, bb + hh / 2, f"{hh:.0%}", ha="center", va="center",
                        fontsize=7.5 * scale,
                        color="white" if lab in ("both correct", "both wrong") else "0.15")
        bottom += h
    ax.set_xticks(np.arange(len(others))); ax.set_xticklabels(others, fontsize=8 * scale)
    ax.set_ylim(0, 1.0); ax.set_ylabel("fraction of matrices")
    ax.set_title(f"{letter}   Agreement on assembled label" if letter
                 else "Agreement on assembled label")
    ax.legend(fontsize=7 * scale, loc="upper center", bbox_to_anchor=(0.5, -0.28),
              ncol=2, frameon=False)


def _figure(methods, params, labels, ok, ms, truth, true_labels, trial_bin, common, N):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.viz.style import set_style, BLUE, BLUE_DEEP, RED, RED_DEEP, MUTE, INK

    colour = {"GRIN": BLUE_DEEP, "Python-MLE": MUTE, "mdsdt": BLUE, "grtools": RED_DEEP}
    stem, ext = "results/figures/comparison_to_r", ".png"

    row_scale = 0.78
    set_style(row_scale)
    fig, ax = plt.subplots(1, 2, figsize=(PRINT_W, PRINT_W * 0.42))
    _panel_convergence(ax[0], methods, ok, trial_bin, colour, row_scale)
    _panel_agreement(ax[1], methods, labels, true_labels, common, colour, row_scale)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.32, top=0.86, wspace=0.35)
    p = f"{stem}{ext}"
    fig.savefig(p); plt.close(fig)
    print(f"figure -> {p}")

    set_style(1.0)
    out2 = f"{stem}_2row{ext}"
    fig, ax = plt.subplots(2, 1, figsize=(PRINT_W * 0.62, PRINT_W * 1.15))
    _panel_convergence(ax[0], methods, ok, trial_bin, colour, 1.0)
    _panel_agreement(ax[1], methods, labels, true_labels, common, colour, 1.0)
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.14, top=0.94, hspace=0.85)
    fig.savefig(out2); plt.close(fig)
    print(f"figure -> {out2}")

    for tag, fn, args in [
            ("a", _panel_convergence, (methods, ok, trial_bin, colour)),
            ("b", _panel_agreement, (methods, labels, true_labels, common, colour))]:
        fig, a = plt.subplots(1, 1, figsize=(PRINT_W, PRINT_W * 0.62))
        fn(a, *args, 1.0, letter="")
        fig.tight_layout()
        p = f"{stem}_{tag}{ext}"
        fig.savefig(p); plt.close(fig)
        print(f"figure -> {p}")

    # A concrete number for the concern that pooled agreement makes GRIN and the
    # baselines look unreliable rather than showing a data-identifiability limit:
    # both-wrong should fall sharply once trials per stimulus are no longer sparse.
    sparse = common & (trial_bin <= 2)     # bands 0-2: 5-10, 10-15, 15-20
    dense = common & (trial_bin >= 6)      # bands 6-8: 75-100, 100-200, 200-500
    for label_, mask in [("sparse (<=20 trials/stimulus)", sparse),
                         ("dense (>=75 trials/stimulus)", dense)]:
        if not mask.any():
            continue
        ref_ex = np.array([a is not None and a == b
                           for a, b in zip(labels["GRIN"][mask], true_labels[mask])])
        for m in methods:
            if m == "GRIN" or m not in labels:
                continue
            o_ex = np.array([a is not None and a == b
                             for a, b in zip(labels[m][mask], true_labels[mask])])
            both_wrong = float((~ref_ex & ~o_ex).mean())
            print(f"  both-wrong GRIN/{m}, {label_}: {both_wrong:.1%} (n={int(mask.sum())})")


if __name__ == "__main__":
    main()

