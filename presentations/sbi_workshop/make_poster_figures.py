"""
make_poster_figures.py — poster-scale re-renders of the figures the poster uses.

Run from the PROJECT ROOT:

    python presentations/sbi_workshop/make_poster_figures.py

Writes to presentations/sbi_workshop/figures/, which poster.tex searches BEFORE
results/figures/, so these override the standard suite automatically. Nothing here
touches results/figures/.

WHY THIS EXISTS
---------------
A poster column is 8.6 in. The standard suite is drawn at scale=1.0 for a manuscript page
held at reading distance; dropped into a column and viewed from three feet, the type is far
too small. Two fixes: scale=POSTER_SCALE, and bespoke compact variants of figures whose
manuscript versions have too many panels to survive the reduction.

WHAT CHANGED IN THIS VERSION
----------------------------
* poster_recovery is now THREE panels, not four, and they are chosen so the panels are the
  two numbers the poster quotes plus one thing the manuscript figure cannot show. See
  poster_recovery() for the reasoning about the zx/zy pairing.
* calibration gets a coverage-only variant. The manuscript version is now three panels at
  16.5 in wide; reduced to a poster column it is unreadable, and the rank histograms are
  the technical half. Coverage on the diagonal is the half a passer-by can read in three
  seconds, which is exactly the claim the column makes.
* speed_accuracy reports GRIN BATCHED and PER-MATRIX. The poster says a posterior update
  "fits inside an inter-stimulus interval" — that is a latency claim, and a batched timing
  divided by the batch size is a throughput number. The single-matrix figure is the one
  that supports the sentence.
* accuracy_crossover is new: the "below ~100 trials the network is also more accurate"
  claim currently appears in the poster as bare text with no evidence behind it.
"""
import os
import sys
import time
import json

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from src.config import MODEL_FILE, DEVICE, Z_MAX, R_MAX, RESULTS_DIR
from src.api import load_model
from src.data.generator import GRTDataGenerator
from src.inference.predict import predict_posterior, predict_point
from src.inference.model_posterior import model_posterior
from src.inference.mle import (fit_full, fit_selected, fit_full_multistart,
                               fit_selected_multistart, fit_full_penalised,
                               fit_selected_penalised)
from src.viz import figures as F
from src.viz.style import (set_style, set_background, BLUE, BLUE_DEEP, RED,
                           RED_DEEP, MUTE, INK)
import src.grt_model as gm

# Type scale for the poster. 1.8 is tuned for an 8.6 in column read from ~3 ft.
# If you widen the columns (fewer than five), drop this toward 1.5.
POSTER_SCALE = 1.8

# Transparent backgrounds. An opaque white figure sits in a white box on a tinted poster.
# Pass a hex string instead of None to match a specific backdrop exactly.
POSTER_BG = None
set_background(POSTER_BG)

# Pseudo-count for the penalised MLE baseline. 0.5 is the Jeffreys prior. See
# scripts/check_mle_health.py: without it, maximum likelihood is UNBOUNDED on any matrix
# with an empty cell, which is most of them, and the comparison is against a divergent
# estimator rather than a real one.
PSEUDO = 0.5

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT_DIR, exist_ok=True)
out = lambda name: os.path.join(OUT_DIR, name)

# Stimulus labels in canonical index order (index i pairs zx_i with zy_i).
STIM = ["A1B1", "A1B2", "A2B1", "A2B2"]
STIM_COLORS = [BLUE_DEEP, BLUE, RED, RED_DEEP]

# ---------------------------------------------------------------------------
# STANDARDISED FIGURE SLOTS — these THREE numbers must match grinposter.sty
# (\figheadh, \figrowh, and the column width \colw). The poster reserves a slot
# of height  FIG_HEAD_IN + rows*FIG_ROW_IN  at column width COL_W_IN; after each
# figure is written we pad it (with TRANSPARENCY, never cropping) to exactly
# that slot's aspect, so it fills the slot with no letterboxing and every N-row
# figure is the same height on the board. Padding is transparent, not cream,
# because set_background(None) above makes the figures themselves transparent —
# an opaque pad would reintroduce the very rectangle that buys us.
COL_W_IN = 8.6      # = \colw  = (45in - 4*0.5in gutters) / 5 columns
FIG_HEAD_IN = 0.60  # = \figheadh
FIG_ROW_IN = 1.75   # = \figrowh


def slot_aspect(rows):
    """Height/width of the on-poster slot for a figure with this many rows."""
    return (FIG_HEAD_IN + rows * FIG_ROW_IN) / COL_W_IN


def standardize(path, rows):
    """Pad a saved figure to the slot aspect for `rows`, in place, transparently.

    Only ever ADDS transparent margin, so nothing is cropped or squashed. Pads
    whichever dimension is short. Warns only if it had to shrink the content
    width by more than 10% (which means the figure is much taller than the row
    count you gave it — bump `rows`).
    """
    try:
        from PIL import Image
    except ImportError:
        print("  [standardize] Pillow not installed — skipping (LaTeX slots still "
              "constrain layout). Install: pip install pillow --break-system-packages")
        return
    im = Image.open(path).convert("RGBA")
    w, h = im.size
    target = slot_aspect(rows)
    cur = h / w
    if abs(cur - target) < 2e-3:
        return
    if cur < target:                              # too short -> pad top & bottom
        new_h = round(w * target)
        canvas = Image.new("RGBA", (w, new_h), (0, 0, 0, 0))
        canvas.paste(im, (0, (new_h - h) // 2), im)
    else:                                          # too tall -> pad L & R
        new_w = round(h / target)
        if (new_w - w) / new_w > 0.10:
            print(f"  [standardize] NOTE {os.path.basename(path)} is much taller than a "
                  f"{rows}-row slot (aspect {cur:.2f} vs {target:.2f}); it will be narrowed "
                  f"to fit. Consider \\gfig[{rows+1}] for it.")
        canvas = Image.new("RGBA", (new_w, h), (0, 0, 0, 0))
        canvas.paste(im, ((new_w - w) // 2, 0), im)
    canvas.save(path)


def _cov_ellipse(ax, xy, color, n_std=2.0, **kw):
    """95% (2 sd) covariance ellipse for a 2-D point cloud."""
    xy = xy[np.isfinite(xy).all(1)]
    if len(xy) < 3:
        return
    mu = xy.mean(0)
    vals, vecs = np.linalg.eigh(np.cov(xy.T))
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    ang = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * n_std * np.sqrt(np.maximum(vals, 0))
    ax.add_patch(Ellipse(mu, w, h, angle=ang, facecolor="none", edgecolor=color,
                         lw=2.0, **kw))


def poster_recovery(true, pred, path, scale=POSTER_SCALE):
    """Compact recovery: pooled sensitivities, pooled correlations, joint per-stimulus error.
    """
    set_style(scale)
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 5.0))

    for a, sl, lim, bound, name, col in (
            (ax[0], slice(0, 8), Z_MAX + 0.3, Z_MAX, "sensitivities  ($z_x$, $z_y$)", BLUE),
            (ax[1], slice(8, 12), 1.0, R_MAX, r"correlations  ($\rho$)", RED_DEEP)):
        t = true[:, sl].ravel(); p = pred[:, sl].ravel()
        a.scatter(t, p, s=7, c=col, alpha=0.18, edgecolors="none", rasterized=True)
        a.plot([-lim, lim], [-lim, lim], color=INK, lw=1.8, ls=(0, (4, 3)), zorder=3)
        for b in (-bound, bound):
            a.axvline(b, color=MUTE, lw=1.0, ls=(0, (1, 3)), zorder=0)
            a.axhline(b, color=MUTE, lw=1.0, ls=(0, (1, 3)), zorder=0)
        r = np.corrcoef(t, p)[0, 1]
        a.text(0.05, 0.95, f"r = {r:.2f}", transform=a.transAxes, va="top", ha="left",
               fontsize=12 * scale, color=INK)
        a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.set_box_aspect(1)
        a.set_title(name); a.set_xlabel("true")
    ax[0].set_ylabel("estimated")

    # panel 3 — joint error per stimulus, in the perceptual plane
    for i, (nm, col) in enumerate(zip(STIM, STIM_COLORS)):
        dx = pred[:, i] - true[:, i]            # zx_i
        dy = pred[:, 4 + i] - true[:, 4 + i]    # zy_i
        d = np.column_stack([dx, dy])
        ax[2].scatter(dx, dy, s=5, color=col, alpha=0.10, edgecolors="none",
                      rasterized=True)
        _cov_ellipse(ax[2], d, col)
        ax[2].scatter([], [], s=60, color=col, label=nm)
    m = float(np.nanpercentile(np.abs(pred[:, :8] - true[:, :8]), 99.5))
    ax[2].axhline(0, color=INK, lw=1.0, ls=(0, (4, 3)))
    ax[2].axvline(0, color=INK, lw=1.0, ls=(0, (4, 3)))
    ax[2].set_xlim(-m, m); ax[2].set_ylim(-m, m); ax[2].set_box_aspect(1)
    ax[2].set_xlabel(r"error in $z_x$"); ax[2].set_ylabel(r"error in $z_y$")
    ax[2].set_title("joint error per stimulus  (2 sd)")
    ax[2].legend(fontsize=9.5 * scale, loc="upper right", handletextpad=0.3,
                 borderpad=0.3, labelspacing=0.25)

    fig.tight_layout()
    fig.savefig(path); plt.close(fig)
    print(f"  wrote {os.path.basename(path)}")


def poster_calibration(samples, true, path, scale=POSTER_SCALE):
    """Coverage-only calibration. One panel, per family, diagonal = honest.

    The manuscript figure is three panels wide and includes the SBC rank histograms, which
    are the right thing for a methods section and the wrong thing for a poster: a passer-by
    cannot evaluate flatness in three seconds, but they can see whether a line sits on a
    diagonal. Structurally-degenerate rho values (rho == 0 under PI, where the rank is not
    a calibration statistic) are excluded here for the same reason as in the full version.
    """
    set_style(scale)
    levels = np.array([0.5, 0.7, 0.8, 0.9, 0.95])
    keep = np.ones_like(true, dtype=bool)
    keep[:, 8:12] = true[:, 8:12] != 0.0

    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    for name, sl, col in ((r"sensitivities ($z$)", slice(0, 8), BLUE_DEEP),
                          (r"correlations ($\rho$)", slice(8, 12), RED_DEEP)):
        emp = []
        for l in levels:
            lo = np.quantile(samples[:, :, sl], (1 - l) / 2, axis=0)
            hi = np.quantile(samples[:, :, sl], (1 + l) / 2, axis=0)
            inside = (true[:, sl] >= lo) & (true[:, sl] <= hi)
            emp.append(inside[keep[:, sl]].mean())
        ax.plot(levels, emp, "o-", color=col, ms=9, lw=2.6, label=name)
    ax.plot([0, 1], [0, 1], color=MUTE, lw=1.8, ls=(0, (4, 3)), label="perfect calibration")
    ax.set_xlim(0.42, 1.0); ax.set_ylim(0.42, 1.0); ax.set_box_aspect(1)
    ax.set_xlabel("nominal credible level"); ax.set_ylabel("empirical coverage")
    ax.set_title("The posterior is honest")
    ax.legend(fontsize=10 * scale, loc="upper left")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    print(f"  wrote {os.path.basename(path)}")


def accuracy_crossover(model, path, trials=(10, 25, 50, 100, 200, 400), n_per_class=20,
                       restarts=20, n_multistart=8, rt_ref_mae=None, scale=POSTER_SCALE,
                       seed=5):
    """GRIN vs MLE accuracy as a function of data — the poster's uncited claim, drawn.

    Column 3 asserts that below ~100 trials per stimulus the network is MORE accurate than
    maximum likelihood, and offers no evidence for it. It is the most interesting sentence
    in that column (speed alone is a bar chart everyone expects) and it is currently the
    only quantitative claim on the poster with nothing behind it. This is that figure: the
    crossover, with the prior regularising exactly where likelihood optimisation diverges.

    Uses fit_full (the saturated fit) rather than fit_selected, purely for runtime; the
    saturated fit is the FAVOURABLE baseline here, so the crossover this shows is a
    conservative estimate of the real one.
    """
    set_style(scale)
    xs, g_mae, m_mae, r_mae, p_mae = [], [], [], [], []
    for t in trials:
        g = GRTDataGenerator(n_per_class=n_per_class, trial_range=(t, t),
                             balanced_trials=True, z_max=Z_MAX, r_max=R_MAX, seed=seed)
        X, yp, Xt, _, _ = g.generate_all_model_cms()
        gp = predict_point(model, X, Xt).numpy()
        mp = np.array([fit_full(X[i], Xt[i])["params"] for i in range(len(X))])
        ok = np.isfinite(mp).all(1)
        idx = np.flatnonzero(ok)[:n_multistart]
        rp = np.array([fit_full_multistart(X[i], Xt[i], n_restarts=restarts)["params"]
                       for i in idx])
        xs.append(t)
        g_mae.append(np.abs(gp[ok] - yp[ok]).mean())
        m_mae.append(np.abs(mp[ok] - yp[ok]).mean())
        r_mae.append(np.abs(rp - yp[idx]).mean() if len(idx) else np.nan)
        pp = np.array([fit_full_penalised(X[i], Xt[i], pseudo=PSEUDO)["params"]
                       for i in range(len(X))])
        p_mae.append(np.abs(pp[ok] - yp[ok]).mean())
        print(f"    {t:4d} trials/stim  GRIN {g_mae[-1]:.3f}   MLE {m_mae[-1]:.3f}"
              f"   ({int((~ok).sum())} MLE failures)")

    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    ax.plot(xs, m_mae, "s-", color=MUTE, lw=2.8, ms=9, label="MLE (1 start)")
    ax.plot(xs, r_mae, "^--", color=RED_DEEP, lw=2.2, ms=8,
            label=f"MLE ({restarts} starts)")
    ax.plot(xs, p_mae, "d-", color=RED, lw=2.4, ms=8, label="MLE (penalised)")
    if rt_ref_mae is not None:
        # +RT model's MAE on its own held-out set — indicative, hence a reference line
        # rather than a curve (it was not evaluated at these specific trial counts).
        ax.axhline(rt_ref_mae, color=BLUE_DEEP, lw=1.8, ls=(0, (1, 2)))
        ax.text(xs[-1], rt_ref_mae, "  +RT model (indicative)", color=BLUE_DEEP,
                va="center", ha="right", fontsize=9 * scale)
    ax.plot(xs, g_mae, "o-", color=BLUE_DEEP, lw=2.8, ms=9, label="GRIN")
    g_arr, m_arr, x_arr = map(np.asarray, (g_mae, m_mae, xs))
    better = g_arr < m_arr
    if better.any() and (~better).any():
        i = int(np.argmax(~better))
        if i > 0:
            xc = float(np.exp(np.interp(0.0, [g_arr[i - 1] - m_arr[i - 1],
                                              g_arr[i] - m_arr[i]],
                                        [np.log(x_arr[i - 1]), np.log(x_arr[i])])))
            ax.axvline(xc, color=RED_DEEP, lw=1.8, ls=(0, (4, 3)))
            ax.text(xc, ax.get_ylim()[1], f"  crossover ≈ {xc:.0f}", color=RED_DEEP,
                    ha="left", va="top", fontsize=11 * scale)
    # ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("trials per stimulus"); ax.set_ylabel("parameter MAE (log)")
    ax.set_title("Where the prior earns its keep")
    ax.legend(fontsize=11 * scale)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    print(f"  wrote {os.path.basename(path)}")


def poster_trust_dashboard(true_show, pred_show, samples_var, true_var, path,
                           scale=POSTER_SCALE):
    """Recovery (3 panels) + a coverage sparkline strip = the 'can you trust it' story.

    Folds the old poster_recovery and poster_calibration into ONE figure, mirroring the
    speed dashboard's shape (main evidence + short context strip). The two halves use the
    two sets they each need — recovery on the clean showcase set, coverage on the
    variable-trial set where calibration is a meaningful test — which is exactly the split
    the standalone versions used; the strip caption states it.

    top:   pooled z scatter | pooled rho scatter | joint per-stimulus error ellipses
    strip: interval coverage vs nominal level, z and rho, against the diagonal
    """
    from matplotlib import gridspec
    set_style(scale)
    fig = plt.figure(figsize=(15.5, 6.4))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.0, 0.5], hspace=0.52, wspace=0.30,
                           figure=fig)
    az = fig.add_subplot(gs[0, 0]); ar = fig.add_subplot(gs[0, 1]); aj = fig.add_subplot(gs[0, 2])

    for a, sl, lim, bound, name, col in (
            (az, slice(0, 8), Z_MAX + 0.3, Z_MAX, "Sensitivities  ($z_x$, $z_y$)", BLUE),
            (ar, slice(8, 12), 1.0, R_MAX, r"Correlations  ($\rho$)", RED_DEEP)):
        t = true_show[:, sl].ravel(); p = pred_show[:, sl].ravel()
        a.scatter(t, p, s=7, c=col, alpha=0.18, edgecolors="none", rasterized=True)
        a.plot([-lim, lim], [-lim, lim], color=INK, lw=1.8, ls=(0, (4, 3)), zorder=3)
        for b in (-bound, bound):
            a.axvline(b, color=MUTE, lw=1.0, ls=(0, (1, 3)), zorder=0)
            a.axhline(b, color=MUTE, lw=1.0, ls=(0, (1, 3)), zorder=0)
        r = np.corrcoef(t, p)[0, 1]
        a.text(0.05, 0.95, f"r = {r:.2f}", transform=a.transAxes, va="top", ha="left",
               fontsize=12 * scale, color=INK)
        a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.set_box_aspect(1)
        a.set_title(name); a.set_xlabel("True")
    az.set_ylabel("Estimated")

    for i, (nm, col) in enumerate(zip(STIM, STIM_COLORS)):
        dx = pred_show[:, i] - true_show[:, i]
        dy = pred_show[:, 4 + i] - true_show[:, 4 + i]
        aj.scatter(dx, dy, s=5, color=col, alpha=0.10, edgecolors="none", rasterized=True)
        _cov_ellipse(aj, np.column_stack([dx, dy]), col)
        aj.scatter([], [], s=60, color=col, label=nm)
    m = float(np.nanpercentile(np.abs(pred_show[:, :8] - true_show[:, :8]), 99.5))
    aj.axhline(0, color=INK, lw=1.0, ls=(0, (4, 3)))
    aj.axvline(0, color=INK, lw=1.0, ls=(0, (4, 3)))
    aj.set_xlim(-m, m); aj.set_ylim(-m, m); aj.set_box_aspect(1)
    aj.set_xlabel(r"Error in $z_x$"); aj.set_ylabel(r"Error in $z_y$")
    aj.set_title("Joint Error")
    # aj.legend(fontsize=9.5 * scale, loc="upper right", handletextpad=0.3,
            #   borderpad=0.3, labelspacing=0.25)

    # --- coverage strip (spans all three columns) ---
    axs = fig.add_subplot(gs[1, :])
    levels = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    keep = np.ones_like(true_var, dtype=bool)
    keep[:, 8:12] = true_var[:, 8:12] != 0.0        # drop rho==0 (rank not a calib. statistic)
    for name, sl, col in ((r"$z$", slice(0, 8), BLUE_DEEP),
                          (r"$\rho$", slice(8, 12), RED_DEEP)):
        emp = []
        for l in levels:
            lo = np.quantile(samples_var[:, :, sl], (1 - l) / 2, axis=0)
            hi = np.quantile(samples_var[:, :, sl], (1 + l) / 2, axis=0)
            inside = (true_var[:, sl] >= lo) & (true_var[:, sl] <= hi)
            emp.append(inside[keep[:, sl]].mean())
        axs.plot(levels, emp, "o-", color=col, ms=6, lw=2.4, label=name)
    axs.plot([0.5, 0.95], [0.5, 0.95], color=MUTE, lw=1.6, ls=(0, (4, 3)))#, label="perfect")
    axs.set_xlim(0.48, 0.97); axs.set_ylim(0.42, 0.99); axs.set_box_aspect(0.14)
    axs.set_xlabel("Nominal credible level"); axs.set_ylabel("Coverage")
    axs.set_title("Posterior intervals", fontsize=11 * scale)
    axs.legend(fontsize=9 * scale, ncol=3, loc="upper left", frameon=False)

    fig.savefig(path); plt.close(fig)
    print(f"  wrote {os.path.basename(path)}  [trust dashboard: recovery + coverage]")


def _is_pi_label(names):
    """Boolean 'is this a PI (independent-correlation) class' for GRIN-style class names.
    None / NaN (a failed fit) -> False, and the caller masks those out via `ok`."""
    out = np.zeros(len(names), bool)
    for i, n in enumerate(np.asarray(names, dtype=object)):
        if n is None or (isinstance(n, float) and n != n):
            continue
        out[i] = gm.MODEL_SPECS[n][0] == "pi"
    return out


def load_poster_eval(model, n_samples=800):
    """The single evaluation every poster figure draws from: GRIN, Python-MLE, and the R
    baselines (mdsdt / grtools) on ONE set — the export_for_r stratified matrices. This is
    what makes the poster internally consistent: recovery, coverage, the speed trade-off, and
    the PI frontier are all the same matrices, and the numbers match comparison_to_r.png.

    Returns (data, None) or (None, reason) if the export isn't present. `data` has:
        X, Xt, truth, trial_bin, maxrho, true_labels, N
        grin_samples                 (S, N, 12) posterior samples  -> coverage
        grin_results                 list of {p_PI,p_sep_A,p_sep_B} -> reliability panel
        params[m], labels[m], ok[m]  per method m in GRIN/Python-MLE/mdsdt/grtools
        called_pi[m]                 bool PI-call per matrix        -> frontier
        ms[...]                      wall-clock ms/matrix
    Baselines are simply absent from the dicts if baseline_fits.csv isn't there yet.
    """
    import pandas as pd
    from src.config import SIMULATED_DATA_DIR, MLE_FITS_DIR
    from src.inference.predict import predict_point, predict_posterior
    from src.inference.mle import fit_selected
    from src.inference.model_posterior import amortized_compare, model_posterior
    from src.viz.labels import to_grin_labels, labels_from_amortized

    csv = os.path.join(SIMULATED_DATA_DIR, "test_set_for_R.csv")
    if not os.path.exists(csv):
        return None, (f"{csv} not found — run:\n"
                      f"    python scripts/export_for_r.py --n 600\n"
                      f"    Rscript scripts/R/fit_baselines.R")
    df = pd.read_csv(csv)
    cm_cols = [f"cm_{s}{r}" for s in range(4) for r in range(4)]
    X = df[cm_cols].to_numpy(); Xt = df[[f"trials_{s}" for s in range(4)]].to_numpy()
    truth = df[gm.PARAM_NAMES].to_numpy(dtype=float)
    trial_bin = df["trial_bin"].to_numpy()
    true_labels = df["model_label"].to_numpy(dtype=object)
    maxrho = np.abs(truth[:, 8:12]).max(1)
    N = len(df)

    params, labels, ok, ms, called_pi = {}, {}, {}, {}, {}

    # --- GRIN: point estimate, posterior samples, constructs, class, timings ---
    post = predict_posterior(model, X, Xt, n_samples=n_samples)
    params["GRIN"] = post["mean"].numpy()
    grin_samples = post["samples"].numpy()
    grin_results = model_posterior(model, X, Xt, n_samples=600)   # p_PI / p_sep_A / p_sep_B
    labels["GRIN"] = labels_from_amortized(amortized_compare(model, X, Xt))
    ok["GRIN"] = np.ones(N, bool)
    called_pi["GRIN"] = np.array([r["p_PI"] for r in grin_results]) > 0.5

    t0 = time.time(); predict_point(model, X, Xt)
    ms["GRIN (batched)"] = 1e3 * (time.time() - t0) / N
    t0 = time.time()
    for i in range(min(50, N)):
        predict_point(model, X[i:i + 1], Xt[i:i + 1])
    ms["GRIN (1 matrix)"] = 1e3 * (time.time() - t0) / min(50, N)

    # --- Python-MLE (selected workflow: the realistic comparator) ---
    t0 = time.time()
    sel = [fit_selected(X[i], Xt[i]) for i in range(N)]
    ms["Python-MLE"] = 1e3 * (time.time() - t0) / N
    params["Python-MLE"] = np.array([f["params"] for f in sel], dtype=float)
    labels["Python-MLE"] = np.array([f["model"] for f in sel], dtype=object)
    ok["Python-MLE"] = np.isfinite(params["Python-MLE"]).all(1)
    called_pi["Python-MLE"] = _is_pi_label(labels["Python-MLE"])

    # --- R baselines from fit_baselines.R ---
    rfits = os.path.join(MLE_FITS_DIR, "baseline_fits.csv")
    if os.path.exists(rfits):
        r = pd.read_csv(rfits).set_index("row_id")
        j = df.set_index("row_id").join(r).reset_index()
        for pkg in ("mdsdt", "grtools"):
            cols = [f"{pkg}_{n}" for n in gm.PARAM_NAMES]
            if any(c not in j.columns for c in cols):
                print(f"  [eval] {pkg}: no parameter columns in baseline_fits.csv — skipped")
                continue
            params[pkg] = j[cols].to_numpy(dtype=float)
            labels[pkg] = to_grin_labels(j[f"{pkg}_model"].to_numpy(dtype=object))
            ok[pkg] = (j[f"{pkg}_ok"].fillna(False).to_numpy(dtype=bool)
                       & np.isfinite(params[pkg]).all(1))
            ms[pkg] = 1e3 * float(np.nanmean(j[f"{pkg}_secs"].to_numpy(dtype=float)))
            called_pi[pkg] = _is_pi_label(labels[pkg])
    else:
        print(f"  [eval] {rfits} not found — mdsdt/grtools omitted. "
              f"Run: Rscript scripts/R/fit_baselines.R")

    return dict(X=X, Xt=Xt, truth=truth, trial_bin=trial_bin, maxrho=maxrho,
                true_labels=true_labels, N=N, grin_samples=grin_samples,
                grin_results=grin_results, params=params, labels=labels, ok=ok,
                ms=ms, called_pi=called_pi), None


def poster_speed_dashboard(data, path, scale=POSTER_SCALE):
    """Speed-vs-accuracy trade-off (incl. mdsdt & grtools) + an accuracy-vs-sample-size
    sparkline. Consumes the shared load_poster_eval() dict, so every point is on the same
    matrices as recovery, coverage, and the PI frontier, and matches comparison_to_r.png.
    Accuracy is scored on the common-converged subset (the fair comparator); the convergence
    gap is stated as a one-line note rather than given its own panel.
    """
    from matplotlib import gridspec
    set_style(scale)
    fig = plt.figure(figsize=(9.6, 5.6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.0, 0.5], hspace=0.5, figure=fig)
    axm = fig.add_subplot(gs[0]); axs = fig.add_subplot(gs[1])

    params, ok, ms, truth = data["params"], data["ok"], data["ms"], data["truth"]
    trial_bin, N = data["trial_bin"], data["N"]
    common = np.ones(N, bool)
    for m in params:
        common &= ok[m]

    # one point per method: (label, ms/matrix, 1-MAE, se, family colour). Axes are the
    # natural way round: time increases left->right (so FAST is left), and accuracy is
    # 1-MAE increasing bottom->top (so GOOD is up). The desirable corner is therefore
    # TOP-LEFT, which is where the "better" arrow points and where GRIN lands.
    fam_col = {"GRIN": BLUE_DEEP, "Python-MLE": MUTE, "mdsdt": BLUE, "grtools": RED_DEEP}
    order = [("GRIN", "GRIN (1 matrix)", "GRIN"),
             ("Python-MLE", "Python-MLE", "Python-MLE"),
             ("mdsdt", "mdsdt", "mdsdt"), ("grtools", "grtools", "grtools")]
    label_of = {"GRIN": "GRIN", "Python-MLE": "MLE",
                "mdsdt": "mdsdt (R)", "grtools": "grtools (R)"}
    xs_all = []
    for key, ms_key, fam in order:
        if key not in params:
            continue
        e = np.abs(params[key][common] - truth[common]).mean(1)
        acc = 1.0 - float(e.mean())
        se = float(e.std(ddof=1) / np.sqrt(len(e)))
        x = ms.get(ms_key, ms.get(key))
        xs_all.append(x)
        axm.errorbar(x, acc, yerr=se, fmt="o", ms=13, color=fam_col[fam],
                     ecolor=fam_col[fam], elinewidth=1.4, capsize=3, zorder=3)
        # GRIN sits in the top-left corner where the "better" arrow lives, so drop its
        # label below-right to avoid a collision; everyone else labels up-right.
        dx, dy = (12, -16) if key == "GRIN" else (10, 8)
        axm.annotate(label_of[key], (x, acc), textcoords="offset points", xytext=(-abs(dx), dy),
                     fontsize=9.5 * scale, color=INK,
                     ha="left" if key == "GRIN" else "right", va="top" if key == "GRIN" else "bottom")
    axm.set_xscale("log")                           # log time; faster (small) is on the left
    axm.set_ylim(0.0, 1.0)                           # 1 - MAE: 0 = bad (bottom), 1 = perfect (top)
    axm.set_xlabel("Time (ms/matrix)")
    axm.set_ylabel(r"1 $-$ MAE")
    axm.set_title("Speed vs Accuracy")
    # "better" indicator tucked into the top-left corner (fast + accurate)
    axm.annotate("", xy=(0.015, 0.98), xytext=(0.12, 0.88), xycoords="axes fraction",
                 arrowprops=dict(arrowstyle="->", color=MUTE, lw=1.8))
    axm.text(0.135, 0.88, "better", transform=axm.transAxes, color=MUTE,
             fontsize=12 * scale, fontweight="bold", ha="left", va="top")
    # convergence, compressed to one honest line
    fails = [(k, 100 * (1 - ok[k].mean())) for k in params if k != "GRIN"]
    if fails:
        note = "; ".join(f"{k} fails {f:.0f}%" for k, f in fails if f >= 0.5)
        if note:
            axm.text(0.9, 0.01, note + "  ·  GRIN never fails", transform=axm.transAxes,
                     ha="right", va="bottom", fontsize=8.5 * scale, color=MUTE)

    # --- sample-size curve: 1-MAE vs trials/stimulus, EVERY method, SAME fixed bins ---
    # Real numbers on the x-axis (not low/mid/high), log-spaced, dense where GRIN's low-N
    # advantage lives. Each method gets a shaded 95% band; where a baseline barely fits, the
    # band balloons and the n-label reads small -- the honest picture, no hidden gaps. GRIN
    # also gets a smooth line over ALL its matrices, since it never fails.
    TPS_EDGES = np.array([5, 10, 15, 20, 30, 50, 75, 100, 200, 500], dtype=float)
    tps = data["Xt"].sum(1) / 4.0                            # trials per stimulus, per matrix
    centres = np.sqrt(TPS_EDGES[:-1] * TPS_EDGES[1:])        # geometric bin centres

    def _band(mask_method):
        """per-bin mean 1-MAE with a bootstrap 95% band and n, over matrices in mask_method."""
        xs, ys, los, his, ns = [], [], [], [], []
        for lo_e, hi_e, xc in zip(TPS_EDGES[:-1], TPS_EDGES[1:], centres):
            m = mask_method & (tps >= lo_e) & (tps < hi_e)
            n = int(m.sum())
            if n < 8:
                continue
            per = 1.0 - np.abs(params[key][m] - truth[m]).mean(1)
            boot = [np.random.default_rng(s).choice(per, len(per)).mean() for s in range(400)]
            xs.append(xc); ys.append(per.mean())
            los.append(np.percentile(boot, 2.5)); his.append(np.percentile(boot, 97.5))
            ns.append(n)
        return map(np.array, (xs, ys, los, his, ns))

    spark_order = [("GRIN", BLUE_DEEP, "o", "GRIN"), ("Python-MLE", MUTE, "s", "MLE"),
                   ("mdsdt", BLUE, "^", "mdsdt"), ("grtools", RED_DEEP, "v", "grtools")]
    for key, col, mk, lab in spark_order:
        if key not in params:
            continue
        x, y, lo, hi, ns = _band(common & ok[key])
        if not len(x):
            continue
        axs.fill_between(x, lo, hi, color=col, alpha=0.13, linewidth=0)
        axs.plot(x, y, mk + "-", color=col, lw=2.0, ms=5)
        axs.annotate(lab, (x[0], y[0]), xytext=(-8, -6 if key == "mdsdt" else 0),
                     textcoords="offset points", fontsize=8 * scale, color=col,
                     fontweight="bold", ha="right", va="center")

    # # GRIN smooth trend over all matrices (rolling mean on sorted trials/stimulus)
    # gm_mask = common & ok["GRIN"]
    # o = np.argsort(tps[gm_mask])
    # xt = tps[gm_mask][o]
    # yt = (1.0 - np.abs(params["GRIN"][gm_mask] - truth[gm_mask]).mean(1))[o]
    # if len(xt) > 40:
    #     w = max(15, len(xt) // 25)
    #     sm = np.convolve(yt, np.ones(w) / w, mode="valid")
    #     axs.plot(xt[w // 2: w // 2 + len(sm)], sm, color=BLUE_DEEP, lw=1.2, alpha=0.5)

    # # n-per-bin as a faint sub-label along the bottom
    # for xc, lo_e, hi_e in zip(centres, TPS_EDGES[:-1], TPS_EDGES[1:]):
    #     n = int((common & (tps >= lo_e) & (tps < hi_e)).sum())
    #     axs.annotate(f"{n}", (xc, 0), xytext=(0, 2), textcoords="offset points",
    #                  ha="center", va="bottom", fontsize=6.5 * scale, color=MUTE,
    #                  annotation_clip=False)

    axs.set_xscale("log")
    axs.set_xticks(TPS_EDGES); axs.set_xticklabels([str(int(e)) for e in TPS_EDGES])
    axs.set_yticks([]); axs.set_yticklabels([])
    axs.minorticks_off()
    axs.set_box_aspect(0.16)
    axs.set_xlabel(r"Trials per stimulus")# (; faint $n$ = matrices per bin)")
    axs.set_ylabel(r"1 $-$ MAE")

    fig.savefig(path); plt.close(fig)
    print(f"  wrote {os.path.basename(path)}  [speed dashboard: trade-off + sample-size curve]")


def _construct_acc_on_R(data, key):
    """(correlation-structure accuracy, separability accuracy) for one method on the shared
    R set, from its class labels vs truth — used as baseline reference marks on the dumbbell.
    Robust to None labels (a failed fit): those rows are dropped before scoring.
    """
    from src.inference.model_posterior import construct_labels
    ok = np.asarray(data["ok"][key], bool)
    tl = np.asarray(data["true_labels"], dtype=object)[ok]
    pl = np.asarray(data["labels"][key], dtype=object)[ok]
    good = np.array([p is not None and (not isinstance(p, float) or p == p) for p in pl], bool)
    tl, pl = tl[good], pl[good]
    if not len(pl):
        return float("nan"), float("nan")
    tc, ta, tb = construct_labels(tl)
    pc, pa, pb = construct_labels(pl)
    corr = float(np.mean(pc == tc))
    sep = float(np.mean((pa == ta) & (pb == tb)))
    return corr, sep


def poster_rt_dumbbell(data, path):
    """Assemble the counts-vs-+RT dumbbell from results/rt_construct_metrics.json (written by
    make_figures_rt.py) plus the baselines' construct accuracy on the shared R set. If the
    RT metrics aren't present, prints what to run and skips — the poster shows a placeholder.
    """
    import json
    from src.config import RESULTS_DIR
    jpath = os.path.join(RESULTS_DIR, "rt_construct_metrics.json")
    if not os.path.exists(jpath):
        set_style(POSTER_SCALE)
        fig, ax = plt.subplots(figsize=(8.6, 3.2)); ax.axis("off")
        ax.text(0.5, 0.5, "RT dumbbell needs rt_construct_metrics.json\n"
                "run:  python scripts/make_figures_rt.py",
                ha="center", va="center", fontsize=11, color=MUTE)
        fig.savefig(path); plt.close(fig)
        print(f"  [rt] {jpath} not found — run make_figures_rt.py")
        return
    M = json.load(open(jpath))
    # baseline construct accuracy on the R set (reference diamonds)
    base_corr, base_sep = {}, {}
    for key, nm in (("Python-MLE", "MLE"), ("mdsdt", "mdsdt"), ("grtools", "grtools")):
        if key in data["labels"]:
            c, s = _construct_acc_on_R(data, key)
            base_corr[nm] = c; base_sep[nm] = s
    ci = lambda d: (d.get("lo"), d.get("hi")) if isinstance(d, dict) and "lo" in d else None
    rows = [
        {"metric": "correlation structure", "counts": M["corr"]["cm"], "rt": M["corr"]["rt"],
         "counts_ci": ci(M["corr"].get("cm_ci")), "rt_ci": ci(M["corr"].get("rt_ci")),
         "baselines": base_corr or None},
        {"metric": "separability", "counts": M["sep"]["cm"], "rt": M["sep"]["rt"],
         "baselines": base_sep or None},
        {"metric": "processing architecture", "counts": M["arch"]["chance"],
         "rt": M["arch"]["rt"], "chance": M["arch"]["chance"]},
        {"metric": "dimension neglect", "counts": M["dimneglect"]["cm"],
         "rt": M["dimneglect"]["rt"], "chance": 0.5},
        {"metric": "perceptual independence", "counts": M["pi"]["cm"], "rt": M["pi"]["rt"],
         "counts_ci": ci(M["pi"].get("cm_ci")), "rt_ci": ci(M["pi"].get("rt_ci")),
         "baselines": base_corr or None},
    ]
    F.rt_vs_counts_dumbbell([r for r in rows if r["counts"] is not None
                             and r["rt"] is not None], path, scale=POSTER_SCALE
    )
                            # note="diamonds: counts-only baselines on the R comparison set")
    standardize(path, rows=2)
    print(f"  wrote {os.path.basename(path)}  [RT dumbbell]")


def main(n_samples=800, frontier_step=0.10):
    """Every poster figure is rendered from ONE shared evaluation (load_poster_eval): the
    export_for_r stratified matrices, where GRIN, Python-MLE, and the R baselines all live.
    Recovery, coverage, the speed trade-off, and the PI frontier are therefore the same
    matrices end to end, and the numbers line up with results/figures/comparison_to_r.png.

    If the export isn't present the script still produces the GRIN-only figures from a
    freshly generated set and tells you what to run for the baseline comparisons.
    """
    device = DEVICE if torch.cuda.is_available() else "cpu"
    model = load_model(MODEL_FILE, device=device)
    print(f"model loaded on {device}; writing to {OUT_DIR}")

    data, reason = load_poster_eval(model, n_samples=n_samples)

    if data is None:
        # ---- fallback: no R export yet. GRIN-only trust + constructs; speed is skipped. ----
        print("  [eval] " + reason)
        print("  building GRIN-only figures from a fresh set; run the export for the rest.")
        vg = GRTDataGenerator(n_per_class=800, z_max=Z_MAX, r_max=R_MAX, seed=99)
        Xv, ypv, Xtv, ycv, ylv = vg.generate_all_model_cms()
        pv = predict_posterior(model, Xv, Xtv, n_samples=n_samples)
        poster_trust_dashboard(ypv, pv["mean"].numpy(), pv["samples"].numpy(), ypv,
                               out("poster_recovery.png"))
        standardize(out("poster_recovery.png"), rows=2)
        res = model_posterior(model, Xv, Xtv, n_samples=600)
        F.construct_probabilities(res, ylv, np.abs(ypv[:, 8:12]).max(1),
                                  out("constructs.png"), scale=POSTER_SCALE,
                                  suptitle=False, legend_loc="upper right",
                                  chance_side="left", wspace=0.06, width=13.5)
        standardize(out("constructs.png"), rows=2)
        print("  wrote poster_recovery.png + constructs.png (GRIN only); "
              "speed_accuracy.png NOT written — run the export + fit_baselines.R")
        return

    truth = data["truth"]
    pred = data["params"]["GRIN"]
    samples = data["grin_samples"]

    # TRUST dashboard (col 2): recovery + coverage, all on the shared set.
    poster_trust_dashboard(truth, pred, samples, truth, out("poster_recovery.png"))
    standardize(out("poster_recovery.png"), rows=2)

    # CONSTRUCTS (col 4): GRIN reliability + a PI frontier that now overlays every method
    # that makes a PI/non-PI call on these same matrices (MLE, mdsdt, grtools).
    extra = []
    for key, colour, marker, lab in (("Python-MLE", MUTE, "s", "MLE"),
                                     ("mdsdt", BLUE, "^", "mdsdt"),
                                     ("grtools", RED_DEEP, "v", "grtools")):
        if key in data["called_pi"]:
            extra.append({"label": lab, "called_pi": data["called_pi"][key],
                          "valid": data["ok"][key], "color": colour, "marker": marker})
    F.construct_probabilities(data["grin_results"], data["true_labels"], data["maxrho"],
                              out("constructs.png"), scale=POSTER_SCALE,
                              suptitle=False, legend_loc="upper right", chance_side="left",
                              wspace=0.06, width=13.5, frontier_step=frontier_step,
                              extra_frontier=extra)
    standardize(out("constructs.png"), rows=2)
    print("  wrote constructs.png"
          + (f"  [frontier vs {', '.join(e['label'] for e in extra)}]" if extra else ""))

    # SPEED dashboard (col 3): trade-off + sample-size spark, same shared set.
    poster_speed_dashboard(data, out("speed_accuracy.png"))
    standardize(out("speed_accuracy.png"), rows=3)

    # RT-vs-counts dumbbell (col 5): what response times add, with the baselines as
    # reference marks. Reads make_figures_rt.py's JSON; wrapped so a late hiccup here
    # cannot discard the recovery/constructs/speed figures already written above.
    try:
        poster_rt_dumbbell(data, out("rt_dumbbell.png"))
    except Exception as e:
        print(f"  [rt] dumbbell skipped ({type(e).__name__}: {e}); other figures are saved")

    # --- model identification, the concrete head-to-head backbone ---
    common = np.ones(data["N"], bool)
    for m in data["ok"]:
        common &= data["ok"][m]
    print("\n=== MODEL IDENTIFICATION (common-converged subset) ===")
    for m in data["labels"]:
        lab = np.asarray(data["labels"][m], dtype=object)
        full = float(np.mean(lab[common] == data["true_labels"][common]))
        cp = data["called_pi"][m]; ispi = data["maxrho"] == 0
        hit = float(np.mean(cp[data["ok"][m] & ispi])) if (data["ok"][m] & ispi).any() else float("nan")
        fa = float(np.mean(cp[data["ok"][m] & ~ispi])) if (data["ok"][m] & ~ispi).any() else float("nan")
        print(f"  {m:12s}  12-way {full:.3f}   PI hit {hit:.2f}   PI false-alarm {fa:.2f}")

    # --- the numbers the poster quotes -------------------------------------------
    z_r = np.corrcoef(truth[:, :8].ravel(), pred[:, :8].ravel())[0, 1]
    rho_r = np.corrcoef(truth[:, 8:].ravel(), pred[:, 8:].ravel())[0, 1]
    common = np.ones(data["N"], bool)
    for m in data["ok"]:
        common &= data["ok"][m]
    print("\n" + "=" * 72)
    print("CONFIRM THESE AGAINST THE \\chk{} VALUES IN poster.tex")
    print("=" * 72)
    print(f"  evaluation set           test_set_for_R.csv, N={data['N']} "
          f"({int(common.sum())} scored by every method)")
    print(f"  z-score recovery         r = {z_r:.3f}")
    print(f"  correlation recovery     r = {rho_r:.3f}")
    for k in ("GRIN (1 matrix)", "Python-MLE", "mdsdt", "grtools"):
        if k in data["ms"]:
            print(f"  time {k:18s} {data['ms'][k]:10.3f} ms/matrix")
    for k in [m for m in data["params"]]:
        e = np.abs(data["params"][k][common] - truth[common]).mean()
        print(f"  MAE  {k:18s} {e:8.4f}   (1-MAE = {1-e:.4f})")
    print("=" * 72)
    print("\nStill to confirm by hand (adaptive + RT runs, not this script): the 55%")
    print("stopping saving, the 46% training saving, architecture 0.88, dimension")
    print("neglect 0.97, PI 0.60 -> 0.65.")



if __name__ == "__main__":
    if not os.path.exists("src"):
        sys.exit("Run this from the PROJECT ROOT: "
                 "python presentations/sbi_workshop/make_poster_figures.py")
    main()
