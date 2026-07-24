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

    WHY THIS SHAPE. The manuscript figure is twelve panels because twelve parameters are
    estimated. A poster column cannot carry twelve panels, and showing four arbitrary ones
    (the previous approach) quotes a per-panel r that does not match the pooled r the poster
    text actually cites.

    Panels 1 and 2 pool by family, so the r printed on each panel IS the number in the
    column text — z_x/z_y together, then all four rho. That removes a quiet mismatch and
    costs nothing, because the poster's claim is about families, not individual parameters.

    Panel 3 answers the pairing question. zx_i and zy_i are not twelve unrelated numbers:
    they are the (x, y) coordinates of stimulus i in the perceptual space that column 1
    promises. Plotting each stimulus's JOINT error (pred - true in both coordinates at once)
    puts four 2-D clouds where eight 1-D panels used to be, and shows something no marginal
    panel can: whether the two coordinate errors are CORRELATED within a stimulus. A tilted
    ellipse means the network trades x-error against y-error for that stimulus — a
    structured failure, not noise. A round one means the errors are independent. Either way
    it is information the twelve-panel figure literally cannot display, so this is
    compaction that adds rather than merely subtracts.
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
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("trials per stimulus (log)"); ax.set_ylabel("parameter MAE (log)")
    ax.set_title("Where the prior earns its keep")
    ax.legend(fontsize=11 * scale)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    print(f"  wrote {os.path.basename(path)}")


def main(n_mle=100, n_mle_multistart=10, restarts=20, crossover=True):
    """n_mle_multistart is deliberately small: the multi-start selected workflow is
    12 classes x `restarts` optimisations per matrix, i.e. ~240 fits. Timings are per
    matrix so the two sample sizes remain comparable."""
    device = DEVICE if torch.cuda.is_available() else "cpu"
    model = load_model(MODEL_FILE, device=device)
    print(f"model loaded on {device}; writing to {OUT_DIR}")

    # --- showcase set: same recipe as make_figures.py, so poster and manuscript
    # describe the same evaluation ------------------------------------------------
    sg = GRTDataGenerator(n_per_class=500, trial_range=(200, 200),
                          balanced_trials=True, z_max=Z_MAX, r_max=R_MAX, seed=7)
    Xs, yps, Xts, ycs, yls = sg.generate_all_model_cms()
    post = predict_posterior(model, Xs, Xts, n_samples=800)
    pred = post["mean"].numpy()

    poster_recovery(yps, pred, out("poster_recovery.png"))

    # --- variable-trial set: calibration + construct probabilities ----------------
    vg = GRTDataGenerator(n_per_class=800, z_max=Z_MAX, r_max=R_MAX, seed=99)
    Xv, ypv, Xtv, ycv, ylv = vg.generate_all_model_cms()
    pv = predict_posterior(model, Xv, Xtv, n_samples=800)

    poster_calibration(pv["samples"].numpy(), ypv, out("calibration.png"))

    res = model_posterior(model, Xv, Xtv, n_samples=600)
    F.construct_probabilities(res, ylv, np.abs(ypv[:, 8:12]).max(1),
                              out("constructs.png"), scale=POSTER_SCALE)
    print("  wrote constructs.png")

    # --- speed / accuracy ---------------------------------------------------------
    # FOUR MLE variants, so the comparison cannot be accused of beating a straw man:
    # the saturated fit and the AIC/BIC-selected workflow, each from a single warm start
    # (what mdsdt does) and each from `restarts` jittered starts keeping the best
    # likelihood (what a careful analyst does). Multi-start costs `restarts` times as much
    # wall clock, and that cost belongs IN the speed comparison rather than being excluded
    # from it. See scripts/check_mle_health.py for why the extra starts do not change the
    # accuracy much: the likelihood is flat along a ridge, so every start reaches the same
    # likelihood at a different place, and there is no better optimum to find.
    rng = np.random.default_rng(0)
    sub = rng.choice(Xs.shape[0], n_mle, replace=False)
    sub_ms = sub[:min(n_mle_multistart, len(sub))]

    def timed(fn, idx):
        t0 = time.time()
        p = np.array([fn(Xs[i], Xts[i])["params"] for i in idx])
        return p, 1e3 * (time.time() - t0) / len(idx)

    mle, mle_ms = timed(fit_full, sub)
    sel, sel_ms = timed(fit_selected, sub)
    pen, pen_ms = timed(lambda c, t: fit_full_penalised(c, t, pseudo=PSEUDO), sub)
    psel, psel_ms = timed(lambda c, t: fit_selected_penalised(c, t, pseudo=PSEUDO), sub)
    print(f"  multi-start MLE on {len(sub_ms)} matrices "
          f"({restarts} restarts; the slow part) ...", flush=True)
    mle_r, mle_r_ms = timed(lambda c, t: fit_full_multistart(c, t, n_restarts=restarts), sub_ms)
    sel_r, sel_r_ms = timed(lambda c, t: fit_selected_multistart(c, t, n_restarts=restarts),
                            sub_ms)

    t0 = time.time()
    predict_point(model, Xs[sub], Xts[sub])
    npe_ms = 1e3 * (time.time() - t0) / len(sub)

    n_single = min(50, len(sub))
    t0 = time.time()
    for i in sub[:n_single]:
        predict_point(model, Xs[i:i + 1], Xts[i:i + 1])
    npe_single_ms = 1e3 * (time.time() - t0) / n_single

    # Single speed-accuracy trade-off scatter (replaces the two-panel bar figure): one
    # point per method, time on x, error on y, standard errors on both. The multi-start
    # variants are dropped from the poster point set — check_mle_health.py shows they do
    # not change accuracy (the likelihood is flat, not mis-optimised), so they would only
    # add near-duplicate points. They remain available for the appendix.
    def pt(p_hat, idx):
        e = np.abs(p_hat - yps[idx]).mean(1)      # per-matrix MAE
        return e.mean(), e.std(ddof=1) / np.sqrt(len(e))

    grin_mae, grin_se = pt(pred[sub], sub)
    labels = ["GRIN (1 matrix)", "MLE (full)", "MLE (full, penalised)",
              "MLE (selected)", "MLE (selected, penalised)"]
    times  = [npe_single_ms, mle_ms, pen_ms, sel_ms, psel_ms]
    t_err  = [0.0, 0.0, 0.0, 0.0, 0.0]            # timings are point estimates here
    maes, m_err = [grin_mae], [grin_se]
    for p_hat, idx in [(mle, sub), (pen, sub), (sel, sub), (psel, sub)]:
        mu, se = pt(p_hat, idx); maes.append(mu); m_err.append(se)

    # OPTIONAL +RT overlay: if the RT suite has been run, results/rt_metrics.json exists
    # and we add the +RT model as one extra point. This lets the poster gesture at the RT
    # work as a future direction without any counts-model figure depending on it. The RT
    # point uses ITS OWN held-out MAE from the JSON (a different eval set from the poster's
    # showcase set), so it is drawn hollow-labelled and described as indicative, not a
    # like-for-like bar — see POSTER_NOTES.
    rt_json = os.path.join(RESULTS_DIR, "rt_metrics.json")
    families = None
    if os.path.exists(rt_json):
        rt = json.load(open(rt_json))
        labels.append("+RT model (1 matrix)")
        times.append(rt["rt_model"]["single_ms"])
        t_err.append(0.0)
        maes.append(rt["rt_model"]["mae"]); m_err.append(rt["rt_model"].get("mae_se", 0.0))
        families = ([l.split(" (")[0] for l in labels[:-1]] + ["+RT model"])
        print(f"  +RT overlay: {rt['rt_model']['single_ms']:.3f} ms, "
              f"MAE {rt['rt_model']['mae']:.3f} (from {rt_json})")

    F.speed_accuracy_tradeoff(
        labels, times, t_err, maes, m_err, out("speed_accuracy.png"), families=families,
        title=f"Speed vs accuracy  ({psel_ms / npe_single_ms:,.0f}× faster, per matrix)",
        scale=POSTER_SCALE)
    print("  wrote speed_accuracy.png"
          + ("  [with +RT overlay]" if os.path.exists(rt_json) else ""))

    if crossover:
        print("  building accuracy_crossover.png (this is the slow one) ...")
        rt_ref = None
        rj = os.path.join(RESULTS_DIR, "rt_metrics.json")
        if os.path.exists(rj):
            rt_ref = json.load(open(rj))["rt_model"]["mae"]
        accuracy_crossover(model, out("accuracy_crossover.png"), rt_ref_mae=rt_ref)

    # --- the numbers the poster quotes -------------------------------------------
    z_r = np.corrcoef(yps[:, :8].ravel(), pred[:, :8].ravel())[0, 1]
    rho_r = np.corrcoef(yps[:, 8:].ravel(), pred[:, 8:].ravel())[0, 1]
    print("\n" + "=" * 72)
    print("CONFIRM THESE AGAINST THE \\chk{} VALUES IN poster.tex")
    print("=" * 72)
    print(f"  z-score recovery         r = {z_r:.3f}")
    print(f"  correlation recovery     r = {rho_r:.3f}")
    print(f"  GRIN, batched            {npe_ms * 1e3:9.1f} us / matrix   (throughput)")
    print(f"  GRIN, single matrix      {npe_single_ms * 1e3:9.1f} us / matrix   (LATENCY —")
    print(f"                             this is the number that supports the")
    print(f"                             'fits inside an inter-stimulus interval' claim)")
    print(f"  MLE (full)               {mle_ms:9.2f} ms / matrix")
    print(f"  MLE (selected)           {sel_ms:9.2f} ms / matrix")
    print(f"  speed-up vs selected     {sel_ms / npe_ms:,.0f}x batched, "
          f"{sel_ms / npe_single_ms:,.0f}x single-matrix")
    print("=" * 72)
    print("\nStill to confirm by hand (they come from the adaptive + RT runs, not from")
    print("this script): the 55% stopping saving, the 46% training saving,")
    print("architecture 0.88, dimension neglect 0.97, PI 0.60 -> 0.65.")


if __name__ == "__main__":
    if not os.path.exists("src"):
        sys.exit("Run this from the PROJECT ROOT: "
                 "python presentations/sbi_workshop/make_poster_figures.py")
    main()
    