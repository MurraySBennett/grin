"""Render the full GRIN counts-only figure suite from a trained model. From the project root:
    python scripts/make_figures.py

Writes to results/figures/:
    recovery.png            12-panel estimated-vs-true, hue by the construct at stake
    error_map.png           recovery MAE + signed bias, by parameter x model class
    confusion.png           12-way model identification (complexity-ordered)
    confusion_constructs.png the three per-construct confusions the 12-way matrix averages away
    calibration.png         SBC rank histogram + interval coverage
    uncertainty.png         posterior width vs trial count
    constructs.png          construct-probability calibration + the PI frontier
    speed_accuracy.png      GRIN vs both MLE workflows

Directory layout across the whole project:
    results/figures/            this suite (counts-only core)
    results/figures/generation/ prior-coverage panels (scripts/generate_data.py --report)
    results/figures/recovery/   method comparison vs mdsdt / grtools (make_recovery_figures.py)
    results/figures/rt/         the RT suite (make_figures_rt.py)

TWO CHANGES WORTH KNOWING ABOUT
  1. Recovery now goes through viz.recovery.recovery_panels, not the old
     figures.parameter_recovery: fixed +/-3.3 and +/-1 axes, hue by the model assumption at
     stake in each panel, per-group statistics, and a plotted subsample rather than all
     6,000 points. The showcase regime is stated in the figure itself.
  2. Model class comes from the amortized comparison head composed into 12 classes, NOT
     from inference.model_selection.infer_class -- that heuristic is over-parsimonious
     (~34%) and never selects the free-correlation models, so its confusion matrix has
     structurally empty columns. Using the head also keeps this figure consistent with the
     shapes in results/figures/recovery/, which are computed the same way.
"""
import os
import time

import numpy as np
import torch

from src.config import MODEL_FILE, FIGURES_DIR, DEVICE, Z_MAX, R_MAX
from src.data.generator import GRTDataGenerator
from src.inference.predict import predict_posterior, predict_point
from src.inference.mle import fit_full, fit_selected
from src.inference.model_posterior import amortized_compare, model_posterior
from src.viz import figures as F
from src.viz import recovery as R
from src.viz.labels import labels_from_amortized

SHOWCASE_REGIME = "showcase set — 200 trials/stimulus, balanced"
PER_CLASS_PLOT = 50          # points per model class actually drawn (all are still generated)


def _plot_subset(labels, per_class=PER_CLASS_PLOT, seed=0):
    """Stratified index subsample so every model class contributes equally to the scatter."""
    rng = np.random.default_rng(seed)
    keep = [rng.choice(np.flatnonzero(labels == c),
                       min(per_class, int((labels == c).sum())), replace=False)
            for c in np.unique(labels)]
    return np.concatenate(keep)


def main():
    device = DEVICE if torch.cuda.is_available() else "cpu"
    from src.api import load_model
    model = load_model(MODEL_FILE, device=device)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig = lambda name: os.path.join(FIGURES_DIR, name)

    # ---- Showcase set: balanced, well-sampled (clean recovery + identification) ----
    sg = GRTDataGenerator(n_per_class=500, trial_range=(200, 200),
                          balanced_trials=True, z_max=Z_MAX, r_max=R_MAX, seed=7)
    Xs, yps, Xts, ycs, yls = sg.generate_all_model_cms()
    yls = np.asarray(yls)
    post = predict_posterior(model, Xs, Xts, n_samples=800)
    pred = post["mean"].numpy()

    keep = _plot_subset(yls)
    R.recovery_panels(yps[keep], pred[keep], fig("recovery.png"),
                      class_names=yls[keep], correct=None, method="GRIN",
                      regime=f"{SHOWCASE_REGIME} (n={len(keep)} of {len(yls)} plotted)",
                      z_max=Z_MAX, r_max=R_MAX)
    F.recovery_error_map(yps, pred, yls, fig("error_map.png"), regime=SHOWCASE_REGIME)

    # GRIN's 12-way class: one forward pass through the comparison head, composed.
    picks = labels_from_amortized(amortized_compare(model, Xs, Xts))
    F.model_confusion(list(yls), list(picks), fig("confusion.png"), regime=SHOWCASE_REGIME)
    F.construct_confusions(list(yls), list(picks), fig("confusion_constructs.png"),
                           regime=SHOWCASE_REGIME)

    # ---- Variable-trial set: calibration + uncertainty scaling ----
    vg = GRTDataGenerator(n_per_class=800, z_max=Z_MAX, r_max=R_MAX, seed=99)
    Xv, ypv, Xtv, ycv, ylv = vg.generate_all_model_cms()
    pv = predict_posterior(model, Xv, Xtv, n_samples=800)
    F.calibration(pv["samples"].numpy(), ypv, fig("calibration.png"))
    F.uncertainty_vs_trials(Xtv.sum(1), pv["std"].numpy(), fig("uncertainty.png"),
                            regime="variable-trial set (TRIAL_RANGE, log-uniform)")
    res = model_posterior(model, Xv, Xtv, n_samples=600)
    F.construct_probabilities(res, ylv, np.abs(ypv[:, 8:12]).max(1), fig("constructs.png"))

    # ---- Speed / accuracy headline vs MLE ----
    # Two MLE baselines, same reasoning as scripts/compare_to_r.py: MLE-full is the
    # saturated 12-parameter fit (fast, but not what practitioners run); MLE-selected fits
    # every model class and keeps the AIC/BIC winner -- the realistic workflow, and the
    # fair comparison. GRIN is timed batched AND per-matrix, because batched throughput
    # against serial fitting is not a like-for-like latency claim.
    sub = np.random.default_rng(0).choice(Xs.shape[0], 150, replace=False)
    t0 = time.time()
    mle = np.array([fit_full(Xs[i], Xts[i])["params"] for i in sub])
    mle_ms = 1e3 * (time.time() - t0) / len(sub)
    t0 = time.time()
    sel = np.array([fit_selected(Xs[i], Xts[i])["params"] for i in sub])
    sel_ms = 1e3 * (time.time() - t0) / len(sub)
    t0 = time.time()
    predict_point(model, Xs[sub], Xts[sub])
    npe_ms = 1e3 * (time.time() - t0) / len(sub)
    t0 = time.time()
    for i in sub[:50]:
        predict_point(model, Xs[i:i + 1], Xts[i:i + 1])
    npe_single_ms = 1e3 * (time.time() - t0) / 50

    F.speed_accuracy_multi(
        ["GRIN (batched)", "GRIN (1 matrix)", "MLE (full)", "MLE (selected)"],
        [npe_ms, npe_single_ms, mle_ms, sel_ms],
        [np.abs(pred[sub] - yps[sub]).mean(),
         np.abs(pred[sub] - yps[sub]).mean(),
         np.abs(mle - yps[sub]).mean(),
         np.abs(sel - yps[sub]).mean()],
        fig("speed_accuracy.png"),
        title_speed=f"Speed — {sel_ms / npe_ms:,.0f}× faster than the realistic MLE workflow",
    )
    print(f"figures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
