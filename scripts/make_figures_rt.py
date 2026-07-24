"""Render the RT-augmented figure suite. Run from the project root:
    python scripts/make_figures_rt.py

Writes to results/figures/rt/ (all RT output lives in its own directory):

  Parity with the counts-only suite (make_figures.py), same evaluation style, RT model:
    rt_recovery.png       — 12-panel predicted-vs-true scatter
    rt_error_map.png      — recovery MAE by parameter x generating model class
    rt_confusion.png      — 12-way model-class identification
    rt_calibration.png    — SBC rank histogram + coverage
    rt_uncertainty.png    — posterior width vs trial count
    rt_constructs.png     — construct-probability calibration + PI frontier (counts parity)
    rt_speed_accuracy.png — speed-accuracy trade-off scatter (+RT vs counts GRIN vs MLE)

  Also writes results/rt_metrics.json — the single source of +RT-model timing and accuracy,
  read by scripts/compare_to_r.py and the poster script so those never re-time the RT model.

  Counts-only baseline vs +RT, evaluated on the SAME held-out matrices (paired):
    cm_baseline_error_map_matched.png — the baseline's own error map, matched condition
    rt_vs_counts_error_gain.png       — diverging heatmap: where RT helps, by class x parameter
    rt_vs_counts_recovery.png         — z / rho recovery, counts vs RT
    rt_vs_counts_paired.png           — per-matrix gain distribution (not just the mean)
    rt_vs_counts_constructs.png       — model class / correlation type / separability, counts vs RT

  RT-specific:
    rt_architecture.png   — 5-way SFT confusion + dimension-neglect detection
    rt_by_architecture.png— recovery error broken down by processing architecture
    rt_lba.png             — LBA parameter recovery

The baseline throughout is the actual shipped counts-only model (MODEL_FILE), not a
freshly retrained proxy, so every "what does RT add" comparison is apples-to-apples on
identical matrices.
"""
import os
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")

from src.config import (FIGURES_DIR, RESULTS_DIR, TRIAL_RANGE, Z_MAX, R_MAX,
                        RT_DRIFT_SD, MODEL_FILE)
from src.data.rt_lba_generator import RTLBAGenerator, ARCHITECTURES, LBA_NAMES, featurize_lba
from src.inference.predict_rt import load_rt_model, predict_rt
from src.inference.predict import predict_point
from src.inference.model_posterior import construct_labels, amortized_compare
from src.viz.labels import labels_from_amortized
from src.inference.mle import fit_selected
from src.models.heads import train_space_to_params
from src.viz import figures as F
from src.viz import recovery as R
import src.grt_model as gm
import json

@torch.no_grad()
def _rt_posterior_samples(model, counts, rtq, trials, n_samples=800):
    """Posterior samples straight from the RT model's own Gaussian head — used for
    calibration / uncertainty figures so this doesn't depend on the counts-only
    predict_posterior's forward-signature assumptions."""
    device = next(model.parameters()).device
    x = featurize_lba(counts, rtq, trials).to(device)
    mean, L, *_ = model(x)
    dist = torch.distributions.MultivariateNormal(mean, scale_tril=L)
    s = dist.sample((n_samples,))                                   # (n_samples, N, 12) train space
    flat = train_space_to_params(s.reshape(-1, s.shape[-1])).reshape(s.shape)
    return flat.cpu().numpy()


def _construct_results(pred):
    """Adapt a predict_rt() output dict into the list-of-dicts construct_probabilities wants.

    construct_probabilities expects one dict per matrix with keys p_PI / p_sep_A / p_sep_B.
    The RT head returns p_corr (softmax over pi/rho1/free) and p_sep_A/B, so p_PI is the
    pi column of p_corr. Same convention used everywhere else in this file.
    """
    p_pi = np.asarray(pred["p_corr"])[:, 0]
    p_a = np.asarray(pred["p_sep_A"]); p_b = np.asarray(pred["p_sep_B"])
    return [{"p_PI": float(p_pi[i]), "p_sep_A": float(p_a[i]), "p_sep_B": float(p_b[i])}
            for i in range(len(p_pi))]


def main(n_per_class=150, seed=999):
    out_dir = os.path.join(FIGURES_DIR, "rt")
    os.makedirs(out_dir, exist_ok=True)
    fig = lambda n: os.path.join(out_dir, n)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- one shared held-out set: counts AND RTs come from the SAME simulated trials ----
    g = RTLBAGenerator(n_per_class=n_per_class, trial_range=TRIAL_RANGE, z_max=Z_MAX,
                       r_max=R_MAX, drift_sd=RT_DRIFT_SD, seed=seed)
    X, RTQ, Xt, yp, ylba, yc, yl, ya = g.generate(verbose=False)
    tc, ta, tb = construct_labels(yl)
    arch_labels = np.array(ARCHITECTURES)[ya]

    # ---- RT model ----
    rt_model = load_rt_model(device=device)
    p = predict_rt(rt_model, X, RTQ, Xt)
    pred_rt = p["params"]
    rt_pred_corr = p["p_corr"].argmax(1)
    rt_pred_sepA = (p["p_sep_A"] > .5).astype(int)
    rt_pred_sepB = (p["p_sep_B"] > .5).astype(int)
    # composed via the shared helper so the RT model's 12-way label is built exactly the
    # way the counts model's is (and the way the recovery-figure shapes are)
    rt_class_names = labels_from_amortized(p)

    # ---- counts-only baseline: the actual shipped model, SAME held-out matrices ----
    from src.api import load_model
    cm_model = load_model(MODEL_FILE, device=device)
    pred_cm = predict_point(cm_model, X, Xt).numpy()
    cm_compare = amortized_compare(cm_model, X, Xt)
    cm_pred_corr = cm_compare["p_corr"].argmax(1)
    cm_pred_sepA = (cm_compare["p_sep_A"] > .5).astype(int)
    cm_pred_sepB = (cm_compare["p_sep_B"] > .5).astype(int)
    cm_class_names = labels_from_amortized(cm_compare)

    # ===================== 1. PARITY: counts-only figures, RT model =====================
    R.recovery_panels(yp, pred_rt, fig("rt_recovery.png"),
                      class_names=yl, correct=None, method="RT model",
                      regime="RT model, held-out matrices", z_max=Z_MAX, r_max=R_MAX)
    M_rt, group_names = F.recovery_error_map(yp, pred_rt, yl, fig("rt_error_map.png"))
    F.model_confusion(list(yl), list(rt_class_names), fig("rt_confusion.png"),
                      regime="RT model, held-out matrices")
    F.construct_confusions(list(yl), list(rt_class_names), fig("rt_confusion_constructs.png"),
                           regime="RT model, held-out matrices")

    samples_rt = _rt_posterior_samples(rt_model, X, RTQ, Xt, n_samples=800)
    F.calibration(samples_rt, yp, fig("rt_calibration.png"))
    F.uncertainty_vs_trials(Xt.sum(1), samples_rt.std(0), fig("rt_uncertainty.png"),
                            regime="RT model, variable-trial held-out set")

    # parity with the counts suite's constructs.png: construct-probability calibration
    # + the PI identifiability frontier, built from the RT model's OWN comparison head
    # (model_posterior only accepts counts+trials, so it cannot be reused for the RT model).
    rt_res = _construct_results(p)
    F.construct_probabilities(rt_res, yl, np.abs(yp[:, 8:12]).max(1),
                              fig("rt_constructs.png"),
                              regime="RT model, variable-trial held-out set")

    # Parity with the counts suite (make_figures.py): both MLE workflows, not just the
    # saturated fit, and the RT model timed batched AND per-matrix. MLE-full is the
    # 12-parameter saturated fit (fast, but not what anyone reports); MLE-selected fits
    # every class and keeps the AIC/BIC winner — the realistic workflow, and the fair
    # speed comparison. Batched throughput against serial fitting is not a latency claim,
    # so the single-matrix number is reported alongside it.
    # ---- speed + accuracy, exported so the poster/comparison can read RT numbers ----
    # Same protocol as make_figures.py and compare_to_r.py: MLE-selected is the realistic
    # workflow; the RT model is timed BATCHED and PER-MATRIX (batched throughput / batch
    # size is not a latency claim). Everything is written to a JSON next to the figures so
    # scripts/compare_to_r.py and the poster can quote the +RT model without re-timing it.
    rng = np.random.default_rng(0)
    sub = rng.choice(len(X), min(150, len(X)), replace=False)
    t0 = time.time()
    sel = np.array([fit_selected(X[i], Xt[i])["params"] for i in sub])
    sel_ms = 1e3 * (time.time() - t0) / len(sub)
    t0 = time.time()
    predict_rt(rt_model, X[sub], RTQ[sub], Xt[sub])
    rt_ms = 1e3 * (time.time() - t0) / len(sub)
    n_single = min(50, len(sub))
    t0 = time.time()
    for i in sub[:n_single]:
        predict_rt(rt_model, X[i:i + 1], RTQ[i:i + 1], Xt[i:i + 1])
    rt_single_ms = 1e3 * (time.time() - t0) / n_single
    # counts-only GRIN single-matrix latency, same matrices, for the trade-off point
    t0 = time.time()
    for i in sub[:n_single]:
        predict_point(cm_model, X[i:i + 1], Xt[i:i + 1])
    cm_single_ms = 1e3 * (time.time() - t0) / n_single

    def _per_matrix_mae(pred_arr, idx):
        e = np.abs(pred_arr - yp[idx]).mean(1)
        return float(e.mean()), float(e.std(ddof=1) / np.sqrt(len(e)))

    rt_mae, rt_se = _per_matrix_mae(pred_rt[sub], sub)
    sel_mae, sel_se = _per_matrix_mae(sel, sub)
    cm_mae, cm_se = _per_matrix_mae(pred_cm[sub], sub)   # counts-only GRIN, same matrices

    # trade-off scatter (parity with the poster's speed_accuracy.png)
    F.speed_accuracy_tradeoff(
        ["+RT model (batched)", "+RT model (1 matrix)", "GRIN counts (1 matrix)",
         "MLE (selected)"],
        [rt_ms, rt_single_ms, cm_single_ms, sel_ms],
        [0.0, 0.0, 0.0, 0.0],
        [rt_mae, rt_mae, cm_mae, sel_mae],
        [rt_se, rt_se, cm_se, sel_se],
        fig("rt_speed_accuracy.png"),
        title=f"+RT model vs MLE  ({sel_ms / rt_single_ms:,.0f}x faster per matrix)",
        annotate=True)

    # JSON export: the single source of RT-model timing/accuracy for downstream scripts
    rt_metrics = {
        "n_timed": int(len(sub)),
        "rt_model": {"batched_ms": rt_ms, "single_ms": rt_single_ms,
                     "mae": rt_mae, "mae_se": rt_se},
        "grin_counts": {"single_ms": cm_single_ms, "mae": cm_mae, "mae_se": cm_se},
        "mle_selected": {"ms": sel_ms, "mae": sel_mae, "mae_se": sel_se},
        "speedup_vs_mle_single": sel_ms / rt_single_ms if rt_single_ms else None,
        "note": "RT model timed on the same held-out matrices as counts GRIN; "
                "MAE is per-matrix mean +/- SE across the timed subset.",
    }
    metrics_path = os.path.join(RESULTS_DIR, "rt_metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(rt_metrics, fh, indent=2)
    print(f"  wrote rt_speed_accuracy.png  +  {metrics_path}")

    # ============= 2. COUNTS-ONLY vs +RT, on the SAME held-out matrices =============
    M_cm, _ = F.recovery_error_map(yp, pred_cm, yl, fig("cm_baseline_error_map_matched.png"),
                                   title="Counts-only baseline — recovery error (matched condition)")
    F.error_gain_map(M_cm, M_rt, group_names, fig("rt_vs_counts_error_gain.png"))

    z_mae_cm = np.abs(pred_cm[:, :8] - yp[:, :8]).mean()
    z_mae_rt = np.abs(pred_rt[:, :8] - yp[:, :8]).mean()
    rho_mae_cm = np.abs(pred_cm[:, 8:] - yp[:, 8:]).mean()
    rho_mae_rt = np.abs(pred_rt[:, 8:] - yp[:, 8:]).mean()
    F.construct_gain_bars(
        ["z-parameters\n(1 \u2212 MAE)", "\u03c1 (correlation)\n(1 \u2212 MAE)"],
        [1 - z_mae_cm, 1 - rho_mae_cm], [1 - z_mae_rt, 1 - rho_mae_rt],
        fig("rt_vs_counts_recovery.png"),
        title="GRT parameter recovery: counts-only vs +RT")

    gain_z = np.abs(pred_cm[:, :8] - yp[:, :8]).mean(1) - np.abs(pred_rt[:, :8] - yp[:, :8]).mean(1)
    gain_rho = np.abs(pred_cm[:, 8:] - yp[:, 8:]).mean(1) - np.abs(pred_rt[:, 8:] - yp[:, 8:]).mean(1)
    F.paired_gain_distribution(gain_z, gain_rho, fig("rt_vs_counts_paired.png"))

    acc_class_cm = float(np.mean(cm_class_names == yl))
    acc_class_rt = float(np.mean(rt_class_names == yl))
    acc_corr_cm = float(np.mean(cm_pred_corr == tc)); acc_corr_rt = float(np.mean(rt_pred_corr == tc))
    acc_sepA_cm = float(np.mean(cm_pred_sepA == ta)); acc_sepA_rt = float(np.mean(rt_pred_sepA == ta))
    acc_sepB_cm = float(np.mean(cm_pred_sepB == tb)); acc_sepB_rt = float(np.mean(rt_pred_sepB == tb))
    F.construct_gain_bars(
        ["model class\n(12-way)", "correlation type\n(PI/RHO1/free)", "separable A", "separable B"],
        [acc_class_cm, acc_corr_cm, acc_sepA_cm, acc_sepB_cm],
        [acc_class_rt, acc_corr_rt, acc_sepA_rt, acc_sepB_rt],
        fig("rt_vs_counts_constructs.png"))

    # ===================== 3. RT-specific: architecture + LBA =====================
    K = len(ARCHITECTURES)
    pa = p["p_arch"].argmax(1)
    cm_arch = np.zeros((K, K))
    for t, q in zip(ya, pa):
        cm_arch[t, q] += 1
    cm_arch_n = cm_arch / cm_arch.sum(1, keepdims=True).clip(min=1)
    pi_acc_rt = float(np.mean((rt_pred_corr == 0) == (tc == 0)))
    pi_acc_cm = float(np.mean((cm_pred_corr == 0) == (tc == 0)))
    # Third element = chance level, drawn as a reference on the bar. rho recovery is
    # reported as MAE on its own axis rather than as 1-MAE next to accuracies: the "1 -"
    # only existed to make it point the same way as the accuracy bars, which is exactly
    # the kind of arithmetic that makes three different quantities look like one.
    gains = {
        "PI\n(yes/no)": (pi_acc_cm, pi_acc_rt, 0.5),
        "architecture": (1 / K, float(np.mean(pa == ya)), 1 / K),
        "\u03c1 recovery\n(MAE)": (rho_mae_cm, rho_mae_rt),
    }
    F.architecture_figure(cm_arch_n, ARCHITECTURES, gains, fig("rt_architecture.png"),
                          regime="RT model vs shipped counts-only model, same held-out matrices")

    F.recovery_error_map(yp, pred_rt, arch_labels, fig("rt_by_architecture.png"),
                         group_names=ARCHITECTURES, xlabel="processing architecture",
                         title="RT recovery error (MAE) by processing architecture")

    F.lba_recovery(ylba, p["lba"], LBA_NAMES, fig("rt_lba.png"),
                   regime="RT model, held-out matrices")

    print(f"RT figures -> {out_dir}")


if __name__ == "__main__":
    main()
    