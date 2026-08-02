"""
grt_figures.py -- custom, publication-quality figures for this port, layered on
top of the mandatory standard diagnostic report (loss/recovery/calibration/
coverage/z-score-contraction, produced by train_*.py per the amortized-workflow
skill). These are classic-GRT-style figures specific to the hierarchical
group/individual structure this port introduces -- not something the skill's
generic diagnostics cover, and not something the bespoke stack (no group model)
has ever needed.

Built on `src.viz.grt_space` (the shared, framework-agnostic perceptual-space
primitive, ported from web/assets/js/grt-plot.js) rather than reimplementing the
ellipse/decision-bound geometry here -- one source of truth for what a "classic
GRT plot" looks like, shared with the web app and available to the bespoke stack
too.

Six functions, one per figure the user asked to see:
  * group_template_figure       -- hierarchical level: the shared group template
                                    alone, or true-vs-recovered for validation.
  * individual_vs_group_figure  -- ONE participant's own (attention-scaled)
                                    perceptual space against the group template,
                                    annotated with their k_A/k_B -- explicitly
                                    requested as its own figure.
  * individual_grid_figure      -- small multiples: every participant in a
                                    session against the shared group template.
  * attention_scalar_forest     -- per-participant k_A/k_B posteriors (pooled via
                                    sample_attention_propagated), forest-plot
                                    style, reference line at k=1 (group-typical).
  * attention_recovery_figure   -- in-silico recovery scatter for log(k_A),
                                    log(k_B), matching src/viz/figures.py's
                                    parameter_recovery conventions.
  * pooled_vs_unpooled_rho_figure -- milestone 2's actual hypothesis (does
                                    pooling participants recover rho better than
                                    one matrix alone?), reusing
                                    src.viz.figures.rt_vs_counts_dumbbell rather
                                    than inventing a new comparison figure.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))                              # bayesflow_port/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))          # repo root

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.viz.style import set_style, BLUE_DEEP, RED_DEEP, MUTE, INK
from src.viz.grt_space import (perceptual_space, perceptual_space_figure,
                               STIM_PALETTE, PREDICTED_COLOR, shared_axis_limit)
import src.grt_model as gm


# --------------------------------------------------------------------------- #
def group_template_figure(group_theta, path, true_group_theta=None,
                          show_marginals=True, title=None, scale=1.0):
    """Hierarchical / group level: the shared perceptual template recovered by
    multiparticipant_workflow.py. Pass true_group_theta (in-silico only) to
    overlay true-vs-recovered (dashed) for validation; omit it for a plain
    display of a real-data group estimate."""
    perceptual_space_figure(
        true_group_theta if true_group_theta is not None else group_theta, path,
        predicted_theta=group_theta if true_group_theta is not None else None,
        show_marginals=show_marginals,
        title=title or "Group-level perceptual structure",
        predicted_label="recovered" if true_group_theta is not None else None,
        scale=scale)


# --------------------------------------------------------------------------- #
def individual_vs_group_figure(participant_theta, group_theta, path, k_A=None,
                               k_B=None, participant_label=None, title=None,
                               scale=1.0):
    """ONE participant's own (attention-scaled) perceptual space (dashed)
    against the shared group template (solid) -- the figure explicitly
    requested to sit alongside the group-level and grid views. Annotates
    k_A/k_B (natural units, 1 = group-typical) when given."""
    set_style(scale)
    lbl = participant_label or "participant"
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    perceptual_space(ax, group_theta, predicted_theta=participant_theta)
    ax.set_xlabel("dimension A"); ax.set_ylabel("dimension B")

    if k_A is not None and k_B is not None:
        ax.text(0.03, 0.03, f"$k_A$ = {k_A:.2f}\n$k_B$ = {k_B:.2f}",
               transform=ax.transAxes, va="bottom", ha="left",
               fontsize=9.5 * scale, color=INK)

    handles = [Line2D([0], [0], color=STIM_PALETTE[i], lw=2.2,
                      label=gm.STIMULUS_ORDER[i]) for i in range(4)]
    handles.append(Line2D([0], [0], color=PREDICTED_COLOR, lw=1.6,
                          linestyle=(0, (5, 4)), label=lbl))
    ax.legend(handles=handles, fontsize=8.5 * scale, loc="upper left",
             bbox_to_anchor=(1.02, 1.0), frameon=False)

    fig.suptitle(title or f"{lbl} vs. group template", x=0.02, ha="left",
                fontweight="bold", fontsize=15 * scale, color=INK)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def individual_grid_figure(participant_thetas, path, group_theta=None,
                           participant_labels=None, ncols=4, title=None, scale=1.0):
    """Small multiples: every participant in a session, each panel showing that
    participant's own perceptual space (solid if group_theta is None, else
    dashed against the group template drawn solid in every panel)."""
    set_style(scale)
    n = len(participant_thetas)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.9 * ncols, 2.9 * nrows),
                             squeeze=False)
    axes = axes.ravel()
    labels = participant_labels or [f"P{i + 1}" for i in range(n)]

    # one shared scale across every panel -- otherwise each panel auto-zooms to
    # its own data and between-panel size differences read as an artefact of
    # that, not a genuine between-participant difference.
    all_thetas = list(participant_thetas) + ([group_theta] if group_theta is not None else [])
    lim = shared_axis_limit(all_thetas)

    for i, ax in enumerate(axes):
        if i >= n:
            ax.axis("off")
            continue
        if group_theta is not None:
            perceptual_space(ax, group_theta, predicted_theta=participant_thetas[i],
                             show_level_ticks=False, title=labels[i], lim=lim)
        else:
            perceptual_space(ax, participant_thetas[i], show_level_ticks=False,
                             title=labels[i], lim=lim)
        ax.set_xticklabels([]); ax.set_yticklabels([])

    handles = [Line2D([0], [0], color=STIM_PALETTE[i], lw=2.2,
                      label=gm.STIMULUS_ORDER[i]) for i in range(4)]
    if group_theta is not None:
        handles += [Line2D([0], [0], color=MUTE, lw=2.0, label="group (solid)"),
                   Line2D([0], [0], color=PREDICTED_COLOR, lw=1.6,
                          linestyle=(0, (5, 4)), label="individual (dashed)")]
    fig.legend(handles=handles, fontsize=8.5 * scale, loc="lower center",
              ncol=min(6, len(handles)), frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(title or "Individual participants vs. group template", x=0.02,
                ha="left", fontweight="bold", fontsize=15 * scale, color=INK)
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def attention_scalar_forest(path, log_kA_samples, log_kB_samples,
                            participant_labels=None, true_k_A=None, true_k_B=None,
                            title=None, scale=1.0):
    """Forest plot of per-participant attention scalars, natural units (1 =
    group-typical). log_kA_samples/log_kB_samples: (n_participants, S) posterior
    draws -- typically the pooled output of
    attention_workflow.sample_attention_propagated, which already accounts for
    stage-1 group-template uncertainty. Pass true_k_A/true_k_B (in-silico only)
    to mark ground truth on top of each interval."""
    set_style(scale)
    log_kA_samples = np.atleast_2d(log_kA_samples)
    log_kB_samples = np.atleast_2d(log_kB_samples)
    n = log_kA_samples.shape[0]
    labels = participant_labels or [f"P{i + 1}" for i in range(n)]
    y = np.arange(n)[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, max(2.2, 0.55 * n + 1.2)), sharey=True)
    for ax, samples, true_vals, name in (
            (axes[0], log_kA_samples, true_k_A, "$k_A$"),
            (axes[1], log_kB_samples, true_k_B, "$k_B$")):
        k = np.exp(samples)
        med = np.median(k, axis=1)
        lo = np.quantile(k, 0.025, axis=1)
        hi = np.quantile(k, 0.975, axis=1)
        ax.hlines(y, lo, hi, color=BLUE_DEEP, lw=2.2, alpha=0.85, zorder=2)
        ax.scatter(med, y, color=BLUE_DEEP, s=32, zorder=3, label="posterior median")
        if true_vals is not None:
            ax.scatter(np.asarray(true_vals), y, marker="x", color=RED_DEEP, s=50,
                      zorder=4, label="true")
        ax.axvline(1.0, color=MUTE, lw=1.3, ls=(0, (4, 3)), zorder=1)
        ax.set_title(name)
        ax.set_xlabel("attention scalar")

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    if true_k_A is not None:
        axes[0].legend(fontsize=8.5 * scale, frameon=False)
    fig.suptitle(title or "Per-participant attention scalars vs. group (k = 1)",
                x=0.02, ha="left", fontweight="bold", fontsize=15 * scale, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def attention_recovery_figure(true_log_kA, pred_log_kA, true_log_kB, pred_log_kB,
                              path, scale=1.0):
    """In-silico recovery scatter for log(k_A), log(k_B), matching
    src/viz/figures.py's parameter_recovery conventions (r + MAE annotated,
    identity diagonal, square-framed panels) so this reads as part of the same
    figure family rather than a one-off. pred_* may be point estimates or
    (N, S) posterior samples (median is taken automatically)."""
    set_style(scale)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.6))
    for ax, t, p, name in ((axes[0], true_log_kA, pred_log_kA, r"$\log k_A$"),
                           (axes[1], true_log_kB, pred_log_kB, r"$\log k_B$")):
        t = np.asarray(t, dtype=float).ravel()
        p = np.asarray(p, dtype=float)
        if p.ndim == 2:
            p = np.median(p, axis=1)
        p = p.ravel()
        ax.scatter(t, p, s=9, c=BLUE_DEEP, alpha=0.3, edgecolors="none", rasterized=True)
        lo = min(t.min(), p.min()); hi = max(t.max(), p.max())
        ax.plot([lo, hi], [lo, hi], color=RED_DEEP, lw=1.6, ls=(0, (4, 3)), zorder=3)
        r = np.corrcoef(t, p)[0, 1]
        mae = np.abs(t - p).mean()
        ax.set_title(name)
        ax.text(0.06, 0.94, f"r = {r:.2f}\nMAE = {mae:.2f}", transform=ax.transAxes,
               va="top", ha="left", fontsize=9 * scale, color=INK)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_box_aspect(1)
        ax.set_xlabel("true")
    axes[0].set_ylabel("posterior median")
    fig.suptitle("Attention-scalar recovery — predicted vs. true (held-out simulations)",
                x=0.02, ha="left", fontweight="bold", fontsize=15 * scale, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def pooled_vs_unpooled_rho_figure(rho_metric_single, rho_metric_pooled, path,
                                  metric_name="NRMSE", stimulus_labels=None,
                                  scale=1.0):
    """Milestone 2's actual hypothesis: does pooling participants recover rho
    better than one matrix alone? Reuses src.viz.figures.rt_vs_counts_dumbbell
    (already built for exactly this "paired before/after" comparison) rather
    than inventing a new figure -- generalised with before_label/after_label/
    xlim precisely so this reuse doesn't inherit the RT-poster's hardcoded
    "counts only"/"+ RT" legend or its [0, 1]-accuracy x-axis, which would
    silently mislabel or clip an error metric like NRMSE. Caller supplies the
    already-extracted per-stimulus rho metric (e.g. the rho_0..rho_3 entries of
    the chosen row in each workflow's metrics.csv) -- this function does not
    parse metrics.csv itself, since BayesFlow's diagnostic-table column naming
    for concatenated inference_variables was not verified as part of this pass
    and guessing at it would risk silently mislabelling the comparison."""
    from src.viz.figures import rt_vs_counts_dumbbell
    labels = stimulus_labels or [f"rho_{i}" for i in range(len(rho_metric_single))]
    lower_is_better = any(k in metric_name.upper() for k in ("MAE", "RMSE", "NRMSE"))
    rows = [{"metric": lbl, "counts": s, "rt": p}
           for lbl, s, p in zip(labels, rho_metric_single, rho_metric_pooled)]
    all_vals = list(rho_metric_single) + list(rho_metric_pooled)
    span = max(all_vals) - min(all_vals) if len(all_vals) > 1 else max(all_vals, default=1.0)
    pad = 0.12 * (span or 1.0)
    xlim = (max(0.0, min(all_vals) - pad), max(all_vals) + pad)
    rt_vs_counts_dumbbell(
        rows, path, scale=scale,
        title="Pooling across participants: rho recovery",
        xlabel=metric_name, before_label="single matrix", after_label="pooled (group)",
        xlim=xlim,
        note="lower is better" if lower_is_better else "higher is better")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Demo/smoke-test only, on PURELY SIMULATED data -- no trained checkpoint is
    # required. Run with: python bayesflow_port/grt_figures.py
    #
    # This is NOT a real-results pipeline: it fabricates a "recovered" group
    # template by jittering the true one, and fabricates attention-scalar
    # "posteriors" by jittering the true log(k_A)/log(k_B) with a fixed SD, since
    # no trained checkpoint for multiparticipant_workflow.py or attention_workflow.py
    # exists yet (see the "Open questions / TODO" section of README.md). Once
    # those are trained, replace the fabricated arrays below with:
    #   - multiparticipant_workflow's build_workflow(...).sample(...) for the
    #     recovered group template posterior;
    #   - attention_workflow.sample_attention_propagated(...) for the pooled
    #     per-participant log(k_A)/log(k_B) posteriors.
    import numpy as np
    from multiparticipant_workflow import simulate_sessions

    out_dir = os.path.join(os.path.dirname(__file__), "results", "grt_figures_demo")
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(0)
    n_participants = 6
    sess = simulate_sessions(n_sessions=1, n_participants=n_participants, rng=rng,
                             attention_sd=0.3)
    group_theta = np.concatenate([sess["group_z"][0], sess["group_rho"][0]])
    k_A, k_B = sess["k_A"][0], sess["k_B"][0]
    zx, zy, rho = group_theta[:4], group_theta[4:8], group_theta[8:]
    participant_thetas = [np.concatenate([zx * k_A[p], zy * k_B[p], rho])
                          for p in range(n_participants)]
    labels = [f"P{i + 1}" for i in range(n_participants)]

    fabricated_recovered_group = group_theta + rng.normal(0, 0.15, 12)
    group_template_figure(fabricated_recovered_group, os.path.join(out_dir, "group_template.png"),
                          true_group_theta=group_theta)

    individual_vs_group_figure(participant_thetas[0], group_theta,
                               os.path.join(out_dir, "individual_vs_group.png"),
                               k_A=k_A[0], k_B=k_B[0], participant_label=labels[0])

    individual_grid_figure(participant_thetas, os.path.join(out_dir, "individual_grid.png"),
                           group_theta=group_theta, participant_labels=labels)

    S = 200
    fabricated_log_kA_samples = np.log(k_A)[:, None] + rng.normal(0, 0.15, (n_participants, S))
    fabricated_log_kB_samples = np.log(k_B)[:, None] + rng.normal(0, 0.15, (n_participants, S))
    attention_scalar_forest(os.path.join(out_dir, "attention_forest.png"),
                            fabricated_log_kA_samples, fabricated_log_kB_samples,
                            participant_labels=labels, true_k_A=k_A, true_k_B=k_B)

    N = 300
    true_lkA = rng.normal(0, 0.3, N); true_lkB = rng.normal(0, 0.3, N)
    fabricated_pred_lkA = true_lkA + rng.normal(0, 0.12, N)
    fabricated_pred_lkB = true_lkB + rng.normal(0, 0.12, N)
    attention_recovery_figure(true_lkA, fabricated_pred_lkA, true_lkB, fabricated_pred_lkB,
                              os.path.join(out_dir, "attention_recovery.png"))

    fabricated_single_matrix_nrmse = [0.42, 0.38, 0.51, 0.44]
    fabricated_pooled_nrmse = [0.24, 0.21, 0.30, 0.26]
    pooled_vs_unpooled_rho_figure(fabricated_single_matrix_nrmse, fabricated_pooled_nrmse,
                                  os.path.join(out_dir, "pooled_vs_unpooled_rho.png"))

    print(f"Wrote 6 demo figures to {out_dir}/ (fabricated data -- see module docstring "
         "above for how to wire in real trained-model outputs).")
