"""figures.py — the GRIN figure suite (recovery, identification, calibration, story)."""
import numpy as np
import matplotlib.pyplot as plt

from .style import (set_style, BLUE, BLUE_DEEP, RED, RED_DEEP, INK, MUTE,
                    CMAP_SEQ, CMAP_DIV, despine_heatmap, clean_colorbar)

try:
    from src import grt_model as gm
except ImportError:
    import grt_model as gm

_PRETTY = {"zx": "$z_x$", "zy": "$z_y$", "rho": r"$\rho$"}
def _pname(name):
    base, i = name.rsplit("_", 1)
    return f"{_PRETTY.get(base, base)}$_{{{i}}}$"


def parameter_recovery(true, pred, path, scale=1.0):
    """12-panel predicted-vs-true scatter, one per identified parameter."""
    set_style(scale)
    names = gm.PARAM_NAMES
    fig, axes = plt.subplots(3, 4, figsize=(13, 9.5))
    for j, ax in enumerate(axes.ravel()):
        t, p = true[:, j], pred[:, j]
        ax.scatter(t, p, s=7, c=BLUE, alpha=0.22, edgecolors="none", rasterized=True)
        lo = min(t.min(), p.min()); hi = max(t.max(), p.max())
        ax.plot([lo, hi], [lo, hi], color=RED_DEEP, lw=1.6, ls=(0, (4, 3)), zorder=3)
        r = np.corrcoef(t, p)[0, 1]; mae = np.abs(t - p).mean()
        ax.set_title(_pname(names[j]))
        ax.text(0.06, 0.94, f"r = {r:.2f}\nMAE = {mae:.2f}", transform=ax.transAxes,
                va="top", ha="left", fontsize=9 * scale, color=INK)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_box_aspect(1)
        if j % 4 == 0: ax.set_ylabel("predicted")
        if j >= 8:     ax.set_xlabel("true")
    fig.suptitle("Parameter recovery — predicted vs. true (held-out simulations)",
                 x=0.02, ha="left", fontweight="bold", fontsize=15 * scale, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(path); plt.close(fig)


def _model_constructs(names):
    """GRT class names -> (corr 0/1/2, ps_x 0/1, ps_y 0/1). None-safe (yields -1)."""
    ci = {"pi": 0, "rho1": 1, "free": 2}
    c, x, y = [], [], []
    for n in np.asarray(names, dtype=object):
        if n is None or (isinstance(n, float) and n != n):
            c.append(-1); x.append(-1); y.append(-1)
        else:
            k, px, py = gm.MODEL_SPECS[n]
            c.append(ci[k]); x.append(int(px)); y.append(int(py))
    return np.array(c), np.array(x), np.array(y)


def model_confusion(true_labels, pred_labels, path, order_by_complexity=True,
                    regime="", title=None, scale=1.0):
    """Row-normalised 12-way model-identification confusion heatmap.

    Two deliberate departures from the obvious version:

    * Rows and columns are ordered by NUMBER OF FREE PARAMETERS, not declaration order, so
      nested neighbours sit next to each other. The interesting structure in this matrix is
      that errors land on adjacent models -- confusing pi_ps_ds with pi_psa_ds is a very
      different event from confusing it with ds -- and complexity ordering puts that on a
      band around the diagonal where it can actually be seen.
    * 12-way exact accuracy is NOT the headline. It penalises distinctions between
      statistically indistinguishable nested models, so it is reported alongside the three
      per-construct accuracies rather than alone in the title. See construct_confusions()
      for the figure where those constructs are the subject.

    Classes the method never selects get a dagger on their column label, because an
    all-white column otherwise looks identical to a class that is simply never confused.
    """
    set_style(scale)
    names = (sorted(gm.MODEL_NAMES, key=lambda m: (gm.n_free_params(m), m))
             if order_by_complexity else list(gm.MODEL_NAMES))
    K = len(names); idx = {n: i for i, n in enumerate(names)}

    cm = np.zeros((K, K))
    n_unlabelled = 0
    for t, p in zip(true_labels, pred_labels):
        if p is None:
            n_unlabelled += 1
            continue
        cm[idx[t], idx[p]] += 1
    total = cm.sum()
    acc = np.trace(cm) / total if total else np.nan
    never = cm.sum(0) == 0
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)

    tc, tx, ty = _model_constructs(true_labels)
    pc, px, py = _model_constructs(pred_labels)
    ok = pc >= 0
    a_corr = float((pc[ok] == tc[ok]).mean()) if ok.any() else np.nan
    a_psa = float((px[ok] == tx[ok]).mean()) if ok.any() else np.nan
    a_psb = float((py[ok] == ty[ok]).mean()) if ok.any() else np.nan

    fig, ax = plt.subplots(figsize=(9.8, 8.8))
    im = ax.imshow(cmn, cmap=CMAP_SEQ, vmin=0, vmax=1)
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels([f"{n} †" if never[j] else n for j, n in enumerate(names)],
                       rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("predicted class" + ("   († never selected)" if never.any() else ""))
    ax.set_ylabel("true class")
    for i in range(K):
        for j in range(K):
            if cmn[i, j] > 0.01:
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                        fontsize=7 * scale, color="white" if cmn[i, j] > 0.55 else INK)
    despine_heatmap(ax)
    clean_colorbar(fig.colorbar(im, fraction=0.046, pad=0.04), "proportion of true class")

    ax.set_title(title or "Model identification")
    sub = (f"12-way exact {acc:.2f}   ·   correlation structure {a_corr:.2f}   ·   "
           f"PS(A) {a_psa:.2f}   ·   PS(B) {a_psb:.2f}")
    if regime:
        sub += f"\n{regime}"
    if n_unlabelled:
        sub += f"   ({n_unlabelled} matrices returned no class)"
    ax.text(0.0, 1.015, sub, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9.5 * scale, color=MUTE)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return {"exact": acc, "corr": a_corr, "ps_x": a_psa, "ps_y": a_psb}


def construct_confusions(true_labels, pred_labels, path, regime="", title=None, scale=1.0):
    """The three per-construct confusions the 12-way matrix averages away.

    The 12 classes are exactly the product of three independent decisions -- correlation
    structure (PI / 1rho / free), separability on A, separability on B -- so the honest
    identification result is three small matrices, not one large sparse one. This is also
    where the known asymmetry shows up plainly: separability is sharp, correlation
    structure is soft.
    """
    set_style(scale)
    tc, tx, ty = _model_constructs(true_labels)
    pc, px, py = _model_constructs(pred_labels)
    keep = pc >= 0
    n_drop = int((~keep).sum())

    specs = [(tc[keep], pc[keep], [r"PI", r"1$\rho$", r"free"], "Correlation structure"),
             (tx[keep], px[keep], ["violated", "holds"], "Separability on A"),
             (ty[keep], py[keep], ["violated", "holds"], "Separability on B")]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
    for ax, (t, p, ticks, name) in zip(axes, specs):
        k = len(ticks)
        cm = np.zeros((k, k))
        for a, b in zip(t, p):
            cm[a, b] += 1
        acc = np.trace(cm) / cm.sum() if cm.sum() else np.nan
        cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
        im = ax.imshow(cmn, cmap=CMAP_SEQ, vmin=0, vmax=1)
        ax.set_xticks(range(k)); ax.set_yticks(range(k))
        ax.set_xticklabels(ticks); ax.set_yticklabels(ticks)
        ax.set_xlabel("inferred"); ax.set_ylabel("true")
        ax.set_title(f"{name}   (acc = {acc:.2f})")
        for i in range(k):
            for j in range(k):
                ax.text(j, i, f"{cmn[i, j]:.2f}\n$n$={int(cm[i, j])}", ha="center",
                        va="center", fontsize=8.5 * scale,
                        color="white" if cmn[i, j] > 0.55 else INK)
        despine_heatmap(ax)
    clean_colorbar(fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02),
                   "proportion of true class")

    fig.suptitle(title or "Model identification by construct", x=0.02, ha="left",
                 fontweight="bold", fontsize=15 * scale, color=INK)
    sub = regime
    if n_drop:
        sub = (sub + "   " if sub else "") + f"({n_drop} matrices returned no class)"
    if sub:
        fig.text(0.02, 0.945, sub, ha="left", va="top", fontsize=10.5 * scale, color=MUTE)
    fig.savefig(path); plt.close(fig)


def recovery_error_map(true, pred, labels, path, regime="", group_names=None,
                       xlabel=None, title=None, scale=1.0):
    """Recovery error by parameter and grouping: magnitude (left) and signed bias (right).

    RETURNS (mae, names): the (12, n_groups) MAE matrix and the group names in plotted
    order, so a paired run can be differenced -- see error_gain_map().

    Three things this figure has to get right, none of which a single heatmap does:

    * z-scores and correlations are NOT on a common scale (+/-3 vs +/-0.9), so a shared
      colour norm makes the rho block look like the best-recovered part of the map when it
      is the worst. The two families are therefore normalised separately, in stacked blocks
      with their own colourbars.
    * Under the pi_* classes the true rho is exactly 0, so those cells measure "does it
      return ~0", not recovery. They are hatched rather than silently rewarded with a dark
      cell. (Only applies when grouping by GRT model class.)
    * MAE cannot distinguish systematic shrinkage from unbiased noise, which is precisely
      the failure mode an amortized posterior mean is prone to. The right panel is signed
      bias (pred - true) on a diverging scale, so shrinkage shows as a coherent block of
      one colour instead of hiding inside the magnitudes.

    group_names: pass an explicit ordered list to group by something other than model class
    (e.g. ARCHITECTURES). When omitted, columns are GRT model classes ordered by number of
    free parameters, and the structurally-zero rho cells are hatched.
    """
    set_style(scale)
    true = np.asarray(true, dtype=float); pred = np.asarray(pred, dtype=float)
    labels = np.asarray(labels)
    pnames = gm.PARAM_NAMES; P = len(pnames)

    by_class = group_names is None
    if by_class:
        # ordered by model complexity, so the map's real gradient reads as a gradient
        names = sorted(gm.MODEL_NAMES, key=lambda m: (gm.n_free_params(m), m))
        xlabel = xlabel or "model class  (ordered by number of free parameters)"
    else:
        names = list(group_names)
        xlabel = xlabel or "group"
    C = len(names)

    mae = np.full((P, C), np.nan)
    bias = np.full((P, C), np.nan)
    fixed = np.zeros((P, C), bool)          # structurally degenerate cells
    for ci, m in enumerate(names):
        msk = labels == m
        if msk.sum():
            d = pred[msk] - true[msk]
            mae[:, ci] = np.abs(d).mean(0)
            bias[:, ci] = d.mean(0)
        if by_class and gm.MODEL_SPECS[m][0] == "pi":
            fixed[8:12, ci] = True          # rho == 0 by construction

    blocks = [("z", slice(0, 8)), ("rho", slice(8, 12))]
    fig = plt.figure(figsize=(17.5, 8.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[8, 4], width_ratios=[1, 1, 0.035],
                          hspace=0.16, wspace=0.14)

    left_axes = []
    for col, (M, cmap, lab, ptitle, annot) in enumerate([
            (mae, CMAP_SEQ, "mean absolute error", "Magnitude — MAE", True),
            (bias, CMAP_DIV, "mean signed error (pred $-$ true)", "Direction — bias", False)]):
        for row, (fam, sl) in enumerate(blocks):
            ax = fig.add_subplot(gs[row, col])
            if col == 0:
                left_axes.append(ax)
            sub = M[sl]
            if cmap is CMAP_DIV:
                v = np.nanmax(np.abs(sub)) or 1.0
                im = ax.imshow(sub, cmap=cmap, aspect="auto", vmin=-v, vmax=v)
            else:
                im = ax.imshow(sub, cmap=cmap, aspect="auto", vmin=0,
                               vmax=np.nanmax(sub) or 1.0)
            ax.set_yticks(range(sub.shape[0]))
            ax.set_yticklabels([_pname(n) for n in pnames[sl]])
            if row == len(blocks) - 1:
                ax.set_xticks(range(C))
                ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8.5 * scale)
                ax.set_xlabel(xlabel)
            else:
                ax.set_xticks([])
                ax.set_title(ptitle if row == 0 else "")
            for i in range(sub.shape[0]):
                for j in range(C):
                    if fixed[sl][i, j]:
                        ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                                   hatch="////", edgecolor=MUTE,
                                                   linewidth=0.0, zorder=3))
                    elif annot and np.isfinite(sub[i, j]):
                        rel = sub[i, j] / (np.nanmax(sub) or 1.0)
                        ax.text(j, i, f"{sub[i, j]:.2f}", ha="center", va="center",
                                fontsize=6.2 * scale,
                                color="white" if rel > 0.6 else INK)
            despine_heatmap(ax)
            if col == 1:
                cax = fig.add_subplot(gs[row, 2])
                clean_colorbar(fig.colorbar(im, cax=cax), lab if row == 0 else None)

    # colourbars for the left column, one per unit family
    for row, ax in enumerate(left_axes):
        cb = fig.colorbar(ax.images[0], ax=ax, fraction=0.028, pad=0.012)
        clean_colorbar(cb, "MAE" if row == 0 else None)

    if fixed.any():
        from matplotlib.patches import Patch
        fig.legend(handles=[Patch(facecolor="white", edgecolor=MUTE, hatch="////",
                                  label=r"$\rho \equiv 0$ by construction (not recovery)")],
                   loc="lower left", bbox_to_anchor=(0.01, -0.005), fontsize=9 * scale,
                   frameon=False)

    fig.suptitle(title or "Recovery error by parameter and model class",
                 x=0.02, ha="left", fontweight="bold", fontsize=15 * scale, color=INK)
    if regime:
        fig.text(0.02, 0.945, regime, ha="left", va="top", fontsize=10.5 * scale, color=MUTE)
    # NOTE: tight_layout is not used here -- the gridspec carries external colourbars,
    # which it cannot measure. Margins are set explicitly instead.
    fig.subplots_adjust(left=0.055, right=0.945, top=0.885, bottom=0.145)
    fig.savefig(path)
    plt.close(fig)
    return mae, names


def error_gain_map(mae_before, mae_after, group_names, path,
                   before_label="counts only", after_label="+ RT",
                   title=None, regime="", scale=1.0):
    """Where does the extra information actually help? (mae_before - mae_after).

    Positive (red) = the second model recovers that parameter better in that group.
    Diverging and symmetric about zero, so "no change" is white and cannot be mistaken for
    "small improvement". z and rho blocks are normalised separately for the same reason
    they are in recovery_error_map -- a 0.05 gain means very different things in the two
    unit families.
    """
    set_style(scale)
    before = np.asarray(mae_before, dtype=float)
    after = np.asarray(mae_after, dtype=float)
    if before.shape != after.shape:
        raise ValueError(f"shape mismatch: {before.shape} vs {after.shape} — the two error "
                         "maps must come from the same parameters and the same groups")
    gain = before - after
    pnames = gm.PARAM_NAMES
    names = list(group_names)
    C = len(names)

    blocks = [slice(0, 8), slice(8, 12)]
    fig = plt.figure(figsize=(11.5, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[8, 4], width_ratios=[1, 0.035],
                          hspace=0.16, wspace=0.06)
    for row, sl in enumerate(blocks):
        ax = fig.add_subplot(gs[row, 0])
        sub = gain[sl]
        v = np.nanmax(np.abs(sub)) or 1.0
        im = ax.imshow(sub, cmap=CMAP_DIV, aspect="auto", vmin=-v, vmax=v)
        ax.set_yticks(range(sub.shape[0]))
        ax.set_yticklabels([_pname(n) for n in pnames[sl]])
        if row == len(blocks) - 1:
            ax.set_xticks(range(C))
            ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8.5 * scale)
        else:
            ax.set_xticks([])
        for i in range(sub.shape[0]):
            for j in range(C):
                if np.isfinite(sub[i, j]):
                    ax.text(j, i, f"{sub[i, j]:+.2f}", ha="center", va="center",
                            fontsize=6.2 * scale,
                            color="white" if abs(sub[i, j]) > 0.72 * v else INK)
        despine_heatmap(ax)
        cax = fig.add_subplot(gs[row, 1])
        clean_colorbar(fig.colorbar(im, cax=cax),
                       f"MAE reduction  ({before_label} $-$ {after_label})" if row == 0 else None)

    fig.suptitle(title or f"Where {after_label} helps — change in recovery error",
                 x=0.02, ha="left", fontweight="bold", fontsize=15 * scale, color=INK)
    sub_txt = f"positive = {after_label} is more accurate"
    if regime:
        sub_txt += f"   ·   {regime}"
    fig.text(0.02, 0.945, sub_txt, ha="left", va="top", fontsize=10.5 * scale, color=MUTE)
    fig.subplots_adjust(left=0.10, right=0.90, top=0.885, bottom=0.155)
    fig.savefig(path)
    plt.close(fig)
    return gain


def construct_gain_bars(labels, before, after, path, before_label="counts only",
                        after_label="+ response times", title=None, ylabel="accuracy",
                        scale=1.0):
    """Paired before/after bars, one group per construct or metric.

    NOTE ON MIXED UNITS: callers sometimes pass a mix of accuracies and (1 - MAE) values.
    Those are not the same quantity and a shared 0-1 axis invites reading them as such, so
    the y-limit is only clamped to [0, 1] when every value genuinely lies in that range,
    and the axis label is caller-supplied rather than hard-coded to "accuracy".
    """
    set_style(scale)
    before = np.asarray(before, dtype=float); after = np.asarray(after, dtype=float)
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(max(7.0, 2.3 * len(labels)), 5))
    ax.bar(x - w / 2, before, w, color=MUTE, label=before_label)
    ax.bar(x + w / 2, after, w, color=BLUE, label=after_label)
    for i, (a, b) in enumerate(zip(before, after)):
        if np.isfinite(a):
            ax.text(i - w / 2, a, f"{a:.2f}", ha="center", va="bottom", fontsize=8.5 * scale)
        if np.isfinite(b):
            ax.text(i + w / 2, b, f"{b:.2f}", ha="center", va="bottom", fontsize=8.5 * scale)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.5 * scale)
    ax.set_ylabel(ylabel)
    vals = np.concatenate([before, after])
    vals = vals[np.isfinite(vals)]
    if vals.size and vals.min() >= 0 and vals.max() <= 1:
        ax.set_ylim(0, 1.08)
    ax.set_title(title or f"{before_label} vs {after_label}")
    ax.legend()
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def paired_gain_distribution(gain_z, gain_rho, path, before_label="counts only",
                             after_label="+ RT", title=None, scale=1.0):
    """Per-matrix improvement distributions — the mean is not the story.

    A mean gain of +0.02 is consistent with "helps a little everywhere" and with "helps a
    lot on a third of matrices and hurts on the rest". These are very different claims, and
    only the distribution distinguishes them. Zero is marked, the median is marked, and the
    share of matrices actually improved is printed, because that share is the honest
    headline for a paired comparison.
    """
    set_style(scale)
    fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.8))
    for a, g, name, col in ((ax[0], np.asarray(gain_z, float), "$z$ parameters", BLUE),
                            (ax[1], np.asarray(gain_rho, float), r"$\rho$ parameters", RED_DEEP)):
        g = g[np.isfinite(g)]
        a.hist(g, bins=40, color=col, edgecolor="white", linewidth=0.3)
        a.axvline(0, color=INK, lw=1.4, ls=(0, (4, 3)))
        if g.size:
            med = np.median(g)
            a.axvline(med, color=MUTE, lw=1.6)
            frac = float((g > 0).mean())
            a.text(0.03, 0.95, f"median {med:+.3f}\n{frac:.0%} of matrices improved",
                   transform=a.transAxes, va="top", ha="left", fontsize=9.5 * scale, color=INK)
        a.set_title(name)
        a.set_xlabel(f"MAE reduction per matrix  ({before_label} $-$ {after_label})")
        a.set_ylabel("matrices")
    fig.suptitle(title or f"Per-matrix gain from {after_label} — distribution, not just the mean",
                 x=0.02, ha="left", fontweight="bold", fontsize=15 * scale, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(path); plt.close(fig)


def calibration(samples, true, path, regime="", n_bins=20, scale=1.0):
    """SBC rank histograms per parameter family + interval coverage per family.

    Why this is not one pooled histogram: pooling all 12 parameters lets miscalibrations
    cancel. If the z posteriors are slightly conservative and the rho posteriors are
    overconfident, the pooled ranks can look flat while neither family is calibrated. Since
    the correlations are the known-weak family, pooling hides exactly the thing worth
    seeing, so z and rho get their own panel and their own coverage curve.

    Why some rho values are excluded: under the PI classes the true rho is EXACTLY 0. The
    SBC rank is then not a calibration statistic at all -- it measures how much posterior
    mass sits below zero, and piles up at the extremes for reasons unrelated to whether the
    posterior is honest. Those entries are dropped and counted in the subtitle rather than
    quietly flattening or spiking the histogram. Detection is `true == 0` exactly, which
    catches every PI draw and (with probability 1) no free-correlation draw.

    The grey band is the 95% interval for the bin counts under correct calibration
    (Binomial(M, 1/n_bins)), so "looks flat" becomes "is within tolerance".
    """
    from scipy.stats import binom

    set_style(scale)
    samples = np.asarray(samples, dtype=float)      # (S, N, 12)
    true = np.asarray(true, dtype=float)            # (N, 12)
    S = samples.shape[0]

    ranks = (samples < true[None]).sum(0) / S       # (N, 12)
    keep = np.ones_like(true, dtype=bool)
    keep[:, 8:12] = true[:, 8:12] != 0.0            # drop structurally-degenerate rho
    n_drop = int((~keep[:, 8:12]).sum())

    fams = [("$z$ parameters", slice(0, 8), BLUE),
            (r"$\rho$ parameters", slice(8, 12), RED_DEEP)]
    levels = np.array([0.5, 0.7, 0.8, 0.9, 0.95])

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.9))
    for k, (name, sl, col) in enumerate(fams):
        r = ranks[:, sl][keep[:, sl]].ravel()
        M = r.size
        ax[k].hist(r, bins=n_bins, range=(0, 1), color=col, edgecolor="white", linewidth=0.4)
        if M:
            lo, hi = binom.ppf([0.025, 0.975], M, 1.0 / n_bins)
            ax[k].axhspan(lo, hi, color=MUTE, alpha=0.22, linewidth=0,
                          label="95% band if calibrated")
            ax[k].axhline(M / n_bins, color=INK, lw=1.3, ls=(0, (4, 3)), label="uniform")
        ax[k].set_title(f"SBC ranks — {name}")
        ax[k].set_xlabel("normalised rank of true value"); ax[k].set_ylabel("count")
        ax[k].legend(fontsize=8.5 * scale)

    for name, sl, col in fams:
        emp = []
        for l in levels:
            lo_q = np.quantile(samples[:, :, sl], (1 - l) / 2, axis=0)
            hi_q = np.quantile(samples[:, :, sl], (1 + l) / 2, axis=0)
            inside = (true[:, sl] >= lo_q) & (true[:, sl] <= hi_q)
            emp.append(inside[keep[:, sl]].mean())
        ax[2].plot(levels, emp, "o-", color=col, ms=6.5, lw=1.9, label=name)
    ax[2].plot([0, 1], [0, 1], color=MUTE, lw=1.4, ls=(0, (4, 3)), label="perfect")
    ax[2].set_title("Interval coverage")
    ax[2].set_xlabel("nominal credible level"); ax[2].set_ylabel("empirical coverage")
    ax[2].set_xlim(0.4, 1); ax[2].set_ylim(0.4, 1); ax[2].set_box_aspect(1)
    ax[2].legend(fontsize=8.5 * scale)

    fig.suptitle("Calibration — are the posteriors honest?", x=0.02, ha="left",
                 fontweight="bold", fontsize=15 * scale, color=INK)
    sub = regime
    if n_drop:
        sub = (sub + "   ·   " if sub else "") + \
              rf"{n_drop} $\rho$ values excluded ($\rho \equiv 0$ under PI — rank is not a "
        sub += "calibration statistic there)"
    if sub:
        fig.text(0.02, 0.945, sub, ha="left", va="top", fontsize=10 * scale, color=MUTE)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(path); plt.close(fig)


def uncertainty_vs_trials(trials_total, post_std, path, regime="", min_per_bin=25,
                          n_bins=12, show_reference=True, scale=1.0):
    """Posterior width vs data, per parameter family, on log-log axes.

    post_std: (N, 12) per-parameter posterior SDs -> one curve per family (preferred), or
              (N,) pre-averaged -> a single curve (back-compatible, but see below).

    Three deliberate choices:

    * PER FAMILY, NOT POOLED. Averaging a z-unit SD with a correlation-unit SD produces a
      number in no units at all. The two families shrink at different rates and start at
      different widths, which is the informative part.
    * MEDIAN AND IQR, NOT MEAN +/- SD. The spread of posterior widths within a trial bin is
      driven mostly by model class and true parameter magnitude, not by trial count, so a
      symmetric SD band around the mean invites reading between-matrix heterogeneity as
      uncertainty about the trend. Quantiles say what they are.
    * LOG-LOG WITH AN n^(-1/2) GUIDE. The claim is that width shrinks like root-n. On
      log-log that is a straight line of slope -0.5, anchored here at each family's first
      bin, so agreement (or the departure from it where the amortization floor bites) is
      readable by eye instead of inferred from a curve's shape.

    Bins are quantile-based but merged forward until each holds at least `min_per_bin`
    matrices, and the surviving counts are printed along the top -- an extreme bin resting
    on four matrices should not look as solid as one resting on four hundred.
    """
    set_style(scale)
    tt = np.asarray(trials_total, dtype=float)
    ps = np.asarray(post_std, dtype=float)
    if ps.ndim == 1:
        fams = [("mean posterior SD (pooled)", None, BLUE_DEEP)]
    else:
        fams = [("$z$ parameters", slice(0, 8), BLUE_DEEP),
                (r"$\rho$ parameters", slice(8, 12), RED_DEEP)]

    # quantile bins, merged forward so none is too thin to trust
    edges = np.unique(np.quantile(tt, np.linspace(0, 1, n_bins + 1)))
    bins, lo = [], edges[0]
    for hi in edges[1:]:
        m = (tt >= lo) & (tt <= hi)
        if m.sum() >= min_per_bin or hi == edges[-1]:
            bins.append((lo, hi, m)); lo = hi
    if not bins:
        bins = [(edges[0], edges[-1], np.ones_like(tt, dtype=bool))]

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    counts = [int(m.sum()) for _, _, m in bins]
    centers = np.array([tt[m].mean() if m.any() else np.nan for _, _, m in bins])

    for name, sl, col in fams:
        v = ps if sl is None else ps[:, sl].mean(1)
        med = np.array([np.median(v[m]) if m.any() else np.nan for _, _, m in bins])
        q1 = np.array([np.quantile(v[m], 0.25) if m.any() else np.nan for _, _, m in bins])
        q3 = np.array([np.quantile(v[m], 0.75) if m.any() else np.nan for _, _, m in bins])
        ax.fill_between(centers, q1, q3, color=col, alpha=0.16, linewidth=0)
        ax.plot(centers, med, "o-", color=col, ms=6, lw=2, label=name)
        if show_reference and np.isfinite(med[0]) and np.isfinite(centers[0]):
            ref = med[0] * (centers / centers[0]) ** -0.5
            ax.plot(centers, ref, ls=(0, (1, 3)), color=col, lw=1.4, alpha=0.85)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("total trials (log scale)")
    ax.set_ylabel("posterior SD — median, IQR band (log scale)")
    ax.set_title("Posterior uncertainty shrinks with data")

    top = ax.get_ylim()[1]
    for c, n in zip(centers, counts):
        if np.isfinite(c):
            ax.text(c, top, f"{n}", ha="center", va="bottom", fontsize=7.2 * scale,
                    color=MUTE, clip_on=False)
    ax.text(0.0, 1.055, "matrices per bin", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=7.6 * scale, color=MUTE)

    from matplotlib.lines import Line2D
    handles, _ = ax.get_legend_handles_labels()
    if show_reference:
        handles.append(Line2D([], [], ls=(0, (1, 3)), color=MUTE, lw=1.4,
                              label=r"$n^{-1/2}$ reference"))
    ax.legend(handles=handles, fontsize=9 * scale)
    if regime:
        fig.text(0.02, 0.965, regime, ha="left", va="top", fontsize=10 * scale, color=MUTE)
    fig.tight_layout(rect=[0, 0, 1, 0.945 if regime else 1.0])
    fig.savefig(path); plt.close(fig)


def speed_accuracy(npe_ms, mle_ms, npe_mae, mle_mae, path, scale=1.0):
    """Headline: amortized network vs maximum likelihood on speed and accuracy."""
    set_style(scale)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    labels = ["GRIN (NPE)", "MLE"]
    ax[0].bar(labels, [npe_ms, mle_ms], color=[BLUE, MUTE], width=0.6)
    ax[0].set_yscale("log"); ax[0].set_ylabel("time per matrix (ms, log)")
    ax[0].set_title(f"Speed  —  {mle_ms/npe_ms:,.0f}× faster")
    for i, v in enumerate([npe_ms, mle_ms]):
        ax[0].text(i, v, f" {v:.3g} ms", ha="center", va="bottom", fontsize=9.5*scale, color=INK)
    ax[1].bar(labels, [npe_mae, mle_mae], color=[BLUE, MUTE], width=0.6)
    ax[1].set_ylabel("mean absolute error"); ax[1].set_title("Accuracy  —  parameter recovery")
    for i, v in enumerate([npe_mae, mle_mae]):
        ax[1].text(i, v, f" {v:.2f}", ha="center", va="bottom", fontsize=9.5*scale, color=INK)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def speed_accuracy_multi(labels, ms, maes, path, title_speed=None, title_acc=None,
                         ylabel_acc="mean absolute error", scale=1.0):
    """Speed and accuracy across N methods (generalises speed_accuracy's two-bar version).

    Speed is log-scaled because the methods differ by orders of magnitude; accuracy is not,
    because the differences there are small and a log axis would exaggerate them. The first
    label is treated as the method under test and coloured accordingly; the rest are
    baselines.
    """
    set_style(scale)
    labels = list(labels)
    ms = np.asarray(ms, dtype=float); maes = np.asarray(maes, dtype=float)
    if not (len(labels) == len(ms) == len(maes)):
        raise ValueError(f"length mismatch: {len(labels)} labels, {len(ms)} times, "
                         f"{len(maes)} errors")
    # Colour by METHOD FAMILY (the token before " ("), not by bar position: "GRIN
    # (batched)" and "GRIN (1 matrix)" are the same method measured two ways, and giving
    # the second one a baseline colour makes it read as a competitor.
    fam_order, fam_of = [], {}
    for l in labels:
        f = l.split(" (")[0]
        if f not in fam_of:
            fam_of[f] = len(fam_order); fam_order.append(f)
    shades = [(BLUE_DEEP, BLUE), (MUTE, MUTE), (RED_DEEP, RED), (INK, MUTE)]
    seen = {}
    colors = []
    for l in labels:
        f = l.split(" (")[0]
        k = seen.get(f, 0); seen[f] = k + 1
        pair = shades[fam_of[f] % len(shades)]
        colors.append(pair[min(k, 1)])

    fs = max(1.0, 0.55 + 0.45 * scale)     # figure grows with the type, or labels collide
    fig, ax = plt.subplots(1, 2, figsize=(max(11.0, 3.6 * len(labels)) * fs, 4.8 * fs))
    ax[0].bar(labels, ms, color=colors, width=0.6)
    ax[0].set_yscale("log"); ax[0].set_ylabel("time per matrix (ms, log)")
    ax[0].set_title(title_speed or "Speed")
    for i, v in enumerate(ms):
        ax[0].text(i, v, f" {v:.3g} ms", ha="center", va="bottom", fontsize=9.5 * scale,
                   color=INK)
    ax[1].bar(labels, maes, color=colors, width=0.6)
    ax[1].set_ylabel(ylabel_acc)
    ax[1].set_title(title_acc or "Accuracy")
    for i, v in enumerate(maes):
        ax[1].text(i, v, f" {v:.3f}", ha="center", va="bottom", fontsize=9.5 * scale, color=INK)
    for a in ax:
        a.tick_params(axis="x", labelsize=9.5 * scale)
        a.set_xticks(range(len(labels)))
        a.set_xticklabels(labels, rotation=22, ha="right")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def _wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion. Behaves at k=0 and k=n."""
    if n <= 0:
        return np.nan, np.nan
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def speed_accuracy_tradeoff(labels, ms, ms_err, mae, mae_err, path, families=None,
                            title=None, annotate=True, scale=1.0):
    """Classic speed-accuracy trade-off: one point per method, time on x, error on y.

    Collapses the two-panel bar version into a single scatter without losing information:
    each method is a point at (time-per-matrix, MAE) with standard-error bars on both axes.
    The desirable corner is BOTTOM-LEFT (fast and accurate), which is stated on the figure
    so the geometry reads correctly at a glance.

    labels/ms/mae      : parallel sequences, one entry per method.
    ms_err/mae_err     : standard errors (same length); pass zeros to omit a bar.
    families           : optional list mapping each method to a family name, so variants of
                         one method (GRIN batched/single, MLE full/selected/penalised) share
                         a colour. Defaults to the token before " (".
    """
    set_style(scale)
    labels = list(labels)
    ms = np.asarray(ms, float); mae = np.asarray(mae, float)
    ms_err = np.asarray(ms_err, float); mae_err = np.asarray(mae_err, float)
    if families is None:
        families = [l.split(" (")[0] for l in labels]
    fam_order = []
    for f in families:
        if f not in fam_order:
            fam_order.append(f)
    palette = [BLUE_DEEP, RED_DEEP, MUTE, BLUE, RED, INK]
    fam_col = {f: palette[i % len(palette)] for i, f in enumerate(fam_order)}

    fig, ax = plt.subplots(figsize=(8.2 * max(1.0, 0.6 + 0.4 * scale),
                                    6.2 * max(1.0, 0.6 + 0.4 * scale)))
    for lab, x, y, xe, ye, fam in zip(labels, ms, mae, ms_err, mae_err, families):
        col = fam_col[fam]
        ax.errorbar(x, y, xerr=xe if xe > 0 else None, yerr=ye if ye > 0 else None,
                    fmt="o", ms=11, color=col, ecolor=col, elinewidth=1.4, capsize=3,
                    zorder=3)
        if annotate:
            ax.annotate(lab, (x, y), textcoords="offset points", xytext=(9, 6),
                        fontsize=9.5 * scale, color=INK)
    # Orientation: fast = LEFT (small ms, conventional), accurate = UP (small MAE). Both
    # axes therefore INVERTED relative to raw values, so the desirable corner is TOP-LEFT.
    # Inverting is cleaner than negating the data because the tick labels stay honest.
    ax.set_xscale("log"); ax.invert_xaxis(); ax.invert_yaxis()
    ax.set_xlabel(r"time per matrix (ms, log)  $\leftarrow$ faster")
    ax.set_ylabel(r"mean absolute error  $\leftarrow$ more accurate")
    ax.set_title(title or "Speed vs accuracy")
    ax.annotate("better", xy=(0.055, 0.93), xycoords="axes fraction",
                fontsize=12 * scale, color=MUTE, fontweight="bold", ha="left", va="center")
    ax.annotate("", xy=(0.02, 0.985), xytext=(0.11, 0.90), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color=MUTE, lw=1.8))
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def construct_probabilities(results, truth, true_maxrho, path, threshold=0.5, regime="",
                            min_per_bin=15, frontier_step=0.05, scale=1.0):
    """Are the per-construct probabilities honest, and where does PI become identifiable?

    Ground truth comes from MODEL_SPECS via _model_constructs(), NOT from substring tests on
    the class name. The previous version asked things like `"psa" in name` and `"ps_" in
    name`, which mislabels: pi_psa_ds and rho1_psa_ds both have PS(A) but do not contain
    "psa" as a leading token, and "ps_" matches pi_ps_ds only incidentally. Model structure
    is declared in MODEL_SPECS; it should be read there.

    Reliability (left): marker area scales with bin count and every bin carries a Wilson
    95% interval, so a point resting on 12 matrices cannot masquerade as one resting on 1200.

    Frontier (right): PI cases have rho EXACTLY 0. They are not a point on a
    correlation-magnitude axis, they are a different condition, so they are drawn as a
    separate marker and NOT joined to the curve -- connecting them implies a continuum
    through zero that does not exist. The remaining bins are quantile-based and plotted at
    the mean |rho| they actually contain, not at invented x positions.
    """
    set_style(scale)
    truth = np.asarray(truth, dtype=object)
    true_maxrho = np.asarray(true_maxrho, dtype=float)
    tc, tx, ty = _model_constructs(truth)

    p_pi = np.array([r["p_PI"] for r in results], dtype=float)
    p_a = np.array([r["p_sep_A"] for r in results], dtype=float)
    p_b = np.array([r["p_sep_B"] for r in results], dtype=float)

    fs = max(1.0, 0.55 + 0.45 * scale)
    fig, ax = plt.subplots(1, 2, figsize=(13 * fs, 5.6 * fs))

    # ---------------- reliability ----------------
    specs = [(p_pi, tc == 0, "independence (PI)", RED_DEEP),
             (p_a, tx == 1, "separability A", BLUE),
             (p_b, ty == 1, "separability B", BLUE_DEEP)]
    edges = np.linspace(0, 1, 11)
    for probs, holds, lab, col in specs:
        xs, ys, los, his, ns = [], [], [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (probs >= lo) & (probs < hi if hi < 1 else probs <= hi)
            n = int(m.sum())
            if n < min_per_bin:
                continue
            k = int(holds[m].sum())
            l, h = _wilson(k, n)
            xs.append(probs[m].mean()); ys.append(k / n)
            los.append(l); his.append(h); ns.append(n)
        if not xs:
            continue
        xs, ys, ns = map(np.array, (xs, ys, ns))
        err = np.vstack([np.clip(ys - np.array(los), 0, None),
                         np.clip(np.array(his) - ys, 0, None)])
        ax[0].errorbar(xs, ys, yerr=err, fmt="none", ecolor=col, elinewidth=1.0,
                       capsize=2, alpha=0.7)
        ax[0].plot(xs, ys, "-", color=col, lw=1.8, label=lab)
        ax[0].scatter(xs, ys, s=12 + 34 * np.sqrt(ns / ns.max()), color=col,
                      edgecolors="white", linewidth=0.6, zorder=3)
    ax[0].plot([0, 1], [0, 1], ls=(0, (4, 3)), color=MUTE)
    ax[0].set_xlabel("predicted probability"); ax[0].set_ylabel("empirical frequency")
    ax[0].set_title("Probability calibration")
    ax[0].set_xlim(0, 1); ax[0].set_ylim(0, 1); ax[0].set_box_aspect(1)
    ax[0].legend(fontsize=9 * scale)
    ax[0].set_xlabel("predicted probability\n"
                     r"marker area $\propto$ bin count; bars = Wilson 95%")

    # ---------------- PI identifiability frontier ----------------
    called_pi = p_pi > threshold
    is_pi = tc == 0
    correct = called_pi == is_pi

    n_pi = int(is_pi.sum())
    if n_pi:
        k = int(correct[is_pi].sum())
        lo, hi = _wilson(k, n_pi)
        ax[1].errorbar([0.0], [k / n_pi], yerr=[[max(k / n_pi - lo, 0)], [max(hi - k / n_pi, 0)]],
                       fmt="D", color=MUTE, ms=8, capsize=3, elinewidth=1.1,
                       label=fr"true PI ($\rho \equiv 0$), $n$={n_pi}")

    nz = ~is_pi
    xs, ys, los, his, ns = [], [], [], [], []
    if nz.sum():
        # FIXED-WIDTH bins in |rho|, not quantiles. Quantile bins put the x positions
        # wherever the data happen to be dense, which makes the frontier's SHAPE an
        # artefact of the sampling. Even steps mean the curve's steepness is readable as
        # steepness. Bins below min_per_bin are dropped rather than drawn thin.
        hi_edge = float(np.nanmax(true_maxrho[nz])) if nz.any() else frontier_step
        qs = np.arange(0.0, hi_edge + frontier_step, frontier_step)
        for lo_e, hi_e in zip(qs[:-1], qs[1:]):
            m = nz & (true_maxrho >= lo_e) & (true_maxrho < hi_e)
            n = int(m.sum())
            if n < min_per_bin:
                continue
            k = int(correct[m].sum())
            l, h = _wilson(k, n)
            xs.append(true_maxrho[m].mean()); ys.append(k / n)
            los.append(l); his.append(h); ns.append(n)
    if xs:
        xs, ys = np.array(xs), np.array(ys)
        err = np.vstack([np.clip(ys - np.array(los), 0, None),
                         np.clip(np.array(his) - ys, 0, None)])
        ax[1].errorbar(xs, ys, yerr=err, fmt="o-", color=RED_DEEP, lw=2.0, ms=5.0,
                       capsize=2, elinewidth=0.9,
                       label=fr"true non-PI, {frontier_step:g} steps in $|\rho|$")
        ax[1].axhline(0.5, color=MUTE, lw=1.2, ls=(0, (1, 3)))
        ax[1].text(0.99, 0.505, "chance", transform=ax[1].get_yaxis_transform(),
                   ha="right", va="bottom", fontsize=8 * scale, color=MUTE)
    ax[1].set_xlabel(r"true correlation magnitude $|\rho|$")
    ax[1].set_ylabel("PI call accuracy")
    ax[1].set_title("PI identifiability frontier")
    ax[1].set_ylim(0, 1.04); ax[1].set_box_aspect(1)
    ax[1].legend(fontsize=8.5 * scale, loc="upper left", framealpha=0.9)
    ax[1].set_xlabel(ax[1].get_xlabel() +
                     f"\ndecision rule: call PI when $p > {threshold:g}$")

    fig.suptitle("Construct probabilities", x=0.02, ha="left", fontweight="bold",
                 fontsize=15 * scale, color=INK)
    if regime:
        fig.text(0.02, 0.945, regime, ha="left", va="top", fontsize=10 * scale, color=MUTE)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(path); plt.close(fig)


def model_comparison_figure(acc_amort, acc_bic, frontier_x, frontier_acc, speedup, path, scale=1.0):
    """Amortized comparison head vs AIC/BIC: per-construct accuracy + the PI frontier."""
    set_style(scale)
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5))
    labels = ["correlation\n(PI/RHO1/free)", "separable A", "separable B"]
    xpos = np.arange(3); w = 0.38
    ax[0].bar(xpos - w/2, acc_amort, w, color=BLUE, label=f"amortized ({speedup:,.0f}× faster)")
    ax[0].bar(xpos + w/2, acc_bic, w, color=MUTE, label="AIC/BIC")
    ax[0].set_xticks(xpos); ax[0].set_xticklabels(labels)
    ax[0].set_ylabel("accuracy"); ax[0].set_ylim(0, 1)
    ax[0].set_title("Amortized comparison vs. gold-standard selection"); ax[0].legend()
    for i, (a, b) in enumerate(zip(acc_amort, acc_bic)):
        ax[0].text(i - w/2, a, f"{a:.2f}", ha="center", va="bottom", fontsize=8.5*scale)
        ax[0].text(i + w/2, b, f"{b:.2f}", ha="center", va="bottom", fontsize=8.5*scale)
    ax[1].plot(frontier_x, frontier_acc, "o-", color=RED_DEEP, lw=2.4)
    ax[1].set_xlabel("true correlation magnitude |ρ|"); ax[1].set_ylabel("PI call accuracy")
    ax[1].set_title("Where the amortization gap lives — the PI frontier")
    ax[1].set_ylim(0, 1.02); ax[1].set_box_aspect(1)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def rt_gain_figure(metrics, frontier, path, scale=1.0):
    """RT-augmented vs counts-only: construct accuracy + the shifted PI frontier.
    metrics: dict with 'labels', 'counts', 'rt'   frontier: dict x, counts, rt."""
    set_style(scale)
    fig, ax = plt.subplots(1, 2, figsize=(12.8, 5))
    x = np.arange(len(metrics["labels"])); w = 0.38
    ax[0].bar(x - w/2, metrics["counts"], w, color=MUTE, label="counts only")
    ax[0].bar(x + w/2, metrics["rt"], w, color=BLUE, label="+ response times")
    ax[0].set_xticks(x); ax[0].set_xticklabels(metrics["labels"])
    ax[0].set_ylabel("accuracy"); ax[0].set_ylim(0, 1)
    ax[0].set_title("Response times sharpen the hard constructs"); ax[0].legend()
    for i, (a, b) in enumerate(zip(metrics["counts"], metrics["rt"])):
        ax[0].text(i - w/2, a, f"{a:.2f}", ha="center", va="bottom", fontsize=8.5*scale)
        ax[0].text(i + w/2, b, f"{b:.2f}", ha="center", va="bottom", fontsize=8.5*scale)
    ax[1].plot(frontier["x"], frontier["counts"], "o--", color=MUTE, lw=2, label="counts only")
    ax[1].plot(frontier["x"], frontier["rt"], "o-", color=RED_DEEP, lw=2.4, label="+ response times")
    ax[1].set_xlabel("true correlation magnitude |ρ|"); ax[1].set_ylabel("PI call accuracy")
    ax[1].set_title("The PI identifiability frontier moves")
    ax[1].set_ylim(0, 1.02); ax[1].legend(); ax[1].set_box_aspect(1)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def architecture_figure(cm_norm, arch_names, gains, path, regime="", scale=1.0):
    """Processing-architecture recovery, plus the honest single-RT gain.

    The gain panel is SPLIT by unit. The previous version put "1 - MAE" and two accuracies
    on one 0-1 axis under the label "accuracy / 1-MAE", which invites reading three
    different quantities off one scale. Accuracies now sit together with their chance
    levels marked; rho recovery is shown as MAE on its own axis, where lower is better and
    no arithmetic has been done to make it point the same way as the accuracies.

    gains: {label: (counts_only_value, plus_rt_value)}. Any key containing "MAE" (case
    insensitive) is routed to the error panel; everything else is treated as an accuracy.
    Chance for a key is taken from `gains[key][2]` if a third element is present.
    """
    set_style(scale)
    K = len(arch_names)
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5), width_ratios=[1.25, 1, 0.8])

    im = ax[0].imshow(cm_norm, cmap=CMAP_SEQ, vmin=0, vmax=1)
    short = [a_.replace("_", "\n") for a_ in arch_names]
    ax[0].set_xticks(range(K)); ax[0].set_yticks(range(K))
    ax[0].set_xticklabels(short, fontsize=8 * scale)
    ax[0].set_yticklabels(short, fontsize=8 * scale)
    ax[0].set_xlabel("inferred"); ax[0].set_ylabel("true")
    bal = float(np.trace(cm_norm)) / K
    ax[0].set_title(f"Processing architecture (balanced acc = {bal:.2f}, chance = {1/K:.2f})")
    for i in range(K):
        for j in range(K):
            if cm_norm[i, j] > .01:
                ax[0].text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                           fontsize=8.5 * scale,
                           color="white" if cm_norm[i, j] > .55 else INK)
    despine_heatmap(ax[0])
    clean_colorbar(fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04), "proportion")

    acc_keys = [k for k in gains if "mae" not in k.lower()]
    err_keys = [k for k in gains if "mae" in k.lower()]

    def paired(axis, keys, ylabel, title, clamp01):
        if not keys:
            axis.set_visible(False); return
        x = np.arange(len(keys)); w = 0.38
        b = [gains[k][0] for k in keys]; r = [gains[k][1] for k in keys]
        axis.bar(x - w / 2, b, w, color=MUTE, label="counts only")
        axis.bar(x + w / 2, r, w, color=BLUE, label="+ single RT")
        for i, k in enumerate(keys):
            for off, v in ((-w / 2, gains[k][0]), (w / 2, gains[k][1])):
                if np.isfinite(v):
                    axis.text(i + off, v, f"{v:.2f}", ha="center", va="bottom",
                              fontsize=8 * scale)
            if len(gains[k]) > 2 and np.isfinite(gains[k][2]):
                axis.hlines(gains[k][2], i - 0.45, i + 0.45, color=INK, lw=1.2,
                            ls=(0, (3, 3)))
        axis.set_xticks(x); axis.set_xticklabels(keys, fontsize=9 * scale)
        axis.set_ylabel(ylabel); axis.set_title(title)
        if clamp01:
            axis.set_ylim(0, 1.08)
        axis.legend(fontsize=9 * scale)

    paired(ax[1], acc_keys, "accuracy", "A single RT: the honest gain", True)
    paired(ax[2], err_keys, "MAE (lower is better)", r"$\rho$ recovery", False)

    fig.suptitle("Processing architecture and what one RT buys", x=0.02, ha="left",
                 fontweight="bold", fontsize=15 * scale, color=INK)
    if regime:
        fig.text(0.02, 0.945, regime, ha="left", va="top", fontsize=10 * scale, color=MUTE)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(path); plt.close(fig)


def lba_recovery(true, pred, names, path, regime="", scale=1.0):
    """LBA parameter recovery — one panel per LBA parameter.

    Shares the recovery-figure conventions: a square frame with identical x and y limits so
    the identity line is a true diagonal, MAE reported alongside r (r alone is blind to
    shrinkage, which is the failure mode that matters for an amortized posterior mean), and
    limits taken from the union of true and predicted so nothing is cropped silently.
    """
    set_style(scale)
    true = np.asarray(true, dtype=float); pred = np.asarray(pred, dtype=float)
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(3.9 * n, 4.3))
    axes = np.atleast_1d(axes)
    for j, (ax, nm) in enumerate(zip(axes, names)):
        t, q = true[:, j], pred[:, j]
        ok = np.isfinite(t) & np.isfinite(q)
        ax.scatter(t[ok], q[ok], s=10, c=BLUE, alpha=0.35, edgecolors="none")
        lo = float(min(t[ok].min(), q[ok].min())); hi = float(max(t[ok].max(), q[ok].max()))
        pad = 0.04 * (hi - lo or 1.0)
        lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], color=RED_DEEP, lw=1.6, ls=(0, (4, 3)), zorder=3)
        r = np.corrcoef(t[ok], q[ok])[0, 1] if ok.sum() > 2 else np.nan
        ax.text(0.05, 0.95, f"r = {r:.2f}\nMAE = {np.abs(t[ok] - q[ok]).mean():.3f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=9 * scale, color=INK)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_box_aspect(1)
        ax.set_title(nm); ax.set_xlabel("true")
        if j == 0:
            ax.set_ylabel("recovered")
    fig.suptitle("LBA parameter recovery", x=0.02, ha="left", fontweight="bold",
                 fontsize=15 * scale, color=INK)
    if regime:
        fig.text(0.02, 0.94, regime, ha="left", va="top", fontsize=10 * scale, color=MUTE)
    fig.tight_layout(rect=[0, 0, 1, 0.92]); fig.savefig(path); plt.close(fig)
