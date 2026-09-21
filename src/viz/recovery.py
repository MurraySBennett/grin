"""
recovery.py — the parameter-recovery family: one panel grid per method, plus the
cross-method summaries.

Every method gets the SAME panel geometry and the SAME axis limits so the figures can be
laid side by side and read as one comparison. Points are coloured by the model assumption
that is actually at stake in that panel (PS(A) in the zx row, PS(B) in the zy row,
correlation structure in the rho row) and, optionally, shaped by whether the method's own
model selection got that construct right.

Axis convention: z panels run to +/-3.3 with the prior boundary drawn at +/-Z_MAX, rho
panels to +/-1 with the boundary at +/-R_MAX. Fixing the frame across all panels is what
makes the sign convention visible — A1/B1 clouds sit in the negative half, A2/B2 in the
positive half — instead of each panel normalising that structure away.
"""
import numpy as np
import matplotlib.pyplot as plt

from .style import (set_style, BLUE, BLUE_DEEP, ROSE, ROSE_DEEP, INK, MUTE,
                    floating_trial_bands)

PRINT_W = 6.5   # \textwidth in GRIN_combined_edited.tex

try:
    from .. import grt_model as gm
except ImportError:
    import grt_model as gm

# ---------------------------------------------------------------- panel vocabulary
_PRETTY = {"zx": "$z_x$", "zy": "$z_y$", "rho": r"$\rho$"}

# hue: which construct is at stake in each row, and how its levels are coloured
_PS_COLORS = {1: BLUE, 0: ROSE_DEEP}
_PS_NAMES = {1: "holds", 0: "violated"}
_CORR_COLORS = {0: BLUE_DEEP, 1: BLUE, 2: ROSE_DEEP}
_CORR_NAMES = {0: r"PI ($\rho=0$)", 1: r"1$\rho$ (shared)", 2: r"free $\rho$"}

# shape: did this method's model selection get the construct right?
_MARK_OK, _MARK_BAD = "o", "X"

METHOD_COLORS = {"GRIN": BLUE_DEEP, "mdsdt": BLUE, "grtools": ROSE_DEEP,
                 "Python-MLE": MUTE}


def _pname(name):
    base, i = name.rsplit("_", 1)
    return f"{_PRETTY.get(base, base)}$_{{{i}}}$"


def _panel_family(j):
    """Parameter index -> ('zx'|'zy'|'rho')."""
    return "zx" if j < 4 else ("zy" if j < 8 else "rho")


def _mae(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return np.abs(a[m] - b[m]).mean() if m.any() else np.nan


# ---------------------------------------------------------------- the panel grid
def recovery_panels(true, pred, path, class_names, correct=None, method="GRIN",
                    regime="", z_max=3.0, r_max=0.9, z_pad=0.3, scale=1.0):
    """12-panel predicted-vs-true grid for ONE method.

    true, pred   : (N, 12) in canonical PARAM_NAMES order.
    class_names  : (N,) GRIN class name of the GENERATING model (ground truth) — drives hue.
    correct      : None            -> no shape coding (single-method showcase);
                   (N,) bool array -> 12-way exact-class correctness, same shapes everywhere;
                   dict            -> per-construct correctness with keys 'ps_x','ps_y','corr',
                                      each an (N,) bool array, applied to its own row.
    regime       : short string naming the evaluation set; goes in the suptitle, because a
                   recovery figure without its data regime on its face is not interpretable.
    """
    set_style(scale)
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    names = gm.PARAM_NAMES

    corr_t, psx_t, psy_t = _constructs(class_names)
    hue_by_row = {"zx": psx_t, "zy": psy_t, "rho": corr_t}

    if correct is None:
        ok_by_row = {"zx": None, "zy": None, "rho": None}
    elif isinstance(correct, dict):
        ok_by_row = {"zx": np.asarray(correct["ps_x"], dtype=bool),
                     "zy": np.asarray(correct["ps_y"], dtype=bool),
                     "rho": np.asarray(correct["corr"], dtype=bool)}
    else:
        g = np.asarray(correct, dtype=bool)
        ok_by_row = {"zx": g, "zy": g, "rho": g}

    fig, axes = plt.subplots(3, 4, figsize=(13, 9.8))
    n_dropped = 0

    for j, ax in enumerate(axes.ravel()):
        fam = _panel_family(j)
        t, p = true[:, j], pred[:, j]
        finite = np.isfinite(t) & np.isfinite(p)
        n_dropped = max(n_dropped, int((~finite).sum()))

        if fam == "rho":
            lim, bound = 1.0, r_max
            levels, colors, lnames = [0, 1, 2], _CORR_COLORS, _CORR_NAMES
        else:
            lim, bound = z_max + z_pad, z_max
            levels, colors, lnames = [1, 0], _PS_COLORS, _PS_NAMES

        hue, ok = hue_by_row[fam], ok_by_row[fam]

        # prior boundary + identity, drawn under the data
        for b in (-bound, bound):
            ax.axvline(b, color=MUTE, lw=0.8, ls=(0, (1, 3)), zorder=0)
            ax.axhline(b, color=MUTE, lw=0.8, ls=(0, (1, 3)), zorder=0)
        ax.plot([-lim, lim], [-lim, lim], color=INK, lw=1.2, ls=(0, (4, 3)),
                alpha=0.55, zorder=1)

        stats = []
        for lv in levels:                      # densest/least interesting level first
            sel = finite & (hue == lv)
            if not sel.any():
                continue
            c = colors[lv]
            if ok is None:
                ax.scatter(t[sel], p[sel], s=14, c=c, alpha=0.55,
                           edgecolors="none", zorder=2)
            else:
                for good, mark, alpha, size in ((True, _MARK_OK, 0.55, 14),
                                                (False, _MARK_BAD, 0.75, 20)):
                    m = sel & (ok == good)
                    if m.any():
                        ax.scatter(t[m], p[m], s=size, marker=mark, c=c, alpha=alpha,
                                   edgecolors="none", zorder=2 + int(not good))
            stats.append((lnames[lv], c, _mae(t[sel], p[sel])))

        ax.set_title(_pname(names[j]))
        y = 0.965
        for lab, c, v in stats:
            ax.text(0.045, y, f"MAE {v:.3f}", transform=ax.transAxes, va="top",
                    ha="left", fontsize=8.2 * scale, color=c)
            y -= 0.075
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_box_aspect(1)
        if j % 4 == 0:
            ax.set_ylabel("estimated")
        if j >= 8:
            ax.set_xlabel("true")

    _row_legends(axes, ok_by_row, scale)

    sub = f"{method}"
    if regime:
        sub += f" — {regime}"
    if n_dropped:
        sub += f"  ({n_dropped} non-finite estimates omitted)"
    fig.suptitle("Parameter recovery — estimated vs. true", x=0.02, ha="left",
                 fontweight="bold", fontsize=15 * scale, color=INK)
    fig.text(0.02, 0.955, sub, ha="left", va="top", fontsize=10.5 * scale, color=MUTE)
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    fig.savefig(path)
    plt.close(fig)


def _constructs(class_names):
    corr_idx = {"pi": 0, "rho1": 1, "free": 2}
    c, sx, sy = [], [], []
    for n in np.asarray(class_names, dtype=object):
        k, px, py = gm.MODEL_SPECS[n]
        c.append(corr_idx[k]); sx.append(int(px)); sy.append(int(py))
    return np.array(c), np.array(sx), np.array(sy)


def _row_legends(axes, ok_by_row, scale):
    """One hue legend per row (the construct at stake changes by row), plus one shape key."""
    from matplotlib.lines import Line2D

    def dot(c, lab, mark=_MARK_OK):
        return Line2D([], [], marker=mark, ls="none", color=c, ms=5.5, label=lab)

    rows = [("zx", "PS(A) ", [(1, _PS_COLORS[1], _PS_NAMES[1]), (0, _PS_COLORS[0], _PS_NAMES[0])]),
            ("zy", "PS(B) ", [(1, _PS_COLORS[1], _PS_NAMES[1]), (0, _PS_COLORS[0], _PS_NAMES[0])]),
            ("rho", "", [(k, _CORR_COLORS[k], _CORR_NAMES[k]) for k in (0, 1, 2)])]
    for r, (fam, prefix, spec) in enumerate(rows):
        handles = [dot(c, prefix + lab) for _, c, lab in spec]
        if ok_by_row[fam] is not None:
            handles += [dot(MUTE, "class correct", _MARK_OK),
                        dot(MUTE, "class wrong", _MARK_BAD)]
        axes[r, 0].legend(handles=handles, loc="lower right", fontsize=7.2 * scale,
                          handletextpad=0.35, borderpad=0.3, labelspacing=0.28)


# ---------------------------------------------------------------- summaries
def _boot_ci(vals, n_boot=2000, seed=0):
    """Percentile bootstrap CI of the mean. Returns (mean, lo, hi)."""
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan
    if v.size < 3:
        return v.mean(), v.mean(), v.mean()
    rng = np.random.default_rng(seed)
    bs = rng.choice(v, size=(n_boot, v.size), replace=True).mean(1)
    return v.mean(), np.quantile(bs, 0.025), np.quantile(bs, 0.975)


def _grouped_bars(ax, groups, methods, values, errs=None, ylabel="", title="",
                  connect=True):
    """values[method][group] -> grouped POINT plot with optional asymmetric intervals.

    Points rather than bars: the informative quantity is the level of each estimate, and
    a bar spends ink from zero up to it for no gain. Dodging within each group keeps the
    methods separable, and a faint connecting line per method carries the trend across
    groups, which grouped bars leave the eye to reconstruct.

    Name retained for backwards compatibility with existing callers.
    """
    x = np.arange(len(groups))
    # Narrower than the group spacing (1.0) by a good margin: at the old 0.8 span, four
    # methods' points sprawled most of the way to the NEXT tick's points, reading as an
    # undifferentiated cloud rather than four small clusters, each obviously "at" its own
    # x position.
    w = 0.42 / max(len(methods), 1)
    for i, m in enumerate(methods):
        off = (i - (len(methods) - 1) / 2) * w
        v = np.asarray(values[m], dtype=float)
        col = METHOD_COLORS.get(m, MUTE)
        if errs is not None:
            lo, hi = np.asarray(errs[m]).T
            ax.vlines(x + off, lo, hi, color=col, lw=1.3, alpha=0.85, zorder=2)
        if connect:
            ax.plot(x + off, v, "-", color=col, lw=1.0, alpha=0.35, zorder=1)
        ax.plot(x + off, v, "o", color=col, ms=5.0, zorder=3, label=m)
    ax.set_xticks(x)
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_ylim(bottom=0)
    if len(groups) > 5:
        ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=8)
    else:
        ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel); ax.set_title(title)


def summary_recovery(results, path, trial_bin_names=("low", "mid", "high"),
                     rho_bin_names=("PI", "weak", "mod", "strong"), scale=1.0):
    """Cross-method recovery summary.

    results: {method: {"true": (N,12), "pred": (N,12), "trial_bin": (N,), "rho_bin": (N,)}}
             All methods MUST be on the same rows, in the same order.
    """
    # Font scale reduced for this row: three \textwidth-tuned panels side by side are
    # only ~2.1in each, so the coded (full-scale) font sizes look outsized relative to
    # the plot area rather than merely "12-14pt as printed" -- see docs/figure_sizing.md.
    row_scale = scale * 0.68
    set_style(row_scale)
    methods = list(results)
    # Native width close to the manuscript's print width rather than the wide "poster"
    # figsize this used to carry -- see docs/figure_sizing.md.
    fig, ax = plt.subplots(1, 3, figsize=(PRINT_W, PRINT_W * 0.36))

    fams = {"$z_x$": slice(0, 4), "$z_y$": slice(4, 8), r"$\rho$": slice(8, 12)}
    vals, errs = {}, {}
    for m in methods:
        r = results[m]
        e = np.abs(r["pred"] - r["true"])
        vals[m] = [np.nanmean(e[:, sl]) for sl in fams.values()]
        errs[m] = [_boot_ci(np.nanmean(e[:, sl], axis=1))[1:] for sl in fams.values()]
    _grouped_bars(ax[0], list(fams), methods, vals, errs, "MAE",
                  "A   By family", connect=False)
    ax[0].legend(fontsize=8 * row_scale, loc="upper center", bbox_to_anchor=(0.5, -0.20),
                ncol=2, frameon=False)

    def _per_group(key, gnames, sl):
        vals, errs = {}, {}
        for m in methods:
            r = results[m]
            per_row = np.nanmean(np.abs(r["pred"][:, sl] - r["true"][:, sl]), axis=1)
            b = np.asarray(r[key])
            # Guard against the failure this function used to have silently: if the
            # caller passes fewer group names than there are bins in the data, the
            # loop below just skips the tail, and the last NAMED group gets read as
            # the top of the range when it is not. Fail loudly instead.
            if b.size and int(np.nanmax(b)) >= len(gnames):
                raise ValueError(
                    f"{key}: data has {int(np.nanmax(b)) + 1} bins but only "
                    f"{len(gnames)} names {tuple(gnames)} were given -- bins "
                    f"{len(gnames)}..{int(np.nanmax(b))} would be dropped silently.")
            vals[m], errs[m] = [], []
            for gi in range(len(gnames)):
                mu, lo, hi = _boot_ci(per_row[b == gi])
                vals[m].append(mu); errs[m].append((lo, hi))
        return vals, errs

    # Panel B (trial count) gets the floating-bar-over-true-numeric-width treatment
    # (see style.py::floating_trial_bands): "5-10" and "10-15" cover the same width but
    # a categorical axis drew them the same distance apart as "20-30", which covers
    # twice the range, and as "100-200", which covers twenty times the range.
    vals, errs = _per_group("trial_bin", trial_bin_names, slice(0, 12))
    series = {m: (vals[m], METHOD_COLORS.get(m, MUTE)) for m in methods}
    floating_trial_bands(ax[1], list(trial_bin_names), series, offset_span=0.16, err=errs)
    ax[1].set_xlabel("trials per stimulus")
    ax[1].set_ylabel("MAE (all params)")
    ax[1].set_ylim(bottom=0)
    ax[1].set_title("B   By trials")

    # Panel C's groups (PI/weak/mod/strong true |rho|) are categorical, not a numeric
    # range with a width to show -- floating bars don't apply here, unlike panel B.
    vals, errs = _per_group("rho_bin", rho_bin_names, slice(8, 12))
    _grouped_bars(ax[2], list(rho_bin_names), methods, vals, errs, r"MAE ($\rho$ only)",
                 r"C   $\rho$ by true $|\rho|$")

    # No figure-level title: that goes in the LaTeX caption, not baked into the PNG.
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.30, top=0.90, wspace=0.55)
    fig.savefig(path)
    plt.close(fig)


def summary_classification(truth_labels, method_labels, path, reference="GRIN", scale=1.0):
    """Per-construct + 12-way accuracy, and the correctness decomposition vs each baseline.

    truth_labels : (N,) generating GRIN class names.
    method_labels: {method: (N,) predicted GRIN class names, None allowed for failed fits}
    """
    set_style(scale)
    methods = list(method_labels)
    ct, sxt, syt = _constructs(truth_labels)

    def parsed(names):
        corr_idx = {"pi": 0, "rho1": 1, "free": 2}
        c, sx, sy, exact = [], [], [], []
        for n, tn in zip(np.asarray(names, dtype=object), np.asarray(truth_labels, dtype=object)):
            if n is None:
                c.append(-1); sx.append(-1); sy.append(-1); exact.append(False)
            else:
                k, px, py = gm.MODEL_SPECS[n]
                c.append(corr_idx[k]); sx.append(int(px)); sy.append(int(py))
                exact.append(n == tn)
        return np.array(c), np.array(sx), np.array(sy), np.array(exact)

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5))

    groups = ["PS(A)", "PS(B)", r"$\rho$ structure", "12-way exact"]
    vals, errs = {}, {}
    for m in methods:
        c, sx, sy, ex = parsed(method_labels[m])
        hits = [sx == sxt, sy == syt, c == ct, ex]
        vals[m] = [h.mean() for h in hits]
        errs[m] = [_boot_ci(h.astype(float))[1:] for h in hits]
    _grouped_bars(ax[0], groups, methods, vals, errs, "accuracy",
                  "Model classification accuracy")
    ax[0].set_ylim(0, 1.05); ax[0].legend(ncol=2)

    # decomposition: agreement is only interesting once you know who was RIGHT
    ref_ex = parsed(method_labels[reference])[3]
    others = [m for m in methods if m != reference]
    cats = [(f"both correct", BLUE_DEEP), (f"{reference} only", BLUE),
            ("baseline only", ROSE), ("both wrong", MUTE)]
    bottom = np.zeros(len(others))
    for ci, (lab, col) in enumerate(cats):
        heights = []
        for m in others:
            o_ex = parsed(method_labels[m])[3]
            frac = [(ref_ex & o_ex), (ref_ex & ~o_ex), (~ref_ex & o_ex), (~ref_ex & ~o_ex)][ci]
            heights.append(frac.mean())
        heights = np.asarray(heights)
        ax[1].bar(np.arange(len(others)), heights, 0.6, bottom=bottom, color=col, label=lab)
        for xi, (h, b) in enumerate(zip(heights, bottom)):
            if h > 0.06:
                ax[1].text(xi, b + h / 2, f"{h:.0%}", ha="center", va="center",
                           fontsize=8.5 * scale,
                           color="white" if col in (BLUE_DEEP, MUTE) else INK)
        bottom += heights
    ax[1].set_xticks(np.arange(len(others))); ax[1].set_xticklabels(others)
    ax[1].set_ylim(0, 1); ax[1].set_ylabel("fraction of matrices")
    ax[1].set_title(f"Where {reference} and each baseline agree — and who was right")
    ax[1].legend(fontsize=9 * scale)

    fig.suptitle("Model classification — method comparison", x=0.02, ha="left",
                 fontweight="bold", fontsize=15 * scale, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path)
    plt.close(fig)
