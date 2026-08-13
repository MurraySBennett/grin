"""
plot.py: reporting and visualisation for GRIN results, at both the individual
and group level. Mirrors grin (the R package)'s plotting suite function-for-
function, same visual identity (src/viz/style.py's palette) -- a figure made
with grintools.plot and one made with grin should read as the same family.

Kept out of grintools/__init__.py deliberately: the core package is
numpy + onnxruntime only (torch-free, dependency-light by design), and this
module needs matplotlib + pandas. Install with the [plot] extra:

    pip install grintools[plot]
    import grintools as gt
    import grintools.plot as gtplot

    result, constructs = gt.infer(M)
    gtplot.plot_space(result)

`tidy()` is the shared foundation: it turns one or many (result, constructs)
pairs -- e.g. from `[gt.infer(m) for m in matrices]` -- into a pandas
DataFrame, which every group-level plot builds on.
"""
import numpy as np

try:
    import matplotlib.pyplot as plt
    import pandas as pd
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "grintools.plot needs matplotlib and pandas. Install with: "
        "pip install grintools[plot]") from e

# --------------------------------------------------------------------------- #
# Palette -- ported from src/viz/style.py so package plots match the paper's.
# Duplicated rather than imported: grintools deliberately does not depend on
# the main research package (src/), so it can be pip-installed standalone.
# --------------------------------------------------------------------------- #
BLUE, BLUE_DEEP = "#5AA9E6", "#2E6CA4"
RED, RED_DEEP = "#F2A5C0", "#C86A93"
INK, MUTE, PAPER = "#2B2B2E", "#9AA0A6", "#FFFFFF"
PALETTE = [BLUE, RED, BLUE_DEEP, RED_DEEP, MUTE]
STIM = ["A1B1", "A1B2", "A2B1", "A2B2"]
STIM_COLORS = dict(zip(STIM, PALETTE[:4]))


def _style(ax):
    """Clean axes: left+bottom spines only, no gridlines, ink-coloured ticks."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(colors=INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)
    ax.set_facecolor(PAPER)
    return ax


# --------------------------------------------------------------------------- #
# Tidy layer
# --------------------------------------------------------------------------- #
def tidy(results, ids=None):
    """Turn one or many (result, constructs) pairs into a tidy DataFrame.

    `results`: a single (result, constructs) tuple, or a list of them (e.g.
    `[gt.infer(m) for m in matrices]`).
    `ids`: optional participant IDs, same length as results; defaults to
    p1, p2, ...

    Returns a pandas DataFrame, one row per participant: id, model_class, the
    12 parameter estimates and their SDs, p_PI/p_sep_A/p_sep_B, and the
    evidence_* flags.
    """
    if isinstance(results, tuple) and len(results) == 2 and hasattr(results[0], "names"):
        results = [results]
    if ids is None:
        ids = [f"p{i + 1}" for i in range(len(results))]
    if len(ids) != len(results):
        raise ValueError(f"len(ids)={len(ids)} != len(results)={len(results)}")

    rows = []
    for pid, (result, constructs) in zip(ids, results):
        row = {"id": pid, "model_class": result.model_class}
        for name, est, sd in zip(result.names, result.params, result.std):
            row[name] = float(est)
            row[f"{name}_sd"] = float(sd)
        row["p_PI"] = constructs["p_PI"]
        row["p_sep_A"] = constructs["p_sep_A"]
        row["p_sep_B"] = constructs["p_sep_B"]
        row["evidence_PI"] = constructs["evidence_PI"]
        row["evidence_sep_A"] = constructs["evidence_sep_A"]
        row["evidence_sep_B"] = constructs["evidence_sep_B"]
        rows.append(row)
    return pd.DataFrame(rows)


def _long_params(df):
    """Wide tidy DataFrame -> long (id, group, param, estimate, sd), one row
    per participant x parameter."""
    groups = {"zx": [f"zx_{i}" for i in range(4)], "zy": [f"zy_{i}" for i in range(4)],
             "rho": [f"rho_{i}" for i in range(4)]}
    out = []
    for grp, names in groups.items():
        for nm in names:
            out.append(pd.DataFrame({
                "id": df["id"], "group": grp, "param": nm,
                "estimate": df[nm], "sd": df[f"{nm}_sd"],
            }))
    return pd.concat(out, ignore_index=True)


def _ellipse_xy(zx, zy, rho, k, n=100):
    theta = np.linspace(0, 2 * np.pi, n)
    L = np.array([[1, 0], [rho, np.sqrt(max(1 - rho**2, 0))]])
    circle = np.stack([np.cos(theta), np.sin(theta)])
    xy = (L @ circle) * k
    return zx + xy[0], zy + xy[1]


def _ci_k(ci):
    from scipy.stats import norm
    return norm.ppf(0.5 + ci / 2)


# --------------------------------------------------------------------------- #
# Individual-level plots
# --------------------------------------------------------------------------- #
def plot_space(result, ci=0.90, ax=None):
    """One participant's perceptual space: 4 stimulus means with correlation
    ellipses, decision bounds at 0 (GRIN's identified coordinates)."""
    ax = ax or plt.subplots(figsize=(6, 6), constrained_layout=True)[1]
    k = _ci_k(ci)
    p = dict(zip(result.names, result.params))
    ax.axvline(0, ls="--", color=MUTE, lw=1)
    ax.axhline(0, ls="--", color=MUTE, lw=1)
    for i, s in enumerate(STIM):
        zx, zy, rho = p[f"zx_{i}"], p[f"zy_{i}"], p[f"rho_{i}"]
        ex, ey = _ellipse_xy(zx, zy, rho, k)
        ax.plot(ex, ey, color=STIM_COLORS[s], label=s)
        ax.scatter([zx], [zy], color=STIM_COLORS[s], s=50, zorder=3)
    ax.set_xlabel("dimension A (zx)"); ax.set_ylabel("dimension B (zy)")
    ax.set_title(f"Perceptual space ({result.model_class})")
    ax.set_aspect("equal")
    ax.legend(title="stimulus", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    return _style(ax)


def plot_params(result, ax=None):
    """One participant's 12 parameter estimates with 90% CIs (dot-and-whisker)."""
    ax = ax or plt.subplots(figsize=(6, 5))[1]
    groups = ["zx"] * 4 + ["zy"] * 4 + ["rho"] * 4
    colors = {"zx": PALETTE[0], "zy": PALETTE[1], "rho": PALETTE[2]}
    y = np.arange(len(result.names))[::-1]
    ax.axvline(0, ls="--", color=MUTE, lw=1)
    for yi, est, lo, hi, grp in zip(y, result.params, result.ci_low, result.ci_high, groups):
        ax.plot([lo, hi], [yi, yi], color=colors[grp], lw=1.5)
        ax.scatter([est], [yi], color=colors[grp], s=40, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(result.names)
    ax.set_xlabel("estimate (90% CI)")
    ax.set_title("Parameter estimates")
    return _style(ax)


def plot_constructs(result, constructs, ax=None):
    """One participant's construct probabilities: correlation structure
    (P(PI)/P(RHO1)/P(free)) and separability (P(sep A)/P(sep B)). Bars for a
    construct the data can't decide (evidence_* is False) are flagged, not
    silently plotted as if informative."""
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    names = ["PI", "RHO1", "free", "sep A", "sep B"]
    probs = list(constructs["p_corr"]) + [constructs["p_sep_A"], constructs["p_sep_B"]]
    evidence = [constructs["evidence_PI"]] * 3 + [constructs["evidence_sep_A"], constructs["evidence_sep_B"]]
    colors = [PALETTE[0]] * 3 + [PALETTE[1]] * 2
    alphas = [1.0 if e else 0.35 for e in evidence]
    bars = ax.bar(names, probs, color=colors)
    for bar, a, ev in zip(bars, alphas, evidence):
        bar.set_alpha(a)
        if not ev:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    "insufficient\nevidence", ha="center", va="bottom", fontsize=8, color=MUTE)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("P(construct)")
    ax.set_title(f"Construct probabilities ({result.model_class})")
    return _style(ax)


# --------------------------------------------------------------------------- #
# Group-level plots
# --------------------------------------------------------------------------- #
def plot_space_group(results, ids=None, facet=True, ci=0.90, ncols=4):
    """Many participants' perceptual spaces: one small-multiple panel each
    (facet=True), or an overlay of all stimulus means + a group-mean ellipse
    per stimulus (facet=False -- individual ellipses omitted; unreadable at
    scale, use facet=True to see individual uncertainty)."""
    df = tidy(results, ids)
    k = _ci_k(ci)
    n = len(df)

    if facet:
        ncols = min(ncols, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows), squeeze=False)
        for i, (_, row) in enumerate(df.iterrows()):
            ax = axes[i // ncols][i % ncols]
            ax.axvline(0, ls="--", color=MUTE, lw=0.8)
            ax.axhline(0, ls="--", color=MUTE, lw=0.8)
            for si, s in enumerate(STIM):
                zx, zy, rho = row[f"zx_{si}"], row[f"zy_{si}"], row[f"rho_{si}"]
                ex, ey = _ellipse_xy(zx, zy, rho, k, n=60)
                ax.plot(ex, ey, color=STIM_COLORS[s], lw=1)
                ax.scatter([zx], [zy], color=STIM_COLORS[s], s=20, zorder=3)
            ax.set_title(row["id"], fontsize=10)
            ax.set_aspect("equal")
            _style(ax)
        for j in range(n, nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")
        fig.suptitle("Perceptual spaces by participant")
        fig.tight_layout()
        return fig
    else:
        ax = plt.subplots(figsize=(6, 6), constrained_layout=True)[1]
        ax.axvline(0, ls="--", color=MUTE, lw=1)
        ax.axhline(0, ls="--", color=MUTE, lw=1)
        for si, s in enumerate(STIM):
            zxs, zys, rhos = df[f"zx_{si}"], df[f"zy_{si}"], df[f"rho_{si}"]
            ax.scatter(zxs, zys, color=STIM_COLORS[s], s=20, alpha=0.35)
            mx, my, mrho = zxs.mean(), zys.mean(), rhos.mean()
            ex, ey = _ellipse_xy(mx, my, mrho, k)
            ax.plot(ex, ey, color=STIM_COLORS[s], lw=2, label=s)
            ax.scatter([mx], [my], color=STIM_COLORS[s], s=80, marker="D", zorder=3)
        ax.set_xlabel("zx"); ax.set_ylabel("zy")
        ax.set_title("Perceptual space, group overlay")
        ax.set_aspect("equal")
        ax.legend(title="stimulus", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
        return _style(ax)


def plot_params_group(results, ids=None, ax=None):
    """Boxplot of each parameter's estimate across many participants,
    faceted by group (zx/zy/rho)."""
    df = tidy(results, ids)
    long = _long_params(df)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    colors = {"zx": PALETTE[0], "zy": PALETTE[1], "rho": PALETTE[2]}
    for ax, grp in zip(axes, ["zx", "zy", "rho"]):
        sub = long[long["group"] == grp]
        names = [f"{grp}_{i}" for i in range(4)]
        data = [sub.loc[sub["param"] == nm, "estimate"].values for nm in names]
        bp = ax.boxplot(data, tick_labels=names, patch_artist=True)
        for box in bp["boxes"]:
            box.set_facecolor(colors[grp]); box.set_alpha(0.6)
        ax.axhline(0, ls="--", color=MUTE, lw=1)
        ax.set_title(grp)
        _style(ax)
    fig.suptitle(f"Parameter distributions across participants (n={len(df)})")
    fig.tight_layout()
    return fig


def plot_model_classes(results, ids=None, ax=None):
    """Bar chart of the inferred GRT model class across many participants."""
    df = tidy(results, ids)
    counts = df["model_class"].value_counts()
    ax = ax or plt.subplots(figsize=(7, 4.5))[1]
    ax.bar(counts.index, counts.values, color=BLUE)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.02 * max(counts.values), str(v), ha="center", color=INK)
    ax.set_ylabel("participants")
    ax.set_title(f"Inferred model class (n={len(df)})")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return _style(ax)


def plot_precision_group(results, ids=None, ax=None):
    """Boxplot of posterior SD across many participants, by parameter group."""
    df = tidy(results, ids)
    long = _long_params(df)
    ax = ax or plt.subplots(figsize=(6, 4.5))[1]
    groups = ["zx", "zy", "rho"]
    data = [long.loc[long["group"] == g, "sd"].values for g in groups]
    bp = ax.boxplot(data, tick_labels=groups, patch_artist=True)
    for box, g in zip(bp["boxes"], groups):
        box.set_facecolor({"zx": PALETTE[0], "zy": PALETTE[1], "rho": PALETTE[2]}[g])
        box.set_alpha(0.6)
    ax.set_ylabel("posterior SD")
    ax.set_title(f"Precision across participants (n={len(df)})")
    return _style(ax)
