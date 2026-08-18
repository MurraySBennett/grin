"""
plot.py: reporting and visualisation for GRIN results, at both the individual
and group level. Mirrors grin (the R package)'s plotting suite function-for-
function -- a figure made with grintools.plot and one made with grin should
read as the same family.

Kept out of grintools/__init__.py deliberately: the core package is
numpy + onnxruntime only (torch-free, dependency-light by design), and this
module needs matplotlib + pandas + scipy. Install with the [plot] extra:

    pip install grintools[plot]
    import grintools as gt
    import grintools.plot as gtplot

    result, constructs = gt.infer(M)
    gtplot.plot_space(result)

Default style is black-on-white -- publication-safe, greyscale-safe, no
colour-vision assumptions. A small set of named colour palettes is opt-in:
pass `palette="name"` to any plot function (see `palette_names()`), or your
own list of hex colours, or set `grintools.plot.DEFAULT_PALETTE = "name"`
once to change the default everywhere. Either way, a single
stimulus/participant space never gets a four-way colour split -- with
everything else fixed, four colours for four points is decoration, not
information; stimuli are told apart by a text label instead.

`tidy()` is the shared foundation: it turns one or many (result, constructs)
pairs -- e.g. from `[gt.infer(m) for m in matrices]` -- into a pandas
DataFrame, which every group-level plot builds on.

Every function returns a plain matplotlib Axes (or Figure, for the
multi-panel ones). If a parameter below doesn't cover what you need, that is
deliberate rather than an oversight: the returned object is an ordinary
matplotlib object and takes ordinary matplotlib calls
(`ax = gtplot.plot_space(result); ax.set_title("...")`), which covers far
more than this package could ever anticipate as an argument. See the
package README's "Editing a figure further" section for worked examples.
"""
import numpy as np

try:
    import matplotlib.pyplot as plt
    import pandas as pd
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "grintools.plot needs matplotlib and pandas. Install with: "
        "pip install grintools[plot]") from e

INK, MUTE, PAPER = "#2B2B2E", "#9AA0A6", "#FFFFFF"
STIM = ["A1B1", "A1B2", "A2B1", "A2B2"]

# --------------------------------------------------------------------------- #
# Named colour palettes. "mono" (INK only) is the default and needs no entry
# here. "contrast" is the colour-vision-deficiency-safe categorical palette of
# Okabe & Ito (2008). Hex values match grin (the R package)'s palettes exactly,
# so a figure made with either package is visually the same family.
# --------------------------------------------------------------------------- #
PALETTES = {
    "contrast": ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"],
    "dusk": ["#0B3954", "#12678A", "#1C9FC9", "#6FD6E8"],
    "ember": ["#4A0E0E", "#9E2B25", "#D9622B", "#F2A65A"],
}

#: Default for every plot function's `palette` argument. Set
#: `grintools.plot.DEFAULT_PALETTE = "name"` once instead of passing
#: `palette="name"` to every call; an explicit `palette=` argument always wins.
DEFAULT_PALETTE = "mono"


def palette_names():
    """List the built-in palette names: "mono" (default) plus PALETTES's keys."""
    return ["mono"] + list(PALETTES)


def _resolve_palette(palette):
    """`palette`: None (defer to DEFAULT_PALETTE), a built-in name, a single hex
    string, or a list of hex colours supplied directly by the caller."""
    if palette is None:
        palette = DEFAULT_PALETTE
    if isinstance(palette, (list, tuple)):
        return list(palette)
    if isinstance(palette, str) and palette.startswith("#"):
        return [palette]
    if palette == "mono":
        return [INK]
    if palette in PALETTES:
        return PALETTES[palette]
    raise ValueError(f"unknown palette {palette!r}; use one of {palette_names()}, "
                     "or pass your own list of hex colours")


def _group_colors(n, palette):
    """n category colours: the resolved palette, repeated/cycled to length n --
    so a mono plot never carries a legend distinguishing "black" from "black"."""
    cols = _resolve_palette(palette)
    if len(cols) == 1:
        return cols * n
    if n <= len(cols):
        return cols[:n]
    reps = -(-n // len(cols))
    return (cols * reps)[:n]


def _style(ax, base_size=12):
    """Clean axes: left+bottom spines only, no gridlines, ink-coloured ticks,
    font sizes scaled off `base_size` (title/labels/ticks)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(colors=INK, labelsize=base_size - 1)
    ax.xaxis.label.set_color(INK); ax.xaxis.label.set_fontsize(base_size)
    ax.yaxis.label.set_color(INK); ax.yaxis.label.set_fontsize(base_size)
    ax.title.set_color(INK); ax.title.set_fontsize(base_size + 2); ax.title.set_fontweight("bold")
    ax.set_facecolor(PAPER)
    return ax


# --------------------------------------------------------------------------- #
# Forward model: identified GRT parameters -> predicted response probabilities.
# Used only for reporting/diagnostics (plot_diagnostics()'s predicted-vs-
# observed panel) -- inference itself never runs this, the trained network
# does. Identical to the Sheppard r-integration used to build the training
# data (see the "Software description" section of the manuscript and
# src/grt_model.py in the main GRIN repo), so a diagnostic plot's "predicted"
# values are computed the same way the network was taught to invert.
# --------------------------------------------------------------------------- #
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(48)


def _norm_cdf(x):
    from scipy.stats import norm
    return norm.cdf(x)


def _bvn_cdf(h, k, rho):
    """Bivariate normal CDF Phi2(h, k; rho), standard normal margins."""
    h = np.asarray(h, dtype=float); k = np.asarray(k, dtype=float); rho = np.asarray(rho, dtype=float)
    base = _norm_cdf(h) * _norm_cdf(k)
    t = rho[..., None] * (_GL_NODES + 1.0) / 2.0
    jac = rho[..., None] / 2.0
    hh = h[..., None]; kk = k[..., None]
    omt2 = 1.0 - t * t
    dens = np.exp(-(hh * hh - 2.0 * t * hh * kk + kk * kk) / (2.0 * omt2)) / (2.0 * np.pi * np.sqrt(omt2))
    return base + np.sum(_GL_WEIGHTS * dens * jac, axis=-1)


def _forward_probabilities(zx, zy, rho):
    """Per-stimulus (zx, zy, rho) -> 4x4 predicted response probabilities
    (rows = stimuli, cols = responses, canonical order)."""
    zx = np.asarray(zx, dtype=float); zy = np.asarray(zy, dtype=float); rho = np.asarray(rho, dtype=float)
    p_x1 = _norm_cdf(-zx); p_y1 = _norm_cdf(-zy)
    p11 = _bvn_cdf(-zx, -zy, rho)
    p12 = p_x1 - p11; p21 = p_y1 - p11; p22 = 1.0 - p_x1 - p_y1 + p11
    return np.clip(np.stack([p11, p12, p21, p22], axis=-1), 0.0, 1.0)


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
    12 parameter estimates and their SDs, p_PI/p_sep_A/p_sep_B, the
    evidence_* flags, and x_bias/y_bias (the decision-criterion bias of
    response_bias()).
    """
    from .io import response_bias as _response_bias
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
        bias = _response_bias(result)
        row["x_bias"] = bias["x_bias"]
        row["y_bias"] = bias["y_bias"]
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


def _marginal_strip(means, rng, orientation, col, base_size, ax=None):
    """A small axes with one Normal(mean, 1) density curve per stimulus, over
    `rng`. `orientation="y"` flips it for use as a right-side vertical strip."""
    ax = ax or plt.subplots(figsize=(2, 5) if orientation == "y" else (5, 2))[1]
    xs = np.linspace(rng[0], rng[1], 200)
    for m in means:
        d = np.exp(-0.5 * (xs - m) ** 2) / np.sqrt(2 * np.pi)
        if orientation == "y":
            ax.plot(d, xs, color=col)
            ax.set_ylim(rng)
        else:
            ax.plot(xs, d, color=col)
            ax.set_xlim(rng)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor(PAPER)
    return ax


# --------------------------------------------------------------------------- #
# Individual-level plots
# --------------------------------------------------------------------------- #
def plot_space(result, ci=0.90, palette=None, title=None, xlabel="dimension A (zx)",
               ylabel="dimension B (zy)", stim_labels=STIM, show_labels=True,
               show_uncertainty=True, show_marginals=False, base_size=12, ax=None):
    """One participant's perceptual space: 4 stimulus means with correlation
    ellipses, decision bounds at 0 (GRIN's identified coordinates). Stimuli
    are told apart by a text label at each point by default, not by colour --
    with one participant on the plot, four colours for four points is
    decoration, not information, and the quadrant is already fixed by the
    sign convention.

    The ellipse and the (optional) error bars show two different kinds of
    uncertainty and the plot never conflates them: the ellipse is the
    predicted spread of a single trial's perceptual sample around the mean
    (fixed at unit variance by the model, shaped by rho); the error bars, if
    shown, are the *posterior* uncertainty about where that mean itself is,
    given the data (`result.std`).

    `show_marginals=True` adds per-dimension marginal density strips (one
    curve per stimulus) above and to the right of the main panel, and returns
    a Figure instead of an Axes; incompatible with passing your own `ax`. See
    also `plot_diagnostics()`, which pairs marginals with a
    predicted-vs-observed reconstruction panel for a fuller view (not a
    goodness-of-fit test of GRT itself -- see that function's docstring).
    """
    if len(stim_labels) != 4:
        raise ValueError("stim_labels must have exactly 4 entries")
    if show_marginals and ax is not None:
        raise ValueError("show_marginals=True builds its own multi-panel figure; "
                         "pass ax=None (the default)")
    k = _ci_k(ci)
    col = _group_colors(1, palette)[0]
    p = dict(zip(result.names, result.params))
    s = dict(zip(result.names, result.std))

    main_ax = ax or plt.subplots(figsize=(6, 6), constrained_layout=(not show_marginals))[1]
    main_ax.axvline(0, ls="--", color=MUTE, lw=1)
    main_ax.axhline(0, ls="--", color=MUTE, lw=1)
    zxs, zys = [], []
    for i, stim in enumerate(stim_labels):
        zx, zy, rho = p[f"zx_{i}"], p[f"zy_{i}"], p[f"rho_{i}"]
        zxs.append(zx); zys.append(zy)
        if show_uncertainty:
            sx, sy = s[f"zx_{i}"], s[f"zy_{i}"]
            main_ax.plot([zx - k * sx, zx + k * sx], [zy, zy], color=col, lw=1.2, zorder=2)
            main_ax.plot([zx, zx], [zy - k * sy, zy + k * sy], color=col, lw=1.2, zorder=2)
        ex, ey = _ellipse_xy(zx, zy, rho, k)
        main_ax.plot(ex, ey, color=col)
        main_ax.scatter([zx], [zy], color=col, s=50, zorder=3)
        if show_labels:
            main_ax.annotate(stim, (zx, zy), color=col, fontsize=base_size - 3, ha="center",
                             va="bottom", xytext=(0, 8), textcoords="offset points")
    if xlabel is not None:
        main_ax.set_xlabel(xlabel)
    if ylabel is not None:
        main_ax.set_ylabel(ylabel)
    unc_note = " and error bars" if show_uncertainty else ""
    default_title = f"Perceptual space ({result.model_class})"
    if not show_marginals:
        subtitle = f"{ci:.0%} ellipses{unc_note}; dashed lines are the decision bounds"
        full_title = (default_title if title is None else title) + f"\n{subtitle}"
        # a single Text object (title \n subtitle), not a separate floating
        # annotation, and the DEFAULT loc (not loc="left"): ax.set_title(...)
        # on the returned Axes -- also default loc -- then cleanly replaces
        # the whole thing, rather than adding a second title next to an
        # orphaned one at a different loc.
        main_ax.set_title(full_title, fontsize=base_size + 1)
    main_ax.set_aspect("equal")
    _style(main_ax, base_size)

    if not show_marginals:
        return main_ax

    ex_all = np.concatenate([_ellipse_xy(zxs[i], zys[i], p[f"rho_{i}"], k)[0] for i in range(4)])
    ey_all = np.concatenate([_ellipse_xy(zxs[i], zys[i], p[f"rho_{i}"], k)[1] for i in range(4)])
    x_rng = (ex_all.min(), ex_all.max()); y_rng = (ey_all.min(), ey_all.max())
    main_ax.set_xlim(x_rng); main_ax.set_ylim(y_rng)

    fig = main_ax.figure
    fig.set_size_inches(7.5, 7.5)
    gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                          left=0.1, right=0.95, bottom=0.08, top=0.9, wspace=0.05, hspace=0.05)
    main_ax.set_position(gs[1, 0].get_position(fig))
    main_ax.set_subplotspec(gs[1, 0])
    top_ax = fig.add_subplot(gs[0, 0], sharex=main_ax)
    right_ax = fig.add_subplot(gs[1, 1], sharey=main_ax)
    _marginal_strip(zxs, x_rng, "x", col, base_size, ax=top_ax)
    _marginal_strip(zys, y_rng, "y", col, base_size, ax=right_ax)
    fig.suptitle(title if title else f"Perceptual space ({result.model_class})", fontsize=base_size + 2)
    return fig


def plot_params(result, palette=None, title=None, param_labels=None, base_size=12, ax=None):
    """One participant's 12 parameter estimates with 90% CIs (dot-and-whisker).
    Position already separates the zx/zy/rho groups, so colour is optional.
    `param_labels`: relabel the y-axis (defaults to result.names)."""
    labels = list(param_labels) if param_labels is not None else list(result.names)
    if len(labels) != len(result.names):
        raise ValueError("param_labels must be the same length as result.names (12)")
    ax = ax or plt.subplots(figsize=(6, 5))[1]
    groups = ["zx"] * 4 + ["zy"] * 4 + ["rho"] * 4
    cols = dict(zip(["zx", "zy", "rho"], _group_colors(3, palette)))
    y = np.arange(len(labels))[::-1]
    ax.axvline(0, ls="--", color=MUTE, lw=1)
    for yi, est, lo, hi, grp in zip(y, result.params, result.ci_low, result.ci_high, groups):
        ax.plot([lo, hi], [yi, yi], color=cols[grp], lw=1.5)
        ax.scatter([est], [yi], color=cols[grp], s=40, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("estimate (90% CI)")
    ax.set_title("Parameter estimates" if title is None else title)
    return _style(ax, base_size)


def plot_constructs(result, constructs, palette=None, title=None, base_size=12, ax=None):
    """One participant's construct probabilities: correlation structure
    (P(PI)/P(RHO1)/P(free)) and separability (P(sep A)/P(sep B)). Bars for a
    construct the data can't decide (evidence_* is False) are flagged, not
    silently plotted as if informative."""
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    names = ["PI", "RHO1", "free", "sep A", "sep B"]
    probs = list(constructs["p_corr"]) + [constructs["p_sep_A"], constructs["p_sep_B"]]
    evidence = [constructs["evidence_PI"]] * 3 + [constructs["evidence_sep_A"], constructs["evidence_sep_B"]]
    panel_colors = _group_colors(2, palette)
    colors = [panel_colors[0]] * 3 + [panel_colors[1]] * 2
    alphas = [1.0 if e else 0.35 for e in evidence]
    bars = ax.bar(names, probs, color=colors)
    for one_bar, a, ev in zip(bars, alphas, evidence):
        one_bar.set_alpha(a)
        if not ev:
            ax.text(one_bar.get_x() + one_bar.get_width() / 2, one_bar.get_height() + 0.02,
                    "insufficient\nevidence", ha="center", va="bottom", fontsize=base_size - 4, color=MUTE)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("P(construct)")
    ax.set_title(f"Construct probabilities ({result.model_class})" if title is None else title)
    return _style(ax, base_size)


def plot_bias(result, ci=0.90, palette=None, title=None, dim_labels=("A", "B"),
             base_size=12, ax=None):
    """Decision-criterion response bias for one participant, with error bars
    carrying its (approximate) uncertainty forward. See
    grintools.response_bias(). For the model-free alternative, see
    plot_empirical_bias()."""
    from .io import response_bias
    if len(dim_labels) != 2:
        raise ValueError("dim_labels must have exactly 2 entries")
    b = response_bias(result)
    k = _ci_k(ci)
    col = _group_colors(1, palette)[0]
    ax = ax or plt.subplots(figsize=(4, 5))[1]
    ax.axhline(0, ls="--", color=MUTE, lw=1)
    ax.bar(dim_labels, [b["x_bias"], b["y_bias"]], color=col, width=0.5)
    ax.errorbar(dim_labels, [b["x_bias"], b["y_bias"]],
               yerr=[k * b["x_bias_se"], k * b["y_bias_se"]],
               fmt="none", ecolor=INK, capsize=4)
    ax.set_ylabel("decision-criterion bias (mean z-score)")
    subtitle = (f"{ci:.0%} CI; 0 = unbiased; positive favours level 2, "
               "negative favours level 1")
    ax.set_title((("Response bias" if title is None else title)) + f"\n{subtitle}",
                fontsize=base_size + 1)
    return _style(ax, base_size)


def plot_empirical_bias(counts, trials=None, palette=None, title=None, dim_labels=("A", "B"),
                        base_size=12, ax=None):
    """Empirical response bias for one participant: how far each dimension's
    "respond level 2" rate sits from the unbiased 0.5, averaged across the
    four stimuli. Works directly from a confusion matrix -- no gt.infer()
    call needed. See grintools.empirical_bias(). For the model-based
    decision-criterion alternative, see plot_bias()."""
    from .io import empirical_bias
    if len(dim_labels) != 2:
        raise ValueError("dim_labels must have exactly 2 entries")
    b = empirical_bias(counts, trials)
    col = _group_colors(1, palette)[0]
    ax = ax or plt.subplots(figsize=(4, 5))[1]
    ax.axhline(0, ls="--", color=MUTE, lw=1)
    ax.bar(dim_labels, [b["x_bias"], b["y_bias"]], color=col, width=0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_ylabel("response bias  (P(respond level 2) - 0.5)")
    subtitle = "0 = unbiased; positive favours level 2, negative favours level 1"
    ax.set_title((("Empirical response bias" if title is None else title)) + f"\n{subtitle}",
                fontsize=base_size + 1)
    return _style(ax, base_size)


def plot_diagnostics(result, counts, trials=None, show_predicted_observed=True,
                     show_marginals=True, palette=None, title=None, base_size=12):
    """Predicted-vs-observed reconstruction and marginal-distribution
    diagnostics for one participant. Needs the ORIGINAL confusion matrix as
    well as the fitted result, because it compares what was observed against
    what the fitted parameters predict -- gt.infer()'s return value alone
    doesn't carry the input matrix back out.

    Deliberately not a "goodness-of-fit" test: the identified 12-parameter
    model is saturated (see the accompanying paper's Introduction and
    identifiability-frontier study), so a single confusion matrix's response
    proportions cannot, in principle, be used to test whether the underlying
    GRT assumptions hold -- essentially any proportion table has SOME fitting
    parameter vector. What this view shows is whether GRIN's OWN fitted
    parameters reconstruct the matrix, informative in one direction only: a
    poor reconstruction is a real signal (network approximation error, or a
    matrix outside the trained envelope) worth a second look, but a good
    reconstruction does not itself validate the GRT assumptions, because the
    saturated model was essentially guaranteed to reconstruct it regardless.

    `show_predicted_observed`: the forward model's predicted response
    probability for each of the 16 stimulus/response cells, plotted against
    the cell's observed proportion. Points near the diagonal indicate a good
    reconstruction; systematic departure for one stimulus (told apart by
    marker shape, not colour) says where the fit is struggling and is worth a
    second look.

    `show_marginals`: the predicted Normal(mean, 1) density on each dimension
    for each of the four stimuli -- the same marginals plot_space()'s
    `show_marginals=True` draws alongside the space plot itself, here paired
    with the reconstruction check instead.

    Returns a single Axes if only one panel is requested, otherwise a Figure.
    """
    if not show_predicted_observed and not show_marginals:
        raise ValueError("nothing to plot: set show_predicted_observed and/or show_marginals=True")
    cm = np.asarray(counts, dtype=float).reshape(4, 4)
    if trials is None:
        trials = cm.sum(axis=1)
    trials = np.asarray(trials, dtype=float).reshape(4)
    observed = cm / trials[:, None]

    p = dict(zip(result.names, result.params))
    zx = np.array([p[f"zx_{i}"] for i in range(4)])
    zy = np.array([p[f"zy_{i}"] for i in range(4)])
    rho = np.array([p[f"rho_{i}"] for i in range(4)])
    col = _group_colors(1, palette)[0]

    n_panels = int(show_predicted_observed) + int(show_marginals) * 2
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.5))
    axes = np.atleast_1d(axes)
    idx = 0

    if show_predicted_observed:
        predicted = _forward_probabilities(zx, zy, rho)
        ax = axes[idx]; idx += 1
        markers = ["o", "^", "s", "D"]
        ax.plot([0, 1], [0, 1], ls="--", color=MUTE, lw=1)
        for i, stim in enumerate(STIM):
            ax.scatter(observed[i], predicted[i], color=col, marker=markers[i], s=45, label=stim)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xlabel("observed proportion"); ax.set_ylabel("predicted probability")
        ax.set_title("Predicted vs. observed")
        ax.legend(title="stimulus", fontsize=base_size - 4, title_fontsize=base_size - 3)
        _style(ax, base_size)

    if show_marginals:
        x_rng = (zx.min() - 3, zx.max() + 3); y_rng = (zy.min() - 3, zy.max() + 3)
        _marginal_strip(zx, x_rng, "x", col, base_size, ax=axes[idx])
        axes[idx].set_title("dimension A marginals"); axes[idx].set_xlabel("zx")
        _style(axes[idx], base_size); idx += 1
        _marginal_strip(zy, y_rng, "x", col, base_size, ax=axes[idx])
        axes[idx].set_title("dimension B marginals"); axes[idx].set_xlabel("zy")
        _style(axes[idx], base_size)

    fig.tight_layout()
    if title:
        fig.suptitle(title, fontsize=base_size + 2)
        fig.subplots_adjust(top=0.85)
    return axes[0] if n_panels == 1 else fig


# --------------------------------------------------------------------------- #
# Group-level plots
# --------------------------------------------------------------------------- #
def plot_space_group(results, ids=None, facet=True, ci=0.90, palette=None, title=None,
                     base_size=12, ncols=4):
    """Many participants' perceptual spaces: one small-multiple panel each
    (facet=True), or an overlay of all stimulus means + a single
    across-participant mean ellipse per stimulus (facet=False).

    facet=False is an EXPLORATORY INSPECTION VIEW ONLY, not a reporting
    figure: GRT's perceptual space is defined per observer, and there is no
    sense in which a "grand mean" ellipse over several independently-fitted
    spaces is itself a fitted GRT model, or a quantity with the
    calibrated-uncertainty guarantees gt.infer() gives an individual
    estimate. Use it to eyeball whether a sample looks roughly homogeneous
    before deciding how to report it properly (per participant, or via
    plot_params_group() / plot_model_classes()) -- not as the figure itself.
    A warning is printed each time this mode is used, as a standing reminder
    rather than a one-off you might miss.
    """
    df = tidy(results, ids)
    k = _ci_k(ci)
    col = _group_colors(1, palette)[0]
    n = len(df)

    if facet:
        ncols_ = min(ncols, n)
        nrows = int(np.ceil(n / ncols_))
        fig, axes = plt.subplots(nrows, ncols_, figsize=(3.2 * ncols_, 3.2 * nrows), squeeze=False)
        for i, (_, row) in enumerate(df.iterrows()):
            ax = axes[i // ncols_][i % ncols_]
            ax.axvline(0, ls="--", color=MUTE, lw=0.8)
            ax.axhline(0, ls="--", color=MUTE, lw=0.8)
            for si, stim in enumerate(STIM):
                zx, zy, rho = row[f"zx_{si}"], row[f"zy_{si}"], row[f"rho_{si}"]
                ex, ey = _ellipse_xy(zx, zy, rho, k, n=60)
                ax.plot(ex, ey, color=col, lw=1)
                ax.scatter([zx], [zy], color=col, s=20, zorder=3)
                ax.annotate(stim, (zx, zy), color=col, fontsize=6, ha="center", va="bottom",
                           xytext=(0, 4), textcoords="offset points")
            ax.set_title(row["id"], fontsize=10)
            ax.set_aspect("equal")
            _style(ax, base_size)
        for j in range(n, nrows * ncols_):
            axes[j // ncols_][j % ncols_].axis("off")
        fig.suptitle("Perceptual spaces by participant" if title is None else title)
        fig.tight_layout()
        return fig

    import warnings
    warnings.warn(
        "plot_space_group(facet=False): exploratory inspection view only -- the "
        "overlaid mean ellipse is not a fitted GRT model or a reporting figure. "
        "See the plot_space_group() docstring.", stacklevel=2)
    ax = plt.subplots(figsize=(6, 6), constrained_layout=True)[1]
    ax.axvline(0, ls="--", color=MUTE, lw=1)
    ax.axhline(0, ls="--", color=MUTE, lw=1)
    for si, stim in enumerate(STIM):
        zxs, zys, rhos = df[f"zx_{si}"], df[f"zy_{si}"], df[f"rho_{si}"]
        ax.scatter(zxs, zys, color=col, s=20, alpha=0.3)
        mx, my, mrho = zxs.mean(), zys.mean(), rhos.mean()
        ex, ey = _ellipse_xy(mx, my, mrho, k)
        ax.plot(ex, ey, color=col, lw=2)
        ax.scatter([mx], [my], color=col, s=80, marker="D", zorder=3)
        ax.annotate(stim, (mx, my), color=col, fontsize=9, ha="center", va="bottom",
                   xytext=(0, 10), textcoords="offset points")
    ax.set_xlabel("zx"); ax.set_ylabel("zy")
    default_title = "Perceptual space, group overlay (exploratory only)"
    subtitle = (f"faint points = individuals; diamonds + {ci:.0%} ellipse = "
               "across-participant mean -- inspection view, not a fitted model")
    ax.set_title((default_title if title is None else title) + f"\n{subtitle}",
                fontsize=base_size + 1)
    ax.set_aspect("equal")
    return _style(ax, base_size)


def plot_params_group(results, ids=None, palette=None, title=None, base_size=12):
    """Boxplot of each parameter's estimate across many participants,
    faceted by group (zx/zy/rho)."""
    df = tidy(results, ids)
    long = _long_params(df)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    cols = dict(zip(["zx", "zy", "rho"], _group_colors(3, palette)))
    for ax, grp in zip(axes, ["zx", "zy", "rho"]):
        sub = long[long["group"] == grp]
        names = [f"{grp}_{i}" for i in range(4)]
        data = [sub.loc[sub["param"] == nm, "estimate"].values for nm in names]
        bp = ax.boxplot(data, tick_labels=names, patch_artist=True)
        for box in bp["boxes"]:
            box.set_facecolor(cols[grp]); box.set_alpha(0.6)
        ax.axhline(0, ls="--", color=MUTE, lw=1)
        ax.set_title(grp)
        _style(ax, base_size)
    fig.suptitle(f"Parameter distributions across participants (n={len(df)})" if title is None else title)
    fig.tight_layout()
    return fig


def plot_model_classes(results, ids=None, palette=None, title=None, base_size=12, ax=None):
    """Bar chart of the inferred GRT model class across many participants."""
    df = tidy(results, ids)
    counts = df["model_class"].value_counts()
    col = _group_colors(1, palette)[0]
    ax = ax or plt.subplots(figsize=(7, 4.5))[1]
    ax.bar(counts.index, counts.values, color=col)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.02 * max(counts.values), str(v), ha="center", color=INK)
    ax.set_ylabel("participants")
    ax.set_title(f"Inferred model class (n={len(df)})" if title is None else title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return _style(ax, base_size)


def plot_precision_group(results, ids=None, palette=None, title=None, base_size=12, ax=None):
    """Boxplot of posterior SD across many participants, by parameter group."""
    df = tidy(results, ids)
    long = _long_params(df)
    ax = ax or plt.subplots(figsize=(6, 4.5))[1]
    groups = ["zx", "zy", "rho"]
    cols = dict(zip(groups, _group_colors(3, palette)))
    data = [long.loc[long["group"] == g, "sd"].values for g in groups]
    bp = ax.boxplot(data, tick_labels=groups, patch_artist=True)
    for box, g in zip(bp["boxes"], groups):
        box.set_facecolor(cols[g])
        box.set_alpha(0.6)
    ax.set_ylabel("posterior SD")
    ax.set_title(f"Precision across participants (n={len(df)})" if title is None else title)
    return _style(ax, base_size)


def plot_bias_group(results, ids=None, palette=None, title=None, base_size=12, ax=None):
    """Group-level companion to plot_bias(): one boxplot per dimension of
    response_bias(), computed per participant from their fitted result. For
    the model-free alternative, see plot_empirical_bias_group()."""
    df = tidy(results, ids)
    col = _group_colors(1, palette)[0]
    ax = ax or plt.subplots(figsize=(4, 5))[1]
    ax.axhline(0, ls="--", color=MUTE, lw=1)
    data = [df["x_bias"].to_numpy(), df["y_bias"].to_numpy()]
    bp = ax.boxplot(data, tick_labels=["A", "B"], patch_artist=True, widths=0.5)
    for box in bp["boxes"]:
        box.set_facecolor(col); box.set_alpha(0.5)
    ax.set_ylabel("decision-criterion bias (mean z-score)")
    ax.set_title(f"Response bias across participants (n={len(df)})"
                if title is None else title)
    return _style(ax, base_size)


def plot_empirical_bias_group(counts_list, trials_list=None, palette=None, title=None,
                              base_size=12, ax=None):
    """Group-level companion to plot_empirical_bias(): one boxplot per
    dimension of empirical_bias() computed on each participant's own
    confusion matrix. Works directly from confusion matrices -- no
    gt.infer() call needed."""
    from .io import empirical_bias
    if trials_list is None:
        trials_list = [None] * len(counts_list)
    if len(trials_list) != len(counts_list):
        raise ValueError("trials_list must be the same length as counts_list")
    b = [empirical_bias(c, t) for c, t in zip(counts_list, trials_list)]
    col = _group_colors(1, palette)[0]
    ax = ax or plt.subplots(figsize=(4, 5))[1]
    ax.axhline(0, ls="--", color=MUTE, lw=1)
    data = [[bi["x_bias"] for bi in b], [bi["y_bias"] for bi in b]]
    bp = ax.boxplot(data, tick_labels=["A", "B"], patch_artist=True, widths=0.5)
    for box in bp["boxes"]:
        box.set_facecolor(col); box.set_alpha(0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_ylabel("response bias  (P(respond level 2) - 0.5)")
    ax.set_title(f"Empirical response bias across participants (n={len(counts_list)})"
                if title is None else title)
    return _style(ax, base_size)
