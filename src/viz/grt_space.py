"""
grt_space.py — the classic GRT perceptual-space plot, in matplotlib.

Static/publication counterpart to web/assets/js/grt-plot.js's renderSpace(): per-
stimulus bivariate-normal ellipses (unit variance, correlation rho_i) centred at
(zx_i, zy_i), decision bounds fixed at 0 (the decisional-separability convention
documented in grt_model.py), and optional marginal-density strips for reading
perceptual separability directly off the figure. Same colour CONVENTION as the
web version -- solid vs. dashed means the same thing -- but drawn from the GRIN
static-figure palette (style.py) rather than the web's own theme, so this sits
visually consistent with the rest of the paper/poster figure suite:

    SOLID, one colour per stimulus  = the reference structure (a group template,
                                       or ground truth)
    DASHED, single colour           = the comparison overlay (a recovered
                                       estimate, or an individual's own structure)

Stimulus order is grt_model.STIMULUS_ORDER: s0=A1B1, s1=A1B2, s2=A2B1, s3=A2B2.
This module is intentionally framework-agnostic (no BayesFlow import) -- it draws
whatever 12-vector it is given, whether that came from the bespoke NPE stack or
from bayesflow_port. Port-specific composites (individual-vs-group, attention
scalars) live in bayesflow_port/grt_figures.py and import the primitives here.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from scipy.stats import norm

from .style import set_style, BLUE, RED, BLUE_DEEP, RED_DEEP, INK, MUTE

try:
    from src import grt_model as gm
except ImportError:
    import grt_model as gm

STIM_PALETTE = [BLUE, RED, BLUE_DEEP, RED_DEEP]   # one per stimulus, s0..s3
PREDICTED_COLOR = INK                              # dashed / comparison overlay


def _unpack(theta):
    """Accepts a 12-vector (grt_model canonical order), a (zx,zy,rho) tuple of
    length-4 arrays, or a dict with those keys. Returns (zx, zy, rho), each (4,)."""
    if isinstance(theta, dict):
        return (np.asarray(theta["zx"], float), np.asarray(theta["zy"], float),
                np.asarray(theta["rho"], float))
    arr = np.asarray(theta, dtype=float)
    if arr.ndim == 1 and arr.shape[0] == 12:
        return gm.unpack(arr)
    if arr.shape == (3, 4):
        return arr[0], arr[1], arr[2]
    raise ValueError(f"cannot interpret theta of shape {arr.shape} as a GRT template "
                     "(expected a 12-vector or a (zx,zy,rho) triple of length-4 arrays)")


def _ellipse_patch(cx, cy, rho, color, dashed=False, lw=None):
    """1-SD equal-density contour of a unit-variance bivariate normal with
    correlation rho -- eigen-decomposition of [[1,r],[r,1]] gives semi-axes
    sqrt(1+r), sqrt(1-r) at 45 degrees. Matches grt-plot.js's drawEllipse exactly."""
    r = float(np.clip(rho, -0.98, 0.98))
    width = 2 * np.sqrt(1 + r)
    height = 2 * np.sqrt(1 - r)
    return Ellipse((cx, cy), width=width, height=height, angle=45,
                   facecolor="none", edgecolor=color,
                   linewidth=lw or (1.5 if dashed else 2.1),
                   linestyle=(0, (5, 4)) if dashed else "solid", zorder=3)


def _axis_limit(zx, zy, rho, predicted=None, min_lim=3.0, pad=1.25):
    xs, ys, rs = [zx], [zy], [rho]
    if predicted is not None:
        pzx, pzy, prho = _unpack(predicted)
        xs.append(pzx); ys.append(pzy); rs.append(prho)
    x = np.concatenate(xs); y = np.concatenate(ys); r = np.abs(np.concatenate(rs))
    radius = np.sqrt(1 + r)
    reach = np.concatenate([np.abs(x) + radius, np.abs(y) + radius])
    return max(min_lim, float(np.nanmax(reach)) * pad)


def shared_axis_limit(thetas, min_lim=3.0, pad=1.25):
    """Axis limit that fits every theta in `thetas` (12-vectors, or (zx,zy,rho)
    triples) -- for a small-multiples grid, where every panel must share one
    scale or between-panel size differences read as an auto-zoom artefact
    instead of a genuine between-participant difference."""
    lims = [_axis_limit(*_unpack(t), min_lim=min_lim, pad=pad) for t in thetas]
    return max(lims)


def perceptual_space(ax, theta, predicted_theta=None, palette=None,
                      predicted_color=None, show_bounds=True, show_points=True,
                      show_level_ticks=True, title=None, lim=None):
    """Draw ONE perceptual-space panel on an existing Axes -- the composable
    primitive used both standalone (via perceptual_space_figure) and inside
    small-multiple grids (bayesflow_port.grt_figures.individual_grid_figure).

    theta: the SOLID reference structure (a group template or ground truth).
    predicted_theta: optional, same shape -- drawn DASHED, overlaid on the same
        axes (a recovered estimate, or an individual's own attention-scaled
        structure, depending on caller).
    """
    palette = palette or STIM_PALETTE
    predicted_color = predicted_color or PREDICTED_COLOR
    zx, zy, rho = _unpack(theta)

    if show_bounds:
        ax.axvline(0, color=MUTE, lw=1.2, ls=(0, (5, 4)), zorder=1)
        ax.axhline(0, color=MUTE, lw=1.2, ls=(0, (5, 4)), zorder=1)

    for i in range(4):
        ax.add_patch(_ellipse_patch(zx[i], zy[i], rho[i], palette[i % len(palette)]))
        if show_points:
            ax.plot(zx[i], zy[i], "o", color=palette[i % len(palette)], ms=4.5, zorder=4)

    if predicted_theta is not None:
        pzx, pzy, prho = _unpack(predicted_theta)
        for i in range(4):
            ax.add_patch(_ellipse_patch(pzx[i], pzy[i], prho[i], predicted_color, dashed=True))
            if show_points:
                ax.plot(pzx[i], pzy[i], "o", color=predicted_color, ms=4.0, zorder=5,
                        markerfacecolor="none", markeredgewidth=1.3)

    lim = lim if lim is not None else _axis_limit(zx, zy, rho, predicted_theta)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_box_aspect(1)
    ax.set_xticks([]); ax.set_yticks([])

    if show_level_ticks:
        tick_kw = dict(fontsize=8.5, color=MUTE)
        ax.text(-lim * 0.97, lim * 0.04, "A1", ha="left", va="bottom", **tick_kw)
        ax.text(lim * 0.97, lim * 0.04, "A2", ha="right", va="bottom", **tick_kw)
        ax.text(lim * 0.04, lim * 0.97, "B2", ha="left", va="top", **tick_kw)
        ax.text(lim * 0.04, -lim * 0.97, "B1", ha="left", va="bottom", **tick_kw)

    if title:
        ax.set_title(title, fontsize=11)
    return ax


def _marginal_curve(ax, means, level_of_this_dim, level_of_other_dim, group_colors,
                    orientation):
    """One marginal-density strip (grtools convention): colour = level of the
    dimension being drawn, linestyle = level of the OTHER dimension (solid for
    its first level, dotted for its second). If the solid and dotted curves of
    the SAME colour do not coincide, that dimension is not separable from the
    other -- this is what the strip exists to show at a glance."""
    z = np.linspace(-3.6, 3.6, 240)
    dens = norm.pdf(z)
    for i in range(4):
        color = group_colors[int(level_of_this_dim[i])]
        dotted = int(level_of_other_dim[i]) == 1
        curve_pos = means[i] + z
        style = dict(color=color, lw=1.4 if dotted else 2.0,
                     linestyle="dotted" if dotted else "solid")
        if orientation == "x":
            ax.plot(curve_pos, dens, **style)
        else:
            ax.plot(dens, curve_pos, **style)


def _legend_handles(palette, predicted_color, stimulus_labels, predicted_label):
    handles = [Line2D([0], [0], color=palette[i % len(palette)], lw=2.2,
                      label=stimulus_labels[i]) for i in range(4)]
    if predicted_label:
        handles.append(Line2D([0], [0], color=predicted_color, lw=1.6,
                              linestyle=(0, (5, 4)), label=predicted_label))
    return handles


def perceptual_space_figure(theta, path, predicted_theta=None, show_marginals=True,
                            title=None, stimulus_labels=None, predicted_label=None,
                            palette=None, predicted_color=None, scale=1.0):
    """A complete, standalone classic-GRT-perceptual-space figure, saved to path.

    theta / predicted_theta: see perceptual_space(). predicted_label, if given
    (e.g. "recovered", "individual"), adds a legend distinguishing solid from
    dashed; stimulus_labels defaults to grt_model.STIMULUS_ORDER.
    """
    set_style(scale)
    palette = palette or STIM_PALETTE
    predicted_color = predicted_color or PREDICTED_COLOR
    labels = stimulus_labels or gm.STIMULUS_ORDER
    zx, zy, rho = _unpack(theta)

    if show_marginals:
        fig = plt.figure(figsize=(7.6 * max(1.0, 0.6 + 0.4 * scale),
                                  7.6 * max(1.0, 0.6 + 0.4 * scale)))
        gs = fig.add_gridspec(2, 2, width_ratios=[4.4, 1], height_ratios=[1, 4.4],
                              wspace=0.06, hspace=0.06)
        ax_top = fig.add_subplot(gs[0, 0])
        ax_main = fig.add_subplot(gs[1, 0])
        ax_right = fig.add_subplot(gs[1, 1])
        ax_corner = fig.add_subplot(gs[0, 1]); ax_corner.axis("off")

        perceptual_space(ax_main, theta, predicted_theta=predicted_theta, palette=palette,
                         predicted_color=predicted_color, show_level_ticks=True)
        lim = ax_main.get_xlim()[1]

        mc = [palette[0], palette[2]]
        _marginal_curve(ax_top, zx, gm.A_LEVEL, gm.B_LEVEL, mc, orientation="x")
        ax_top.set_xlim(-lim, lim); ax_top.set_xticks([]); ax_top.set_yticks([])
        for s in ("top", "right", "left"):
            ax_top.spines[s].set_visible(False)
        ax_top.set_title("dimension A marginals", fontsize=9, color=MUTE, pad=2)

        _marginal_curve(ax_right, zy, gm.B_LEVEL, gm.A_LEVEL, mc, orientation="y")
        ax_right.set_ylim(-lim, lim); ax_right.set_xticks([]); ax_right.set_yticks([])
        for s in ("top", "right", "bottom"):
            ax_right.spines[s].set_visible(False)
        ax_right.set_ylabel("dimension B marginals", fontsize=9, color=MUTE)
        ax_right.yaxis.set_label_position("right")

        ax_main.set_xlabel("dimension A"); ax_main.set_ylabel("dimension B")
        # the blank corner cell (gs[0,1]) is otherwise unused -- a legend anchored
        # to ax_main here would land underneath ax_right's density curves instead.
        ax_corner.legend(handles=_legend_handles(palette, predicted_color, labels,
                                                 predicted_label),
                         fontsize=8.5 * scale, loc="center", frameon=False)
    else:
        fig, ax_main = plt.subplots(figsize=(6.0 * max(1.0, 0.6 + 0.4 * scale),
                                             6.0 * max(1.0, 0.6 + 0.4 * scale)))
        perceptual_space(ax_main, theta, predicted_theta=predicted_theta, palette=palette,
                         predicted_color=predicted_color)
        ax_main.set_xlabel("dimension A"); ax_main.set_ylabel("dimension B")
        ax_main.legend(handles=_legend_handles(palette, predicted_color, labels,
                                               predicted_label),
                       fontsize=8.5 * scale, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                       frameon=False)

    fig.suptitle(title or "Perceptual space", x=0.02, ha="left", fontweight="bold",
                fontsize=15 * scale, color=INK)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
