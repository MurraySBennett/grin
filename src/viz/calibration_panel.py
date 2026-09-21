"""
calibration_panel.py — the manuscript's calibration figure.

Three panels, portrait-friendly, built from scripts/calibration_breakdown.py's JSON:

  A  SBC rank histograms, z and rho overlaid as step outlines against the
     uniform band. Overlaying is the point: the two families depart from
     uniform in OPPOSITE directions, which a pooled histogram hides.
  B  Coverage curve, empirical against nominal, one line per family.
  C  90% coverage by trials per stimulus, as horizontal segments spanning each
     trial-count band's own width on a continuous log axis, rather than one point per
     categorical "5-10"/"10-15"/... label -- the segments make the (equal) coverage
     value visible over the (unequal) width of data each band actually represents, and
     a shared boundary (e.g. the 10 between "5-10" and "10-15") is then a single tick
     instead of printed twice.

No bar charts anywhere: dots/segments carry the estimate and the vertical rule carries
the Monte Carlo interval.

Saved as a combined 1x3 row (manuscript default), a combined 3x1 stack, and each panel
individually -- see docs/figure_sizing.md. The combined row uses a REDUCED scale
(ROW_SCALE below): matching native figsize to print width fixed the earlier shrink
problem, but coded font sizes are tuned for a full \\textwidth panel, and three of those
side by side are only ~2.1in each -- full-size fonts there look outsized and compress
the actual plot area rather than just being "12-14pt as printed". The 3-row and
individual-panel variants have room for the full scale.

The in-plot "U-shape: intervals too narrow" / "above: conservative" / "below:
overconfident" annotations from the first version of this figure are gone -- that
content belongs in the LaTeX caption, not fighting three panels for space. See
CAPTION_NOTES below for the sentences to fold into \\caption{}.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

from .style import set_style, BLUE, BLUE_DEEP, RED_DEEP, MUTE, INK, floating_trial_bands

FAMS = [("$z$ (marginal sensitivities)", "z", BLUE_DEEP),
        (r"$\rho$ (within-stimulus correlations)", "rho", RED_DEEP)]
PRINT_W = 6.5    # \textwidth in GRIN_combined_edited.tex
ROW_SCALE = 0.68  # font scale for the cramped 1x3-panel-per-\textwidth layout only

CAPTION_NOTES = (
    "In (A), the z rank histogram is peaked (intervals wider than necessary) and the "
    "rho histogram is U-shaped (intervals too narrow). In (B), the z curve sits above "
    "the identity line (conservative) and the rho curve below it (overconfident)."
)


def _panel_a(ax, ranks, keep, scale, n_bins=20, letter="A", legend=True):
    sl = {"z": slice(0, 8), "rho": slice(8, 12)}
    edges = np.linspace(0, 1, n_bins + 1)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    M = None
    for label, key, col in FAMS:
        r = ranks[:, sl[key]][keep[:, sl[key]]].ravel()
        dens, _ = np.histogram(r, bins=edges)
        dens = dens / dens.sum() * n_bins           # density: 1.0 == uniform
        ax.step(np.r_[0, ctr, 1], np.r_[dens[0], dens, dens[-1]],
                where="mid", color=col, lw=2.0, label=label.split(" (")[0])
        M = r.size
    lo, hi = binom.ppf([0.025, 0.975], M, 1.0 / n_bins) / (M / n_bins)
    ax.axhspan(lo, hi, color=MUTE, alpha=0.20, lw=0)
    ax.axhline(1.0, color=INK, lw=1.0, ls=(0, (4, 3)))
    ax.set_xlabel("normalised rank")
    ax.set_ylabel("density (1.0 = calibrated)")
    ax.set_title(f"{letter}   SBC ranks" if letter else "SBC ranks")
    if legend:
        ax.legend(fontsize=8 * scale, loc="upper center", bbox_to_anchor=(0.5, -0.30),
                  ncol=2, frameon=False)


def _panel_b(ax, bd, scale, letter="B"):
    levels = [float(l) for l in bd["meta"]["levels"]]
    ax.plot([0.4, 1], [0.4, 1], color=MUTE, lw=1.2, ls=(0, (4, 3)), zorder=1)
    for label, key, col in FAMS:
        emp = [bd["by_family"][key][str(l)]["coverage"] for l in levels]
        ax.plot(levels, emp, "-", color=col, lw=1.8, zorder=2)
        ax.plot(levels, emp, "o", color=col, ms=6, zorder=3, label=label.split(" (")[0])
    ax.set_xlim(0.42, 1.0); ax.set_ylim(0.42, 1.0); ax.set_box_aspect(1)
    ax.set_xlabel("nominal credible level")
    ax.set_ylabel("empirical coverage")
    ax.set_title(f"{letter}   Coverage" if letter else "Coverage")


def _panel_c(ax, bd, scale, letter="C"):
    bands = list(bd["by_trial_band"].keys())
    ax.axhline(0.9, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=1)
    series, err = {}, {}
    for label, key, col in FAMS:
        vals = [bd["by_trial_band"][b][key]["coverage"] for b in bands]
        ses = [bd["by_trial_band"][b][key]["mc_se"] for b in bands]
        series[label.split(" (")[0]] = (vals, col)
        err[label.split(" (")[0]] = [(c - 1.96 * se, c + 1.96 * se) for c, se in zip(vals, ses)]
    edges = floating_trial_bands(ax, bands, series, offset_span=0.10, err=err, lw=3.2)
    ax.tick_params(axis="x", labelsize=7.5 * scale)
    ax.set_xlabel("trials per stimulus")
    ax.set_ylabel("90% coverage")
    ax.set_title(f"{letter}   By trial count" if letter else "By trial count")
    ax.set_ylim(0.78, 1.0)


def calibration_breakdown(bd, ranks=None, keep=None, path=None, scale=1.0, n_bins=20):
    """bd: the dict written by scripts/calibration_breakdown.py.
    ranks/keep: (N,12) arrays, for panel A. Panel A is skipped if absent."""
    stem, ext = (os.path.splitext(path) if path else (None, None))

    set_style(ROW_SCALE)
    fig, ax = plt.subplots(1, 3, figsize=(PRINT_W, PRINT_W * 0.40))
    if ranks is not None:
        # legend shown once (panel A) -- z/rho colour coding is identical across all
        # three panels, so repeating it three times only ate space.
        _panel_a(ax[0], ranks, keep, ROW_SCALE)
    _panel_b(ax[1], bd, ROW_SCALE)
    _panel_c(ax[2], bd, ROW_SCALE)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.30, top=0.85, wspace=0.55)
    if path:
        fig.savefig(path); plt.close(fig)
        print(f"figure -> {path}")

        set_style(scale)
        out3 = f"{stem}_3row{ext}"
        fig, ax = plt.subplots(3, 1, figsize=(PRINT_W * 0.62, PRINT_W * 1.9))
        if ranks is not None:
            _panel_a(ax[0], ranks, keep, scale)
        _panel_b(ax[1], bd, scale)
        _panel_c(ax[2], bd, scale)
        fig.subplots_adjust(left=0.17, right=0.95, bottom=0.05, top=0.96, hspace=0.95)
        fig.savefig(out3); plt.close(fig)
        print(f"figure -> {out3}")

        panels = [("a", _panel_a, (ranks, keep)) if ranks is not None else None,
                  ("b", _panel_b, (bd,)), ("c", _panel_c, (bd,))]
        for item in panels:
            if item is None:
                continue
            tag, fn, args = item
            fig, a = plt.subplots(1, 1, figsize=(PRINT_W, PRINT_W * 0.68))
            fn(a, *args, scale, letter="")
            if tag == "b":
                a.legend(fontsize=9 * scale, loc="lower right", frameon=False)
            fig.tight_layout()
            p = f"{stem}_{tag}{ext}"
            fig.savefig(p); plt.close(fig)
            print(f"figure -> {p}")
        print(f"\ncaption notes (fold into \\caption{{}}): {CAPTION_NOTES}")
        return None
    return fig
