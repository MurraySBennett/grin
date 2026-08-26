"""
calibration_panel.py — the manuscript's calibration figure.

Three panels, portrait-friendly, built from scripts/calibration_breakdown.py's JSON:

  A  SBC rank histograms, z and rho overlaid as step outlines against the
     uniform band. Overlaying is the point: the two families depart from
     uniform in OPPOSITE directions, which a pooled histogram hides.
  B  Coverage curve, empirical against nominal, one line per family.
  C  90% coverage by trials per stimulus, as points (not bars) -- the
     informative quantity is the level, so the level is what is drawn.

No bar charts anywhere: dots carry the estimate and the vertical rule carries
the Monte Carlo interval.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

from .style import set_style, BLUE, BLUE_DEEP, RED_DEEP, MUTE, INK

FAMS = [("$z$ (marginal sensitivities)", "z", BLUE_DEEP),
        (r"$\rho$ (within-stimulus correlations)", "rho", RED_DEEP)]


def calibration_breakdown(bd, ranks=None, keep=None, path=None, scale=1.0, n_bins=20):
    """bd: the dict written by scripts/calibration_breakdown.py.
    ranks/keep: (N,12) arrays, for panel A. Panel A is skipped if absent."""
    set_style(scale)
    fig, ax = plt.subplots(1, 3, figsize=(13.2, 3.9))

    # ---- A: SBC rank histograms, step outlines ---------------------------
    if ranks is not None:
        sl = {"z": slice(0, 8), "rho": slice(8, 12)}
        edges = np.linspace(0, 1, n_bins + 1)
        ctr = 0.5 * (edges[:-1] + edges[1:])
        for label, key, col in FAMS:
            r = ranks[:, sl[key]][keep[:, sl[key]]].ravel()
            dens, _ = np.histogram(r, bins=edges)
            dens = dens / dens.sum() * n_bins           # density: 1.0 == uniform
            ax[0].step(np.r_[0, ctr, 1], np.r_[dens[0], dens, dens[-1]],
                       where="mid", color=col, lw=2.0, label=label.split(" (")[0])
            M = r.size
        lo, hi = binom.ppf([0.025, 0.975], M, 1.0 / n_bins) / (M / n_bins)
        ax[0].axhspan(lo, hi, color=MUTE, alpha=0.20, lw=0)
        ax[0].axhline(1.0, color=INK, lw=1.0, ls=(0, (4, 3)))
        ax[0].set_xlabel("normalised rank of true value")
        ax[0].set_ylabel("density (1.0 = calibrated)")
        ax[0].set_title("A   SBC ranks")
        ax[0].legend(fontsize=8.5 * scale, loc="lower center")
        ax[0].annotate("U-shape:\nintervals too narrow", xy=(0.5, 1.28), ha="center",
                       fontsize=8 * scale, color=RED_DEEP)

    # ---- B: coverage curve ------------------------------------------------
    levels = [float(l) for l in bd["meta"]["levels"]]
    ax[1].plot([0.4, 1], [0.4, 1], color=MUTE, lw=1.2, ls=(0, (4, 3)), zorder=1)
    for label, key, col in FAMS:
        emp = [bd["by_family"][key][str(l)]["coverage"] for l in levels]
        ax[1].plot(levels, emp, "-", color=col, lw=1.8, zorder=2)
        ax[1].plot(levels, emp, "o", color=col, ms=6, zorder=3,
                   label=label.split(" (")[0])
    ax[1].set_xlim(0.42, 1.0); ax[1].set_ylim(0.42, 1.0); ax[1].set_box_aspect(1)
    ax[1].set_xlabel("nominal credible level")
    ax[1].set_ylabel("empirical coverage")
    ax[1].set_title("B   Interval coverage")
    ax[1].annotate("above the line:\nconservative", xy=(0.60, 0.90), fontsize=8 * scale,
                   color=BLUE_DEEP)
    ax[1].annotate("below the line:\noverconfident", xy=(0.72, 0.52), fontsize=8 * scale,
                   color=RED_DEEP)

    # ---- C: 90% coverage by trial band, dots with MC intervals -----------
    bands = list(bd["by_trial_band"].keys())
    x = np.arange(len(bands))
    ax[2].axhline(0.9, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=1)
    for off, (label, key, col) in zip((-0.13, 0.13), FAMS):
        c = np.array([bd["by_trial_band"][b][key]["coverage"] for b in bands], float)
        se = np.array([bd["by_trial_band"][b][key]["mc_se"] for b in bands], float)
        ax[2].vlines(x + off, c - 1.96 * se, c + 1.96 * se, color=col, lw=1.4, zorder=2)
        ax[2].plot(x + off, c, "o", color=col, ms=5.5, zorder=3,
                   label=label.split(" (")[0])
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(bands, rotation=45, ha="right", fontsize=8 * scale)
    ax[2].set_xlabel("trials per stimulus")
    ax[2].set_ylabel("empirical coverage of the 90% interval")
    ax[2].set_title("C   Coverage by data regime")
    ax[2].set_ylim(0.78, 1.0)

    fig.tight_layout()
    if path:
        fig.savefig(path)
        plt.close(fig)
    return fig
