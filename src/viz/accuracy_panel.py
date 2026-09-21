"""
accuracy_panel.py — performance against observed per-dimension accuracy.

The companion to the trial-count stratification used elsewhere. Accuracy is the
quantity a researcher sets by choosing stimulus separation, and can watch during a
pilot or staircase block, so it is the axis on which a design decision is actually
made. Three panels, all lines and points:

  A  parameter error for each family. The two families are expected to pull in
     opposite directions -- correlations identified best near chance, sensitivities
     best near ceiling -- and the crossing region is the design recommendation.
  B  construct classification accuracy against the same axis.
  C  correlation error against accuracy, split by trial count, to show whether the
     informative band moves as data accumulate or only gets narrower.

Saved as a combined 1x3 row (the manuscript default), a combined 3x1 stack, and each
panel individually -- see docs/figure_sizing.md. Native figsize matches the intended
print width directly rather than a wide "poster" size shrunk by LaTeX. The combined row
uses ROW_SCALE (< 1) because three \textwidth-tuned panels side by side are only ~2.1in
each -- full-size fonts there look outsized, not just "12-14pt as printed" (see
docs/figure_sizing.md). Legends live inside each panel now (not below the axes): with
one shared x-axis label instead of three repeats, there's no separate margin for a
below-axis legend to sit in without either colliding with the shared label or bloating
the figure -- in-panel corners are picked per panel from where the curves are NOT.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from .style import set_style, BLUE, BLUE_DEEP, RED_DEEP, MUTE, INK

BAND_LO, BAND_HI = 0.60, 0.80     # the frontier analysis's recommended window
PRINT_W = 6.5                     # \textwidth in GRIN_combined_edited.tex
ROW_SCALE = 0.68                  # font scale for the cramped 1x3-per-\textwidth layout


def _centres(rows, key_lo="lo", key_hi="hi"):
    return np.array([0.5 * (r[key_lo] + r[key_hi]) for r in rows])


def _panel_a(ax, x, rows, scale, letter="A", xlabel=True):
    ax.axvspan(BAND_LO, BAND_HI, color=MUTE, alpha=0.15, lw=0, zorder=0)
    # rho is bounded on (-1,1) so its absolute error is already interpretable; the
    # sensitivities are unbounded, so absolute error there confounds precision with
    # the size of what is being estimated. Plot rho's absolute error against the
    # sensitivities' RELATIVE error, which is the quantity the Cramer-Rao argument
    # in the frontier analysis actually makes a claim about.
    rz = np.array([r["rel_err_z"] for r in rows])
    mr = np.array([r["mae_rho"] for r in rows])
    ax.plot(x, mr, "-o", color=RED_DEEP, ms=5.5, lw=1.8, label=r"$\rho$")
    ax.plot(x, rz, "-o", color=BLUE_DEEP, ms=5.5, lw=1.8, label="$z$")
    if xlabel:
        ax.set_xlabel("accuracy / dimension")
    ax.set_ylabel("error")
    ax.set_title(f"{letter}   Recovery" if letter else "Recovery")
    # The shaded band is explained in the LaTeX caption ("shading marks the 60-80%
    # window..."), so an in-plot annotation repeating that would just be one more thing
    # competing for room in an already-narrow panel -- left out deliberately here.
    ax.legend(fontsize=8 * scale, loc="upper right", frameon=False, handlelength=1.4)
    ax.set_ylim(bottom=0)


def _panel_b(ax, x, rows, scale, letter="B", xlabel=True):
    ax.axvspan(BAND_LO, BAND_HI, color=MUTE, alpha=0.15, lw=0, zorder=0)
    for key, lab, col in (("acc_PI", "PI", RED_DEEP),
                          ("acc_sepA", "PS(A)", BLUE_DEEP),
                          ("acc_sepB", "PS(B)", BLUE)):
        ax.plot(x, [r[key] for r in rows], "-o", color=col, ms=5.5, lw=1.8, label=lab)
    ax.axhline(0.5, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=1)
    if xlabel:
        ax.set_xlabel("accuracy / dimension")
    ax.set_ylabel("classification accuracy")
    ax.set_title(f"{letter}   Constructs" if letter else "Constructs")
    ax.legend(fontsize=8 * scale, loc="lower right", frameon=False, handlelength=1.4)


def _panel_c(ax, cells, scale, letter="C", xlabel=True):
    ax.axvspan(BAND_LO, BAND_HI, color=MUTE, alpha=0.15, lw=0, zorder=0)
    tps_bands = sorted({(c["tps_lo"], c["tps_hi"]) for c in cells})
    cmap = [BLUE, BLUE_DEEP, RED_DEEP, INK]
    for i, (lo, hi) in enumerate(tps_bands):
        sub = [c for c in cells if c["tps_lo"] == lo and c["tps_hi"] == hi]
        if len(sub) < 2:
            continue
        xs = _centres(sub, "acc_lo", "acc_hi")
        ax.plot(xs, [c["mae_rho"] for c in sub], "-o", ms=4.5, lw=1.6,
                color=cmap[i % len(cmap)], label=f"{lo:g}–{hi:g}")
    if xlabel:
        ax.set_xlabel("accuracy / dimension")
    ax.set_ylabel(r"MAE, $\rho$")
    ax.set_title(f"{letter}   $\\rho$ error by trial count" if letter
                 else "$\\rho$ error by trial count")
    ax.legend(fontsize=7.5 * scale, loc="lower right", frameon=False,
              handlelength=1.4, title="trials", title_fontsize=7.5 * scale)
    ax.set_ylim(bottom=0)


def accuracy_stratified_figure(out, path, scale=1.0):
    rows = out["by_accuracy"]
    x = _centres(rows)
    cells = out["by_accuracy_x_trials"]
    stem, ext = os.path.splitext(path)

    # ---- combined, 1 row x 3 (manuscript default) --------------------------
    set_style(ROW_SCALE)
    fig, ax = plt.subplots(1, 3, figsize=(PRINT_W, PRINT_W * 0.38))
    _panel_a(ax[0], x, rows, ROW_SCALE, xlabel=False)
    _panel_b(ax[1], x, rows, ROW_SCALE, xlabel=False)
    _panel_c(ax[2], cells, ROW_SCALE, xlabel=False)
    fig.text(0.5, 0.02, "accuracy / dimension", ha="center", va="bottom",
             fontsize=10 * ROW_SCALE)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.20, top=0.86, wspace=0.55)
    fig.savefig(path); plt.close(fig)
    print(f"figure -> {path}")

    # ---- combined, 3 rows x 1 (single-column alternative) -------------------
    set_style(scale)
    out3 = f"{stem}_3row{ext}"
    fig, ax = plt.subplots(3, 1, figsize=(PRINT_W * 0.62, PRINT_W * 1.9))
    _panel_a(ax[0], x, rows, scale)
    _panel_b(ax[1], x, rows, scale)
    _panel_c(ax[2], cells, scale)
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.06, top=0.96, hspace=0.55)
    fig.savefig(out3); plt.close(fig)
    print(f"figure -> {out3}")

    # ---- individual panels --------------------------------------------------
    for tag, fn, args in [("a", _panel_a, (x, rows)), ("b", _panel_b, (x, rows)),
                          ("c", _panel_c, (cells,))]:
        fig, a = plt.subplots(1, 1, figsize=(PRINT_W, PRINT_W * 0.62))
        fn(a, *args, scale, letter="")
        fig.tight_layout()
        p = f"{stem}_{tag}{ext}"
        fig.savefig(p); plt.close(fig)
        print(f"figure -> {p}")

    return path
