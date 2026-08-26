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
"""
import numpy as np
import matplotlib.pyplot as plt

from .style import set_style, BLUE, BLUE_DEEP, RED_DEEP, MUTE, INK

BAND_LO, BAND_HI = 0.60, 0.80     # the frontier analysis's recommended window


def _centres(rows, key_lo="lo", key_hi="hi"):
    return np.array([0.5 * (r[key_lo] + r[key_hi]) for r in rows])


def accuracy_stratified_figure(out, path, scale=1.0):
    set_style(scale)
    rows = out["by_accuracy"]
    x = _centres(rows)
    fig, ax = plt.subplots(1, 3, figsize=(13.4, 4.0))

    for a in ax:
        a.axvspan(BAND_LO, BAND_HI, color=MUTE, alpha=0.15, lw=0, zorder=0)

    # ---- A: parameter error, both families, on comparable scales ---------
    # rho is bounded on (-1,1) so its absolute error is already interpretable; the
    # sensitivities are unbounded, so absolute error there confounds precision with
    # the size of what is being estimated. Plot rho's absolute error against the
    # sensitivities' RELATIVE error, which is the quantity the Cramer-Rao argument
    # in the frontier analysis actually makes a claim about.
    rz = np.array([r["rel_err_z"] for r in rows])
    mr = np.array([r["mae_rho"] for r in rows])
    ax[0].plot(x, mr, "-o", color=RED_DEEP, ms=5.5, lw=1.8,
               label=r"$\rho$: absolute error")
    ax[0].plot(x, rz, "-o", color=BLUE_DEEP, ms=5.5, lw=1.8,
               label="$z$: error relative to $|z|$")
    ax[0].set_xlabel("observed accuracy per dimension")
    ax[0].set_ylabel("error")
    ax[0].set_title("A   Parameter recovery")
    ax[0].legend(fontsize=8.5 * scale)
    ax[0].set_ylim(bottom=0)

    # ---- B: construct classification -------------------------------------
    for key, lab, col in (("acc_PI", "independence", RED_DEEP),
                          ("acc_sepA", "separability A", BLUE_DEEP),
                          ("acc_sepB", "separability B", BLUE)):
        ax[1].plot(x, [r[key] for r in rows], "-o", color=col, ms=5.5, lw=1.8, label=lab)
    ax[1].axhline(0.5, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax[1].set_xlabel("observed accuracy per dimension")
    ax[1].set_ylabel("classification accuracy")
    ax[1].set_title("B   Construct recovery")
    ax[1].legend(fontsize=8.5 * scale, loc="lower right")

    # ---- C: correlation error by accuracy, split by trial count ----------
    cells = out["by_accuracy_x_trials"]
    tps_bands = sorted({(c["tps_lo"], c["tps_hi"]) for c in cells})
    cmap = [BLUE, BLUE_DEEP, RED_DEEP, INK]
    for i, (lo, hi) in enumerate(tps_bands):
        sub = [c for c in cells if c["tps_lo"] == lo and c["tps_hi"] == hi]
        if len(sub) < 2:
            continue
        xs = _centres(sub, "acc_lo", "acc_hi")
        ax[2].plot(xs, [c["mae_rho"] for c in sub], "-o", ms=4.5, lw=1.6,
                   color=cmap[i % len(cmap)], label=f"{lo:g}–{hi:g} trials")
    ax[2].set_xlabel("observed accuracy per dimension")
    ax[2].set_ylabel(r"MAE, $\rho$")
    ax[2].set_title("C   Correlation error by data regime")
    ax[2].legend(fontsize=8 * scale)
    ax[2].set_ylim(bottom=0)

    ax[0].annotate("recommended\ndesign window", xy=(0.70, ax[0].get_ylim()[1] * 0.92),
                   ha="center", va="top", fontsize=8 * scale, color=MUTE)

    fig.tight_layout()
    fig.savefig(path); plt.close(fig)
    print(f"figure -> {path}")
    return path
