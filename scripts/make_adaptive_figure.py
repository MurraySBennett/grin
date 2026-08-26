"""
Adaptive stopping figure for the manuscript, from results/adaptive_stopping.json.

Two panels, both curves -- the quantity of interest is a level against a
continuous target, so nothing here is a bar chart:

  A  trials per stimulus needed to reach a posterior-precision target, adaptive
     against the smallest fixed budget that gets the SAME FRACTION of observers
     there. Log y, because the fixed budget grows geometrically.
  B  the resulting saving, as a function of the target, with the reported
     operating point marked and the region where observers start to be censored
     shaded.

    python scripts/make_adaptive_figure.py
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt

from src.config import FIGURES_DIR
from src.viz.style import set_style, BLUE_DEEP, RED_DEEP, MUTE, INK

SRC = os.path.join("results", "adaptive_stopping.json")
OUT = os.path.join(FIGURES_DIR, "adaptive_stopping.png")
REPORTED = 0.35


def main():
    d = json.load(open(SRC))
    by = d["by_sd_max"]
    targets = np.array(sorted((float(k) for k in by), reverse=True))
    key = lambda t: by[f"{t:g}"]

    adaptive = np.array([key(t)["adaptive_mean_trials"] for t in targets])
    fixed = np.array([key(t)["fixed_matched_coverage"] for t in targets])
    saving = np.array([key(t)["saving_matched"] for t in targets])
    censored = np.array([key(t)["never_reached_frac"] for t in targets])

    set_style()
    fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.0))

    # ---- A: trials required ------------------------------------------------
    ax[0].plot(targets, fixed, "-", color=RED_DEEP, lw=2.0)
    ax[0].plot(targets, fixed, "o", color=RED_DEEP, ms=6, label="fixed budget")
    ax[0].plot(targets, adaptive, "-", color=BLUE_DEEP, lw=2.0)
    ax[0].plot(targets, adaptive, "o", color=BLUE_DEEP, ms=6, label="adaptive stopping")
    ax[0].set_yscale("log")
    ax[0].invert_xaxis()
    ax[0].set_xlabel("posterior precision target (max posterior SD)")
    ax[0].set_ylabel("trials per stimulus")
    ax[0].set_title("A   Cost of reaching a precision target")
    ax[0].legend(loc="upper left")
    ax[0].axvline(REPORTED, color=MUTE, lw=1.0, ls=(0, (4, 3)), zorder=0)

    # ---- B: saving ---------------------------------------------------------
    ax[1].plot(targets, 100 * saving, "-", color=BLUE_DEEP, lw=2.0)
    ax[1].plot(targets, 100 * saving, "o", color=BLUE_DEEP, ms=6)
    ax[1].invert_xaxis()
    ax[1].set_xlabel("posterior precision target (max posterior SD)")
    ax[1].set_ylabel("trials saved per observer (%)")
    ax[1].set_title("B   Saving, as a function of the target")

    j = int(np.argmin(np.abs(targets - REPORTED)))
    ax[1].annotate(f"reported operating point\n{100*saving[j]:.1f}% at SD $\\leq$ {REPORTED}",
                   xy=(targets[j], 100 * saving[j]),
                   xytext=(targets[j] - 0.06, 100 * saving[j] - 26),
                   fontsize=8.5, color=INK,
                   arrowprops=dict(arrowstyle="-", color=MUTE, lw=1.0))
    # shade only where censoring is material; 0.25% of observers is not
    cens = targets[censored > 0.01]
    if cens.size:
        ax[1].axvspan(cens.max(), targets.min(), color=MUTE, alpha=0.16, lw=0)
        ax[1].annotate("some observers\nnever reach the target",
                       xy=(cens.max() - 0.005, 20), fontsize=8, color=MUTE,
                       ha="right", va="bottom")
    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(OUT); plt.close(fig)
    print(f"wrote {OUT}")
    for t, a, f_, s, c in zip(targets, adaptive, fixed, saving, censored):
        print(f"  SD<= {t:.2f}   adaptive {a:7.1f}   fixed {f_:6.0f}   "
              f"saving {100*s:5.1f}%   censored {100*c:4.1f}%")


if __name__ == "__main__":
    main()
