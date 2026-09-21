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

Saved several ways, all from the same two panel-drawing functions so they can never
drift apart:
  adaptive_stopping.png            combined, 1 row x 2 panels (the manuscript default)
  adaptive_stopping_2row.png       combined, 2 rows x 1 panel (if a journal wants it taller
                                    and narrower -- e.g. a single-column layout)
  adaptive_stopping_a.png          panel A alone
  adaptive_stopping_b.png          panel B alone

Every saved file uses a native figsize matched to its OWN intended print width, not a
shared wide "poster" figsize shrunk down by LaTeX -- see docs/figure_sizing.md. That is
what keeps fonts at their coded point size once actually printed, rather than the
~0.4-0.5x shrink a 10-13in-wide figure suffers when placed at a ~6.5in manuscript
column width.

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

# Target print width for a full-width manuscript figure (APA man class, 1in margins on
# US letter): matches \textwidth in GRIN_combined_edited.tex. Individual panels and the
# 2-row combo use this width too, so nothing needs LaTeX-side rescaling.
PRINT_W = 6.5


def _panel_a(ax, targets, adaptive, fixed, letter="A", capped=None):
    ax.plot(targets, fixed, "-", color=RED_DEEP, lw=2.0, zorder=2)
    ax.plot(targets, adaptive, "-", color=BLUE_DEEP, lw=2.0, zorder=2)
    ax.plot(targets, adaptive, "o", color=BLUE_DEEP, ms=6, label="adaptive stopping",
            zorder=3)
    if capped is not None and capped.any():
        # The fixed-budget grid tested tops out at 640 trials/stimulus. At the two
        # tightest targets, 640 is already the smallest grid value that gets enough
        # observers there -- the curve going flat is that ceiling binding, not a real
        # plateau in the underlying relationship. Hollow markers say so on sight instead
        # of leaving a reader to wonder why the red line stops climbing.
        ax.plot(targets[~capped], fixed[~capped], "o", color=RED_DEEP, ms=6,
                label="fixed budget", zorder=3)
        ax.plot(targets[capped], fixed[capped], "o", color=RED_DEEP, ms=6,
                markerfacecolor="none", mew=1.6, label="fixed budget (grid-capped)",
                zorder=3)
    else:
        ax.plot(targets, fixed, "o", color=RED_DEEP, ms=6, label="fixed budget", zorder=3)
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("posterior-SD target")
    ax.set_ylabel("trials per stimulus")
    ax.set_title(f"{letter}   Cost of reaching a target" if letter
                 else "Cost of reaching a target")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.axvline(REPORTED, color=MUTE, lw=1.0, ls=(0, (4, 3)), zorder=0)


def _panel_b(ax, targets, saving, censored, letter="B"):
    ax.plot(targets, 100 * saving, "-", color=BLUE_DEEP, lw=2.0)
    ax.plot(targets, 100 * saving, "o", color=BLUE_DEEP, ms=6)
    ax.invert_xaxis()
    ax.set_xlabel("posterior-SD target")
    ax.set_ylabel("trials saved per observer (%)")
    ax.set_title(f"{letter}   Resulting saving" if letter
                 else "Resulting saving")

    j = int(np.argmin(np.abs(targets - REPORTED)))
    # Centred beneath its point and inside the axes, rather than off to one side where
    # it could be mistaken for describing a neighbouring point.
    ax.annotate(f"illustrative threshold\n{100*saving[j]:.1f}% at SD $\\leq$ {REPORTED}",
                xy=(targets[j], 100 * saving[j]),
                xytext=(targets[j], 100 * saving[j] - 30),
                ha="center", fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTE, lw=1.0))
    cens = censored > 0.01
    if cens.any():
        # A span from the affected point(s) to targets.min() degenerates to zero width
        # when only the single most-extreme target crosses the threshold (exactly the
        # case here, since that target IS targets.min()) -- span to the actual axis
        # edge instead, which stays correct regardless of how many points are affected.
        edge = min(ax.get_xlim())  # data-coordinate edge nearest the small-SD side,
                                    # correct whichever way invert_xaxis() flipped it
        ax.axvspan(targets[cens].min(), edge, color=MUTE, alpha=0.16, lw=0, zorder=0)
        ax.plot(targets[cens], 100 * saving[cens], "o", ms=11, mfc="none",
               mec=MUTE, mew=1.4, zorder=1)
        ax.annotate("some observers never reach the\ntarget within the largest budget\n"
                    "simulated (640 trials)",
                    xy=(targets[cens].max(), 100 * saving[cens].max()),
                    xycoords="data", textcoords="axes fraction", xytext=(0.55, 0.08),
                    fontsize=7.5, color=MUTE, ha="center", va="bottom",
                    arrowprops=dict(arrowstyle="-", color=MUTE, lw=0.8))


def main():
    d = json.load(open(SRC))
    by = d["by_sd_max"]
    targets = np.array(sorted((float(k) for k in by), reverse=True))
    key = lambda t: by[f"{t:g}"]

    adaptive = np.array([key(t)["adaptive_mean_trials"] for t in targets])
    fixed = np.array([key(t)["fixed_matched_coverage"] for t in targets])
    saving = np.array([key(t)["saving_matched"] for t in targets])
    censored = np.array([key(t)["never_reached_frac"] for t in targets])
    # The fixed-budget grid this was matched against (see the generating simulation):
    # a point sits AT the ceiling because no larger grid value was tested, which reads
    # very differently from a point that would sit there even on an unbounded grid.
    grid_max = max(d.get("levels", [640]))
    capped = fixed >= grid_max

    set_style()
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ---- combined, 1 row x 2 (manuscript default) --------------------------
    fig, ax = plt.subplots(1, 2, figsize=(PRINT_W, PRINT_W * 0.5))
    _panel_a(ax[0], targets, adaptive, fixed, capped=capped)
    _panel_b(ax[1], targets, saving, censored)
    fig.tight_layout(w_pad=6.0)
    fig.savefig(OUT); plt.close(fig)
    print(f"wrote {OUT}")

    # ---- combined, 2 rows x 1 (narrow single-column alternative) -----------
    out2 = OUT.replace(".png", "_2row.png")
    fig, ax = plt.subplots(2, 1, figsize=(PRINT_W * 0.62, PRINT_W * 1.15))
    _panel_a(ax[0], targets, adaptive, fixed, capped=capped)
    _panel_b(ax[1], targets, saving, censored)
    fig.tight_layout(h_pad=2.0)
    fig.savefig(out2); plt.close(fig)
    print(f"wrote {out2}")

    # ---- individual panels ---------------------------------------------
    for tag, fn, letter in [("a", _panel_a, ""), ("b", _panel_b, "")]:
        fig, a = plt.subplots(1, 1, figsize=(PRINT_W, PRINT_W * 0.72))
        if tag == "a":
            fn(a, targets, adaptive, fixed, letter=letter, capped=capped)
        else:
            fn(a, targets, saving, censored, letter=letter)
        fig.tight_layout()
        p = OUT.replace(".png", f"_{tag}.png")
        fig.savefig(p); plt.close(fig)
        print(f"wrote {p}")

    for t, a, f_, s, c in zip(targets, adaptive, fixed, saving, censored):
        print(f"  SD<= {t:.2f}   adaptive {a:7.1f}   fixed {f_:6.0f}   "
              f"saving {100*s:5.1f}%   censored {100*c:4.1f}%")


if __name__ == "__main__":
    main()
