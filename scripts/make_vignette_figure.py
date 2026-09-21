"""
Figure 1: four ways of failing an identification task, at matched overall accuracy.

The point of the figure is that the four perceptual spaces are plainly different while
the four confusion matrices they produce are not obviously different by eye -- which is
the case for fitting a GRT model rather than reading accuracy off the data.

Cases (all with overall accuracy matched to within ACC_TOL by a bisection on one
free parameter per case):

  1 low sensitivity on A     both dimensions perceived, one poorly
  2 separability failure     the mean on A shifts with the level of B
  3 decisional failure       perception is fine; the bound on A shifts with B
  4 dimension neglect        A is processed, B is guessed

Case 3 is drawn with tilted bounds because that is what a failure of decisional
separability IS; it is outside the parameterisation GRIN estimates (which assumes
decisional separability throughout) and is included because it is one of the four
states a researcher needs to tell apart, not because the estimator recovers it.

Saved as the combined 2x4 grid (manuscript default: case = column), a combined 4x2
grid (case = row, if a journal wants it taller and narrower), and each case
individually as its own perceptual-space + confusion-matrix pair -- see
docs/figure_sizing.md. Figsize matches intended print width directly so the coded
font sizes land at their true point size on the page, rather than a wide "poster"
figure LaTeX shrinks down.

    python scripts/make_vignette_figure.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import norm

from src.config import FIGURES_DIR
from src.viz.style import set_style, BLUE, RED, BLUE_DEEP, RED_DEEP, INK, MUTE, CMAP_SEQ
import src.grt_model as gm

OUT = os.path.join(FIGURES_DIR, "vignette.png")
TARGET_ACC = 0.48   # the ceiling for the neglect case is 0.50 (B guessed), so the
                    # four cases can only be matched below it
ACC_TOL = 0.004
STIM_COL = [BLUE, RED, BLUE_DEEP, RED_DEEP]
LABELS = ["A$_1$B$_1$", "A$_1$B$_2$", "A$_2$B$_1$", "A$_2$B$_2$"]
PRINT_W = 6.5   # \textwidth in GRIN_combined_edited.tex


def _acc(P):
    return float(np.mean(np.diag(P)))


def _case_low_sensitivity(s):
    # zy fixed lower than the original 1.4: at 1.4, matching the shared 48% target left
    # the bisected A-sensitivity at ~0.06 -- visually indistinguishable from zero, so the
    # panel read as a second "collapsed dimension" case rather than a genuinely low-but-
    # nonzero one. At 0.75, the bisection lands near 0.3: still clearly the weakest of the
    # four cases, but visibly separated ellipses rather than an accidental second neglect.
    zx = np.array([-s, -s, s, s]); zy = np.array([-0.75, 0.75, -0.75, 0.75])
    return zx, zy, np.zeros(4)


def _case_separability(s):
    # The mean on A depends on the level of B: |z_x| is 2.5x larger when B is at level 2.
    # The ratio is held fixed and the overall scale is what the bisection moves, so the
    # separability failure stays the same size as accuracy is matched to the other cases.
    zx = np.array([-s, -2.5 * s, s, 2.5 * s])
    zy = np.array([-0.9 * s, 0.9 * s, -0.9 * s, 0.9 * s])
    return zx, zy, np.zeros(4)


def _case_decisional(s):
    # perception is separable and independent; the failure is in the bound, applied below
    zx = np.array([-s, -s, s, s]); zy = np.array([-s, s, -s, s])
    return zx, zy, np.zeros(4)


def _case_neglect(s):
    zx = np.array([-s, -s, s, s]); zy = np.array([-0.02, 0.02, -0.02, 0.02])
    return zx, zy, np.zeros(4)


def _probs_tilted(zx, zy, slope):
    """Response probabilities when the bound on A tilts with the perceived level of B.
    Monte Carlo, because a tilted bound has no orthant-probability shortcut."""
    rng = np.random.default_rng(7)
    n = 400_000
    P = np.zeros((4, 4))
    for i in range(4):
        x = rng.normal(zx[i], 1.0, n); y = rng.normal(zy[i], 1.0, n)
        a = x > slope * y            # tilted bound on A
        b = y > 0.0
        idx = (a.astype(int) * 2) + b.astype(int)
        P[i] = np.bincount(idx, minlength=4) / n
    return P


def _fit_accuracy(fn, lo, hi, tilt=None):
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        zx, zy, rho = fn(mid)
        P = _probs_tilted(zx, zy, tilt) if tilt is not None else \
            gm.forward_probabilities(zx, zy, rho)
        a = _acc(np.asarray(P))
        if abs(a - TARGET_ACC) < ACC_TOL:
            break
        if a < TARGET_ACC:
            lo = mid
        else:
            hi = mid
    return fn(mid), np.asarray(P), a


def _draw_space(ax, zx, zy, rho, tilt=None):
    if tilt is None:
        ax.axvline(0, color=MUTE, lw=1.3, ls=(0, (5, 4)))
    else:
        yy = np.array([-4.2, 4.2])
        ax.plot(tilt * yy, yy, color=RED_DEEP, lw=1.6, ls=(0, (5, 4)))
    ax.axhline(0, color=MUTE, lw=1.3, ls=(0, (5, 4)))
    for i in range(4):
        # 1 SD only. A second, fainter 2 SD ellipse was drawn here before; nobody in the
        # GRT literature plots both on the same figure, and doing so read as not knowing
        # the convention rather than as extra information.
        ax.add_patch(Ellipse((zx[i], zy[i]), 2, 2, angle=0,
                             fill=False, edgecolor=STIM_COL[i], lw=1.5))
        ax.plot(zx[i], zy[i], "o", color=STIM_COL[i], ms=4)
    ax.set_xlim(-4.2, 4.2); ax.set_ylim(-4.2, 4.2)
    ax.set_box_aspect(1); ax.set_xticks([]); ax.set_yticks([])


def _draw_matrix(ax, P, label_fs=7.5, tick_fs=6.5):
    ax.imshow(P, cmap=CMAP_SEQ, vmin=0, vmax=0.75)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{100*P[i,j]:.0f}", ha="center", va="center",
                    fontsize=label_fs, color=INK if P[i, j] < 0.45 else "white")
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(LABELS, fontsize=tick_fs); ax.set_yticklabels(LABELS, fontsize=tick_fs)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)


CASES = [
    ("Low sensitivity on A", _case_low_sensitivity, (0.05, 3.0), None,
     "one feature perceived poorly"),
    ("Separability failure", _case_separability, (0.05, 3.0), None,
     "A's mean shifts with B"),
    ("Dimension neglect", _case_neglect, (0.05, 3.5), None,
     "B guessed, not perceived"),
    # Last, deliberately: nothing in this paper resolves decisional failure (every
    # estimator here assumes decisional separability), so it shouldn't sit alongside
    # the three cases the rest of the figure motivates as if it were one of them.
    ("Decisional failure", _case_decisional, (0.05, 3.0), 0.55,
     "A's bound shifts with B"),
]


def _draw_case(ax_top, ax_bot, title, sub, a, zx, zy, rho, tilt, P,
               title_fs=9.5, sub_fs=7.5, label_fs=7.5, tick_fs=6.5):
    _draw_space(ax_top, zx, zy, rho, tilt)
    _draw_matrix(ax_bot, P, label_fs=label_fs, tick_fs=tick_fs)
    ax_top.set_title(f"{title}\n", fontsize=title_fs)
    ax_top.text(0.5, 1.02, sub, transform=ax_top.transAxes, ha="center",
               va="bottom", fontsize=sub_fs, color=MUTE, style="italic")


def main():
    set_style()
    fitted = []
    for title, fn, bounds, tilt, sub in CASES:
        (zx, zy, rho), P, a = _fit_accuracy(fn, *bounds, tilt=tilt)
        fitted.append((title, sub, a, zx, zy, rho, tilt, P))
        print(f"{title:24s} accuracy {a:.4f}")

    # ---- combined, case = column (manuscript default) -----------------------
    fig, ax = plt.subplots(2, 4, figsize=(PRINT_W, PRINT_W * 0.535),
                           gridspec_kw=dict(height_ratios=[1.05, 1.0],
                                            width_ratios=[1.18, 1, 1, 1]))
    for k, (title, sub, a, zx, zy, rho, tilt, P) in enumerate(fitted):
        _draw_case(ax[0][k], ax[1][k], title, sub, a, zx, zy, rho, tilt, P)
    ax[1][0].set_ylabel("confusion matrix (%)", fontsize=9)
    # All four cases are matched to the same target accuracy by construction (see
    # TARGET_ACC/ACC_TOL above) -- stating that once for the whole figure is the point;
    # repeating it under every panel was the kind of redundancy that just adds clutter.
    fig.text(0.5, 0.015, f"overall accuracy {100*TARGET_ACC:.0f}% in all four panels",
             ha="center", va="bottom", fontsize=9, color=MUTE)
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.13, top=0.80,
                        wspace=0.85, hspace=0.30)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(OUT); plt.close(fig)
    print(f"wrote {OUT}")

    # ---- combined, case = row (narrow single-column alternative) -----------
    out2 = OUT.replace(".png", "_stacked.png")
    fig, ax = plt.subplots(4, 2, figsize=(PRINT_W * 0.62, PRINT_W * 1.85),
                           gridspec_kw=dict(width_ratios=[1.05, 1.0]))
    for k, (title, sub, a, zx, zy, rho, tilt, P) in enumerate(fitted):
        _draw_case(ax[k][0], ax[k][1], title, sub, a, zx, zy, rho, tilt, P,
                  title_fs=12, sub_fs=9, label_fs=8, tick_fs=7.5)
    fig.text(0.5, 0.008, f"overall accuracy {100*TARGET_ACC:.0f}% in all four panels",
             ha="center", va="bottom", fontsize=9.5, color=MUTE)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.95,
                        wspace=0.15, hspace=0.55)
    fig.savefig(out2); plt.close(fig)
    print(f"wrote {out2}")

    # ---- individual: one case, its own space + matrix pair -----------------
    for k, (title, sub, a, zx, zy, rho, tilt, P) in enumerate(fitted):
        tag = title.lower().replace(" ", "_")
        fig, ax2 = plt.subplots(1, 2, figsize=(PRINT_W * 0.6, PRINT_W * 0.34))
        _draw_case(ax2[0], ax2[1], title, sub, a, zx, zy, rho, tilt, P,
                  title_fs=13, sub_fs=10, label_fs=8.5, tick_fs=8)
        # standalone panel: no other copy of this number exists in the figure, so it
        # belongs here rather than only in a caption the reader may not have to hand.
        ax2[1].set_xlabel(f"overall accuracy {100*a:.0f}%", fontsize=10)
        fig.tight_layout()
        p = OUT.replace(".png", f"_{tag}.png")
        fig.savefig(p); plt.close(fig)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
