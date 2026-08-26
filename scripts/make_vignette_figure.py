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


def _acc(P):
    return float(np.mean(np.diag(P)))


def _case_low_sensitivity(s):
    zx = np.array([-s, -s, s, s]); zy = np.array([-1.4, 1.4, -1.4, 1.4])
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
        for k in (1.0, 2.0):
            ax.add_patch(Ellipse((zx[i], zy[i]), 2 * k, 2 * k, angle=0,
                                 fill=False, edgecolor=STIM_COL[i],
                                 lw=1.5 if k == 1 else 0.8,
                                 alpha=1.0 if k == 1 else 0.45))
        ax.plot(zx[i], zy[i], "o", color=STIM_COL[i], ms=4)
    ax.set_xlim(-4.2, 4.2); ax.set_ylim(-4.2, 4.2)
    ax.set_box_aspect(1); ax.set_xticks([]); ax.set_yticks([])


def _draw_matrix(ax, P):
    ax.imshow(P, cmap=CMAP_SEQ, vmin=0, vmax=0.75)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{100*P[i,j]:.0f}", ha="center", va="center",
                    fontsize=7.5, color=INK if P[i, j] < 0.45 else "white")
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(LABELS, fontsize=6.5); ax.set_yticklabels(LABELS, fontsize=6.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)


def main():
    cases = [
        ("Low sensitivity on A", _case_low_sensitivity, (0.05, 3.0), None,
         "one feature is perceived poorly"),
        ("Separability failure", _case_separability, (0.05, 3.0), None,
         "the mean on A shifts with B"),
        ("Decisional failure", _case_decisional, (0.05, 3.0), 0.55,
         "the bound on A shifts with B"),
        ("Dimension neglect", _case_neglect, (0.05, 3.5), None,
         "B is guessed, not perceived"),
    ]
    set_style()
    fig, ax = plt.subplots(2, 4, figsize=(11.4, 6.1),
                           gridspec_kw=dict(height_ratios=[1.05, 1.0]))
    for k, (title, fn, bounds, tilt, sub) in enumerate(cases):
        (zx, zy, rho), P, a = _fit_accuracy(fn, *bounds, tilt=tilt)
        _draw_space(ax[0][k], zx, zy, rho, tilt)
        _draw_matrix(ax[1][k], P)
        ax[0][k].set_title(f"{title}\n", fontsize=11)
        ax[0][k].text(0.5, 1.02, sub, transform=ax[0][k].transAxes, ha="center",
                      va="bottom", fontsize=8.5, color=MUTE, style="italic")
        ax[1][k].set_xlabel(f"overall accuracy {100*a:.0f}%", fontsize=9)
        print(f"{title:24s} accuracy {a:.4f}")
    ax[0][0].set_ylabel("perceptual space", fontsize=10)
    ax[1][0].set_ylabel("confusion matrix (%)", fontsize=10)
    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(OUT); plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
