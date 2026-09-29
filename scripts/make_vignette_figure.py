"""
Figure 1: the identification design, and five observers at matched overall accuracy.

The point of the figure is that the perceptual spaces are plainly different while the
confusion matrices they produce are not obviously different by eye -- which is the case
for fitting a GRT model rather than reading accuracy off the data.

The panels are organised as the construct factorial GRIN actually reports, rather than
as an assortment of ways to do badly:

  0 the identification design   four lesions, border regularity x colour uniformity
  1 both constructs hold        PI and PS hold; the distributions are simply close
  2 separability fails          the mean on A shifts with the level of B
  3 independence fails          one shared within-stimulus correlation
  4 both fail together          the shift and the correlation at once
  5 decisional separability     perception is fine; the bound on A tilts with B

Panel 4 is what makes componentwise reporting legible: GRIN returns three heads, not a
jointly normalised twelve-class mode, and panel 4 is the case that distinction is for.

Panel 5 is drawn with a tilted bound because that is what a failure of decisional
separability IS. It is outside the parameterisation GRIN estimates (which assumes
decisional separability throughout) and is placed last, set apart, because the paper
introduces it and does not estimate it -- not because the estimator recovers it.

Stimulus images are four ISIC lesions chosen from the BTL feature-strength scores in
projects/melanoma-identification (pwc/data/estimates/btl_cv_data.csv) at crossed
extremes of perceived border irregularity and colour variation, filtered to CC-0 and
CC-BY. See data/vignette_stimuli/MANIFEST.json for ids, licences and attribution.

    python scripts/make_vignette_figure.py
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Ellipse

from src.config import FIGURES_DIR, PROJECT_ROOT
from src.viz.style import set_style, BLUE, RED, BLUE_DEEP, RED_DEEP, INK, MUTE, CMAP_SEQ
import src.grt_model as gm

OUT = os.path.join(FIGURES_DIR, "vignette.png")
STIM_DIR = os.path.join(PROJECT_ROOT, "data", "vignette_stimuli")

# The old 48% target was forced by a dimension-neglect panel whose ceiling was 50%. That
# panel is gone (it was never explored anywhere else in the paper), so the target is now
# free, and 60% is \compOverall -- the overall accuracy at which the frontier analysis
# says the PI and separability classification curves cross. The opening figure and the
# design guidance therefore describe the same operating point.
TARGET_ACC = 0.60
ACC_TOL = 0.004

STIM_COL = [BLUE, RED, BLUE_DEEP, RED_DEEP]
LABELS = ["A$_1$B$_1$", "A$_1$B$_2$", "A$_2$B$_1$", "A$_2$B$_2$"]
PRINT_W = 6.5   # \textwidth in the manuscript

# A = border regularity (x), B = colour uniformity (y). Level 1 is the low end of each
# axis: A1 regular, A2 irregular; B1 uniform, B2 varied. Stimulus order is the canonical
# (A1B1, A1B2, A2B1, A2B2), so indices 1 and 3 are the B2 (varied colour) stimuli.
XLAB = "perceived border irregularity"
YLAB = "perceived colour variation"

# Size of the separability failure, as an absolute shift in the A percept when the colour
# is varied. Absolute rather than proportional to s: a shift that scaled with s would
# fight the bisection, because growing s to raise accuracy would move A1B2 further onto
# the wrong side of the bound and the target would become unreachable.
#
# 0.7 leaves A1B2 just inside the correct side of the bound (zx = -0.10). Larger values
# read more dramatically in the space but push the matrix to a 48% off-diagonal cell,
# which starts to make the failure self-diagnosing by eye and undercuts the figure's
# own claim. A separability failure is always the more visible of the two constructs
# in a confusion matrix -- marginal sensitivities are informed by every trial, whereas
# a correlation only moves the residual concordant/discordant balance -- so the honest
# figure shows that asymmetry rather than suppressing it.
PS_SHIFT = 0.7
# One shared within-stimulus correlation, the 1rho class of the twelve.
PI_RHO = 0.60
DS_TILT = 0.55

STIM_CELLS = [
    ("A1B1_regular_uniform",   "A$_1$B$_1$"),
    ("A1B2_regular_varied",    "A$_1$B$_2$"),
    ("A2B1_irregular_uniform", "A$_2$B$_1$"),
    ("A2B2_irregular_varied",  "A$_2$B$_2$"),
]


def _acc(P):
    return float(np.mean(np.diag(P)))


# ---- the five observers -----------------------------------------------------
# Each takes the one free parameter the bisection moves and returns (zx, zy, rho).

def _case_baseline(s):
    """PI and PS both hold. Nothing interacts; the distributions are just close
    together. This is the reference the other panels are read against, and the only
    one whose follow-up is plainly more practice."""
    zx = np.array([-s, -s, s, s])
    zy = np.array([-s, s, -s, s])
    return zx, zy, np.zeros(4)


def _case_ps(s):
    """Perceptual separability fails: the mean of the border percept shifts by a
    constant when the colour is varied, so the border looks more irregular at B2.
    The A1-A2 separation is preserved -- both levels slide together."""
    zx = np.array([-s, -s + PS_SHIFT, s, s + PS_SHIFT])
    zy = np.array([-s, s, -s, s])
    return zx, zy, np.zeros(4)


def _case_pi(s):
    """Perceptual independence fails: both features are perceived at full acuity and
    neither mean depends on the other, but the two percepts covary within a stimulus."""
    zx = np.array([-s, -s, s, s])
    zy = np.array([-s, s, -s, s])
    return zx, zy, np.full(4, PI_RHO)


def _case_both(s):
    """Both fail at once. Componentwise reporting is what separates this from the two
    single-failure panels; a single twelve-class label would not."""
    zx = np.array([-s, -s + PS_SHIFT, s, s + PS_SHIFT])
    zy = np.array([-s, s, -s, s])
    return zx, zy, np.full(4, PI_RHO)


def _case_ds(s):
    """Perception is separable and independent; the failure is in the bound, which is
    applied below by _probs_tilted rather than encoded in these parameters."""
    zx = np.array([-s, -s, s, s])
    zy = np.array([-s, s, -s, s])
    return zx, zy, np.zeros(4)


CASES = [
    ("Both constructs hold", _case_baseline, (0.05, 3.5), None,
     "close distributions"),
    ("Separability fails", _case_ps, (0.05, 3.5), None,
     "border mean shifts with colour"),
    ("Independence fails", _case_pi, (0.05, 3.5), None,
     "one shared correlation"),
    ("Both fail together", _case_both, (0.05, 3.5), None,
     "shift and correlation at once"),
    ("Decisional separability fails", _case_ds, (0.05, 3.5), DS_TILT,
     "introduced here, not estimated"),
]


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
    a = float("nan")
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        zx, zy, rho = fn(mid)
        P = _probs_tilted(zx, zy, tilt) if tilt is not None else \
            gm.forward_probabilities(zx, zy, rho)
        a = _acc(np.asarray(P))
        if abs(a - TARGET_ACC) < ACC_TOL:
            return fn(mid), np.asarray(P), a
        if a < TARGET_ACC:
            lo = mid
        else:
            hi = mid
    raise RuntimeError(
        f"bisection did not reach {TARGET_ACC:.2f} (closest {a:.4f}); "
        "widen the bracket or soften the failure size")


def _draw_space(ax, zx, zy, rho, tilt=None, axis_labels=False, label_fs=6.5):
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
        #
        # The ellipse now follows rho. It used to be hardcoded circular, which was
        # invisible while every case had rho = 0 and would have silently drawn the
        # independence panel as though it were independent. For unit variances the
        # covariance [[1, r], [r, 1]] has eigenvalues 1 +/- r on the +/-45 degree axes.
        r = float(rho[i])
        ax.add_patch(Ellipse((zx[i], zy[i]),
                             2 * np.sqrt(1 + r), 2 * np.sqrt(1 - r), angle=45,
                             fill=False, edgecolor=STIM_COL[i], lw=1.5))
        ax.plot(zx[i], zy[i], "o", color=STIM_COL[i], ms=4)
    ax.set_xlim(-4.2, 4.2); ax.set_ylim(-4.2, 4.2)
    ax.set_box_aspect(1); ax.set_xticks([]); ax.set_yticks([])
    if axis_labels:
        ax.set_xlabel(XLAB, fontsize=label_fs, color=MUTE)
        ax.set_ylabel(YLAB, fontsize=label_fs, color=MUTE)


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


def _square(img):
    """Centre crop to square so the four lesions are shown on the same footing."""
    h, w = img.shape[:2]
    m = min(h, w)
    t, l = (h - m) // 2, (w - m) // 2
    return img[t:t + m, l:l + m]


def _draw_design(axes, title_fs=9.5, sub_fs=7.5, tick_fs=6.5):
    """The identification design: the 2x2 of stimuli every observer below is judging.

    This is a property of the experiment, not of an observer, so it gets the first slot
    and no confusion matrix -- dropped into the run of case panels with a matrix under
    it, it would read as a sixth observer."""
    for ax, (stem, lab) in zip(axes, STIM_CELLS):
        p = os.path.join(STIM_DIR, f"{stem}.jpg")
        ax.imshow(_square(mpimg.imread(p)))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(MUTE); sp.set_linewidth(0.8)
        ax.set_title(lab, fontsize=tick_fs + 0.5, color=INK, pad=2)
    axes[0].set_ylabel("regular\nborder", fontsize=tick_fs, color=MUTE)
    axes[2].set_ylabel("irregular\nborder", fontsize=tick_fs, color=MUTE)
    axes[2].set_xlabel("uniform colour", fontsize=tick_fs, color=MUTE)
    axes[3].set_xlabel("varied colour", fontsize=tick_fs, color=MUTE)


def _draw_case(ax_top, ax_bot, title, sub, a, zx, zy, rho, tilt, P,
               title_fs=9.5, sub_fs=7.5, label_fs=7.5, tick_fs=6.5,
               axis_labels=False, dim=False):
    _draw_space(ax_top, zx, zy, rho, tilt, axis_labels=axis_labels, label_fs=tick_fs)
    _draw_matrix(ax_bot, P, label_fs=label_fs, tick_fs=tick_fs)
    ax_top.set_title(f"{title}\n", fontsize=title_fs,
                     color=MUTE if dim else INK)
    ax_top.text(0.5, 1.02, sub, transform=ax_top.transAxes, ha="center",
                va="bottom", fontsize=sub_fs, color=MUTE, style="italic")


def main():
    set_style()
    fitted = []
    for title, fn, bounds, tilt, sub in CASES:
        (zx, zy, rho), P, a = _fit_accuracy(fn, *bounds, tilt=tilt)
        fitted.append((title, sub, a, zx, zy, rho, tilt, P))
        print(f"{title:30s} accuracy {a:.4f}   s-space zx={np.round(zx,2)} rho={rho[0]:.2f}")

    # ---- combined: 2 rows x 3 columns, design first, DS last ------------------
    fig = plt.figure(figsize=(PRINT_W, PRINT_W * 1.26))
    outer = fig.add_gridspec(2, 3, wspace=0.42, hspace=0.30,
                             left=0.075, right=0.985, bottom=0.055, top=0.925)

    # slot (0,0): the design. Nested to the same 1.05/1.0 split the case panels use, with
    # the stimuli occupying only the upper block, so they line up with the perceptual
    # spaces beside them instead of drifting apart down a full-height cell.
    gs_d0 = outer[0, 0].subgridspec(2, 1, height_ratios=[1.05, 1.0], hspace=0.30)
    gs_d = gs_d0[0].subgridspec(2, 2, wspace=0.12, hspace=0.30)
    dax = [fig.add_subplot(gs_d[i, j]) for i in range(2) for j in range(2)]
    _draw_design(dax)
    dax[0].text(0.0, 1.42, "The identification design", transform=dax[0].transAxes,
                ha="left", va="bottom", fontsize=9.5, color=INK)
    dax[0].text(0.0, 1.26, "four lesions, two features crossed",
                transform=dax[0].transAxes, ha="left", va="bottom",
                fontsize=7.5, color=MUTE, style="italic")

    # The design slot has no confusion matrix under it, which leaves the one block of
    # free space in the figure. A reader meeting GRT here needs to be told how to read a
    # panel once; saying it in the figure beats saying it in a caption they may not have
    # to hand, or not at all.
    key = fig.add_subplot(gs_d0[1]); key.axis("off")
    key.text(0.0, 1.0, "How to read each observer", transform=key.transAxes,
             ha="left", va="top", fontsize=8.5, color=INK)
    key.text(0.0, 0.88,
             "Above: the perceptual space.\n"
             "One ellipse per stimulus (1 SD);\n"
             "dashed lines are the bounds.\n\n"
             "Below: the confusion matrix it\n"
             "implies. Rows are stimuli shown,\n"
             "columns responses made, so the\n"
             "diagonal is correct.\n\n"
             "All five observers share one\n"
             "overall accuracy. The spaces\n"
             "differ; whether the matrices\n"
             "reveal which construct failed\n"
             "is the question.",
             transform=key.transAxes, ha="left", va="top",
             fontsize=6.5, color=MUTE, linespacing=1.4)

    # slots 1-5: the observers, in reading order
    slots = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    for k, (title, sub, a, zx, zy, rho, tilt, P) in enumerate(fitted):
        r, c = slots[k]
        gs_c = outer[r, c].subgridspec(2, 1, height_ratios=[1.05, 1.0], hspace=0.30)
        ax_top = fig.add_subplot(gs_c[0]); ax_bot = fig.add_subplot(gs_c[1])
        _draw_case(ax_top, ax_bot, title, sub, a, zx, zy, rho, tilt, P,
                   axis_labels=(k == 0), dim=(k == len(fitted) - 1))
        # Left-column panels only. Labelling the first case panel too put the text in the
        # design column, on top of the reading key.
        if c == 0:
            ax_bot.set_ylabel("confusion matrix (%)", fontsize=7.5, color=MUTE)

    # All five observers are matched to the same target accuracy by construction (see
    # TARGET_ACC/ACC_TOL above) -- stating that once for the whole figure is the point;
    # repeating it under every panel was the kind of redundancy that just adds clutter.
    fig.text(0.5, 0.012,
             f"overall accuracy {100*TARGET_ACC:.0f}% for all five observers",
             ha="center", va="bottom", fontsize=9, color=MUTE)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(OUT, dpi=400); plt.close(fig)
    print(f"wrote {OUT}")

    # ---- individual: one case, its own space + confusion-matrix pair ----------
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
        fig.savefig(p, dpi=400); plt.close(fig)
        print(f"wrote {p}")

    # ---- the design panel on its own, for talks -------------------------------
    fig, ax3 = plt.subplots(2, 2, figsize=(PRINT_W * 0.42, PRINT_W * 0.46))
    _draw_design([ax3[i][j] for i in range(2) for j in range(2)], tick_fs=8)
    fig.tight_layout()
    p = OUT.replace(".png", "_design.png")
    fig.savefig(p, dpi=400); plt.close(fig)
    print(f"wrote {p}")

    man = json.load(open(os.path.join(STIM_DIR, "MANIFEST.json")))
    print("\nstimulus licences (attribution required for anything but CC-0):")
    for cell, m in man.items():
        print(f"  {cell:26s} {m['isic_id']}  {m['licence']:6s}  {m['attribution']}")


if __name__ == "__main__":
    main()
