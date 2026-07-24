"""
style.py — the GRIN figure identity.

A restrained blue / rose / soft-white palette on clean axes: only the left and
bottom spines, no gridlines, clearly labelled axes, ticks, titles and legends.
Call set_style() once before plotting. Use the palette constants and colormaps
so colour is consistent across every figure (poster, talk, paper).
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- core palette ---------------------------------------------------------
BLUE      = "#5AA9E6"     # sky
BLUE_DEEP = "#2E6CA4"
RED       = "#F2A5C0"     # soft rose/red
RED_DEEP  = "#C86A93"
# Back-compat aliases: the palette was renamed RED/RED_DEEP, but older modules (and any
# saved notebooks) still import ROSE/ROSE_DEEP. Keep both names bound to the same colours
# so a rename never silently changes a figure's appearance or raises ImportError.
ROSE      = RED
ROSE_DEEP = RED_DEEP
INK       = "#2B2B2E"     # text / spines
MUTE      = "#9AA0A6"     # reference lines, secondary
PAPER     = "#FFFFFF"

# categorical order for multi-series plots
PALETTE = [BLUE, RED, BLUE_DEEP, RED_DEEP, MUTE]

# sequential (0 -> max), e.g. confusion / error heatmaps
CMAP_SEQ = LinearSegmentedColormap.from_list("grin_seq", ["#FFFFFF", "#CDE6F7", BLUE, BLUE_DEEP])
# diverging (signed), blue <-> white <-> rose
CMAP_DIV = LinearSegmentedColormap.from_list("grin_div", [BLUE_DEEP, "#FFFFFF", RED_DEEP])


_BG = PAPER          # background used by every subsequent set_style() call


def set_background(color):
    """Set the figure background for all later set_style() calls.

    Pass None for a TRANSPARENT background, which is what you want on a poster or slide
    with a coloured backdrop -- an opaque white figure otherwise sits in a white box on a
    tinted page. Pass a hex string to match a specific backdrop exactly.

    Call this once before rendering; the figure functions call set_style() internally with
    only a scale, so the background has to live at module level rather than be threaded
    through every signature.
    """
    global _BG
    _BG = color


def set_style(scale=1.0, bg="__inherit__"):
    """Apply the theme. scale>1 enlarges type for posters/talks.

    bg: "__inherit__" uses whatever set_background() last specified; None forces
    transparent; a colour string forces that colour.
    """
    b = _BG if bg == "__inherit__" else bg
    transparent = b is None
    face = "none" if transparent else b
    mpl.rcParams.update({
        "figure.facecolor": face, "axes.facecolor": face, "savefig.facecolor": face,
        "savefig.transparent": transparent,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False,
        "axes.edgecolor": INK, "axes.linewidth": 1.1,
        "axes.titlelocation": "left", "axes.titleweight": "bold",
        "axes.titlesize": 13 * scale, "axes.titlecolor": INK, "axes.titlepad": 8,
        "axes.labelcolor": INK, "axes.labelsize": 11.5 * scale, "axes.labelpad": 5,
        "text.color": INK,
        "xtick.color": INK, "ytick.color": INK,
        "xtick.labelsize": 9.5 * scale, "ytick.labelsize": 9.5 * scale,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 4, "ytick.major.size": 4,
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "legend.frameon": False, "legend.fontsize": 10 * scale,
        "figure.dpi": 120, "savefig.dpi": 220, "savefig.bbox": "tight",
    })


def despine_heatmap(ax):
    for s in ax.spines.values():
        s.set_visible(False)


def clean_colorbar(cbar, label=None):
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=3, labelsize=9)
    if label:
        cbar.set_label(label)
