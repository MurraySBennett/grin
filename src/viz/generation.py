"""
generation.py — prior-coverage figures for the simulated corpus.

Replaces the ad-hoc plotting that used to live inside GRTDataGenerator._plot_coverage.
Two changes of substance beyond styling: the prior boundaries (Z_MAX, R_MAX) are drawn on
the panels whose whole job is to show whether the prior covers the space, and
trials_per_stimulus — computed and printed but previously never plotted — gets a panel.

Writes the composite `coverage_report.png` AND each panel as its own file under
`<figures>/generation/`, so single panels can be lifted into a talk without a screenshot.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .style import set_style, BLUE, BLUE_DEEP, ROSE_DEEP, INK, MUTE


def _chance_line(ax, x=0.25, label="chance (0.25)"):
    ax.axvline(x, color=ROSE_DEEP, lw=1.4, ls=(0, (4, 3)), label=label)
    ax.legend(fontsize=8.5)


def _bounds(ax, *xs, label="prior bound"):
    for i, x in enumerate(xs):
        ax.axvline(x, color=MUTE, lw=1.2, ls=(0, (1, 3)),
                   label=label if i == 0 else None)
    ax.legend(fontsize=8.5)


# Each entry: key -> (title, xlabel, draw_fn(ax, stats, z_max, r_max))
def _p_overall(ax, s, z_max, r_max):
    ax.hist(s["overall_accuracy"], bins=40, color=BLUE, edgecolor="white", linewidth=0.3)
    _chance_line(ax)
    ax.set_xlabel("proportion correct"); ax.set_ylabel("count")
    ax.set_title("Overall accuracy")


def _p_perstim(ax, s, z_max, r_max):
    ax.hist(np.asarray(s["per_stimulus_accuracy"]).ravel(), bins=40, color=BLUE,
            edgecolor="white", linewidth=0.3)
    _chance_line(ax)
    ax.set_xlabel("proportion correct"); ax.set_ylabel("count")
    ax.set_title("Per-stimulus accuracy")


def _p_congruency(ax, s, z_max, r_max):
    ax.hist(s["congruency_asymmetry"], bins=40, color=BLUE, edgecolor="white", linewidth=0.3)
    ax.axvline(0, color=MUTE, lw=1.2, ls=(0, (1, 3)))
    ax.set_xlabel("A1B1/A2B2 accuracy $-$ A1B2/A2B1 accuracy"); ax.set_ylabel("count")
    ax.set_title("Congruency asymmetry")


def _p_bias(ax, s, z_max, r_max):
    bins = np.histogram_bin_edges(
        np.concatenate([s["x_response_bias"], s["y_response_bias"]]), bins=40)
    ax.hist(s["x_response_bias"], bins=bins, histtype="step", lw=1.8, color=BLUE,
            label="dimension A (x)")
    ax.hist(s["y_response_bias"], bins=bins, histtype="step", lw=1.8, color=ROSE_DEEP,
            label="dimension B (y)")
    ax.axvline(0, color=MUTE, lw=1.2, ls=(0, (1, 3)))
    ax.set_xlabel("response bias"); ax.set_ylabel("count")
    ax.set_title("Response bias"); ax.legend(fontsize=8.5)


def _p_zscore(ax, s, z_max, r_max):
    ax.hist(s["abs_zscore"], bins=40, color=BLUE, edgecolor="white", linewidth=0.3)
    _bounds(ax, z_max, label=f"prior bound ($Z_{{max}}$ = {z_max:g})")
    ax.set_xlabel("$|z|$ (sensitivity)"); ax.set_ylabel("count")
    ax.set_title("Sensitivity coverage")


def _p_corr(ax, s, z_max, r_max):
    ax.hist(s["correlation"], bins=40, color=BLUE, edgecolor="white", linewidth=0.3)
    _bounds(ax, -r_max, r_max, label=f"prior bound ($R_{{max}}$ = {r_max:g})")
    ax.set_xlabel(r"perceptual correlation $\rho$"); ax.set_ylabel("count")
    ax.set_title("Correlation coverage")


def _p_trials(ax, s, z_max, r_max):
    t = np.asarray(s["trials_per_stimulus"]).ravel()
    t = t[t > 0]
    ax.hist(t, bins=np.logspace(np.log10(max(t.min(), 1)), np.log10(t.max()), 40),
            color=BLUE, edgecolor="white", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_xlabel("trials per stimulus (log scale)"); ax.set_ylabel("count")
    ax.set_title("Trial-count coverage")


PANELS = [
    ("overall_accuracy", _p_overall),
    ("per_stimulus_accuracy", _p_perstim),
    ("congruency_asymmetry", _p_congruency),
    ("response_bias", _p_bias),
    ("sensitivity", _p_zscore),
    ("correlation", _p_corr),
    ("trials_per_stimulus", _p_trials),
]


def coverage_figures(stats, composite_path, panel_dir=None, z_max=3.0, r_max=0.9,
                     scale=1.0):
    """Write the composite coverage report and (optionally) one file per panel.

    stats: the dict returned by GRTDataGenerator.report_coverage().
    """
    set_style(scale)

    fig, axes = plt.subplots(3, 3, figsize=(14.5, 11))
    flat = axes.ravel()
    for ax, (_, fn) in zip(flat, PANELS):
        fn(ax, stats, z_max, r_max)
    for ax in flat[len(PANELS):]:
        ax.set_visible(False)
    fig.suptitle("Prior coverage of the simulated corpus", x=0.02, ha="left",
                 fontweight="bold", fontsize=15 * scale, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(composite_path)
    plt.close(fig)

    if panel_dir:
        os.makedirs(panel_dir, exist_ok=True)
        for name, fn in PANELS:
            f, a = plt.subplots(figsize=(6.4, 4.6))
            fn(a, stats, z_max, r_max)
            f.tight_layout()
            f.savefig(os.path.join(panel_dir, f"{name}.png"))
            plt.close(f)
    return composite_path
