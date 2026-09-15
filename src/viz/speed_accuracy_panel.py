"""
speed_accuracy_panel.py — speed and parameter-recovery accuracy as a function of trial
count, WITH uncertainty. Replaces the old Table 3 (one aggregate speed number per
method) and Table 4 (MAE by trial band, no uncertainty) with a three-panel figure:
richer than either table on its own, and the two quantities share an x-axis so the
"where does GRIN's advantage actually come from" story reads as one picture rather than
two separately-scaled tables.

  A  ms per matrix by trial count (log y). GRIN is reported at single-matrix latency
     only -- batched throughput is a different claim (amortised over many matrices at
     once) and isn't "time to answer for one observer who just walked in"; the batched
     number is still true and still reported, but as a caption note, not a data series
     competing with four single-call numbers on the same axis.
  B  marginal-sensitivity MAE by trial count (linear y), on paired complete cases.
  C  within-stimulus-correlation MAE by trial count, on the same complete cases.

Error bars/bands are percentile-bootstrap 95% CIs of the mean (src/viz/recovery.py's
_boot_ci, same convention fig:recovery already uses) -- not +-1 SD. SD on a log-scale,
right-skewed latency distribution produced a nonsensical band in the first version of
this figure (GRIN's speed band reached down to 1e-4ms after clipping a negative lower
bound); the bootstrap CI stays inside the actual data range regardless of skew.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from .style import set_style, BLUE, BLUE_DEEP, RED_DEEP, MUTE, floating_trial_bands

PRINT_W = 6.5
METHOD_COL = {"GRIN": BLUE_DEEP, "mdsdt": BLUE, "grtools": RED_DEEP, "Python-MLE": MUTE}
ORDER = ["GRIN", "mdsdt", "grtools", "Python-MLE"]


def _panel_speed(ax, out, scale, letter="A"):
    bands = out["bands"]
    series, err = {}, {}
    for m in ORDER:
        d = out["methods"][m]
        mean = [v if v is not None else np.nan for v in d["speed_ms_mean"]]
        lo = d["speed_ms_lo"]
        hi = d["speed_ms_hi"]
        series[m] = (mean, METHOD_COL[m])
        err[m] = [None if (a is None or b is None) else (a, b) for a, b in zip(lo, hi)]
    floating_trial_bands(ax, bands, series, offset_span=0.16, err=err)
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 6)          # headroom: every method is ~flat across trial
                                      # count, so no empty region exists at the data's
                                      # own scale for the legend to sit in
    ax.tick_params(axis="x", labelsize=7.5 * scale)
    ax.set_xlabel("trials per stimulus")
    ax.set_ylabel("ms per matrix (log)")
    ax.set_title(f"{letter}   Speed by trial count" if letter else "Speed by trial count")
    ax.legend(fontsize=7.5 * scale, loc="upper right", frameon=False, handlelength=1.4,
              ncol=2, columnspacing=1.0)


def _panel_accuracy(ax, out, scale, family, letter="B"):
    bands = out["bands"]
    series, err = {}, {}
    for m in ORDER:
        d = out["methods"][m]
        fd = d["family_mae"][family]
        mean = [v if v is not None else np.nan for v in fd["mae_mean"]]
        lo = fd["mae_lo"]
        hi = fd["mae_hi"]
        series[m] = (mean, METHOD_COL[m])
        err[m] = [None if (a is None or b is None) else (a, b) for a, b in zip(lo, hi)]
    floating_trial_bands(ax, bands, series, offset_span=0.16, err=err)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="x", labelsize=7.5 * scale)
    ax.set_xlabel("trials per stimulus")
    label = "sensitivity" if family == "z" else "correlation"
    ax.set_ylabel(f"{label} MAE")
    ax.set_title(f"{letter}   {label.capitalize()} accuracy" if letter
                 else f"{label.capitalize()} accuracy")


def speed_accuracy_figure(out, path, scale=1.0):
    row_scale = scale * 0.78
    set_style(row_scale)
    stem, ext = os.path.splitext(path)

    fig, ax = plt.subplots(1, 3, figsize=(PRINT_W, PRINT_W * 0.40))
    _panel_speed(ax[0], out, row_scale)
    _panel_accuracy(ax[1], out, row_scale, "z", letter="B")
    _panel_accuracy(ax[2], out, row_scale, "rho", letter="C")
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.30, top=0.86, wspace=0.48)
    fig.savefig(path); plt.close(fig)
    print(f"figure -> {path}")

    set_style(scale)
    out2 = f"{stem}_2row{ext}"
    fig, ax = plt.subplots(3, 1, figsize=(PRINT_W * 0.62, PRINT_W * 1.55))
    _panel_speed(ax[0], out, scale)
    _panel_accuracy(ax[1], out, scale, "z", letter="B")
    _panel_accuracy(ax[2], out, scale, "rho", letter="C")
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.09, top=0.96, hspace=0.78)
    fig.savefig(out2); plt.close(fig)
    print(f"figure -> {out2}")

    panels = [
        ("a", lambda a: _panel_speed(a, out, scale, letter="")),
        ("b", lambda a: _panel_accuracy(a, out, scale, "z", letter="")),
        ("c", lambda a: _panel_accuracy(a, out, scale, "rho", letter="")),
    ]
    for tag, fn in panels:
        fig, a = plt.subplots(1, 1, figsize=(PRINT_W, PRINT_W * 0.62))
        fn(a)
        fig.tight_layout()
        p = f"{stem}_{tag}{ext}"
        fig.savefig(p); plt.close(fig)
        print(f"figure -> {p}")
    return path
