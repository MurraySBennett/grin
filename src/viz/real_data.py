"""
real_data.py — the manuscript's real-data comparison figures.

Three figures, none of them a bar chart:

  real_data_spaces.png    the classic GRT perceptual space, one row per observer and
                          one column per method, so the fitted representations can be
                          compared as representations rather than as parameter tables.
  real_data_params.png    every parameter for every observer, as GRIN's 95% credible
                          interval with the three point estimates overlaid on it. The
                          question this answers is whether model-class agreement hides
                          disagreement about the representation, which it can.
  real_data_thinning.png  each method's distance from its own full-data estimate as the
                          matrix is resampled to fewer trials, plus its convergence rate
                          over the same range. With no ground truth, self-consistency
                          under thinning is the available stability criterion.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .style import set_style, BLUE, BLUE_DEEP, RED_DEEP, MUTE, INK
from .grt_space import perceptual_space, shared_axis_limit

METHODS = [("grin", "GRIN", BLUE_DEEP), ("mdsdt", "mdsdt", BLUE),
           ("grtools", "grtools", RED_DEEP), ("python_mle", "Python MLE", MUTE)]
PLABELS = ([f"$z_{{x{i}}}$" for i in range(4)] + [f"$z_{{y{i}}}$" for i in range(4)]
           + [f"$\\rho_{i}$" for i in range(4)])


def _spaces(names, methods, path, scale=1.0):
    set_style(scale)
    avail = [m for m in METHODS if not np.all(np.isnan(methods[m[0]]))]
    nr, nc = len(names), len(avail)
    fig, ax = plt.subplots(nr, nc, figsize=(2.5 * nc, 2.5 * nr), squeeze=False)
    for i, nm in enumerate(names):
        thetas = [methods[k][i] for k, _, _ in avail if not np.any(np.isnan(methods[k][i]))]
        lim = shared_axis_limit(thetas) if thetas else 3.0
        for j, (key, label, _) in enumerate(avail):
            a = ax[i][j]
            th = methods[key][i]
            if np.any(np.isnan(th)):
                a.text(0.5, 0.5, "did not\nconverge", ha="center", va="center",
                       transform=a.transAxes, color=MUTE, fontsize=9)
                a.set_xticks([]); a.set_yticks([]); a.set_box_aspect(1)
                for sp in a.spines.values():
                    sp.set_color(MUTE)
            else:
                perceptual_space(a, th, show_level_ticks=(i == 0), lim=lim)
            if i == 0:
                a.set_title(label, fontsize=11 * scale)
            if j == 0:
                a.set_ylabel(nm, fontsize=10 * scale)
    fig.suptitle("Fitted perceptual spaces, five published observers",
                 x=0.02, ha="left", fontweight="bold", fontsize=14 * scale, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path); plt.close(fig)
    return path


def _params(names, g, methods, path, scale=1.0):
    set_style(scale)
    n = len(names)
    fig, ax = plt.subplots(1, n, figsize=(2.9 * n, 5.0), squeeze=False, sharey=True)
    y = np.arange(12)[::-1]
    for i, nm in enumerate(names):
        a = ax[0][i]
        a.hlines(y, g["lo"][i], g["hi"][i], color=BLUE_DEEP, lw=4.5, alpha=0.30,
                 zorder=1)
        a.plot(g["mean"][i], y, "o", color=BLUE_DEEP, ms=5.5, zorder=4, label="GRIN")
        for key, label, col in METHODS[1:]:
            th = methods[key][i]
            if np.all(np.isnan(th)):
                continue
            a.plot(th, y, "|", color=col, ms=9, mew=1.8, zorder=3, label=label)
        a.axvline(0, color=MUTE, lw=0.9, ls=(0, (4, 3)), zorder=0)
        a.set_yticks(y); a.set_title(nm, fontsize=10 * scale)
        a.set_xlabel("estimate")
        if i == 0:
            a.set_yticklabels(PLABELS, fontsize=9 * scale)
            a.legend(fontsize=8 * scale, loc="lower left")
    fig.suptitle("Parameter estimates: GRIN's 95% credible interval, "
                 "with each baseline's point estimate",
                 x=0.02, ha="left", fontweight="bold", fontsize=14 * scale, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path); plt.close(fig)
    return path


def _thinning(sub, full, path, model=None, scale=1.0):
    """sub: the subsample table. full: {dataset: {method: 12-vector}} full-data fits.

    Returns the plotted values as a dict so the manuscript quotes exactly what the
    figure shows, rather than a separately-computed number that can drift from it.
    """
    set_style(scale)
    plotted = {}
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.1))
    levels = sorted(sub["tps_target"].dropna().unique())

    for key, label, col in METHODS:
        cols = ([f"{key}_zx_{i}" for i in range(4)] + [f"{key}_zy_{i}" for i in range(4)]
                + [f"{key}_rho_{i}" for i in range(4)])
        if not all(c in sub.columns for c in cols):
            print(f"  (thinning: no columns for {label}, skipping)")
            continue
        med, conv = [], []
        for lv in levels:
            d = sub[sub["tps_target"] == lv]
            dev = []
            for _, row in d.iterrows():
                ref = full.get(row["dataset"], {}).get(key)
                if ref is None:
                    continue
                th = row[cols].to_numpy(float)
                if np.any(np.isnan(th)) or np.any(np.isnan(ref)):
                    continue
                dev.append(np.abs(th - ref).mean())
            med.append(np.median(dev) if dev else np.nan)
            okcol = f"{key}_ok"
            conv.append(d[okcol].astype(str).str.upper().isin(["TRUE", "1"]).mean()
                        if okcol in d.columns else np.nan)
        ax[0].plot(levels, med, "-o", color=col, ms=5, lw=1.8, label=label)
        ax[1].plot(levels, 100 * np.asarray(conv, float), "-o", color=col, ms=5, lw=1.8,
                   label=label)
        plotted[label] = dict(levels=[float(l) for l in levels],
                              median_drift=[None if not np.isfinite(v) else float(v)
                                            for v in med],
                              convergence=[None if not np.isfinite(c) else float(c)
                                           for c in conv])

    from matplotlib.ticker import NullFormatter
    for a in ax:
        a.set_xscale("log")
        a.set_xticks(levels)
        a.set_xticklabels([f"{int(l)}" for l in levels])
        # a log axis adds its own minor ticks, which overprint the band labels
        a.xaxis.set_minor_formatter(NullFormatter())
        a.tick_params(axis="x", which="minor", length=0)
        a.set_xlabel("trials per stimulus after resampling")
        a.invert_xaxis()
    ax[0].set_ylabel("mean absolute distance from own full-data fit")
    ax[0].set_title("A   Stability as data thin")
    ax[1].set_ylabel("fits converging (%)")
    ax[1].set_title("B   Convergence as data thin")
    ax[1].set_ylim(0, 103)
    ax[0].legend(fontsize=8.5 * scale)
    # several methods sit at 100% convergence and overplot; say so rather than let the
    # reader infer a line is missing
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    import json
    with open(str(path).replace(".png", ".json"), "w") as f:
        json.dump(plotted, f, indent=2)
    return path


def real_data_figures(names, X, Xt, g, methods, figdir, subsample_path=None, model=None):
    os.makedirs(figdir, exist_ok=True)
    made = [_spaces(names, methods, os.path.join(figdir, "real_data_spaces.png")),
            _params(names, g, methods, os.path.join(figdir, "real_data_params.png"))]
    if subsample_path:
        sub = pd.read_csv(subsample_path)
        # full-data reference per method, from the arrays compare_real_data.py already
        # built -- these cover all four methods, whereas the R fit table covers only two
        full = {nm: {k: methods[k][i] for k, _, _ in METHODS}
                for i, nm in enumerate(names)}
        made.append(_thinning(sub, full,
                              os.path.join(figdir, "real_data_thinning.png"), model))
    for p in made:
        print(f"figure -> {p}")
    return made
