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
                          matrix is resampled to fewer trials, with observer-aware
                          hierarchical bootstrap intervals. With no ground truth,
                          self-consistency under thinning is the available criterion.
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
    # All five real datasets are the example matrices distributed WITH mdsdt (see the
    # Empirical illustration section), so that is the one column that is actually the
    # source of the data rather than just another fit of it -- level ticks (A1/A2/B1/B2)
    # go there, once per row, instead of blanket-labelling the whole first row.
    source_col = next((j for j, (k, _, _) in enumerate(avail) if k == "mdsdt"), None)
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
                # Black, not the 4-colour stimulus palette: with the ellipses already
                # separated spatially and the confusion-matrix figure carrying the
                # colour-coded version, the extra hue here was decoration, not signal.
                perceptual_space(a, th, palette=[INK] * 4,
                                 show_level_ticks=(j == source_col), lim=lim)
            if i == 0:
                a.set_title(label, fontsize=11 * scale)
            if j == 0:
                a.set_ylabel(nm, fontsize=10 * scale)
    # No figure title -- that's LaTeX caption content, not baked into the PNG.
    fig.tight_layout()
    fig.savefig(path); plt.close(fig)
    return path


def _params(names, g, methods, path, scale=1.0):
    set_style(scale)
    n = len(names)
    # Native width close to (rather than ~2x) the manuscript's print width: 5 forest-plot
    # columns genuinely need more than \textwidth/5 each to stay legible, so this accepts
    # a modest remaining shrink (~0.8x) rather than the ~0.45x the original 14.5in-wide
    # figure suffered -- see docs/figure_sizing.md.
    fig, ax = plt.subplots(1, n, figsize=(1.62 * n, 4.6), squeeze=False, sharey=True)
    y = np.arange(12)[::-1]
    # The three baselines' tick marks used to all sit dead-centre on GRIN's row, so when
    # two methods' point estimates were close (the common case -- they mostly agree),
    # their ticks landed on top of each other and were unreadable as separate marks. A
    # small fixed vertical dodge per baseline turns "one smudge" back into three legible
    # ticks even when their x-positions are nearly identical.
    dodge = {"mdsdt": 0.16, "grtools": 0.0, "python_mle": -0.16}
    for i, nm in enumerate(names):
        a = ax[0][i]
        a.hlines(y, g["lo"][i], g["hi"][i], color=BLUE_DEEP, lw=4.5, alpha=0.30,
                 zorder=1)
        a.plot(g["mean"][i], y, "o", color=BLUE_DEEP, ms=5.5, zorder=4, label="GRIN")
        for key, label, col in METHODS[1:]:
            th = methods[key][i]
            if np.all(np.isnan(th)):
                continue
            a.plot(th, y + dodge.get(key, 0.0), "|", color=col, ms=7, mew=2.0,
                  zorder=3, label=label)
        a.axvline(0, color=MUTE, lw=0.9, ls=(0, (4, 3)), zorder=0)
        a.set_yticks(y); a.set_title(nm, fontsize=10.5 * scale)
        a.tick_params(axis="x", labelsize=8.5 * scale)
        if i == 0:
            a.set_yticklabels(PLABELS, fontsize=10 * scale)
    # One shared x-label instead of "estimate" repeated under every one of the 5
    # panels -- the panels already share a y-axis (sharey=True), so they read as one
    # coordinate system, not 5 independent ones each needing its own label.
    fig.text(0.5, 0.01, "estimate", ha="center", va="bottom", fontsize=9.5 * scale)
    # A legend drawn inside the first panel (loc="lower left") sat on top of the rho
    # rows' data points -- every row has an estimate somewhere in that corner for at
    # least one dataset. Pull it out to a horizontal strip at the top instead, where it
    # cannot collide with plotted data. No figure title -- that's LaTeX caption content.
    handles, labels = ax[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=len(handles), fontsize=8.5 * scale, frameon=False)
    fig.subplots_adjust(top=0.90, bottom=0.10)
    fig.savefig(path); plt.close(fig)
    return path


PRINT_W = 6.5   # \textwidth in GRIN_combined_edited.tex
THIN_BOOTSTRAP_DRAWS = 5000
THIN_BOOTSTRAP_SEED = 20260902


def _hierarchical_median_interval(groups, rng, n_draws=THIN_BOOTSTRAP_DRAWS):
    """95% bootstrap interval for the pooled median, respecting observer clustering.

    Observers are sampled first, then repetitions within each sampled observer. This
    avoids treating the repeated thinnings of one five-observer dataset as independent
    observers. The interval remains descriptive because there are only five source
    observers.
    """
    groups = [np.asarray(g, float) for g in groups if len(g)]
    if not groups:
        return np.nan, np.nan
    draws = np.empty(n_draws, float)
    for b in range(n_draws):
        selected = rng.integers(0, len(groups), size=len(groups))
        sample = np.concatenate([
            rng.choice(groups[i], size=len(groups[i]), replace=True) for i in selected
        ])
        draws[b] = np.median(sample)
    return tuple(np.quantile(draws, [0.025, 0.975]))


def _thinning_data(sub, full, family):
    """Pure data pass, no plotting: returns (levels, per_method, plotted) so the
    figure-drawing and the JSON dump can never compute different numbers from the
    same inputs."""
    if family not in {"z", "rho"}:
        raise ValueError("family must be 'z' or 'rho'")
    family_slice = slice(0, 8) if family == "z" else slice(8, 12)
    levels = sorted(sub["tps_target"].dropna().unique())
    per_method = {}
    plotted = {}
    for method_id, (key, label, col) in enumerate(METHODS):
        cols = ([f"{key}_zx_{i}" for i in range(4)] + [f"{key}_zy_{i}" for i in range(4)]
                + [f"{key}_rho_{i}" for i in range(4)])
        if not all(c in sub.columns for c in cols):
            print(f"  (thinning: no columns for {label}, skipping)")
            continue
        med, lo, hi, conv = [], [], [], []
        for lv in levels:
            d = sub[sub["tps_target"] == lv]
            dev = []
            observer_groups = []
            for dataset, observer_rows in d.groupby("dataset", sort=False):
                ref = full.get(dataset, {}).get(key)
                if ref is None or np.any(np.isnan(ref[family_slice])):
                    continue
                observer_dev = []
                for _, row in observer_rows.iterrows():
                    th = row[cols].to_numpy(float)
                    if np.any(np.isnan(th)):
                        continue
                    observer_dev.append(np.abs(
                        th[family_slice] - ref[family_slice]).mean())
                if observer_dev:
                    observer_groups.append(observer_dev)
                    dev.extend(observer_dev)
            med.append(np.median(dev) if dev else np.nan)
            rng = np.random.default_rng(
                THIN_BOOTSTRAP_SEED + method_id * 10000 + int(lv)
            )
            lv_lo, lv_hi = _hierarchical_median_interval(observer_groups, rng)
            lo.append(lv_lo); hi.append(lv_hi)
            okcol = f"{key}_ok"
            conv.append(d[okcol].astype(str).str.upper().isin(["TRUE", "1"]).mean()
                        if okcol in d.columns else np.nan)
        per_method[key] = (label, col, med, lo, hi, conv)
        plotted[label] = dict(levels=[float(l) for l in levels],
                              median_drift=[None if not np.isfinite(v) else float(v)
                                            for v in med],
                              ci95_low=[None if not np.isfinite(v) else float(v)
                                        for v in lo],
                              ci95_high=[None if not np.isfinite(v) else float(v)
                                         for v in hi],
                              convergence=[None if not np.isfinite(c) else float(c)
                                           for c in conv])
    return levels, per_method, plotted


def _panel_drift(ax, levels, per_method, scale, family, letter="A", legend=True):
    from matplotlib.ticker import NullFormatter
    for key, (label, col, med, lo, hi, conv) in per_method.items():
        ax.fill_between(levels, lo, hi, color=col, alpha=0.11, linewidth=0, zorder=1)
        ax.plot(levels, med, "-o", color=col, ms=5, lw=1.8, label=label, zorder=2)
    ax.set_xscale("log")
    ax.set_xticks(levels)
    ax.set_xticklabels([f"{int(l)}" for l in levels])
    ax.xaxis.set_minor_formatter(NullFormatter())  # log axis else overprints band labels
    ax.tick_params(axis="x", which="minor", length=0)
    ax.set_xlabel("trials/stimulus, resampled")
    ax.invert_xaxis()
    label = "sensitivity" if family == "z" else "correlation"
    ax.set_ylabel(f"{label} drift")
    ax.set_title(f"{letter}   {label.capitalize()} stability" if letter
                 else f"{label.capitalize()} stability")
    if legend:
        # Shown once (this panel only) -- both panels plot the exact same four methods
        # in the exact same colours, so a second copy in panel B was pure repetition.
        ax.legend(fontsize=7.5 * scale, loc="upper left", frameon=False, handlelength=1.4)


def _panel_convergence(ax, levels, per_method, scale, letter="B", legend=False):
    from matplotlib.ticker import NullFormatter
    for key, (label, col, med, lo, hi, conv) in per_method.items():
        ax.plot(levels, 100 * np.asarray(conv, float), "-o", color=col, ms=5, lw=1.8,
                label=label)
    ax.set_xscale("log")
    ax.set_xticks(levels)
    ax.set_xticklabels([f"{int(l)}" for l in levels])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="minor", length=0)
    ax.set_xlabel("trials/stimulus, resampled")
    ax.invert_xaxis()
    # "Converging" alone assumes the reader already knows this means each package's OWN
    # pass/fail check on the resampled fit, not agreement with any reference -- spelling
    # that out in the axis label means the panel doesn't depend on the caption to be
    # read correctly.
    ax.set_ylabel("meet own convergence check (%)")
    ax.set_title(f"{letter}   Convergence as data thin" if letter else "Convergence as data thin")
    ax.set_ylim(0, 103)
    if legend:
        ax.legend(fontsize=7.5 * scale, loc="upper left", frameon=False, handlelength=1.4)


def _thinning(sub, full, path, model=None, scale=1.0):
    """sub: the subsample table. full: {dataset: {method: 12-vector}} full-data fits.

    Returns the plotted values as a dict so the manuscript quotes exactly what the
    figure shows, rather than a separately-computed number that can drift from it.

    Single panel (stability only) -- a second "convergence as data thin" panel used to
    sit alongside this one, but three of the four methods trivially meet their own
    convergence check at every level and the fourth's rate is a property of its restart
    policy already discussed at length in the manuscript's speed/convergence section;
    repeating that here added a panel without adding information. _panel_convergence is
    kept (below) in case a reviewer response wants it back, just not called by default.
    """
    levels_z, per_method_z, plotted_z = _thinning_data(sub, full, "z")
    levels_rho, per_method_rho, plotted_rho = _thinning_data(sub, full, "rho")
    if levels_z != levels_rho:
        raise RuntimeError("sensitivity and correlation thinning levels differ")

    # Keep the parameter families distinct: averaging eight z-scores with four
    # correlations imposes an arbitrary 8:4 weighting and hides family-specific drift.
    row_scale = scale * 0.78
    set_style(row_scale)
    fig, axes = plt.subplots(1, 2, figsize=(PRINT_W, PRINT_W * 0.44))
    _panel_drift(axes[0], levels_z, per_method_z, row_scale, "z", letter="A")
    _panel_drift(axes[1], levels_rho, per_method_rho, row_scale, "rho", letter="B",
                 legend=False)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.24, top=0.88, wspace=0.35)
    fig.savefig(path); plt.close(fig)
    print(f"figure -> {path}")

    import json
    with open(str(path).replace(".png", ".json"), "w") as f:
        json.dump({"sensitivity": plotted_z, "correlation": plotted_rho}, f, indent=2)
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
