"""
Export a STRATIFIED sample of simulated matrices for the R baselines (grtools / mdsdt).

    python scripts/export_for_r.py --n 600

Why a sample, not the whole set: MLE fitting in R is ~0.1-0.5 s/matrix, so fitting a
million matrices would take weeks and add nothing — the comparison is a statistical claim.
A few hundred to a few thousand matrices gives tight CIs on every metric. We stratify by
TRIAL COUNT (where NPE and MLE differ most), MODEL CLASS, and EFFECT SIZE (the PI frontier),
so the interesting structure is represented rather than averaged away.

Columns:
  row_id                      : join key back to GRIN's predictions
  cm_00..cm_33                : 16 counts, stimulus-major, order a1b1,a1b2,a2b1,a2b2
                                (this IS mdsdt's/grtools' stimulus order = our canonical order)
  trials_0..3                 : per-stimulus totals
  zx_0..3, zy_0..3, rho_0..3  : the 12 identified GROUND-TRUTH parameters
  model_label                 : true GRT class
  trial_bin, rho_bin          : the strata
"""
import argparse, os
import numpy as np
import pandas as pd

from src.config import SIMULATED_DATA_DIR, TRIAL_RANGE, Z_MAX, R_MAX
from src.data.generator import GRTDataGenerator
import src.grt_model as gm


# Fixed bin edges in TRIALS PER STIMULUS (not quantiles): dense where GRIN's low-N
# advantage lives, coarse where the curve flattens and nothing new happens. The x-axis
# then carries real numbers instead of "low/mid/high", and the SAME edges are used for
# every method so a sparse baseline bin shows as a wide CI + low n, not a hidden gap.
#
# MODULE SCOPE, not a local in main(), so consumers import the edges rather than
# restating them. scripts/compare_to_r.py previously carried its own 3-name list
# ("low", "mid", "high") against the 9 bins written here; because it indexed
# range(len(names)) it silently scored only bins 0-2 and labelled 15-20 trials per
# stimulus as "high". One definition, imported, makes that drift unrepresentable.
TPS_EDGES = np.array([5, 10, 15, 20, 30, 50, 75, 100, 200, 500], dtype=float)
N_TRIAL_BINS = len(TPS_EDGES) - 1
TRIAL_BIN_LABELS = [f"{int(TPS_EDGES[i])}-{int(TPS_EDGES[i + 1])}"
                    for i in range(N_TRIAL_BINS)]


def main(n=2000, seed=999):
    per_class = max(n // 12, 10)
    g = GRTDataGenerator(n_per_class=per_class * 3, trial_range=TRIAL_RANGE,
                         z_max=Z_MAX, r_max=R_MAX, seed=seed)
    X, yp, Xt, yc, yl = g.generate_all_model_cms()

    tps = Xt.sum(1) / 4.0                                    # mean trials per stimulus
    trial_bin = np.clip(np.digitize(tps, TPS_EDGES) - 1, 0, len(TPS_EDGES) - 2)
    maxrho = np.abs(yp[:, 8:12]).max(1)
    rho_bin = np.digitize(maxrho, [0.001, 0.3, 0.6])        # PI / weak / mod / strong

    n_tbin = len(TPS_EDGES) - 1
    # Target an EVEN number of matrices per trial bin (that is what tightens the spark),
    # drawn across class x rho within each trial bin. per_tbin is generous because the fine
    # low-N bins are the whole point; raise --n if a real experiment needs even more.
    per_tbin = max(1, n // n_tbin)
    rng = np.random.default_rng(seed)
    keep = []
    for tb in range(n_tbin):
        pool_tb = np.flatnonzero(trial_bin == tb)
        if not len(pool_tb):
            continue
        sub = []
        for cls in np.unique(yl):
            for rb in np.unique(rho_bin):
                idx = pool_tb[(yl[pool_tb] == cls) & (rho_bin[pool_tb] == rb)]
                if len(idx):
                    take = max(1, per_tbin // (12 * 4))
                    sub.extend(rng.choice(idx, min(take, len(idx)), replace=False))
        # top up this trial bin toward per_tbin from whatever else is in it
        if len(sub) < per_tbin:
            rest = np.setdiff1d(pool_tb, sub, assume_unique=False)
            if len(rest):
                sub.extend(rng.choice(rest, min(per_tbin - len(sub), len(rest)),
                                      replace=False))
        keep.extend(sub)
    keep = np.array(sorted(set(keep)))[:n]

    cols = {"row_id": np.arange(len(keep))}
    cols.update({f"cm_{s}{r}": X[keep, s * 4 + r] for s in range(4) for r in range(4)})
    cols.update({f"trials_{s}": Xt[keep, s] for s in range(4)})
    cols.update({nm: yp[keep, j] for j, nm in enumerate(gm.PARAM_NAMES)})
    cols["model_label"] = yl[keep]
    cols["trial_bin"] = trial_bin[keep]
    cols["rho_bin"] = rho_bin[keep]
    cols["tps"] = tps[keep]                                  # exact trials/stimulus, for the curve

    path = os.path.join(SIMULATED_DATA_DIR, "test_set_for_R.csv")
    pd.DataFrame(cols).to_csv(path, index=False)
    print(f"wrote {len(keep)} stratified matrices -> {path}")
    tb_counts = np.bincount(trial_bin[keep], minlength=n_tbin)
    labels = TRIAL_BIN_LABELS
    print("  trials/stim bins:  " + "  ".join(f"{l}:{c}" for l, c in zip(labels, tb_counts)))
    print(f"  rho bins: {np.bincount(rho_bin[keep])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=999)
    a = ap.parse_args()
    main(a.n, a.seed)
