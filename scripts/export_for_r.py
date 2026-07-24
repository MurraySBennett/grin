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


def main(n=600, seed=999):
    per_class = max(n // 12, 10)
    g = GRTDataGenerator(n_per_class=per_class * 3, trial_range=TRIAL_RANGE,
                         z_max=Z_MAX, r_max=R_MAX, seed=seed)
    X, yp, Xt, yc, yl = g.generate_all_model_cms()

    total = Xt.sum(1)
    trial_bin = np.digitize(total, np.quantile(total, [0.33, 0.66]))     # low / mid / high
    maxrho = np.abs(yp[:, 8:12]).max(1)
    rho_bin = np.digitize(maxrho, [0.001, 0.3, 0.6])                     # PI / weak / mod / strong

    # stratified draw: even coverage of (class x trial_bin x rho_bin)
    rng = np.random.default_rng(seed)
    keep = []
    for cls in np.unique(yl):
        for tb in np.unique(trial_bin):
            for rb in np.unique(rho_bin):
                idx = np.flatnonzero((yl == cls) & (trial_bin == tb) & (rho_bin == rb))
                if len(idx):
                    take = max(1, n // (12 * 3 * 4))
                    keep.extend(rng.choice(idx, min(take, len(idx)), replace=False))
    keep = np.array(sorted(set(keep)))[:n]

    cols = {"row_id": np.arange(len(keep))}
    cols.update({f"cm_{s}{r}": X[keep, s * 4 + r] for s in range(4) for r in range(4)})
    cols.update({f"trials_{s}": Xt[keep, s] for s in range(4)})
    cols.update({nm: yp[keep, j] for j, nm in enumerate(gm.PARAM_NAMES)})
    cols["model_label"] = yl[keep]
    cols["trial_bin"] = trial_bin[keep]
    cols["rho_bin"] = rho_bin[keep]

    path = os.path.join(SIMULATED_DATA_DIR, "test_set_for_R.csv")
    pd.DataFrame(cols).to_csv(path, index=False)
    print(f"wrote {len(keep)} stratified matrices -> {path}")
    print(f"  trial bins: {np.bincount(trial_bin[keep])}  rho bins: {np.bincount(rho_bin[keep])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=999)
    a = ap.parse_args()
    main(a.n, a.seed)
    