"""Generate GRIN training data.

    python scripts/generate_data.py --report          # counts-only dataset (the core pipeline)
    python scripts/generate_data.py --rt              # RT dataset (architecture + LBA)
    python scripts/generate_data.py --report --rt     # both

Counts-only is what most users need (and what legacy datasets support). The --rt dataset
additionally simulates trial-level response times, enabling processing-architecture and
LBA-parameter inference. It is slower to generate because it simulates every trial.
"""
import argparse
import numpy as np

from src.config import (DATASET_FILE, COVERAGE_FIGURE, N_PER_CLASS, TRIAL_RANGE,
                        TRIAL_IMBALANCE, Z_MAX, R_MAX, DATA_SEED, RT_DATASET_FILE, RT_DRIFT_SD)
from src.data.generator import GRTDataGenerator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-class", type=int, default=N_PER_CLASS)
    p.add_argument("--seed", type=int, default=DATA_SEED)
    p.add_argument("--report", action="store_true", help="write the coverage figure")
    p.add_argument("--rt", action="store_true", help="also generate the RT dataset")
    p.add_argument("--rt-only", action="store_true", help="generate ONLY the RT dataset")
    a = p.parse_args()

    if not a.rt_only:
        gen = GRTDataGenerator(n_per_class=a.n_per_class, trial_range=TRIAL_RANGE,
                               z_max=Z_MAX, r_max=R_MAX, seed=a.seed,
                               imbalance=TRIAL_IMBALANCE)
        X, y_params, X_trials, y_cls, y_label = gen.generate_all_model_cms()
        np.savez(DATASET_FILE, X=X, X_trials=X_trials, y_params=y_params,
                 y_model_cls=y_cls, y_cls_label=y_label)
        print(f"[counts] {X.shape[0]} matrices -> {DATASET_FILE}")
        if a.report:
            gen.coverage_report(X, X_trials, y_params, y_cls, y_label, figure_path=COVERAGE_FIGURE)
            print(f"[counts] coverage figure -> {COVERAGE_FIGURE}")

    if a.rt or a.rt_only:
        # Identical settings to the counts pipeline — same prior, same trial range,
        # same imbalance, same N.
        from src.data.rt_lba_generator import RTLBAGenerator
        g = RTLBAGenerator(n_per_class=a.n_per_class, trial_range=TRIAL_RANGE,
                           z_max=Z_MAX, r_max=R_MAX, drift_sd=RT_DRIFT_SD, seed=a.seed,
                           imbalance=TRIAL_IMBALANCE)
        print(f"[rt] simulating trial-level RTs ({a.n_per_class}/class x 12)...")
        X, RTQ, Xt, yp, ylba, ycls, ylab, yarch = g.generate()
        np.savez(RT_DATASET_FILE, X=X, RTQ=RTQ, X_trials=Xt, y_params=yp, y_lba=ylba,
                 y_model_cls=ycls, y_cls_label=ylab, y_arch=yarch)
        print(f"[rt] {X.shape[0]} matrices -> {RT_DATASET_FILE}")


if __name__ == "__main__":
    main()
