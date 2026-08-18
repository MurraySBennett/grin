"""Generate an experimental 3x3 GRIN training corpus.

This does not overwrite the production 2x2 corpus.  Example smoke run:
    python scripts/generate_data_3x3.py --n-per-class 100 --output data/simulated/grt_3x3_smoke.npz
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.generator_3x3 import GRT3x3DataGenerator, GRT3x3HeteroDataGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-class", type=int, default=100_000)
    parser.add_argument("--trial-min", type=int, default=1)
    parser.add_argument("--trial-max", type=int, default=1000)
    parser.add_argument("--imbalance", type=float, default=0.35)
    parser.add_argument("--variance", choices=("unit", "free"), default="unit")
    parser.add_argument("--sd-min", type=float, default=0.5)
    parser.add_argument("--sd-max", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output", default="data/simulated/grt_3x3_dataset.npz")
    args = parser.parse_args()

    generator_class = GRT3x3DataGenerator if args.variance == "unit" else GRT3x3HeteroDataGenerator
    extra = {} if args.variance == "unit" else {"sd_range": (args.sd_min, args.sd_max)}
    generator = generator_class(
        n_per_class=args.n_per_class,
        trial_range=(args.trial_min, args.trial_max),
        imbalance=args.imbalance,
        seed=args.seed,
        **extra,
    )
    X, y_params, X_trials, y_model_cls, y_cls_label = generator.generate_all_model_cms()
    parent = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(parent, exist_ok=True)
    np.savez_compressed(
        args.output,
        X=X,
        X_trials=X_trials,
        y_params=y_params,
        y_model_cls=y_model_cls,
        y_cls_label=y_cls_label,
        design="3x3",
        variance_model=args.variance,
        seed=args.seed,
        trial_range=np.array([args.trial_min, args.trial_max]),
        imbalance=args.imbalance,
    )
    print(f"wrote {len(X):,} {args.variance}-variance matrices to {args.output}")


if __name__ == "__main__":
    main()
