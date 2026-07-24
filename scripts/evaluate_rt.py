"""Evaluate the trained RT model. Run from the project root:
    python scripts/evaluate_rt.py
Reports: GRT recovery, construct accuracy, 5-way architecture recovery,
dimension-neglect detection, and LBA parameter recovery.
"""
import numpy as np
import torch

from src.config import TRIAL_RANGE, Z_MAX, R_MAX, RT_DRIFT_SD
from src.data.rt_lba_generator import RTLBAGenerator, ARCHITECTURES, LBA_NAMES
from src.inference.predict_rt import load_rt_model, predict_rt, dimension_neglect
from src.inference.model_posterior import construct_labels


def main(n_per_class=120, seed=999):
    model = load_rt_model()
    g = RTLBAGenerator(n_per_class=n_per_class, trial_range=TRIAL_RANGE,
                       z_max=Z_MAX, r_max=R_MAX, drift_sd=RT_DRIFT_SD, seed=seed)
    X, RTQ, Xt, yp, ylba, yc, yl, ya = g.generate(verbose=False)
    p = predict_rt(model, X, RTQ, Xt)
    tc, ta, tb = construct_labels(yl)

    print("=== GRT recovery ===")
    print(f"   z-score MAE {np.abs(p['params'][:, :8] - yp[:, :8]).mean():.3f} | "
          f"rho MAE {np.abs(p['params'][:, 8:] - yp[:, 8:]).mean():.3f}")

    print("\n=== constructs ===")
    pc = p["p_corr"].argmax(1)
    print(f"   correlation structure {np.mean(pc == tc):.2f} | PI {np.mean((pc == 0) == (tc == 0)):.2f}")
    print(f"   separability A {np.mean((p['p_sep_A'] > .5).astype(int) == ta):.2f} | "
          f"B {np.mean((p['p_sep_B'] > .5).astype(int) == tb):.2f}")

    print("\n=== processing architecture (5-way SFT) ===")
    pa = p["p_arch"].argmax(1)
    print(f"   overall {np.mean(pa == ya):.2f}  (chance {1/len(ARCHITECTURES):.2f})")
    for i, a in enumerate(ARCHITECTURES):
        m = ya == i
        if m.sum():
            print(f"     {a:26s} {np.mean(pa[m] == i):.2f}")

    print("\n=== dimension neglect (the training-relevant one) ===")
    st = [i for i, a in enumerate(ARCHITECTURES) if "self_terminating" in a]
    true_neglect = np.isin(ya, st)
    pred_neglect = dimension_neglect(p) > 0.5
    print(f"   detection accuracy {np.mean(pred_neglect == true_neglect):.2f} | "
          f"hit rate {np.mean(pred_neglect[true_neglect]):.2f} | "
          f"false alarm {np.mean(pred_neglect[~true_neglect]):.2f}")

    print("\n=== LBA parameters ===")
    for j, nm in enumerate(LBA_NAMES):
        r = np.corrcoef(ylba[:, j], p["lba"][:, j])[0, 1]
        print(f"   {nm:12s} r={r:+.2f}  MAE {np.abs(ylba[:, j] - p['lba'][:, j]).mean():.3f}")


if __name__ == "__main__":
    main()
