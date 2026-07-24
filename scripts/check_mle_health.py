"""
check_mle_health.py — is the MLE baseline failing, or is maximum likelihood itself
badly behaved for this model?

    python -m scripts.check_mle_health

WHAT THIS ANSWERS
-----------------
The comparison figures show GRIN beating maximum likelihood at every sample size. Two very
different explanations:

  (A) our MLE implementation is bad -- the optimiser gets stuck, so we are beating a straw
      man and the result is worthless;
  (B) maximum likelihood is genuinely poorly behaved for the saturated GRT model on a
      single 4x4 matrix, and the prior is what rescues the estimate.

The columns below distinguish them. The decisive one is NLL SPREAD: the difference in
likelihood between the best and worst of several restarts from different starting points.

  * NLL spread LARGE and MAE improving with restarts -> explanation (A). The optimiser was
    missing better optima; multi-start fixes it and the numbers change.
  * NLL spread ~ZERO but PARAMETER estimates differing a lot between restarts ->
    explanation (B). Every start reaches the same likelihood at a different place, which
    means the likelihood is FLAT along a ridge. There is no better optimum to find, so
    multi-start cannot help, and the fits running off to |z| > Z_MAX and |rho| -> 1 are
    maximum likelihood doing exactly what it is supposed to do on a flat surface.

ESCAPE columns count fits landing outside the prior's support: |z| beyond Z_MAX (a
sensitivity more extreme than any participant we simulate) or |rho| above 0.99 (a
degenerate perceptual distribution collapsed onto a line). Under (B) these are common, and
they are the mechanism behind the MAE gap -- not a bug.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import MODEL_FILE, Z_MAX, R_MAX
from src.api import load_model
from src.data.generator import GRTDataGenerator
from src.inference.predict import predict_point
from src.inference import mle as M
import src.grt_model as gm

TRIALS = (10, 50, 200, 800)
N_MATRICES = 40        # per trial level
N_RESTARTS = 5         # restarts used for the spread diagnostic (20 adds cost, not insight)
RHO_DEGENERATE = 0.99


def main(n=N_MATRICES, restarts=N_RESTARTS):
    model = load_model(MODEL_FILE)
    print(f"prior support:  |z| <= {Z_MAX}   |rho| <= {R_MAX}")
    print(f"{n} matrices per level, {restarts} restarts each\n")
    head = (f"{'trials':>7} {'GRIN':>7} {'MLE 1':>7} {'MLE ms':>7} "
            f"{'|z|>Zmax':>9} {'|r|>.99':>8} {'MAE spread':>11} {'NLL spread':>11} "
            f"{'1-start ms':>11} {'ms-start ms':>12}")
    print(head); print("-" * len(head))

    for T in TRIALS:
        g = GRTDataGenerator(n_per_class=max(n // 12, 1), trial_range=(T, T),
                             balanced_trials=True, z_max=Z_MAX, r_max=R_MAX, seed=11)
        X, yp, Xt, _, _ = g.generate_all_model_cms()
        N = len(X)
        grin = predict_point(model, X, Xt).numpy()

        t0 = time.time()
        single = np.array([M.fit_full(X[i], Xt[i])["params"] for i in range(N)])
        ms_single = 1e3 * (time.time() - t0) / N

        t0 = time.time()
        multi, mae_spread, nll_spread = [], [], []
        for i in range(N):
            base = M._init_from_data("ds", np.asarray(X[i], float).reshape(4, 4),
                                     np.asarray(Xt[i], float))
            rng = np.random.default_rng(1000 + i)
            sols = []
            for k in range(restarts):
                init = base if k == 0 else base + rng.normal(0, 0.75, base.shape)
                f = M.fit_class(X[i], Xt[i], "ds", init=init)
                sols.append((f["nll"], f["params"]))
            nlls = np.array([s[0] for s in sols])
            maes = np.array([np.abs(s[1] - yp[i]).mean() for s in sols])
            multi.append(sols[int(np.argmin(nlls))][1])
            mae_spread.append(maes.max() - maes.min())
            nll_spread.append(nlls.max() - nlls.min())
        ms_multi = 1e3 * (time.time() - t0) / N
        multi = np.array(multi)

        esc_z = 100 * (np.abs(multi[:, :8]).max(1) > Z_MAX).mean()
        esc_r = 100 * (np.abs(multi[:, 8:]).max(1) > RHO_DEGENERATE).mean()
        print(f"{T:7d} {np.abs(grin - yp).mean():7.3f} "
              f"{np.abs(single - yp).mean():7.3f} {np.abs(multi - yp).mean():7.3f} "
              f"{esc_z:8.0f}% {esc_r:7.0f}% {np.median(mae_spread):11.3f} "
              f"{np.median(nll_spread):11.1e} {ms_single:11.1f} {ms_multi:12.1f}")

    print("\nMLE 1  = single warm start (what mdsdt does).")
    print("MLE ms = best of the restarts by likelihood (the careful workflow).")
    print("\nVERDICT: if NLL spread is ~1e-6 while MAE spread is large, the likelihood is")
    print("flat along a ridge. Multi-start cannot help, the escape percentages are real,")
    print("and the prior -- not a better optimiser -- is what makes the estimate behave.")


if __name__ == "__main__":
    main()
