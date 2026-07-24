import json, os, sys, numpy as np

# Run from the repo root: `python tests/gen_fit_reference.py`. This imports the
# REAL project package (not a local copy), so it stays honest as grt_model.py /
# mle.py evolve — if their public interface changes, this script breaks loudly
# instead of silently comparing against a stale copy.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import grt_model as gm
from src.inference.mle import fit_class, fit_and_select

rng = np.random.default_rng(3)
cases = []
# simulate from a spread of TRUE classes and trial counts, incl. sparse data
for true_model in ["pi_ps_ds", "rho1_ps_ds", "psa_ds", "ds", "pi_ds", "rho1_psb_ds"]:
    for n_tr in (25, 100, 400):
        zx, zy, rho = gm.sample_prior(true_model, 1, rng)
        P = gm.forward_probabilities(zx, zy, rho)[0]
        counts = np.stack([rng.multinomial(n_tr, P[s]) for s in range(4)])
        trials = counts.sum(1)
        fits = {m: fit_class(counts, trials, m) for m in gm.MODEL_NAMES}
        best, _ = fit_and_select(counts, trials, "bic")
        cases.append({
            "true_model": true_model, "n_trials": int(n_tr),
            "counts": counts.tolist(), "trials": trials.tolist(),
            "true_params": list(map(float, gm.pack(zx, zy, rho)[0])),
            "fits": {m: {"loglik": float(f["loglik"]), "aic": float(f["aic"]),
                         "bic": float(f["bic"]), "k": int(f["k"]),
                         "params": list(map(float, f["params"]))}
                     for m, f in fits.items()},
            "best_bic": best["model"],
        })
json.dump(cases, open("tests/fit_reference.json", "w"))
print(f"{len(cases)} cases x 12 classes = {12*len(cases)} MLE fits")
