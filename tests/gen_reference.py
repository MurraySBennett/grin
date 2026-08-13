"""
Generate the ground-truth reference for tests/core.test.mjs, which checks
web/assets/js/grt-core.js -- a hand-ported copy of src/grt_model.py -- against
the real Python package. Run from the repo root: `python tests/gen_reference.py`.

This imports the REAL project package (not a local copy), so it stays honest as
grt_model.py evolves -- if its public interface changes, this script breaks loudly
instead of silently comparing against a stale copy (same principle as
get_fit_reference.py).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import grt_model as gm  # noqa: E402  (needs the path insert above)


def _loglik(counts, probs):
    return float((np.asarray(counts) * np.log(np.clip(probs, 1e-300, None))).sum())


rng = np.random.default_rng(0)
cases = []

# One random draw per model class, at a few trial-count scales (drives forward
# model + logLik + validate together on realistic, class-constrained params).
for name in gm.MODEL_NAMES:
    zx, zy, rho = gm.sample_prior(name, 3, rng)
    probs = gm.forward_probabilities(zx, zy, rho)             # (3,4,4)
    params = gm.pack(zx, zy, rho)                              # (3,12)
    ok, _ = gm.validate(zx, zy, rho, name)
    for i in range(3):
        for n_tr in (30, 300):
            counts = np.round(probs[i] * n_tr / probs[i].sum(1, keepdims=True)).astype(int)
            cases.append({
                "model": name,
                "params": params[i].tolist(),
                "probs": probs[i].tolist(),
                "counts": counts.tolist(),
                "loglik": _loglik(counts, probs[i]),
                "valid": bool(ok),
            })

# Hand-picked edge cases: zero correlation, near-boundary correlation, zero
# sensitivity -- the corners the random draws might miss.
edge_params = [
    {"zx": [0, 0, 0, 0], "zy": [0, 0, 0, 0], "rho": [0, 0, 0, 0]},
    {"zx": [-2.5, -2.5, 2.5, 2.5], "zy": [-1.0, 1.0, -1.0, 1.0], "rho": [0.85, -0.85, 0.85, -0.85]},
    {"zx": [-0.1, -0.1, 0.1, 0.1], "zy": [-0.1, 0.1, -0.1, 0.1], "rho": [0.0, 0.0, 0.0, 0.0]},
]
for ep in edge_params:
    zx = np.array(ep["zx"])[None]; zy = np.array(ep["zy"])[None]; rho = np.array(ep["rho"])[None]
    probs = gm.forward_probabilities(zx, zy, rho)
    params = gm.pack(zx, zy, rho)
    ok, _ = gm.validate(zx, zy, rho, "ds")
    counts = np.round(probs[0] * 300 / probs[0].sum(1, keepdims=True)).astype(int)
    cases.append({
        "model": "ds", "params": params[0].tolist(), "probs": probs[0].tolist(),
        "counts": counts.tolist(), "loglik": _loglik(counts, probs[0]), "valid": bool(ok),
    })

with open(os.path.join(os.path.dirname(__file__), "reference.json"), "w", encoding="utf-8") as f:
    json.dump(cases, f)
print(f"{len(cases)} forward-model cases -> tests/reference.json")
