"""
Generate the ground-truth reference for tests/io.test.mjs, which checks
web/assets/js/grt-io.js's aggregate() -- a hand-ported copy of the RT-quantile
step in src/data/rt_lba_generator.py's _simulate_group -- against numpy. Run from
the repo root: `python tests/gen_io_reference.py`.

Independent of the real generator class (no simulation needed here, just the
aggregation arithmetic itself), but implements the IDENTICAL formula documented in
rt_lba_generator.py and mirrored in grt-io.js's aggregate(): per (stimulus,
response) cell, sort RTs ascending and take the nearest-rank index
round_half_to_even(q * (k - 1)); empty cells get all-zero quantiles.
"""
import json
import os

import numpy as np

QUANTILES = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
RT_BOUNDS = (0.1, 10.0)


def aggregate(trials):
    """trials: list of {stimulus, response, rt}. Mirrors grt-io.js's aggregate()."""
    counts = np.zeros((4, 4), dtype=int)
    cell_rts = [[[] for _ in range(4)] for _ in range(4)]
    for t in trials:
        counts[t["stimulus"], t["response"]] += 1
        rt = min(RT_BOUNDS[1], max(RT_BOUNDS[0], t["rt"]))
        cell_rts[t["stimulus"]][t["response"]].append(rt)

    rtq = np.zeros((4, 4, len(QUANTILES)))
    for s in range(4):
        for r in range(4):
            v = sorted(cell_rts[s][r])
            k = len(v)
            if k == 0:
                continue
            for qi, q in enumerate(QUANTILES):
                idx = int(np.clip(np.rint(q * (k - 1)), 0, k - 1))
                rtq[s, r, qi] = v[idx]
    return counts, rtq


rng = np.random.default_rng(1)
cases = []
# A spread of cell-count regimes: sparse, moderate, one empty cell, one singleton.
for n_trials, seed in [(40, 1), (200, 2), (12, 3)]:
    r = np.random.default_rng(seed)
    trials = []
    for _ in range(n_trials):
        s = int(r.integers(0, 4))
        resp = int(r.integers(0, 4))
        # occasionally skip response==stimulus's "wrong" cells to create empty cells
        if resp == (3 - s) and r.random() < 0.7:
            continue
        rt = float(np.clip(r.lognormal(mean=-0.5, sigma=0.4), 0.05, 12.0))
        trials.append({"stimulus": s, "response": resp, "rt": rt})
    counts, rtq = aggregate(trials)
    cases.append({
        "trials": trials,
        "counts": counts.tolist(),
        "rtq": rtq.reshape(-1).tolist(),
    })

with open(os.path.join(os.path.dirname(__file__), "io_reference.json"), "w", encoding="utf-8") as f:
    json.dump(cases, f)
print(f"{len(cases)} aggregation cases -> tests/io_reference.json")
