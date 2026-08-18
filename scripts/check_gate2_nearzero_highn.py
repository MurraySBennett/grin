"""High-n confirmation of gate 2's near-zero-drift edge case
(docs/dynamic_grt_rt_design.md S5, gate 2).

`scripts/check_dynamic_grt_gates.py` found the near-zero-drift condition
(drift=0.0, boundary=1.0, rate=2.0) at n=8000 borderline: 99.45% path-coupled
fine/coarse response agreement vs a 99.5% threshold. The Monte Carlo standard
error at n=8000 is large enough (~0.08 pp) that 99.45% is not distinguishable
from the threshold. This script reruns ONLY that condition at much larger n,
in memory-bounded batches on the same path-coupled fine/coarse design as the
original gate, to see whether the estimate holds near 99.45% or moves as
sampling noise shrinks.

Reuses `_run_first_passage_with_increments` from check_dynamic_grt_gates.py
so the boundary-crossing/interpolation logic is identical to the original gate
-- this script only changes n and adds batching.

Run from the project root:
    python scripts/check_gate2_nearzero_highn.py [--n TOTAL_N] [--batch-size BATCH]

Writes results/dynamic_grt_gate2_nearzero_highn.json.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import RESULTS_DIR
from scripts.check_dynamic_grt_gates import _run_first_passage_with_increments

OUT_FILE = os.path.join(RESULTS_DIR, "dynamic_grt_gate2_nearzero_highn.json")

# Same condition as the "near-zero drift" row in check_dynamic_grt_gates.convergence_check.
LABEL = "near-zero drift"
DRIFT = 0.0
BOUNDARY = 1.0
RATE = 2.0
DT_FINE = 0.0025
COARSEN = 2
MAX_INTERNAL_TIME = 12.0
THRESHOLD = 0.995


def run_batch(n, dt_fine, coarsen, max_internal_time, rng):
    max_steps_fine = int(np.ceil(max_internal_time / dt_fine))
    drift = np.full(n, DRIFT)
    fine_noise = rng.standard_normal((max_steps_fine, n)) * np.sqrt(dt_fine)
    fine_increment = drift[None, :] * dt_fine + fine_noise
    del fine_noise

    resp_fine, time_fine, active_fine = _run_first_passage_with_increments(
        fine_increment, dt_fine, BOUNDARY)

    usable_steps = (max_steps_fine // coarsen) * coarsen
    coarse_increment = (fine_increment[:usable_steps]
                         .reshape(-1, coarsen, n).sum(axis=1))
    del fine_increment
    resp_coarse, time_coarse, active_coarse = _run_first_passage_with_increments(
        coarse_increment, dt_fine * coarsen, BOUNDARY)
    del coarse_increment

    both_complete = (~active_fine) & (~active_coarse)
    agree = resp_fine[both_complete] == resp_coarse[both_complete]
    time_diff = np.abs(time_fine[both_complete] - time_coarse[both_complete])
    return {
        "n": n,
        "n_both_complete": int(both_complete.sum()),
        "n_agree": int(agree.sum()),
        "censored_fine": int(active_fine.sum()),
        "censored_coarse": int(active_coarse.sum()),
        "time_diff": time_diff,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500_000, help="total trials")
    ap.add_argument("--batch-size", type=int, default=25_000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    n_batches = int(np.ceil(args.n / args.batch_size))
    print(f"Gate 2 near-zero-drift confirmation: total n={args.n:,} "
          f"in {n_batches} batches of {args.batch_size:,} "
          f"(drift={DRIFT}, boundary={BOUNDARY}, rate={RATE}, "
          f"dt {DT_FINE}->{DT_FINE * COARSEN})")

    t_start = time.time()
    total_both_complete = 0
    total_agree = 0
    total_censored_fine = 0
    total_censored_coarse = 0
    total_n = 0
    all_time_diffs = []

    for b in range(n_batches):
        batch_n = min(args.batch_size, args.n - total_n)
        rng = np.random.default_rng(args.seed + b)
        t0 = time.time()
        result = run_batch(batch_n, DT_FINE, COARSEN, MAX_INTERNAL_TIME, rng)
        total_n += batch_n
        total_both_complete += result["n_both_complete"]
        total_agree += result["n_agree"]
        total_censored_fine += result["censored_fine"]
        total_censored_coarse += result["censored_coarse"]
        all_time_diffs.append(result["time_diff"])
        running_agreement = total_agree / total_both_complete
        se = np.sqrt(running_agreement * (1 - running_agreement) / total_both_complete)
        print(f"  batch {b + 1}/{n_batches}: n={total_n:,} cumulative, "
              f"agreement={running_agreement:.5f} +/- {1.96 * se:.5f} (95% CI), "
              f"({time.time() - t0:.1f}s)")

    time_diff = np.concatenate(all_time_diffs)
    response_agreement = total_agree / total_both_complete
    se = np.sqrt(response_agreement * (1 - response_agreement) / total_both_complete)
    ci_lo, ci_hi = response_agreement - 1.96 * se, response_agreement + 1.96 * se
    hitting_time_diff_median = float(np.median(time_diff)) if time_diff.size else None
    hitting_time_diff_p95 = float(np.quantile(time_diff, 0.95)) if time_diff.size else None
    passes_threshold = bool(response_agreement >= THRESHOLD)
    ci_excludes_threshold_below = bool(ci_hi < THRESHOLD)
    ci_excludes_threshold_above = bool(ci_lo >= THRESHOLD)

    out = {
        "condition": {"label": LABEL, "drift": DRIFT, "boundary": BOUNDARY, "rate": RATE,
                       "dt_fine": DT_FINE, "dt_coarse": DT_FINE * COARSEN},
        "total_n_requested": args.n,
        "batch_size": args.batch_size,
        "n_batches": n_batches,
        "n_both_complete": total_both_complete,
        "censor_rate_fine": total_censored_fine / total_n,
        "censor_rate_coarse": total_censored_coarse / total_n,
        "response_agreement": response_agreement,
        "response_agreement_se": se,
        "response_agreement_95ci": [ci_lo, ci_hi],
        "threshold": THRESHOLD,
        "passes_threshold_point_estimate": passes_threshold,
        "ci_confirms_below_threshold": ci_excludes_threshold_below,
        "ci_confirms_at_or_above_threshold": ci_excludes_threshold_above,
        "hitting_time_diff_median": hitting_time_diff_median,
        "hitting_time_diff_p95": hitting_time_diff_p95,
        "hitting_time_diff_criteria": {
            "median_limit": 2 * DT_FINE * COARSEN,
            "p95_limit": 10 * DT_FINE * COARSEN,
            "median_pass": bool(hitting_time_diff_median is not None and hitting_time_diff_median <= 2 * DT_FINE * COARSEN),
            "p95_pass": bool(hitting_time_diff_p95 is not None and hitting_time_diff_p95 <= 10 * DT_FINE * COARSEN),
        },
        "runtime_seconds": time.time() - t_start,
        "original_n8000_result": {
            "response_agreement": 0.9945,
            "hitting_time_diff_median": 0.0006445375074992238,
            "hitting_time_diff_p95": 0.03386727661643085,
        },
    }

    print(f"\nFinal: n_both_complete={total_both_complete:,}, "
          f"response_agreement={response_agreement:.5f} "
          f"(95% CI [{ci_lo:.5f}, {ci_hi:.5f}])")
    print(f"  threshold: {THRESHOLD}")
    if ci_excludes_threshold_below:
        print("  -> CI is entirely BELOW threshold: near-zero-drift genuinely fails at 99.5%, "
              "not a Monte Carlo artifact.")
    elif ci_excludes_threshold_above:
        print("  -> CI is entirely AT/ABOVE threshold: the n=8000 shortfall was Monte Carlo "
              "noise; condition passes at high n.")
    else:
        print("  -> CI still straddles the threshold; even this n does not resolve it "
              "(unlikely at n=500k+, but possible).")
    print(f"  hitting-time diff median {hitting_time_diff_median:.6f}s "
          f"(limit {2 * DT_FINE * COARSEN:.5f}s), "
          f"p95 {hitting_time_diff_p95:.6f}s (limit {10 * DT_FINE * COARSEN:.5f}s)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {OUT_FILE}")
    print(f"done in {out['runtime_seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
