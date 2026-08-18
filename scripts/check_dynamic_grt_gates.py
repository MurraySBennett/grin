"""Validation gates 2-3 for the dynamic-GRT reference simulator
(docs/dynamic_grt_rt_design.md S5): discretisation convergence and prior-predictive
plausibility. Exercises ONLY the scalar reference simulator (src/data/rt_dynamic_grt.py)
-- no vectorised generator, no network, no training. Safe to run on a laptop.

Run from the project root:
    python scripts/check_dynamic_grt_gates.py

Writes results/dynamic_grt_gates.json (full machine-readable output) and prints a
human-readable summary. Exits nonzero if either gate fails its tolerance, so this can
gate a later step without a human re-reading the JSON every time.

Gate 1 (reference correctness) is covered by tests/test_rt_dynamic_grt.py, not here.
Gates 4-8 (static-dynamic bridge, identifiability, architecture evidence,
misspecification, empirical check) need the vectorised generator and/or trained network
and are explicitly out of scope for this script.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import RESULTS_DIR, R_MAX, Z_MAX
from src.data.rt_dynamic_grt import (
    ARCHITECTURES,
    DynamicRTParameters,
    sample_dynamic_rt_parameters,
    simulate_dynamic_grt_trials,
)

OUT_FILE = os.path.join(RESULTS_DIR, "dynamic_grt_gates.json")

# Gate 2's pass/fail tolerances are set inline in convergence_check(), scaled to its
# path-coupled dt_fine/coarsen arguments rather than fixed constants here -- see that
# function's docstring for why (comparing two independent runs at different dt, an
# earlier draft of this script's approach, confounds discretisation error with Monte
# Carlo sampling noise).


def _quantiles(x, qs=(0.1, 0.5, 0.9)):
    return [float(v) for v in np.quantile(x, qs)] if x.size else [None] * len(qs)


def _skew(x):
    if x.size < 3:
        return None
    c = x - x.mean()
    m2 = np.mean(c**2)
    m3 = np.mean(c**3)
    return float(m3 / m2**1.5) if m2 > 0 else None


def _run_first_passage_with_increments(increments, dt, boundary):
    """Same boundary-crossing/interpolation logic as _first_passage_1d, but takes
    pre-generated per-step evidence increments instead of drawing its own noise --
    lets the caller couple two resolutions to the SAME underlying Wiener path."""
    n = increments.shape[1]
    evidence = np.zeros(n)
    response = np.full(n, -1, dtype=np.int8)
    hit_time = np.full(n, np.nan)
    active = np.ones(n, dtype=bool)
    for step in range(increments.shape[0]):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        previous = evidence[idx]
        updated = previous + increments[step, idx]
        evidence[idx] = updated
        crossed = np.abs(updated) >= boundary
        if not np.any(crossed):
            continue
        crossed_idx = idx[crossed]
        previous_crossed = previous[crossed]
        updated_crossed = updated[crossed]
        positive = updated_crossed >= 0
        target = np.where(positive, boundary, -boundary)
        denom = updated_crossed - previous_crossed
        frac = np.divide(target - previous_crossed, denom,
                          out=np.ones_like(target), where=denom != 0)
        frac = np.clip(frac, 0.0, 1.0)
        response[crossed_idx] = positive.astype(np.int8)
        hit_time[crossed_idx] = (step + frac) * dt
        active[crossed_idx] = False
    return response, hit_time, active


# --- Gate 2: discretisation convergence -------------------------------------------
def convergence_check(n=8000, coarsen=2, dt_fine=0.0025, max_internal_time=12.0, seed=2026):
    """Path-coupled convergence check on the channel-level first-passage step
    (architecture only recombines already-generated channel times, so the
    discretisation error lives entirely in _first_passage_1d, not in the GRT/
    architecture wrapper).

    Both resolutions run on the SAME underlying fine-grid Wiener path: the coarse
    step's increment is the exact sum of `coarsen` consecutive fine increments, which
    is the correct Brownian rescaling. This isolates discretisation (time-stepping)
    error from Monte Carlo sampling noise -- comparing two INDEPENDENT runs at
    different dt (which an earlier draft of this script did) confounds the two, and
    at these sample sizes sampling noise alone can produce spurious-looking
    "failures" of order 1-2%.
    """
    combos = [
        ("near-zero drift", 0.0, 1.00, 2.0),
        ("moderate drift", 1.2, 0.90, 3.0),
        ("strong drift, small boundary", 2.6, 0.75, 5.0),
        ("weak drift, large boundary", 0.15, 1.30, 1.0),
    ]
    max_steps_fine = int(np.ceil(max_internal_time / dt_fine))
    rows = []
    all_pass = True
    for i, (label, drift_val, boundary, rate) in enumerate(combos):
        rng = np.random.default_rng(seed + i)
        drift = np.full(n, drift_val)
        fine_noise = rng.standard_normal((max_steps_fine, n)) * np.sqrt(dt_fine)
        fine_increment = drift[None, :] * dt_fine + fine_noise

        resp_fine, time_fine, active_fine = _run_first_passage_with_increments(
            fine_increment, dt_fine, boundary)

        usable_steps = (max_steps_fine // coarsen) * coarsen
        coarse_increment = (fine_increment[:usable_steps]
                             .reshape(-1, coarsen, n).sum(axis=1))
        resp_coarse, time_coarse, active_coarse = _run_first_passage_with_increments(
            coarse_increment, dt_fine * coarsen, boundary)

        both_complete = (~active_fine) & (~active_coarse)
        response_agree = float(np.mean(resp_fine[both_complete] == resp_coarse[both_complete]))
        time_diff = np.abs(time_fine[both_complete] - time_coarse[both_complete])
        row = {
            "condition": {"label": label, "drift": drift_val, "boundary": boundary, "rate": rate,
                           "dt_fine": dt_fine, "dt_coarse": dt_fine * coarsen},
            "n": n,
            "censor_rate_fine": float(active_fine.mean()),
            "censor_rate_coarse": float(active_coarse.mean()),
            "response_agreement": response_agree,
            "hitting_time_diff_median": float(np.median(time_diff)) if time_diff.size else None,
            "hitting_time_diff_p95": float(np.quantile(time_diff, 0.95)) if time_diff.size else None,
        }
        ok = (
            response_agree >= 0.995
            and row["hitting_time_diff_median"] is not None
            and row["hitting_time_diff_median"] <= 2 * dt_fine * coarsen
            and row["hitting_time_diff_p95"] <= 10 * dt_fine * coarsen
        )
        row["pass"] = ok
        all_pass = all_pass and ok
        rows.append(row)
    return all_pass, rows


# --- Gate 3: prior-predictive plausibility -----------------------------------------
def prior_predictive_check(n_draws=150, n_trials=600, seed=3026):
    """Simulate across the full GRT + nuisance prior and summarise RT plausibility.
    This does not pass/fail against a fixed number -- it is inspected by a human, per
    the design doc's gate 3 -- but it does flag the two failure modes that would be
    silent otherwise: near-total censoring, and a leading edge below t0."""
    rng = np.random.default_rng(seed)
    nuisance = sample_dynamic_rt_parameters(rng, n_draws)
    zx = rng.uniform(-Z_MAX, Z_MAX, n_draws)
    zy = rng.uniform(-Z_MAX, Z_MAX, n_draws)
    rho = rng.uniform(-R_MAX, R_MAX, n_draws)

    rows = []
    flags = []
    for i in range(n_draws):
        params = DynamicRTParameters(t0=float(nuisance[i, 0]), boundary=float(nuisance[i, 1]),
                                      rate=float(nuisance[i, 2]))
        expected_response = int(zx[i] > 0) * 2 + int(zy[i] > 0)
        for arch in ARCHITECTURES:
            trials = simulate_dynamic_grt_trials(
                float(zx[i]), float(zy[i]), float(rho[i]), n_trials, arch, params,
                np.random.default_rng(seed + 1 + i))
            complete = ~trials.censored
            rt = trials.rt[complete]
            censor_rate = float(trials.censored.mean())
            if censor_rate > 0.01:
                flags.append(f"draw {i} ({arch}): censor_rate={censor_rate:.3f} > 1%")
            if rt.size and rt.min() < params.t0 - 1e-9:
                flags.append(f"draw {i} ({arch}): min RT {rt.min():.4f} < t0 {params.t0:.4f}")

            is_correct = trials.response[complete] == expected_response
            row = {
                "draw": i, "architecture": arch,
                "zx": float(zx[i]), "zy": float(zy[i]), "rho": float(rho[i]),
                "t0": params.t0, "boundary": params.boundary, "rate": params.rate,
                "censor_rate": censor_rate,
                "rt_q10_q50_q90": _quantiles(rt),
                "rt_skew": _skew(rt),
                "accuracy": float(is_correct.mean()) if rt.size else None,
                "rt_median_correct": float(np.median(rt[is_correct])) if is_correct.any() else None,
                "rt_median_error": float(np.median(rt[~is_correct])) if (~is_correct).any() else None,
            }
            rows.append(row)
    return rows, flags


def _speed_accuracy_summary(rows):
    """Correlate boundary (caution) with accuracy and median RT, pooled across
    architectures -- the classic speed-accuracy signature the design doc's gate 3 asks
    for: higher caution should mean slower and more accurate."""
    boundary = np.array([r["boundary"] for r in rows])
    acc = np.array([r["accuracy"] for r in rows], dtype=float)
    med_rt = np.array([r["rt_q10_q50_q90"][1] for r in rows], dtype=float)
    ok = np.isfinite(acc) & np.isfinite(med_rt)
    return {
        "corr_boundary_accuracy": float(np.corrcoef(boundary[ok], acc[ok])[0, 1]),
        "corr_boundary_median_rt": float(np.corrcoef(boundary[ok], med_rt[ok])[0, 1]),
        "n": int(ok.sum()),
    }


def _correct_vs_error_summary(rows):
    diffs = [r["rt_median_error"] - r["rt_median_correct"] for r in rows
             if r["rt_median_error"] is not None and r["rt_median_correct"] is not None]
    if not diffs:
        return {"n_comparable": 0}
    diffs = np.array(diffs)
    return {
        "n_comparable": int(diffs.size),
        "n_total": len(rows),
        "mean_error_minus_correct_rt": float(diffs.mean()),
        "fraction_error_slower": float((diffs > 0).mean()),
    }


def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gate", choices=("2", "3", "both"), default="both",
                    help="which gate(s) to run (default: both)")
    p.add_argument("--gate2-n", type=int, default=8000,
                    help="trials per gate-2 condition (default 8000; laptop-sized). "
                         "The near-zero-drift condition was borderline (99.45%% vs a "
                         "99.5%% response-agreement threshold) at this default -- rerun "
                         "with e.g. 200000+ to check whether that's Monte Carlo noise.")
    p.add_argument("--gate2-coarsen", type=int, default=2)
    p.add_argument("--gate2-dt-fine", type=float, default=0.0025)
    p.add_argument("--gate2-max-time", type=float, default=12.0)
    p.add_argument("--gate3-draws", type=int, default=150,
                    help="prior draws for gate 3 (default 150)")
    p.add_argument("--gate3-trials", type=int, default=600,
                    help="trials per draw per architecture for gate 3 (default 600)")
    p.add_argument("--out", default=OUT_FILE,
                    help="output JSON path (default results/dynamic_grt_gates.json -- "
                         "pass a different path for a high-n confirmation run that "
                         "shouldn't overwrite the baseline result)")
    return p.parse_args()


def main():
    args = _parse_args()
    t0 = time.time()
    conv_pass, conv_rows = True, []
    if args.gate in ("2", "both"):
        print(f"Gate 2: discretisation convergence (path-coupled fine vs coarse step, "
              f"n={args.gate2_n})...")
        conv_pass, conv_rows = convergence_check(
            n=args.gate2_n, coarsen=args.gate2_coarsen,
            dt_fine=args.gate2_dt_fine, max_internal_time=args.gate2_max_time)
    for row in conv_rows:
        c = row["condition"]
        tag = "PASS" if row["pass"] else "FAIL"
        print(f"  [{tag}] {c['label']} (drift={c['drift']:+.2f}, boundary={c['boundary']:.2f}, "
              f"dt {c['dt_fine']:.5f}->{c['dt_coarse']:.5f}): "
              f"response agreement {row['response_agreement']:.4f}, "
              f"hitting-time diff median {row['hitting_time_diff_median']:.5f}s "
              f"(p95 {row['hitting_time_diff_p95']:.5f}s), "
              f"censor fine/coarse {row['censor_rate_fine']:.4f}/{row['censor_rate_coarse']:.4f}")
    if conv_rows:
        print(f"Gate 2 overall: {'PASS' if conv_pass else 'FAIL'}\n")

    out = {"gate2_convergence": {"pass": conv_pass, "conditions": conv_rows}}

    if args.gate in ("3", "both"):
        print(f"Gate 3: prior-predictive plausibility "
              f"({args.gate3_draws} prior draws x 2 architectures x {args.gate3_trials} trials)...")
        pp_rows, pp_flags = prior_predictive_check(n_draws=args.gate3_draws, n_trials=args.gate3_trials)
        sa = _speed_accuracy_summary(pp_rows)
        ce = _correct_vs_error_summary(pp_rows)
        all_censor = np.array([r["censor_rate"] for r in pp_rows])
        all_rt_med = np.array([r["rt_q10_q50_q90"][1] for r in pp_rows if r["rt_q10_q50_q90"][1] is not None])
        all_skew = np.array([r["rt_skew"] for r in pp_rows if r["rt_skew"] is not None])
        print(f"  censoring: mean {all_censor.mean():.4f}, max {all_censor.max():.4f} "
              f"({(all_censor > 0.01).sum()}/{len(all_censor)} draws over 1%)")
        print(f"  median RT across draws: {np.median(all_rt_med):.3f}s "
              f"(range {all_rt_med.min():.3f}-{all_rt_med.max():.3f}s)")
        print(f"  RT skew across draws: median {np.median(all_skew):.2f} "
              f"({(all_skew > 0).mean() * 100:.0f}% right-skewed)")
        print(f"  speed-accuracy: corr(boundary, accuracy) = {sa['corr_boundary_accuracy']:+.3f}, "
              f"corr(boundary, median RT) = {sa['corr_boundary_median_rt']:+.3f}  "
              "(both should be positive: more caution -> slower AND more accurate)")
        print(f"  error vs correct RT: {ce['fraction_error_slower']*100:.0f}% of comparable draws "
              f"have slower median error RT (n={ce['n_comparable']}/{ce['n_total']})")
        if pp_flags:
            print(f"  FLAGS ({len(pp_flags)}):")
            for f in pp_flags[:20]:
                print(f"    - {f}")
            if len(pp_flags) > 20:
                print(f"    ... and {len(pp_flags) - 20} more")
        else:
            print("  no flags (no censoring >1%, no RT below t0)")

        out["gate3_prior_predictive"] = {
            "n_draws": args.gate3_draws, "n_trials_per_draw": args.gate3_trials,
            "censoring": {"mean": float(all_censor.mean()), "max": float(all_censor.max())},
            "rt_median_across_draws": {"median": float(np.median(all_rt_med)),
                                        "min": float(all_rt_med.min()), "max": float(all_rt_med.max())},
            "rt_skew_across_draws": {"median": float(np.median(all_skew)),
                                      "fraction_right_skewed": float((all_skew > 0).mean())},
            "speed_accuracy": sa,
            "correct_vs_error_rt": ce,
            "flags": pp_flags,
            "rows": pp_rows,
        }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {args.out}")
    print(f"done in {time.time() - t0:.1f}s")
    return 0 if conv_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
