"""Validation gate 4 for the dynamic-GRT RT model (docs/dynamic_grt_rt_design.md S5):
static-dynamic bridge. Quantifies how dynamic-GRT response probabilities differ from
static-GRT orthant probabilities (src/grt_model.py) over a grid of (zx, zy, rho, boundary).

Analytic shortcut (no path simulation needed for the response marginal): for drift v and
a symmetric absorbing boundary a on a unit-diffusion Wiener process, the classical
two-barrier first-passage result gives

    P(hit +a before -a | drift v) = expit(2 v a)          (sigmoid, standard BM result)

which recovers the static model's deterministic sign(v) rule as a -> infinity and is exact
for every finite a -- no discretisation error, unlike the path simulator. The two channels
are conditionally independent given the trial-level drift draw V ~ N((zx,zy), Sigma), so

    P(response = ++) = E_V[ expit(2 Vx a) expit(2 Vy a) ],   etc.

which is a 2D Gaussian expectation, evaluated by Gauss-Hermite quadrature (exact up to
quadrature error, not Monte Carlo noise). This is cross-checked against the scalar path
simulator (src/data/rt_dynamic_grt.py) at a few spot points before being trusted for the
full grid.

Run from the project root:
    python scripts/check_dynamic_grt_gate4_bridge.py

Writes results/dynamic_grt_gate4_bridge.json.
"""
import json
import os
import sys
import time

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.integrate import quad, dblquad
from scipy.special import expit

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import RESULTS_DIR, Z_MAX, R_MAX
from src.grt_model import forward_probabilities
from src.data.rt_dynamic_grt import (
    DynamicRTParameters,
    simulate_dynamic_grt_trials,
)

OUT_FILE = os.path.join(RESULTS_DIR, "dynamic_grt_gate4_bridge.json")

N_NODES = 64
_HE_X, _HE_W = hermegauss(N_NODES)  # weight exp(-x^2/2), so E[f(Z)] = sum(w*f(x)) / sqrt(2 pi)


def dynamic_response_probabilities(zx, zy, rho, boundary, n_nodes=N_NODES):
    """Exact (quadrature) dynamic-GRT response probabilities for arrays of grid points.

    zx, zy, rho, boundary: broadcastable 1-D arrays, shape (G,).
    Returns array shape (G, 4) in the same [a1b1, a1b2, a2b1, a2b2] order as
    src.grt_model.forward_probabilities.
    """
    zx = np.asarray(zx, dtype=float)
    zy = np.asarray(zy, dtype=float)
    rho = np.asarray(rho, dtype=float)
    boundary = np.asarray(boundary, dtype=float)
    if n_nodes == N_NODES:
        x, w = _HE_X, _HE_W
    else:
        x, w = hermegauss(n_nodes)
    weight2d = np.outer(w, w) / (2.0 * np.pi)          # (n_nodes, n_nodes)

    z1 = x[None, :, None]                                # (1, n_nodes, 1)
    z2 = x[None, None, :]                                # (1, 1, n_nodes)
    vx = zx[:, None, None] + z1                          # (G, n_nodes, 1)
    vy = (zy[:, None, None] + rho[:, None, None] * z1
          + np.sqrt(1.0 - rho[:, None, None] ** 2) * z2)  # (G, n_nodes, n_nodes)
    hx = expit(2.0 * vx * boundary[:, None, None])        # (G, n_nodes, 1)
    hy = expit(2.0 * vy * boundary[:, None, None])        # (G, n_nodes, n_nodes)

    p_pp = np.einsum("ij,gij->g", weight2d, hx * hy)
    p_pn = np.einsum("ij,gij->g", weight2d, hx * (1.0 - hy))
    p_np = np.einsum("ij,gij->g", weight2d, (1.0 - hx) * hy)
    p_nn = np.einsum("ij,gij->g", weight2d, (1.0 - hx) * (1.0 - hy))
    return np.clip(np.stack([p_nn, p_np, p_pn, p_pp], axis=-1), 0.0, 1.0)


def _bvn_density(vx, vy, zx, zy, rho):
    z = ((vx - zx) ** 2 - 2 * rho * (vx - zx) * (vy - zy) + (vy - zy) ** 2) / (1 - rho ** 2)
    return np.exp(-z / 2.0) / (2 * np.pi * np.sqrt(1 - rho ** 2))


def dynamic_response_probabilities_adaptive(zx, zy, rho, boundary, span=14.0, epsabs=1e-10):
    """Reference implementation of the same quantity as dynamic_response_probabilities,
    using SciPy adaptive quadrature instead of fixed-order Gauss-Hermite. Slow (one
    2-D adaptive integral plus two 1-D integrals per point) but free of the numerical
    instability numpy's hermegauss shows above ~300 nodes, so this is the trustworthy
    reference for spot-checking the fast quadrature and for asymptotic (large-boundary)
    checks where the sigmoid transition is too steep for a fixed low-order Hermite rule.
    Scalar inputs only.
    """
    p_x_pos, _ = quad(lambda vx: expit(2 * vx * boundary) *
                       np.exp(-(vx - zx) ** 2 / 2) / np.sqrt(2 * np.pi),
                       zx - span, zx + span, epsabs=epsabs)
    p_y_pos, _ = quad(lambda vy: expit(2 * vy * boundary) *
                       np.exp(-(vy - zy) ** 2 / 2) / np.sqrt(2 * np.pi),
                       zy - span, zy + span, epsabs=epsabs)
    p_pp, _ = dblquad(
        lambda vy, vx: _bvn_density(vx, vy, zx, zy, rho) * expit(2 * vx * boundary) * expit(2 * vy * boundary),
        zx - span, zx + span, zy - span, zy + span, epsabs=epsabs)
    p_pn = p_x_pos - p_pp
    p_np = p_y_pos - p_pp
    p_nn = 1.0 - p_x_pos - p_y_pos + p_pp
    return np.clip(np.array([p_nn, p_np, p_pn, p_pp]), 0.0, 1.0)


def spot_check_against_simulator(n=400_000, seed=4026):
    """Cross-check the analytic quadrature against the (already gate-2/3-validated)
    scalar path simulator at a handful of points spanning the prior, including edge
    cases (near-zero z, extreme rho, small/large boundary)."""
    points = [
        # (zx, zy, rho, boundary)
        (0.0, 0.0, 0.0, 1.0),
        (1.2, -0.8, 0.5, 1.0),
        (2.5, 2.5, -0.7, 0.9),
        (0.3, -0.3, 0.0, 0.6),
        (-1.5, 1.5, 0.8, 1.4),
        (0.0, 0.0, 0.0, 3.0),      # boundary -> large: should approach static 25/25/25/25 split
        (1.0, 1.0, 0.0, 0.3),      # small boundary: heavy diffusive smearing toward 50/50-ish
    ]
    zx = np.array([p[0] for p in points])
    zy = np.array([p[1] for p in points])
    rho = np.array([p[2] for p in points])
    boundary = np.array([p[3] for p in points])
    analytic = dynamic_response_probabilities(zx, zy, rho, boundary)

    rows = []
    max_abs_diff = 0.0
    for i, (zx_i, zy_i, rho_i, b_i) in enumerate(points):
        params = DynamicRTParameters(t0=0.2, boundary=float(b_i), rate=2.0)
        trials = simulate_dynamic_grt_trials(
            zx_i, zy_i, rho_i, n, "parallel_exhaustive", params,
            np.random.default_rng(seed + i))
        complete = ~trials.censored
        counts = np.bincount(trials.response[complete], minlength=4)
        sim_probs = counts / complete.sum()
        diff = np.abs(sim_probs - analytic[i])
        se = np.sqrt(np.clip(analytic[i] * (1 - analytic[i]), 0, None) / complete.sum())
        max_abs_diff = max(max_abs_diff, float(diff.max()))
        rows.append({
            "zx": zx_i, "zy": zy_i, "rho": rho_i, "boundary": b_i,
            "n": int(complete.sum()), "censor_rate": float(trials.censored.mean()),
            "analytic": [float(v) for v in analytic[i]],
            "simulated": [float(v) for v in sim_probs],
            "abs_diff": [float(v) for v in diff],
            "max_diff_in_mc_se": float((diff / np.maximum(se, 1e-12)).max()),
        })
    return rows, max_abs_diff


def build_grid():
    zx_vals = np.array([-2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5])
    zy_vals = np.array([-2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5])
    rho_vals = np.array([-0.6, -0.3, 0.0, 0.3, 0.6])
    boundary_vals = np.array([0.5, 0.75, 1.0, 1.3, 1.5, 2.0, 3.0])
    zx, zy, rho, boundary = np.meshgrid(zx_vals, zy_vals, rho_vals, boundary_vals, indexing="ij")
    return (zx.ravel(), zy.ravel(), rho.ravel(), boundary.ravel(),
            dict(zx=zx_vals, zy=zy_vals, rho=rho_vals, boundary=boundary_vals))


def main():
    t_start = time.time()
    print("Spot-checking analytic quadrature against the scalar path simulator "
          "(n=400,000/point)...")
    spot_rows, max_spot_diff = spot_check_against_simulator()
    for row in spot_rows:
        print(f"  (zx={row['zx']:+.2f}, zy={row['zy']:+.2f}, rho={row['rho']:+.2f}, "
              f"a={row['boundary']:.2f}): max|diff|={max(row['abs_diff']):.4f} "
              f"({row['max_diff_in_mc_se']:.1f} MC SE), censor={row['censor_rate']:.4f}")
    print(f"  max abs diff across all spot points: {max_spot_diff:.4f}\n")

    print("Full grid: 7 zx x 7 zy x 5 rho x 7 boundary = 1715 points "
          "(analytic quadrature, no simulation)...")
    zx, zy, rho, boundary, axes = build_grid()
    dynamic_probs = dynamic_response_probabilities(zx, zy, rho, boundary)
    static_probs = forward_probabilities(zx, zy, rho)  # boundary-independent by construction

    abs_diff = np.abs(dynamic_probs - static_probs)
    tv_distance = 0.5 * abs_diff.sum(axis=1)   # total variation distance per grid point
    max_diff = abs_diff.max(axis=1)

    print(f"  TV distance: median {np.median(tv_distance):.4f}, "
          f"mean {tv_distance.mean():.4f}, max {tv_distance.max():.4f}")
    print(f"  max per-cell abs diff: median {np.median(max_diff):.4f}, max {max_diff.max():.4f}")

    print("\n  TV distance by boundary (should shrink toward 0 as boundary grows -- "
          "recovering the static deterministic-sign limit):")
    by_boundary = []
    for b in axes["boundary"]:
        mask = boundary == b
        by_boundary.append({"boundary": float(b),
                             "tv_median": float(np.median(tv_distance[mask])),
                             "tv_mean": float(tv_distance[mask].mean()),
                             "tv_max": float(tv_distance[mask].max())})
        print(f"    boundary={b:.2f}: TV median={by_boundary[-1]['tv_median']:.4f}, "
              f"mean={by_boundary[-1]['tv_mean']:.4f}, max={by_boundary[-1]['tv_max']:.4f}")

    print("\n  TV distance by |rho| (perceptual correlation vs. the bridge gap):")
    by_rho = []
    for r in axes["rho"]:
        mask = rho == r
        by_rho.append({"rho": float(r),
                        "tv_median": float(np.median(tv_distance[mask])),
                        "tv_mean": float(tv_distance[mask].mean())})
        print(f"    rho={r:+.2f}: TV median={by_rho[-1]['tv_median']:.4f}, "
              f"mean={by_rho[-1]['tv_mean']:.4f}")

    print("\n  TV distance by |z| (distance of the percept from the bound):")
    z_mag = np.maximum(np.abs(zx), np.abs(zy))
    z_bins = [(0.0, 0.5), (0.5, 1.5), (1.5, 2.5), (2.5, 3.0)]
    by_z = []
    for lo, hi in z_bins:
        mask = (z_mag >= lo) & (z_mag < hi if hi < 3.0 else z_mag <= hi)
        if not np.any(mask):
            continue
        by_z.append({"z_range": [lo, hi],
                      "tv_median": float(np.median(tv_distance[mask])),
                      "tv_mean": float(tv_distance[mask].mean()), "n": int(mask.sum())})
        print(f"    max(|zx|,|zy|) in [{lo},{hi}]: TV median={by_z[-1]['tv_median']:.4f}, "
              f"mean={by_z[-1]['tv_mean']:.4f} (n={by_z[-1]['n']})")

    # Cross-check the fast Gauss-Hermite quadrature against the adaptive-quadrature
    # reference over the grid's actual boundary range (0.5-3.0), where the sigmoid
    # transition is gentle enough for 64 Hermite nodes to be reliable.
    print("\nCross-checking Gauss-Hermite (64 nodes) against adaptive-quadrature reference "
          "over the grid's boundary range...")
    rng_check = np.random.default_rng(4027)
    check_idx = rng_check.choice(zx.size, size=12, replace=False)
    gh_vs_adaptive = []
    max_gh_diff = 0.0
    for idx in check_idx:
        ref = dynamic_response_probabilities_adaptive(
            float(zx[idx]), float(zy[idx]), float(rho[idx]), float(boundary[idx]))
        diff = float(np.abs(dynamic_probs[idx] - ref).max())
        max_gh_diff = max(max_gh_diff, diff)
        gh_vs_adaptive.append({
            "zx": float(zx[idx]), "zy": float(zy[idx]), "rho": float(rho[idx]),
            "boundary": float(boundary[idx]), "gh": [float(v) for v in dynamic_probs[idx]],
            "adaptive": [float(v) for v in ref], "max_abs_diff": diff,
        })
    print(f"  max |GH - adaptive| over {len(check_idx)} random grid points: {max_gh_diff:.6f} "
          "(GH is trustworthy on the grid if this is small)")

    # Slow-convergence asymptotic check, using the adaptive reference (robust well past
    # where fixed-order Hermite becomes numerically unstable, see boundary >= ~8 above):
    # sigmoid transition width in drift units is ~1/(2*boundary), so with unit-variance
    # drift, convergence to the static (deterministic-sign) limit is slow in boundary,
    # not a step at any single value -- this checks the trend continues correctly.
    print("\nAsymptotic check (adaptive quadrature, representative point "
          "zx=1.5, zy=-1.0, rho=0.3): TV distance vs static should shrink monotonically "
          "as boundary grows (never needs to hit exactly 0, since scipy adaptive quad "
          "cannot literally take boundary -> infinity):")
    asymptotic_rows = []
    static_ref = forward_probabilities(np.array([1.5]), np.array([-1.0]), np.array([0.3]))[0]
    for b in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
        probs = dynamic_response_probabilities_adaptive(1.5, -1.0, 0.3, b)
        tv = float(0.5 * np.abs(probs - static_ref).sum())
        asymptotic_rows.append({"boundary": b, "tv_distance": tv})
        print(f"    boundary={b:6.1f}: TV distance={tv:.5f}")
    monotone_enough = all(asymptotic_rows[i]["tv_distance"] >= asymptotic_rows[i + 1]["tv_distance"] - 1e-6
                           for i in range(len(asymptotic_rows) - 1))
    print(f"  monotonically non-increasing: {monotone_enough} "
          f"({'confirms the static limit as boundary -> infinity' if monotone_enough else 'UNEXPECTED -- investigate'})")

    out = {
        "method": "analytic 2D Gauss-Hermite quadrature of the two-barrier BM first-passage "
                   "probability expit(2*v*a); no path simulation for the response marginal",
        "n_quadrature_nodes": N_NODES,
        "spot_check_vs_simulator": {
            "n_per_point": 400_000,
            "max_abs_diff_across_points": max_spot_diff,
            "rows": spot_rows,
        },
        "gh_vs_adaptive_crosscheck": {
            "max_abs_diff": max_gh_diff,
            "rows": gh_vs_adaptive,
        },
        "large_boundary_asymptotic_check": {
            "point": {"zx": 1.5, "zy": -1.0, "rho": 0.3},
            "rows": asymptotic_rows,
            "monotonically_non_increasing": monotone_enough,
        },
        "grid": {
            "axes": {k: [float(v) for v in vals] for k, vals in axes.items()},
            "n_points": int(zx.size),
            "tv_distance": {
                "median": float(np.median(tv_distance)),
                "mean": float(tv_distance.mean()),
                "max": float(tv_distance.max()),
            },
            "max_per_cell_abs_diff": {
                "median": float(np.median(max_diff)),
                "max": float(max_diff.max()),
            },
            "by_boundary": by_boundary,
            "by_rho": by_rho,
            "by_z_magnitude": by_z,
        },
        "runtime_seconds": time.time() - t_start,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {OUT_FILE}")
    print(f"done in {out['runtime_seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
