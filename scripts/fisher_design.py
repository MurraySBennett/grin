"""Reference Fisher-information calculation for the design-guidance section.

The calculation isolates one stimulus row of the 2x2 identification matrix.  It
varies symmetric marginal sensitivity ``z`` (so accuracy on each dimension is
Phi(z)), fixes the decision bounds at zero and unit marginal variances, and by
default fixes rho=0.  For each grid point it estimates the derivatives of the
four response probabilities with respect to (zx, zy, rho) by central differences
and inverts the full 3x3 multinomial Fisher-information matrix.  Thus the rho
bound treats both marginal sensitivities as unknown nuisance parameters.

The number of trials merely scales all standard errors by 1/sqrt(n), so the
reported relative curves and operating points do not depend on the illustrative
choice n=100.  Sensitivity precision is expressed as SE(z)/|z|; the rho curve is
expressed relative to its minimum.  The JSON records every setting and grid value.

    python scripts/fisher_design.py

Writes results/validation/fisher_design.json.
"""
import json
import os

import numpy as np
from scipy.stats import norm

import src.grt_model as gm

OUT = os.path.join("results", "validation", "fisher_design.json")
EPS = 1e-5
PROB_FLOOR = 1e-12


def _stimulus_probabilities(theta):
    zx, zy, rho = np.asarray(theta, dtype=float)
    # All four rows are identical here; take one. At rho=0, changing either
    # design-consistent sign only permutes response cells and leaves information intact.
    return gm.forward_probabilities(
        np.repeat(zx, 4), np.repeat(zy, 4), np.repeat(rho, 4)
    )[0]


def fisher_covariance(theta, n_trials=100):
    """Inverse multinomial information for theta=(zx, zy, rho)."""
    theta = np.asarray(theta, dtype=float)
    p = np.clip(_stimulus_probabilities(theta), PROB_FLOOR, 1.0)
    gradient = np.empty((4, 3))
    for j in range(3):
        hi, lo = theta.copy(), theta.copy()
        hi[j] += EPS
        lo[j] -= EPS
        gradient[:, j] = (
            _stimulus_probabilities(hi) - _stimulus_probabilities(lo)
        ) / (2 * EPS)
    information = n_trials * (gradient.T @ (gradient / p[:, None]))
    return np.linalg.inv(information)


def main(rho=0.0, n_trials=100):
    accuracy = np.linspace(0.501, 0.990, 490)
    z = norm.ppf(accuracy)
    se_z, se_rho = [], []
    for value in z:
        cov = fisher_covariance((-value, -value, rho), n_trials=n_trials)
        se_z.append(np.sqrt(cov[0, 0]))
        se_rho.append(np.sqrt(cov[2, 2]))
    se_z = np.asarray(se_z)
    se_rho = np.asarray(se_rho)
    cv_z = se_z / z
    relative_rho = se_rho / se_rho.min()
    within = accuracy[relative_rho <= 1.5]

    out = {
        "meta": {
            "varied": "symmetric |zx|=|zy|=z; per-dimension accuracy=Phi(z)",
            "fixed": {
                "rho": float(rho),
                "decision_bounds": [0.0, 0.0],
                "marginal_variances": [1.0, 1.0],
                "trials_per_stimulus": int(n_trials),
            },
            "estimated_parameters": ["zx", "zy", "rho"],
            "derivative": f"central difference, step={EPS}",
            "note": "Trial count scales SE but does not change either reported operating point.",
        },
        "summary": {
            "rho_information_min_accuracy": float(accuracy[np.argmin(se_rho)]),
            "rho_within_1_5x_min_through_accuracy": float(within.max()),
            "sensitivity_cv_min_accuracy": float(accuracy[np.argmin(cv_z)]),
        },
        "grid": [
            {
                "accuracy": float(a),
                "z": float(zz),
                "se_z": float(sz),
                "cv_z": float(cv),
                "se_rho": float(sr),
                "se_rho_relative_to_min": float(rr),
            }
            for a, zz, sz, cv, sr, rr in zip(
                accuracy, z, se_z, cv_z, se_rho, relative_rho
            )
        ],
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)
        handle.write("\n")
    print(json.dumps(out["summary"], indent=2))
    print(f"wrote {OUT}")
    return out


if __name__ == "__main__":
    main()
