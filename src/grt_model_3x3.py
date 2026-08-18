"""Experimental identified Gaussian GRT model for a 3x3 identification design.

This module is deliberately separate from :mod:`src.grt_model`: the released 2x2
model and its checkpoint contract remain unchanged.

There are two perceptual dimensions (A and B), each with three stimulus and
response levels.  Under decisional separability, the response regions are formed
by two axis-aligned bounds on each dimension.  We identify location and scale by
fixing the first bound at zero and every stimulus covariance marginal variance at
one.  The second bound is a positive, estimated spacing.  The canonical vector is

    [mu_x(9), mu_y(9), rho(9), bound_x, bound_y]

for 29 parameters.  A 9x9 confusion matrix has 9 * (9 - 1) = 72 data degrees of
freedom, so this Gaussian family is not saturated.
"""

import numpy as np
from scipy.stats import norm

from src.grt_model import bvn_cdf


N_LEVELS = 3
N_STIMULI = N_RESPONSES = N_LEVELS ** 2
DATA_DF = N_STIMULI * (N_RESPONSES - 1)
N_PARAMS = 29

STIMULUS_ORDER = [f"A{a + 1}B{b + 1}" for a in range(3) for b in range(3)]
RESPONSE_ORDER = [f"a{a + 1}b{b + 1}" for a in range(3) for b in range(3)]
A_LEVEL = np.repeat(np.arange(3), 3)
B_LEVEL = np.tile(np.arange(3), 3)

PARAM_NAMES = (
    [f"mux_{i}" for i in range(9)]
    + [f"muy_{i}" for i in range(9)]
    + [f"rho_{i}" for i in range(9)]
    + ["bound_x", "bound_y"]
)

# The structural vocabulary matches the 2x2 model.  PS(A) means that the A
# location depends only on A's physical level, and likewise for PS(B).
MODEL_SPECS = {
    "pi_ps_ds": ("pi", True, True),
    "pi_psa_ds": ("pi", True, False),
    "pi_psb_ds": ("pi", False, True),
    "rho1_ps_ds": ("rho1", True, True),
    "rho1_psa_ds": ("rho1", True, False),
    "rho1_psb_ds": ("rho1", False, True),
    "pi_ds": ("pi", False, False),
    "ps_ds": ("free", True, True),
    "rho1_ds": ("rho1", False, False),
    "psa_ds": ("free", True, False),
    "psb_ds": ("free", False, True),
    "ds": ("free", False, False),
}
MODEL_NAMES = list(MODEL_SPECS)


def n_free_params(model_name):
    corr, ps_x, ps_y = MODEL_SPECS[model_name]
    n_x = 3 if ps_x else 9
    n_y = 3 if ps_y else 9
    n_rho = {"pi": 0, "rho1": 1, "free": 9}[corr]
    return n_x + n_y + n_rho + 2  # two positive boundary spacings


def _cdf_grid(mu_x, mu_y, rho, bound_x, bound_y):
    """CDF at the four response-region edges on each dimension."""
    hx = np.stack([-mu_x, bound_x[..., None] - mu_x], axis=-1)
    hy = np.stack([-mu_y, bound_y[..., None] - mu_y], axis=-1)
    interior = bvn_cdf(hx[..., :, :, None], hy[..., :, None, :],
                       rho[..., :, None, None])

    grid = np.zeros(mu_x.shape + (4, 4), dtype=float)
    grid[..., 1:3, 1:3] = interior
    grid[..., 1:3, 3] = norm.cdf(hx)
    grid[..., 3, 1:3] = norm.cdf(hy)
    grid[..., 3, 3] = 1.0
    return grid


def forward_probabilities(mu_x, mu_y, rho, bound_x, bound_y):
    """Map canonical parameters to probabilities of shape ``(..., 9, 9)``."""
    mu_x = np.asarray(mu_x, dtype=float)
    mu_y = np.asarray(mu_y, dtype=float)
    rho = np.asarray(rho, dtype=float)
    bound_x = np.asarray(bound_x, dtype=float)
    bound_y = np.asarray(bound_y, dtype=float)
    if mu_x.shape[-1] != 9 or mu_y.shape[-1] != 9 or rho.shape[-1] != 9:
        raise ValueError("mu_x, mu_y, and rho must end in nine stimulus values")
    if np.any(bound_x <= 0) or np.any(bound_y <= 0):
        raise ValueError("second decision bounds must be positive")

    grid = _cdf_grid(mu_x, mu_y, rho, bound_x, bound_y)
    cells = np.diff(np.diff(grid, axis=-2), axis=-1)  # (..., 9, 3, 3)
    cells = np.clip(cells, 0.0, 1.0)
    cells /= cells.sum(axis=(-2, -1), keepdims=True)
    return cells.reshape(mu_x.shape[:-1] + (9, 9))


def _ordered_means(levels, bound, n, rng, z_max):
    means = np.empty((n, len(levels)), dtype=float)
    lo = rng.uniform(0.05, z_max, means.shape)
    mid = rng.uniform(0.05, 0.95, means.shape) * bound[:, None]
    hi = bound[:, None] + rng.uniform(0.05, z_max, means.shape)
    means[:] = np.where(levels[None] == 0, -lo,
                        np.where(levels[None] == 1, mid, hi))
    return means


def sample_prior(model_name, n, rng, z_max=3.0, r_max=0.9,
                 bound_range=(0.75, 3.0)):
    """Sample ordered 3x3 representations satisfying a structural model class."""
    corr, ps_x, ps_y = MODEL_SPECS[model_name]
    bx = rng.uniform(*bound_range, n)
    by = rng.uniform(*bound_range, n)

    if ps_x:
        base_x = _ordered_means(np.arange(3), bx, n, rng, z_max)
        mu_x = base_x[:, A_LEVEL]
    else:
        mu_x = _ordered_means(A_LEVEL, bx, n, rng, z_max)
    if ps_y:
        base_y = _ordered_means(np.arange(3), by, n, rng, z_max)
        mu_y = base_y[:, B_LEVEL]
    else:
        mu_y = _ordered_means(B_LEVEL, by, n, rng, z_max)

    if corr == "pi":
        rho = np.zeros((n, 9))
    elif corr == "rho1":
        rho = np.repeat(rng.uniform(-r_max, r_max, (n, 1)), 9, axis=1)
    else:
        rho = rng.uniform(-r_max, r_max, (n, 9))
    return mu_x, mu_y, rho, bx, by


def pack(mu_x, mu_y, rho, bound_x, bound_y):
    bx = np.asarray(bound_x)[..., None]
    by = np.asarray(bound_y)[..., None]
    return np.concatenate([mu_x, mu_y, rho, bx, by], axis=-1)


def unpack(params):
    p = np.asarray(params)
    return p[..., :9], p[..., 9:18], p[..., 18:27], p[..., 27], p[..., 28]


def validate(mu_x, mu_y, rho, bound_x, bound_y, model_name, atol=1e-10):
    problems = []
    corr, ps_x, ps_y = MODEL_SPECS[model_name]
    if np.any(np.asarray(bound_x) <= 0) or np.any(np.asarray(bound_y) <= 0):
        problems.append("decision-bound spacings must be positive")
    if np.any(np.abs(rho) >= 1):
        problems.append("rho must lie strictly between -1 and 1")
    if corr == "pi" and not np.allclose(rho, 0, atol=atol):
        problems.append("PI requires rho=0")
    if corr == "rho1" and not np.allclose(rho, rho[..., :1], atol=atol):
        problems.append("RHO1 requires a common correlation")
    if ps_x:
        for a in range(3):
            if not np.allclose(mu_x[..., A_LEVEL == a], mu_x[..., A_LEVEL == a][..., :1], atol=atol):
                problems.append(f"PS(A) violated at A level {a + 1}")
    if ps_y:
        for b in range(3):
            if not np.allclose(mu_y[..., B_LEVEL == b], mu_y[..., B_LEVEL == b][..., :1], atol=atol):
                problems.append(f"PS(B) violated at B level {b + 1}")
    return not problems, problems
