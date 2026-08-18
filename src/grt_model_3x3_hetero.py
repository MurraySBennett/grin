"""Experimental heteroscedastic Gaussian GRT model for a 3x3 design.

The two decision bounds on each response dimension are fixed at 0 and 1 to set
the otherwise arbitrary latent location and scale.  Each of the nine stimulus
distributions then has two means, two positive marginal standard deviations,
and one correlation:

    [mu_x(9), mu_y(9), sd_x(9), sd_y(9), rho(9)]

This is a 45-parameter family inside a 9x9 table with 72 data degrees of freedom.
Decision bounds remain global and axis aligned, so decisional separability is an
assumption of this model, not an estimated construct.
"""

import numpy as np
from scipy.stats import norm

from src.grt_model import bvn_cdf
from src.grt_model_3x3 import A_LEVEL, B_LEVEL, DATA_DF, MODEL_SPECS, MODEL_NAMES


N_LEVELS = 3
N_STIMULI = N_RESPONSES = 9
N_PARAMS = 45
PARAM_NAMES = (
    [f"mux_{i}" for i in range(9)]
    + [f"muy_{i}" for i in range(9)]
    + [f"sdx_{i}" for i in range(9)]
    + [f"sdy_{i}" for i in range(9)]
    + [f"rho_{i}" for i in range(9)]
)


def n_free_params(model_name):
    corr, ps_x, ps_y = MODEL_SPECS[model_name]
    # PS constrains the complete marginal: both its mean and its variance.
    n_x = 2 * (3 if ps_x else 9)
    n_y = 2 * (3 if ps_y else 9)
    n_rho = {"pi": 0, "rho1": 1, "free": 9}[corr]
    return n_x + n_y + n_rho


def forward_probabilities(mu_x, mu_y, sd_x, sd_y, rho):
    """Map canonical parameters to response probabilities of shape (..., 9, 9)."""
    mu_x, mu_y, sd_x, sd_y, rho = map(
        lambda value: np.asarray(value, dtype=float), (mu_x, mu_y, sd_x, sd_y, rho)
    )
    for name, value in (("mu_x", mu_x), ("mu_y", mu_y), ("sd_x", sd_x),
                        ("sd_y", sd_y), ("rho", rho)):
        if value.shape[-1] != 9:
            raise ValueError(f"{name} must end in nine stimulus values")
    if np.any(sd_x <= 0) or np.any(sd_y <= 0):
        raise ValueError("marginal standard deviations must be positive")

    hx = np.stack([-mu_x / sd_x, (1.0 - mu_x) / sd_x], axis=-1)
    hy = np.stack([-mu_y / sd_y, (1.0 - mu_y) / sd_y], axis=-1)
    interior = bvn_cdf(hx[..., :, :, None], hy[..., :, None, :],
                       rho[..., :, None, None])
    grid = np.zeros(mu_x.shape + (4, 4), dtype=float)
    grid[..., 1:3, 1:3] = interior
    grid[..., 1:3, 3] = norm.cdf(hx)
    grid[..., 3, 1:3] = norm.cdf(hy)
    grid[..., 3, 3] = 1.0
    cells = np.diff(np.diff(grid, axis=-2), axis=-1)
    cells = np.clip(cells, 0.0, 1.0)
    cells /= cells.sum(axis=(-2, -1), keepdims=True)
    return cells.reshape(mu_x.shape[:-1] + (9, 9))


def _sample_means(levels, n, rng, z_max):
    values = np.empty((n, len(levels)))
    low = -rng.uniform(0.05, z_max, values.shape)
    middle = rng.uniform(0.05, 0.95, values.shape)
    high = 1.0 + rng.uniform(0.05, z_max, values.shape)
    values[:] = np.where(levels[None] == 0, low,
                         np.where(levels[None] == 1, middle, high))
    return values


def _sample_sd(shape, rng, sd_range):
    lo, hi = sd_range
    if lo <= 0 or hi <= lo:
        raise ValueError("sd_range must contain increasing positive values")
    return np.exp(rng.uniform(np.log(lo), np.log(hi), shape))


def sample_prior(model_name, n, rng, z_max=3.0, r_max=0.9,
                 sd_range=(0.5, 2.0)):
    corr, ps_x, ps_y = MODEL_SPECS[model_name]
    if ps_x:
        base_mu_x = _sample_means(np.arange(3), n, rng, z_max)
        base_sd_x = _sample_sd((n, 3), rng, sd_range)
        mu_x, sd_x = base_mu_x[:, A_LEVEL], base_sd_x[:, A_LEVEL]
    else:
        mu_x = _sample_means(A_LEVEL, n, rng, z_max)
        sd_x = _sample_sd((n, 9), rng, sd_range)
    if ps_y:
        base_mu_y = _sample_means(np.arange(3), n, rng, z_max)
        base_sd_y = _sample_sd((n, 3), rng, sd_range)
        mu_y, sd_y = base_mu_y[:, B_LEVEL], base_sd_y[:, B_LEVEL]
    else:
        mu_y = _sample_means(B_LEVEL, n, rng, z_max)
        sd_y = _sample_sd((n, 9), rng, sd_range)

    if corr == "pi":
        rho = np.zeros((n, 9))
    elif corr == "rho1":
        rho = np.repeat(rng.uniform(-r_max, r_max, (n, 1)), 9, axis=1)
    else:
        rho = rng.uniform(-r_max, r_max, (n, 9))
    return mu_x, mu_y, sd_x, sd_y, rho


def pack(mu_x, mu_y, sd_x, sd_y, rho):
    return np.concatenate([mu_x, mu_y, sd_x, sd_y, rho], axis=-1)


def unpack(params):
    p = np.asarray(params)
    return p[..., :9], p[..., 9:18], p[..., 18:27], p[..., 27:36], p[..., 36:45]


def validate(mu_x, mu_y, sd_x, sd_y, rho, model_name, atol=1e-10):
    problems = []
    corr, ps_x, ps_y = MODEL_SPECS[model_name]
    if np.any(sd_x <= 0) or np.any(sd_y <= 0):
        problems.append("marginal standard deviations must be positive")
    if np.any(np.abs(rho) >= 1):
        problems.append("rho must lie strictly between -1 and 1")
    if corr == "pi" and not np.allclose(rho, 0, atol=atol):
        problems.append("PI requires rho=0")
    if corr == "rho1" and not np.allclose(rho, rho[..., :1], atol=atol):
        problems.append("RHO1 requires a common correlation")
    if ps_x:
        for a in range(3):
            mask = A_LEVEL == a
            if not np.allclose(mu_x[..., mask], mu_x[..., mask][..., :1], atol=atol):
                problems.append(f"PS(A) mean violated at A level {a + 1}")
            if not np.allclose(sd_x[..., mask], sd_x[..., mask][..., :1], atol=atol):
                problems.append(f"PS(A) variance violated at A level {a + 1}")
    if ps_y:
        for b in range(3):
            mask = B_LEVEL == b
            if not np.allclose(mu_y[..., mask], mu_y[..., mask][..., :1], atol=atol):
                problems.append(f"PS(B) mean violated at B level {b + 1}")
            if not np.allclose(sd_y[..., mask], sd_y[..., mask][..., :1], atol=atol):
                problems.append(f"PS(B) variance violated at B level {b + 1}")
    return not problems, problems
