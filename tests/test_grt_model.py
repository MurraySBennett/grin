"""Unit tests for src/grt_model.py -- the single source of truth for GRIN's GRT
parameterization (also the thing web/assets/js/grt-core.js is a hand-port of;
see tests/core.test.mjs for that side)."""
import os
import sys

import numpy as np
import pytest
from scipy.stats import multivariate_normal, norm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import grt_model as gm


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_forward_probabilities_matches_scipy_directly(rng):
    """Independent check of the forward map (not the Sheppard-integration internals
    it's actually built from): compare against scipy's own bivariate normal CDF for
    a spread of (zx, zy, rho), rather than re-deriving the same integration."""
    for _ in range(20):
        zx = rng.uniform(-3, 3)
        zy = rng.uniform(-3, 3)
        rho = rng.uniform(-0.95, 0.95)
        got = gm.forward_probabilities(np.array([zx]), np.array([zy]), np.array([rho]))[0]

        cov = [[1, rho], [rho, 1]]
        p_both = multivariate_normal.cdf([-zx, -zy], mean=[0, 0], cov=cov)
        p_x1 = norm.cdf(-zx)
        p_y1 = norm.cdf(-zy)
        want = [p_both, p_x1 - p_both, p_y1 - p_both, 1 - p_x1 - p_y1 + p_both]

        assert got == pytest.approx(want, abs=1e-6)


def test_forward_probabilities_rows_sum_to_one(rng):
    for name in gm.MODEL_NAMES:
        zx, zy, rho = gm.sample_prior(name, 50, rng)
        probs = gm.forward_probabilities(zx, zy, rho)
        assert probs.sum(axis=-1) == pytest.approx(1.0, abs=1e-8)


def test_pack_unpack_round_trip(rng):
    zx = rng.uniform(-3, 3, (10, 4))
    zy = rng.uniform(-3, 3, (10, 4))
    rho = rng.uniform(-0.9, 0.9, (10, 4))
    vec = gm.pack(zx, zy, rho)
    assert vec.shape == (10, 12)
    zx2, zy2, rho2 = gm.unpack(vec)
    assert zx2 == pytest.approx(zx)
    assert zy2 == pytest.approx(zy)
    assert rho2 == pytest.approx(rho)


@pytest.mark.parametrize("name", gm.MODEL_NAMES)
def test_sample_prior_satisfies_its_own_class_constraints(name, rng):
    zx, zy, rho = gm.sample_prior(name, 200, rng)
    ok, problems = gm.validate(zx, zy, rho, name)
    assert ok, problems


@pytest.mark.parametrize("name", gm.MODEL_NAMES)
def test_sample_prior_sign_convention(name, rng):
    """Level 1 sits below the bound (negative), level 2 above it (positive) --
    the design-consistent sign convention documented in grt_model.py."""
    zx, zy, _ = gm.sample_prior(name, 200, rng)
    assert np.all(zx[:, gm.A_LEVEL == 0] < 0)
    assert np.all(zx[:, gm.A_LEVEL == 1] > 0)
    assert np.all(zy[:, gm.B_LEVEL == 0] < 0)
    assert np.all(zy[:, gm.B_LEVEL == 1] > 0)


def test_validate_flags_a_ps_a_violation():
    zx = np.array([[-1.0, -2.0, 1.0, 1.0]])  # zx_0 != zx_1: PS(A) violated
    zy = np.array([[-1.0, 1.0, -1.0, 1.0]])
    rho = np.array([[0.0, 0.0, 0.0, 0.0]])
    ok, problems = gm.validate(zx, zy, rho, "pi_ps_ds")
    assert not ok
    assert any("PS(A)" in p for p in problems)


def test_validate_flags_out_of_range_correlation():
    zx = np.array([[-1.0, -1.0, 1.0, 1.0]])
    zy = np.array([[-1.0, 1.0, -1.0, 1.0]])
    rho = np.array([[1.5, 0.0, 0.0, 0.0]])
    ok, problems = gm.validate(zx, zy, rho, "ds")
    assert not ok
    assert any("rho" in p for p in problems)


def test_n_free_params_never_exceeds_data_df():
    for name in gm.MODEL_NAMES:
        assert gm.n_free_params(name) <= gm.DATA_DF


def test_n_free_params_matches_hand_count():
    # pi_ps_ds: 2 (zx) + 2 (zy) + 0 (pi) = 4
    assert gm.n_free_params("pi_ps_ds") == 4
    # ds (fully free, no PS): 4 + 4 + 4 = 12 (saturated)
    assert gm.n_free_params("ds") == 12
    # rho1_ds: 4 + 4 + 1 = 9
    assert gm.n_free_params("rho1_ds") == 9


def test_bvn_cdf_reduces_to_independence_at_zero_correlation(rng):
    h = rng.uniform(-2, 2, 20)
    k = rng.uniform(-2, 2, 20)
    got = gm.bvn_cdf(h, k, np.zeros(20))
    want = norm.cdf(h) * norm.cdf(k)
    assert got == pytest.approx(want, abs=1e-10)
