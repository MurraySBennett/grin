import numpy as np
import pytest
from scipy.stats import multivariate_normal, norm

from src import grt_model_3x3_hetero as gm


@pytest.fixture
def rng():
    return np.random.default_rng(4545)


def test_dimensions_and_free_parameter_counts():
    assert gm.N_PARAMS == 45
    assert gm.DATA_DF == 72
    assert gm.n_free_params("ds") == 45
    assert gm.n_free_params("pi_ps_ds") == 12


@pytest.mark.parametrize("name", gm.MODEL_NAMES)
def test_prior_obeys_mean_and_variance_constraints(name, rng):
    theta = gm.sample_prior(name, 30, rng)
    valid, problems = gm.validate(*theta, name)
    assert valid, problems


def test_forward_rows_are_probabilities(rng):
    theta = gm.sample_prior("ds", 20, rng)
    probabilities = gm.forward_probabilities(*theta)
    assert probabilities.shape == (20, 9, 9)
    assert np.all(probabilities >= 0)
    assert probabilities.sum(axis=-1) == pytest.approx(1.0, abs=1e-10)


def test_one_row_matches_scipy(rng):
    mux, muy, sdx, sdy, rho = gm.sample_prior("ds", 1, rng)
    got = gm.forward_probabilities(mux, muy, sdx, sdy, rho)[0, 0].reshape(3, 3)
    distribution = multivariate_normal(
        mean=[mux[0, 0], muy[0, 0]],
        cov=[[sdx[0, 0] ** 2, rho[0, 0] * sdx[0, 0] * sdy[0, 0]],
             [rho[0, 0] * sdx[0, 0] * sdy[0, 0], sdy[0, 0] ** 2]],
    )
    edges = [-np.inf, 0.0, 1.0, np.inf]

    def cdf(x, y):
        if x == -np.inf or y == -np.inf:
            return 0.0
        if x == np.inf and y == np.inf:
            return 1.0
        if x == np.inf:
            return norm.cdf(y, loc=muy[0, 0], scale=sdy[0, 0])
        if y == np.inf:
            return norm.cdf(x, loc=mux[0, 0], scale=sdx[0, 0])
        return distribution.cdf([x, y])

    want = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            want[i, j] = (cdf(edges[i + 1], edges[j + 1])
                          - cdf(edges[i], edges[j + 1])
                          - cdf(edges[i + 1], edges[j])
                          + cdf(edges[i], edges[j]))
    assert got == pytest.approx(want, abs=2e-6)


def test_ps_validation_detects_variance_only_violation(rng):
    theta = list(gm.sample_prior("pi_ps_ds", 1, rng))
    theta[2][0, 1] *= 1.2  # same A level as stimulus 0, different B level
    valid, problems = gm.validate(*theta, "pi_ps_ds")
    assert not valid
    assert any("PS(A) variance" in problem for problem in problems)


def test_pack_round_trip(rng):
    theta = gm.sample_prior("rho1_ps_ds", 4, rng)
    packed = gm.pack(*theta)
    assert packed.shape == (4, 45)
    for original, recovered in zip(theta, gm.unpack(packed)):
        assert recovered == pytest.approx(original)
