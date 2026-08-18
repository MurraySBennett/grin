import numpy as np
import pytest
from scipy.stats import multivariate_normal

from src import grt_model_3x3 as gm


@pytest.fixture
def rng():
    return np.random.default_rng(3003)


def test_dimensions_and_parameter_count():
    assert gm.N_STIMULI == gm.N_RESPONSES == 9
    assert gm.N_PARAMS == 29
    assert gm.DATA_DF == 72
    assert max(map(gm.n_free_params, gm.MODEL_NAMES)) == 29


@pytest.mark.parametrize("name", gm.MODEL_NAMES)
def test_prior_obeys_structural_constraints(name, rng):
    theta = gm.sample_prior(name, 30, rng)
    valid, problems = gm.validate(*theta, name)
    assert valid, problems


def test_forward_rows_are_probabilities(rng):
    theta = gm.sample_prior("ds", 25, rng)
    probabilities = gm.forward_probabilities(*theta)
    assert probabilities.shape == (25, 9, 9)
    assert np.all(probabilities >= 0)
    assert probabilities.sum(axis=-1) == pytest.approx(1.0, abs=1e-10)


def test_one_row_matches_scipy_rectangle_probabilities(rng):
    mu_x, mu_y, rho, bx, by = gm.sample_prior("ds", 1, rng)
    got = gm.forward_probabilities(mu_x, mu_y, rho, bx, by)[0, 0].reshape(3, 3)
    x_edges = [-np.inf, 0.0, bx[0], np.inf]
    y_edges = [-np.inf, 0.0, by[0], np.inf]
    distribution = multivariate_normal(mean=[mu_x[0, 0], mu_y[0, 0]],
                                       cov=[[1, rho[0, 0]], [rho[0, 0], 1]])

    def cdf(x, y):
        if x == -np.inf or y == -np.inf:
            return 0.0
        if x == np.inf and y == np.inf:
            return 1.0
        if x == np.inf:
            from scipy.stats import norm
            return norm.cdf(y, loc=mu_y[0, 0])
        if y == np.inf:
            from scipy.stats import norm
            return norm.cdf(x, loc=mu_x[0, 0])
        return distribution.cdf([x, y])

    want = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            want[i, j] = (cdf(x_edges[i + 1], y_edges[j + 1])
                          - cdf(x_edges[i], y_edges[j + 1])
                          - cdf(x_edges[i + 1], y_edges[j])
                          + cdf(x_edges[i], y_edges[j]))
    assert got == pytest.approx(want, abs=2e-6)


def test_pack_round_trip(rng):
    theta = gm.sample_prior("rho1_ps_ds", 5, rng)
    packed = gm.pack(*theta)
    assert packed.shape == (5, 29)
    for original, recovered in zip(theta, gm.unpack(packed)):
        assert recovered == pytest.approx(original)
